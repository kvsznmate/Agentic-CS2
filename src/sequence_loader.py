"""sequence_loader.py — sliding-window sequences for the recurrent baseline (M1 modelling).

The `data_loader.py` serves INDIVIDUAL (input, action) pairs. A recurrent model
(the WASD-from-FPV LSTM baseline) needs SEQUENCES: T consecutive frames as input,
predicting the action at the LAST frame of the window (many-to-one, T=8 by
default — ~0.5 s at 15 FPS). This module builds those windows on top of the
existing loader WITHOUT duplicating any of its logic — it reuses SessionDataset
for decoding, cropping, the stored-radar path, action assembly, the whole-session
split (D-021), and the optional keep-mask (D-026).

INVARIANTS THIS MUST NOT BREAK (split + keep-mask enforced in data_loader;
gameplay filter added in D-040):
  1. WHOLE-SESSION SPLIT (D-021). A window must never span two sessions — frames
     from different sessions are not temporally continuous. Windows are built
     PER SESSION and never cross a session boundary.
  2. KEEP-MASK CONTIGUITY (D-026). When use_keep_mask=True, blank/no-radar frames
     (menu/halftime/dead) are dropped. A window must be 8 frames that are
     *actually consecutive in time* — so a window may only be drawn from a RUN of
     kept frames with no dropped frame in the middle. We do NOT stitch across a
     gap (that would put a pre-menu frame next to a post-menu frame and call them
     0.5 s apart). Runs shorter than T contribute no windows.
  3. GAMEPLAY-FILTER CONTIGUITY (D-031/D-035/D-040). When use_gameplay_filter=True,
     dead/spectating/freezetime frames are dropped the SAME way, feeding the SAME
     run-level mask as the keep-mask — so a window never bridges a death/respawn
     or freezetime gap either. Both filters combine (logical AND) and both are
     applied at the RUN level here, never in the inner per-frame dataset (which
     would renumber locals and hide the gaps).

Concretely: for each session we take its kept local indices (all of them if no
mask), split them into maximal runs of consecutive integers, and within each run
emit every length-T window. A window is T consecutive FRAMES; its label is the
action vector of the window's last frame.

WHY BUILD ON SessionDataset RATHER THAN RE-READ NPZ: the loader already handles
v1/v2/v3, the radar-vs-FPV input selection, the fill-and-assert batch safety, and
the mask. Re-implementing frame reads here would be a second code path that could
drift from the format (the exact class of bug DATA_FORMAT.md's single-source rule
guards against). So this class holds a SessionDataset per split side and asks it
for frames by global index; it only adds the windowing index on top.

Memory: like SessionDataset, frames are decompressed and cached per session on
first access. A batch of B windows of T frames materialises B*T frame crops; at
the current data scale this is fine. For a much larger corpus, the same LRU/stream
note in data_loader applies here too.

Usage (see model_lstm.py for the trainer that consumes this):
    from src.sequence_loader import build_sequence_datasets
    train_seq, holdout_seq = build_sequence_datasets(
        crop="full", seq_len=8, target_keys=("w","a","s","d"),
        use_keep_mask=True)
    for X, Y in train_seq.iter_batches(batch_size=32):
        # X: (B, T, H, W, 3) uint8 BGR ; Y: (B, len(target_keys)) float32 0/1
        ...

LOOK TARGETS (navigation yaw — dx/dy, D-036). The movement model gained a second
output branch predicting mouse motion (dx/dy) so the navigation feed can ROTATE
the player, not only strafe (a WASD-only mover can only move along the axis it
spawned facing). These are the SAME raw device deltas the recorder logs (per
DATA_FORMAT.md: dx/dy in device units, +x right, +y down), served at each window's
LAST frame to match the many-to-one label convention. They are exposed WITHOUT
touching the existing (X, Y) button path:
  * get_batch(...)            -> (X, Y_keys)                unchanged
  * get_batch_with_look(...)  -> (X, Y_keys, Y_look)        additive
  * look_balance()            -> per-axis mean/std/mean-abs  for the zero-motion
                                 baseline the trainer reports look error against
IMPORTANT (scope boundary, D-036): dx/dy here are NAVIGATION yaw for the movement
feed. They are NOT combat aim. Aim is the separate detector-gated model (#10 ->
#11) that the arbiter switches to when an enemy is on screen. Predicting dx/dy in
THIS model does not touch that gate. See DECISIONS D-036.
"""

import numpy as np

from src import data_loader as dl


DEFAULT_SEQ_LEN = 8                       # ~0.5 s at 15 FPS (LOOP_FPS)
DEFAULT_TARGET_KEYS = ("w", "a", "s", "d")  # movement-only baseline


def _consecutive_runs(sorted_locals):
    """Split a sorted int array into maximal runs of consecutive integers.

    e.g. [0,1,2, 5,6, 9] -> [array([0,1,2]), array([5,6]), array([9])].
    Used to find stretches of kept frames with NO dropped frame in the middle, so
    a window never bridges a keep-mask gap (D-026 contiguity invariant).
    """
    if len(sorted_locals) == 0:
        return []
    arr = np.asarray(sorted_locals)
    breaks = np.nonzero(np.diff(arr) != 1)[0] + 1
    return np.split(arr, breaks)


class SequenceDataset:
    """Sliding-window sequence view over a set of sessions (many-to-one).

    Wraps a data_loader.SessionDataset (which it uses for all frame decoding,
    cropping, radar selection, action assembly, and the keep-mask). Adds a window
    index: a list of (session_i, [local frame indices of length T]) where the T
    locals are consecutive in time and lie within one session's kept run.

    The label for a window is the action vector at the window's LAST local index,
    restricted to `target_keys` columns (WASD for the baseline).
    """

    def __init__(self, session_paths, crop="full", seq_len=DEFAULT_SEQ_LEN,
                 target_keys=DEFAULT_TARGET_KEYS, use_keep_mask=False,
                 use_gameplay_filter=False):
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self.seq_len = int(seq_len)
        self.target_keys = tuple(target_keys)
        self.crop = crop
        # The underlying per-frame dataset. Any frame-EXCLUDING filter (the D-026
        # keep-mask OR the D-031/D-035 gameplay filter) would, if applied in the
        # inner dataset, drop frames from ITS global index and thereby RENUMBER
        # locals — hiding exactly the gaps a sequence must not stitch across. So
        # the inner dataset is built UNFILTERED (both filters off) and we apply
        # BOTH ourselves at the RUN level below, where we can see where the gaps
        # are. That is why neither use_keep_mask nor use_gameplay_filter is passed
        # to the inner SessionDataset.
        self._ds = dl.SessionDataset(session_paths, crop=crop, use_keep_mask=False)
        self._use_keep_mask = use_keep_mask
        self._use_gameplay_filter = use_gameplay_filter

        # Resolve target-key column indices once, from the action layout. Requires
        # at least one session; action_layout reads key_names from the data.
        # Also resolve the dx/dy columns for the look branch (D-036): the same
        # assembled action vector carries dx/dy, so we read their columns by name
        # here and slice them in get_batch_with_look / look_balance. Resolving
        # from action_layout (not a hardcoded offset) keeps this correct if the
        # vector layout ever changes, per DATA_FORMAT.md's "read columns by name".
        if self._ds.n_sessions == 0:
            self._target_cols = []
            self._look_cols = []
        else:
            layout = dl.action_layout(self._ds._arrays(0))
            missing = [k for k in self.target_keys if k not in layout]
            if missing:
                raise ValueError(
                    f"target_keys {missing} not in action layout {layout}. "
                    f"Movement keys are a subset of key_names.")
            self._target_cols = [layout.index(k) for k in self.target_keys]
            missing_look = [k for k in ("dx", "dy") if k not in layout]
            if missing_look:
                raise ValueError(
                    f"look targets {missing_look} not in action layout {layout}. "
                    f"dx/dy must be present to train the navigation-look branch.")
            self._look_cols = [layout.index("dx"), layout.index("dy")]

        # Build the window index: (session_i, np.array of T consecutive locals).
        # The inner dataset numbers frames 0..n-1 per session in its _index; we
        # mirror that numbering by walking sessions in the same order and using
        # each session's own length. To map a (session_i, local) to the inner
        # dataset's GLOBAL index, we precompute each session's global base offset.
        self._global_base = []
        base = 0
        for si in range(self._ds.n_sessions):
            self._global_base.append(base)
            base += self._ds._session_lengths[si]

        self._windows = []   # list of (session_i, locals_array[T])
        n_runs = n_short = 0
        n_drop_alive = n_drop_freeze = n_drop_keep = 0
        for si, path in enumerate(self._ds.session_paths):
            n = self._ds._session_lengths[si]
            # Per-session KEEP mask over the session's ORIGINAL frame indices; then
            # windows are formed only from maximal runs of CONSECUTIVE kept frames
            # (via _consecutive_runs), so a window never bridges a dropped-frame
            # gap (the D-026 contiguity invariant — now honoured for the gameplay
            # filter too). Both filters exclude frames the same way, so both feed
            # this single mask and combine by logical AND.
            keep_combined = np.ones(n, dtype=bool)
            if self._use_gameplay_filter:
                # Prime the inner cache once (the batches reuse it, so this adds no
                # extra decompression) and read the v4/v5 GSI arrays from it. Drop
                # a frame if it is dead/spectating (alive==0) OR frozen at spawn
                # (round_phase=='freezetime') — i.e. keep only live playtime
                # (D-031/D-035). Sessions lacking the fields (v1/v2/v3) are simply
                # not gameplay-filtered, never dropped wholesale.
                arrs = self._ds._arrays(si)
                if "alive" in arrs:
                    alive = arrs["alive"].astype(bool)
                    n_drop_alive += int(n - int(alive.sum()))
                    keep_combined &= alive
                if "round_phase" in arrs:
                    not_freeze = arrs["round_phase"].astype(str) != dl._FREEZETIME_PHASE
                    # count only freezetime frames still kept after the alive step,
                    # so the two numbers don't double-count the same excluded frame
                    n_drop_freeze += int((keep_combined & ~not_freeze).sum())
                    keep_combined &= not_freeze
            if self._use_keep_mask:
                keep = dl.load_keep_mask(path, expected_frames=n)
                if keep is not None:
                    n_drop_keep += int((keep_combined & ~keep).sum())
                    keep_combined &= keep
            locals_all = np.nonzero(keep_combined)[0]
            for run in _consecutive_runs(locals_all):
                n_runs += 1
                if len(run) < self.seq_len:
                    n_short += 1
                    continue
                # every length-T window within this contiguous run
                for start in range(0, len(run) - self.seq_len + 1):
                    self._windows.append((si, run[start:start + self.seq_len]))
        self._stats = {"runs": n_runs, "runs_too_short": n_short,
                       "dropped_alive": n_drop_alive,
                       "dropped_freezetime": n_drop_freeze,
                       "dropped_keepmask": n_drop_keep}
        if self._use_gameplay_filter and (n_drop_alive or n_drop_freeze):
            print(f"  sequence gameplay filter (D-031/D-035): dropped "
                  f"{n_drop_alive} dead/spectating + {n_drop_freeze} freezetime "
                  f"frame(s) before windowing (live playtime only).")
        if self._use_keep_mask and n_drop_keep:
            print(f"  sequence keep-mask (D-026): dropped {n_drop_keep} "
                  f"blank/no-radar frame(s) before windowing.")

    def __len__(self):
        return len(self._windows)

    @property
    def n_sessions(self):
        return self._ds.n_sessions

    @property
    def n_targets(self):
        return len(self._target_cols)

    def names(self):
        return self._ds.names()

    def _global_index(self, session_i, local_i):
        return self._global_base[session_i] + int(local_i)

    def get_batch(self, window_positions):
        """Return (X, Y) for the given window indices.

        X : (B, T, h, w, 3) uint8 — input frames per window (FPV crop or radar).
        Y : (B, n_targets) float32 — 0/1 held-state of target_keys at each
            window's LAST frame.

        Frames are fetched from the inner SessionDataset by global index. We fetch
        ALL T*B frames in one inner get_batch call (grouped by session inside the
        loader) then reshape to (B, T, ...), so decoding/caching happens once per
        contributing session per batch rather than per frame.
        """
        B = len(window_positions)
        T = self.seq_len
        if B == 0:
            # Shapes still well-defined so callers can allocate/skip cleanly.
            th, tw = self._ds._out_hw()
            return (np.empty((0, T, th, tw, 3), np.uint8),
                    np.empty((0, self.n_targets), np.float32))

        # Flatten all (window, t) frame requests into one global-index list, in
        # row-major (window then time) order so the reshape below is correct.
        flat_global = []
        last_global = []
        for wp in window_positions:
            si, locals_arr = self._windows[wp]
            for li in locals_arr:
                flat_global.append(self._global_index(si, li))
            last_global.append(self._global_index(si, locals_arr[-1]))

        Xflat, _Yflat = self._ds.get_batch(flat_global)     # (B*T, h, w, 3)
        h, w = Xflat.shape[1], Xflat.shape[2]
        X = Xflat.reshape(B, T, h, w, 3)

        # Labels: action vector at each window's LAST frame, target columns only.
        # Fetch the last frames' actions via the loader's assembled vector so the
        # column semantics match action_layout exactly.
        _Xlast, Ylast_full = self._ds.get_batch(last_global)  # (B, 15)
        Y = Ylast_full[:, self._target_cols].astype(np.float32)
        return X, Y

    def get_batch_with_look(self, window_positions):
        """Like get_batch, but ALSO return the look (dx/dy) targets (D-036).

        X      : (B, T, h, w, 3) uint8 — input frames per window.
        Y_keys : (B, n_targets) float32 — 0/1 held-state of target_keys at the
                 window's LAST frame (identical to get_batch's Y).
        Y_look : (B, 2) float32 — RAW dx, dy (device units) at the window's LAST
                 frame. NOT standardized here: standardization stats are computed
                 by the trainer from the TRAIN split only and applied there, so
                 this loader stays a pure data source and never leaks held-out
                 statistics into the train transform (D-021 discipline extended to
                 the look targets). +x is physical right, +y down (DATA_FORMAT.md).

        Frames and last-frame actions are fetched exactly as in get_batch (one
        grouped inner call each), so the button path's behaviour is unchanged and
        the look columns are just an additional slice of the SAME assembled
        action vector — they cannot fall out of sync with Y_keys.
        """
        B = len(window_positions)
        T = self.seq_len
        if B == 0:
            th, tw = self._ds._out_hw()
            return (np.empty((0, T, th, tw, 3), np.uint8),
                    np.empty((0, self.n_targets), np.float32),
                    np.empty((0, 2), np.float32))

        flat_global = []
        last_global = []
        for wp in window_positions:
            si, locals_arr = self._windows[wp]
            for li in locals_arr:
                flat_global.append(self._global_index(si, li))
            last_global.append(self._global_index(si, locals_arr[-1]))

        Xflat, _Yflat = self._ds.get_batch(flat_global)
        h, w = Xflat.shape[1], Xflat.shape[2]
        X = Xflat.reshape(B, T, h, w, 3)

        _Xlast, Ylast_full = self._ds.get_batch(last_global)  # (B, 15)
        Y_keys = Ylast_full[:, self._target_cols].astype(np.float32)
        Y_look = Ylast_full[:, self._look_cols].astype(np.float32)   # raw dx/dy
        return X, Y_keys, Y_look

    def look_balance(self):
        """Per-axis dx/dy statistics over all windows' LAST frames (D-036).

        Returns {"dx": {"mean", "std", "mean_abs"}, "dy": {...}, "n": N}, all in
        RAW device units. Two uses in the trainer:
          * mean/std are the STANDARDIZATION stats — but the trainer recomputes
            these from the TRAIN split alone and reuses them for held-out (never
            the other way), so calling this on the train dataset is how those
            stats are obtained; calling it on held-out is only for reporting.
          * mean_abs is the ZERO-MOTION BASELINE: predicting dx=dy=0 gives a mean
            absolute error of exactly mean_abs, so the model's look MAE is honest
            only when compared against it (the dx/dy analogue of the button
            majority-class baseline). A model barely beating mean_abs has learned
            almost nothing about turning.
        Reads only the last frame of each window, batched to bound memory.
        """
        if len(self) == 0:
            nan = float("nan")
            return {"dx": {"mean": nan, "std": nan, "mean_abs": nan},
                    "dy": {"mean": nan, "std": nan, "mean_abs": nan}, "n": 0}
        last_global = []
        for si, locals_arr in self._windows:
            last_global.append(self._global_index(si, locals_arr[-1]))
        cols = []
        for s in range(0, len(last_global), 8192):
            gi = last_global[s:s + 8192]
            _X, Yfull = self._ds.get_batch(gi)
            cols.append(Yfull[:, self._look_cols].astype(np.float64))
        allv = np.concatenate(cols, axis=0)   # (N, 2) raw dx/dy
        out = {"n": int(allv.shape[0])}
        for i, ax in enumerate(("dx", "dy")):
            v = allv[:, i]
            out[ax] = {"mean": float(v.mean()), "std": float(v.std()),
                       "mean_abs": float(np.abs(v).mean())}
        return out

    def eventful_mask(self, rare_keys=("a", "s", "d", "space"), dx_abs_thresh=None):
        """Per-window boolean: is this window 'eventful' (a turn / rare action)?

        Returns (mask[N] bool, info dict). A window is eventful if, at its LAST
        frame, EITHER any of `rare_keys` is held OR |dx| exceeds dx_abs_thresh.
        These are the windows under-represented in forward-heavy data (D-036/keys
        imbalance): rare movement keys (A/S/D/space) and real turns (large dx).
        Used ONLY by the trainer to OVERSAMPLE such windows so the model sees more
        of them per epoch — it does NOT change the labels, the loss, or eval.

        rare_keys deliberately EXCLUDES w/shift (already frequent — oversampling
        them would defeat the purpose). dx_abs_thresh defaults to this dataset's
        dx std (a turn 'bigger than typical'); pass a number to override. Reads
        only last-frame vectors, batched, like target_balance/look_balance.

        NOTE: computed on THIS dataset only. The trainer calls it on the TRAIN
        SequenceDataset; it cannot move a window across the D-021 split (each side
        is a separate dataset over disjoint sessions).
        """
        n = len(self)
        if n == 0:
            return np.zeros(0, dtype=bool), {"n": 0, "eventful": 0}
        # Resolve which target columns count as 'rare' (intersection with the
        # dataset's target_keys, by name, so it's robust to key-set changes).
        rare_idx = [i for i, k in enumerate(self.target_keys) if k in rare_keys]
        last_global = [self._global_index(si, la[-1]) for si, la in self._windows]
        rare_hit = np.zeros(n, dtype=bool)
        dx_vals = np.zeros(n, dtype=np.float64)
        pos = 0
        for s in range(0, len(last_global), 8192):
            gi = last_global[s:s + 8192]
            _X, Yfull = self._ds.get_batch(gi)
            b = len(gi)
            if rare_idx:
                keycols = Yfull[:, [self._target_cols[i] for i in rare_idx]]
                rare_hit[pos:pos + b] = (keycols > 0.5).any(axis=1)
            dx_vals[pos:pos + b] = Yfull[:, self._look_cols[0]]   # dx column
            pos += b
        if dx_abs_thresh is None:
            dx_abs_thresh = float(np.std(dx_vals)) if n > 1 else 0.0
        dx_hit = np.abs(dx_vals) > dx_abs_thresh
        mask = rare_hit | dx_hit
        info = {"n": n, "eventful": int(mask.sum()),
                "rare_hit": int(rare_hit.sum()), "dx_hit": int(dx_hit.sum()),
                "dx_abs_thresh": float(dx_abs_thresh)}
        return mask, info

    def iter_batches(self, batch_size=32, shuffle=True, seed=None, drop_last=False):
        """Yield (X, Y) batches over all windows.

        Shuffling is over WINDOWS within this dataset only — it cannot move a
        window across the train/holdout boundary (each side is a separate
        SequenceDataset over disjoint sessions), and it cannot break window
        contiguity (each window's T frames were fixed as consecutive at build
        time; shuffling only reorders whole windows).
        """
        n = len(self)
        order = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        end = n - (n % batch_size) if (drop_last and batch_size) else n
        for start in range(0, end, batch_size):
            yield self.get_batch(order[start:start + batch_size].tolist())

    def target_balance(self):
        """Fraction of windows where each target key is held at the last frame.

        Cheap pass for sanity + the honest baseline: if 'w' is held in 85% of
        windows, a model predicting 'always w' scores 85% on it, so this is the
        majority-class reference the trainer reports lift against. Reads only the
        last frame of each window.
        """
        if len(self) == 0:
            return {k: float("nan") for k in self.target_keys}
        last_global = []
        for si, locals_arr in self._windows:
            last_global.append(self._global_index(si, locals_arr[-1]))
        # Batch the reads to bound memory.
        held = np.zeros(self.n_targets, dtype=np.float64)
        total = 0
        for s in range(0, len(last_global), 8192):
            gi = last_global[s:s + 8192]
            _X, Yfull = self._ds.get_batch(gi)
            held += Yfull[:, self._target_cols].sum(axis=0)
            total += len(gi)
        return {k: float(held[i] / total) for i, k in enumerate(self.target_keys)}


def build_sequence_datasets(rec_dir=dl._REC_DIR, crop="full", seq_len=DEFAULT_SEQ_LEN,
                            target_keys=DEFAULT_TARGET_KEYS,
                            holdout_frac=dl.DEFAULT_HOLDOUT_FRAC,
                            manual_holdout=None, use_keep_mask=False,
                            use_gameplay_filter=False):
    """Construct (train_seq, holdout_seq) with the leak-free whole-session split.

    Mirrors data_loader.build_datasets but yields SequenceDatasets. The split is
    the SAME deterministic whole-session split (D-021) — the two sides are built
    from disjoint session lists, so no window can straddle the boundary. crop
    selects the input feed: "full"/"centre"/(t,l,h,w) for the FPV baseline, or
    "radar" for the #7 radar-vs-movement comparison (same trainer, different feed).

    use_gameplay_filter (D-040): when True, windows are built only from live-
    playtime frames — GSI-alive AND not-freezetime (D-031/D-035) — with windows
    never bridging a dropped-frame gap (handled at the run level inside
    SequenceDataset). Default OFF, mirroring use_keep_mask, so the #7 radar gate
    and the committed baseline don't move unless asked. Applied identically to
    train and held-out so the split's meaning is unchanged. v4+/v5 only; older
    sessions carry no alive/round_phase and are simply not filtered.
    """
    train_paths, holdout_paths = dl.split_sessions(rec_dir, holdout_frac, manual_holdout)
    train_seq = SequenceDataset(train_paths, crop=crop, seq_len=seq_len,
                                target_keys=target_keys, use_keep_mask=use_keep_mask,
                                use_gameplay_filter=use_gameplay_filter)
    holdout_seq = SequenceDataset(holdout_paths, crop=crop, seq_len=seq_len,
                                  target_keys=target_keys, use_keep_mask=use_keep_mask,
                                  use_gameplay_filter=use_gameplay_filter)
    return train_seq, holdout_seq
