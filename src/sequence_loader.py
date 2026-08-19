"""sequence_loader.py — sliding-window sequences for the recurrent baseline (M1 modelling).

The `data_loader.py` serves INDIVIDUAL (input, action) pairs. A recurrent model
(the WASD-from-FPV LSTM baseline) needs SEQUENCES: T consecutive frames as input,
predicting the action at the LAST frame of the window (many-to-one, T=8 by
default — ~0.5 s at 15 FPS). This module builds those windows on top of the
existing loader WITHOUT duplicating any of its logic — it reuses SessionDataset
for decoding, cropping, the stored-radar path, action assembly, the whole-session
split (D-021), and the optional keep-mask (D-026).

TWO INVARIANTS THIS MUST NOT BREAK (both already enforced in data_loader):
  1. WHOLE-SESSION SPLIT (D-021). A window must never span two sessions — frames
     from different sessions are not temporally continuous. Windows are built
     PER SESSION and never cross a session boundary.
  2. KEEP-MASK CONTIGUITY (D-026). When use_keep_mask=True, blank/no-radar frames
     (menu/halftime/dead) are dropped. A window must be 8 frames that are
     *actually consecutive in time* — so a window may only be drawn from a RUN of
     kept frames with no dropped frame in the middle. We do NOT stitch across a
     gap (that would put a pre-menu frame next to a post-menu frame and call them
     0.5 s apart). Runs shorter than T contribute no windows.

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
                 target_keys=DEFAULT_TARGET_KEYS, use_keep_mask=False):
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self.seq_len = int(seq_len)
        self.target_keys = tuple(target_keys)
        self.crop = crop
        # The underlying per-frame dataset. use_keep_mask here would EXCLUDE masked
        # frames from ITS global index; we instead pass use_keep_mask=False to the
        # inner dataset and do masking ourselves at the RUN level, because we need
        # to know WHERE the gaps are to avoid bridging them. (Excluding frames in
        # the inner index would renumber locals and hide the gaps.) So we read the
        # mask directly per session below.
        self._ds = dl.SessionDataset(session_paths, crop=crop, use_keep_mask=False)
        self._use_keep_mask = use_keep_mask

        # Resolve target-key column indices once, from the action layout. Requires
        # at least one session; action_layout reads key_names from the data.
        if self._ds.n_sessions == 0:
            self._target_cols = []
        else:
            layout = dl.action_layout(self._ds._arrays(0))
            missing = [k for k in self.target_keys if k not in layout]
            if missing:
                raise ValueError(
                    f"target_keys {missing} not in action layout {layout}. "
                    f"Movement keys are a subset of key_names.")
            self._target_cols = [layout.index(k) for k in self.target_keys]

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
        for si, path in enumerate(self._ds.session_paths):
            n = self._ds._session_lengths[si]
            if self._use_keep_mask:
                keep = dl.load_keep_mask(path, expected_frames=n)
                locals_all = (np.nonzero(keep)[0] if keep is not None
                              else np.arange(n))
            else:
                locals_all = np.arange(n)
            for run in _consecutive_runs(locals_all):
                n_runs += 1
                if len(run) < self.seq_len:
                    n_short += 1
                    continue
                # every length-T window within this contiguous run
                for start in range(0, len(run) - self.seq_len + 1):
                    self._windows.append((si, run[start:start + self.seq_len]))
        self._stats = {"runs": n_runs, "runs_too_short": n_short}

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
                            manual_holdout=None, use_keep_mask=False):
    """Construct (train_seq, holdout_seq) with the leak-free whole-session split.

    Mirrors data_loader.build_datasets but yields SequenceDatasets. The split is
    the SAME deterministic whole-session split (D-021) — the two sides are built
    from disjoint session lists, so no window can straddle the boundary. crop
    selects the input feed: "full"/"centre"/(t,l,h,w) for the FPV baseline, or
    "radar" for the #7 radar-vs-movement comparison (same trainer, different feed).
    """
    train_paths, holdout_paths = dl.split_sessions(rec_dir, holdout_frac, manual_holdout)
    train_seq = SequenceDataset(train_paths, crop=crop, seq_len=seq_len,
                                target_keys=target_keys, use_keep_mask=use_keep_mask)
    holdout_seq = SequenceDataset(holdout_paths, crop=crop, seq_len=seq_len,
                                  target_keys=target_keys, use_keep_mask=use_keep_mask)
    return train_seq, holdout_seq
