"""data_loader.py — dataset loader for self-recorded sessions (Issue #6).

Reads the recordings produced by `recorder.py` and emits (input, action-label)
batches for training. Reads ALL on-disk formats defined in DATA_FORMAT.md:
  * v3 session folder  (manifest + chunk_*.npz WITH a per-frame `radar` array,
                        D-024 — FPV + separate high-res radar),
  * v2 session folder  (manifest + chunk_*.npz, FPV only, D-018), and
  * v1 single .npz      (a legacy one-chunk FPV-only session).

It is the other half of #6 (the recording half is `recorder.py --record`). Three
things #6's acceptance asks for, all here:
  1. a loader emitting (input, action-label) batches,
  2. with configurable inputs (full FPV / centre FPV / the high-res radar),
  3. a reserved held-out split — reserved BEFORE the data is used, so the #7
     (radar) and #10 (detection) gates that read this data stay honest.

──────────────────────────────────────────────────────────────────────────────
TWO-RESOLUTION DATA (D-024) — the `radar` input is a STORED ARRAY, not a crop
──────────────────────────────────────────────────────────────────────────────
Up to v2, a session stored only the 150x270 FPV `frames`, and "the radar" was a
sub-RECTANGLE carved from that downscaled frame — which turned out too low-res to
read position (the #7 finding). v3 fixes this at capture time: each frame stores
a SEPARATE high-res radar crop (128x128) taken from the full-res grab before the
FPV downscale (D-024). So in this loader:

  * input="full"/"centre"/(t,l,h,w)  -> a crop of the FPV `frames` (as before).
  * input="radar"                    -> the STORED `radar` array (v3), served
                                        directly. NOT a crop of `frames`.

A v1/v2 session has no `radar` array, so input="radar" on it raises a clear
error (rather than silently falling back to the old low-res FPV crop). That's
deliberate: the radar gate (#7) must run on the high-res radar, and mixing in
old FPV-crop radars would corrupt it. v1/v2 remain fully usable for the FPV
inputs. See DATA_FORMAT.md for the schema and RADAR_OUT_HW for the size.

──────────────────────────────────────────────────────────────────────────────
CRASHED / INCOMPLETE SESSIONS ARE SKIPPED AT DISCOVERY (D-022)
──────────────────────────────────────────────────────────────────────────────
A recording interrupted mid-write leaves a v2/v3 folder whose `manifest.json` has
`complete: false` and possibly `chunks: []`, plus a stray `chunk_NNNNN.npz.tmp.npz`
never atomically renamed. This is the crash-safety design working as intended
(D-018/D-019): the manifest never claims frames that weren't finalized. But such
a folder is NOT usable data.

`discover_sessions()` therefore EXCLUDES incomplete/empty sessions, so they never
enter the split at all. This matters because the split is by hash BEFORE any
frames load: if an empty session were discovered, it could be hashed into
held-out and become a phantom 0-frame test set — the failure that motivated this
rule. Skips are reported (see `discover_sessions(report=True)` / the CLI). A v1
single `.npz` has no manifest/flag, so it is assumed complete.

──────────────────────────────────────────────────────────────────────────────
THE HELD-OUT SPLIT (D-021) — read this before changing it
──────────────────────────────────────────────────────────────────────────────
The split is BY WHOLE SESSION, never by frame. Frames inside one session are
consecutive ticks of the same continuous motion, so they are highly correlated;
a random per-FRAME split would place near-duplicate frames in both train and
held-out and silently inflate every downstream metric (radar probe #7, detector
#10, aim #11). Splitting whole sessions removes that leak.

Assignment is DETERMINISTIC by a hash of the session name, so the same session
always lands on the same side regardless of what else is recorded later (adding
sessions never reshuffles existing ones), and no shuffle seed or state file is
needed. CAVEAT: at small session counts a ~20% hash split may put 0 or 2 sessions
in held-out by luck — reproducible + leak-free, not exactly-20%-right-now.
`manual_holdout=` overrides for the tiny-data phase; it is not the default.

Read with np.load(allow_pickle=False) — the format is plain arrays only.
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np

from src import capture_config as cfg


_REC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "recordings")

# Per-frame arrays that must stay index-aligned (must match DATA_FORMAT.md).
# `radar` is present only in v3 sessions (D-024); it is handled as an OPTIONAL
# per-frame array (see _PER_FRAME_OPTIONAL) so v1/v2 still load.
_PER_FRAME = ["frames", "timestamps", "keys", "lclick", "rclick", "dx", "dy"]
_PER_FRAME_OPTIONAL = ["radar"]

# Schema versions this loader understands. 1 = legacy single file, 2 = chunked
# FPV-only, 3 = chunked FPV + high-res radar (D-024). Anything else is refused
# loudly rather than loaded on a guess (DATA_FORMAT.md extension rule 1).
_SUPPORTED_SCHEMA = {1, 2, 3}

# FPV model input geometry, from D-012 / DATA_FORMAT.md. (H, W).
FRAME_H, FRAME_W = 150, 270

# Stored radar geometry, from D-024 / capture_config. (H, W). The radar is served
# from the stored `radar` array at this size; it is NOT cropped from the FPV.
RADAR_H, RADAR_W = cfg.RADAR_OUT_HW

# ── FPV crop rectangles as (top, left, height, width) in stored-FPV pixels ────
# "full" needs no rectangle (whole FPV frame). "centre" is a provisional FPV
# window for the detection/aim models — tune against real frames in #11.
CENTRE_CROP_DEFAULT = (27, 55, 96, 160)  # (top, left, h, w) -> rows 27..123, cols 55..215

# NOTE (D-024): the old RADAR_CROP_DEFAULT — a rectangle carved from the 150x270
# FPV — is RETIRED. The radar is now a separate stored array (input="radar"),
# not an FPV crop. Kept out on purpose so nothing silently uses the dead low-res
# path. To crop a custom sub-region of the FPV, pass a (top,left,h,w) tuple.

# Held-out fraction for the deterministic hash split (D-021).
DEFAULT_HOLDOUT_FRAC = 0.20

# Hash bucket resolution — session name -> integer in [0, _HASH_BUCKETS).
_HASH_BUCKETS = 1000


# ─────────────────────────────────────────────────────────────────────────────
# Session discovery
# ─────────────────────────────────────────────────────────────────────────────

def _session_usability(path):
    """Judge whether a session path is usable data. Returns (ok, reason).

    v2/v3 folder: usable iff its manifest parses, `complete` is not False, and it
    lists at least one chunk. A crashed recording (D-018/D-019 crash-safety)
    yields `complete: false` and/or `chunks: []` — correctly flagged unusable
    here so it never reaches the split (D-022).

    v1 file: assumed usable (no manifest/flag).
    """
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.json")
        try:
            with open(manifest_path) as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            return False, f"manifest unreadable ({e.__class__.__name__})"
        if m.get("complete") is False:
            return False, "incomplete (complete=false — crashed/interrupted recording)"
        if not m.get("chunks"):
            return False, "no chunks listed (empty session)"
        return True, ""
    return True, ""


def discover_sessions(rec_dir=_REC_DIR, report=False):
    """Return a sorted list of USABLE session paths in rec_dir.

    A session is either a v2/v3 folder (has manifest.json) or a v1 bare .npz.
    Incomplete/empty sessions are EXCLUDED (see _session_usability and D-022) so
    they never enter the split. Sorted by name for stable ordering.

    report=True: also print a line for each skipped session (path + reason).
    """
    if not os.path.isdir(rec_dir):
        return []
    folders = [p for p in glob.glob(os.path.join(rec_dir, "*"))
               if os.path.isdir(p) and os.path.isfile(os.path.join(p, "manifest.json"))]
    files = glob.glob(os.path.join(rec_dir, "*.npz"))
    candidates = sorted(folders + files)

    usable, skipped = [], []
    for path in candidates:
        ok, reason = _session_usability(path)
        (usable if ok else skipped).append((path, reason))

    if report and skipped:
        print(f"Skipped {len(skipped)} unusable session(s):")
        for path, reason in skipped:
            print(f"  {session_name(path):<28} {reason}")
        print()

    return [p for p, _ in usable]


def list_skipped_sessions(rec_dir=_REC_DIR):
    """Return [(path, reason), ...] for sessions discovery excludes."""
    if not os.path.isdir(rec_dir):
        return []
    folders = [p for p in glob.glob(os.path.join(rec_dir, "*"))
               if os.path.isdir(p) and os.path.isfile(os.path.join(p, "manifest.json"))]
    files = glob.glob(os.path.join(rec_dir, "*.npz"))
    out = []
    for path in sorted(folders + files):
        ok, reason = _session_usability(path)
        if not ok:
            out.append((path, reason))
    return out


def session_name(path):
    """Canonical session name for hashing/reporting: folder name, or file stem."""
    base = os.path.basename(path.rstrip(os.sep))
    if base.endswith(".npz"):
        base = base[:-4]
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic whole-session split (D-021)
# ─────────────────────────────────────────────────────────────────────────────

def _session_bucket(name):
    """Map a session name to a stable integer bucket in [0, _HASH_BUCKETS).

    Uses blake2b (stable across processes/machines, unlike Python's salted
    hash()) so the split is reproducible everywhere and never depends on run
    order or a stored seed.
    """
    h = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % _HASH_BUCKETS


def is_holdout(name, holdout_frac=DEFAULT_HOLDOUT_FRAC, manual_holdout=None):
    """True if this session belongs to the held-out split.

    manual_holdout (optional): explicit session names to force into held-out —
    the tiny-data override from D-021. Takes precedence for named sessions; all
    others fall back to the hash. Deliberately an override, not the default.
    """
    if manual_holdout and name in set(manual_holdout):
        return True
    return _session_bucket(name) < holdout_frac * _HASH_BUCKETS


def split_sessions(rec_dir=_REC_DIR, holdout_frac=DEFAULT_HOLDOUT_FRAC,
                   manual_holdout=None):
    """Partition discovered (usable) sessions into (train, holdout) lists of paths.

    Pure function of the usable session names present + the fraction. No frame is
    ever split across the boundary — the leak-free guarantee. Incomplete sessions
    are already excluded by discover_sessions (D-022).
    """
    train, holdout = [], []
    for path in discover_sessions(rec_dir):
        name = session_name(path)
        (holdout if is_holdout(name, holdout_frac, manual_holdout) else train).append(path)
    return train, holdout


# ─────────────────────────────────────────────────────────────────────────────
# Reading one session (v1 file or v2/v3 folder) into arrays
# ─────────────────────────────────────────────────────────────────────────────

def _check_schema(arr_scalar, where):
    v = int(arr_scalar)
    if v not in _SUPPORTED_SCHEMA:
        raise ValueError(
            f"{where}: schema_version={v} is not supported "
            f"(this loader knows {sorted(_SUPPORTED_SCHEMA)}). "
            f"Update data_loader.py and DATA_FORMAT.md together before reading it.")
    return v


def _load_npz(path):
    out = {}
    with np.load(path, allow_pickle=False) as d:
        for k in d.files:
            out[k] = d[k]
    return out


def load_session_arrays(path):
    """Load a session (v1 file or v2/v3 folder) into one dict of index-aligned arrays.

    For a folder: reads manifest.json, then concatenates the per-frame arrays of
    the listed chunks IN MANIFEST ORDER (DATA_FORMAT.md reconstruction rule).
    Metadata (key_names/geom/…) is taken from the first chunk. The `radar` array
    (v3, D-024) is concatenated when present and simply absent for v1/v2.

    Discovery already filters out incomplete sessions (D-022); the empty-manifest
    guard below still stands for direct callers who bypass discovery.

    Returns a dict with the _PER_FRAME arrays (+ `radar` if present) plus
    metadata; every per-frame array shares length N (verified — alignment IS the
    sync guarantee).
    """
    if os.path.isdir(path):
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        chunk_names = manifest.get("chunks", [])
        if not chunk_names:
            raise ValueError(
                f"Session {path} lists no chunks in its manifest "
                f"(complete={manifest.get('complete')}). This is a crashed/empty "
                f"recording; discover_sessions() excludes it — load it directly "
                f"only if you know why.")
        want = _PER_FRAME + _PER_FRAME_OPTIONAL
        parts = {k: [] for k in want}
        meta = {}
        for cname in chunk_names:
            chunk = _load_npz(os.path.join(path, cname))
            if "schema_version" in chunk:
                _check_schema(chunk["schema_version"], f"{path}/{cname}")
            for k in want:
                if k in chunk:
                    parts[k].append(chunk[k])
            for k in ("key_names", "schema_version", "geom", "loop_fps_target"):
                if k in chunk and k not in meta:
                    meta[k] = chunk[k]
        arrays = {k: np.concatenate(v, axis=0) for k, v in parts.items() if v}
        arrays.update(meta)
    else:
        arrays = _load_npz(path)
        if "schema_version" in arrays:
            _check_schema(arrays["schema_version"], path)

    # Alignment assertion — the on-disk sync guarantee, re-checked in memory.
    # Includes `radar` when present, so a v3 session with a mis-sized radar array
    # is caught rather than served.
    check = _PER_FRAME + [k for k in _PER_FRAME_OPTIONAL if k in arrays]
    counts = {k: arrays[k].shape[0] for k in check if k in arrays}
    if len(set(counts.values())) > 1:
        raise ValueError(
            f"{path}: per-frame arrays have mismatched lengths {counts} — "
            f"the frame/action alignment is broken; refusing to serve this "
            f"session rather than train on misaligned data.")
    return arrays


# ─────────────────────────────────────────────────────────────────────────────
# Inputs (FPV crops or the stored radar) and action-vector assembly
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_input(crop):
    """Map an input spec to a plan: ('radar', None) or ('fpv', rect|None).

    crop may be:
      * "full"   -> ('fpv', None)                  whole FPV frame
      * "centre" -> ('fpv', CENTRE_CROP_DEFAULT)   centred FPV window
      * "radar"  -> ('radar', None)                the STORED high-res radar (v3)
      * (t,l,h,w)-> ('fpv', (t,l,h,w))             custom FPV crop (e.g. #11)
    """
    if crop is None or crop == "full":
        return ("fpv", None)
    if crop == "centre":
        return ("fpv", CENTRE_CROP_DEFAULT)
    if crop == "radar":
        return ("radar", None)
    if isinstance(crop, (tuple, list)) and len(crop) == 4:
        return ("fpv", tuple(int(x) for x in crop))
    raise ValueError(
        f"Unknown input {crop!r}. Use 'full', 'centre', 'radar', or a "
        f"(top, left, h, w) FPV crop tuple.")


def _apply_crop(frames, rect):
    """Crop a batch of FPV frames (B,H,W,C) to rect=(top,left,h,w); None = unchanged."""
    if rect is None:
        return frames
    top, left, h, w = rect
    if top < 0 or left < 0 or top + h > FRAME_H or left + w > FRAME_W:
        raise ValueError(
            f"FPV crop {rect} falls outside the {FRAME_H}x{FRAME_W} frame. "
            f"Fix the rectangle (rows 0..{FRAME_H}, cols 0..{FRAME_W}).")
    return frames[:, top:top + h, left:left + w, :]


def assemble_action_vector(arrays, idx):
    """Build the action-label matrix for row indices idx.

    Layout (columns), from DATA_FORMAT.md's action composition:
        [ keys(11) , lclick(1) , rclick(1) , dx(1) , dy(1) ]  -> 15 columns.
    keys are 0/1 in key_names order; dx/dy are raw device deltas (kept float32
    for the model). Returned as float32 (B, 15). Use action_layout() to read
    columns by name rather than hardcoding offsets.
    """
    keys = arrays["keys"][idx].astype(np.float32)          # (B, 11)
    lclick = arrays["lclick"][idx].astype(np.float32)[:, None]
    rclick = arrays["rclick"][idx].astype(np.float32)[:, None]
    dx = arrays["dx"][idx].astype(np.float32)[:, None]
    dy = arrays["dy"][idx].astype(np.float32)[:, None]
    return np.concatenate([keys, lclick, rclick, dx, dy], axis=1)


def action_layout(arrays):
    """Return the column names of the assembled action vector, in order."""
    key_names = [s.decode() if isinstance(s, bytes) else str(s)
                 for s in arrays["key_names"]]
    return key_names + ["lclick", "rclick", "dx", "dy"]


# ─────────────────────────────────────────────────────────────────────────────
# The dataset object
# ─────────────────────────────────────────────────────────────────────────────

class SessionDataset:
    """Indexable (input, action-vector) dataset over a set of sessions.

    `input_kind` selects what X is:
      * "full"/"centre"/(t,l,h,w) -> a crop of the FPV `frames`.
      * "radar"                   -> the stored high-res `radar` array (v3, D-024).

    Loads the given session paths, records where each session's frames sit in a
    global index, and serves batches. Frames are decompressed per session on
    first access and cached in memory — fine for the current data scale; for much
    larger corpora this becomes a memory concern (noted in build_datasets).

    IMPORTANT: this class serves EXACTLY the sessions it is handed. It has no
    notion of train/holdout itself — that separation is enforced by
    build_datasets(), which constructs two independent SessionDatasets from the
    split. A held-out session simply never enters the training dataset's path
    list, so it cannot leak through iteration.
    """

    def __init__(self, session_paths, crop="full"):
        self.session_paths = list(session_paths)
        self.crop = crop
        self._input_kind, self._rect = _resolve_input(crop)
        self._cache = {}  # path -> arrays dict (lazy)

        self._index = []            # list of (session_i, local_i)
        self._session_lengths = []
        for si, path in enumerate(self.session_paths):
            n = self._session_length(path)
            self._session_lengths.append(n)
            self._index.extend((si, li) for li in range(n))

    @staticmethod
    def _session_length(path):
        """Frame count for a session, cheaply (manifest for folders; load for v1)."""
        if os.path.isdir(path):
            with open(os.path.join(path, "manifest.json")) as f:
                m = json.load(f)
            if m.get("total_frames") is not None:
                return int(m["total_frames"])
            total = 0
            for cname in m.get("chunks", []):
                with np.load(os.path.join(path, cname), allow_pickle=False) as d:
                    total += int(d["frames"].shape[0])
            return total
        with np.load(path, allow_pickle=False) as d:
            return int(d["frames"].shape[0])

    def _arrays(self, session_i):
        path = self.session_paths[session_i]
        if path not in self._cache:
            self._cache[path] = load_session_arrays(path)
        return self._cache[path]

    def __len__(self):
        return len(self._index)

    @property
    def n_sessions(self):
        return len(self.session_paths)

    def names(self):
        return [session_name(p) for p in self.session_paths]

    def _out_hw(self):
        """(H, W) of the served input X, for allocating batches."""
        if self._input_kind == "radar":
            return RADAR_H, RADAR_W
        if self._rect:
            return self._rect[2], self._rect[3]
        return FRAME_H, FRAME_W

    def _session_has_radar(self, session_i):
        """True iff this session's loaded arrays contain a `radar` array (v3)."""
        return "radar" in self._arrays(session_i)

    def _session_input(self, arrays, locals_arr):
        """Return the input array X for one session at the given local indices."""
        if self._input_kind == "radar":
            if "radar" not in arrays:
                raise ValueError(
                    "input='radar' requested but this session has no `radar` "
                    "array — it is a v1/v2 (FPV-only) recording. The high-res "
                    "radar exists only in v3 sessions (D-024). Re-record with "
                    "`python -m src.recorder --record`, or select an FPV input "
                    "('full'/'centre') for this session.")
            return arrays["radar"][locals_arr]               # (b, RADAR_H, RADAR_W, 3)
        frames = arrays["frames"][locals_arr]                # (b, 150,270,3)
        return _apply_crop(frames, self._rect)

    def get_batch(self, global_indices):
        """Return (X, Y) for the given global frame indices.

        X : (B, h, w, 3) uint8 BGR — the selected FPV crop, or the stored radar.
        Y : (B, 15) float32        — assembled action vectors (action_layout()).

        Frames are grouped by session so each session's arrays are touched once
        per batch. Row order of the output matches the order of global_indices.

        Safety (hardened after a real scare, 2026-08): X is filled from actual
        session data and then EVERY row is asserted to have been written, so an
        uninitialised buffer can never escape as if it were data — even on a
        future code path that skips a session. For input='radar', radar
        availability is checked UP FRONT for every contributing session, so a
        batch that mixes a v3 and a v1/v2 session fails with one clear error
        BEFORE any allocation, rather than partway through the fill. (The earlier
        version used np.empty and a per-session-late check; the fill-and-assert
        below is the belt-and-suspenders replacement.)
        """
        by_session = {}
        for out_pos, gi in enumerate(global_indices):
            si, li = self._index[gi]
            by_session.setdefault(si, ([], []))
            by_session[si][0].append(li)
            by_session[si][1].append(out_pos)

        # UP-FRONT radar availability check across ALL contributing sessions, so
        # a mixed v3 + v1/v2 radar batch raises once, before allocation.
        if self._input_kind == "radar":
            missing = [session_name(self.session_paths[si])
                       for si in by_session if not self._session_has_radar(si)]
            if missing:
                raise ValueError(
                    f"input='radar' requested but {len(missing)} contributing "
                    f"session(s) have no `radar` array (v1/v2, FPV-only): "
                    f"{missing}. The high-res radar exists only in v3 sessions "
                    f"(D-024). Exclude those sessions or select an FPV input.")

        B = len(global_indices)
        th, tw = self._out_hw()
        # np.full with a sentinel (not np.empty): if any row were left unwritten,
        # it would be an obvious, catchable 255 block rather than silent garbage.
        # The written-mask assert below is the real guard; the sentinel is a
        # visible fallback if that assert were ever removed.
        X = np.full((B, th, tw, 3), 255, dtype=np.uint8)
        Y = np.zeros((B, 15), dtype=np.float32)
        written = np.zeros(B, dtype=bool)

        for si, (locals_, out_positions) in by_session.items():
            arrays = self._arrays(si)
            locals_arr = np.array(locals_)
            xin = self._session_input(arrays, locals_arr)
            acts = assemble_action_vector(arrays, locals_arr)
            for row, out_pos in enumerate(out_positions):
                X[out_pos] = xin[row]
                Y[out_pos] = acts[row]
                written[out_pos] = True

        if not written.all():
            # Should be impossible: every global index maps to exactly one
            # session, which fills its row. If this fires, the index/grouping is
            # broken — refuse rather than return partially-uninitialised data.
            n_missing = int((~written).sum())
            raise AssertionError(
                f"get_batch left {n_missing}/{B} rows unwritten — refusing to "
                f"return a batch with uninitialised rows. This indicates a bug in "
                f"index grouping, not bad data.")
        return X, Y

    def iter_batches(self, batch_size=64, shuffle=True, seed=None, drop_last=False):
        """Yield (X, Y) batches over all frames in this dataset.

        shuffle applies to FRAME order WITHIN this dataset only — it cannot move
        frames across the train/holdout boundary, because a held-out session's
        frames are not in this dataset's index at all.
        """
        n = len(self)
        order = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        end = n - (n % batch_size) if drop_last else n
        for start in range(0, end, batch_size):
            yield self.get_batch(order[start:start + batch_size])


def build_datasets(rec_dir=_REC_DIR, crop="full", holdout_frac=DEFAULT_HOLDOUT_FRAC,
                   manual_holdout=None):
    """Construct (train_ds, holdout_ds) with the leak-free whole-session split.

    The entry point trainers should call. The two datasets are built from
    DISJOINT session lists (split_sessions), so held-out frames are physically
    absent from the training dataset — enforced structurally, not by a flag
    someone can forget. Crashed/empty sessions are excluded upstream (D-022).

    Memory note: SessionDataset caches decompressed frames per session on access.
    With the v3 radar array each cached session is a bit larger (+128x128x3/frame,
    D-024). Fine at the current scale; for a much larger corpus switch the cache
    to an LRU or stream chunks — flagged here so it's a conscious change.
    """
    train_paths, holdout_paths = split_sessions(rec_dir, holdout_frac, manual_holdout)
    train_ds = SessionDataset(train_paths, crop=crop)
    holdout_ds = SessionDataset(holdout_paths, crop=crop)
    return train_ds, holdout_ds


# ─────────────────────────────────────────────────────────────────────────────
# CLI: report the split and sanity-check a batch (no training here)
# ─────────────────────────────────────────────────────────────────────────────

def _summarise(argv=None):
    p = argparse.ArgumentParser(
        description="Report the train/held-out split and sanity-check one batch.")
    p.add_argument("--crop", default="full", choices=["full", "centre", "radar"],
                   help="which input to serve: full/centre FPV crop, or the "
                        "stored high-res radar (v3 only). Default full.")
    p.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC,
                   help=f"held-out fraction for the hash split (default {DEFAULT_HOLDOUT_FRAC})")
    p.add_argument("--batch", type=int, default=8, help="sanity-check batch size")
    args = p.parse_args(argv)

    sessions = discover_sessions(report=True)
    if not sessions:
        print(f"No usable sessions found in {_REC_DIR}. Record one with "
              f"`python -m src.recorder --record`.")
        return

    train_paths, holdout_paths = split_sessions(holdout_frac=args.holdout_frac)
    print(f"Found {len(sessions)} usable session(s) in {_REC_DIR}.")
    print(f"Split (whole-session, deterministic hash, holdout_frac="
          f"{args.holdout_frac}):\n")
    print(f"  TRAIN ({len(train_paths)}):")
    for pth in train_paths:
        print(f"    {session_name(pth):<28} bucket={_session_bucket(session_name(pth))}")
    print(f"  HELD-OUT ({len(holdout_paths)}):")
    for pth in holdout_paths:
        print(f"    {session_name(pth):<28} bucket={_session_bucket(session_name(pth))}")
    print()

    if not holdout_paths:
        print("NOTE: held-out is empty at this session count — the hash put every "
              "usable session in train. Expected with very few sessions (D-021); "
              "resolves as you record more, or pass manual_holdout= in code.")
    if not train_paths:
        print("No training sessions after split — record more, or pass manual_holdout=.")
        return

    try:
        train_ds, holdout_ds = build_datasets(crop=args.crop, holdout_frac=args.holdout_frac)
        print(f"Input '{args.crop}': train {len(train_ds)} frames across "
              f"{train_ds.n_sessions} session(s); held-out {len(holdout_ds)} frames "
              f"across {holdout_ds.n_sessions} session(s).")
        # Prefer a TRAIN batch; if TRAIN can't serve this input (e.g. radar but
        # the train sessions are v2), fall back to HELD-OUT so the CLI still
        # demonstrates the input where data supports it.
        demo_ds, demo_side = (train_ds, "train") if len(train_ds) else (holdout_ds, "held-out")
        X, Y = demo_ds.get_batch(list(range(min(args.batch, len(demo_ds)))))
        print(f"\nOne {demo_side} batch:")
        print(f"  X ({'radar' if args.crop=='radar' else 'frames'}): shape {X.shape}, "
              f"dtype {X.dtype} (BGR)")
        print(f"  Y (actions): shape {Y.shape}, dtype {Y.dtype}")
        print(f"  action columns: {action_layout(demo_ds._arrays(0))}")
        print("\nLoader OK. (This CLI does no training — it verifies shapes + split.)")
    except (ValueError, AssertionError) as e:
        # Most likely: input='radar' on v1/v2 sessions (no radar array). This is
        # EXPECTED and correct for FPV-only data — it is not a loader failure.
        print(f"\nCould not serve input '{args.crop}': {e}")


if __name__ == "__main__":
    _summarise()
