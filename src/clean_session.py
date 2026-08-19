"""clean_session.py — dataset-hygiene pass: mark blank/no-radar frames (Issue #5-adjacent).

WHAT THIS IS. Recording runs continuously through a match (recorder.py --record,
D-018) — which is correct: you should NOT restart CS2 every round. The cost is
that a session contains frames where there is no useful gameplay on the radar:
the buy menu / settings menu open, halftime, the death/spectate screen. Those
frames have a BLANK (near-uniform) radar crop, and their WASD/aim actions are
decoupled from map position. They are contamination for anything that learns
navigation from the radar (#7) and clutter for review.

This tool finds those frames and writes a per-frame KEEP-MASK next to each
session, WITHOUT deleting or rewriting any recorded data. The loader can then
optionally skip masked-out frames when building batches (opt-in; see
data_loader.build_datasets(use_keep_mask=...)). Nothing about the on-disk
recording changes; the mask is a sidecar you can regenerate or ignore.

WHY A SIDECAR MASK, NOT A FILTERED COPY (decision, see DECISIONS.md D-026):
  * Non-destructive: recordings are expensive (D-018 crash-safety exists so we
    don't lose them); a cleaner that deletes frames works against that.
  * Reversible + re-runnable: "blank" is a heuristic with a threshold. If the cut
    is ever slightly off, or the definition of junk changes, regenerate the mask
    in seconds — no re-recording, no discarded data.
  * One definition of "blank": the mask is derived from radar_probe's
    _radar_variance + _gameplay_threshold — the SAME functions the probe uses —
    so training, the probe, and the review tool agree on what a junk frame is,
    instead of two detectors that drift apart.
  * Cheap: one small boolean array + a JSON report per session, vs storing the
    whole ~9 GB/hour dataset twice (DATA_FORMAT.md storage budget).

WHAT THE MASK CANNOT CATCH (read this — it is why review_session.py exists):
The cut is on RADAR variance, so it only catches frames where the RADAR is blank.
A frame can have a perfectly normal radar and still not be useful play — e.g. the
buy menu overlays the FPV while the minimap is still drawn behind it, so a
buy-menu frame can have high radar variance and pass as "gameplay." "Blank radar"
and "not useful" are not the same set. This tool is a first-pass mask on the
signal we can measure cheaply; use `python -m src.review_session` to SEE what it
keeps and whether a class of junk (menu-open-but-radar-present) slips through. If
it does, that needs a different signal (an FPV-side menu detector), tracked
separately — not variance.

THRESHOLD IS DERIVED FROM DATA, ONCE, ACROSS ALL SESSIONS (measure-first rule):
The blank/present variance cut is found by Otsu on the pooled variance histogram
of ALL discovered sessions (like --gameplay-report), then applied to every
session. Deriving it once — not per session — keeps the cut consistent
file-to-file (a session that happens to be ALL gameplay must not get a cut fitted
to its own single mode). The tool refuses to trust a cut the data doesn't support
(Otsu's bimodality flag): if the pooled distribution isn't clearly bimodal it
warns and, unless --force, does not write masks.

Usage:
  python -m src.clean_session --report        # pooled variance + cut, write nothing
  python -m src.clean_session --all           # write keep_mask.npz for every session
  python -m src.clean_session --session NAME   # just one session (folder name or path)
  python -m src.clean_session --all --force    # write even if not clearly bimodal
  python -m src.clean_session --all --dump-blank 8   # also PNG-dump a few dropped frames

The mask file (per session folder):
  keep_mask.npz   arrays:
      keep            (N,) bool     True = gameplay (keep), False = blank/no-radar
      variance        (N,) float32  per-frame radar variance (so the cut can be
                                    re-applied at a different threshold without
                                    recomputing)
      threshold       () float32    the variance cut used
      schema_version  () int        mask format version (1)
      source_frames   () int        N the mask was built for (guards staleness)
  clean_report.json  human-readable summary (counts, cut, bimodality, per-chunk).

The mask is index-aligned to the session's concatenated per-frame arrays
(DATA_FORMAT.md order), exactly like every other per-frame array — row i of `keep`
corresponds to frame i of `frames`/`radar`/`keys`/…
"""

import argparse
import glob
import json
import os

import numpy as np

from src import data_loader as dl
from src import radar_probe as rp


# Mask sidecar filenames, written into each session folder.
MASK_FILE = "keep_mask.npz"
REPORT_FILE = "clean_report.json"

# Mask format version (independent of the recording schema_version).
MASK_SCHEMA = 1

# Batch size for streaming variance over a session (mirrors the probe/report).
_BATCH = 4096


def _iter_session_variance(ds):
    """Yield per-frame radar variance for a SessionDataset, in global-index order.

    Streams in batches so a big session never has all its radar frames in memory
    at once. Uses radar_probe._radar_variance on the stored radar (crop='radar'),
    the same measure --gameplay-report and --probe --filter-blank use.
    """
    n = len(ds)
    for start in range(0, n, _BATCH):
        gi = list(range(start, min(start + _BATCH, n)))
        Xb, _ = ds.get_batch(gi)                 # (b, RADAR_H, RADAR_W, 3) BGR
        yield start, rp._radar_variance(Xb)      # (b,)


def _pooled_variance_and_cut(sessions):
    """Compute pooled radar variance across all sessions and the Otsu cut.

    Returns (per_session_variance, threshold, diag) where per_session_variance is
    {path: np.ndarray(N,)}. The threshold is derived ONCE from the pooled
    histogram (radar_probe._gameplay_threshold), so it is identical for every
    session — see the module docstring on why per-session cuts are wrong.
    """
    per_session = {}
    pooled = []
    for path in sessions:
        ds = dl.SessionDataset([path], crop="radar")
        parts = []
        for _start, v in _iter_session_variance(ds):
            parts.append(v)
        var = (np.concatenate(parts) if parts
               else np.empty(0, dtype=np.float64))
        per_session[path] = var
        if var.size:
            pooled.append(var)
    if not pooled:
        return per_session, 0.0, {"reason": "no frames", "bimodal": False}
    allv = np.concatenate(pooled)
    thr, diag = rp._gameplay_threshold(allv)
    diag = dict(diag)
    diag["pooled_frames"] = int(allv.size)
    diag["n_blank_pooled"] = int((allv <= thr).sum())
    return per_session, float(thr), diag


def _chunk_lengths(path):
    """Per-chunk frame counts for a session folder (for the report), or [N] for v1."""
    if os.path.isdir(path):
        with open(os.path.join(path, "manifest.json")) as f:
            m = json.load(f)
        lengths = []
        for cname in m.get("chunks", []):
            with np.load(os.path.join(path, cname), allow_pickle=False) as d:
                lengths.append(int(d["frames"].shape[0]))
        return lengths
    with np.load(path, allow_pickle=False) as d:
        return [int(d["frames"].shape[0])]


def _mask_dir(path):
    """Where the sidecar files go for a session.

    For a v2/v3 folder: inside the folder. For a legacy v1 .npz file: a sibling
    folder named "<stem>.mask/" (a bare file has nowhere to put a sidecar).
    """
    if os.path.isdir(path):
        return path
    stem = path[:-4] if path.endswith(".npz") else path
    d = stem + ".mask"
    os.makedirs(d, exist_ok=True)
    return d


def write_mask_for_session(path, variance, threshold, dump_blank=0):
    """Write keep_mask.npz + clean_report.json for one session. Returns a summary dict.

    keep = variance > threshold  (present-radar frames are kept). The variance
    array is stored alongside so the cut can be re-thresholded later without
    recomputing. Nothing in the session's chunks is modified.
    """
    n = int(variance.shape[0])
    keep = (variance > threshold)
    n_keep = int(keep.sum())
    n_drop = n - n_keep

    out_dir = _mask_dir(path)
    mask_path = os.path.join(out_dir, MASK_FILE)
    tmp = mask_path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        keep=keep.astype(bool),
        variance=variance.astype(np.float32),
        threshold=np.array(threshold, dtype=np.float32),
        schema_version=np.array(MASK_SCHEMA),
        source_frames=np.array(n),
    )
    os.replace(tmp, mask_path)

    report = {
        "session": dl.session_name(path),
        "mask_schema": MASK_SCHEMA,
        "total_frames": n,
        "kept": n_keep,
        "dropped": n_drop,
        "dropped_frac": (n_drop / n) if n else 0.0,
        "threshold_variance": threshold,
        "chunk_lengths": _chunk_lengths(path),
    }
    rpath = os.path.join(out_dir, REPORT_FILE)
    rtmp = rpath + ".tmp"
    with open(rtmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(rtmp, rpath)

    if dump_blank > 0 and n_drop > 0:
        _dump_blank_frames(path, keep, dump_blank)

    return report


def _dump_blank_frames(path, keep, count):
    """Save a few of the DROPPED (blank-radar) frames as PNGs, to eyeball the cut."""
    try:
        import cv2
    except ImportError:
        print("  (cv2 unavailable — skipping --dump-blank)")
        return
    dump_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "capture_debug")
    os.makedirs(dump_dir, exist_ok=True)
    ds = dl.SessionDataset([path], crop="radar")
    drop_idx = np.where(~keep)[0]
    pick = drop_idx[np.linspace(0, len(drop_idx) - 1,
                                min(count, len(drop_idx))).astype(int)]
    X, _ = ds.get_batch(list(pick))
    stub = dl.session_name(path)
    print(f"  dumping {len(pick)} dropped(blank) radar frames to {dump_dir}:")
    for k, j in enumerate(pick):
        big = cv2.resize(X[k], (X[k].shape[1] * 3, X[k].shape[0] * 3),
                         interpolation=cv2.INTER_NEAREST)
        out = os.path.join(dump_dir, f"{stub}_blank_{int(j):06d}.png")
        cv2.imwrite(out, big)
        print(f"    frame {int(j):>6}: {out}")


def report_only(sessions):
    """Print the pooled variance distribution + the cut, writing nothing.

    A thin wrapper that reuses the probe's own reporting for consistency. This is
    the same picture as `python -m src.radar_probe --gameplay-report`, shown here
    so `clean_session --report` is self-contained.
    """
    per_session, thr, diag = _pooled_variance_and_cut(sessions)
    allv = np.concatenate([v for v in per_session.values() if v.size]) \
        if any(v.size for v in per_session.values()) else np.empty(0)
    if allv.size == 0:
        print("No frames across the discovered sessions.")
        return thr, diag
    n_blank = int((allv <= thr).sum())
    print(f"Pooled radar variance over {allv.size} frames, "
          f"{len(sessions)} session(s):")
    print(f"  min {allv.min():.1f}  median {np.median(allv):.1f}  max {allv.max():.1f}")
    lv = np.log10(allv + 1.0)
    hist, edges = np.histogram(lv, bins=12)
    print("\nlog10(variance+1) distribution:")
    for i in range(len(hist)):
        bar = "#" * int(40 * hist[i] / max(hist.max(), 1))
        print(f"  {edges[i]:5.2f}-{edges[i+1]:5.2f} | {hist[i]:6d} {bar}")
    print(f"\nChosen blank/present cut: variance = {thr:.2f} ({diag['reason']}).")
    print(f"  Would mark {n_blank} frames ({100*n_blank/allv.size:.1f}%) BLANK/no-radar,")
    print(f"  keep {allv.size - n_blank} ({100*(allv.size-n_blank)/allv.size:.1f}%) as gameplay.")
    if diag.get("bimodal", False):
        print("  Distribution is bimodal (blank vs present separate cleanly).")
    else:
        print("  WARNING: distribution NOT clearly bimodal — the cut may be")
        print("  unreliable on this data. Masks will not be written without --force.")
    return thr, diag


def clean(sessions, force=False, dump_blank=0):
    """Derive the pooled cut and write a keep-mask for each session.

    Refuses to write if the pooled distribution isn't clearly bimodal (Otsu's
    separation flag) unless force=True — the 'measure first, then set the bar'
    guard: a cut on a non-bimodal distribution would drop the wrong frames.
    """
    if not sessions:
        print(f"No usable sessions in {dl._REC_DIR}. Record with "
              f"`python -m src.recorder --record`.")
        return

    per_session, thr, diag = _pooled_variance_and_cut(sessions)
    if diag.get("pooled_frames", 0) == 0:
        print("Usable sessions found but they contain 0 frames — nothing to mask.")
        return

    print(f"Pooled cut: variance = {thr:.2f} ({diag['reason']}), from "
          f"{diag['pooled_frames']} frames across {len(sessions)} session(s).")
    if not diag.get("bimodal", False):
        print("WARNING: pooled variance distribution is NOT clearly bimodal.")
        print("  A variance cut may not cleanly separate blank from present here.")
        print("  Inspect with `python -m src.clean_session --report` (or the probe's")
        print("  --gameplay-report). Re-run with --force to write masks anyway.")
        if not force:
            print("Refusing to write masks (no --force). Nothing changed.")
            return

    print(f"\nWriting keep-masks (sidecar; recordings untouched):")
    total_n = total_keep = total_drop = 0
    for path in sessions:
        var = per_session[path]
        if var.size == 0:
            print(f"  {dl.session_name(path):<28} 0 frames — skipped")
            continue
        rep = write_mask_for_session(path, var, thr, dump_blank=dump_blank)
        total_n += rep["total_frames"]
        total_keep += rep["kept"]
        total_drop += rep["dropped"]
        print(f"  {rep['session']:<28} keep {rep['kept']:>6} / drop "
              f"{rep['dropped']:>5} ({rep['dropped_frac']*100:4.1f}%)  -> {MASK_FILE}")
    print(f"\nDone. {total_keep} kept / {total_drop} dropped of {total_n} frames "
          f"({100*total_drop/max(total_n,1):.1f}% marked blank).")
    print("Recordings are unchanged — only sidecar masks were written.")
    print("SEE what was kept/dropped before trusting it for training:")
    print("  python -m src.review_session            # scrub a session, junk frames tinted")
    print("Then, to train on kept frames only, build the loader with the mask:")
    print("  build_datasets(..., use_keep_mask=True)   # opt-in; default stays off")


def _resolve_sessions(session_arg):
    """Return the list of session paths to operate on (all, or one named)."""
    if session_arg is None:
        return dl.discover_sessions(report=True)
    # A specific session: accept a folder path, a bare name, or a v1 .npz path.
    if os.path.isdir(session_arg) or os.path.isfile(session_arg):
        return [session_arg]
    cand = os.path.join(dl._REC_DIR, os.path.basename(session_arg))
    if os.path.isdir(cand):
        return [cand]
    cand_npz = cand if cand.endswith(".npz") else cand + ".npz"
    if os.path.isfile(cand_npz):
        return [cand_npz]
    raise FileNotFoundError(f"Could not find session: {session_arg}")


def _build_parser():
    p = argparse.ArgumentParser(
        description="Dataset-hygiene pass: mark blank/no-radar frames with a "
                    "sidecar keep-mask (non-destructive). See DATA_FORMAT.md / D-026.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--report", action="store_true",
                   help="show pooled radar-variance distribution + the cut; write nothing")
    g.add_argument("--all", action="store_true",
                   help="write a keep-mask for every usable session")
    g.add_argument("--session", type=str, metavar="NAME",
                   help="write a keep-mask for one session (folder name or path)")
    p.add_argument("--force", action="store_true",
                   help="write masks even if the variance distribution isn't clearly bimodal")
    p.add_argument("--dump-blank", type=int, default=0, metavar="N",
                   help="also save N dropped(blank) radar frames per session as PNG")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.report:
        sessions = dl.discover_sessions(report=True)
        report_only(sessions)
    elif args.all:
        sessions = dl.discover_sessions(report=True)
        clean(sessions, force=args.force, dump_blank=args.dump_blank)
    elif args.session is not None:
        sessions = _resolve_sessions(args.session)
        clean(sessions, force=args.force, dump_blank=args.dump_blank)
    else:
        print("Choose: --report (show the cut), --all (mask every session), or "
              "--session NAME (one session). See `python -m src.clean_session -h`.")


if __name__ == "__main__":
    main()
