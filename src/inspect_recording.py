"""inspect_recording.py — summarise and spot-check a recording.

A reusable tool for looking inside recordings produced by `recorder.py`. Reads
both formats: a **v2 session folder** (manifest.json + chunk_*.npz, D-018) and a
legacy **v1 single .npz**. Use it to confirm a session is well-formed and
actually captured gameplay before trusting it: array shapes + alignment,
key/click activity, mouse-delta ranges, real frame rate from timestamps, and
(optionally) a few frames dumped to PNG so you can SEE the images are real.

Why both numbers and images: matching array shapes prove the record is
structurally sound (frames and actions index-aligned), but only eyeballing a
frame proves the capture actually grabbed the game. This tool does both.

Usage:
  python -m src.inspect_recording                    # newest recording (folder or file)
  python -m src.inspect_recording <path-or-name>     # a specific one
  python -m src.inspect_recording --dump 6           # also save 6 frames to PNG

For v2 folders, frames/actions are concatenated across chunks in manifest order
before analysis, so the summary is over the WHOLE session.
"""

import argparse
import glob
import json
import os

import numpy as np


_REC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "recordings")
_DUMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "capture_debug")

_PER_FRAME = ["frames", "timestamps", "keys", "lclick", "rclick", "dx", "dy"]


def _resolve_target(arg):
    """Return a path to inspect: a v2 session folder OR a v1 .npz file.

    With no arg, picks the newest of either kind in data/recordings.
    """
    if arg is None:
        # Newest among: session folders (with a manifest) and bare .npz files.
        folders = [p for p in glob.glob(os.path.join(_REC_DIR, "*"))
                   if os.path.isdir(p) and os.path.isfile(os.path.join(p, "manifest.json"))]
        files = glob.glob(os.path.join(_REC_DIR, "*.npz"))
        candidates = folders + files
        if not candidates:
            raise FileNotFoundError(
                f"No recordings found in {_REC_DIR}. Record one with "
                f"`python -m src.recorder --record`.")
        return max(candidates, key=os.path.getmtime)
    # Explicit: accept a folder, a file, or a bare name for either.
    if os.path.isdir(arg) or os.path.isfile(arg):
        return arg
    in_rec = os.path.join(_REC_DIR, os.path.basename(arg))
    if os.path.isdir(in_rec):
        return in_rec
    cand = arg if arg.endswith(".npz") else arg + ".npz"
    in_rec_file = os.path.join(_REC_DIR, os.path.basename(cand))
    if os.path.isfile(in_rec_file):
        return in_rec_file
    raise FileNotFoundError(f"Could not find recording: {arg}")


def _load_npz(path):
    """Load one .npz into a plain dict of arrays."""
    out = {}
    with np.load(path, allow_pickle=False) as d:
        for k in d.files:
            out[k] = d[k]
    return out


def _load_target(target):
    """Load a v2 folder (concatenated) or v1 file into one arrays dict.

    Returns (arrays_dict, meta_dict). meta_dict carries session-level info for
    display (chunk count, completeness, on-disk size).
    """
    if os.path.isdir(target):
        manifest_path = os.path.join(target, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        chunk_names = manifest.get("chunks", [])
        if not chunk_names:
            raise ValueError(f"Session {target} has no chunks listed in manifest.")
        # Load each chunk; concatenate per-frame arrays in manifest order.
        per_frame_lists = {k: [] for k in _PER_FRAME}
        meta_arrays = {}
        chunk_lengths = []
        total_size = os.path.getsize(manifest_path)
        for cname in chunk_names:
            cpath = os.path.join(target, cname)
            total_size += os.path.getsize(cpath)
            chunk = _load_npz(cpath)
            if "frames" in chunk:
                chunk_lengths.append(int(chunk["frames"].shape[0]))
            for k in _PER_FRAME:
                if k in chunk:
                    per_frame_lists[k].append(chunk[k])
            # keep metadata from the first chunk
            for k in ("key_names", "schema_version", "geom", "loop_fps_target"):
                if k in chunk and k not in meta_arrays:
                    meta_arrays[k] = chunk[k]
        arrays = {}
        for k, parts in per_frame_lists.items():
            if parts:
                arrays[k] = np.concatenate(parts, axis=0)
        arrays.update(meta_arrays)
        meta = {
            "kind": "v2 session folder",
            "chunks": len(chunk_names),
            "complete": manifest.get("complete"),
            "size_mb": total_size / (1024 * 1024),
            "manifest_total_frames": manifest.get("total_frames"),
            "chunk_lengths": chunk_lengths,
        }
        return arrays, meta
    else:
        arrays = _load_npz(target)
        meta = {
            "kind": "v1 single file",
            "chunks": 1,
            "complete": True,
            "size_mb": os.path.getsize(target) / (1024 * 1024),
            "manifest_total_frames": None,
            "chunk_lengths": [int(arrays["frames"].shape[0])] if "frames" in arrays else [],
        }
        return arrays, meta


def _fmt_hz(timestamps):
    """Derive real FPS and gap stats from the saved perf_counter timestamps.

    Note: for a v2 session the timestamps are concatenated across chunks; the
    gap between the last frame of one chunk and the first of the next is a real
    inter-frame gap (recording was continuous), so this stays valid.
    """
    if timestamps.size < 2:
        return None
    diffs = np.diff(timestamps)
    mean_dt = diffs.mean()
    fps = 1.0 / mean_dt if mean_dt > 0 else float("nan")
    return {
        "fps": fps,
        "mean_dt_ms": mean_dt * 1000.0,
        "max_dt_ms": diffs.max() * 1000.0,
        "min_dt_ms": diffs.min() * 1000.0,
        "jitter_ms": diffs.std() * 1000.0,
    }


def inspect(target, dump=0):
    arrays, meta = _load_target(target)
    print(f"Inspecting: {target}")
    print(f"Kind: {meta['kind']}"
          + (f", {meta['chunks']} chunk(s)" if meta['chunks'] else "")
          + (f", complete={meta['complete']}" if meta['complete'] is not None else ""))
    print(f"On-disk size: {meta['size_mb']:.1f} MB")
    if meta.get("manifest_total_frames") is not None:
        print(f"Manifest total_frames: {meta['manifest_total_frames']}")
    print()

    d = arrays  # analysis below expects a dict-like of arrays
    if True:
        keys_present = list(d.keys())
        print("Arrays (name, shape, dtype):")
        for k in keys_present:
            print(f"  {k:<12} {str(d[k].shape):<22} {d[k].dtype}")
        print()

        # ── Alignment check: every per-frame array should share frame count ──
        frames = d["frames"] if "frames" in d else None
        n = frames.shape[0] if frames is not None else None
        per_frame = ["frames", "timestamps", "keys", "lclick", "rclick", "dx", "dy"]
        counts = {k: d[k].shape[0] for k in per_frame if k in d}
        aligned = len(set(counts.values())) == 1
        print("Alignment (per-frame arrays should all have the same length):")
        for k, c in counts.items():
            print(f"  {k:<12} {c}")
        print(f"  -> {'ALIGNED — frames and actions match row-for-row.' if aligned else 'MISALIGNED — lengths differ! This recording is suspect.'}\n")

        # ── Frame sanity ──
        if frames is not None:
            print(f"Frames: {n} total, shape per frame "
                  f"{frames.shape[1:]} (H, W, C), dtype {frames.dtype}")
            # Cheap content check: a stuck/black capture shows near-zero variance.
            sample_idx = np.linspace(0, n - 1, min(n, 50)).astype(int)
            sample = frames[sample_idx].astype(np.float32)
            print(f"  pixel value range over a 50-frame sample: "
                  f"min={sample.min():.0f} max={sample.max():.0f} "
                  f"mean={sample.mean():.1f}")
            if sample.max() - sample.min() < 5:
                print("  WARNING: almost no pixel variation — capture may have been "
                      "black/stuck. Dump frames (--dump) and look.")
            print()

        # ── Timing ──
        if "timestamps" in d:
            hz = _fmt_hz(d["timestamps"])
            if hz:
                print("Timing (from saved capture timestamps):")
                print(f"  real rate: {hz['fps']:.1f} FPS "
                      f"(mean {hz['mean_dt_ms']:.1f} ms/frame, "
                      f"jitter {hz['jitter_ms']:.1f} ms)")
                print(f"  slowest frame gap: {hz['max_dt_ms']:.1f} ms; "
                      f"fastest: {hz['min_dt_ms']:.1f} ms")
                print()

                # ── Stall localisation: are the big gaps at chunk boundaries? ──
                # This is the diagnostic for the #4 flush-stall question. If the
                # synchronous chunk flush is freezing the capture loop, the huge
                # inter-frame gaps will land exactly at the chunk boundary frames
                # (1800, 3600, ...). If the big gaps are elsewhere, the stall is
                # something else (game/system hitch), handled differently.
                ts = d["timestamps"]
                diffs = np.diff(ts) * 1000.0  # ms gaps, index i = gap AFTER frame i
                lengths = meta.get("chunk_lengths") or []
                # boundary frame indices = cumulative chunk lengths (except the last)
                boundaries = list(np.cumsum(lengths))[:-1] if len(lengths) > 1 else []
                budget_ms = 1000.0 / (int(d["loop_fps_target"]) if "loop_fps_target" in d else LOOP_FPS)
                stall_ms = 3.0 * budget_ms  # "stall" = gap > 3x the frame budget

                # Top few gaps overall, with where they sit.
                order = np.argsort(diffs)[::-1][:5]
                print("  Largest inter-frame gaps (gap AFTER the listed frame index):")
                for i in order:
                    i = int(i)
                    at_boundary = i + 1 in boundaries  # gap after frame i bridges to i+1
                    tag = "  <-- CHUNK BOUNDARY" if at_boundary else ""
                    print(f"    frame {i:>6}: {diffs[i]:8.1f} ms{tag}")

                if boundaries:
                    # Gap at each chunk boundary specifically.
                    print("  Gap at each chunk boundary:")
                    boundary_gaps = []
                    for b in boundaries:
                        gi = b - 1  # gap AFTER the last frame of the chunk
                        if 0 <= gi < len(diffs):
                            boundary_gaps.append(diffs[gi])
                            print(f"    boundary at frame {b:>6}: {diffs[gi]:8.1f} ms")
                    # Verdict.
                    big = [g for g in diffs if g > stall_ms]
                    big_at_boundary = sum(1 for b in boundaries
                                          if 0 <= b - 1 < len(diffs) and diffs[b - 1] > stall_ms)
                    print()
                    if boundary_gaps and max(boundary_gaps) > stall_ms:
                        frac = big_at_boundary / max(len(big), 1)
                        print(f"  DIAGNOSIS: chunk-boundary gaps exceed {stall_ms:.0f} ms "
                              f"(3x budget). {big_at_boundary} of {len(boundaries)} "
                              f"boundaries stall; {frac*100:.0f}% of all stalls sit at "
                              f"boundaries.")
                        print("  => Consistent with the SYNCHRONOUS FLUSH blocking the "
                              "loop. Fix: move the chunk write off the capture loop.")
                    else:
                        print("  DIAGNOSIS: chunk-boundary gaps are NOT the large ones — "
                              "the stall is elsewhere (game/system hitch), not the flush.")
                else:
                    print("  (single chunk / no interior boundaries — no boundary test.)")
                print()

        # ── Action activity ──
        if "keys" in d:
            keys = d["keys"]
            names = ([s.decode() if isinstance(s, bytes) else str(s)
                      for s in d["key_names"]] if "key_names" in d
                     else [str(i) for i in range(keys.shape[1])])
            held_counts = keys.sum(axis=0)
            print("Key activity (frames each key was held):")
            any_key = False
            for name, c in zip(names, held_counts):
                if c > 0:
                    pct = 100.0 * c / keys.shape[0]
                    print(f"  {name:<6} {int(c):>6}  ({pct:.1f}% of frames)")
                    any_key = True
            if not any_key:
                print("  (no keys held in this session)")
            print()

        if "lclick" in d and "rclick" in d:
            lc = int(d["lclick"].sum())
            rc = int(d["rclick"].sum())
            tot = d["lclick"].shape[0]
            print(f"Clicks: left held {lc} frames ({100*lc/tot:.1f}%), "
                  f"right held {rc} frames ({100*rc/tot:.1f}%)\n")

        if "dx" in d and "dy" in d:
            dx, dy = d["dx"], d["dy"]
            moved = int(np.count_nonzero(dx) + np.count_nonzero(dy))
            print("Mouse deltas:")
            print(f"  dx range [{int(dx.min())}, {int(dx.max())}], "
                  f"mean |dx| {np.abs(dx).mean():.1f}")
            print(f"  dy range [{int(dy.min())}, {int(dy.max())}], "
                  f"mean |dy| {np.abs(dy).mean():.1f}")
            nz_dx = int(np.count_nonzero(dx))
            print(f"  frames with dx!=0: {nz_dx} "
                  f"({100*nz_dx/dx.shape[0]:.1f}%)")
            if dx.max() == 0 and dx.min() == 0:
                print("  WARNING: all dx are zero — mouse aim was not captured.")
            print()

        # ── Optional: dump frames to PNG for visual inspection ──
        if dump > 0 and frames is not None:
            try:
                import cv2
            except ImportError:
                print("Cannot dump frames: cv2 not available.")
                return
            os.makedirs(_DUMP_DIR, exist_ok=True)
            idx = np.linspace(0, n - 1, min(dump, n)).astype(int)
            stub = os.path.basename(str(target).rstrip(os.sep))
            stub = os.path.splitext(stub)[0]  # drop .npz if a v1 file
            print(f"Dumping {len(idx)} frames to {_DUMP_DIR}:")
            for j in idx:
                out = os.path.join(_DUMP_DIR, f"{stub}_frame_{int(j):05d}.png")
                cv2.imwrite(out, frames[j])
                print(f"  {out}")
            print("\nOpen those PNGs — they should show real gameplay at 270x150. "
                  "If they look right, the recording genuinely captured the game.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Summarise and spot-check a recording (v2 folder or v1 .npz).")
    p.add_argument("recording", nargs="?", default=None,
                   help="path or name of a session folder / .npz (default: newest)")
    p.add_argument("--dump", type=int, default=0, metavar="N",
                   help="also save N evenly-spaced frames as PNG for visual inspection")
    args = p.parse_args(argv)
    target = _resolve_target(args.recording)
    inspect(target, dump=args.dump)


if __name__ == "__main__":
    main()
