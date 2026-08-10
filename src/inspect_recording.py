"""inspect_recording.py — summarise and spot-check a recorded .npz session.

A reusable tool for looking inside recordings produced by `recorder.py --record`.
Use it to confirm a session is well-formed and actually captured gameplay before
trusting it: array shapes + alignment, key/click activity, mouse-delta ranges,
the real frame rate derived from timestamps, and (optionally) a few frames dumped
to PNG so you can SEE the images are real, not garbage.

Why both numbers and images: matching array shapes prove the record is
structurally sound (frames and actions index-aligned), but only eyeballing a
frame proves the capture actually grabbed the game. This tool does both.

Usage:
  python -m src.inspect_recording                         # newest recording
  python -m src.inspect_recording <path-or-name>          # a specific one
  python -m src.inspect_recording --dump 6                # also save 6 frames to PNG
  python -m src.inspect_recording <name> --dump 6

This is an inspection/QA tool, not part of the record or train path. It also
reports the real per-frame timing (from saved timestamps), which is relevant to
the open question of the recording loop's FPS.
"""

import argparse
import glob
import os

import numpy as np


_REC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "recordings")
_DUMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "capture_debug")


def _resolve_path(arg):
    """Turn a user argument into a real .npz path, or find the newest recording."""
    if arg is None:
        candidates = sorted(glob.glob(os.path.join(_REC_DIR, "*.npz")),
                            key=os.path.getmtime)
        if not candidates:
            raise FileNotFoundError(
                f"No .npz recordings found in {_REC_DIR}. Record one first with "
                f"`python -m src.recorder --record`.")
        return candidates[-1]  # newest by mtime
    # Accept a full path, a path relative to cwd, or a bare name (with/without .npz)
    if os.path.isfile(arg):
        return arg
    cand = arg if arg.endswith(".npz") else arg + ".npz"
    in_recdir = os.path.join(_REC_DIR, os.path.basename(cand))
    if os.path.isfile(in_recdir):
        return in_recdir
    raise FileNotFoundError(f"Could not find recording: {arg}")


def _fmt_hz(timestamps):
    """Derive real FPS and gap stats from the saved perf_counter timestamps."""
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


def inspect(path, dump=0):
    print(f"Inspecting: {path}")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB\n")

    with np.load(path, allow_pickle=False) as d:
        keys_present = list(d.files)
        print("Arrays in file (name, shape, dtype):")
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
            stub = os.path.splitext(os.path.basename(path))[0]
            print(f"Dumping {len(idx)} frames to {_DUMP_DIR}:")
            for j in idx:
                out = os.path.join(_DUMP_DIR, f"{stub}_frame_{int(j):05d}.png")
                cv2.imwrite(out, frames[j])
                print(f"  {out}")
            print("\nOpen those PNGs — they should show real gameplay at 270x150. "
                  "If they look right, the recording genuinely captured the game.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Summarise and spot-check a recorded .npz session.")
    p.add_argument("recording", nargs="?", default=None,
                   help="path or name of the .npz (default: newest in data/recordings)")
    p.add_argument("--dump", type=int, default=0, metavar="N",
                   help="also save N evenly-spaced frames as PNG for visual inspection")
    args = p.parse_args(argv)
    path = _resolve_path(args.recording)
    inspect(path, dump=args.dump)


if __name__ == "__main__":
    main()
