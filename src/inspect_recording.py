"""inspect_recording.py — summarise and spot-check a recording.

A reusable tool for looking inside recordings produced by `recorder.py`. Reads
all formats: a **v4 session folder** (v3 + per-frame GSI `alive`/`round_phase`,
D-031), a **v3 session folder** (manifest.json + chunk_*.npz WITH a per-frame
`radar` array, D-024), a **v2 session folder** (FPV-only chunks, D-018), and a
legacy **v1 single .npz**. Use it to confirm a session is well-formed and
actually captured gameplay before trusting it: array shapes + alignment (incl.
the radar array on v3+), the GSI alive/round_phase summary on v4, key/click
activity, mouse-delta ranges, real frame rate from timestamps, and (optionally)
a few frames dumped to PNG so you can SEE the images are real.

Note: `alive`/`round_phase` (v4) are NOT added to the alignment set below on
purpose — the alignment check runs over `_PER_FRAME` and older tools/sessions
lack them; they are summarised separately and their length is implicitly checked
by the loader, which is authoritative.

Why both numbers and images: matching array shapes prove the record is
structurally sound (frames and actions index-aligned), but only eyeballing a
frame proves the capture actually grabbed the game. This tool does both.

Usage:
  python -m src.inspect_recording                    # newest recording (folder or file)
  python -m src.inspect_recording <path-or-name>     # a specific one
  python -m src.inspect_recording --dump 6           # also save 6 frames to PNG
  python -m src.inspect_recording --dump-radar 6     # also save 6 radar crops to PNG (v3)

For v2/v3 folders, frames/actions are concatenated across chunks in manifest
order before analysis, so the summary is over the WHOLE session.

The reported "kind" is derived from the session's actual `schema_version`
(1 -> v1 single file, 2 -> v2 FPV-only folder, 3 -> v3 FPV+radar folder), NOT
hardcoded — an earlier version hardcoded "v2 session folder" for any folder,
which misreported v3 sessions as v2 and hid the radar array. Fixed.
"""

import argparse
import glob
import json
import os

import numpy as np


_REC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "recordings")
_DUMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "capture_debug")

# Per-frame arrays that must stay index-aligned. `radar` is present only on v3
# (D-024); it's included here so it is displayed AND alignment-checked when present.
_PER_FRAME = ["frames", "radar", "timestamps", "keys", "lclick", "rclick", "dx", "dy"]

# Fallback frame budget if a session somehow lacks loop_fps_target.
_DEFAULT_LOOP_FPS = 15


def _resolve_target(arg):
    """Return a path to inspect: a v2/v3 session folder OR a v1 .npz file.

    With no arg, picks the newest of either kind in data/recordings.
    """
    if arg is None:
        folders = [p for p in glob.glob(os.path.join(_REC_DIR, "*"))
                   if os.path.isdir(p) and os.path.isfile(os.path.join(p, "manifest.json"))]
        files = glob.glob(os.path.join(_REC_DIR, "*.npz"))
        candidates = folders + files
        if not candidates:
            raise FileNotFoundError(
                f"No recordings found in {_REC_DIR}. Record one with "
                f"`python -m src.recorder --record`.")
        return max(candidates, key=os.path.getmtime)
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


def _kind_from_schema(schema_version, is_folder):
    """Human label derived from the actual schema_version (not hardcoded).

    1 -> v1 single file, 2 -> v2 FPV-only folder, 3 -> v3 FPV+radar folder.
    Unknown versions are reported explicitly rather than silently mislabelled.
    """
    if schema_version is None:
        return "unknown-schema folder" if is_folder else "unknown-schema file"
    v = int(schema_version)
    if v == 1:
        return "v1 single file"
    if v == 2:
        return "v2 session folder (FPV only)"
    if v == 3:
        return "v3 session folder (FPV + radar)"
    if v == 4:
        return "v4 session folder (FPV + radar + GSI alive)"
    if v == 5:
        return "v5 session folder (FPV + radar + GSI alive + state features)"
    return f"schema-v{v} {'folder' if is_folder else 'file'} (newer than this tool knows)"


def _load_target(target):
    """Load a v2/v3 folder (concatenated) or v1 file into one arrays dict.

    Returns (arrays_dict, meta_dict). meta_dict carries session-level info for
    display (chunk count, completeness, on-disk size, and the schema-derived kind).
    """
    if os.path.isdir(target):
        manifest_path = os.path.join(target, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        chunk_names = manifest.get("chunks", [])
        if not chunk_names:
            raise ValueError(f"Session {target} has no chunks listed in manifest.")
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
            for k in ("key_names", "schema_version", "geom", "loop_fps_target"):
                if k in chunk and k not in meta_arrays:
                    meta_arrays[k] = chunk[k]
        arrays = {}
        for k, parts in per_frame_lists.items():
            if parts:
                arrays[k] = np.concatenate(parts, axis=0)
        arrays.update(meta_arrays)
        # Prefer the manifest's schema_version, fall back to a chunk's.
        schema = manifest.get("schema_version")
        if schema is None and "schema_version" in meta_arrays:
            schema = int(meta_arrays["schema_version"])
        meta = {
            "kind": _kind_from_schema(schema, is_folder=True),
            "schema_version": schema,
            "chunks": len(chunk_names),
            "complete": manifest.get("complete"),
            "size_mb": total_size / (1024 * 1024),
            "manifest_total_frames": manifest.get("total_frames"),
            "chunk_lengths": chunk_lengths,
        }
        return arrays, meta
    else:
        arrays = _load_npz(target)
        schema = int(arrays["schema_version"]) if "schema_version" in arrays else None
        meta = {
            "kind": _kind_from_schema(schema, is_folder=False),
            "schema_version": schema,
            "chunks": 1,
            "complete": True,
            "size_mb": os.path.getsize(target) / (1024 * 1024),
            "manifest_total_frames": None,
            "chunk_lengths": [int(arrays["frames"].shape[0])] if "frames" in arrays else [],
        }
        return arrays, meta


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


def inspect(target, dump=0, dump_radar=0):
    arrays, meta = _load_target(target)
    print(f"Inspecting: {target}")
    print(f"Kind: {meta['kind']}"
          + (f", {meta['chunks']} chunk(s)" if meta['chunks'] else "")
          + (f", complete={meta['complete']}" if meta['complete'] is not None else ""))
    print(f"On-disk size: {meta['size_mb']:.1f} MB")
    if meta.get("manifest_total_frames") is not None:
        print(f"Manifest total_frames: {meta['manifest_total_frames']}")
    print()

    d = arrays
    keys_present = list(d.keys())
    print("Arrays (name, shape, dtype):")
    for k in keys_present:
        print(f"  {k:<15} {str(d[k].shape):<22} {d[k].dtype}")
    print()

    # ── Alignment check: every per-frame array should share frame count ──
    # Now includes `radar` when present (v3), so a mis-sized radar array is caught.
    frames = d["frames"] if "frames" in d else None
    n = frames.shape[0] if frames is not None else None
    counts = {k: d[k].shape[0] for k in _PER_FRAME if k in d}
    aligned = len(set(counts.values())) == 1
    print("Alignment (per-frame arrays should all have the same length):")
    for k, c in counts.items():
        print(f"  {k:<15} {c}")
    print(f"  -> {'ALIGNED — frames and actions match row-for-row.' if aligned else 'MISALIGNED — lengths differ! This recording is suspect.'}\n")

    # ── Frame sanity ──
    if frames is not None:
        print(f"Frames (FPV): {n} total, shape per frame "
              f"{frames.shape[1:]} (H, W, C), dtype {frames.dtype}")
        sample_idx = np.linspace(0, n - 1, min(n, 50)).astype(int)
        sample = frames[sample_idx].astype(np.float32)
        print(f"  pixel value range over a 50-frame sample: "
              f"min={sample.min():.0f} max={sample.max():.0f} "
              f"mean={sample.mean():.1f}")
        if sample.max() - sample.min() < 5:
            print("  WARNING: almost no pixel variation — capture may have been "
                  "black/stuck. Dump frames (--dump) and look.")
        print()

    # ── GSI alive / round_phase sanity (v4, D-031) ──
    if "alive" in d:
        alive = d["alive"].astype(bool)
        n_alive = int(alive.sum())
        tot = alive.shape[0]
        print(f"GSI alive (v4, D-031): {n_alive}/{tot} frames alive "
              f"({100*n_alive/max(tot,1):.1f}%), {tot - n_alive} dead/spectating/menu.")
        # Count alive/dead transitions as a coarse sanity signal (a real session
        # should show at least one death if you died; all-alive or all-dead is a
        # flag worth an eyeball).
        if tot > 1:
            flips = int(np.count_nonzero(np.diff(alive.astype(np.int8))))
            print(f"  alive<->dead transitions: {flips}")
        if n_alive == 0:
            print("  WARNING: NO alive frames — GSI may not have tracked life, or "
                  "the whole session was dead/spectate/menu. Verify with "
                  "`--verify-gsi` before trusting the gameplay filter.")
        elif n_alive == tot:
            print("  NOTE: every frame is alive — fine if you never died, but if you "
                  "did, check the spectating guard with `--verify-gsi`.")
        if "round_phase" in d:
            rp = d["round_phase"]
            vals, counts = np.unique(rp.astype(str), return_counts=True)
            dist = ", ".join(f"{v!r}:{int(c)}" for v, c in zip(vals, counts))
            print(f"  round_phase distribution: {dist}")
        print()

    # ── GSI state features sanity (v5, D-033) ──
    if "health" in d:
        health = d["health"]
        tot = health.shape[0]
        # Health over frames the session considers gameplay (alive), if we have it,
        # else over all frames. Sentinel 0 doubles as dead; the alive flag
        # disambiguates, so summarise health on alive frames for a meaningful range.
        if "alive" in d:
            am = d["alive"].astype(bool)
            hp_live = health[am]
            scope = f"{hp_live.shape[0]} alive frames"
        else:
            hp_live = health
            scope = f"{tot} frames"
        if hp_live.size:
            print(f"GSI state features (v5, D-033):")
            print(f"  health over {scope}: min={int(hp_live.min())} "
                  f"max={int(hp_live.max())} mean={hp_live.mean():.1f}")
        else:
            print(f"GSI state features (v5, D-033): no alive frames to summarise "
                  f"health over.")
        if "active_weapon" in d:
            wv, wc = np.unique(d["active_weapon"].astype(str), return_counts=True)
            # Show the most common few, so a full inventory doesn't flood output.
            order = np.argsort(wc)[::-1]
            top = ", ".join(f"{wv[i]!r}:{int(wc[i])}" for i in order[:8])
            print(f"  active_weapon (top): {top}")
            if (d["active_weapon"].astype(str) == "").all():
                print("  WARNING: active_weapon is empty on every frame — GSI weapon "
                      "block may not be arriving. Check the .cfg subscribes "
                      "player_weapons.")
        if "ammo_clip" in d and "ammo_reserve" in d:
            clip = d["ammo_clip"]; res = d["ammo_reserve"]
            has_ammo = clip >= 0  # sentinel -1 = no-ammo weapon (knife/C4)/unknown
            n_has = int(has_ammo.sum())
            if n_has:
                print(f"  ammo (over {n_has} frames with an ammo weapon): "
                      f"clip [{int(clip[has_ammo].min())},{int(clip[has_ammo].max())}], "
                      f"reserve [{int(res[has_ammo].min())},{int(res[has_ammo].max())}]")
            else:
                print("  ammo: no frames with an ammo-bearing weapon (all "
                      "knife/C4/none, or ammo not reported).")
        print()

    # ── Radar sanity (v3) ──
    if "radar" in d:
        radar = d["radar"]
        print(f"Radar (v3, D-024): {radar.shape[0]} total, shape per crop "
              f"{radar.shape[1:]} (H, W, C), dtype {radar.dtype}")
        r_idx = np.linspace(0, radar.shape[0] - 1, min(radar.shape[0], 50)).astype(int)
        r_sample = radar[r_idx].astype(np.float32)
        print(f"  pixel value range over a 50-frame sample: "
              f"min={r_sample.min():.0f} max={r_sample.max():.0f} "
              f"mean={r_sample.mean():.1f}")
        if r_sample.max() - r_sample.min() < 5:
            print("  WARNING: almost no pixel variation in the radar — the radar "
                  "crop may be off or black. Dump with --dump-radar and look.")
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

            ts = d["timestamps"]
            diffs = np.diff(ts) * 1000.0  # ms gaps, index i = gap AFTER frame i
            lengths = meta.get("chunk_lengths") or []
            boundaries = list(np.cumsum(lengths))[:-1] if len(lengths) > 1 else []
            loop_fps = int(d["loop_fps_target"]) if "loop_fps_target" in d else _DEFAULT_LOOP_FPS
            budget_ms = 1000.0 / loop_fps
            stall_ms = 3.0 * budget_ms

            order = np.argsort(diffs)[::-1][:5]
            print("  Largest inter-frame gaps (gap AFTER the listed frame index):")
            for i in order:
                i = int(i)
                at_boundary = i + 1 in boundaries
                tag = "  <-- CHUNK BOUNDARY" if at_boundary else ""
                print(f"    frame {i:>6}: {diffs[i]:8.1f} ms{tag}")

            if boundaries:
                print("  Gap at each chunk boundary:")
                boundary_gaps = []
                for b in boundaries:
                    gi = b - 1
                    if 0 <= gi < len(diffs):
                        boundary_gaps.append(diffs[gi])
                        print(f"    boundary at frame {b:>6}: {diffs[gi]:8.1f} ms")
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

    # ── Optional: dump FPV frames to PNG ──
    if dump > 0 and frames is not None:
        _dump_images(frames, target, dump, suffix="frame")
    # ── Optional: dump radar crops to PNG (v3) ──
    if dump_radar > 0:
        if "radar" in d:
            _dump_images(d["radar"], target, dump_radar, suffix="radar", upscale=3)
        else:
            print("--dump-radar requested but this session has no radar array "
                  "(v1/v2). Nothing to dump.")


def _dump_images(stack, target, count, suffix, upscale=1):
    """Save `count` evenly-spaced frames from `stack` (N,H,W,3) as PNGs."""
    try:
        import cv2
    except ImportError:
        print(f"Cannot dump {suffix}: cv2 not available.")
        return
    n = stack.shape[0]
    os.makedirs(_DUMP_DIR, exist_ok=True)
    idx = np.linspace(0, n - 1, min(count, n)).astype(int)
    stub = os.path.basename(str(target).rstrip(os.sep))
    stub = os.path.splitext(stub)[0]
    print(f"Dumping {len(idx)} {suffix} image(s) to {_DUMP_DIR}:")
    for j in idx:
        img = stack[j]
        if upscale > 1:
            img = cv2.resize(img, (img.shape[1] * upscale, img.shape[0] * upscale),
                             interpolation=cv2.INTER_NEAREST)
        out = os.path.join(_DUMP_DIR, f"{stub}_{suffix}_{int(j):05d}.png")
        cv2.imwrite(out, img)
        print(f"  {out}")
    print()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Summarise and spot-check a recording (v1 file, or v2/v3 folder).")
    p.add_argument("recording", nargs="?", default=None,
                   help="path or name of a session folder / .npz (default: newest)")
    p.add_argument("--dump", type=int, default=0, metavar="N",
                   help="also save N evenly-spaced FPV frames as PNG")
    p.add_argument("--dump-radar", type=int, default=0, metavar="N",
                   help="also save N evenly-spaced radar crops as PNG (v3 only)")
    args = p.parse_args(argv)
    target = _resolve_target(args.recording)
    inspect(target, dump=args.dump, dump_radar=args.dump_radar)


if __name__ == "__main__":
    main()
