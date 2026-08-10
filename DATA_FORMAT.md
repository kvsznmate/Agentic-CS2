# DATA_FORMAT.md — Agentic-CS2 on-disk recording schema

**Status: v1, locked 2026-08 (Issue #5).** This is the authoritative schema for
self-recorded session files. Every loader, trainer, and inspection tool reads
this format; change it only by bumping `schema_version` and updating this file
in the same commit (per the living-docs rule in CLAUDE.md).

Related decisions: **D-012** (capture geometry: fullscreen 1920x1080, full-frame
crop, 150x270 model input), **D-015** (action fields: keys/clicks + raw mouse
deltas), **D-016** (loop rate). This file is the schema those decisions produce.

---

## File type and naming

- One **session** = one `.npz` file (NumPy compressed archive), written by
  `src/recorder.py`.
- Location: `data/recordings/` (gitignored — recordings are large and local).
- Name: `session_YYYYMMDD_HHMMSS.npz` by default, or a caller-supplied stub.
- Written with `np.savez_compressed`; read with `np.load(path, allow_pickle=False)`.
  **`allow_pickle=False` is required** — the format uses only plain arrays
  precisely so it never needs pickle (safer, portable, loader-friendly).

---

## Arrays in each file

`N` = number of frames in the session. Every **per-frame** array has length `N`
along axis 0, and row `i` of every per-frame array corresponds to the **same
tick** — this index-alignment IS the frame/action synchronization the M0 gate
proved (#3). Do not sort or filter one array without the others.

### Per-frame arrays (length N)

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `frames` | (N, 150, 270, 3) | uint8 | The captured game image per tick. **BGR** channel order (OpenCV-native, D-012/Q4). 150 high × 270 wide × 3 channels. This is the FULL downscaled screen — the radar corner is inside it (see "Why full frame" below). |
| `timestamps` | (N,) | float64 | `time.perf_counter()` taken immediately after each frame grab — the alignment anchor. Seconds, monotonic, arbitrary origin (differences are meaningful, absolute value is not). |
| `keys` | (N, 11) | uint8 | 0/1 held-state of the 11 logged keys per tick, in the fixed order given by `key_names`. |
| `lclick` | (N,) | uint8 | 0/1 left mouse button held this tick. |
| `rclick` | (N,) | uint8 | 0/1 right mouse button held this tick. |
| `dx` | (N,) | int32 | Raw mouse movement in device units accumulated over the interval **ending** at this frame. +x = physical right. (D-015: relative deltas, not absolute view angle.) |
| `dy` | (N,) | int32 | Raw mouse movement, vertical. +y = physical down. |

### Metadata arrays (fixed-size, not length N)

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `key_names` | (11,) | str (`<U…`) | The key label for each column of `keys`, in order: `w, a, s, d, space, ctrl, shift, 1, 2, 3, r`. Always read column meaning FROM THIS ARRAY, never assume the order. |
| `schema_version` | () scalar | int | Format version. **1** for this document. A loader should check this and refuse/adapt if it sees a version it doesn't know. |
| `geom` | (…) | str | Human-readable capture geometry stamp, e.g. `"fullscreen 1920x1080 -> crop full-frame -> 150x270 BGR"`. Records the conditions the frames were captured under so a file is self-describing without external context. |
| `loop_fps_target` | () scalar | int | The loop's target FPS at record time (D-016: 15). The REAL rate is derivable from `timestamps`; this is just what was aimed for. |

---

## The action vector (how to read an action for frame i)

The "action label" for frame `i`, as consumed by training, is assembled from the
per-frame arrays at index `i`:

```
action_i = {
    "keys":   keys[i]            # 11 binary values, order per key_names
    "lclick": lclick[i]          # binary
    "rclick": rclick[i]          # binary
    "dx":     dx[i]              # int, raw mouse x since previous frame
    "dy":     dy[i]              # int, raw mouse y since previous frame
}
```

There is no separate "action vector" array on disk — it is composed from the
columns above. A future loader (#6) may concatenate these into a single vector;
the canonical source is these named arrays. Keeping them separate on disk (rather
than a pre-concatenated blob) means the layout can change without rewriting files.

---

## Why the FULL frame is stored (and no radar crop yet)

The `frames` array is the whole downscaled screen, not a pre-cropped sub-region.
Two reasons, both deliberate:

1. **The radar must stay recoverable.** The radar/minimap occupies a corner of
   the frame. Storing the full frame keeps it available for the M2 navigation
   gate (#7), which is what will decide whether the radar even carries usable
   signal. If we cropped the radar away at record time, #7 could not be run.
   (Note from on-machine inspection: at 150x270 the radar is very low-res — you
   can tell an enemy's rough region but not exact position. Whether that's
   *enough* signal is exactly the #7 question; not prejudged here.)

2. **The per-model crops belong downstream, not on disk.** D-012 deferred the
   centre-vs-radar sub-crops to the loader (#6: "configurable crops — full FPV,
   centre, radar"). The loader carves sub-regions from the stored full frame at
   training time. So this schema stores ONE full frame per tick; it does **not**
   define a radar crop rectangle. That belongs to #6/#7 once the radar is
   understood. Deliberately NOT baking a radar representation into the format
   before #7 proves it out.

---

## What is NOT in this format (and why)

- **No enemy-detection labels.** Self-recording yields actions for free but
  nothing about where enemies are on screen (D-003). Detection labels are a
  separate data problem (M3, #8–#9) with their own creation method. When they
  exist, they attach to these files **by frame index** — a parallel per-frame
  array keyed to the same `i`. The format is designed to allow that addition
  without touching existing fields (it would be a new array + a schema bump).
- **No view angles / player position / health / ammo.** The study read these
  from game memory via RAM offsets — dead on Source 2 (D-002). We do not have
  them and do not fake them. Aim is represented by raw mouse deltas only.
- **No audio.** Out of scope for the current architecture.

---

## Extending the format (the rules)

1. Bump `schema_version` and document the new version in this file, same commit.
2. Prefer **adding** a new named array over changing an existing one — old
   loaders keep working, new loaders use the new field.
3. Anything attached per-frame (e.g. detection labels later) MUST be length `N`
   and index-aligned to `frames`, so the sync guarantee extends to it.
4. Never introduce a field that requires `allow_pickle=True` to read.

---

## Minimal loader sketch (reference)

```python
import numpy as np

with np.load(path, allow_pickle=False) as d:
    assert int(d["schema_version"]) == 1
    frames = d["frames"]            # (N,150,270,3) uint8 BGR
    key_names = [str(s) for s in d["key_names"]]
    # per-frame action columns:
    keys, lclick, rclick = d["keys"], d["lclick"], d["rclick"]
    dx, dy = d["dx"], d["dy"]
    ts = d["timestamps"]
    N = frames.shape[0]
    assert all(a.shape[0] == N for a in (keys, lclick, rclick, dx, dy, ts))
```

`src/inspect_recording.py` is the fuller, human-facing version of this check.
