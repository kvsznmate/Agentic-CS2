# DATA_GUIDE.md — How to make data for Agentic-CS2

This is the step-by-step guide to producing the project's dataset: environment
setup, one-time calibration, and the recording workflow. Follow it top to bottom
the first time; after that, the day-to-day loop is just the two commands in
**Step 6**.

**What "data" means here.** One recording session = one *v3 session folder* under
`data/recordings/`, holding synchronized, per-frame:
- the **FPV** image (150×270 BGR) — what the detection/aim model sees,
- a separate high-res **radar** image (128×128 BGR) — what the navigation model sees,
- the **action** taken that frame — keys, mouse buttons, and raw mouse dx/dy.

Both images come from one screen grab per frame, so they share a timestamp and
can't drift (decision D-024). The action is logged in the same tick and aligned
to that frame. This is the whole point of the capture design: **frame N's image
and frame N's action provably correspond.**

> **Why this matters / read first:** the reference study read game state (position,
> view angles) from RAM — dead on Source 2, so we can't. Everything we capture
> comes from the screen + our own input devices. The action labels are free (we
> log our own play); the images are legible because we grab at full resolution
> before downscaling. See `DECISIONS.md` D-002, D-010, D-012, D-015, D-024 for
> the reasoning. `PROJECT_ISSUES.md` M0/M1 are the milestones this guide covers.

---

## Prerequisites

- **Windows** with an NVIDIA GPU (developed on an RTX 4050). Capture and input
  logging are Windows-specific.
- **CS2 installed**, and the ability to launch a **local server with bots** —
  never online matchmaking (simulated input can trip anti-cheat; decision D-007).
- **Miniconda / Anaconda** installed.
- Enough free disk: the recorder refuses to start below **5 GB free**, and the
  first dataset target is **~20 GB** (decision D-020). Clear space first.

---

## Step 1 — Set up the environment (one time)

The stack is pinned: **Python 3.10 · TensorFlow 2.10.1 · CUDA 11.2 · cuDNN 8.1 ·
numpy 1.26 · mss 7.0.1 · OpenCV 4.10**, all on native Windows (decision D-011 —
TF 2.10 is the last TensorFlow with native-Windows GPU support).

From the project root (`C:\Users\<you>\Desktop\Agentic-CS2`):

```bash
conda env create -f environment.yml
conda activate agentic-cs2
python -m src.smoke_test
```

A clean run ends with:

```
RESULT: all core imports succeeded. GPU visible.
```

If it says `warn TensorFlow sees NO GPU`, that's a setup problem, not expected —
update your NVIDIA Windows driver first (`nvidia-smi` should list the card), then
confirm the CUDA/cuDNN conda packages installed. GPU is not required to *record*
data (recording is CPU + screen capture only), but you'll want it for training,
so fix it now.

> **Every command below assumes `conda activate agentic-cs2` is active** and that
> you're in the project root. All commands are run as modules (`python -m src.…`).

---

## Step 2 — Prepare CS2 (one time, then before each session)

1. Set CS2 to **fullscreen at 1920×1080** (native). The capture geometry is
   calibrated for this; a different resolution or windowed mode will need
   re-calibration (Step 3). See D-012.
2. Make sure the **radar/minimap is visible** in the top-left as normal.
3. Launch a **local practice/offline server with bots** on your target map
   (Dust2 is the project's first map — D-006). Not an online match.

Keep CS2 **focused and in-game** whenever you run a capture command — the tools
grab whatever is on screen and read your live input.

---

## Step 3 — Calibrate the capture (one time per machine / display change)

This confirms the tool is grabbing the game (not the desktop) and that the crop
is right. You only redo this if you change monitor, resolution, or HUD layout.

### 3a. Confirm the full-frame crop

With CS2 in the foreground (alt-tab to run the command, or run it on a second
monitor):

```bash
python -m src.capture --calibrate
```

This writes to `data/capture_debug/`:
- `full_monitor.png` — your whole screen with the crop rectangle drawn as a green box,
- `cropped.png` — exactly what the crop yields.

**Open `full_monitor.png`.** The green box should sit exactly on the game image —
no desktop, no title bar. If it's off, edit `CROP_LEFT/TOP/WIDTH/HEIGHT` in
`src/capture_config.py` and re-run until it's right. If the wrong screen was
grabbed entirely, change `MONITOR_INDEX` (the command prints the list of monitors
mss can see).

### 3b. Confirm the radar is legible

The radar is captured as its own high-res crop (D-024). Its rectangle is already
measured and baked into `capture_config.py` (`L=10, T=10, W=260, H=260`), so on
the standard setup you're just confirming it:

```bash
python -m src.capture --radar-calibrate
```

Writes to `data/capture_debug/`:
- `radar_on_full.png` — the full grab with the radar box drawn,
- `radar_src.png` — the raw radar crop,
- `radar_out.png` and `radar_out_upscaled.png` — the crop at the stored 128×128 size.

**Open `radar_out_upscaled.png`.** You should be able to read roughly where you
are on the map. If the box is off (different HUD scale, say), you can retighten it:

```bash
# add a labelled pixel grid to read the minimap bounds:
python -m src.capture --radar-calibrate --grid
# test a new rectangle without editing the config:
python -m src.capture --radar-calibrate --grid --radar-rect <L> <T> <W> <H>
```

When `radar_src.png` is tight on the minimap, bake those four numbers into
`capture_config.RADAR_SRC_*`. **This must be right before you record — the
rectangle is frozen into every recorded file** (it's part of what makes the data
self-describing). On the standard 1920×1080 setup you shouldn't need to touch it.

### 3c. (Optional) Eyeball the live feed and the frame rate

```bash
python -m src.capture --preview      # live window of the model-input feed; press q to quit
python -m src.capture --benchmark    # sustained capture FPS (FPV-only path)
```

`--benchmark` should report roughly ~25 FPS (capture in isolation); the bar is
≥20 (D-014). This is just a health check — the real recording rate is measured in
Step 5.

---

## Step 4 — Verify alignment (do this before your FIRST real recording)

**This is the most important check in the whole pipeline.** It proves that the
input logged at frame N actually corresponds to the screen at frame N. A loop
that merely runs does not prove this — so we test it with a scripted motion.

With CS2 focused and in-game (cursor locked, as in real play):

```bash
python -m src.recorder --verify
```

Follow the prompts: it asks you to **sweep the mouse steadily RIGHT**, then
**steadily LEFT**. It then checks that the logged mouse dx is positive during the
right sweep and negative during the left, and that the switch in the data lines
up with when you actually switched.

- **`RESULT: PASS`** — alignment is trustworthy. Re-run it once or twice; sync is
  worth over-checking. Proceed.
- **`RESULT: MOSTLY OK`** — direction separated but your switch timing was loose;
  re-run and switch promptly when prompted.
- **`RESULT: FAIL`** — **stop.** Do not record real data. Either raw mouse deltas
  aren't reading in-game (run `python -m src.raw_mouse --selftest` with CS2
  focused) or the loop isn't pairing inputs to frames. A sync that can't be shown
  reliable is a project-level kill condition (PROJECT_ISSUES #3), not a detail.

You only need a PASS once on a given setup, but re-verify if you change anything
about capture, input, or the machine.

---

## Step 5 — Do a short pilot recording (recommended before a long session)

Record a couple of minutes to confirm the full v3 path works end-to-end and to
see your real frame rate, before committing to a long session.

```bash
python -m src.recorder --record --seconds 120
```

Play normally in your bot server. Press **F8** to stop early; otherwise it stops
at 120 s. It prints a running frame count, chunks written, FPS, and long-gap
count, then finalizes.

Then **inspect what landed on disk**:

```bash
python -m src.inspect_recording data/recordings/<session_folder_name>
```

(The recorder prints the exact folder name and the inspect command when it
finishes.) Confirm it reports a **v3 session folder (FPV + radar)**, shows a
`radar (N,128,128,3)` row, and that frames are aligned.

**Sanity-check the radar images the model will actually train on:**

```bash
python -m src.radar_probe --dump-radar 12
```

This saves 12 stored radar crops (upscaled) to `data/capture_debug/`. Open them
and confirm you can read your map position. If yes, the data is good to scale up.

> **On frame rate:** a very short session looks slow because of warm-up (the first
> ~45 frames carry the largest gaps). The reliable number is the steady-state rate
> over a multi-minute run; the recording loop targets ~15 FPS and is bounded by
> the screen grab, not by input logging (decisions D-016, D-024). If you want the
> per-stage cost breakdown, run `python -m src.recorder --profile` (measures, saves
> nothing).

---

## Step 6 — Record the real dataset (the day-to-day loop)

This is the workflow you repeat until you've accumulated ~20 GB (≈ 5–7 hours of
play; decision D-020). Do it in multiple sessions — each session is a separate
folder, and more independent sessions is *better* (the train/held-out split is by
whole session).

Each time:

1. Launch CS2 fullscreen 1920×1080, start your **local bot server** on the map.
2. Activate the env and, from the project root:

   ```bash
   python -m src.recorder --record
   ```

   With no `--seconds`, it records **until you press F8**. Play normally. You can
   name a session if you like: `--record --name dust2_session_03`.

3. When done (F8), it finalizes the folder and prints where it is. Optionally
   inspect it (Step 5) to confirm it's clean.

**Repeat across several sessions** until you're at target volume. Check your total
any time with:

```bash
python -m src.data_loader --crop radar
```

This reports every usable session, the train/held-out split, and loads one batch
to confirm the loader is happy. (`--crop full` does the same for the FPV feed.)

> **Play varied movement.** Deathmatch reflexes tend to be mostly forward-running.
> For the navigation signal to be measurable, deliberately include strafing (A/D),
> turning, and holding angles — otherwise those keys are barely pressed and there's
> nothing to learn or measure from them.

### Tips for long sessions

- **F8 stops cleanly at any time** — the session is flushed and marked complete;
  you never lose a clean recording by stopping.
- **Crash-safe:** recordings are written in ~2-minute chunks, each finalized
  atomically, so a crash loses at most the current chunk (decisions D-018, D-019).
  An interrupted session is simply excluded from the dataset automatically (D-022).
- **Disk floor:** the recorder refuses to start below 5 GB free and stops cleanly
  if free space crosses that floor mid-session. Keep an eye on space toward 20 GB.
- **Don't use `--record-single`** for the dataset. That's a legacy one-file mode
  that stores *no radar* (FPV only); it exists only for a quick round-trip smoke
  test. The real dataset needs `--record` (v3, with radar).

---

## What you end up with

```
data/recordings/
├── session_20260811_141839/
│   ├── manifest.json          # schema_version:3, chunk list, total_frames, complete:true
│   ├── chunk_00000.npz         # ~1800 frames: frames + radar + actions, index-aligned
│   ├── chunk_00001.npz
│   └── ...
├── session_20260811_142126/
│   └── ...
└── ...
```

Each `chunk_*.npz` holds, all the same length N (row i = one synchronized tick):
`frames` (N,150,270,3), `radar` (N,128,128,3), `timestamps`, `keys` (N,11),
`lclick`, `rclick`, `dx`, `dy`, plus self-describing metadata (`schema_version`,
`geom`, `key_names`, `loop_fps_target`). The authoritative schema is
`DATA_FORMAT.md`.

---

## Using the data (quick reference)

The loader gives you two independent feeds from the same recordings — you pick
which via `crop`:

```python
from src import data_loader as dl

# Radar-only (navigation model): X is (B, 128, 128, 3)
train_ds, holdout_ds = dl.build_datasets(crop="radar")

# FPV-only (detection/aim model): X is (B, 150, 270, 3)
train_ds, holdout_ds = dl.build_datasets(crop="full")   # or "centre"

for X, Y in train_ds.iter_batches(batch_size=64):
    ...   # Y is (B, 15): [keys(11), lclick, rclick, dx, dy]; names via dl.action_layout(...)
```

The train/held-out split is by **whole session**, deterministic, and leak-free
(decision D-021) — held-out sessions never appear in the training stream. Adding
more sessions never reshuffles existing ones. `crop="radar"` requires v3 sessions
(older FPV-only recordings correctly refuse it).

---

## Command reference

| Command | What it does |
|---|---|
| `python -m src.smoke_test` | Verify the env imports + GPU is visible (Step 1) |
| `python -m src.capture --calibrate` | Check/fix the full-frame crop geometry |
| `python -m src.capture --radar-calibrate [--grid] [--radar-rect L T W H]` | Check/tighten the radar crop |
| `python -m src.capture --preview` | Live window of the model-input feed (q to quit) |
| `python -m src.capture --benchmark` | Sustained capture FPS (health check) |
| `python -m src.raw_mouse --selftest` | Confirm raw mouse deltas read in-game |
| `python -m src.recorder --verify` | **Scripted-motion alignment check — do before first recording** |
| `python -m src.recorder --dryrun` | Run the loop with a live input readout, save nothing |
| `python -m src.recorder --profile` | Per-stage per-frame cost breakdown, save nothing |
| `python -m src.recorder --record [--seconds N] [--name NAME]` | **Record a v3 session (FPV + radar). F8 to stop** |
| `python -m src.inspect_recording data/recordings/<folder>` | Summarize + spot-check a recorded session |
| `python -m src.radar_probe --dump-radar N` | Save N stored radar crops to eyeball legibility |
| `python -m src.data_loader --crop {full,centre,radar}` | Report the split + load one batch to sanity-check |

---

## Troubleshooting

- **`--verify` FAILs** → don't record. Run `python -m src.raw_mouse --selftest`
  with CS2 focused; if deltas don't track motion, it's the raw-input path (DPI,
  focus, fullscreen). Alignment is a hard gate (PROJECT_ISSUES #3).
- **Green box off the game in `--calibrate`** → fix `CROP_*` (or `MONITOR_INDEX`)
  in `src/capture_config.py`, re-run.
- **Radar unreadable in `--dump-radar` / `--radar-calibrate`** → retighten the box
  with `--radar-calibrate --grid --radar-rect …`, then bake into
  `capture_config.RADAR_SRC_*`. Must be done before recording.
- **Recorder won't start ("REFUSING TO START: … GB free")** → free disk to above
  5 GB.
- **Frame rate looks low on a short clip** → it's warm-up; judge from a
  multi-minute run. See D-016.
- **`--crop radar` errors on old sessions** → those are FPV-only (v1/v2) sessions
  recorded before the radar existed; they can't serve radar by design. Record new
  v3 sessions with `--record`.
