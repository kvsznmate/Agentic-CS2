# Agentic-CS2

**Teaching an AI agent to play Counter-Strike 2 by watching how we play it.**

---

## The goal

An agent that can move around a map sensibly and take a fight when it sees an
enemy, trained entirely on gameplay we recorded ourselves.

The aim is not a superhuman aimbot. It's a believable, human-style player on
one map, built honestly from scratch so that every part of it can be measured
and explained.

---

## How it works (the idea)

A human CS player is effectively doing two jobs at once:

1. **Navigating**: glancing at the minimap, deciding where to go, and choosing
   which angle to watch.
2. **Fighting**: spotting an enemy on screen, then aiming and shooting.

The agent copies that split. It has **two separate "brains"**, each looking at a
different part of the screen, plus a **referee** that decides which one is in
control:

```
                 ┌──────────────────────────────┐
  minimap  ───►  │  Navigation                   │──┐
  (radar)        │  "where do I go, what angle?" │  │
                 └──────────────────────────────┘  │     ┌───────────┐
                                                   ├───► │  Arbiter  │ ───► keyboard
                 ┌──────────────────────────────┐  │     │ (referee) │      + mouse
  first-person ► │  Combat                       │──┘     └───────────┘
  view           │  "is there an enemy? aim."    │
                 └──────────────────────────────┘
```

- **Navigation** reads the **radar** (minimap) to work out where the player is
  and how to move. It also checks a hand-written **probability panel** for the
  map: our own CS knowledge, written down, of where enemies tend to be at
  different points in a round.

- **Combat** reads the **first-person view** to detect enemies and aim at them.
  (The radar doesn't show enemies, so detection has to come from the main
  image.)

- **The arbiter** gives the mouse to exactly one of them per frame. If an
  enemy is visible, combat takes over and movement freezes. Otherwise,
  navigation drives.

### Where the data comes from


- the **screen** (a downscaled first-person image, plus a separate high-resolution
  crop of the radar),
- **what we pressed**: keys, mouse clicks and raw mouse movement, synced to
  that exact frame,
- **game state** from CS2's official Game State Integration (GSI): whether we're
  alive, the round phase, health, weapon and ammo.

Movement labels come for free: the keys we pressed *are* the answers the model
learns from. Enemy-detection labels don't. Nothing in a recording says where
the enemies were on screen, so producing those labels is a separate problem.

### Built on research, but our own

The approach builds on an earlier behavioural-cloning study on CS:GO (Pearce &
Zhu, 2022). We use its ideas but not its dataset, code or agent, and we don't
benchmark against it. CS2 runs on a new engine (Source 2), so most of the old
capture and memory-reading tricks no longer work and had to be rebuilt.

---

## Current state

The project is organised around three **go/no-go gates**: cheap experiments that
could kill the idea early, run before anything expensive gets built.

| Gate | Question | Status |
|---|---|---|
| **1. Capture + sync** | Can we record screen and input on CS2, frame-accurately? | ✅ **Passed.** ~15 FPS recording, inputs verified to line up with frames, 0% dropped frames over a 5-min test |
| **2. Radar signal** | Can movement be predicted from the minimap? | ◐ **Tooling ready, waiting on data.** Early results are provisional until the dataset is big enough |
| **3. Detection** | Can we label enemies and train a detector that works? | ☐ **Not started.** Labelling method still to be chosen |

### What works today

- **Recording pipeline:** long, unattended, crash-safe sessions with FPV, radar,
  inputs and GSI state saved together (data format v5, see `DATA_FORMAT.md`).
- **Data hygiene:** dead, spectating, freezetime and menu frames can be filtered
  out, and a session viewer lets you scrub through a recording like a video.
- **Data loader:** with tests, and a fixed train/held-out split by session so
  results can't leak.
- **First model:** a CNN + LSTM that looks at the last ~0.5 s of frames and
  predicts movement keys (WASD) plus where to turn the view. It's a movement-only
  baseline with no combat yet.
- **Live demo:** `play_movement` runs that model inside CS2 on a local bot
  server and presses the keys itself, so you can watch what it learned.

### Honest caveats

- **The dataset is still small.** The target is ~20 GB (≈5–7 hours of play) and
  recording is ongoing. Any model result before that point is provisional.
- **The first model mostly learned to walk forward.** An early run clearly
  beat chance on W but was near chance on A/S/D. That's typical with too little
  data, and it's why the radar verdict can't be committed yet.
- **Some newer pieces haven't been checked on the real machine yet,** such as
  the gameplay filter in the trainer. `PROJECT_ISSUES.md` tracks exactly what's
  verified and what isn't.
- **Nothing on the combat side exists yet:** no enemy detection, aim model,
  probability panel or arbiter.

### What's next

1. Record up to the ~20 GB dataset target.
2. Re-run the radar test at full volume and commit a go/no-go verdict.
3. Choose a way to produce enemy labels and start the detection track.

The full roadmap (milestones M0–M9, every issue, and its status) is in
[`PROJECT_ISSUES.md`](PROJECT_ISSUES.md).

---

## Safety: when it runs, and where

Injecting keyboard and mouse input into an online CS2 match can trigger a
**VAC ban**. So:

- **Recording** (read-only: screen capture, input logging, GSI) is done in
  normal matchmaking. It doesn't send any input to the game.
- **The agent**, i.e. anything that *presses keys or moves the mouse*, runs
  **only on an offline / local bot server**. The play loop starts disarmed
  (F9 arms it, F8 quits).

Running the agent online is a possible future step, at accepted risk. It is
**not** current practice. See `DECISIONS.md` D-007 / D-041.

---

## Getting started

**Hardware/OS:** native Windows, NVIDIA GPU (developed on an RTX 4050), CS2 at
1920×1080 fullscreen.

```bash
conda env create -f environment.yml
conda activate agentic-cs2
python -m src.smoke_test
```

A good run ends with `RESULT: all core imports succeeded. GPU visible.`

The stack is **Python 3.10 · TensorFlow 2.10.1 · CUDA 11.2 · cuDNN 8.1**. TF
2.10 is the last TensorFlow with GPU support on native Windows, and staying on
Windows keeps the model on the same OS as the game (D-011). If the smoke test
says TensorFlow sees no GPU, update your NVIDIA driver first, then check that the
CUDA/cuDNN conda packages installed. `nvidia-smi` confirms the driver can see
the card.

**Then:**

- **Want to record gameplay?** Follow [`HOW_TO_RECORD.md`](HOW_TO_RECORD.md)
  (no programming needed), or the more technical
  [`DATA_GUIDE.md`](DATA_GUIDE.md). For CS2 GSI setup, see
  [`gsi/README.md`](gsi/README.md).
- **Want every command?** See [`CLI.md`](CLI.md). Everything runs as
  `python -m src.<module>` from the repo root.

---

## Repo map

```
Agentic-CS2/
├── src/
│   ├── capture.py, capture_config.py   # screen grab: FPV + radar crop
│   ├── raw_mouse.py                    # raw mouse deltas (works under CS2's cursor lock)
│   ├── gsi_listener.py, gsi_probe.py   # CS2 game-state (alive, phase, health, ammo)
│   ├── recorder.py                     # the synchronized recording loop
│   ├── inspect_recording.py            # summarise / sanity-check a session
│   ├── review_session.py               # scrub a session like a video
│   ├── clean_session.py                # mark junk frames (menus, blank radar)
│   ├── data_loader.py, sequence_loader.py   # load training batches / sequences
│   ├── radar_probe.py                  # Gate 2: does the radar carry signal?
│   ├── model_lstm.py                   # movement + view-turn model (train / eval)
│   ├── look_probe.py                   # diagnose what a trained model actually learned
│   ├── key_output.py                   # simulated key / mouse output (offline only)
│   ├── play_movement.py                # live demo: model drives CS2 on a bot server
│   └── smoke_test.py, test_data_loader.py
├── gsi/                # CS2 GSI config file + setup notes
├── data/               # recordings and models (local only, git-ignored)
├── environment.yml     # pinned conda environment
├── index.html, Images/ # visual recording guide
└── *.md                # docs, listed below
```

## Docs map

| File | What's in it |
|---|---|
| [`PROJECT_ISSUES.md`](PROJECT_ISSUES.md) | Roadmap: milestones, issues, what's done and what's verified |
| [`DECISIONS.md`](DECISIONS.md) | Every design decision and the reason behind it (D-001 …) |
| [`DATA_FORMAT.md`](DATA_FORMAT.md) | The exact on-disk recording format |
| [`HOW_TO_RECORD.md`](HOW_TO_RECORD.md) | Beginner-friendly recording walkthrough |
| [`DATA_GUIDE.md`](DATA_GUIDE.md) | Technical data-making guide (setup, calibration, workflow) |
| [`CLI.md`](CLI.md) | Every runnable command |
| [`CLAUDE.md`](CLAUDE.md) | Working rules for the AI collaborator on this repo |

---

## Ground rules we stick to

- **Data first.** Making good data is the hard part, so it gets first-class
  attention.
- **One map, end to end,** before trying to generalise.
- **Self-recording is the backbone.** Learning from spectated pro demos would
  scale better, but it means recovering a stranger's inputs on a new engine. That
  is a huge job, so it stays an optional late stretch goal.
- **Set the bar before the experiment.** Every gate has a pass mark committed in
  advance, so results can't be talked into a pass afterwards.
