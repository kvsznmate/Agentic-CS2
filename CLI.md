# CLI.md — Agentic-CS2 command reference

Every runnable command in the project, grouped by what you're actually doing. All
are run from the repo root (`C:\Users\matek_yulq090\Desktop\Agentic-CS2`) with the
`agentic-cs2` conda env active, as Python modules:

```
python -m src.<module> [flags]
```

Conventions used below:
- **F8** quits any live loop (recorder, play). **F9** arms/disarms `play_movement`.
- Flags shown with their real defaults. `--seconds` omitted usually means "run
  until you stop it."
- Decision references like *(D-031)* point at `DECISIONS.md`; issue refs like
  *(#7)* point at `PROJECT_ISSUES.md`.
- **Modules with no CLI** (libraries, imported by others): `capture_config.py`,
  `sequence_loader.py`, `smoke_test.py`, `__init__.py`. Tests
  (`test_data_loader.py`) run under `pytest`, not as a module.

---

## 0. Quick reference (everything, one line each)

```
# Capture / geometry
python -m src.capture --calibrate                 # save full grab + crop to fix geometry
python -m src.capture --radar-calibrate           # crop radar from full-res grab, check legibility
python -m src.capture --radar-calibrate --grid    #   ...with a labelled pixel grid
python -m src.capture --radar-calibrate --radar-rect L T W H   # test a source rect before baking it in
python -m src.capture --benchmark                 # sustained grab+crop+resize FPS vs the bar
python -m src.capture --preview                   # live window of the model-input feed

# Input plumbing (prove the pieces read/write before recording)
python -m src.raw_mouse --selftest                # live dx/dy — raw mouse deltas read cleanly?
python -m src.key_output --selftest               # press W/A/S/D on a countdown — does the char move?
python -m src.key_output --probe-focus            # inject 'w' into the focused window (Notepad diagnostic)
python -m src.key_output --tap w --seconds 1.0    # hold one key for N seconds

# GSI (alive / gameplay signal)
python -m src.gsi_probe                            # feasibility harness: raw GSI readout + JSONL log
python -m src.gsi_listener --selftest              # shared listener alone, print state changes
python -m src.recorder --verify-gsi                # confirm GSI reaches the RECORDER + alive rule

# The M0 gate: capture + sync
python -m src.recorder --verify                    # scripted-motion alignment check — RUN FIRST
python -m src.recorder --dryrun                    # loop + live input readout, saves nothing
python -m src.recorder --profile                   # per-stage loop timing (what limits FPS)

# Recording
python -m src.recorder --record                    # extended chunked v5 session (until F8)
python -m src.recorder --record --seconds 300      #   ...capped at 5 minutes
python -m src.recorder --record --name pilot01     #   ...with a name stub
python -m src.recorder --record-single             # LEGACY v1 single-file FPV-only round-trip

# Inspecting / cleaning / reviewing recordings
python -m src.recorder --inspect                   # convenience wrapper for inspect_recording
python -m src.inspect_recording                    # summarise newest recording
python -m src.inspect_recording <name-or-path>     # summarise a specific one
python -m src.inspect_recording --dump 6           #   ...also save 6 FPV frames as PNG
python -m src.inspect_recording --dump-radar 6     #   ...also save 6 radar crops as PNG
python -m src.clean_session --report               # pooled radar-variance cut, write nothing
python -m src.clean_session --all                  # write keep-mask for every usable session
python -m src.clean_session --session <name>       # write keep-mask for one session
python -m src.review_session                        # scrubber: FPV+radar, seekbar, per-frame actions

# Data loader / gates / models
python -m src.data_loader                           # discover sessions, sanity-check a batch
python -m src.data_loader --crop radar --gameplay-filter
python -m src.radar_probe --gameplay-report         # radar-variance distribution + blank/present cut
python -m src.radar_probe --probe                   # #7 gate: WASD-from-radar linear probe + verdict
python -m src.model_lstm --summary                  # build + print the movement model, no training
python -m src.model_lstm --train                    # train WASD-from-FPV baseline (D-027)
python -m src.model_lstm --train --crop radar       # #7 stronger test: movement-from-radar (D-028)
python -m src.play_movement                         # drive the trained baseline on a LOCAL bot server
```

---

## 1. Capture & geometry — `src.capture`

Set up and verify screen capture before anything else. Geometry (crop rectangle,
radar rectangle) is machine-specific and baked into every recording, so calibrate
first. *(D-010 mss capture; D-012 fullscreen 1920×1080 / 150×270 FPV; D-013
INTER_LINEAR; D-014 ≥20 FPS bar; D-024 two-resolution radar crop.)*

| Command | What it does |
|---|---|
| `python -m src.capture --calibrate` | Save a full grab + the current crop so you can check/fix the game-region geometry. |
| `python -m src.capture --radar-calibrate` | Crop the radar from a full-res grab and save it, to check the minimap is legible at 128×128. |
| `python -m src.capture --radar-calibrate --grid` | As above, plus a labelled pixel grid so you can read tight minimap bounds. |
| `python -m src.capture --radar-calibrate --radar-rect L T W H` | Test a specific source rectangle (four ints) instead of `capture_config`'s — iterate before baking it in. |
| `python -m src.capture --benchmark` | Measure sustained grab+crop+resize FPS against the gate bar. |
| `python -m src.capture --benchmark --seconds 30` | Benchmark for a custom duration (default 10 s). |
| `python -m src.capture --preview` | Live window of the cropped+resized model-input feed. |

---

## 2. Input plumbing — `src.raw_mouse`, `src.key_output`

Prove the input pieces work in isolation before wiring them into recording or
play. Both have a `--selftest` for exactly this. *(D-015 raw mouse deltas; D-029
SendInput scan codes, offline-only.)*

### `src.raw_mouse` — read mouse aim (deltas)
| Command | What it does |
|---|---|
| `python -m src.raw_mouse --selftest` | Print live dx/dy so you can verify raw deltas read cleanly (works under the in-game cursor lock). |
| `python -m src.raw_mouse --selftest --seconds 15` | Selftest for a custom duration (default 15 s). |

### `src.key_output` — write keystrokes to the game
| Command | What it does |
|---|---|
| `python -m src.key_output --selftest` | Press W/A/S/D in turn on a countdown — watch your character move. This is the gate before trusting key output. |
| `python -m src.key_output --selftest --key w` | Selftest only one key. |
| `python -m src.key_output --probe-focus` | Inject `w` into the focused window (type into Notepad) — the CS2-vs-system diagnostic when `--selftest` shows nothing in-game. |
| `python -m src.key_output --tap w --seconds 1.0` | Hold one key (`w/a/s/d/space/ctrl/shift/1/2/3/r`) for `--seconds` (default 1.0). |

---

## 3. GSI — alive / gameplay signal — `src.gsi_probe`, `src.gsi_listener`

Game State Integration tells the recorder whether you're alive, dead, or
spectating. Requires `gamestate_integration_agenticcs2.cfg` in
`<Steam>/.../game/csgo/cfg/`, saved BOM-free, with CS2 restarted after adding it.
Defaults: host `127.0.0.1`, port `3000`, token `agentic_cs2_local`. *(D-030
feasibility; D-031 folded into recording; D-032 the steamid-match alive rule.)*

### `src.gsi_probe` — standalone feasibility harness
Raw GSI readout (alive/weapon/phase changes), cadence stats, and a JSONL log of
every payload to `data/gsi_probe/`. Use this to inspect real payloads.
| Command | What it does |
|---|---|
| `python -m src.gsi_probe` | Run the probe with defaults. |
| `python -m src.gsi_probe --port 3000 --host 127.0.0.1 --token agentic_cs2_local` | Override bind host/port/token (must match the `.cfg`). |

### `src.gsi_listener` — the shared listener (used by the recorder)
| Command | What it does |
|---|---|
| `python -m src.gsi_listener --selftest` | Run the listener alone and print state changes — confirms the shared module receives + parses GSI. |
| `python -m src.gsi_listener --selftest --seconds 30` | Selftest, stop after N seconds (default: until Ctrl-C). |

> To confirm GSI reaches the **recorder** specifically (not just the standalone
> listener), use `python -m src.recorder --verify-gsi` — see §4.

---

## 4. The M0 gate: capture + sync — `src.recorder` (verify/dryrun/profile)

The most important gate: can we record synchronized screen+input at all? Run
`--verify` first; a sync that can't be shown reliable is a kill-flag condition
*(#3)*. *(D-015 single synchronous loop; D-016 15 FPS bar.)*

| Command | What it does |
|---|---|
| `python -m src.recorder --verify` | **Run this first.** Scripted-motion alignment check: sweep right then left when prompted; it verifies logged mouse direction and the switch-point line up with your motion. |
| `python -m src.recorder --verify --seconds 12` | Alignment check with a custom duration (default 12 s). |
| `python -m src.recorder --verify-gsi` | Confirm GSI reaches the recorder and the alive rule reads right — alive while playing, DEAD when you die AND while spectating a living teammate. Use a **bot match** (not deathmatch) and die + spectate to fully exercise it. |
| `python -m src.recorder --verify-gsi --seconds 30` | GSI verify with a custom duration (default 30 s). |
| `python -m src.recorder --dryrun` | Run the loop with a live input readout, writing nothing — quick smoke test that all input sources are alive together. |
| `python -m src.recorder --profile` | Time each loop stage (mouse/keys/grab/assembly) to find what limits FPS. Saves nothing. |

---

## 5. Recording — `src.recorder` (record)

Writes a **v5 session folder** (FPV + radar + GSI alive/round_phase + state
features health/weapon/ammo) under `data/recordings/`. Chunks every 1800 frames
(~2 min), refuses to start below 5 GB free, and **won't start until GSI's first
POST** (so a mis-configured `.cfg` fails loudly instead of recording a silent
zero-frame session). Runs until **F8** unless `--seconds` is given. *(D-018
chunked; D-019 threaded writes; D-024 radar; D-031 GSI liveness gate; D-033 state
features; D-034 boundary-stall diagnostics printed at end of run.)*

| Command | What it does |
|---|---|
| `python -m src.recorder --record` | Record an extended chunked v5 session until F8. |
| `python -m src.recorder --record --seconds 300` | Record for a fixed duration (here 5 minutes). |
| `python -m src.recorder --record --name pilot01` | Record with a name stub (folder becomes `session_...` / your stub). |
| `python -m src.recorder --record-single` | **LEGACY v1**, FPV-only, single `.npz`. Quick self-contained round-trip check only — does NOT store radar or GSI. Not for the real dataset. |
| `python -m src.recorder --record-single --seconds 60 --name rt_test` | Legacy single-file record with duration + name. |

At the end of a `--record` run you'll see a **Writer:** line reporting per-chunk
compress+write time and whether chunk-boundary stalls are backpressure or OS
write-back *(D-034)* — read it if FPS looks low.

---

## 6. Inspect / clean / review recordings

### `src.inspect_recording` — summarise a session
Array shapes + alignment, real FPS from timestamps, chunk-boundary stall
diagnosis, key/click/mouse activity, GSI alive summary, and (v5) the
health/weapon/ammo state-feature summary. Reads v1–v5.
| Command | What it does |
|---|---|
| `python -m src.inspect_recording` | Summarise the newest recording (folder or `.npz`). |
| `python -m src.inspect_recording <name-or-path>` | Summarise a specific session (folder name, or full path). |
| `python -m src.inspect_recording --dump 6` | Also save 6 evenly-spaced FPV frames as PNG (to `data/capture_debug/`). |
| `python -m src.inspect_recording --dump-radar 6` | Also save 6 evenly-spaced radar crops as PNG (v3+). |
| `python -m src.recorder --inspect` | Convenience wrapper (same tool, one entry point). Accepts `--name`, `--dump`, `--dump-radar`. |

### `src.clean_session` — non-destructive keep-mask for blank frames
Computes per-frame radar variance, derives one blank/present cut across all
sessions, and writes `keep_mask.npz` + `clean_report.json` per session. Never
modifies recordings; the loader uses masks only opt-in. *(D-026.)*
| Command | What it does |
|---|---|
| `python -m src.clean_session --report` | Show the pooled radar-variance distribution and where the cut falls. Writes nothing. Run this first. |
| `python -m src.clean_session --all` | Write a keep-mask for every usable session. |
| `python -m src.clean_session --all --force` | Write masks even if the variance distribution isn't clearly bimodal (override the safety guard). |
| `python -m src.clean_session --session <name>` | Write a keep-mask for one session (folder name or path). |
| `python -m src.clean_session --all --dump-blank 10` | Also save 10 dropped (blank) radar frames per session as PNG, to eyeball what got cut. |

### `src.review_session` — visual scrubber
FPV + radar side by side, seekbar, per-frame action readout; tints frames the
keep-mask would drop so you can verify the cut by eye. *(D-026.)*
| Command | What it does |
|---|---|
| `python -m src.review_session` | Review the newest usable session. |
| `python -m src.review_session --session <name>` | Review a specific session (folder name or path). |
| `python -m src.review_session --fps 15` | Set initial playback FPS (default 15; adjust live with `[` and `]`). |
| `python -m src.review_session --no-mask` | Ignore any keep-mask; don't tint frames. |

---

## 7. Data loader — `src.data_loader`

Discovers sessions, builds the deterministic whole-session train/held-out split,
and serves (input-crop, action-label) batches. Running it as a module does a
discovery + batch sanity check. *(D-021 hash split; D-022 crashed-session
exclusion; D-026 keep-mask; D-031 gameplay filter.)*
| Command | What it does |
|---|---|
| `python -m src.data_loader` | Discover sessions, print the split, and sanity-check a batch. |
| `python -m src.data_loader --crop full` | Serve the full FPV crop (default). Other choices: `centre`, `radar` (v3+). |
| `python -m src.data_loader --crop radar` | Serve the stored high-res radar crop. |
| `python -m src.data_loader --gameplay-filter` | Serve only GSI-alive gameplay frames (v4+); reports how many frames each session keeps. |
| `python -m src.data_loader --batch 8` | Set the sanity-check batch size (default 8). |
| `python -m src.data_loader --holdout-frac 0.20` | Held-out fraction for the hash split (default 0.20). |

---

## 8. Gate #7 — radar navigation signal — `src.radar_probe`

The M2 go/no-go: does the radar carry movement signal? Linear probe with an
honest, group-split, volume-gated verdict. *(D-023 provisional-below-floor; D-025
32×32 grayscale + numerically honest solver; D-028 the training-based stronger
test.)*
| Command | What it does |
|---|---|
| `python -m src.radar_probe --gameplay-report` | Show the radar-variance distribution and where the blank/present cut would fall, without fitting. Run before `--probe --filter-blank`. |
| `python -m src.radar_probe --probe` | Run the WASD-from-radar linear probe and print a verdict. |
| `python -m src.radar_probe --probe --filter-blank` | Drop blank/no-radar frames before fitting — tells "no signal" apart from "blank frames masking signal." |
| `python -m src.radar_probe --probe --balanced-split` | PROBE-ONLY: split by frame volume instead of the committed name-hash, so both sides get comparable data at small session counts. Leak-free, but NOT the committed split and NOT for training. |
| `python -m src.radar_probe --probe --holdout-frac 0.4` | Session holdout fraction for the probe's group split (default 0.4). |
| `python -m src.radar_probe --dump-radar 8` | Save 8 stored radar crops as PNGs to eyeball legibility. |

---

## 9. Models — `src.model_lstm`, `src.play_movement`

### `src.model_lstm` — WASD-from-FPV movement baseline
CNN+LSTM, many-to-one, 4 independent sigmoids (W/A/S/D). Reports held-out lift
vs the majority-class baseline; stamps PROVISIONAL below the data-volume floor.
The same script runs the #7 radar comparison via `--crop radar`. *(D-027 the
baseline; D-028 radar comparison.)*
| Command | What it does |
|---|---|
| `python -m src.model_lstm --summary` | Build and print the model architecture; no training. |
| `python -m src.model_lstm --train` | Train the WASD-from-FPV baseline. |
| `python -m src.model_lstm --train --crop radar` | Train on the radar feed instead — the #7 movement-from-radar comparison. |
| `python -m src.model_lstm --train --crop centre` | Train on the centre FPV crop. |
| `python -m src.model_lstm --train --seq-len 8` | Frames per window (default 8, ~0.5 s at 15 FPS). |
| `python -m src.model_lstm --train --epochs 10 --batch 32` | Training epochs (default 10) and batch size (default 32). |
| `python -m src.model_lstm --train --use-keep-mask` | Exclude blank/no-radar frames via the D-026 keep-mask. |
| `python -m src.model_lstm --train --holdout-frac 0.20` | Held-out fraction for the split (default 0.20). |

### `src.play_movement` — drive the baseline (LOCAL bot server only)
Paced ~15 FPS inference loop that actuates the trained model. Starts **DISARMED**;
**F9** arms/disarms, **F8** quits, all exits release keys. **Offline/local bot
servers only** *(D-007)* — never online. Expect behavioural-cloning drift in
open-loop control (a known BC property). *(D-029.)*
| Command | What it does |
|---|---|
| `python -m src.play_movement` | Drive the newest model for the default `full` crop. |
| `python -m src.play_movement --model <path.keras>` | Use a specific `.keras` model (default: newest for `--crop`). |
| `python -m src.play_movement --crop radar` | Tell the loop which feed the model expects — must match how it was trained. |
| `python -m src.play_movement --threshold 0.4` | Probability ≥ this presses the key (default 0.4). Every key above the bar fires, so diagonals (W+D) work. Raise (~0.6) to suppress over-eager keys; lower to let more through. Tune live. |

---

## 10. Tests

```
pytest src/test_data_loader.py          # loader/split unit tests (D-021/D-022)
pytest                                   # whole suite
```
Other in-repo test files referenced by decisions (e.g. `test_seq_logic.py` for the
sequence-windowing invariants, D-027) run the same way if present.

---

## Typical end-to-end order

1. `src.capture --calibrate` / `--radar-calibrate` — geometry.
2. `src.raw_mouse --selftest`, `src.key_output --selftest` — input pieces.
3. `src.gsi_probe` then `src.recorder --verify-gsi` — GSI alive signal.
4. `src.recorder --verify` — **the sync gate (#3).**
5. `src.recorder --record` — record the dataset.
6. `src.inspect_recording` (+ `src.clean_session --report`, `src.review_session`) — check it.
7. `src.data_loader`, `src.radar_probe --probe` — **radar gate (#7).**
8. `src.model_lstm --train` — movement baseline; `src.play_movement` — see it act.

Gates in order: **#3 capture+sync → #7 radar signal → #10 detection.** Don't build
behind an unresolved gate.
