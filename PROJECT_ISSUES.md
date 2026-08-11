# CS Two-Feed Agent — Milestones & Issues

A behavioural-cloning FPS agent that splits perception into two feeds: a **radar/probability panel** driving navigation, and a **first-person vision model** driving enemy detection and aim, combined by an **arbiter**. We train our own models on our own data captured from the current live game (CS2). We do not use or compare against any prior work.

**The central reality of this project:** we have no dataset. Creating one is the largest risk and the first several milestones. Data collection — not modeling — is where this project succeeds or fails. The plan reflects that: it leads with capture, treats detection-labeling as its own high-risk track, and quarantines spectating as optional so it can never block the critical path.

**Ordering principle:** derisking, not components. The cheapest project-killing tests come first and gate everything expensive behind them.

**Two hard scope guards, learned the hard way:**
- **Self-recording is the spine; spectating is an optional stretch track.** Self-recording gives us our own actions for free. Spectating requires recovering a *stranger's* actions, which on CS2 means rebuilding a dead memory-inference pipeline on a new engine — potentially larger than the whole rest of the project. It must never block the critical path.
- **Action labels are free; detection labels are a second data problem.** Playing the game logs our inputs automatically, but tells us nothing about where enemies were on screen. Detection labels need their own creation method and their own milestone.

---

## Milestones

### M0 — Capture pipeline (GO/NO-GO)  ✅ PASSED (2026-08)
**Goal:** prove we can capture clean, synchronized screen + input data on CS2 at all.
**Exit criteria:**
- ✅ Screen capture working on CS2 (mss full-screen grab + crop; the legacy D3D9 method is dead). — #2 closed.
- ✅ Our own keyboard/mouse inputs logged and time-synced to frames. — #3 closed; alignment verified across 3 runs.
- ✅ A short recorded session round-trips to disk and reloads intact. — 5-min session, 4833 frames, reloaded intact.
**Benchmark:** a 5-minute self-recorded session yields frame-action pairs with verified alignment (input at frame N actually corresponds to screen at frame N); dropped-frame rate under a set bar. — **MET** (alignment PASS; 0.0% long-gaps vs <2% bar; ~15 FPS vs ≥15 bar, D-016).
**Outcome:** the project's central risk — can we make synchronized data on Source 2 at all — is resolved YES. Capture is mss+crop (D-010/D-012), aim is raw mouse deltas readable under the in-game cursor lock (D-015), the loop is a single synchronous design with input logging essentially free on top of capture (D-016). Next: M1 (#4 extended-session tooling, #5 lock the data format).

### M1 — Self-recorded action dataset
**Goal:** enough of our own play to train movement + aim.
**Exit criteria:**
- Recording tooling usable for extended sessions without babysitting.
- A first dataset of self-play recorded, stored, and documented (size target set in M1 issues).
- Data format finalized (frame image + action vector + **separate high-res radar image**, D-024).
**Benchmark:** N hours / M frames recorded (target committed in #4 after a pilot); a loader emits (input, action-label) batches; held-out split reserved.
**Scope note:** actions come free from self-recording. Detection labels do NOT — that's M3.

### M2 — GO/NO-GO: Radar carries navigation signal
**Goal:** prove the radar is extractable and predicts movement, on our data.
**Exit criteria:**
- Radar image reliably recovers self-position from the CS2 HUD.
- Movement predictable from radar above chance on our recordings.
**Benchmark:** self-position on ≥95% of sampled frames; a linear probe beats chance on coarse movement direction (exact bar set in #7 after first measurement).
**Kill condition:** if radar shows no movement signal, the navigation path is dead — rethink before M5.
**Tooling status:** `src/radar_probe.py` implements the whole gate (legibility dump + WASD linear probe). The probe verdict is **PROVISIONAL** below a data-volume floor — a committed M2 result needs the full dataset (D-020). **Radar geometry is already SETTLED (D-024):** the radar is a dedicated high-res 128x128 crop captured before the FPV downscale, its rectangle measured and baked into `capture_config` via `--radar-calibrate`, so #7's only open work is the signal verdict at volume — not box-hunting.

### M3 — GO/NO-GO: Detection labels + working detector
**Goal:** the load-bearing combat component, from scratch, including its labels.
**Exit criteria:**
- A method to obtain enemy-on-screen labels chosen and validated (manual annotation, or an automated ground-truth source — this is a real sub-project).
- A labeled detection subset created.
- Detection model hits a pre-declared threshold on held-out data.
**Benchmark:** labeling throughput sustainable enough to reach a usable set; detection meets its declared precision/recall bar (set in #10 before training).
**Kill condition:** self-recording gives nothing for detection. If we can't produce labels at usable quality/throughput, or the detector can't hit threshold, the arbiter design collapses — pivot.

### M4 — Aim & combat behaviour
**Goal:** aim at and shoot a detected enemy.
**Exit criteria:**
- Aim model trained on our data; hits a committed absolute accuracy target.
- Combat sub-policy (detect + aim + fire) produces sensible outputs when an enemy is present.
**Benchmark:** aim-head meets committed target on held-out data; in scripted playback, fire triggers on-target above a set rate.

### M5 — Probability panel (one map)
**Goal:** a queryable enemy-location prior for one map, hand-authored.
**Exit criteria:**
- Panel structure encodes timing zones (position × round-time → likelihood).
- Navigation controller queries it and returns a direction + hold-angle.
**Benchmark:** query <1 ms; scripted scenarios give map-sensible directions.
**Scope guard:** hand-authored heuristics only. Round-timing can't be learned from unstructured deathmatch-style self-play — don't try.

### M6 — Integration & arbiter
**Goal:** both feeds together, one actuator, clean handoff, real-time.
**Exit criteria:**
- Arbiter gates mouse ownership on detection (enemy → combat drives + freeze movement; else → navigation drives).
- Debug overlay shows both feeds + current mouse owner per frame.
- Full loop sustains target FPS.
**Benchmark:** zero dual-ownership frames in test; loop holds target FPS with both models active.

### M7 — Whole-agent evaluation
**Goal:** measure how well the complete agent plays, on our own terms.
**Exit criteria:**
- Agent evaluated against a fixed, self-defined protocol.
- Results + honest limitations written up.
**Benchmark:** agent meets a committed absolute bar on a frozen protocol (e.g. kills-per-minute vs medium bots on one map, or scripted-scenario success rate — defined in #18). A bar we set for ourselves.

### M8 — (Stretch) Spectator data pipeline
**Goal:** scale data by recovering other players' actions — the paper's approach, on CS2.
**Exit criteria:** a working method to infer a spectated player's inputs on CS2, feeding the same dataset format.
**Scope guard / warning:** this is a from-scratch reverse-engineering effort on Source 2 (dead RAM offsets, changed GSI). It is plausibly larger than all of M0–M7 combined. Pursue ONLY after a self-recorded agent works end-to-end, and never let it block the critical path.

### M9 — (Stretch) Multi-map
**Goal:** author a second map's panel to test generalization. A multiplier, not a prerequisite.

---

## Issues

Labels: `infra` `data` `gate` `perception` `navigation` `combat` `integration` `eval` `stretch`

---

### M0 — Capture pipeline (GO/NO-GO)

#### #1 Repo, environment, dependency pinning  ✅ built (pending env-create verification)
**Labels:** `infra` · **Depends on:** —
Pin the stack (Python + capture libs + framework + numpy + OpenCV). Document setup. Note version drift can change model behaviour.
**Acceptance:** fresh checkout + docs yields a working env; smoke test imports core modules.
**Delivered:** `environment.yml` (conda; native-Windows **TF 2.10 + CUDA 11.2 + Py 3.10**, GPU on the RTX 4050 — DECISIONS D-011, superseding D-009), `src/__init__.py`, `src/smoke_test.py` (import + version + GPU-detection check), `.gitignore`, setup docs in `README.md`. Capture lib settled as mss + crop (D-010). **Still to verify on-machine:** run `conda env create -f environment.yml` then `python -m src.smoke_test` and confirm `RESULT: all core imports succeeded. GPU visible.` — the env build itself hasn't been executed here, only specified. GPU detection depends on a current NVIDIA driver + the CUDA/cuDNN conda packages installing cleanly (see README GPU setup).

#### #2 [GATE] CS2 screen capture  ✅ CLOSED (verified on-machine, 2026-08)
**Labels:** `gate` `data` `infra` · **Depends on:** #1
Capture the CS2 window. The legacy D3D9 BitBlt method is dead on Source 2 — use full-screen grab (mss) + crop to the game region, output the fixed model input size. Verify against different in-game scenes.
**Acceptance:** clean, correctly-cropped frames captured live from CS2 at a usable rate. — **MET.**
**Delivered:** `src/capture.py` (mss grab + crop + resize to **150x270** BGR, `INTER_LINEAR`; persistent mss instance; `grab()` returns frame **plus a perf_counter timestamp** so #3 can sync inputs to it without a rewrite; **`grab_with_radar()` added in D-024** returns FPV + a separate high-res 128x128 radar crop from the same grab) and `src/capture_config.py` (per-machine geometry, incl. the baked radar rectangle). Entry points: `--calibrate`, `--radar-calibrate` (measure/confirm the radar box), `--benchmark` (with grab-vs-resize timing split), `--preview`.
**Setup (D-012):** CS2 fullscreen at native **1920x1080**; crop **full-frame**; FPV model input **150x270 (16:9)** so the downscale carries no aspect distortion.
**Verified on-machine:** `--calibrate` green box covers the full frame; `cropped.png` undistorted; `--preview` clean across scenes with the radar visible; `--benchmark` **~25 FPS** sustained (FPV-only path).
**Rate bar (D-014):** committed at **≥ 20 FPS**, met with margin at ~25. Deliberately not chasing 30 on capture-in-isolation — the meaningful budget is the full loop (capture + logging + inference), measured later. **dxcam** recorded as the deferred faster-capture lever (D-014) if the full loop later needs it.
**Findings along the way:** profiling (D-013) overturned the assumption that mss was the sole bottleneck — the `INTER_AREA` resize was ~half the cost; switched to `INTER_LINEAR` for the FPV. Both mss (~26 ms) and resize (~14 ms) are slower than ideal on this hardware (likely non-SIMD PyPI OpenCV + mss's generic Windows path), acceptable at the committed bar.
**Scope reminder:** capture is only the FIRST HALF of the M0 gate. #3 (sync) is the real risk — #2 passing is NOT M0 passing.

#### #3 [GATE] Synced input logging  ✅ CLOSED — verified on-machine (2026-08). **M0 GATE PASSED.**
**Labels:** `gate` `data` · **Depends on:** #2
Log our own keyboard + mouse (including mouse deltas) and align each input to the correct captured frame. Frame/input synchronization is the hard part, not the capture. Verify alignment explicitly.
**Acceptance:** a short session produces frame-action pairs where input at frame N provably matches screen at frame N; dropped-frame rate under a committed bar. — **MET.**
**Delivered:** `src/raw_mouse.py` (raw-input WM_INPUT mouse listener, accumulates device dx/dy, read per frame; `--selftest`), `src/recorder.py` (single synchronous loop; `--verify` / `--record` / `--dryrun` / `--profile`; `.npz` writer with reload round-trip), `src/inspect_recording.py` (summarise + spot-check a recording). Approach in DECISIONS **D-015**.
**Verified on-machine:**
- Raw mouse deltas read cleanly **in-game under CS2's cursor lock** (the load-bearing bet of D-015) — `raw_mouse --selftest` confirmed dx/dy track physical motion where the study's GetCursorPos would freeze.
- Alignment PASS, **consistent across 3 verify runs**: right-sweep 94–96% +dx, left-sweep 82–86% −dx, means cleanly separated (e.g. +40 vs −49) every run. Direction (the actual sync proof) never smeared; only a secondary reaction-time boundary metric ever flagged, since fixed (fixed-time tolerance).
- Round-trip: a 5-min session saved 4833 frames to `.npz`, reloaded intact, arrays index-aligned (confirmed via `inspect_recording`).
**Dropped-frame bar:** long-gap frames **0.0%** over the 5-min session — well under the < 2% bar. **MET.**
**FPS (D-016):** recording runs at **~15 FPS**, committed bar ≥ 15, met. Profiling proved the loop is 100% bounded by the ~37 ms mss grab; input logging adds ~0.07 ms (free). Matches the study's 16 FPS working loop. dxcam is the documented lever (D-014) if the full agent loop later needs more.
**This closes M0** — the first and most important gate. Synchronized screen+input capture on CS2 is proven. Everything downstream (radar signal #7, detection #10, the models) now has a real data pipeline to build on.

---

### M1 — Self-recorded action dataset

#### #4 Recording tooling for extended sessions  ✅ CLOSED (2026-08)
**Labels:** `data` `infra` · **Depends on:** #3
Turn the capture+log prototype into something usable for long sessions unattended: start/stop, disk management, crash resilience. Run a pilot; set the dataset size target (hours/frames) from pilot throughput.
**Acceptance:** an extended session records without babysitting; size target committed. — **MET.**
**Delivered:** `recorder.py` `record_session()` (behind `--record`) writing **chunked sessions** via `ChunkedSessionWriter` with a **background writer thread** (D-019) so capture never stalls at a chunk boundary. Bounded memory (Option A, ~2-chunk depth), crash-safe (atomic `.tmp.npz`-then-rename per chunk + manifest; `finally`-block finalize so F8/timeout/Ctrl-C/exception all close cleanly, marked `complete` only if all writes succeeded). Disk management: refuses to start below 5 GB free, stops cleanly if free space crosses the floor mid-session. `--record-single` retains the legacy single-file writer; `inspect_recording.py` reads all formats and localises stalls to chunk boundaries. Rationale: DECISIONS **D-018** (chunking) + **D-019** (threaded write). **Note:** the session format has since advanced to v3 (D-024) to carry the radar array; the chunking/crash-safety machinery here is unchanged.
**Verified on-machine:** two 5-min sessions recorded, chunked (2–3 chunks), reloaded intact, aligned. After D-019 the boundary stall is gone — gaps at frames 1800/3600 are 82/69 ms (were 10,657 ms), whole-session slowest gap 127 ms, jitter 5.3 ms. Clean thread shutdown.
**Throughput + SIZE TARGET (committed):** measured ~**3–4 GB/hour** at ~13 FPS FPV-only (5-min run = 313 MB / 3862 frames). **Dataset target: record as much as practical, ~20 GB (≈ 5–7 hours of play).** User will clear disk space to accommodate. This is the volume #6 records. **Note (D-024):** v3's radar array adds ~+40%/frame, so ~20 GB now buys somewhat less playtime; budget accordingly.
**Deliberately waived:** the explicit mid-session kill/crash test was consciously skipped (user call) — crash-safety is low-risk because each chunk is fully written + atomically renamed the moment it lands, so completed chunks can't be corrupted by a later kill, and clean-shutdown finalize is already verified. Noted here so a future session sees it was a choice, not an oversight. **UPDATE (2026-08):** a real mid-write crash later occurred (`session_20260810_151611`) — the atomic-write design contained it cleanly (manifest stayed `complete:false`, no good chunk corrupted), and the loader now excludes such sessions at discovery (D-022). So the waived test was exercised by reality and the design held.

#### #5 Finalize data format  ✅ CLOSED (2026-08); ⟳ REVISED to v3 (D-024)
**Labels:** `data` · **Depends on:** #3
Lock the on-disk schema: frame image, action vector (keys, clicks, mouse dx/dy), and the radar. Design for the loader and for later detection-label attachment.
**Acceptance:** documented schema; sample files validate against it. — **MET.**
**Delivered:** `DATA_FORMAT.md` at the repo root. Originally **v1**; **revised to v3 (D-024)** to add a separate per-frame high-res **`radar`** array (N,128,128,3 uint8 BGR) alongside the FPV `frames` (N,150,270,3), both index-aligned, both from one grab. Metadata `key_names`, `schema_version`, `geom` (now records the radar source rectangle too), `loop_fps_target`. `recorder.py` writes the self-description fields. Reads via `np.load(allow_pickle=False)`; validated by the round-trip check + `inspect_recording.py` + `test_data_loader.py`.
**Two deliberate scope calls, as revised by D-024:** (1) the FPV frame is stored **whole** (crops deferred to the loader per D-012), BUT the **radar is now a separate captured array, not a loader-side crop** — because the minimap's needed resolution only exists before the FPV downscale, so it must be captured, not recovered later. This supersedes v1's "one full frame, all crops deferred" stance for the radar specifically; the FPV half stands. (2) detection labels are **not** in the format — separate data problem (M3); schema accepts them later as a new index-aligned per-frame array + a schema bump, no change to existing fields.
**Back-compat:** v1 (single FPV `.npz`) and v2 (chunked FPV-only) remain readable; they simply have no radar array, and the loader refuses the radar input on them with a clear error.

#### #6 Record the first self-play dataset + loader  ◐ IN PROGRESS — loader built + tested (incl. v3 radar); dataset recording underway
**Labels:** `data` `infra` · **Depends on:** #4, #5
Record the target volume of our own play. Build a loader emitting (input, action-label) batches with configurable inputs (full FPV, centre FPV, the radar). Reserve a held-out split.
**Acceptance:** dataset recorded + documented; loader unit-tested for shapes; held-out split reserved.

**Status of the three acceptance parts:**
- **Loader — ✅ BUILT (v3-aware).** `src/data_loader.py`: reads ALL on-disk formats (v3 folder with radar, v2 FPV-only folder, v1 single file), concatenates chunks in manifest order, asserts per-frame index-alignment on load **including the radar array** (refuses a misaligned or mis-sized-radar session rather than serving it), and **excludes crashed/incomplete sessions at discovery** (D-022). Emits `(X, Y)` batches via `SessionDataset` / `get_batch` / `iter_batches`; `X` is the selected input, `Y` is a 15-col action vector `[keys(11), lclick, rclick, dx, dy]` (column names via `action_layout()`). Inputs: `full` (150x270 FPV), `centre` (FPV crop), a caller-supplied `(top,left,h,w)` FPV crop, or **`radar`** — which serves the STORED high-res 128x128 array directly (D-024), NOT a crop of the FPV. On v1/v2 (no radar array) the radar input raises a clear error. `get_batch` is hardened (2026-08): up-front radar-availability check across all contributing sessions, sentinel-fill + written-mask assert so no uninitialised row can escape.
- **Held-out split — ✅ RESERVED (D-021).** Whole-session, deterministic by name-hash, ~20%; leak-free by construction (`build_datasets` returns train + held-out over disjoint session lists). Reserved BEFORE the data is used, so #7 and #10 stay honest. Rationale + caveats in DECISIONS **D-021**.
- **Unit tests — ✅ (all green on-machine).** `src/test_data_loader.py` (runs without pytest): split deterministic/leak-free/stable-when-sessions-added; crashed sessions excluded + never in a split (D-022); FPV batch shapes; action-vector layout + values; row-order-matches-request; misaligned-session rejection; disjoint train/holdout; **v3 radar tests (D-024): v3 serves the real stored radar and rows match the on-disk array; v1/v2 refuse radar; a mixed v2+v3 radar batch fails up-front; a mis-sized radar array is rejected; get_batch never returns uninitialised rows.** Verified end-to-end on real data: a v3 session (`session_20260811_141839`) serves `crop=radar` as real (B,128,128,3) BGR.
- **Dataset recorded + documented — ◐ IN PROGRESS.** Target ~20 GB / ≈5–7 h (D-020). Prior ~30 min of v2/v1 data is **FPV-only (no radar array)** — archived to `data/recordings_v2_archive/`, unusable for the #7 radar probe; the dataset is being **re-recorded as v3** (FPV + radar). A couple of confirmed v3 sessions exist so far; recording continues via `python -m src.recorder --record`.

**#6 does NOT close until the ~20 GB of v3 data is recorded + documented.** The loader, split, and tests are done; the dataset volume is the remaining bar. Only then are #7 (radar), #10 (detection), #11 (aim) truly unblocked on real data volume — though the loader is available now for their scaffolding.

---

### M2 — GO/NO-GO: Radar carries navigation signal

#### #7 [GATE] Radar signal sanity check  ◐ IN PROGRESS — tool built + updated for v3; geometry SETTLED, committed verdict awaits data volume
**Labels:** `gate` `perception` `navigation` · **Depends on:** #6
On our own frames, the go/no-go: check whether WASD is predictable from the radar sequence (by eye + a quick linear probe). Measure first, then set the M2 bar.
**Acceptance:** self-position legible in the radar image; a committed verdict + threshold. If below chance, RAISE THE KILL FLAG before M5.

**Key finding — this is a FROM-SCRATCH problem, not a reference reproduction.** The reference study never recovered position from the radar image: it read player coords (`localpos1/2/3`) straight from game memory via RAM offsets (its "map coverage" is a histogram of those memory-read coords, not anything from pixels). All dead on Source 2 (D-002). So we have NO ground-truth position and must ask whether the radar PIXELS carry usable navigation signal at all — which is exactly what makes this a genuine gate that can fail.

**Geometry is SETTLED (D-024).** The earlier plan had #7 hunt for a radar crop rectangle in the 150x270 FPV. That was retired: at the FPV downscale the radar was too low-res to read position, so D-024 made the radar a **dedicated high-res 128x128 crop captured before the FPV downscale**, its source rectangle (`L=10,T=10,W=260,H=260`) measured with `python -m src.capture --radar-calibrate`, confirmed legible on an uploaded crop, and baked into `capture_config` + recorded in DATA_FORMAT.md. So #7 no longer owns any geometry — legibility is confirmed and the box is fixed. The only open work is the **signal verdict at volume**.

**Delivered:** `src/radar_probe.py` — updated for D-024 (consumes the stored radar via the loader's `crop="radar"`; no rectangle logic):
- `--dump-radar N` (legibility): saves N stored radar crops (raw + upscaled) so self-position legibility can be re-confirmed on real sessions.
- `--probe` (signal): fits a per-key linear logistic probe for w/a/s/d from the flattened stored radar, on a **whole-session train/test split** (D-021 style — never per-frame, which would leak correlated frames and fake a PASS), and reports per-key accuracy vs the majority-class baseline plus mean lift.
**HONEST-GATE behaviour:** the probe prints its verdict as **PROVISIONAL** below a data-volume floor (`MIN_PROBE_FRAMES=20000`, `MIN_PROBE_SESSIONS=3`). A provisional PASS is a smoke test only and must NOT be recorded in DECISIONS.md as the M2 outcome — the committed verdict + threshold come from a re-run at full data volume (D-020). Rationale in DECISIONS **D-023**. A linear probe is a floor (a CNN could do better), so a committed "no signal" warrants one CNN attempt before the KILL flag; a committed clear positive is a real GO. Requires v3 sessions (the radar array); on v1/v2-only data the probe reports cleanly that it needs v3 data.

**Remaining to close #7:** (1) after the v3 dataset reaches committed volume, run `--probe`, commit the M2 threshold + verdict in DECISIONS.md; (2) if no signal, attempt a CNN probe, then KILL-or-proceed. (Geometry/legibility is done.)

---

### M3 — GO/NO-GO: Detection labels + working detector

#### #8 Choose and validate a detection-labeling method
**Labels:** `gate` `data` `combat` · **Depends on:** #5
Self-recording yields no enemy-position labels — this issue decides how to get them. Evaluate options: manual annotation of a subset, semi-automated labeling, or an automated ground-truth source. Assess throughput and label quality on a pilot. This is a real sub-project, not a task.
**Acceptance:** a chosen method with measured throughput + quality on a pilot batch.

#### #9 Create the labeled detection subset
**Labels:** `data` `combat` · **Depends on:** #8
Produce enough labeled enemy-on-screen data to train a detector, using the chosen method. Track label noise.
**Acceptance:** a labeled subset of committed size; quality documented.

#### #10 [GATE] Train the enemy-detection model
**Labels:** `gate` `combat` · **Depends on:** #9, #6
Train "enemy on screen" from the FPV centre crop. Declare the precision/recall threshold in this issue before training so the gate is honest.
**Acceptance:** detection meets the pre-declared threshold on held-out data. If not, RAISE THE KILL FLAG.

---

### M4 — Aim & combat behaviour

#### #11 Train the aim model
**Labels:** `combat` · **Depends on:** #6
Train mouse aiming from our FPV data. Commit an absolute held-out accuracy target up front (set after inspecting our mouse-delta distributions — they'll differ from any prior work since it's our own play).
**Acceptance:** aim-head meets committed target on held-out data.

#### #12 Combat sub-policy (aim + fire)
**Labels:** `combat` · **Depends on:** #10, #11
Wire detect + aim + fire into one behaviour: enemy present → aim toward it and fire under a sensible policy (respect fire-rate, don't spray at nothing).
**Acceptance:** on scripted playback with enemies, crosshair converges and fire triggers above a set on-target rate.

---

### M5 — Probability panel (one map)

#### #13 Design the panel data structure
**Labels:** `navigation` · **Depends on:** #7
Structure mapping (self-position, round-time) → enemy-location likelihood over map regions; cheap to query per frame.
**Acceptance:** documented schema + in-memory structure queryable <1 ms.

#### #14 Hand-author one map's timing zones
**Labels:** `navigation` · **Depends on:** #13
Encode CS knowledge: which regions hold enemies at which round-times (mid-doors timing generalised). One map.
**Acceptance:** panel populated; spot-checked against known timings.

#### #15 Navigation controller
**Labels:** `navigation` · **Depends on:** #14, #7
Consume panel + radar position; output WASD + hold-angle. Rule-based to start.
**Acceptance:** sensible movement + hold-angle on scripted scenarios.

---

### M6 — Integration & arbiter

#### #16 Arbiter / mouse-ownership gate
**Labels:** `integration` · **Depends on:** #10, #15
Enemy detected → combat owns mouse + freeze movement; else → navigation owns + holds angle. One owner per frame.
**Acceptance:** handoff unit-tested on synthetic signals; zero dual-ownership frames.

#### #17 Two-feed debug overlay + performance pass
**Labels:** `integration` `infra` · **Depends on:** #16
Draw both paths + current mouse owner live. Then profile: two forward passes per loop ~doubles compute — hit target FPS by shrinking/slowing the navigation model as needed. **Note:** D-024's radar crop already trimmed the recording loop's headroom (unpaced ceiling ~24 FPS with radar vs ~27 without); the full two-model loop will be tighter still, so dxcam (D-014) is the more likely lever here.
**Acceptance:** overlay usable for drift diagnosis; loop sustains target FPS; timing documented.

---

### M7 — Whole-agent evaluation

#### #18 Define protocol + target, then evaluate and write up
**Labels:** `eval` · **Depends on:** #16
No baseline, so define our own frozen test (e.g. kills-per-minute vs medium bots on one map, or scripted-scenario success). Commit an absolute "working" bar. Run several times (report variance). Write up methods, results incl. negatives, and limitations — chiefly self-recorded-data scale and the offline scope.
**Acceptance:** repeatable protocol + committed target; results across runs; a report a third party can follow.

---

### Stretch

#### #19 (Stretch) Spectator data pipeline on CS2
**Labels:** `stretch` `data` · **Depends on:** #18
Recover a spectated player's actions on CS2 to scale data the paper's way. WARNING: from-scratch reverse-engineering on Source 2 (dead RAM offsets, changed GSI); plausibly larger than M0–M7 combined. Only after a self-recorded agent works end-to-end.
**Acceptance:** a working inference method feeding the same dataset format.

#### #20 (Stretch) Second-map panel
**Labels:** `stretch` `navigation` · **Depends on:** #18
Author a second map's timing zones to test generalization.
**Acceptance:** second map authored + integrated; navigation sensible on it.

---

## Benchmark summary

Every number is a bar we set for ourselves — no external comparison.

| Milestone | Benchmark | Set when |
|---|---|---|
| M0 | 5-min session → verified frame-action alignment; dropped frames under bar | Bar set in #3 |
| M1 | N hours / M frames recorded; loader emits batches | Target set in #4 after pilot |
| M2 | Self-position ≥95% of frames; movement predictable above chance | Bar set in #7 |
| M3 | Sustainable label throughput; detection meets precision/recall | Declared in #10 before training |
| M4 | Aim-head hits committed target; fire on-target above set rate | Set in #11/#12 |
| M5 | Panel query <1 ms; map-sensible directions | Fixed |
| M6 | Zero dual-ownership frames; loop holds target FPS | Fixed |
| M7 | Agent meets committed absolute bar on frozen protocol | Set in #18 |

---

## Dependency map

```
#1 ─ #2 ─ #3 (GATE: capture+sync) ─┬─ #4 ─┐
                                   ├─ #5 ─┼─ #6 ─┬─ #7 (GATE: radar)
                                   │      │      ├─ #11
                                   │      │      └─ #10
                                   │      └─ #8 ─ #9 ─ #10 (GATE: detection)
                                   
#7 ─ #13 ─ #14 ─ #15 ─┐
#10 ─┬─ #12           │
#11 ─┴─ #12 ─────────┴─ #16 ─ #17 ─ #18 ─┬─ #19 (stretch: spectate)
                                          └─ #20 (stretch: 2nd map)
```

**Three gates, in order: #3 (capture+sync) → #7 (radar signal) → #10 (detection).** #3 is the new first gate and the most important — with no synchronized data, there is no project. All three are cheap relative to what they protect, and all three come before the panel and arbiter.
