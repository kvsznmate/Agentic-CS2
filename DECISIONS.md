# DECISIONS.md — Agentic-CS2

The project's decision log. Each entry is a settled direction and the reasoning behind it. **Purpose: stop settled questions from being reopened every session.** Before proposing a direction, read this file. If you believe a decision here should change, say so explicitly — name the decision, say why it no longer holds, and get agreement before acting. Do not silently contradict a decision below.

**Format:** newest at top. Each entry: date · decision · why · (status if later changed).

---

## 2026-08 · Seeded from project-planning conversation

The entries below were established while scoping the project. They are the current ground truth.

### D-014 · #2 capture rate bar set at 20 FPS (met); dxcam is the deferred lever if the full loop needs more
**Decision:** The committed "usable rate" bar for the #2 capture gate is **sustained ≥ 20 FPS**, measured by `--benchmark`. Current measured capture rate is **~25 FPS** (mss ~26 ms grab + cv2 ~14 ms resize per frame, 1920x1080 -> 150x270), so the bar is **met with margin**. We do NOT chase 30 FPS on capture-in-isolation. **dxcam** (Windows Desktop Duplication API) is recorded as the known faster-capture lever, deferred, to be pulled only if the *full agent loop* proves too slow.
**Why:** #2 is the easy gate; #3 (frame/input sync) is the hard one and the real M0 risk. Optimising capture alone past 30 spends effort on a number that model inference will later eat into anyway — the meaningful budget is the *whole loop* (capture + logging + two-model inference), which can't be measured until those exist. The reference study's entire loop ran at 16 FPS and produced a working agent, so ~25 FPS for capture alone is comfortably in proven-workable territory; 20 is set as the floor (just above the study's 16, with headroom) so normal drift or #3's logging overhead doesn't push us under our own bar for no real reason. Profiling (D-013) already showed both mss (~26 ms) and the resize (~14 ms) are slower than ideal on this hardware, consistent with a non-SIMD PyPI OpenCV build and mss's generic Windows copy path — both fixable, neither worth fixing now. **The lever, for a future session:** if the full loop is too slow, swap mss -> dxcam (typically several times faster for full-screen grabs, and hands back frames in a cheaper form). That would change D-010 (currently commits to mss) and needs its own calibration pass, so it is a decision, not a silent swap. Deferring it avoids over-investing in the easy gate and lets the real full-loop budget — measured later, not guessed — decide whether it's even needed.

### D-013 · Capture resize uses INTER_LINEAR, not INTER_AREA
**Decision:** The downscale in `capture.py`'s `grab()` uses `cv2.INTER_LINEAR`. Not `INTER_AREA`, despite AREA being the textbook choice for shrinking images.
**Why:** Benchmarked on this machine (2026-08), `INTER_AREA` resizing 1920x1080 down to 270x150 cost ~24 ms/frame — roughly half the total per-frame budget and the single largest contributor to a sub-bar 20 FPS. At a ~7x per-axis reduction `INTER_AREA` averages over every source pixel in each output cell, which is expensive; `INTER_LINEAR` is far cheaper and on a 150x270 frame feeding a CNN the quality/anti-aliasing difference is negligible. This was found by adding a grab-vs-resize timing split to `--benchmark` after a raw 20 FPS reading, and it overturned the initial assumption that mss was the bottleneck — mss (~26 ms) and the resize (~24 ms) were near-equal, and the resize was the fixable half. **Note for a future session:** do not switch this back to `INTER_AREA` "for quality" without re-checking the FPS cost; the speed hit is real and was measured. If frame quality ever proves to matter for aim, revisit with a measurement, not a default.

### D-012 · Fullscreen 1920x1080 capture, full-frame crop, 16:9 model input (150x270)
**Decision:** CS2 is run **fullscreen at the monitor's native 1920x1080**. The capture crop is the **full frame** (LEFT=0, TOP=0, 1920x1080) — stored as an absolute monitor-relative rectangle in `src/capture_config.py`, separate from capture logic, confirmed via `python -m src.capture --calibrate`. The model input size is changed from the study's 150x280 (~4:3) to **150x270 (H,W), which is 16:9**. Frames leave in BGR (Q4).
**Why:** Calibration killed the original plan of matching the study's 1024x768 *windowed* setup. On this machine the native monitor is 1920x1080 (16:9); the study's inward offsets (824x498) landed the crop in a corner and cut away most of the game. Running CS2 fullscreen at native res makes the full-monitor grab pure game edge to edge, so the crop is trivially full-frame — no window-position hunting, no desktop/taskbar to exclude. **The aspect consequence is the real content of this decision:** the source is now 16:9 but the study's 150x280 target is ~4:3, so a naive resize would horizontally distort every frame (a quiet quality bug, worst for aim precision). Changing the input to 150x270 (1920/1080=1.778; 150x270=1.80, ~1.2% off, both dims cleanly divisible for CNN pooling) makes 1920x1080 -> 150x270 a pure uniform downscale with no distortion. Exact parity with the study's 150x280 first-layer size is dropped deliberately — D-001 already says we don't run or compare against their model, so parity buys nothing while an undistorted frame at real resolution buys a lot. **Trades noted:** (1) fullscreen 1920x1080 departs from the study's 4:3 1024x768 (Q1), accepted for a clean undistorted capture; (2) full-frame keeps peripheral margins the study discarded to concentrate resolution on the centre — that centre-vs-edges trade is deferred to the per-model sub-crops in the loader (#6: full FPV / centre / radar), tuned against real data, not resolved at capture where keeping the whole frame protects the radar from being amputated. **Supersedes** the earlier 1024x768-windowed framing of Q1 for capture geometry; 150x270 is the value #5 (DATA_FORMAT.md) must lock authoritatively.

### D-011 · Python stack is native-Windows TF 2.10 + CUDA 11.2 + Py 3.10, with GPU on the RTX 4050
**Decision:** The environment is Python 3.10, TensorFlow 2.10.1, cudatoolkit 11.2, cudnn 8.1.0, numpy 1.26, plus mss / opencv-python / pillow / matplotlib, via conda (`environment.yml`). CUDA libs come from conda-forge; TF and the vision/capture pip deps are pip-installed into the conda env. Everything runs on native Windows. Supersedes D-009.
**Why:** The machine has an **RTX 4050** (Ada, compute capability 8.9). This one fact collapsed the earlier options: (1) D-009's TF 2.3 / CUDA 10.1 stack cannot see an Ada card at all — CUDA 10.1 tops out at Turing (CC 7.5) — so on this hardware D-009 was silently CPU-only. (2) "Modern TF on the GPU on native Windows" does not exist: **TF 2.10 was the last release with native-Windows GPU support**; from 2.11 GPU requires WSL2 or CPU-only. (3) WSL2 + modern TF would give newer TF but split the agent across two OSes — model in Linux, capture (#2) and input (#3) on Windows where CS2 runs — precisely at the capture+sync gate that is the project's #1 risk. So the stack that (a) uses the 4050 and (b) keeps the whole pipeline on one OS is **TF 2.10 native Windows**. Official pairing (per TF install guide): `cudatoolkit=11.2`, `cudnn=8.1.0`, Python 3.9–3.11; confirmed working on Ada with a current NVIDIA driver. Conda is still required because pip won't install CUDA libs on Windows. Parity with the reference's TF *version* is dropped — D-001 already says we don't run their agent or load their `.h5` weights, so version parity buys nothing, whereas a working GPU on the actual hardware buys a lot. **Trade accepted:** TF 2.10 is a 2022 release (not bleeding-edge), and the CUDA toolchain must match exactly or the GPU won't be detected — acceptable, since M2–M4 models are small and the real bottleneck is data, not FLOPs.

### D-010 · Capture library is `mss` full-screen grab + crop (win32 BitBlt is dead)
**Decision:** Screen capture uses `mss` to grab the full screen, then crops to the CS2 game region and resizes to the fixed model input. The reference study's `screen_input.py` win32 `BitBlt`/`GetWindowDC` path is not carried over.
**Why:** This was already the ground truth in PROJECT_ISSUES #2 but had no decision entry, so it kept living only in issue text. CS2 is Source 2; the legacy D3D9-era `BitBlt` capture the reference relied on is dead/unreliable on it. `mss` is already a dependency in the reference's own `e2e.yml` (7.0.1), so it carries over cleanly. Recording it here so it isn't re-litigated.

### D-009 · Python stack pinned as a faithful conda repro of the reference (TF 2.3 / Py 3.7)
**Status:** **SUPERSEDED by D-011 on 2026-08.** Reason: the target machine has an RTX 4050 (Ada, CC 8.9), which the CUDA 10.1 toolchain in this stack cannot address — making this stack CPU-only on the actual hardware. Replaced with a native-Windows TF 2.10 + CUDA 11.2 stack that uses the GPU. The entry is kept below for the decision trail.
**Decision:** The environment reproduces the reference study's `e2e.yml` stack — Python 3.7, TensorFlow 2.3, numpy 1.18, OpenCV 4.4, mss 7.0.1 — via conda (`environment.yml`), not via `pyproject.toml`/uv. The pieces tied to the reference's dead memory-inference and win32-capture paths (`pymem`, and the `BitBlt` use of `pywin32`) are dropped from the core; `pywin32` is kept only for simulated key output later.
**Why:** Chosen deliberately over a modern stack (see below) for exact parity with the reference methods. The originally-requested combination — TF 2.3 + Py 3.7 + pyproject/uv + a modern CUDA build — is not installable as a set: Py 3.7 is EOL and below uv/Poetry's floor; TF 2.3's only official GPU path is CUDA 10.1 + cuDNN 7.6, which installs via conda (not pip/uv) and does not target current NVIDIA cards (RTX 30/40/50). Conda is therefore mandatory for this stack, which is why the reference itself shipped a conda env. **GPU caveat:** the CUDA 10.1 GPU path works only on older GPUs; on current cards the env runs CPU-only until the stack is modernized. Reconsider (move to modern TF or PyTorch on Py 3.11) if the EOL interpreter or CPU-only training becomes a real blocker.

### D-008 · The agent maintains this log and the other living docs
**Decision:** Updating `DECISIONS.md`, `DATA_FORMAT.md`, and `PROJECT_ISSUES.md` is part of doing the work, not a separate chore. New decisions are appended here as/before they are acted on.
**Why:** The plan reversed direction several times during scoping (see below). Without a written record, each new session re-litigates settled questions and re-suggests rejected approaches. The log is the memory that prevents that.

### D-007 · Offline / local bot servers only
**Decision:** The agent runs only against local servers with bots, never online matchmaking.
**Why:** Simulated keyboard/mouse input and any memory reading can trip anti-cheat. Offline is also how the reference research ran. Safety + it matches our actual need.

### D-006 · One map end-to-end before multi-map
**Decision:** Nail a single map (Dust2) through the whole pipeline before authoring any others.
**Why:** Multi-map is a multiplier, not a prerequisite. Per-map panel authoring is real work; doing several before one works end-to-end is scope ahead of a working core.

### D-005 · The probability panel is hand-authored, not learned
**Decision:** Enemy-location-over-round-time priors are encoded as heuristics from CS knowledge, not learned from our dataset.
**Why:** Round-timing enemy priors are a competitive-play concept. Our self-recorded data is unstructured (deathmatch-style, no coordinated round timing), so the signal to learn them isn't in the data. Hand-authoring is legitimate and is the only viable path with our data.

### D-004 · Detection comes from the first-person image, never the radar
**Decision:** Enemy detection is done by the vision model on the FPV. The radar/panel is only for navigation and positioning.
**Why:** In CS2 the radar only shows enemies a teammate currently sees. A solo agent's radar shows essentially no enemies. Any design that expects the radar to reveal enemies is broken. This also cleanly separates the two feeds: radar → "where am I / where should I be," vision → "is someone actually there."

### D-003 · Detection labels are a separate data problem; action labels are free
**Decision:** Treat enemy-detection label creation as its own milestone with its own method and risk. Do not assume it comes from recording.
**Why:** Self-recording logs our own inputs automatically (action labels free), but tells us nothing about where enemies are on screen. Detection labels require their own creation method (annotation or an automated ground-truth source). This is the most underestimated part of the data work.

### D-002 · Self-recording is the spine; spectating is optional and late
**Decision:** Build the dataset by recording our own play. Spectating (recovering another player's actions) is a stretch track that must never block the critical path.
**Why:** Self-recording gives us our own actions for free — no inference needed. Spectating requires recovering a *stranger's* actions, which on CS2 means rebuilding the memory-inference pipeline the reference study did with RAM offsets + GSI — all dead on Source 2. That rebuild is plausibly larger than the rest of the project combined. It is quarantined as optional so it can't sink the timeline.

### D-001 · Build our own project and dataset; do not run or compare against the reference study
**Decision:** We build on the reference study's methods and findings, but train our own models on our own CS2 data. We do not use its dataset, run its agent, or benchmark against it. All success criteria are absolute bars we set for ourselves.
**Why:** Our architecture (two-feed panel + arbiter) and scope differ from the reference study's single end-to-end model. Reproducing or comparing against it would pull in dead dependencies (its CSGO-era capture, RAM offsets, GSI) and a comparison that doesn't map cleanly onto a different design. Training our own keeps scope focused on our actual research question. Superseded the earlier idea of reusing the study's dataset.

---

## Template for new entries

```
### D-00X · <short decision title>
**Decision:** <what was decided>
**Why:** <one or two lines of reasoning>
**Status (optional):** <e.g. "Superseded by D-0YY on 2026-09 because ...">
```
