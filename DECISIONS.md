# DECISIONS.md — Agentic-CS2

The project's decision log. Each entry is a settled direction and the reasoning behind it. **Purpose: stop settled questions from being reopened every session.** Before proposing a direction, read this file. If you believe a decision here should change, say so explicitly — name the decision, say why it no longer holds, and get agreement before acting. Do not silently contradict a decision below.

**Format:** newest at top. Each entry: date · decision · why · (status if later changed).

---

## 2026-08 · Seeded from project-planning conversation

The entries below were established while scoping the project. They are the current ground truth.

### D-010 · Capture library is `mss` full-screen grab + crop (win32 BitBlt is dead)
**Decision:** Screen capture uses `mss` to grab the full screen, then crops to the CS2 game region and resizes to the fixed model input. The reference study's `screen_input.py` win32 `BitBlt`/`GetWindowDC` path is not carried over.
**Why:** This was already the ground truth in PROJECT_ISSUES #2 but had no decision entry, so it kept living only in issue text. CS2 is Source 2; the legacy D3D9-era `BitBlt` capture the reference relied on is dead/unreliable on it. `mss` is already a dependency in the reference's own `e2e.yml` (7.0.1), so it carries over cleanly. Recording it here so it isn't re-litigated.

### D-009 · Python stack pinned as a faithful conda repro of the reference (TF 2.3 / Py 3.7)
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
