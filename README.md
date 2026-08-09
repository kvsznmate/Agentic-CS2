# Agentic-CS2

A behavioural-cloning FPS agent for **Counter-Strike 2**. Perception is split
into two feeds — a **radar / probability panel** driving navigation, and a
**first-person vision model** driving enemy detection and aim — combined by an
**arbiter** that hands the mouse to exactly one feed per frame (enemy seen →
combat drives and movement freezes; otherwise → navigation drives).

We train our own models on our own data captured from live CS2. We build on the
findings of a reference study (Pearce & Zhu, 2022) but do **not** use its
dataset, run its agent, or benchmark against it. See `DECISIONS.md` for the
reasoning behind every settled direction, and `PROJECT_ISSUES.md` for the
milestone/issue plan.

> **Status:** early. The first gate — capturing clean, *synchronized*
> screen+input data on CS2 — is not yet passed. Do not build behind an open
> gate; see the three gates in `CLAUDE.md`.

---

## Environment setup (Issue #1)

The Python stack runs on **native Windows with GPU support on the RTX 4050**:
**Python 3.10 · TensorFlow 2.10.1 · CUDA 11.2 · cuDNN 8.1 · numpy 1.26 · mss 7.0.1 ·
OpenCV 4.10**. Why this exact stack (and not a newer TF, or WSL2) is recorded in
`DECISIONS.md` **D-011** — short version: **TF 2.10 is the last TensorFlow with
GPU support on native Windows**, and staying native keeps the model on the same
OS as the CS2 capture/input layer.

```bash
conda env create -f environment.yml
conda activate agentic-cs2
python -m src.smoke_test
```

`src/smoke_test.py` verifies every pinned dependency imports, reports loaded
versions against what's expected, and reports whether the GPU is visible. A
clean run printing `RESULT: all core imports succeeded. GPU visible.` satisfies
Issue #1's acceptance ("fresh checkout + docs yields a working env; smoke test
imports core modules").

### GPU setup — the 4050 should be detected

Unlike a CPU-only stack, here the GPU is the point. `cudatoolkit=11.2` and
`cudnn=8.1.0` come from conda-forge (pip cannot install CUDA libs on Windows);
TensorFlow is pip-installed into the env. For the 4050 to be seen you also need:

- a **current NVIDIA Windows driver** (the driver — not the toolkit — is what
  gives an Ada card support under the 11.2 runtime libraries; update via GeForce
  Experience or nvidia.com), and
- the **Microsoft Visual C++ 2015–2022 redistributable** (usually already
  present on a gaming machine).

If `smoke_test.py` prints `warn TensorFlow sees NO GPU`, that's a setup issue,
not expected behaviour — check the driver first, then that the CUDA/cuDNN conda
packages actually installed. `nvidia-smi` at a terminal confirms the driver sees
the card.

> **Note on native-Windows GPU + TensorFlow:** GPU support ends at TF 2.10; TF
> 2.11+ needs WSL2 or is CPU-only. We deliberately stay on 2.10 to keep the
> whole agent on one OS (see D-011). If the project ever outgrows 2.10, moving
> to WSL2 + modern TF is the escape hatch — at the cost of bridging capture and
> input across the Windows/Linux boundary.

---

## Why the reference's capture/memory code is not reused

CS2 runs on **Source 2**. The reference study targeted CSGO and relied on
techniques that are dead or changed on Source 2:

- **Screen capture** used legacy win32 `BitBlt` (`screen_input.py`). We use
  `mss` full-screen grab + crop instead (`DECISIONS.md` **D-010**).
- **Memory inference** (RAM offsets via `pymem`/hazedumper, GSI) is dead on
  Source 2. It is dropped from the core stack. This is why **spectating** —
  which would need that pipeline rebuilt — is quarantined as an optional,
  late stretch track and must never block the critical path (**D-002**).

The hard problem for us is **frame/input synchronization**, not capture itself.

---

## Repo layout

```
Agentic-CS2/
├── environment.yml     # conda env (the pinned Python stack)
├── src/
│   ├── __init__.py
│   └── smoke_test.py   # Issue #1 acceptance: verifies the env imports
├── CLAUDE.md           # lean auto-loaded project map + gates + rules
├── DECISIONS.md        # settled directions + why (read before proposing)
├── PROJECT_ISSUES.md   # milestones & issues; progress tracker
└── README.md           # this file
```

## Scope guards (full rationale in `DECISIONS.md`)

- Data creation is the hardest part, not the modeling.
- Self-recording is the spine; spectating is optional and late.
- Action labels are free from recording; detection labels are a separate
  data problem (their own milestone, M3).
- The radar does **not** show enemies — detection comes from the FPV image.
- The probability panel is **hand-authored**, not learned.
- One map end-to-end before generalizing.
- **Offline / local bot servers only — never online matchmaking** (D-007).
