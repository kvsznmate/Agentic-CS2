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

The Python stack is a **faithful conda reproduction** of the reference study's
environment: **Python 3.7 · TensorFlow 2.3 · numpy 1.18 · OpenCV 4.4 · mss 7.0.1**,
on Windows. This choice (and why it is conda, not `pyproject.toml`/uv) is
recorded in `DECISIONS.md` **D-009**.

```bash
conda env create -f environment.yml
conda activate agentic-cs2
python -m src.smoke_test
```

`src/smoke_test.py` verifies every pinned dependency imports and reports the
loaded versions against what's expected. A clean run printing
`RESULT: all core imports succeeded.` satisfies Issue #1's acceptance
("fresh checkout + docs yields a working env; smoke test imports core modules").

### ⚠ GPU caveat — read before expecting GPU training

TensorFlow 2.3's only official GPU path is **CUDA 10.1 + cuDNN 7.6**, which the
conda env installs. That toolchain targets **older NVIDIA GPUs only**. On
current cards (**RTX 30/40/50-series**, compute capability 8.6+) CUDA 10.1 will
not drive the hardware and **training falls back to CPU**. If you have a modern
GPU and need GPU training, that requires modernizing the stack (modern
TensorFlow, or PyTorch, on Python 3.11) — revisit **D-009** before doing so.
`smoke_test.py` reports whether TensorFlow actually sees a GPU.

To force a CPU-only build regardless of hardware, follow the comment block at
the top of `environment.yml`.

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
