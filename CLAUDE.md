# CLAUDE.md — Agentic-CS2

Auto-loaded every session. Keep this file short. It is a map, not a manual — detailed rules live in the files it points to.

---

## What this project is

**Agentic-CS2** is a behavioural-cloning FPS agent for Counter-Strike 2 that splits perception into two feeds:
- a **radar / probability panel** driving navigation (where to move, which angle to hold), and
- a **first-person vision model** driving enemy detection and aim,
- combined by an **arbiter** that hands the mouse to exactly one feed per frame (enemy seen → combat drives + movement freezes; else → navigation drives).

We train our own models on our own data captured from live CS2. We build on the findings of a reference study but do **not** use its dataset, run its agent, or compare against it.

---

## Directories

- **Build here:** `C:\Users\matek_yulq090\Desktop\Agentic-CS2`
- **Reference study (READ-ONLY):** `C:\Users\matek_yulq090\Desktop\Counter-Strike_Behavioural_Cloning#code-overview` — consult for methods; never edit, never copy wholesale.

---

## Working rules (non-negotiable)

1. **Read before acting.** Use the filesystem connector to read the real current state of any file before answering or editing. Never work from memory of a file's contents.
2. **Write to disk, not to chat.** Deliverables are new/edited files in the Agentic-CS2 directory. A short illustrative snippet in chat is fine; pasting whole files instead of writing them is not.
3. **Report concretely.** After changes, say which files you touched and what changed. Point to files; don't re-paste them.

---

## The three gates (derisking order)

Work is sequenced by killing risk early. Do not build behind an unresolved gate.
1. **Capture + sync** — can we record synchronized screen+input data on CS2 at all?
2. **Radar signal** — can movement be predicted from the radar?
3. **Detection** — can we produce enemy labels and train a detector to threshold?

If a request sits behind an open gate, say so and recommend resolving the gate first.

---

## Living documents — keep these current

These files are the project's memory. Updating them is part of doing the work, not an afterthought.

| File | What it holds | When to update |
|---|---|---|
| `DECISIONS.md` | Every settled direction + one-line *why* | Whenever a direction is chosen or reversed — **append before/as you act on it** |
| `DATA_FORMAT.md` | Authoritative on-disk data schema (frame, action vector, radar crop, sync) | The moment the format is set or changed (issue #5) |
| `PROJECT_ISSUES.md` | Milestone + issue plan; progress tracker | When an issue opens/closes or the plan diverges from reality |

**Before proposing anything, check `DECISIONS.md` — do not reopen a settled question without flagging that it's settled and why.**

---

## Hard scope guards (see DECISIONS.md for rationale)

- Data creation is the hardest part, not the modeling. Treat it as first-class.
- Self-recording is the spine; **spectating is optional and late** — never let it block the critical path.
- Action labels are free from recording; **detection labels are a separate data problem.**
- The radar does **not** show enemies — detection comes from the first-person image.
- The probability panel is **hand-authored**, not learned.
- CS2 is Source 2 — legacy D3D9 capture, RAM offsets, and GSI are dead/changed. Sync is the hard part, not capture.
- One map end-to-end before generalizing.
- Offline/local bot servers only — never online matchmaking.

---

## Behaviour

Be a collaborator, not just an implementer. Push back on bad ideas, out-of-scope requests, and gate violations — candor over agreeableness. Prefer absolute, committed benchmarks set before the work. Separate verified fact from assumption; when you haven't read or measured something, say so.
