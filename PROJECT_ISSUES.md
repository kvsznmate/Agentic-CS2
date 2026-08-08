# CS Two-Feed Agent — Milestones & Issues

A behavioural-cloning FPS agent that splits perception into two feeds: a **radar/probability panel** driving navigation, and a **first-person vision model** driving enemy detection and aim, combined by an **arbiter**. We train our own models on our own data captured from the current live game (CS2). We do not use or compare against any prior work.

**The central reality of this project:** we have no dataset. Creating one is the largest risk and the first several milestones. Data collection — not modeling — is where this project succeeds or fails. The plan reflects that: it leads with capture, treats detection-labeling as its own high-risk track, and quarantines spectating as optional so it can never block the critical path.

**Ordering principle:** derisking, not components. The cheapest project-killing tests come first and gate everything expensive behind them.

**Two hard scope guards, learned the hard way:**
- **Self-recording is the spine; spectating is an optional stretch track.** Self-recording gives us our own actions for free. Spectating requires recovering a *stranger's* actions, which on CS2 means rebuilding a dead memory-inference pipeline on a new engine — potentially larger than the whole rest of the project. It must never block the critical path.
- **Action labels are free; detection labels are a second data problem.** Playing the game logs our inputs automatically, but tells us nothing about where enemies were on screen. Detection labels need their own creation method and their own milestone.

---

## Milestones

### M0 — Capture pipeline (GO/NO-GO)
**Goal:** prove we can capture clean, synchronized screen + input data on CS2 at all.
**Exit criteria:**
- Screen capture working on CS2 (mss full-screen grab + crop; the legacy D3D9 method is dead).
- Our own keyboard/mouse inputs logged and time-synced to frames.
- A short recorded session round-trips to disk and reloads intact.
**Benchmark:** a 5-minute self-recorded session yields frame-action pairs with verified alignment (input at frame N actually corresponds to screen at frame N); dropped-frame rate under a set bar.
**Kill condition:** if we can't capture synchronized data reliably, there is no project. Everything downstream depends on this. Resolve before anything else.

### M1 — Self-recorded action dataset
**Goal:** enough of our own play to train movement + aim.
**Exit criteria:**
- Recording tooling usable for extended sessions without babysitting.
- A first dataset of self-play recorded, stored, and documented (size target set in M1 issues).
- Data format finalized (frame image + action vector + radar crop region).
**Benchmark:** N hours / M frames recorded (target committed in #4 after a pilot); a loader emits (input-crop, action-label) batches; held-out split reserved.
**Scope note:** actions come free from self-recording. Detection labels do NOT — that's M3.

### M2 — GO/NO-GO: Radar carries navigation signal
**Goal:** prove the radar is extractable and predicts movement, on our data.
**Exit criteria:**
- Radar crop reliably recovers self-position from the CS2 HUD.
- Movement predictable from radar above chance on our recordings.
**Benchmark:** self-position on ≥95% of sampled frames; a linear probe beats chance on coarse movement direction (exact bar set in #7 after first measurement).
**Kill condition:** if radar shows no movement signal, the navigation path is dead — rethink before M5.

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

#### #1 Repo, environment, dependency pinning
**Labels:** `infra` · **Depends on:** —
Pin the stack (Python + capture libs + framework + numpy + OpenCV). Document setup. Note version drift can change model behaviour.
**Acceptance:** fresh checkout + docs yields a working env; smoke test imports core modules.

#### #2 [GATE] CS2 screen capture
**Labels:** `gate` `data` `infra` · **Depends on:** #1
Capture the CS2 window. The legacy D3D9 BitBlt method is dead on Source 2 — use full-screen grab (mss) + crop to the game region, output the fixed model input size. Verify against different in-game scenes.
**Acceptance:** clean, correctly-cropped frames captured live from CS2 at a usable rate.

#### #3 [GATE] Synced input logging
**Labels:** `gate` `data` · **Depends on:** #2
Log our own keyboard + mouse (including mouse deltas) and align each input to the correct captured frame. Frame/input synchronization is the hard part, not the capture. Verify alignment explicitly.
**Acceptance:** a short session produces frame-action pairs where input at frame N provably matches screen at frame N; dropped-frame rate under a committed bar. **If sync can't be made reliable, RAISE THE KILL FLAG.**

---

### M1 — Self-recorded action dataset

#### #4 Recording tooling for extended sessions
**Labels:** `data` `infra` · **Depends on:** #3
Turn the capture+log prototype into something usable for long sessions unattended: start/stop, disk management, crash resilience. Run a pilot; set the dataset size target (hours/frames) from pilot throughput.
**Acceptance:** an extended session records without babysitting; size target committed.

#### #5 Finalize data format
**Labels:** `data` · **Depends on:** #3
Lock the on-disk schema: frame image, action vector (keys, clicks, mouse dx/dy), and the radar crop region. Design for the loader and for later detection-label attachment.
**Acceptance:** documented schema; sample files validate against it.

#### #6 Record the first self-play dataset + loader
**Labels:** `data` `infra` · **Depends on:** #4, #5
Record the target volume of our own play. Build a loader emitting (input-crop, action-label) batches with configurable crops (full FPV, centre, radar). Reserve a held-out split.
**Acceptance:** dataset recorded + documented; loader unit-tested for shapes; held-out split reserved.

---

### M2 — GO/NO-GO: Radar carries navigation signal

#### #7 [GATE] Radar crop + signal sanity check
**Labels:** `gate` `perception` `navigation` · **Depends on:** #6
Fixed-coordinate crop of the CS2 radar; recover self-position. Then the go/no-go: on a few hundred of our frames, check whether WASD is predictable from the radar sequence (by eye + a quick linear probe). Measure first, then set the M2 bar.
**Acceptance:** self-position on ≥95% of a sample; a committed verdict + threshold. If below chance, RAISE THE KILL FLAG before M5.

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
Draw both paths + current mouse owner live. Then profile: two forward passes per loop ~doubles compute — hit target FPS by shrinking/slowing the navigation model as needed.
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
