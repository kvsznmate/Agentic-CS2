# DATA_FORMAT.md — Agentic-CS2 on-disk recording schema

**Status: v5, revised 2026-08 (D-033).** This is the authoritative
schema for self-recorded session files. Every loader, trainer, and inspection
tool reads this format; change it only by bumping `schema_version` and updating
this file in the same commit (per the living-docs rule in CLAUDE.md).

**Circular radar mask (2026-08, D-039) — NOT a schema bump.** The stored `radar`
array now has its CORNERS FORCED TO BLACK: a filled circle (in 128×128 space, centre
~(67,67), radius 63) is kept and everything outside it is zeroed, at capture time in
`grab_with_radar`. Reason: the square radar crop unavoidably includes game SCENE in its
corners (walls/floor outside the round minimap), and those pixels CHANGE as the player
moves, so a model could learn from them as a spurious signal; masking makes the corners
constant (black) so they carry no information. Like the D-038 re-crop, this changes only
PIXEL VALUES, not the array's shape or dtype (still 128×128×3 uint8 BGR) and no reader
behaves differently, so `schema_version` stays **5** — consistent with D-038's rule that a
bump is for shape/dtype/contract changes, not pixel-content changes. The mask geometry is
recorded in each file's `geom` (e.g. `... -> 128x128 BGR, circular mask c(67,67) r61`) and
lives in `capture_config.RADAR_MASK_*`. It is applied at CAPTURE and baked in permanently;
older v5 files (and all v1–v4) simply have unmasked radars — trust each file's `geom`.

**Radar-rectangle change (2026-08, D-038) — NOT a schema bump.** The radar is now
recorded CENTERED (see D-038), and the radar SOURCE rectangle was re-measured from
`L10 T10 W260 H260` to **`L28 T32 W320 H320`** to fit the centered disc, which the old
box clipped. This changes only the pixel region cropped and its `geom` stamp — the
stored `radar` array is still 128×128 BGR, same shape and dtype — so `schema_version`
stays **5** and no array contract changed. Consequence: v5 files recorded before and
after this change carry DIFFERENT `geom` rectangles though both say v5. This is exactly
why `geom` is stamped per file: always trust a file's own `geom`, never a constant.
Sessions recorded with the old rectangle AND the non-centered radar must be re-recorded
(they were confirmed disposable when this change was made).

**v5 vs v4 (what changed and why):** v5 adds four per-frame GSI **state features**
— own-player state the model can CONDITION ON, sampled from the same GSI stream
as the v4 alive flag (D-033): **`health`** (uint8 0–100), **`active_weapon`**
(fixed-width string, sentinel `""`), **`ammo_clip`** and **`ammo_reserve`**
(int16, sentinel **`-1`** for weapons with no ammo concept like knife/C4, or
unknown). These are **model INPUTS, not action labels**: they belong on the `X`
(input) side alongside the frame, so a future combat sub-policy (#12) can learn
e.g. "clip empty → the human reloads" or "low HP → plays back". They are NOT part
of the action vector `Y` (which stays keys/clicks/mouse). The loader exposes them
via `load_state_features()`, deliberately separate from `get_batch`'s action `Y`.
**Sentinels are load-bearing:** health/ammo of `0` are REAL states (dead / empty
clip), so "absent" must be a distinct value — hence `-1` for ammo and the
disambiguation of health-0 via the `alive` flag. **No current model reads these**
(the D-027 movement baseline is unchanged); v5 records them now because they are
free from the GSI stream already sampled and can only be captured live. The FPV,
radar, alive, and round_phase fields are unchanged from v4.

**v4 vs v3 (what changed and why):** v4 adds two per-frame fields sampled from
CS2 Game State Integration (GSI) at capture time (D-030/D-031): **`alive`** (uint8
0/1 — the GSI own-POV alive flag: health>0 AND not spectating) and
**`round_phase`** (fixed-width string — the coarse freezetime/live/over phase, or
the sentinel `"?"` when GSI carried none). These let non-gameplay frames (dead,
spectating a teammate, menu/warmup) be filtered out downstream with an
engine-authoritative signal instead of the radar-variance heuristic alone (issue
#21). The FPV `frames` and `radar` arrays are unchanged from v3. The GSI listener
is sampled once per frame off the capture thread; the alive flag is
forward-filled between GSI updates (GSI is throttled/event-driven and coarser
than the frame rate), and the recorder does not begin a session until GSI's first
POST, so every recorded frame carries a real alive value. See D-031 for the alive
rule, the liveness gate, and the known staleness limitation.

**Back-compat (v4):** v1/v2/v3 sessions remain valid and readable; they simply
have no `alive`/`round_phase` arrays, so the loader's gameplay filter
(`use_gameplay_filter=True`) does not apply to them (those frames are not
filtered by it rather than dropped). All FPV and radar inputs still work on
v1/v2/v3.

**v3 vs v2 (what changed and why):** v2 stored only the 150x270 FPV frame per
tick, and "the radar" was a sub-rectangle cropped from that downscaled frame.
On-machine that radar crop was too low-res to read self-position — the #7 finding.
v3 adds a SEPARATE, higher-resolution **`radar`** array per frame: a crop of the
CS2 minimap taken from the FULL-resolution grab BEFORE the FPV downscale, resized
to 128x128 (D-024). The FPV `frames` array is unchanged (still 150x270). Both
feeds come from ONE screen grab (`Capture.grab_with_radar`), so they are
inherently synchronized. Everything else about the v2 container — the chunked
session folder, crash-safety, manifest — is unchanged; `radar` is just another
index-aligned per-frame array. (v4 then added `alive`/`round_phase` the same way
— see the top of this file.)

**Back-compat:** v1 (single `.npz`, FPV only) and v2 (chunked, FPV only) remain
valid and readable. They simply have no `radar` array, so the loader's radar
input is unavailable for them and raises a clear error rather than silently
serving an old low-res FPV crop. All FPV inputs still work on v1/v2.

**v2 vs v1 (recap):** v1 stored a whole session as one `.npz` (unbounded memory,
whole-session loss on crash). v2 made a session a **folder of chunk files**
flushed periodically (bounded memory; a crash loses at most one in-progress
chunk — Issue #4, D-018). The per-frame array schema inside each chunk was
byte-for-byte v1; only the container changed.

Related decisions: **D-012** (FPV capture geometry: fullscreen 1920x1080,
full-frame crop, 150x270 FPV input), **D-024** (two-resolution capture: separate
high-res radar crop), **D-031** (per-frame GSI alive/round_phase + recorder
integration), **D-015** (action fields: keys/clicks + raw mouse deltas),
**D-016** (loop rate), **D-017** (v1 array schema), **D-018** (chunked sessions),
**D-019** (threaded chunk writes), **D-021** (whole-session held-out split),
**D-022** (crashed sessions excluded at discovery), **D-026** (radar-variance
keep-mask, the secondary hygiene layer). This file is the schema those decisions
produce.

---

## Session structure (v2/v3: a folder of chunks)

A **session** is a folder under `data/recordings/`, named
`session_YYYYMMDD_HHMMSS/` (or a caller-supplied stub). Inside:

```
data/recordings/session_20260810_131408/
  manifest.json          # session-level metadata + chunk list (see below)
  chunk_00000.npz        # first chunk of frames+radar+actions
  chunk_00001.npz        # next chunk
  ...
  chunk_00042.npz.tmp.npz # (only if a crash happened mid-write; ignored by loaders)
  keep_mask.npz          # (OPTIONAL sidecar, D-026) blank/no-radar frame mask
  clean_report.json      # (OPTIONAL sidecar, D-026) human-readable mask summary
```

- Each `chunk_NNNNN.npz` holds a contiguous slice of the session's frames in the
  **per-frame array schema below**. Chunks are ordered by their zero-padded
  index; concatenating them in order reconstructs the full session, and the
  sync/index-alignment guarantee holds *within* each chunk.
- Each chunk is written to `chunk_NNNNN.tmp.npz` then atomically renamed to
  `chunk_NNNNN.npz` on success. A leftover `.tmp.npz` marks a chunk whose write
  was interrupted by a crash — loaders ignore it (only manifest-listed chunks are
  read); the rest of the session is intact.
- `manifest.json` records session-level info and the authoritative chunk list.
- Written with `np.savez_compressed`; read with `np.load(path, allow_pickle=False)`.
  **`allow_pickle=False` is required** — the format uses only plain arrays.
- **Back-compat:** a bare `session_*.npz` file (v1) is still valid — treat it as a
  single-chunk, FPV-only session with no manifest.

### manifest.json

```json
{
  "schema_version": 5,
  "session": "session_20260810_131408",
  "geom": "fullscreen 1920x1080 -> crop L0T0W1920H1080 -> FPV 270x150 BGR; radar src L28T32W320H320 -> 128x128 BGR, circular mask c(67,67) r61",
  "loop_fps_target": 15,
  "chunks": ["chunk_00000.npz", "chunk_00001.npz"],
  "total_frames": 4833,
  "complete": true
}
```

`complete` is `false` while recording and set `true` on clean shutdown — so a
session interrupted by a crash is identifiable (its manifest says
`complete: false`, or lists no chunks). Discovery excludes such sessions (D-022).
The per-chunk `schema_version`/`geom`/`loop_fps_target` fields (below) are also
written inside each chunk, so a chunk is self-describing in isolation.

### keep_mask.npz — OPTIONAL dataset-hygiene sidecar (D-026)

A session folder MAY contain a `keep_mask.npz` + `clean_report.json`, written by
`python -m src.clean_session`. These are **sidecars**: they never alter the
chunks, and a session is complete and fully usable without them. The mask marks
**blank/no-radar** frames (buy menu, halftime, dead/spectate screen) — frames
whose radar crop is near-uniform and whose actions are decoupled from map
position — so training can exclude them.

`keep_mask.npz` arrays:

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `keep` | (N,) | bool | `True` = keep (gameplay), `False` = blank/no-radar. **Index-aligned to the session's concatenated per-frame arrays** — row `i` corresponds to frame `i` of `frames`/`radar`/`keys`/… (same alignment rule as every per-frame array). |
| `variance` | (N,) | float32 | Per-frame radar grayscale variance the cut was applied to (stored so the threshold can be re-applied at a different value without recomputing). |
| `threshold` | () | float32 | The variance cut used: `keep = variance > threshold`. |
| `schema_version` | () | int | Mask format version (currently `1`), independent of the recording `schema_version`. |
| `source_frames` | () | int | The `N` the mask was built for. A loader compares this to the session length and IGNORES a mask whose `source_frames` differs (a stale mask, e.g. after re-recording) rather than misaligning frames. |

How the cut is chosen: `clean_session.py` pools per-frame radar variance across
ALL sessions and finds the blank/present split by Otsu on the log-variance
histogram (the same `_radar_variance` + `_gameplay_threshold` the #7 probe uses),
deriving ONE threshold applied to every session. It refuses to write masks if the
pooled distribution isn't clearly bimodal (unless `--force`).

How the loader uses it: **opt-in only.** `build_datasets(use_keep_mask=True)` /
`SessionDataset(use_keep_mask=True)` exclude `keep == False` frames by leaving
them out of the global index (a dropped frame is simply never served — the same
mechanism as the train/holdout split). The **default is off**: with
`use_keep_mask=False` (or no mask present) every frame is served exactly as
before, so the mask cannot silently change what the #7 gate or a trainer reads.

Scope limit: the cut is on RADAR variance, so it only catches frames where the
*radar* is blank. Junk with a normal radar (e.g. the buy menu open while the
minimap still renders behind it) is NOT caught by this mask and needs a separate
FPV-side signal. `python -m src.review_session` visualises the mask (dropped
frames tinted) so this class of miss can be spotted by eye.

---

## Arrays in each chunk

`N` = number of frames **in that chunk**. Every **per-frame** array has length
`N` along axis 0, and row `i` of every per-frame array corresponds to the **same
tick** — this index-alignment IS the frame/action synchronization the M0 gate
proved (#3), and it now extends to the `radar` array too (same tick, same grab).
Do not sort or filter one array without the others.

### Per-frame arrays (length N)

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `frames` | (N, 150, 270, 3) | uint8 | The FPV game image per tick. **BGR** (OpenCV-native, D-012). 150 high × 270 wide. The full downscaled screen; used for the detection/aim FPV models. |
| `radar` | (N, 128, 128, 3) | uint8 | **v3 (D-024).** The CS2 minimap per tick, cropped from the FULL-resolution grab (source rectangle in `geom`) BEFORE the FPV downscale, resized to 128×128. **BGR.** Square, undistorted (square source → square target). **Corners masked to black (v5+, D-039):** a circular mask (centre ~(67,67), radius 61 in 128-space) zeroes the game-scene corners outside the round minimap, so only the disc carries signal; exact mask in `geom`. Used for the navigation/radar gate (#7). NOT a crop of `frames`; a separate higher-res image. Absent in v1/v2. |
| `timestamps` | (N,) | float64 | `time.perf_counter()` taken immediately after each frame grab — the alignment anchor. Seconds, monotonic, arbitrary origin. |
| `keys` | (N, 11) | uint8 | 0/1 held-state of the 11 logged keys per tick, in the fixed order given by `key_names`. |
| `lclick` | (N,) | uint8 | 0/1 left mouse button held this tick. |
| `rclick` | (N,) | uint8 | 0/1 right mouse button held this tick. |
| `dx` | (N,) | int32 | Raw mouse movement (device units) accumulated over the interval **ending** at this frame. +x = physical right. (D-015: relative deltas, not absolute view angle.) |
| `dy` | (N,) | int32 | Raw mouse movement, vertical. +y = physical down. |
| `alive` | (N,) | uint8 | **v4 (D-031, rule corrected in D-032).** GSI own-POV alive flag: `1` iff at this frame CS2's `player` block was OUR OWN POV (`player.steamid == provider.steamid`) AND reported `health > 0`. So dead frames AND frames spectating a living teammate (foreign steamid) are `0`. Sampled once per frame from the GSI listener and forward-filled between GSI updates. Used by the loader's authoritative gameplay filter (issue #21). Absent in v1/v2/v3. |
| `round_phase` | (N,) | str (`<U16`) | **v4 (D-031).** Coarse GSI round phase at this frame (`freezetime`/`live`/`over`), or the sentinel `"?"` when the GSI update carried no phase. Fixed-width (16). Note: only the coarse phase is available from live solo play — the exact round-time countdown is observer-only (D-030), a constraint on M5. Absent in v1/v2/v3. |
| `health` | (N,) | uint8 | **v5 (D-033).** Own-player health 0–100 from GSI, forward-filled between updates. A model INPUT (state to condition on), not an action label. `0` means dead OR no-local-state; disambiguate via `alive` (False in both cases, but `alive` distinguishes them from live play). Absent in v1–v4. |
| `active_weapon` | (N,) | str (`<U24`) | **v5 (D-033).** GSI active weapon name (e.g. `weapon_ak47`), or sentinel `""` when unknown/menu. Fixed-width (24). A model INPUT. Absent in v1–v4. |
| `ammo_clip` | (N,) | int16 | **v5 (D-033).** Rounds in the active weapon's clip, or sentinel **`-1`** when the weapon has no ammo concept (knife/C4) or ammo is unknown. `0` is a REAL state (empty clip) and is preserved distinct from `-1`. A model INPUT. Absent in v1–v4. |
| `ammo_reserve` | (N,) | int16 | **v5 (D-033).** Reserve/spare ammo for the active weapon as GSI reports it, sentinel **`-1`** as for `ammo_clip`. A model INPUT. Absent in v1–v4. |

Both `frames` and `radar` are captured in the SAME `grab_with_radar()` call, so
they share the tick's single `timestamps[i]` anchor — the two feeds cannot drift
relative to each other. The `alive`/`round_phase` sample is taken on that same
tick (a non-blocking read of the GSI listener's latest-value slot), so it is
index-aligned to the frame like every other per-frame array. It reflects the most
recent GSI state as of that frame, forward-filled since GSI updates arrive
coarser than the frame interval (D-031).

### Metadata arrays (fixed-size, not length N)

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `key_names` | (11,) | str (`<U…`) | The key label for each column of `keys`, in order: `w, a, s, d, space, ctrl, shift, 1, 2, 3, r`. Always read column meaning FROM THIS ARRAY, never assume the order. |
| `schema_version` | () scalar | int | Format version. **5** for this document (chunked FPV + radar + GSI alive/round_phase + GSI state features health/weapon/ammo). v4 = through alive/round_phase (no state features); v3 = FPV + radar, no GSI; v2 = FPV only; v1 = legacy standalone file. A loader should check this and refuse/adapt on an unknown value. |
| `geom` | (…) | str | Human-readable capture-geometry stamp, now including BOTH the FPV crop and the **radar source rectangle + output size**, e.g. `"fullscreen 1920x1080 -> crop L0T0W1920H1080 -> FPV 270x150 BGR; radar src L28T32W320H320 -> 128x128 BGR, circular mask c(67,67) r61"`. Records the conditions the frames were captured under so a file is self-describing. |
| `loop_fps_target` | () scalar | int | The loop's target FPS at record time (D-016: 15). The REAL rate is derivable from `timestamps`. |

---

## The radar source rectangle (how the `radar` array is produced)

The `radar` array is not a crop of `frames`. It is produced at capture time
(`Capture.grab_with_radar`, D-024) as:

1. one full-resolution grab of the monitor (1920×1080),
2. crop the minimap source rectangle **`(RADAR_SRC_LEFT, RADAR_SRC_TOP,
   RADAR_SRC_WIDTH, RADAR_SRC_HEIGHT)`** from that full-res image
   (currently `L=28, T=32, W=320, H=320`, a square tight on the CENTERED minimap
   disc, measured via `python -m src.capture --radar-calibrate`; see D-038),
3. resize that crop to **`RADAR_OUT_HW`** (currently 128×128) with `INTER_AREA`,
4. (D-039) apply the circular mask `RADAR_MASK` — zero every pixel outside the disc
   (centre `RADAR_MASK_CENTER` ~(67,67), radius `RADAR_MASK_RADIUS` 61) so the
   game-scene corners are black. Baked into the stored array at capture.

The exact rectangle and output size live in `src/capture_config.py` and are
stamped into each file's `geom`. They are **machine-specific** (they depend on the
CS2 radar HUD scale, the centered-radar cvars, and the monitor), so a recording made on a different setup
carries its own `geom`; always trust the file's `geom`, not a hardcoded constant,
when interpreting old data. The square source → square target keeps the minimap
undistorted, the same principle D-012 applies to the FPV.

---

## The action vector (how to read an action for frame i)

The "action label" for frame `i`, as consumed by training, is assembled from the
per-frame arrays at index `i`:

```
action_i = {
    "keys":   keys[i]            # 11 binary values, order per key_names
    "lclick": lclick[i]          # binary
    "rclick": rclick[i]          # binary
    "dx":     dx[i]              # int, raw mouse x since previous frame
    "dy":     dy[i]              # int, raw mouse y since previous frame
}
```

There is no separate "action vector" array on disk — it is composed from the
columns above. The loader (#6) concatenates these into a single 15-column vector
`[keys(11), lclick, rclick, dx, dy]`; the canonical source is these named arrays.
Keeping them separate on disk means the layout can change without rewriting files.

---

## The two inputs the loader serves (v3)

From these arrays the loader (`data_loader.py`) serves either feed:

- **FPV inputs** — a crop of `frames`: `"full"` (whole 150×270), `"centre"`, or a
  caller-supplied `(top, left, h, w)` rectangle. For the detection/aim models
  (#10/#11).
- **Radar input** — the stored `radar` array served directly (`"radar"`), at
  128×128. For the navigation/radar gate (#7). On a v1/v2 session this raises a
  clear error, because those files have no `radar` array.

---

## Why the FULL FPV frame is stored (and the radar is separate)

1. **The FPV frame stays whole** so the detection/aim sub-crops (centre, etc.)
   are tuned downstream against real data (#11), not baked in at capture.
2. **The radar is a separate high-res array, not an FPV crop**, because the FPV
   downscale destroys the minimap detail needed to read position (the #7 finding;
   D-024). Storing it separately, cropped from the full-res grab, is what keeps
   the navigation signal legible. The cost is ~49 KB/frame on top of the FPV's
   ~121 KB (~+40% before compression), reflected in the storage budget below.

---

## What is NOT in this format (and why)

- **No enemy-detection labels.** Self-recording yields actions for free but
  nothing about where enemies are on screen (D-003). Detection labels are a
  separate data problem (M3, #8–#9). When they exist, they attach to these files
  **by frame index** — a parallel per-frame array keyed to the same `i`, a new
  array + a schema bump, no change to existing fields.
- **No view angles / player position / health / ammo.** The study read these from
  game memory via RAM offsets — dead on Source 2 (D-002). Aim is raw mouse deltas
  only. (Note: player position is exactly what #7 tests whether we can recover
  from the `radar` pixels, since we cannot read it from memory.)
- **No audio.** Out of scope for the current architecture.

---

## Storage budget (v3)

Uncompressed per frame: FPV 150×270×3 = ~121 KB, radar 128×128×3 = ~49 KB, plus a
few bytes of action data ≈ **~170 KB/frame**. At 15 FPS that is ~2.6 MB/s, or
**~9 GB/hour** uncompressed (`np.savez_compressed` reduces this on disk, variably).
This reshapes the D-020 target: the ~20 GB first-dataset goal buys proportionally
less wall-clock play than the v2 FPV-only estimate. Record toward a usable amount;
revisit the target if training needs more.

---

## Extending the format (the rules)

1. Bump `schema_version` and document the new version in this file, same commit.
2. Prefer **adding** a new named array over changing an existing one — old loaders
   keep working, new loaders use the new field. (v3's `radar` followed this rule.)
3. Anything attached per-frame (e.g. detection labels later) MUST be length `N`
   and index-aligned to `frames`, so the sync guarantee extends to it.
4. Never introduce a field that requires `allow_pickle=True` to read.

---

## Minimal loader sketch (reference)

```python
import numpy as np

with np.load(path, allow_pickle=False) as d:
    v = int(d["schema_version"])          # 1 (v1 file), 2 (FPV folder), 3 (FPV+radar)
    frames = d["frames"]                  # (N,150,270,3) uint8 BGR
    radar = d["radar"] if "radar" in d.files else None   # (N,128,128,3) uint8 BGR (v3)
    key_names = [str(s) for s in d["key_names"]]
    keys, lclick, rclick = d["keys"], d["lclick"], d["rclick"]
    dx, dy = d["dx"], d["dy"]
    ts = d["timestamps"]
    N = frames.shape[0]
    per_frame = [keys, lclick, rclick, dx, dy, ts]
    if radar is not None:
        per_frame.append(radar)
    assert all(a.shape[0] == N for a in per_frame)
```

For a chunked v2/v3 session, read `manifest.json` and concatenate the listed
chunks in order; the snippet above is the per-chunk (or per-v1-file) view.
`src/data_loader.py` is the full loader (both formats, the split, the crops);
`src/inspect_recording.py` is the human-facing check.
```
