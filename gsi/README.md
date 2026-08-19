# GSI — CS2 Game State Integration (Step 1 feasibility)

Purpose: get an **authoritative alive/dead + phase signal from CS2 itself**, to
drive an "alive-only / gameplay-only" recording filter — replacing manual
curation and the post-hoc radar-variance (`--filter-blank`) heuristic with ground
truth from the game.

This folder is the **feasibility harness only**. It proves the signal exists
before any recorder or `DATA_FORMAT` change is made. It is intentionally
standalone and does not touch `recorder.py`.

## Files

- `gamestate_integration_agenticcs2.cfg` — the GSI subscription config CS2 reads.
- `../src/gsi_probe.py` — a standalone HTTP listener that receives CS2's GSI
  POSTs, prints alive/dead + phase transitions, and logs every payload to
  `data/gsi_probe/gsi_*.jsonl`.

## Run order (matters)

1. **Start the listener first:**
   ```
   python -m src.gsi_probe
   ```
   It binds `http://127.0.0.1:3000` and waits.

2. **Install the config into CS2's cfg directory**, then **fully restart CS2**:
   - Typical path (verify on your machine):
     `<Steam>\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg\`
     (CS2 still uses the legacy `csgo` folder name under the hood.)
   - Find `<Steam>` via registry `HKCU\Software\Valve\Steam` → `SteamPath`.
   - **Save the file WITHOUT a UTF-8 BOM** or CS2 silently ignores it. (If you
     copy it as-is it's already BOM-free; be careful if you re-save it in an
     editor that adds a BOM.)

3. **Load a bot game / offline server.** Take damage, die, spectate a teammate,
   go through freezetime/warmup. Watch the listener's live readout.

4. **Ctrl-C** to stop. It prints a feasibility summary and the JSONL log path.

## What a PASS looks like

- `LOCAL player_state seen: True` — the local alive flag exists.
- `health field seen: True`, and health tracks reality (full alive, 0 on death).
- `deaths observed:` increments when you die.
- Update cadence is reported (mean/median/p95 gap) — compare to ~67 ms/frame at
  15 FPS. Sparse GSI updates mean the alive flag is coarse and must be
  **forward-filled** between frames, not treated as per-frame precise.

## Known caveats / open questions this harness is meant to answer

- **CS2 vs CSGO field shape.** The Valve GSI docs are CSGO-branded. If payloads
  arrive but `LOCAL player_state` isn't seen, the CS2 field nesting differs —
  inspect the JSONL and adjust `_extract()` in `gsi_probe.py`.
- **Playing vs spectating.** When you die and spectate, GSI's `player` block may
  refer to the observed player (look for `observer_slot`). The harness flags
  this; the exact CS2 behaviour is what we're here to observe, since it's the
  basis of "don't record spectating as gameplay."
- **Sync.** GSI is push/throttled, not frame-locked. The listener stamps every
  update with `time.perf_counter()` — the **same clock** `recorder.py` stamps
  frames with — so alignment is possible later. Folding GSI into the recorder is
  a separate design step (the capture loop is deliberately single-threaded;
  GSI is push-based), to be decided only after this feasibility check passes.

## After this passes

Next steps (NOT done here): decide the `DATA_FORMAT` change (a per-frame
`alive`/`round_phase` field, issue #5), and how to fold the GSI stream into
recording with frame-sync. Those are real decisions to record in `DECISIONS.md`
once feasibility is confirmed.
