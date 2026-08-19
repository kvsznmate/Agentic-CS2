"""gsi_probe.py — CS2 Game State Integration feasibility harness (Step 1).

The QUESTION this answers, before we build anything on GSI:
  Does CS2 emit the LOCAL PLAYER'S alive/dead state (player_state.health) via
  Game State Integration on THIS machine, reliably enough to drive an
  "alive-only" recording filter — and does it distinguish playing from
  spectating? If yes, this replaces manual curation and the post-hoc radar-
  variance filter with an authoritative, engine-level signal. If GSI turns out
  dead/broken on this CS2 install, we fall back to the visual heuristic.

WHAT THIS IS AND ISN'T:
  * IS: a standalone HTTP endpoint that receives CS2's GSI POSTs, timestamps
    each with time.perf_counter() — the SAME clock recorder.py stamps frames
    with (record["t"]) — prints alive/dead + phase transitions live, and logs
    every payload to JSONL so you can inspect timing/fields afterward.
  * ISN'T: wired into the recorder. The recorder is a single synchronous loop by
    deliberate design (D-015, "way one"); GSI is push-based HTTP and cannot go
    into that loop without threading, which is a real integration decision for
    LATER. Step 1 only needs to prove the signal exists, which needs no recorder
    or DATA_FORMAT change. Keep this standalone.

WHY perf_counter TIMESTAMPS MATTER: GSI updates are event-driven and throttled
(see the .cfg), NOT frame-locked. To EVENTUALLY align an alive-flag to frame N,
we need GSI updates on the same monotonic clock as the frames. Stamping here with
perf_counter both (a) proves that alignment path is available and (b) lets this
harness measure GSI update cadence vs the ~15 FPS capture rate — the real
question for whether per-frame alive-labelling is even meaningful.

HOW TO RUN (order matters):
  1. python -m src.gsi_probe            # start THIS first (defaults to :3000)
  2. copy gamestate_integration_agenticcs2.cfg into the CS2 cfg dir, restart CS2
  3. load a map / bot game, take damage, die, spectate — watch the readout
  4. Ctrl-C to stop; a summary + a JSONL path are printed

WHAT TO LOOK FOR (the feasibility verdict):
  * "LOCAL player_state seen" must go True — that's the alive flag existing.
  * health should track reality: full when alive, 0 (or block changes) on death.
  * on death while solo, note whether player_state drops out or reports the
    spectated player — this is the playing-vs-spectating distinction we need.
  * update cadence: how many updates/sec, and the gaps — sparse updates mean the
    alive flag is coarse and must be forward-filled between frames, not assumed
    per-frame precise.
"""

import argparse
import json
import os
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# JSONL log of every received payload (with our perf_counter + wall time), so the
# feasibility run leaves an inspectable artifact. Lives under data/ (gitignored).
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "gsi_probe")


class _State:
    """Mutable run state shared across requests (ThreadingHTTPServer => 1 handler
    at a time in practice for a single client, but keep it simple and additive).
    """

    def __init__(self, token, logfile):
        self.token = token
        self.logfile = logfile
        self.t0 = time.perf_counter()
        self.count = 0
        self.first_perf = None
        self.last_perf = None
        self.gaps = []               # inter-update gaps (s), for cadence stats
        self.saw_local_state = False
        self.saw_health = False
        self.last_health = None
        self.last_alive = None
        self.last_weapon = None
        self.last_round_phase = None
        self.last_map_phase = None
        self.deaths = 0
        self.spectating_seen = False
        self.our_steamid = None


def _extract(payload):
    """Pull the fields we care about out of a GSI payload dict.

    Returns a dict with health/alive/phases/steamid where available, plus a
    'has_local_state' flag. Defensive: GSI blocks are optional and their exact
    nesting may differ slightly on CS2 vs the CSGO docs, so we .get() everywhere
    and never assume a key exists. If the shape is different than expected, the
    raw payload is still logged to JSONL for inspection.
    """
    out = {"has_local_state": False, "health": None, "alive": None,
           "round_phase": None, "map_phase": None, "steamid": None,
           "spectating": None, "active_weapon": None,
           "ammo_clip": None, "ammo_reserve": None}

    provider = payload.get("provider") or {}
    out["steamid"] = provider.get("steamid")

    player = payload.get("player") or {}
    # When spectating/observing, 'player' refers to the OBSERVED player and
    # carries an 'observer_slot'; our own POV while alive typically does not.
    # This is our playing-vs-spectating tell (see docs: allplayers/observer-only
    # fields). We surface it rather than interpret it hard, since exact CS2
    # behaviour is what this harness is meant to observe.
    if "observer_slot" in player:
        out["spectating"] = True
    player_state = player.get("state") or {}
    if player_state:
        out["has_local_state"] = True
        hp = player_state.get("health")
        out["health"] = hp
        if hp is not None:
            out["alive"] = hp > 0

    # player_weapons is a dict weapon_0/weapon_1/... ; the one with state
    # 'active' is what we're holding. Ammo fields are absent for knife/grenades,
    # so .get() and leave None when not present. Defensive against CS2 vs CSGO
    # shape differences (the raw payload is logged regardless).
    weapons = player.get("weapons") or {}
    for _slot, w in weapons.items():
        if not isinstance(w, dict):
            continue
        if w.get("state") == "active":
            out["active_weapon"] = w.get("name")
            out["ammo_clip"] = w.get("ammo_clip")
            out["ammo_reserve"] = w.get("ammo_reserve")
            break

    rnd = payload.get("round") or {}
    out["round_phase"] = rnd.get("phase")
    mp = payload.get("map") or {}
    out["map_phase"] = mp.get("phase")
    return out


class _Handler(BaseHTTPRequestHandler):
    # Injected by the server factory.
    state: _State = None

    def log_message(self, *a):
        pass  # silence default per-request stderr logging

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        t_perf = time.perf_counter()
        # ALWAYS respond 2XX quickly; GSI won't send the next update until it gets
        # one, and treats non-2XX as a failure (drops delta computation).
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()

        st = self.state
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            print(f"  [warn] non-JSON POST ({len(raw)} bytes) — ignoring")
            return

        # Optional token check (the cfg sets auth.token). Don't hard-fail on it in
        # feasibility mode; just note a mismatch once.
        got_token = (payload.get("auth") or {}).get("token")
        if st.token and got_token is not None and got_token != st.token:
            print(f"  [warn] auth token mismatch (got {got_token!r})")

        st.count += 1
        if st.first_perf is None:
            st.first_perf = t_perf
        if st.last_perf is not None:
            st.gaps.append(t_perf - st.last_perf)
        st.last_perf = t_perf

        info = _extract(payload)
        if info["steamid"] and not st.our_steamid:
            st.our_steamid = info["steamid"]

        # Log the full payload + our timestamps for later inspection.
        rec = {
            "t_perf": t_perf,               # SAME clock as recorder frame['t']
            "t_wall": datetime.now().isoformat(timespec="milliseconds"),
            "n": st.count,
            "extract": info,
            "payload": payload,
        }
        st.logfile.write(json.dumps(rec) + "\n")
        st.logfile.flush()

        # ── live readout: only print on CHANGES, so the console is legible ──
        changed = []
        if info["has_local_state"]:
            if not st.saw_local_state:
                st.saw_local_state = True
                changed.append("LOCAL player_state: NOW SEEN")
            if info["health"] is not None:
                st.saw_health = True
                if info["health"] != st.last_health:
                    changed.append(f"health {st.last_health} -> {info['health']}")
                    # death edge
                    if (st.last_alive is True) and (info["alive"] is False):
                        st.deaths += 1
                        changed.append(f"*** DEATH (#{st.deaths}) ***")
                    st.last_health = info["health"]
                    st.last_alive = info["alive"]
        if info["active_weapon"] != st.last_weapon:
            ammo = ""
            if info["ammo_clip"] is not None:
                ammo = f" [{info['ammo_clip']}/{info['ammo_reserve']}]"
            changed.append(f"weapon -> {info['active_weapon']}{ammo}")
            st.last_weapon = info["active_weapon"]
        if info["spectating"] and not st.spectating_seen:
            st.spectating_seen = True
            changed.append("SPECTATING/observer payload seen (observer_slot present)")
        if info["round_phase"] != st.last_round_phase:
            changed.append(f"round.phase -> {info['round_phase']}")
            st.last_round_phase = info["round_phase"]
        if info["map_phase"] != st.last_map_phase:
            changed.append(f"map.phase -> {info['map_phase']}")
            st.last_map_phase = info["map_phase"]

        if changed:
            rel = t_perf - st.t0
            alive_s = ("ALIVE" if info["alive"] else "DEAD") if info["alive"] is not None else "?"
            print(f"[{rel:7.2f}s | #{st.count:4d} | {alive_s:5}] " + "; ".join(changed))


def _make_server(host, port, state):
    handler = type("_BoundHandler", (_Handler,), {"state": state})
    return ThreadingHTTPServer((host, port), handler)


def run(host="127.0.0.1", port=3000, token="agentic_cs2_local"):
    os.makedirs(_LOG_DIR, exist_ok=True)
    stamp = time.strftime("gsi_%Y%m%d_%H%M%S")
    logpath = os.path.join(_LOG_DIR, stamp + ".jsonl")
    logfile = open(logpath, "w")
    state = _State(token=token, logfile=logfile)
    server = _make_server(host, port, state)

    print(f"GSI feasibility listener on http://{host}:{port}")
    print(f"  Logging every payload to: {logpath}")
    print( "  Start THIS first, then (re)start CS2 with the .cfg in place.")
    print( "  Load a bot game, take damage, die, spectate. Ctrl-C to stop.\n")
    print( "  Waiting for CS2 POSTs... (nothing below = CS2 isn't sending;")
    print( "   check the .cfg is in the cfg dir, no BOM, and CS2 was restarted)\n")

    t_start = time.perf_counter()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        logfile.close()

    # ── summary / feasibility verdict ──
    dur = time.perf_counter() - t_start
    n = state.count
    print("\n" + "=" * 70)
    print("GSI FEASIBILITY SUMMARY")
    print(f"  ran {dur:.0f}s, received {n} payloads "
          f"({n/dur:.1f}/s)" if dur > 0 else f"  received {n} payloads")
    if n == 0:
        print("  RESULT: NO DATA — CS2 sent nothing. GSI not proven on this box.")
        print("    Check, in order: .cfg is in <Steam>/.../game/csgo/cfg/, saved")
        print("    WITHOUT a UTF-8 BOM, CS2 was fully restarted after adding it,")
        print("    the listener was running before launch, port 3000 not blocked.")
        print(f"  (empty log at {logpath})")
        print("=" * 70)
        return False

    if state.gaps:
        g = sorted(state.gaps)
        mean_gap = sum(g) / len(g)
        p50 = g[len(g)//2]
        p95 = g[min(len(g)-1, int(0.95*len(g)))]
        print(f"  update cadence: mean gap {mean_gap*1000:.0f} ms, "
              f"median {p50*1000:.0f} ms, p95 {p95*1000:.0f} ms")
        print(f"    (frames arrive ~{1000/15:.0f} ms apart at 15 FPS — compare)")
    print(f"  LOCAL player_state seen: {state.saw_local_state}")
    print(f"  health field seen:       {state.saw_health} "
          f"(last={state.last_health})")
    print(f"  deaths observed:         {state.deaths}")
    print(f"  spectating payload seen: {state.spectating_seen}")
    print(f"  our steamid:             {state.our_steamid}")
    print()
    if state.saw_local_state and state.saw_health:
        print("  RESULT: FEASIBLE — CS2 emits our local health/alive state. This")
        print("  can drive an authoritative alive-only filter. Next: decide the")
        print("  DATA_FORMAT change (per-frame alive/phase) and how to fold GSI")
        print("  into recording (it's push/threaded vs the sync capture loop).")
        ok = True
    else:
        print("  RESULT: PARTIAL — payloads arrived but the local health/alive")
        print("  field wasn't clearly seen. Inspect the JSONL: the CS2 field")
        print("  nesting may differ from the CSGO docs. Adjust _extract() to the")
        print("  real shape and re-run before building on it.")
        ok = False
    print(f"  full payload log: {logpath}")
    print("=" * 70)
    return ok


def _build_parser():
    p = argparse.ArgumentParser(
        description="CS2 GSI feasibility listener (Step 1): prove local alive/dead "
                    "state is emitted, before building recorder/schema on it.")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind host (must match the .cfg uri; default 127.0.0.1)")
    p.add_argument("--port", type=int, default=3000,
                   help="bind port (must match the .cfg uri; default 3000)")
    p.add_argument("--token", default="agentic_cs2_local",
                   help="expected auth token (must match the .cfg; informational)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    run(host=args.host, port=args.port, token=args.token)


if __name__ == "__main__":
    main()
