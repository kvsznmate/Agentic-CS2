"""gsi_probe.py — CS2 GSI feasibility harness (Step 1), now a thin layer on the
shared listener.

WHAT CHANGED (D-031): the HTTP server + payload parsing that used to live here
were extracted to `src/gsi_listener.py` so the recorder and this probe share ONE
definition of "receive + parse GSI." This file is now just the FEASIBILITY
HARNESS on top of that listener: it prints alive/dead + weapon + phase changes
live, measures GSI update cadence against the ~15 FPS frame rate, and logs every
payload to JSONL for later inspection. Operator-facing behaviour is unchanged.

The QUESTION this answered (now CONFIRMED on-machine, D-030): does CS2 emit the
LOCAL player's alive/dead state via GSI on this machine, reliably enough to drive
an alive-only recording filter, and does it distinguish playing from spectating?
Feasibility passed; this harness remains for re-checking GSI on a new setup or
after a config change.

HOW TO RUN (order matters):
  1. python -m src.gsi_probe            # start THIS first (defaults to :3000)
  2. copy gamestate_integration_agenticcs2.cfg into the CS2 cfg dir, restart CS2
  3. load a map / bot game, take damage, die, spectate — watch the readout
  4. Ctrl-C to stop; a summary + a JSONL path are printed

WHAT TO LOOK FOR (the feasibility verdict):
  * "LOCAL player_state seen" must go True — the alive flag existing.
  * health should track reality: full alive, 0 on death.
  * on die-and-spectate, the readout should flip to DEAD and print SPECTATING
    (detected by player.steamid != provider.steamid — the own-POV tell, D-032;
    NOT observer_slot, which is present even in your own alive payload here).
  * update cadence vs ~67 ms/frame at 15 FPS — sparse updates mean the flag is
    coarse and is forward-filled between frames (which the recorder handles).
"""

import argparse
import json
import os
import time

from src.gsi_listener import (GsiListener, DEFAULT_HOST, DEFAULT_PORT,
                              DEFAULT_TOKEN)


# JSONL log of every received payload (with perf_counter + wall time), so a
# feasibility run leaves an inspectable artifact. Under data/ (gitignored).
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "gsi_probe")


class _ProbeHarness:
    """Stateful callback + summary for a feasibility run.

    Wraps a GsiListener with an on_update callback that (a) prints state changes
    legibly, (b) tracks cadence + death/spectate/weapon transitions, and (c) logs
    every payload to JSONL. All of this is DIAGNOSTICS around the shared listener;
    none of it is needed by the recorder.
    """

    def __init__(self, logfile):
        self.logfile = logfile
        self.t0 = time.perf_counter()
        self.count = 0
        self.first_perf = None
        self.last_perf = None
        self.gaps = []
        self.saw_local_state = False
        self.saw_health = False
        self.deaths = 0
        self.spectating_seen = False
        self.our_steamid = None
        self._last = {}  # last-seen values for change printing

    def on_update(self, state, t_perf, t_wall, payload):
        self.count += 1
        if self.first_perf is None:
            self.first_perf = t_perf
        if self.last_perf is not None:
            self.gaps.append(t_perf - self.last_perf)
        self.last_perf = t_perf

        if state.get("has_local_state"):
            self.saw_local_state = True
        if state.get("health") is not None:
            self.saw_health = True
        if state.get("steamid") and not self.our_steamid:
            self.our_steamid = state["steamid"]

        # Log the full payload + timestamps for later inspection.
        rec = {
            "t_perf": t_perf,            # SAME clock as recorder frame['t']
            "t_wall": t_wall,
            "n": self.count,
            "extract": state,
            "payload": payload,
        }
        self.logfile.write(json.dumps(rec) + "\n")
        self.logfile.flush()

        # ── live readout: only print on CHANGES, so the console is legible ──
        changed = []
        if state.get("has_local_state") and "local_state" not in self._last:
            self._last["local_state"] = True
            changed.append("LOCAL player_state: NOW SEEN")
        # Death is an ALIVE-flag transition (D-032), not merely a health change:
        # you can go alive -> spectating-a-teammate (health looks like 100 on the
        # observed player) without ever logging health 0, and that is still a death
        # for our purposes. Keying off `alive` catches both the health-0 path and
        # the straight-to-spectate path.
        h = state.get("health")
        if h != self._last.get("health"):
            if h is not None and self._last.get("health") is not None:
                changed.append(f"health {self._last.get('health')} -> {h}")
            self._last["health"] = h
        a = state.get("alive")
        if a != self._last.get("alive"):
            if self._last.get("alive") is True and a is not True:
                self.deaths += 1
                why = "spectating" if state.get("spectating") else "health 0"
                changed.append(f"*** DEATH (#{self.deaths}, {why}) ***")
            self._last["alive"] = a
        w = state.get("active_weapon")
        if w != self._last.get("weapon"):
            ammo = ""
            if state.get("ammo_clip") is not None:
                ammo = f" [{state.get('ammo_clip')}/{state.get('ammo_reserve')}]"
            changed.append(f"weapon -> {w}{ammo}")
            self._last["weapon"] = w
        # Spectating tell = steamid mismatch (own_pov False), NOT observer_slot.
        if state.get("spectating") and not self.spectating_seen:
            self.spectating_seen = True
            changed.append("SPECTATING seen (player.steamid != provider.steamid)")
        for key, label in (("round_phase", "round.phase"), ("map_phase", "map.phase")):
            if state.get(key) != self._last.get(key):
                changed.append(f"{label} -> {state.get(key)}")
                self._last[key] = state.get(key)

        if changed:
            rel = t_perf - self.t0
            a = state.get("alive")
            alive_s = ("ALIVE" if a else "DEAD") if a is not None else "?"
            print(f"[{rel:7.2f}s | #{self.count:4d} | {alive_s:5}] " + "; ".join(changed))

    def print_summary(self, dur):
        n = self.count
        print("\n" + "=" * 70)
        print("GSI FEASIBILITY SUMMARY")
        print(f"  ran {dur:.0f}s, received {n} payloads "
              f"({n/dur:.1f}/s)" if dur > 0 else f"  received {n} payloads")
        if n == 0:
            print("  RESULT: NO DATA — CS2 sent nothing. GSI not proven on this box.")
            print("    Check, in order: .cfg is in <Steam>/.../game/csgo/cfg/, saved")
            print("    WITHOUT a UTF-8 BOM, CS2 was fully restarted after adding it,")
            print("    the listener was running before launch, port 3000 not blocked.")
            print("=" * 70)
            return False
        if self.gaps:
            g = sorted(self.gaps)
            mean_gap = sum(g) / len(g)
            p50 = g[len(g) // 2]
            p95 = g[min(len(g) - 1, int(0.95 * len(g)))]
            print(f"  update cadence: mean gap {mean_gap*1000:.0f} ms, "
                  f"median {p50*1000:.0f} ms, p95 {p95*1000:.0f} ms")
            print(f"    (frames arrive ~{1000/15:.0f} ms apart at 15 FPS — compare)")
        print(f"  LOCAL player_state seen: {self.saw_local_state}")
        print(f"  health field seen:       {self.saw_health}")
        print(f"  deaths observed:         {self.deaths}")
        print(f"  spectating payload seen: {self.spectating_seen}")
        print(f"  our steamid:             {self.our_steamid}")
        print()
        if self.saw_local_state and self.saw_health:
            print("  RESULT: FEASIBLE — CS2 emits our local health/alive state. This")
            print("  can drive an authoritative alive-only filter (now wired into the")
            print("  recorder, D-031).")
            ok = True
        else:
            print("  RESULT: PARTIAL — payloads arrived but the local health/alive")
            print("  field wasn't clearly seen. Inspect the JSONL: the CS2 field")
            print("  nesting may differ from the CSGO docs. Adjust extract_state() in")
            print("  gsi_listener.py to the real shape and re-run before building on it.")
            ok = False
        print("=" * 70)
        return ok


def run(host=DEFAULT_HOST, port=DEFAULT_PORT, token=DEFAULT_TOKEN):
    os.makedirs(_LOG_DIR, exist_ok=True)
    stamp = time.strftime("gsi_%Y%m%d_%H%M%S")
    logpath = os.path.join(_LOG_DIR, stamp + ".jsonl")
    logfile = open(logpath, "w")

    harness = _ProbeHarness(logfile)
    gsi = GsiListener(host=host, port=port, token=token, on_update=harness.on_update)

    print(f"GSI feasibility listener on http://{host}:{port}")
    print(f"  Logging every payload to: {logpath}")
    print( "  Start THIS first, then (re)start CS2 with the .cfg in place.")
    print( "  Load a bot game, take damage, die, spectate. Ctrl-C to stop.\n")
    print( "  Waiting for CS2 POSTs... (nothing below = CS2 isn't sending;")
    print( "   check the .cfg is in the cfg dir, no BOM, and CS2 was restarted)\n")

    t_start = time.perf_counter()
    gsi.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        gsi.stop()
        logfile.close()

    dur = time.perf_counter() - t_start
    ok = harness.print_summary(dur)
    print(f"  full payload log: {logpath}")
    return ok


def _build_parser():
    p = argparse.ArgumentParser(
        description="CS2 GSI feasibility listener (Step 1): confirm local "
                    "alive/dead state is emitted. Thin harness over gsi_listener.")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help="bind host (must match the .cfg uri; default 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="bind port (must match the .cfg uri; default 3000)")
    p.add_argument("--token", default=DEFAULT_TOKEN,
                   help="expected auth token (must match the .cfg; informational)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    run(host=args.host, port=args.port, token=args.token)


if __name__ == "__main__":
    main()
