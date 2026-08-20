"""gsi_listener.py — shared CS2 GSI listener (Issue #21, D-030/D-031).

A background HTTP listener that receives CS2 Game State Integration POSTs and
exposes the LATEST parsed self-state through a narrow, thread-safe slot. Both the
standalone feasibility harness (`gsi_probe.py`) and the recorder
(`recorder.py --record`) consume this ONE listener, so there is a single
definition of "how we receive and parse GSI" rather than two that could drift.

── WHY THIS SHAPE (and why it's safe for the sync-critical recorder) ─────────
The recorder's capture loop is a single synchronous loop by deliberate design
(D-015, "way one") — the whole M0 gate exists to prove input-at-frame-N matches
screen-at-frame-N, and that proof rests on the loop staying synchronous. GSI, by
contrast, is PUSH-based, throttled, event-driven HTTP: CS2 decides when to POST,
and you cannot block the capture loop waiting on it.

So this listener follows the EXACT pattern raw_mouse.RawMouseListener already
uses for the mouse: a producer on its own thread, and a narrow interface the
loop samples once per frame. The only shared state is a lock-protected "latest
value" slot. The listener thread touches NOTHING the capture loop owns — not the
capture, not the frame timestamps, not the mouse listener. This is the same
threading-confined-to-I/O reasoning that made D-019 (threaded chunk writes) safe:
the sync path stays fully synchronous; only receipt of an external signal is
off-thread. `read_latest()` is a snapshot (it does NOT drain, unlike the mouse's
read_and_reset) because GSI is a STATE, not a stream of increments to sum — at
frame N you want "what is the current alive state," which is the latest value.

── WHAT IT DECIDES vs WHAT IT LEAVES TO THE CALLER ───────────────────────────
This module PARSES GSI and tracks the newest self-state (health, own-POV alive,
round/map phase, weapon/ammo, and the perf_counter time of the last update). It
deliberately does NOT decide dataset policy. The alive RULE — specifically that
"alive" means it is OUR OWN POV (the payload's `player.steamid` equals
`provider.steamid`) AND health>0, so spectating a living teammate after death
reads as NOT alive (D-030/D-031, corrected 2026-08-20) — is computed here because
it is about correctly interpreting the payload, not about dataset inclusion. But
WHEN to start/stop recording, how to treat frames before the first update, and
how to filter on the flag are the recorder's/loader's decisions.

NOTE (the own-POV tell, learned from real CS2 payloads): the spectating guard is
the STEAMID MATCH, not `observer_slot`. On this CS2 build `player.observer_slot`
is present in your OWN alive payload too (it is your slot number), so it is
useless as a spectating signal — an earlier version used it and mislabelled every
in-game frame as spectating. When you die and the camera moves to a teammate, GSI
swaps `player` to the OBSERVED player and `player.steamid` no longer equals
`provider.steamid` (which is always your account). That mismatch is the reliable
spectating tell; `activity` stays "playing" while you spectate, so it can't
substitute. See DECISIONS D-032.

── FORWARD-FILL IS INHERENT, NOT A FEATURE ───────────────────────────────────
GSI is throttled ("throttle" 0.1 in the .cfg — at most every 100 ms, and only on
change; "heartbeat" 10 s otherwise) while the capture loop runs ~15 FPS (~67 ms
frames). So MOST frames arrive with no fresh GSI update. read_latest() simply
returns the last received state — which IS forward-filling, for free, via the
slot. `age_seconds` (returned alongside) reports how stale that state is
(perf_counter since the last POST) so a caller CAN detect a dead/quiet GSI if it
wants; the recorder's chosen policy for staleness is documented in the recorder.

── TIMESTAMPS ────────────────────────────────────────────────────────────────
Every received POST is stamped with time.perf_counter() — the SAME monotonic
clock recorder.py stamps frames with (record["t"], via Capture.grab). That shared
clock is what makes it meaningful to associate the latest GSI state with a frame.

── STANDALONE USE ────────────────────────────────────────────────────────────
    python -m src.gsi_listener --selftest    # prove GSI reaches this process
This runs the listener alone and prints state changes, a thin echo of what the
probe does — handy to confirm CS2 is POSTing before wiring anything to it.
"""

import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# Defaults must match gsi/gamestate_integration_agenticcs2.cfg (uri/port/token).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3000
DEFAULT_TOKEN = "agentic_cs2_local"


def extract_state(payload):
    """Pull the fields we care about out of one GSI payload dict.

    Returns a dict with health / own-POV alive / phases / weapon+ammo / steamid,
    plus flags for whether a local player_state was present and whether we are
    spectating (viewing another player). Defensive: GSI blocks are optional and
    their exact nesting can differ on CS2 vs the CSGO-branded docs, so we .get()
    everywhere and never assume a key exists; a caller keeping the raw payload can
    always re-parse if a block's shape surprises us (D-030's CS2-vs-CSGO caveat).

    THE ALIVE RULE (D-030/D-031, CORRECTED in D-032 2026-08-20): `alive` is True
    iff the payload is OUR OWN POV **and** a local player_state reports health > 0.
    Own-POV is `player.steamid == provider.steamid`. This is the load-bearing
    spectating guard: when you die and spectate a living teammate, GSI's `player`
    block becomes the OBSERVED player (health > 0) with a DIFFERENT steamid, so the
    steamid check makes those frames read DEAD — keeping someone else's screen
    paired with our idle/dead inputs (poison for behavioural cloning) out of the
    dataset. `alive` is None only when we can't judge (own POV but no health field,
    e.g. the brief dead frame before the camera moves — treated as not-alive by the
    recorder).

    Why NOT `observer_slot`: on this CS2 build `observer_slot` is present in the
    OWN alive payload as well (it is the player's own slot index), so it does not
    distinguish playing from spectating and an earlier rule using it forced every
    in-game frame to alive=False. It is still surfaced below as informational.
    """
    out = {
        "has_local_state": False,
        "spectating": False,      # True = viewing ANOTHER player (steamid mismatch)
        "own_pov": False,         # True = player block is us (steamid match)
        "health": None,
        "alive": None,            # tri-valued at PARSE time: True/False/None(unknown)
        "round_phase": None,
        "map_phase": None,
        "activity": None,
        "observer_slot": None,    # informational only; NOT the spectating tell
        "active_weapon": None,
        "ammo_clip": None,
        "ammo_reserve": None,
        "steamid": None,          # provider steamid = our account
        "player_steamid": None,   # steamid of whoever the player block describes
    }

    provider = payload.get("provider") or {}
    provider_sid = provider.get("steamid")
    out["steamid"] = provider_sid

    player = payload.get("player") or {}
    player_sid = player.get("steamid")
    out["player_steamid"] = player_sid
    out["activity"] = player.get("activity")
    out["observer_slot"] = player.get("observer_slot")  # kept for diagnostics only

    # Own-POV tell: the player block describes US iff its steamid matches the
    # provider's (our account). A mismatch means we're viewing another player
    # (spectating). If either steamid is missing we can't assert spectating, so we
    # fall back to own_pov=True when a local state exists (menu/self payloads that
    # omit one of the ids still describe us) — the health check then governs alive.
    if player_sid is not None and provider_sid is not None:
        out["own_pov"] = (player_sid == provider_sid)
    else:
        out["own_pov"] = True
    out["spectating"] = not out["own_pov"]

    player_state = player.get("state") or {}
    if player_state:
        out["has_local_state"] = True
        hp = player_state.get("health")
        out["health"] = hp
        if hp is not None:
            # Own-POV AND health>0. Spectating a living teammate => alive False
            # (own_pov is False there), despite their health>0 (the D-032 guard).
            out["alive"] = out["own_pov"] and (hp > 0)

    # Active weapon + ammo (combat context; not used by the alive filter). The
    # active weapon has state == 'active'; ammo fields are absent for knife/nades.
    weapons = player.get("weapons") or {}
    for _slot, w in weapons.items():
        if isinstance(w, dict) and w.get("state") == "active":
            out["active_weapon"] = w.get("name")
            out["ammo_clip"] = w.get("ammo_clip")
            out["ammo_reserve"] = w.get("ammo_reserve")
            break

    rnd = payload.get("round") or {}
    out["round_phase"] = rnd.get("phase")
    mp = payload.get("map") or {}
    out["map_phase"] = mp.get("phase")
    return out


class _LatestSlot:
    """Lock-protected newest-state slot. The ONLY shared state with any reader.

    Holds the most recent parsed state, the perf_counter time it arrived, and a
    running count of received updates. Overwrite semantics: a new POST replaces
    the slot (GSI is a state, not a stream to accumulate). Every method takes the
    lock; reads are cheap snapshots.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = None          # last extract_state() dict, or None pre-contact
        self._t_perf = None         # perf_counter of the last update
        self._t_wall = None         # wall-clock ISO of the last update (diagnostics)
        self._count = 0             # total updates received

    def update(self, state, t_perf, t_wall):
        with self._lock:
            self._state = state
            self._t_perf = t_perf
            self._t_wall = t_wall
            self._count += 1

    def read(self):
        """Snapshot: (state_or_None, t_perf_or_None, count)."""
        with self._lock:
            return self._state, self._t_perf, self._count

    def count(self):
        with self._lock:
            return self._count


class _Handler(BaseHTTPRequestHandler):
    listener = None  # injected by the server factory

    def log_message(self, *a):
        pass  # silence default per-request stderr logging

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        t_perf = time.perf_counter()
        # Respond 2XX immediately: GSI won't send the next update until it gets a
        # response and treats non-2XX as failure. Keep this fast and before any
        # parsing so a slow parse never delays the ack.
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()

        lst = self.listener
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            lst._note_bad_payload(len(raw))
            return

        got_token = (payload.get("auth") or {}).get("token")
        if lst.token and got_token is not None and got_token != lst.token:
            lst._note_token_mismatch(got_token)

        state = extract_state(payload)
        t_wall = datetime.now().isoformat(timespec="milliseconds")
        lst._on_update(state, t_perf, t_wall, payload)


class GsiListener:
    """Owns the GSI HTTP server on a background thread; exposes the latest state.

    Usage (mirrors RawMouseListener):
        gsi = GsiListener()
        gsi.start()
        gsi.wait_for_first_update(timeout=15.0)   # optional: block until CS2 POSTs
        ...
        state, age = gsi.read_latest()            # once per frame; non-blocking
        ...
        gsi.stop()

    read_latest() returns (state_dict_or_None, age_seconds_or_None). `state` is
    None before the first POST (no contact yet). `age_seconds` is how long since
    the last update (perf_counter delta) — a caller can treat a large age as a
    dead/quiet GSI. This class does NOT itself drop or default frames; that policy
    lives in the recorder (D-031).

    Optional on_update callback: called from the listener thread for each parsed
    update as on_update(state, t_perf, t_wall, payload). The probe uses it to
    print/log; the recorder does NOT need it (it samples read_latest() per frame).
    Keep any callback fast and non-blocking — it runs on the HTTP thread.
    """

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, token=DEFAULT_TOKEN,
                 on_update=None):
        self.host = host
        self.port = port
        self.token = token
        self._on_update_cb = on_update
        self._slot = _LatestSlot()
        self._server = None
        self._thread = None
        self._first_update_evt = threading.Event()
        # Diagnostics; incremented only on the HTTP thread, read after stop().
        self._bad_payloads = 0
        self._token_mismatch = None

    # ── called from the HTTP thread ──
    def _on_update(self, state, t_perf, t_wall, payload):
        self._slot.update(state, t_perf, t_wall)
        if not self._first_update_evt.is_set():
            self._first_update_evt.set()
        if self._on_update_cb is not None:
            try:
                self._on_update_cb(state, t_perf, t_wall, payload)
            except Exception as e:  # noqa: BLE001 - a bad callback must not kill the server
                print(f"  [gsi] on_update callback raised ({e!r}); continuing.")

    def _note_bad_payload(self, nbytes):
        self._bad_payloads += 1
        print(f"  [gsi] non-JSON POST ({nbytes} bytes) — ignoring")

    def _note_token_mismatch(self, got):
        if self._token_mismatch is None:
            self._token_mismatch = got
            print(f"  [gsi] auth token mismatch (got {got!r}) — check the .cfg")

    # ── lifecycle ──
    def start(self):
        if self._thread is not None:
            return
        handler = type("_BoundGsiHandler", (_Handler,), {"listener": self})
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="gsi-listener", daemon=True)
        self._thread.start()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._server = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ── the narrow interface the recorder samples ──
    def read_latest(self):
        """Return (state_or_None, age_seconds_or_None). Non-blocking snapshot.

        state       : the latest extract_state() dict, or None if no POST yet.
        age_seconds : perf_counter() - time_of_last_update, or None pre-contact.

        This is forward-fill by construction: between GSI updates the slot still
        holds the last state, so the caller naturally reads the carried-forward
        value. age_seconds lets the caller judge staleness.
        """
        state, t_perf, _count = self._slot.read()
        if state is None or t_perf is None:
            return None, None
        return state, time.perf_counter() - t_perf

    def has_first_update(self):
        return self._first_update_evt.is_set()

    def wait_for_first_update(self, timeout=None):
        """Block until the first GSI POST arrives (or timeout). True if it arrived.

        The recorder uses this as a liveness gate: if CS2 isn't POSTing (config
        not installed, saved with a BOM, CS2 not restarted, port blocked — the
        gsi/README failure list), this returns False and the recorder refuses to
        start, turning a silent zero-frame session into a loud early failure.
        """
        return self._first_update_evt.wait(timeout=timeout)

    def update_count(self):
        return self._slot.count()


def _selftest(host=DEFAULT_HOST, port=DEFAULT_PORT, token=DEFAULT_TOKEN, seconds=None):
    """Run the listener alone and print state changes — confirm CS2 is POSTing.

    A thin echo of gsi_probe for when you just want to know "is GSI reaching this
    process at all" before wiring it into the recorder. Start THIS, then (re)start
    CS2 with the .cfg in place, load a bot game, take damage/die/spectate.
    """
    last = {}

    def _on_update(state, t_perf, t_wall, payload):
        changed = []
        for k in ("alive", "health", "round_phase", "map_phase",
                  "active_weapon", "spectating"):
            if state.get(k) != last.get(k):
                changed.append(f"{k}={state.get(k)}")
                last[k] = state.get(k)
        if changed:
            print(f"[{time.strftime('%H:%M:%S')}] " + "  ".join(changed))

    print(f"GSI listener self-test on http://{host}:{port} (token {token!r}).")
    print("Start this FIRST, then (re)start CS2 with the .cfg in place.")
    print("Waiting for POSTs... (nothing below = CS2 isn't sending; check the")
    print(" .cfg is in the cfg dir, BOM-free, and CS2 was restarted). Ctrl-C to stop.\n")

    gsi = GsiListener(host=host, port=port, token=token, on_update=_on_update)
    gsi.start()
    t_end = (time.perf_counter() + seconds) if seconds else None
    try:
        while t_end is None or time.perf_counter() < t_end:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        gsi.stop()
    n = gsi.update_count()
    print(f"\nStopped. Received {n} update(s). "
          f"{'GSI is reaching this process.' if n else 'NO updates — GSI not proven here.'}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Shared CS2 GSI listener (Issue #21). --selftest confirms CS2 "
                    "is POSTing to this process.")
    p.add_argument("--selftest", action="store_true",
                   help="run the listener alone and print state changes")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--token", default=DEFAULT_TOKEN)
    p.add_argument("--seconds", type=float, default=None,
                   help="stop after N seconds (default: run until Ctrl-C)")
    args = p.parse_args(argv)
    if args.selftest:
        _selftest(host=args.host, port=args.port, token=args.token, seconds=args.seconds)
    else:
        print("Nothing to do. Use --selftest to confirm GSI reaches this process.")


if __name__ == "__main__":
    main()
