"""key_output.py — simulated keyboard output via SendInput SCAN CODES (M6-adjacent).

The FIRST component in the project that WRITES input to the game rather than
reading it. Everything upstream (raw_mouse, recorder) only observes; this presses
keys. It exists so a trained policy can actuate — starting with the movement
baseline (play_movement.py), later the arbiter (#16).

WHY SCAN CODES, NOT VIRTUAL-KEY CODES (the load-bearing decision, D-029):
The obvious approach — SendInput/keybd_event with a virtual-key code (VK_W) —
often does NOTHING in games. Source 2 (like most engines) reads the keyboard
through DirectInput / raw input, which keys off the hardware SCAN CODE, not the
Windows virtual-key. A VK-only injection updates the Win32 async key state (so it
"works" in Notepad and even registers in GetAsyncKeyState) but the game never
sees it. So we send with KEYEVENTF_SCANCODE and the actual scan code, exactly as
a physical keyboard would. This is the single most likely thing to fail silently,
which is why this module is standalone with a --selftest: verify a real character
moves in CS2 BEFORE trusting any policy wired to it.

WHAT THIS IS AND ISN'T:
  * IS: press / release / hold-for-a-duration of the movement keys (W/A/S/D) plus
    the handful the recorder logs, by scan code, via SendInput. ALSO (D-036):
    relative mouse motion via `mouse_move_relative`, for NAVIGATION yaw — the
    movement feed's view-turning, NOT combat aim. Mouse output is a NEW actuator
    and, per the D-029 lesson, is UNPROVEN until `--selftest-mouse` shows the CS2
    view actually turns; do not wire a policy to it before that passes.
  * ISN'T: combat aim (that is the separate detector-gated model, #10 -> #11, via
    the arbiter). ISN'T any anti-cheat evasion — this is plain SendInput, used
    ONLY offline on a local bot server (D-007). Online use is forbidden.

SAFETY (D-007): simulated input can trip anti-cheat. This must only ever drive an
offline/local server. This module can't enforce that by itself (it just presses
keys); the POLICY loop (play_movement.py) carries the arm/disarm guard and the
local-only warning. Keep it that way.

Scan codes here are SET-1 ("make" codes) for a US QWERTY layout — the standard
PC keyboard set. W/A/S/D and the others below are layout-position keys, so scan
codes are the right abstraction (they follow the physical key, not the letter).

Usage:
  python -m src.key_output --selftest          # press W ~1s, then each of A/S/D — WATCH your character
  python -m src.key_output --selftest --key a  # test one specific key
  python -m src.key_output --tap w --seconds 2 # hold W for 2 seconds
"""

import argparse
import ctypes
import time
from ctypes import wintypes


# ── SendInput plumbing via ctypes ────────────────────────────────────────────
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# ULONG_PTR is a POINTER-SIZED UNSIGNED INTEGER (not a pointer to a ULONG). Win32
# declares dwExtraInfo as ULONG_PTR (8 bytes on x64); define it correctly so the
# field carries an integer value, never a live pointer.
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_uint64
else:
    ULONG_PTR = ctypes.c_uint32

# INPUT type
INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
# KEYBDINPUT flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# MOUSEINPUT flags (D-036, Piece 2). MOUSEEVENTF_MOVE is a RELATIVE move by
# (dx, dy) in mouse units — exactly the form the model predicts (per-frame device
# deltas), and the same relative semantics raw_mouse.py CAPTURED. We deliberately
# do NOT use MOUSEEVENTF_ABSOLUTE: the model learned relative motion, and CS2
# consumes relative mouse input for view control. Whether Source 2 actually reads
# SendInput-injected relative motion is UNPROVEN until --selftest-mouse confirms it
# in-game (the D-029 discipline: injected input is the thing that fails silently).
MOUSEEVENTF_MOVE = 0x0001

# SET-1 make (press) scan codes, US QWERTY, by physical key. Release is the same
# code with KEYEVENTF_KEYUP. These follow the physical position, which is what the
# game reads — the whole point of D-029.
SCAN_CODES = {
    "w": 0x11, "a": 0x1E, "s": 0x1F, "d": 0x20,
    "space": 0x39, "ctrl": 0x1D, "shift": 0x2A,
    "1": 0x02, "2": 0x03, "3": 0x04, "r": 0x13,
}

# Keys that live in the "extended" block of the keyboard need KEYEVENTF_EXTENDEDKEY
# alongside their scan code. None of the movement/recorder keys above are extended
# (left ctrl/shift are non-extended), so this stays empty for now — documented so a
# future addition (e.g. arrow keys, right-ctrl) knows to set the flag.
EXTENDED_KEYS = set()


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),        # value, not POINTER(ULONG)
    ]


class MOUSEINPUT(ctypes.Structure):
    # Included NOT because we send mouse input, but because the real Win32 INPUT
    # union sizes to its LARGEST member, MOUSEINPUT. Omitting it makes
    # sizeof(INPUT) smaller than the OS's INPUT, so SendInput's cbSize check
    # rejects every event -> 0 inserted, GetLastError=0. This was the actual bug.
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUTunion),
    ]


# Pin SendInput's signature now that INPUT exists, so ctypes marshals count,
# array pointer, and cbSize at correct x64 widths. cbSize MUST be sizeof(INPUT).
user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
_INPUT_SIZE = ctypes.sizeof(INPUT)


def _send_scan(scan_code, key_up, extended=False):
    """Send ONE key event (press or release) by scan code via SendInput.

    Builds an INPUT_KEYBOARD event with KEYEVENTF_SCANCODE so the OS delivers it
    as a hardware-style scan-code event — the form the game reads (D-029). wVk is
    left 0 on purpose: with KEYEVENTF_SCANCODE the scan code in wScan is
    authoritative and the virtual key is ignored.
    """
    flags = KEYEVENTF_SCANCODE
    if key_up:
        flags |= KEYEVENTF_KEYUP
    if extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    ki = KEYBDINPUT(wVk=0, wScan=scan_code, dwFlags=flags, time=0, dwExtraInfo=0)
    inp = INPUT(type=INPUT_KEYBOARD, union=_INPUTunion(ki=ki))
    ctypes.set_last_error(0)
    n = user32.SendInput(1, ctypes.byref(inp), _INPUT_SIZE)
    if n != 1:
        # SendInput returns the number of events inserted; 0 means it was blocked.
        # Read the actual Win32 error so we can tell WHY, rather than guess:
        #   ERROR_ACCESS_DENIED (5)  -> UIPI: a lower-integrity process is blocked
        #                               from injecting into a higher-integrity
        #                               foreground window (the classic case).
        #   0                        -> not an access error; often the target simply
        #                               isn't accepting injected input (e.g. an
        #                               anti-cheat filtering the input stream), or a
        #                               marshalling problem. The Notepad probe
        #                               (--probe-focus) distinguishes these.
        err = ctypes.get_last_error()
        raise OSError(
            f"SendInput inserted {n}/1 events for scan {scan_code:#x} "
            f"(GetLastError={err}: {_winerr_name(err)}; sizeof(INPUT)={_INPUT_SIZE}). "
            f"Blocked before reaching any window.")


def _winerr_name(err):
    """Human label for the handful of Win32 error codes SendInput realistically returns."""
    return {
        0: "ERROR_SUCCESS (no error flagged despite 0 inserted — target rejected input)",
        5: "ERROR_ACCESS_DENIED (UIPI / integrity level)",
        87: "ERROR_INVALID_PARAMETER (bad INPUT struct — a bug in our call)",
    }.get(err, f"unknown code {err}")


def mouse_move_relative(dx, dy):
    """Inject ONE relative mouse move of (dx, dy) mouse units via SendInput (D-036).

    This is the mouse counterpart to _send_scan: the FIRST mouse output in the
    project. It sends a MOUSEEVENTF_MOVE event with relative dx/dy — the same
    relative-delta form the model predicts and that raw_mouse.py captured, so a
    predicted (dx, dy) can be replayed as view motion. dx>0 is rightward, dy>0 is
    downward (matching the raw-capture sign convention in DATA_FORMAT.md).

    Values are rounded to int and clamped to a sane per-call range so a
    mispredicted spike can't hurl the view across the map in one frame (a safety
    bound, not a model assumption). Sending (0, 0) is a no-op and skipped.

    UNPROVEN UNTIL VERIFIED (D-029 discipline): whether CS2 actually turns from
    this injected motion must be confirmed with --selftest-mouse in-game before any
    policy is wired to it. Like scan-code output, this can insert successfully at
    the Win32 level yet be ignored by the game's raw-input path; the self-test is
    what distinguishes 'works' from 'silently does nothing'.
    """
    idx = int(round(dx))
    idy = int(round(dy))
    # Per-call clamp: a single frame's view move shouldn't exceed this many mouse
    # units. Generous enough for fast turns, bounded enough that a bad prediction
    # can't fling the camera. Not a model claim — a physical safety rail.
    LIMIT = 600
    idx = max(-LIMIT, min(LIMIT, idx))
    idy = max(-LIMIT, min(LIMIT, idy))
    if idx == 0 and idy == 0:
        return
    mi = MOUSEINPUT(dx=idx, dy=idy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE,
                    time=0, dwExtraInfo=0)
    inp = INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=mi))
    ctypes.set_last_error(0)
    n = user32.SendInput(1, ctypes.byref(inp), _INPUT_SIZE)
    if n != 1:
        err = ctypes.get_last_error()
        raise OSError(
            f"SendInput inserted {n}/1 mouse events for move ({idx},{idy}) "
            f"(GetLastError={err}: {_winerr_name(err)}; sizeof(INPUT)={_INPUT_SIZE}). "
            f"Blocked before reaching any window.")


def _resolve(key):
    if key not in SCAN_CODES:
        raise ValueError(f"Unknown key {key!r}. Known: {sorted(SCAN_CODES)}")
    return SCAN_CODES[key], (key in EXTENDED_KEYS)


def press(key):
    """Press (and hold) a key by name. Must be paired with release()."""
    scan, ext = _resolve(key)
    _send_scan(scan, key_up=False, extended=ext)


def release(key):
    """Release a previously-pressed key by name."""
    scan, ext = _resolve(key)
    _send_scan(scan, key_up=True, extended=ext)


def tap(key, seconds=0.1):
    """Press a key, hold for `seconds`, release. Blocking."""
    press(key)
    try:
        time.sleep(seconds)
    finally:
        release(key)   # always release, even if interrupted


class KeyController:
    """Tracks which keys are currently held and drives press/release to a target set.

    The policy loop thinks in terms of "which keys should be down THIS frame"; this
    class turns that into the minimal set of press/release calls and guarantees no
    key is left stuck down. `apply(desired)` presses keys newly wanted and releases
    keys no longer wanted. `release_all()` is the safety stop.

    Holding is edge-triggered on PURPOSE: a key already down is NOT re-pressed each
    frame (re-pressing a held scan code can register as key-repeat and is wasteful).
    Only transitions are sent.
    """

    def __init__(self, keys=("w", "a", "s", "d")):
        self.managed = tuple(keys)
        self._down = set()

    def apply(self, desired):
        """Press/release so exactly `desired` (an iterable of key names) is held.

        Only keys in self.managed are touched; a desired key outside the managed
        set is ignored (with no error) so the caller can't accidentally actuate a
        key this controller wasn't set up to own.
        """
        want = {k for k in desired if k in self.managed}
        # Release keys that are down but no longer wanted.
        for k in list(self._down):
            if k not in want:
                release(k)
                self._down.discard(k)
        # Press keys newly wanted.
        for k in want:
            if k not in self._down:
                press(k)
                self._down.add(k)

    def release_all(self):
        """Release every key this controller currently holds. The safety stop."""
        for k in list(self._down):
            try:
                release(k)
            finally:
                self._down.discard(k)

    @property
    def held(self):
        return set(self._down)


def probe_focus(seconds=1.5):
    """Inject W into WHATEVER window is focused — the CS2-vs-everything-else test.

    This is the diagnostic fork when --selftest fails with SendInput returning 0.
    Open Notepad (or any text field), run this, and click into Notepad during the
    countdown so it is focused when the key fires. Then:
      * If a 'w' is TYPED in Notepad -> SendInput itself works; our scan-code call
        is correct. The failure is SPECIFIC to CS2 (integrity level not cleared
        for it, or its anti-cheat filtering injected input even offline). That is
        a platform/target finding, not a bug in this module.
      * If NOTHING is typed AND you see the same GetLastError here -> the block is
        not CS2-specific; it's our SendInput call or a system-wide policy. We debug
        the call (marshalling, elevation) rather than CS2.
    Uses the SAME _send_scan path as the real output, so it tests the real code.
    """
    print("FOCUS PROBE — injects 'w' into whatever window is focused.")
    print("Open Notepad, then CLICK INTO IT during the countdown so it's focused.")
    print("(This is the test that separates 'CS2/anti-cheat blocks it' from")
    print(" 'our SendInput call is wrong' — see which happens.)\n")
    print("Firing 'w' in:", end=" ", flush=True)
    for c in (3, 2, 1):
        print(c, end=" ", flush=True)
        time.sleep(1.0)
    print("\n")
    try:
        tap("w", seconds=seconds)
    except OSError as e:
        print(f"  SendInput FAILED even into the focused window: {e}")
        print("  -> The block is NOT CS2-specific. Since you're already elevated,")
        print("     this points at the SendInput call itself or a system input")
        print("     policy, not anti-cheat. Tell the developer this exact error.")
        return
    print("  SendInput reported success (1/1 event inserted).")
    print("  Did a 'w' appear in Notepad?")
    print("    YES -> our output works; the earlier failure is SPECIFIC to CS2")
    print("           (integrity level for CS2, or anti-cheat filtering injected")
    print("           input). That reframes the live demo — talk to the developer.")
    print("    NO  -> input inserted but not delivered to the focused app; unusual,")
    print("           worth reporting.")


def selftest(only_key=None):
    """Press keys with a countdown so you can WATCH the character in CS2.

    This is the make-or-break check for D-029: if scan-code output is right, your
    character moves; if VK-only injection were used instead, nothing would happen
    in-game even though Windows would think the key was pressed. Focus CS2, stand
    in an open area on a LOCAL server, and run this.

    Presses W (forward) for ~1.5 s, then A, S, D in turn (skippable with --key).
    You should see the character move each time. If NOTHING moves, scan-code
    output isn't reaching the game — do not proceed to play_movement.py; fix this
    first (common causes: CS2 not focused, needs run-as-admin to match the game's
    integrity level, or a different keyboard layout).
    """
    keys = [only_key] if only_key else ["w", "a", "s", "d"]
    labels = {"w": "FORWARD", "a": "LEFT", "s": "BACK", "d": "RIGHT"}
    print("KEY OUTPUT SELF-TEST (scan-code SendInput, D-029).")
    print("Focus CS2, stand in an open area on a LOCAL server (D-007: offline only).")
    print("Each key is held ~1.5 s. WATCH your character move.\n")
    print("Starting in:", end=" ", flush=True)
    for c in (3, 2, 1):
        print(c, end=" ", flush=True)
        time.sleep(1.0)
    print("\n")
    for k in keys:
        print(f"  pressing {k.upper()} ({labels.get(k, k)}) for 1.5 s...", flush=True)
        try:
            tap(k, seconds=1.5)
        except OSError as e:
            print(f"    SendInput FAILED: {e}")
            print("    -> events are being blocked before reaching CS2. Next steps:")
            print("       1. Run the terminal as Administrator (match CS2's integrity).")
            print("       2. If already admin, run the FOCUS PROBE to tell whether the")
            print("          block is CS2-specific or system-wide:")
            print("            python -m src.key_output --probe-focus   (type into Notepad)")
            return
        time.sleep(0.4)
    print("\nDone. Did the character move on EACH key?")
    print("  YES -> scan-code output works; play_movement.py can drive the game.")
    print("  NO  -> output isn't reaching CS2. Do NOT wire the model to it yet.")
    print("         Check: CS2 focused? terminal as Admin? US-QWERTY layout?")


def selftest_mouse(seconds=None):
    """Prove injected RELATIVE mouse motion turns the CS2 view (D-036, D-029 rule).

    The mouse counterpart to selftest(). Mouse output (mouse_move_relative) is a
    NEW, UNPROVEN actuator: SendInput can report success while the game ignores the
    motion, exactly the silent-failure mode scan codes had. So before any policy
    drives the mouse, this must show the VIEW ACTUALLY TURNS in-game.

    It sweeps the view RIGHT in small steps, then LEFT back, then DOWN then UP, so
    you can watch each axis. What to check, on a LOCAL server (D-007), CS2 focused,
    standing still:
      * RIGHT phase -> your view should pan smoothly to the right,
      * LEFT phase  -> pan back to the left (roughly returning to start),
      * DOWN/UP     -> the view pitches down then up.
    If the view does NOT move but SendInput reports success, CS2 is ignoring
    injected mouse motion (raw-input filtering / focus / integrity) — the mouse
    path is NOT usable and the model must not be wired to it. If it moves the WRONG
    way, the sign convention needs flipping in play_movement, not here (this sends
    exactly what DATA_FORMAT.md defines: +dx right, +dy down).

    Deliberately uses the SAME mouse_move_relative path the policy will use, so a
    pass here means the policy's actuation is trustworthy.
    """
    step = 8               # mouse units per injected move (small = smooth)
    n_steps = 40           # steps per phase
    pause = 0.02           # seconds between steps (~smooth at ~50 Hz)
    print("MOUSE OUTPUT SELF-TEST (relative SendInput, D-036).")
    print("Focus CS2, stand still on a LOCAL server (D-007: offline only).")
    print("WATCH YOUR VIEW: it should pan right, back left, then down, then up.")
    print("This is the make-or-break check that injected mouse motion reaches CS2 —")
    print("the mouse analogue of the scan-code test. Ctrl-C to abort.\n")
    print("Starting in:", end=" ", flush=True)
    for c in (3, 2, 1):
        print(c, end=" ", flush=True)
        time.sleep(1.0)
    print("\n")

    phases = [("RIGHT", step, 0), ("LEFT", -step, 0),
              ("DOWN", 0, step), ("UP", 0, -step)]
    try:
        for label, sx, sy in phases:
            print(f"  sweeping {label} ({n_steps} steps of ({sx},{sy}))...",
                  flush=True)
            for _ in range(n_steps):
                mouse_move_relative(sx, sy)
                time.sleep(pause)
            time.sleep(0.3)
    except OSError as e:
        print(f"\n    SendInput FAILED for mouse move: {e}")
        print("    -> injected mouse events are being blocked before reaching CS2.")
        print("       Same triage as the key path: run the terminal as Admin to")
        print("       match CS2's integrity level; if already admin, the block is")
        print("       likely anti-cheat/raw-input filtering. Do NOT wire the look")
        print("       head to the mouse until this passes.")
        return
    print("\nDone. Did your VIEW pan right, then back left, then down, then up?")
    print("  YES -> injected relative mouse motion reaches CS2; the look head can")
    print("         drive the view (wire it in play_movement.py behind the arm guard).")
    print("  NO (view didn't move, no error) -> CS2 is ignoring injected mouse")
    print("         motion (raw-input filtering / focus / integrity). The mouse path")
    print("         is NOT usable as-is; do NOT drive the model's dx/dy to it.")
    print("  WRONG DIRECTION -> the path works; the sign is handled where the model")
    print("         output is applied, not in key_output.")


def _build_parser():
    p = argparse.ArgumentParser(
        description="Simulated keyboard output via SendInput scan codes (D-029). "
                    "Test in isolation with --selftest BEFORE using play_movement.py.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true",
                   help="press W/A/S/D in turn with a countdown — watch your character move")
    g.add_argument("--selftest-mouse", action="store_true",
                   help="sweep the VIEW right/left/down/up via injected relative mouse "
                        "motion — the D-036 mouse-output check; watch your view turn")
    g.add_argument("--probe-focus", action="store_true",
                   help="inject 'w' into the focused window (type into Notepad) — the "
                        "CS2-vs-system diagnostic when --selftest is blocked")
    g.add_argument("--tap", type=str, metavar="KEY",
                   help="hold one key (w/a/s/d/space/ctrl/shift/1/2/3/r) for --seconds")
    p.add_argument("--key", type=str, default=None,
                   help="with --selftest: test only this key")
    p.add_argument("--seconds", type=float, default=1.0,
                   help="with --tap: hold duration (default 1.0)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.selftest:
        selftest(only_key=args.key)
    elif args.selftest_mouse:
        selftest_mouse()
    elif args.probe_focus:
        probe_focus(seconds=args.seconds if args.seconds != 1.0 else 1.5)
    elif args.tap:
        print(f"Holding {args.tap!r} for {args.seconds:.1f}s (focus CS2 on a local server).")
        time.sleep(1.0)
        tap(args.tap, seconds=args.seconds)
        print("Done.")
    else:
        print("Choose --selftest (recommended first) or --tap KEY. "
              "See `python -m src.key_output -h`.")


if __name__ == "__main__":
    main()
