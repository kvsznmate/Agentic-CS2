"""raw_mouse.py — background raw-input mouse listener for Issue #3.

Reads the *physical* mouse movement (raw dx/dy from the device) rather than the
cursor position. This is the load-bearing difference from the reference study:
their `mouse_check()` read `GetCursorPos` (the OS cursor), which in CS2 is locked
and hidden during play and so reports a frozen point. The player's aim (view
angle) they instead read out of game memory via RAM offsets — a Source 2 dead end
for us (D-002). So we recover aim from raw device deltas via the Windows raw
input API (WM_INPUT). See DECISIONS D-015.

── WHY AN ACCUMULATOR, NOT AN EVENT STREAM ───────────────────────────────────
A high-DPI / high-polling mouse (1000+ Hz) emits a WM_INPUT message per motion
tick — thousands/sec. Processing each is both pointless for us and a known
performance hazard (fast mouse movement has been measured tanking game FPS even
with batched handling). We don't need per-event data; the model needs "how far
did the view move between frame N and N+1." So this listener runs a hidden
message window on its own thread and simply ACCUMULATES dx/dy. The recorder
calls read_and_reset() once per frame to get the summed movement since the last
frame and zero the accumulator. Frame-rate-aligned aim signal, no event flood.

── HONEST STATUS ─────────────────────────────────────────────────────────────
This is the piece most likely to need on-machine iteration (mouse DPI, polling
rate, whether CS2 fullscreen changes message delivery, focus behaviour). It is
deliberately its own module with a --selftest so it can be debugged in isolation
from the capture loop. Run:  python -m src.raw_mouse --selftest
"""

import ctypes
import threading
import time
from ctypes import wintypes


# ── Win32 raw input plumbing via ctypes ───────────────────────────────────
# We register the mouse as a raw input device against a hidden message-only
# window, run a message pump on a dedicated thread, and add up the deltas that
# arrive in WM_INPUT. Everything here is standard Win32; the fiddly part is that
# raw input REQUIRES a window + message loop, which is why this can't be a
# simple poll.

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

WM_INPUT = 0x00FF
WM_QUIT = 0x0012
RIDEV_INPUTSINK = 0x00000100  # receive input even when not foreground
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
HID_USAGE_PAGE_GENERIC = 0x01
HID_USAGE_GENERIC_MOUSE = 0x02


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class _RAWMOUSE_BUTTONS(ctypes.Structure):
    _fields_ = [("usButtonFlags", wintypes.USHORT),
                ("usButtonData", wintypes.USHORT)]


class _RAWMOUSE_U(ctypes.Union):
    # Defined at module level (not nested in RAWMOUSE): a nested class isn't a
    # visible name to a sibling _fields_ list evaluated in the class body, which
    # raised NameError. Module-level names resolve fine.
    _fields_ = [("ulButtons", wintypes.ULONG),
                ("buttons", _RAWMOUSE_BUTTONS)]


class RAWMOUSE(ctypes.Structure):
    # Only the fields we need; layout matches the Win32 RAWMOUSE.
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("u", _RAWMOUSE_U),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", ctypes.c_long),
        ("lLastY", ctypes.c_long),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("mouse", RAWMOUSE),
    ]


# Window-proc signature. Return type is LRESULT (pointer-width), NOT c_long:
# on 64-bit Windows DefWindowProcW returns a 64-bit LRESULT, and declaring the
# callback as c_long (32-bit) overflowed on every message. wintypes has no
# LRESULT, so use the pointer-width signed type.
LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)

# Declare DefWindowProcW explicitly so ctypes marshals the 64-bit wParam/lParam
# and return value correctly instead of guessing (the guess overflowed).
user32.DefWindowProcW.restype = LRESULT
user32.DefWindowProcW.argtypes = [
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
]

# Same reasoning for GetRawInputData: declare the signature so the handle and
# the two pointer args marshal correctly on 64-bit. hRawInput is passed as the
# lParam of WM_INPUT.
user32.GetRawInputData.restype = wintypes.UINT
user32.GetRawInputData.argtypes = [
    wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p,
    ctypes.POINTER(wintypes.UINT), wintypes.UINT
]


class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class RawMouseListener:
    """Accumulates raw mouse dx/dy on a background thread.

    Thread-safe. Start it, then call read_and_reset() once per frame to get the
    (dx, dy) summed since the previous call and clear the running total.

        listener = RawMouseListener()
        listener.start()
        ...
        dx, dy = listener.read_and_reset()   # once per frame
        ...
        listener.stop()

    dx > 0 is rightward physical movement, dy > 0 is downward (raw device axes;
    verify sign on-machine with --selftest, since this is exactly the kind of
    thing that must be confirmed, not assumed).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._dx = 0
        self._dy = 0
        self._event_count = 0  # for diagnostics: how many WM_INPUT arrived
        self._thread = None
        self._hwnd = None
        self._thread_id = None
        self._ready = threading.Event()
        # Keep a reference to the WNDPROC or it gets garbage-collected and the
        # window proc call crashes — a classic ctypes footgun.
        self._wndproc = WNDPROC(self._on_message)

    def _on_message(self, hwnd, msg, wparam, lparam):
        if msg == WM_INPUT:
            size = wintypes.UINT(0)
            user32.GetRawInputData(
                wintypes.HANDLE(lparam), RID_INPUT, None,
                ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
            buf = ctypes.create_string_buffer(size.value)
            got = user32.GetRawInputData(
                wintypes.HANDLE(lparam), RID_INPUT, buf,
                ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
            if got == size.value:
                ri = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
                if ri.header.dwType == RIM_TYPEMOUSE:
                    with self._lock:
                        self._dx += ri.mouse.lLastX
                        self._dy += ri.mouse.lLastY
                        self._event_count += 1
            # fall through to DefWindowProc for cleanup per Win32 guidance
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _run(self):
        self._thread_id = kernel32.GetCurrentThreadId()
        hinst = kernel32.GetModuleHandleW(None)
        class_name = "AgenticCS2RawMouseWnd"

        wndclass = WNDCLASS()
        wndclass.lpfnWndProc = self._wndproc
        wndclass.hInstance = hinst
        wndclass.lpszClassName = class_name
        atom = user32.RegisterClassW(ctypes.byref(wndclass))
        if not atom:
            # Class may already be registered from a prior run in the same
            # process; that's fine, CreateWindow will still work by name.
            pass

        # Message-only window: parent = HWND_MESSAGE (-3). It never shows and
        # only exists to receive WM_INPUT.
        HWND_MESSAGE = wintypes.HWND(-3)
        self._hwnd = user32.CreateWindowExW(
            0, class_name, "raw_mouse", 0, 0, 0, 0, 0,
            HWND_MESSAGE, None, hinst, None)
        if not self._hwnd:
            self._ready.set()
            raise ctypes.WinError(ctypes.get_last_error())

        rid = RAWINPUTDEVICE(
            HID_USAGE_PAGE_GENERIC, HID_USAGE_GENERIC_MOUSE,
            RIDEV_INPUTSINK, self._hwnd)
        if not user32.RegisterRawInputDevices(
                ctypes.byref(rid), 1, ctypes.sizeof(RAWINPUTDEVICE)):
            self._ready.set()
            raise ctypes.WinError(ctypes.get_last_error())

        self._ready.set()

        # Standard message pump. GetMessage blocks until a message arrives;
        # PostThreadMessage(WM_QUIT) from stop() breaks it.
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="raw-mouse")
        self._thread.start()
        # Wait until the window + device registration is done (or failed) so a
        # caller that immediately reads doesn't race the setup.
        self._ready.wait(timeout=5.0)

    def read_and_reset(self):
        """Return (dx, dy) accumulated since last call, and zero the total."""
        with self._lock:
            dx, dy = self._dx, self._dy
            self._dx = 0
            self._dy = 0
            return dx, dy

    def event_count_and_reset(self):
        """Diagnostic: how many raw events arrived since last call."""
        with self._lock:
            c = self._event_count
            self._event_count = 0
            return c

    def stop(self):
        if self._thread_id is not None:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None


def _selftest(seconds=15.0):
    """Prove raw deltas read cleanly — run this BEFORE trusting the recorder.

    Prints accumulated dx/dy a few times a second while you move the mouse.
    What to check:
      * Move RIGHT only  -> dx steadily positive, dy ~0.
      * Move LEFT only   -> dx steadily negative.
      * Move DOWN only   -> dy positive (note the sign; raw down is +).
      * Hold still       -> both ~0, event count ~0.
    CRUCIALLY: do this with CS2 open and in-game (cursor locked), not just on
    the desktop. The whole point is that raw input keeps working during the
    cursor lock where the study's GetCursorPos would freeze. If dx/dy stay 0
    in-game, that's the thing to solve before building on this.
    """
    print("Raw mouse self-test. Move the mouse; watch dx/dy. Ctrl+C to stop.")
    print("Test on the desktop first, then RE-RUN with CS2 in-game (cursor "
          "locked) — that's the real check.\n")
    listener = RawMouseListener()
    listener.start()
    try:
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            time.sleep(0.2)
            dx, dy = listener.read_and_reset()
            n = listener.event_count_and_reset()
            bar_x = ("+" if dx >= 0 else "-") * min(abs(dx), 40)
            print(f"dx={dx:+5d} dy={dy:+5d}  events={n:4d}  {bar_x}")
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
    print("\nStopped. If dx/dy tracked your movement in-game, the aim signal is "
          "readable and the recorder can trust read_and_reset().")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Raw-input mouse listener (Issue #3).")
    p.add_argument("--selftest", action="store_true",
                   help="print live dx/dy so you can verify raw deltas read cleanly")
    p.add_argument("--seconds", type=float, default=15.0)
    args = p.parse_args()
    if args.selftest:
        _selftest(seconds=args.seconds)
    else:
        print("Nothing to do. Use --selftest to verify raw mouse reading.")
