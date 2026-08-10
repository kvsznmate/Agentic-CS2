"""recorder.py — synchronized frame + input recording (Issue #3, the M0 GATE).

The heart of M0. Runs ONE synchronous loop (way one, chosen deliberately over a
threaded design — the point of this gate is to PROVE alignment, and the simplest
loop is the one whose correctness can actually be verified). Each tick:

  1. read accumulated raw mouse dx/dy since the last frame (and reset),
  2. read keyboard + mouse-button state,
  3. grab a frame (which returns its own capture timestamp),
  4. write one record pairing all of the above.

Why this order matters: the frame grab is LAST, mirroring the reference study's
loop, which grabbed the image last on purpose so the image's time-lag matches
when a forward pass will run at inference. The mouse delta covers the interval
that ENDS at this frame — "how far the view moved arriving at frame N."

WHAT THIS GATE MUST PROVE (PROJECT_ISSUES #3): that input at frame N provably
corresponds to screen at frame N. That is not self-evident from a loop that
merely runs — it has to be demonstrated. Hence --verify below, which records a
short session while YOU perform a scripted motion, then checks the logged inputs
match what you did, frame by frame. If sync can't be shown reliable, the issue
says RAISE THE KILL FLAG — so this file is built to expose failure, not hide it.

DECISIONS honoured:
  D-015  aim captured from raw mouse deltas (raw_mouse.py), not cursor position
         or memory-read view angles (both dead for us). Keys/clicks via pywin32.
  single synchronous loop, per the way-one decision.

Entry points:
  python -m src.recorder --verify     # scripted-motion alignment check (DO FIRST)
  python -m src.recorder --record     # record a session to disk
  python -m src.recorder --dryrun     # loop + live readout, write nothing
"""

import argparse
import os
import time

import numpy as np
import win32api

from src.capture import Capture
from src import capture_config as cfg
from src.raw_mouse import RawMouseListener


# Output location for recordings. data/ is gitignored (recordings are large and
# local — see .gitignore and PROJECT_ISSUES #4/#5).
_REC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "recordings")

# Target loop rate. Profiling (D-016) showed the loop is entirely bounded by the
# ~37 ms mss grab; input logging is free. Realised recording rate is ~15 FPS, so
# the loop targets 15 rather than an unreachable 20. Matches the study's 16 FPS
# working loop. dxcam is the documented lever (D-014) if the full agent loop
# (with model inference) later needs more headroom.
LOOP_FPS = 15

# ── Keys we log ───────────────────────────────────────────────────────────
# Movement + jump/crouch + reload + weapon slots, mirroring the study's action
# set (config.py n_keys=11: w,s,a,d,space,ctrl,shift,1,2,3,r). These are the
# ACTUAL in-game keys — unlike the study's record script, which had the player
# press tfgh so it could simulate wasd for a spectated bot. We record our own
# real play (D-002), so we log the real keys directly.
# Virtual-key codes: letters/digits use ord(uppercase). Named keys are explicit.
VK_SPACE = 0x20
VK_CONTROL = 0x11
VK_SHIFT = 0x10
LOGGED_KEYS = [
    ("w", ord("W")), ("a", ord("A")), ("s", ord("S")), ("d", ord("D")),
    ("space", VK_SPACE), ("ctrl", VK_CONTROL), ("shift", VK_SHIFT),
    ("1", ord("1")), ("2", ord("2")), ("3", ord("3")), ("r", ord("R")),
]
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
# Quit key for the loop (checked each tick). F8 is unlikely to be bound in CS2.
VK_QUIT = 0x77  # F8


def _key_down(vk):
    """True if the virtual key is currently held.

    GetAsyncKeyState high bit = currently down. Works during gameplay (the study
    relied on the same call), unlike cursor position which the game locks.
    """
    return (win32api.GetAsyncKeyState(vk) & 0x8000) != 0


def _read_keys():
    """Return a 0/1 vector over LOGGED_KEYS in fixed order."""
    return [1 if _key_down(vk) else 0 for _, vk in LOGGED_KEYS]


def _read_clicks():
    """Return (left, right) as 0/1 for mouse buttons currently held."""
    left = 1 if _key_down(VK_LBUTTON) else 0
    right = 1 if _key_down(VK_RBUTTON) else 0
    return left, right


class Recorder:
    """Owns the capture, the mouse listener, and the per-frame record assembly."""

    def __init__(self):
        self.cap = Capture()
        self.mouse = RawMouseListener()

    def __enter__(self):
        self.cap.__enter__()
        self.mouse.start()
        return self

    def __exit__(self, *exc):
        self.mouse.stop()
        self.cap.__exit__(*exc)

    def read_frame_record(self):
        """One synchronized tick -> (frame, record_dict).

        Order is deliberate (see module docstring): mouse delta first (closes the
        interval ending at this frame), then keys/clicks, then the frame grab
        last so its timing matches inference. All share the frame's capture
        timestamp as the single alignment anchor.
        """
        dx, dy = self.mouse.read_and_reset()
        keys = _read_keys()
        lclick, rclick = _read_clicks()
        frame, t_cap = self.cap.grab()
        record = {
            "t": t_cap,          # capture timestamp (perf_counter) — the anchor
            "keys": keys,        # 0/1 over LOGGED_KEYS order
            "lclick": lclick,
            "rclick": rclick,
            "dx": dx,            # summed raw mouse dx since last frame
            "dy": dy,
        }
        return frame, record


def _pace(loop_start, fps):
    """Sleep until the frame's time budget is up, to hold ~fps."""
    target = loop_start + 1.0 / fps
    while time.perf_counter() < target:
        time.sleep(0.0005)


def dryrun(seconds=20.0):
    """Run the loop with a live readout, writing nothing. Sanity check the feed.

    Prints keys/clicks/dx/dy each tick so you can watch inputs register in real
    time. Good first smoke test that all three input sources are alive together
    before committing to a recording or the formal verify.
    """
    print("Dry run — synchronized loop, nothing saved. Press F8 to stop.")
    print("Move, press WASD, click — watch the values change.\n")
    key_names = [name for name, _ in LOGGED_KEYS]
    with Recorder() as rec:
        n = 0
        t_start = time.perf_counter()
        while True:
            loop_start = time.perf_counter()
            if _key_down(VK_QUIT):
                break
            _frame, r = rec.read_frame_record()
            n += 1
            if n % 5 == 0:  # don't spam every frame
                held = [key_names[i] for i, v in enumerate(r["keys"]) if v]
                print(f"keys={held!s:<28} L={r['lclick']} R={r['rclick']} "
                      f"dx={r['dx']:+5d} dy={r['dy']:+5d}", end="\r")
            if time.perf_counter() - t_start > seconds:
                break
            _pace(loop_start, LOOP_FPS)
    print("\nDry run done.")


def profile(seconds=20.0):
    """Time each loop stage separately to find what limits the recording FPS.

    `--record` reported ~14.8 FPS, below the ~25 capture-alone rate and our 20
    target. This breaks the per-frame cost into its parts — mouse read, key/click
    read, frame grab, record assembly — so we know whether the mss grab dominates
    (expected, per D-013/D-014; then a lower floor is just honest) or something
    cheaper is the culprit. Runs the REAL work each tick but saves nothing and
    does NOT pace, so the numbers reflect raw stage cost, not the sleep.

    Play normally (move + look) so the mouse/key paths do representative work.
    Press F8 to stop.
    """
    print("Profiling the record loop. Play normally (move + look). F8 to stop.")
    print("Saving nothing; this measures per-stage cost, unpaced.\n")
    n = 0
    mouse_t = keys_t = grab_t = asm_t = 0.0
    with Recorder() as rec:
        t_start = time.perf_counter()
        while True:
            if _key_down(VK_QUIT):
                break
            t0 = time.perf_counter()
            dx, dy = rec.mouse.read_and_reset()
            t1 = time.perf_counter()
            keys = _read_keys()
            lclick, rclick = _read_clicks()
            t2 = time.perf_counter()
            frame, t_cap = rec.cap.grab()
            t3 = time.perf_counter()
            # record assembly + list append, mirroring record()'s real work
            _rec = {"t": t_cap, "keys": keys, "lclick": lclick,
                    "rclick": rclick, "dx": dx, "dy": dy}
            _ = np.asarray(frame, dtype=np.uint8)  # the per-frame array cost
            t4 = time.perf_counter()
            mouse_t += t1 - t0
            keys_t += t2 - t1
            grab_t += t3 - t2
            asm_t += t4 - t3
            n += 1
            if time.perf_counter() - t_start > seconds:
                break
    if n == 0:
        print("No frames.")
        return
    total = mouse_t + keys_t + grab_t + asm_t
    per_frame_ms = total / n * 1000.0
    fps = n / total if total else float("nan")
    print(f"\nProfiled {n} frames, unpaced.")
    print(f"Per-frame total: {per_frame_ms:.2f} ms  ->  {fps:.1f} FPS ceiling "
          f"(unpaced, no sleep)")
    print("Stage breakdown (ms/frame and share):")
    for label, t in [("mouse read", mouse_t), ("keys+clicks", keys_t),
                     ("frame grab (mss+resize)", grab_t),
                     ("record assembly", asm_t)]:
        ms = t / n * 1000.0
        print(f"  {label:<26} {ms:6.2f} ms  ({ms/per_frame_ms*100:4.0f}%)")
    print()
    if grab_t > 0.6 * total:
        print("Diagnosis: the frame grab dominates. This is the known slow mss +")
        print("resize path (D-013/D-014) — expected, not a loop bug. A lower FPS")
        print("floor is honest; dxcam is the documented lever if the full agent")
        print("loop later needs more (D-014).")
    else:
        print("Diagnosis: the grab is NOT the sole cost — another stage is a large")
        print("share. Worth a closer look before enshrining a floor; see which row")
        print("above is unexpectedly heavy.")
    return fps


def record(seconds=60.0, name=None):
    """Record a session to disk as an .npz: frames + aligned action arrays.

    A session that round-trips to disk and reloads intact. Frames and action
    arrays are stored index-aligned: row i of every per-frame array corresponds
    to frame i.

    FORMAT: this writes the authoritative v1 schema locked in Issue #5 — see
    DATA_FORMAT.md at the repo root for the full field list, dtypes, and the
    self-description fields (schema_version / geom / loop_fps_target). Any change
    to what's written here must bump schema_version and update DATA_FORMAT.md in
    the same commit.
    """
    os.makedirs(_REC_DIR, exist_ok=True)
    if name is None:
        name = time.strftime("session_%Y%m%d_%H%M%S")
    path = os.path.join(_REC_DIR, name + ".npz")

    frames = []
    ts = []
    keys = []
    lclicks = []
    rclicks = []
    dxs = []
    dys = []

    print(f"Recording to {path}. Press F8 to stop (or after {seconds:.0f}s).")
    print("Play normally. Keep CS2 focused and in-game.\n")
    dropped = 0
    with Recorder() as rec:
        n = 0
        t_start = time.perf_counter()
        last_t = None
        while True:
            loop_start = time.perf_counter()
            if _key_down(VK_QUIT):
                break
            frame, r = rec.read_frame_record()
            frames.append(frame)
            ts.append(r["t"])
            keys.append(r["keys"])
            lclicks.append(r["lclick"])
            rclicks.append(r["rclick"])
            dxs.append(r["dx"])
            dys.append(r["dy"])
            # crude dropped-frame proxy: gaps much longer than the frame budget
            if last_t is not None:
                gap = r["t"] - last_t
                if gap > 1.8 / LOOP_FPS:
                    dropped += 1
            last_t = r["t"]
            n += 1
            if n % 20 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  {n} frames, {n/elapsed:.1f} FPS, {dropped} long-gaps",
                      end="\r")
            if time.perf_counter() - t_start > seconds:
                break
            _pace(loop_start, LOOP_FPS)

    frames = np.asarray(frames, dtype=np.uint8)
    ts = np.asarray(ts, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.uint8)
    gw, gh = cfg.GAME_RES
    ih, iw = cfg.MODEL_INPUT_HW
    geom = (f"fullscreen {gw}x{gh} -> crop "
            f"L{cfg.CROP_LEFT}T{cfg.CROP_TOP}W{cfg.CROP_WIDTH}H{cfg.CROP_HEIGHT} "
            f"-> {iw}x{ih} {cfg.COLOR_FORMAT}")
    actions = {
        # ── schema self-description (DATA_FORMAT.md v1, Issue #5) ──
        "schema_version": np.array(1),
        "geom": np.array(geom),
        "loop_fps_target": np.array(LOOP_FPS),
        # ── per-frame arrays (length N, index-aligned to frames) ──
        "timestamps": ts,
        "keys": keys,                       # (N, len(LOGGED_KEYS))
        "key_names": np.array([n for n, _ in LOGGED_KEYS]),
        "lclick": np.asarray(lclicks, dtype=np.uint8),
        "rclick": np.asarray(rclicks, dtype=np.uint8),
        "dx": np.asarray(dxs, dtype=np.int32),
        "dy": np.asarray(dys, dtype=np.int32),
    }
    np.savez_compressed(path, frames=frames, **actions)

    elapsed = time.perf_counter() - t_start
    print(f"\nSaved {len(frames)} frames to {path}")
    print(f"  {elapsed:.1f}s, mean {len(frames)/elapsed:.1f} FPS, "
          f"{dropped} long-gap frames ({100*dropped/max(len(frames),1):.1f}%)")

    # Immediately reload to prove the round-trip (an M0 exit criterion).
    with np.load(path, allow_pickle=False) as d:
        assert d["frames"].shape[0] == len(frames)
        assert d["keys"].shape[0] == len(frames)
        assert d["dx"].shape[0] == len(frames)
    print("  Round-trip OK: reloaded and shapes match.")
    return path


def verify(seconds=12.0):
    """SCRIPTED-MOTION ALIGNMENT CHECK — this is what actually tests the gate.

    A loop that merely runs does not prove input matches frame. This does: it
    records a short session while you perform a prescribed motion, then checks
    the logged signal matches the instruction, timed correctly.

    The check is mouse-based because mouse delta is the hardest and most
    sync-sensitive signal. You'll be prompted to sweep the mouse RIGHT for a few
    seconds, then LEFT. We then verify:
      * during the RIGHT phase, logged dx is predominantly positive,
      * during the LEFT phase, logged dx is predominantly negative,
      * the sign flip in the data lines up (within a frame or two) with when you
        actually switched — i.e. the input timeline tracks the frame timeline.

    If the phases don't separate cleanly, sync (or raw-mouse reading) is not
    trustworthy — investigate before recording real data; per #3, consider the
    kill flag.
    """
    print("ALIGNMENT VERIFY. Follow the prompts. Keep CS2 focused and in-game\n"
          "(cursor locked) so this tests the real conditions.\n")
    half = seconds / 2.0
    records = []
    phase_marks = []  # (index_at_phase_start, label)

    with Recorder() as rec:
        # PHASE 1: sweep right
        print(">>> Now sweep the mouse steadily to the RIGHT until told to stop.")
        t_start = time.perf_counter()
        phase_marks.append((0, "RIGHT"))
        while time.perf_counter() - t_start < half:
            loop_start = time.perf_counter()
            _frame, r = rec.read_frame_record()
            records.append(r)
            _pace(loop_start, LOOP_FPS)

        # PHASE 2: sweep left
        print(">>> Now sweep the mouse steadily to the LEFT until told to stop.")
        phase_marks.append((len(records), "LEFT"))
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < half:
            loop_start = time.perf_counter()
            _frame, r = rec.read_frame_record()
            records.append(r)
            _pace(loop_start, LOOP_FPS)

    print("\nStopped. Analysing alignment...\n")
    dx = np.array([r["dx"] for r in records], dtype=np.int64)
    n = len(records)
    split = phase_marks[1][0]

    right_dx = dx[:split]
    left_dx = dx[split:]

    def _summ(arr, label):
        if arr.size == 0:
            return f"  {label}: no frames"
        pos = int((arr > 0).sum())
        neg = int((arr < 0).sum())
        return (f"  {label} phase: {arr.size} frames, "
                f"mean dx={arr.mean():+.1f}, {pos} positive / {neg} negative")

    print(_summ(right_dx, "RIGHT"))
    print(_summ(left_dx, "LEFT"))

    # Verdicts. Judge direction using ONLY frames where the mouse actually moved
    # (dx != 0). A hand sweep has natural pauses between strokes that log dx=0;
    # counting those zeros against the ratio is wrong — the question is "when you
    # moved, did it move the right way," not "did you move every single frame."
    # The earlier version divided by all frames incl. zeros and produced false
    # FAILs on clean data. This measures the actual thing.
    r_moved = right_dx[right_dx != 0]
    l_moved = left_dx[left_dx != 0]
    r_pos_frac = (r_moved > 0).mean() if r_moved.size else 0.0
    l_neg_frac = (l_moved < 0).mean() if l_moved.size else 0.0

    print(f"  RIGHT: of {r_moved.size} moving frames, {r_pos_frac*100:.0f}% "
          f"were rightward (+dx).")
    print(f"  LEFT:  of {l_moved.size} moving frames, {l_neg_frac*100:.0f}% "
          f"were leftward (-dx).")

    # Pass each phase if the means clearly separate AND, among moving frames, the
    # intended direction dominates (>=80%). Means-separation is the primary
    # evidence sync works; the fraction guards against a near-random mix.
    ok_right = right_dx.size > 0 and right_dx.mean() > 0 and r_pos_frac >= 0.8
    ok_left = left_dx.size > 0 and left_dx.mean() < 0 and l_neg_frac >= 0.8

    # Where does the data actually switch from right-dominant to left-dominant?
    # NOT "first negative frame" — hand motion jitters, and a single negative
    # sample at the start is normal, not the transition. Instead find the split
    # index k that best separates "mostly positive before k" from "mostly
    # negative after k", by maximising correctly-classified frames. That locates
    # the real behavioural boundary robustly, then we check it lands near where
    # you were actually told to switch (`split`).
    best_k = None
    best_score = -1
    for k in range(1, n):
        correct = int((dx[:k] > 0).sum() + (dx[k:] < 0).sum())
        if correct > best_score:
            best_score = correct
            best_k = k
    # Fraction of frames consistent with a single switch at best_k — a cleanliness
    # measure. ~1.0 means the two phases are almost perfectly separated.
    separation = best_score / n if n else 0.0

    print()
    print(f"  You switched direction at frame {split}. Best-fit switch in the")
    print(f"  logged data is at frame {best_k} "
          f"({separation*100:.0f}% of frames consistent with one clean switch).")
    if best_k is not None:
        drift = abs(best_k - split)
        print(f"  Offset between the two: {drift} frame(s) "
              f"(small = the input timeline tracks the frame timeline).")

    # A trustworthy result: both phases separate by direction AND the data's
    # own switch point sits close to where you actually switched. The tolerance
    # is a fixed TIME (~1.5s), not a frame-percentage: the offset is dominated by
    # human reaction to the "switch" prompt, which is a wall-clock lag, so a
    # frame-fraction tolerance wrongly tightened on shorter/faster sessions.
    # (Earlier frame-% version flagged clean data as MOSTLY-OK on short runs.)
    reaction_tol_frames = int(1.5 * LOOP_FPS)
    boundary_ok = best_k is not None and abs(best_k - split) <= reaction_tol_frames

    print()
    if ok_right and ok_left and boundary_ok:
        print("RESULT: PASS — logged mouse motion matches the scripted direction in")
        print("each phase, and the data's switch point lines up with when you")
        print("actually switched. Frame/input alignment looks trustworthy.")
        print("(Re-run a couple of times; sync is worth over-checking.)")
        passed = True
    elif ok_right and ok_left and not boundary_ok:
        print("RESULT: MOSTLY OK — both phases separate by direction (sync of sign")
        print("is working), but the data's switch point is further from the")
        print("boundary than expected. Usually means you started/stopped sweeping")
        print("off the prompt timing rather than a real sync fault. Re-run and try")
        print("to switch direction promptly when prompted.")
        passed = False
    else:
        print("RESULT: FAIL — the phases did NOT separate by direction.")
        print("  Either raw deltas aren't reading in-game (run")
        print("  `python -m src.raw_mouse --selftest` with CS2 focused), or the")
        print("  loop isn't pairing inputs with frames correctly. Do NOT record")
        print("  real data until this passes — per PROJECT_ISSUES #3, a sync that")
        print("  can't be shown reliable is a kill-flag condition, not a detail.")
        passed = False
    return passed


def _build_parser():
    p = argparse.ArgumentParser(
        description="Synchronized frame+input recorder (Issue #3, M0 gate).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--verify", action="store_true",
                   help="scripted-motion alignment check — RUN THIS FIRST")
    g.add_argument("--record", action="store_true",
                   help="record a session to disk (.npz)")
    g.add_argument("--dryrun", action="store_true",
                   help="run the loop with a live readout, save nothing")
    g.add_argument("--profile", action="store_true",
                   help="time each loop stage to find what limits FPS (saves nothing)")
    p.add_argument("--seconds", type=float, default=None,
                   help="duration; defaults per mode")
    p.add_argument("--name", type=str, default=None,
                   help="optional recording filename stub")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.verify:
        verify(seconds=args.seconds or 12.0)
    elif args.record:
        record(seconds=args.seconds or 60.0, name=args.name)
    elif args.dryrun:
        dryrun(seconds=args.seconds or 20.0)
    elif args.profile:
        profile(seconds=args.seconds or 20.0)
    else:
        print("Choose a mode: --verify (do first), --dryrun, or --record.")
        print("See `python -m src.recorder -h`.")


if __name__ == "__main__":
    main()
