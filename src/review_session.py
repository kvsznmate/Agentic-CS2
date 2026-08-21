"""review_session.py — scrub a recording like a video (FPV + radar side by side).

WHY THIS EXISTS. Until now a session could only be spot-checked as a handful of
dumped PNGs (inspect_recording --dump). That makes it hard to answer the two
questions that actually matter before trusting a session:
  1. Did the capture record real, continuous gameplay (not a stuck/black feed)?
  2. Which frames are junk (buy menu, halftime, dead/spectate) — and does the
     blank-radar keep-mask (clean_session.py) actually catch them?

This tool plays a session back: the FPV and the high-res radar shown together,
a seekbar, play/pause, single-frame stepping, a live readout of that frame's
logged actions (keys/clicks/mouse dx,dy), AND the per-frame GSI state on v4/v5
sessions (alive/dead, round phase, and on v5 the health/weapon/ammo state
features) so you can SEE input, image, and game-state together — the same
alignment the M0 gate proved (#3), now visible frame by frame. The GSI line is
coloured green when alive and red when dead/spectating. Older formats (v1-v3)
simply omit the GSI line.

If a keep-mask exists for the session (from `python -m src.clean_session`), frames
the mask marks as BLANK/junk are tinted red and labelled, so you can:
  * confirm the cut is dropping the right frames, and
  * hunt for the junk it CANNOT catch — e.g. the buy menu open while the minimap
    still renders behind it, which keeps a normal radar variance and so passes the
    variance cut as "gameplay." That class of junk needs an FPV-side signal, not
    variance; this viewer is how you find out whether it's present in your data.

Reads every on-disk format via data_loader (v3 FPV+radar, v2 FPV-only, v1 file).
On a v1/v2 session there is no radar; the radar pane shows a placeholder and only
the FPV is scrubbed.

CONTROLS (focus the OpenCV window):
  space        play / pause
  . or  ->     step one frame forward   (also Right arrow, where supported)
  , or  <-     step one frame back      (also Left arrow)
  m            jump to next mask-DROPPED (blank) frame
  n            jump to next mask-KEPT  (gameplay) frame
  [ / ]        slow down / speed up playback
  g            toggle the radar-variance / mask overlay text
  s            save the current composite frame as a PNG (to data/capture_debug)
  q or Esc     quit
Trackbar: drag to seek anywhere in the session.

Usage:
  python -m src.review_session                  # newest usable session
  python -m src.review_session --session NAME    # a specific session (folder or path)
  python -m src.review_session --fps 30          # initial playback rate
  python -m src.review_session --no-mask         # ignore any keep-mask overlay

Performance: frames are decompressed per session and held in memory by
SessionDataset (fine at current scale). Playback is paced in the display loop;
the file's own recorded FPS is shown for reference but you can play faster/slower.
"""

import argparse
import json
import os
import time

import numpy as np

from src import data_loader as dl
from src import clean_session as cs


_DUMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "capture_debug")

# Display sizes. FPV is 150x270; radar is 128x128. We upscale both for viewing.
_FPV_SCALE = 3          # 150x270 -> 450x810
_RADAR_VIEW = 384       # radar shown at 384x384 (128 * 3)
_PANEL_BG = 30          # dark grey background for the composite


def _load_mask(path):
    """Load a session's keep-mask sidecar if present + valid. Returns dict or None.

    Validates source_frames against the session length so a STALE mask (written
    for a different frame count, e.g. the session was re-recorded) is ignored with
    a warning rather than misaligning the overlay.
    """
    mask_dir = path if os.path.isdir(path) else (
        (path[:-4] if path.endswith(".npz") else path) + ".mask")
    mpath = os.path.join(mask_dir, cs.MASK_FILE)
    if not os.path.isfile(mpath):
        return None
    try:
        with np.load(mpath, allow_pickle=False) as d:
            mask = {k: d[k] for k in d.files}
    except Exception as e:  # noqa: BLE001
        print(f"  (keep-mask present but unreadable: {e!r} — ignoring)")
        return None
    return mask


def _fmt_actions(arrays, i, key_names):
    """One-line action readout for frame i: held keys, clicks, mouse delta."""
    held = [key_names[j] for j, v in enumerate(arrays["keys"][i]) if v]
    lc = int(arrays["lclick"][i]); rc = int(arrays["rclick"][i])
    dx = int(arrays["dx"][i]); dy = int(arrays["dy"][i])
    keys_s = "+".join(held) if held else "-"
    clicks = ("L" if lc else "") + ("R" if rc else "")
    clicks = clicks if clicks else "-"
    return f"keys[{keys_s}]  click[{clicks}]  mouse(dx={dx:+d}, dy={dy:+d})"


def _fmt_gsi(arrays, i):
    """One-line GSI state readout for frame i, or None if this session has no GSI.

    Degrades by format: v1-v3 have no GSI at all (returns None -> caller draws
    nothing); v4 has `alive`/`round_phase` but no state features; v5 adds
    `health`/`active_weapon`/`ammo_clip`/`ammo_reserve` (D-033). Each field is
    guarded so a v4 session doesn't crash on the missing v5 arrays.

    The alive flag is the own-POV rule (D-032): alive=1 means our own POV AND
    health>0, so a frame spectating a living teammate reads alive=0. We surface
    playing-vs-SPECTATING too when the array is present, since 'dead' and
    'spectating' are different reasons a frame is non-gameplay and it helps to see
    which. Ammo shows only for weapons that HAVE ammo (sentinel -1 -> knife/C4,
    shown as '-').
    """
    if "alive" not in arrays:
        return None  # v1/v2/v3: no GSI recorded

    alive = int(arrays["alive"][i])
    life = "ALIVE" if alive else "dead"
    phase = ""
    if "round_phase" in arrays:
        rp = arrays["round_phase"][i]
        rp = rp.decode() if isinstance(rp, bytes) else str(rp)
        phase = f"  round[{rp}]"

    # State features (v5). All-or-nothing group, but guard each anyway.
    state_bits = ""
    if "health" in arrays:
        hp = int(arrays["health"][i])
        state_bits += f"  hp={hp:3d}"
    if "active_weapon" in arrays:
        wp = arrays["active_weapon"][i]
        wp = wp.decode() if isinstance(wp, bytes) else str(wp)
        wp = wp if wp else "-"
        state_bits += f"  wpn[{wp}]"
    if "ammo_clip" in arrays and "ammo_reserve" in arrays:
        clip = int(arrays["ammo_clip"][i])
        reserve = int(arrays["ammo_reserve"][i])
        ammo = f"{clip}/{reserve}" if clip >= 0 else "-"   # -1 sentinel = no ammo concept
        state_bits += f"  ammo[{ammo}]"

    return f"GSI: {life}{phase}{state_bits}"


def _life_state(arrays, i):
    """(label, colour BGR) for the frame's life state, or None if no GSI.

    Keyed purely off the stored `alive` flag. Note the recorded format keeps a
    BARE boolean (D-031): spectating a living teammate is already folded into
    alive=0 by the D-032 rule, but on disk we cannot tell 'dead on own POV' from
    'spectating' apart — both are alive=0. So the dead label says both, honestly,
    rather than claiming a distinction the data doesn't carry.
    """
    if "alive" not in arrays:
        return None
    if int(arrays["alive"][i]):
        return ("ALIVE", (120, 230, 120))
    return ("dead / spectating", (80, 80, 255))


def _compose(fpv, radar, has_radar):
    """Build the side-by-side composite canvas (FPV left, radar right).

    Returns a BGR uint8 image. Leaves a header/footer margin for text drawn by
    the caller. FPV and radar are upscaled to their view sizes; radar pane shows a
    placeholder when the session has no radar (v1/v2).
    """
    import cv2
    fh, fw = fpv.shape[:2]
    fpv_big = cv2.resize(fpv, (fw * _FPV_SCALE, fh * _FPV_SCALE),
                         interpolation=cv2.INTER_NEAREST)
    fbh, fbw = fpv_big.shape[:2]

    if has_radar and radar is not None:
        radar_big = cv2.resize(radar, (_RADAR_VIEW, _RADAR_VIEW),
                               interpolation=cv2.INTER_NEAREST)
    else:
        radar_big = np.full((_RADAR_VIEW, _RADAR_VIEW, 3), 60, np.uint8)
        cv2.putText(radar_big, "no radar (v1/v2)", (30, _RADAR_VIEW // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    pane_h = max(fbh, radar_big.shape[0])
    header, footer, gap, margin = 34, 80, 24, 16
    canvas_h = header + pane_h + footer
    canvas_w = margin + fbw + gap + radar_big.shape[1] + margin
    canvas = np.full((canvas_h, canvas_w, 3), _PANEL_BG, np.uint8)

    fy = header
    canvas[fy:fy + fbh, margin:margin + fbw] = fpv_big
    rx = margin + fbw + gap
    ry = header
    canvas[ry:ry + radar_big.shape[0], rx:rx + radar_big.shape[1]] = radar_big

    cv2.putText(canvas, "FPV (model input, 3x)", (margin, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "radar (128, 3x)", (rx, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas, (header, pane_h, footer, margin)


def _next_in_state(keep, start, target_state, forward=True):
    """Index of the next frame whose keep-state == target_state, from start.

    target_state True -> next KEPT frame; False -> next DROPPED(blank) frame.
    Returns start unchanged if none found. Wraps is NOT done — bounded search.
    """
    n = keep.shape[0]
    rng = range(start + 1, n) if forward else range(start - 1, -1, -1)
    for i in rng:
        if bool(keep[i]) == target_state:
            return i
    return start


def review(path, init_fps=15.0, use_mask=True):
    try:
        import cv2
    except ImportError:
        print("cv2 is required for the review tool but is not installed.")
        return

    arrays = dl.load_session_arrays(path)
    n = int(arrays["frames"].shape[0])
    if n == 0:
        print("Session has 0 frames.")
        return
    has_radar = "radar" in arrays
    key_names = [s.decode() if isinstance(s, bytes) else str(s)
                 for s in arrays["key_names"]]

    # Real recorded FPS from timestamps, for reference in the header.
    ts = arrays["timestamps"]
    real_fps = (1.0 / np.diff(ts).mean()) if ts.size > 1 else float("nan")

    mask = _load_mask(path) if use_mask else None
    keep = variance = None
    if mask is not None:
        src_n = int(mask.get("source_frames", n))
        if src_n != n:
            print(f"  (keep-mask was built for {src_n} frames but session has {n} "
                  f"— stale mask, ignoring overlay)")
            mask = None
        else:
            keep = mask["keep"].astype(bool)
            variance = mask.get("variance")
            thr = float(mask.get("threshold", 0.0))
            n_drop = int((~keep).sum())
            print(f"  keep-mask: {n_drop}/{n} frames marked blank "
                  f"({100*n_drop/n:.1f}%), cut variance={thr:.1f}. "
                  f"Dropped frames tinted red.")
    if mask is None and use_mask:
        print("  (no keep-mask for this session — run `python -m src.clean_session "
              "--session <name>` to create one. Showing frames untinted.)")

    win = f"review: {dl.session_name(path)}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    state = {"i": 0, "playing": False, "fps": float(init_fps), "show_overlay": True}

    def _on_seek(pos):
        state["i"] = max(0, min(n - 1, pos))
    cv2.createTrackbar("frame", win, 0, n - 1, _on_seek)

    print("\nControls: space play/pause | . next , prev | m next blank | n next kept |")
    print("          [ slower ] faster | g overlay | s save PNG | q quit\n")

    last_draw = 0.0
    while True:
        i = state["i"]
        fpv = arrays["frames"][i]
        radar = arrays["radar"][i] if has_radar else None
        canvas, (header, pane_h, footer, margin) = _compose(fpv, radar, has_radar)
        ch = canvas.shape[0]

        # Red tint if this frame is mask-dropped (blank/junk).
        dropped = (keep is not None) and (not bool(keep[i]))
        if dropped:
            overlay = canvas.copy()
            overlay[:] = (0, 0, 160)
            canvas = cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0)

        # Footer text: frame counter, actions, GSI state, timing, mask state.
        y0 = header + pane_h + 20
        status = f"frame {i+1}/{n}   rec {real_fps:4.1f} FPS   play {state['fps']:4.1f} FPS"
        if state["playing"]:
            status += "  [PLAYING]"
        cv2.putText(canvas, status, (margin, y0 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 210, 170), 1, cv2.LINE_AA)
        # Glanceable life tag on the status row (green ALIVE / red dead), if GSI.
        life = _life_state(arrays, i)
        # Mask state stays on the status row, right side (where it began).
        if state["show_overlay"] and keep is not None:
            tag = "BLANK / dropped" if dropped else "gameplay / kept"
            col = (80, 80, 255) if dropped else (120, 230, 120)
            vtxt = ""
            if variance is not None:
                vtxt = f"  var={float(variance[i]):.0f} (cut {thr:.0f})"
            cv2.putText(canvas, tag + vtxt, (margin + 360, y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(canvas, _fmt_actions(arrays, i, key_names), (margin, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        # GSI state line (v4/v5): alive, round phase, hp, weapon, ammo on its own
        # full-width row so long weapon names / ammo don't collide with anything.
        # Coloured green alive / red dead for glanceability. Absent on v1-v3,
        # where _fmt_gsi returns None and we draw nothing.
        gsi_line = _fmt_gsi(arrays, i)
        if gsi_line is not None:
            gcol = life[1] if life is not None else (210, 200, 140)
            cv2.putText(canvas, gsi_line, (margin, y0 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, gcol, 1, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        if cv2.getTrackbarPos("frame", win) != i:
            cv2.setTrackbarPos("frame", win, i)

        # Pace playback; keep the UI responsive with a short waitKey either way.
        if state["playing"]:
            now = time.perf_counter()
            due = 1.0 / max(state["fps"], 1e-3)
            wait_ms = max(1, int((due - (now - last_draw)) * 1000))
        else:
            wait_ms = 30
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord("q"), 27):            # q or Esc
            break
        elif key == ord(" "):
            state["playing"] = not state["playing"]
            last_draw = time.perf_counter()
        elif key in (ord("."), 83):          # '.' or Right
            state["playing"] = False
            state["i"] = min(n - 1, i + 1)
        elif key in (ord(","), 81):          # ',' or Left
            state["playing"] = False
            state["i"] = max(0, i - 1)
        elif key == ord("m") and keep is not None:
            state["playing"] = False
            state["i"] = _next_in_state(keep, i, target_state=False, forward=True)
        elif key == ord("n") and keep is not None:
            state["playing"] = False
            state["i"] = _next_in_state(keep, i, target_state=True, forward=True)
        elif key == ord("]"):
            state["fps"] = min(120.0, state["fps"] * 1.5)
        elif key == ord("["):
            state["fps"] = max(1.0, state["fps"] / 1.5)
        elif key == ord("g"):
            state["show_overlay"] = not state["show_overlay"]
        elif key == ord("s"):
            os.makedirs(_DUMP_DIR, exist_ok=True)
            out = os.path.join(_DUMP_DIR,
                               f"{dl.session_name(path)}_review_{i:06d}.png")
            cv2.imwrite(out, canvas)
            print(f"  saved {out}")

        # Advance when playing and the frame's time is up.
        if state["playing"]:
            now = time.perf_counter()
            if now - last_draw >= 1.0 / max(state["fps"], 1e-3):
                last_draw = now
                if i >= n - 1:
                    state["playing"] = False       # stop at the end
                else:
                    state["i"] = i + 1

    cv2.destroyAllWindows()


def _resolve_target(session_arg):
    """Newest usable session if none named, else the named folder/path."""
    if session_arg is None:
        sessions = dl.discover_sessions(report=True)
        if not sessions:
            raise FileNotFoundError(
                f"No usable sessions in {dl._REC_DIR}. Record one with "
                f"`python -m src.recorder --record`.")
        return max(sessions, key=os.path.getmtime)
    if os.path.isdir(session_arg) or os.path.isfile(session_arg):
        return session_arg
    cand = os.path.join(dl._REC_DIR, os.path.basename(session_arg))
    if os.path.isdir(cand):
        return cand
    cand_npz = cand if cand.endswith(".npz") else cand + ".npz"
    if os.path.isfile(cand_npz):
        return cand_npz
    raise FileNotFoundError(f"Could not find session: {session_arg}")


def _build_parser():
    p = argparse.ArgumentParser(
        description="Scrub a recording (FPV + radar) with a seekbar; tint blank/junk "
                    "frames from the keep-mask. See clean_session.py for the mask.")
    p.add_argument("--session", type=str, default=None, metavar="NAME",
                   help="session folder name or path (default: newest usable session)")
    p.add_argument("--fps", type=float, default=15.0,
                   help="initial playback FPS (default 15; adjust live with [ and ])")
    p.add_argument("--no-mask", action="store_true",
                   help="ignore any keep-mask; do not tint frames")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    target = _resolve_target(args.session)
    review(target, init_fps=args.fps, use_mask=not args.no_mask)


if __name__ == "__main__":
    main()
