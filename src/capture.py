"""capture.py — CS2 screen capture (Issue #2, GATE).

Grabs the CS2 game image via mss (full-screen grab + crop) and resizes it to the
fixed model input size. This is the capture HALF of the M0 capture+sync gate;
input logging and frame/input synchronization are Issue #3 and are NOT here. The
grab function is shaped so #3 can attach to it without a rewrite: every grab
returns the frame AND the timestamp taken as close as possible to the grab, so
inputs can later be aligned to that timestamp.

DECISIONS honoured:
  D-010  capture is mss full-screen grab + crop; the reference's win32 BitBlt
         path (screen_input.py) is dead on Source 2 and not carried over.
  D-012  CS2 fullscreen at native 1920x1080; crop is full-frame; model input is
         150x270 (H,W), 16:9, so the downscale carries no aspect distortion.
  Q4     frames leave in BGR.

Machine-specific geometry (monitor, crop rectangle, input size) lives in
capture_config.py. Confirm the crop with:

    python -m src.capture --calibrate     # save a full grab + crop to inspect

Other entry points:
    python -m src.capture --benchmark     # measure sustained grab+crop+resize FPS
    python -m src.capture --preview       # live window of the cropped/resized feed

Usable-rate bar for this gate (proposed, see PROJECT_ISSUES #2 / benchmark
summary): sustained grab+crop+resize >= 30 FPS on this machine, comfortably
above the study's 16 FPS loop and leaving headroom for #3's logging and the
models. --benchmark reports the measured number so the bar is checked honestly.
"""

import argparse
import os
import time

import cv2
import numpy as np
from mss import mss

from src import capture_config as cfg


# Where calibration / debug images are written. Under the repo but gitignored
# (data/ is in .gitignore), so inspection artefacts never get committed.
_DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "capture_debug")


class Capture:
    """Owns a persistent mss instance and the resolved crop geometry.

    An mss() instance is reused across grabs on purpose: constructing one per
    frame is a real bottleneck, and a persistent instance is what makes a high
    grab rate achievable. Use as a context manager so the instance is closed:

        with Capture() as cap:
            frame, t = cap.grab()
    """

    def __init__(self):
        self.monitor_index = cfg.MONITOR_INDEX
        self.crop_left = cfg.CROP_LEFT
        self.crop_top = cfg.CROP_TOP
        self.crop_width = cfg.CROP_WIDTH
        self.crop_height = cfg.CROP_HEIGHT
        # capture_config stores (H, W); cv2.resize wants (W, H).
        h, w = cfg.MODEL_INPUT_HW
        self._resize_wh = (w, h)

        self._sct = mss()
        self._monitor = self._resolve_monitor()
        # The absolute-coordinate region mss will grab: the crop rectangle,
        # offset by the monitor's own origin so LEFT/TOP are relative to the
        # monitor's top-left rather than the whole virtual desktop.
        self._grab_region = {
            "left": self._monitor["left"] + self.crop_left,
            "top": self._monitor["top"] + self.crop_top,
            "width": self.crop_width,
            "height": self.crop_height,
        }

    def _resolve_monitor(self):
        monitors = self._sct.monitors  # [0] = virtual union, [1..] = physical
        if self.monitor_index < 1 or self.monitor_index >= len(monitors):
            raise ValueError(
                f"MONITOR_INDEX={self.monitor_index} is out of range. "
                f"mss sees {len(monitors) - 1} physical monitor(s): "
                f"valid indices are 1..{len(monitors) - 1}. "
                f"Run `python -m src.capture --calibrate` to list them."
            )
        return monitors[self.monitor_index]

    def grab(self):
        """Return (frame, timestamp).

        frame      : np.ndarray, shape (H, W, 3), BGR uint8, resized to the
                     model input size (cfg.MODEL_INPUT_HW).
        timestamp  : float, time.perf_counter() taken immediately after the
                     raw grab — the anchor #3 will sync inputs against.

        The timestamp is read right after the grab (not after resize) so it
        reflects when the pixels were actually captured, which is what input
        alignment cares about.
        """
        raw = self._sct.grab(self._grab_region)
        t = time.perf_counter()
        # mss returns BGRA in a buffer; view as array and drop alpha -> BGR.
        img = np.asarray(raw)[:, :, :3]
        # INTER_LINEAR, not INTER_AREA: at a ~7x downscale (1920x1080 -> 270x150)
        # INTER_AREA averages over every source pixel per output cell, which
        # measured at ~24 ms/frame and was the main bottleneck (benchmark,
        # 2026-08). INTER_LINEAR is far cheaper and the quality difference on a
        # 150x270 CNN input is negligible. See D-013.
        frame = cv2.resize(img, self._resize_wh, interpolation=cv2.INTER_LINEAR)
        return frame, t

    def grab_full_monitor(self):
        """Grab the entire selected monitor, uncropped, as BGR uint8.

        Used only by calibration so you can see the whole screen and read off
        where the game image actually sits. Not part of the hot path.
        """
        raw = self._sct.grab(self._monitor)
        return np.asarray(raw)[:, :, :3]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._sct.close()


def _ensure_debug_dir():
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    return _DEBUG_DIR


def calibrate():
    """Save a full-monitor grab and the current crop so the crop can be checked.

    Writes two images to data/capture_debug/:
      full_monitor.png  — the whole selected monitor, with the current crop
                          rectangle drawn on it as a green box.
      cropped.png       — exactly what the crop currently yields (before resize).
    Then prints the monitor list and how to correct capture_config if the box
    isn't sitting on the game image. This is the honest check that makes #2 a
    real gate rather than an assumption.
    """
    out = _ensure_debug_dir()
    with Capture() as cap:
        monitors = cap._sct.monitors
        print("Monitors mss can see (index: geometry):")
        print(f"  0: {monitors[0]}   <- virtual union of all monitors (do not use)")
        for i in range(1, len(monitors)):
            marker = "  <- MONITOR_INDEX (current)" if i == cap.monitor_index else ""
            print(f"  {i}: {monitors[i]}{marker}")
        print()

        full = cap.grab_full_monitor().copy()
        # Draw the crop rectangle (monitor-relative coords) on the full grab.
        x0, y0 = cap.crop_left, cap.crop_top
        x1, y1 = x0 + cap.crop_width, y0 + cap.crop_height
        h_full, w_full = full.shape[:2]
        in_bounds = 0 <= x0 < x1 <= w_full and 0 <= y0 < y1 <= h_full
        cv2.rectangle(full, (x0, y0), (x1, y1), (0, 255, 0), 3)
        full_path = os.path.join(out, "full_monitor.png")
        cv2.imwrite(full_path, full)

        cropped = cap.grab_full_monitor()[y0:y1, x0:x1] if in_bounds else None
        crop_path = os.path.join(out, "cropped.png")
        if cropped is not None and cropped.size:
            cv2.imwrite(crop_path, cropped)

    print(f"Selected monitor: index {cap.monitor_index}, size "
          f"{cap._monitor['width']}x{cap._monitor['height']}.")
    print(f"Current crop rectangle (monitor-relative): "
          f"left={cfg.CROP_LEFT} top={cfg.CROP_TOP} "
          f"width={cfg.CROP_WIDTH} height={cfg.CROP_HEIGHT}.")
    gw, gh = cfg.GAME_RES
    if (cfg.CROP_WIDTH, cfg.CROP_HEIGHT) == (gw, gh):
        print("  note: crop size equals full GAME_RES — you are grabbing the whole "
              "window including any HUD borders/title bar. That may be fine, but "
              "the study cropped inward; confirm against the saved crop.")
    if not in_bounds:
        print("  WARNING: the crop rectangle falls outside the monitor bounds. "
              "cropped.png was NOT written. Fix CROP_* in capture_config.py.")
    print()
    print(f"Wrote:\n  {full_path}\n  (green box = current crop)")
    if cropped is not None and cropped.size:
        print(f"  {crop_path}\n  (exactly what the crop yields, before resize)")
    print()
    print("To correct: open full_monitor.png. The green box should sit exactly on "
          "the game image — no desktop, no window title bar, no HUD you don't want. "
          "If it's off, adjust CROP_LEFT/TOP/WIDTH/HEIGHT in src/capture_config.py "
          "and re-run --calibrate until the box is right. If the wrong screen was "
          "grabbed entirely, change MONITOR_INDEX using the list above.")


def benchmark(seconds=10.0, warmup=20):
    """Measure sustained FPS AND break the per-frame cost into grab vs resize.

    Runs the real hot path for `seconds` and reports achieved FPS, plus how much
    of each frame is the raw mss grab versus the cv2 resize. That split is the
    whole point: if the grab dominates, the bottleneck is mss (a capture-backend
    problem, possibly a D-010 change); if the resize dominates, it's our
    processing. `warmup` grabs are discarded first. Purely a measurement — no
    frames are kept. Times the same operations grab() does, inline, so the
    breakdown reflects the real path.
    """
    bar = 20.0  # committed #2 gate bar (D-014); ~25 FPS measured clears it.
    with Capture() as cap:
        shape = None
        for _ in range(warmup):
            frame, _ = cap.grab()
            shape = frame.shape

        n = 0
        grab_total = 0.0
        resize_total = 0.0
        t_start = time.perf_counter()
        t_end = t_start + seconds
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            raw = cap._sct.grab(cap._grab_region)
            img = np.asarray(raw)[:, :, :3]
            t1 = time.perf_counter()
            cv2.resize(img, cap._resize_wh, interpolation=cv2.INTER_LINEAR)
            t2 = time.perf_counter()
            grab_total += (t1 - t0)
            resize_total += (t2 - t1)
            n += 1
        elapsed = time.perf_counter() - t_start

    fps = n / elapsed if elapsed > 0 else float("nan")
    per_frame_ms = (elapsed / n * 1000.0) if n else float("nan")
    grab_ms = (grab_total / n * 1000.0) if n else float("nan")
    resize_ms = (resize_total / n * 1000.0) if n else float("nan")
    print(f"Grabbed {n} frames in {elapsed:.2f}s")
    print(f"Output frame shape: {shape} (H, W, C), BGR")
    print(f"Sustained rate: {fps:.1f} FPS  ({per_frame_ms:.2f} ms/frame)")
    print(f"  raw grab (mss):   {grab_ms:.2f} ms/frame  ({grab_ms / per_frame_ms * 100:.0f}%)")
    print(f"  resize (cv2):     {resize_ms:.2f} ms/frame  ({resize_ms / per_frame_ms * 100:.0f}%)")
    print(f"Gate bar (proposed): >= {bar:.0f} FPS  ->  "
          f"{'PASS' if fps >= bar else 'BELOW BAR'}")
    if fps < bar:
        if grab_ms > resize_ms * 3:
            print("  Diagnosis: the raw mss grab dominates — this is a capture-backend")
            print("  limit, not our processing. mss full-screen grab is slow on Windows;")
            print("  a faster backend (e.g. dxcam / Desktop Duplication) is the likely")
            print("  fix, but that changes D-010 and is a decision, not a silent swap.")
        else:
            print("  Diagnosis: the resize is a large share — unexpected for 150x270.")
            print("  Worth checking cv2 build / interpolation before touching capture.")
    return fps


def preview():
    """Live window of the cropped+resized feed — a quick 'is this the game?' check.

    Upscales the tiny model-input frame so it's viewable. Press q to quit. Useful
    right after calibration to watch the feed across different in-game scenes,
    which is exactly what #2's acceptance ("verified against different scenes")
    asks for.
    """
    print("Live preview of the model-input feed. Move around in CS2 to check the "
          "crop across scenes. Press q in the preview window to quit.")
    with Capture() as cap:
        while True:
            frame, _ = cap.grab()
            # Upscale for human viewing only; the model would see `frame` as-is.
            view = cv2.resize(frame, (frame.shape[1] * 3, frame.shape[0] * 3),
                              interpolation=cv2.INTER_NEAREST)
            cv2.imshow("capture preview (model input, 3x)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


def _build_parser():
    p = argparse.ArgumentParser(
        description="CS2 screen capture (Issue #2). Default action runs a short "
                    "benchmark. Use --calibrate first on a new machine.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--calibrate", action="store_true",
                   help="save a full grab + current crop to inspect and fix geometry")
    g.add_argument("--benchmark", action="store_true",
                   help="measure sustained grab+crop+resize FPS against the gate bar")
    g.add_argument("--preview", action="store_true",
                   help="live window of the cropped+resized model-input feed")
    p.add_argument("--seconds", type=float, default=10.0,
                   help="benchmark duration in seconds (default 10)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.calibrate:
        calibrate()
    elif args.preview:
        preview()
    else:
        # Default to benchmark; it's the safe, side-effect-free action.
        benchmark(seconds=args.seconds)


if __name__ == "__main__":
    main()
