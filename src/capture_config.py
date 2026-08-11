"""capture_config.py — machine-specific capture geometry for Issue #2.

This file holds the numbers that depend on *this* machine: which monitor CS2 is
on, the rectangle of the game image inside the full-screen grab, and the fixed
size frames are resized to. They are kept separate from capture.py so the
per-machine geometry lives in one obvious place and can be changed without
touching capture logic.

── HOW WE GOT HERE (calibrated 2026-08) ──────────────────────────────────────
The reference study (config.py / screen_input.py) cropped a 1024x768 *windowed*
CSGO with inward offsets (824x498) to concentrate the model on the centre. We
initially seeded that crop, but calibration on this machine showed it landing
in a corner and cutting away most of the game. This machine's native monitor is
1920x1080 (16:9), and CS2 is now run FULLSCREEN at that native resolution, so
the full-monitor grab is pure game edge to edge and the crop is simply the whole
frame. See DECISIONS.md D-012.

── THE 16:9 -> MODEL-INPUT DECISION (D-012) ──────────────────────────────────
The study's model input was 150x280 (H,W), which is ~4:3. Our source is now 16:9
(1920x1080). Squashing 16:9 straight into a 4:3 target would horizontally
distort every frame — a quiet quality bug, worst for aim. So the model input is
changed to a 16:9 shape, 150x270 (H,W): 1920/1080 = 1.778, and 150x270 = 1.80,
a ~1.2% deviation that's visually negligible while keeping both dimensions
cleanly divisible for CNN pooling. Resizing 1920x1080 -> 150x270 is then a pure
uniform downscale with NO aspect distortion. We drop exact parity with the
study's 150x280 input deliberately — D-001 says we don't run or compare against
their model, so matching their first-layer size buys nothing, whereas an
undistorted frame at our real resolution buys a lot.

Run calibration to confirm geometry:  python -m src.capture --calibrate
It saves a full grab + the current crop to disk so you can eyeball them.

── HOW THE CROP IS DEFINED ───────────────────────────────────────────────────
The crop is an absolute pixel rectangle within the chosen monitor
(CROP_LEFT/TOP/WIDTH/HEIGHT). Storing it this way (rather than per-edge offsets)
avoids a hidden dependence on the monitor's resolution.
"""

# ── Monitor selection ─────────────────────────────────────────────────────
# mss numbers monitors from 1 (monitor 0 is the "all monitors" virtual union).
# On a dual-monitor setup, set this to whichever physical monitor CS2 runs on.
# Calibration prints the monitor list so you can pick correctly.
MONITOR_INDEX = 2

# ── Game / display resolution (for reference / sanity checks) ─────────────
# CS2 is run FULLSCREEN at the monitor's native 1920x1080 (16:9). Not used
# directly for cropping (the rectangle below is), but calibration warns if the
# crop rectangle is inconsistent with this.
GAME_RES = (1920, 1080)  # (width, height)

# ── The crop rectangle, in pixels, within MONITOR_INDEX ───────────────────
# FULL-FRAME. CS2 fullscreen at native 1920x1080 fills the monitor edge to edge,
# so the full-monitor grab is already pure game — no desktop, taskbar, or title
# bar to exclude. Crop = the whole frame. Confirm via --calibrate: the green box
# should cover the entire image.
#
# NOTE — full-frame keeps the peripheral margins the study discarded to spend the
# model's limited resolution on the centre. That centre-vs-edges trade is
# deferred to the per-model sub-crops in the loader (#6: full FPV / centre /
# radar), tuned against real recorded data. Keeping the whole frame at capture
# ensures nothing downstream (especially the radar) is amputated.
CROP_LEFT = 0
CROP_TOP = 0
CROP_WIDTH = 1920
CROP_HEIGHT = 1080

# ── Model input size (H, W) — 16:9, see D-012 ─────────────────────────────
# Changed from the study's 150x280 (~4:3) to 150x270 (16:9) so that downscaling
# the 1920x1080 grab introduces NO aspect distortion. Height, then width. This
# is a soft value here; the authoritative on-disk schema is locked later in #5
# (DATA_FORMAT.md), which is where any change to this must be recorded.
MODEL_INPUT_HW = (150, 270)  # (height, width)

# ── Colour format ─────────────────────────────────────────────────────────
# Frames leave the capture layer in BGR (OpenCV-native, matches the study).
# mss returns BGRA; capture.py drops alpha to BGR. Recorded here because the
# channel order becomes part of the data contract finalized in #5.
COLOR_FORMAT = "BGR"


# ══════════════════════════════════════════════════════════════════════════
# RADAR REGION — for the two-resolution redesign (separate high-res radar crop)
# ══════════════════════════════════════════════════════════════════════════
# WHY THIS EXISTS: at the 150x270 model input the radar corner is too low-res to
# read self-position (the #7 finding). The fix: store a SEPARATE, higher-res
# radar crop taken from the FULL 1920x1080 grab BEFORE the aggressive FPV
# downscale — the FPV stays 150x270, the radar gets its own array sized to its
# own need (planned DATA_FORMAT.md v2 -> v3). See #7 / DECISIONS (entry added
# once the format lands).
#
# ── MEASURED (2026-08), still PENDING ONE CONFIRMATION PASS ──
# --radar-calibrate confirmed the radar is LEGIBLE when cropped from full-res
# (uploaded radar_out.png: minimap, geometry, dots all readable). The initial
# generous 320x320@(0,0) box included the HUD surround (dark border + empty
# strip below the map). Tightened by grid-reading radar_src_grid.png: the minimap
# disc runs ~(10,10) to ~(270,270) in the crop's local coords. Because that crop
# started at full-frame (0,0), local == source here, so the source rectangle is
# left=10 top=10 width=260 height=260. SQUARE minimap -> RADAR_OUT_HW stays
# square (128x128), so the resize carries no aspect distortion (same principle as
# D-012 for the FPV).
#
# STILL TO DO before this is baked into the recorder/format: run one more
#     python -m src.capture --radar-calibrate --grid
# and confirm radar_src.png is tight on the minimap with no dead border (these
# bounds were eyeballed off a 20px grid, so good to ~±10px). Only then wire it
# into the recorder + DATA_FORMAT.md v3 — the rectangle is baked into every file
# at CAPTURE time, so it must be right before recording resumes.
RADAR_SRC_LEFT = 10     # px, within the 1920x1080 frame  (measured; confirm once)
RADAR_SRC_TOP = 10      # px                               (measured; confirm once)
RADAR_SRC_WIDTH = 260   # px  (10..270)                    (measured; confirm once)
RADAR_SRC_HEIGHT = 260  # px  (10..270)                    (measured; confirm once)

# Size the radar crop is resized to for storage + the model (H, W). Square,
# matching the square source region, so no aspect distortion. 128x128 is the
# working choice for "enough to read position without bloating storage."
RADAR_OUT_HW = (128, 128)  # (height, width)
