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
# ── MEASURED (2026-08) FOR THE CENTERED RADAR — D-038 ──
# The radar is now recorded CENTERED (cl_radar_centered 1 + cl_radar_always_centered 1),
# zoomed to a local player-relative window rather than the whole-map overview. In this
# mode the self marker is fixed at the CENTRE of the disc, so self-position is known a
# priori and the #7 probe no longer has to identify which marker is ours. This changed
# the on-screen radar object, so the crop was re-measured (D-038), superseding the
# overview-radar rectangle the old D-024 values were measured against.
#
# HOW THESE NUMBERS WERE FOUND: on a centered full-res grab (radar_on_full.png), the
# minimap disc was fit as a circle (Hough): centre (188,192), radius 156, i.e. the disc
# spans x[32..344] y[36..348] (~312 px across). The old L10 T10 W260 H260 crop covered
# only the upper-left of this disc and clipped the right side + bottom — exactly where the
# map and the self marker sit. The rectangle below is that fitted disc plus a ~4px margin
# per side, kept SQUARE so the resize to RADAR_OUT_HW carries no aspect distortion (the
# D-012/D-024 principle). Verified by cropping: the whole disc sits inside with even
# margin on all four edges; a little scene shows in the square's corners (unavoidable when
# bounding a circle with a square — a circular mask could zero it later if wanted).
#
# CONFIDENCE / CAVEATS: measured from ONE grab via a Hough fit that was a touch generous,
# so good to ~±4px; the margin absorbs that, so it will not clip. These bounds are
# MACHINE- AND HUD-SPECIFIC: they depend on cl_hud_radar_scale and the centered-radar
# cvars above. If any of those change, re-measure with `python -m src.capture
# --radar-calibrate`. The rectangle is baked into every file at CAPTURE time (stamped into
# each file's `geom`), so it must be right before recording resumes — the pre-D-038 dataset
# was recorded with the old rectangle AND the non-centered radar and must be re-recorded.
RADAR_SRC_LEFT = 28     # px, within the 1920x1080 frame  (centered radar, D-038)
RADAR_SRC_TOP = 32      # px                               (centered radar, D-038)
RADAR_SRC_WIDTH = 320   # px  (x 28..348)                  (centered radar, D-038)
RADAR_SRC_HEIGHT = 320  # px  (y 32..352)                  (centered radar, D-038)

# Size the radar crop is resized to for storage + the model (H, W). Square,
# matching the square source region, so no aspect distortion. 128x128 is the
# working choice for "enough to read position without bloating storage."
RADAR_OUT_HW = (128, 128)  # (height, width)

# ── CIRCULAR RADAR MASK (D-039) ───────────────────────────────────────────
# The square radar crop necessarily includes GAME SCENE in its corners (the tan
# walls/floor visible outside the round minimap). Those corner pixels CHANGE as
# the player moves, so a model could learn from them as a spurious signal. This
# mask forces the corners to black so only the round minimap remains, killing the
# corners as a variable input. Applied at CAPTURE time in grab_with_radar(), so
# the stored `radar` array already has black corners. NOT a schema bump (D-039):
# the array stays 128x128x3 BGR, only pixel VALUES change — same rule as D-038.
#
# GEOMETRY: a filled circle in the RADAR_OUT_HW (128x128) output space. Centre and
# radius were MEASURED from the centered-radar dump (radar_out_upscaled.png):
# the disc sits at ~(67,67) with radius ~66 in 128-space. Radius was set to 63 from
# that screenshot, then CORRECTED to 61 after checking a REAL stored radar_out.png:
# at r=63 the CS2 radar's own yellow border RING (plus a sliver of scene just
# outside it) survived along the bottom-right arc; r=61 cuts just inside that rim,
# removing it while leaving all minimap content (walls, marker, FOV cone, labels)
# intact. Tightening below 61 removes nothing further (the residual warm pixels are
# the in-map yellow labels) and only risks clipping map. The centre is slightly off
# (64,64) because the HUD disc itself is not perfectly centred in the crop.
#
# CONFIDENCE: centre/radius were first measured from a screenshot preview, then
# CONFIRMED and corrected (63 -> 61) against a REAL stored radar_out.png from
# --radar-calibrate. Still MACHINE/HUD-specific: they depend on cl_hud_radar_scale
# and the centered-radar cvars. Because the mask is baked into every recorded file
# permanently, RE-CONFIRM against a fresh grab if the HUD scale changes: run
# `python -m src.capture --radar-calibrate` and check radar_out.png has black
# corners/edges and no minimap clipped. If off, adjust the three constants below
# and re-check — they are the single place the mask geometry lives.
RADAR_MASK_ENABLED = True   # baked into stored radar at capture (D-039)
RADAR_MASK_CENTER = (67, 67)  # (x, y) of disc centre in RADAR_OUT_HW space
RADAR_MASK_RADIUS = 61        # px; pixels farther than this from centre -> black


def _build_radar_mask():
    """Precompute the (H,W) boolean radar mask once (True = keep, False = black).

    Built at import from RADAR_MASK_CENTER/RADIUS in RADAR_OUT_HW space so
    grab_with_radar() applies a cheap boolean index per frame rather than
    recomputing the circle. Returns None when RADAR_MASK_ENABLED is False, so the
    capture path can skip masking entirely (and older-style unmasked behaviour is
    a one-flag change).
    """
    if not RADAR_MASK_ENABLED:
        return None
    import numpy as _np
    h, w = RADAR_OUT_HW
    cx, cy = RADAR_MASK_CENTER
    yy, xx = _np.ogrid[:h, :w]
    keep = (xx - cx) ** 2 + (yy - cy) ** 2 <= RADAR_MASK_RADIUS ** 2
    return keep


# Precomputed once; (H,W) bool with True inside the disc, or None if disabled.
RADAR_MASK = _build_radar_mask()
