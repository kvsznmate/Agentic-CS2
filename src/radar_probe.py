"""radar_probe.py — Issue #7 (M2 GATE): navigation-signal check on the radar.

The M2 go/no-go. Two questions, in order:

  1. GEOMETRY / LEGIBILITY — is self-position legible in the stored radar image?
     (Resolved at capture time now: the radar is a dedicated high-res 128x128
     crop, D-024, its rectangle measured and baked into capture_config via
     `python -m src.capture --radar-calibrate`. This tool just dumps the stored
     crops so you can re-confirm legibility on real sessions.)
  2. SIGNAL (the actual gate) — can movement (WASD) be predicted from the radar
     pixels above chance? By eye + a quick linear probe. (A real go/no-go; its
     verdict is only trustworthy at real data volume.)

WHY THIS IS A FROM-SCRATCH PROBLEM FOR US (not a reference reproduction):
The reference study never recovered position from the radar image. It read the
player's map coordinates straight from game memory (localpos1/2/3 via RAM
offsets; its "map coverage" is a histogram of those memory-read coords). All of
that is dead on Source 2 (D-002). So we have no ground-truth position and must
ask whether the RADAR PIXELS carry usable navigation signal at all. That is
exactly what this gate tests, and why it can genuinely fail.

WHAT CHANGED WITH D-024 (two-resolution capture):
Earlier, "the radar" was a sub-rectangle cropped from the 150x270 FPV frame, and
this tool took a `--radar-rect` to hunt for that box at load time. That box was
too low-res to read position (the #7 finding), so D-024 made the radar a SEPARATE
high-res array captured before the FPV downscale, resized to 128x128, stored per
frame. Consequences for this tool: it no longer owns any rectangle (geometry is
baked at capture, confirmed with `--radar-calibrate`); it consumes the stored
radar via the loader's `crop="radar"`; and the old `--radar-rect` override and
box-tightening guidance are gone.

NUMERICS + PERFORMANCE NOTE (both fixed 2026-08, learned the hard way):
The probe's logistic solver is hand-rolled numpy (no sklearn assumption). Two
separate problems surfaced on the first real runs, on the same file:

  (1) DIVERGENCE. The first run overflowed `exp` and produced anti-correlated
      accuracies (a key held 7% of the time "predicted" at 22% — a −70pp lift),
      the fingerprint of a DIVERGED optimizer, NOT of a signal-free radar. Fixed:
      sign-stable sigmoid (no exp overflow), lr 0.5→0.05, iters→400, gradient-norm
      clipping, a real variance floor, stronger L2, AND a divergence guard that
      makes the probe REFUSE to report per-key numbers if the fit didn't converge.

  (2) COST. Even fixed, fitting on the FULL 128x128x3 = 49,152-dim flattened radar
      over thousands of frames for 400 iterations, in float64, is a multi-TFLOP,
      multi-GB grind on this machine's non-SIMD numpy — it ran >20 min with no
      output and had to be killed. Fixed by D-025: the probe DOWNSAMPLES the radar
      to 32x32 GRAYSCALE (~1,024 dims, ~48x smaller) before fitting, and prints
      the loss every few iterations so "working" is visibly distinct from "stuck".

Together these mean the probe now either gives honest numbers quickly or tells you
plainly it couldn't fit — the same "a tool that can mislead must fail loudly"
lesson as the inspector episode, applied to the estimator. The 32x32 downsample is
a real methodological choice (it changes what the probe measures), recorded as
D-025: reading WHICH BLOB IS WHERE on a minimap does not need full resolution; if a
linear probe can find position→movement signal at all it survives 32x32, and if it
can't show even at full res, 49k noisy dims weren't going to rescue it.

──────────────────────────────────────────────────────────────────────────────
HONEST-GATE RULE (read before trusting any PASS this prints)
──────────────────────────────────────────────────────────────────────────────
The signal probe emits a verdict, but a behavioural-cloning probe on a small,
highly-correlated sample can read "signal" from very little — and M5 (the entire
hand-authored navigation panel) is gated on this being REAL. So this tool marks
its verdict **PROVISIONAL** whenever it runs on fewer than MIN_PROBE_FRAMES
frames (or fewer than MIN_PROBE_SESSIONS sessions). A PROVISIONAL pass is a
smoke-test, NOT the committed M2 gate result, and must not be logged to
DECISIONS.md as the gate outcome. Re-run at real data volume (the ~20 GB / 5–7 h
of D-020) to get a COMMITTED verdict worth recording. This is not caution for its
own sake — a false PASS here is precisely the expensive mistake the derisking
order exists to prevent.

Group-aware evaluation: the probe splits TRAIN vs TEST by whole session (never by
frame), mirroring D-021, because consecutive frames are near-duplicates and a
per-frame split would leak them across the split and inflate the score — the very
way a thin-data probe fabricates a false PASS.

Depends on: data_loader.py (the stored radar via crop="radar", the session split,
action columns). Requires v3 sessions (the radar array only exists in v3, D-024);
on v1/v2-only data the loader will refuse crop="radar" with a clear error.

Usage:
  python -m src.radar_probe --dump-radar 12   # save stored radar crops to eyeball
  python -m src.radar_probe --probe            # run the WASD linear probe + verdict
"""

import argparse
import os

import numpy as np

from src import data_loader as dl


_DUMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "capture_debug")

# ── Honest-gate thresholds ────────────────────────────────────────────────────
# Below either of these, the probe verdict is PROVISIONAL, not committed. These
# are floors for the verdict to *mean* something, not targets — the real dataset
# (D-020) is far above them. Set deliberately, not tuned: a few thousand frames
# across at least a couple of independent sessions is the minimum for a
# group-split probe to be more than noise. The true M2 bar is set in this issue
# only after the first COMMITTED run, per the plan ("measure first, then set the
# bar").
MIN_PROBE_FRAMES = 20000     # ~22 min at 15 FPS
MIN_PROBE_SESSIONS = 3       # need enough independent sessions for a group split

# ── Lift-interpretation thresholds (revised 2026-08 after the balanced-split run) ─
# Two separate things a converged fit's held-out lift can mean, by MAGNITUDE:
#
#   * STRONGLY negative (below BROKEN_LIFT_EPS): the probe generalises much WORSE
#     than the majority-class baseline. A signal-free radar sits at ~0 lift, so a
#     deeply negative lift means the probe learned TRAIN priors that actively
#     mislead on TEST — a broken measurement (train/test distribution mismatch),
#     not a radar finding. Report NO RESULT and refuse a verdict. This is the
#     session_20260818 first run (-12pp on a lopsided name-hash split).
#
#   * MILDLY negative to ~zero (between BROKEN_LIFT_EPS and the SIGNAL band): this
#     is the FLAT/WEAK-signal regime — what a linear probe looks like when there
#     is little or no LINEARLY-decodable position→movement signal. It is a real
#     (if provisional) finding, NOT a broken run. Report it as weak/no-signal and
#     route to the legibility check + more data + a CNN attempt (the linear probe
#     is only a floor, per D-023/#7). This is the balanced-split run (-2.1pp on a
#     fair split): near zero, dragged just under 0 by w/d — not "broken".
#
# The earlier single NEG_LIFT_EPS=-0.02 was too tight and mislabelled the -2.1pp
# fair-split result as a "broken split", telling the user to re-run the balanced
# split they had JUST run. Splitting the threshold in two fixes that: only a
# deeply negative lift is treated as broken; a near-zero one is treated as a
# genuine flat result. A false PASS is still the expensive mistake to avoid, so
# the SIGNAL bar stays strict and unchanged.
BROKEN_LIFT_EPS = -0.05      # mean lift below -5pp => distribution mismatch, NO RESULT
SIGNAL_LIFT_EPS = 0.03       # mean lift above +3pp => signal present (committed-volume)

# ── Probe feature resolution (D-025) ──────────────────────────────────────────
# The probe downsamples the stored radar to this size, GRAYSCALE, before fitting.
# 32x32x1 = 1024 dims vs the stored 128x128x3 = 49,152 — ~48x smaller, so the
# hand-rolled numpy logreg fits in seconds instead of grinding for 20+ min. This
# is a deliberate methodological choice, not just speed: a linear probe asks only
# "is there ANY linearly-decodable position→movement signal," and blob position on
# a minimap survives a 32x32 grayscale reduction. If signal can't show even at full
# res, 49k noisy dims wouldn't rescue it. Raise PROBE_SIZE if a future check wants
# more detail; it trades speed for resolution.
PROBE_SIZE = 32              # probe operates on PROBE_SIZE x PROBE_SIZE grayscale


def _radar_hw():
    """(H, W) of the stored radar image, from the loader (D-024)."""
    return dl.RADAR_H, dl.RADAR_W


def _balanced_split(paths, target_test_frac=0.4):
    """PROBE-ONLY split: largest sessions to TRAIN, the rest to TEST. Leak-free.

    WHY THIS EXISTS (added 2026-08): the committed split is the loader's
    deterministic name-hash (D-021) — correct for training (stable, leak-free) but
    it can land pathologically for a *probe* at a tiny session count. In the
    session_20260818 episode it put the single 24,867-frame session entirely in
    TEST and trained on ~7k frames of unrelated small sessions, so the probe
    measured cross-distribution transfer, not radar signal (mean lift -12pp).

    This helper builds a fairer PROBE split for that situation. It fills TRAIN
    largest-session-first up to (1 - target_test_frac) of all frames, and puts the
    remaining (smaller) sessions in TEST. So the probe LEARNS from the richest
    data and is scored on genuinely held-out OTHER sessions. It stays leak-free
    (whole sessions, never per-frame) and deterministic (sorted by frame count
    then name).

    IMPORTANT HONEST NOTE: when one session dominates the corpus (as now: the big
    session is ~63% of all frames), NO whole-session split can be both leak-free
    AND distribution-matched — that session must sit entirely on one side. Putting
    it in TRAIN is the least-bad diagnostic: it asks 'does a probe fit on our
    biggest session generalise to the others?', which is the right question. It is
    NOT the committed split and NOT used by any trainer — only by
    `--probe --balanced-split`, to tell 'no signal' apart from 'bad split'. The
    real fix for a trustworthy verdict is more sessions of comparable size
    (D-020 volume), not a cleverer split.
    """
    sized = sorted(
        ((p, dl.SessionDataset._session_length(p)) for p in paths),
        key=lambda pn: (-pn[1], dl.session_name(pn[0])),
    )
    total = sum(n for _, n in sized)
    if total == 0:
        return [], []
    train, test = [], []
    train_n = 0
    train_goal = total * (1.0 - target_test_frac)
    for p, n in sized:
        # Fill TRAIN largest-first up to its frame goal; the rest go to TEST.
        if train_n < train_goal:
            train.append(p); train_n += n
        else:
            test.append(p)
    # Guarantee both sides are non-empty when there are >=2 sessions (e.g. if the
    # single largest session already met the train goal and swept everything).
    if len(sized) >= 2 and (not train or not test):
        if not test:
            test.append(train.pop())      # train is sorted large->small: move smallest
        else:
            train.append(test.pop(0))     # test is sorted large->small: move its largest
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — LEGIBILITY: dump the stored radar crops so a human can confirm them
# ─────────────────────────────────────────────────────────────────────────────

def dump_radar(n=12, upscale=3):
    """Save N stored radar crops (evenly spaced across usable sessions) as PNGs.

    The legibility half of the gate: open the PNGs and confirm you can tell
    roughly WHERE the player is on the map from the radar. This is a prerequisite
    to the signal probe meaning anything. Unlike the pre-D-024 version, this does
    NOT hunt for a crop rectangle — the radar geometry is baked at capture
    (confirmed with `python -m src.capture --radar-calibrate`); here we just look
    at what was actually stored (128x128, D-024) on real sessions.

    Requires v3 sessions (they carry the radar array). On v1/v2-only data the
    loader refuses crop="radar" and this reports that cleanly.

    Writes both the raw crop and an upscaled copy (nearest-neighbour) so the crop
    is comfortably viewable.
    """
    try:
        import cv2
    except ImportError:
        print("cv2 not available — cannot dump PNGs.")
        return

    rh, rw = _radar_hw()
    sessions = dl.discover_sessions(report=True)
    if not sessions:
        print(f"No usable sessions in {dl._REC_DIR}. Record with "
              f"`python -m src.recorder --record` (v3 includes the radar).")
        return

    ds = dl.SessionDataset(sessions, crop="radar")  # stored radar, not an FPV crop
    total = len(ds)
    if total == 0:
        print("Usable sessions found but they contain 0 frames.")
        return

    os.makedirs(_DUMP_DIR, exist_ok=True)
    idx = np.linspace(0, total - 1, min(n, total)).astype(int)
    try:
        X, _ = ds.get_batch(list(idx))  # (n, rh, rw, 3) stored radar crops
    except ValueError as e:
        # crop="radar" on v1/v2 sessions (no radar array).
        print(f"Cannot dump radar: {e}")
        print("Record a v3 session first: `python -m src.recorder --record`.")
        return

    print(f"Stored radar image: {rw}x{rh} (D-024, baked at capture).")
    print(f"Dumping {len(idx)} radar crops to {_DUMP_DIR}:")
    for k, j in enumerate(idx):
        raw = X[k]
        big = cv2.resize(raw, (rw * upscale, rh * upscale),
                         interpolation=cv2.INTER_NEAREST)
        praw = os.path.join(_DUMP_DIR, f"radar_{int(j):06d}_raw.png")
        pbig = os.path.join(_DUMP_DIR, f"radar_{int(j):06d}_{upscale}x.png")
        cv2.imwrite(praw, raw)
        cv2.imwrite(pbig, big)
        print(f"  frame {int(j):>6}: {pbig}")
    print()
    print("OPEN THE *x.png FILES. One question to answer by eye:")
    print("  Can you tell roughly WHERE on the map the player is from the radar?")
    print("  If yes, the signal probe is worth running. If the radar is somehow")
    print("  unreadable even at this resolution, that's a finding for this gate")
    print("  (check the capture rectangle with `--radar-calibrate`) — note it")
    print("  before running the probe.")
    print("(The radar rectangle is already baked in capture_config and recorded")
    print(" in DATA_FORMAT.md — nothing to tighten here.)")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — SIGNAL: linear probe, WASD-from-radar, group-split, honest verdict
# ─────────────────────────────────────────────────────────────────────────────

def _key_index(train_ds, key):
    """Column index of a key in the assembled action vector (via action_layout)."""
    layout = dl.action_layout(train_ds._arrays(0))
    return layout.index(key)


def _downsample_gray(batch, size=PROBE_SIZE):
    """Reduce a (B, H, W, 3) BGR uint8 batch to (B, size*size) float32 in [0,1].

    Grayscale (luminance-ish mean over channels) then area-resize to size x size.
    This is the D-025 probe feature: small + grayscale so the linear fit is fast,
    while preserving the blob-position information a linear probe can use. Uses cv2
    if available (INTER_AREA, correct for downscaling); falls back to a plain
    numpy stride reduction if cv2 is missing, so the probe still runs.
    """
    B = batch.shape[0]
    gray = batch.astype(np.float32).mean(axis=3)          # (B, H, W)
    try:
        import cv2
        out = np.empty((B, size, size), dtype=np.float32)
        for i in range(B):
            out[i] = cv2.resize(gray[i], (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        H, W = gray.shape[1], gray.shape[2]
        ys = (np.linspace(0, H, size + 1)).astype(int)
        xs = (np.linspace(0, W, size + 1)).astype(int)
        out = np.empty((B, size, size), dtype=np.float32)
        for i in range(B):
            for a in range(size):
                for bcol in range(size):
                    out[i, a, bcol] = gray[i, ys[a]:ys[a+1], xs[bcol]:xs[bcol+1]].mean()
    return (out.reshape(B, -1) / 255.0).astype(np.float32)


# ── Gameplay-vs-not filter (added 2026-08 after the blank-radar finding) ─────
# Some recorded frames have NO radar on screen — settings menu open mid-match,
# spectating another player, halftime. The radar crop is then near-uniform
# (blank/dark), and the WASD keys in those frames are decoupled from any map
# position (menu nav, or nothing). Those frames are STRUCTURED CONTAMINATION for
# the probe: 'blank image -> mostly no keys' is a pattern the linear model can
# learn that has nothing to do with radar navigation signal, and it can drag a
# real result toward zero. This filter identifies present-radar (gameplay) frames
# so the probe can fit on them only.
#
# HOW: a present minimap has spatial structure (map geometry, the dot, colour), so
# its grayscale PIXEL VARIANCE is high; a blank/menu radar is near-uniform, so its
# variance is near zero. The two populations are bimodal and separate cleanly. We
# do NOT hardcode the cut: `_gameplay_threshold` finds the valley between the two
# modes from the actual variance histogram (a simple min-density split around the
# gap), so it adapts to this machine's radar. `--gameplay-report` prints the
# distribution + chosen cut so it can be eyeballed before being trusted — the same
# 'measure first, then set the bar' + 'a tool that can mislead must narrate
# itself' discipline the rest of this file follows.
#
# SCOPE: this is a PROBE-ONLY diagnostic (opt-in via --filter-blank), to tell
# 'no radar signal' apart from 'blank frames masking it'. The blank-frame problem
# is bigger than the probe — it also contaminates the detector (#10) and aim (#11)
# — so the REAL fix belongs upstream in the loader/preprocessing as a dataset-
# hygiene pass, tracked separately. This helper is the cheap here-and-now check.

def _radar_variance(batch):
    """Per-frame grayscale pixel variance of a (B,H,W,3) uint8 radar batch -> (B,).

    High for a present minimap (spatial structure), near zero for a blank/menu
    radar. Computed on the full-res stored radar (not the downsample) so the
    blank/present distinction is as clean as possible.
    """
    gray = batch.astype(np.float32).mean(axis=3)          # (B,H,W)
    return gray.reshape(gray.shape[0], -1).var(axis=1)     # (B,)


def _gameplay_threshold(variances):
    """Find the variance cut separating blank (low) from present (high) radar.

    Uses OTSU'S METHOD on the log-variance histogram: the threshold that maximises
    between-class variance (equivalently, minimises within-class variance) of the
    two resulting groups. This is the standard, robust way to split a bimodal
    distribution into two classes, and it does NOT depend on peak-finding tricks
    (an earlier valley-between-tallest-peaks heuristic put the cut on the wrong
    side when one mode held most of the mass — caught in test). We work in log10
    space because radar variances span orders of magnitude (near-0 blanks vs
    thousands for a present minimap).

    Also reports a `bimodal` flag from the separation quality: if the two classes
    Otsu produces are not well separated (their means are close relative to their
    spread), the distribution probably isn't really two modes, and the caller
    should eyeball --gameplay-report before trusting the cut.

    Returns (threshold_in_variance_units, diagnostic_dict).
    """
    v = np.asarray(variances, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, {"reason": "no data", "bimodal": False}
    v = np.maximum(v, 0.0)                     # variance is >=0; guard bad inputs
    lv = np.log10(v + 1.0)                     # +1 admits exact-zero-variance blanks
    lo, hi = lv.min(), lv.max()
    if hi - lo < 1e-6:
        return 0.0, {"reason": "single value (no blank/present separation)",
                     "bimodal": False}

    nb = 256
    hist, edges = np.histogram(lv, bins=nb, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    w = hist.astype(np.float64) / total        # class-probability per bin

    # Otsu: for each possible cut between bins, compute between-class variance
    # sigma_b^2 = w0*w1*(mu0-mu1)^2, and pick the cut that maximises it.
    w0 = np.cumsum(w)                           # weight of class 0 (<= cut)
    w1 = 1.0 - w0                               # weight of class 1 (> cut)
    # Cumulative means.
    csum = np.cumsum(w * centers)
    mu_total = csum[-1]
    # Guard against divide-by-zero at the ends.
    valid = (w0 > 1e-12) & (w1 > 1e-12)
    mu0 = np.where(valid, csum / np.where(w0 > 0, w0, 1), 0.0)
    mu1 = np.where(valid, (mu_total - csum) / np.where(w1 > 0, w1, 1), 0.0)
    sigma_b = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, 0.0)
    cut_bin = int(np.argmax(sigma_b))
    thr_log = centers[cut_bin]
    thr = float(10 ** thr_log - 1.0)

    # Separation quality: is this REALLY two modes, or did Otsu just cut one mode
    # in half? Otsu always returns a cut, so we must judge separation ourselves.
    # Two genuinely different populations here (blank var~1 vs present var~thousands)
    # sit orders of magnitude apart: >~2 decades in log10. Carving a single mode in
    # half only separates its halves by a fraction of a decade. So require a LARGE
    # absolute gap between the class means in log10 space, not just a multiple of
    # the (small) within-class spread — the latter falsely flagged a unimodal set
    # as bimodal in test. MIN_MODE_GAP_DECADES is deliberately strict.
    MIN_MODE_GAP_DECADES = 1.5
    left = lv[lv <= thr_log]
    right = lv[lv > thr_log]
    if left.size and right.size:
        gap = right.mean() - left.mean()       # in log10 units (decades)
        bimodal = gap >= MIN_MODE_GAP_DECADES
        frac_low = left.size / lv.size
    else:
        bimodal = False
        frac_low = (left.size / lv.size) if lv.size else 0.0
    return thr, {"reason": "otsu split", "bimodal": bool(bimodal),
                 "frac_below": float(frac_low),
                 "gap_decades": float(gap) if (left.size and right.size) else 0.0}


def gameplay_report(holdout_frac=0.4, balanced_split=False):
    """Print the radar-variance distribution + the chosen blank/present cut.

    A look-before-you-trust step for the blank-frame filter: shows how many frames
    would be dropped as 'no radar' and where the threshold lands, WITHOUT fitting
    anything. Run this before --probe --filter-blank to sanity-check the cut.
    """
    all_paths = dl.discover_sessions(report=True)
    if not all_paths:
        print(f"No usable sessions in {dl._REC_DIR}.")
        return
    ds = dl.SessionDataset(all_paths, crop="radar")
    n = len(ds)
    if n == 0:
        print("Usable sessions found but 0 frames.")
        return
    print(f"Computing radar variance over {n} frames across "
          f"{ds.n_sessions} session(s)...")
    vs = []
    try:
        for start in range(0, n, 4096):
            gi = list(range(start, min(start + 4096, n)))
            Xb, _ = ds.get_batch(gi)
            vs.append(_radar_variance(Xb))
    except ValueError as e:
        print(f"Cannot read radar: {e}")
        return
    v = np.concatenate(vs)
    thr, diag = _gameplay_threshold(v)
    n_blank = int((v <= thr).sum())
    n_play = int((v > thr).sum())
    print(f"\nRadar variance: min {v.min():.1f}  median {np.median(v):.1f}  "
          f"max {v.max():.1f}")
    # Coarse text histogram on log scale.
    lv = np.log10(v + 1.0)
    hist, edges = np.histogram(lv, bins=12)
    print("\nlog10(variance+1) distribution:")
    for i in range(len(hist)):
        bar = "#" * int(40 * hist[i] / max(hist.max(), 1))
        print(f"  {edges[i]:5.2f}–{edges[i+1]:5.2f} | {hist[i]:6d} {bar}")
    print(f"\nChosen blank/present cut: variance = {thr:.2f}  ({diag['reason']}).")
    print(f"  Would mark {n_blank} frames ({100*n_blank/len(v):.1f}%) as BLANK/no-radar")
    print(f"  and keep {n_play} frames ({100*n_play/len(v):.1f}%) as gameplay.")
    if not diag.get("bimodal", False):
        print("  WARNING: distribution is not clearly bimodal — the cut may be")
        print("  unreliable. Eyeball the histogram above before trusting")
        print("  --probe --filter-blank; the blank frames may be few, or the")
        print("  variance measure may not separate them here.")
    else:
        print("  Distribution is bimodal (blank vs present separate cleanly).")


def _gather_Xy(dataset, key_cols, blank_threshold=None):
    """Build probe features + targets from a dataset.

    X : (N, PROBE_SIZE*PROBE_SIZE) float32 in [0,1] — the stored radar, reduced to
        small grayscale (D-025). A linear probe on these features is deliberately
        the WEAKEST reasonable model: if even this finds signal, the radar carries
        usable movement information; if it can't, that's meaningful (though not
        proof a CNN couldn't — noted in the verdict).
    y : (N, len(key_cols)) float32 0/1 — held-state of each probed key.

    blank_threshold (optional): if given, frames whose full-res radar variance is
    <= this value are dropped as blank/no-radar (settings/spectate/halftime) before
    features are built (the Option-B gameplay filter). Returns only present-radar
    frames. Also returns the count kept/dropped for reporting.
    """
    n = len(dataset)
    Xs, ys = [], []
    n_kept = n_dropped = 0
    for start in range(0, n, 4096):
        gi = list(range(start, min(start + 4096, n)))
        Xb, Yb = dataset.get_batch(gi)                    # Xb: (b, 128,128,3) radar
        if blank_threshold is not None:
            var = _radar_variance(Xb)                     # (b,)
            keep = var > blank_threshold
            n_dropped += int((~keep).sum())
            n_kept += int(keep.sum())
            Xb = Xb[keep]
            Yb = Yb[keep]
            if Xb.shape[0] == 0:
                continue
        Xs.append(_downsample_gray(Xb))                   # (b', PROBE_SIZE^2)
        ys.append(Yb[:, key_cols])
    if not Xs:
        raise ValueError("no frames left after blank-radar filtering — the "
                         "threshold removed everything; check --gameplay-report.")
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    # Guard: features should be finite in [0,1]. A NaN/Inf here would silently
    # poison the fit; catch it rather than propagate garbage.
    if not np.isfinite(X).all():
        raise ValueError("radar features contain non-finite values — bad data.")
    return X, y, (n_kept, n_dropped)


def _sigmoid(z):
    """Numerically stable logistic sigmoid.

    exp(-z) overflows for very negative z (the first-run bug). Branch on sign so
    we only ever exponentiate a non-positive argument: for z>=0 use 1/(1+e^-z),
    for z<0 use e^z/(1+e^z). Mathematically identical, never overflows.
    """
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _fit_logreg(X, y, l2=5.0, iters=400, lr=0.05, clip=5.0, verbose=True):
    """Tiny batch logistic-regression (one weight vector per target), numpy-only.

    Standardise features, gradient-descent a logistic model with L2. Enough to
    answer "better than chance?", which is all the gate asks. Returns
    (W, b, mu, sd, info) where info reports convergence so the caller can refuse
    to trust a diverged fit.

    Numerics (fixes for the diverged first run — see module docstring):
      * sign-stable sigmoid (no exp overflow);
      * lr=0.05 (was 0.5, which diverged) + iters=400;
      * per-step gradient-norm clipping at `clip` so one bad step can't blow up;
      * variance floor sd>=1e-2 AFTER /255 scaling;
      * stronger default L2 (5.0) for an underdetermined regime.
    Performance: operates on the D-025 small grayscale features (~1k dims), and
    prints the loss every ~50 iters when verbose so a slow fit is legible, never a
    silent black box (the >20-min-no-output failure).

    info = {"converged": bool, "loss0": float, "loss1": float, "reason": str}.
    converged is False if the loss is non-finite at any point or ends higher than
    it began (divergence) — the probe treats that as "no trustworthy fit".
    """
    X = X.astype(np.float32)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd = np.maximum(sd, 1e-2)          # real variance floor (X is already /255)
    Xs = ((X - mu) / sd).astype(np.float32)
    N, D = Xs.shape
    K = y.shape[1]
    W = np.zeros((D, K), np.float32)
    b = np.zeros((1, K), np.float32)

    def _loss():
        z = Xs @ W + b
        p = np.clip(_sigmoid(z), 1e-7, 1 - 1e-7)
        ce = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        reg = 0.5 * l2 * float((W * W).sum()) / N
        return float(ce + reg)

    loss0 = _loss()
    if verbose:
        print(f"    fit: {N} rows x {D} dims, {iters} iters "
              f"(loss start {loss0:.4f})")
    converged = True
    reason = "ok"
    for it in range(iters):
        z = Xs @ W + b
        p = _sigmoid(z)
        g = (p - y).astype(np.float32)                    # (N,K)
        gW = Xs.T @ g / N + l2 * W / N
        gb = g.mean(0, keepdims=True)
        gnorm = float(np.sqrt((gW * gW).sum() + (gb * gb).sum()))
        if not np.isfinite(gnorm):
            converged = False
            reason = "non-finite gradient"
            break
        if gnorm > clip:
            gW *= clip / gnorm
            gb *= clip / gnorm
        W -= lr * gW
        b -= lr * gb
        if verbose and (it + 1) % 50 == 0:
            print(f"    iter {it+1:>4}/{iters}  loss {_loss():.4f}")

    loss1 = _loss()
    if not np.isfinite(loss1):
        converged = False
        reason = "non-finite loss"
    elif loss1 > loss0 + 1e-6:
        converged = False
        reason = f"loss increased ({loss0:.4f} -> {loss1:.4f})"

    info = {"converged": converged, "loss0": loss0, "loss1": loss1, "reason": reason}
    return W, b, mu, sd, info


def _predict(X, W, b, mu, sd):
    z = ((X.astype(np.float32) - mu) / sd) @ W + b
    return _sigmoid(z)


def probe(holdout_frac=0.4, l2=5.0, balanced_split=False, filter_blank=False):
    """Run the WASD-from-radar linear probe with a group-aware split + verdict.

    Steps:
      * Split sessions TRAIN/TEST by whole session (D-021 style) — never by frame.
        Default: the loader's committed name-hash split. With balanced_split=True,
        use the PROBE-ONLY frame-balanced split (_balanced_split) instead, which
        keeps the split leak-free but gives both sides comparable frame volume —
        use it to tell 'no radar signal' apart from 'the hash gave a lopsided
        split at this tiny session count' (the session_20260818 episode).
      * Optionally (filter_blank=True) drop blank/no-radar frames — settings menu,
        spectating, halftime — before fitting, since they are structured
        contamination that can drag the result toward zero. The blank/present
        variance cut is derived from the data (see _gameplay_threshold); check it
        first with --gameplay-report.
      * Build probe features for each side: the stored radar reduced to small
        grayscale (D-025), via crop="radar".
      * Fit a linear logistic probe per movement key (w/a/s/d) on TRAIN.
      * If the fit did not converge, REFUSE to report per-key numbers (they would
        mislead — the diverged-first-run lesson). Otherwise score on TEST:
        per-key accuracy vs the majority-class baseline (the honest "above chance"
        reference — predicting the more common state).
      * Emit a verdict, marked PROVISIONAL if data volume is below the floor.
    """
    rh, rw = _radar_hw()
    if balanced_split:
        all_paths = dl.discover_sessions()
        train_paths, test_paths = _balanced_split(all_paths, target_test_frac=holdout_frac)
        split_desc = "frame-balanced, PROBE-ONLY (not the committed D-021 split)"
    else:
        train_paths, test_paths = dl.split_sessions(holdout_frac=holdout_frac)
        split_desc = "committed name-hash, D-021"
    n_sessions = len(train_paths) + len(test_paths)

    print(f"Radar signal probe — stored radar {rw}x{rh}, probe features "
          f"{PROBE_SIZE}x{PROBE_SIZE} grayscale (D-025).")
    print(f"Split: {split_desc}.")
    print(f"Sessions: {len(train_paths)} train / {len(test_paths)} test "
          f"(whole-session, leak-free).")
    if not train_paths or not test_paths:
        print("\nNeed at least one session on EACH side of the split. With very few "
              "sessions the hash can put them all on one side — record more, or "
              "pass manual_holdout= in code. Cannot run the probe.")
        return

    train_ds = dl.SessionDataset(train_paths, crop="radar")
    test_ds = dl.SessionDataset(test_paths, crop="radar")
    n_train, n_test = len(train_ds), len(test_ds)
    n_total = n_train + n_test

    movement_keys = ["w", "a", "s", "d"]
    try:
        key_cols = [_key_index(train_ds, k) for k in movement_keys]

        # If filtering blank frames, derive ONE variance cut across BOTH splits so
        # train and test are filtered on the same criterion. Computed here (not
        # per-split) so the threshold doesn't drift between the two sides.
        blank_thr = None
        if filter_blank:
            print("Computing radar-variance cut for blank-frame filtering...")
            allv = []
            for ds in (train_ds, test_ds):
                m = len(ds)
                for start in range(0, m, 4096):
                    gi = list(range(start, min(start + 4096, m)))
                    Xb, _ = ds.get_batch(gi)
                    allv.append(_radar_variance(Xb))
            allv = np.concatenate(allv)
            blank_thr, diag = _gameplay_threshold(allv)
            n_blank = int((allv <= blank_thr).sum())
            print(f"  blank/present cut: variance={blank_thr:.2f} ({diag['reason']}); "
                  f"{n_blank}/{len(allv)} frames ({100*n_blank/len(allv):.1f}%) "
                  f"are blank/no-radar and will be DROPPED.")
            if not diag.get("bimodal", False):
                print("  WARNING: variance distribution not clearly bimodal — run "
                      "`--gameplay-report` to eyeball the cut before trusting this.")

        print(f"Frames: {n_train} train / {n_test} test. Building probe features "
              f"({PROBE_SIZE}x{PROBE_SIZE} = {PROBE_SIZE*PROBE_SIZE} dims)"
              f"{' (blank frames filtered)' if filter_blank else ''}...")
        Xtr, ytr, (kept_tr, drop_tr) = _gather_Xy(train_ds, key_cols, blank_thr)
        Xte, yte, (kept_te, drop_te) = _gather_Xy(test_ds, key_cols, blank_thr)
        if filter_blank:
            print(f"  after filtering: {Xtr.shape[0]} train "
                  f"(dropped {drop_tr}), {Xte.shape[0]} test (dropped {drop_te}).")
            # Update the effective counts so the volume-floor check reflects the
            # frames actually USED, not the raw session totals.
            n_train, n_test = Xtr.shape[0], Xte.shape[0]
            n_total = n_train + n_test
    except ValueError as e:
        # crop="radar" on v1/v2 sessions (no radar array), non-finite features,
        # or the blank filter removed everything.
        print(f"\nCannot run the probe: {e}")
        print("The probe needs v3 sessions (radar array). Record with "
              "`python -m src.recorder --record`.")
        return

    W, b, mu, sd, info = _fit_logreg(Xtr, ytr, l2=l2)

    # DIVERGENCE GUARD — refuse to report numbers from a fit that didn't converge.
    # A diverged fit produces confident-looking but meaningless accuracies (the
    # first-run failure). Better to say "the probe didn't fit" than to print a
    # verdict that could be mistaken for a radar finding.
    if not info["converged"]:
        print(f"\n  Training loss: {info['loss0']:.4f} -> {info['loss1']:.4f}")
        print("=" * 70)
        print("VERDICT: NO RESULT — the linear probe did NOT converge "
              f"({info['reason']}).")
        print("  This is a SOLVER outcome, not a radar finding: it says nothing")
        print("  about whether the radar carries signal. Likely causes: too few")
        print("  frames for the fit, or degenerate features. Re-run with more")
        print("  data; if it persists at volume, lower the learning rate / raise")
        print("  L2 in _fit_logreg, or switch the probe to a small CNN. Do NOT")
        print("  record anything in DECISIONS.md from a non-converged probe.")
        print("=" * 70)
        return None, True

    pte = _predict(Xte, W, b, mu, sd)
    pred = (pte >= 0.5).astype(np.float32)

    print(f"\nTraining loss: {info['loss0']:.4f} -> {info['loss1']:.4f} (converged).")
    print("Per-key results on held-out sessions "
          "(probe acc vs majority-class baseline):")
    lifts = []
    for i, k in enumerate(movement_keys):
        y = yte[:, i]
        base_rate = max(y.mean(), 1 - y.mean())      # majority-class accuracy
        acc = (pred[:, i] == y).mean()
        lift = acc - base_rate
        lifts.append(lift)
        held_frac = ytr[:, i].mean()                 # how often this key is held
        print(f"  {k}: acc {acc*100:5.1f}%  baseline {base_rate*100:5.1f}%  "
              f"lift {lift*100:+5.1f}pp   (held {held_frac*100:.0f}% of train)")

    mean_lift = float(np.mean(lifts))
    print(f"\nMean accuracy lift over baseline across w/a/s/d: "
          f"{mean_lift*100:+.1f}pp")

    # ── BROKEN-MEASUREMENT GUARD (strongly negative lift only) ──
    # A converged fit that generalises MUCH worse than the majority-class baseline
    # (below BROKEN_LIFT_EPS) is a broken measurement, not a 'no signal' finding:
    # a signal-free radar sits at ~0 lift, so a deeply negative lift means the
    # probe learned TRAIN priors that actively mislead on TEST (train/test
    # distribution mismatch). This is the session_20260818 first run (-12pp on a
    # lopsided name-hash split). Refuse a verdict. NOTE: a MILDLY negative lift
    # (e.g. -2pp) is NOT caught here — that is the flat/weak-signal regime handled
    # below, not a broken run. The advice here depends on whether this run was
    # already balanced, so we don't tell the user to re-run a split they just ran.
    if mean_lift < BROKEN_LIFT_EPS:
        print()
        print("=" * 70)
        print(f"VERDICT: NO RESULT — mean held-out lift is {mean_lift*100:+.1f}pp, "
              f"below {BROKEN_LIFT_EPS*100:+.0f}pp.")
        print("  Strongly WORSE than chance is the fingerprint of a TRAIN/TEST")
        print("  DISTRIBUTION MISMATCH, not a signal-free radar (which sits at ~0")
        print("  lift). The fit converged, so this is a SPLIT problem, not a")
        print("  solver one.")
        if not balanced_split:
            print("  LIKELY CAUSE: the committed name-hash split landed lopsided at")
            print("  this small session count. DO THIS:")
            print("    1. Re-run with the frame-balanced probe split:")
            print("         python -m src.radar_probe --probe --balanced-split")
            print("    2. Confirm the radar is legible if you haven't:")
            print("         python -m src.radar_probe --dump-radar 20")
        else:
            print("  This run was ALREADY balanced, so a lopsided split is not the")
            print("  cause. A deeply negative lift on a fair split points to bad")
            print("  data on one side. DO THIS:")
            print("    1. Confirm the radar is legible (do this first):")
            print("         python -m src.radar_probe --dump-radar 20")
            print("    2. If legible, suspect a corrupt/misaligned session or a")
            print("       key-column mismatch; inspect the sessions individually.")
        print("  Do NOT record anything in DECISIONS.md from this run.")
        print("=" * 70)
        return mean_lift, True

    # ── VERDICT (lift is not broken; classify it) ──
    # provisional = below the data-volume floor for a COMMITTED gate result.
    provisional = (n_total < MIN_PROBE_FRAMES) or (n_sessions < MIN_PROBE_SESSIONS)
    # A near-zero-or-mildly-negative lift is the flat/weak-signal regime; a lift
    # above SIGNAL_LIFT_EPS is a real positive. This split is the same at any
    # volume; only whether it can be COMMITTED depends on `provisional`.
    has_signal = mean_lift > SIGNAL_LIFT_EPS
    print()
    print("=" * 70)
    if provisional:
        band = "SIGNAL (provisional)" if has_signal else "FLAT / WEAK (provisional)"
        print(f"VERDICT: {band} — data volume below the floor for a committed gate")
        print(f"  result ({n_total} frames / {n_sessions} sessions; need "
              f"≥{MIN_PROBE_FRAMES} frames and ≥{MIN_PROBE_SESSIONS} sessions).")
        print(f"  Mean lift {mean_lift*100:+.1f}pp over baseline. Treat this as a")
        print("  SMOKE TEST only — do NOT record it as the M2 outcome in")
        print("  DECISIONS.md. Re-run at full data volume (D-020) for a committed")
        print("  verdict that can gate M5.")
        if has_signal:
            print("  What it's worth now: the lift is already clearly positive — the")
            print("  approach is promising; earn the committed GO at volume.")
        else:
            print("  What it's worth now: near zero on this little data is a YELLOW")
            print("  FLAG, not a KILL. In order: (1) confirm the radar is legible")
            print("  (`--dump-radar 20`); (2) record more sessions of comparable")
            print("  size toward D-020 and re-run; (3) if a fair, full-volume linear")
            print("  probe still reads flat, try ONE small CNN before any KILL — a")
            print("  linear probe is only a floor (D-023/#7).")
    else:
        if has_signal:
            print(f"VERDICT (committed-volume): SIGNAL PRESENT")
            print(f"  Mean lift {mean_lift*100:+.1f}pp over baseline on "
                  f"{n_total} frames / {n_sessions} sessions.")
            print("  ACTION: this is a real GO. Record the committed M2 threshold +")
            print("  this verdict in DECISIONS.md, and proceed to M5.")
        else:
            print(f"VERDICT (committed-volume): NO CLEAR SIGNAL")
            print(f"  Mean lift {mean_lift*100:+.1f}pp over baseline on "
                  f"{n_total} frames / {n_sessions} sessions — at or below the")
            print(f"  +{SIGNAL_LIFT_EPS*100:.0f}pp signal bar.")
            print("  ACTION: a linear probe is only a FLOOR, so this does NOT by")
            print("  itself KILL the navigation path. Try ONE small CNN probe first")
            print("  (D-023/#7). Only if THAT is also flat do you raise the KILL")
            print("  flag — and either way, record the outcome in DECISIONS.md.")
    print("=" * 70)
    return mean_lift, provisional


def _build_parser():
    p = argparse.ArgumentParser(
        description="Issue #7 (M2 gate): stored-radar legibility + WASD signal probe.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dump-radar", type=int, metavar="N",
                   help="save N stored radar crops as PNGs to eyeball legibility")
    g.add_argument("--probe", action="store_true",
                   help="run the WASD-from-radar linear probe and print a verdict")
    g.add_argument("--gameplay-report", action="store_true",
                   help="show the radar-variance distribution and where the "
                        "blank/present (no-radar vs gameplay) cut would fall, "
                        "without fitting. Run before --probe --filter-blank.")
    p.add_argument("--holdout-frac", type=float, default=0.4,
                   help="session holdout fraction for the probe's group split")
    p.add_argument("--balanced-split", action="store_true",
                   help="PROBE-ONLY diagnostic: split by frame VOLUME instead of "
                        "the committed name-hash, so both sides get comparable "
                        "data. Use to tell 'no radar signal' apart from a lopsided "
                        "hash split at a small session count. Still leak-free "
                        "(whole sessions); NOT the committed split, NOT for training.")
    p.add_argument("--filter-blank", action="store_true",
                   help="drop blank/no-radar frames (settings menu, spectating, "
                        "halftime) before fitting the probe. The variance cut is "
                        "derived from the data; check it first with "
                        "--gameplay-report. Diagnostic: tells 'no signal' apart "
                        "from 'blank frames masking signal'.")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.dump_radar is not None:
        dump_radar(n=args.dump_radar)
    elif args.gameplay_report:
        gameplay_report(holdout_frac=args.holdout_frac,
                        balanced_split=args.balanced_split)
    elif args.probe:
        probe(holdout_frac=args.holdout_frac, balanced_split=args.balanced_split,
              filter_blank=args.filter_blank)
    else:
        print("Choose: --dump-radar N (legibility), --gameplay-report (blank-frame "
              "check), or --probe (signal). See `python -m src.radar_probe -h`.")


if __name__ == "__main__":
    main()
