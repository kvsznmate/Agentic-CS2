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


def _gather_Xy(dataset, key_cols):
    """Build probe features + targets from a dataset.

    X : (N, PROBE_SIZE*PROBE_SIZE) float32 in [0,1] — the stored radar, reduced to
        small grayscale (D-025). A linear probe on these features is deliberately
        the WEAKEST reasonable model: if even this finds signal, the radar carries
        usable movement information; if it can't, that's meaningful (though not
        proof a CNN couldn't — noted in the verdict).
    y : (N, len(key_cols)) float32 0/1 — held-state of each probed key.
    """
    n = len(dataset)
    Xs, ys = [], []
    for start in range(0, n, 4096):
        gi = list(range(start, min(start + 4096, n)))
        Xb, Yb = dataset.get_batch(gi)                    # Xb: (b, 128,128,3) radar
        Xs.append(_downsample_gray(Xb))                   # (b, PROBE_SIZE^2)
        ys.append(Yb[:, key_cols])
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    # Guard: features should be finite in [0,1]. A NaN/Inf here would silently
    # poison the fit; catch it rather than propagate garbage.
    if not np.isfinite(X).all():
        raise ValueError("radar features contain non-finite values — bad data.")
    return X, y


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


def probe(holdout_frac=0.4, l2=5.0):
    """Run the WASD-from-radar linear probe with a group-aware split + verdict.

    Steps:
      * Split sessions TRAIN/TEST by whole session (D-021 style) — never by frame.
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
    train_paths, test_paths = dl.split_sessions(holdout_frac=holdout_frac)
    n_sessions = len(train_paths) + len(test_paths)

    print(f"Radar signal probe — stored radar {rw}x{rh}, probe features "
          f"{PROBE_SIZE}x{PROBE_SIZE} grayscale (D-025).")
    print(f"Sessions: {len(train_paths)} train / {len(test_paths)} test "
          f"(whole-session split, D-021).")
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
        print(f"Frames: {n_train} train / {n_test} test. Building probe features "
              f"({PROBE_SIZE}x{PROBE_SIZE} = {PROBE_SIZE*PROBE_SIZE} dims)...")
        Xtr, ytr = _gather_Xy(train_ds, key_cols)
        Xte, yte = _gather_Xy(test_ds, key_cols)
    except ValueError as e:
        # crop="radar" on v1/v2 sessions (no radar array), or non-finite features.
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

    # ── Verdict, with the honest-gate provisional flag ──
    provisional = (n_total < MIN_PROBE_FRAMES) or (n_sessions < MIN_PROBE_SESSIONS)
    print()
    print("=" * 70)
    if provisional:
        print("VERDICT: PROVISIONAL — data volume below the floor for a committed")
        print(f"  gate result ({n_total} frames / {n_sessions} sessions; need "
              f"≥{MIN_PROBE_FRAMES} frames and ≥{MIN_PROBE_SESSIONS} sessions).")
        print("  Treat the numbers above as a SMOKE TEST only. Do NOT record this")
        print("  as the M2 gate outcome in DECISIONS.md. Re-run after recording")
        print("  the full dataset (D-020) for a verdict that can gate M5.")
        print("  What it's worth now: if the lift is already clearly positive, the")
        print("  approach is promising; if it's ~0 or negative on this little data,")
        print("  that's a yellow flag to investigate (radar illegible? capture box")
        print("  off?), not yet a KILL.")
    else:
        # Real-volume verdict. The exact PASS threshold is a bar to COMMIT in this
        # issue now that we can measure it (the plan: measure first, then set it).
        verdict = "SIGNAL PRESENT" if mean_lift > 0.03 else "NO CLEAR SIGNAL"
        print(f"VERDICT (committed-volume): {verdict}")
        print(f"  Mean lift {mean_lift*100:+.1f}pp over baseline on "
              f"{n_total} frames / {n_sessions} sessions.")
        print("  ACTION: record the committed M2 threshold + this verdict in")
        print("  DECISIONS.md now (a linear probe is a floor — a CNN could do")
        print("  better, so 'no clear signal' here warrants one CNN attempt before")
        print("  raising the KILL flag per #7, but a clear positive is a real GO).")
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
    p.add_argument("--holdout-frac", type=float, default=0.4,
                   help="session holdout fraction for the probe's group split")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.dump_radar is not None:
        dump_radar(n=args.dump_radar)
    elif args.probe:
        probe(holdout_frac=args.holdout_frac)
    else:
        print("Choose: --dump-radar N (legibility) or --probe (signal). "
              "See `python -m src.radar_probe -h`.")


if __name__ == "__main__":
    main()
