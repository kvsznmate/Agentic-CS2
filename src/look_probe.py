"""look_probe.py — diagnose what a trained movement+look model actually learned.

WHY THIS EXISTS (the missing instrument):
Training and eval only report AGGREGATES — mean per-key lift, mean dx/dy MAE. An
aggregate hides the failure modes we actually hit at play time. In particular:
  * The keys head can score fine on average while only ever firing W (the common
    key) and suppressing A/S/D — the imbalance trap. A per-key confusion view
    shows this directly.
  * The look head can "beat" a weak baseline while really just predicting a small
    CONSTANT near the mean every frame — which at play time (deltas applied every
    loop tick) becomes an endless slow drift, not turning. A single MAE number
    cannot tell "learned to turn" from "outputs the mean". The distribution and
    the prediction-vs-truth correlation can.

So this probe loads a trained model and, on a SAMPLE of TRAIN windows (this asks
"what did it fit", not "does it generalise" — that's the trainer's held-out job),
prints:
  KEYS, per key: predicted-positive rate vs true-positive rate, and precision/
    recall. If pred-rate ~ 0 for A/S/D while true-rate isn't, the model suppresses
    turns (explains "only goes straight").
  LOOK, per axis (dx/dy): std of PREDICTIONS vs std of TRUTH (mean-collapse shows
    as pred-std << truth-std), Pearson correlation of predicted vs actual (near 0
    = not tracking, just a constant), the model's mean output (a nonzero constant
    here = the drift bias), and a few concrete rows (true vs predicted) at the
    hardest real turns so you can eyeball it.

This is a DIAGNOSTIC, not a gate — like radar_probe.py it just measures. It
changes nothing on disk and trains nothing. It reuses the SAME loaders, the SAME
output-name resolution, and the SAME sidecar standardization inversion as the
trainer/eval and play_movement, so what it reports matches what those do (D-036).

Usage:
  python -m src.look_probe                       # newest full look-model, 2000 windows
  python -m src.look_probe --crop radar          # probe the radar model instead
  python -m src.look_probe --model PATH          # a specific .keras
  python -m src.look_probe --sample 4000         # more windows (slower)
  python -m src.look_probe --split holdout       # probe held-out windows instead of train
"""

import argparse
import glob
import os

import numpy as np

from src import sequence_loader as sl
from src import data_loader as dl
from src.model_lstm import MOVEMENT_KEYS, _MODEL_DIR


def _resolve_model(model_arg, crop):
    """Newest trained model for this crop, or an explicit path (matches play_movement)."""
    if model_arg:
        if not os.path.exists(model_arg):
            raise SystemExit(f"Model not found: {model_arg}")
        return model_arg
    pattern = os.path.join(_MODEL_DIR, f"wasd_lstm_{crop}_T*.keras")
    matches = glob.glob(pattern)
    if not matches:
        raise SystemExit(
            f"No trained model for crop='{crop}' in {_MODEL_DIR} "
            f"(looked for {os.path.basename(pattern)}). Train one first.")
    return max(matches, key=os.path.getmtime)


def _load_look_stats(model_path):
    """Load the dx/dy standardization sidecar (mean/std), or None (matches D-036)."""
    stats_path = (model_path[:-len(".keras")] if model_path.endswith(".keras")
                  else model_path) + ".look_stats.npz"
    if not os.path.isfile(stats_path):
        return None
    with np.load(stats_path, allow_pickle=False) as d:
        return {"mean": d["mean"].astype(np.float32),
                "std": d["std"].astype(np.float32)}


def _output_indices(model):
    """Resolve (keys_idx, look_idx) by output NAME; look_idx None if single-head."""
    names = getattr(model, "output_names", None)
    if isinstance(names, (list, tuple)) and len(names) > 1:
        keys_idx = names.index("move_keys") if "move_keys" in names else 0
        look_idx = names.index("look") if "look" in names else None
        return keys_idx, look_idx
    return 0, None


def _collect_predictions(model, seq, sample, eval_batch, keys_idx, look_idx,
                         look_stats):
    """Run the model over a sample of windows; return truth + predictions.

    Returns dict with:
      yk_true (N,K), yk_prob (N,K)  — key truth 0/1 and predicted probabilities
      yl_true (N,2), yl_pred (N,2)  — raw dx/dy truth and predicted (un-standardized)
    Uses get_batch_with_look and micro-batched model() calls (bounded VRAM, as in
    the trainer's eval).
    """
    n = len(seq)
    if n == 0:
        raise SystemExit("No windows in this split — nothing to probe.")
    # Deterministic sample of window positions (a fixed stride, not random, so the
    # probe is reproducible run-to-run).
    take = min(sample, n)
    idx = np.linspace(0, n - 1, take).astype(int)
    idx = np.unique(idx)

    yk_true, yk_prob, yl_true, yl_pred = [], [], [], []
    for s in range(0, len(idx), eval_batch):
        wp = idx[s:s + eval_batch].tolist()
        X, Yk, Yl_raw = seq.get_batch_with_look(wp)
        out = model(X, training=False)
        if isinstance(out, (list, tuple)):
            pk = np.asarray(out[keys_idx])
            pl_std = np.asarray(out[look_idx]) if look_idx is not None else None
        else:
            pk = np.asarray(out)
            pl_std = None
        yk_true.append(Yk)
        yk_prob.append(pk)
        yl_true.append(Yl_raw)
        if pl_std is not None:
            if look_stats is not None:
                pl_raw = pl_std * look_stats["std"] + look_stats["mean"]
            else:
                pl_raw = pl_std   # standardized units (no sidecar) — flagged later
            yl_pred.append(pl_raw)

    res = {
        "yk_true": np.concatenate(yk_true, axis=0),
        "yk_prob": np.concatenate(yk_prob, axis=0),
        "yl_true": np.concatenate(yl_true, axis=0),
        "yl_pred": np.concatenate(yl_pred, axis=0) if yl_pred else None,
    }
    return res


def _report_keys(res, threshold):
    yk_true = res["yk_true"]
    yk_prob = res["yk_prob"]
    yk_pred = (yk_prob >= threshold).astype(np.float32)
    n = yk_true.shape[0]
    print(f"\nKEYS — prediction vs truth over {n} windows "
          f"(play threshold {threshold:.2f}):")
    print(f"  {'key':<6} {'true+rate':>9} {'pred+rate':>9} {'precision':>9} "
          f"{'recall':>7} {'meanP':>7}")
    for i, k in enumerate(MOVEMENT_KEYS):
        t = yk_true[:, i]
        p = yk_pred[:, i]
        true_rate = t.mean()
        pred_rate = p.mean()
        tp = float((p * t).sum())
        prec = tp / max(p.sum(), 1e-9)
        rec = tp / max(t.sum(), 1e-9)
        mean_prob = yk_prob[:, i].mean()
        print(f"  {k:<6} {true_rate*100:8.1f}% {pred_rate*100:8.1f}% "
              f"{prec*100:8.1f}% {rec*100:6.1f}% {mean_prob:6.2f}")
    print("  Reading it: pred+rate ~ 0% while true+rate is not => the model")
    print("  SUPPRESSES that key (never presses it). Low recall on A/D = it")
    print("  rarely turns even when you did. meanP is the average sigmoid output;")
    print("  if it sits below the threshold for A/D, lowering --threshold at play")
    print("  time will surface more turns (if there is any signal to surface).")


def _pearson(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")   # one side constant -> correlation undefined
    return float(np.corrcoef(a, b)[0, 1])


def _report_look(res, look_stats):
    yl_true = res["yl_true"]
    yl_pred = res["yl_pred"]
    if yl_pred is None:
        print("\nLOOK — model has no look head; nothing to probe.")
        return
    unit = "device units" if look_stats is not None else "STANDARDIZED units (no sidecar!)"
    n = yl_true.shape[0]
    print(f"\nLOOK — prediction vs truth over {n} windows ({unit}):")
    print(f"  {'axis':<4} {'truth_std':>9} {'pred_std':>9} {'corr':>7} "
          f"{'pred_mean':>10} {'true_mean':>10}")
    collapse_flag = False
    for i, ax in enumerate(("dx", "dy")):
        t = yl_true[:, i]
        p = yl_pred[:, i]
        corr = _pearson(t, p)
        ratio = p.std() / max(t.std(), 1e-9)
        if ratio < 0.25:
            collapse_flag = True
        print(f"  {ax:<4} {t.std():8.1f} {p.std():8.1f} "
              f"{corr:6.2f} {p.mean():+9.2f} {t.mean():+9.2f}")
    print("  Reading it:")
    print("   * pred_std MUCH smaller than truth_std  => MEAN-COLLAPSE: the model")
    print("     outputs a nearly-constant value, not turning. (This is the classic")
    print("     cause of 'slow constant drift' at play time.)")
    print("   * corr near 0 => predictions don't track your actual turns at all;")
    print("     corr clearly positive => it learned the DIRECTION of turning.")
    print("   * pred_mean far from 0 => a constant bias applied every frame; at")
    print("     play time that becomes an endless slow pan in that direction.")
    if collapse_flag:
        print("  VERDICT: at least one axis looks MEAN-COLLAPSED (pred_std < 25% of")
        print("  truth_std). The look head has not learned to turn on this data —")
        print("  see the concrete rows below and the recommendations.")

    # Concrete rows: the hardest real turns (largest |true dx|), so you can eyeball
    # what the model does exactly when the player turned hard.
    order = np.argsort(-np.abs(yl_true[:, 0]))[:8]
    print("\n  Hardest real turns (by |true dx|) — did the model follow?")
    print(f"    {'true dx':>8} {'pred dx':>8}   {'true dy':>8} {'pred dy':>8}")
    for j in order:
        print(f"    {yl_true[j,0]:8.1f} {yl_pred[j,0]:8.1f}   "
              f"{yl_true[j,1]:8.1f} {yl_pred[j,1]:8.1f}")


def probe(model_arg=None, crop="full", sample=2000, split="train", eval_batch=8,
          threshold=0.4):
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"TensorFlow import failed ({e!r}). conda activate agentic-cs2.")

    model_path = _resolve_model(model_arg, crop)
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    keys_idx, look_idx = _output_indices(model)
    look_stats = _load_look_stats(model_path) if look_idx is not None else None
    if look_idx is not None and look_stats is None:
        print("  WARNING: look head present but NO look-stats sidecar found; look")
        print("  numbers will be in STANDARDIZED units and pred_mean/std are not")
        print("  directly comparable to device-unit truth. Retrain to emit it.")

    seq_len = int(model.input_shape[1])
    train_seq, hold_seq = sl.build_sequence_datasets(
        crop=crop, seq_len=seq_len, target_keys=MOVEMENT_KEYS)
    seq = train_seq if split == "train" else hold_seq
    print(f"Probing {split} split: {len(seq)} windows available "
          f"(sampling up to {sample}).")

    res = _collect_predictions(model, seq, sample, eval_batch, keys_idx, look_idx,
                               look_stats)
    _report_keys(res, threshold)
    _report_look(res, look_stats)
    print("\n(Probe only — nothing was trained or written. This measures what the")
    print(" model FIT on the sampled split; generalisation is the trainer's")
    print(" held-out job. D-036.)")


def _build_parser():
    p = argparse.ArgumentParser(
        description="Diagnose what a trained movement+look model actually predicts.")
    p.add_argument("--model", default=None, help="path to a .keras (default: newest for crop)")
    p.add_argument("--crop", default="full", help="full/centre/radar (must match the model)")
    p.add_argument("--sample", type=int, default=2000, help="windows to probe (default 2000)")
    p.add_argument("--split", choices=("train", "holdout"), default="train",
                   help="probe train (what it fit) or holdout (default train)")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="key probability -> press cutoff for the keys report (default 0.4)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    probe(model_arg=args.model, crop=args.crop, sample=args.sample,
          split=args.split, threshold=args.threshold)


if __name__ == "__main__":
    main()
