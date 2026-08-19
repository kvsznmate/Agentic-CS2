"""model_lstm.py — WASD-from-FPV recurrent baseline (first model of the project).

The baseline behavioural-cloning model: given a short SEQUENCE of FPV frames,
predict which movement keys (W/A/S/D) the player is holding at the LAST frame
(many-to-one, T=8 ~ 0.5 s at 15 FPS). This is the honest floor for the project's
central question — "does a model learn to act from OUR pixels and OUR recorded
actions?" — using the labels that come FREE from self-recording (D-003: the keys
you pressed are the labels; nothing is hand-annotated).

WHY THIS MODEL, THIS SCOPE (agreed in conversation, tied to the plan):
  * MOVEMENT ONLY (WASD). Tightest baseline; actions are free; and it doubles as
    the honest version of the #7 radar gate — run the SAME model on crop="radar"
    and compare its WASD lift to the FPV model's (radar->movement is exactly what
    #7 asks). The trainer takes --crop, so one script does both jobs.
  * RECURRENT (CNN encoder + LSTM), per the chosen architecture and the reference
    study's recurrent design. Movement is temporal (momentum, counter-strafing) —
    a single frame can't see it. many-to-one keeps the head simple.
  * WASD = 4 INDEPENDENT SIGMOIDS, not a 4-way softmax. You can hold W+D at once
    (forward strafe), so this is MULTI-LABEL. A softmax would forbid diagonals —
    most of CS movement. Loss is per-key binary cross-entropy.

THE HONEST METRIC (why accuracy alone is a trap):
WASD is heavily imbalanced — W is held most of the time. A model that predicts
"always W, never A/S/D" scores high on raw accuracy while learning nothing. So
the trainer reports, per key, the model's held-out accuracy AND the majority-class
baseline (predict the more common state), and the LIFT between them. Lift is the
real signal; per-key output makes "it just always-holds-W" visible. This mirrors
radar_probe.py's baseline-lift reporting, deliberately.

DATA-VOLUME HONESTY (D-023 applied to modelling):
At the current ~15k frames / 15 sessions, this pipeline can prove it LEARNS, but a
good number could be overfitting on highly-correlated frames — an LSTM memorises
small correlated data fast. So a strong result here is PROVISIONAL, not a committed
"this is how well BC works." The deliverable now is a WORKING, HONEST training +
eval pipeline; the committed benchmark comes at real volume (D-020). The trainer
prints this caveat with the numbers rather than letting a shiny lift stand alone.
It also always reports the TRAIN lift next to the HELD-OUT lift, because a large
train/held-out gap is the visible fingerprint of memorisation.

STACK: tf.keras on TensorFlow 2.10 (environment.yml / D-011 — last native-Windows
GPU TF, and the paper's framework). numpy<2 per the pins. Nothing here needs a
newer TF API. Sequences come from sequence_loader.py, which enforces the
whole-session split (D-021) and optional keep-mask (D-026) and guarantees each
window is T temporally-consecutive frames.

Usage:
  python -m src.model_lstm --train                     # FPV baseline (crop=full)
  python -m src.model_lstm --train --crop centre       # centred FPV crop
  python -m src.model_lstm --train --crop radar        # the #7 radar->WASD comparison
  python -m src.model_lstm --train --use-keep-mask      # exclude blank frames (D-026)
  python -m src.model_lstm --train --epochs 15 --batch 32 --seq-len 8
  python -m src.model_lstm --summary                   # build + print model, no training
"""

import argparse
import os
import time

import numpy as np

from src import sequence_loader as sl
from src import data_loader as dl


# Where trained baselines are saved (gitignored under data/).
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")

# Movement keys this baseline predicts, in output-column order. Beyond WASD this
# includes shift (walk) and space (jump), each an INDEPENDENT sigmoid (multi-label
# — you hold shift+W, or tap space while moving). All are already logged by the
# recorder (LOGGED_KEYS) and present in every session's `keys` array, so NO
# re-recording is needed; they just become training targets here.
#   * shift: rare + tactical — expect a WEAK signal, like A/S/D.
#   * space: JUMP is a near-INSTANTANEOUS event (held ~1 frame at 15 FPS), so it's
#     the rarest target by far and a poor fit for "predict held-state at the last
#     frame." Expect its lift to be ~0 and the model may never fire it above the
#     play threshold. Included for completeness / future data, not because it's
#     likely to learn now. This is an honest limitation, not a bug.
MOVEMENT_KEYS = ("w", "a", "s", "d", "shift", "space")

# Volume floors below which a result is PROVISIONAL (mirrors radar_probe D-023).
MIN_TRAIN_WINDOWS = 15000
MIN_SESSIONS = 3


def _import_tf():
    """Import TensorFlow lazily with a clear message if the env isn't set up.

    Kept out of module import so `-h` and syntax checks don't require TF. The env
    (environment.yml, D-011) pins tensorflow==2.10.1; if import fails we say how to
    fix it rather than dumping a raw ImportError.
    """
    try:
        import tensorflow as tf  # noqa: F401
        return tf
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"TensorFlow import failed ({e!r}).\n"
            f"Activate the project env and verify the GPU stack:\n"
            f"  conda activate agentic-cs2\n"
            f"  python -m src.smoke_test\n"
            f"environment.yml pins tensorflow==2.10.1 (D-011).")


def build_model(seq_len, frame_hw, n_outputs, channels=3):
    """CNN-encoder + LSTM, many-to-one, n_outputs independent sigmoids.

    Input : (T, H, W, C) uint8-range frames (cast+scaled inside the model so the
            data pipeline can stay uint8 BGR end-to-end).
    Output: (n_outputs,) in [0,1] — per-key hold probability at the last frame.

    Design notes:
      * A TimeDistributed CNN applies the SAME small encoder to each of the T
        frames, producing a per-frame feature vector; the LSTM consumes the T
        features and returns only its final state (return_sequences=False) — the
        many-to-one shape. Weight-sharing across time is what makes this a
        sequence model rather than T separate CNNs.
      * The CNN is deliberately SMALL (three conv blocks). The FPV is low-res
        (150x270) and the dataset is small; a heavy encoder would overfit before
        it generalised. This is a baseline, not a final architecture.
      * Scaling (/255) is a Rescaling layer INSIDE the model, so inference gets
        the same preprocessing as training automatically and the loader keeps
        serving uint8 (smaller batches, one source of truth for the frame dtype).
      * BGR is fed as-is. The model doesn't care about channel order as long as it
        is CONSISTENT between train and inference, and the loader is BGR
        everywhere (D-012). No BGR->RGB conversion is introduced (it would be a
        silent inconsistency risk for zero benefit here).
    """
    tf = _import_tf()
    from tensorflow.keras import layers, models

    H, W = frame_hw
    inp = layers.Input(shape=(seq_len, H, W, channels), name="frame_seq")
    x = layers.Rescaling(1.0 / 255.0)(inp)

    def conv_block(filters):
        # Factory so the encoder is easy to read/tune; each block halves H,W.
        return [
            layers.Conv2D(filters, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
        ]

    # Per-frame CNN encoder, shared across time via TimeDistributed.
    encoder_layers = (
        conv_block(16) + conv_block(32) + conv_block(64) +
        [layers.GlobalAveragePooling2D()]
    )
    for lyr in encoder_layers:
        x = layers.TimeDistributed(lyr)(x)        # -> (T, feature_dim)

    x = layers.LSTM(128, return_sequences=False)(x)   # many-to-one final state
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_outputs, activation="sigmoid", name="move_keys")(x)

    model = models.Model(inp, out, name="move_fpv_lstm_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",     # per-key independent (multi-label)
        metrics=["binary_accuracy"],
    )
    return model


def _batched_arrays(seq_ds, batch_size, shuffle, seed=None):
    """Yield (X, Y) numpy batches from a SequenceDataset (Keras-friendly generator).

    Kept as a plain Python generator (not tf.data) to stay simple and framework-
    version-robust on TF 2.10; at this data scale the Python overhead is
    negligible next to the conv forward/backward pass.
    """
    yield from seq_ds.iter_batches(batch_size=batch_size, shuffle=shuffle, seed=seed)


def _evaluate_lift(model, seq_ds, batch_size, baseline_rates=None):
    """Per-key held-out accuracy, majority-class baseline, and lift.

    baseline_rates: dict key->held_fraction from the TRAIN set. The majority-class
    baseline accuracy for a key is max(rate, 1-rate) evaluated on THIS set's
    labels — the honest 'predict the more common state' reference. If not given,
    it is computed from this dataset itself.

    Returns a dict per key: {"acc", "baseline", "lift", "held"} plus "n".
    """
    n = len(seq_ds)
    if n == 0:
        return {"n": 0}
    keys = seq_ds.target_keys
    n_correct = np.zeros(len(keys))
    n_held = np.zeros(len(keys))
    total = 0
    # Deterministic pass (no shuffle) so eval is stable.
    for X, Y in seq_ds.iter_batches(batch_size=batch_size, shuffle=False):
        p = model.predict(X, verbose=0)
        pred = (p >= 0.5).astype(np.float32)
        n_correct += (pred == Y).sum(axis=0)
        n_held += Y.sum(axis=0)
        total += Y.shape[0]
    acc = n_correct / max(total, 1)
    held = n_held / max(total, 1)
    out = {"n": total}
    for i, k in enumerate(keys):
        rate = baseline_rates[k] if baseline_rates else held[i]
        base = max(rate, 1.0 - rate)
        out[k] = {"acc": float(acc[i]), "baseline": float(base),
                  "lift": float(acc[i] - base), "held": float(held[i])}
    return out


def _print_lift_table(title, res, keys):
    print(f"\n{title} (n={res.get('n', 0)} windows):")
    if res.get("n", 0) == 0:
        print("  (no windows)")
        return
    print(f"  {'key':<4} {'acc':>7} {'baseline':>9} {'lift':>8}   held")
    lifts = []
    for k in keys:
        r = res[k]
        lifts.append(r["lift"])
        print(f"  {k:<4} {r['acc']*100:6.1f}% {r['baseline']*100:8.1f}% "
              f"{r['lift']*100:+7.1f}pp   {r['held']*100:4.0f}%")
    print(f"  mean lift over {'/'.join(keys)}: {np.mean(lifts)*100:+.1f}pp")
    return float(np.mean(lifts))


def train(crop="full", seq_len=8, epochs=10, batch_size=32,
          use_keep_mask=False, holdout_frac=dl.DEFAULT_HOLDOUT_FRAC,
          manual_holdout=None, save=True):
    """Train the WASD baseline and report honest per-key held-out lift.

    Steps: build leak-free sequence splits (D-021) with the chosen input feed;
    report window counts + the train-set key balance (the baseline reference);
    train; then evaluate per-key lift on held-out, printed next to the TRAIN lift
    so memorisation (large gap) is visible; and stamp the whole thing PROVISIONAL
    if below the volume floor.
    """
    tf = _import_tf()

    frame_hw = (dl.RADAR_H, dl.RADAR_W) if crop == "radar" else _fpv_hw(crop)
    print(f"Building sequence datasets: crop='{crop}', seq_len={seq_len}, "
          f"targets={MOVEMENT_KEYS}, keep_mask={use_keep_mask}.")
    try:
        train_seq, hold_seq = sl.build_sequence_datasets(
            crop=crop, seq_len=seq_len, target_keys=MOVEMENT_KEYS,
            holdout_frac=holdout_frac, manual_holdout=manual_holdout,
            use_keep_mask=use_keep_mask)
    except ValueError as e:
        # e.g. crop="radar" on v1/v2-only data (no radar array).
        raise SystemExit(
            f"Could not build sequences for crop='{crop}': {e}\n"
            f"(crop='radar' needs v3 sessions with the radar array — D-024.)")

    n_train, n_hold = len(train_seq), len(hold_seq)
    n_sessions = train_seq.n_sessions + hold_seq.n_sessions
    print(f"Windows: {n_train} train / {n_hold} held-out "
          f"(from {train_seq.n_sessions} + {hold_seq.n_sessions} sessions).")
    print(f"  (runs too short for T={seq_len} contribute nothing: "
          f"train {train_seq._stats['runs_too_short']}/{train_seq._stats['runs']} runs, "
          f"held-out {hold_seq._stats['runs_too_short']}/{hold_seq._stats['runs']} runs.)")
    if n_train == 0:
        raise SystemExit("No training windows. Record more, lower --seq-len, or "
                         "check the keep-mask isn't dropping everything.")
    if n_hold == 0:
        print("WARNING: held-out is EMPTY (the whole-session hash put every session "
              "in train at this count). Lift will be TRAIN-ONLY and cannot show "
              "generalisation. Record more sessions or pass manual_holdout=. "
              "Continuing so the pipeline runs, but do not read the result as a gate.")

    # Train-set key balance = the majority-class baseline reference for eval.
    train_balance = train_seq.target_balance()
    print("\nTrain-set key balance (held fraction at window's last frame):")
    for k in MOVEMENT_KEYS:
        print(f"  {k}: {train_balance[k]*100:5.1f}%  "
              f"(always-{'hold' if train_balance[k]>0.5 else 'release'} baseline "
              f"= {max(train_balance[k],1-train_balance[k])*100:.1f}%)")

    model = build_model(seq_len, frame_hw, n_outputs=len(MOVEMENT_KEYS))
    model.summary(print_fn=lambda s: print("  " + s))

    # Keras generators over our numpy batches. steps computed from window counts.
    steps = max(1, n_train // batch_size)
    print(f"\nTraining: {epochs} epochs x {steps} steps (batch {batch_size}) "
          f"on {frame_hw[0]}x{frame_hw[1]} frames...")
    t0 = time.perf_counter()
    for ep in range(epochs):
        ep_loss = ep_acc = 0.0
        seen = 0
        for X, Y in _batched_arrays(train_seq, batch_size, shuffle=True, seed=ep):
            metrics = model.train_on_batch(X, Y, return_dict=True)
            b = X.shape[0]
            ep_loss += metrics["loss"] * b
            ep_acc += metrics["binary_accuracy"] * b
            seen += b
        print(f"  epoch {ep+1:>2}/{epochs}  loss {ep_loss/max(seen,1):.4f}  "
              f"binary_acc {ep_acc/max(seen,1):.4f}  ({seen} windows)")
    train_secs = time.perf_counter() - t0
    print(f"Trained in {train_secs:.0f}s.")

    # ── Honest evaluation: per-key lift on held-out AND train ──
    keys = MOVEMENT_KEYS
    train_res = _evaluate_lift(model, train_seq, batch_size, baseline_rates=train_balance)
    hold_res = (_evaluate_lift(model, hold_seq, batch_size, baseline_rates=train_balance)
                if n_hold > 0 else {"n": 0})
    train_mean = _print_lift_table("TRAIN lift", train_res, keys)
    hold_mean = _print_lift_table("HELD-OUT lift", hold_res, keys)

    # ── Verdict framing (provisional below the volume floor) ──
    provisional = (n_train < MIN_TRAIN_WINDOWS) or (n_sessions < MIN_SESSIONS) or (n_hold == 0)
    print("\n" + "=" * 70)
    if hold_res.get("n", 0) == 0:
        print("RESULT: PIPELINE OK, NO GENERALISATION MEASURED — held-out was empty.")
        print("  The model trained and the eval path works, but with no held-out")
        print("  sessions there is no honest generalisation number. Record more")
        print("  sessions (or pass manual_holdout=) and re-run.")
    else:
        gap = (train_mean - hold_mean) if (train_mean is not None and hold_mean is not None) else 0.0
        tag = "PROVISIONAL" if provisional else "COMMITTED-VOLUME"
        print(f"RESULT ({tag}): held-out mean WASD lift {hold_mean*100:+.1f}pp "
              f"(train {train_mean*100:+.1f}pp, gap {gap*100:.1f}pp).")
        if provisional:
            print(f"  Below the volume floor ({n_train} windows / {n_sessions} "
                  f"sessions; need >={MIN_TRAIN_WINDOWS} windows and "
                  f">={MIN_SESSIONS} sessions). This is a SMOKE TEST that the")
            print("  pipeline learns, NOT a committed baseline. Do not record it as")
            print("  an M1/M2 result. Re-run at D-020 volume for a real number.")
        print("  Reading it: a positive held-out lift = the model learned movement")
        print("  from the FPV beyond always-guessing the common key. A large")
        print("  train-vs-held-out GAP = overfitting (expected at this data size).")
        if crop == "radar":
            print("  crop='radar': this is the #7 movement-from-radar comparison.")
            print("  Compare this held-out lift to the FPV run's to judge whether")
            print("  the radar carries movement signal a sequence model can use.")
    print("=" * 70)

    if save and hold_res.get("n", 0) >= 0:
        os.makedirs(_MODEL_DIR, exist_ok=True)
        # Stub includes the key count so a 5-key (with-shift) model does NOT
        # overwrite an existing 4-key WASD model file, and play_movement can tell
        # them apart. e.g. wasd_lstm_full_T8_k5.keras.
        stub = f"wasd_lstm_{crop}_T{seq_len}_k{len(MOVEMENT_KEYS)}"
        path = os.path.join(_MODEL_DIR, stub + ".keras")
        try:
            model.save(path)
            print(f"\nSaved model: {path}")
        except Exception as e:  # noqa: BLE001
            print(f"\n(Model save skipped: {e!r})")
    return model


def _fpv_hw(crop):
    """Served (H, W) for an FPV crop spec, matching data_loader's cropping."""
    if crop in (None, "full"):
        return (dl.FRAME_H, dl.FRAME_W)
    if crop == "centre":
        _t, _l, h, w = dl.CENTRE_CROP_DEFAULT
        return (h, w)
    if isinstance(crop, (tuple, list)) and len(crop) == 4:
        return (int(crop[2]), int(crop[3]))
    # "radar" is handled by the caller; anything else is unknown.
    raise SystemExit(f"Unknown crop {crop!r} for FPV sizing.")


def summary(crop="full", seq_len=8):
    """Build and print the model without training — a quick shape/param check."""
    frame_hw = (dl.RADAR_H, dl.RADAR_W) if crop == "radar" else _fpv_hw(crop)
    model = build_model(seq_len, frame_hw, n_outputs=len(MOVEMENT_KEYS))
    model.summary(print_fn=print)
    print(f"\nInput: ({seq_len}, {frame_hw[0]}, {frame_hw[1]}, 3)  ->  "
          f"output: {len(MOVEMENT_KEYS)} sigmoids {MOVEMENT_KEYS}")


def _build_parser():
    p = argparse.ArgumentParser(
        description="WASD-from-FPV recurrent baseline (movement-only BC). Also runs "
                    "the #7 radar->movement comparison via --crop radar.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--train", action="store_true", help="train the baseline")
    g.add_argument("--summary", action="store_true",
                   help="build + print the model, no training")
    p.add_argument("--crop", default="full",
                   help="input feed: full/centre FPV crop (default full), or 'radar' "
                        "for the #7 movement-from-radar comparison")
    p.add_argument("--seq-len", type=int, default=8, help="frames per window (default 8)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--use-keep-mask", action="store_true",
                   help="exclude blank/no-radar frames via the D-026 keep-mask")
    p.add_argument("--holdout-frac", type=float, default=dl.DEFAULT_HOLDOUT_FRAC)
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.summary:
        summary(crop=args.crop, seq_len=args.seq_len)
    elif args.train:
        train(crop=args.crop, seq_len=args.seq_len, epochs=args.epochs,
              batch_size=args.batch, use_keep_mask=args.use_keep_mask,
              holdout_frac=args.holdout_frac)
    else:
        print("Choose --train or --summary. See `python -m src.model_lstm -h`.")


if __name__ == "__main__":
    main()
