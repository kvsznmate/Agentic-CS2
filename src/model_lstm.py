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

NAVIGATION YAW (dx/dy, D-036): by DEFAULT the model also predicts mouse motion
(dx/dy) via a second output branch, because a WASD-only mover can only strafe
along the axis it spawned facing — it cannot turn to traverse the map. dx/dy here
are NAVIGATION yaw for the movement feed, NOT combat aim (aim is the separate
detector-gated model #10 -> #11 the arbiter switches to on enemy contact). The
look targets are standardized using TRAIN-split stats (saved as a sidecar for
play-time to invert) and evaluated as per-axis MAE vs a zero-motion baseline. Pass
--no-look to train the original WASD-only single-head baseline unchanged.

Usage:
  python -m src.model_lstm --train                     # FPV + navigation-yaw (default)
  python -m src.model_lstm --train --no-look           # WASD-only baseline (original)
  python -m src.model_lstm --train --crop centre       # centred FPV crop
  python -m src.model_lstm --train --crop radar        # the #7 radar->movement comparison
  python -m src.model_lstm --train --use-keep-mask      # exclude blank frames (D-026)
  python -m src.model_lstm --train --look-loss-weight 0.5  # rebalance look vs button loss
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


_TF_GPU_CONFIGURED = False


def _configure_gpu(tf):
    """Enable GPU memory growth once, before any allocation (OOM mitigation).

    On a memory-tight laptop GPU (the RTX 4050 here exposes only ~3.4 GB to TF
    after Windows/desktop/CS2 take their share), TF 2.10's default of grabbing a
    large block of VRAM at startup can fail with 'Memory allocation failure' ->
    'failed to create cublas handle' -> a cascade of cudnn/cublas errors mid-
    training (seen on a 128x128 radar run at batch 32). set_memory_growth makes TF
    allocate incrementally instead, so it takes only what each step needs and
    doesn't over-reserve. This is NOT a full fix on its own — if a single batch's
    activations genuinely exceed free VRAM it will still OOM, and the remedies
    remain (close CS2/other GPU users; lower --batch) — but it removes the
    preallocation failure mode and is the correct default for this stack.

    Must run before the GPU is initialised; _import_tf calls it at first import.
    Guarded so it runs once and is harmless with no GPU or if growth was already
    set (TF forbids changing it after initialisation, so we swallow that).
    """
    global _TF_GPU_CONFIGURED
    if _TF_GPU_CONFIGURED:
        return
    _TF_GPU_CONFIGURED = True
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            print(f"  (GPU memory growth enabled on {len(gpus)} device(s) "
                  f"to avoid startup over-allocation.)")
    except (RuntimeError, ValueError) as e:
        # RuntimeError: GPU already initialised (growth can't change now) - fine.
        print(f"  (GPU memory-growth setup skipped: {e.__class__.__name__}.)")


def _import_tf():
    """Import TensorFlow lazily with a clear message if the env isn't set up.

    Kept out of module import so `-h` and syntax checks don't require TF. The env
    (environment.yml, D-011) pins tensorflow==2.10.1; if import fails we say how to
    fix it rather than dumping a raw ImportError. GPU memory growth is configured
    here (once), at the first import, before anything touches the device.
    """
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"TensorFlow import failed ({e!r}).\n"
            f"Activate the project env and verify the GPU stack:\n"
            f"  conda activate agentic-cs2\n"
            f"  python -m src.smoke_test\n"
            f"environment.yml pins tensorflow==2.10.1 (D-011).")
    _configure_gpu(tf)
    return tf


def build_model(seq_len, frame_hw, n_outputs, channels=3, n_look=0,
                look_loss_weight=1.0):
    """CNN-encoder + LSTM, many-to-one. Button sigmoids, optional look branch.

    Input : (T, H, W, C) uint8-range frames (cast+scaled inside the model so the
            data pipeline can stay uint8 BGR end-to-end).
    Output:
      * n_look == 0 (default, UNCHANGED baseline): a single output tensor
        `move_keys` of n_outputs sigmoids in [0,1] — per-key hold probability at
        the last frame. Byte-for-byte the original model; existing callers and
        saved models are unaffected.
      * n_look > 0 (D-036): TWO named outputs — `move_keys` (n_outputs sigmoids,
        as above) AND `look` (n_look linear units) predicting STANDARDIZED mouse
        motion (dx/dy) at the last frame, for NAVIGATION yaw. The button head is
        identical; the look head is a separate branch off the SAME LSTM feature
        vector, so adding it cannot change what the button head computes.

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
      * LOOK HEAD (D-036): a small dense -> linear n_look output. LINEAR (no
        activation) because dx/dy are unbounded real deltas; the targets are
        STANDARDIZED by the trainer (train-split mean/std), so a plain linear
        output regressed under MSE is the right shape. It is NAVIGATION yaw, not
        combat aim — see the module note and D-036. loss weight is tunable so the
        MSE term can be balanced against the per-key BCE if one dominates.
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

    feat = layers.LSTM(128, return_sequences=False)(x)   # many-to-one final state
    h = layers.Dropout(0.3)(feat)
    h = layers.Dense(64, activation="relu")(h)
    keys_out = layers.Dense(n_outputs, activation="sigmoid", name="move_keys")(h)

    if n_look and n_look > 0:
        # Separate look branch off the SAME shared feature vector `feat`. A small
        # dense then a LINEAR output for standardized dx/dy (D-036). Kept modest
        # to match the baseline's small-data footing.
        lh = layers.Dense(32, activation="relu")(feat)
        look_out = layers.Dense(n_look, activation=None, name="look")(lh)
        model = models.Model(inp, [keys_out, look_out],
                             name="move_fpv_lstm_nav")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss={"move_keys": "binary_crossentropy", "look": "mse"},
            loss_weights={"move_keys": 1.0, "look": float(look_loss_weight)},
            metrics={"move_keys": "binary_accuracy", "look": "mae"},
        )
        return model

    # n_look == 0: the ORIGINAL single-head baseline, unchanged.
    model = models.Model(inp, keys_out, name="move_fpv_lstm_baseline")
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


def _evaluate_lift(model, seq_ds, batch_size, baseline_rates=None, eval_batch=8):
    """Per-key held-out accuracy, majority-class baseline, and lift.

    baseline_rates: dict key->held_fraction from the TRAIN set. The majority-class
    baseline accuracy for a key is max(rate, 1-rate) evaluated on THIS set's
    labels — the honest 'predict the more common state' reference. If not given,
    it is computed from this dataset itself.

    eval_batch: how many WINDOWS to run through model() per forward pass. Kept
    SMALL (default 8) and independent of the training batch because a window is T
    frames, so a TimeDistributed conv sees eval_batch*T frames at once; on the
    large FPV crop (150x270) a full training-size batch made that first-conv
    activation ~0.6 GB in ONE tensor and OOM'd on the 3.4 GB GPU after training
    had fragmented VRAM (the radar 128x128 run survived the same code). Small
    fixed-size predict calls keep the peak activation bounded regardless of frame
    size, which is what makes eval robust on this machine (D-027).

    Returns a dict per key: {"acc", "baseline", "lift", "held"} plus "n".
    """
    n = len(seq_ds)
    if n == 0:
        return {"n": 0}
    keys = seq_ds.target_keys
    n_correct = np.zeros(len(keys))
    n_held = np.zeros(len(keys))
    total = 0
    # A two-output (navigation-yaw) model returns [move_keys, look]; the single-
    # head baseline returns one array. Resolve the move_keys output index ONCE by
    # name (not assumed order), matching how _evaluate_look and play_movement do
    # it, so we score the BUTTON head and never the look head (D-036).
    keys_out_idx = 0
    try:
        if isinstance(model.output_names, (list, tuple)) and len(model.output_names) > 1:
            keys_out_idx = list(model.output_names).index("move_keys")
    except (ValueError, AttributeError):
        keys_out_idx = 0
    # Deterministic pass (no shuffle) so eval is stable. Fetch in the loader's
    # batch_size chunks, but run model() in eval_batch-sized micro-batches via
    # __call__ (not .predict, which builds/retains a predict function and can add
    # its own batching); training=False, and we slice so peak VRAM is bounded.
    for X, Y in seq_ds.iter_batches(batch_size=batch_size, shuffle=False):
        preds = []
        for s in range(0, X.shape[0], eval_batch):
            xb = X[s:s + eval_batch]
            out = model(xb, training=False)
            ob = out[keys_out_idx] if isinstance(out, (list, tuple)) else out
            preds.append(np.asarray(ob))
        p_keys = np.concatenate(preds, axis=0) if preds else np.zeros((0, len(keys)))
        pred = (p_keys >= 0.5).astype(np.float32)
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


# ── Look (dx/dy navigation-yaw) standardization + evaluation (D-036) ──

def _look_stats_from_balance(balance):
    """Build (mean[2], std[2]) float32 arrays from a look_balance() dict.

    std is floored at 1.0 so a near-constant axis (e.g. dy barely moving during
    pure navigation) cannot divide-by-~0 and blow up the standardized targets.
    A floor of 1 device-count is negligible against real turning deltas and keeps
    the transform invertible and stable.
    """
    mean = np.array([balance["dx"]["mean"], balance["dy"]["mean"]], dtype=np.float32)
    std = np.array([balance["dx"]["std"], balance["dy"]["std"]], dtype=np.float32)
    std = np.maximum(std, 1.0).astype(np.float32)
    return mean, std


def _standardize(look_raw, mean, std):
    """(raw dx/dy) -> standardized, using train-split mean/std (D-036)."""
    return (look_raw - mean) / std


def _evaluate_look(model, seq_ds, batch_size, look_mean, look_std, baseline_abs,
                   eval_batch=8):
    """Per-axis look MAE in RAW device units, vs the zero-motion baseline.

    The model emits STANDARDIZED dx/dy; we invert with (train) mean/std back to
    device units before scoring, so the reported error reads as real mouse counts
    and is comparable across runs. baseline_abs is the per-axis mean-abs of THIS
    set's raw dx/dy (predict-zero error). A model is only meaningful where its MAE
    is clearly below baseline_abs — the dx/dy analogue of beating the button
    majority-class baseline.

    eval_batch: windows per forward pass, small and fixed for the same VRAM reason
    as _evaluate_lift (large FPV frames * T make a full-batch conv activation OOM
    on the 3.4 GB GPU post-training).

    Returns {"dx": {"mae", "baseline", "improve"}, "dy": {...}, "n"}. `improve` is
    baseline - mae (positive = better than predicting no motion).
    """
    n = len(seq_ds)
    if n == 0:
        return {"n": 0}
    # Resolve the look output index by name (not assumed order), mirroring
    # _evaluate_lift. build_model orders outputs [move_keys, look], so this is 1
    # today, but resolving by name keeps it correct if the order ever changes.
    look_out_idx = 1
    try:
        if isinstance(model.output_names, (list, tuple)):
            look_out_idx = list(model.output_names).index("look")
    except (ValueError, AttributeError):
        look_out_idx = 1
    abs_err = np.zeros(2, dtype=np.float64)
    total = 0
    # Micro-batched model() calls (not .predict) to bound peak VRAM — see
    # _evaluate_lift's note on the FPV OOM.
    for Xb, _Yk, Yl_raw in _iter_look_batches(seq_ds, batch_size, shuffle=False):
        preds = []
        for s in range(0, Xb.shape[0], eval_batch):
            out = model(Xb[s:s + eval_batch], training=False)
            ob = out[look_out_idx] if isinstance(out, (list, tuple)) else out
            preds.append(np.asarray(ob))
        pred_look_std = (np.concatenate(preds, axis=0) if preds
                         else np.zeros((0, 2), np.float32))
        pred_raw = pred_look_std * look_std + look_mean       # invert standardize
        abs_err += np.abs(pred_raw - Yl_raw).sum(axis=0)
        total += Yl_raw.shape[0]
    mae = abs_err / max(total, 1)
    out = {"n": total}
    for i, ax in enumerate(("dx", "dy")):
        base = baseline_abs[ax]
        out[ax] = {"mae": float(mae[i]), "baseline": float(base),
                   "improve": float(base - mae[i])}
    return out


def _print_look_table(title, res):
    print(f"\n{title} (n={res.get('n', 0)} windows):")
    if res.get("n", 0) == 0:
        print("  (no windows)")
        return
    print(f"  {'axis':<4} {'MAE':>9} {'zero-mot':>10} {'improve':>9}")
    for ax in ("dx", "dy"):
        r = res[ax]
        print(f"  {ax:<4} {r['mae']:8.1f}  {r['baseline']:9.1f}  "
              f"{r['improve']:+8.1f}   (device units)")


def _iter_look_batches(seq_ds, batch_size, shuffle, seed=None):
    """Yield (X, Y_keys, Y_look_raw) batches from a SequenceDataset (D-036).

    Uses get_batch_with_look so the look (raw dx/dy) targets ride along with the
    button targets from the SAME assembled action vector. Mirrors the ordering of
    iter_batches so shuffling/behaviour matches the button path.
    """
    n = len(seq_ds)
    order = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    end = n
    for start in range(0, end, batch_size):
        wp = order[start:start + batch_size].tolist()
        yield seq_ds.get_batch_with_look(wp)


def _make_keras_sequence(seq_ds, batch_size, predict_look, look_mean, look_std,
                         oversample=1, eventful_mask=None):
    """Wrap a SequenceDataset as a keras.utils.Sequence for model.fit (perf fix).

    WHY THIS EXISTS: the training loop originally called model.train_on_batch() in
    a Python for-loop over ~800 batches/epoch. On TF 2.10 that path rebuilds/leaks
    per-call state and, on this memory-tight machine, degraded into disk-swap
    thrash mid-run (a ~30-min stall between epochs, tf.data 'Optimization loop
    failed: CANCELLED' warnings, then a burst of epochs). model.fit() over a
    keras.utils.Sequence uses Keras's CACHED compiled train function instead of
    re-entering the train step from Python each batch, which is the supported,
    non-leaking path — and, crucially, it keeps the EXACT compiled loss/metrics
    from build_model (we do NOT re-derive the two-head weighted loss by hand,
    which could silently diverge from model.compile).

    The Sequence yields the SAME batches the old loop did:
      * predict_look=True  -> (X, {"move_keys": Yk, "look": Yl_standardized})
      * predict_look=False -> (X, Yk)
    Standardization of the look targets uses the TRAIN-split mean/std passed in
    (D-036), identical to the old loop. Shuffling is per-epoch via on_epoch_end,
    matching the old `seed=ep` reshuffle.

    OVERSAMPLING (D-037-followup, the turn-imbalance fix): with oversample>1 and
    an eventful_mask (per-window bool from SequenceDataset.eventful_mask), windows
    flagged eventful (rare movement key held, or a real turn / large |dx|) are
    REPEATED `oversample` times in the per-epoch index, so the model sees more
    turning per epoch. This is a DATA-MIX change only:
      * it does NOT touch the loss or the labels (no divergence-from-compile risk),
      * it applies ONLY here (the train adapter) — EVAL still visits each window
        once, unweighted, so reported lift/MAE stay honest and comparable,
      * it cannot cross the D-021 split (repeats indices WITHIN this train dataset).
    WHY (probe finding): on forward-heavy data the keys head suppressed A/S/D and
    the dx head mean-collapsed — because turn windows are rare. Repeating them is
    the lower-risk lever than reweighting the loss. CAVEAT: repetition cannot
    create turn variety that isn't in the data; it raises exposure to the turns
    that ARE there, and can overfit them — more turning DATA is the real fix. With
    oversample=1 (default) this is the original behaviour exactly.
    """
    _import_tf()
    from tensorflow import keras

    n = len(seq_ds)
    # Build the BASE index (which window positions exist, with repeats). Eventful
    # windows appear `oversample` times; everything else once. Shuffling happens
    # per-epoch over this base list. If no mask or oversample<=1, it's arange(n).
    if oversample and oversample > 1 and eventful_mask is not None and n > 0:
        reps = np.ones(n, dtype=np.int64)
        reps[np.asarray(eventful_mask, dtype=bool)] = int(oversample)
        base_index = np.repeat(np.arange(n), reps)
    else:
        base_index = np.arange(n)

    class _SeqAdapter(keras.utils.Sequence):
        def __init__(self):
            self._ds = seq_ds
            self._bs = batch_size
            self._base = base_index
            self._order = self._base.copy()
            self._epoch = 0
            self._rng = np.random.default_rng(0)
            self._rng.shuffle(self._order)

        def __len__(self):
            # Batches over the (possibly oversampled) index; final partial kept.
            m = len(self._order)
            return (m + self._bs - 1) // self._bs

        def __getitem__(self, i):
            wp = self._order[i * self._bs:(i + 1) * self._bs].tolist()
            if predict_look:
                X, Yk, Yl_raw = self._ds.get_batch_with_look(wp)
                Yl = _standardize(Yl_raw, look_mean, look_std)
                return X, {"move_keys": Yk, "look": Yl}
            X, Yk = self._ds.get_batch(wp)
            return X, Yk

        def on_epoch_end(self):
            # Reshuffle the base index each epoch (reseeded, like the old loop).
            self._epoch += 1
            self._rng = np.random.default_rng(self._epoch)
            self._order = self._base.copy()
            self._rng.shuffle(self._order)

    return _SeqAdapter()


def train(crop="full", seq_len=8, epochs=10, batch_size=32,
          use_keep_mask=False, holdout_frac=dl.DEFAULT_HOLDOUT_FRAC,
          manual_holdout=None, save=True, predict_look=True, look_loss_weight=1.0,
          oversample=1):
    """Train the movement baseline and report honest held-out metrics.

    Steps: build leak-free sequence splits (D-021) with the chosen input feed;
    report window counts + the train-set key balance (the baseline reference);
    train; then evaluate per-key lift on held-out, printed next to the TRAIN lift
    so memorisation (large gap) is visible; and stamp the whole thing PROVISIONAL
    if below the volume floor.

    predict_look (D-036): when True (default), the model ALSO predicts navigation
    yaw (dx/dy) via a second output branch, so the movement feed can rotate the
    player, not just strafe. dx/dy targets are STANDARDIZED using stats computed
    from the TRAIN split ONLY (never held-out — D-021 discipline extended to the
    target transform); the stats are saved next to the model so play-time can
    invert them. Look is evaluated as per-axis MAE vs a zero-motion baseline, in
    device units, train and held-out side by side. Set predict_look=False to train
    the ORIGINAL WASD-only single-head baseline unchanged.

    SCOPE (D-036): dx/dy here are NAVIGATION yaw for the movement feed, NOT combat
    aim. Aim is the separate detector-gated model (#10 -> #11) the arbiter switches
    to on enemy contact; nothing here front-runs that gate.
    """
    tf = _import_tf()

    frame_hw = (dl.RADAR_H, dl.RADAR_W) if crop == "radar" else _fpv_hw(crop)
    print(f"Building sequence datasets: crop='{crop}', seq_len={seq_len}, "
          f"targets={MOVEMENT_KEYS}, keep_mask={use_keep_mask}, "
          f"predict_look={predict_look}.")
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

    # ── Look (dx/dy) standardization stats — TRAIN SPLIT ONLY (D-036) ──
    # Computed from the train sequences and reused unchanged for held-out and at
    # inference. Never from held-out: that would leak the test distribution into
    # the training transform, the same leak D-021's whole-session split prevents,
    # just moved into preprocessing. Saved with the model so play_movement.py can
    # invert the standardization to real device deltas.
    look_mean = look_std = None
    train_look_bal = hold_look_bal = None
    if predict_look:
        train_look_bal = train_seq.look_balance()
        look_mean, look_std = _look_stats_from_balance(train_look_bal)
        print("\nLook (dx/dy) standardization — from TRAIN split only (D-036):")
        print(f"  dx: mean {look_mean[0]:+.2f}, std {look_std[0]:.2f}, "
              f"zero-motion MAE baseline {train_look_bal['dx']['mean_abs']:.1f}")
        print(f"  dy: mean {look_mean[1]:+.2f}, std {look_std[1]:.2f}, "
              f"zero-motion MAE baseline {train_look_bal['dy']['mean_abs']:.1f}")
        if n_hold > 0:
            hold_look_bal = hold_seq.look_balance()

    n_look = 2 if predict_look else 0
    model = build_model(seq_len, frame_hw, n_outputs=len(MOVEMENT_KEYS),
                        n_look=n_look, look_loss_weight=look_loss_weight)
    model.summary(print_fn=lambda s: print("  " + s))

    # Train via model.fit over a keras.utils.Sequence (perf fix — see
    # _make_keras_sequence). The old hand-rolled train_on_batch loop stalled into
    # disk-swap on this machine; fit uses Keras's cached compiled train function
    # and the SAME compiled loss/metrics, so numbers are unchanged, memory is not.
    # Oversampling of 'eventful' (turn / rare-key) windows, off by default
    # (D-037-followup). Computed on the TRAIN split only; EVAL is never oversampled
    # so its lift/MAE stay honest. See _make_keras_sequence for the rationale and
    # the caveat that this cannot invent turn variety absent from the data.
    ev_mask = None
    if oversample and oversample > 1:
        ev_mask, ev_info = train_seq.eventful_mask()
        base = ev_info["n"]
        eff = base + (oversample - 1) * ev_info["eventful"]
        print(f"\nOversampling eventful windows x{oversample} (D-037-followup): "
              f"{ev_info['eventful']}/{base} windows flagged eventful "
              f"({ev_info['rare_hit']} rare-key, {ev_info['dx_hit']} big-|dx| "
              f"> {ev_info['dx_abs_thresh']:.0f}); effective train windows/epoch "
              f"{base} -> {eff}. EVAL is NOT oversampled (honest lift/MAE).")
        if ev_info["eventful"] == 0:
            print("  (No eventful windows found — oversampling is a no-op. Is the "
                  "data all forward-walking?)")

    steps = max(1, n_train // batch_size)
    print(f"\nTraining: {epochs} epochs x ~{steps} steps (batch {batch_size}) "
          f"on {frame_hw[0]}x{frame_hw[1]} frames...")
    train_gen = _make_keras_sequence(train_seq, batch_size, predict_look,
                                     look_mean, look_std,
                                     oversample=oversample, eventful_mask=ev_mask)
    t0 = time.perf_counter()
    # workers=1 / no multiprocessing: the SequenceDataset reads frames via a
    # shared handle that isn't fork-safe, and single-worker is what removed the
    # leak; per-epoch Keras logging (verbose=2) prints one line per epoch, close
    # to the old readout.
    model.fit(train_gen, epochs=epochs, verbose=2, shuffle=False,
              workers=1, use_multiprocessing=False)
    train_secs = time.perf_counter() - t0
    print(f"Trained in {train_secs:.0f}s.")

    # ── Honest evaluation: per-key lift on held-out AND train ──
    keys = MOVEMENT_KEYS
    train_res = _evaluate_lift(model, train_seq, batch_size, baseline_rates=train_balance)
    hold_res = (_evaluate_lift(model, hold_seq, batch_size, baseline_rates=train_balance)
                if n_hold > 0 else {"n": 0})
    train_mean = _print_lift_table("TRAIN lift", train_res, keys)
    hold_mean = _print_lift_table("HELD-OUT lift", hold_res, keys)

    # ── Look evaluation (D-036): per-axis MAE vs zero-motion, device units ──
    if predict_look:
        train_look_res = _evaluate_look(
            model, train_seq, batch_size, look_mean, look_std,
            baseline_abs={"dx": train_look_bal["dx"]["mean_abs"],
                          "dy": train_look_bal["dy"]["mean_abs"]})
        _print_look_table("TRAIN look MAE", train_look_res)
        if n_hold > 0:
            hold_look_res = _evaluate_look(
                model, hold_seq, batch_size, look_mean, look_std,
                baseline_abs={"dx": hold_look_bal["dx"]["mean_abs"],
                              "dy": hold_look_bal["dy"]["mean_abs"]})
            _print_look_table("HELD-OUT look MAE", hold_look_res)
            print("  Reading it: MAE clearly BELOW the zero-motion column = the")
            print("  model predicts turning better than assuming no turn. dx")
            print("  (horizontal yaw) should improve most; dy (pitch) barely moves")
            print("  in pure navigation, so a near-zero dy improvement is expected,")
            print("  not a bug (D-036).")

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
        # Stub includes the key count AND whether a look head is present, so a
        # nav model (with look) does NOT overwrite a WASD-only model file, and
        # play_movement can tell them apart. e.g. wasd_lstm_full_T8_k6_look.keras.
        look_tag = "_look" if predict_look else ""
        stub = f"wasd_lstm_{crop}_T{seq_len}_k{len(MOVEMENT_KEYS)}{look_tag}"
        path = os.path.join(_MODEL_DIR, stub + ".keras")
        try:
            model.save(path)
            print(f"\nSaved model: {path}")
        except Exception as e:  # noqa: BLE001
            print(f"\n(Model save skipped: {e!r})")
        # Save the look standardization stats as a SIDECAR next to the model
        # (D-036). play_movement.py must load these to invert the standardized
        # dx/dy the look head emits back into real device deltas; without them the
        # look output is uninterpretable. Kept as a small .npz rather than baked
        # into the .keras so it is trivial to read without importing the model.
        # A WASD-only run writes no sidecar (there is nothing to invert).
        if predict_look and look_mean is not None:
            stats_path = os.path.join(_MODEL_DIR, stub + ".look_stats.npz")
            try:
                np.savez(stats_path, mean=look_mean, std=look_std,
                         axes=np.array(["dx", "dy"]),
                         seq_len=np.array(seq_len), crop=np.array(str(crop)))
                print(f"Saved look-standardization stats: {stats_path}")
                print("  (play_movement.py inverts dx/dy with these; D-036.)")
            except Exception as e:  # noqa: BLE001
                print(f"(Look-stats save skipped: {e!r})")
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


def summary(crop="full", seq_len=8, predict_look=True):
    """Build and print the model without training — a quick shape/param check."""
    frame_hw = (dl.RADAR_H, dl.RADAR_W) if crop == "radar" else _fpv_hw(crop)
    n_look = 2 if predict_look else 0
    model = build_model(seq_len, frame_hw, n_outputs=len(MOVEMENT_KEYS),
                        n_look=n_look)
    model.summary(print_fn=print)
    if predict_look:
        print(f"\nInput: ({seq_len}, {frame_hw[0]}, {frame_hw[1]}, 3)  ->  outputs: "
              f"move_keys={len(MOVEMENT_KEYS)} sigmoids {MOVEMENT_KEYS} + "
              f"look=2 linear (dx,dy standardized; navigation yaw, D-036)")
    else:
        print(f"\nInput: ({seq_len}, {frame_hw[0]}, {frame_hw[1]}, 3)  ->  "
              f"output: {len(MOVEMENT_KEYS)} sigmoids {MOVEMENT_KEYS} "
              f"(WASD-only, no look head)")


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
    # Look head (navigation yaw, D-036). ON by default: the movement feed needs
    # to rotate the player, not only strafe. --no-look reproduces the original
    # WASD-only single-head baseline exactly.
    p.add_argument("--no-look", dest="predict_look", action="store_false",
                   help="train WASD-only (no dx/dy navigation-yaw head); "
                        "reproduces the original single-head baseline")
    p.set_defaults(predict_look=True)
    p.add_argument("--look-loss-weight", type=float, default=1.0,
                   help="weight on the look (dx/dy MSE) loss relative to the "
                        "per-key BCE (default 1.0; raise/lower if one term "
                        "dominates training)")
    p.add_argument("--oversample", type=int, default=1,
                   help="repeat 'eventful' windows (rare movement key held, or a "
                        "real turn / large |dx|) this many times per epoch, to "
                        "counter the forward-walking bias that suppresses A/S/D and "
                        "collapses dx (D-037-followup). 1 = off (default). Affects "
                        "TRAIN only; eval stays honest. Caveat: cannot invent turns "
                        "absent from the data — more turning DATA is the real fix.")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.summary:
        summary(crop=args.crop, seq_len=args.seq_len,
                predict_look=args.predict_look)
    elif args.train:
        train(crop=args.crop, seq_len=args.seq_len, epochs=args.epochs,
              batch_size=args.batch, use_keep_mask=args.use_keep_mask,
              holdout_frac=args.holdout_frac, predict_look=args.predict_look,
              look_loss_weight=args.look_loss_weight, oversample=args.oversample)
    else:
        print("Choose --train or --summary. See `python -m src.model_lstm -h`.")


if __name__ == "__main__":
    main()
