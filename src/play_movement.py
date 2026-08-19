"""play_movement.py — drive CS2 movement from the trained WASD baseline (demo).

The live counterpart to training: grab frames, run the movement model, and press
W/A/S/D so you can WATCH what the baseline actually learned — on an empty/local
map, not a real match. This is a DEMO / diagnostic, NOT the agent: no mouse/aim,
no detection, no arbiter (those are M4/M6). It answers one question by eye: does
the movement the model predicts look like anything, or does it just walk forward
and drift?

READ THIS BEFORE RUNNING — WHAT TO EXPECT (so you don't misread it):
  * The baseline learned W strongly and A/S/D weakly (the eval: +23.5pp on W,
    ~0 on the rest). So expect mostly-forward motion with little strafing. That
    is the model being honest about what it learned, not a bug.
  * BEHAVIOURAL-CLONING DRIFT is expected. The model trained on YOUR frames; once
    it acts, it produces its OWN frames, which drift from anything it saw, and
    errors compound (covariate shift). So after a few seconds of open-loop
    control it may wander into a wall or do something degenerate. That is the
    classic BC failure mode, NOT necessarily a broken model. Watch the first few
    seconds after each (re)arm for the cleanest read.
  * The model needs SEQ_LEN frames before its first prediction — a brief pause at
    the start while the buffer fills. Expected.

TIMING MUST MATCH TRAINING (the subtle correctness point):
The model was trained at ~15 FPS (LOOP_FPS) with an 8-frame window, so it expects
frames spaced ~66 ms apart. This loop PACES itself to the same rate, reusing the
recorder's `_pace`, so the 8-frame window covers the same real-time span it did in
training. Running unpaced (faster) would feed out-of-distribution sequences and
make the model look worse than it is. Preprocessing is IDENTICAL to training by
construction: frames come from the same `Capture.grab()` (BGR, 150x270, uint8),
and the model's own Rescaling layer does /255 — so nothing here re-scales or
reorders channels.

SAFETY — OFFLINE / LOCAL ONLY (D-007). This injects synthetic keystrokes, which
can trip anti-cheat online. Run it ONLY against a local server with bots / an
empty map (e.g. `map de_dust2` from the console, sv_cheats optional). NEVER on
online matchmaking. Controls make you the one in charge:
  * The agent starts DISARMED. It predicts but does NOT press keys until you arm.
  * F9 toggles ARM/DISARM. Disarming instantly releases all keys.
  * F8 quits (and releases all keys).
So you can always cut it by hitting F9/F8 if it does something silly.

Usage:
  python -m src.play_movement                     # newest model for crop=full
  python -m src.play_movement --model PATH         # a specific .keras model
  python -m src.play_movement --crop full          # must match the model's input feed
  python -m src.play_movement --threshold 0.4      # prob -> hold cutoff (default 0.4, multi-key)
"""

import argparse
import glob
import os
import time
from collections import deque

import numpy as np
import win32api

from src.capture import Capture
from src import capture_config as cfg
from src.recorder import _pace, _key_down, VK_QUIT, LOOP_FPS
from src.key_output import KeyController
from src import data_loader as dl
from src.model_lstm import MOVEMENT_KEYS   # single source of truth for the key set


_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")

# MOVEMENT_KEYS is imported from model_lstm so the play loop's keys ALWAYS match
# what the model was trained to output (e.g. adding shift there adds it here). It
# now includes shift; key_output has shift's scan code (0x2A) already. We still
# CHECK the loaded model's actual output width against this list at load time, so
# a stale 4-key model can't be silently driven with a 5-key mapping.

# Arm/disarm hotkey. F9 is unlikely to be bound in CS2 (F8 is the quit key, reused
# from the recorder for muscle-memory consistency).
VK_ARM = 0x78   # F9


def _resolve_model(model_arg, crop):
    """Pick the model file to load: an explicit path, or the newest for this crop.

    With no --model, looks for data/models/wasd_lstm_<crop>_T*.keras (the trainer's
    naming) and takes the newest, so `--crop full` loads the FPV model and
    `--crop radar` the radar one without needing the exact filename.
    """
    if model_arg:
        if not os.path.exists(model_arg):
            raise FileNotFoundError(f"Model not found: {model_arg}")
        return model_arg
    # Trainer names models wasd_lstm_<crop>_T<seq>_k<nkeys>.keras (k-suffix added
    # when shift made it 5 keys). Match with-or-without the k-suffix so both old
    # 4-key models and new 5-key ones are found; newest wins.
    pattern = os.path.join(_MODEL_DIR, f"wasd_lstm_{crop}_T*.keras")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No trained model for crop='{crop}' in {_MODEL_DIR} "
            f"(looked for {os.path.basename(pattern)}). Train one first:\n"
            f"  python -m src.model_lstm --train --crop {crop}")
    return max(matches, key=os.path.getmtime)


def _infer_seq_len(model):
    """Read the sequence length T the model expects from its input shape.

    Input shape is (None, T, H, W, 3); we need T to size the rolling buffer so it
    matches how the model was trained (mismatch = wrong-length sequences).
    """
    shape = model.input_shape           # (None, T, H, W, C)
    if len(shape) != 5 or shape[1] is None:
        raise ValueError(f"Unexpected model input shape {shape}; expected "
                         f"(None, T, H, W, 3).")
    return int(shape[1])


def _crop_frame(frame, crop):
    """Apply the SAME crop the loader/trainer used for this feed.

    For 'full' the frame is unchanged; 'centre' uses the loader's default FPV
    rectangle; a 4-tuple is a custom FPV crop. 'radar' is NOT handled here — a
    radar-driven demo would need the radar array from grab_with_radar(), which is
    out of scope for this movement demo (the model you'll usually run here is the
    FPV one). Keeping crops in sync with data_loader avoids feeding the model a
    differently-cropped frame than it trained on.
    """
    if crop in (None, "full"):
        return frame
    if crop == "centre":
        t, l, h, w = dl.CENTRE_CROP_DEFAULT
        return frame[t:t + h, l:l + w]
    if isinstance(crop, (tuple, list)) and len(crop) == 4:
        t, l, h, w = crop
        return frame[t:t + h, l:l + w]
    raise ValueError(f"play_movement supports full/centre/custom FPV crops, not "
                     f"{crop!r}. (A radar-driven demo is a separate path.)")


def play(model_arg=None, crop="full", threshold=0.4):
    # Lazy TF import (keeps -h and non-TF envs usable).
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"TensorFlow import failed ({e!r}). "
                         f"Activate the env and run `python -m src.smoke_test`.")

    model_path = _resolve_model(model_arg, crop)
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    seq_len = _infer_seq_len(model)

    # Guard: the model's output width must match our key list, or the probs won't
    # line up with the keys (e.g. a 4-key WASD model loaded with a 5-key mapping
    # would misassign shift). Fail clearly instead of pressing the wrong keys.
    n_out = int(model.output_shape[-1])
    if n_out != len(MOVEMENT_KEYS):
        raise SystemExit(
            f"Model outputs {n_out} keys but MOVEMENT_KEYS has {len(MOVEMENT_KEYS)} "
            f"({MOVEMENT_KEYS}). This model was trained with a different key set "
            f"(likely the older 4-key WASD model, before shift was added). Retrain "
            f"with the current keys:  python -m src.model_lstm --train --crop {crop}")

    use_radar = (crop == "radar")
    if use_radar:
        ih, iw = cfg.RADAR_OUT_HW
    else:
        ih, iw = cfg.MODEL_INPUT_HW if crop in (None, "full") else _crop_hw(crop)
    feed_name = "radar (128x128 minimap)" if use_radar else f"'{crop}' FPV"
    print(f"Model expects T={seq_len} frames of the {feed_name} feed.")

    print("\n" + "=" * 68)
    print("MOVEMENT DEMO — LOCAL / OFFLINE SERVER ONLY (D-007).")
    print(f"This presses {'/'.join(k.upper() for k in MOVEMENT_KEYS)} in CS2. "
          f"Do NOT run on online matchmaking.")
    print("  F9 = ARM / DISARM (starts DISARMED; disarm releases all keys)")
    print("  F8 = quit (releases all keys)")
    print("Stand on an empty/local map, focus CS2, then press F9 to arm.")
    if use_radar:
        print("RADAR-DRIVEN: the model sees ONLY the minimap, not the world. This is")
        print("the #7 navigation-signal probe made physical — expect WEAKER, odder")
        print("motion than the FPV agent (deciding movement from a top-down blip with")
        print("no view of what's ahead). Whatever it does IS the honest answer to")
        print("'can you move from the radar alone' — not a polished mover.")
    else:
        print("Expect mostly-forward motion (the model learned W best) and some")
        print("drift after a few seconds (behavioural-cloning covariate shift).")
    print("=" * 68 + "\n")

    keyctl = KeyController(keys=MOVEMENT_KEYS)
    buf = deque(maxlen=seq_len)             # rolling window of recent frames
    armed = False
    prev_arm_down = False
    n = 0
    last_pred = None

    try:
        with Capture() as cap:
            while True:
                loop_start = time.perf_counter()

                if _key_down(VK_QUIT):       # F8
                    break

                # Edge-detect F9 so one press toggles once (not every frame held).
                arm_down = _key_down(VK_ARM)
                if arm_down and not prev_arm_down:
                    armed = not armed
                    if not armed:
                        keyctl.release_all()
                    print(f"\n{'ARMED — model is driving' if armed else 'DISARMED — keys released'}")
                prev_arm_down = arm_down

                frame, radar, _t = cap.grab_with_radar()  # SAME path as recording (D-024)
                buf.append(radar if use_radar else _crop_frame(frame, crop))
                n += 1

                # Need a full window before predicting.
                if len(buf) < seq_len:
                    _pace(loop_start, LOOP_FPS)
                    continue

                # (1, T, H, W, 3) uint8 — the model's Rescaling does /255 inside.
                x = np.expand_dims(np.stack(buf, axis=0), axis=0)
                probs = model.predict(x, verbose=0)[0]     # (4,)
                desired = {k for k, p in zip(MOVEMENT_KEYS, probs) if p >= threshold}

                if armed:
                    keyctl.apply(desired)

                # Live readout (throttled) so you can see decisions vs. actuation.
                if n % 5 == 0:
                    bar = "  ".join(f"{k}={p:.2f}" for k, p in zip(MOVEMENT_KEYS, probs))
                    held = "+".join(sorted(keyctl.held)) if keyctl.held else "-"
                    state = "ARMED" if armed else "disarmed"
                    print(f"  [{state}] {bar}   held[{held}]      ", end="\r", flush=True)

                _pace(loop_start, LOOP_FPS)
    finally:
        # ALWAYS release keys on the way out, however we exit — never leave the
        # game with a key stuck down.
        keyctl.release_all()
        print("\nStopped. All keys released.")


def _crop_hw(crop):
    if crop == "centre":
        _t, _l, h, w = dl.CENTRE_CROP_DEFAULT
        return (h, w)
    if isinstance(crop, (tuple, list)) and len(crop) == 4:
        return (int(crop[2]), int(crop[3]))
    return cfg.MODEL_INPUT_HW


def _build_parser():
    p = argparse.ArgumentParser(
        description="Drive CS2 movement from the trained WASD baseline (LOCAL server "
                    "only, D-007). A demo of what the model learned — no aim, no arbiter.")
    p.add_argument("--model", type=str, default=None,
                   help="path to a .keras model (default: newest for --crop)")
    p.add_argument("--crop", type=str, default="full",
                   help="input feed the model expects: full/centre/custom FPV crop, "
                        "or 'radar' for a radar-driven agent (the #7 probe, weaker by "
                        "design). Must match how the model was trained.")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="probability >= this presses the key (default 0.4). MULTI-KEY: "
                        "every key above the bar fires, so diagonals (e.g. W+D) work. "
                        "Raise it (e.g. 0.6) to suppress over-eager keys; lower it to "
                        "let more through. Tune live to find the sweet spot.")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    play(model_arg=args.model, crop=args.crop, threshold=args.threshold)


if __name__ == "__main__":
    main()
