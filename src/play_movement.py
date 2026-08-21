"""play_movement.py — drive CS2 movement from the trained WASD baseline (demo).

The live counterpart to training: grab frames, run the movement model, and press
W/A/S/D so you can WATCH what the baseline actually learned — on an empty/local
map, not a real match. This is a DEMO / diagnostic, NOT the agent: no detection,
no arbiter (those are M4/M6). It answers one question by eye: does the movement
the model predicts look like anything, or does it just walk forward and drift?

NAVIGATION-YAW MODELS (D-036): if the loaded model has a look head (dx/dy), this
demo DISPLAYS the predicted mouse motion each frame (un-standardized to real
device deltas via the model's `.look_stats.npz` sidecar) next to the key probs. By
default it does NOT move the mouse — dx/dy is display-only. Pass --drive-mouse to
ALSO actuate the view via the look head, which is gated three ways: the flag must
be set, the model must have a look head, and its sidecar stats must be present
(so the mouse is driven with real device units, never standardized values). Mouse
actuation uses `key_output.mouse_move_relative`, a NEW actuator that must first be
confirmed with `python -m src.key_output --selftest-mouse` (does the CS2 view
actually turn from injected motion?) — the D-029 discipline. Disarm (F9) or quit
(F8) stops all key AND mouse output instantly.

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
  python -m src.play_movement --drive-mouse        # ALSO turn the view from the look head
                                                   #   (after key_output --selftest-mouse passes)
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
from src.key_output import KeyController, mouse_move_relative
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
    `--crop radar` the radar one without needing the exact filename. Both WASD-only
    (`..._kN.keras`) and navigation-yaw (`..._kN_look.keras`) names match; newest
    wins, so a freshly-trained look model is picked up automatically (D-036).
    """
    if model_arg:
        if not os.path.exists(model_arg):
            raise FileNotFoundError(f"Model not found: {model_arg}")
        return model_arg
    # Trainer names models wasd_lstm_<crop>_T<seq>_k<nkeys>[_look].keras. Match
    # with-or-without the k-suffix and the _look suffix so old 4-key models, newer
    # WASD-only models, and navigation-yaw models are all found; newest wins.
    pattern = os.path.join(_MODEL_DIR, f"wasd_lstm_{crop}_T*.keras")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No trained model for crop='{crop}' in {_MODEL_DIR} "
            f"(looked for {os.path.basename(pattern)}). Train one first:\n"
            f"  python -m src.model_lstm --train --crop {crop}")
    return max(matches, key=os.path.getmtime)


def _load_look_stats(model_path):
    """Load the look-standardization sidecar for a model, or None if absent (D-036).

    The trainer saves `<model-stub>.look_stats.npz` next to a navigation-yaw model,
    holding the TRAIN-split dx/dy mean/std used to standardize the look targets.
    play-time MUST invert that transform (raw = std_out * std + mean) to turn the
    look head's standardized output back into real device deltas. A WASD-only model
    has no sidecar -> returns None, and the caller treats the model as button-only.

    The sidecar path is the model path with `.keras` replaced by `.look_stats.npz`,
    matching how the trainer names it (same stub). We do NOT infer look-vs-not from
    this file alone (the model's output structure is the source of truth); this
    just supplies the inversion constants when a look head is present.
    """
    if model_path.endswith(".keras"):
        stats_path = model_path[:-len(".keras")] + ".look_stats.npz"
    else:
        stats_path = model_path + ".look_stats.npz"
    if not os.path.isfile(stats_path):
        return None
    try:
        with np.load(stats_path, allow_pickle=False) as d:
            mean = d["mean"].astype(np.float32)
            std = d["std"].astype(np.float32)
    except (OSError, KeyError, ValueError) as e:
        print(f"  (look-stats sidecar {os.path.basename(stats_path)} unreadable: "
              f"{e.__class__.__name__} — look output will be shown standardized)")
        return None
    return {"mean": mean, "std": std, "path": stats_path}


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


def play(model_arg=None, crop="full", threshold=0.4, drive_mouse=False):
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

    # Detect whether this is a navigation-yaw model (two outputs: move_keys + look)
    # or the WASD-only baseline (one output). A functional model with two heads has
    # a LIST output_shape; one head is a single shape tuple. This is the source of
    # truth for look-vs-not — not the sidecar file (D-036).
    out_shape = model.output_shape
    has_look = isinstance(out_shape, list) and len(out_shape) == 2

    # Resolve the key-head output width for the guard below. For a two-output
    # model, the button head is the one named 'move_keys'; find it by name so we
    # don't assume output ordering. For a single-output model it's the lone shape.
    if has_look:
        # model.output_names aligns with model.outputs order; locate move_keys.
        try:
            keys_idx = list(model.output_names).index("move_keys")
        except (ValueError, AttributeError):
            # Fall back to assuming the first head is the key head (that is how
            # build_model orders them), but say so rather than silently guessing.
            print("  (could not find an output named 'move_keys'; assuming the "
                  "first output is the key head.)")
            keys_idx = 0
        n_out = int(out_shape[keys_idx][-1])
        look_idx = 1 - keys_idx
        n_look = int(out_shape[look_idx][-1])
    else:
        keys_idx = 0
        look_idx = None
        n_out = int(out_shape[-1])
        n_look = 0

    # Guard: the KEY head's width must match our key list, or the probs won't line
    # up with the keys (e.g. a 4-key WASD model loaded with a 5-key mapping would
    # misassign shift). Fail clearly instead of pressing the wrong keys.
    if n_out != len(MOVEMENT_KEYS):
        raise SystemExit(
            f"Model key-head outputs {n_out} keys but MOVEMENT_KEYS has "
            f"{len(MOVEMENT_KEYS)} ({MOVEMENT_KEYS}). This model was trained with a "
            f"different key set (likely the older 4-key WASD model, before shift was "
            f"added). Retrain with the current keys:  "
            f"python -m src.model_lstm --train --crop {crop}")

    # Load the look-standardization sidecar (D-036) if this is a look model. The
    # look head emits STANDARDIZED dx/dy; we invert with these train-split stats to
    # real device deltas for display (Piece 1) and later for actuation (Piece 2).
    look_stats = None
    if has_look:
        if n_look != 2:
            print(f"  (look head has width {n_look}, expected 2 for dx/dy; showing "
                  f"it raw/standardized.)")
        look_stats = _load_look_stats(model_path)
        if look_stats is not None:
            print(f"  Loaded look-standardization stats (dx/dy mean/std) from "
                  f"{os.path.basename(look_stats['path'])}.")
        else:
            print("  NOTE: this is a navigation-yaw model but its look-stats sidecar "
                  "was not found. dx/dy will be shown STANDARDIZED (not device "
                  "units). Re-train to regenerate the sidecar for real-unit output.")

    # Resolve whether we will ACTUATE the mouse this run (D-036, Piece 2). Guarded
    # hard: mouse driving requires (a) the user to opt in with --drive-mouse, (b) a
    # look head to drive it, and (c) the sidecar stats to produce real device
    # deltas. Missing any of these downgrades to display-only rather than moving
    # the mouse on bad or standardized values. The opt-in exists because injected
    # mouse motion is only trustworthy AFTER `key_output --selftest-mouse` confirms
    # CS2 turns from it — the D-029 discipline; this flag is the user asserting they
    # ran that check.
    actuate_mouse = False
    if drive_mouse:
        if not has_look:
            print("  --drive-mouse ignored: this model has no look head (WASD-only), "
                  "so there is no dx/dy to drive the mouse with.")
        elif look_stats is None:
            print("  --drive-mouse ignored: no look-stats sidecar, so dx/dy would be "
                  "standardized (not real device units) — refusing to drive the mouse "
                  "with uncalibrated values. Re-train to regenerate the sidecar.")
        else:
            actuate_mouse = True

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
    if actuate_mouse:
        print("  MOUSE: ARMED runs will ALSO move the view via the look head (dx/dy).")
        print("         Confirm you have run `python -m src.key_output --selftest-mouse`")
        print("         and the view actually turned — otherwise disarm and check that")
        print("         first. Disarm (F9) or quit (F8) stops all output instantly.")
    elif has_look:
        print("  MOUSE: look head dx/dy is DISPLAYED only (no mouse motion). Pass")
        print("         --drive-mouse to also actuate it (after --selftest-mouse passes).")
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
                pred = model.predict(x, verbose=0)

                # Unpack per output shape. A two-output (navigation-yaw) model
                # returns a LIST [move_keys, look]; the single-head baseline returns
                # one array. Use the resolved keys_idx/look_idx so ordering is not
                # assumed (D-036).
                if has_look:
                    probs = pred[keys_idx][0]                 # (n_keys,)
                    look_std_out = pred[look_idx][0]         # (2,) standardized
                    # Invert standardization to REAL device deltas when we have the
                    # sidecar stats; otherwise show the standardized value and say so.
                    if look_stats is not None:
                        look_dxdy = (look_std_out * look_stats["std"]
                                     + look_stats["mean"])
                    else:
                        look_dxdy = look_std_out
                else:
                    probs = pred[0]                          # (n_keys,)
                    look_dxdy = None

                desired = {k for k, p in zip(MOVEMENT_KEYS, probs) if p >= threshold}

                if armed:
                    keyctl.apply(desired)
                    # Navigation-yaw actuation (D-036, Piece 2). Only when the user
                    # opted in AND a calibrated look head is present (actuate_mouse).
                    # look_dxdy is already in REAL device units here (sidecar was
                    # required for actuate_mouse). mouse_move_relative clamps per
                    # call as a safety rail. Disarming (F9) stops this instantly
                    # because the whole block is gated on `armed`.
                    if actuate_mouse and look_dxdy is not None:
                        mouse_move_relative(look_dxdy[0], look_dxdy[1])

                # Live readout (throttled) so you can see decisions vs. actuation.
                if n % 5 == 0:
                    bar = "  ".join(f"{k}={p:.2f}" for k, p in zip(MOVEMENT_KEYS, probs))
                    held = "+".join(sorted(keyctl.held)) if keyctl.held else "-"
                    state = "ARMED" if armed else "disarmed"
                    if look_dxdy is not None:
                        unit = "" if look_stats is not None else " [std]"
                        look_str = (f"  look dx={look_dxdy[0]:+6.1f} "
                                    f"dy={look_dxdy[1]:+6.1f}{unit}")
                    else:
                        look_str = ""
                    print(f"  [{state}] {bar}   held[{held}]{look_str}   ",
                          end="\r", flush=True)

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
    p.add_argument("--drive-mouse", action="store_true",
                   help="ALSO move the view from the look head's dx/dy (navigation "
                        "yaw, D-036). Off by default (dx/dy is display-only). Only "
                        "use after `python -m src.key_output --selftest-mouse` "
                        "confirms injected mouse motion turns the CS2 view. Requires "
                        "a look model with its look-stats sidecar; ignored otherwise.")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    play(model_arg=args.model, crop=args.crop, threshold=args.threshold,
         drive_mouse=args.drive_mouse)


if __name__ == "__main__":
    main()
