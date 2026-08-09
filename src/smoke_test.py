"""Environment smoke test — Issue #1 acceptance ("smoke test imports core modules").

Run after creating the conda env:

    conda activate agentic-cs2
    python -m src.smoke_test

Verifies that every pinned dependency in environment.yml imports, reports the
version actually loaded against the version DECISIONS.md D-011 expects, and
reports whether TensorFlow sees the GPU. On this stack (TF 2.10 native Windows
+ CUDA 11.2, RTX 4050) the GPU SHOULD be detected — if it isn't, that's a setup
problem to fix, not the expected state (see D-011 / README GPU section).

Exit code 0 = all core imports succeeded; 1 = at least one failed.
"""

import importlib
import sys

# (import name, pip/conda name, version attribute, expected version per D-011)
CHECKS = [
    ("numpy", "numpy", "__version__", "1.26.4"),
    ("scipy", "scipy", "__version__", None),
    ("cv2", "opencv-python", "__version__", "4.10.0"),
    ("tensorflow", "tensorflow", "__version__", "2.10.1"),
    ("mss", "mss", "__version__", "7.0.1"),
    ("PIL", "pillow", "__version__", "9.5.0"),
    ("matplotlib", "matplotlib", "__version__", "3.7.5"),
]

# pywin32 is Windows-only and exposes no single version attr; check import only.
OPTIONAL_WIN = [("win32api", "pywin32")]


def _check(import_name, dist_name, version_attr, expected):
    try:
        mod = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - we want to report any failure
        return False, f"FAIL  {dist_name:<14} could not import ({import_name}): {exc}"
    actual = getattr(mod, version_attr, "?")
    # expected=None means "pin not fixed, any version is fine"; just report it.
    if expected is None:
        return True, f"ok    {dist_name:<14} {actual}"
    # OpenCV reports a 4-part version (e.g. 4.10.0.84); match on the prefix.
    matches = str(actual).startswith(expected)
    flag = "ok  " if matches else "warn"
    note = "" if matches else f"  (expected ~{expected})"
    return True, f"{flag}  {dist_name:<14} {actual}{note}"


def main():
    print(f"Agentic-CS2 smoke test — Python {sys.version.split()[0]}")
    print("-" * 60)

    all_ok = True
    for import_name, dist_name, version_attr, expected in CHECKS:
        ok, line = _check(import_name, dist_name, version_attr, expected)
        all_ok = all_ok and ok
        print(line)

    for import_name, dist_name in OPTIONAL_WIN:
        try:
            importlib.import_module(import_name)
            print(f"ok    {dist_name:<14} imported")
        except Exception as exc:  # noqa: BLE001
            print(f"note  {dist_name:<14} not available ({exc}) — Windows-only, "
                  "only needed for key output later")

    # GPU visibility. On this stack the GPU is expected to be present (that's the
    # whole reason for TF 2.10 + CUDA 11.2 on the 4050). No GPU = setup problem.
    gpu_ok = False
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu_ok = True
            print(f"\nok    GPU detected: {len(gpus)} device(s) — {[g.name for g in gpus]}")
        else:
            print("\nwarn  TensorFlow sees NO GPU. On this stack the 4050 should be "
                  "visible.\n      Check: current NVIDIA driver, CUDA 11.2 / cuDNN 8.1 "
                  "in the env, MSVC redist.\n      See D-011 / README GPU section.")
    except Exception as exc:  # noqa: BLE001
        print(f"\nwarn  Could not query TensorFlow devices: {exc}")

    print("-" * 60)
    if all_ok:
        gpu_note = "GPU visible." if gpu_ok else "but GPU NOT visible — see warning above."
        print(f"RESULT: all core imports succeeded. {gpu_note}")
        sys.exit(0)
    else:
        print("RESULT: at least one core import FAILED — env is not ready.")
        sys.exit(1)


if __name__ == "__main__":
    main()
