"""Environment smoke test — Issue #1 acceptance ("smoke test imports core modules").

Run after creating the conda env:

    conda activate agentic-cs2
    python -m src.smoke_test

Verifies that every pinned dependency in environment.yml imports, reports the
version actually loaded against the version DECISIONS.md D-009 expects, and
reports whether TensorFlow sees a GPU (expected: no GPU on modern cards under
the CUDA 10.1 stack — see the GPU CAVEAT in environment.yml).

Exit code 0 = all core imports succeeded; 1 = at least one failed.
"""

import importlib
import sys

# (import name, pip/conda name, version attribute, expected version per D-009)
CHECKS = [
    ("numpy", "numpy", "__version__", "1.18.5"),
    ("scipy", "scipy", "__version__", "1.4.1"),
    ("cv2", "opencv", "__version__", "4.4.0"),
    ("tensorflow", "tensorflow", "__version__", "2.3.0"),
    ("mss", "mss", "__version__", "7.0.1"),
    ("PIL", "pillow", "__version__", "9.3.0"),
    ("matplotlib", "matplotlib", "__version__", "3.5.3"),
]

# pywin32 is Windows-only and exposes no single version attr; check import only.
OPTIONAL_WIN = [("win32api", "pywin32")]


def _check(import_name, dist_name, version_attr, expected):
    try:
        mod = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - we want to report any failure
        return False, f"FAIL  {dist_name:<12} could not import ({import_name}): {exc}"
    actual = getattr(mod, version_attr, "?")
    flag = "ok " if actual == expected else "warn"
    note = "" if actual == expected else f"  (expected {expected})"
    return True, f"{flag}   {dist_name:<12} {actual}{note}"


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
            print(f"ok    {dist_name:<12} imported")
        except Exception as exc:  # noqa: BLE001
            print(f"note  {dist_name:<12} not available ({exc}) — Windows-only, "
                  "only needed for key output later")

    # GPU visibility — informational, never a failure (see GPU CAVEAT).
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"\nTensorFlow sees {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        else:
            print("\nTensorFlow sees no GPU — CPU-only. Expected on modern cards "
                  "under the CUDA 10.1 stack (see environment.yml GPU CAVEAT).")
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not query TensorFlow devices: {exc}")

    print("-" * 60)
    if all_ok:
        print("RESULT: all core imports succeeded.")
        sys.exit(0)
    else:
        print("RESULT: at least one core import FAILED — env is not ready.")
        sys.exit(1)


if __name__ == "__main__":
    main()
