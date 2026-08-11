"""test_data_loader.py — unit tests for the Issue #6 loader.

Runs without pytest (plain asserts + a runner) so it works in the pinned env
regardless of test deps. Builds tiny SYNTHETIC sessions in a temp dir in ALL
on-disk formats (v1 single .npz, v2 chunked folder, v3 chunked folder WITH a
radar array), then checks the loader's load-bearing contracts:

  1. THE SPLIT (D-021): whole-session, leak-free, deterministic, and stable when
     new sessions are added (existing assignments don't move).
  2. CRASHED SESSIONS (D-022): a crashed/incomplete session is excluded at
     discovery and never enters either split.
  3. THE LOADER (FPV): correct batch shapes for each FPV crop; action vector
     layout and values; alignment enforcement; v1/v2/v3 read identically for FPV.
  4. THE RADAR PATH (D-024): a v3 session serves the STORED radar as real
     (B,128,128,3) data; v1/v2 REFUSE radar (they have no radar array); a v3
     session with a mis-sized radar array is caught; a batch mixing a v3 and a
     v1/v2 session fails UP FRONT; get_batch never returns uninitialised rows.

Item 4 exists because a real scare (2026-08) turned on exactly these behaviours:
an un-updated inspector misreported a v3 session as v2, and it took a direct
diagnostic to prove the loader served real radar data. These tests make that
confusion impossible to repeat silently — the contracts are now asserted.

Run:  python -m src.test_data_loader
"""

import json
import os
import shutil
import tempfile

import numpy as np

from src import data_loader as dl


# ── synthetic session builders ───────────────────────────────────────────────

def _fake_arrays(n, seed, with_radar=False):
    """A dict of valid per-frame + metadata arrays, matching DATA_FORMAT.md.

    with_radar=True adds a (n, RADAR_H, RADAR_W, 3) `radar` array (v3, D-024).
    """
    rng = np.random.default_rng(seed)
    a = {
        "frames": rng.integers(0, 256, size=(n, dl.FRAME_H, dl.FRAME_W, 3),
                               dtype=np.uint8),
        "timestamps": np.cumsum(rng.uniform(0.05, 0.08, size=n)).astype(np.float64),
        "keys": rng.integers(0, 2, size=(n, 11), dtype=np.uint8),
        "key_names": np.array(["w", "a", "s", "d", "space", "ctrl", "shift",
                               "1", "2", "3", "r"]),
        "lclick": rng.integers(0, 2, size=n, dtype=np.uint8),
        "rclick": rng.integers(0, 2, size=n, dtype=np.uint8),
        "dx": rng.integers(-50, 51, size=n, dtype=np.int32),
        "dy": rng.integers(-50, 51, size=n, dtype=np.int32),
    }
    if with_radar:
        a["radar"] = rng.integers(0, 256, size=(n, dl.RADAR_H, dl.RADAR_W, 3),
                                  dtype=np.uint8)
    return a


def _write_v1(rec_dir, name, n, seed):
    a = _fake_arrays(n, seed)
    path = os.path.join(rec_dir, name + ".npz")
    np.savez_compressed(path, schema_version=np.array(1),
                        geom=np.array("synthetic"), loop_fps_target=np.array(15),
                        **a)
    return path


def _write_folder(rec_dir, name, chunk_sizes, seed, schema, with_radar):
    """Write a chunked session folder at the given schema (2 or 3)."""
    sdir = os.path.join(rec_dir, name)
    os.makedirs(sdir, exist_ok=True)
    chunks = []
    total = 0
    for i, n in enumerate(chunk_sizes):
        a = _fake_arrays(n, seed + i, with_radar=with_radar)
        cname = f"chunk_{i:05d}.npz"
        np.savez_compressed(os.path.join(sdir, cname),
                            schema_version=np.array(schema),
                            geom=np.array("synthetic"),
                            loop_fps_target=np.array(15), **a)
        chunks.append(cname)
        total += n
    manifest = {"schema_version": schema, "session": name, "geom": "synthetic",
                "loop_fps_target": 15, "chunks": chunks, "total_frames": total,
                "complete": True}
    with open(os.path.join(sdir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return sdir, total


def _write_v2(rec_dir, name, chunk_sizes, seed):
    return _write_folder(rec_dir, name, chunk_sizes, seed, schema=2, with_radar=False)


def _write_v3(rec_dir, name, chunk_sizes, seed):
    return _write_folder(rec_dir, name, chunk_sizes, seed, schema=3, with_radar=True)


def _write_crashed_v2(rec_dir, name, n=20, seed=1):
    """Write a CRASHED session: empty/incomplete manifest + a stray .tmp.npz.

    Mirrors exactly what a mid-write crash leaves on disk (D-018/D-019): the
    manifest lists no chunks and complete=false, and there's an un-renamed
    chunk_00000.npz.tmp.npz. This is the shape of the real session_20260810_151611.
    """
    sdir = os.path.join(rec_dir, name)
    os.makedirs(sdir, exist_ok=True)
    a = _fake_arrays(n, seed)
    np.savez_compressed(os.path.join(sdir, "chunk_00000.tmp.npz"),
                        schema_version=np.array(2), geom=np.array("synthetic"),
                        loop_fps_target=np.array(15), **a)
    manifest = {"schema_version": 2, "session": name, "geom": "synthetic",
                "loop_fps_target": 15, "chunks": [], "total_frames": 0,
                "complete": False}
    with open(os.path.join(sdir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return sdir


def _holdout_names(assignment):
    return {k for k, v in assignment.items() if v}


# ── tests: split ─────────────────────────────────────────────────────────────

def test_split_is_deterministic_and_leakfree():
    names = [f"session_2026080{i}_120000" for i in range(9)]
    frac = 0.34
    a = {n: dl.is_holdout(n, frac) for n in names}
    b = {n: dl.is_holdout(n, frac) for n in names}
    assert a == b, "split not deterministic across calls"
    holdo = _holdout_names(a)
    train = set(names) - holdo
    assert not (train & holdo), "a session is in both splits"
    assert len(train) + len(holdo) == len(names)


def test_split_stable_when_sessions_added():
    frac = 0.25
    original = [f"session_A_{i}" for i in range(6)]
    assign_before = {n: dl.is_holdout(n, frac) for n in original}
    added = original + [f"session_B_{i}" for i in range(6)]
    assign_after = {n: dl.is_holdout(n, frac) for n in added}
    for n in original:
        assert assign_before[n] == assign_after[n], (
            f"assignment for {n} changed when new sessions were added")


def test_manual_holdout_overrides_hash():
    frac = 0.20
    name = "session_forced"
    assert dl.is_holdout(name, frac, manual_holdout={name}) is True


# ── tests: crashed sessions (D-022) ──────────────────────────────────────────

def test_crashed_session_excluded_from_discovery():
    tmp = tempfile.mkdtemp()
    try:
        _write_v2(tmp, "session_good", [10], seed=1)
        _write_crashed_v2(tmp, "session_crashed", n=15, seed=2)
        found = dl.discover_sessions(rec_dir=tmp)
        names = {dl.session_name(p) for p in found}
        assert "session_good" in names, "good session was not discovered"
        assert "session_crashed" not in names, (
            "crashed session leaked into discovery — it would become a phantom "
            "split member")
        skipped = dl.list_skipped_sessions(rec_dir=tmp)
        skipped_names = {dl.session_name(p) for p, _ in skipped}
        assert "session_crashed" in skipped_names
    finally:
        shutil.rmtree(tmp)


def test_crashed_session_never_in_split():
    tmp = tempfile.mkdtemp()
    try:
        for i in range(5):
            _write_v2(tmp, f"session_ok_{i}", [8], seed=i)
        _write_crashed_v2(tmp, "session_crashed", n=15, seed=99)
        train_paths, hold_paths = dl.split_sessions(rec_dir=tmp, holdout_frac=0.4)
        all_names = {dl.session_name(p) for p in train_paths + hold_paths}
        assert "session_crashed" not in all_names, (
            "crashed session appeared in a split")
        assert all(f"session_ok_{i}" in all_names for i in range(5))
    finally:
        shutil.rmtree(tmp)


# ── tests: loader FPV path (v1/v2/v3) ────────────────────────────────────────

def test_v1_v2_v3_load_and_fpv_shapes():
    """v1 file, v2 folder, and v3 folder all load; FPV crops size correctly.

    v3 must load its extra `radar` array without disturbing the FPV arrays.
    """
    tmp = tempfile.mkdtemp()
    try:
        v1 = _write_v1(tmp, "session_v1", 10, seed=1)
        arrays_v1 = dl.load_session_arrays(v1)
        assert arrays_v1["frames"].shape == (10, dl.FRAME_H, dl.FRAME_W, 3)
        assert "radar" not in arrays_v1

        sdir2, total2 = _write_v2(tmp, "session_v2", [6, 4], seed=100)
        arrays_v2 = dl.load_session_arrays(sdir2)
        assert total2 == 10 and arrays_v2["frames"].shape[0] == 10
        assert "radar" not in arrays_v2

        sdir3, total3 = _write_v3(tmp, "session_v3", [6, 4], seed=200)
        arrays_v3 = dl.load_session_arrays(sdir3)
        assert total3 == 10 and arrays_v3["frames"].shape[0] == 10
        # v3's radar array loaded, concatenated across chunks, right shape:
        assert arrays_v3["radar"].shape == (10, dl.RADAR_H, dl.RADAR_W, 3)
        # every per-frame array (incl. radar) shares length:
        for k in ["frames", "radar", "timestamps", "keys", "lclick", "rclick",
                  "dx", "dy"]:
            assert arrays_v3[k].shape[0] == 10, f"{k} not length 10 after concat"

        # FPV crops on v1 (same code path for all schemas)
        ds = dl.SessionDataset([v1], crop="full")
        X, Y = ds.get_batch([0, 1, 2, 3])
        assert X.shape == (4, dl.FRAME_H, dl.FRAME_W, 3)
        assert Y.shape == (4, 15) and Y.dtype == np.float32

        ds_c = dl.SessionDataset([v1], crop="centre")
        _, _, h, w = dl.CENTRE_CROP_DEFAULT
        Xc, _ = ds_c.get_batch([0, 1])
        assert Xc.shape == (2, h, w, 3)
    finally:
        shutil.rmtree(tmp)


# ── tests: RADAR path (D-024) — the ones the 2026-08 scare motivated ──────────

def test_v3_serves_real_radar():
    """crop='radar' on a v3 session returns the STORED radar as (B,128,128,3),
    and the values MATCH the on-disk array (not uninitialised, not the FPV)."""
    tmp = tempfile.mkdtemp()
    try:
        sdir, _ = _write_v3(tmp, "session_v3r", [8], seed=7)
        ds = dl.SessionDataset([sdir], crop="radar")
        X, Y = ds.get_batch([0, 1, 2])
        assert X.shape == (3, dl.RADAR_H, dl.RADAR_W, 3)
        assert X.dtype == np.uint8
        assert Y.shape == (3, 15)
        # The served rows must equal the stored radar rows exactly.
        arrays = dl.load_session_arrays(sdir)
        for row, gi in enumerate([0, 1, 2]):
            assert np.array_equal(X[row], arrays["radar"][gi]), (
                f"served radar row {row} does not match stored radar[{gi}] — "
                f"the loader is not serving the real stored array")
    finally:
        shutil.rmtree(tmp)


def test_v1_and_v2_refuse_radar():
    """crop='radar' on v1 or v2 (no radar array) must RAISE, not fabricate."""
    tmp = tempfile.mkdtemp()
    try:
        v1 = _write_v1(tmp, "session_v1n", 6, seed=1)
        sdir2, _ = _write_v2(tmp, "session_v2n", [6], seed=2)
        for path, label in [(v1, "v1"), (sdir2, "v2")]:
            ds = dl.SessionDataset([path], crop="radar")
            raised = False
            try:
                ds.get_batch([0, 1])
            except ValueError:
                raised = True
            assert raised, (f"{label} session did NOT refuse radar — it must "
                            f"raise, never serve a fabricated radar batch")
    finally:
        shutil.rmtree(tmp)


def test_mixed_v2_v3_radar_batch_fails_up_front():
    """A batch spanning a v3 and a v2 session, crop='radar', must raise ONE clear
    error before allocation — not partially fill then fail."""
    tmp = tempfile.mkdtemp()
    try:
        sdir3, _ = _write_v3(tmp, "session_mix_v3", [5], seed=10)
        sdir2, _ = _write_v2(tmp, "session_mix_v2", [5], seed=20)
        # Hand both to ONE dataset so a batch can span both sessions.
        ds = dl.SessionDataset([sdir3, sdir2], crop="radar")
        # Global indices 0..4 -> v3 session; 5..9 -> v2 session. A batch covering
        # both must fail because the v2 half has no radar.
        raised = False
        try:
            ds.get_batch([0, 5])
        except ValueError as e:
            raised = True
            assert "radar" in str(e).lower()
        assert raised, ("mixed v3+v2 radar batch did not raise — the up-front "
                        "availability check is not working")
    finally:
        shutil.rmtree(tmp)


def test_v3_mis_sized_radar_is_rejected():
    """A v3 session whose radar array length disagrees with frames must be caught
    by the alignment assertion, not served."""
    tmp = tempfile.mkdtemp()
    try:
        # Build a v3 chunk by hand with radar shorter than frames.
        sdir = os.path.join(tmp, "session_badradar")
        os.makedirs(sdir, exist_ok=True)
        a = _fake_arrays(10, seed=1, with_radar=True)
        a["radar"] = a["radar"][:8]  # break alignment: 8 radar vs 10 frames
        np.savez_compressed(os.path.join(sdir, "chunk_00000.npz"),
                            schema_version=np.array(3), geom=np.array("synthetic"),
                            loop_fps_target=np.array(15), **a)
        manifest = {"schema_version": 3, "session": "session_badradar",
                    "geom": "synthetic", "loop_fps_target": 15,
                    "chunks": ["chunk_00000.npz"], "total_frames": 10,
                    "complete": True}
        with open(os.path.join(sdir, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        raised = False
        try:
            dl.load_session_arrays(sdir)
        except ValueError:
            raised = True
        assert raised, "mis-sized radar array was not caught by alignment check"
    finally:
        shutil.rmtree(tmp)


def test_get_batch_never_returns_uninitialised_rows():
    """Sanity on the fill-and-assert guard: a normal batch has all rows written,
    and (indirectly) no row retains the 255 sentinel by accident.

    We can't easily force an unwritten row without a bug, so this checks the
    positive path: every row of a v3 radar batch equals real stored data (so the
    sentinel-fill is fully overwritten)."""
    tmp = tempfile.mkdtemp()
    try:
        sdir, _ = _write_v3(tmp, "session_fill", [12], seed=5)
        ds = dl.SessionDataset([sdir], crop="radar")
        arrays = dl.load_session_arrays(sdir)
        req = [11, 0, 7, 3]
        X, _ = ds.get_batch(req)
        for row, gi in enumerate(req):
            assert np.array_equal(X[row], arrays["radar"][gi]), (
                f"row {row} not overwritten with real data (sentinel leak?)")
    finally:
        shutil.rmtree(tmp)


# ── tests: action vector, ordering, alignment, split disjointness ────────────

def test_action_vector_values_and_layout():
    tmp = tempfile.mkdtemp()
    try:
        v1 = _write_v1(tmp, "session_act", 5, seed=7)
        arrays = dl.load_session_arrays(v1)
        layout = dl.action_layout(arrays)
        assert layout == ["w", "a", "s", "d", "space", "ctrl", "shift",
                          "1", "2", "3", "r", "lclick", "rclick", "dx", "dy"]
        Y = dl.assemble_action_vector(arrays, np.array([2, 3]))
        assert Y.shape == (2, 15)
        i = 2
        expected = np.concatenate([
            arrays["keys"][i], [arrays["lclick"][i]], [arrays["rclick"][i]],
            [arrays["dx"][i]], [arrays["dy"][i]]]).astype(np.float32)
        assert np.array_equal(Y[0], expected)
    finally:
        shutil.rmtree(tmp)


def test_batch_row_order_matches_request():
    tmp = tempfile.mkdtemp()
    try:
        v1 = _write_v1(tmp, "session_order", 20, seed=3)
        ds = dl.SessionDataset([v1], crop="full")
        req = [5, 0, 19, 12]
        X, _ = ds.get_batch(req)
        arrays = dl.load_session_arrays(v1)
        for row, gi in enumerate(req):
            assert np.array_equal(X[row], arrays["frames"][gi]), (
                f"row {row} is not the requested frame {gi}")
    finally:
        shutil.rmtree(tmp)


def test_misaligned_session_is_rejected():
    tmp = tempfile.mkdtemp()
    try:
        a = _fake_arrays(10, seed=1)
        a["dx"] = a["dx"][:8]  # break alignment deliberately
        path = os.path.join(tmp, "session_bad.npz")
        np.savez_compressed(path, schema_version=np.array(1),
                            geom=np.array("synthetic"),
                            loop_fps_target=np.array(15), **a)
        raised = False
        try:
            dl.load_session_arrays(path)
        except ValueError:
            raised = True
        assert raised, "misaligned session was not rejected"
    finally:
        shutil.rmtree(tmp)


def test_build_datasets_disjoint():
    tmp = tempfile.mkdtemp()
    try:
        for i in range(8):
            _write_v1(tmp, f"session_split_{i}", 5, seed=i)
        train_ds, hold_ds = dl.build_datasets(rec_dir=tmp, holdout_frac=0.4)
        train_names = set(train_ds.names())
        hold_names = set(hold_ds.names())
        assert not (train_names & hold_names), "session appears in both splits"
        assert len(train_names) + len(hold_names) == 8
    finally:
        shutil.rmtree(tmp)


def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {e!r}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
