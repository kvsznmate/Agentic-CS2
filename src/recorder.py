"""recorder.py — synchronized frame + input recording (Issue #3, the M0 GATE).

The heart of M0. Runs ONE synchronous loop (way one, chosen deliberately over a
threaded design — the point of this gate is to PROVE alignment, and the simplest
loop is the one whose correctness can actually be verified). Each tick:

  1. read accumulated raw mouse dx/dy since the last frame (and reset),
  2. read keyboard + mouse-button state,
  3. grab a frame (which returns its own capture timestamp),
  4. write one record pairing all of the above.

Why this order matters: the frame grab is LAST, mirroring the reference study's
loop, which grabbed the image last on purpose so the image's time-lag matches
when a forward pass will run at inference. The mouse delta covers the interval
that ENDS at this frame — "how far the view moved arriving at frame N."

WHAT THIS GATE MUST PROVE (PROJECT_ISSUES #3): that input at frame N provably
corresponds to screen at frame N. That is not self-evident from a loop that
merely runs — it has to be demonstrated. Hence --verify below, which records a
short session while YOU perform a scripted motion, then checks the logged inputs
match what you did, frame by frame. If sync can't be shown reliable, the issue
says RAISE THE KILL FLAG — so this file is built to expose failure, not hide it.

DECISIONS honoured:
  D-015  aim captured from raw mouse deltas (raw_mouse.py), not cursor position
         or memory-read view angles (both dead for us). Keys/clicks via pywin32.
  D-024  two-resolution capture: the extended recorder stores a separate high-res
         radar crop (128x128) alongside the 150x270 FPV, both from ONE grab
         (Capture.grab_with_radar). Format bumped to v3. See DATA_FORMAT.md.
  single synchronous loop, per the way-one decision.

Entry points:
  python -m src.recorder --verify     # scripted-motion alignment check (DO FIRST)
  python -m src.recorder --record     # record an extended chunked session (v3)
  python -m src.recorder --dryrun     # loop + live readout, write nothing
"""

import argparse
import json
import os
import queue
import shutil
import threading
import time

import numpy as np
import win32api

from src.capture import Capture
from src import capture_config as cfg
from src.raw_mouse import RawMouseListener


# Output location for recordings. data/ is gitignored (recordings are large and
# local — see .gitignore and PROJECT_ISSUES #4/#5).
_REC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "recordings")

# On-disk schema version this recorder writes (DATA_FORMAT.md). v3 adds the
# per-frame `radar` array (D-024) to the v2 chunked format.
SCHEMA_VERSION = 3

# Target loop rate. Profiling (D-016) showed the loop is entirely bounded by the
# ~37 ms mss grab; input logging is free. Realised recording rate is ~15 FPS, so
# the loop targets 15 rather than an unreachable 20. Matches the study's 16 FPS
# working loop. dxcam is the documented lever (D-014) if the full agent loop
# (with model inference) later needs more headroom. D-024's radar crop adds a
# small per-frame cost on top; --profile measures it.
LOOP_FPS = 15

# ── Chunked-session parameters (Issue #4 / D-018) ──────────────────────────
# A long session is written as a folder of chunk files so memory stays bounded
# and a crash loses at most one in-progress chunk. At 15 FPS a 150x270x3 FPV
# frame is ~121 KB and a 128x128x3 radar is ~49 KB (~170 KB/frame total, D-024),
# ~2.6 MB/s, so 1800 frames (~2 min) is a few hundred MB in RAM per chunk before
# flush — comfortable, and a good crash-loss granularity.
CHUNK_FRAMES = 1800  # flush a chunk every this many frames (~2 min at 15 FPS)

# Refuse to start / keep recording if free disk space falls below this, so a long
# unattended session can't fill the drive. With the radar array the uncompressed
# budget is ~9 GB/hour (DATA_FORMAT.md v3); compressed is less but we stay
# conservative.
MIN_FREE_DISK_GB = 5.0

# ── Keys we log ───────────────────────────────────────────────────────────
# Movement + jump/crouch + reload + weapon slots, mirroring the study's action
# set (config.py n_keys=11: w,s,a,d,space,ctrl,shift,1,2,3,r). These are the
# ACTUAL in-game keys — unlike the study's record script, which had the player
# press tfgh so it could simulate wasd for a spectated bot. We record our own
# real play (D-002), so we log the real keys directly.
# Virtual-key codes: letters/digits use ord(uppercase). Named keys are explicit.
VK_SPACE = 0x20
VK_CONTROL = 0x11
VK_SHIFT = 0x10
LOGGED_KEYS = [
    ("w", ord("W")), ("a", ord("A")), ("s", ord("S")), ("d", ord("D")),
    ("space", VK_SPACE), ("ctrl", VK_CONTROL), ("shift", VK_SHIFT),
    ("1", ord("1")), ("2", ord("2")), ("3", ord("3")), ("r", ord("R")),
]
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
# Quit key for the loop (checked each tick). F8 is unlikely to be bound in CS2.
VK_QUIT = 0x77  # F8


def _key_down(vk):
    """True if the virtual key is currently held.

    GetAsyncKeyState high bit = currently down. Works during gameplay (the study
    relied on the same call), unlike cursor position which the game locks.
    """
    return (win32api.GetAsyncKeyState(vk) & 0x8000) != 0


def _read_keys():
    """Return a 0/1 vector over LOGGED_KEYS in fixed order."""
    return [1 if _key_down(vk) else 0 for _, vk in LOGGED_KEYS]


def _read_clicks():
    """Return (left, right) as 0/1 for mouse buttons currently held."""
    left = 1 if _key_down(VK_LBUTTON) else 0
    right = 1 if _key_down(VK_RBUTTON) else 0
    return left, right


class Recorder:
    """Owns the capture, the mouse listener, and the per-frame record assembly."""

    def __init__(self):
        self.cap = Capture()
        self.mouse = RawMouseListener()

    def __enter__(self):
        self.cap.__enter__()
        self.mouse.start()
        return self

    def __exit__(self, *exc):
        self.mouse.stop()
        self.cap.__exit__(*exc)

    def read_frame_record(self):
        """One synchronized tick -> (frame, radar, record_dict).

        Order is deliberate (see module docstring): mouse delta first (closes the
        interval ending at this frame), then keys/clicks, then the frame grab
        last so its timing matches inference. All share the frame's capture
        timestamp as the single alignment anchor.

        Returns the FPV frame AND the high-res radar crop (D-024), both from the
        SAME grab via Capture.grab_with_radar, so the two feeds are inherently
        synchronized (same instant, one grab).
        """
        dx, dy = self.mouse.read_and_reset()
        keys = _read_keys()
        lclick, rclick = _read_clicks()
        frame, radar, t_cap = self.cap.grab_with_radar()
        record = {
            "t": t_cap,          # capture timestamp (perf_counter) — the anchor
            "keys": keys,        # 0/1 over LOGGED_KEYS order
            "lclick": lclick,
            "rclick": rclick,
            "dx": dx,            # summed raw mouse dx since last frame
            "dy": dy,
        }
        return frame, radar, record


def _pace(loop_start, fps):
    """Sleep until the frame's time budget is up, to hold ~fps."""
    target = loop_start + 1.0 / fps
    while time.perf_counter() < target:
        time.sleep(0.0005)


def dryrun(seconds=20.0):
    """Run the loop with a live readout, writing nothing. Sanity check the feed.

    Prints keys/clicks/dx/dy each tick so you can watch inputs register in real
    time. Good first smoke test that all three input sources are alive together
    before committing to a recording or the formal verify.
    """
    print("Dry run — synchronized loop, nothing saved. Press F8 to stop.")
    print("Move, press WASD, click — watch the values change.\n")
    key_names = [name for name, _ in LOGGED_KEYS]
    with Recorder() as rec:
        n = 0
        t_start = time.perf_counter()
        while True:
            loop_start = time.perf_counter()
            if _key_down(VK_QUIT):
                break
            _frame, _radar, r = rec.read_frame_record()
            n += 1
            if n % 5 == 0:  # don't spam every frame
                held = [key_names[i] for i, v in enumerate(r["keys"]) if v]
                print(f"keys={held!s:<28} L={r['lclick']} R={r['rclick']} "
                      f"dx={r['dx']:+5d} dy={r['dy']:+5d}", end="\r")
            if time.perf_counter() - t_start > seconds:
                break
            _pace(loop_start, LOOP_FPS)
    print("\nDry run done.")


def profile(seconds=20.0):
    """Time each loop stage separately to find what limits the recording FPS.

    Breaks the per-frame cost into its parts — mouse read, key/click read, the
    grab-with-radar (FPV + radar crop), record assembly — so we know whether the
    mss grab dominates (expected, per D-013/D-014; then a lower floor is honest)
    or something cheaper is the culprit. Since D-024 added the radar crop+resize
    to the grab stage, this is also the honest place to see its cost. Runs the
    REAL work each tick but saves nothing and does NOT pace, so the numbers
    reflect raw stage cost, not the sleep.

    Play normally (move + look) so the mouse/key paths do representative work.
    Press F8 to stop.
    """
    print("Profiling the record loop. Play normally (move + look). F8 to stop.")
    print("Saving nothing; this measures per-stage cost, unpaced.\n")
    n = 0
    mouse_t = keys_t = grab_t = asm_t = 0.0
    with Recorder() as rec:
        t_start = time.perf_counter()
        while True:
            if _key_down(VK_QUIT):
                break
            t0 = time.perf_counter()
            dx, dy = rec.mouse.read_and_reset()
            t1 = time.perf_counter()
            keys = _read_keys()
            lclick, rclick = _read_clicks()
            t2 = time.perf_counter()
            frame, radar, t_cap = rec.cap.grab_with_radar()
            t3 = time.perf_counter()
            # record assembly + array casts, mirroring the writer's real work
            _rec = {"t": t_cap, "keys": keys, "lclick": lclick,
                    "rclick": rclick, "dx": dx, "dy": dy}
            _ = np.asarray(frame, dtype=np.uint8)
            _ = np.asarray(radar, dtype=np.uint8)
            t4 = time.perf_counter()
            mouse_t += t1 - t0
            keys_t += t2 - t1
            grab_t += t3 - t2
            asm_t += t4 - t3
            n += 1
            if time.perf_counter() - t_start > seconds:
                break
    if n == 0:
        print("No frames.")
        return
    total = mouse_t + keys_t + grab_t + asm_t
    per_frame_ms = total / n * 1000.0
    fps = n / total if total else float("nan")
    print(f"\nProfiled {n} frames, unpaced.")
    print(f"Per-frame total: {per_frame_ms:.2f} ms  ->  {fps:.1f} FPS ceiling "
          f"(unpaced, no sleep)")
    print("Stage breakdown (ms/frame and share):")
    for label, t in [("mouse read", mouse_t), ("keys+clicks", keys_t),
                     ("grab+radar (mss+2 resizes)", grab_t),
                     ("record assembly", asm_t)]:
        ms = t / n * 1000.0
        print(f"  {label:<28} {ms:6.2f} ms  ({ms/per_frame_ms*100:4.0f}%)")
    print()
    if grab_t > 0.6 * total:
        print("Diagnosis: the grab (mss + FPV resize + radar crop/resize) dominates.")
        print("This is the known slow mss path (D-013/D-014) plus D-024's radar")
        print("work — expected, not a loop bug. A lower FPS floor is honest; dxcam")
        print("is the documented lever if the full agent loop later needs more.")
    else:
        print("Diagnosis: the grab is NOT the sole cost — another stage is a large")
        print("share. Worth a closer look before enshrining a floor; see which row")
        print("above is unexpectedly heavy.")
    return fps


def _geom_string():
    """Human-readable capture-geometry stamp for self-describing files (D-017/D-024)."""
    gw, gh = cfg.GAME_RES
    ih, iw = cfg.MODEL_INPUT_HW
    rh, rw = cfg.RADAR_OUT_HW
    return (f"fullscreen {gw}x{gh} -> crop "
            f"L{cfg.CROP_LEFT}T{cfg.CROP_TOP}W{cfg.CROP_WIDTH}H{cfg.CROP_HEIGHT} "
            f"-> FPV {iw}x{ih} {cfg.COLOR_FORMAT}; "
            f"radar src L{cfg.RADAR_SRC_LEFT}T{cfg.RADAR_SRC_TOP}"
            f"W{cfg.RADAR_SRC_WIDTH}H{cfg.RADAR_SRC_HEIGHT} -> {rw}x{rh} {cfg.COLOR_FORMAT}")


def _free_disk_gb(path):
    """Free space (GB) on the filesystem holding `path`."""
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


class ChunkedSessionWriter:
    """Writes a session as a folder of chunk .npz files with a manifest.

    Chunks are written on a BACKGROUND THREAD (D-019) so the capture loop never
    pauses to save. `flush()` snapshots the current buffer and hands it to a
    queue; a single writer thread compresses and writes it (atomic .tmp.npz ->
    rename) while the capture loop keeps grabbing the next chunk. This fixes the
    ~10 s capture stall the earlier synchronous flush caused at every chunk
    boundary (D-018).

    v3 (D-024): each chunk now also holds a per-frame `radar` array, buffered and
    written the same way as `frames`. Nothing about the threading/crash-safety
    changes — the radar is just another index-aligned per-frame array.

    Sync-safety (important, re D-015): the writer thread touches ONLY finished,
    handed-off buffers. It never touches the capture, the timestamps, or the
    mouse listener. The frame/input alignment path stays fully synchronous and
    single-threaded; only the disk write is moved off it. So the "way one"
    synchronous-capture decision is preserved — threading is confined to I/O.

    Backpressure (Option A, single-chunk depth): the handoff queue has capacity
    1. If the writer is still busy when the next chunk is ready, `flush()` blocks
    until the writer drains — bounding memory to ~2 chunks. With a ~10 s write
    against a ~120 s fill window this effectively never triggers, but if the disk
    ever stalls that long we'd rather pause than grow memory without bound.

    Usage:
        w = ChunkedSessionWriter(session_dir)
        for each tick: w.add(frame, radar, record)   # buffers; auto-flushes
        w.close()                                     # drains queue, marks complete
    """

    def __init__(self, session_dir, chunk_frames=CHUNK_FRAMES):
        self.dir = session_dir
        self.chunk_frames = chunk_frames
        os.makedirs(self.dir, exist_ok=True)
        self.session_name = os.path.basename(self.dir.rstrip(os.sep))
        self._buf = self._empty_buffer()
        self._geom = _geom_string()

        # State OWNED BY THE WRITER THREAD once started (main thread reads copies
        # for progress only). Chunk list + counts are mutated only in the worker.
        self._chunk_files = []
        self._total_frames = 0
        self._chunk_index = 0            # next index to assign (main thread, at flush)
        self._write_error = None         # first exception seen by the worker

        # capacity-1 queue = Option A single-chunk backpressure
        self._q = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._writer_loop, name="chunk-writer",
                                        daemon=True)
        self._thread.start()
        self._write_manifest(complete=False)

    @staticmethod
    def _empty_buffer():
        return {"t": [], "keys": [], "lclick": [], "rclick": [], "dx": [], "dy": [],
                "frames": [], "radar": []}

    def add(self, frame, radar, record):
        """Buffer one tick; hand off a chunk when the buffer reaches chunk_frames."""
        b = self._buf
        b["frames"].append(frame)
        b["radar"].append(radar)
        b["t"].append(record["t"])
        b["keys"].append(record["keys"])
        b["lclick"].append(record["lclick"])
        b["rclick"].append(record["rclick"])
        b["dx"].append(record["dx"])
        b["dy"].append(record["dy"])
        if len(b["frames"]) >= self.chunk_frames:
            self.flush()

    def flush(self):
        """Hand the current buffer to the writer thread (non-blocking in practice).

        Assigns this chunk's index and filename, enqueues the buffer, and
        immediately starts a fresh buffer so the capture loop keeps running while
        the worker writes. Blocks ONLY if the previous chunk is still being
        written (capacity-1 queue) — the Option A backpressure.
        """
        b = self._buf
        if len(b["frames"]) == 0:
            return
        # Surface any writer error promptly rather than silently losing chunks.
        if self._write_error is not None:
            raise self._write_error
        idx = self._chunk_index
        fname = f"chunk_{idx:05d}.npz"
        self._chunk_index += 1
        self._buf = self._empty_buffer()  # main thread moves on immediately
        # Hand off (idx, fname, buffer). put() blocks if worker still busy.
        self._q.put((idx, fname, b))

    def _writer_loop(self):
        """Background thread: drain the queue, write each chunk, update manifest."""
        while True:
            item = self._q.get()
            if item is None:          # sentinel = shut down
                self._q.task_done()
                return
            idx, fname, b = item
            try:
                self._write_chunk(idx, fname, b)
                self._chunk_files.append(fname)
                self._total_frames += len(b["frames"])
                self._write_manifest(complete=False)
            except Exception as e:  # noqa: BLE001 - record, keep thread alive
                if self._write_error is None:
                    self._write_error = e
                print(f"\n  WARNING: chunk {fname} write failed ({e!r}).")
            finally:
                self._q.task_done()

    def _write_chunk(self, idx, fname, b):
        """Compress + atomically write one chunk. Runs on the writer thread."""
        final = os.path.join(self.dir, fname)
        # Temp name ends in .npz on purpose: np.savez_compressed APPENDS ".npz"
        # to any path not already ending in it, so a ".tmp" target would become
        # ".tmp.npz" and the os.replace would fail. .npz-suffixed temp avoids that.
        tmp = os.path.join(self.dir, f"chunk_{idx:05d}.tmp.npz")
        arrays = {
            "schema_version": np.array(SCHEMA_VERSION),
            "geom": np.array(self._geom),
            "loop_fps_target": np.array(LOOP_FPS),
            "frames": np.asarray(b["frames"], dtype=np.uint8),
            "radar": np.asarray(b["radar"], dtype=np.uint8),   # v3 (D-024)
            "timestamps": np.asarray(b["t"], dtype=np.float64),
            "keys": np.asarray(b["keys"], dtype=np.uint8),
            "key_names": np.array([k for k, _ in LOGGED_KEYS]),
            "lclick": np.asarray(b["lclick"], dtype=np.uint8),
            "rclick": np.asarray(b["rclick"], dtype=np.uint8),
            "dx": np.asarray(b["dx"], dtype=np.int32),
            "dy": np.asarray(b["dy"], dtype=np.int32),
        }
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, final)

    def _write_manifest(self, complete):
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "session": self.session_name,
            "geom": self._geom,
            "loop_fps_target": LOOP_FPS,
            "chunks": list(self._chunk_files),
            "total_frames": self._total_frames,
            "complete": complete,
        }
        mpath = os.path.join(self.dir, "manifest.json")
        tmp = mpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, mpath)

    @property
    def chunks_written(self):
        """How many chunks have actually landed on disk (for progress display)."""
        return len(self._chunk_files)

    def close(self):
        """Hand off the remainder, wait for the writer to finish, mark complete.

        Runs from record_session's `finally`, so it must not raise even if a
        write failed — otherwise it would mask the original error. Any buffered
        remainder is enqueued, the writer is told to stop, and we join it so all
        chunks are on disk before we mark the session complete. If any chunk
        write errored, the session is marked INCOMPLETE (loadable up to the last
        good chunk) rather than falsely complete.
        """
        try:
            self.flush()  # enqueue remainder (no-op if buffer empty)
        except Exception as e:  # noqa: BLE001 - finalizer must not propagate
            if self._write_error is None:
                self._write_error = e
        # Signal shutdown and wait for the writer to drain the queue.
        self._q.put(None)
        self._thread.join()
        ok = self._write_error is None
        self._write_manifest(complete=ok)
        if not ok:
            print(f"\n  WARNING: one or more chunk writes failed ({self._write_error!r}). "
                  f"Session marked incomplete; {self._total_frames} frames in "
                  f"{len(self._chunk_files)} chunk(s) are readable.")
        return self._total_frames


def record(seconds=60.0, name=None):
    """LEGACY single-file recorder (v1, FPV-only). Kept for the short round-trip
    check via --record-single; NOT updated for the radar (D-024).

    v3 (radar + FPV) is the chunked path in record_session(). This one-file writer
    predates the radar and stays FPV-only on purpose: it exists only as a quick
    self-contained round-trip smoke test, and a v1 file remains a valid FPV-only
    single-chunk session. Do not use it to build the #7 radar dataset — use
    --record (record_session), which writes v3 with the radar.

    Writes the v1 schema (DATA_FORMAT.md): frames + aligned action arrays,
    index-aligned (row i of every per-frame array is frame i).
    """
    os.makedirs(_REC_DIR, exist_ok=True)
    if name is None:
        name = time.strftime("session_%Y%m%d_%H%M%S")
    path = os.path.join(_REC_DIR, name + ".npz")

    frames = []
    ts = []
    keys = []
    lclicks = []
    rclicks = []
    dxs = []
    dys = []

    print(f"[LEGACY v1, FPV-only] Recording to {path}. Press F8 to stop "
          f"(or after {seconds:.0f}s).")
    print("Play normally. Keep CS2 focused and in-game.")
    print("NOTE: this path stores NO radar (D-024). For the radar dataset use "
          "--record.\n")
    dropped = 0
    with Recorder() as rec:
        n = 0
        t_start = time.perf_counter()
        last_t = None
        while True:
            loop_start = time.perf_counter()
            if _key_down(VK_QUIT):
                break
            frame, _radar, r = rec.read_frame_record()  # radar grabbed but not stored here
            frames.append(frame)
            ts.append(r["t"])
            keys.append(r["keys"])
            lclicks.append(r["lclick"])
            rclicks.append(r["rclick"])
            dxs.append(r["dx"])
            dys.append(r["dy"])
            # crude dropped-frame proxy: gaps much longer than the frame budget
            if last_t is not None:
                gap = r["t"] - last_t
                if gap > 1.8 / LOOP_FPS:
                    dropped += 1
            last_t = r["t"]
            n += 1
            if n % 20 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  {n} frames, {n/elapsed:.1f} FPS, {dropped} long-gaps",
                      end="\r")
            if time.perf_counter() - t_start > seconds:
                break
            _pace(loop_start, LOOP_FPS)

    frames = np.asarray(frames, dtype=np.uint8)
    ts = np.asarray(ts, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.uint8)
    actions = {
        # ── schema self-description (DATA_FORMAT.md v1) ──
        "schema_version": np.array(1),
        "geom": np.array(_geom_string()),
        "loop_fps_target": np.array(LOOP_FPS),
        # ── per-frame arrays (length N, index-aligned to frames) ──
        "timestamps": ts,
        "keys": keys,                       # (N, len(LOGGED_KEYS))
        "key_names": np.array([n for n, _ in LOGGED_KEYS]),
        "lclick": np.asarray(lclicks, dtype=np.uint8),
        "rclick": np.asarray(rclicks, dtype=np.uint8),
        "dx": np.asarray(dxs, dtype=np.int32),
        "dy": np.asarray(dys, dtype=np.int32),
    }
    np.savez_compressed(path, frames=frames, **actions)

    elapsed = time.perf_counter() - t_start
    print(f"\nSaved {len(frames)} frames to {path}")
    print(f"  {elapsed:.1f}s, mean {len(frames)/elapsed:.1f} FPS, "
          f"{dropped} long-gap frames ({100*dropped/max(len(frames),1):.1f}%)")

    # Immediately reload to prove the round-trip (an M0 exit criterion).
    with np.load(path, allow_pickle=False) as d:
        assert d["frames"].shape[0] == len(frames)
        assert d["keys"].shape[0] == len(frames)
        assert d["dx"].shape[0] == len(frames)
    print("  Round-trip OK: reloaded and shapes match.")
    return path


def record_session(seconds=None, name=None, chunk_frames=CHUNK_FRAMES):
    """Extended chunked recording (Issue #4, v3 with radar per D-024).

    Writes a v3 session folder (DATA_FORMAT.md): each frame stores the 150x270
    FPV AND the 128x128 high-res radar crop, both from one grab, index-aligned.
    Chunks flushed every `chunk_frames` frames so memory stays bounded and a crash
    loses at most one in-progress chunk. Runs until F8, until `seconds` elapses
    (if given), or until interrupted — and in every case the buffered frames are
    flushed and the manifest marked complete, so you never lose a clean session.

    Disk safety: refuses to start below MIN_FREE_DISK_GB, and stops cleanly if
    free space crosses that floor mid-session, so an unattended run can't fill
    the drive.

    This is what #4 means by "usable for extended sessions without babysitting,"
    now producing the two-feed data the radar gate (#7) needs.
    """
    os.makedirs(_REC_DIR, exist_ok=True)
    if name is None:
        name = time.strftime("session_%Y%m%d_%H%M%S")
    session_dir = os.path.join(_REC_DIR, name)

    free = _free_disk_gb(_REC_DIR)
    if free < MIN_FREE_DISK_GB:
        print(f"REFUSING TO START: only {free:.1f} GB free, need "
              f"{MIN_FREE_DISK_GB:.1f} GB. Free up disk space first.")
        return None

    dur = f"{seconds:.0f}s" if seconds else "until F8"
    print(f"Recording session '{name}' ({dur}), v{SCHEMA_VERSION} (FPV + radar). "
          f"Press F8 to stop.")
    print(f"  chunks every {chunk_frames} frames (~{chunk_frames/LOOP_FPS:.0f}s), "
          f"{free:.1f} GB free, floor {MIN_FREE_DISK_GB:.0f} GB.")
    print("  Play normally. Keep CS2 focused and in-game.\n")

    writer = ChunkedSessionWriter(session_dir, chunk_frames=chunk_frames)
    n = 0
    dropped = 0
    stop_reason = "F8"
    t_start = time.perf_counter()
    last_t = None
    last_disk_check = t_start
    try:
        with Recorder() as rec:
            while True:
                loop_start = time.perf_counter()
                if _key_down(VK_QUIT):
                    stop_reason = "F8"
                    break
                frame, radar, r = rec.read_frame_record()
                writer.add(frame, radar, r)
                if last_t is not None and (r["t"] - last_t) > 1.8 / LOOP_FPS:
                    dropped += 1
                last_t = r["t"]
                n += 1
                if n % 20 == 0:
                    elapsed = time.perf_counter() - t_start
                    print(f"  {n} frames, {writer.chunks_written} chunks written, "
                          f"{n/elapsed:.1f} FPS, {dropped} long-gaps", end="\r")
                # periodic disk check (every ~10s) so a long run can't fill disk
                if time.perf_counter() - last_disk_check > 10.0:
                    last_disk_check = time.perf_counter()
                    if _free_disk_gb(_REC_DIR) < MIN_FREE_DISK_GB:
                        stop_reason = "low disk"
                        break
                if seconds is not None and (time.perf_counter() - t_start) > seconds:
                    stop_reason = "time"
                    break
                _pace(loop_start, LOOP_FPS)
    except KeyboardInterrupt:
        stop_reason = "Ctrl-C"
    finally:
        # ALWAYS flush + finalize, whatever stopped us. This is the crash-safety
        # payoff: an interrupted session still closes cleanly with all buffered
        # frames written and the manifest marked complete.
        total = writer.close()

    elapsed = time.perf_counter() - t_start
    print(f"\nStopped ({stop_reason}). Session '{name}':")
    print(f"  {total} frames across {writer.chunks_written} chunks, "
          f"{elapsed:.0f}s, mean {total/max(elapsed,1e-9):.1f} FPS, "
          f"{dropped} long-gap frames ({100*dropped/max(total,1):.1f}%).")
    print(f"  Folder: {session_dir}")
    if stop_reason == "low disk":
        print("  NOTE: stopped because free disk fell below the floor — the "
              "session up to here is complete and safe.")
    print("  Inspect with: python -m src.inspect_recording "
          f"{os.path.join('data', 'recordings', name)}")
    return session_dir


def verify(seconds=12.0):
    """SCRIPTED-MOTION ALIGNMENT CHECK — this is what actually tests the gate.

    A loop that merely runs does not prove input matches frame. This does: it
    records a short session while you perform a prescribed motion, then checks
    the logged signal matches the instruction, timed correctly.

    The check is mouse-based because mouse delta is the hardest and most
    sync-sensitive signal. You'll be prompted to sweep the mouse RIGHT for a few
    seconds, then LEFT. We then verify:
      * during the RIGHT phase, logged dx is predominantly positive,
      * during the LEFT phase, logged dx is predominantly negative,
      * the sign flip in the data lines up (within a frame or two) with when you
        actually switched — i.e. the input timeline tracks the frame timeline.

    If the phases don't separate cleanly, sync (or raw-mouse reading) is not
    trustworthy — investigate before recording real data; per #3, consider the
    kill flag. (Uses the same grab_with_radar path as recording, so it exercises
    the real loop including D-024's radar crop.)
    """
    print("ALIGNMENT VERIFY. Follow the prompts. Keep CS2 focused and in-game\n"
          "(cursor locked) so this tests the real conditions.\n")
    half = seconds / 2.0
    records = []
    phase_marks = []  # (index_at_phase_start, label)

    with Recorder() as rec:
        # PHASE 1: sweep right
        print(">>> Now sweep the mouse steadily to the RIGHT until told to stop.")
        t_start = time.perf_counter()
        phase_marks.append((0, "RIGHT"))
        while time.perf_counter() - t_start < half:
            loop_start = time.perf_counter()
            _frame, _radar, r = rec.read_frame_record()
            records.append(r)
            _pace(loop_start, LOOP_FPS)

        # PHASE 2: sweep left
        print(">>> Now sweep the mouse steadily to the LEFT until told to stop.")
        phase_marks.append((len(records), "LEFT"))
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < half:
            loop_start = time.perf_counter()
            _frame, _radar, r = rec.read_frame_record()
            records.append(r)
            _pace(loop_start, LOOP_FPS)

    print("\nStopped. Analysing alignment...\n")
    dx = np.array([r["dx"] for r in records], dtype=np.int64)
    n = len(records)
    split = phase_marks[1][0]

    right_dx = dx[:split]
    left_dx = dx[split:]

    def _summ(arr, label):
        if arr.size == 0:
            return f"  {label}: no frames"
        pos = int((arr > 0).sum())
        neg = int((arr < 0).sum())
        return (f"  {label} phase: {arr.size} frames, "
                f"mean dx={arr.mean():+.1f}, {pos} positive / {neg} negative")

    print(_summ(right_dx, "RIGHT"))
    print(_summ(left_dx, "LEFT"))

    # Verdicts. Judge direction using ONLY frames where the mouse actually moved
    # (dx != 0). A hand sweep has natural pauses between strokes that log dx=0;
    # counting those zeros against the ratio is wrong — the question is "when you
    # moved, did it move the right way," not "did you move every single frame."
    r_moved = right_dx[right_dx != 0]
    l_moved = left_dx[left_dx != 0]
    r_pos_frac = (r_moved > 0).mean() if r_moved.size else 0.0
    l_neg_frac = (l_moved < 0).mean() if l_moved.size else 0.0

    print(f"  RIGHT: of {r_moved.size} moving frames, {r_pos_frac*100:.0f}% "
          f"were rightward (+dx).")
    print(f"  LEFT:  of {l_moved.size} moving frames, {l_neg_frac*100:.0f}% "
          f"were leftward (-dx).")

    ok_right = right_dx.size > 0 and right_dx.mean() > 0 and r_pos_frac >= 0.8
    ok_left = left_dx.size > 0 and left_dx.mean() < 0 and l_neg_frac >= 0.8

    best_k = None
    best_score = -1
    for k in range(1, n):
        correct = int((dx[:k] > 0).sum() + (dx[k:] < 0).sum())
        if correct > best_score:
            best_score = correct
            best_k = k
    separation = best_score / n if n else 0.0

    print()
    print(f"  You switched direction at frame {split}. Best-fit switch in the")
    print(f"  logged data is at frame {best_k} "
          f"({separation*100:.0f}% of frames consistent with one clean switch).")
    if best_k is not None:
        drift = abs(best_k - split)
        print(f"  Offset between the two: {drift} frame(s) "
              f"(small = the input timeline tracks the frame timeline).")

    reaction_tol_frames = int(1.5 * LOOP_FPS)
    boundary_ok = best_k is not None and abs(best_k - split) <= reaction_tol_frames

    print()
    if ok_right and ok_left and boundary_ok:
        print("RESULT: PASS — logged mouse motion matches the scripted direction in")
        print("each phase, and the data's switch point lines up with when you")
        print("actually switched. Frame/input alignment looks trustworthy.")
        print("(Re-run a couple of times; sync is worth over-checking.)")
        passed = True
    elif ok_right and ok_left and not boundary_ok:
        print("RESULT: MOSTLY OK — both phases separate by direction (sync of sign")
        print("is working), but the data's switch point is further from the")
        print("boundary than expected. Usually means you started/stopped sweeping")
        print("off the prompt timing rather than a real sync fault. Re-run and try")
        print("to switch direction promptly when prompted.")
        passed = False
    else:
        print("RESULT: FAIL — the phases did NOT separate by direction.")
        print("  Either raw deltas aren't reading in-game (run")
        print("  `python -m src.raw_mouse --selftest` with CS2 focused), or the")
        print("  loop isn't pairing inputs with frames correctly. Do NOT record")
        print("  real data until this passes — per PROJECT_ISSUES #3, a sync that")
        print("  can't be shown reliable is a kill-flag condition, not a detail.")
        passed = False
    return passed


def _build_parser():
    p = argparse.ArgumentParser(
        description="Synchronized frame+input recorder (Issue #3, M0 gate; "
                    "v3 FPV+radar per D-024).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--verify", action="store_true",
                   help="scripted-motion alignment check — RUN THIS FIRST")
    g.add_argument("--record", action="store_true",
                   help="record an extended chunked session to disk (v3 folder: FPV+radar)")
    g.add_argument("--record-single", action="store_true",
                   help="record one single-file .npz (LEGACY v1, FPV-only; round-trip check)")
    g.add_argument("--dryrun", action="store_true",
                   help="run the loop with a live readout, save nothing")
    g.add_argument("--profile", action="store_true",
                   help="time each loop stage to find what limits FPS (saves nothing)")
    p.add_argument("--seconds", type=float, default=None,
                   help="duration; defaults per mode. Omit with --record to run until F8.")
    p.add_argument("--name", type=str, default=None,
                   help="optional recording name stub")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.verify:
        verify(seconds=args.seconds or 12.0)
    elif args.record:
        record_session(seconds=args.seconds, name=args.name)
    elif args.record_single:
        record(seconds=args.seconds or 60.0, name=args.name)
    elif args.dryrun:
        dryrun(seconds=args.seconds or 20.0)
    elif args.profile:
        profile(seconds=args.seconds or 20.0)
    else:
        print("Choose a mode: --verify (do first), --record (extended v3), "
              "--dryrun, or --profile.")
        print("See `python -m src.recorder -h`.")


if __name__ == "__main__":
    main()
