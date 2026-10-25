"""RGBD recording: H.264 RGB, lossless 16-bit depth, JSON manifest."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import queue
import subprocess
import threading
from pathlib import Path

from vacuum.sim.types import CameraFrame

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"
META_NAME = "meta.json"
RGB_NAME = "rgb.mp4"
DEPTH_NAME = "depth.mkv"


class Recorder:
    """Consume camera frames on a worker thread and write them to disk.

    RGB is encoded as H.264 (small, easy to inspect). Depth is kept lossless as
    16-bit grayscale using FFV1. Per-frame timing and pose live in the manifest,
    so drops or variable timing are visible offline.
    """

    def __init__(
        self,
        directory: Path,
        width: int,
        height: int,
        fps: float = 15.0,
        camera_info: dict | None = None,
        scene: str = "",
        seed: int | None = None,
        queue_size: int = 60,
    ) -> None:
        self.directory = Path(directory)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.camera_info = dict(camera_info or {})
        self.scene = scene
        self.seed = seed
        self.queue: queue.Queue[CameraFrame | None] = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self.frames_written = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._rgb_proc: subprocess.Popen | None = None
        self._depth_proc: subprocess.Popen | None = None
        self._manifest = None
        self.started_at = _dt.datetime.now().astimezone()

    # ------------------------------------------------------------------ setup
    def start(self) -> Recorder:
        self.directory.mkdir(parents=True, exist_ok=True)
        size = f"{self.width}x{self.height}"
        rgb_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", str(self.fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self.directory / RGB_NAME),
        ]
        depth_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", size, "-r", str(self.fps), "-i", "-",
            "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray16le",
            str(self.directory / DEPTH_NAME),
        ]
        try:
            self._write_meta(started=True)
            self._manifest = open(self.directory / MANIFEST_NAME, "w", encoding="utf-8")
            self._rgb_proc = subprocess.Popen(
                rgb_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self._depth_proc = subprocess.Popen(
                depth_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
            self._thread.start()
        except Exception:
            self._cleanup_partial_start()
            raise
        log.info("recording to %s", self.directory)
        return self

    # ------------------------------------------------------------------ frames
    def add(self, frame: CameraFrame) -> None:
        if self._stop.is_set() or frame.width != self.width or frame.height != self.height:
            return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                self.dropped += 1

    def close(self) -> Path:
        if self._closed:
            return self.directory
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15.0)
            if self._thread.is_alive():
                self.error = self.error or "recorder worker did not stop within 15 seconds"
                for proc in (self._rgb_proc, self._depth_proc):
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                self._thread.join(timeout=2.0)
        for proc in (self._rgb_proc, self._depth_proc):
            if proc is None:
                continue
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
                return_code = proc.wait(timeout=10.0)
                stderr = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
                if return_code != 0:
                    detail = f": {stderr}" if stderr else ""
                    self.error = self.error or f"ffmpeg exited with status {return_code}{detail}"
            except Exception as exc:  # noqa: BLE001 - best-effort shutdown
                proc.kill()
                self.error = self.error or f"ffmpeg shutdown failed: {exc}"
        if self._manifest is not None:
            self._manifest.close()
            self._manifest = None
        self._closed = True
        self._write_meta(started=False)
        log.info(
            "recording finished: %s (%d frames, %d dropped)",
            self.directory, self.frames_written, self.dropped,
        )
        return self.directory

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        import io

        from PIL import Image

        while True:
            try:
                frame = self.queue.get(timeout=0.25)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if frame is None:
                break
            try:
                rgb = Image.open(io.BytesIO(frame.jpeg)).convert("RGB").resize(
                    (self.width, self.height)
                )
                depth_bytes = frame.depth_raw
                expected = self.width * self.height * 2
                if len(depth_bytes) != expected:
                    self.error = f"depth payload {len(depth_bytes)} != {expected}"
                    continue
                assert self._rgb_proc and self._rgb_proc.stdin
                assert self._depth_proc and self._depth_proc.stdin
                self._rgb_proc.stdin.write(rgb.tobytes())
                self._depth_proc.stdin.write(depth_bytes)
                record = {
                    "frame": self.frames_written,
                    "t": frame.t,
                    "seq": frame.seq,
                    "pose": frame.pose.as_list(),
                    "dropped_before": self.dropped,
                }
                assert self._manifest is not None
                self._manifest.write(json.dumps(record) + "\n")
                self.frames_written += 1
                if self.frames_written % max(1, int(self.fps)) == 0:
                    self._manifest.flush()
            except (BrokenPipeError, OSError) as exc:
                self.error = f"ffmpeg pipe failed: {exc}"
                log.error(self.error)
                break
            except Exception as exc:  # noqa: BLE001 - keep recorder alive
                self.error = f"frame write failed: {exc}"
                log.exception("frame write failed")
        if self._manifest is not None:
            self._manifest.flush()
        for proc in (self._rgb_proc, self._depth_proc):
            if proc is not None and proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

    def _cleanup_partial_start(self) -> None:
        """Release resources if construction of either encoder fails."""
        self._stop.set()
        for proc in (self._rgb_proc, self._depth_proc):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
        if self._manifest is not None:
            self._manifest.close()
            self._manifest = None
        self._closed = True
        self.error = "recording failed during startup"
        self._write_meta(started=False)

    # -------------------------------------------------------------------- meta
    def _write_meta(self, started: bool) -> None:
        if not started and self.directory.exists():
            path = self.directory / META_NAME
            meta = {}
            if path.exists():
                try:
                    meta = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    meta = {}
        else:
            meta = {
                "created_at": self.started_at.isoformat(),
                "scene": self.scene,
                "seed": self.seed,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "camera_info": self.camera_info,
                "files": {
                    "rgb": RGB_NAME,
                    "depth": DEPTH_NAME,
                    "manifest": MANIFEST_NAME,
                },
                "format": {
                    "rgb": "H.264 MP4, RGB24, constant nominal fps",
                    "depth": "FFV1 MKV, gray16le, millimetres (multiply by depth_scale)",
                    "manifest": "JSONL: frame index, sim time t, seq, robot pose [x, y, yaw]",
                },
            }
        meta["started"] = started
        meta["frames_written"] = self.frames_written
        meta["dropped"] = self.dropped
        if self.frames_written:
            meta["duration_s"] = self.frames_written / self.fps
        if self.error:
            meta["error"] = self.error
        (self.directory / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
