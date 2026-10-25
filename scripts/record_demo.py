#!/usr/bin/env python3
"""Render the GUI demo to an MP4 without needing a desktop (offscreen Qt).

This is how the submitted screen-cast was produced: the demo drives three
scenes while a QTimer grabs the window at 15 fps and pipes frames to ffmpeg.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo_drive import Demo  # noqa: E402

from vacuum.ui.main_window import MainWindow  # noqa: E402

WIDTH = 1600
HEIGHT = 1000
FPS = 15


def packed_rgb(image: QImage) -> bytes:
    image = image.convertToFormat(QImage.Format_RGB888)
    width, height = image.width(), image.height()
    stride = image.bytesPerLine()
    raw = image.constBits().tobytes()
    return b"".join(raw[y * stride : y * stride + width * 3] for y in range(height))


class WindowRecorder:
    def __init__(self, window: MainWindow, output: Path) -> None:
        self.window = window
        self.output = output
        self.frames = 0
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
            ],
            stdin=subprocess.PIPE,
        )
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture)
        self.timer.start(int(1000 / FPS))

    def capture(self) -> None:
        if self.proc.stdin is None:
            return
        image = self.window.grab().toImage()
        if image.width() != WIDTH or image.height() != HEIGHT:
            image = image.scaled(WIDTH, HEIGHT)
        try:
            self.proc.stdin.write(packed_rgb(image))
        except (BrokenPipeError, OSError):
            self.timer.stop()
            return
        self.frames += 1

    def close(self) -> None:
        self.timer.stop()
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("../screen_recording.mp4"))
    parser.add_argument("--start-delay-ms", type=int, default=1_500)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv[:1])
    window = MainWindow(recordings_dir=Path("recordings"))
    window.resize(WIDTH, HEIGHT)
    window.show()
    recorder = WindowRecorder(window, args.out)
    demo = Demo(window)
    app.aboutToQuit.connect(recorder.close)
    QTimer.singleShot(args.start_delay_ms, demo.start)
    code = app.exec()
    recorder.close()
    print(f"wrote {args.out} ({recorder.frames} frames)", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
