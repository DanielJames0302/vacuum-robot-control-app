#!/usr/bin/env python3
"""Scripted GUI demo used to record the screen-cast (development helper).

Loads three scenes, drives a few goals while the visualiser is visible, then
quits. Start the simulator first, then run this with a display available.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vacuum.ui.main_window import MainWindow  # noqa: E402

STEPS = [
    ("01", ["kitchen", "living"]),
    ("04", ["desk"]),
    ("06", ["office", "sofa"]),
]
GOAL_TIMEOUT_MS = 120_000
SETTLE_MS = 2_500


class Demo:
    def __init__(self, window: MainWindow) -> None:
        self.window = window
        self.step = 0
        self.goal = 0
        self.active = False
        self.started = False
        self.awaiting_scene = True
        window.worker.scene_loaded.connect(self._scene_loaded)
        window.worker.snapshot_ready.connect(self._snapshot)
        self._goal_timer = QTimer()
        self._goal_timer.setSingleShot(True)
        self._goal_timer.timeout.connect(self._goal_timeout)

    def start(self) -> None:
        # MainWindow auto-loads the first scene; wait for that signal instead
        # of issuing a second load.
        print("demo: waiting for initial scene load", flush=True)

    def _load_step(self) -> None:
        if self.step >= len(STEPS):
            self._finish()
            return
        scene_id, _ = STEPS[self.step]
        index = self.window.scene_combo.findData(scene_id)
        if index < 0:
            print(f"scene {scene_id} not offered by the simulator", flush=True)
            self.step += 1
            self._load_step()
            return
        self.window.scene_combo.setCurrentIndex(index)
        self.window.use_seed.setChecked(True)
        self.window.seed_spin.setValue(42)
        self.awaiting_scene = True
        self.window._load_scene()  # noqa: SLF001 - demo helper
        print(f"demo: loading scene {scene_id}", flush=True)

    def _scene_loaded(self, payload: dict) -> None:
        if not self.awaiting_scene:
            return
        self.awaiting_scene = False
        if not self.started:
            self.started = True
        QTimer.singleShot(SETTLE_MS, self._next_goal)

    def _next_goal(self) -> None:
        _, goals = STEPS[self.step]
        if self.goal >= len(goals):
            self.step += 1
            self.goal = 0
            QTimer.singleShot(SETTLE_MS, self._load_step)
            return
        name = goals[self.goal]
        index = None
        for i in range(self.window.goal_combo.count()):
            if self.window.goal_combo.itemText(i).replace(" ✓", "") == name:
                index = i
                break
        if index < 0:
            print(f"demo: goal {name!r} not in scene", flush=True)
            self.goal += 1
            self._next_goal()
            return
        self.window.goal_combo.setCurrentIndex(index)
        self.active = True
        self.window._go()  # noqa: SLF001 - demo helper
        self._goal_timer.start(GOAL_TIMEOUT_MS)
        print(f"demo: scene {STEPS[self.step][0]} goal {name}", flush=True)

    def _snapshot(self, snapshot: dict) -> None:
        if not self.active:
            return
        state = snapshot.get("state")
        if state in ("done", "aborted"):
            self.active = False
            self._goal_timer.stop()
            print(f"demo: goal finished ({state})", flush=True)
            self.goal += 1
            QTimer.singleShot(SETTLE_MS, self._next_goal)

    def _goal_timeout(self) -> None:
        if not self.active:
            return
        print("demo: goal timeout; cancelling", flush=True)
        self.window._cancel()  # noqa: SLF001 - demo helper
        self.active = False
        self.goal += 1
        QTimer.singleShot(SETTLE_MS, self._next_goal)

    def _finish(self) -> None:
        print("demo: finished", flush=True)
        self.window.close()
        QApplication.instance().quit()


def main() -> int:
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Vacuum Robot Console")
    window = MainWindow(recordings_dir=Path("recordings"))
    window.setGeometry(40, 40, 1600, 1000)
    window.show()
    demo = Demo(window)
    QTimer.singleShot(1_500, demo.start)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
