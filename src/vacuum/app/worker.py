"""Qt worker thread: owns the controller, emits snapshots to the GUI."""
from __future__ import annotations

import queue
import time
import traceback

from PySide6.QtCore import QThread, Signal

from vacuum.app.controller import MissionController

SNAPSHOT_PERIOD = 1.0 / 20.0


class SimWorker(QThread):
    scene_loaded = Signal(object)
    snapshot_ready = Signal(object)
    preview_ready = Signal(object)
    log_message = Signal(str)
    error = Signal(str)

    def __init__(self, controller: MissionController) -> None:
        super().__init__()
        self.controller = controller
        self._actions: queue.Queue[tuple] = queue.Queue()
        self._running = True

    # --------------------------------------------------------------- commands
    def load_scene(self, scene_id: str, seed: int | None) -> None:
        self._actions.put(("load", scene_id, seed))

    def reset_scene(self, seed: int | None) -> None:
        self._actions.put(("reset", seed))

    def start_goal(self, name: str) -> None:
        self._actions.put(("goal", name))

    def cancel_goal(self) -> None:
        self._actions.put(("cancel",))

    def start_recording(self) -> None:
        self._actions.put(("record_start",))

    def stop_recording(self) -> None:
        self._actions.put(("record_stop",))

    def shutdown(self) -> None:
        self._actions.put(("stop",))
        self._running = False

    # ------------------------------------------------------------------- loop
    def run(self) -> None:  # noqa: C901 - small, explicit event loop
        last_snapshot = 0.0
        last_preview = None
        if not self.controller.ping():
            self.error.emit(
                "Cannot reach the simulator on the configured ports. "
                "Start it with: docker run --rm -p 5555-5558:5555-5558 "
                "-p 8080:8080 ghcr.io/eurekarobotics/vacuum-sim:1.0.0 --scene 01 --viewer"
            )
        else:
            self.log_message.emit("Connected to simulator")

        while self._running:
            self._process_actions()
            try:
                self.controller.tick()
            except Exception as exc:  # noqa: BLE001 - keep the GUI alive
                self.error.emit(f"controller error: {exc}")
                self.log_message.emit(traceback.format_exc())
                time.sleep(0.05)

            for event in self.controller.drain_events():
                self.log_message.emit(event)

            now = time.monotonic()
            if now - last_snapshot >= SNAPSHOT_PERIOD:
                last_snapshot = now
                self.snapshot_ready.emit(self.controller.snapshot())
                if self.controller.fault:
                    self.error.emit(self.controller.fault)

            preview = self.controller.latest_preview
            if preview is not None and preview is not last_preview:
                last_preview = preview
                self.preview_ready.emit(preview)
            time.sleep(0.005)

        self.controller.close()

    def _process_actions(self) -> None:
        while True:
            try:
                action = self._actions.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_action(action)
            except Exception as exc:  # noqa: BLE001
                self.error.emit(f"{action[0]} failed: {exc}")
                self.log_message.emit(traceback.format_exc())

    def _handle_action(self, action: tuple) -> None:
        kind = action[0]
        if kind == "load":
            _, scene_id, seed = action
            bundle = self.controller.load_scene(scene_id, seed)
            self.scene_loaded.emit(
                {
                    "scene": bundle.scene,
                    "occupancy": bundle.occupancy,
                    "robot_info": bundle.robot_info,
                    "camera_info": bundle.camera_info,
                }
            )
        elif kind == "reset":
            _, seed = action
            bundle = self.controller.reset_scene(seed)
            self.scene_loaded.emit(
                {
                    "scene": bundle.scene,
                    "occupancy": bundle.occupancy,
                    "robot_info": bundle.robot_info,
                    "camera_info": bundle.camera_info,
                }
            )
        elif kind == "goal":
            _, name = action
            if not self.controller.start_goal(name):
                self.error.emit(f"could not plan a path to {name}")
        elif kind == "cancel":
            self.controller.cancel()
        elif kind == "record_start":
            directory = self.controller.start_recording()
            if directory is not None:
                self.log_message.emit(f"recording started: {directory}")
        elif kind == "record_stop":
            self.controller.stop_recording()
        elif kind == "stop":
            self._running = False
