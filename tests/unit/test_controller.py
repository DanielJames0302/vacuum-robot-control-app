from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("zmq")

import vacuum.app.controller as controller_module
from vacuum.app.controller import MissionController
from vacuum.model.gridmap import OccupancyMap
from vacuum.model.pose import Pose
from vacuum.sim.types import CameraFrame, RobotState, SceneData


class FakeSim:
    def __init__(self, **_kwargs) -> None:
        self.messages = []
        self.state_gaps = 0
        self.camera_gaps = 0
        self.command_ok = True

    def poll(self, timeout_ms=0):
        messages, self.messages = self.messages, []
        return messages

    def send_wheels(self, _left, _right):
        return self.command_ok

    def get_map(self):
        return {}, b"png"

    def get_robot_info(self):
        return {}

    def get_camera_info(self):
        return {}

    def list_scenes(self):
        return [{"id": "05", "name": "Advanced", "difficulty": "advanced"}]

    def drain_streams(self) -> None:
        pass

    def close(self) -> None:
        pass

    def stop(self) -> None:
        pass


def state(seq: int, scene: str = "01", t: float = 1.0) -> RobotState:
    return RobotState(t=t, seq=seq, scene=scene, pose=Pose(0.5, 0.5, 0.0))


def camera(seq: int, t: float = 1.0) -> CameraFrame:
    return CameraFrame(t, seq, Pose(0.5, 0.5, 0.0), 1, 1, b"jpeg", b"\0\0")


def make_controller(monkeypatch) -> MissionController:
    monkeypatch.setattr(controller_module, "SimClient", FakeSim)
    controller = MissionController(use_depth=False)
    occupancy = OccupancyMap(np.zeros((20, 20), dtype=bool), 0.1, (0.0, 0.0))
    controller.scene = SceneData(id="01", start=Pose(0.5, 0.5, 0.0))
    controller.engine.set_scene(occupancy, {}, [], controller.scene.start)
    controller._awaiting_fresh_state = True
    return controller


def test_tick_uses_newest_valid_state_and_rejects_stale(monkeypatch) -> None:
    controller = make_controller(monkeypatch)
    controller.sim.messages = [
        ("state", state(8, scene="02")),
        ("state", state(2)),
        ("state", state(1)),
    ]
    controller.tick()
    assert controller.latest_state is not None
    assert controller.latest_state.seq == 2
    assert controller.snapshot()["stale_states"] == 2


def test_camera_waits_for_current_state_and_rejects_backward_sequence(monkeypatch) -> None:
    controller = make_controller(monkeypatch)
    controller.sim.messages = [("camera", camera(9))]
    controller.tick()
    assert controller.latest_camera is None

    controller.sim.messages = [("state", state(1)), ("camera", camera(2))]
    controller.tick()
    assert controller.latest_camera is not None
    controller.sim.messages = [("camera", camera(1))]
    controller.tick()
    assert controller.snapshot()["stale_cameras"] == 2


def test_dropped_wheel_command_becomes_fault(monkeypatch) -> None:
    controller = make_controller(monkeypatch)
    controller.sim.command_ok = False
    controller.sim.messages = [("state", state(1))]
    controller.tick()
    assert controller.fault == "command queue full; wheel command was dropped"


def test_depth_is_enabled_by_default_for_advanced_scene(monkeypatch) -> None:
    monkeypatch.setattr(controller_module, "SimClient", FakeSim)
    occupancy = OccupancyMap(np.zeros((20, 20), dtype=bool), 0.1, (0.0, 0.0))
    monkeypatch.setattr(
        OccupancyMap,
        "from_png",
        classmethod(lambda _cls, _meta, _png: occupancy),
    )
    controller = MissionController()
    controller._apply_scene_result(
        "05",
        42,
        {"goals": [], "start": {"x": 0.5, "y": 0.5, "yaw": 0.0}},
    )
    assert controller.engine.use_depth is True


def test_no_depth_override_still_disables_advanced_scene(monkeypatch) -> None:
    monkeypatch.setattr(controller_module, "SimClient", FakeSim)
    occupancy = OccupancyMap(np.zeros((20, 20), dtype=bool), 0.1, (0.0, 0.0))
    monkeypatch.setattr(
        OccupancyMap,
        "from_png",
        classmethod(lambda _cls, _meta, _png: occupancy),
    )
    controller = MissionController(use_depth=False)
    controller._apply_scene_result(
        "05",
        42,
        {"goals": [], "start": {"x": 0.5, "y": 0.5, "yaw": 0.0}},
    )
    assert controller.engine.use_depth is False
