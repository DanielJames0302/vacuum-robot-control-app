from __future__ import annotations

import numpy as np
import pytest

from vacuum.control.engine import GOAL_POSITION_TOLERANCE, MissionEngine, MissionState
from vacuum.model.gridmap import OccupancyMap
from vacuum.model.pose import Pose
from vacuum.sim.types import Goal, RobotState

ROBOT_INFO = {
    "radius": 0.17,
    "wheel_base": 0.23,
    "wheel_radius": 0.035,
    "max_wheel_speed": 10.0,
    "max_linear_speed": 0.35,
    "max_angular_speed": 2.0,
}


def open_map() -> OccupancyMap:
    return OccupancyMap(np.zeros((40, 40), dtype=bool), resolution=0.1, origin=(0.0, 0.0))


def make_engine() -> MissionEngine:
    engine = MissionEngine()
    engine.set_scene(
        open_map(),
        ROBOT_INFO,
        [Goal("target", Pose(3.5, 3.5, 0.0), has_yaw=False)],
        Pose(0.5, 0.5, 0.0),
    )
    return engine


def state(pose: Pose, t: float, seq: int = 1, bumper: bool = False) -> RobotState:
    return RobotState(t=t, seq=seq, scene="test", pose=pose, bumper=bumper)


def test_plan_follow_and_mark_goal() -> None:
    engine = make_engine()
    assert engine.plan_goal("target")
    assert engine.state is MissionState.DRIVING
    assert len(engine.path) > 2

    output = engine.step(state(Pose(0.5, 0.5, 0.0), 0.1))
    assert output.twist.linear > 0.0

    marked = None
    for index in range(8):
        output = engine.step(state(Pose(3.5, 3.5, 0.0), 1.0 + index, seq=index + 2))
        if output.mark_goal is not None:
            marked = output
            break
    assert marked is not None
    assert marked.mark_goal == "target"
    assert marked.state is MissionState.WAIT_MARK

    engine.confirm_mark(True)
    assert "target" in engine.marked
    assert engine.state is MissionState.DONE


def test_goal_tolerance_is_five_centimetres() -> None:
    assert GOAL_POSITION_TOLERANCE == 0.05


def test_bumper_starts_recovery_and_reverses() -> None:
    engine = make_engine()
    assert engine.plan_goal("target")
    output = engine.step(state(Pose(0.5, 0.5, 0.0), 5.0, bumper=True))
    assert output.state is MissionState.RECOVERING
    assert output.twist.linear < 0.0
    assert engine.bump_count == 1


def test_cancel_returns_to_ready() -> None:
    engine = make_engine()
    assert engine.plan_goal("target")
    engine.cancel()
    assert engine.state is MissionState.READY
    assert engine.path == []


def test_unknown_goal_rejected() -> None:
    engine = make_engine()
    assert not engine.plan_goal("nope")
    assert engine.state is MissionState.READY


def test_depth_hits_require_two_observations() -> None:
    engine = make_engine()
    points = np.array([[1.0, 1.0, 0.1]], dtype=np.float32)
    assert engine.observe_depth(points) == 0
    added = engine.observe_depth(points)
    assert added == 1
    assert engine.depth_mask is not None
    assert engine.depth_mask[29, 10]


def test_depth_observation_ignores_out_of_bounds_points() -> None:
    engine = make_engine()
    points = np.array([[1.0, 1.0, 0.1], [100.0, -100.0, 0.1]], dtype=np.float32)
    assert engine.observe_depth(points) == 0
    assert engine.observe_depth(points) == 1


def test_duplicate_depth_points_count_one_cell() -> None:
    engine = make_engine()
    points = np.repeat(np.array([[1.0, 1.0, 0.1]], dtype=np.float32), 20, axis=0)
    assert engine.observe_depth(points) == 0
    assert engine.observe_depth(points) == 1


def test_scene_limits_do_not_stick_across_loads() -> None:
    engine = MissionEngine()
    slow = dict(ROBOT_INFO, max_linear_speed=0.1, max_angular_speed=0.4)
    engine.set_scene(open_map(), slow, [], Pose())
    assert engine.config.max_speed == pytest.approx(0.09)
    engine.set_scene(open_map(), ROBOT_INFO, [], Pose())
    assert engine.config.max_speed == 0.30
    assert engine.config.max_omega == 1.5
