from __future__ import annotations

import math

import pytest

from vacuum.control.pure_pursuit import (
    DriveConfig,
    PurePursuit,
    Twist,
    VelocityLimiter,
    rotate_towards,
)
from vacuum.model.pose import Pose


def test_straight_path_moves_forward() -> None:
    pursuit = PurePursuit()
    pursuit.set_path([(0.0, 0.0), (2.0, 0.0)])
    twist, finished = pursuit.update(Pose(0.0, 0.0, 0.0))
    assert twist.linear > 0.1
    assert abs(twist.angular) < 0.1
    assert not finished


def test_carrot_behind_turns_in_place() -> None:
    pursuit = PurePursuit()
    pursuit.set_path([(0.0, 0.0), (-2.0, 0.0)])
    twist, _ = pursuit.update(Pose(0.0, 0.0, 0.0))
    assert twist.linear == 0.0
    assert twist.angular != 0.0


def test_goal_reached() -> None:
    pursuit = PurePursuit()
    pursuit.set_path([(0.0, 0.0), (0.5, 0.0)])
    twist, finished = pursuit.update(Pose(0.5, 0.0, 0.0))
    assert finished
    assert twist.linear == 0.0


def test_wheel_saturation() -> None:
    config = DriveConfig(max_wheel_speed=10.0)
    left, right = Twist(3.0, 3.0).wheels(config)
    assert max(abs(left), abs(right)) <= 10.0


def test_rotate_towards_settles() -> None:
    assert rotate_towards(math.radians(170), math.radians(-170)).angular > 0
    assert rotate_towards(0.0, 0.0).angular == 0.0


def test_velocity_limiter_uses_elapsed_simulation_time() -> None:
    limiter = VelocityLimiter(DriveConfig(max_linear_accel=1.0, max_angular_accel=2.0))
    assert limiter.update(Twist(1.0, 2.0), 10.0) == Twist()
    limited = limiter.update(Twist(1.0, 2.0), 10.1)
    assert limited.linear == pytest.approx(0.1)
    assert limited.angular == pytest.approx(0.2)


def test_velocity_limiter_resets_on_time_rollback() -> None:
    limiter = VelocityLimiter(DriveConfig())
    limiter.update(Twist(1.0, 1.0), 2.0)
    limiter.update(Twist(1.0, 1.0), 2.1)
    assert limiter.update(Twist(1.0, 1.0), 0.0) == Twist()
