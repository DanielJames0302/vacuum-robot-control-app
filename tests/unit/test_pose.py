from __future__ import annotations

import math

import pytest

from vacuum.model.pose import Pose, angle_diff, interpolate, wrap_angle


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, math.pi),
        (3.0 * math.pi, math.pi),
        (1.5 * math.pi, -0.5 * math.pi),
    ],
)
def test_wrap_angle(angle: float, expected: float) -> None:
    assert wrap_angle(angle) == pytest.approx(expected)


def test_angle_diff_takes_short_way() -> None:
    assert angle_diff(math.radians(350), math.radians(10)) == pytest.approx(
        math.radians(-20)
    )
    assert angle_diff(math.radians(-170), math.radians(170)) == pytest.approx(
        math.radians(20)
    )


def test_local_world_roundtrip() -> None:
    pose = Pose(1.2, -0.4, math.radians(37))
    world = pose.to_world(0.8, -0.3)
    local = pose.to_local(*world)
    assert local == pytest.approx((0.8, -0.3))


def test_advanced_and_heading() -> None:
    pose = Pose(1.0, 2.0, math.pi / 2)
    ahead = pose.advanced(0.5)
    assert (ahead.x, ahead.y) == pytest.approx((1.0, 2.5))
    assert pose.distance_to(ahead) == pytest.approx(0.5)
    assert pose.heading_to(Pose(4.0, 2.0)) == pytest.approx(0.0)


def test_interpolate_yaw_crosses_pi() -> None:
    mid = interpolate(Pose(0, 0, math.radians(170)), Pose(2, 0, math.radians(-170)), 0.5)
    assert mid.x == pytest.approx(1.0)
    assert abs(wrap_angle(mid.yaw)) == pytest.approx(math.pi)
