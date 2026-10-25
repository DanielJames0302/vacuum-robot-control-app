from __future__ import annotations

import numpy as np
import pytest

from vacuum.model.depth import depth_points_world
from vacuum.model.pose import Pose
from vacuum.sim.types import CameraFrame

CAMERA = {
    "fx": 100.0,
    "fy": 200.0,
    "cx": 1.5,
    "cy": 1.5,
    "depth_scale": 0.001,
    "T_base_camera": [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.08],
        [0.0, 0.0, 0.0, 1.0],
    ],
}


def frame(depth_mm: int, width: int = 4, height: int = 4) -> CameraFrame:
    raw = np.full((height, width), depth_mm, dtype="<u2").tobytes()
    return CameraFrame(
        t=1.0,
        seq=2,
        pose=Pose(1.0, 2.0, np.pi / 2),
        width=width,
        height=height,
        jpeg=b"",
        depth_raw=raw,
    )


def test_flat_wall_becomes_collidable_points() -> None:
    points = depth_points_world(frame(1000), CAMERA, stride=1)
    assert len(points) == 16
    assert points[:, 2].min() >= 0.035
    assert points[:, 2].max() <= 0.25
    # Camera x maps to robot x (forward), then the 90 deg pose maps it to +y.
    assert np.allclose(points[:, 0], 1.0, atol=0.02)
    assert np.allclose(points[:, 1], 3.0, atol=1e-4)


def test_floor_and_far_points_are_filtered() -> None:
    assert depth_points_world(frame(100), CAMERA, stride=1).size == 0
    assert depth_points_world(frame(5000), CAMERA, stride=1).size == 0


def test_stride_subsamples() -> None:
    points = depth_points_world(frame(1000, 8, 8), CAMERA, stride=2)
    assert len(points) == 16


def test_points_above_robot_height_are_filtered() -> None:
    camera = dict(CAMERA)
    camera["T_base_camera"] = [
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 0.20],
        [0, 0, 0, 1],
    ]
    assert depth_points_world(frame(1000), camera, stride=1, max_height=0.10).size == 0


def test_malformed_depth_payload_is_rejected() -> None:
    bad = frame(1000)
    bad.depth_raw = b"too short"
    with pytest.raises(ValueError):
        depth_points_world(bad, CAMERA)
