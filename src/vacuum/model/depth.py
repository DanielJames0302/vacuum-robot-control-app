"""Back-project RGBD depth into world-frame points for obstacle detection."""
from __future__ import annotations

import numpy as np

from vacuum.model.pose import Pose
from vacuum.sim.types import CameraFrame


def depth_points_world(
    frame: CameraFrame,
    camera_info: dict,
    stride: int = 6,
    min_depth: float = 0.15,
    max_depth: float = 3.0,
    min_height: float = 0.035,
    max_height: float = 0.25,
    max_forward: float = 2.5,
) -> np.ndarray:
    """Return Nx3 world points that are likely to collide with the robot.

    Only points between ``min_height`` and ``max_height`` in the robot frame are
    kept, so the floor and tabletops (which the robot can pass under) are
    ignored. The camera pose in the frame header is used for the transform.
    """
    depth = np.frombuffer(frame.depth_raw, dtype="<u2").reshape(frame.height, frame.width)
    depth = depth[::stride, ::stride].astype(np.float32) * float(
        camera_info.get("depth_scale", 0.001)
    )
    valid = (depth > min_depth) & (depth < max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    rows = np.arange(0, frame.height, stride)
    cols = np.arange(0, frame.width, stride)
    u, v = np.meshgrid(cols, rows)
    fx = float(camera_info.get("fx", 415.69))
    fy = float(camera_info.get("fy", 415.69))
    cx = float(camera_info.get("cx", frame.width / 2.0 - 0.5))
    cy = float(camera_info.get("cy", frame.height / 2.0 - 0.5))

    z = depth
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy

    transform = np.asarray(
        camera_info.get(
            "T_base_camera",
            [[0, 0, 1, 0.16], [-1, 0, 0, 0], [0, -1, 0, 0.08], [0, 0, 0, 1]],
        ),
        dtype=np.float32,
    )
    x_robot = transform[0, 0] * x_cam + transform[0, 1] * y_cam + transform[0, 2] * z + transform[0, 3]
    y_robot = transform[1, 0] * x_cam + transform[1, 1] * y_cam + transform[1, 2] * z + transform[1, 3]
    z_robot = transform[2, 0] * x_cam + transform[2, 1] * y_cam + transform[2, 2] * z + transform[2, 3]

    keep = valid & (z_robot >= min_height) & (z_robot <= max_height)
    keep &= (x_robot >= 0.1) & (x_robot <= max_forward)
    if not np.any(keep):
        return np.empty((0, 3), dtype=np.float32)

    pose: Pose = frame.pose
    cos_yaw, sin_yaw = np.cos(pose.yaw), np.sin(pose.yaw)
    x_world = pose.x + cos_yaw * x_robot[keep] - sin_yaw * y_robot[keep]
    y_world = pose.y + sin_yaw * x_robot[keep] + cos_yaw * y_robot[keep]
    z_world = z_robot[keep]
    return np.stack([x_world, y_world, z_world], axis=1)
