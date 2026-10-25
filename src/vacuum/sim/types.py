"""Typed views over the simulator's JSON messages."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

from vacuum.model.pose import Pose


@dataclass
class Goal:
    name: str
    pose: Pose
    has_yaw: bool = False

    @classmethod
    def from_json(cls, data: dict) -> Goal:
        values = list(data["pose"])
        yaw = float(values[2]) if len(values) > 2 else 0.0
        return cls(
            name=str(data["name"]),
            pose=Pose(float(values[0]), float(values[1]), yaw),
            has_yaw=len(values) > 2,
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "pose": self.pose.as_list(),
            "has_yaw": self.has_yaw,
        }


@dataclass
class SceneData:
    id: str
    name: str = ""
    difficulty: str = ""
    start: Pose = field(default_factory=Pose)
    goals: list[Goal] = field(default_factory=list)


@dataclass
class RobotState:
    t: float
    seq: int
    scene: str
    pose: Pose
    left_ticks: int = 0
    right_ticks: int = 0
    bumper: bool = False
    goto_id: int | None = None
    goto_status: str | None = None

    @classmethod
    def from_json(cls, data: dict) -> RobotState:
        goto = data.get("goto") or {}
        wheels = data.get("wheels") or {}
        return cls(
            t=float(data["t"]),
            seq=int(data["seq"]),
            scene=str(data.get("scene", "")),
            pose=Pose.from_dict(data["pose"]),
            left_ticks=int(wheels.get("left_ticks", 0)),
            right_ticks=int(wheels.get("right_ticks", 0)),
            bumper=bool(data.get("bumper", False)),
            goto_id=goto.get("id"),
            goto_status=goto.get("status"),
        )


@dataclass
class CameraFrame:
    t: float
    seq: int
    pose: Pose
    width: int
    height: int
    jpeg: bytes
    depth_raw: bytes

    @classmethod
    def from_frames(cls, header: dict, jpeg: bytes, depth_raw: bytes) -> CameraFrame:
        return cls(
            t=float(header["t"]),
            seq=int(header["seq"]),
            pose=Pose.from_dict(header["pose"]),
            width=int(header["width"]),
            height=int(header["height"]),
            jpeg=jpeg,
            depth_raw=depth_raw,
        )

    def rgb_array(self) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(self.jpeg)).convert("RGB"))

    def depth_array(self, depth_scale: float = 0.001) -> np.ndarray:
        depth = np.frombuffer(self.depth_raw, dtype="<u2")
        if depth.size != self.width * self.height:
            raise ValueError(
                f"depth payload has {depth.size} values, expected {self.width * self.height}"
            )
        return depth.reshape(self.height, self.width).astype(np.float32) * depth_scale
