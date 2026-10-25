"""Pose and angle helpers.

World frame: x/y on the floor, z up, yaw counter-clockwise from +x in (-pi, pi].
Robot frame: x forward, y left, z up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TWO_PI = 2.0 * math.pi


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    wrapped = math.fmod(angle + math.pi, TWO_PI)
    if wrapped < 0.0:
        wrapped += TWO_PI
    result = wrapped - math.pi
    return math.pi if result <= -math.pi else result


def angle_diff(target: float, current: float) -> float:
    """Shortest signed rotation from ``current`` to ``target``."""
    return wrap_angle(target - current)


def deg(radians: float) -> float:
    return math.degrees(radians)


def rad(degrees: float) -> float:
    return math.radians(degrees)


@dataclass(frozen=True)
class Pose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> Pose:
        return cls(float(data["x"]), float(data["y"]), float(data.get("yaw", 0.0)))

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "yaw": self.yaw}

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.yaw]

    def distance_to(self, other: Pose) -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def heading_to(self, other: Pose) -> float:
        return math.atan2(other.y - self.y, other.x - self.x)

    def to_local(self, x: float, y: float) -> tuple[float, float]:
        """Map a world point into this pose's frame."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x - self.x, y - self.y
        return c * dx + s * dy, -s * dx + c * dy

    def to_world(self, lx: float, ly: float) -> tuple[float, float]:
        """Map a point in this pose's frame to world coordinates."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * lx - s * ly, self.y + s * lx + c * ly

    def advanced(self, distance: float) -> Pose:
        return Pose(
            self.x + distance * math.cos(self.yaw),
            self.y + distance * math.sin(self.yaw),
            self.yaw,
        )

    def yaw_error_to(self, other: Pose) -> float:
        return angle_diff(other.yaw, self.yaw)


def interpolate(a: Pose, b: Pose, t: float) -> Pose:
    return Pose(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, wrap_angle(a.yaw + angle_diff(b.yaw, a.yaw) * t))
