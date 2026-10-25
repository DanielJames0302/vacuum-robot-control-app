"""Differential-drive path following and low-level helpers."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from vacuum.model.pose import Pose, angle_diff


@dataclass
class DriveConfig:
    max_speed: float = 0.30
    max_omega: float = 1.5
    lookahead: float = 0.25
    min_lookahead: float = 0.20
    waypoint_tolerance: float = 0.20
    goal_tolerance: float = 0.08
    slowdown_distance: float = 0.45
    wheel_base: float = 0.23
    wheel_radius: float = 0.035
    max_wheel_speed: float = 10.0
    max_linear_accel: float = 1.0
    max_angular_accel: float = 6.0


@dataclass
class Twist:
    linear: float = 0.0
    angular: float = 0.0

    def wheels(self, config: DriveConfig) -> tuple[float, float]:
        left = (self.linear - self.angular * config.wheel_base / 2.0) / config.wheel_radius
        right = (self.linear + self.angular * config.wheel_base / 2.0) / config.wheel_radius
        peak = max(abs(left), abs(right))
        if peak > config.max_wheel_speed:
            scale = config.max_wheel_speed / peak
            left *= scale
            right *= scale
        return left, right


class PurePursuit:
    """Geometric path follower; command slew limiting is applied by the controller."""

    def __init__(self, config: DriveConfig | None = None) -> None:
        self.config = config or DriveConfig()
        self.path: list[tuple[float, float]] = []
        self.index = 0
        self._lengths: list[float] = []
        self._remaining = 0.0
        self.visibility: Callable[[tuple[float, float], tuple[float, float]], bool] | None = None

    def set_path(self, points: list[tuple[float, float]]) -> None:
        self.path = list(points)
        self.index = 0
        self._lengths = []
        total = 0.0
        for a, b in zip(self.path, self.path[1:], strict=False):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            self._lengths.append(length)
            total += length
        self._remaining = total

    @property
    def finished(self) -> bool:
        return not self.path or self.index >= len(self.path) - 1

    def _nearest_index(self, pose: Pose) -> int:
        if not self.path:
            return 0
        best_index = self.index
        best_distance = math.inf
        start = max(0, self.index - 5)
        end = min(len(self.path), self.index + 40)
        for i in range(start, end):
            px, py = self.path[i]
            distance = math.hypot(px - pose.x, py - pose.y)
            if distance < best_distance:
                best_distance = distance
                best_index = i
        return best_index

    def _lookahead_point(self, pose: Pose, index: int) -> tuple[float, float]:
        if index >= len(self.path) - 1:
            return self.path[-1]
        config = self.config
        distance_to_path = math.hypot(
            self.path[index][0] - pose.x, self.path[index][1] - pose.y
        )
        target = min(config.lookahead, config.min_lookahead + distance_to_path * 0.5)
        target = max(target, config.min_lookahead)

        # Never chase a carrot through a wall. Walk forward along the path and
        # stop at the furthest point that is directly visible from the robot.
        travelled = 0.0
        previous = self.path[index]
        chosen = self.path[index]
        for i in range(index, len(self.path) - 1):
            segment = self._lengths[i]
            point = self.path[i + 1]
            if self.visibility is not None and i > index:
                if not self.visibility((pose.x, pose.y), point):
                    break
            remaining = target - travelled
            if remaining <= segment and segment > 1e-9:
                t = remaining / segment
                ax, ay = previous
                bx, by = point
                candidate = (ax + (bx - ax) * t, ay + (by - ay) * t)
                if self.visibility is None or self.visibility((pose.x, pose.y), candidate):
                    return candidate
                return chosen
            travelled += segment
            previous = point
            chosen = point
        return chosen

    def update(self, pose: Pose) -> tuple[Twist, bool]:
        """Return a twist and whether the final path point has been reached."""
        config = self.config
        if not self.path:
            return Twist(), True

        self.index = self._nearest_index(pose)

        # Advance past waypoints that are already close enough.
        while self.index < len(self.path) - 1:
            px, py = self.path[self.index]
            if math.hypot(px - pose.x, py - pose.y) <= config.waypoint_tolerance:
                self.index += 1
            else:
                break

        goal_x, goal_y = self.path[-1]
        goal_distance = math.hypot(goal_x - pose.x, goal_y - pose.y)
        if self.index >= len(self.path) - 1 and goal_distance <= config.goal_tolerance:
            return Twist(), True

        target_x, target_y = self._lookahead_point(pose, self.index)
        heading = math.atan2(target_y - pose.y, target_x - pose.x)
        alpha = angle_diff(heading, pose.yaw)

        # If the carrot is behind us, turn in place. Creeping forward while
        # spinning around is how differential robots scrape walls.
        if abs(alpha) > math.radians(55.0):
            omega = max(-1.2, min(1.2, 1.6 * alpha))
            return Twist(0.0, omega), False

        speed = config.max_speed
        speed *= max(0.20, 1.0 - 0.85 * min(1.0, abs(alpha) / 1.4))
        speed *= min(1.0, max(0.25, goal_distance / config.slowdown_distance))

        lookahead_distance = max(
            config.min_lookahead,
            math.hypot(target_x - pose.x, target_y - pose.y),
        )
        omega = 2.0 * speed * math.sin(alpha) / lookahead_distance
        omega = max(-config.max_omega, min(config.max_omega, omega))
        return Twist(speed, omega), False


class VelocityLimiter:
    """Limit commanded velocity changes using simulation time."""

    def __init__(self, config: DriveConfig) -> None:
        self.config = config
        self._last = Twist()
        self._last_t: float | None = None

    def reset(self, t: float | None = None) -> None:
        self._last = Twist()
        self._last_t = t

    def update(self, target: Twist, now: float) -> Twist:
        if self._last_t is None or now <= self._last_t:
            self._last_t = now
            self._last = Twist()
            return self._last
        dt = min(now - self._last_t, 0.25)
        self._last_t = now
        linear = _move_towards(
            self._last.linear, target.linear, self.config.max_linear_accel * dt
        )
        angular = _move_towards(
            self._last.angular, target.angular, self.config.max_angular_accel * dt
        )
        self._last = Twist(linear, angular)
        return self._last


def _move_towards(current: float, target: float, maximum_delta: float) -> float:
    delta = max(-maximum_delta, min(maximum_delta, target - current))
    return current + delta


def rotate_towards(current_yaw: float, target_yaw: float, gain: float = 2.0) -> Twist:
    error = angle_diff(target_yaw, current_yaw)
    omega = max(-1.2, min(1.2, gain * error))
    if abs(error) < math.radians(2.0):
        return Twist(0.0, 0.0)
    return Twist(0.0, omega)


def turn_rate_for_test(angular: float, config: DriveConfig) -> tuple[float, float]:
    """Small helper used by unit tests and diagnostics."""
    return Twist(0.0, angular).wheels(config)
