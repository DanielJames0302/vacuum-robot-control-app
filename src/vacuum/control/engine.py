"""Mission state machine: plan, drive, align, mark, recover."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from vacuum.control.pure_pursuit import DriveConfig, PurePursuit, Twist, rotate_towards
from vacuum.model.gridmap import OccupancyMap, PlanningGrid
from vacuum.model.pose import Pose, angle_diff
from vacuum.planning.astar import plan_world_path
from vacuum.planning.smoother import line_of_sight, resample, simplify_path
from vacuum.sim.types import Goal, RobotState

GOAL_POSITION_TOLERANCE = 0.05  # metres, per the assignment
GOAL_YAW_TOLERANCE = math.radians(5.0)


class MissionState(str, Enum):
    IDLE = "idle"
    READY = "ready"
    DRIVING = "driving"
    FINAL_APPROACH = "final approach"
    ALIGNING = "aligning"
    MARKING = "verifying goal"
    WAIT_MARK = "marking goal"
    RECOVERING = "recovering"
    DONE = "done"
    ABORTED = "aborted"


@dataclass
class EngineOutput:
    twist: Twist = field(default_factory=Twist)
    mark_goal: str | None = None
    state: MissionState = MissionState.IDLE
    status: str = ""


class MissionEngine:
    def __init__(self, config: DriveConfig | None = None) -> None:
        self._base_config = config or DriveConfig()
        self.config = DriveConfig(**vars(self._base_config))
        self.occupancy: OccupancyMap | None = None
        self.robot_info: dict = {}
        self.goals: dict[str, Goal] = {}
        self.goal_order: list[str] = []
        self.grid: PlanningGrid | None = None
        self.pursuit = PurePursuit(self.config)

        self.state = MissionState.IDLE
        self.current_goal: Goal | None = None
        self.path: list[tuple[float, float]] = []
        self._path_length = 0.0
        self.marked: set[str] = set()
        self.events: list[str] = []

        self._verify_count = 0
        self._mark_attempts = 0
        self._recovery_start = 0.0
        self._recovery_attempts = 0
        self._last_recovery_end = -10.0
        self._best_distance = math.inf
        self._last_progress_t: float | None = None
        self._progress_index = 0
        self._ignore_depth = False
        self._extra_obstacles: list[tuple[float, float, float]] = []
        self.use_depth = True
        self._depth_hits: np.ndarray | None = None
        self._depth_dirty = False
        self._last_depth_replan = -10.0
        self._last_pose = Pose(0.0, 0.0, 0.0)
        self._last_state_t = 0.0
        self.bump_count = 0
        self._status = "waiting for scene"

    # ----------------------------------------------------------------- setup
    def set_scene(
        self,
        occupancy: OccupancyMap,
        robot_info: dict,
        goals: list[Goal],
        start: Pose,
    ) -> None:
        self.occupancy = occupancy
        self.robot_info = dict(robot_info)
        self.goals = {goal.name: goal for goal in goals}
        self.goal_order = [goal.name for goal in goals]
        self.config = DriveConfig(**vars(self._base_config))
        self.pursuit.config = self.config
        self.config.wheel_base = float(robot_info.get("wheel_base", self.config.wheel_base))
        self.config.wheel_radius = float(robot_info.get("wheel_radius", self.config.wheel_radius))
        self.config.max_wheel_speed = float(
            robot_info.get("max_wheel_speed", self.config.max_wheel_speed)
        )
        self.config.max_speed = min(
            self.config.max_speed, float(robot_info.get("max_linear_speed", 0.35)) * 0.9
        )
        self.config.max_omega = min(
            self.config.max_omega, float(robot_info.get("max_angular_speed", 2.0)) * 0.9
        )
        self.config.max_linear_accel = float(
            robot_info.get("max_linear_acceleration", robot_info.get("max_linear_accel", 1.0))
        )
        self.config.max_angular_accel = float(
            robot_info.get("max_angular_acceleration", robot_info.get("max_angular_accel", 6.0))
        )
        self._start = start
        self._last_pose = start
        self._last_state_t = 0.0
        self.bump_count = 0
        self._depth_hits = np.zeros((occupancy.height, occupancy.width), dtype=np.uint16)
        self._depth_dirty = False
        self.reset_mission(keep_scene=True)
        self.state = MissionState.READY
        self._status = "scene loaded, pick a goal"
        self.events.append(f"scene loaded; {len(goals)} goals")

    def reset_mission(self, keep_scene: bool = True) -> None:
        self.state = MissionState.READY if keep_scene else MissionState.IDLE
        self.current_goal = None
        self.path = []
        self._path_length = 0.0
        self.pursuit.set_path([])
        self.marked.clear()
        self._verify_count = 0
        self._mark_attempts = 0
        self._recovery_attempts = 0
        self._extra_obstacles.clear()
        self.grid = None
        self._status = "ready"
        self._best_distance = math.inf
        self._last_progress_t = None

    @property
    def robot_radius(self) -> float:
        return float(self.robot_info.get("radius", 0.17))

    @property
    def robot_height(self) -> float:
        return float(self.robot_info.get("height", 0.09))

    def _planning_grid(self, include_depth: bool = True) -> PlanningGrid | None:
        if self.occupancy is None:
            return None
        use_depth = self.use_depth and include_depth and not self._ignore_depth
        return self.occupancy.build_planning_grid(
            self.robot_radius + 0.01,
            margin=0.05,
            extra_obstacles=self._extra_obstacles,
            extra_occupied=self.depth_mask if use_depth else None,
        )

    # --------------------------------------------------------------- planning
    def plan_goal(self, name: str) -> bool:
        if self.occupancy is None or name not in self.goals:
            self.events.append(f"unknown goal {name!r}")
            return False
        if name in self.marked:
            self.events.append(f"goal {name!r} was already marked in this run")
            return False
        goal = self.goals[name]
        self._ignore_depth = False
        start_pose = self._last_pose if hasattr(self, "_last_pose") else self._start
        points = self._build_path(start_pose, goal.pose)
        if points is None:
            self.events.append(f"no path found to {name!r}")
            return False
        self.current_goal = goal
        self._set_path(points)
        self.state = MissionState.DRIVING
        self._verify_count = 0
        self._mark_attempts = 0
        self._recovery_attempts = 0
        self._best_distance = math.inf
        self._last_progress_t = None
        self._progress_index = 0
        self._status = f"driving to {name}"
        self.events.append(f"planned {len(points)} waypoints to {name}")
        return True

    def cancel(self) -> None:
        self.pursuit.set_path([])
        self.path = []
        self._path_length = 0.0
        self.current_goal = None
        self.state = MissionState.READY
        self._status = "cancelled"
        self.events.append("mission cancelled")

    def _replan(self) -> bool:
        if self.current_goal is None or self.occupancy is None:
            return False
        points = self._build_path(self._last_pose, self.current_goal.pose)
        if points is None:
            return False
        self._set_path(points)
        self._best_distance = math.inf
        self._last_progress_t = None
        self._progress_index = 0
        return True

    def _build_path(self, start: Pose, goal: Pose) -> list[tuple[float, float]] | None:
        """Plan and apply the common remembered-obstacle/depth fallbacks."""
        if self.occupancy is None:
            return None
        self.grid = self._planning_grid()
        if self.grid is None:
            return None
        points = plan_world_path(self.occupancy, self.grid, start, goal)
        if points is None and self._extra_obstacles:
            self.events.append("no path with remembered obstacles; retrying without them")
            self._extra_obstacles.clear()
            self.grid = self._planning_grid()
            if self.grid is not None:
                points = plan_world_path(self.occupancy, self.grid, start, goal)
        if points is None and self.use_depth and not self._ignore_depth:
            self.events.append("no path with camera obstacles; ignoring them for this goal")
            self._ignore_depth = True
            self.grid = self._planning_grid()
            if self.grid is not None:
                points = plan_world_path(self.occupancy, self.grid, start, goal)
        if points is None:
            return None
        assert self.grid is not None
        points = simplify_path(self.occupancy, self.grid, points)
        return resample(points, spacing=0.05)

    def _set_path(self, points: list[tuple[float, float]]) -> None:
        self.path = points
        self._path_length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(points, points[1:], strict=False)
        )
        self._install_visibility()
        self.pursuit.set_path(points)

    # ------------------------------------------------------------------- step
    def step(self, state: RobotState, now: float | None = None) -> EngineOutput:
        self._last_pose = state.pose
        self._last_state_t = state.t
        now = state.t if now is None else now

        if self.state in (MissionState.IDLE, MissionState.READY, MissionState.DONE,
                          MissionState.ABORTED):
            return EngineOutput(Twist(), None, self.state, self._status)

        if self.state is MissionState.WAIT_MARK:
            return EngineOutput(Twist(), None, self.state, self._status)

        if self.state is MissionState.RECOVERING:
            return self._step_recovery(state)

        if state.bumper and now - self._last_recovery_end > 1.0:
            return self._start_recovery(state, "bumper contact")

        if self.state is MissionState.DRIVING:
            return self._step_driving(state, now)
        if self.state is MissionState.FINAL_APPROACH:
            return self._step_final_approach(state, now)
        if self.state is MissionState.ALIGNING:
            return self._step_aligning(state)
        if self.state is MissionState.MARKING:
            return self._step_marking(state)

        return EngineOutput(Twist(), None, self.state, self._status)

    def _install_visibility(self) -> None:
        if self.occupancy is None or self.grid is None:
            self.pursuit.visibility = None
            return
        occupancy, grid = self.occupancy, self.grid
        self.pursuit.visibility = lambda a, b: line_of_sight(
            occupancy, grid, a, b, min_clearance=1
        )

    # ------------------------------------------------------------ depth fusion
    @property
    def depth_mask(self) -> np.ndarray | None:
        if self._depth_hits is None:
            return None
        return self._depth_hits >= 2

    def observe_depth(self, points_world: np.ndarray) -> int:
        """Accumulate collidable-height points observed by the RGBD camera."""
        if not self.use_depth or self.occupancy is None or self._depth_hits is None:
            return 0
        if len(points_world) == 0:
            return 0
        xs = points_world[:, 0]
        ys = points_world[:, 1]
        cols = np.floor((xs - self.occupancy.origin[0]) / self.occupancy.resolution).astype(int)
        rows = (
            self.occupancy.height
            - 1
            - np.floor((ys - self.occupancy.origin[1]) / self.occupancy.resolution).astype(int)
        )
        inside = (
            (rows >= 0)
            & (cols >= 0)
            & (rows < self.occupancy.height)
            & (cols < self.occupancy.width)
        )
        if not np.any(inside):
            return 0
        # Ignore points that land on (or next to) obstacles already on the map;
        # only genuinely new obstacles should trigger replanning.
        known = self.occupancy.inflate(self.robot_radius + 0.05)
        inside_rows = rows[inside]
        inside_cols = cols[inside]
        interesting_inside = ~known[inside_rows, inside_cols]
        if not np.any(interesting_inside):
            return 0
        flat = np.unique(
            inside_rows[interesting_inside] * self.occupancy.width
            + inside_cols[interesting_inside]
        )
        hits_flat = self._depth_hits.reshape(-1)
        previous = hits_flat[flat].copy()
        np.add.at(hits_flat, flat, 1)
        np.minimum(hits_flat, 200, out=hits_flat)
        added = int(np.count_nonzero((previous < 2) & (hits_flat[flat] >= 2)))
        if added:
            self._depth_dirty = True
        return added

    def _depth_near_path(self, horizon: float = 1.6, radius_m: float = 0.40) -> bool:
        if self._depth_hits is None or self.occupancy is None or not self.path:
            return False
        mask = self._depth_hits >= 2
        if not np.any(mask):
            return False
        radius_cells = int(math.ceil(radius_m / self.occupancy.resolution))
        pose = self._last_pose
        for x, y in self.path[self.pursuit.index:]:
            if math.hypot(x - pose.x, y - pose.y) > horizon:
                break
            cell = self.occupancy.world_to_grid(x, y)
            if cell is None:
                continue
            row, col = cell
            r0, r1 = max(0, row - radius_cells), min(self.occupancy.height, row + radius_cells + 1)
            c0, c1 = max(0, col - radius_cells), min(self.occupancy.width, col + radius_cells + 1)
            if np.any(mask[r0:r1, c0:c1]):
                return True
        return False

    def _clearance_at(self, pose: Pose) -> int:
        if self.grid is None or self.occupancy is None:
            return 10
        cell = self.occupancy.world_to_grid(pose.x, pose.y)
        if cell is None:
            return 0
        return int(self.grid.clearance[cell])

    # ------------------------------------------------------------- behaviours
    def _step_driving(self, state: RobotState, now: float) -> EngineOutput:
        assert self.current_goal is not None
        goal = self.current_goal.pose
        distance = state.pose.distance_to(goal)

        if self.pursuit.index > self._progress_index:
            self._progress_index = self.pursuit.index
            self._last_progress_t = now
        elif self._last_progress_t is None:
            self._last_progress_t = now
        elif now - self._last_progress_t > 12.0:
            self.events.append("no path progress for 12 s; treating it as blocked")
            return self._start_recovery(state, "stuck")

        if distance < 0.14:
            self.state = MissionState.FINAL_APPROACH
            self._status = "final approach"
            return self._step_final_approach(state, now)

        if self._depth_dirty and now - self._last_depth_replan > 3.0:
            self._last_depth_replan = now
            self._depth_dirty = False
            if self._depth_near_path() and self._replan():
                self.events.append("replanned around obstacle seen by the camera")

        twist, finished = self.pursuit.update(state.pose)
        if finished:
            self.state = MissionState.FINAL_APPROACH
            return self._step_final_approach(state, now)

        # Slow down in tight spots so the follower does not clip corners.
        clearance = self._clearance_at(state.pose)
        if clearance <= 2 and twist.linear > 0.0:
            twist.linear *= 0.55
        elif clearance <= 4 and twist.linear > 0.0:
            twist.linear *= 0.8

        self._status = f"driving to {self.current_goal.name} ({distance:.2f} m)"
        return EngineOutput(twist, None, self.state, self._status)

    def _step_final_approach(self, state: RobotState, now: float) -> EngineOutput:
        assert self.current_goal is not None
        goal = self.current_goal.pose
        distance = state.pose.distance_to(goal)
        heading_error = angle_diff(state.pose.heading_to(goal), state.pose.yaw)

        if distance <= 0.035:
            self.state = MissionState.ALIGNING if self.current_goal.has_yaw else MissionState.MARKING
            self._verify_count = 0
            return self.step(state, now)

        speed = min(0.10, max(0.025, 1.4 * distance))
        if abs(heading_error) > math.radians(50.0):
            speed = min(speed, 0.02)
            omega = 1.0 if heading_error > 0 else -1.0
        else:
            omega = max(-0.9, min(0.9, 2.5 * heading_error))
        self._status = f"approaching {self.current_goal.name} ({distance * 100:.1f} cm)"
        return EngineOutput(Twist(speed, omega), None, self.state, self._status)

    def _step_aligning(self, state: RobotState) -> EngineOutput:
        assert self.current_goal is not None
        goal = self.current_goal.pose
        yaw_error = angle_diff(goal.yaw, state.pose.yaw)
        if abs(yaw_error) < math.radians(2.0):
            self.state = MissionState.MARKING
            self._verify_count = 0
            return self._step_marking(state)
        twist = rotate_towards(state.pose.yaw, goal.yaw)
        self._status = f"aligning yaw ({math.degrees(yaw_error):+.1f} deg)"
        return EngineOutput(twist, None, self.state, self._status)

    def _step_marking(self, state: RobotState) -> EngineOutput:
        assert self.current_goal is not None
        goal = self.current_goal
        distance = state.pose.distance_to(goal.pose)
        yaw_error = abs(angle_diff(goal.pose.yaw, state.pose.yaw)) if goal.has_yaw else 0.0

        within_position = distance <= GOAL_POSITION_TOLERANCE
        within_yaw = (not goal.has_yaw) or yaw_error <= GOAL_YAW_TOLERANCE
        if within_position and within_yaw:
            self._verify_count += 1
            if self._verify_count >= 4:
                self.state = MissionState.WAIT_MARK
                self._status = f"marking {goal.name}"
                self.events.append(f"goal {goal.name!r} reached; calling mark_goal")
                return EngineOutput(Twist(), goal.name, self.state, self._status)
            self._status = f"verifying {goal.name} ({self._verify_count}/4)"
            return EngineOutput(Twist(), None, self.state, self._status)

        self._verify_count = 0
        if not within_position:
            if distance > 0.10:
                self.state = MissionState.DRIVING
                self._replan()
            else:
                self.state = MissionState.FINAL_APPROACH
            return self.step(state, state.t)
        if not within_yaw:
            self.state = MissionState.ALIGNING
            return self.step(state, state.t)
        return EngineOutput(Twist(), None, self.state, self._status)

    # ---------------------------------------------------------------- recovery
    def _start_recovery(self, state: RobotState, reason: str) -> EngineOutput:
        if self._recovery_attempts >= 8:
            self.state = MissionState.ABORTED
            self._status = "goal aborted: too many recovery attempts"
            self.events.append(self._status)
            return EngineOutput(Twist(), None, self.state, self._status)
        self._recovery_attempts += 1
        if reason == "bumper contact":
            self.bump_count += 1
        if reason == "bumper contact":
            front = state.pose.advanced(0.22)
            self._extra_obstacles.append((front.x, front.y, 0.15))
        self._recovery_start = state.t
        self._last_recovery_end = state.t
        self.state = MissionState.RECOVERING
        self.pursuit.set_path([])
        self._status = f"recovering from {reason}"
        self.events.append(
            f"recovery {self._recovery_attempts}: {reason} near "
            f"({state.pose.x:.2f}, {state.pose.y:.2f})"
        )
        return EngineOutput(Twist(-0.10, 0.0), None, self.state, self._status)

    def _step_recovery(self, state: RobotState) -> EngineOutput:
        elapsed = state.t - self._recovery_start
        if elapsed < 1.0:
            return EngineOutput(Twist(-0.12, 0.0), None, self.state, self._status)
        if elapsed < 1.8:
            return EngineOutput(Twist(0.0, 0.9), None, self.state, self._status)

        if self._replan():
            self.state = MissionState.DRIVING
            self._last_recovery_end = state.t
            self._last_progress_t = None
            self._status = "replanned after recovery"
            return EngineOutput(Twist(), None, self.state, self._status)

        self._recovery_attempts += 1
        if self._recovery_attempts >= 8:
            self.state = MissionState.ABORTED
            self._status = "goal aborted after repeated recovery failures"
            self.events.append(self._status)
            return EngineOutput(Twist(), None, self.state, self._status)

        self._recovery_start = state.t
        self._last_recovery_end = state.t
        self.events.append("replan failed; backing off and rotating before retrying")
        return EngineOutput(Twist(-0.12, 0.0), None, self.state, self._status)

    # ------------------------------------------------------------------ misc
    def confirm_mark(self, success: bool) -> None:
        if self.current_goal is None:
            return
        name = self.current_goal.name
        if success:
            self.marked.add(name)
            self.state = MissionState.DONE
            self._status = f"reached {name}"
            self.events.append(f"mark_goal({name!r}) accepted")
        else:
            self._mark_attempts += 1
            if self._mark_attempts >= 3:
                self.state = MissionState.ABORTED
                self._status = f"mark_goal failed for {name}"
                self.events.append(self._status)
            else:
                self.state = MissionState.ALIGNING
                self._status = "retrying mark_goal"
                self._verify_count = 0

    @property
    def distance_to_goal(self) -> float | None:
        if self.current_goal is None or not hasattr(self, "_last_pose"):
            return None
        return self._last_pose.distance_to(self.current_goal.pose)

    def snapshot(self) -> dict:
        goal = self.current_goal
        return {
            "state": self.state.value,
            "status": self._status,
            "goal": goal.name if goal else None,
            "goal_pose": goal.pose.as_list() if goal else None,
            "goal_has_yaw": goal.has_yaw if goal else False,
            "path": list(self.path),
            "path_length": self._path_length,
            "distance": self.distance_to_goal,
            "marked": sorted(self.marked),
            "bumps": self.bump_count,
            "depth_cells": int(np.count_nonzero(self.depth_mask)) if self._depth_hits is not None else 0,
        }

    def drain_events(self) -> list[str]:
        events = self.events
        self.events = []
        return events
