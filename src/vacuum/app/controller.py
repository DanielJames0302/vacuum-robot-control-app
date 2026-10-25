"""Synchronous mission controller shared by the Qt worker and headless CLI."""
from __future__ import annotations

import datetime as _dt
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from vacuum.control.engine import EngineOutput, MissionEngine
from vacuum.control.pure_pursuit import Twist, VelocityLimiter
from vacuum.model.depth import depth_points_world
from vacuum.model.gridmap import OccupancyMap
from vacuum.model.pose import Pose
from vacuum.recording.recorder import Recorder
from vacuum.sim.client import SimClient
from vacuum.sim.types import CameraFrame, Goal, RobotState, SceneData

log = logging.getLogger(__name__)

COMMAND_PERIOD = 1.0 / 30.0


@dataclass(frozen=True)
class SceneBundle:
    scene: SceneData
    occupancy: OccupancyMap
    robot_info: dict
    camera_info: dict


class ControllerSnapshot(TypedDict):
    connected: bool
    fault: str | None
    scene: str | None
    scene_name: str
    pose: list[float] | None
    bumper: bool
    t: float
    state: str
    status: str
    goal: str | None
    goal_pose: list[float] | None
    goal_has_yaw: bool
    path: list[tuple[float, float]]
    path_length: float
    distance: float | None
    marked: list[str]
    bumps: int
    depth_cells: int
    use_depth: bool
    robot_radius: float
    goals: list[dict]
    state_gaps: int
    camera_gaps: int
    stale_states: int
    stale_cameras: int
    recording: bool
    recording_dir: str | None
    recorded_frames: int


class MissionController:
    """Owns all simulator sockets and the mission engine.

    Call :meth:`tick` from a single thread at 20-50 Hz. It polls the streams,
    advances the engine, and sends wheel commands at 30 Hz (the simulator's
    watchdog stops the robot after 250 ms without commands).
    """

    def __init__(
        self,
        host: str = "localhost",
        port_base: int = 5555,
        recordings_dir: Path = Path("recordings"),
        log_callback: Callable[[str], None] | None = None,
        use_depth: bool | None = None,
    ) -> None:
        self.sim = SimClient(host=host, port_base=port_base)
        self.engine = MissionEngine()
        self._depth_override = use_depth
        self.recordings_dir = Path(recordings_dir)
        self.log_callback = log_callback
        self.scene: SceneData | None = None
        self.occupancy: OccupancyMap | None = None
        self.robot_info: dict = {}
        self.camera_info: dict = {}
        self.latest_state: RobotState | None = None
        self.latest_camera: CameraFrame | None = None
        self.latest_preview: CameraFrame | None = None
        self._preview_counter = 0
        self._camera_counter = 0
        self.recorder: Recorder | None = None
        self.recording_dir: Path | None = None
        self.seed: int | None = None
        self._last_output = EngineOutput()
        self._last_command_time = 0.0
        self._last_state_time = 0.0
        self._last_state_seq: int | None = None
        self._last_camera_seq: int | None = None
        self._awaiting_fresh_state = False
        self._stale_states = 0
        self._stale_cameras = 0
        self._fault: str | None = None
        self._events: list[str] = []
        self._limiter = VelocityLimiter(self.engine.config)

    # ------------------------------------------------------------------ setup
    def _log(self, message: str) -> None:
        log.info(message)
        self._events.append(message)
        if self.log_callback is not None:
            try:
                self.log_callback(message)
            except Exception:  # noqa: BLE001 - logging must not break control
                log.exception("log callback failed")

    def drain_events(self) -> list[str]:
        events = self._events
        self._events = []
        return events

    def ping(self) -> bool:
        try:
            self.sim.ping()
            self._fault = None
            return True
        except Exception as exc:  # noqa: BLE001
            self._fault = f"simulator unreachable: {exc}"
            return False

    def list_scenes(self) -> list[dict]:
        return self.sim.list_scenes()

    def load_scene(self, scene_id: str, seed: int | None = None) -> SceneBundle:
        self.stop_recording()
        result = self.sim.load_scene(scene_id, seed)
        return self._apply_scene_result(scene_id, seed, result)

    def reset_scene(self, seed: int | None = None) -> SceneBundle:
        self.stop_recording()
        if self.scene is None:
            raise RuntimeError("no scene loaded")
        result = self.sim.reset(seed)
        return self._apply_scene_result(self.scene.id, seed, result)

    def _apply_scene_result(
        self, scene_id: str, seed: int | None, result: dict
    ) -> SceneBundle:
        goals = [Goal.from_json(goal) for goal in result.get("goals", [])]
        meta, png = self.sim.get_map()
        occupancy = OccupancyMap.from_png(meta, png)
        robot_info = self.sim.get_robot_info()
        camera_info = self.sim.get_camera_info()

        scenes = self.sim.list_scenes()
        info = next((s for s in scenes if str(s.get("id")) == str(scene_id)), {})
        scene = SceneData(
            id=str(scene_id),
            name=str(info.get("name", "")),
            difficulty=str(info.get("difficulty", "")),
            start=Pose.from_dict(result["start"]),
            goals=goals,
        )
        self.engine.use_depth = True if self._depth_override is None else self._depth_override
        self._log(
            f"depth obstacle memory {'enabled' if self.engine.use_depth else 'disabled'}"
            f" for {scene.difficulty or 'unknown'} scene"
        )
        self.scene = scene
        self.occupancy = occupancy
        self.robot_info = robot_info
        self.camera_info = camera_info
        self.seed = seed
        self.sim.drain_streams()
        self.latest_state = None
        self.latest_camera = None
        self.latest_preview = None
        self._last_state_seq = None
        self._last_camera_seq = None
        self._awaiting_fresh_state = True
        self._stale_states = 0
        self._stale_cameras = 0
        self.engine.set_scene(occupancy, robot_info, goals, scene.start)
        self._limiter = VelocityLimiter(self.engine.config)
        self._log(
            f"loaded scene {scene_id} ({scene.name}); start "
            f"({scene.start.x:.2f}, {scene.start.y:.2f}, {scene.start.yaw:.2f})"
        )
        return SceneBundle(scene, occupancy, robot_info, camera_info)

    # -------------------------------------------------------------- missions
    def start_goal(self, name: str) -> bool:
        started = self.engine.plan_goal(name)
        if started:
            self._log(f"mission started: {name}")
        return started

    def cancel(self) -> None:
        self.engine.cancel()
        self._limiter.reset()
        self._send(Twist())
        self._log("cancelled")

    # ---------------------------------------------------------------- ticking
    def tick(self) -> None:
        now = time.monotonic()
        messages = self.sim.poll(timeout_ms=5)
        valid_states: list[RobotState] = []
        camera_messages: list[CameraFrame] = []
        for kind, message in messages:
            if kind == "state":
                assert isinstance(message, RobotState)
                if not self._valid_state(message):
                    self._stale_states += 1
                    continue
                valid_states.append(message)
            elif kind == "camera":
                assert isinstance(message, CameraFrame)
                camera_messages.append(message)

        if valid_states:
            state = max(valid_states, key=lambda item: item.seq)
            self._stale_states += len(valid_states) - 1
            self.latest_state = state
            self._last_state_time = now
            self._last_state_seq = state.seq
            self._awaiting_fresh_state = False
            try:
                output = self.engine.step(state)
                output.twist = self._limiter.update(output.twist, state.t)
                self._last_output = output
            except Exception as exc:  # noqa: BLE001 - never kill the loop
                self._fault = f"engine error: {exc}"
                log.exception("engine step failed")
                self._last_output = EngineOutput(Twist(), None)
            if self._last_output.mark_goal:
                self._handle_mark(self._last_output.mark_goal)

        new_camera = False
        for message in camera_messages:
            if not self._valid_camera(message):
                self._stale_cameras += 1
                continue
            self._last_camera_seq = message.seq
            self.latest_camera = message
            new_camera = True
            if self.recorder is not None:
                self.recorder.add(message)
            self._camera_counter += 1
            if self.engine.use_depth and self._camera_counter % 5 == 0:
                try:
                    points = depth_points_world(
                        message,
                        self.camera_info,
                        max_height=self.engine.robot_height + 0.01,
                    )
                    self.engine.observe_depth(points)
                except Exception as exc:  # noqa: BLE001
                    log.debug("depth projection failed: %s", exc)
            self._preview_counter += 1
            if self._preview_counter % 3 == 0:
                self.latest_preview = message

        for event in self.engine.drain_events():
            self._log(event)

        if now - self._last_command_time >= COMMAND_PERIOD:
            self._last_command_time = now
            self._send(self._last_output.twist if self.latest_state else Twist())

        if new_camera and self.recorder is not None and self.recorder.error:
            self._fault = self.recorder.error

    def _send(self, twist: Twist) -> None:
        try:
            if not self.sim.send_wheels(*twist.wheels(self.engine.config)):
                self._fault = "command queue full; wheel command was dropped"
        except Exception as exc:  # noqa: BLE001
            self._fault = f"command send failed: {exc}"

    def _valid_state(self, state: RobotState) -> bool:
        if self.scene is None or state.scene != self.scene.id:
            return False
        if self._last_state_seq is not None and state.seq <= self._last_state_seq:
            return False
        if self._awaiting_fresh_state:
            # A late packet from the prior run can arrive after the socket drain.
            # Fresh state starts near the exact pose returned by load/reset.
            if state.pose.distance_to(self.scene.start) > 0.08:
                return False
        return True

    def _valid_camera(self, frame: CameraFrame) -> bool:
        if self.latest_state is None or self._awaiting_fresh_state:
            return False
        if self._last_camera_seq is not None and frame.seq <= self._last_camera_seq:
            return False
        return abs(frame.t - self.latest_state.t) <= 2.0

    def _handle_mark(self, name: str) -> None:
        # A timeout is ambiguous: the simulator may have recorded the mark before
        # its reply was lost. Retries are bounded in MissionEngine, and the design
        # note documents that this transport cannot provide exactly-once RPC.
        success = False
        try:
            self.sim.mark_goal(name)
            success = True
        except Exception as exc:  # noqa: BLE001
            self._fault = f"mark_goal({name}) failed: {exc}"
            self._log(self._fault)
        self.engine.confirm_mark(success)

    # -------------------------------------------------------------- recording
    def start_recording(self) -> Path | None:
        if self.scene is None or self.camera_info is None:
            self._log("cannot record before a scene is loaded")
            return None
        if self.recorder is not None:
            return self.recording_dir
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = self.recordings_dir / f"scene{self.scene.id}_{stamp}"
        try:
            recorder = Recorder(
                directory,
                width=int(self.camera_info.get("width", 640)),
                height=int(self.camera_info.get("height", 480)),
                fps=float(self.camera_info.get("rate", 15.0)),
                camera_info=self.camera_info,
                scene=self.scene.id,
                seed=self.seed,
            )
            recorder.start()
        except Exception as exc:  # noqa: BLE001
            self._fault = f"recording failed to start: {exc}"
            self._log(self._fault)
            return None
        self.recorder = recorder
        self.recording_dir = directory
        self._log(f"recording to {directory}")
        return directory

    def stop_recording(self) -> Path | None:
        if self.recorder is None:
            return None
        directory = self.recorder.close()
        self._log(f"recording saved to {directory}")
        self.recorder = None
        self.recording_dir = None
        return directory

    # --------------------------------------------------------------- snapshot
    @property
    def connected(self) -> bool:
        return (
            self.latest_state is not None
            and (time.monotonic() - self._last_state_time) < 1.0
        )

    @property
    def fault(self) -> str | None:
        return self._fault

    def snapshot(self) -> ControllerSnapshot:
        state = self.latest_state
        engine = self.engine.snapshot()
        return {
            "connected": self.connected,
            "fault": self._fault,
            "scene": self.scene.id if self.scene else None,
            "scene_name": self.scene.name if self.scene else "",
            "pose": state.pose.as_list() if state else None,
            "bumper": state.bumper if state else False,
            "t": state.t if state else 0.0,
            "state": engine["state"],
            "status": engine["status"],
            "goal": engine["goal"],
            "goal_pose": engine["goal_pose"],
            "goal_has_yaw": engine["goal_has_yaw"],
            "path": engine["path"],
            "path_length": engine["path_length"],
            "distance": engine["distance"],
            "marked": engine["marked"],
            "bumps": engine["bumps"],
            "depth_cells": engine.get("depth_cells", 0),
            "use_depth": self.engine.use_depth,
            "robot_radius": self.engine.robot_radius,
            "goals": [goal.as_dict() for goal in (self.scene.goals if self.scene else [])],
            "state_gaps": self.sim.state_gaps,
            "camera_gaps": self.sim.camera_gaps,
            "stale_states": self._stale_states,
            "stale_cameras": self._stale_cameras,
            "recording": self.recorder is not None,
            "recording_dir": str(self.recording_dir) if self.recording_dir else None,
            "recorded_frames": self.recorder.frames_written if self.recorder else 0,
        }

    def close(self) -> None:
        self.stop_recording()
        try:
            self.sim.stop()
        except Exception:  # noqa: BLE001
            pass
        self.sim.close()
