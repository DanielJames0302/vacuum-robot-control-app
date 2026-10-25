"""ZeroMQ transport for the vacuum simulator.

Ports (relative to ``port_base``):
    0 - RPC (REQ)
    1 - commands (PUSH)
    2 - robot state (SUB)
    3 - RGBD camera (SUB)

All sockets in a :class:`SimClient` must be used from a single thread.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import zmq

from vacuum.sim.rpc import RpcClient
from vacuum.sim.types import CameraFrame, RobotState

log = logging.getLogger(__name__)


class SimClient:
    def __init__(
        self,
        host: str = "localhost",
        port_base: int = 5555,
        rpc_timeout_ms: int = 3000,
    ) -> None:
        self.host = host
        self.port_base = port_base
        self.context = zmq.Context.instance()

        self.rpc = RpcClient(host, port_base, timeout_ms=rpc_timeout_ms, context=self.context)

        self.command = self.context.socket(zmq.PUSH)
        self.command.setsockopt(zmq.LINGER, 0)
        self.command.connect(self._url(1))

        self.state = self.context.socket(zmq.SUB)
        self.state.setsockopt(zmq.LINGER, 0)
        self.state.setsockopt(zmq.RCVHWM, 1000)
        self.state.setsockopt(zmq.SUBSCRIBE, b"")
        self.state.connect(self._url(2))

        self.camera = self.context.socket(zmq.SUB)
        self.camera.setsockopt(zmq.LINGER, 0)
        self.camera.setsockopt(zmq.RCVHWM, 30)
        self.camera.setsockopt(zmq.SUBSCRIBE, b"")
        self.camera.connect(self._url(3))

        self._poller = zmq.Poller()
        self._poller.register(self.state, zmq.POLLIN)
        self._poller.register(self.camera, zmq.POLLIN)

        self.last_state_seq: int | None = None
        self.last_camera_seq: int | None = None
        self.state_gaps = 0
        self.camera_gaps = 0
        self.malformed = 0

    def _url(self, offset: int) -> str:
        return f"tcp://{self.host}:{self.port_base + offset}"

    # ------------------------------------------------------------------ RPC
    def call(self, method: str, params: dict[str, Any] | None = None, timeout_ms: int | None = None):
        return self.rpc.call(method, params, timeout_ms=timeout_ms)

    def ping(self) -> dict:
        result, _ = self.call("ping", timeout_ms=1000)
        return result or {}

    def list_scenes(self) -> list[dict]:
        result, _ = self.call("list_scenes")
        return list(result or [])

    def load_scene(self, scene_id: str, seed: int | None = None) -> dict:
        params: dict[str, Any] = {"id": scene_id}
        if seed is not None:
            params["seed"] = int(seed)
        result, _ = self.call("load_scene", params, timeout_ms=15000)
        self.reset_stream_tracking()
        return result or {}

    def reset(self, seed: int | None = None) -> dict:
        params: dict[str, Any] = {}
        if seed is not None:
            params["seed"] = int(seed)
        result, _ = self.call("reset", params, timeout_ms=15000)
        self.reset_stream_tracking()
        return result or {}

    def get_goals(self) -> list[dict]:
        result, _ = self.call("get_goals")
        return list(result or [])

    def get_map(self) -> tuple[dict, bytes]:
        result, blobs = self.call("get_map", timeout_ms=10000)
        if not blobs:
            raise RuntimeError("get_map returned no PNG payload")
        return result or {}, blobs[0]

    def get_robot_info(self) -> dict:
        result, _ = self.call("get_robot_info")
        return result or {}

    def get_camera_info(self) -> dict:
        result, _ = self.call("get_camera_info")
        return result or {}

    def mark_goal(self, name: str) -> dict:
        result, _ = self.call("mark_goal", {"name": name}, timeout_ms=3000)
        return result or {}

    # ------------------------------------------------------------- commands
    def send_command(self, command: dict[str, Any]) -> bool:
        try:
            self.command.send_json(command, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            log.warning("command queue full, dropping %s", command)
            return False

    def send_wheels(self, left: float, right: float) -> bool:
        return self.send_command({"type": "wheels", "left": float(left), "right": float(right)})

    def send_goto(self, goto_id: int, x: float, y: float, yaw: float | None = None) -> bool:
        command: dict[str, Any] = {"type": "goto", "id": int(goto_id), "x": float(x), "y": float(y)}
        if yaw is not None:
            command["yaw"] = float(yaw)
        return self.send_command(command)

    def send_cancel(self, goto_id: int) -> bool:
        return self.send_command({"type": "cancel", "id": int(goto_id)})

    def stop(self) -> None:
        self.send_wheels(0.0, 0.0)

    # --------------------------------------------------------------- streams
    def drain_streams(self) -> None:
        """Discard messages queued before a load/reset (they belong to the old run)."""
        for socket in (self.state, self.camera):
            while True:
                try:
                    socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

    def reset_stream_tracking(self) -> None:
        self.last_state_seq = None
        self.last_camera_seq = None
        self.state_gaps = 0
        self.camera_gaps = 0

    def poll(self, timeout_ms: int = 10) -> list[tuple[str, RobotState | CameraFrame]]:
        """Return everything that arrived, newest-last per stream."""
        events = dict(self._poller.poll(timeout_ms))
        messages: list[tuple[str, RobotState | CameraFrame]] = []

        if self.state in events:
            while True:
                try:
                    frames = self.state.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    state = RobotState.from_json(json.loads(frames[0]))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    self.malformed += 1
                    continue
                if self.last_state_seq is None:
                    self.last_state_seq = state.seq
                elif state.seq > self.last_state_seq:
                    if state.seq > self.last_state_seq + 1:
                        self.state_gaps += state.seq - self.last_state_seq - 1
                    self.last_state_seq = state.seq
                messages.append(("state", state))

        if self.camera in events:
            while True:
                try:
                    frames = self.camera.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(frames) < 3:
                    self.malformed += 1
                    continue
                try:
                    header = json.loads(frames[0])
                    frame = CameraFrame.from_frames(header, frames[1], frames[2])
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    self.malformed += 1
                    continue
                if self.last_camera_seq is None:
                    self.last_camera_seq = frame.seq
                elif frame.seq > self.last_camera_seq:
                    if frame.seq > self.last_camera_seq + 1:
                        self.camera_gaps += frame.seq - self.last_camera_seq - 1
                    self.last_camera_seq = frame.seq
                messages.append(("camera", frame))

        return messages

    def close(self) -> None:
        for socket in (self.command, self.state, self.camera):
            socket.close(linger=0)
        self.rpc.close()
