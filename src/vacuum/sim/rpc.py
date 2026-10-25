"""ZeroMQ RPC helper for the simulator's request/reply channel."""
from __future__ import annotations

import json
from typing import Any

import zmq


class RpcError(RuntimeError):
    """The simulator answered with ``ok: false``."""


class RpcTimeout(RuntimeError):
    """No reply arrived in time."""


class RpcClient:
    """REQ socket wrapper with timeouts and automatic socket recovery.

    A ZeroMQ REQ socket must strictly alternate send/recv. If a reply is lost,
    the socket is stuck; recreating it is the simplest reliable recovery.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 3000,
        context: zmq.Context | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._owns_context = context is None
        self.context = context or zmq.Context.instance()
        self.socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(f"tcp://{self.host}:{self.port}")
        self.socket = socket

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> tuple[Any, list[bytes]]:
        if self.socket is None:
            self._connect()
        assert self.socket is not None
        timeout = self.timeout_ms if timeout_ms is None else timeout_ms
        payload = json.dumps({"method": method, "params": params or {}}).encode()
        self.socket.send(payload)

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        events = dict(poller.poll(timeout))
        if self.socket not in events:
            self._connect()
            raise RpcTimeout(f"no reply to {method!r} within {timeout} ms")

        frames = self.socket.recv_multipart()
        try:
            reply = json.loads(frames[0])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RpcError(f"malformed reply to {method!r}: {exc}") from exc
        if not reply.get("ok"):
            raise RpcError(str(reply.get("error", "unknown simulator error")))
        return reply.get("result"), frames[1:]

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if self._owns_context:
            self.context.term()
