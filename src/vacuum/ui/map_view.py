"""Top-down map canvas with path, robot, goals and trail."""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from vacuum.model.gridmap import OccupancyMap
from vacuum.sim.types import Goal

FREE_COLOR = 0xD9
OCCUPIED_COLOR = 0x28


class MapView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.occupancy: OccupancyMap | None = None
        self.goals: list[Goal] = []
        self.image: QImage | None = None
        self.path: list[tuple[float, float]] = []
        self.trail: deque[tuple[float, float]] = deque(maxlen=3000)
        self.pose: tuple[float, float, float] | None = None
        self.robot_radius = 0.17
        self.bumper = False
        self.goal: str | None = None
        self.marked: set[str] = set()
        self.scene_id: str | None = None
        self._scale = 1.0
        self._x0 = 0.0
        self._y0 = 0.0

    # ------------------------------------------------------------------ data
    def set_scene(self, occupancy: OccupancyMap, goals: list[Goal], scene_id: str) -> None:
        self.occupancy = occupancy
        self.goals = goals
        self.scene_id = scene_id
        self.path = []
        self.trail.clear()
        self.pose = None
        self.marked.clear()
        grey = np.where(occupancy.occupied, OCCUPIED_COLOR, FREE_COLOR).astype(np.uint8)
        grey = np.ascontiguousarray(grey)
        self.image = QImage(
            grey.data, occupancy.width, occupancy.height, occupancy.width, QImage.Format_Grayscale8
        ).copy()
        self.update()

    def set_snapshot(self, snapshot: dict) -> None:
        if snapshot.get("scene") != self.scene_id:
            self.trail.clear()
        pose = snapshot.get("pose")
        previous = self.pose
        self.pose = tuple(pose) if pose else None
        if self.pose is not None:
            if previous is not None and math.hypot(
                self.pose[0] - previous[0], self.pose[1] - previous[1]
            ) > 1.5:
                self.trail.clear()
            self.trail.append((self.pose[0], self.pose[1]))
        self.path = [tuple(p) for p in snapshot.get("path") or []]
        self.goal = snapshot.get("goal")
        self.marked = set(snapshot.get("marked") or [])
        self.robot_radius = float(snapshot.get("robot_radius", self.robot_radius))
        self.bumper = bool(snapshot.get("bumper", False))
        self.update()

    # ------------------------------------------------------------- transform
    def _update_transform(self) -> None:
        if self.occupancy is None:
            return
        x_min, x_max, y_min, y_max = self.occupancy.bounds
        margin = 16.0
        avail_w = max(1.0, self.width() - 2 * margin)
        avail_h = max(1.0, self.height() - 2 * margin)
        self._scale = min(avail_w / (x_max - x_min), avail_h / (y_max - y_min))
        self._x0 = (self.width() - (x_max - x_min) * self._scale) / 2.0
        self._y0 = (self.height() - (y_max - y_min) * self._scale) / 2.0
        self._x_min = x_min
        self._y_max = y_max

    def _to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(
            self._x0 + (x - self._x_min) * self._scale,
            self._y0 + (self._y_max - y) * self._scale,
        )

    # ----------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#17181c"))

        if self.occupancy is None or self.image is None:
            painter.setPen(QColor("#9aa0a6"))
            painter.setFont(QFont("Sans", 13))
            painter.drawText(self.rect(), Qt.AlignCenter, "Load a scene to begin")
            return

        self._update_transform()
        x_min, x_max, y_min, y_max = self.occupancy.bounds
        target = QRectF(
            self._x0,
            self._y0,
            (x_max - x_min) * self._scale,
            (y_max - y_min) * self._scale,
        )
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(target, self.image)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Planned path.
        if len(self.path) > 1:
            pen = QPen(QColor("#00e5ff"), 2.2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(QPolygonF([self._to_screen(x, y) for x, y in self.path]))

        # Robot trail.
        if len(self.trail) > 1:
            painter.setPen(QPen(QColor(79, 195, 247, 90), 1.4))
            painter.drawPolyline(QPolygonF([self._to_screen(x, y) for x, y in self.trail]))

        # Goals.
        font = QFont("Sans", 9)
        painter.setFont(font)
        for goal in self.goals:
            point = self._to_screen(goal.pose.x, goal.pose.y)
            marked = goal.name in self.marked
            active = goal.name == self.goal
            color = QColor("#66bb6a") if marked else QColor("#ffca28")
            if active:
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(point, 11, 11)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 70)))
            painter.drawEllipse(point, 7, 7)
            if goal.has_yaw:
                end = self._to_screen(
                    goal.pose.x + 0.28 * math.cos(goal.pose.yaw),
                    goal.pose.y + 0.28 * math.sin(goal.pose.yaw),
                )
                painter.setPen(QPen(color, 2))
                painter.drawLine(point, end)
            painter.setPen(QColor("#f5f5f5"))
            label = goal.name + (" ✓" if marked else "")
            painter.drawText(point + QPointF(11, -8), label)

        # Robot.
        if self.pose is not None:
            x, y, yaw = self.pose
            centre = self._to_screen(x, y)
            radius_px = max(4.0, self.robot_radius * self._scale)
            if self.bumper:
                painter.setPen(QPen(QColor("#ef5350"), 3))
            else:
                painter.setPen(QPen(QColor("#0b1a22"), 2))
            painter.setBrush(QBrush(QColor("#4fc3f7")))
            painter.drawEllipse(centre, radius_px, radius_px)
            nose = self._to_screen(x + 0.30 * math.cos(yaw), y + 0.30 * math.sin(yaw))
            painter.setPen(QPen(QColor("#ffffff"), 2.4))
            painter.drawLine(centre, nose)
            painter.setBrush(QBrush(QColor("#0d47a1")))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(centre, 2.4, 2.4)

        # Scale bar.
        bar_m = 1.0
        bar_px = bar_m * self._scale
        painter.setPen(QPen(QColor("#c8d0d8"), 2))
        bx = self._x0 + 12.0
        by = self._y0 + (y_max - y_min) * self._scale - 12.0
        painter.drawLine(QPointF(bx, by), QPointF(bx + bar_px, by))
        painter.setFont(QFont("Sans", 8))
        painter.drawText(QPointF(bx, by - 5), "1 m")
