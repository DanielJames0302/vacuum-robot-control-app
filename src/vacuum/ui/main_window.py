"""Qt main window."""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vacuum.app.controller import ControllerSnapshot, MissionController
from vacuum.app.worker import SimWorker
from vacuum.sim.types import SceneData
from vacuum.ui.map_view import MapView

DEFAULT_SCENES = [{"id": f"{i:02d}", "name": f"Scene {i:02d}", "difficulty": ""} for i in range(1, 7)]


class MainWindow(QMainWindow):
    def __init__(
        self,
        host: str = "localhost",
        port_base: int = 5555,
        recordings_dir: Path = Path("recordings"),
    ) -> None:
        super().__init__()
        self.setWindowTitle("Vacuum Robot Console")
        self.resize(1180, 760)

        self.scenes = self._fetch_scenes(host, port_base)
        self.scene: SceneData | None = None
        self.last_fault: str | None = None

        controller = MissionController(
            host=host,
            port_base=port_base,
            recordings_dir=recordings_dir,
        )
        self.worker = SimWorker(controller)

        self._build_ui()
        self._connect_signals()
        self.worker.start()
        QTimer.singleShot(900, self._auto_load)

    # ------------------------------------------------------------------- UI
    def _fetch_scenes(self, host: str, port_base: int) -> list[dict]:
        try:
            from vacuum.sim.rpc import RpcClient

            client = RpcClient(host, port_base, timeout_ms=800)
            try:
                scenes = client.call("list_scenes")[0] or []
                return list(scenes)
            finally:
                client.close()
        except Exception:  # noqa: BLE001 - simulator may not be up yet
            return list(DEFAULT_SCENES)

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.map_view = MapView()
        layout.addWidget(self.map_view, stretch=1)

        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(4, 4, 4, 4)
        panel_layout.setSpacing(6)

        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QScrollArea.NoFrame)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel_scroll.setMinimumWidth(340)
        panel_scroll.setMaximumWidth(430)
        panel_scroll.setWidget(panel)
        layout.addWidget(panel_scroll)
        self.setCentralWidget(central)

        # Simulator controls.
        sim_group = QGroupBox("Simulator")
        sim_form = QFormLayout(sim_group)
        self.scene_combo = QComboBox()
        for scene in self.scenes:
            label = f"{scene.get('id')} — {scene.get('name', '')}"
            if scene.get("difficulty"):
                label += f" ({scene['difficulty']})"
            self.scene_combo.addItem(label, scene.get("id"))
        sim_form.addRow("Scene", self.scene_combo)

        seed_row = QHBoxLayout()
        self.use_seed = QCheckBox("Seed")
        self.use_seed.setChecked(True)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setValue(42)
        seed_row.addWidget(self.use_seed)
        seed_row.addWidget(self.seed_spin, stretch=1)
        seed_widget = QWidget()
        seed_widget.setLayout(seed_row)
        seed_row.setContentsMargins(0, 0, 0, 0)
        sim_form.addRow("", seed_widget)

        load_row = QHBoxLayout()
        self.load_button = QPushButton("Load scene")
        self.reset_button = QPushButton("Reset run")
        load_row.addWidget(self.load_button)
        load_row.addWidget(self.reset_button)
        sim_form.addRow(load_row)
        panel_layout.addWidget(sim_group)

        # Goal controls.
        goal_group = QGroupBox("Mission")
        goal_layout = QVBoxLayout(goal_group)
        self.goal_combo = QComboBox()
        self.goal_combo.setEnabled(False)
        goal_layout.addWidget(self.goal_combo)
        buttons = QHBoxLayout()
        self.go_button = QPushButton("Plan && Go")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.go_button)
        buttons.addWidget(self.cancel_button)
        goal_layout.addLayout(buttons)
        panel_layout.addWidget(goal_group)

        # Recording.
        rec_group = QGroupBox("Camera recording")
        rec_layout = QVBoxLayout(rec_group)
        self.record_button = QPushButton("Start recording")
        self.record_button.setCheckable(True)
        rec_layout.addWidget(self.record_button)
        self.record_label = QLabel("off")
        self.record_label.setWordWrap(True)
        self.record_label.setStyleSheet("color: #9aa0a6;")
        rec_layout.addWidget(self.record_label)
        panel_layout.addWidget(rec_group)

        panel_layout.addWidget(self._build_status_panel())
        panel_layout.addWidget(self._build_camera_panel())
        panel_layout.addWidget(self._build_log_panel(), stretch=1)

        QShortcut(QKeySequence("Space"), self, activated=self._go)
        QShortcut(QKeySequence("Esc"), self, activated=self._cancel)

    def _build_status_panel(self) -> QGroupBox:
        group = QGroupBox("Status")
        status_form = QFormLayout(group)
        self.connection_label = QLabel("waiting…")
        self.scene_label = QLabel("—")
        self.scene_label.setWordWrap(True)
        self.state_label = QLabel("idle")
        self.goal_label = QLabel("—")
        self.distance_label = QLabel("—")
        self.pose_label = QLabel("—")
        self.bumper_label = QLabel("clear")
        self.path_label = QLabel("—")
        self.depth_label = QLabel("—")
        self.stream_label = QLabel("—")
        self.stream_label.setWordWrap(True)
        for row, widget in (
            ("Connection", self.connection_label),
            ("Scene", self.scene_label),
            ("State", self.state_label),
            ("Goal", self.goal_label),
            ("Distance", self.distance_label),
            ("Pose", self.pose_label),
            ("Bumper", self.bumper_label),
            ("Path", self.path_label),
            ("Depth memory", self.depth_label),
            ("Streams", self.stream_label),
        ):
            status_form.addRow(row, widget)
        return group

    def _build_camera_panel(self) -> QGroupBox:
        group = QGroupBox("Camera")
        camera_layout = QVBoxLayout(group)
        self.camera_label = QLabel("no frames yet")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(280, 158)
        self.camera_label.setMaximumHeight(200)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.camera_label.setStyleSheet("background: #101114; color: #9aa0a6;")
        camera_layout.addWidget(self.camera_label)
        return group

    def _build_log_panel(self) -> QGroupBox:
        group = QGroupBox("Log")
        log_layout = QVBoxLayout(group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(100)
        self.log_view.setMaximumBlockCount(400)
        self.log_view.setStyleSheet("font-family: monospace; font-size: 10px;")
        log_layout.addWidget(self.log_view)
        return group

    def _connect_signals(self) -> None:
        self.worker.scene_loaded.connect(self._on_scene_loaded)
        self.worker.snapshot_ready.connect(self._on_snapshot)
        self.worker.preview_ready.connect(self._on_preview)
        self.worker.log_message.connect(self._append_log)
        self.worker.error.connect(self._on_error)

        self.load_button.clicked.connect(self._load_scene)
        self.reset_button.clicked.connect(self._reset_scene)
        self.go_button.clicked.connect(self._go)
        self.cancel_button.clicked.connect(self._cancel)
        self.record_button.toggled.connect(self._toggle_recording)
        self.goal_combo.currentIndexChanged.connect(self._update_buttons)

    # --------------------------------------------------------------- actions
    def _seed(self) -> int | None:
        return self.seed_spin.value() if self.use_seed.isChecked() else None

    def _auto_load(self) -> None:
        self._load_scene()

    def _load_scene(self) -> None:
        scene_id = self.scene_combo.currentData()
        if scene_id:
            self.worker.load_scene(str(scene_id), self._seed())

    def _reset_scene(self) -> None:
        self.worker.reset_scene(self._seed())

    def _go(self) -> None:
        name = self.goal_combo.currentText().replace(" ✓", "")
        if name and self.goal_combo.isEnabled():
            self.worker.start_goal(name)

    def _cancel(self) -> None:
        self.worker.cancel_goal()

    def _toggle_recording(self, checked: bool) -> None:
        if checked:
            self.worker.start_recording()
        else:
            self.worker.stop_recording()

    def _update_buttons(self) -> None:
        running = self.state_label.text() not in ("ready", "idle", "done", "aborted")
        self.cancel_button.setEnabled(running)

    # --------------------------------------------------------------- signals
    def _on_scene_loaded(self, payload: dict) -> None:
        scene: SceneData = payload["scene"]
        self.scene = scene
        self.map_view.set_scene(payload["occupancy"], scene.goals, scene.id)
        self.goal_combo.blockSignals(True)
        self.goal_combo.clear()
        for goal in scene.goals:
            self.goal_combo.addItem(goal.name)
        self.goal_combo.blockSignals(False)
        self.goal_combo.setEnabled(bool(scene.goals))
        self.scene_label.setText(f"{scene.id} {scene.name}".strip())
        self.setWindowTitle(f"Vacuum Robot Console — {scene.id} {scene.name}".strip())
        self.state_label.setText("ready")
        self.goal_label.setText("—")
        self.distance_label.setText("—")
        self._append_log(
            f"Scene {scene.id} loaded. Start "
            f"({scene.start.x:.2f}, {scene.start.y:.2f}, {scene.start.yaw:.2f})"
        )

    def _on_snapshot(self, snapshot: ControllerSnapshot) -> None:
        self.map_view.set_snapshot(snapshot)

        connected = snapshot.get("connected")
        self.connection_label.setText("connected" if connected else "waiting for state…")
        self.connection_label.setStyleSheet(
            "color: #66bb6a;" if connected else "color: #ef5350;"
        )
        self.state_label.setText(snapshot.get("state", "—"))
        self._update_buttons()

        goal = snapshot.get("goal")
        self.goal_label.setText(goal or "—")
        distance = snapshot.get("distance")
        self.distance_label.setText(f"{distance * 100:.1f} cm" if distance is not None else "—")

        pose = snapshot.get("pose")
        if pose:
            self.pose_label.setText(
                f"x {pose[0]:.2f}  y {pose[1]:.2f}  yaw {math.degrees(pose[2]):+.0f}°"
            )
        self.bumper_label.setText("CONTACT" if snapshot.get("bumper") else "clear")
        self.bumper_label.setStyleSheet(
            "color: #ef5350; font-weight: bold;" if snapshot.get("bumper") else ""
        )

        path = snapshot.get("path") or []
        if len(path) > 1:
            length = snapshot.get("path_length", 0.0)
            self.path_label.setText(f"{len(path)} pts, {length:.2f} m")
        else:
            self.path_label.setText("—")

        depth_on = bool(snapshot.get("use_depth"))
        depth_cells = int(snapshot.get("depth_cells") or 0)
        self.depth_label.setText(
            f"on ({depth_cells} cells)" if depth_on else "off"
        )
        self.stream_label.setText(
            f"gaps {snapshot.get('state_gaps', 0)}/{snapshot.get('camera_gaps', 0)}, "
            f"stale {snapshot.get('stale_states', 0)}/{snapshot.get('stale_cameras', 0)}"
        )

        # Update goal labels with marked ticks.
        marked = set(snapshot.get("marked") or [])
        for index in range(self.goal_combo.count()):
            name = self.goal_combo.itemText(index).replace(" ✓", "")
            text = name + (" ✓" if name in marked else "")
            if self.goal_combo.itemText(index) != text:
                self.goal_combo.setItemText(index, text)

        # Recording state follows the controller (e.g. failure to start).
        recording = bool(snapshot.get("recording"))
        if self.record_button.isChecked() != recording:
            self.record_button.blockSignals(True)
            self.record_button.setChecked(recording)
            self.record_button.blockSignals(False)
            self.record_button.setText("Stop recording" if recording else "Start recording")
        if recording:
            self.record_label.setText(
                f"{snapshot.get('recorded_frames', 0)} frames → "
                f"{snapshot.get('recording_dir', '')}"
            )
        else:
            self.record_label.setText("off")

        fault = snapshot.get("fault")
        if fault and fault != self.last_fault:
            self.last_fault = fault
            self._append_log(f"FAULT: {fault}")
        if not fault:
            self.last_fault = None

    def _on_preview(self, frame) -> None:
        image = QImage.fromData(frame.jpeg, "JPEG")
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _on_error(self, message: str) -> None:
        self._append_log(f"ERROR: {message}")

    # ---------------------------------------------------------------- closing
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.worker.shutdown()
        self.worker.wait(4000)
        super().closeEvent(event)
