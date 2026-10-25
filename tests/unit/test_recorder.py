from __future__ import annotations

import io
import json
import shutil
from unittest.mock import Mock

import pytest
from PIL import Image

from vacuum.model.pose import Pose
from vacuum.recording.recorder import DEPTH_NAME, MANIFEST_NAME, META_NAME, RGB_NAME, Recorder
from vacuum.sim.types import CameraFrame

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def make_frame(index: int, width: int = 16, height: int = 12) -> CameraFrame:
    image = Image.new("RGB", (width, height), (index * 20, 40, 80))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    depth = (1000 + index).to_bytes(2, "little") * (width * height)
    return CameraFrame(
        t=float(index) / 5.0,
        seq=index,
        pose=Pose(0.1 * index, 0.2, 0.3),
        width=width,
        height=height,
        jpeg=buffer.getvalue(),
        depth_raw=depth,
    )


def test_recorder_writes_rgb_depth_manifest_and_meta(tmp_path) -> None:
    recorder = Recorder(
        tmp_path / "run",
        width=16,
        height=12,
        fps=5.0,
        camera_info={"depth_scale": 0.001},
        scene="01",
        seed=7,
    )
    recorder.start()
    for index in range(3):
        recorder.add(make_frame(index))
    directory = recorder.close()

    assert (directory / RGB_NAME).exists()
    assert (directory / DEPTH_NAME).exists()
    assert (directory / MANIFEST_NAME).exists()
    assert (directory / META_NAME).exists()

    records = [
        json.loads(line)
        for line in (directory / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert [record["frame"] for record in records] == [0, 1, 2]
    assert records[1]["t"] == pytest.approx(0.2)
    assert records[2]["pose"] == pytest.approx([0.2, 0.2, 0.3])

    meta = json.loads((directory / META_NAME).read_text(encoding="utf-8"))
    assert meta["scene"] == "01"
    assert meta["seed"] == 7
    assert meta["started"] is False
    assert meta["frames_written"] == 3
    assert meta["duration_s"] == pytest.approx(0.6)


def test_partial_start_failure_cleans_up_first_encoder(tmp_path, monkeypatch) -> None:
    first = Mock()
    first.poll.return_value = None
    popen = Mock(side_effect=[first, OSError("cannot start depth encoder")])
    monkeypatch.setattr("vacuum.recording.recorder.subprocess.Popen", popen)
    recorder = Recorder(tmp_path / "failed", 16, 12)
    with pytest.raises(OSError):
        recorder.start()
    first.kill.assert_called_once()
    first.wait.assert_called_once()
    meta = json.loads((tmp_path / "failed" / META_NAME).read_text(encoding="utf-8"))
    assert meta["started"] is False
    assert "startup" in meta["error"]


def test_nonzero_encoder_exit_is_written_to_meta(tmp_path) -> None:
    recorder = Recorder(tmp_path / "failed-exit", 16, 12)
    recorder.directory.mkdir()
    recorder._write_meta(started=True)
    proc = Mock()
    proc.stdin = io.BytesIO()
    proc.stderr = io.BytesIO(b"encoder failed")
    proc.wait.return_value = 2
    recorder._rgb_proc = proc
    recorder.close()
    meta = json.loads((recorder.directory / META_NAME).read_text(encoding="utf-8"))
    assert "status 2" in meta["error"]
