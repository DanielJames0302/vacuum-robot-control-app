#!/usr/bin/env python3
"""Check that a recording directory is complete and internally consistent."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED = ("meta.json", "manifest.jsonl", "rgb.mp4", "depth.mkv")


def ffprobe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_read_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"{path.name}: no video stream")
    return streams[0]


def load_manifest(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest line {number}: {exc}") from exc
    return records


def validate(directory: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED:
        if not (directory / name).exists():
            errors.append(f"missing {name}")
        elif (directory / name).stat().st_size == 0:
            errors.append(f"{name} is empty")
    if errors:
        return errors

    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    records = load_manifest(directory / "manifest.jsonl")

    if not records:
        errors.append("manifest has no frames")
        return errors
    if meta.get("started"):
        errors.append("meta.started is true; recording was not closed cleanly")
    if meta.get("frames_written") != len(records):
        errors.append(
            f"meta.frames_written={meta.get('frames_written')} but manifest has {len(records)}"
        )
    if meta.get("dropped"):
        errors.append(f"{meta['dropped']} frames were dropped by the recorder queue")
    if meta.get("error"):
        errors.append(f"recorder error: {meta['error']}")

    previous_t = -1.0
    previous_seq = -1
    for index, record in enumerate(records):
        if record.get("frame") != index:
            errors.append(f"manifest frame {index} has frame={record.get('frame')}")
            break
        t = float(record.get("t", -1.0))
        if t < previous_t:
            errors.append(f"manifest time went backwards at frame {index}")
            break
        previous_t = t
        seq = int(record.get("seq", -1))
        if seq <= previous_seq:
            errors.append(f"manifest seq is not increasing at frame {index}")
            break
        previous_seq = seq
        pose = record.get("pose")
        if not isinstance(pose, list) or len(pose) != 3:
            errors.append(f"manifest frame {index} has an invalid pose")
            break

    if shutil.which("ffprobe") is None:
        errors.append("ffprobe is not installed; cannot inspect the videos")
        return errors

    try:
        rgb = ffprobe(directory / "rgb.mp4")
        depth = ffprobe(directory / "depth.mkv")
    except (subprocess.CalledProcessError, ValueError) as exc:
        errors.append(f"ffprobe failed: {exc}")
        return errors

    expected_size = (int(meta.get("width", 0)), int(meta.get("height", 0)))
    if (int(rgb.get("width", 0)), int(rgb.get("height", 0))) != expected_size:
        errors.append("rgb.mp4 dimensions do not match meta.json")
    if (int(depth.get("width", 0)), int(depth.get("height", 0))) != expected_size:
        errors.append("depth.mkv dimensions do not match meta.json")
    if rgb.get("codec_name") != "h264":
        errors.append(f"rgb codec is {rgb.get('codec_name')}, expected h264")
    if depth.get("codec_name") != "ffv1":
        errors.append(f"depth codec is {depth.get('codec_name')}, expected ffv1")
    if depth.get("pix_fmt") != "gray16le":
        errors.append(f"depth pixel format is {depth.get('pix_fmt')}, expected gray16le")

    for name, stream in (("rgb.mp4", rgb), ("depth.mkv", depth)):
        count = stream.get("nb_read_frames")
        if count is not None and int(count) != len(records):
            errors.append(f"{name} has {count} frames but manifest has {len(records)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="recording directory to validate")
    args = parser.parse_args()
    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 2
    errors = validate(args.directory)
    if errors:
        print(f"INVALID: {args.directory}")
        for error in errors:
            print(f"  - {error}")
        return 1
    meta = json.loads((args.directory / "meta.json").read_text(encoding="utf-8"))
    print(
        f"OK: {args.directory} | scene {meta.get('scene')} | "
        f"{meta.get('frames_written')} frames | {meta.get('duration_s', 0):.1f}s | "
        f"{meta.get('width')}x{meta.get('height')} @ {meta.get('fps')} fps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
