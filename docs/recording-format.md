# Recording format

Each recording is a self-contained directory:

```
recordings/example_scene01/
  rgb.mp4
  depth.mkv
  manifest.jsonl
  meta.json
```

## `rgb.mp4`

- H.264, `yuv420p`, 640x480, nominal 15 fps
- One video frame per stored camera message
- Good for playback and quick visual checks; use the manifest for timing

## `depth.mkv`

- FFV1 level 3, `gray16le`, 640x480, nominal 15 fps
- Values are the simulator's raw depth in **millimetres**
- Convert to metres with `depth_scale` from `meta.json` (normally `0.001`)
- Zero means invalid/no return
- Lossless: no depth quantisation beyond the simulator's 16-bit input

The depth video has the same frame count and order as `rgb.mp4`. Frames are
aligned by index, not by container timestamps.

## `manifest.jsonl`

One JSON object per line, in storage order:

```json
{"frame": 0, "t": 12.34, "seq": 185, "pose": [0.14, -0.34, 2.253], "dropped_before": 0}
```

| Field | Meaning |
|---|---|
| `frame` | zero-based index in both videos |
| `t` | simulator time in seconds since the last load/reset |
| `seq` | camera stream sequence number |
| `pose` | robot pose `[x, y, yaw]` in the world frame (metres, radians) |
| `dropped_before` | cumulative frames dropped by the recorder queue before this frame |

`t` is the authoritative frame time. It may be non-uniform even though the video
container advertises 15 fps. `pose` is the pose carried in the camera message
header, i.e. the pose at capture time.

## `meta.json`

Important fields:

| Field | Meaning |
|---|---|
| `created_at` | local timestamp when the recording started |
| `scene`, `seed` | scene id and start-pose seed (if provided) |
| `width`, `height`, `fps` | frame size and nominal rate |
| `camera_info` | `fx`, `fy`, `cx`, `cy`, `depth_scale`, `max_depth`, `T_base_camera` |
| `files` | names of the RGB, depth and manifest files |
| `started` | `false` after a clean close |
| `frames_written`, `dropped`, `duration_s` | recording statistics |
| `error` | encoder/queue error, if any |
| `format` | short human-readable description |

`T_base_camera` is a 4x4 row-major transform mapping camera-frame points to the
robot frame (OpenCV camera convention: x right, y down, z forward).

## Reading a recording in Python

```python
import json
from pathlib import Path

import cv2
import numpy as np

run = Path("recordings/example_scene01")
meta = json.loads((run / "meta.json").read_text())
scale = meta["camera_info"]["depth_scale"]

records = [json.loads(line) for line in (run / "manifest.jsonl").read_text().splitlines()]
rgb = cv2.VideoCapture(str(run / "rgb.mp4"))
depth = cv2.VideoCapture(str(run / "depth.mkv"))

for record in records:
    ok_rgb, rgb_frame = rgb.read()
    ok_depth, depth_frame = depth.read()
    if not (ok_rgb and ok_depth):
        break
    depth_m = depth_frame[:, :, 0].astype(np.float32) * scale
    t = record["t"]
    pose = record["pose"]
    # use rgb_frame, depth_m, t, pose ...
```

OpenCV returns the 16-bit depth frame as a single-channel `uint16` image (the
example indexes channel 0 defensively). If a decoder returns a different dtype,
cast before applying `depth_scale`.

## Inspect or extract frames

```bash
ffprobe -v error -show_streams recordings/example_scene01/rgb.mp4
ffprobe -v error -show_streams recordings/example_scene01/depth.mkv

# Decode one depth frame to a 16-bit PNG
ffmpeg -i recordings/example_scene01/depth.mkv -vf "select=eq(n\,100)" \
  -frames:v 1 -pix_fmt gray16le /tmp/depth_100.png

# Decode the matching RGB frame
ffmpeg -i recordings/example_scene01/rgb.mp4 -vf "select=eq(n\,100)" \
  -frames:v 1 /tmp/rgb_100.png
```

## Validation

```bash
python scripts/validate_recording.py recordings/example_scene01
```

The validator checks required files, non-empty payloads, manifest frame/time/seq
ordering, pose shape, `meta.json` consistency, codecs (`h264`, `ffv1`), pixel
format (`gray16le`), dimensions, and that both videos contain exactly as many
frames as the manifest.

## Known constraints

- The two video files carry identical constant-fps timestamps; only the manifest
  preserves real capture time.
- The recorder drops frames if the queue is full and reports the count in
  `meta.json`; it does not stretch time to hide drops.
- Depth is recorded raw; no hole filling or filtering is applied, so consumers can
  choose their own preprocessing.
