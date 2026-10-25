# Design note

## Architecture

The application controls the simulator through its ZeroMQ interface and treats it
like hardware: messages may be late, duplicated, lost, or malformed. The same
`MissionController` is used by the Qt GUI and headless runner.

```
sim/         RPC, commands, state and RGBD transport
model/       poses, occupancy grids and depth projection
planning/    A* and line-of-sight path smoothing
control/     pure pursuit and mission state machine
recording/   FFmpeg encoders and per-frame manifest
app/         shared orchestration and Qt worker thread
ui/, cli/    graphical and headless front ends
```

All simulator sockets belong to one controller thread. The GUI receives typed
snapshots and never accesses sockets directly. State batches are reduced to the
newest valid sample before control is evaluated. Wrong-scene, duplicate and
backward samples are discarded and reported separately from sequence gaps.
After load/reset, the controller waits for a state near the exact returned start
pose before accepting camera data; this prevents in-flight messages from the
previous run entering perception or a recording.

## Planning and motion

The supplied PNG is converted to a boolean occupancy grid using the documented
world/image convention. Obstacles are inflated by the robot radius plus a safety
margin, and the image border is blocked. A* uses eight-connected motion, octile
distance, rejects diagonal corner cutting, and adds a clearance cost. The cell
path is greedily shortened where line of sight is safe and resampled every 5 cm.

A pure-pursuit follower tracks the path with a 25 cm lookahead. It turns in place
when the target is behind it, slows near obstacles and the goal, and uses an
explicit simulation-time slew limiter based on the robot's runtime acceleration
limits. Inside 14 cm, a slower final approach takes over. Goals are accepted only
after four consecutive samples within 5 cm and, when applicable, 5 degrees.

The map can omit real obstacles. Camera depth is back-projected with the supplied
intrinsics and `T_base_camera`. Only points intersecting the robot's runtime
height (plus 1 cm measurement tolerance) are remembered; furniture above the
vacuum is not falsely blocked. A cell needs observations in two frames before it
affects planning. Bumper contact records a temporary obstacle, reverses, rotates,
and replans. Lack of path progress for 12 seconds invokes the same bounded
recovery. Depth memory is enabled by default on every scene after seed-42 tests
reached every goal in scenes 04–06 without contact. Sparse permanent points can
still alter global topology: scenes 04/05 took long alternative routes after
camera-triggered replans. `--no-depth` remains available for comparison, and a
decaying local costmap is the intended long-term replacement.

## Recording and visualisation

RGB is stored as H.264 and 16-bit depth as lossless FFV1. A JSONL manifest stores
simulation time, sequence number, capture pose and cumulative dropped frames for
every encoded pair. `meta.json` contains camera calibration, filenames, counts,
and encoder errors. Simulation time—not nominal container frame rate—is the
offline timing authority.

Recording runs in a bounded worker queue so encoding cannot block control. Start
is transactional: partial encoder startup is cleaned up. Stop drains accepted
frames, waits for the writer before closing shared resources, checks both FFmpeg
exit codes, and records finalization failures. If overloaded, the oldest queued
frame is dropped and the loss is explicit in the manifest.

The visualiser shows the occupancy map, robot heading and trail, goal poses,
planned path, bumper, mission state, depth-memory size, stream gaps, stale-message
counts, recording progress, RGB preview and a bounded event log.

## Trade-offs and limitations

- Simulator pose is treated as ground truth; there is no localisation or SLAM.
- Depth memory does not decay or ray-clear free space, so it remains conservative.
- Recovery is reactive rather than wall-following or route-diversifying.
- A* replans globally instead of incrementally; this is adequate for these maps.
- `mark_goal` over REQ/REP cannot provide exactly-once knowledge: if its reply is
  lost, the simulator may have accepted the call even though the client retries.
- Camera videos use a constant nominal rate; exact timing requires the manifest.
- Advanced scenes are best-effort and may still bump; the assignment prioritises
  reliable performance on scenes 01–03.

With more time, I would add a ray-cleared decaying local costmap, deterministic
recorded-stream replay tests, a recovery executive with route diversity, and an
offline RGB/depth/pose playback tool.
