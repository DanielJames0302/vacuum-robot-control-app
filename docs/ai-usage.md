# AI usage

I used an AI coding assistant (OpenAI Codex / opencode CLI, model
`deepseek-v4.1-flash-free`) as a pair-programming tool during this assignment.

## What it was used for

- Scaffolding the Python package, entry points and module layout.
- Drafting transport, map, planning, control, recording and Qt UI code.
- Debugging simulator-integration issues, especially stale stream messages after
  `load_scene`, simulation-time vs wall-clock timers, and recovery loops.
- Drafting unit tests and the recording validator.
- Drafting and editing this documentation, including the README and design note.

## What was verified

I ran every change against the real simulator. The final code was checked with:

```bash
ruff check src tests scripts
pytest
vacuum-headless --scene 01 --seed 42 --goals all
# ... through scene 06
python scripts/validate_recording.py recordings/example_scene01
```

The scene results in the README come from those runs, not from the assistant.
The assistant did not have access to hidden tests or the grading harness.

## Decisions kept under human control

- Python + PySide6, rather than C++.
- Global A* with clearance cost, line-of-sight smoothing and pure pursuit.
- The goal-verification policy (four consecutive in-tolerance states before
  `mark_goal`).
- The easy/advanced depth-memory policy and its trade-off.
- The recording format (H.264 RGB + lossless FFV1 depth + JSONL manifest).
- Which limitations to document rather than hide.

No third-party code was copied into the project; dependencies are listed in the
README and `pyproject.toml`.
