#!/usr/bin/env python3
"""Run every scene with the headless runner and print an aggregate table."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_scene(scene: str, seed: int, timeout: float, recordings: Path) -> dict:
    command = [
        sys.executable, "-m", "vacuum.cli.headless",
        "--scene", scene, "--seed", str(seed), "--goals", "all",
        "--timeout", str(timeout), "--recordings-dir", str(recordings),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    summary = None
    for line in result.stdout.splitlines():
        if line.startswith("summary:"):
            summary = json.loads(line.split(":", 1)[1].strip())
    if summary is None:
        summary = {
            "scene": scene,
            "seed": seed,
            "elapsed_s": 0.0,
            "results": {},
            "reached": 0,
            "total": 0,
            "error": result.stderr.strip()[-400:],
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default="01,02,03,04,05,06")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--recordings-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summaries = []
    for scene in [item.strip() for item in args.scenes.split(",") if item.strip()]:
        print(f"--- scene {scene} ---", flush=True)
        summary = run_scene(scene, args.seed, args.timeout, args.recordings_dir)
        summaries.append(summary)
        print(
            f"scene {scene}: {summary['reached']}/{summary['total']} reached "
            f"in {summary.get('elapsed_s', 0):.1f}s, "
            f"bumps {summary.get('bumps', '?')} "
            f"({summary.get('results', {})})",
            flush=True,
        )

    print("\nscene  reached/total  elapsed_s  bumps")
    for summary in summaries:
        print(
            f"{summary.get('scene', '?'):>5}  "
            f"{summary.get('reached', 0)}/{summary.get('total', 0):<11}  "
            f"{summary.get('elapsed_s', 0):>8.1f}  {summary.get('bumps', '?'):>5}"
        )
    if args.out:
        args.out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    failures = sum(
        1 for summary in summaries if summary.get("reached") != summary.get("total")
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
