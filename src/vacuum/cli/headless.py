"""Headless mission runner: drive all (or selected) goals without a GUI.

Examples:
    python -m vacuum.cli.headless --scene 01 --seed 42 --goals all
    python -m vacuum.cli.headless --scene 02 --seed 7 --goals kitchen,living --record
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from vacuum.app.controller import MissionController

log = logging.getLogger("vacuum.headless")


def _parse_goals(spec: str, available: list[str]) -> list[str]:
    if spec.strip().lower() == "all":
        return available
    wanted = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in available]
    if unknown:
        raise SystemExit(f"unknown goals: {', '.join(unknown)}; available: {', '.join(available)}")
    return wanted


def run(args: argparse.Namespace) -> int:
    controller = MissionController(
        host=args.host,
        port_base=args.port_base,
        recordings_dir=args.recordings_dir,
        log_callback=lambda message: print(f"[controller] {message}", flush=True),
        use_depth=False if args.no_depth else (True if args.depth else None),
    )
    results: dict[str, str] = {}
    started = time.monotonic()
    try:
        if not controller.ping():
            print(
                f"cannot reach simulator at {args.host}:{args.port_base}; start it with:\n"
                "  docker run --rm -p 5555-5558:5555-5558 -p 8080:8080 "
                "ghcr.io/eurekarobotics/vacuum-sim:1.0.0 --scene 01 --viewer",
                file=sys.stderr,
            )
            return 2

        bundle = controller.load_scene(args.scene, args.seed)
        goals = [goal.name for goal in bundle.scene.goals]
        print(
            f"scene {bundle.scene.id} ({bundle.scene.name}) start="
            f"({bundle.scene.start.x:.2f}, {bundle.scene.start.y:.2f}, "
            f"{bundle.scene.start.yaw:.2f})",
            flush=True,
        )
        wanted = _parse_goals(args.goals, goals)
        print(f"goals: {', '.join(wanted)}", flush=True)

        if args.record:
            directory = controller.start_recording()
            if directory:
                print(f"recording to {directory}", flush=True)

        for name in wanted:
            if not controller.start_goal(name):
                results[name] = "unreachable"
                print(f"=== {name}: could not plan", flush=True)
                continue
            print(f"=== {name}: driving", flush=True)
            goal_start = time.monotonic()
            last_print = 0.0
            while True:
                controller.tick()
                snapshot = controller.snapshot()
                state = snapshot["state"]
                if state in ("done", "aborted"):
                    results[name] = "reached" if state == "done" else "aborted"
                    break
                if time.monotonic() - goal_start > args.timeout:
                    controller.cancel()
                    results[name] = "timeout"
                    break
                now = time.monotonic()
                if now - last_print >= 2.0:
                    last_print = now
                    pose = snapshot["pose"]
                    distance = snapshot["distance"]
                    pose_text = (
                        f"({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:+.2f})" if pose else "?"
                    )
                    dist_text = f"{distance:.2f} m" if distance is not None else "?"
                    print(
                        f"  {state:16s} pose={pose_text} dist={dist_text} "
                        f"bumps={snapshot['bumps']}",
                        flush=True,
                    )
                time.sleep(0.01)

            snapshot = controller.snapshot()
            print(
                f"=== {name}: {results[name]} after {time.monotonic() - goal_start:.1f}s "
                f"(bumps {snapshot['bumps']})",
                flush=True,
            )

    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    finally:
        controller.stop_recording()
        controller.close()

    elapsed = time.monotonic() - started
    summary = {
        "scene": args.scene,
        "seed": args.seed,
        "elapsed_s": round(elapsed, 2),
        "results": results,
        "reached": sum(1 for value in results.values() if value == "reached"),
        "total": len(results),
        "bumps": controller.snapshot().get("bumps", 0),
    }
    print("summary:", json.dumps(summary), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if summary["reached"] == summary["total"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless vacuum mission runner")
    parser.add_argument("--scene", default="01", help="scene id (01-06)")
    parser.add_argument("--seed", type=int, default=None, help="start-pose seed")
    parser.add_argument("--goals", default="all", help="'all' or comma-separated goal names")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port-base", type=int, default=5555)
    parser.add_argument("--timeout", type=float, default=150.0, help="seconds per goal")
    parser.add_argument("--record", action="store_true", help="record the RGBD camera")
    depth = parser.add_mutually_exclusive_group()
    depth.add_argument(
        "--depth", action="store_true", help="enable depth-based obstacle memory (the default)"
    )
    depth.add_argument("--no-depth", action="store_true", help="force depth-based obstacle memory off")
    parser.add_argument("--recordings-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--json-out", type=Path, default=None, help="write summary JSON here")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
