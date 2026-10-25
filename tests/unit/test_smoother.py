from __future__ import annotations

import numpy as np

from vacuum.model.gridmap import OccupancyMap
from vacuum.model.pose import Pose
from vacuum.planning.astar import plan_world_path
from vacuum.planning.smoother import line_of_sight, resample, simplify_path


def build_open() -> tuple[OccupancyMap, object]:
    occ = OccupancyMap(np.zeros((40, 40), dtype=bool), resolution=0.1, origin=(0.0, 0.0))
    return occ, occ.build_planning_grid(robot_radius=0.05, margin=0.0)


def test_resample_preserves_endpoints_and_spacing() -> None:
    points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.3)]
    out = resample(points, spacing=0.05)
    assert out[0] == points[0]
    assert out[-1] == points[-1]
    assert all(
        np.hypot(b[0] - a[0], b[1] - a[1]) <= 0.0500001
        for a, b in zip(out, out[1:], strict=False)
    )


def test_line_of_sight_open_and_blocked() -> None:
    occ, grid = build_open()
    assert line_of_sight(occ, grid, (0.5, 0.5), (3.5, 3.5), min_clearance=1)
    occupied = occ.occupied.copy()
    occupied[4:35, 20] = True
    occ = type(occ)(occupied, resolution=occ.resolution, origin=occ.origin)
    grid = occ.build_planning_grid(robot_radius=0.05, margin=0.0)
    assert not line_of_sight(occ, grid, (0.5, 0.5), (3.5, 3.5), min_clearance=1)


def test_simplify_reduces_points_and_keeps_endpoints() -> None:
    occ, grid = build_open()
    path = plan_world_path(occ, grid, Pose(0.5, 0.5), Pose(3.5, 3.5))
    assert path is not None
    simplified = simplify_path(occ, grid, path)
    assert len(simplified) < len(path)
    assert simplified[0] == path[0]
    assert simplified[-1] == path[-1]
