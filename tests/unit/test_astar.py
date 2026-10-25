from __future__ import annotations

import math

import numpy as np

from vacuum.model.gridmap import OccupancyMap, PlanningGrid
from vacuum.model.pose import Pose
from vacuum.planning.astar import astar, path_length, plan_world_path


def make_grid(size: int = 12) -> PlanningGrid:
    blocked = np.zeros((size, size), dtype=bool)
    blocked[0, :] = blocked[-1, :] = True
    blocked[:, 0] = blocked[:, -1] = True
    clearance = np.ones_like(blocked, dtype=np.uint8) * 5
    return PlanningGrid(blocked=blocked, clearance=clearance, resolution=1.0)


def test_straight_path_endpoints() -> None:
    grid = make_grid()
    cells = astar(grid, (2, 2), (2, 9))
    assert cells is not None
    assert cells[0] == (2, 2)
    assert cells[-1] == (2, 9)
    assert all(not grid.is_blocked(*cell) for cell in cells)


def test_astar_routes_around_wall() -> None:
    grid = make_grid()
    grid.blocked[2:10, 6] = True
    cells = astar(grid, (2, 2), (2, 9))
    assert cells is not None
    assert any(col == 5 for _, col in cells)
    assert cells[-1] == (2, 9)


def test_astar_no_path_returns_none() -> None:
    grid = make_grid()
    grid.blocked[1:11, 6] = True
    assert astar(grid, (2, 2), (2, 9)) is None


def test_world_path_and_length() -> None:
    occ = OccupancyMap(np.zeros((40, 40), dtype=bool), resolution=0.1, origin=(0.0, 0.0))
    grid = occ.build_planning_grid(robot_radius=0.05, margin=0.0)
    path = plan_world_path(occ, grid, Pose(0.5, 0.5), Pose(3.5, 3.5))
    assert path is not None
    assert path[0] == (0.5, 0.5)
    assert path[-1] == (3.5, 3.5)
    assert path_length(path) >= math.hypot(3.0, 3.0)
