"""A* on an inflated occupancy grid."""
from __future__ import annotations

import heapq
import math
from collections.abc import Iterable

from vacuum.model.gridmap import OccupancyMap, PlanningGrid
from vacuum.model.pose import Pose

Cell = tuple[int, int]

_NEIGHBOURS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def _octile(a: Cell, b: Cell) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2.0) - 2.0) * min(dr, dc)


def astar(
    grid: PlanningGrid,
    start: Cell,
    goal: Cell,
    clearance_weight: float = 2.0,
    max_expansions: int = 500_000,
) -> list[Cell] | None:
    """Plan on ``grid``; returns cells from start to goal inclusive."""
    if grid.is_blocked(*start) or grid.is_blocked(*goal):
        return None
    height, width = grid.blocked.shape
    n = height * width
    g_score = [math.inf] * n
    came_from = [-1] * n
    closed = bytearray(n)

    start_idx = start[0] * width + start[1]
    goal_idx = goal[0] * width + goal[1]
    g_score[start_idx] = 0.0
    heap: list[tuple[float, float, int]] = [(float(_octile(start, goal)), 0.0, start_idx)]

    expansions = 0
    while heap:
        _, g, idx = heapq.heappop(heap)
        if closed[idx]:
            continue
        closed[idx] = 1
        if idx == goal_idx:
            return _reconstruct(came_from, idx, width)
        expansions += 1
        if expansions > max_expansions:
            return None

        row, col = divmod(idx, width)
        for dr, dc, step in _NEIGHBOURS:
            nr, nc = row + dr, col + dc
            if nr < 0 or nc < 0 or nr >= height or nc >= width:
                continue
            if grid.blocked[nr, nc]:
                continue
            if dr != 0 and dc != 0:
                if grid.blocked[row, col + dc] or grid.blocked[row + dr, col]:
                    continue
            nidx = nr * width + nc
            if closed[nidx]:
                continue
            clearance = float(grid.clearance[nr, nc])
            penalty = clearance_weight / (1.0 + clearance)
            tentative = g + step * (1.0 + penalty)
            if tentative < g_score[nidx] - 1e-9:
                g_score[nidx] = tentative
                came_from[nidx] = idx
                heapq.heappush(
                    heap,
                    (tentative + _octile((nr, nc), goal), tentative, nidx),
                )
    return None


def _reconstruct(came_from: list[int], idx: int, width: int) -> list[Cell]:
    cells: list[Cell] = []
    while idx >= 0:
        cells.append(divmod(idx, width))
        idx = came_from[idx]
    cells.reverse()
    return cells


def plan_world_path(
    occupancy: OccupancyMap,
    grid: PlanningGrid,
    start: Pose,
    goal: Pose,
    clearance_weight: float = 2.0,
) -> list[tuple[float, float]] | None:
    """Plan between two world poses, returning world points (start and goal exact)."""
    start_cell = occupancy.world_to_grid(start.x, start.y)
    goal_cell = occupancy.world_to_grid(goal.x, goal.y)
    if start_cell is None or goal_cell is None:
        return None
    start_free = occupancy.nearest_free_cell(*start_cell, grid.blocked)
    goal_free = occupancy.nearest_free_cell(*goal_cell, grid.blocked)
    if start_free is None or goal_free is None:
        return None
    cells = astar(grid, start_free, goal_free, clearance_weight=clearance_weight)
    if cells is None:
        return None
    points = [occupancy.grid_to_world(r, c) for r, c in cells]
    points[0] = (start.x, start.y)
    points[-1] = (goal.x, goal.y)
    return points


def path_length(points: Iterable[tuple[float, float]]) -> float:
    pts = list(points)
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:], strict=False))
