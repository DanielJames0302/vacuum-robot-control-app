"""Path simplification and resampling."""
from __future__ import annotations

import math

from vacuum.model.gridmap import OccupancyMap, PlanningGrid

Point = tuple[float, float]


def _bresenham(r0: int, c0: int, r1: int, c1: int):
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    step_r = 1 if r1 >= r0 else -1
    step_c = 1 if c1 >= c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        yield r, c
        if r == r1 and c == c1:
            return
        err2 = 2 * err
        if err2 > -dc:
            err -= dc
            r += step_r
        if err2 < dr:
            err += dr
            c += step_c


def line_of_sight(
    occupancy: OccupancyMap,
    grid: PlanningGrid,
    a: Point,
    b: Point,
    min_clearance: int = 3,
) -> bool:
    """True if the straight segment stays clear of inflated obstacles.

    Intermediate cells must have at least ``min_clearance`` cells of margin so
    that a path follower cutting the corner slightly still fits. Diagonal steps
    also reject squeezing between two blocked cells.
    """
    cell_a = occupancy.world_to_grid(a[0], a[1])
    cell_b = occupancy.world_to_grid(b[0], b[1])
    if cell_a is None or cell_b is None:
        return False
    cells = list(_bresenham(cell_a[0], cell_a[1], cell_b[0], cell_b[1]))
    for index, (row, col) in enumerate(cells):
        if grid.is_blocked(row, col):
            return False
        if 0 < index < len(cells) - 1:
            if int(grid.clearance[row, col]) < min_clearance:
                return False
        if index > 0:
            prev_row, prev_col = cells[index - 1]
            if prev_row != row and prev_col != col:
                if grid.is_blocked(prev_row, col) or grid.is_blocked(row, prev_col):
                    return False
    return True


def simplify_path(
    occupancy: OccupancyMap,
    grid: PlanningGrid,
    points: list[Point],
    max_skip: int = 80,
) -> list[Point]:
    """Greedy line-of-sight shortcutting, preserving the first and last point."""
    if len(points) <= 2:
        return list(points)
    simplified = [points[0]]
    index = 0
    while index < len(points) - 1:
        best = index + 1
        limit = min(index + max_skip, len(points) - 1)
        for candidate in range(limit, index + 1, -1):
            if line_of_sight(occupancy, grid, points[index], points[candidate]):
                best = candidate
                break
        simplified.append(points[best])
        index = best
    return simplified


def resample(points: list[Point], spacing: float = 0.05) -> list[Point]:
    """Insert points so consecutive samples are at most ``spacing`` apart."""
    if len(points) < 2 or spacing <= 0.0:
        return list(points)
    out = [points[0]]
    carry = 0.0
    for a, b in zip(points, points[1:], strict=False):
        dx, dy = b[0] - a[0], b[1] - a[1]
        segment = math.hypot(dx, dy)
        if segment < 1e-9:
            continue
        travelled = spacing - carry
        while travelled <= segment:
            t = travelled / segment
            out.append((a[0] + dx * t, a[1] + dy * t))
            travelled += spacing
        carry = segment - (travelled - spacing)
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out
