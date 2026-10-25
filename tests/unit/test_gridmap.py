from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from vacuum.model.gridmap import OccupancyMap


def test_grid_world_roundtrip() -> None:
    occ = OccupancyMap(np.zeros((10, 20), dtype=bool), resolution=0.05, origin=(-1.0, -2.0))
    for row, col in ((0, 0), (9, 19), (4, 7)):
        x, y = occ.grid_to_world(row, col)
        assert occ.world_to_grid(x, y) == (row, col)


def test_row_zero_is_high_y() -> None:
    occ = OccupancyMap(np.zeros((10, 10), dtype=bool), resolution=0.1, origin=(0.0, 0.0))
    x, y = occ.grid_to_world(0, 0)
    assert x == pytest.approx(0.05)
    assert y == pytest.approx(0.95)


def test_from_png_and_bounds() -> None:
    array = np.full((5, 8), 255, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    occ = OccupancyMap.from_png(
        {"resolution": 0.2, "origin": [1.0, 2.0]}, buffer.getvalue()
    )
    assert occ.width == 8
    assert occ.height == 5
    assert occ.bounds == (1.0, 2.6, 2.0, 3.0)
    assert occ.occupied.all()


def test_inflate_blocks_neighbours() -> None:
    occupied = np.zeros((7, 7), dtype=bool)
    occupied[3, 3] = True
    occ = OccupancyMap(occupied, resolution=0.1, origin=(0.0, 0.0))
    inflated = occ.inflate(0.25)
    assert inflated[3, 3]
    assert inflated[3, 5]
    assert not inflated[3, 6]


def test_planning_grid_border_and_clearance() -> None:
    occ = OccupancyMap(np.zeros((10, 10), dtype=bool), resolution=0.1, origin=(0.0, 0.0))
    grid = occ.build_planning_grid(robot_radius=0.1, margin=0.0)
    assert grid.blocked[0, :].all()
    assert grid.blocked[:, -1].all()
    assert grid.clearance[5, 5] > grid.clearance[1, 1]


def test_extra_obstacle_stamp() -> None:
    occ = OccupancyMap(np.zeros((20, 20), dtype=bool), resolution=0.1, origin=(0.0, 0.0))
    grid = occ.build_planning_grid(0.05, margin=0.0, extra_obstacles=[(1.0, 1.0, 0.2)])
    assert grid.is_blocked(10, 10)


def test_nearest_free_cell() -> None:
    occ = OccupancyMap(np.zeros((10, 10), dtype=bool), resolution=0.1, origin=(0.0, 0.0))
    blocked = np.zeros((10, 10), dtype=bool)
    blocked[4, 4] = True
    row, col = occ.nearest_free_cell(4, 4, blocked)
    assert abs(row - 4) + abs(col - 4) == 1
    assert not blocked[row, col]
