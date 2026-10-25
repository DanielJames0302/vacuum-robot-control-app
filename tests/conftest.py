from __future__ import annotations

import numpy as np
import pytest

from vacuum.model.gridmap import OccupancyMap


@pytest.fixture
def open_map() -> OccupancyMap:
    """4 m x 4 m free map with 10 cm cells, origin at (0, 0)."""
    return OccupancyMap(np.zeros((40, 40), dtype=bool), resolution=0.1, origin=(0.0, 0.0))


@pytest.fixture
def map_with_wall() -> OccupancyMap:
    """4 m x 4 m map with a vertical wall at x ~= 2 m, gap at the bottom."""
    occupied = np.zeros((40, 40), dtype=bool)
    occupied[5:35, 20] = True
    return OccupancyMap(occupied, resolution=0.1, origin=(0.0, 0.0))
