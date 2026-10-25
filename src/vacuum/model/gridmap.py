"""Occupancy grid with world<->grid transforms, inflation and clearance."""
from __future__ import annotations

import io
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

from vacuum.model.pose import Pose


@dataclass
class PlanningGrid:
    """Inflated occupancy plus a clearance field, ready for A*."""

    blocked: np.ndarray  # bool, True = not traversable
    clearance: np.ndarray  # uint8, Manhattan-ish distance to blocked (cells)
    resolution: float

    def is_blocked(self, row: int, col: int) -> bool:
        if row < 0 or col < 0 or row >= self.blocked.shape[0] or col >= self.blocked.shape[1]:
            return True
        return bool(self.blocked[row, col])


class OccupancyMap:
    """Map layout from the simulator.

    Row 0 is the top of the image and corresponds to the largest y.
    ``origin`` is the world position of the bottom-left corner of pixel
    (row H-1, col 0).
    """

    def __init__(
        self,
        occupied: np.ndarray,
        resolution: float,
        origin: tuple[float, float],
    ) -> None:
        if occupied.ndim != 2:
            raise ValueError("occupancy image must be 2-D")
        self.occupied = occupied.astype(bool)
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        self.height, self.width = self.occupied.shape
        self.free = ~self.occupied
        self._base_inflated: dict[float, np.ndarray] = {}

    @classmethod
    def from_png(cls, meta: dict, png: bytes) -> OccupancyMap:
        image = Image.open(io.BytesIO(png))
        array = np.asarray(image.convert("L"))
        return cls(
            occupied=array > 127,
            resolution=float(meta["resolution"]),
            origin=(float(meta["origin"][0]), float(meta["origin"][1])),
        )

    # -------------------------------------------------------------- geometry
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) in world coordinates."""
        x_min, y_min = self.origin
        return (
            x_min,
            x_min + self.width * self.resolution,
            y_min,
            y_min + self.height * self.resolution,
        )

    @property
    def center(self) -> tuple[float, float]:
        x_min, x_max, y_min, y_max = self.bounds
        return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

    def world_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        """World point -> (row, col), or None outside the map."""
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = self.height - 1 - int(math.floor((y - self.origin[1]) / self.resolution))
        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            return None
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Centre of cell (row, col) in world coordinates."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (self.height - 1 - row + 0.5) * self.resolution
        return x, y

    def cell_center_pose(self, row: int, col: int) -> Pose:
        x, y = self.grid_to_world(row, col)
        return Pose(x, y, 0.0)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and not self.occupied[row, col]

    def is_free_world(self, x: float, y: float) -> bool:
        cell = self.world_to_grid(x, y)
        return cell is not None and not self.occupied[cell]

    # ------------------------------------------------------------- planning
    def inflate(self, radius_m: float) -> np.ndarray:
        """Return a bool grid with obstacles dilated by ``radius_m``."""
        radius_m = round(float(radius_m), 4)
        cached = self._base_inflated.get(radius_m)
        if cached is not None:
            return cached
        inflated = inflate_mask(self.occupied, radius_m / self.resolution)
        self._base_inflated[radius_m] = inflated
        return inflated

    def build_planning_grid(
        self,
        robot_radius: float,
        margin: float = 0.03,
        extra_obstacles: list[tuple[float, float, float]] | None = None,
        extra_occupied: np.ndarray | None = None,
    ) -> PlanningGrid:
        radius_m = robot_radius + margin
        if extra_occupied is not None and bool(np.any(extra_occupied)):
            merged = self.occupied | extra_occupied
            blocked_extra = inflate_mask(merged, radius_m / self.resolution)
        else:
            blocked_extra = self.inflate(radius_m).copy()
        if extra_obstacles:
            for x, y, radius in extra_obstacles:
                stamp_circle(blocked_extra, self, x, y, radius + radius_m)
        # The map border is not reachable; treat it as blocked so paths stay inside.
        blocked_extra[0, :] = True
        blocked_extra[-1, :] = True
        blocked_extra[:, 0] = True
        blocked_extra[:, -1] = True
        clearance = distance_to_blocked(blocked_extra, cap=10)
        return PlanningGrid(blocked=blocked_extra, clearance=clearance, resolution=self.resolution)

    def nearest_free_cell(
        self,
        row: int,
        col: int,
        blocked: np.ndarray,
        max_radius: int = 60,
    ) -> tuple[int, int] | None:
        """BFS for the closest non-blocked cell, ignoring the map border."""
        if self.in_bounds(row, col) and not blocked[row, col]:
            return row, col
        seen = np.zeros_like(blocked, dtype=bool)
        queue: deque[tuple[int, int, int]] = deque([(row, col, 0)])
        while queue:
            r, c, dist = queue.popleft()
            if dist > max_radius:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not self.in_bounds(nr, nc) or seen[nr, nc]:
                    continue
                if not blocked[nr, nc]:
                    return nr, nc
                seen[nr, nc] = True
                queue.append((nr, nc, dist + 1))
        return None

    # ------------------------------------------------------------- rendering
    def preview(self, scale: int = 1) -> np.ndarray:
        """Grayscale image with 0 = occupied, 255 = free (row 0 at top)."""
        image = np.where(self.occupied, 0, 255).astype(np.uint8)
        if scale > 1:
            image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
        return image


def inflate_mask(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    """Dilate a boolean mask with a disc of ``radius_cells``."""
    radius = int(math.ceil(max(0.0, radius_cells)))
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    height, width = mask.shape
    r2 = radius_cells * radius_cells
    for dr in range(-radius, radius + 1):
        if dr >= 0:
            dst_r = slice(dr, height)
            src_r = slice(0, height - dr)
        else:
            dst_r = slice(0, height + dr)
            src_r = slice(-dr, height)
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > r2:
                continue
            if dc >= 0:
                dst_c = slice(dc, width)
                src_c = slice(0, width - dc)
            else:
                dst_c = slice(0, width + dc)
                src_c = slice(-dc, width)
            out[dst_r, dst_c] |= mask[src_r, src_c]
    return out


def stamp_circle(
    mask: np.ndarray,
    occupancy: OccupancyMap,
    x: float,
    y: float,
    radius_m: float,
) -> None:
    """Set a filled disc (world coordinates) to True in ``mask``."""
    cell = occupancy.world_to_grid(x, y)
    if cell is None:
        return
    row, col = cell
    radius_cells = radius_m / occupancy.resolution
    extent = int(math.ceil(radius_cells))
    r0, r1 = max(0, row - extent), min(occupancy.height, row + extent + 1)
    c0, c1 = max(0, col - extent), min(occupancy.width, col + extent + 1)
    if r0 >= r1 or c0 >= c1:
        return
    yy = (np.arange(r0, r1) - row)[:, None]
    xx = (np.arange(c0, c1) - col)[None, :]
    mask[r0:r1, c0:c1] |= (yy * yy + xx * xx) <= radius_cells * radius_cells


def distance_to_blocked(blocked: np.ndarray, cap: int = 10) -> np.ndarray:
    """Approximate distance (cells) to the nearest blocked cell, capped at ``cap``."""
    height, width = blocked.shape
    dist = np.full((height, width), cap, dtype=np.uint8)
    dist[blocked] = 0
    for _ in range(cap):
        padded = np.pad(dist, 1, mode="constant", constant_values=cap)
        neighbours = np.minimum.reduce(
            [
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
            ]
        )
        updated = np.minimum(dist, neighbours + 1)
        if np.array_equal(updated, dist):
            break
        dist = updated
    dist[blocked] = 0
    return dist
