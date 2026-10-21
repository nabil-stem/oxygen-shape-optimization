"""Deterministic axis-aligned rectangles for the reference-solver milestone."""

import numpy as np

from oxygen_shape.geometry.raster import FaceKey, RasterGeometry, Side


def rectangle(nx: int, ny: int, h: float, *, slab: bool = False) -> RasterGeometry:
    if (
        isinstance(nx, bool)
        or isinstance(ny, bool)
        or not isinstance(nx, int)
        or not isinstance(ny, int)
        or nx < 1
        or ny < 1
    ):
        raise ValueError("nx and ny must be positive integers")
    source = tuple(FaceKey(row, 0, Side.WEST) for row in range(ny))
    neumann = (
        tuple(
            FaceKey(row, col, side)
            for row, side in ((0, Side.SOUTH), (ny - 1, Side.NORTH))
            for col in range(nx)
        )
        if slab
        else ()
    )
    return RasterGeometry(np.ones((ny, nx), dtype=bool), h, source, neumann)
