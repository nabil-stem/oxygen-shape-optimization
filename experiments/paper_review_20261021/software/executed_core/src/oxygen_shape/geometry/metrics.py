"""Raster geometry measures with an explicit morphology diagnostic."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_opening
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.raster import RasterGeometry


@dataclass(frozen=True)
class GeometryMetrics:
    area: float
    robin_perimeter: float
    source_length: float
    neumann_perimeter: float
    total_perimeter: float
    bounding_diameter: float
    occupied_width: float
    occupied_height: float
    opening_removed_fraction: float
    opening_radius: float
    minimum_neck: float | None = None


def geometry_metrics(geometry: RasterGeometry, opening_radius_cells: int = 1) -> GeometryMetrics:
    if (
        isinstance(opening_radius_cells, bool)
        or not isinstance(opening_radius_cells, int)
        or opening_radius_cells < 1
    ):
        raise ValueError("Opening radius must be a positive integer number of cells")
    faces = extract_boundary(geometry)
    h = geometry.h
    counts = {kind: sum(face.kind == kind for face in faces) for kind in BoundaryKind}
    cells = np.argwhere(geometry.inside)
    corners = np.concatenate([cells + offset for offset in ((0, 0), (0, 1), (1, 0), (1, 1))])
    # Convex-hull extrema suffice for the exact diameter of the raster cell union.
    corners = np.unique(corners, axis=0) * h
    hull = corners[ConvexHull(corners).vertices]
    diameter = float(pdist(hull).max())
    radius = opening_radius_cells
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    structure = x * x + y * y <= radius * radius
    opened = binary_opening(geometry.inside, structure=structure, border_value=0)
    removed = float(np.count_nonzero(geometry.inside & ~opened) / geometry.inside.sum())
    return GeometryMetrics(
        geometry.area,
        counts[BoundaryKind.ROBIN] * h,
        counts[BoundaryKind.SOURCE_DIRICHLET] * h,
        counts[BoundaryKind.NEUMANN] * h,
        len(faces) * h,
        diameter,
        float((cells[:, 1].max() - cells[:, 1].min() + 1) * h),
        float((cells[:, 0].max() - cells[:, 0].min() + 1) * h),
        removed,
        radius * h,
    )
