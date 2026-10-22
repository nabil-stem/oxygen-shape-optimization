"""Explicit square obstacles in a connected air region; no additional sources."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import label

from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.constraints import feature_violations, validate_geometry
from oxygen_shape.geometry.raster import OFFSETS, FaceKey, RasterGeometry, Side


def hole_labels(geometry: RasterGeometry) -> tuple[np.ndarray, int]:
    """Label bounded complement components; 8-connected exterior, 4-connected air.

    Point contacts are not regular polygonal boundaries and remain rejected by
    the independent FEM graph check. Padding makes box-edge exclusions exterior.
    """
    labels, _ = label(~np.pad(geometry.inside, 1), structure=np.ones((3, 3), dtype=bool))
    exterior = labels[0, 0]
    interior = labels[1:-1, 1:-1]
    result = np.zeros(geometry.inside.shape, dtype=np.int32)
    ids = [int(i) for i in np.unique(interior) if i not in (0, exterior)]
    for index, component in enumerate(ids, 1):
        result[interior == component] = index
    return result, len(ids)


def boundary_hole_ids(geometry: RasterGeometry) -> np.ndarray:
    """Zero for exterior faces, 1..K for faces incident on each excluded island."""
    labels, _ = hole_labels(geometry)
    ny, nx = labels.shape
    result = []
    for face in extract_boundary(geometry):
        dy, dx = OFFSETS[face.key.side]
        row, col = face.key.row + dy, face.key.col + dx
        result.append(int(labels[row, col]) if 0 <= row < ny and 0 <= col < nx else 0)
    return np.asarray(result, dtype=np.int32)


def hole_boundary(geometry: RasterGeometry, kind: Literal["robin", "neumann"]) -> RasterGeometry:
    """Change only bounded-island walls, preserving all exterior classifications."""
    if kind not in ("robin", "neumann"):
        raise ValueError("Hole boundary must be robin or neumann")
    faces = extract_boundary(geometry)
    ids = boundary_hole_ids(geometry)
    if any(i and f.kind == BoundaryKind.SOURCE_DIRICHLET for f, i in zip(faces, ids, strict=True)):
        raise ValueError("A protected source cannot be reclassified as a hole wall")
    neumann = tuple(
        f.key
        for f, i in zip(faces, ids, strict=True)
        if (i and kind == "neumann") or (not i and f.kind == BoundaryKind.NEUMANN)
    )
    return RasterGeometry(geometry.inside, geometry.h, geometry.source_faces, neumann)


@dataclass(frozen=True, order=True)
class SquareHole:
    row: int
    col: int
    width: int

    def __post_init__(self) -> None:
        if any(isinstance(x, bool) or not isinstance(x, int) for x in vars(self).values()):
            raise ValueError("Square-hole indices and width must be integers")
        if self.row < 0 or self.col < 0 or self.width < 1:
            raise ValueError("Square-hole indices must be nonnegative and width positive")


def perforate(
    outer: RasterGeometry,
    holes: tuple[SquareHole, ...],
    *,
    gap: float = 0.04,
    protected_length: float = 0.12,
    minimum_feature_width: float = 0.04,
) -> RasterGeometry:
    """Remove separated, grid-aligned squares from a no-hole outer air region.

    gap is a conservative L-infinity clearance measured between cell edges,
    both to the exterior and to other squares. The opening-core air-feature
    rule is additionally checked. No resource repair or boundary smoothing occurs.
    """
    validate_geometry(outer).require_valid()
    if hole_labels(outer)[1]:
        raise ValueError("Supply a no-hole outer region and the complete square list")
    if any(not np.isfinite(x) or x <= 0 for x in (gap, protected_length, minimum_feature_width)):
        raise ValueError("Clearances and feature width must be positive and finite")
    margin = max(1, int(np.ceil(gap / outer.h - 1e-12)))
    mask = outer.inside.copy()
    occupied = np.zeros_like(mask)
    ny, nx = mask.shape
    for hole in holes:
        r, c, w = hole.row, hole.col, hole.width
        if w * outer.h < minimum_feature_width - 1e-12:
            raise ValueError("Square width is below the declared hole feature cutoff")
        if c * outer.h < protected_length - 1e-12:
            raise ValueError("Square enters the protected source collar")
        if r < margin or c < margin or r + w + margin > ny or c + w + margin > nx:
            raise ValueError("Square violates exterior clearance")
        expanded = np.s_[r - margin : r + w + margin, c - margin : c + w + margin]
        if not np.all(outer.inside[expanded]):
            raise ValueError("Square violates exterior clearance")
        if occupied[expanded].any():
            raise ValueError("Squares overlap or violate inter-square clearance")
        occupied[r : r + w, c : c + w] = True
        mask[r : r + w, c : c + w] = False
    geometry = outer.with_mask(mask)
    validate_geometry(geometry, reference=outer).require_valid()
    violations = feature_violations(geometry, minimum_feature_width)
    if violations:
        raise ValueError("; ".join(violations))
    if hole_labels(geometry)[1] != len(holes):
        raise ValueError("Each square must remain a separate enclosed island")
    return geometry


def maze_chamber(n: int = 100) -> RasterGeometry:
    """A chamber plus the standard 0.1 inlet; square islands create maze routes."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 40 or n % 20:
        raise ValueError("Maze chamber grid must be a multiple of 20, at least 40")
    x, y = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
    inside = ((x < 0.20) & (y >= 0.45) & (y < 0.55)) | (
        (x >= 0.18) & (x < 0.90) & (y >= 0.15) & (y < 0.85)
    )
    source = tuple(FaceKey(int(r), 0, Side.WEST) for r in np.flatnonzero(inside[:, 0]))
    return RasterGeometry(inside, 1 / n, source)
