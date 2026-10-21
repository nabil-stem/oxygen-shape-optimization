"""Deterministic baseline masks sharing a fixed source in the unit box.

Fixed-area masks use nested score sublevel sets, with deterministic selection on
the final equal-score layer. This gives exact cell-count matching; the final
layer can differ from the ideal continuous shape by one raster layer.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from oxygen_shape.geometry.constraints import validate_geometry
from oxygen_shape.geometry.raster import FaceKey, RasterGeometry, Side

ShapeName = Literal[
    "rectangle", "disk", "square", "sinusoidal", "branching", "hilbert", "deep_cavity"
]
SHAPE_NAMES: tuple[ShapeName, ...] = (
    "rectangle",
    "disk",
    "square",
    "sinusoidal",
    "branching",
    "hilbert",
    "deep_cavity",
)


@dataclass(frozen=True)
class BaselineGeometry:
    geometry: RasterGeometry
    shape: ShapeName
    comparison: str
    target: float
    matching_error: float
    matching_tolerance: float


def hilbert_vertices(order: int) -> NDArray[np.float64]:
    """Standard discrete Hilbert path on a 2**order square lattice, scaled to [0,1]."""
    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 4:
        raise ValueError("Hilbert order must be an integer between 1 and 4")
    side = 2**order
    points = []
    for distance in range(side * side):
        x = y = 0
        remaining, scale = distance, 1
        while scale < side:
            rx, ry = (remaining // 2) & 1, (remaining ^ (remaining // 2)) & 1
            if ry == 0:
                if rx == 1:
                    x, y = scale - 1 - x, scale - 1 - y
                x, y = y, x
            x += scale * rx
            y += scale * ry
            remaining //= 4
            scale *= 2
        points.append((x / (side - 1), y / (side - 1)))
    return np.asarray(points)


def _polyline_distance(
    x: NDArray,
    y: NDArray,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> NDArray[np.float64]:
    distance = np.full(x.shape, np.inf)
    for (ax, ay), (bx, by) in segments:
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            raise ValueError("Zero-length skeleton segment")
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / length_squared, 0, 1)
        distance = np.minimum(distance, np.hypot(x - ax - t * dx, y - ay - t * dy))
    return distance


def _score_field(n: int, shape: ShapeName, hilbert_order: int) -> tuple[NDArray, NDArray]:
    x, y = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
    # The inlet is identical in every family. The distal cavity adds a longer neck.
    inlet_end = 0.70 if shape == "deep_cavity" else 0.55
    mandatory = (x < inlet_end) & (y >= 0.45) & (y < 0.55)
    eligible = (x >= 0.12) & (x < 0.95) & (y >= 0.05) & (y < 0.95)
    if shape == "rectangle":
        score = np.abs(y - 0.5)
        eligible &= (x >= 0.18) & (x < 0.92)
    elif shape == "square":
        score = np.maximum(np.abs(x - 0.55), np.abs(y - 0.5))
    elif shape == "disk":
        score = np.hypot(x - 0.55, y - 0.5)
    elif shape == "sinusoidal":
        phase = (x - 0.18) / 0.74
        score = np.abs(y - 0.5) / (1 + 0.28 * np.sin(4 * np.pi * phase))
        eligible &= (x >= 0.18) & (x < 0.92)
    elif shape == "branching":
        segments = [
            ((0.18, 0.5), (0.50, 0.5)),
            ((0.50, 0.5), (0.82, 0.22)),
            ((0.50, 0.5), (0.82, 0.78)),
            ((0.70, 0.325), (0.90, 0.40)),
            ((0.70, 0.675), (0.90, 0.60)),
        ]
        score = _polyline_distance(x, y, segments)
    elif shape == "hilbert":
        points = 0.22 + 0.66 * hilbert_vertices(hilbert_order)
        segments = [((0.18, 0.5), tuple(points[0]))]
        segments.extend((tuple(a), tuple(b)) for a, b in zip(points[:-1], points[1:], strict=True))
        score = _polyline_distance(x, y, segments)
    elif shape == "deep_cavity":
        score = np.abs(y - 0.5)
        eligible &= (x >= 0.65) & (x < 0.95)
    else:
        raise ValueError(f"Unknown baseline shape: {shape}")
    return np.where(eligible, score, np.inf), mandatory


def _geometry(mask: NDArray, n: int) -> RasterGeometry:
    source = tuple(FaceKey(row, 0, Side.WEST) for row in range(9 * n // 20, 11 * n // 20))
    geometry = RasterGeometry(mask, 1 / n, source)
    validate_geometry(geometry).require_valid()
    return geometry


def _check_grid(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 20 or n % 20:
        raise ValueError("Baseline resolution must be a positive multiple of 20")


def matched_area(
    shape: ShapeName,
    n: int,
    target_area: float = 0.24,
    *,
    hilbert_order: int = 2,
) -> BaselineGeometry:
    _check_grid(n)
    if not np.isfinite(target_area) or target_area <= 0:
        raise ValueError("Target area must be positive and finite")
    score, mandatory = _score_field(n, shape, hilbert_order)
    target_cells = round(target_area * n * n)
    if abs(target_cells / (n * n) - target_area) > 1e-12:
        raise ValueError("Target area must be representable by an integer number of cells")
    available = np.flatnonzero(np.isfinite(score) & ~mandatory)
    needed = target_cells - int(mandatory.sum())
    if not 0 <= needed <= available.size:
        raise ValueError("Target area is incompatible with the inlet or bounding box")
    # Stable tie order is row-major; there is no random tie breaking or cell repair.
    order = np.argsort(score.ravel()[available], kind="stable")
    mask = mandatory.copy()
    mask.ravel()[available[order[:needed]]] = True
    geometry = _geometry(mask, n)
    return BaselineGeometry(
        geometry, shape, "fixed_area", target_area, abs(geometry.area - target_area), 1e-12
    )


def matched_perimeter(
    shape: ShapeName,
    n: int,
    target_perimeter: float = 3.0,
) -> BaselineGeometry:
    """Match rectangle/square membrane length within an explicit 4h tolerance.

    Whole score layers preserve simple shapes. Enumerating layers avoids assuming
    monotonic perimeter for a general mask; this routine is limited to two shapes.
    """
    from oxygen_shape.geometry.metrics import geometry_metrics

    _check_grid(n)
    if shape not in ("rectangle", "square"):
        raise ValueError("Fixed-perimeter cohort supports rectangle and square only")
    if not np.isfinite(target_perimeter) or target_perimeter <= 0:
        raise ValueError("Target perimeter must be positive and finite")
    score, mandatory = _score_field(n, shape, 2)
    # Scores rounded only to group floating-point representations of the same layer.
    layers = np.unique(np.round(score[np.isfinite(score)], 12))
    best_mask, best_error = None, np.inf
    for threshold in layers:
        mask = mandatory | (score <= threshold + 1e-12)
        # Counting exposed faces is exact and avoids repeated hull calculations.
        padded = np.pad(mask, 1)
        perimeter = (
            np.count_nonzero(padded[1:] != padded[:-1])
            + np.count_nonzero(padded[:, 1:] != padded[:, :-1])
        ) / n - 0.1
        error = abs(perimeter - target_perimeter)
        if error < best_error:
            best_mask, best_error = mask, error
    tolerance = 4 / n
    if best_mask is None or best_error > tolerance + 1e-12:
        raise ValueError("Requested membrane length cannot be matched within 4h")
    geometry = _geometry(best_mask, n)
    measured = geometry_metrics(geometry).robin_perimeter
    return BaselineGeometry(
        geometry,
        shape,
        "fixed_perimeter",
        target_perimeter,
        abs(measured - target_perimeter),
        tolerance,
    )
