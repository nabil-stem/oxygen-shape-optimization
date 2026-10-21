"""Finite, literature-inspired controls; none is an anatomical reconstruction.

All masks have exactly matched area and the same inlet through x=0.18. The
signed-score threshold is the declared construction, not an unreported repair.
Individual feasibility, including fine-grid feature checks, remains mandatory.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from oxygen_shape.geometry.baselines import _polyline_distance, hilbert_vertices
from oxygen_shape.geometry.constraints import validate_geometry
from oxygen_shape.geometry.raster import FaceKey, RasterGeometry, Side

FAMILIES = (
    "rectangle",
    "hexagon",
    "koch_1",
    "koch_2",
    "maze",
    "dichotomous",
    "acinar_sacs",
    "asymmetric_sacs",
    "hilbert_2",
)


@dataclass(frozen=True)
class ResearchGeometry:
    geometry: RasterGeometry
    family: str
    seed: int
    parameters: dict


def quadratic_koch(level: int) -> np.ndarray:
    """Five-segment square-Koch generator, finite level 0..3, length (5/3)^level."""
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 3:
        raise ValueError("Koch level must be an integer in [0,3]")
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    pattern = np.array(
        [[0.0, 0.0], [1 / 3, 0.0], [1 / 3, 1 / 3], [2 / 3, 1 / 3], [2 / 3, 0.0], [1.0, 0.0]]
    )
    for _ in range(level):
        parts = []
        for a, b in zip(points[:-1], points[1:], strict=True):
            direction = b - a
            normal = np.array([-direction[1], direction[0]])
            parts.append((a + pattern[:, :1] * direction + pattern[:, 1:] * normal)[:-1])
        points = np.vstack([*parts, points[-1:]])
    return points


def _polygon_score(x: np.ndarray, y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    segments = [
        (tuple(a), tuple(b)) for a, b in zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    ]
    distance = _polyline_distance(x, y, segments)
    inside = Path(vertices).contains_points(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
    return np.where(inside, -distance, distance)


def _tree(asymmetric: bool) -> tuple[list, list]:
    segments = [((0.18, 0.5), (0.34, 0.5))]
    frontier = [(0.34, 0.5)]
    for level, (end_x, span) in enumerate(((0.55, 0.22), (0.84, 0.11))):
        following = []
        for index, start in enumerate(frontier):
            for sign in (-1, 1):
                offset = span * (1 + (0.12 * (-1) ** (index + level) if asymmetric else 0))
                end = (end_x, start[1] + sign * offset)
                segments.append((start, end))
                following.append(end)
        frontier = following
    return segments, frontier


def make_research_geometry(
    family: str,
    n: int = 100,
    target_area: float = 0.24,
    *,
    seed: int = 0,
) -> ResearchGeometry:
    if family not in FAMILIES:
        raise ValueError(f"Unknown research family {family!r}")
    if isinstance(n, bool) or not isinstance(n, int) or n < 40 or n % 20:
        raise ValueError("Research grid must be a multiple of 20, at least 40")
    if not np.isfinite(target_area) or not 0 < target_area < 0.7:
        raise ValueError("Area must be positive and below .7")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # Vary the distal geometry, keeping every protected inlet pixel identical.
    shift = float(rng.uniform(-0.012, 0.012)) if seed else 0.0
    stretch_x = float(rng.uniform(0.94, 1.04)) if seed else 1.0
    stretch_y = float(rng.uniform(0.97, 1.03)) if seed else 1.0
    center_x = 0.56 + (float(rng.uniform(-0.015, 0.015)) if seed else 0.0)
    x, y = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
    mandatory = (x < 0.18) & (y >= 0.45) & (y < 0.55)
    eligible = (x >= 0.12) & (x < 0.96) & (y >= 0.04) & (y < 0.96)
    inlet_distance = _polyline_distance(x, y, [((0.12, 0.5), (0.38, 0.5))]) - 0.05
    parameters = {
        "distal_shift": shift,
        "stretch_x": stretch_x,
        "stretch_y": stretch_y,
        "center_x": center_x,
        "common_inlet_end": 0.18,
        "construction": "exact-cell-count signed-score sublevel",
    }
    if family == "rectangle":
        score = np.maximum(
            np.abs(y - 0.5 - shift) - 0.15 * stretch_y, np.abs(x - center_x) - 0.35 * stretch_x
        )
    elif family == "hexagon":
        angles = np.arange(6) * np.pi / 3
        rotation = float(rng.uniform(-0.12, 0.12)) if seed else 0.0
        angles += rotation
        parameters["hexagon_rotation"] = rotation
        score = (
            np.max(
                np.stack(
                    [
                        (x - center_x) / stretch_x * np.cos(a)
                        + (y - 0.5 - shift) / stretch_y * np.sin(a)
                        for a in angles
                    ]
                ),
                axis=0,
            )
            - 0.26
        )
    elif family.startswith("koch_"):
        level = int(family[-1])
        curve = quadratic_koch(level)
        upper = np.c_[
            center_x + 0.70 * stretch_x * (curve[:, 0] - 0.5),
            0.5 + shift + stretch_y * (0.14 + 0.224 * curve[:, 1]),
        ]
        lower = np.c_[upper[::-1, 0], 1 + 2 * shift - upper[::-1, 1]]
        score = _polygon_score(x, y, np.vstack([upper, lower]))
        parameters.update(level=level, vertical_affine_scale=0.32 * stretch_y / stretch_x)
        if level == 2:
            from scipy.ndimage import gaussian_filter

            score = gaussian_filter(score, sigma=0.012 * n)
            parameters["signed_distance_smoothing_length"] = 0.012
    elif family in ("dichotomous", "acinar_sacs", "asymmetric_sacs"):
        segments, leaves = _tree(family == "asymmetric_sacs")

        def transform(point):
            return (
                0.18 + (point[0] - 0.18) * stretch_x,
                0.5 + (point[1] - 0.5) * stretch_y + shift,
            )

        segments = [(transform(a), transform(b)) for a, b in segments]
        leaves = [transform(p) for p in leaves]
        score = _polyline_distance(x, y, segments)
        if family != "dichotomous":
            # Hexagonal terminal chambers are proxies; they do not reproduce the
            # Fujita/Koshiyama airway construction or its anatomical statistics.
            for lx, ly in leaves:
                angles = np.arange(6) * np.pi / 3
                chamber = (
                    np.max(
                        np.stack([(x - lx) * np.cos(a) + (y - ly) * np.sin(a) for a in angles]),
                        axis=0,
                    )
                    - 0.035
                )
                score = np.minimum(score, chamber)
        parameters.update(generations=2, terminal_chambers=family != "dichotomous")
    elif family == "maze":
        points = np.array(
            [
                [0.18, 0.5],
                [0.24, 0.5],
                [0.24, 0.23],
                [0.86, 0.23],
                [0.86, 0.5],
                [0.39, 0.5],
                [0.39, 0.77],
                [0.86, 0.77],
            ]
        )
        points[1:, 1] += shift
        points[1:, 0] = 0.18 + (points[1:, 0] - 0.18) * stretch_x
        points[1:, 1] = 0.5 + (points[1:, 1] - 0.5) * stretch_y
        score = _polyline_distance(
            x, y, [(tuple(a), tuple(b)) for a, b in zip(points[:-1], points[1:], strict=True)]
        )
        parameters["lanes"] = 3
    else:
        points = np.array([0.36, 0.19 + shift]) + hilbert_vertices(2) * np.array(
            [0.52 * stretch_x, 0.62 * stretch_y]
        )
        segments = [
            ((0.18, 0.5), (0.24, 0.5)),
            ((0.24, 0.5), (0.24, 0.19 + shift)),
            ((0.24, 0.19 + shift), tuple(points[0])),
        ]
        segments += [(tuple(a), tuple(b)) for a, b in zip(points[:-1], points[1:], strict=True)]
        score = _polyline_distance(x, y, segments)
        parameters["order"] = 2
    if family in ("rectangle", "hexagon", "koch_1", "koch_2"):
        score = np.minimum(score, inlet_distance)
    score = np.where(eligible & ~mandatory, score, np.inf)
    target_cells = round(target_area * n * n)
    if abs(target_cells / n**2 - target_area) > 1e-12:
        raise ValueError("Area must be representable on the selected grid")
    candidates = np.flatnonzero(np.isfinite(score))
    needed = target_cells - int(mandatory.sum())
    if not 0 <= needed <= len(candidates):
        raise ValueError("Area incompatible with the inlet or bounding box")
    selected = candidates[np.argsort(score.ravel()[candidates], kind="stable")[:needed]]
    mask = mandatory.copy()
    mask.ravel()[selected] = True
    source = tuple(FaceKey(row, 0, Side.WEST) for row in range(9 * n // 20, 11 * n // 20))
    geometry = RasterGeometry(mask, 1 / n, source)
    validate_geometry(geometry).require_valid()
    parameters.update(
        threshold=float(score.ravel()[selected[-1]]),
        row_major_tie_break=True,
        target_area=target_area,
    )
    return ResearchGeometry(geometry, family, seed, parameters)
