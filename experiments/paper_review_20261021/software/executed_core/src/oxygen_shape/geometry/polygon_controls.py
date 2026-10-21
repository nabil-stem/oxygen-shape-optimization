"""Explicit physical polygons for separating raster-boundary and PDE errors."""

import numpy as np
from matplotlib.path import Path

from oxygen_shape.geometry.raster import FaceKey, RasterGeometry, Side
from oxygen_shape.geometry.research_families import quadratic_koch


def polygon_area(vertices):
    x, y = np.asarray(vertices).T
    return float(abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))) / 2)


def chamber_pslg(family: str, target_area: float = 0.24) -> dict:
    """Source is exactly x=0, y=.45..55; all remaining polygon edges are Robin."""
    if family not in ("rectangle", "hexagon", "koch_1", "koch_2"):
        raise ValueError("Unsupported physical polygon control")

    def points(scale):
        if family == "hexagon":
            left = 0.56 - 0.32 * scale
            joint = left + 0.05 / np.sqrt(3)
            bottom, top = 0.5 - 0.16 * np.sqrt(3) * scale, 0.5 + 0.16 * np.sqrt(3) * scale
            return np.array(
                [
                    [0.0, 0.45],
                    [joint, 0.45],
                    [0.56 - 0.16 * scale, bottom],
                    [0.56 + 0.16 * scale, bottom],
                    [0.56 + 0.32 * scale, 0.5],
                    [0.56 + 0.16 * scale, top],
                    [0.56 - 0.16 * scale, top],
                    [joint, 0.55],
                    [0.0, 0.55],
                ]
            )
        level = 0 if family == "rectangle" else int(family[-1])
        curve = quadratic_koch(level)
        if level == 2:
            # The raw 5/3 generator has point contacts at level two. Make the
            # physical control a declared regularized polygon, not a singular
            # self-contacting boundary or an implicitly filled triangulation.
            rounded = curve.copy()
            rounded[1:-1] = 0.8 * curve[1:-1] + 0.1 * (curve[:-2] + curve[2:])
            curve = rounded
        upper = np.c_[
            0.56 + (0.70 * curve[:, 0] - 0.35) * scale, 0.5 + (0.14 + 0.224 * curve[:, 1]) * scale
        ]
        lower = upper.copy()
        lower[:, 1] = 1 - upper[:, 1]
        left = upper[0, 0]
        return np.vstack([[0.0, 0.45], [left, 0.45], lower, upper[::-1], [left, 0.55], [0.0, 0.55]])

    low, high = 0.4, 1.3
    if not polygon_area(points(low)) < target_area < polygon_area(points(high)):
        raise ValueError("Polygon area outside the supported scale interval")
    for _ in range(60):
        mid = (low + high) / 2
        if polygon_area(points(mid)) < target_area:
            low = mid
        else:
            high = mid
    vertices = points((low + high) / 2)
    segments = np.c_[np.arange(len(vertices)), np.roll(np.arange(len(vertices)), -1)]
    markers = np.full((len(segments), 1), 2, dtype=np.int32)
    markers[-1] = 1
    return dict(vertices=vertices, segments=segments, segment_markers=markers)


def rasterize_chamber(pslg: dict, n: int) -> RasterGeometry:
    if n < 20 or n % 20:
        raise ValueError("Polygon raster grid must be a multiple of 20")
    x, y = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
    mask = Path(pslg["vertices"]).contains_points(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
    source = tuple(FaceKey(row, 0, Side.WEST) for row in range(9 * n // 20, 11 * n // 20))
    return RasterGeometry(mask, 1 / n, source)
