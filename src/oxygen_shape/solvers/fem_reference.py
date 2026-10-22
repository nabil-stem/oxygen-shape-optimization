"""Independent conforming P1 FEM with consistent Robin boundary mass matrices.

Triangle preserves the supplied polygon edges. Meshing a raster boundary checks
the same staircase domain; it does not convert it into a smooth anatomical wall.
No finite-volume conductance or matrix assembly is reused here.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.raster import RasterGeometry, Side
from oxygen_shape.geometry.square_holes import hole_labels


@dataclass(frozen=True)
class FEMResult:
    vertices: np.ndarray
    triangles: np.ndarray
    concentration: np.ndarray
    boundary_segments: np.ndarray
    boundary_markers: np.ndarray
    Q: float
    source_influx: float
    conservation_error: float
    residual: float
    area: float
    robin_perimeter: float
    energy: float


def raster_pslg(geometry: RasterGeometry) -> dict:
    """Exact integer-grid boundary graph, with source/Robin/Neumann markers 1/2/3."""
    vertices, segments, markers, lookup = [], [], [], {}
    for face in extract_boundary(geometry):
        r, c, side = face.key.row, face.key.col, face.key.side
        a, b = {
            Side.WEST: ((c, r), (c, r + 1)),
            Side.EAST: ((c + 1, r), (c + 1, r + 1)),
            Side.SOUTH: ((c, r), (c + 1, r)),
            Side.NORTH: ((c, r + 1), (c + 1, r + 1)),
        }[side]
        pair = []
        for point in (a, b):
            if point not in lookup:
                lookup[point] = len(vertices)
                vertices.append(point)
            pair.append(lookup[point])
        segments.append(pair)
        markers.append(
            {BoundaryKind.SOURCE_DIRICHLET: 1, BoundaryKind.ROBIN: 2, BoundaryKind.NEUMANN: 3}[
                face.kind
            ]
        )
    degree = np.bincount(np.asarray(segments).ravel(), minlength=len(vertices))
    if not np.all(degree == 2):
        raise ValueError("Non-manifold point contacts cannot define a simple FEM polygon")
    pslg = {
        "vertices": np.asarray(vertices) * geometry.h,
        "segments": np.asarray(segments, np.int32),
        "segment_markers": np.asarray(markers, np.int32)[:, None],
    }
    labels, count = hole_labels(geometry)
    if count:
        # A cell center is strictly inside the excluded component, unlike a
        # component centroid which could lie outside a concave hole.
        seeds = [np.argwhere(labels == i)[0][::-1] + 0.5 for i in range(1, count + 1)]
        pslg["holes"] = np.asarray(seeds) * geometry.h
    return pslg


def solve_pslg(pslg: dict, Bi: float, max_area: float = 1e-4) -> FEMResult:
    """Solve -Laplacian u=0, u=1 on marker 1, du/dn+Bi*u=0 on marker 2."""
    if not np.isfinite(Bi) or Bi < 0 or not np.isfinite(max_area) or max_area <= 0:
        raise ValueError("Bi must be nonnegative and max_area positive, both finite")
    supplied = np.asarray(pslg["vertices"], float)
    if not np.isfinite(supplied).all() or len(np.unique(np.round(supplied, 12), axis=0)) != len(
        supplied
    ):
        raise ValueError("Coincident/nonfinite PSLG vertices; regularize self-contacts explicitly")
    try:
        import triangle
    except ImportError as error:
        raise ImportError("Install requirements-validation.txt for independent FEM") from error
    mesh = triangle.triangulate(pslg, f"pq28a{max_area:.12f}Q")
    vertices = np.asarray(mesh["vertices"], float)
    triangles = np.asarray(mesh["triangles"], int)
    if len(np.unique(triangles)) != len(vertices):
        raise ValueError("Triangulation has unused vertices; check boundary degeneracy")
    edges = np.asarray(mesh["segments"], int)
    markers = np.asarray(mesh["segment_markers"], int).ravel()
    points = vertices[triangles]
    twice_area = (points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1]) - (
        points[:, 2, 0] - points[:, 0, 0]
    ) * (points[:, 1, 1] - points[:, 0, 1])
    areas = np.abs(twice_area) / 2
    if np.any(areas <= 0):
        raise ArithmeticError("Degenerate finite element")
    b = points[:, [1, 2, 0], 1] - points[:, [2, 0, 1], 1]
    c = points[:, [2, 0, 1], 0] - points[:, [1, 2, 0], 0]
    local = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (
        4 * areas[:, None, None]
    )
    rows = np.repeat(triangles, 3, axis=1).ravel()
    cols = np.tile(triangles, (1, 3)).ravel()
    diffusion = coo_matrix(
        (local.ravel(), (rows, cols)), shape=(len(vertices), len(vertices))
    ).tocsr()
    robin = edges[markers == 2]
    lengths = np.linalg.norm(vertices[robin[:, 1]] - vertices[robin[:, 0]], axis=1)
    mass = (Bi * lengths[:, None, None] / 6 * np.array([[2.0, 1.0], [1.0, 2.0]])).ravel()
    rr = np.repeat(robin, 2, axis=1).ravel()
    cc = np.tile(robin, (1, 2)).ravel()
    matrix = diffusion + coo_matrix((mass, (rr, cc)), shape=diffusion.shape).tocsr()
    fixed = np.unique(edges[markers == 1])
    if not len(fixed):
        raise ValueError("FEM needs at least one source boundary segment")
    free = np.setdiff1d(np.arange(len(vertices)), fixed)
    u = np.ones(len(vertices))
    if Bi > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            u[free] = spsolve(
                matrix[free][:, free], -np.asarray(matrix[free][:, fixed].sum(axis=1)).ravel()
            )
    if not np.isfinite(u).all():
        raise ArithmeticError("Nonfinite FEM solution")
    reaction = matrix @ u
    residual = float(np.max(np.abs(reaction[free]))) if len(free) else 0.0
    Q = float(Bi * np.sum(lengths * u[robin].mean(axis=1)))
    influx = float(reaction[fixed].sum())
    if Bi == 0:
        influx = 0.0  # Exact constant solution, matching the reference zero-flux limit.
    error = abs(Q - influx) / max(abs(Q), abs(influx), 1e-14)
    if error > 1e-8 or residual > 1e-9:
        raise ArithmeticError(f"FEM failed conservation/residual: {error=}, {residual=}")
    return FEMResult(
        vertices,
        triangles,
        u,
        edges,
        markers,
        Q,
        influx,
        error,
        residual,
        float(areas.sum()),
        float(lengths.sum()),
        float(u @ (matrix @ u)),
    )


def solve_raster_fem(geometry: RasterGeometry, Bi: float, max_area: float = 1e-4) -> FEMResult:
    result = solve_pslg(raster_pslg(geometry), Bi, max_area)
    if abs(result.area - geometry.area) > 1e-10 * max(1.0, geometry.area):
        raise ArithmeticError("FEM triangulation changed raster air area, including holes")
    return result
