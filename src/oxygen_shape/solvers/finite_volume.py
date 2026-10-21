"""Cell-centered, conservative finite-volume discretization on square cells."""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from oxygen_shape.geometry.boundary import BoundaryFace, BoundaryKind, extract_boundary
from oxygen_shape.geometry.constraints import validate_geometry
from oxygen_shape.geometry.raster import OFFSETS, RasterGeometry
from oxygen_shape.physics.metrics import FluxMetrics
from oxygen_shape.physics.model import DiffusionModel, SolverSettings
from oxygen_shape.solvers.diagnostics import conservation_error, require_conservation


@dataclass(frozen=True)
class LinearSystem:
    matrix: csr_matrix
    rhs: NDArray[np.float64]
    cell_ids: NDArray[np.int64]
    faces: tuple[BoundaryFace, ...]
    face_ids: NDArray[np.int64]
    conductances: NDArray[np.float64]


@dataclass(frozen=True)
class SolveResult:
    concentration: NDArray[np.float64]
    metrics: FluxMetrics
    faces: tuple[BoundaryFace, ...]
    outward_face_flux: NDArray[np.float64]
    wall_concentration: NDArray[np.float64]
    conductances: NDArray[np.float64]


def robin_conductance(Bi: float, h: float) -> float:
    if not np.isfinite(Bi) or Bi < 0 or not np.isfinite(h) or h <= 0:
        raise ValueError("Bi must be finite and nonnegative; h must be positive and finite")
    return 0.0 if Bi == 0 else float(h / (h / 2.0 + 1.0 / Bi))


def assemble(geometry: RasterGeometry, model: DiffusionModel) -> LinearSystem:
    validate_geometry(geometry).require_valid()
    faces = extract_boundary(geometry)
    ids = np.full(geometry.inside.shape, -1, dtype=np.int64)
    count = int(geometry.inside.sum())
    ids[geometry.inside] = np.arange(count)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    diagonal = np.zeros(count)
    rhs = np.zeros(count)
    ny, nx = ids.shape
    for row, col in np.argwhere(geometry.inside):
        i = int(ids[row, col])
        for dy, dx in OFFSETS.values():
            y, x = row + dy, col + dx
            if 0 <= y < ny and 0 <= x < nx and ids[y, x] >= 0:
                rows.append(i)
                cols.append(int(ids[y, x]))
                values.append(-1.0)
                diagonal[i] += 1.0
    face_ids = np.array([ids[f.key.row, f.key.col] for f in faces], dtype=np.int64)
    gr = robin_conductance(model.Bi, geometry.h)
    conductances = np.array(
        [
            2.0
            if f.kind == BoundaryKind.SOURCE_DIRICHLET
            else gr
            if f.kind == BoundaryKind.ROBIN
            else 0.0
            for f in faces
        ]
    )
    np.add.at(diagonal, face_ids, conductances)
    source = np.array([f.kind == BoundaryKind.SOURCE_DIRICHLET for f in faces])
    np.add.at(rhs, face_ids[source], conductances[source])
    rows.extend(range(count))
    cols.extend(range(count))
    values.extend(diagonal.tolist())
    matrix = coo_matrix((values, (rows, cols)), shape=(count, count)).tocsr()
    return LinearSystem(matrix, rhs, ids, faces, face_ids, conductances)


def solve(
    geometry: RasterGeometry,
    model: DiffusionModel,
    settings: SolverSettings | None = None,
) -> SolveResult:
    settings = settings or SolverSettings()
    system = assemble(geometry, model)
    # With Bi=0, the unique source-pinned solution is exactly constant. Preserve
    # the exact zero-flux limit instead of amplifying roundoff in a relative ratio.
    if model.Bi == 0:
        u = np.ones(system.rhs.size)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            u = np.asarray(spsolve(system.matrix, system.rhs))
    if not np.all(np.isfinite(u)):
        raise ArithmeticError("Solver returned nonfinite concentrations")
    residual = float(np.max(np.abs(system.matrix @ u - system.rhs)))
    if residual > settings.residual_tolerance:
        raise ArithmeticError(f"Linear-system residual {residual:.4g} exceeds tolerance")
    if u.min() < -settings.bounds_tolerance or u.max() > 1 + settings.bounds_tolerance:
        raise ArithmeticError("Concentration violates the discrete maximum principle")
    kinds = np.array([f.kind.value for f in system.faces])
    source = kinds == BoundaryKind.SOURCE_DIRICHLET.value
    robin = kinds == BoundaryKind.ROBIN.value
    adjacent = u[system.face_ids]
    flux = system.conductances * (adjacent - source.astype(float))
    q_source, q_robin = float(-flux[source].sum()), float(flux[robin].sum())
    require_conservation(q_source, q_robin, settings)
    perimeter = float(robin.sum() * geometry.h)
    ideal = model.Bi * perimeter
    eta = q_robin / ideal if ideal > 0 else None
    wall = adjacent - flux / 2.0  # half-cell resistance per face is 1/2 in 2D
    field = np.full(geometry.inside.shape, np.nan)
    field[geometry.inside] = u
    metrics = FluxMetrics(
        q_robin,
        q_source,
        ideal,
        eta,
        geometry.area,
        perimeter,
        float(source.sum() * geometry.h),
        conservation_error(q_source, q_robin),
        residual,
    )
    return SolveResult(field, metrics, system.faces, flux, wall, system.conductances)
