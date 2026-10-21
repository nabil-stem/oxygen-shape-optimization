"""Independent amplitude-scaling diagnostic, never an optimizer source setting."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve

from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.physics.model import DiffusionModel, SolverSettings
from oxygen_shape.solvers.diagnostics import conservation_error, require_conservation
from oxygen_shape.solvers.finite_volume import assemble


@dataclass(frozen=True)
class AmplitudeResult:
    v: np.ndarray
    normalized_u: np.ndarray
    flux: float
    normalized_flux: float
    eta: float | None
    conservation_error: float
    normalized_residual: float


def solve_source_amplitude(
    geometry: RasterGeometry,
    model: DiffusionModel,
    amplitude: float,
    settings: SolverSettings | None = None,
) -> AmplitudeResult:
    """Solve A*v=a*b independently; v uses a fixed concentration reference.

    The physical normalization for the changed drive is u=v/a, with u_D=1.
    See docs/source_amplitude_derivation.md and the analytical slab check.
    """
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("The reference driving concentration factor must be positive and finite")
    settings = settings or SolverSettings()
    system = assemble(geometry, model)
    values = (
        np.full(system.rhs.size, amplitude)
        if model.Bi == 0
        else np.asarray(spsolve(system.matrix, amplitude * system.rhs))
    )
    normalized = values / amplitude
    residual = float(np.max(np.abs(system.matrix @ values - amplitude * system.rhs)) / amplitude)
    if not np.isfinite(values).all() or residual > settings.residual_tolerance:
        raise ArithmeticError("Amplitude diagnostic failed its finite/residual check")
    if (
        normalized.min() < -settings.bounds_tolerance
        or normalized.max() > 1 + settings.bounds_tolerance
    ):
        raise ArithmeticError("Amplitude diagnostic violates the normalized maximum principle")
    source = np.array([f.kind == BoundaryKind.SOURCE_DIRICHLET for f in system.faces])
    robin = np.array([f.kind == BoundaryKind.ROBIN for f in system.faces])
    face_flux = system.conductances * (values[system.face_ids] - amplitude * source.astype(float))
    Q = float(face_flux[robin].sum())
    influx = float(-face_flux[source].sum())
    require_conservation(influx, Q, settings)
    perimeter = float(robin.sum() * geometry.h)
    ideal = amplitude * model.Bi * perimeter
    field = np.full(geometry.inside.shape, np.nan)
    field[geometry.inside] = values
    return AmplitudeResult(
        field,
        field / amplitude,
        Q,
        Q / amplitude,
        Q / ideal if ideal else None,
        conservation_error(influx, Q),
        residual,
    )
