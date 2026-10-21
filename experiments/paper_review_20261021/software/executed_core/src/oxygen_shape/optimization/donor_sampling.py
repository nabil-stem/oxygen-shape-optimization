"""Flux is only a proposal feature; candidate effects require a new PDE solve."""

import numpy as np
from numpy.typing import NDArray

from oxygen_shape.geometry.boundary import BoundaryKind
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.solvers.finite_volume import SolveResult


def eligible_wall_flux(
    geometry: RasterGeometry, solution: SolveResult, protected_length: float
) -> NDArray[np.float64]:
    """Return j=Bi*u_wall in exactly the order used by boundary_samples."""
    return np.asarray(
        [
            solution.outward_face_flux[i] / geometry.h
            for i, face in enumerate(solution.faces)
            if face.kind == BoundaryKind.ROBIN
            and (face.key.col + 0.5) * geometry.h >= protected_length + 2 * geometry.h
        ],
        dtype=float,
    )


def low_flux_weights(flux: NDArray, fraction: float) -> NDArray[np.float64]:
    """Uniform face measure within the bottom quantile, including threshold ties."""
    flux = np.asarray(flux, dtype=float)
    if flux.ndim != 1 or not flux.size or not np.isfinite(flux).all():
        raise ValueError("Flux must be a nonempty finite vector")
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Low-flux fraction must lie in (0, 1)")
    weights = (flux <= np.quantile(flux, fraction)).astype(float)
    return weights / weights.sum()
