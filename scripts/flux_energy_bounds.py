"""Complementary-energy lower estimates from saved cell fields; no PDE solve."""

from dataclasses import dataclass

import numpy as np

from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.raster import Side


@dataclass(frozen=True)
class FluxLowerEstimate:
    lower: float
    complementary_value: float
    source_influx: float
    wall_absorption: float
    bulk_energy: float
    wall_resistance_energy: float
    divergence_residual_l1: float
    divergence_residual_max: float
    floating_guard: float


def primal_upper_estimate(vertices, triangles, concentration, edges, markers, Bi):
    """Integrate the represented P1 function directly, without an assembled matrix."""
    points = np.asarray(vertices)[triangles]
    u = np.asarray(concentration)
    values = u[triangles]
    twice_area = np.abs(
        (points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1])
        - (points[:, 2, 0] - points[:, 0, 0]) * (points[:, 1, 1] - points[:, 0, 1])
    )
    b = points[:, [1, 2, 0], 1] - points[:, [2, 0, 1], 1]
    c = points[:, [2, 0, 1], 0] - points[:, [1, 2, 0], 0]
    bulk = float(
        np.sum(((values * b).sum(axis=1) ** 2 + (values * c).sum(axis=1) ** 2) / (2 * twice_area))
    )
    robin = edges[np.asarray(markers).ravel() == 2]
    length = np.linalg.norm(vertices[robin[:, 0]] - vertices[robin[:, 1]], axis=1)
    a, z = u[robin].T
    wall = float(Bi * np.sum(length * (a * a + a * z + z * z)) / 3)
    fixed = np.unique(edges[np.asarray(markers).ravel() == 1])
    if not len(fixed) or not np.all(u[fixed] == 1):
        raise ValueError("Primal bound requires exactly imposed source value one")
    energy = bulk + wall
    guard = 64 * np.finfo(float).eps * (len(triangles) + len(edges)) * (1 + energy)
    return energy + guard, energy, guard


def lower_flux_estimate(geometry, field, Bi):
    if not np.isfinite(Bi) or Bi <= 0:
        raise ValueError("Complementary Robin estimate requires finite Bi>0")
    mask = geometry.inside
    u = np.asarray(field, float)
    if u.shape != mask.shape or not np.isfinite(u[mask]).all():
        raise ValueError("Saved field does not match the physical raster domain")
    flux = np.zeros((*mask.shape, 4))  # west, east, south, north integrated outward flux
    horizontal = mask[:, :-1] & mask[:, 1:]
    value = np.where(horizontal, u[:, :-1] - u[:, 1:], 0.0)
    flux[:, :-1, 1] = value
    flux[:, 1:, 0] = -value
    vertical = mask[:-1, :] & mask[1:, :]
    value = np.where(vertical, u[:-1, :] - u[1:, :], 0.0)
    flux[:-1, :, 3] = value
    flux[1:, :, 2] = -value
    index = {Side.WEST: 0, Side.EAST: 1, Side.SOUTH: 2, Side.NORTH: 3}
    conductance = Bi * geometry.h / (1 + Bi * geometry.h / 2)
    source, wall = [], []
    for face in extract_boundary(geometry):
        r, c = face.key.row, face.key.col
        if face.kind == BoundaryKind.SOURCE_DIRICHLET:
            value = 2 * (u[r, c] - 1)
            source.append(value)
        elif face.kind == BoundaryKind.ROBIN:
            value = conductance * u[r, c]
            wall.append(value)
        else:
            value = 0.0
        flux[r, c, index[face.key.side]] = value
    w, e, s, n = flux[mask].T
    bulk = float(np.sum(w * w - w * e + e * e + s * s - s * n + n * n) / 3)
    reactive = float(np.sum(np.square(wall)) / (Bi * geometry.h))
    influx, absorption = float(-np.sum(source)), float(np.sum(wall))
    residual = np.sum(flux[mask], axis=1)
    residual_l1 = float(np.abs(residual).sum())
    complementary = 2 * influx - bulk - reactive
    # Conservative practical guard, distinct from a directed-rounding proof.
    guard = (
        64
        * np.finfo(float).eps
        * (int(mask.sum()) + len(source) + len(wall))
        * (1 + abs(influx) + bulk + reactive)
    )
    lower = complementary - 2 * residual_l1 - guard
    return FluxLowerEstimate(
        lower,
        complementary,
        influx,
        absorption,
        bulk,
        reactive,
        residual_l1,
        float(np.abs(residual).max()),
        float(guard),
    )
