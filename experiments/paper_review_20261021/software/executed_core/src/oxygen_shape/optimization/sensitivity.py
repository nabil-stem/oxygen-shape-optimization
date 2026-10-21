"""Signed, empirical pair responses; these are not shape derivatives."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PairResponse:
    addition: tuple[float, float]
    donor: tuple[float, float]
    delta_Q: float
    reallocated_area: float

    def __post_init__(self) -> None:
        values = (*self.addition, *self.donor, self.delta_Q, self.reallocated_area)
        if len(self.addition) != 2 or len(self.donor) != 2 or not np.isfinite(values).all():
            raise ValueError("Pair responses require finite 2D points and values")
        if self.reallocated_area <= 0:
            raise ValueError("Reallocated area must be positive; net area is zero")

    @property
    def sensitivity(self) -> float:
        return self.delta_Q / self.reallocated_area


def mixture_probabilities(weights: NDArray, epsilon: float) -> NDArray[np.float64]:
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 2
        or not weights.size
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("Pair weights must be a nonempty finite nonnegative matrix")
    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Exploration probability must be in (0,1]")
    uniform = np.full(weights.shape, 1 / weights.size)
    if weights.max() == 0:
        return uniform
    scaled = weights / weights.max()
    return epsilon * uniform + (1 - epsilon) * scaled / scaled.sum()


def sensitivity_probabilities(
    points: NDArray,
    responses: list[PairResponse],
    bandwidth: float,
    epsilon: float,
) -> NDArray[np.float64]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Boundary points must be a nonempty finite (N,2) array")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth must be finite and positive")
    benefit = np.zeros((len(points), len(points)))
    for response in responses:
        addition = np.exp(-np.sum((points - response.addition) ** 2, axis=1) / (2 * bandwidth**2))
        donor = np.exp(-np.sum((points - response.donor) ** 2, axis=1) / (2 * bandwidth**2))
        benefit += np.outer(addition, donor) * response.sensitivity
    return mixture_probabilities(np.maximum(benefit, 0), epsilon)


def flux_probabilities(flux: NDArray, epsilon: float) -> NDArray[np.float64]:
    flux = np.asarray(flux, dtype=float)
    if flux.ndim != 1 or not flux.size or not np.isfinite(flux).all() or np.any(flux < 0):
        raise ValueError("Robin flux must be a nonempty finite nonnegative vector")
    if flux.max() == 0:
        return mixture_probabilities(np.zeros((len(flux), len(flux))), epsilon)
    scaled = flux / flux.max()
    return mixture_probabilities(np.outer(scaled, 1 - scaled), epsilon)
