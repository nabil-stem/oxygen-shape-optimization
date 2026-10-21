"""Exact normalized slab solution with source at x=0 and Robin wall at x=L."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def slab_flux(Bi: float, length: float = 1.0) -> float:
    if not np.isfinite(Bi) or Bi < 0 or not np.isfinite(length) or length <= 0:
        raise ValueError("Bi must be finite and nonnegative; length must be finite and positive")
    return 0.0 if Bi == 0 else float(1.0 / (length + 1.0 / Bi))


def slab_solution(x: ArrayLike, Bi: float, length: float = 1.0) -> NDArray[np.float64]:
    q = slab_flux(Bi, length)
    coordinates = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(coordinates)) or np.any((coordinates < 0) | (coordinates > length)):
        raise ValueError("Slab coordinates must lie in [0, length]")
    return 1.0 - q * coordinates
