"""Small tabular off-policy Q-learning baseline, using NumPy only."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.geometry.raster import RasterGeometry
from oxygen_shape.solvers.finite_volume import SolveResult

STATE_COUNT = 8
ACTION_COUNT = 12  # six boundary regions, two radius bins


def observe(
    geometry: RasterGeometry, result: SolveResult, reference_Q: float, perimeter_limit: float
) -> int:
    x = (np.arange(geometry.inside.shape[1]) + 0.5) * geometry.h
    distal = result.concentration[:, x > 0.65]
    finite = distal[np.isfinite(distal)]
    utilization = float(finite.mean()) if len(finite) else 0.0
    gain_bit = int(result.metrics.physical_flux_Q / reference_Q >= 1.10)
    perimeter_bit = int(raster_perimeter(geometry) / perimeter_limit >= 0.85)
    concentration_bit = int(utilization >= 0.1)
    return 4 * gain_bit + 2 * perimeter_bit + concentration_bit


@dataclass
class QPolicy:
    values: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((STATE_COUNT, ACTION_COUNT))
    )
    visits: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros((STATE_COUNT, ACTION_COUNT), dtype=np.int64)
    )

    def choose(self, state: int, rng: np.random.Generator, epsilon: float) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(ACTION_COUNT))
        row = self.values[state]
        # Random tie breaking avoids a baked-in preference for action zero.
        eligible = np.flatnonzero(row >= row.max() - 1e-12)
        return int(rng.choice(eligible))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        *,
        alpha: float,
        gamma: float,
        terminal: bool,
    ) -> float:
        target = reward + (0.0 if terminal else gamma * float(self.values[next_state].max()))
        error = target - self.values[state, action]
        self.values[state, action] += alpha * error
        self.visits[state, action] += 1
        return float(error)

    def save(self, path: Path) -> None:
        np.savez(path, values=self.values, visits=self.visits, schema_version=np.array(1))

    @classmethod
    def load(cls, path: Path) -> "QPolicy":
        with np.load(path, allow_pickle=False) as data:
            values, visits = data["values"].copy(), data["visits"].copy()
            if (
                values.shape != (STATE_COUNT, ACTION_COUNT)
                or visits.shape != values.shape
                or not np.isfinite(values).all()
                or np.any(visits < 0)
                or int(data["schema_version"]) != 1
            ):
                raise ValueError("Invalid Q-policy checkpoint")
            return cls(values, visits)
