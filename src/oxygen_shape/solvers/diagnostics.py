"""Conservation checks include an absolute floor for near-zero flux."""

import numpy as np

from oxygen_shape.physics.model import SolverSettings


def conservation_error(source: float, robin: float, epsilon: float = 1e-14) -> float:
    return abs(source - robin) / max(abs(source), abs(robin), epsilon)


def require_conservation(source: float, robin: float, settings: SolverSettings) -> None:
    difference = abs(source - robin)
    threshold = settings.conservation_atol + settings.conservation_rtol * max(
        abs(source), abs(robin)
    )
    if not np.isfinite(difference) or difference > threshold:
        raise ArithmeticError(
            f"Flux conservation failed: mismatch={difference:.4g}, limit={threshold:.4g}"
        )
