"""Convert saved histories to curves using positional data, not index alignment."""

import numpy as np
import pandas as pd


def gain_by_solve(history: pd.DataFrame) -> pd.Series:
    """Keep the initial zero and final gain at their actual charged solve counts."""
    solves = history["pde_solves"].to_numpy(dtype=float)
    flux = history["best_Q"].to_numpy(dtype=float)
    if (
        not len(flux)
        or not np.isfinite(solves).all()
        or not np.isfinite(flux).all()
        or flux[0] <= 0
        or (solves < 1).any()
        or (solves != np.floor(solves)).any()
        or (np.diff(solves) <= 0).any()
        or (np.diff(flux) < -1e-13).any()
    ):
        raise ValueError("Finite best-flux history and increasing integer solve counts required")
    return pd.Series(
        100 * (flux / flux[0] - 1),
        index=solves.astype(np.int64),
        name="best_coarse_gain_percent",
    )
