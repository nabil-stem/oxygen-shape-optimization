import numpy as np
import pandas as pd
import pytest

from oxygen_shape.viz.history import gain_by_solve


@pytest.mark.parametrize("index", [[0, 1, 2], [8, 19, 40]])
def test_curve_keeps_initial_and_final_values_at_actual_solve_counts(index):
    history = pd.DataFrame({"pde_solves": [1, 2, 3], "best_Q": [0.4, 0.44, 0.48]}, index=index)
    curve = gain_by_solve(history)
    np.testing.assert_array_equal(curve.index, [1, 2, 3])
    np.testing.assert_allclose(curve.to_numpy(), [0, 10, 20], atol=1e-12)
    assert not curve.isna().any()


@pytest.mark.parametrize("solves", [[1, 1, 3], [1, 2.5, 3]])
def test_curve_rejects_ambiguous_or_fractional_solve_counts(solves):
    with pytest.raises(ValueError, match="integer solve counts"):
        gain_by_solve(pd.DataFrame({"pde_solves": solves, "best_Q": [0.4, 0.44, 0.48]}))
