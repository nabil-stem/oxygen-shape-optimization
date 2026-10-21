import numpy as np
import pytest

from oxygen_shape.optimization.config import RLConfig
from oxygen_shape.optimization.q_learning import QPolicy
from oxygen_shape.optimization.random_search import run_search


def test_q_learning_bootstraps_future_reward_and_handles_terminal():
    policy = QPolicy()
    # Two-step MDP: state 0/action 0 has zero immediate reward but reaches
    # state 1/action 1 with reward 2; action 1 in state 0 has reward only 1.
    for _ in range(100):
        policy.update(1, 1, 2, 0, alpha=0.2, gamma=0.9, terminal=True)
        policy.update(0, 0, 0, 1, alpha=0.2, gamma=0.9, terminal=False)
        policy.update(0, 1, 1, 0, alpha=0.2, gamma=0.9, terminal=True)
    assert policy.values[0, 0] == pytest.approx(1.8, abs=1e-6)
    assert policy.values[0, 1] == pytest.approx(1, abs=1e-6)
    assert policy.choose(0, np.random.default_rng(1), epsilon=0) == 0


def test_checkpoint_roundtrip_and_frozen_evaluation(tmp_path):
    config = RLConfig(
        training_episodes=1, training_solve_budget=4, solve_budget=4, seed=42, evaluation_seeds=[0]
    )
    policy = QPolicy()
    trained = run_search(config, policy=policy, epsilon=1, training=True)
    assert policy.visits.sum() == trained.attempts
    assert np.any(policy.values != 0)
    path = tmp_path / "policy.npz"
    policy.save(path)
    restored = QPolicy.load(path)
    values, visits = restored.values.copy(), restored.visits.copy()
    run_search(config, policy=restored, epsilon=0.1, training=False)
    np.testing.assert_array_equal(restored.values, values)
    np.testing.assert_array_equal(restored.visits, visits)


def test_training_seeds_cannot_overlap_evaluation():
    with pytest.raises(ValueError, match="disjoint"):
        RLConfig(seed=42, training_seed_offset=0, evaluation_seeds=[42])


def test_training_attempt_cap_is_terminal_and_keeps_actual_cost():
    config = RLConfig(training_episodes=1, solve_budget=25, max_attempts_per_solve=1)
    policy = QPolicy()
    result = run_search(config, policy=policy, epsilon=1, training=True)
    assert result.attempts <= 25
    assert 1 <= result.number_of_pde_solves <= 25
    assert policy.visits.sum() == result.attempts


def test_seeded_learning_reproduces_weights_and_trajectory():
    config = RLConfig(solve_budget=4, seed=42)
    first, second = QPolicy(), QPolicy()
    a = run_search(config, policy=first, epsilon=0.5, training=True)
    b = run_search(config, policy=second, epsilon=0.5, training=True)
    assert a.proposals == b.proposals
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.visits, second.visits)


def test_frozen_policy_uses_uniform_fallback_after_infeasible_proposals():
    config = RLConfig(solve_budget=5, seed=42, policy_stall_limit=1, max_attempts_per_solve=100)
    policy = QPolicy()
    policy.values[:, 11] = 1
    values = policy.values.copy()
    result = run_search(config, policy=policy, epsilon=0)
    assert result.number_of_pde_solves == 5
    assert any(row["uniform_fallback"] for row in result.proposals)
    np.testing.assert_array_equal(policy.values, values)
