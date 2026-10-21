# RL proposal policy: implementation and use

This user-requested extension follows the validated deterministic solver,
baseline shape study, and constrained random disk optimizer. It implements
**tabular Q-learning** in NumPy. The policy chooses where and at what radius to
propose a disk edit; the finite-volume solver remains the physical evaluator.
The low-concentration deletion hypothesis and sensitivity-guided comparison are
still separate research milestones.

## State, action, transition, reward

The environment contains the full geometry and latest solved concentration field.
The agent observes a coarse state index:

```
s = 4 * (Q / Q_initial >= 1.10)
  + 2 * (Robin_perimeter / perimeter_limit >= 0.85)
  +     (mean interior u at x>0.65 >= 0.10)
```

These three binary features give eight observation states. Empty distal air
has mean zero by convention. The features include actual global flux and
concentration data from the current solved domain; no candidate concentration
is guessed by the policy.

There are twelve actions: six spatial regions times two radius bins. The region
index is `2*x_third + upper_half`, with x thirds of the unit box and the split
y=0.5. A radius is sampled uniformly within the lower or upper half of the
configured interval. Choose an eligible addition face uniformly within that
region, and an independent removal donor uniformly over all eligible faces.
Angular exploration, exact paired area balance, and validity checks are the
same as in [the random disk optimizer](optimization_v0.md).

An invalid action leaves the geometry unchanged and receives reward -0.01.
A valid candidate is solved and accepted only when its Q exceeds the current Q
by the configured absolute minimum gain. A valid rejected candidate receives
zero reward. An accepted candidate receives

```
reward = 100 * (Q_candidate - Q_current) / Q_initial
```

The physical optimization objective and score remain Q and Q/Q_initial.
Training reward also discourages infeasible proposals; it is not substituted for
the reported physical objective or constraints. The policy never commands
automatic deletion of a low-u region. Every accepted change is evaluated globally.

## Learning equation and assumptions

```
target = reward + gamma * max_a Q_table[next_state, a]
Q_table[state, action] += alpha * (target - Q_table[state, action])
```

At a terminal transition the bootstrap term is zero. The default learning rate
is 0.15 and discount is 0.9. Training is epsilon-greedy, with epsilon decreasing
linearly from 1.0 to 0.15 over forty episodes. Equal-valued greedy actions are
chosen at random, so an untrained zero table has no hard-coded preferred action.
The final training checkpoint is used; held-out evaluation is not used to select
an earlier checkpoint or tune the policy.

This is the action-value update of
[Watkins and Dayan, *Q-learning* (1992)](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).
The eight-state observation aggregates different geometries and omits remaining
budget. It is a partially observed approximation, not a full-state Markov model.
The finite training schedule and constant learning rate also do not establish
the assumptions of the paper's convergence theorem. No optimal-policy or global
shape-optimum claim follows from this implementation.

A step is a proposal attempt, including invalid proposals. Consequently gamma
discounts per attempt. The policy learns both flux gains and proposal feasibility.
Equal PDE-solve budgets are only one comparison axis; invalid attempts and wall
time must also be reported.

## Training and evaluation protocol

Training starts each episode from the configured initial geometry. The seed for
episode i is `training_seed_offset + seed + i`. Each episode has a solve cap
(including its initial solve) and a proposal-attempt cap. Reaching either cap
ends training for that episode, with actual counts retained. An early run exposed
feasibility exhaustion, so training now handles that case as an explicit terminal
transition instead of discarding the learned checkpoint.

After training the table is frozen. Five held-out seeds compare:

1. Uniform eligible boundary-measure sampling with a uniform radius.
2. An untrained zero-table policy, using the same twelve-action parameterization.
3. The trained policy, with evaluation epsilon 0.1.

During frozen evaluation, after fifty consecutive invalid proposals the trained
and untrained policies switch to the uniform boundary sampler until one valid
candidate resets the counter. This is a hybrid learned proposal policy with a
uniform feasibility safeguard; fallback attempts are counted separately.
Training uses no fallback, so every training action remains an explicit table
action. The pure-policy pilot failed to reach its solve budget (24 of 41 solves
after 4100 attempts on seed 0). The revised protocol uses fresh seeds 100–104,
retains the same checkpoint, and records the failed pilot rather than pooling it
with the final benchmark. No reward-based tuning or model selection was done.

All three use identical starting domains, constraints, acceptance thresholds,
and actual evaluation PDE budgets. A budget shortfall fails evaluation loudly;
it is not silently compared with a completed run. We record training cost,
evaluation cost, refinement cost, proposal attempts, wall time, and results for
every seed. The benchmark plot excludes training from its x-axis and labels that
choice; the report includes the omitted training cost explicitly.

Initial and final geometries are re-solved on 2x and 4x grids. Shape constraints
must still pass there. This refines the same physical raster cell union, so area,
perimeter, source, and staircase orientation are preserved. Mesh changes in the
gain supply an observed sensitivity check, not a rigorous continuum error bound.

Held-out seeds assess stochastic search on one initial geometry at one Bi.
They do not establish generalization to new geometries, sources, or regimes.

## Commands and checkpoint

```powershell
# Random-search benchmark
.\.venv\Scripts\python.exe -m oxygen_shape.cli optimize --config configs/optimize_disks.yaml

# Train, freeze, and benchmark against both controls
.\.venv\Scripts\python.exe -m oxygen_shape.cli train-rl --config configs/train_rl.yaml

# Reuse the saved checkpoint without retraining
.\.venv\Scripts\python.exe -m oxygen_shape.cli evaluate-rl --config configs/train_rl.yaml --policy models/disk_q_policy.npz

# Re-run the five-seed comparison from an existing training directory
.\.venv\Scripts\python.exe -m oxygen_shape.cli benchmark-rl --config configs/train_rl.yaml --training-run "PATH_TO_TRAINING_RUN"
```

`evaluate-rl` uses the seed in the configuration, rejects a training seed, and
checks that geometry, physics, constraints, and action radii match the checkpoint
metadata. Evaluation budgets and seed may change. Keep `policy_metadata.json`
beside `policy.npz` when copying the checkpoint.

The checkpoint contains learned values, visit counts, and a schema version.
It uses no pickle. The metadata defines the state/action encoding, radii, original
training configuration, and frozen-evaluation status. The CLI re-evaluation also
stores a SHA-256 checksum of the loaded checkpoint.

Relevant modules:

- `optimization/q_learning.py`: observation, epsilon-greedy policy, temporal-difference update, checkpoint.
- `optimization/random_search.py`: shared PDE-scored transition and greedy acceptance.
- `geometry/disk_edits.py`: random disk proposals, independent of learned values and concentration.
- `geometry/constraints.py`: source, area, perimeter, connectivity, holes, and feature checks.
- `experiments/rl.py`: training episodes and saved checkpoints.
- `experiments/rl_benchmark.py`: held-out benchmark with frozen-weight verification.
- `experiments/policy_evaluation.py`: reuse a learned checkpoint.

Tests verify the update on a two-step MDP with delayed reward, terminal handling,
deterministic seeded trajectories and weights, checkpoint round-trip, frozen
evaluation, disjoint seeds, equal evaluation budgets, and physical constraints.
Results are recorded separately in [the experiment report](rl_results.md).
