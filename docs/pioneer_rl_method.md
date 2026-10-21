# RL training with strong controls and validated starting geometries

Protocol fixed 2026-10-19 before preparation/training. This user-requested
extension takes priority over the pending equal-wall-time sampler study. It
does not close the earlier scientific advantage gate or authorize advanced
Monte Carlo/3D work. Good starts mean numerically validated improvements under
the same constraints, not globally optimal shapes.

## Preparation and readiness

The training bank has two canonical curriculum cases and ten pinned uniform
seed-8000--8004 improvements across deep-cavity/rectangle families. These are
development data already studied, not independent evaluation evidence. Source
hashes, metadata, source collars, physical constraints and recomputed Q are
checked. Every starting mask is evaluated at n=100/200/400.

Before any learner exists, create one validation start and two test starts per
family using generation seeds 61000 and 62000/62001, respectively. For each
seed/family, run screened-uniform, minimum-perimeter and current-flux pool
controls with 17 PDE solves and a strict 6000-raw-draw cap. Select the largest
n=400 Q among distinct eligible results with refined gain above both 1e-7 and
the last observed refinement change. Keep every generation run and its cost.
Failure to obtain a distinct improving start stops preparation; no budget or
quality threshold is relaxed. All methods later see identical frozen starts.

Check mask-hash disjointness across training/validation/test. The default bank
therefore has 12/2/4 starts. It covers two geometry families, not a population
of human acini. Pioneers selected this way are a challenging controlled
benchmark; selecting by numerical quality before training is not model-based
test selection. No validation/test trajectories are used for learning.

Preflight includes an analytical slab check, all five control environments,
a short discarded learning rollout on each canonical training case, checkpoint
round-trip and frozen reload evaluation. Final learners start with zero weights.
The complete regression suite must pass before the production command starts.

## Environment and controls

The full environment state is the current geometry/PDE solution, a random
candidate pool, remaining PDE/raw budgets and duplicate history. The learner
sees a 24-feature approximation. Each action selects one candidate in an
eight-draw pool. Cheap area/collar/perimeter checks mask the exploitation set;
a 25% unconditional slot-zero branch retains the original primitive's support.
If none passes the cheap checks, use slot zero. Full topology/feature checks
still run on the selected edit. Only actual global delta-Q>1e-7 accepts a shape.

The five controls are:

1. Uniform: one raw disk pair per decision.
2. Screened-uniform: uniform among cheaply screened pool members.
3. Perimeter pool: smallest Robin perimeter among screened members.
4. Flux pool: largest j_add*(max(j)-j_donor) among screened members, using only
   the current solved field. This is a heuristic, not a shape derivative, and
   is a pool-ranking control rather than the older joint probability sampler.
5. Untrained: zero action values through the same feature/pool environment.
   Its choices coincide with screened-uniform under identical seeds; its
   feature-computation overhead remains measurable.

The learner chooses by its action values among screened members, with internal
epsilon-greedy exploration during training, in addition to the fixed uniform
branch. Ties are uniform. Frozen evaluation uses internal epsilon=0 while
retaining the 25% branch. All methods obey identical geometry and PDE budgets;
all raw draws, invalid edits, selected checks, candidate solves and wall time
remain visible. No rejected candidate's PDE field is guessed.

Per-rollout `wall_seconds` times the environment, including raw proposal work,
full checks, solves and per-candidate NPZ writing. It excludes configuration
setup, final CSV/state serialization and later refinement. The root summary
separately records elapsed time for its latest invocation, so these rollout
times are not a complete workflow-cost or training-amortization comparison.

## Features, reward and update

Feature names are versioned in candidate_policy.py. State terms are bias,
remaining PDE/raw fractions, relative Q gain, perimeter fraction, eta and mean
interior u. Candidate terms are normalized radius, reallocated area, perimeter
change, addition/donor positions, current wall fluxes, their heuristic product
and cheap-pass indicator. Six interactions allow budget/state-dependent action
ranking. No candidate Q, fine-grid response, future acceptance or test score is
an input. Clipping is explicit only for relative gain; geometric and flux scales
use the declared dimensionless box, area, budget and Bi.

The score is linear: q_w(s,a)=w^T phi(s,a). Reward is -0.01 for a geometrically
invalid/duplicate selection, zero for a valid rejected edit, and
100*(Q_candidate-Q_current)/Q_start for an accepted edit. The physical objective
remains Q, separate from training reward and Q/Q_start optimization score.

For the actual mixed behavior policy pi_w, use the one-step Expected Sarsa
semi-gradient target and update:

\[
\delta=r+\gamma\sum_{a'}\pi_w(a'|s')w^T\phi(s',a')-w^T\phi(s,a),
\quad w\leftarrow w+\frac{\alpha}{1+\|\phi(s,a)\|^2}\delta\phi(s,a).
\]

At a terminal transition, the bootstrap term is zero. A next candidate pool is
generated only when another decision is possible, and every one of its raw
draws counts toward the same cap. The generated next pool is then used by the
next actual transition; there is no free hypothetical lookahead. A final
truncated pool consumes the remaining allowance and its selected action is
processed before termination. Gamma discounts per selected decision, including
invalid selections. The state-dependent step size bounds the feature-scaled
update magnitude; it is not a convergence theorem.

The default alpha is 0.05 and gamma 0.95. Five replicas use seeds 50000--50004,
each with 24 episodes. Episode seed is replica*1000+episode. Each curriculum
cycle uses a deterministic replica/cycle-specific permutation of training starts.
Internal epsilon decreases linearly from 1 to 0.15. Episodes have 17 solves,
including their initial solve, and 6000 raw draws. Exhausted episodes remain
explicit terminal transitions. Weights must remain finite and bounded; divergent
updates fail loudly while the last completed-episode checkpoint remains intact.

This is an approximate on-policy research learner, not an implementation of
Gradient Expected Sarsa or an assurance of optimal-policy convergence. Function
approximation and policy assumptions matter; for example, the off-policy
instability studied by [Yang et al. (2020)](https://arxiv.org/html/2012.07199)
does not provide a guarantee for this different control setup. The observations
alias different geometries; adding budgets does not make them a sufficient
Markov state. Numerical two-step/delayed-reward and terminal tests verify the
implemented update before use.

## Frozen evaluation and checkpoints

Use final weights from all five replicas. Do not choose a best episode or
replica from validation/test outcomes. Validation rollout seed 71000 and test
seeds 72000/72001 are separate from training and pioneer generation. Compare
five controls plus five final models on 2 validation starts and 4 test starts:
20 validation and 80 test rollouts at the default settings. Every evaluation
initial/final state is solved at 200/400. Report all methods, failures, paired
start/seed wins/ties/losses and variability; shared controls across replicas
are not independent experiments. Include preparation, preflight and training
costs separately from evaluation. No automatic policy promotion occurs.

Atomic NPZ checkpoints contain feature schema, weights, update count and the
completed-episode history in one file, loaded without pickle. Their identity
pins the protocol, bank and executed sources. Resuming starts from the next
uncommitted episode with its deterministic seed; an interrupted episode is
replayed. Completed evaluations are skipped after frozen-model checks.
Preparation must finish before training can be resumed; a preparation failure
retains its evidence and requires a fresh preparation run.

```powershell
# Prepare bank and run readiness checks only
.\.venv\Scripts\python.exe -m oxygen_shape.cli train-rl-candidates --config configs/train_rl_candidates.yaml --prepare-only

# Prepare, train five replicas, and evaluate all controls/final models
.\.venv\Scripts\python.exe -m oxygen_shape.cli train-rl-candidates --config configs/train_rl_candidates.yaml

# Continue a prepared or interrupted compatible run
.\.venv\Scripts\python.exe -m oxygen_shape.cli train-rl-candidates --config configs/train_rl_candidates.yaml --resume "RUN_DIRECTORY"
```

The old tabular policy and commands remain available and unchanged. No GPU or
new deep-learning dependency is added. All results concern dimensionless 2D
diffusion with the existing Robin/source model and feature constraints.

The [first completed study](pioneer_rl_results.md) documents a post-run replica
label correction and preserves its original executed source identity. The live
reporting fix consequently fails that completed run's strict source guard on
resume; future compatible runs support the commands above. Its stored policies
still load with their recorded identity, without retraining or numerical changes.
