# RL with admissible candidate pools

Frozen before outcomes: `configs/train_rl_feasible.yaml`.

Train five independent linear Expected Sarsa policies, 24 episodes each, cycling
through 12 starts (two canonical and ten pinned earlier improvements). Use the
validated 24 incumbent-only features. Initialize a fixed flux-product prior of
one and train its additive residual weights at learning rate 0.05. The smoke
learner is discarded. No candidate PDE solution enters selection features.

Each pooled decision refills until four distinct fully admissible candidates
exist, after at least eight raw draws, or 256 draws are spent. Each episode has
17 PDE solves including its initial solve and a hard 6000 raw-draw cap. Preserve
the unconditional first raw draw with probability 0.25; otherwise use an
epsilon-greedy selection among eligible candidates, epsilon 1 to 0.15. Separate
proposal and selection random streams. Charge the next pool once before the
Expected Sarsa bootstrap and consume that exact pool on the next transition.

Reward is 100 times the accepted global increment in Q divided by initial Q.
Invalid and nonimproving actions receive zero. With discount one and bounded
episodes, total undiscounted reward telescopes to 100 times relative final Q
gain; this is not a claim that approximate function approximation must converge.

Keep the source normalized to one, Bi=1, area 0.24, perimeter at most four,
protected collar 0.12, connected hole-free airspace, and feature width 0.04
checked at 100/200/400. Plot contrast transforms never enter the solver or RL.

Before learning, generate two validation and four test geometries with fresh
generation seeds and three fixed controls. Select by refined flux using the
existing improvement/grid-sensitivity gate; reject all known historical masks
listed in the frozen config as well as cross-split duplicates. Validation and
test outcomes do not select replicas, hyperparameters or a promoted model.

Evaluate all five frozen final replicas and five controls on every held-out
start with two fresh rollout seeds: 40 validation and 80 test runs. Controls
are raw uniform, screened pool uniform, minimum-perimeter ranking, flux ranking,
and the untrained zero-score policy. The latter duplicates screened uniform
in distribution and is a consistency check. Flux ranking tests whether learning
adds value to its own fixed initialization. All pooled methods use identical
refill rules. Raw uniform uses one raw proposal per decision. Report PDE counts,
all raw draws, geometry checks, completion, Q, eta, refinement, and wall time.
Shared starts/controls across replicas are not independent samples. No claim of
equal-time performance or global optimality follows from equal PDE budgets.

Atomic checkpoints commit after each completed episode, including policy prior,
update count and episode records. Resume checks source, config, numerical
environment and bank identity. Uncommitted rollout directories are preserved
before replay; discarded PDE work is reported separately with bounds. Initial
bank preparation must finish before checkpoint-based resume is available.

The planned cost is 414 preparation, 40 preflight, at most 2040 training, 2040
evaluation, and 480 endpoint-refinement PDE solves: 5014 total, plus separately
reported interruption costs if any. Earlier reused pioneer generation costs
are historical, not new solves. The CPU implementation follows profiling: this
linear model is inexpensive; geometry validation is the measured bottleneck.
