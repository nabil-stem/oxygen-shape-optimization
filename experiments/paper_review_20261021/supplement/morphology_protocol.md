# Frozen development protocol: literature-inspired geometries and long trajectories

This extends the publication critique with executable comparisons. It is an
exploratory development study; no biological optimality or general RL superiority
is presupposed. Hyperparameters are fixed before the new study's outcomes.
The short implementation profile on the seed-0 hexagonal control is development
knowledge and is separate from the study's new geometry seeds.

Physical objective: maximize dimensionless total steady Robin absorption Q.
Source u=1, source length 0.1, Bi=1 for searches. Area=0.24, unit bounding box,
perimeter <=8, protected collar x<0.12, no holes, source connectivity, operational
feature width 0.04 at n=100/200/400. The cap is shared across all families and
differs from historical cap-4 experiments. Actual perimeter and eta are always
reported; eta never drives acceptance or training labels.

Families: rectangle, hexagon, square-Koch level 1, regularized square-Koch level
2, serpentine maze, two-generation dichotomous tree, hexagonal terminal sacs,
asymmetric terminal sacs, finite Hilbert order 2. Generators exactly match cell
count with declared stable ties; no outcome-selected geometric repairs occur.

Seven training families exclude maze and Hilbert. A sequential response regressor
is trained on 65 PDE solves per family, geometry seed 310001. Its 28 features
include the previous 24 incumbent/candidate features and four source-path
distance features. Labels are the actually evaluated signed global response
100*(Q_candidate-Q_current)/Q_initial. Unselected candidates are not treated as
known outcomes. This is ridge response regression, not an RL value function.

Evaluation uses five paired search seeds and geometry seeds 310101..310105.
No evaluation geometry is selected for how a method performs. Duplicate mask
hashes must be reported rather than called independent anatomy. The seven
familiar training families and two unseen training families are identified.
Distal aspect ratios, positions, rotations or path spans vary as declared by
each generator. Duplicate, canonical, training-overlapping or invalid masks are
rejected before any outcomes are observed; retries add 1000 to the candidate
seed, with a finite cap and a saved rejection manifest. Forty-five evaluation
masks must be distinct and disjoint from the seven training masks.
Online adaptation during a search is part of the learned method, not a frozen
policy generalization test.

Three methods use the same primitive, admissibility checks and PDE budgets:
fixed flux-guided greedy search, fixed flux-guided annealing, and warm-started
response-guided annealing. Exploration samples eligible proposals only; the old
unconditional invalid-action floor is removed in this separate new environment.
Candidate pools refill to two valid choices or a charged finite draw limit.
Raw proposals mix paired disks (75%) and area/perimeter-preserving single-cell
exchanges (25%), identically for all methods. The latter are a declared raster
primitive, not disk operations: matching exposed-neighbor counts makes their
net perimeter change exactly zero. Feature and connectivity checks still apply.
Hard perimeter-band failures are screened before expensive feature validation.

Annealing accepts positive delta Q and can accept a decrease with probability
exp(delta_Q/(Q_initial*T)). Temperature decreases geometrically from 0.003 to
0.00002. The asymmetric proposal means this is heuristic annealing, with no
claim of equilibrium sampling. Current and best states are separate; every 256
charged solves a restart returns to the saved best without solving again.

Main comparison: 9 families x 5 seeds x 3 methods x 129 solves = 17,415 PDE solves.
Predeclared long trajectories: sac family x 5 seeds x 3 methods x 1,025 solves
=15,375 solves, including 1,024 evaluated edits per complete trajectory.
Exact-perimeter controls: rectangle, hexagon and sac family x 5 seeds x 3 methods
x65 solves =2,925 solves. In these controls the starting perimeter is held exactly
within 1e-12; starting perimeters differ across families. Raw budget exhaustion
is retained and reported, not quietly dropped or given extra budget.

The 455 warm-start training solves are additional. A fixed-pool diagnostic
evaluates four candidates in five pools on each of nine representative starts;
all evaluations are charged. These labels do not retrain the warm-start model.
The candidate oracle is a diagnostic, never a free search baseline.

The baseline bank is swept over Bi=.01,.1,1,10,100 at n=100/200/400. Endpoint
and starting masks are refined at n=200/400 and checked with independent P1 FEM
on the identical staircase at two mesh densities. Four explicit physical
polygons separately compare body-fitted FEM with regenerated raster masks;
this quantifies geometry representation bias without concealing it with a Bi
correction. Independent FEM theory and tests are in fem_reference_derivation.md.

Primary descriptive comparisons use paired Q400 differences at equal coarse
PDE budgets; independent FEM differences provide a second numerical comparison.
Report distributions by family and seed, actual membrane changes, raw proposals,
geometry checks, runtime, training cost and truncations. Interquartile plot bands
are variability, not confidence intervals or 45 independent anatomical replicates.
No causal equal-runtime speedup is claimed from two concurrent workers.

Every solved proposal is committed to an atomic, self-contained resume journal
with candidate/current/best masks and fields, selected features, RNG states and
regression parameters. A pending marker bounds uncommitted solver work after a
crash. Source snapshots and config hashes are frozen. Resuming a finished job
performs zero PDE solves; source or identity changes fail loudly.

Choose the best returned endpoint by independent fine-mesh FEM Q and report its
Q400, both refinement changes, actual perimeter and gains over its own start.
This is post-hoc selection among returned endpoints, not a global optimum or an
independent test of a winning policy. Attribute flux spatially before arguing
why the shape performs better. The final interpretation must allow nature-inspired
controls to lose; physiological conclusions require additional validation.
