# What the morphology benchmark can establish

The study compares controlled finite two-dimensional raster domains. It can
demonstrate increased total steady Robin absorption Q under its declared
resource constraints. A sac-like starting generator is a finite mathematical
proxy, not an anatomical reconstruction or evidence of evolutionary optimality.

## Resource matching and computational effort

All starts have area 0.24 and share the same source, bounding box and perimeter
cap 8. Their actual starting perimeters differ. A high-flux start is not an
equal-membrane winner unless that comparison explicitly matches perimeter.
The exact-perimeter controls test improvements relative to each individual
start; they do not make different families share one membrane length.

Main family/method comparisons use 129 PDE solves per search. The separately
predeclared long sac-family study uses 1,025. A best endpoint selected across
both phases has unequal search effort across starting families. It can be a
useful best observed design but cannot establish family or policy superiority.
Training and independent validation costs are additional and must be charged.

## Operational feature rule

The value 0.04 is an opening-core criterion, not a guaranteed lower bound on
every local geometric width. With radius r=0.02, the raster domain is opened by
a disk. Its retained core must remain connected and intersect the source. Every
air pixel must lie within r/2+sqrt(2)*h of that core. This is tested on the
n=100 mask and the same physical mask subdivided to n=200 and n=400.

This rejects disconnected narrow necks and long thin appendages under an
explicit finite-grid rule, while permitting short residual boundary bumps.
Single-cell area/perimeter-preserving exchanges can create such bumps. There
is no curvature penalty in the frozen study. Independent FEM validates their
staircase PDE, not the physical relevance of pixel-scale roughness. Calling
these endpoints smooth shapes with a strict 0.04 minimum feature size would
overstate the implemented admissible set.

## Numerical and boundary-model distinctions

Refining a saved raster mask preserves its staircase perimeter. It checks PDE
discretization on that polygon. Regenerating a physical smooth or oblique shape
at increasing resolution changes the represented boundary and is a separate
test. The physical-polygon controls expose Cartesian membrane bias directly;
it is not corrected by changing Bi after seeing outcomes.

Complementary/primal energy estimates bound flux on the same raster polygon
in exact arithmetic, with practical floating guards in the implementation.
They do not certify an anatomical surface, nor do overlapping estimates identify
a unique best geometry. The selected endpoints are coarse-best candidates;
fine-grid selection does not examine every internal proposal.

At very high Bi, the source-touching Robin corners can dominate total absorption.
The analytical [strip derivation](source_corner_limit.md) explains why a finite
large-Bi plateau cannot be assumed for this boundary partition. The separate
[conducting-collar ablation](collar_ablation.md) asks whether the conclusion
depends on where absorption begins. Its geometry is fixed and not reoptimized.

## Learning and statistical scope

The new model regresses measured signed global flux changes. It is not a deep
RL value function. Online adaptation in evaluation is part of that search
method. Its seven training masks do not overlap the 45 evaluation masks, and
two evaluation families are absent from training. Distinct masks and different
search seeds are not independent samples of natural anatomy.

Each family has five paired geometry/search-seed combinations, rather than a
full crossing of five search seeds on every geometry. The methods share the
same start and seed within each pair, which supports paired comparisons, but
the design does not separately estimate geometry variability and search-seed
variability. Plot bands contain both. Only one warm-training sequence is used;
generalization over independent training replicas is not measured.

Results are exploratory development evidence. A favorable post-hoc winner does
not replace paired equal-budget comparisons. Unfavorable learned-search or
nature-inspired outcomes must be retained. A publication claim should distinguish
solver correctness, useful constrained designs, algorithm comparisons and
physiological interpretation; each needs its own evidence.
