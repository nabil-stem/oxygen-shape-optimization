# Conducting-collar model sensitivity

The original frozen search classifies every exposed non-source wall as Robin.
The protected source collar is a geometry constraint, not an impermeable region.
At high Bi this permits strong absorption immediately beside the source and can
make proximal source corners dominate the objective.

A separate fixed-geometry ablation marks only the proximal wall at x<0.06 or
x<0.12 as Neumann. Distal absorbing walls remain Robin with the same Bi. This
represents an idealized nonalveolated conducting segment; it does not represent
an alveolar membrane as reflecting and is not an anatomically calibrated length.
The supplied Fujita/Koshiyama paper motivates examining the spatial onset of
alveolation, rather than proving either collar value physiologically correct.
[Original paper](https://doi.org/10.1080/15502287.2025.2463673).

For a rectangular nonabsorbing source neck of width H and length l, integrate
Laplace's equation across the neck. With zero lateral normal flux, the cross-
sectional average obeys mean(u)''=0. Consequently Q=H*(1-mean(u(l)))/l. The maximum
principle gives 0<=mean(u(l))<=1, hence Q<=H/l. For H=0.1 the upper bounds are
1.666667 at l=0.06 and 0.833333 at l=0.12. These bounds apply to this idealized
conducting neck, not the original absorbing collar. The 1D Robin slab verifies
the relation and the implementation checks it for every ablation solve.

Reevaluate the nine representative initial geometries and five distinct selected
endpoints at Bi=0.1,1,10,100, n=200/400, under all three boundary rules. Use two
independent FEM densities at Bi=1. Preserve the original search and its budgets;
record all extra deterministic evaluation costs in a separate postprocessing
directory. These are sensitivity results, not reoptimized designs under a new
physical model. Across boundary rules, reactive membrane length changes; within
each rule the removed proximal wall length is common to all admissible shapes.

If rankings or screening change, report that dependence before interpreting the
results biologically. Keep source concentration fixed at u=1. Altering a boundary
condition must have a stated physical purpose, not merely brighten a field plot.
