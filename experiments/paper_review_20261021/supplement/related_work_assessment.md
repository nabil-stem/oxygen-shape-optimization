# Related work and a narrower contribution

Reviewed 21 October 2026, including primary full texts and author preprints.
This is a targeted literature assessment, not a systematic review or proof
that no closer study exists. The governing optimization idea is established.

| Primary work | Relevant overlap | Difference and consequence for our claims |
|---|---|---|
| [Marck, Nadin and Privat (2014)](https://doi.org/10.1137/130941377), [author preprint](https://arxiv.org/abs/1310.2214) | Fin shape maximization of inlet heat flux with reactive lateral exchange; volume/surface constraints; oscillatory maximizing sequences and nonexistence in their reduced model | The conclusion explicitly discusses a multidimensional Laplace model with fixed inlet and Robin walls. We cannot present this PDE/objective combination as new. Their 1D theorems do not directly prove our finite 2D raster optimum or nonexistence. |
| [Alexandersen and Sigmund (2021 preprint version)](https://arxiv.org/abs/2012.04310) | Optimal cooling-fin geometry, tradeoff between conduction and surface exchange, and equivalence of fixed-power and fixed-root-temperature formulations in their model | Their reduced one-dimensional model and surface assumptions differ. Its optimum Bi result is not a theorem that our fixed-area arbitrary chamber must have Bi=1. |
| [Yan, Wang and Sigmund (2018)](https://doi.org/10.1016/j.ijheatmasstransfer.2018.01.114) | Demonstrates that visually plausible tree structures can be inferior to lamellar structures for specified conduction objectives | It concerns distributed volumetric heat generation, material distributions and thermal compliance/max-temperature criteria. It cautions against nature-based intuition but does not establish the best shape for our interior Robin absorption problem. |
| [Bucur, Nahon, Nitsch and Trombetti (2022)](https://doi.org/10.1007/s00526-022-02298-1) | Variational Laplace/Robin energy with a held-temperature inner body and geometry/resource constraints | They minimize heat loss for insulation and allow a different source geometry/admissible set. Similar energy identities and shape-design framing are prior work, while their minimizers are not maximizers of our flux problem. |
| [Blommaert et al. (2025 preprint)](https://arxiv.org/abs/2504.03436) | Geometry-constrained adjoint optimization of heat transfer, minimum spacing and entrance effects | Their periodic fin-array convection model and flow constraints differ. It supports using credible conventional optimization and checking inlet assumptions; it is not an identical Robin-diffusion baseline. |
| [Bucur and Giacomini (2016)](https://doi.org/10.1016/j.anihpc.2015.07.001), [primary full text](https://ems.press/content/serial-article-files/16954) | Free-boundary Robin energies, relaxed admissible sets and existence | Their problem minimizes an energy with an internal prescribed region and optional volume cost. It does not establish a maximum for our fixed boundary-source/area/perimeter raster class. It reinforces that regularity, relaxation and Robin energy shape optimization are established topics. |

The pulmonary, prefractal, partial-absorption and source-corner primary sources
are recorded separately in morphology_literature.md, source_corner_limit.md and
complementary_flux_bounds.md. Read these formulations together rather than
equating all transport problems with one another.

## Contribution that the present evidence could support

An open, reproducible finite-geometry benchmark can combine controlled resource
constraints, globally evaluated signed shape changes, equal-budget search
comparisons, independent FEM checks, practical energy bounds for ranking and
an explicit inlet-wall sensitivity. The scientific result may be a limitation
or negative finding: learned selection or nature-inspired starting complexity
need not improve total transfer under the tested assumptions.

This combination is a candidate empirical contribution, not yet an established
novel algorithm or publication-readiness guarantee. Many solver calls and a
working learning implementation are not novelty. The manuscript must name the
specific question answered beyond existing studies and report how assumptions
change its answer. The added uniform and fully evaluated-pool greedy controls
close a comparison gap; neither is claimed as a new optimization method.

No broad inference that natural acini are suboptimal follows. The model lacks
three-dimensional packing, dynamic ventilation, blood-side transport and many
biological constraints. Conversely, a good sac-proxy endpoint does not prove
natural optimality. The present objective remains total Q, with eta diagnostic.
