# Pulmonary acinus morphology references

These notes distinguish findings reported by the authors from project decisions.
Relevant equations, tables and figures were checked against the published pages. Page numbers
below are PDF page numbers, including the publisher cover in the 2025 file.

## Source identities

1. Ishikawa A, Koshiyama K (2022). Mathematical modeling of pulmonary acinus
   structure: Verification of acinar shape effects on pathway structure using
   rat lungs. Respiratory Physiology & Neurobiology 302, 103900.
   DOI: https://doi.org/10.1016/j.resp.2022.103900.
   SHA-256: `318b84e09aafd77d3e85a360e8596896870fdba456dc209d092957b9f634d62d`.
2. Fujita K, Koshiyama K (2025). Polyhedron-based pulmonary acinus structure
   modeling considering alveolar distribution along airways for including
   respiratory bronchioles. International Journal for Computational Methods in
   Engineering Science and Mechanics 26(3), 231-240.
   DOI: https://doi.org/10.1080/15502287.2025.2463673.
   SHA-256: `059aa07e258729d75e2f0711da0ea45d3a237a37cdf1e6c549c793b3ba618021`.

## Reported methods and findings

### 2022: outer shape changes the airway path distribution

Methods pp. 2-3 and Fig. 1: seeds define Voronoi compartments; Delaunay
adjacency supplies possible airway connections. A connected, cycle-free tree
selects common compartment faces to remove. Hexagonal seeds give a 2D example;
body-centered cubic seeds give truncated-octahedral compartments in 3D.
The authors optimize a weighted combination of terminal-alveolus count and
reciprocal mean inlet-to-alveolus path length with simulated annealing
(Eqs. 1-3). These are assumed gas-exchange proxies, not computed oxygen flux.

Results pp. 4-6, Figs. 5-9 and Tables 2-3: cone, inverted cone, double cone,
chestnut-like cone and cuboid differ in path distributions and tortuosity.
Aspect ratio also matters. Ten samples per shape are analyzed. The reported
alveolar-surface-to-volume ratios are similar across these homogeneous models,
despite differences in airway accessibility. Table 1's realized model volumes
are not identical to the experimental target, so this is not an exact
matched-volume optimization benchmark.

Limitations pp. 6-8: the model tends to overestimate alveolar number and
alveolar surface per volume and underestimates experimental mean tortuosity.
The authors explicitly leave validation of actual gas-exchange optimality to
future work. Tissue mechanics, dynamics and geometric heterogeneity remain
important limitations.

### 2025: alveolar location can be controlled separately from total count

Methods pp. 3-4, Eqs. 4-7: a logistic weight

\[
s_i = \frac{S}{1+\exp[-k(L_i-L_0)]},\qquad L_i=l_i/L_x
\]

modifies terminal counts and the weighted mean path length in the structural
objective. With S=2, k=0 recovers unit weights. Here l_i is distance along the
airway tree, not straight-line Euclidean distance. The weight is not a membrane
permeability or a fractional absorbing-wall coverage; it can exceed one.

Results pp. 5-8, Figs. 2, 4-8: length weighting reduces proximal terminal
alveoli and lengthens proximal airways while leaving the total terminal count
roughly similar. The midpoint L_0 particularly affects path distributions and
tortuosity. The 2D demonstration uses three samples; main 3D statistics use ten
samples. The reported respiratory-bronchiole-like morphology depends on the
selected parameters and the model's compartment interpretation.

Human model pp. 4, 7-9: each human polyhedron represents an alveolar duct or sac,
not an individual alveolus. Table 1 uses about 560 compartments for that coarse
model. The authors state that explicit alveolar walls and detailed proximal
airway-diameter variation are absent (p. 9). Good agreement with selected
morphometric statistics does not validate a fully resolved gas-exchange surface.

### Values requiring care before reuse

The 2025 methods paragraph on p. 4 gives a seed estimate near 2061 from
522e6 / 194e3 cubic micrometers. Direct division gives approximately 2691.
Table 1 separately reports 2332 realized rat-model seeds. These are different
quantities and should not be silently substituted for each other. Extracted
micro symbols also corrupt units: text extraction renders Table 2's path-length
heading as mm, but visual inspection of PDF p. 6 confirms the printed unit is
micrometers. Values around 750-1217 are thus consistent with that printed unit;
this is an extraction error, not a published unit error. No dimensional
calibration is imported into the code from these values.

## Project interpretation and implementation decisions

| Source information | Consequence for this codebase |
|---|---|
| Structural scores are count/path proxies | Keep the validated Laplace-Robin PDE and global Q acceptance rule |
| Shape affects access even with similar surface/volume | Continue reporting Q, eta, area, perimeter and constraints separately |
| Proximal and distal alveolar distributions differ | Future wall-coverage studies need explicit face types and a declared coverage budget |
| Human polyhedra represent sacs/ducts | Do not count coarse external polyhedron faces as resolved human alveolar area |
| Path length is measured along the airway tree | A future raster analogy needs a declared geodesic/skeleton definition; Euclidean x is not equivalent |
| Homogeneous geometry has known limitations | Keep the current raster model labeled a controlled 2D study |

These are our decisions, not numerical conclusions of either paper. In
particular, neither paper establishes that a minimum-perimeter proposal pool
will improve Q. That is the development hypothesis tested next, motivated by
our observed rejection data. Geometry-guided proposal selection must still
pass all existing source, topology and feature checks and be globally solved.

Do not make Robin walls reflecting to mimic reduced alveolarization. A later
heterogeneous model could explicitly distinguish impermeable conducting faces
from absorbing alveolar faces, but it would require a new declared physical
experiment and validation. The papers' logistic structural weights cannot be
reinterpreted as local Bi without a separate model and derivation.

The Voronoi/tree approach is a useful future geometry family, not a reason to
skip 2D validation or begin 3D now. Internal septa require representation and
topology decisions beyond the present no-hole raster benchmark. No physics,
source boundary, resource budget or existing RL checkpoint is changed here.
