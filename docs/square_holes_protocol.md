# Square-island extension: controlled maze perforations

This is a new exploratory protocol, separate from the hole-free v0.1.0 study.
It asks how square exclusions change the global absorbed flux, where an island
is useful, and whether several islands shield one another. The sign of an effect
is an outcome, not a prescribed target. Empty squares are excluded material in
the 2D air mask; they are not additional oxygen inlets or pores between alveoli.

Run from the repository root after installing the validation dependencies:

```text
python -m oxygen_shape.cli study-holes --config configs/square_holes.yaml
```

The same command with `--resume PATH_TO_RUN` replays committed states from
checked cached fields. Config/source changes or altered committed caches fail.
The numerical source snapshot and config remain fixed during the study. Every
new FVM/FEM solve is recorded in a hashed ledger; a pending marker bounds lost
work after an interruption. Reports and figures are generated from saved fields.

## Comparisons fixed before this run

- Five outer families: a chamber in which islands create maze routes, the prior
  rectangle, a hexagonal chamber, a sac proxy, and the prior serpentine maze. The last may not have
  room for legal islands. Keep this no-fit outcome and the rejected roster.
  The new chamber starts with area 0.522; the prior families start with area
  0.24. Comparisons are paired within each outer shape. Cross-family raw flux
  rankings would not be resource matched.
- Single squares of widths 0.04 and 0.08 on a fixed 0.08 position lattice.
  For each family/width, all positions have the same air area, square perimeter,
  exterior boundary and source. Every eligible position is evaluated.
- A fixed unperforated outer chamber and declared sequences of 1/2/4/6 squares.
  These change air area and membrane; actual resources must be reported.
- Four six-square layouts: spread, proximal, distal and staggered. These match
  air area, inner and outer membrane, source, bounding box and square width.
- Close and separated pairs, plus their exact individual-square counterparts,
  for finite interaction measurements. Do not infer interactions by summing
  isolated fluxes on different outer domains.
- Five search seeds and two paid greedy relocation heuristics, starting from
  the same six-square maze: uniform and concentration-informed proposals. Each
  gets 129 FVM solves including its own initial solve. All rejected evaluated
  candidates are charged. Geometric duplicate/invalid proposals use no solve
  and are recorded separately. Uniform exploration remains at least 0.25 in
  the informed proposal law. Neither method is a learned RL policy.

Every perforated mask receives paired reflecting-hole and reactive-hole solves.
Store Q0, QN, QR, inner-wall and exterior-wall absorption, Q_ideal, eta, area,
Robin length, source length, residuals and conservation. The
[derivation](square_holes_derivation.md) proves QN<=Q0 and QR>=QN for this setup
and gives the exact obstruction/activation decomposition. Global solved Q alone
determines acceptance. No square is removed because a local field looks dark.

## Constraints and validation

Source u=1 on the unchanged inlet of length 0.1, unit bounding box, uniform Bi,
connected air and protected x<0.12. Islands are exact grid-aligned squares.
Minimum square width 0.04; conservative L-infinity air clearance at least 0.04
between islands and the exterior. The existing opening-core/residual air-feature
rule is also checked. These are finite grid rules, not anatomical constraints.
Default optimizer no-hole restrictions elsewhere are unchanged.

Sweep Bi=0.01,0.1,1,10,100 at n=100. At Bi=0.01,1,100 select both coarse best
and worst single-square positions within each family/width; validate every
declared layout at those Bi and each returned search endpoint at search Bi=1.
Subdivide the identical polygons to n=200/400 and check independent P1 FEM at
maximum areas 8e-5/2e-5. Do not call selected-extrema checking an exhaustive fine
map. Square interfaces are exact; exterior raster staircases remain staircases.

Before trusting irregular perforations, test a polygonal annulus against its
analytical radial Robin solution, meshed air area, flux conservation, reflecting
and zero-Bi limits, and source/feature protection. A report must retain numerical
uncertainty when gains are comparable to refinement changes.

## Prior art and scope

[Freus & Freus (2015)](https://doi.org/10.17512/jamcm.2015.2.01) already use the
Laplace equation, Robin conditions on holes, and topological sensitivity to
design a heat exchanger. The heat/mass transfer analogy makes this particularly
close prior art. Their method is a topological derivative plus boundary elements;
our extension measures finite square edits with conservative FVM and independent
FEM. The existence of holes or Robin topology optimization is not new.

[Grebenkov & Traytak (2019)](https://doi.org/10.1016/j.jcp.2018.10.033) treat
Laplace problems with multiple spherical reactive sinks/obstacles and several
boundary types. [Grebenkov (2020)](https://doi.org/10.1063/5.0012719) develops
partially reactive multi-trap calculations. Their 3D spherical settings differ
from the present planar square controls.
[Grebenkov & Ward (2026)](https://doi.org/10.1017/S0956792525100284) analyze
competition among small targets on planar-domain boundaries, including Robin
conditions. Target interaction and partial absorption have substantial prior art.

Actual collateral air passages are studied in
[long-range lung diffusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC2862294/).
That work concerns helium diffusion and alternative air routes, not the present
steady oxygen wall-uptake objective. No physiological or first-study conclusion
is inferred from our empty-square controls. A possible contribution is an openly
reproducible, resource-matched finite-topology benchmark and its measured limits.
