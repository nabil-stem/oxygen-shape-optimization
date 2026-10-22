# Deterministic reference: model and discretization

Scope: Milestones 0 and 1. All coordinates and parameters below are dimensionless.
The air domain is connected, diffusivity is one, and all source faces have `u=1`.
Rows increase upward; cell centers are `((col+1/2)h, (row+1/2)h)`.

## Scaling and physical objective

With `u=(C-C_b)/(C_0-C_b)` and `x_hat=x/L`, the bulk equation is
`Laplacian(u)=0` and the membrane condition is `du/dn + Bi*u=0`,
where `Bi=WL/D`. The total dimensionless membrane flux is
`Q=integral(Bi*u_wall ds_hat)` and `Q_ideal=Bi*P_R`.
Efficiency is `Q/Q_ideal`; at zero permeability it is undefined, stored as null.
In spatial dimension `d`, dimensional flux is `D*(C_0-C_b)*L^(d-2)*Q`.
The 2D result corresponds to transfer per unit out-of-plane depth.

## Interior and boundary resistances

Integrating `-Laplacian(u)=0` over a square cell gives the sum of outward face
fluxes equal to zero. An interior face has area (length) `h` and center distance
`h`, hence conductance one and flux `u_i-u_j`. Each pair cancels exactly in the
global balance. A source is half a cell away, giving conductance two and outward
flux `2*(u_i-1)`. A zero-Neumann face contributes zero.

On a Robin face write the outward flux density `j` in two ways:

```
j = (u_i-u_wall)/(h/2) = Bi*u_wall
u_wall = u_i/(1+Bi*h/2)
integrated face flux = h*j = G_R*u_i
G_R = h/(h/2+1/Bi),  Bi>0
G_R = 0,             Bi=0
```

This agrees with AGENTS.md. It includes both diffusion across a half-cell and
reaction at the wall. In particular `G_R -> 2` as `Bi -> infinity` at fixed `h`.
The implementation uses the resistance form to avoid overflow in `Bi*h`.
The face wall value is recovered as `u_wall=u_i-flux/2`, also valid for
Dirichlet and zero-Neumann faces. Efficiency averages wall values, not cell values.

For each cell, interior conductances and boundary conductances contribute to the
diagonal; interior off-diagonals are minus one; source conductances enter the RHS.
The symmetric matrix satisfies

```
v^T A v = sum_interior_pairs (v_i-v_j)^2
          + sum_source_faces 2*v_i^2 + sum_robin_faces G_R*v_i^2.
```

For a connected domain with a source this is positive definite, including `Bi=0`.
Disconnected geometries and missing sources are rejected before assembly.
`spsolve` is the reference solver. At `Bi=0`, the unique solution is exactly one;
we assign this exact constant after assembly to preserve exactly zero flux.

## Fluxes and diagnostics

The stored conductance array is reused for both assembly and boundary fluxes.
`Q_R=sum_R G_R*u_i`, `Q_D=sum_D 2*(1-u_i)`. Summing cell equations gives
`Q_R-Q_D=0`. The reported relative error is
`abs(Q_D-Q_R)/max(abs(Q_D),abs(Q_R),1e-14)`.
Acceptance uses `abs(Q_D-Q_R) <= atol + rtol*max(abs(Q_D),abs(Q_R))` to
avoid rejecting near-zero flux purely because of relative roundoff. We also check
the unscaled infinity norm of `A*u-b` and the discrete maximum principle
`0 <= u <= 1`, without clipping the solution.

## Slab and what refinement can establish

For `u''=0`, `u(0)=1`, `u'(ell)+Bi*u(ell)=0`, write `u=1-q*x`.
The Robin condition gives `q=Bi/(1+Bi*ell)` and
`u(ell)=1/(1+Bi*ell)`. The center-to-center and half-cell flux formulas above are
exact for a linear function. Consequently the slab is reproduced to roundoff on
every grid; there is no nonzero truncation error from which to estimate an order.
Validation reports center, membrane, and flux-density errors against `h` honestly.
The thin 2D strip uses three rows, Neumann top/bottom, left source, right Robin;
its transverse width changes during refinement, so compare `Q/width`, not raw Q.

A separate rectangle with Robin top, bottom, and right has a non-linear field.
Its test refines the same physical rectangle and compares flux and nested-grid
concentrations. This is a self-convergence check, not an analytical 2D proof.
The source/Robin corner junction can reduce observed convergence order.
Axis-aligned rectangular metrics are exact; curved staircases have perimeter
bias and require additional study before shape comparisons.

## Source protection and scope limits

Boundary conditions belong to exposed faces. Validation rejects labels on buried
or removed faces, duplicated/conflicting labels, missing sources, and diagonal
connectivity. Comparing a candidate to a reference requires identical grid,
spacing, and source face identities; their exposure is checked again.
No optimization proposals are implemented. Area, perimeter, feature-size, and
roughness constraints for optimization remain work for later milestones.

## Baseline geometry matching (Milestone 2)

All baseline cases occupy the unit box and share the same exposed west source
segment from y=0.45 to y=0.55. Resolutions are multiples of twenty, so the source
endpoints are represented exactly. A shared inlet corridor connects to a body
defined by a nested scalar score field: coordinate distance for rectangles and
squares, Euclidean distance for disks, modulated transverse distance for
sinusoidal walls, or distance to a polyline for branching and Hilbert-like tubes.
A distal chamber behind a long narrow corridor supplies the deep-cavity example.

For target area A, select exactly K=A/h^2 cells, including the mandatory inlet.
Add remaining cells in increasing score order. Only the final equal-score layer
is selected in deterministic row-major order. This introduces a possible one-cell
symmetry defect but avoids hiding area mismatch. Connectivity and source exposure
are checked after selection. Unsupported noninteger K is rejected.

The fixed-perimeter group uses rectangles and squares only. Enumerate whole
score layers and choose the layer with exposed Robin length closest to the target.
The declared matching tolerance is 4h, accounting for discrete jumps in these
simple axis-aligned boundaries. Actual length, mismatch, tolerance, and varying
area are stored, so this is not an exact smooth-domain perimeter comparison.

The raster bounding diameter is the greatest distance between vertices of the
convex hull of occupied cell corners. A morphology diagnostic measures cells
removed by a one-cell-radius circular opening. This diagnostic includes corner
effects and is not a certified neck width; minimum neck is explicitly unreported.
Screened wall fraction is the boundary-measure fraction whose reconstructed
wall concentration is below 0.1. Effective membrane length is Q/Bi; neither
quantity defines an optimization acceptance rule.

Exposed-face perimeter has an orientation-dependent staircase bias. Refining a
digitized disk does not by itself recover its Euclidean perimeter or guarantee
the correct smooth-boundary Robin flux. The baseline retains the specified face
conductance without a subcell correction and labels this limitation in its report.

## Finite paired-edit sensitivity estimate

The fixed-area optimizer reassigns a positive area a from a donor disk to an
addition disk. The net area change is zero, so dividing delta-Q by net area is
undefined. Its empirical response is instead s=delta-Q/a, with delta-Q measured
by a new global PDE solve on the complete paired candidate. This response
includes both wall changes and is not an isolated boundary shape derivative.

For two eligible points p_i,d_j and an observed pair p_k,d_k, the Gaussian joint
kernel factors as K_k(i,j)=exp(-|p_i-p_k|^2/(2*l^2)) times
exp(-|d_j-d_k|^2/(2*l^2)). Sum K_k*s_k with signs intact, then clip the sum below
at zero to obtain proposal weights w. Normalize them and mix with the uniform
joint boundary-face distribution: P=epsilon/N^2+(1-epsilon)*w/sum(w).
If sum(w)=0, use uniform. Equal Cartesian face lengths make 1/N^2 the uniform
product of the two boundary measures. Every pair thus retains probability at
least epsilon/N^2. A two-point analytical-response test checks signs, joint pair
dependence, normalization, scaling invariance and this lower bound.

The flux-guided comparison uses w(i,j)=j_i*(max(j)-j_j) and the same mixture.
Equal flux produces zero contrast and reduces to uniform. Neither weighting
formula changes acceptance: only the globally re-solved Q determines the best
candidate. The batch holds geometry fixed while probing and discards responses
before another round. Full historical protocol and limitations: [sensitivity method](https://github.com/nabil-stem/oxygen-shape-optimization/blob/v0.1.0/docs/sensitivity_method.md).

## Held-out ranking and radius diagnostic

For a candidate with features z=(addition_x,y, donor_x,y, addition_radius), use
the spatial joint kernel above and optionally multiply by a Gaussian kernel in
radius. Compare signed sums of calibration delta-Q with sums of delta-Q divided
by reallocated area. These are heuristic scores, not calibrated flux estimates.
Clip the summed scores below at zero before rank evaluation: this preserves
the optimizer's probability ordering, including ties at its exploration floor.

For a top fraction f of n held-out candidates, choose k=ceil(f*n). Give unit
weight to scores strictly above the kth largest score and split the remaining
weight equally among tied scores at that threshold; then divide weights by k.
Weights sum to one, never depend on observed responses, and reduce to uniform
when every score ties. Consequently an all-tied predictor gets the pool mean,
rather than being credited for an arbitrary ordering of the test set. Tests
verify these limits and equivalence to the current sensitivity probability map.
See the historical [prediction protocol](https://github.com/nabil-stem/oxygen-shape-optimization/blob/v0.1.0/docs/prediction_method.md) for the frozen split and independent radius replay.

## Screened-uniform proposal law

See [the distribution derivation](screening_distribution.md). For iid raw
proposals, cheap-pass probability a and m slots, screened-uniform selection
multiplies the probability of every fully valid edit by the same factor
epsilon+(1-epsilon)*(1-(1-a)^m)/a, provided all fully valid edits pass the cheap
screens. Conditioning on full validity therefore recovers ordinary rejection
sampling's valid-edit law. Minimum-perimeter ranking generally does not.
The current all-slots-generated implementation cannot improve valid yield per
raw draw; runtime can still improve through fewer expensive full checks.
An exact rational finite-example check is in scripts/check_screening_distribution.py.
Finite raw/time caps and non-iid or response-informed selection require separate
care; this argument is not a flux-optimality or finite-budget speedup proof.
