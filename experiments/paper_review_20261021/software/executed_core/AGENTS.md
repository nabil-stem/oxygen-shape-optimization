# AGENTS.md — Oxygen Diffusion Shape Optimization

> **Purpose:** This file is the single source of truth for an agent working on this research codebase.
>
> **Project status:** exploratory 2D research prototype first; 3D only after validation.
>
> **Primary scientific question:**  
> **Given oxygen diffusivity, membrane permeability, a fixed oxygen source, and geometric constraints, what geometry maximizes the total steady diffusive oxygen flux through the permeable wall?**
>
> **Important:** Do not jump directly to a sophisticated Monte Carlo solver. Build and validate a deterministic reference solver first.

---

## 0. Agent operating rules

You are working on a scientific computing / PDE-constrained shape-optimization project.

Follow these rules strictly:

1. **Correctness before speed.**
2. **Separate the PDE solver from the geometry optimizer.**
3. **Every important numerical result must be testable and reproducible.**
4. **Do not claim that a geometry is globally optimal.** We are searching for better geometries under explicit constraints.
5. **Do not delete a region merely because the local concentration is small.** A geometric change must be evaluated by its effect on the global objective.
6. **Do not confuse Walk on Spheres circles with the geometry itself.**
   - WoS/WoSt circles/stars are integration / sampling domains used to solve a PDE.
   - The physical geometry is the optimization variable.
7. **Do not model a permeable alveolar wall as a purely reflecting wall.**
   - Neumann wall -> reflecting.
   - Robin wall -> partially absorbing / reactive boundary.
8. **Do not move to 3D until the 2D implementation passes all validation tests.**
9. **Use SI-independent dimensionless variables internally whenever possible.**
10. **Record random seeds, configuration values, solver tolerances, grid resolution, and Git commit hash with every experiment.**
11. When unsure about a numerical formula, derive it in a markdown note and verify it against an analytical test before using it.
12. Prefer small modules, typed interfaces, and tests over a large notebook-only implementation.

---

# 1. Research problem

We model steady oxygen diffusion inside an air-filled domain \(\Omega\).

The domain boundary is partitioned into:

- \(\Gamma_D\): source / inlet boundary with prescribed oxygen concentration.
- \(\Gamma_R\): permeable wall through which oxygen is absorbed.
- \(\Gamma_N\): optional impermeable wall with zero normal flux.

The geometry \(\Omega\) is allowed to change, while the source boundary and selected geometric constraints remain fixed.

The optimization goal is to maximize the total oxygen flux through \(\Gamma_R\).

---

# 2. Physical model

Let

- \(C(\mathbf{x})\): oxygen concentration,
- \(D\): oxygen diffusivity in air,
- \(W\): membrane permeability,
- \(C_0\): concentration imposed at the source,
- \(C_b\): effective blood-side concentration.

In the first model, assume:

- steady state,
- no convection inside the optimized domain,
- constant \(D\),
- constant \(W\),
- blood acts as a prescribed sink.

The bulk equation is

\[
\nabla^2 C = 0
\qquad \text{in } \Omega.
\]

Source boundary:

\[
C=C_0
\qquad \text{on } \Gamma_D.
\]

Permeable wall:

\[
-D\frac{\partial C}{\partial n}
=
W(C-C_b)
\qquad \text{on } \Gamma_R.
\]

Optional impermeable wall:

\[
\frac{\partial C}{\partial n}=0
\qquad \text{on } \Gamma_N.
\]

The total transferred oxygen flux is

\[
\Phi(\Omega)
=
\int_{\Gamma_R}
W(C-C_b)\,dS.
\]

By conservation at steady state, this should equal the net oxygen flux entering through the source, up to numerical error.

---

# 3. Non-dimensionalization

Use

\[
u=
\frac{C-C_b}{C_0-C_b}.
\]

Choose a characteristic geometric length \(L\), and define

\[
\hat{\mathbf{x}}=\frac{\mathbf{x}}{L}.
\]

Then

\[
\nabla_{\hat x}^2u=0.
\]

At the source,

\[
u=1.
\]

At a permeable wall,

\[
\frac{\partial u}{\partial \hat n}
+
Bi\,u
=
0,
\]

where

\[
Bi=\frac{WL}{D}.
\]

It is also useful to define

\[
\Lambda=\frac{D}{W},
\]

so that

\[
Bi=\frac{L}{\Lambda}.
\]

\(\Lambda\) is the characteristic "unscreened" length used in pulmonary diffusion literature.

Interpretation:

- \(Bi\ll1\): membrane-limited regime. The concentration tends to remain relatively uniform, and geometry should matter less if membrane measure is held fixed.
- \(Bi\gg1\): transport-limited / strongly absorbing regime. Screening becomes important; inaccessible wall regions can contribute little.
- \(Bi\sim1\): transport and membrane resistance are comparable.

Define the dimensionless total flux

\[
Q(\Omega)
=
\int_{\hat\Gamma_R}
Bi\,u\,d\hat S.
\]

The dimensional total flux can be reconstructed from the chosen scale. For optimization, the dimensionless \(Q\) is the primary quantity.

If the membrane concentration were everywhere equal to the source value, the ideal dimensionless flux would be

\[
Q_{\mathrm{ideal}}
=
Bi\,|\hat\Gamma_R|.
\]

Define a wall-utilization / screening efficiency

\[
\eta
=
\frac{Q}{Q_{\mathrm{ideal}}}.
\]

For the normalized model with \(C_b=0\), this is equivalent to the average value of \(u\) over the permeable wall.

**Important:** maximizing \(Q\) and maximizing \(\eta\) are not the same problem.

---

# 4. What exactly are we optimizing?

The scientifically meaningful problem is

\[
\max_{\Omega\in\mathcal A} Q(\Omega)
\]

subject to the PDE and a set of admissibility constraints \(\mathcal A\).

Without geometric constraints, the optimization can become meaningless: an algorithm can create increasingly wrinkled or fractal boundaries to add membrane measure.

Therefore **every optimization experiment must state its geometric constraints explicitly**.

---

# 5. Recommended geometry constraints

Start with a controlled 2D problem.

At minimum enforce:

1. **Fixed source**
   - location,
   - length,
   - concentration.

2. **Connected air domain**
   - every interior component must be connected to the source,
   - reject disconnected islands and inaccessible cavities unless deliberately studying them.

3. **Fixed bounding box**
   - geometry cannot grow indefinitely.

4. **Area constraint**
   - \(A(\Omega)\approx A_0\), with a strict numerical tolerance.

5. **Perimeter / membrane budget**
   - either fix \(P_R\),
   - or impose an upper bound,
   - or use a penalty.

6. **Minimum feature size**
   - prohibit arbitrarily thin branches / necks.

7. **Optional curvature / roughness regularization**
   - discourage pixel-scale sawtooth boundaries.

A useful first scientific comparison is:

### Experiment family A — fixed area, bounded membrane length

\[
A=A_0,\qquad
P_R\le P_{\max}.
\]

### Experiment family B — fixed membrane length, bounded footprint

\[
P_R=P_0,\qquad
\Omega\subset B.
\]

### Experiment family C — penalized biological cost

\[
F(\Omega)
=
Q(\Omega)
-\lambda_A R_A
-\lambda_P R_P
-\lambda_{\mathrm{rough}}R_{\mathrm{rough}}.
\]

Use family C for algorithm development, but use hard or clearly reported constraints for final scientific comparisons.

---

# 6. Main conceptual correction to the original "circle" idea

The original intuition contains two different ideas that must not be mixed.

## 6.1 Circles used by Walk on Spheres

In classical WoS, from a point \(\mathbf{x}_k\), one finds a maximal ball contained in the domain and samples a point on its boundary.

In 2D,

\[
\theta\sim U(0,2\pi),
\]

\[
\mathbf{x}_{k+1}
=
\mathbf{x}_k
+
R_k
(\cos\theta,\sin\theta).
\]

That circle is a **numerical sampling object**. It is not added to the physical geometry.

## 6.2 Circles/disks used as geometry-edit primitives

Separately, we can define an optimizer that modifies the geometry through Boolean disk operations:

\[
\Omega'=\Omega\cup B(\mathbf{c},r)
\]

or

\[
\Omega'=\Omega\setminus B(\mathbf{c},r).
\]

These disks **do modify the geometry**.

This is a valid and interesting search representation, but every proposal must be evaluated through the PDE objective and constraints.

---

# 7. Why "remove the region where \(u\approx0\)" is not a valid optimization rule

Do **not** use

```text
if u is small:
    delete region
```

as the optimization law.

Reasons:

1. The PDE is global.
2. Removing a region changes the concentration everywhere.
3. A low-concentration region can still alter access to another wall region.
4. The local concentration is not a shape derivative.
5. A low local flux does not automatically imply that deleting the region increases total flux.
6. Deletion may violate area, membrane-length, connectivity, or minimum-neck constraints.

Instead use:

\[
\Delta Q
=
Q(\Omega')-Q(\Omega).
\]

Accept/reject a geometric proposal based on its actual effect on the objective and constraints.

Later, replace repeated trial solves by a proper sensitivity estimate.

---

# 8. Proposed research algorithm

The project will have three levels.

---

## Level 1 — brute-force local shape proposals

For each iteration:

1. Solve the PDE on current geometry \(\Omega_k\).
2. Compute \(Q_k\), \(\eta_k\), area, perimeter, and diagnostics.
3. Sample one or more geometry modifications.
4. Validate geometric constraints.
5. Solve the PDE on each candidate.
6. Compute
   \[
   \Delta Q=Q_{\mathrm{candidate}}-Q_k.
   \]
7. Accept the best valid candidate if it improves the selected score.
8. Save the state.

This is expensive but conceptually clean.

---

## Level 2 — sensitivity-guided disk proposals

Estimate a local empirical shape sensitivity.

For a small local perturbation around boundary point \(p\),

\[
S(p)
\approx
\frac{
Q(\Omega^{+}_{p,\varepsilon})
-
Q(\Omega^{-}_{p,\varepsilon})
}{
\Delta A
}.
\]

This is not yet a formal shape derivative; it is a numerical marginal-benefit estimate.

Use \(S(p)\) to bias future proposals.

A robust sampling distribution is a mixture:

\[
p(p)
=
\varepsilon p_{\mathrm{uniform}}(p)
+
(1-\varepsilon)p_{\mathrm{guided}}(p).
\]

This preserves exploration and prevents premature concentration in one region.

---

## Level 3 — true shape derivatives / differentiable Monte Carlo

After the deterministic prototype is validated, investigate:

- adjoint shape derivatives,
- Differential Walk on Spheres,
- robust derivative estimation with Walk on Stars,
- differentiable treatment of moving Robin boundaries.

Do **not** assume existing Differential WoS code directly solves the exact case of a moving oxygen-absorbing Robin boundary. Verify the mathematical assumptions first.

---

# 9. Geometry proposal system

The optimizer should support multiple proposal families.

## 9.1 Boundary-attached disk addition

Choose a boundary point \(p\).

Estimate local outward normal \(n(p)\).

Sample an angular deviation

\[
\theta\sim U(-\theta_{\max},\theta_{\max})
\]

around the outward normal.

Define

\[
d(\theta)
=
R(\theta)n(p).
\]

Place the disk center approximately at

\[
c=p+\alpha r\,d(\theta),
\]

with \(\alpha\) chosen so the new disk overlaps the existing domain sufficiently.

Then

\[
\Omega'=\Omega\cup B(c,r).
\]

Reject if:

- it produces an isolated component,
- it violates bounding-box constraints,
- it creates a neck below the minimum width,
- it changes the protected source.

This is better than sampling a completely arbitrary angle in \([0,2\pi]\) because the move remains attached to the current boundary.

---

## 9.2 Boundary-attached disk subtraction

Choose boundary point \(p\).

Place a disk primarily inside the domain and remove it:

\[
\Omega'=\Omega\setminus B(c,r).
\]

Reject if:

- the domain disconnects from the source,
- the source is modified,
- a forbidden hole appears,
- minimum feature size is violated.

Do not prefer a subtraction just because local \(u\) is small. Measure \(\Delta Q\).

---

## 9.3 Area-balanced paired move

For fixed-area optimization, a useful proposal is:

1. add one boundary-attached disk,
2. remove approximately the same area elsewhere,
3. repair small rasterization mismatch,
4. require
   \[
   \frac{|A'-A_0|}{A_0}<\tau_A.
   \]

This allows shape redistribution rather than uncontrolled growth.

---

## 9.4 Frontier growth / branching move

To preserve the original exploration idea, support a chain-growth mode.

Given a current frontier point \(p_k\), choose

\[
\theta_k
\sim
U(-\theta_{\max},\theta_{\max})
\]

relative to the local continuation direction, and create an overlapping disk at

\[
p_{k+1}
=
p_k+s(\cos\theta_k,\sin\theta_k).
\]

This can generate branched or tortuous structures.

However:

- the move is only a geometry proposal,
- it must obey constraints,
- it must be scored through \(Q\),
- growth-only experiments are exploratory and are **not** fair fixed-resource optimizations.

---

# 10. Random sampling strategy

Implement the following sequence.

### Stage A — uniform exploration

- uniform boundary point,
- uniform proposal type,
- radius sampled from a predefined range,
- angle sampled uniformly inside a feasible cone.

Purpose: establish a non-biased baseline.

### Stage B — flux-informed heuristic

Use local Robin flux density

\[
j(s)=Bi\,u(s)
\]

only as a heuristic feature.

Possible strategies:

- sample high-\(j\) regions for outward growth,
- sample low-\(j\) regions for candidate pruning,
- always validate using the true global \(\Delta Q\).

Do not interpret local \(j\) as a mathematically correct shape derivative.

### Stage C — sensitivity-guided exploration

Use measured local \(\Delta Q\) or a proper shape derivative.

---

# 11. Deterministic reference solver: Cartesian finite volume

Before WoS/WoSt, implement a robust reference solver on a Cartesian grid.

Use a cell-centered finite-volume discretization because it naturally handles flux conservation.

Let square cell width be \(h\), with normalized diffusivity \(D=1\).

For every interior cell \(i\), sum fluxes through its faces.

---

## 11.1 Interior-interior face

For neighboring cells \(i,j\),

\[
G_{ij}=1
\]

in 2D on a uniform square grid after the face-length / center-distance factors cancel.

Contribution:

\[
G_{ij}(u_i-u_j).
\]

---

## 11.2 Source Dirichlet face

The source face is \(h/2\) from the cell center.

For source value \(u_D=1\),

\[
G_D=2.
\]

Contribution:

\[
G_D(u_i-1).
\]

---

## 11.3 Neumann face

For zero Neumann condition,

\[
q_n=0.
\]

Contribution:

\[
0.
\]

---

## 11.4 Robin face

The cell center is separated from the membrane by a half-cell diffusion resistance and the membrane has its own reaction resistance.

For normalized Robin coefficient \(Bi\),

\[
q=Bi\,u_{\mathrm{wall}},
\]

and the effective integrated face conductance is

\[
G_R
=
\frac{h}
{h/2+1/Bi}
=
\frac{Bi\,h}
{1+Bi\,h/2}.
\]

Contribution to the cell balance:

\[
G_R u_i.
\]

For \(Bi=0\), use \(G_R=0\).

This treatment should converge to the continuum Robin condition as \(h\to0\).

---

## 11.5 Cell equation

For each interior cell:

\[
\sum_{j\in N(i)}
G_{ij}(u_i-u_j)
+
\sum_{f\in D(i)}
G_D(u_i-1)
+
\sum_{f\in R(i)}
G_Ru_i
=
0.
\]

Assemble a sparse linear system

\[
Au=b.
\]

Use `scipy.sparse`.

Default solver:

- `scipy.sparse.linalg.spsolve` for small/medium cases,
- later CG or another appropriate iterative SPD solver after verifying matrix properties.

---

# 12. Flux computation

Do not estimate total flux from an image gradient unless needed for visualization.

Use the same face fluxes used by the finite-volume discretization.

Robin total flux:

\[
Q_R
=
\sum_{f\in\Gamma_R}
G_R u_{i(f)}.
\]

Source influx:

\[
Q_D
=
\sum_{f\in\Gamma_D}
G_D(1-u_{i(f)}).
\]

At steady state,

\[
Q_D\approx Q_R.
\]

Define conservation error:

\[
e_{\mathrm{cons}}
=
\frac{|Q_D-Q_R|}
{\max(Q_D,Q_R,\epsilon)}.
\]

This is a mandatory diagnostic.

---

# 13. Analytical validation: 1D slab

Before any irregular geometry, validate against the exact 1D solution.

Problem:

\[
u''(x)=0,\qquad x\in[0,L],
\]

\[
u(0)=1,
\]

\[
u'(L)+\kappa u(L)=0.
\]

Exact solution:

\[
u(x)
=
1-\frac{\kappa}{1+\kappa L}x.
\]

At the membrane,

\[
u(L)
=
\frac{1}{1+\kappa L}.
\]

Flux magnitude:

\[
q
=
\frac{\kappa}{1+\kappa L}
\]

in normalized diffusivity units.

For \(L=1\), \(\kappa=Bi\):

\[
q
=
\frac{Bi}{1+Bi}.
\]

Required test:

- run multiple grid resolutions,
- verify convergence of numerical \(u\),
- verify convergence of flux,
- plot error vs \(h\).

No irregular-domain work should be trusted before this passes.

---

# 14. Additional mandatory numerical tests

## Test A — constant Dirichlet sanity

If every boundary is Dirichlet \(u=1\), solution must be \(u=1\).

## Test B — zero permeability

For

\[
Bi=0,
\]

Robin walls reduce to Neumann walls and

\[
Q=0.
\]

## Test C — increasing permeability

For a fixed simple geometry, total flux should increase as \(Bi\) increases and approach a transport-limited regime.

## Test D — mass conservation

Require small relative difference between source influx and wall absorption.

## Test E — grid refinement

Run at several resolutions and verify that:

- \(Q\) stabilizes,
- concentration fields converge,
- geometry metrics converge or rasterization error is quantified.

## Test F — symmetry

For a symmetric geometry and source, the numerical solution should respect symmetry within tolerance.

## Test G — connectivity rejection

Candidate shapes disconnected from the source must be rejected.

## Test H — protected source

Geometry edits must never change source location or source length.

---

# 15. Baseline geometries

Implement deterministic generators for:

1. rectangle,
2. circle / disk-like chamber,
3. square with matched area,
4. square with matched perimeter,
5. smooth sinusoidally perturbed wall,
6. branching geometry,
7. Hilbert-like / space-filling benchmark,
8. intentionally screened deep-cavity geometry.

Do not compare shapes unless the comparison states what quantity is matched:

- area,
- perimeter,
- source length,
- bounding diameter,
- or combinations.

Generate side-by-side reports with:

- geometry,
- concentration map,
- boundary flux density,
- \(Q\),
- \(Q_{\mathrm{ideal}}\),
- \(\eta\),
- area,
- Robin perimeter,
- conservation error.

---

# 16. Main research hypotheses

These are hypotheses, not conclusions.

### H1 — geometry matters most when transport limitation is significant

At low \(Bi\), if membrane measure is fixed, shapes should behave more similarly.

At larger \(Bi\), screening should amplify geometry-dependent differences.

### H2 — maximizing membrane length alone is insufficient

Additional wall length can become screened and contribute little to \(Q\).

### H3 — a good geometry balances access and membrane measure

The optimizer should tend toward shapes that expose absorbing boundary while maintaining diffusive accessibility from the source.

### H4 — branching / space-filling structures may help only up to a scale

Features much smaller than the characteristic screening scale may provide diminishing returns.

### H5 — uniform random search can find improvements but is inefficient

A physics/sensitivity-guided sampler should reach a given \(Q\) with fewer PDE solves.

### H6 — "delete low concentration" is not equivalent to optimizing flux

Direct \(\Delta Q\) experiments should produce counterexamples where locally low-\(u\) geometry still affects the global objective.

---

# 17. Optimizer v0: random local search

Implement a transparent baseline before any machine learning or advanced optimizer.

Pseudo-code:

```python
geometry = initial_geometry
state = evaluate(geometry)

for iteration in range(max_iterations):
    candidates = []

    for _ in range(n_candidates):
        proposal = propose_random_disk_edit(
            geometry=geometry,
            rng=rng,
            constraints=constraints,
        )

        if not is_valid(proposal):
            continue

        candidate_state = evaluate(proposal)
        candidates.append(candidate_state)

    best = select_best(candidates)

    if best is not None and best.score > state.score + min_improvement:
        geometry = best.geometry
        state = best

    save_iteration(...)
```

For stochastic experiments, also support optional simulated annealing later, but greedy improvement is the first baseline.

---

# 18. Objective and penalties

Keep the physical objective and optimization score separate.

Always store:

```text
physical_flux_Q
ideal_flux_Q
efficiency_eta
area
robin_perimeter
roughness
constraint_violations
optimization_score
```

Example development score:

\[
F
=
\frac{Q}{Q_{\mathrm{ref}}}
-\lambda_A
\left(
\frac{A-A_0}{A_0}
\right)^2
-\lambda_P
\max\left(
0,
\frac{P-P_{\max}}{P_{\max}}
\right)^2
-\lambda_R R_{\mathrm{rough}}.
\]

Do not publish conclusions based only on \(F\). Report the physical \(Q\) and all constraints.

---

# 19. Geometry representation: v0

Use a raster mask first.

Suggested encoding:

```python
inside: bool[ny, nx]
```

Boundary conditions are attached to **exposed cell faces**, not merely to pixels.

Each exposed face must be classified as one of:

```text
SOURCE_DIRICHLET
ROBIN
NEUMANN
```

Default:

- protected inlet segment -> source,
- all other exposed geometry boundary -> Robin,
- optional masks / rules can mark Neumann faces.

Advantages:

- easy Boolean disk operations,
- easy connectivity tests,
- compatible with finite-volume solver,
- easy visualization,
- deterministic.

Later representations:

- polygon,
- spline,
- signed-distance field,
- level set,
- triangle mesh in 3D.

Do not start there.

---

# 20. Minimum-feature and geometry checks

Implement:

```python
validate_geometry(geometry, constraints) -> ValidationReport
```

Checks:

1. source unchanged,
2. one connected interior component reachable from source,
3. no invalid isolated air islands,
4. optional no internal holes,
5. area tolerance,
6. perimeter limit,
7. bounding box,
8. minimum local thickness,
9. minimum number of cells in a neck,
10. nonempty Robin boundary.

For minimum thickness in raster v0, use a distance transform or morphological opening test.

---

# 21. Boundary sampling

Provide:

```python
sample_boundary_face_uniform(...)
```

and later:

```python
sample_boundary_face_weighted(weights, epsilon_uniform=...)
```

Uniform means uniform over eligible **boundary measure**, not uniform over image rows or connected components.

For disk edits, estimate local normal from the boundary geometry or signed distance field.

---

# 22. Why use the normal direction for disk proposals?

Sampling

\[
\theta\sim U(0,2\pi)
\]

is correct for the exit direction of a 2D WoS circle centered at the current walk point.

It is **not automatically the best distribution for geometry growth**.

For geometry edits, use a local coordinate system:

- normal \(n\),
- tangent \(t\).

Sample

\[
\delta\theta\sim U(-\theta_{\max},\theta_{\max})
\]

and form

\[
d=
\cos(\delta\theta)n
+
\sin(\delta\theta)t.
\]

This preserves the user's idea of angular exploration while making connected boundary growth more likely.

For specialized branching experiments, allow a wider angular distribution.

---

# 23. Visualization requirements

Every run should be able to produce:

1. geometry mask,
2. concentration heatmap,
3. concentration contours,
4. Robin boundary colored by local flux,
5. source boundary highlighted,
6. accepted disk addition/removal overlay,
7. \(Q\) vs optimization iteration,
8. \(\eta\) vs iteration,
9. area vs iteration,
10. perimeter vs iteration,
11. conservation error vs iteration.

Use Matplotlib.

Do not use fixed decorative colors as part of scientific logic; plotting choices should be easy to change.

---

# 24. Reproducibility

Every experiment directory should contain:

```text
config.yaml
metrics.csv
final_geometry.npy
initial_geometry.npy
summary.json
figures/
run.log
```

`summary.json` should include:

```json
{
  "seed": 42,
  "grid_shape": [200, 200],
  "Bi": 1.0,
  "solver": "finite_volume",
  "solver_tolerance": null,
  "initial_Q": null,
  "final_Q": null,
  "initial_eta": null,
  "final_eta": null,
  "area": null,
  "robin_perimeter": null,
  "conservation_error": null,
  "git_commit": null
}
```

Fill values programmatically.

---

# 25. Recommended repository structure

```text
oxygen-shape-optimization/
│
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── slab_1d.yaml
│   ├── rectangle_2d.yaml
│   ├── baseline_shapes.yaml
│   └── optimize_disks.yaml
│
├── src/
│   └── oxygen_shape/
│       ├── __init__.py
│       ├── cli.py
│       │
│       ├── physics/
│       │   ├── model.py
│       │   ├── nondimensional.py
│       │   └── metrics.py
│       │
│       ├── geometry/
│       │   ├── raster.py
│       │   ├── generators.py
│       │   ├── boundary.py
│       │   ├── disk_edits.py
│       │   ├── connectivity.py
│       │   └── constraints.py
│       │
│       ├── solvers/
│       │   ├── finite_volume.py
│       │   ├── analytical_1d.py
│       │   └── diagnostics.py
│       │
│       ├── optimization/
│       │   ├── objective.py
│       │   ├── random_search.py
│       │   ├── samplers.py
│       │   └── sensitivity.py
│       │
│       ├── experiments/
│       │   ├── runner.py
│       │   └── reporting.py
│       │
│       └── viz/
│           ├── fields.py
│           ├── geometry.py
│           └── history.py
│
├── tests/
│   ├── test_analytical_1d.py
│   ├── test_fvm_conservation.py
│   ├── test_fvm_symmetry.py
│   ├── test_robin_limits.py
│   ├── test_geometry_connectivity.py
│   ├── test_source_protection.py
│   └── test_disk_edits.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── experiments/
│   └── .gitkeep
│
└── docs/
    ├── derivations.md
    ├── numerical_validation.md
    └── research_log.md
```

---

# 26. Python stack

Target:

```text
Python >= 3.10
```

Core dependencies:

```text
numpy
scipy
matplotlib
scikit-image
pyyaml
pydantic
pandas
typer
```

Development:

```text
pytest
pytest-cov
ruff
```

Optional later:

```text
numba
cupy
shapely
```

Do not add GPU dependencies until profiling shows that the CPU implementation is a bottleneck.

---

# 27. CLI design

Target commands:

```bash
python -m oxygen_shape.cli validate-1d --config configs/slab_1d.yaml
```

```bash
python -m oxygen_shape.cli solve --config configs/rectangle_2d.yaml
```

```bash
python -m oxygen_shape.cli compare-shapes --config configs/baseline_shapes.yaml
```

```bash
python -m oxygen_shape.cli optimize --config configs/optimize_disks.yaml
```

Each command must:

1. print a concise result,
2. write a reproducible experiment folder,
3. fail loudly on invalid geometry or conservation failure.

---

# 28. Milestone plan

## Milestone 0 — repository and infrastructure

Deliver:

- package scaffold,
- `pyproject.toml`,
- config loading,
- typed data structures,
- pytest setup,
- CLI skeleton,
- experiment-output helper.

Acceptance:

```bash
pytest -q
```

passes.

---

## Milestone 1 — analytical + finite-volume solver

Deliver:

- exact 1D Robin solution,
- 2D raster geometry,
- boundary-face extraction,
- finite-volume system assembly,
- Robin / Dirichlet / Neumann faces,
- total source and Robin flux,
- conservation diagnostic.

Acceptance:

- 1D convergence demonstrated,
- conservation error documented,
- \(Bi=0\) test passes,
- symmetry test passes.

**Do not implement shape optimization before this milestone is validated.**

---

## Milestone 2 — baseline shapes and screening experiments

Deliver:

- baseline geometry generators,
- shape metrics,
- \(Q\) and \(\eta\),
- parameter sweep over \(Bi\),
- comparison plots.

Required experiment:

```text
Bi = 0.01, 0.1, 1, 10, 100
```

for multiple geometry families under matched constraints.

Goal:

observe how geometry sensitivity changes with the screening regime.

---

## Milestone 3 — random disk optimizer

Deliver:

- add-disk proposal,
- remove-disk proposal,
- paired area-balanced proposal,
- geometry validation,
- deterministic seed support,
- greedy candidate selection,
- history plots.

Acceptance:

- optimizer can improve at least one deliberately poor initial geometry,
- constraints remain satisfied,
- results reproduce with same seed.

---

## Milestone 4 — investigate the original hypothesis

Explicitly test:

> "Removing low-concentration or low-flux regions improves total oxygen flux."

Procedure:

1. identify lowest-\(u\) / lowest-\(j\) boundary regions,
2. make small controlled deletions,
3. re-solve,
4. measure \(\Delta Q\),
5. compare with random deletions,
6. find both supporting and counterexample cases.

Do not assume the hypothesis is true.

---

## Milestone 5 — sensitivity-guided search

Deliver:

- empirical local perturbation sensitivity,
- sensitivity map,
- weighted sampling,
- comparison:
  - uniform sampler,
  - flux-guided sampler,
  - sensitivity-guided sampler.

Compare by:

- best \(Q\) after N PDE solves,
- wall-clock time,
- variance over seeds.

---

## Milestone 6 — grid-free Monte Carlo solver research

Only after Milestones 1–5.

Study and prototype:

1. classical Walk on Spheres for simple Dirichlet validation,
2. Walk on Stars for Neumann/Dirichlet,
3. Walkin' Robin for Robin boundaries.

Validate Monte Carlo values against the finite-volume reference on the same geometry.

Metrics:

- bias,
- variance,
- runtime,
- scaling with geometry complexity,
- pointwise concentration error,
- total-flux estimation strategy.

Do not replace the deterministic reference solver.

---

## Milestone 7 — differentiable geometry research

Investigate:

- adjoint PDE shape derivative,
- Differential Walk on Spheres,
- derivative Walk on Stars,
- differentiable Robin boundary treatment.

Goal:

replace brute-force candidate re-solving by an efficient boundary sensitivity estimate.

This is a research milestone, not an MVP requirement.

---

## Milestone 8 — 3D extension

Only after 2D scientific conclusions are stable.

Possible geometry representation:

- voxel mask for first prototype,
- then mesh / implicit surface.

Repeat the same validation philosophy.

---

# 29. Experimental matrix

For each geometry family:

```text
grid resolution: 100, 200, 400
Bi: 0.01, 0.1, 1, 10, 100
seeds: at least 5 for stochastic optimization
```

Track:

```text
Q
Q_ideal
eta
area
Robin perimeter
bounding diameter
minimum neck
conservation error
number of PDE solves
wall-clock time
```

For optimizer studies, plot median and variability across seeds.

---

# 30. Scientific questions to answer

The codebase should eventually answer:

1. Under fixed area and membrane budget, what classes of shapes yield the largest total flux?
2. How does the preferred geometry change with \(Bi\)?
3. At what scale does added geometric complexity stop producing useful flux?
4. Can the optimizer rediscover branching / space-filling structures?
5. Does a Hilbert-like geometry outperform simpler structures under fair constraints?
6. How much membrane is effectively screened?
7. Does deleting low-flux wall improve total flux after resource reallocation?
8. How strongly does minimum feature size change the optimized shape?
9. Are random disk primitives expressive enough to find good shapes?
10. How much faster is sensitivity-guided search than uniform random search?
11. Can WoSt / Walkin' Robin evaluate irregular optimized geometries more efficiently than raster finite-volume methods?
12. Can a grid-free differentiable solver optimize a moving Robin wall reliably?

---

# 31. Important distinction: total flux vs efficiency

A geometry with more membrane can have a larger total flux but lower utilization efficiency.

Always report both.

\[
Q = \text{total useful oxygen transfer}
\]

\[
\eta=\frac{Q}{Q_{\mathrm{ideal}}}
= \text{fraction of ideal membrane utilization}.
\]

Example interpretation:

- high \(Q\), low \(\eta\): lots of membrane, strongly screened,
- lower \(Q\), high \(\eta\): smaller membrane, efficiently used,
- high \(Q\), high \(\eta\): ideal design target under the chosen resource constraints.

---

# 32. Optional multi-objective formulation

Instead of selecting a single arbitrary penalty, build a Pareto front.

Objectives:

\[
\max Q,
\]

\[
\min P_R,
\]

\[
\min A,
\]

\[
\min R_{\mathrm{rough}}.
\]

This can reveal whether there is a family of efficient geometries rather than one unique optimum.

Do this only after the single-objective baseline is stable.

---

# 33. Numerical pitfalls

Watch for:

### Staircase boundary bias

Raster geometry changes perimeter with grid resolution.

Mitigation:

- report resolution,
- compare multiple grids,
- eventually use subcell boundary reconstruction or polygon representation.

### Small features disappearing

A disk smaller than a few cells is not meaningful.

Require:

```text
r >= 3h
```

for early experiments.

### Perimeter cheating

Pixelated high-frequency roughness can artificially increase membrane measure.

Use minimum-feature constraints and roughness control.

### Source artifacts

Do not allow edits near the protected source unless explicitly testing source geometry.

### Solver singularity

If the geometry has no Dirichlet source and only zero-Neumann boundaries, the Laplace problem is not uniquely pinned.

Reject such cases.

### Monte Carlo noise

Never compare candidate geometries using noisy estimates without uncertainty control.

### Optimization overfitting to grid

Verify final candidates at a finer resolution.

---

# 34. What counts as evidence of an improvement?

A candidate geometry is scientifically "better" only if:

1. it satisfies the same declared constraints,
2. it has higher converged \(Q\),
3. the improvement exceeds numerical uncertainty,
4. conservation error is acceptable,
5. the ranking persists under grid refinement,
6. the result is not caused by a raster perimeter artifact.

For stochastic solvers, include confidence intervals.

---

# 35. Relation to pulmonary acinus physics

The project is motivated by oxygen transport in the pulmonary acinus.

In the simplified steady diffusion regime:

- oxygen enters through a prescribed source/front,
- diffuses through air,
- is absorbed across a permeable alveolar interface,
- irregular geometry creates uneven accessibility,
- some wall regions can be screened.

The optimization should therefore search for geometries that improve **accessible absorbing surface**, not simply total surface.

Biological interpretation must remain cautious:

- the true lung is 3D,
- breathing is time-dependent,
- convection can matter upstream,
- blood-side oxygen is not always a perfect fixed sink,
- tissue mechanics and vascular structure impose additional constraints.

The 2D model is a controlled physics problem, not a full physiological lung model.

---

# 36. Research references

Read these before implementing the advanced stages.

## Pulmonary diffusion and screening

### Felici, Filoche, Sapoval — Diffusional screening in the human pulmonary acinus

Journal of Applied Physiology, 2003.

DOI:

https://doi.org/10.1152/japplphysiol.00913.2002

Key relevance:

- steady Laplace model,
- fixed oxygen concentration source,
- Robin permeability boundary,
- diffusional screening,
- characteristic length \(\Lambda=D/W\).

### Sapoval, Filoche, Weibel — Smaller is better—but not too small

PNAS, 2002.

DOI:

https://doi.org/10.1073/pnas.122352499

Key relevance:

- physical scale of the pulmonary acinus,
- geometry/permeability/diffusivity coupling,
- space-filling geometry and screening.

---

## Grid-free PDE solvers

### Sawhney, Miller, Gkioulekas, Crane — Walk on Stars

ACM Transactions on Graphics / SIGGRAPH 2023.

DOI:

https://doi.org/10.1145/3592398

Key relevance:

- mixed Dirichlet / Neumann elliptic PDEs,
- reflected Brownian interpretation,
- grid-free geometry handling.

Simple tutorial implementation:

https://github.com/GeometryCollective/wost-simple

---

### Miller, Sawhney, Crane, Gkioulekas — Walkin' Robin

SIGGRAPH 2024 / ACM Transactions on Graphics.

Project:

https://imaging.cs.cmu.edu/walk_on_stars_robin/

Key relevance:

- Robin, Neumann, and Dirichlet boundary conditions,
- partially reflecting / absorbing random-walk interpretation,
- suitable conceptual foundation for permeable walls.

---

## Differentiable PDE simulation

### Miller, Sawhney, Crane, Gkioulekas — Differential Walk on Spheres

ACM Transactions on Graphics, 2024.

DOI:

https://doi.org/10.1145/3687913

Project:

https://imaging.cs.cmu.edu/differential_walk_on_spheres/

Experimental code:

https://github.com/baileymiller/differential_wos

Key relevance:

- derivatives of PDE solutions with respect to geometry / parameters,
- stochastic-gradient shape optimization,
- mesh/grid-free differentiation.

**Caution:** verify exact applicability to a moving Robin wall before adopting it.

---

# 37. Antigravity workflow

This repository is intended to be developed with an agentic coding environment.

When using Antigravity:

1. Keep this `AGENTS.md` at the repository root.
2. Ask the agent to work milestone-by-milestone.
3. Require it to run tests after each meaningful edit.
4. Ask for a compact implementation report after each milestone.
5. Review generated derivations in `docs/derivations.md`.
6. Do not allow the agent to silently change the mathematical model.
7. If a test contradicts the specification, investigate rather than weakening the test.

Antigravity agents can operate over the editor, terminal, files, and browser, so use the browser primarily for checking primary papers/documentation and the terminal for verification.

---

# 38. First Antigravity prompt

Use this prompt after placing this file in an empty repository:

```text
Read AGENTS.md completely and treat it as the scientific and engineering specification.

We are starting from zero.

Implement ONLY Milestone 0 and Milestone 1.

Requirements:
1. Create the Python package and repository structure.
2. Use Python >= 3.10.
3. Implement the analytical 1D Laplace + Robin solution.
4. Implement the 2D cell-centered finite-volume solver described in AGENTS.md.
5. Represent exposed boundary faces explicitly as Dirichlet source, Robin wall, or zero-Neumann wall.
6. Compute Robin flux and source influx using the exact same face discretization used in the system assembly.
7. Implement conservation-error diagnostics.
8. Add unit tests for:
   - exact 1D convergence,
   - Bi = 0,
   - symmetry,
   - mass conservation,
   - source protection.
9. Add one simple rectangle example and generate a concentration plot.
10. Write the discretization derivation and assumptions to docs/derivations.md.
11. Run pytest and show the final test results.
12. Do not implement WoS, WoSt, optimization, GPU code, or 3D yet.

If you discover that a formula in AGENTS.md is inconsistent, do not silently change it. Derive the correction, document it, and explain the change in your final report.

Finish by giving:
- files created,
- tests run,
- numerical validation results,
- known limitations,
- exact command I should run next.
```

---

# 39. Second Antigravity prompt — after Milestone 1 passes

```text
Read AGENTS.md and the current codebase.

Now implement ONLY Milestone 2.

Build deterministic baseline geometry generators and a reproducible Bi sweep.

Compare at least:
- rectangle,
- disk-like chamber,
- square,
- sinusoidally perturbed wall,
- branching geometry,
- Hilbert-like benchmark.

For every comparison, explicitly state which quantities are matched.
Never compare shapes with different resource constraints without labeling that difference.

Produce:
- concentration maps,
- Robin flux-density plots,
- Q,
- Q_ideal,
- eta,
- area,
- Robin perimeter,
- conservation error,
- grid-refinement check.

Use Bi = [0.01, 0.1, 1, 10, 100].

Do not implement optimization yet.

Add tests for all new geometry metrics and save a report under docs/baseline_results.md.
```

---

# 40. Third Antigravity prompt — disk optimization

```text
Read AGENTS.md and verify Milestones 1 and 2 still pass.

Implement Milestone 3: constrained random disk shape optimization.

Key rule:
DO NOT remove geometry simply because u or local flux is small.

A disk addition/removal is only a candidate geometry edit.
The candidate must be:
1. geometrically valid,
2. solved using the validated finite-volume PDE solver,
3. evaluated using global total flux Q,
4. accepted only according to the declared constrained optimization rule.

Implement:
- boundary-attached disk addition,
- boundary-attached disk removal,
- area-balanced paired move,
- uniform boundary sampling,
- feasible angular sampling relative to local boundary normal,
- fixed random seed,
- geometry validation,
- optimization history and plots.

Start with a deliberately poor geometry and demonstrate that the optimizer can improve Q without violating constraints.

Create docs/optimization_v0.md with:
- exact objective,
- exact constraints,
- proposal distributions,
- acceptance rule,
- numerical results,
- failure cases.

Do not implement Differential WoS yet.
```

---

# 41. Decision gate before advanced Monte Carlo work

Before starting WoS/WoSt, answer these questions from the deterministic experiments:

- Does geometry materially change \(Q\) at the tested \(Bi\)?
- Do we actually see screening?
- Is added perimeter becoming inactive?
- Are disk proposals capable of discovering meaningful improvements?
- Which constraints dominate the solution?
- Is the finite-volume solver too slow for the desired optimization scale?
- Which quantity needs a Monte Carlo estimator: pointwise \(u\), total \(Q\), or shape sensitivity?

If these questions are not answered, do not add a more complex solver.

---

# 42. Definition of success for the first research prototype

The first prototype is successful when we can show, reproducibly:

1. a validated Laplace + Robin solver,
2. mass conservation,
3. a screening transition as \(Bi\) changes,
4. fair comparisons among several constrained geometries,
5. a disk-based search that improves flux without cheating via area/perimeter,
6. evidence showing where the naive "delete low-\(u\)" rule works and where it fails,
7. a sensitivity-guided method that beats uniform random proposals in PDE-solve budget.

That is already a meaningful research platform.

WoSt, Walkin' Robin, differentiable Monte Carlo, GPU acceleration, and 3D are subsequent research layers.

---

# 43. Final mental model

Keep these four layers separate:

```text
PHYSICS
Laplace diffusion + source + Robin absorption
        ↓
SOLVER
Finite volume first; WoSt/Robin later
        ↓
OBJECTIVE
Total absorbed oxygen flux Q
        ↓
GEOMETRY OPTIMIZER
Disk edits → empirical sensitivity → shape derivatives
```

The original random-circle intuition belongs primarily to the **geometry proposal layer** if the circles modify \(\Omega\), and to the **solver layer** only when they are genuine WoS/WoSt sampling regions.

The scientifically correct loop is:

```text
propose shape
→ enforce constraints
→ solve PDE
→ measure global flux
→ compare
→ accept/reject
→ learn where to propose next
```

not:

```text
find low concentration
→ delete it automatically
```

This distinction is the foundation of the project.
