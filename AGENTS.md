# Contributor and agent instructions

This is an exploratory 2D PDE-constrained shape-optimization codebase.
The scientific question is to maximize total steady Robin-wall flux under
explicit source, area, membrane, topology and feature constraints.

## Scientific invariants

- Correctness precedes speed. Keep the PDE solver separate from the optimizer.
- Use dimensionless variables internally: source u=1, Bi=WL/D, and
  du/dn+Bi*u=0 on reactive walls. A Robin wall is not a reflecting Neumann wall.
- The objective is total absorbed Q. Report eta=Q/(Bi*P_R), actual area and
  perimeter independently; maximizing eta is a different problem.
- Evaluate geometry edits through their global effect on Q and admissibility.
  Low local concentration or wall flux alone is not a deletion rule.
- Fix and protect the source. Enforce the declared bounding box, connectivity,
  area tolerance, membrane budget, hole policy and operational feature rule.
- Distinguish fixed membrane length from a shared upper bound. Report actual
  resources, search/training costs, failed proposals and incomplete budgets.
- Do not claim global optimality, anatomical validity or learning superiority
  from a selected endpoint. Keep unfavorable results and interpretation limits.
- Raster refinement checks the same staircase domain; it does not validate a
  smooth anatomical wall. Practical floating-point energy estimates are not
  interval-arithmetic certificates.
- Walk-on-Spheres sampling balls are solver objects, distinct from disks used
  as geometry edits. Do not introduce Monte Carlo or 3D before the necessary
  deterministic and 2D validation gates are satisfied.

## Numerical work

The finite-volume reference uses interior conductance 1, source conductance 2,
and Robin conductance Bi*h/(1+Bi*h/2). Compute source and wall flux from those
same conductances and check conservation, residuals and concentration bounds.
Derive uncertain formulas in a technical note and check an analytical case
before adopting them. See docs/derivations.md and docs/numerical_validation.md.

Record seeds, configs, tolerances, resolution, code revision/source hashes and
solver costs. Keep executed sources/configurations fixed while a study runs.
Do not overwrite historical manifests or silently resume using changed code.
Use the archived executed sources for identity-guarded historical continuation.

## Repository workflow

Read README.md, docs/reproducibility.md and the relevant protocol before edits.
Prefer small typed modules and meaningful numerical tests. For code changes run:

```text
python -m ruff check src tests
python -m pytest -q
```

Use portable paths and configs that do not depend on a contributor's machine.
Keep runtime outputs, personal launchers, scratch notes and copied environments
out of Git. Curated public results belong in paper/; complete executed evidence
belongs in a versioned release. The v0.1.0 assets and tag are historical records:
do not rewrite them as part of a later documentation or repository cleanup.

Do not add GPU dependencies without profiling evidence. A larger model or more
training does not replace fair baselines, held-out evaluation or solver checks.
