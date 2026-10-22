# Square islands in a diffusion-limited maze chamber

Nabil ES-SAADI · Independent researcher · Exploratory, unreviewed extension, October 2026.

**Relocating six equal square islands improves total uptake by
0.3834% over the declared starting layout at Bi=1,
with air area and reactive wall length fixed.** This is a finite-search result,
not a global optimum, an RL advantage, or evidence that a pulmonary acinus is optimal.
The new chamber has different resources from the v0.1.0 morphology benchmark;
their raw Q values must not be ranked together.

## What was measured

The air satisfies steady Laplace diffusion, source u=1, and reactive Robin walls
du/dn+Bi*u=0. Empty squares are excluded regions, not oxygen sources or pores
connecting alveoli. Paired controls assign either reflecting (N) or reactive (R)
conditions to the identical island walls while preserving the exterior.
The objective is absorbed Q; eta=Q/(Bi*P_R), area and perimeter are reported independently.

The fixed protocol evaluated 840 paired single-square/Bi comparisons
at n=100: 168 legal positions across two square widths, three eligible outer
families and five permeabilities. Both the sac proxy and the narrow maze had no
legal positions on the declared lattice under the width/clearance rules.
These outcomes and every rejected roster remain in the data. Thirteen declared
multi-square layouts, ten 129-solve searches and 340 refinement
rows complete the matrix. The 85 selected case/Bi combinations were checked
at n=200/400 on the identical polygons and with independent triangular FEM at
maximum areas 8e-5/2e-5. This does not refine every map position.

## Main comparisons

All four six-square layouts and every returned search endpoint have air area
0.4836, reactive perimeter 5.02, six square widths of 0.08 and source length 0.10.
The unperforated chamber instead has area 0.522 and reactive perimeter 3.10.
Thus a comparison against no islands includes **61.94% more reactive wall and
7.36% less air area**. The relocation gain above is the resource-matched comparison.

Finest FEM results at Bi=1, unperforated Q=0.50298613:

| Layout | Reactive Q_R | Reflecting Q_N | Q_R gain versus unperforated |
|---|---:|---:|---:|
| spread_6 | 0.51574557 | 0.49810800 | 2.5367% |
| proximal_6 | 0.51548408 | 0.48861447 | 2.4848% |
| distal_6 | 0.51095013 | 0.50150618 | 1.5833% |
| staggered_6 | 0.51500356 | 0.49928460 | 2.3892% |
| Best returned search endpoint | 0.51772317 | 0.49533899 | 2.9299% |

The best returned finest-FEM endpoint is `search/uniform_greedy_420002`;
the same mask was returned by 4 runs across both proposal methods.
its mask hash is `7deda1c9a3f845625aa8e70f5cb79e720804f8c74abd592ded2cf698b192842c`. All endpoint coordinates and hashes
are in [endpoint_geometries.json](endpoint_geometries.json), with complete
[refinement comparisons](tables/endpoint_refinement.csv). Returned runs produce
6 distinct endpoint masks. The largest FEM mesh
increment across these endpoints is 0.000856 in Q;
their finest-FEM range is 0.000202, and the largest finest
FVM/FEM discrepancy is 0.000777.
These are numerical sensitivity measurements, not rigorous error bounds.
Small endpoint differences should not be presented as a certified unique ranking.
The paired relocation gain is 0.3727% at n=200 and 0.3770% at n=400;
the two FEM meshes give 0.3845% and 0.3834%. The improvement over the starting
layout persists across all tested discretizations, while endpoint ordering
has a much smaller margin.

At n=100, the proximal layout absorbs about 0.15047 on its island walls, versus
0.12833 for the spread layout, yet its total Q is lower. Much of the island uptake
replaces exterior uptake. The controlled identity
Q_R-Q0=(Q_N-Q0)+(Q_R-Q_N) separates obstruction from membrane activation;
inner-wall uptake by itself is not the benefit. The energy argument proves
Q_N<=Q0 and Q_R>=Q_N under the stated assumptions, without prescribing the sign
of Q_R-Q0. See the [derivation](../../docs/square_holes_derivation.md).

For the selected endpoint versus the starting six-square spread layout, finest
FEM gives an additional obstruction effect of
-0.002769 in Q and an additional
activation effect of 0.004747.
Their net is 0.001978. The relocation improves total uptake
despite a larger reflecting-obstacle penalty; its activation increment more
than compensates. This comparison uses global paired solutions, not a local
dark-region deletion rule or extra membrane.

At low Bi, the declared six-square spread layout gives about 54.2% more Q at
n=100, alongside the additional wall budget. At high Bi, changes can be extremely
small because downstream walls see little concentration. Tiny positive map
differences near numerical uncertainty do not establish a resolved benefit.
The count sequence changes resources and cannot isolate a topology-only gain.

The two finite pair arrangements have exact individual-square controls and
[refined interaction measurements](tables/pair_refinement.csv). Their locations
also differ relative to the source, so they are not a separation-only causal sweep.
The uniform and concentration-informed relocation rules accept edits only after
a global Q solve. Neither rule is a learned RL policy or a topological derivative.
Five seeds from one starting maze do not establish algorithmic superiority.
New square positions come from the fixed 0.08 proposal lattice. Some declared
starting positions are off that lattice and can remain when those squares are
not moved; this is the same action set for both methods, not exhaustive layout search.

![Concentration fields with a shared linear scale](figures/square_fields_linear.png)

[Shared power-scale view](figures/square_fields_power.png) ·
[Obstruction and activation](figures/hole_effects.png) ·
[Position maps](figures/square_position_maps.png) ·
[Paid search histories](figures/hole_relocation.png).
All five figures also have vector PDFs. Concentration is dimensionless with the
same source value; the power-scale view changes the display only.
Field panels, position maps and search curves show n=100 data; the table above
reports the finest FEM values.

## Evidence, cost and recovery

The completed run committed **3493 FVM and
418 FEM solves**. All ten search budgets completed,
including paid rejected candidates. The searches used 1290
charged solves and recorded 2924 geometry/duplicate
rejections without a PDE solve. Per-call times, full journals and masks remain
in the raw release. This is CPU sparse-solver work; no GPU or new ML dependency was used.
The charged search total comprises ten initial states, 85 accepted edits and
1,195 evaluated edits that were not accepted.

An earlier attempt was interrupted and its ledger and pending marker became
zero-filled. It contains 3,708 cache files but cannot certify committed counts
or solver times. It was preserved, never resumed or adopted. A fresh full run
used unchanged numerical sources/configuration except for storing outputs outside
OneDrive. **Completed-run cost totals exclude that additional, unverifiable work**;
the interrupted attempt is supplied separately as failure evidence.

The independent saved-field audit checked every committed cache hash, declared
mask, executed source snapshot, concentration bounds, FVM face balances and FEM
boundary labels, element residuals, source/wall flux and energy. It performed
zero PDE solves. Maximum relative conservation error was
6.87e-11. All 344 tests passed locally and on
Windows and Ubuntu, including intentional field/label corruption checks, in
[the audit regression](https://github.com/nabil-stem/oxygen-shape-optimization/actions/runs/39737994635).
The frozen numerical code also passed all 341 then-current tests on both platforms in
[this CI run](https://github.com/nabil-stem/oxygen-shape-optimization/actions/runs/39735020737).

A Windows separate-folder replay disabled PDE solver entry points and regenerated
all five PNGs byte for byte. All ten public CSV tables reproduced within
1.34e-15 absolute difference; endpoint coordinates and derived ranking tables
were byte-identical. Small face-flux partition differences come from evaluating
equivalent Robin-conductance formulas in a different floating-point order.
The five vector PDFs were regenerated; their file bytes are not claimed identical.
See [validation_summary.json](validation_summary.json) for complete records.

The [dated raw release](https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/square-holes-20261022)
contains complete successful and interrupted evidence, frozen executed sources,
test records, configs and per-file manifests. v0.1.0 remains unchanged.
[Copied files](public_view_manifest.json) retain their raw bytes and hashes;
derived tables use the formulas in `scripts/curate_square_holes.py`.

## Reproduce and scope the contribution

From a compatible checkout:

```text
python -m oxygen_shape.cli study-holes --config configs/square_holes.yaml
python scripts/audit_square_holes.py PATH_TO_COMPLETED_RUN audit.json
python scripts/curate_square_holes.py PATH_TO_COMPLETED_RUN NEW_PUBLIC_FOLDER
```

Keep executed sources and configuration fixed. A completed, intact cached run
can regenerate reports with zero new solves using `--resume PATH_TO_RUN`.
The raw archive contains the exact executed snapshot for historical continuation;
do not rewrite a ledger to bypass identity guards. See the
[protocol](../../docs/square_holes_protocol.md) and
[reproduction guide](../../docs/reproducibility.md).

[Freus & Freus (2015)](https://doi.org/10.17512/jamcm.2015.2.01) already studied
Laplace/Robin holes for heat-exchanger topology design using a topological
derivative and boundary elements. [Grebenkov & Traytak (2019)](https://doi.org/10.1016/j.jcp.2018.10.033)
treat multiple partially reactive sinks/obstacles. Holes, partial reactivity and
screening are not new discoveries here. A possible research contribution is this
resource-controlled finite-square benchmark, its obstruction/activation accounting,
numerical checks and openly retained limitations.

The model omits transient breathing, convection, three-dimensional anatomy,
blood-side transport and smooth-wall validation. No human scientific review,
global optimality, clinical conclusion, physiological acinar efficiency claim or
archival DOI is asserted. Original data/figures/text: CC BY 4.0; project code: MIT.
