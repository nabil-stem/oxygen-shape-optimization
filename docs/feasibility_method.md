# Geometry-only perimeter-pool proposal study

Protocol fixed before running the new sampler, 2026-10-19. The supplied
pulmonary morphology papers inform interpretation; they do not validate this
sampler or replace the physical objective (see pulmonary_papers.md).

## Revision

Draw eight independent proposals with the existing area-balanced disk primitive,
uniform eligible face pairs and radius distribution. With probability 0.25,
choose slot zero unchanged. Otherwise select the smallest Robin-perimeter
candidate whose area and protected source collar satisfy the cheap checks and
whose perimeter is in budget. Choose uniformly among numerical ties. If no
candidate passes these screens, choose slot zero. Do not repair any geometry.
Run the full unchanged 100/200/400-grid constraint check on the selected edit.
This is a proposal distribution, not a membrane-minimizing objective: only a
full PDE solve and a positive delta-Q can accept an edit in search.

The uniform branch preserves support of the original primitive. Each raw draw,
including invalid and unselected edits, is charged and logged. A selected-edit
yield gain is not a gain per raw draw; eight proposals are not one unit of work.
Minimum-perimeter selection can discard useful high-perimeter edits and can
reduce flux, which the search comparison must expose.

## Fixed-geometry feasibility audit

Use the same two pinned seed-8000 outputs and initial controls as the preceding
diagnostic, with identical constraints. Five fresh streams 8600-8604 generate
64 pools of eight draws per case. Compare slot zero against the mixture selector
on the same pool. Fully validate every raw edit for the audit, without using
those validation outcomes in selection and without candidate PDE solves.
Log selected yield, all raw yield, face/radius coverage, duplicates, rejection
categories and costs. This shared-pool audit does not compare production wall
times: baseline would generate only its first draw in deployment.

## Frozen equal-PDE-budget search

Freeze pool size, mixture probability and selection rule before inspecting the
audit. Do not tune from its outcomes. On each optimized input, run five new seeds
8800-8804 for both uniform and perimeter-pool methods, rotating method order.
Each search has 17 PDE solves including its own initial solve, and a hard cap
of 6000 raw draws. A final truncated pool uses the remaining raw allowance;
incomplete runs stay incomplete and must not masquerade as equal-budget pairs.
Uniform draws one proposal at a time. Both use greedy global Q acceptance,
the same tolerance and unchanged physical constraints. Reject duplicate valid
masks at a fixed incumbent before solving; clear that cache after acceptance.

Record all raw proposals and selected checks, valid and invalid time costs,
physical Q and eta histories, wall time, accepted moves, solve counts and fine
validation of initial/final geometries at n=200/400. Count every refinement solve
separately. Compare complete matched seeds and report all failures. One frozen
input per family and five seeds are exploratory evidence, not a general
optimality or RL-speedup claim. Existing scores and checkpoints stay unchanged.
