# Distinct valid proposal pools: frozen development protocol

The completed reward/prior study found multiple eligible choices in only about
4% of fully screened pools. This study isolates candidate availability before
changing or retraining the policy. It changes no physical constraint or solver.

## Algorithms

All methods draw the same ordinary paired-disk primitive, with uniform boundary
face pairs and radius U[0.05,0.08]. Each raw draw is charged. Cheap checks precede
the full geometry validator, including feature checks at 100/200/400. Duplicate
masks, the incumbent and candidates already solved at the current incumbent are
ineligible. Source and impermeable boundary classifications are protected.

1. fixed8: draw exactly eight raw proposals (or the remaining raw budget).
2. refill2: draw at least eight, then stop when two distinct valid proposals are
   available or 256 raw proposals have been drawn.
3. refill4: the same with four distinct valid alternatives.

Each pool is ranked either uniformly or by the existing incumbent-wall-flux
heuristic j_add*(max(j)-j_donor)/Bi^2. No candidate PDE is used in ranking. All
six methods retain probability 0.25 on the first unconditional raw proposal;
the remaining probability chooses the best eligible score, uniformly over ties.
If no eligible choice exists, select the first raw draw and reject it. A selected
candidate must pass all checks before a PDE solve; acceptance requires actual
global Q improvement >1e-7. Refill pools are discarded after one selection. This
deliberately measures selection opportunities rather than persistent reservoirs.

Uniform ranking with rejection sampling is not claimed to preserve the original
raw law after distinct-mask filtering or a stopping rule. Flux ranking is a
heuristic, not a derivative. Quotas are targets, never a reason to relax a check.

## Frozen matrix and accounting

Config: `configs/feasible_pools.yaml`. Six known development geometries: two
canonical and four improved masks from the completed revision study. The former
test masks are now development inputs; their file checksums are pinned and their
bytes copied before outcomes. Fresh rollout seeds 181000--181004. Six methods
times six starts times five seeds = 180 runs. No selection or training uses this
study as a held-out test. Execution order reverses on odd seeds to reduce timing
order effects; wall times remain descriptive, not equal-time experiments.

Per run: 17 coarse PDE solves including the initial solve; 6000 raw draws; full
geometry checks and all rejected draws charged. Endpoints independently solved
at n=200 and n=400, four extra PDE solves. Maximum 3780 new experimental solves,
no training solves or new bank-generation solves. Budget exhaustion is reported
as an outcome and retained in all tables. Strict conservation diagnostics apply.

Record physical Q, ideal Q, eta, area, Robin perimeter, every raw proposal and
its seed, probabilities, full-check counts/time, pool eligible multiplicity,
accepted edits, raw and PDE totals, endpoint refinement and workflow time.
Snapshot executed sources and software environment; Git hash may be null before
the first commit. Every solved candidate's mask and concentration are retained.

Report matched refined Q wins/ties/losses using tolerance 1e-7; controls reused
across two quota comparisons are not independent samples. Report raw valid
fraction separately from the fraction of pools offering multiple alternatives.
More opportunities alone are not scientific evidence of better flux, improved
raw yield, successful RL, generalization or speed. No automatic policy promotion.

[valid_pool_capacity.md](valid_pool_capacity.md) derives a simple binomial
benchmark for why larger pools can expose choices while spending many more
raw draws. It is not a fitted result or a guarantee for changing incumbents.
