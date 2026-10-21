# Empirical sensitivity-guided paired disk search

## Prespecified protocol

Use the existing deterministic finite-volume solver and unchanged constraints.
Benchmark uniform, flux-guided and empirical-sensitivity sampling with the same
batch-greedy schedule. Each round freezes the incumbent geometry, evaluates up
to eight valid candidates, and accepts the best only if global Q improves by
more than 1e-7. Every probe is also eligible for acceptance. The total budget is
41 PDE solves including the initial solve; there are no free sensitivity solves.
All methods validate the final geometry at n=200/400 outside the search budget.
This batched uniform control differs from the earlier one-candidate greedy v0.

For sensitivity search, the first four valid candidates in a round are uniform
probes. Further candidates use a mixture with uniform probability 0.25. Discard
the map when the incumbent changes; never reuse observations on a new geometry.
Partial final rounds still commit their best valid candidate. Invalid proposals
consume attempts/time but no PDE solve. After fifty consecutive invalid draws,
use uniform pairs until the next valid solve; the safeguard applies to both
guided methods. Attempt exhaustion remains an explicitly incomplete run.

## Empirical estimate, not a continuum shape derivative

A probe moves area a_k>0 from donor boundary point d_k to addition point p_k,
while maintaining exactly zero net area change. Define its signed response

    s_k = [Q(Omega_k_candidate) - Q(Omega_round)] / a_k.

Here a_k is the **reallocated** area, not the net area (which is zero). A paired
edit changes two wall locations simultaneously. It does not identify separate
addition/removal derivatives, nor does a finite-radius response establish an
infinitesimal shape derivative. Radii remain in [0.05,0.08].

On eligible boundary-face centers (p_i,d_j), define

    K_k(i,j) = exp(-[|p_i-p_k|^2 + |d_j-d_k|^2]/[2*l^2]),
    B(i,j) = sum_k K_k(i,j) s_k,
    w(i,j) = max(B(i,j),0),
    P(i,j) = epsilon/N^2 + (1-epsilon)*w(i,j)/sum(w).

The fixed bandwidth l=0.15 uses unit-box coordinates. Signed responses include
losses; clipping happens only after aggregation. The joint map preserves which
addition/donor pair was tested instead of assigning the pair's gain to either
location alone. If there is no positive weight, fall back to uniform. Every
pair keeps positive exploration probability, and maps are saved with points,
probe responses and probabilities. This is a spatial smoothing heuristic.

The flux control has joint weights j_i * (max(j)-j_j), with the same uniform
mixture and fallback; equal flux reduces to uniform. Flux is measured using the
reference solver's Robin face flux, never interpreted as a derivative.

## Validation and evaluation

Check the weighting formula on a constructed two-point response with known
signs, uniform-limit behavior, scale invariance, preservation of exploration
mass, and exact budget accounting. Test that probes use one fixed base geometry,
that each candidate is globally re-solved, and that all accepted states satisfy
the existing constraints. Record maps, all invalid attempts, signed responses,
search/refinement cost and time. Verify reproducibility with fixed seeds.

The initial comparison uses five fresh seeds 8000–8004, two valid shape families
(deep cavity and rectangle), and Bi=1. All three methods use identical budgets.
Report median/IQR, paired differences, finest-grid wins, target-reaching solve
counts and elapsed time. Keep incomplete runs visible and do not tune settings
against these scores. Transfer to other Bi values is a subsequent experiment.
Evidence of an advantage is an outcome to test, not a promised result.
