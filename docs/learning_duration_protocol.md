# Controlled continuation to 96 episodes

The user approves the diagnosis and longer-training plan on 2026-10-20.
Frozen configuration: `configs/learning_duration.yaml`.

Keep the 24 incumbent/candidate features, linear Expected Sarsa update, fixed
flux prior, learning rate, PDE model, source, admissibility rules, pool refill
and per-episode budgets unchanged. Reuse the five audited policies after 24
episodes from `20261019T133653820304Z_pioneer_rl_2382a80e`. Continue those same
five seed streams to 96 episodes, cycling through the same twelve training
starts and using the original permutation rule. This isolates additional
training at the final exploration level; it does not compare new architectures.

Do not stretch exploration decay over 96 episodes: that would change the first
24 episodes as well as training duration. For zero-based episode e, use

epsilon(e) = epsilon_start + (epsilon_end - epsilon_start)
             * min(e, 23) / 23.

Match the reference arithmetic operation order to preserve floating-point values.
For all later episodes use the resulting final exploration level. The unchanged
unconditional raw-proposal exploration floor remains 0.25. The epsilon-greedy
component among admissible proposals ends near 0.15; these are distinct mixtures.

Reconstruct the episode-12 policies from recorded features, rewards and Expected
Sarsa bootstrap targets. Parse CSV floats with round-trip precision. Replaying
all 24 episodes must reproduce each original final policy digest exactly before
reusing the prefix; this has passed for all five policies. Under unchanged
features, seed/curriculum and exploration schedule, reuse avoids 2040 repeated
training PDE solves. It is an exact prefix, not new training evidence.

Evaluate frozen policies at episodes 12, 24, 48 and 96. Requested checkpoint
versions are saved as training progresses; evaluation runs after
the fixed training schedule finishes and cannot steer it. All six previous
validation/test geometries now explicitly become development cases. Use two
fresh fixed rollout seeds (263000, 263001) on every case and a frozen flux
control with identical refill/PDE/raw budgets. Evaluate each control once per
case/seed, then reuse those results for the four checkpoint comparisons.
There are 240 model evaluations and twelve control evaluations. These share
geometries and controls across replicas and checkpoints and are not independent
samples. No fresh final test is used. Do not select or promote a final model.

Refine every evaluated initial/final endpoint to n=200 and n=400. Record Q,
eta, area, Robin perimeter, conservation, raw draws, validation count, completion,
workflow time and policy digest. Include incomplete outcomes and their costs;
report any complete-pair comparisons explicitly. Reward on the cycling training
curriculum is not a learning curve. Use frozen-checkpoint development outcomes
to assess whether extra training helps.

Maximum planned new experimental PDE cost if all budgets complete:

| Phase | New solves |
|---|---:|
| Preflight | 40 |
| Continue 5 replicas for 72 new episodes each | 6120 |
| 252 development rollouts, 17 coarse + 4 refinement solves | 5292 |
| Total | 11452 |

The reused 2040 training solves and previous preparation costs are historical
and separate. No new bank-generation or final-test solves are planned. Any
interrupted replay work has separately reported bounds. Save every completed
episode atomically and keep immutable checkpoint versions. Resume checks the
new config/source/bank/numerical environment plus pinned historical files.

The diagnostic uses saved fields and proposal metadata without solving a new
PDE. It exactly replays 2573 updates and reconstructs all 1252 learned evaluation
decisions. Final learned ranking differs from its fixed flux prior in 58.55% of
those decisions. Only selected candidates have solved outcomes, so these data
cannot establish the true gain of unselected counterfactual actions. TD error
is a bootstrapped value error, not a concentration or immediate-flux error.

After this development experiment, decide whether extra duration helped or
whether observations/model capacity deserve a separate controlled study. Any
future final comparison must reserve fresh masks after its design is frozen.
