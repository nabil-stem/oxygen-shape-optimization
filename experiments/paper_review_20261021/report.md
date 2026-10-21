# Paper evidence tables

285 unique returned endpoints; 2 endpoints / 2 distinct masks overlap the highest practical lower flux estimate. The highest fine-FEM value does not by itself prove a unique winner. All ranges concern the same staircase, with practical floating guards.

## Equal-budget five-method comparison

Known development starts, 45 pairs, 129 coarse solves per trajectory. Added controls were specified after the original results. Win counts use the descriptive 1e-7 threshold; resolved outcomes compare complementary energy ranges.

| method | wins | ties | losses | mean_delta_Q400 | mean_delta_FEM | resolved_win | overlap | resolved_loss |
|---|---|---|---|---|---|---|---|---|
| pool_greedy | 13 | 0 | 32 | -0.012937560568689202 | -0.0130428431654177 | 12 | 3 | 30 |
| uniform_greedy | 7 | 0 | 38 | -0.014769815031687797 | -0.014902563213674972 | 6 | 2 | 37 |
| flux_greedy | 0 | 45 | 0 | 0.0 | 0.0 | 0 | 45 | 0 |
| flux_anneal | 23 | 0 | 22 | 0.0002088280991023428 | 0.00022374516826655243 | 23 | 1 | 21 |
| learned_anneal | 19 | 0 | 26 | -0.0041416218849723435 | -0.004191418220892334 | 18 | 1 | 26 |

## Separate phases

| phase | method | endpoints | resolved_improvements | median_FEM_gain_percent | median_membrane_change_percent | incomplete |
|---|---|---|---|---|---|---|
| comparison | flux_anneal | 45 | 45 | 11.623273583734782 | 41.94756554307118 | 0 |
| comparison | flux_greedy | 45 | 45 | 12.308892238009527 | 32.31939163498099 | 0 |
| comparison | learned_anneal | 45 | 45 | 11.413213513593057 | 44.36363636363636 | 0 |
| comparison | pool_greedy | 45 | 45 | 9.534779058473909 | 40.15151515151516 | 0 |
| comparison | uniform_greedy | 45 | 45 | 8.157744509914334 | 33.70786516853934 | 0 |
| deep | flux_anneal | 5 | 5 | 25.692559792841617 | 20.521172638436468 | 0 |
| deep | flux_greedy | 5 | 5 | 25.99090884808579 | 23.12703583061888 | 0 |
| deep | learned_anneal | 5 | 5 | 23.510232271001907 | 26.0 | 0 |
| matched_perimeter | flux_anneal | 15 | 15 | 2.495802993677332 | 0.0 | 0 |
| matched_perimeter | flux_greedy | 15 | 14 | 1.808634830425726 | 0.0 | 0 |
| matched_perimeter | learned_anneal | 15 | 15 | 2.7919289096149047 | 0.0 | 0 |

Different phase budgets and actual membrane lengths preclude attributing cross-phase winners to morphology or method alone. Fixed perimeter is within each start, not a shared length across families.

Committed costs: original study 38,069, additional classical controls 12,330, collar/strip checks 322; total 50,721 solver calls. Warm training is included once; imported endpoints are not charged twice. Journal replay and this collection add no PDE solves.
