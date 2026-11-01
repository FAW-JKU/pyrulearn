# Native rule learners vs. their external references

## Setup

**Full run** (16 datasets). Quick sanity check instead: `python demos/native_vs_interfaced.py` with no arguments.

Four native rule learners against the external reference implementations
they are modeled on, on 15 small/medium and
1 large binary datasets from `pyrulearn.experiments.catalog`.
Two more pairs are covered by their own demos instead: Slipper vs.
IMod:Slipper (`ripper_comparison.py`'s own preliminary check), and PyLORD
vs. the reference JavaLORD (`demos/seco_learners_comparison.py`). A
fifth pair, ORB vs. RKD:RuleBoosting, is checked
separately below instead of in the main comparison -- see "Why optimal
rule boosting isn't in the main comparison". Across 2 small datasets (3-fold), ORB and RKD:RuleBoosting agreed closely on accuracy (mean 89.78% vs. 89.89%, 0.11 points apart; mean fit time 61.07s vs. 10.53s) -- but neither's exhaustive search scales to the main comparison's wider, one-hot-encoded feature counts within the 300s cap, a limit of the exhaustive-search approach itself rather than of either implementation, so the pair is checked here instead of in the main table below.

- **RuleFit** -- A sparse (L1) logistic regression over a random forest's leaves (100 trees, depth 4) -- the candidate pool native RuleFit doesn't yet grow itself (see the module docstring). Its own sparsity knob is a plain regularization strength (`C`, optionally `cv`-selected by predictive score) with no cap on the final rule count -- unlike IMod:RuleFit (right), it ends up keeping noticeably more, smaller rules for similar accuracy (measured: 70.8 rules/268.7 conditions vs. 25.3/59.6, mean). Neither a smaller fixed `C` nor `cv=` reproduces IMod:RuleFit's rule count reliably across datasets (tested: `cv=5` went from 67->28 rules on one dataset but 29->39, the wrong way, on another) -- it's a different sparsity *mechanism*, not a mismatched default, so left unmatched here.
  **IMod:RuleFit** -- imodels' RuleFit: gradient-boosted trees as the rule pool, then a sparse regression over their leaves, deliberately capped at `max_rules=30` -- it searches the regularization path for the sparsest fit under that ceiling (`get_best_alpha_under_max_rules`), rather than picking strength by predictive score alone.
- **Boomer** -- ENDER configured as BOOMER's single-label case: logistic loss, L2-regularized Newton steps, up to 1,000 rules.
  **MLRL:Boomer** -- The reference BOOMER implementation (`mlrl-boomer`), with its instance/feature sampling turned off to match native Boomer's full-data search (see PAIRS above), and, in this demo only, run on a single thread: by default it uses every core of the machine, which the native learners (all single-threaded) don't.
- **CBA** -- Classification Based on Associations: mine class association rules (here capped at max_len=3, see ROADMAP.md), sort by precedence, select by database coverage (CBA-CB, "M1").
  **PArc:CBA** -- The reference CBA implementation (`pyarc`), mining with Borgelt's `fim`; its defaults mirror the native CBA's.
- **Pypper** -- IREP* growth-and-pruning plus `ReplaceReviseOptimization`, run per class -- a re-implementation of RIPPER, not a port.
  **Weka:JRip** -- Weka's reference RIPPER implementation, via a subprocess (`pyrulearn.interfaces.weka.WekaJRip`); also compared at a larger scale, and against wittgenstein's RIPPER too, in `demos/ripper_comparison.py`.

Protocol: 5-fold stratified cross-validation (3-fold for the large adult) (`pyrulearn.experiments.runner.run_cv`), one
`DataSpec` per training fold (`build_dataspec(discretizer=TreeDiscretizer(max_intervals=8))`),
the test fold binarized against that same `DataSpec`. Each fit is capped
at 300s; a time-out or an error counts as a failure (and
as last in the ranking), not as the end of the run -- CBA/PArc:CBA's
itemset mining is the likeliest to hit this on wider data.

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: test accuracy, number of rules and conditions, fit time.

Threads: every learner here runs on one thread. `mlrl-boomer` would use every core of the machine by
default (about 70 for a single fit on the machine these results were measured on), which no native
learner does, so `MLRL:Boomer` is limited to one thread
(`parallel_rule_refinement`, `parallel_statistic_update` and `parallel_prediction` set to `false`)
to compare like with like. This limit is set by this demo only: `MLRLBoomer` itself keeps MLRL's
default and uses every core, and with that it would be faster than reported here.
Fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

![accuracy vs. rule-set complexity](native_vs_interfaced_plots/native_vs_interfaced_accuracy.png)

![fit time, native vs. interfaced](native_vs_interfaced_plots/native_vs_interfaced_fit_time.png)

## Why optimal rule boosting isn't in the main comparison

Across 2 small datasets (3-fold), ORB and RKD:RuleBoosting agreed closely on accuracy (mean 89.78% vs. 89.89%, 0.11 points apart; mean fit time 61.07s vs. 10.53s) -- but neither's exhaustive search scales to the main comparison's wider, one-hot-encoded feature counts within the 300s cap, a limit of the exhaustive-search approach itself rather than of either implementation, so the pair is checked here instead of in the main table below.

| learner | accuracy | fit_time |
|---|---|---|
| ORB | 89.78% | 61.067 |
| RKD:RuleBoosting | 89.89% | 10.527 |

## Native vs. interfaced, pair by pair (mean across the 15 small/medium datasets and their folds)

| pair | learner | accuracy | n_rules | n_conditions | fit_time (s) | failures |
|---|---|--:|--:|--:|--:|--:|
| RuleFit | RuleFit | 83.70% | 71.6 | 271.6 | 2.60 | 0 |
| RuleFit | IMod:RuleFit | 82.80% | 25.3 | 59.5 | 2.54 | 0 |
| Boomer | Boomer | 84.17% | 231.0 | 1291.2 | 1.75 | 0 |
| Boomer | MLRL:Boomer | 83.74% | 237.3 | 1302.9 | 4.75 | 0 |
| CBA | CBA | 83.22% | 63.7 | 183.1 | 46.80 | 0 |
| CBA | PArc:CBA | 83.28% | 63.5 | 182.7 | 22.15 | 0 |
| Pypper | Pypper | 84.06% | 3.6 | 9.2 | 0.05 | 0 |
| Pypper | Weka:JRip | 84.11% | 3.9 | 10.4 | 0.51 | 0 |

## Summary by learner (mean rank across the same datasets)

| learner | accuracy | n_rules | n_conditions | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| Boomer | 84.17% | 231.0 | 1291.2 | 1.75 | 1.9 | 3.63 | 0 |
| RuleFit | 83.70% | 71.6 | 271.6 | 2.60 | 2.7 | 4.07 | 0 |
| Weka:JRip | 84.11% | 3.9 | 10.4 | 0.51 | 2.5 | 4.20 | 0 |
| MLRL:Boomer | 83.74% | 237.3 | 1302.9 | 4.75 | 1.9 | 4.47 | 0 |
| Pypper | 84.06% | 3.6 | 9.2 | 0.05 | 3.0 | 4.50 | 0 |
| IMod:RuleFit | 82.80% | 25.3 | 59.5 | 2.54 | 1.5 | 4.90 | 0 |
| PArc:CBA | 83.28% | 63.5 | 182.7 | 22.15 | 0.5 | 5.10 | 0 |
| CBA | 83.22% | 63.7 | 183.1 | 46.80 | 0.9 | 5.13 | 0 |

Sorted by mean rank. `wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied for last.

## The large dataset: adult

3-fold cross-validation, same 300s cap. `folds` counts the folds that finished; the other columns are means over those folds only.

| learner | folds | accuracy | n_rules | n_conditions | fit_time (s) |
|---|--:|--:|--:|--:|--:|
| Boomer | 3/3 | 86.81% | 751.0 | 6815.0 | 67.12 |
| CBA | 0/3 | -- | -- | -- | -- |
| IMod:RuleFit | 3/3 | 84.97% | 26.0 | 42.7 | 59.68 |
| MLRL:Boomer | 3/3 | 86.88% | 768.7 | 7241.0 | 207.47 |
| PArc:CBA | 0/3 | -- | -- | -- | -- |
| Pypper | 3/3 | 84.23% | 12.3 | 69.0 | 1.72 |
| RuleFit | 3/3 | 86.59% | 405.3 | 1612.0 | 90.89 |
| Weka:JRip | 3/3 | 84.80% | 19.3 | 110.3 | 44.00 |

## Per-dataset results, per learner (mean across folds)

| dataset | learner | accuracy | n_rules | n_conditions | fit_time |
|---|---|---|---|---|---|
| SPECT | Boomer | 81.63% | 128.2 | 581.0 | 0.673 |
| SPECT | CBA | 80.92% | 33.8 | 81.8 | 1.891 |
| SPECT | IMod:RuleFit | 81.66% | 21.2 | 49.8 | 1.711 |
| SPECT | MLRL:Boomer | 81.25% | 126.8 | 563.2 | 0.551 |
| SPECT | PArc:CBA | 80.54% | 31.8 | 76.6 | 0.657 |
| SPECT | Pypper | 79.80% | 1.0 | 5.2 | 0.018 |
| SPECT | RuleFit | 82.77% | 35.2 | 132.8 | 0.806 |
| SPECT | Weka:JRip | 82.77% | 1.0 | 5.2 | 0.352 |
| breast-cancer | Boomer | 70.99% | 263.8 | 1445.8 | 0.934 |
| breast-cancer | CBA | 68.19% | 69.2 | 199.6 | 4.714 |
| breast-cancer | IMod:RuleFit | 72.37% | 27.2 | 70.4 | 1.789 |
| breast-cancer | MLRL:Boomer | 72.40% | 229.0 | 1095.0 | 0.841 |
| breast-cancer | PArc:CBA | 69.93% | 68.8 | 198.6 | 1.700 |
| breast-cancer | Pypper | 74.45% | 1.2 | 2.0 | 0.017 |
| breast-cancer | RuleFit | 69.24% | 68.8 | 269.2 | 1.037 |
| breast-cancer | Weka:JRip | 72.40% | 1.2 | 2.4 | 0.363 |
| colic | Boomer | 83.98% | 290.2 | 1406.4 | 1.389 |
| colic | CBA | 79.34% | 79.0 | 232.4 | 79.525 |
| colic | IMod:RuleFit | 75.29% | 28.6 | 60.2 | 2.115 |
| colic | MLRL:Boomer | 78.01% | 342.6 | 2178.2 | 4.073 |
| colic | PArc:CBA | 79.07% | 80.0 | 235.4 | 32.181 |
| colic | Pypper | 84.79% | 2.0 | 4.2 | 0.042 |
| colic | RuleFit | 80.17% | 73.8 | 277.2 | 1.569 |
| colic | Weka:JRip | 84.52% | 2.6 | 5.2 | 0.397 |
| credit-approval | Boomer | 84.78% | 374.4 | 2420.4 | 1.643 |
| credit-approval | CBA | 85.22% | 130.2 | 379.6 | 37.806 |
| credit-approval | IMod:RuleFit | 86.09% | 24.2 | 55.0 | 1.681 |
| credit-approval | MLRL:Boomer | 84.78% | 356.6 | 2114.8 | 3.208 |
| credit-approval | PArc:CBA | 85.07% | 129.4 | 377.4 | 14.668 |
| credit-approval | Pypper | 85.80% | 2.4 | 5.4 | 0.030 |
| credit-approval | RuleFit | 84.93% | 92.8 | 362.0 | 1.663 |
| credit-approval | Weka:JRip | 86.23% | 4.2 | 8.8 | 0.424 |
| credit-g | Boomer | 74.70% | 576.6 | 4022.4 | 2.420 |
| credit-g | CBA | 72.50% | 148.6 | 441.2 | 122.527 |
| credit-g | IMod:RuleFit | 72.90% | 25.6 | 61.0 | 2.517 |
| credit-g | MLRL:Boomer | 75.20% | 607.4 | 4186.4 | 5.369 |
| credit-g | PArc:CBA | 72.60% | 148.8 | 441.8 | 84.214 |
| credit-g | Pypper | 71.10% | 2.4 | 7.6 | 0.040 |
| credit-g | RuleFit | 73.90% | 216.8 | 854.4 | 2.598 |
| credit-g | Weka:JRip | 70.20% | 3.0 | 12.0 | 0.536 |
| dresses-sales | Boomer | 57.20% | 356.8 | 2385.0 | 1.384 |
| dresses-sales | CBA | 57.40% | 155.2 | 448.4 | 186.682 |
| dresses-sales | IMod:RuleFit | 59.60% | 28.2 | 78.4 | 1.923 |
| dresses-sales | MLRL:Boomer | 59.00% | 296.4 | 1635.6 | 3.140 |
| dresses-sales | PArc:CBA | 58.00% | 153.2 | 442.8 | 68.503 |
| dresses-sales | Pypper | 62.20% | 1.2 | 1.4 | 0.029 |
| dresses-sales | RuleFit | 57.60% | 125.6 | 480.2 | 1.571 |
| dresses-sales | Weka:JRip | 61.60% | 1.6 | 2.2 | 0.436 |
| heart-c | Boomer | 79.88% | 235.0 | 1176.6 | 0.951 |
| heart-c | CBA | 79.55% | 60.6 | 178.0 | 17.319 |
| heart-c | IMod:RuleFit | 81.83% | 26.8 | 52.6 | 1.536 |
| heart-c | MLRL:Boomer | 79.22% | 261.0 | 1290.8 | 1.396 |
| heart-c | PArc:CBA | 79.55% | 61.8 | 181.6 | 6.850 |
| heart-c | Pypper | 76.56% | 2.2 | 5.4 | 0.021 |
| heart-c | RuleFit | 78.56% | 65.2 | 247.2 | 1.392 |
| heart-c | Weka:JRip | 77.89% | 3.0 | 6.8 | 0.355 |
| heart-h | Boomer | 80.27% | 162.4 | 769.2 | 0.861 |
| heart-h | CBA | 78.91% | 61.4 | 172.0 | 13.959 |
| heart-h | IMod:RuleFit | 74.86% | 26.4 | 64.8 | 1.450 |
| heart-h | MLRL:Boomer | 78.23% | 174.6 | 830.4 | 0.970 |
| heart-h | PArc:CBA | 78.56% | 61.8 | 173.2 | 5.001 |
| heart-h | Pypper | 79.59% | 1.6 | 2.4 | 0.018 |
| heart-h | RuleFit | 80.62% | 57.2 | 211.6 | 1.042 |
| heart-h | Weka:JRip | 78.22% | 2.2 | 4.2 | 0.356 |
| heart-statlog | Boomer | 77.41% | 200.8 | 970.6 | 1.077 |
| heart-statlog | CBA | 80.00% | 51.0 | 149.8 | 16.740 |
| heart-statlog | IMod:RuleFit | 80.00% | 24.8 | 49.8 | 1.691 |
| heart-statlog | MLRL:Boomer | 77.04% | 236.8 | 1127.6 | 1.530 |
| heart-statlog | PArc:CBA | 79.63% | 51.2 | 150.4 | 6.455 |
| heart-statlog | Pypper | 79.26% | 3.2 | 7.2 | 0.039 |
| heart-statlog | RuleFit | 79.26% | 56.8 | 212.4 | 1.078 |
| heart-statlog | Weka:JRip | 78.15% | 3.4 | 8.0 | 0.377 |
| hepatitis | Boomer | 78.06% | 94.0 | 361.8 | 0.776 |
| hepatitis | CBA | 76.77% | 18.8 | 51.2 | 15.479 |
| hepatitis | IMod:RuleFit | 72.26% | 30.6 | 72.6 | 1.803 |
| hepatitis | MLRL:Boomer | 78.71% | 86.0 | 321.4 | 0.838 |
| hepatitis | PArc:CBA | 76.77% | 18.8 | 51.2 | 5.186 |
| hepatitis | Pypper | 76.13% | 2.0 | 3.2 | 0.025 |
| hepatitis | RuleFit | 76.13% | 28.8 | 91.6 | 0.714 |
| hepatitis | Weka:JRip | 79.35% | 2.2 | 4.8 | 0.346 |
| kr-vs-kp | Boomer | 99.50% | 121.4 | 684.2 | 2.536 |
| kr-vs-kp | CBA | 97.75% | 41.8 | 122.0 | 6.268 |
| kr-vs-kp | IMod:RuleFit | 96.65% | 19.2 | 52.4 | 2.337 |
| kr-vs-kp | MLRL:Boomer | 99.50% | 150.4 | 831.2 | 5.102 |
| kr-vs-kp | PArc:CBA | 97.75% | 41.8 | 122.0 | 2.880 |
| kr-vs-kp | Pypper | 98.97% | 13.8 | 43.6 | 0.139 |
| kr-vs-kp | RuleFit | 98.69% | 61.4 | 231.6 | 5.233 |
| kr-vs-kp | Weka:JRip | 99.12% | 13.4 | 40.4 | 0.694 |
| mushroom | Boomer | 100.00% | 41.6 | 144.4 | 6.372 |
| mushroom | CBA | 100.00% | 9.2 | 27.6 | 157.882 |
| mushroom | IMod:RuleFit | 99.91% | 20.6 | 43.8 | 9.269 |
| mushroom | MLRL:Boomer | 100.00% | 56.0 | 187.0 | 31.117 |
| mushroom | PArc:CBA | 100.00% | 9.2 | 27.6 | 78.639 |
| mushroom | Pypper | 99.96% | 6.6 | 9.8 | 0.089 |
| mushroom | RuleFit | 100.00% | 16.8 | 61.4 | 9.491 |
| mushroom | Weka:JRip | 99.98% | 6.0 | 9.4 | 1.437 |
| sick | Boomer | 98.52% | 274.6 | 1610.8 | 3.346 |
| sick | CBA | 97.67% | 52.0 | 138.0 | 38.035 |
| sick | IMod:RuleFit | 98.20% | 28.6 | 69.2 | 4.365 |
| sick | MLRL:Boomer | 98.01% | 291.4 | 1779.0 | 10.522 |
| sick | PArc:CBA | 97.69% | 51.6 | 136.8 | 24.038 |
| sick | Pypper | 98.06% | 3.4 | 11.2 | 0.064 |
| sick | RuleFit | 98.22% | 77.0 | 296.2 | 7.119 |
| sick | Weka:JRip | 98.14% | 4.8 | 15.4 | 0.776 |
| tic-tac-toe | Boomer | 100.00% | 245.8 | 1059.8 | 1.176 |
| tic-tac-toe | CBA | 100.00% | 10.0 | 30.0 | 2.639 |
| tic-tac-toe | IMod:RuleFit | 94.78% | 20.4 | 52.4 | 1.594 |
| tic-tac-toe | MLRL:Boomer | 100.00% | 253.8 | 1093.4 | 1.894 |
| tic-tac-toe | PArc:CBA | 100.00% | 10.0 | 30.0 | 1.039 |
| tic-tac-toe | Pypper | 98.33% | 8.0 | 24.2 | 0.088 |
| tic-tac-toe | RuleFit | 99.37% | 68.8 | 262.4 | 2.310 |
| tic-tac-toe | Weka:JRip | 98.12% | 8.4 | 26.0 | 0.469 |
| vote | Boomer | 95.63% | 99.8 | 329.8 | 0.726 |
| vote | CBA | 94.02% | 34.4 | 94.6 | 0.592 |
| vote | IMod:RuleFit | 95.63% | 27.0 | 59.8 | 2.298 |
| vote | MLRL:Boomer | 94.71% | 91.2 | 309.6 | 0.767 |
| vote | PArc:CBA | 94.02% | 34.4 | 94.6 | 0.251 |
| vote | Pypper | 95.86% | 2.6 | 5.6 | 0.053 |
| vote | RuleFit | 96.09% | 28.4 | 84.2 | 1.301 |
| vote | Weka:JRip | 94.94% | 2.2 | 5.0 | 0.365 |

