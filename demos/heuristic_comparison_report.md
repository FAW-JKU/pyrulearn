# Rule learning heuristics in separate-and-conquer (after Janssen & Fürnkranz)

## Setup

**Full run** (36 datasets, 10-fold). Quick sanity check instead: `python demos/heuristic_comparison.py` with no arguments.

Rule learning heuristics compared inside one plain separate-and-conquer
learner, after Janssen & Fürnkranz (Machine Learning, 2010). The learner
follows the paper's Algorithms 1 and 2: greedy top-down search that
refines until no refinement is left and returns the best rule
encountered (`BeamSearch(beam_width=1)`), no stopping criterion and no
pruning; covering stops once a new rule covers no more positives than
negatives; multi-class problems use ordered class binarization (least
frequent class first, the largest class as the default rule).

Heuristics: the paper's standard ones (Precision, Laplace, Accuracy,
WRA, Correlation), its tuned parametrized ones (cost c=0.437, relative
cost c_r=0.342, F-measure beta=0.5, m-estimate m=22.466), and an
m-estimate sweep over m in {2, 4, 8, 16, 32}.
Benchmarks: Weka's JRip and pyrulearn's Pypper, both with their own
pruning, and both also use ordered class binarization. What each
heuristic prefers is visualized by its coverage-space isometrics in the
companion demo, [heuristic_isometrics](heuristic_isometrics_report.md).

Datasets: 36 of the paper's datasets that are in
`pyrulearn.experiments.catalog` (20 binary,
16 multi-class). Numeric attributes are discretized per
training fold (`build_dataspec(discretizer=TreeDiscretizer(max_intervals=8))`) instead
of the paper's tests at every midpoint. Each fit is capped at
600s; a timeout counts as a failure.

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: test accuracy, number of rules and conditions, fit time.

Every learner runs single-threaded; fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

![accuracy vs. theory size](heuristic_comparison_plots/heuristic_comparison_tradeoff.png)

![m-estimate sweep](heuristic_comparison_plots/heuristic_comparison_m_sweep.png)

## Summary by learner (mean across every dataset and fold)

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| m-estimate (m=16) | 83.82% | 17.2 | 60.8 | 3.3 | 0.08 | 6.3 | 6.40 | 0 |
| m-estimate (m=8) | 83.42% | 22.4 | 77.3 | 3.1 | 0.09 | 1.1 | 7.88 | 0 |
| Pypper | 83.36% | 7.1 | 19.0 | 2.4 | 0.07 | 4.7 | 6.62 | 0 |
| m-estimate (m=32) | 83.36% | 12.9 | 48.4 | 3.6 | 0.08 | 2.3 | 6.72 | 0 |
| m-estimate (m=22.466) | 83.17% | 14.9 | 54.4 | 3.4 | 0.08 | 1.3 | 7.42 | 0 |
| m-estimate (m=4) | 82.92% | 28.2 | 91.9 | 3.0 | 0.09 | 1.4 | 8.08 | 0 |
| Weka:JRip | 82.67% | 7.7 | 20.3 | 2.4 | 0.52 | 6.2 | 7.65 | 0 |
| Correlation | 82.65% | 14.4 | 52.1 | 3.3 | 0.06 | 4.1 | 8.31 | 0 |
| m-estimate (m=2) | 82.62% | 35.6 | 115.3 | 2.9 | 0.09 | 1.4 | 8.24 | 0 |
| relative cost (c_r=0.342) | 82.30% | 9.6 | 35.3 | 3.3 | 0.06 | 0.5 | 8.88 | 0 |
| F-measure (beta=0.5) | 81.77% | 19.5 | 61.5 | 2.9 | 0.08 | 1.1 | 9.22 | 0 |
| Laplace | 81.67% | 43.6 | 134.0 | 2.7 | 0.09 | 0.4 | 10.43 | 0 |
| Precision | 81.53% | 68.0 | 205.5 | 2.7 | 0.15 | 1.1 | 10.57 | 0 |
| cost (c=0.437) | 81.40% | 44.4 | 143.1 | 2.6 | 0.10 | 1.1 | 9.93 | 0 |
| Accuracy | 81.14% | 39.1 | 126.2 | 2.6 | 0.10 | 0.1 | 10.18 | 0 |
| WRA | 80.51% | 4.7 | 16.2 | 3.2 | 0.04 | 3.0 | 9.47 | 0 |

`wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied for last.

## Per-dataset results, per learner (mean across folds)

| dataset | learner | accuracy | n_rules | n_conditions | fit_time |
|---|---|---|---|---|---|
| anneal | Accuracy | 94.76% | 16.5 | 30.6 | 0.054 |
| anneal | Correlation | 93.99% | 11.0 | 22.2 | 0.049 |
| anneal | F-measure (beta=0.5) | 96.43% | 17.3 | 36.2 | 0.066 |
| anneal | Laplace | 96.32% | 22.1 | 55.6 | 0.068 |
| anneal | Precision | 96.32% | 26.8 | 73.0 | 0.069 |
| anneal | Pypper | 94.32% | 8.9 | 22.5 | 0.096 |
| anneal | WRA | 91.64% | 6.1 | 15.5 | 0.032 |
| anneal | Weka:JRip | 89.98% | 6.7 | 10.9 | 0.545 |
| anneal | cost (c=0.437) | 94.20% | 16.9 | 30.6 | 0.037 |
| anneal | m-estimate (m=16) | 96.55% | 15.3 | 32.1 | 0.068 |
| anneal | m-estimate (m=2) | 96.99% | 21.8 | 50.9 | 0.078 |
| anneal | m-estimate (m=22.466) | 95.99% | 13.0 | 26.0 | 0.063 |
| anneal | m-estimate (m=32) | 94.65% | 11.8 | 20.1 | 0.051 |
| anneal | m-estimate (m=4) | 97.66% | 20.3 | 46.1 | 0.075 |
| anneal | m-estimate (m=8) | 96.66% | 18.1 | 37.7 | 0.063 |
| anneal | relative cost (c_r=0.342) | 93.09% | 8.0 | 16.5 | 0.044 |
| audiology | Accuracy | 80.99% | 40.4 | 85.1 | 0.082 |
| audiology | Correlation | 84.05% | 34.0 | 72.5 | 0.080 |
| audiology | F-measure (beta=0.5) | 77.49% | 42.3 | 85.5 | 0.081 |
| audiology | Laplace | 74.33% | 55.0 | 105.2 | 0.076 |
| audiology | Precision | 75.22% | 67.0 | 130.3 | 0.113 |
| audiology | Pypper | 75.71% | 11.1 | 17.3 | 0.111 |
| audiology | WRA | 82.77% | 23.2 | 59.1 | 0.117 |
| audiology | Weka:JRip | 74.84% | 14.7 | 26.2 | 0.459 |
| audiology | cost (c=0.437) | 79.17% | 44.0 | 93.3 | 0.066 |
| audiology | m-estimate (m=16) | 83.18% | 31.8 | 70.0 | 0.077 |
| audiology | m-estimate (m=2) | 77.94% | 44.6 | 88.2 | 0.082 |
| audiology | m-estimate (m=22.466) | 83.62% | 29.1 | 68.0 | 0.086 |
| audiology | m-estimate (m=32) | 84.98% | 27.7 | 63.9 | 0.108 |
| audiology | m-estimate (m=4) | 79.66% | 39.0 | 82.7 | 0.073 |
| audiology | m-estimate (m=8) | 80.55% | 34.9 | 75.4 | 0.072 |
| audiology | relative cost (c_r=0.342) | 82.75% | 24.3 | 62.1 | 0.124 |
| balance-scale | Accuracy | 77.61% | 65.6 | 240.8 | 0.139 |
| balance-scale | Correlation | 78.72% | 13.3 | 41.7 | 0.037 |
| balance-scale | F-measure (beta=0.5) | 80.63% | 16.2 | 47.5 | 0.049 |
| balance-scale | Laplace | 78.88% | 92.7 | 326.9 | 0.167 |
| balance-scale | Precision | 77.27% | 125.4 | 425.5 | 0.234 |
| balance-scale | Pypper | 78.38% | 7.9 | 23.8 | 0.072 |
| balance-scale | WRA | 72.65% | 2.5 | 6.5 | 0.014 |
| balance-scale | Weka:JRip | 80.31% | 8.1 | 25.2 | 0.439 |
| balance-scale | cost (c=0.437) | 80.63% | 70.1 | 255.3 | 0.129 |
| balance-scale | m-estimate (m=16) | 82.70% | 13.0 | 39.2 | 0.045 |
| balance-scale | m-estimate (m=2) | 73.10% | 78.0 | 335.2 | 0.180 |
| balance-scale | m-estimate (m=22.466) | 79.83% | 11.1 | 33.1 | 0.036 |
| balance-scale | m-estimate (m=32) | 80.94% | 8.4 | 25.1 | 0.033 |
| balance-scale | m-estimate (m=4) | 77.26% | 27.8 | 89.8 | 0.073 |
| balance-scale | m-estimate (m=8) | 76.64% | 21.8 | 71.1 | 0.061 |
| balance-scale | relative cost (c_r=0.342) | 76.62% | 4.9 | 12.2 | 0.020 |
| breast-cancer | Accuracy | 72.38% | 10.9 | 27.2 | 0.024 |
| breast-cancer | Correlation | 67.49% | 3.3 | 35.9 | 0.029 |
| breast-cancer | F-measure (beta=0.5) | 67.54% | 8.4 | 38.3 | 0.052 |
| breast-cancer | Laplace | 70.58% | 22.7 | 58.4 | 0.033 |
| breast-cancer | Precision | 68.84% | 36.3 | 95.1 | 0.057 |
| breast-cancer | Pypper | 74.84% | 1.0 | 2.0 | 0.025 |
| breast-cancer | WRA | 68.46% | 1.2 | 6.1 | 0.015 |
| breast-cancer | Weka:JRip | 71.37% | 1.3 | 2.5 | 0.370 |
| breast-cancer | cost (c=0.437) | 71.32% | 14.8 | 34.9 | 0.024 |
| breast-cancer | m-estimate (m=16) | 63.26% | 8.9 | 41.3 | 0.052 |
| breast-cancer | m-estimate (m=2) | 70.23% | 24.4 | 69.8 | 0.046 |
| breast-cancer | m-estimate (m=22.466) | 63.24% | 8.9 | 43.1 | 0.049 |
| breast-cancer | m-estimate (m=32) | 66.48% | 6.5 | 35.6 | 0.045 |
| breast-cancer | m-estimate (m=4) | 67.84% | 21.9 | 70.0 | 0.060 |
| breast-cancer | m-estimate (m=8) | 67.78% | 15.8 | 68.2 | 0.057 |
| breast-cancer | relative cost (c_r=0.342) | 66.79% | 10.7 | 38.7 | 0.045 |
| breast-w | Accuracy | 95.71% | 8.9 | 14.9 | 0.023 |
| breast-w | Correlation | 94.56% | 6.1 | 11.8 | 0.020 |
| breast-w | F-measure (beta=0.5) | 95.57% | 11.0 | 23.5 | 0.033 |
| breast-w | Laplace | 94.99% | 21.0 | 34.1 | 0.040 |
| breast-w | Precision | 94.57% | 42.9 | 69.2 | 0.072 |
| breast-w | Pypper | 95.57% | 3.9 | 8.1 | 0.032 |
| breast-w | WRA | 94.56% | 2.9 | 7.1 | 0.024 |
| breast-w | Weka:JRip | 95.42% | 4.5 | 8.5 | 0.499 |
| breast-w | cost (c=0.437) | 96.00% | 10.0 | 17.0 | 0.025 |
| breast-w | m-estimate (m=16) | 95.28% | 6.7 | 24.8 | 0.035 |
| breast-w | m-estimate (m=2) | 95.28% | 16.8 | 29.8 | 0.033 |
| breast-w | m-estimate (m=22.466) | 96.00% | 6.0 | 22.3 | 0.038 |
| breast-w | m-estimate (m=32) | 96.28% | 5.5 | 21.5 | 0.039 |
| breast-w | m-estimate (m=4) | 94.85% | 13.0 | 26.7 | 0.030 |
| breast-w | m-estimate (m=8) | 94.99% | 9.2 | 25.5 | 0.028 |
| breast-w | relative cost (c_r=0.342) | 94.13% | 3.0 | 5.9 | 0.023 |
| colic | Accuracy | 80.18% | 24.7 | 61.0 | 0.055 |
| colic | Correlation | 83.73% | 13.7 | 42.0 | 0.037 |
| colic | F-measure (beta=0.5) | 82.35% | 15.9 | 44.2 | 0.050 |
| colic | Laplace | 76.39% | 37.8 | 98.3 | 0.072 |
| colic | Precision | 76.91% | 61.7 | 158.2 | 0.116 |
| colic | Pypper | 83.72% | 3.0 | 7.0 | 0.038 |
| colic | WRA | 83.72% | 1.4 | 3.5 | 0.013 |
| colic | Weka:JRip | 84.50% | 3.4 | 8.4 | 0.398 |
| colic | cost (c=0.437) | 79.08% | 26.8 | 64.2 | 0.046 |
| colic | m-estimate (m=16) | 79.08% | 10.6 | 39.3 | 0.051 |
| colic | m-estimate (m=2) | 76.12% | 28.4 | 84.5 | 0.073 |
| colic | m-estimate (m=22.466) | 80.46% | 8.5 | 35.3 | 0.047 |
| colic | m-estimate (m=32) | 81.55% | 7.0 | 30.2 | 0.052 |
| colic | m-estimate (m=4) | 77.21% | 22.8 | 72.7 | 0.060 |
| colic | m-estimate (m=8) | 74.47% | 13.8 | 50.4 | 0.055 |
| colic | relative cost (c_r=0.342) | 83.18% | 5.4 | 21.4 | 0.030 |
| credit-approval | Accuracy | 82.32% | 17.7 | 44.3 | 0.047 |
| credit-approval | Correlation | 81.88% | 14.1 | 43.4 | 0.055 |
| credit-approval | F-measure (beta=0.5) | 81.74% | 22.8 | 82.4 | 0.111 |
| credit-approval | Laplace | 80.14% | 49.7 | 144.3 | 0.104 |
| credit-approval | Precision | 80.29% | 97.1 | 273.0 | 0.212 |
| credit-approval | Pypper | 85.36% | 2.7 | 6.3 | 0.055 |
| credit-approval | WRA | 83.62% | 1.0 | 3.4 | 0.018 |
| credit-approval | Weka:JRip | 84.78% | 4.2 | 10.1 | 0.459 |
| credit-approval | cost (c=0.437) | 83.04% | 21.2 | 54.1 | 0.053 |
| credit-approval | m-estimate (m=16) | 83.48% | 14.7 | 81.6 | 0.110 |
| credit-approval | m-estimate (m=2) | 82.46% | 41.8 | 131.5 | 0.108 |
| credit-approval | m-estimate (m=22.466) | 83.33% | 9.1 | 49.5 | 0.099 |
| credit-approval | m-estimate (m=32) | 83.62% | 7.6 | 39.5 | 0.086 |
| credit-approval | m-estimate (m=4) | 82.46% | 37.2 | 130.5 | 0.119 |
| credit-approval | m-estimate (m=8) | 81.59% | 24.8 | 101.0 | 0.132 |
| credit-approval | relative cost (c_r=0.342) | 85.36% | 2.4 | 9.8 | 0.035 |
| credit-g | Accuracy | 68.00% | 139.8 | 458.2 | 0.333 |
| credit-g | Correlation | 67.40% | 29.1 | 176.5 | 0.133 |
| credit-g | F-measure (beta=0.5) | 67.40% | 34.0 | 182.0 | 0.181 |
| credit-g | Laplace | 68.10% | 134.0 | 431.3 | 0.264 |
| credit-g | Precision | 69.40% | 193.0 | 619.1 | 0.410 |
| credit-g | Pypper | 70.40% | 3.4 | 13.0 | 0.065 |
| credit-g | WRA | 69.00% | 1.2 | 9.1 | 0.021 |
| credit-g | Weka:JRip | 71.20% | 3.6 | 12.9 | 0.553 |
| credit-g | cost (c=0.437) | 68.10% | 159.5 | 513.3 | 0.328 |
| credit-g | m-estimate (m=16) | 66.20% | 45.0 | 244.4 | 0.229 |
| credit-g | m-estimate (m=2) | 70.90% | 100.6 | 338.7 | 0.210 |
| credit-g | m-estimate (m=22.466) | 67.00% | 37.0 | 225.8 | 0.236 |
| credit-g | m-estimate (m=32) | 68.10% | 29.9 | 208.7 | 0.239 |
| credit-g | m-estimate (m=4) | 68.30% | 78.9 | 288.7 | 0.197 |
| credit-g | m-estimate (m=8) | 67.00% | 61.4 | 263.0 | 0.212 |
| credit-g | relative cost (c_r=0.342) | 67.70% | 19.8 | 129.7 | 0.153 |
| diabetes | Accuracy | 74.48% | 23.1 | 64.4 | 0.058 |
| diabetes | Correlation | 74.48% | 6.1 | 35.6 | 0.040 |
| diabetes | F-measure (beta=0.5) | 73.05% | 7.0 | 34.1 | 0.054 |
| diabetes | Laplace | 72.53% | 59.7 | 219.4 | 0.137 |
| diabetes | Precision | 72.14% | 101.7 | 375.0 | 0.234 |
| diabetes | Pypper | 74.22% | 3.8 | 12.6 | 0.053 |
| diabetes | WRA | 73.29% | 1.1 | 2.6 | 0.014 |
| diabetes | Weka:JRip | 72.53% | 3.9 | 12.4 | 0.452 |
| diabetes | cost (c=0.437) | 74.62% | 29.5 | 85.7 | 0.069 |
| diabetes | m-estimate (m=16) | 75.26% | 15.1 | 79.7 | 0.086 |
| diabetes | m-estimate (m=2) | 71.48% | 53.8 | 232.3 | 0.158 |
| diabetes | m-estimate (m=22.466) | 72.79% | 10.3 | 52.4 | 0.068 |
| diabetes | m-estimate (m=32) | 73.56% | 8.6 | 44.4 | 0.059 |
| diabetes | m-estimate (m=4) | 71.22% | 38.4 | 172.1 | 0.147 |
| diabetes | m-estimate (m=8) | 73.70% | 24.5 | 115.7 | 0.111 |
| diabetes | relative cost (c_r=0.342) | 73.84% | 6.7 | 23.4 | 0.041 |
| glass | Accuracy | 64.05% | 27.0 | 73.2 | 0.056 |
| glass | Correlation | 63.55% | 20.7 | 62.9 | 0.057 |
| glass | F-measure (beta=0.5) | 65.91% | 28.0 | 81.0 | 0.073 |
| glass | Laplace | 65.04% | 33.7 | 91.9 | 0.053 |
| glass | Precision | 66.97% | 50.0 | 139.7 | 0.078 |
| glass | Pypper | 67.38% | 6.2 | 14.3 | 0.061 |
| glass | WRA | 69.70% | 4.7 | 20.0 | 0.040 |
| glass | Weka:JRip | 66.84% | 6.6 | 16.1 | 0.400 |
| glass | cost (c=0.437) | 62.19% | 30.2 | 81.3 | 0.047 |
| glass | m-estimate (m=16) | 72.03% | 17.9 | 58.0 | 0.067 |
| glass | m-estimate (m=2) | 66.95% | 30.2 | 85.7 | 0.061 |
| glass | m-estimate (m=22.466) | 65.04% | 14.9 | 51.0 | 0.069 |
| glass | m-estimate (m=32) | 64.52% | 12.3 | 44.2 | 0.061 |
| glass | m-estimate (m=4) | 67.40% | 25.4 | 74.7 | 0.064 |
| glass | m-estimate (m=8) | 70.13% | 22.6 | 70.9 | 0.064 |
| glass | relative cost (c_r=0.342) | 68.27% | 13.6 | 47.3 | 0.058 |
| hayes-roth | Accuracy | 76.25% | 6.6 | 10.1 | 0.011 |
| hayes-roth | Correlation | 84.38% | 6.2 | 9.4 | 0.011 |
| hayes-roth | F-measure (beta=0.5) | 82.50% | 7.5 | 12.7 | 0.016 |
| hayes-roth | Laplace | 83.75% | 9.4 | 18.9 | 0.012 |
| hayes-roth | Precision | 83.75% | 11.8 | 24.8 | 0.015 |
| hayes-roth | Pypper | 84.38% | 6.1 | 9.6 | 0.031 |
| hayes-roth | WRA | 75.00% | 5.0 | 6.1 | 0.010 |
| hayes-roth | Weka:JRip | 83.12% | 5.8 | 8.5 | 0.336 |
| hayes-roth | cost (c=0.437) | 84.38% | 8.2 | 15.1 | 0.012 |
| hayes-roth | m-estimate (m=16) | 84.38% | 7.1 | 11.5 | 0.013 |
| hayes-roth | m-estimate (m=2) | 85.00% | 8.9 | 17.0 | 0.013 |
| hayes-roth | m-estimate (m=22.466) | 83.75% | 6.8 | 10.9 | 0.012 |
| hayes-roth | m-estimate (m=32) | 83.75% | 6.6 | 10.3 | 0.012 |
| hayes-roth | m-estimate (m=4) | 85.62% | 7.9 | 14.0 | 0.013 |
| hayes-roth | m-estimate (m=8) | 86.25% | 7.6 | 12.8 | 0.013 |
| hayes-roth | relative cost (c_r=0.342) | 84.38% | 6.8 | 10.6 | 0.013 |
| heart-c | Accuracy | 71.63% | 22.4 | 65.9 | 0.039 |
| heart-c | Correlation | 73.27% | 10.6 | 45.0 | 0.040 |
| heart-c | F-measure (beta=0.5) | 74.28% | 15.7 | 53.4 | 0.050 |
| heart-c | Laplace | 75.97% | 33.9 | 96.0 | 0.054 |
| heart-c | Precision | 74.96% | 59.5 | 154.2 | 0.094 |
| heart-c | Pypper | 79.23% | 3.4 | 9.0 | 0.033 |
| heart-c | WRA | 73.96% | 2.4 | 9.7 | 0.019 |
| heart-c | Weka:JRip | 81.23% | 3.4 | 7.6 | 0.395 |
| heart-c | cost (c=0.437) | 76.26% | 28.2 | 80.6 | 0.052 |
| heart-c | m-estimate (m=16) | 75.61% | 13.3 | 55.9 | 0.052 |
| heart-c | m-estimate (m=2) | 78.56% | 25.8 | 78.7 | 0.047 |
| heart-c | m-estimate (m=22.466) | 76.32% | 9.6 | 40.7 | 0.042 |
| heart-c | m-estimate (m=32) | 76.97% | 8.8 | 38.1 | 0.047 |
| heart-c | m-estimate (m=4) | 78.24% | 21.3 | 69.2 | 0.049 |
| heart-c | m-estimate (m=8) | 76.29% | 16.6 | 63.3 | 0.049 |
| heart-c | relative cost (c_r=0.342) | 74.92% | 6.6 | 23.5 | 0.033 |
| heart-h | Accuracy | 76.53% | 20.1 | 58.2 | 0.040 |
| heart-h | Correlation | 73.13% | 10.0 | 39.3 | 0.032 |
| heart-h | F-measure (beta=0.5) | 76.85% | 13.3 | 51.6 | 0.051 |
| heart-h | Laplace | 77.90% | 30.0 | 82.4 | 0.050 |
| heart-h | Precision | 75.83% | 46.7 | 119.2 | 0.068 |
| heart-h | Pypper | 76.82% | 1.8 | 4.0 | 0.021 |
| heart-h | WRA | 75.53% | 1.7 | 6.4 | 0.018 |
| heart-h | Weka:JRip | 77.89% | 2.2 | 4.3 | 0.377 |
| heart-h | cost (c=0.437) | 75.54% | 22.8 | 65.7 | 0.042 |
| heart-h | m-estimate (m=16) | 77.87% | 10.2 | 37.9 | 0.047 |
| heart-h | m-estimate (m=2) | 78.61% | 22.7 | 68.1 | 0.050 |
| heart-h | m-estimate (m=22.466) | 75.48% | 7.2 | 28.7 | 0.039 |
| heart-h | m-estimate (m=32) | 77.89% | 5.3 | 28.3 | 0.035 |
| heart-h | m-estimate (m=4) | 74.46% | 18.4 | 59.8 | 0.052 |
| heart-h | m-estimate (m=8) | 76.86% | 13.3 | 50.3 | 0.050 |
| heart-h | relative cost (c_r=0.342) | 77.90% | 4.3 | 16.1 | 0.024 |
| heart-statlog | Accuracy | 73.33% | 18.2 | 50.0 | 0.030 |
| heart-statlog | Correlation | 76.67% | 10.7 | 39.1 | 0.033 |
| heart-statlog | F-measure (beta=0.5) | 75.56% | 14.3 | 45.3 | 0.046 |
| heart-statlog | Laplace | 74.07% | 29.6 | 81.6 | 0.045 |
| heart-statlog | Precision | 72.22% | 54.2 | 139.2 | 0.080 |
| heart-statlog | Pypper | 81.48% | 2.9 | 6.9 | 0.028 |
| heart-statlog | WRA | 75.93% | 2.3 | 9.0 | 0.017 |
| heart-statlog | Weka:JRip | 79.26% | 3.4 | 7.6 | 0.370 |
| heart-statlog | cost (c=0.437) | 73.70% | 23.5 | 63.9 | 0.035 |
| heart-statlog | m-estimate (m=16) | 77.04% | 11.3 | 41.9 | 0.040 |
| heart-statlog | m-estimate (m=2) | 74.44% | 24.1 | 69.2 | 0.040 |
| heart-statlog | m-estimate (m=22.466) | 74.81% | 8.8 | 34.5 | 0.038 |
| heart-statlog | m-estimate (m=32) | 78.89% | 7.5 | 32.1 | 0.034 |
| heart-statlog | m-estimate (m=4) | 77.41% | 19.5 | 61.8 | 0.043 |
| heart-statlog | m-estimate (m=8) | 80.00% | 15.3 | 53.5 | 0.044 |
| heart-statlog | relative cost (c_r=0.342) | 75.56% | 8.3 | 31.1 | 0.031 |
| hepatitis | Accuracy | 75.75% | 11.7 | 28.7 | 0.020 |
| hepatitis | Correlation | 73.79% | 6.7 | 20.0 | 0.016 |
| hepatitis | F-measure (beta=0.5) | 73.21% | 8.5 | 22.9 | 0.021 |
| hepatitis | Laplace | 73.83% | 12.6 | 27.5 | 0.016 |
| hepatitis | Precision | 73.21% | 18.2 | 39.6 | 0.022 |
| hepatitis | Pypper | 80.75% | 2.1 | 4.5 | 0.020 |
| hepatitis | WRA | 76.75% | 1.2 | 4.9 | 0.007 |
| hepatitis | Weka:JRip | 78.75% | 2.1 | 4.0 | 0.341 |
| hepatitis | cost (c=0.437) | 75.75% | 11.9 | 27.0 | 0.017 |
| hepatitis | m-estimate (m=16) | 77.75% | 5.6 | 16.5 | 0.017 |
| hepatitis | m-estimate (m=2) | 77.58% | 9.8 | 24.5 | 0.016 |
| hepatitis | m-estimate (m=22.466) | 78.38% | 5.4 | 16.8 | 0.015 |
| hepatitis | m-estimate (m=32) | 78.25% | 4.4 | 13.9 | 0.014 |
| hepatitis | m-estimate (m=4) | 76.42% | 8.1 | 21.6 | 0.015 |
| hepatitis | m-estimate (m=8) | 78.33% | 6.2 | 17.7 | 0.020 |
| hepatitis | relative cost (c_r=0.342) | 78.33% | 3.6 | 11.8 | 0.012 |
| hypothyroid | Accuracy | 98.99% | 13.4 | 35.7 | 0.086 |
| hypothyroid | Correlation | 98.94% | 8.8 | 24.5 | 0.073 |
| hypothyroid | F-measure (beta=0.5) | 98.83% | 13.9 | 42.9 | 0.111 |
| hypothyroid | Laplace | 98.28% | 33.2 | 107.4 | 0.176 |
| hypothyroid | Precision | 98.25% | 82.7 | 267.9 | 0.354 |
| hypothyroid | Pypper | 99.36% | 3.9 | 9.8 | 0.075 |
| hypothyroid | WRA | 97.08% | 2.1 | 7.7 | 0.045 |
| hypothyroid | Weka:JRip | 99.20% | 3.4 | 8.9 | 0.775 |
| hypothyroid | cost (c=0.437) | 98.78% | 17.0 | 48.2 | 0.098 |
| hypothyroid | m-estimate (m=16) | 99.13% | 10.9 | 39.2 | 0.111 |
| hypothyroid | m-estimate (m=2) | 98.67% | 23.0 | 81.0 | 0.134 |
| hypothyroid | m-estimate (m=22.466) | 99.05% | 10.5 | 40.2 | 0.108 |
| hypothyroid | m-estimate (m=32) | 98.91% | 8.3 | 31.4 | 0.095 |
| hypothyroid | m-estimate (m=4) | 98.86% | 18.8 | 71.0 | 0.117 |
| hypothyroid | m-estimate (m=8) | 98.81% | 14.1 | 52.9 | 0.115 |
| hypothyroid | relative cost (c_r=0.342) | 99.20% | 3.8 | 12.5 | 0.056 |
| ionosphere | Accuracy | 86.33% | 14.1 | 24.4 | 0.030 |
| ionosphere | Correlation | 87.47% | 10.2 | 18.8 | 0.024 |
| ionosphere | F-measure (beta=0.5) | 90.90% | 13.0 | 24.0 | 0.050 |
| ionosphere | Laplace | 91.47% | 15.7 | 25.9 | 0.023 |
| ionosphere | Precision | 92.32% | 19.4 | 33.0 | 0.033 |
| ionosphere | Pypper | 89.17% | 4.4 | 7.0 | 0.032 |
| ionosphere | WRA | 89.17% | 2.0 | 2.3 | 0.046 |
| ionosphere | Weka:JRip | 86.88% | 5.8 | 10.0 | 0.454 |
| ionosphere | cost (c=0.437) | 88.90% | 14.6 | 25.4 | 0.030 |
| ionosphere | m-estimate (m=16) | 90.59% | 9.8 | 20.2 | 0.052 |
| ionosphere | m-estimate (m=2) | 92.02% | 14.5 | 24.3 | 0.022 |
| ionosphere | m-estimate (m=22.466) | 90.31% | 9.2 | 20.7 | 0.048 |
| ionosphere | m-estimate (m=32) | 89.17% | 7.6 | 24.6 | 0.062 |
| ionosphere | m-estimate (m=4) | 92.02% | 12.8 | 22.6 | 0.023 |
| ionosphere | m-estimate (m=8) | 92.30% | 11.2 | 20.9 | 0.021 |
| ionosphere | relative cost (c_r=0.342) | 87.18% | 4.6 | 11.8 | 0.073 |
| iris | Accuracy | 92.00% | 4.4 | 9.1 | 0.007 |
| iris | Correlation | 93.33% | 3.7 | 8.9 | 0.009 |
| iris | F-measure (beta=0.5) | 92.00% | 4.8 | 10.3 | 0.012 |
| iris | Laplace | 92.00% | 7.1 | 13.6 | 0.011 |
| iris | Precision | 92.00% | 10.9 | 18.8 | 0.017 |
| iris | Pypper | 93.33% | 2.0 | 2.8 | 0.016 |
| iris | WRA | 93.33% | 2.3 | 3.9 | 0.010 |
| iris | Weka:JRip | 95.33% | 2.5 | 3.9 | 0.354 |
| iris | cost (c=0.437) | 92.00% | 4.6 | 9.6 | 0.009 |
| iris | m-estimate (m=16) | 92.67% | 3.8 | 8.1 | 0.009 |
| iris | m-estimate (m=2) | 92.67% | 5.6 | 11.1 | 0.010 |
| iris | m-estimate (m=22.466) | 92.67% | 3.7 | 8.0 | 0.010 |
| iris | m-estimate (m=32) | 92.67% | 3.7 | 7.9 | 0.010 |
| iris | m-estimate (m=4) | 92.67% | 5.1 | 11.1 | 0.010 |
| iris | m-estimate (m=8) | 92.00% | 3.8 | 7.7 | 0.011 |
| iris | relative cost (c_r=0.342) | 92.67% | 3.7 | 8.0 | 0.010 |
| kr-vs-kp | Accuracy | 96.93% | 41.2 | 192.3 | 0.167 |
| kr-vs-kp | Correlation | 98.25% | 9.7 | 27.0 | 0.050 |
| kr-vs-kp | F-measure (beta=0.5) | 98.12% | 13.4 | 43.7 | 0.077 |
| kr-vs-kp | Laplace | 99.28% | 40.7 | 139.6 | 0.139 |
| kr-vs-kp | Precision | 99.22% | 67.8 | 224.9 | 0.239 |
| kr-vs-kp | Pypper | 99.12% | 14.3 | 45.4 | 0.152 |
| kr-vs-kp | WRA | 94.34% | 3.0 | 8.1 | 0.029 |
| kr-vs-kp | Weka:JRip | 99.31% | 14.6 | 46.2 | 0.812 |
| kr-vs-kp | cost (c=0.437) | 96.75% | 51.1 | 235.4 | 0.198 |
| kr-vs-kp | m-estimate (m=16) | 99.50% | 21.5 | 67.3 | 0.093 |
| kr-vs-kp | m-estimate (m=2) | 99.28% | 32.1 | 108.5 | 0.119 |
| kr-vs-kp | m-estimate (m=22.466) | 99.44% | 20.0 | 64.6 | 0.102 |
| kr-vs-kp | m-estimate (m=32) | 99.47% | 17.6 | 67.6 | 0.099 |
| kr-vs-kp | m-estimate (m=4) | 99.31% | 29.5 | 98.8 | 0.123 |
| kr-vs-kp | m-estimate (m=8) | 99.34% | 26.7 | 95.9 | 0.113 |
| kr-vs-kp | relative cost (c_r=0.342) | 95.21% | 6.0 | 20.0 | 0.045 |
| lymph | Accuracy | 80.24% | 12.7 | 27.1 | 0.026 |
| lymph | Correlation | 86.52% | 8.5 | 21.3 | 0.020 |
| lymph | F-measure (beta=0.5) | 83.86% | 11.2 | 26.4 | 0.022 |
| lymph | Laplace | 76.33% | 19.6 | 42.0 | 0.027 |
| lymph | Precision | 77.81% | 29.3 | 62.4 | 0.038 |
| lymph | Pypper | 77.05% | 3.1 | 6.8 | 0.029 |
| lymph | WRA | 85.00% | 5.6 | 17.7 | 0.020 |
| lymph | Weka:JRip | 79.86% | 4.6 | 9.2 | 0.351 |
| lymph | cost (c=0.437) | 83.67% | 14.9 | 34.4 | 0.022 |
| lymph | m-estimate (m=16) | 79.14% | 9.2 | 25.2 | 0.027 |
| lymph | m-estimate (m=2) | 79.00% | 15.4 | 32.6 | 0.022 |
| lymph | m-estimate (m=22.466) | 78.48% | 8.5 | 22.5 | 0.026 |
| lymph | m-estimate (m=32) | 79.81% | 7.7 | 22.2 | 0.024 |
| lymph | m-estimate (m=4) | 80.48% | 13.0 | 29.2 | 0.021 |
| lymph | m-estimate (m=8) | 78.43% | 10.1 | 25.2 | 0.022 |
| lymph | relative cost (c_r=0.342) | 83.05% | 6.5 | 18.8 | 0.019 |
| molecular-biology_promoters | Accuracy | 83.36% | 4.1 | 9.7 | 0.010 |
| molecular-biology_promoters | Correlation | 81.27% | 3.2 | 9.7 | 0.011 |
| molecular-biology_promoters | F-measure (beta=0.5) | 78.64% | 5.4 | 12.0 | 0.015 |
| molecular-biology_promoters | Laplace | 73.45% | 6.7 | 13.0 | 0.014 |
| molecular-biology_promoters | Precision | 77.36% | 13.9 | 27.5 | 0.030 |
| molecular-biology_promoters | Pypper | 83.18% | 2.1 | 3.8 | 0.016 |
| molecular-biology_promoters | WRA | 80.45% | 2.3 | 9.0 | 0.011 |
| molecular-biology_promoters | Weka:JRip | 77.45% | 2.5 | 5.4 | 0.349 |
| molecular-biology_promoters | cost (c=0.437) | 78.36% | 5.2 | 12.3 | 0.013 |
| molecular-biology_promoters | m-estimate (m=16) | 85.00% | 4.1 | 10.4 | 0.012 |
| molecular-biology_promoters | m-estimate (m=2) | 72.64% | 6.2 | 12.3 | 0.015 |
| molecular-biology_promoters | m-estimate (m=22.466) | 82.27% | 3.7 | 10.6 | 0.011 |
| molecular-biology_promoters | m-estimate (m=32) | 84.09% | 3.6 | 10.6 | 0.012 |
| molecular-biology_promoters | m-estimate (m=4) | 83.09% | 5.3 | 11.5 | 0.013 |
| molecular-biology_promoters | m-estimate (m=8) | 79.27% | 4.6 | 10.7 | 0.013 |
| molecular-biology_promoters | relative cost (c_r=0.342) | 80.18% | 3.0 | 9.5 | 0.011 |
| monks-problems-1 | Accuracy | 80.45% | 1.6 | 2.2 | 0.004 |
| monks-problems-1 | Correlation | 77.00% | 1.4 | 1.8 | 0.004 |
| monks-problems-1 | F-measure (beta=0.5) | 80.78% | 2.4 | 5.3 | 0.008 |
| monks-problems-1 | Laplace | 100.00% | 10.6 | 33.0 | 0.019 |
| monks-problems-1 | Precision | 99.82% | 17.3 | 53.3 | 0.031 |
| monks-problems-1 | Pypper | 95.30% | 3.6 | 10.3 | 0.025 |
| monks-problems-1 | WRA | 77.00% | 1.4 | 1.8 | 0.005 |
| monks-problems-1 | Weka:JRip | 93.04% | 3.7 | 10.4 | 0.364 |
| monks-problems-1 | cost (c=0.437) | 84.60% | 3.3 | 7.8 | 0.007 |
| monks-problems-1 | m-estimate (m=16) | 92.79% | 6.9 | 22.0 | 0.019 |
| monks-problems-1 | m-estimate (m=2) | 100.00% | 11.2 | 36.6 | 0.025 |
| monks-problems-1 | m-estimate (m=22.466) | 92.48% | 5.4 | 19.0 | 0.014 |
| monks-problems-1 | m-estimate (m=32) | 88.17% | 3.8 | 11.6 | 0.011 |
| monks-problems-1 | m-estimate (m=4) | 100.00% | 11.0 | 36.0 | 0.023 |
| monks-problems-1 | m-estimate (m=8) | 98.75% | 10.2 | 34.4 | 0.023 |
| monks-problems-1 | relative cost (c_r=0.342) | 88.87% | 5.0 | 14.8 | 0.013 |
| monks-problems-2 | Accuracy | 65.89% | 90.3 | 513.5 | 0.191 |
| monks-problems-2 | Correlation | 87.19% | 8.5 | 54.8 | 0.023 |
| monks-problems-2 | F-measure (beta=0.5) | 67.24% | 0.9 | 3.9 | 0.007 |
| monks-problems-2 | Laplace | 67.23% | 84.6 | 491.3 | 0.164 |
| monks-problems-2 | Precision | 63.07% | 127.3 | 741.5 | 0.288 |
| monks-problems-2 | Pypper | 72.22% | 4.4 | 24.2 | 0.058 |
| monks-problems-2 | WRA | 65.74% | 0.9 | 3.6 | 0.005 |
| monks-problems-2 | Weka:JRip | 80.57% | 7.6 | 44.6 | 0.398 |
| monks-problems-2 | cost (c=0.437) | 68.39% | 105.2 | 615.8 | 0.223 |
| monks-problems-2 | m-estimate (m=16) | 82.87% | 14.3 | 84.9 | 0.038 |
| monks-problems-2 | m-estimate (m=2) | 74.22% | 58.6 | 357.5 | 0.158 |
| monks-problems-2 | m-estimate (m=22.466) | 77.05% | 11.8 | 69.8 | 0.032 |
| monks-problems-2 | m-estimate (m=32) | 80.37% | 9.3 | 55.6 | 0.028 |
| monks-problems-2 | m-estimate (m=4) | 74.88% | 34.6 | 207.4 | 0.088 |
| monks-problems-2 | m-estimate (m=8) | 81.21% | 21.8 | 132.1 | 0.056 |
| monks-problems-2 | relative cost (c_r=0.342) | 64.58% | 18.0 | 97.1 | 0.047 |
| monks-problems-3 | Accuracy | 95.86% | 2.3 | 3.6 | 0.007 |
| monks-problems-3 | Correlation | 95.68% | 2.4 | 5.4 | 0.009 |
| monks-problems-3 | F-measure (beta=0.5) | 96.40% | 2.8 | 5.7 | 0.009 |
| monks-problems-3 | Laplace | 95.67% | 6.5 | 13.9 | 0.011 |
| monks-problems-3 | Precision | 97.10% | 9.3 | 18.9 | 0.016 |
| monks-problems-3 | Pypper | 98.92% | 3.0 | 5.0 | 0.020 |
| monks-problems-3 | WRA | 96.40% | 2.0 | 2.0 | 0.007 |
| monks-problems-3 | Weka:JRip | 98.92% | 3.0 | 5.0 | 0.352 |
| monks-problems-3 | cost (c=0.437) | 96.04% | 2.2 | 3.2 | 0.007 |
| monks-problems-3 | m-estimate (m=16) | 98.92% | 3.0 | 5.0 | 0.008 |
| monks-problems-3 | m-estimate (m=2) | 97.48% | 5.9 | 13.3 | 0.013 |
| monks-problems-3 | m-estimate (m=22.466) | 98.92% | 3.0 | 5.0 | 0.008 |
| monks-problems-3 | m-estimate (m=32) | 98.92% | 3.0 | 5.0 | 0.008 |
| monks-problems-3 | m-estimate (m=4) | 98.38% | 4.5 | 9.3 | 0.010 |
| monks-problems-3 | m-estimate (m=8) | 98.56% | 3.2 | 6.0 | 0.009 |
| monks-problems-3 | relative cost (c_r=0.342) | 96.40% | 2.0 | 2.0 | 0.006 |
| mushroom | Accuracy | 100.00% | 7.8 | 13.6 | 0.065 |
| mushroom | Correlation | 100.00% | 6.9 | 10.9 | 0.063 |
| mushroom | F-measure (beta=0.5) | 100.00% | 10.9 | 12.9 | 0.079 |
| mushroom | Laplace | 100.00% | 10.9 | 12.9 | 0.071 |
| mushroom | Precision | 100.00% | 15.0 | 18.0 | 0.099 |
| mushroom | Pypper | 99.96% | 5.2 | 10.6 | 0.091 |
| mushroom | WRA | 99.96% | 3.0 | 11.1 | 0.045 |
| mushroom | Weka:JRip | 99.98% | 6.0 | 10.1 | 1.593 |
| mushroom | cost (c=0.437) | 100.00% | 7.8 | 13.6 | 0.067 |
| mushroom | m-estimate (m=16) | 100.00% | 10.9 | 12.9 | 0.090 |
| mushroom | m-estimate (m=2) | 100.00% | 10.9 | 12.9 | 0.086 |
| mushroom | m-estimate (m=22.466) | 100.00% | 10.9 | 12.9 | 0.087 |
| mushroom | m-estimate (m=32) | 100.00% | 10.9 | 12.9 | 0.080 |
| mushroom | m-estimate (m=4) | 100.00% | 10.9 | 12.9 | 0.086 |
| mushroom | m-estimate (m=8) | 100.00% | 10.9 | 12.9 | 0.096 |
| mushroom | relative cost (c_r=0.342) | 99.89% | 4.0 | 12.1 | 0.069 |
| primary-tumor | Accuracy | 33.90% | 73.3 | 264.2 | 0.202 |
| primary-tumor | Correlation | 40.41% | 59.7 | 295.7 | 0.200 |
| primary-tumor | F-measure (beta=0.5) | 39.53% | 67.9 | 279.9 | 0.250 |
| primary-tumor | Laplace | 34.81% | 89.3 | 351.5 | 0.204 |
| primary-tumor | Precision | 35.98% | 117.2 | 455.9 | 0.277 |
| primary-tumor | Pypper | 38.66% | 8.6 | 24.8 | 0.128 |
| primary-tumor | WRA | 41.91% | 9.9 | 40.2 | 0.077 |
| primary-tumor | Weka:JRip | 36.00% | 6.4 | 21.9 | 0.418 |
| primary-tumor | cost (c=0.437) | 32.74% | 76.2 | 273.3 | 0.187 |
| primary-tumor | m-estimate (m=16) | 41.90% | 39.5 | 179.3 | 0.185 |
| primary-tumor | m-estimate (m=2) | 38.05% | 77.8 | 323.1 | 0.235 |
| primary-tumor | m-estimate (m=22.466) | 43.97% | 34.6 | 151.8 | 0.162 |
| primary-tumor | m-estimate (m=32) | 41.31% | 28.3 | 127.0 | 0.149 |
| primary-tumor | m-estimate (m=4) | 36.58% | 62.8 | 269.5 | 0.223 |
| primary-tumor | m-estimate (m=8) | 41.02% | 49.1 | 215.5 | 0.198 |
| primary-tumor | relative cost (c_r=0.342) | 43.09% | 20.3 | 84.8 | 0.117 |
| segment | Accuracy | 90.30% | 164.4 | 560.3 | 0.545 |
| segment | Correlation | 95.32% | 26.5 | 104.8 | 0.165 |
| segment | F-measure (beta=0.5) | 93.98% | 39.6 | 144.8 | 0.255 |
| segment | Laplace | 93.81% | 90.9 | 298.9 | 0.277 |
| segment | Precision | 93.16% | 170.1 | 536.4 | 0.537 |
| segment | Pypper | 94.81% | 16.1 | 51.3 | 0.180 |
| segment | WRA | 90.69% | 7.8 | 29.8 | 0.112 |
| segment | Weka:JRip | 94.59% | 21.5 | 65.0 | 1.354 |
| segment | cost (c=0.437) | 90.35% | 173.5 | 588.2 | 0.541 |
| segment | m-estimate (m=16) | 93.85% | 38.7 | 138.5 | 0.252 |
| segment | m-estimate (m=2) | 94.20% | 70.5 | 234.4 | 0.257 |
| segment | m-estimate (m=22.466) | 94.29% | 35.9 | 132.9 | 0.219 |
| segment | m-estimate (m=32) | 94.03% | 31.5 | 114.4 | 0.225 |
| segment | m-estimate (m=4) | 93.94% | 57.9 | 203.4 | 0.225 |
| segment | m-estimate (m=8) | 93.72% | 49.4 | 181.8 | 0.224 |
| segment | relative cost (c_r=0.342) | 91.65% | 17.9 | 63.7 | 0.188 |
| solar-flare | Accuracy | 74.20% | 20.2 | 47.9 | 0.062 |
| solar-flare | Correlation | 74.49% | 5.2 | 13.8 | 0.057 |
| solar-flare | F-measure (beta=0.5) | 71.58% | 12.0 | 48.9 | 0.102 |
| solar-flare | Laplace | 74.39% | 43.4 | 136.4 | 0.126 |
| solar-flare | Precision | 74.48% | 67.9 | 215.4 | 0.164 |
| solar-flare | Pypper | 74.39% | 5.5 | 10.7 | 0.078 |
| solar-flare | WRA | 73.46% | 3.0 | 4.2 | 0.040 |
| solar-flare | Weka:JRip | 71.21% | 6.5 | 16.9 | 0.524 |
| solar-flare | cost (c=0.437) | 73.83% | 20.7 | 48.7 | 0.060 |
| solar-flare | m-estimate (m=16) | 73.17% | 10.9 | 38.4 | 0.092 |
| solar-flare | m-estimate (m=2) | 73.45% | 38.3 | 141.3 | 0.158 |
| solar-flare | m-estimate (m=22.466) | 73.27% | 9.4 | 34.5 | 0.083 |
| solar-flare | m-estimate (m=32) | 73.64% | 7.3 | 29.8 | 0.076 |
| solar-flare | m-estimate (m=4) | 72.70% | 28.0 | 112.9 | 0.142 |
| solar-flare | m-estimate (m=8) | 73.17% | 19.9 | 70.8 | 0.121 |
| solar-flare | relative cost (c_r=0.342) | 72.99% | 3.0 | 5.8 | 0.037 |
| sonar | Accuracy | 72.62% | 14.8 | 31.4 | 0.094 |
| sonar | Correlation | 72.64% | 8.3 | 22.6 | 0.077 |
| sonar | F-measure (beta=0.5) | 74.05% | 11.8 | 26.7 | 0.059 |
| sonar | Laplace | 64.93% | 25.3 | 42.7 | 0.085 |
| sonar | Precision | 61.02% | 49.4 | 87.3 | 0.165 |
| sonar | Pypper | 75.05% | 3.0 | 5.8 | 0.035 |
| sonar | WRA | 75.57% | 3.5 | 24.7 | 0.162 |
| sonar | Weka:JRip | 72.60% | 4.7 | 10.1 | 0.455 |
| sonar | cost (c=0.437) | 66.33% | 18.5 | 36.3 | 0.080 |
| sonar | m-estimate (m=16) | 74.50% | 8.6 | 22.6 | 0.042 |
| sonar | m-estimate (m=2) | 67.81% | 19.5 | 33.6 | 0.062 |
| sonar | m-estimate (m=22.466) | 73.00% | 7.8 | 21.7 | 0.038 |
| sonar | m-estimate (m=32) | 68.79% | 6.3 | 20.9 | 0.059 |
| sonar | m-estimate (m=4) | 65.31% | 14.0 | 26.4 | 0.048 |
| sonar | m-estimate (m=8) | 75.00% | 11.4 | 24.5 | 0.049 |
| sonar | relative cost (c_r=0.342) | 74.57% | 5.8 | 29.5 | 0.075 |
| soybean | Accuracy | 90.05% | 55.0 | 124.3 | 0.134 |
| soybean | Correlation | 93.13% | 33.4 | 73.4 | 0.085 |
| soybean | F-measure (beta=0.5) | 91.38% | 46.6 | 101.3 | 0.137 |
| soybean | Laplace | 89.76% | 62.6 | 142.5 | 0.113 |
| soybean | Precision | 88.45% | 96.4 | 213.7 | 0.162 |
| soybean | Pypper | 91.07% | 25.8 | 49.2 | 0.187 |
| soybean | WRA | 93.86% | 20.5 | 52.7 | 0.086 |
| soybean | Weka:JRip | 77.77% | 25.5 | 46.0 | 0.600 |
| soybean | cost (c=0.437) | 89.77% | 60.1 | 139.1 | 0.124 |
| soybean | m-estimate (m=16) | 92.99% | 37.5 | 81.3 | 0.115 |
| soybean | m-estimate (m=2) | 90.78% | 52.7 | 119.7 | 0.122 |
| soybean | m-estimate (m=22.466) | 92.84% | 34.9 | 73.9 | 0.108 |
| soybean | m-estimate (m=32) | 92.11% | 32.9 | 71.4 | 0.104 |
| soybean | m-estimate (m=4) | 91.96% | 47.8 | 108.7 | 0.116 |
| soybean | m-estimate (m=8) | 90.93% | 42.7 | 95.6 | 0.112 |
| soybean | relative cost (c_r=0.342) | 93.27% | 26.4 | 64.7 | 0.103 |
| tic-tac-toe | Accuracy | 87.48% | 76.8 | 320.9 | 0.178 |
| tic-tac-toe | Correlation | 88.85% | 12.8 | 62.7 | 0.039 |
| tic-tac-toe | F-measure (beta=0.5) | 82.46% | 10.8 | 38.6 | 0.029 |
| tic-tac-toe | Laplace | 96.97% | 20.9 | 78.0 | 0.045 |
| tic-tac-toe | Precision | 97.39% | 25.5 | 95.3 | 0.056 |
| tic-tac-toe | Pypper | 98.02% | 8.6 | 27.5 | 0.063 |
| tic-tac-toe | WRA | 69.00% | 1.5 | 3.0 | 0.009 |
| tic-tac-toe | Weka:JRip | 98.12% | 8.4 | 26.6 | 0.468 |
| tic-tac-toe | cost (c=0.437) | 86.02% | 94.6 | 393.4 | 0.175 |
| tic-tac-toe | m-estimate (m=16) | 97.29% | 11.4 | 40.5 | 0.032 |
| tic-tac-toe | m-estimate (m=2) | 96.45% | 15.6 | 60.7 | 0.041 |
| tic-tac-toe | m-estimate (m=22.466) | 97.60% | 10.2 | 34.8 | 0.030 |
| tic-tac-toe | m-estimate (m=32) | 97.50% | 9.9 | 33.0 | 0.028 |
| tic-tac-toe | m-estimate (m=4) | 96.97% | 15.0 | 59.2 | 0.036 |
| tic-tac-toe | m-estimate (m=8) | 96.66% | 13.5 | 49.5 | 0.037 |
| tic-tac-toe | relative cost (c_r=0.342) | 92.48% | 8.1 | 25.2 | 0.024 |
| vehicle | Accuracy | 66.79% | 157.3 | 499.2 | 0.487 |
| vehicle | Correlation | 70.45% | 26.3 | 125.6 | 0.266 |
| vehicle | F-measure (beta=0.5) | 70.58% | 43.2 | 168.5 | 0.434 |
| vehicle | Laplace | 70.71% | 137.0 | 411.8 | 0.318 |
| vehicle | Precision | 72.36% | 219.6 | 633.1 | 0.517 |
| vehicle | Pypper | 67.74% | 14.0 | 39.3 | 0.132 |
| vehicle | WRA | 60.04% | 5.4 | 31.5 | 0.140 |
| vehicle | Weka:JRip | 68.45% | 13.9 | 38.4 | 0.720 |
| vehicle | cost (c=0.437) | 66.90% | 179.2 | 556.1 | 0.508 |
| vehicle | m-estimate (m=16) | 72.46% | 37.8 | 169.0 | 0.590 |
| vehicle | m-estimate (m=2) | 71.53% | 102.2 | 325.9 | 0.334 |
| vehicle | m-estimate (m=22.466) | 69.87% | 31.3 | 174.1 | 0.471 |
| vehicle | m-estimate (m=32) | 69.51% | 24.6 | 137.7 | 0.434 |
| vehicle | m-estimate (m=4) | 71.52% | 81.2 | 278.1 | 0.340 |
| vehicle | m-estimate (m=8) | 70.35% | 60.1 | 272.6 | 0.520 |
| vehicle | relative cost (c_r=0.342) | 70.70% | 16.8 | 93.5 | 0.312 |
| vote | Accuracy | 94.50% | 3.7 | 10.7 | 0.009 |
| vote | Correlation | 93.81% | 3.4 | 10.2 | 0.010 |
| vote | F-measure (beta=0.5) | 94.71% | 4.6 | 13.8 | 0.016 |
| vote | Laplace | 94.02% | 10.4 | 38.5 | 0.020 |
| vote | Precision | 94.25% | 17.6 | 63.0 | 0.031 |
| vote | Pypper | 95.18% | 1.8 | 3.5 | 0.019 |
| vote | WRA | 94.04% | 2.7 | 8.5 | 0.008 |
| vote | Weka:JRip | 95.42% | 3.1 | 7.6 | 0.353 |
| vote | cost (c=0.437) | 94.50% | 3.7 | 10.7 | 0.008 |
| vote | m-estimate (m=16) | 93.57% | 5.6 | 18.3 | 0.015 |
| vote | m-estimate (m=2) | 94.03% | 8.5 | 31.1 | 0.021 |
| vote | m-estimate (m=22.466) | 94.95% | 4.9 | 16.3 | 0.013 |
| vote | m-estimate (m=32) | 94.73% | 4.9 | 16.0 | 0.012 |
| vote | m-estimate (m=4) | 93.80% | 6.5 | 24.6 | 0.020 |
| vote | m-estimate (m=8) | 94.03% | 5.1 | 17.4 | 0.015 |
| vote | relative cost (c_r=0.342) | 93.81% | 2.9 | 9.2 | 0.008 |
| vowel | Accuracy | 80.10% | 181.8 | 508.3 | 0.327 |
| vowel | Correlation | 81.62% | 72.0 | 262.7 | 0.244 |
| vowel | F-measure (beta=0.5) | 82.32% | 108.7 | 333.8 | 0.275 |
| vowel | Laplace | 80.10% | 195.3 | 528.2 | 0.340 |
| vowel | Precision | 79.09% | 277.4 | 749.7 | 0.478 |
| vowel | Pypper | 74.85% | 48.1 | 173.6 | 0.416 |
| vowel | WRA | 62.12% | 23.3 | 132.3 | 0.189 |
| vowel | Weka:JRip | 76.06% | 49.4 | 162.8 | 0.983 |
| vowel | cost (c=0.437) | 79.90% | 212.7 | 585.4 | 0.349 |
| vowel | m-estimate (m=16) | 80.10% | 93.0 | 303.9 | 0.260 |
| vowel | m-estimate (m=2) | 81.11% | 165.4 | 458.1 | 0.301 |
| vowel | m-estimate (m=22.466) | 81.21% | 82.7 | 282.3 | 0.258 |
| vowel | m-estimate (m=32) | 79.80% | 72.4 | 261.8 | 0.251 |
| vowel | m-estimate (m=4) | 82.02% | 142.5 | 406.2 | 0.307 |
| vowel | m-estimate (m=8) | 81.11% | 117.3 | 350.5 | 0.284 |
| vowel | relative cost (c_r=0.342) | 72.12% | 45.9 | 202.9 | 0.269 |
| wine | Accuracy | 95.00% | 4.2 | 8.4 | 0.009 |
| wine | Correlation | 94.97% | 4.3 | 8.6 | 0.009 |
| wine | F-measure (beta=0.5) | 92.75% | 5.7 | 10.2 | 0.012 |
| wine | Laplace | 92.19% | 6.7 | 11.3 | 0.010 |
| wine | Precision | 92.19% | 11.1 | 19.1 | 0.016 |
| wine | Pypper | 92.09% | 2.8 | 4.3 | 0.022 |
| wine | WRA | 96.63% | 3.5 | 7.3 | 0.008 |
| wine | Weka:JRip | 94.41% | 2.7 | 5.1 | 0.360 |
| wine | cost (c=0.437) | 96.67% | 4.8 | 9.4 | 0.008 |
| wine | m-estimate (m=16) | 94.38% | 5.7 | 10.0 | 0.011 |
| wine | m-estimate (m=2) | 92.19% | 6.4 | 11.0 | 0.012 |
| wine | m-estimate (m=22.466) | 94.38% | 5.1 | 9.3 | 0.010 |
| wine | m-estimate (m=32) | 94.38% | 4.9 | 9.0 | 0.009 |
| wine | m-estimate (m=4) | 91.63% | 6.2 | 10.7 | 0.011 |
| wine | m-estimate (m=8) | 94.38% | 5.9 | 10.3 | 0.012 |
| wine | relative cost (c_r=0.342) | 94.97% | 4.0 | 8.2 | 0.009 |
| zoo | Accuracy | 92.00% | 9.8 | 22.8 | 0.017 |
| zoo | Correlation | 93.00% | 7.7 | 16.7 | 0.012 |
| zoo | F-measure (beta=0.5) | 93.00% | 8.8 | 19.5 | 0.014 |
| zoo | Laplace | 92.00% | 9.4 | 21.1 | 0.013 |
| zoo | Precision | 92.00% | 12.4 | 27.9 | 0.018 |
| zoo | Pypper | 89.09% | 5.4 | 8.1 | 0.035 |
| zoo | WRA | 96.00% | 6.2 | 13.2 | 0.011 |
| zoo | Weka:JRip | 89.00% | 6.6 | 11.1 | 0.325 |
| zoo | cost (c=0.437) | 92.00% | 9.8 | 22.8 | 0.014 |
| zoo | m-estimate (m=16) | 93.00% | 8.2 | 18.4 | 0.015 |
| zoo | m-estimate (m=2) | 93.00% | 8.7 | 19.4 | 0.015 |
| zoo | m-estimate (m=22.466) | 92.00% | 7.7 | 16.7 | 0.011 |
| zoo | m-estimate (m=32) | 93.00% | 7.5 | 16.2 | 0.013 |
| zoo | m-estimate (m=4) | 93.00% | 8.5 | 19.2 | 0.015 |
| zoo | m-estimate (m=8) | 93.00% | 8.3 | 18.6 | 0.013 |
| zoo | relative cost (c_r=0.342) | 93.00% | 7.9 | 17.3 | 0.012 |

