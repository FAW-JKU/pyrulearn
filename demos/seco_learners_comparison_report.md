# Separate-and-conquer rule learners compared

## Setup

**Full run** (29 datasets). Quick sanity check instead: `python demos/seco_learners_comparison.py` with no arguments.

pyrulearn's separate-and-conquer rule learners compared with each other
and with two external baselines, on 18 binary and
11 multi-class datasets from `pyrulearn.experiments.catalog`.
All of them learn one rule at a time and remove the examples it covers;
they differ in how a rule is searched for, when it stops, and whether and
how the rule set is pruned. Every learner runs with its default settings,
including its default way of handling several classes.

The datasets have mostly symbolic attributes (at least half nominal), the
kind these learners were designed for -- numeric attributes and their
discretization are the topic of the numeric-discretization demo. Artificial
concepts (the monks problems, m-of-n, Hayes-Roth, LED) and datasets with
many classes (left for the multi-class demo) are not included.

- **CN2** -- Beam search (width 5) with the Laplace estimate; CN2's likelihood-ratio significance test filters the rules a search can return. Learns rules for every class (one-vs-rest rule set).
- **PFoil** -- FOIL: greedy search that adds the condition with the highest FOIL gain; Quinlan's encoding-length restriction stops rules that cost more bits than the examples they explain. Rules for every class.
- **PFossil** -- FOSSIL: hill climbing on the correlation between rule and class; rules below a correlation of 0.3 are dropped. Rules for every class.
- **Pypper** -- pyrulearn's RIPPER: grow on two thirds of the data, prune on the rest (IREP*), then optimize the rule set (replace/revise). Rules for the classes from least to most frequent, the most frequent one as default.
- **Opus(WRAcc)** -- Opus: in every covering step the rule with the highest weighted relative accuracy among all conjunctions of up to 3 conditions, found by exhaustive branch-and-bound search (OPUS). Rules for every class.
- **PyLORD** -- A simplified LORD: for every training example, the best rule covering it (greedy m-estimate search, then pruning); a test example is classified by the best rule that covers it. Many overlapping rules.
- **Weka:JRip** -- Weka's RIPPER, run as a subprocess on the same binarized data. Fit time includes the JVM start.
- **JavaLORD** -- The reference LORD implementation (Java), run as a subprocess on the same binarized data, and, in this demo only, on a single thread: by default it uses half the machine's cores, which no other learner here does.

AQR, the AQ baseline CN2 was designed to improve on, and Opus with the
Laplace estimate were checked separately and left out -- see "Why AQR and
Opus(Laplace) aren't in the main comparison".
Across 4 small datasets (3-fold), AQR was less accurate than CN2 (mean accuracy 71.38% vs. 78.27%), learned rules 7.0x as long (17.4 vs. 2.5 conditions per rule; 493.7 vs. 89.8 conditions in total), and took 14x as long per fit (5.5s vs. 0.39s, mean) -- requiring every rule to be consistent overfits noisy data, and on larger datasets with numeric attributes the fit times would dominate the whole comparison. Opus with the Laplace estimate (at most 3 conditions per rule), which finds the rule CN2's beam search only approximates, was less accurate than CN2 (mean accuracy 77.12%), learned 1.3x as many rules (44.9 vs. 34.2, mean) and took 134x as long per fit (52.1s, mean). Webb (1993) found classifiers that optimize the Laplace estimate exactly often less accurate than those of a heuristic search that fails to -- the result behind Quinlan & Cameron-Jones's 'oversearching'. The main comparison runs Opus with WRAcc instead.

Protocol: 10-fold stratified cross-validation (5-fold for the large adult, connect-4)
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(discretizer=TreeDiscretizer(max_intervals=8))`), the test fold binarized
against that same `DataSpec` -- every learner sees the identical binary
features per fold (as vertical bitmaps, the library's default representation; the learners
that need the dense matrix convert them). Each fit is capped at 900s; a
time-out or an error counts as a failure (and as last in the ranking),
not as the end of the run. The cap is deliberately kept for the large
datasets too: which learners scale to tens of thousands of examples is
part of the result.

Every learner runs single-threaded; the reference Java LORD, which by default uses half the machine's cores, is limited to one thread here (`thread_count=1`, for this comparison only). Fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: test accuracy, number of rules and conditions, fit time.

![accuracy vs. rule-set complexity](seco_learners_comparison_plots/seco_learners_comparison_accuracy.png)

![fit time per dataset](seco_learners_comparison_plots/seco_learners_comparison_fit_time.png)

![critical-difference diagram (accuracy)](seco_learners_comparison_plots/seco_learners_comparison_cd.png)

## Why AQR and Opus(Laplace) aren't in the main comparison

Across 4 small datasets (3-fold), AQR was less accurate than CN2 (mean accuracy 71.38% vs. 78.27%), learned rules 7.0x as long (17.4 vs. 2.5 conditions per rule; 493.7 vs. 89.8 conditions in total), and took 14x as long per fit (5.5s vs. 0.39s, mean) -- requiring every rule to be consistent overfits noisy data, and on larger datasets with numeric attributes the fit times would dominate the whole comparison. Opus with the Laplace estimate (at most 3 conditions per rule), which finds the rule CN2's beam search only approximates, was less accurate than CN2 (mean accuracy 77.12%), learned 1.3x as many rules (44.9 vs. 34.2, mean) and took 134x as long per fit (52.1s, mean). Webb (1993) found classifiers that optimize the Laplace estimate exactly often less accurate than those of a heuristic search that fails to -- the result behind Quinlan & Cameron-Jones's 'oversearching'. The main comparison runs Opus with WRAcc instead.

| dataset | learner | accuracy | n_rules | n_conditions | conds_per_rule | fit_time |
|---|---|---|---|---|---|---|
| breast-cancer | AQR | 57.70% | 27.3 | 489.0 | 17.7 | 2.719 |
| breast-cancer | CN2 | 71.68% | 25.7 | 67.3 | 2.6 | 0.207 |
| breast-cancer | Opus(Laplace) | 66.44% | 58.7 | 171.7 | 2.9 | 32.203 |
| credit-approval | AQR | 75.94% | 37.0 | 1155.0 | 31.3 | 15.560 |
| credit-approval | CN2 | 84.06% | 61.7 | 172.0 | 2.8 | 1.017 |
| credit-approval | Opus(Laplace) | 83.33% | 78.0 | 227.3 | 2.9 | 156.400 |
| heart-statlog | AQR | 71.85% | 17.3 | 274.3 | 15.3 | 2.994 |
| heart-statlog | CN2 | 79.26% | 33.0 | 87.0 | 2.6 | 0.260 |
| heart-statlog | Opus(Laplace) | 80.00% | 28.7 | 84.7 | 3.0 | 18.469 |
| hepatitis | AQR | 80.03% | 10.3 | 56.3 | 5.4 | 0.536 |
| hepatitis | CN2 | 78.08% | 16.3 | 33.0 | 2.0 | 0.075 |
| hepatitis | Opus(Laplace) | 78.70% | 14.3 | 35.3 | 2.5 | 1.462 |

## Summary by learner (mean across the 27 datasets and their folds)

Without the two large datasets (adult, connect-4), where most learners hit the time cap -- see the section on them below. On these 27 datasets every learner has a result for (almost) every fold, so the means compare like with like.

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | 82.35% | 66.0 | 229.3 | 3.1 | 2.75 | 4.8 | 3.74 | 0 |
| PFoil | 82.22% | 56.5 | 242.5 | 3.6 | 0.16 | 2.8 | 4.13 | 0 |
| PyLORD | 82.64% | 133.4 | 694.4 | 4.0 | 2.26 | 4.2 | 4.17 | 0 |
| PFossil | 81.93% | 13.1 | 46.9 | 4.0 | 0.04 | 4.5 | 4.28 | 0 |
| Pypper | 81.32% | 4.8 | 16.1 | 2.8 | 0.09 | 4.0 | 4.46 | 0 |
| JavaLORD | 81.80% | 129.0 | 617.5 | 3.7 | 8.41 | 2.2 | 4.57 | 0 |
| Weka:JRip | 80.97% | 5.6 | 18.5 | 2.8 | 0.83 | 2.0 | 4.83 | 0 |
| Opus(WRAcc) | 78.13% | 9.8 | 25.7 | 2.6 | 18.16 | 2.5 | 5.81 | 0 |

Sorted by mean rank. `wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied for last. Means over successful fits only; `failures` counts the fits that timed out or raised.

### Small datasets only (17, fewer than 1,000 examples)

`SPECT`, `analcatdata_dmft`, `anneal`, `breast-cancer`, `colic`, `credit-approval`, `cylinder-bands`, `dresses-sales`, `heart-c`, `heart-h`, `heart-statlog`, `hepatitis`, `lymph`, `molecular-biology_promoters`, `tic-tac-toe`, `vote`, `zoo`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | 79.39% | 43.1 | 119.7 | 2.7 | 3.43 | 4.5 | 3.38 | 0 |
| PFoil | 79.03% | 43.9 | 152.1 | 3.1 | 0.08 | 1.5 | 4.21 | 0 |
| PyLORD | 79.54% | 92.2 | 399.6 | 3.5 | 0.67 | 2.0 | 4.24 | 0 |
| PFossil | 78.47% | 11.8 | 39.3 | 3.9 | 0.03 | 2.5 | 4.59 | 0 |
| Pypper | 77.57% | 3.0 | 7.3 | 2.4 | 0.04 | 1.0 | 4.68 | 0 |
| Weka:JRip | 77.29% | 3.6 | 8.5 | 2.4 | 0.51 | 1.0 | 4.91 | 0 |
| Opus(WRAcc) | 76.68% | 9.5 | 24.7 | 2.6 | 15.86 | 2.5 | 5.00 | 0 |
| JavaLORD | 78.17% | 87.0 | 356.2 | 3.2 | 0.65 | 2.0 | 5.00 | 0 |

### Medium datasets only (10, 1,000 to 10,000 examples)

`car`, `cmc`, `credit-g`, `dna`, `hypothyroid`, `kr-vs-kp`, `mushroom`, `sick`, `solar-flare`, `splice`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| PFossil | 87.81% | 15.2 | 59.9 | 4.2 | 0.07 | 2.0 | 3.75 | 0 |
| JavaLORD | 87.96% | 200.6 | 1061.6 | 4.5 | 21.59 | 0.2 | 3.85 | 0 |
| PFoil | 87.66% | 78.0 | 396.3 | 4.5 | 0.28 | 1.2 | 4.00 | 0 |
| PyLORD | 87.92% | 203.6 | 1195.6 | 4.9 | 4.98 | 2.2 | 4.05 | 0 |
| Pypper | 87.69% | 7.9 | 31.1 | 3.4 | 0.17 | 3.0 | 4.10 | 0 |
| CN2 | 87.39% | 104.9 | 415.6 | 3.8 | 1.58 | 0.2 | 4.35 | 0 |
| Weka:JRip | 87.22% | 8.8 | 35.5 | 3.5 | 1.38 | 1.0 | 4.70 | 0 |
| Opus(WRAcc) | 80.58% | 10.2 | 27.3 | 2.7 | 22.07 | 0.0 | 7.20 | 0 |

## Mean accuracy by target type

| learner | binary | multi-class |
|---|--:|--:|
| CN2 | 83.66% | 80.12% |
| PFoil | 83.47% | 80.10% |
| PyLORD | 84.25% | 79.92% |
| PFossil | 82.88% | 80.32% |
| Pypper | 83.01% | 78.45% |
| JavaLORD | 83.35% | 79.17% |
| Weka:JRip | 82.61% | 78.17% |
| Opus(WRAcc) | 81.00% | 73.25% |

## The large datasets: adult, connect-4

5-fold cross-validation, with the same 900s cap per fit as everywhere else -- which learners get through tens of thousands of examples at all is the result here. `folds` counts the folds that finished; accuracy, size and fit time are means over those folds only.

| dataset | learner | folds | accuracy | n_rules | n_conditions | fit_time (s) |
|---|---|--:|--:|--:|--:|--:|
| adult | CN2 | 5/5 | 78.80% | 2255.0 | 10899.2 | 213.04 |
| adult | JavaLORD | 0/5 | -- | -- | -- | -- |
| adult | Opus(WRAcc) | 5/5 | 79.46% | 14.4 | 43.2 | 269.50 |
| adult | PFoil | 5/5 | 85.88% | 1695.8 | 16910.0 | 53.66 |
| adult | PFossil | 5/5 | 82.22% | 9.2 | 56.6 | 0.49 |
| adult | PyLORD | 0/5 | -- | -- | -- | -- |
| adult | Pypper | 5/5 | 84.22% | 13.2 | 79.2 | 2.96 |
| adult | Weka:JRip | 5/5 | 84.87% | 22.8 | 138.0 | 68.29 |
| connect-4 | CN2 | 5/5 | 74.62% | 8126.8 | 40187.2 | 392.94 |
| connect-4 | JavaLORD | 0/5 | -- | -- | -- | -- |
| connect-4 | Opus(WRAcc) | 5/5 | 66.34% | 26.2 | 78.6 | 313.08 |
| connect-4 | PFoil | 5/5 | 81.83% | 3531.2 | 34317.4 | 131.03 |
| connect-4 | PFossil | 5/5 | 65.83% | 0.0 | 0.0 | 0.24 |
| connect-4 | PyLORD | 5/5 | 81.88% | 11759.0 | 93973.6 | 754.92 |
| connect-4 | Pypper | 5/5 | 76.86% | 73.0 | 631.0 | 23.39 |
| connect-4 | Weka:JRip | 5/5 | 76.79% | 98.2 | 855.8 | 613.98 |

## Per-dataset results, per learner (mean across folds)

| dataset | learner | accuracy | n_rules | n_conditions | fit_time |
|---|---|---|---|---|---|
| SPECT | CN2 | 79.74% | 26.8 | 105.6 | 0.341 |
| SPECT | JavaLORD | 84.30% | 60.9 | 253.9 | 0.277 |
| SPECT | Opus(WRAcc) | 78.65% | 9.8 | 20.6 | 0.395 |
| SPECT | PFoil | 82.05% | 25.8 | 82.1 | 0.035 |
| SPECT | PFossil | 78.68% | 9.7 | 41.1 | 0.017 |
| SPECT | PyLORD | 83.93% | 61.7 | 276.2 | 0.481 |
| SPECT | Pypper | 79.00% | 0.9 | 5.0 | 0.023 |
| SPECT | Weka:JRip | 80.87% | 1.0 | 5.5 | 0.434 |
| analcatdata_dmft | CN2 | 19.56% | 100.8 | 386.6 | 0.921 |
| analcatdata_dmft | JavaLORD | 19.08% | 470.8 | 2781.4 | 0.949 |
| analcatdata_dmft | Opus(WRAcc) | 21.69% | 24.6 | 71.1 | 2.672 |
| analcatdata_dmft | PFoil | 19.07% | 253.7 | 973.8 | 0.543 |
| analcatdata_dmft | PFossil | 19.95% | 0.3 | 3.3 | 0.023 |
| analcatdata_dmft | PyLORD | 18.58% | 480.7 | 3038.0 | 2.961 |
| analcatdata_dmft | Pypper | 18.94% | 1.0 | 2.8 | 0.075 |
| analcatdata_dmft | Weka:JRip | 19.45% | 1.0 | 3.5 | 0.472 |
| anneal | CN2 | 98.77% | 33.9 | 88.0 | 0.228 |
| anneal | JavaLORD | 90.87% | 28.9 | 46.1 | 0.957 |
| anneal | Opus(WRAcc) | 97.44% | 11.9 | 23.7 | 0.523 |
| anneal | PFoil | 97.21% | 25.9 | 68.7 | 0.056 |
| anneal | PFossil | 99.22% | 17.8 | 30.5 | 0.034 |
| anneal | PyLORD | 98.44% | 47.0 | 144.5 | 0.797 |
| anneal | Pypper | 94.20% | 8.2 | 20.1 | 0.102 |
| anneal | Weka:JRip | 89.98% | 6.7 | 10.9 | 0.605 |
| breast-cancer | CN2 | 69.93% | 35.0 | 91.0 | 0.500 |
| breast-cancer | JavaLORD | 71.37% | 92.7 | 417.9 | 0.385 |
| breast-cancer | Opus(WRAcc) | 71.34% | 7.9 | 22.9 | 2.168 |
| breast-cancer | PFoil | 72.02% | 43.2 | 156.8 | 0.064 |
| breast-cancer | PFossil | 69.26% | 7.4 | 30.0 | 0.021 |
| breast-cancer | PyLORD | 72.04% | 93.9 | 447.1 | 0.488 |
| breast-cancer | Pypper | 74.84% | 1.0 | 2.0 | 0.018 |
| breast-cancer | Weka:JRip | 71.37% | 1.3 | 2.5 | 0.443 |
| car | CN2 | 94.91% | 82.1 | 440.2 | 0.406 |
| car | JavaLORD | 95.83% | 104.1 | 637.1 | 1.374 |
| car | Opus(WRAcc) | 81.94% | 8.0 | 19.0 | 0.159 |
| car | PFoil | 96.70% | 75.4 | 449.1 | 0.248 |
| car | PFossil | 93.87% | 16.2 | 59.1 | 0.048 |
| car | PyLORD | 95.37% | 106.2 | 697.2 | 1.977 |
| car | Pypper | 93.63% | 17.1 | 86.4 | 0.280 |
| car | Weka:JRip | 94.33% | 18.2 | 90.1 | 0.637 |
| cmc | CN2 | 54.11% | 183.2 | 777.2 | 1.963 |
| cmc | JavaLORD | 52.76% | 651.9 | 4472.5 | 2.811 |
| cmc | Opus(WRAcc) | 48.13% | 16.5 | 47.9 | 4.749 |
| cmc | PFoil | 51.40% | 211.0 | 1266.9 | 0.641 |
| cmc | PFossil | 52.61% | 2.1 | 9.2 | 0.017 |
| cmc | PyLORD | 52.08% | 662.7 | 5357.8 | 5.688 |
| cmc | Pypper | 54.79% | 4.1 | 12.6 | 0.127 |
| cmc | Weka:JRip | 51.94% | 3.6 | 14.5 | 0.547 |
| colic | CN2 | 82.91% | 65.0 | 161.6 | 0.571 |
| colic | JavaLORD | 78.26% | 74.5 | 205.9 | 0.634 |
| colic | Opus(WRAcc) | 86.70% | 9.5 | 22.8 | 3.001 |
| colic | PFoil | 84.52% | 38.5 | 110.0 | 0.062 |
| colic | PFossil | 82.89% | 19.8 | 50.7 | 0.032 |
| colic | PyLORD | 83.45% | 85.0 | 250.7 | 0.462 |
| colic | Pypper | 85.08% | 1.8 | 4.0 | 0.030 |
| colic | Weka:JRip | 84.50% | 3.4 | 8.4 | 0.533 |
| credit-approval | CN2 | 86.23% | 92.1 | 270.0 | 1.736 |
| credit-approval | JavaLORD | 83.62% | 126.5 | 499.1 | 1.552 |
| credit-approval | Opus(WRAcc) | 84.78% | 8.6 | 24.8 | 4.918 |
| credit-approval | PFoil | 84.49% | 57.5 | 220.2 | 0.112 |
| credit-approval | PFossil | 82.17% | 8.8 | 38.9 | 0.026 |
| credit-approval | PyLORD | 84.20% | 127.8 | 525.7 | 1.030 |
| credit-approval | Pypper | 83.91% | 3.8 | 11.0 | 0.053 |
| credit-approval | Weka:JRip | 84.78% | 4.2 | 10.1 | 0.719 |
| credit-g | CN2 | 73.70% | 187.5 | 539.3 | 6.768 |
| credit-g | JavaLORD | 75.70% | 257.7 | 982.3 | 2.198 |
| credit-g | Opus(WRAcc) | 72.60% | 10.7 | 31.7 | 35.484 |
| credit-g | PFoil | 73.10% | 94.4 | 450.0 | 0.265 |
| credit-g | PFossil | 73.20% | 3.4 | 29.3 | 0.025 |
| credit-g | PyLORD | 75.80% | 257.9 | 994.7 | 1.806 |
| credit-g | Pypper | 71.10% | 2.8 | 10.0 | 0.066 |
| credit-g | Weka:JRip | 71.20% | 3.6 | 12.9 | 0.705 |
| cylinder-bands | CN2 | 71.67% | 88.3 | 140.6 | 39.106 |
| cylinder-bands | JavaLORD | 73.15% | 130.0 | 303.4 | 1.517 |
| cylinder-bands | Opus(WRAcc) | 69.44% | 7.9 | 22.6 | 168.030 |
| cylinder-bands | PFoil | 75.00% | 48.8 | 180.6 | 0.131 |
| cylinder-bands | PFossil | 70.19% | 9.4 | 64.3 | 0.042 |
| cylinder-bands | PyLORD | 78.52% | 127.9 | 305.8 | 0.955 |
| cylinder-bands | Pypper | 65.37% | 4.1 | 13.3 | 0.077 |
| cylinder-bands | Weka:JRip | 65.00% | 6.1 | 16.8 | 0.790 |
| dna | CN2 | 89.11% | 153.6 | 675.6 | 1.739 |
| dna | JavaLORD | 91.78% | 323.0 | 1471.8 | 59.812 |
| dna | Opus(WRAcc) | 67.73% | 11.2 | 28.6 | 54.690 |
| dna | PFoil | 92.50% | 89.0 | 469.4 | 0.424 |
| dna | PFossil | 93.69% | 27.8 | 137.4 | 0.144 |
| dna | PyLORD | 91.78% | 320.8 | 1463.3 | 7.844 |
| dna | Pypper | 92.22% | 13.2 | 68.2 | 0.354 |
| dna | Weka:JRip | 91.96% | 17.3 | 97.4 | 3.006 |
| dresses-sales | CN2 | 62.40% | 24.2 | 37.1 | 12.271 |
| dresses-sales | JavaLORD | 58.80% | 171.3 | 514.3 | 1.026 |
| dresses-sales | Opus(WRAcc) | 59.20% | 10.4 | 30.0 | 76.110 |
| dresses-sales | PFoil | 58.20% | 67.1 | 243.2 | 0.143 |
| dresses-sales | PFossil | 59.40% | 2.2 | 20.4 | 0.021 |
| dresses-sales | PyLORD | 60.60% | 190.3 | 607.5 | 0.860 |
| dresses-sales | Pypper | 61.20% | 1.1 | 1.2 | 0.033 |
| dresses-sales | Weka:JRip | 61.00% | 1.9 | 3.1 | 0.537 |
| heart-c | CN2 | 82.22% | 51.7 | 148.4 | 0.696 |
| heart-c | JavaLORD | 80.57% | 60.9 | 215.9 | 0.449 |
| heart-c | Opus(WRAcc) | 76.89% | 8.2 | 23.3 | 1.997 |
| heart-c | PFoil | 78.92% | 29.6 | 100.0 | 0.048 |
| heart-c | PFossil | 79.54% | 18.2 | 69.4 | 0.038 |
| heart-c | PyLORD | 80.57% | 62.1 | 223.6 | 0.390 |
| heart-c | Pypper | 81.55% | 3.0 | 7.6 | 0.037 |
| heart-c | Weka:JRip | 81.23% | 3.4 | 7.6 | 0.471 |
| heart-h | CN2 | 79.25% | 50.1 | 143.8 | 0.858 |
| heart-h | JavaLORD | 75.51% | 60.9 | 219.0 | 0.449 |
| heart-h | Opus(WRAcc) | 77.21% | 8.0 | 23.3 | 1.053 |
| heart-h | PFoil | 78.93% | 30.4 | 97.3 | 0.045 |
| heart-h | PFossil | 81.28% | 17.2 | 52.9 | 0.026 |
| heart-h | PyLORD | 77.89% | 62.6 | 244.8 | 0.351 |
| heart-h | Pypper | 78.54% | 1.9 | 4.5 | 0.024 |
| heart-h | Weka:JRip | 77.89% | 2.2 | 4.3 | 0.456 |
| heart-statlog | CN2 | 78.52% | 47.7 | 134.1 | 0.509 |
| heart-statlog | JavaLORD | 78.89% | 53.4 | 184.3 | 0.395 |
| heart-statlog | Opus(WRAcc) | 80.74% | 8.2 | 23.4 | 1.687 |
| heart-statlog | PFoil | 77.04% | 26.5 | 84.7 | 0.045 |
| heart-statlog | PFossil | 80.74% | 17.6 | 60.3 | 0.037 |
| heart-statlog | PyLORD | 78.89% | 53.4 | 187.2 | 0.385 |
| heart-statlog | Pypper | 79.26% | 2.3 | 5.7 | 0.037 |
| heart-statlog | Weka:JRip | 79.26% | 3.4 | 7.6 | 0.468 |
| hepatitis | CN2 | 82.67% | 22.2 | 51.1 | 0.189 |
| hepatitis | JavaLORD | 81.38% | 28.2 | 66.6 | 0.236 |
| hepatitis | Opus(WRAcc) | 80.75% | 7.4 | 20.3 | 0.467 |
| hepatitis | PFoil | 77.58% | 17.1 | 37.0 | 0.020 |
| hepatitis | PFossil | 78.75% | 15.0 | 38.6 | 0.021 |
| hepatitis | PyLORD | 83.38% | 29.8 | 73.0 | 0.113 |
| hepatitis | Pypper | 79.42% | 1.7 | 3.4 | 0.041 |
| hepatitis | Weka:JRip | 78.75% | 2.1 | 4.0 | 0.440 |
| hypothyroid | CN2 | 98.89% | 34.3 | 109.5 | 0.339 |
| hypothyroid | JavaLORD | 98.33% | 38.0 | 137.4 | 5.524 |
| hypothyroid | Opus(WRAcc) | 95.63% | 10.1 | 24.6 | 1.752 |
| hypothyroid | PFoil | 99.05% | 31.5 | 100.2 | 0.116 |
| hypothyroid | PFossil | 98.89% | 21.5 | 55.8 | 0.084 |
| hypothyroid | PyLORD | 98.81% | 47.7 | 162.8 | 3.859 |
| hypothyroid | Pypper | 99.28% | 3.3 | 8.9 | 0.096 |
| hypothyroid | Weka:JRip | 99.20% | 3.4 | 8.9 | 0.916 |
| kr-vs-kp | CN2 | 99.34% | 68.6 | 295.9 | 0.435 |
| kr-vs-kp | JavaLORD | 99.41% | 47.5 | 224.6 | 6.131 |
| kr-vs-kp | Opus(WRAcc) | 93.96% | 8.0 | 20.0 | 1.016 |
| kr-vs-kp | PFoil | 99.34% | 40.8 | 178.8 | 0.156 |
| kr-vs-kp | PFossil | 99.37% | 14.5 | 56.7 | 0.061 |
| kr-vs-kp | PyLORD | 99.44% | 49.7 | 259.4 | 3.797 |
| kr-vs-kp | Pypper | 99.09% | 13.6 | 44.0 | 0.163 |
| kr-vs-kp | Weka:JRip | 99.31% | 14.6 | 46.2 | 0.958 |
| lymph | CN2 | 83.95% | 18.5 | 50.2 | 0.072 |
| lymph | JavaLORD | 81.05% | 26.9 | 72.9 | 0.237 |
| lymph | Opus(WRAcc) | 74.38% | 8.0 | 19.8 | 0.972 |
| lymph | PFoil | 82.57% | 17.1 | 40.5 | 0.026 |
| lymph | PFossil | 81.71% | 16.2 | 45.0 | 0.025 |
| lymph | PyLORD | 81.05% | 26.8 | 73.1 | 0.166 |
| lymph | Pypper | 73.71% | 2.9 | 5.9 | 0.033 |
| lymph | Weka:JRip | 79.86% | 4.6 | 9.2 | 0.401 |
| molecular-biology_promoters | CN2 | 83.00% | 11.9 | 22.7 | 0.072 |
| molecular-biology_promoters | JavaLORD | 86.82% | 16.9 | 33.0 | 0.354 |
| molecular-biology_promoters | Opus(WRAcc) | 77.64% | 4.7 | 12.2 | 4.246 |
| molecular-biology_promoters | PFoil | 86.09% | 9.0 | 18.6 | 0.018 |
| molecular-biology_promoters | PFossil | 82.27% | 7.4 | 19.7 | 0.018 |
| molecular-biology_promoters | PyLORD | 86.00% | 16.4 | 31.9 | 0.175 |
| molecular-biology_promoters | Pypper | 83.18% | 2.1 | 3.8 | 0.063 |
| molecular-biology_promoters | Weka:JRip | 77.45% | 2.5 | 5.4 | 0.492 |
| mushroom | CN2 | 100.00% | 23.0 | 30.0 | 0.140 |
| mushroom | JavaLORD | 100.00% | 23.0 | 38.7 | 44.316 |
| mushroom | Opus(WRAcc) | 98.58% | 4.3 | 12.6 | 0.763 |
| mushroom | PFoil | 100.00% | 11.5 | 24.7 | 0.106 |
| mushroom | PFossil | 99.95% | 12.3 | 23.0 | 0.110 |
| mushroom | PyLORD | 100.00% | 23.0 | 38.7 | 7.238 |
| mushroom | Pypper | 99.96% | 5.2 | 10.6 | 0.123 |
| mushroom | Weka:JRip | 99.98% | 6.0 | 10.1 | 2.043 |
| sick | CN2 | 97.91% | 83.8 | 294.4 | 1.209 |
| sick | JavaLORD | 97.85% | 96.2 | 402.4 | 8.928 |
| sick | Opus(WRAcc) | 96.66% | 10.5 | 27.4 | 3.814 |
| sick | PFoil | 97.99% | 63.5 | 241.2 | 0.267 |
| sick | PFossil | 98.30% | 16.4 | 59.4 | 0.079 |
| sick | PyLORD | 97.83% | 103.9 | 531.4 | 5.727 |
| sick | Pypper | 98.22% | 3.5 | 11.9 | 0.080 |
| sick | Weka:JRip | 98.22% | 5.3 | 18.1 | 1.069 |
| solar-flare | CN2 | 73.82% | 112.1 | 504.7 | 1.396 |
| solar-flare | JavaLORD | 73.26% | 209.1 | 1171.2 | 0.933 |
| solar-flare | Opus(WRAcc) | 74.77% | 11.2 | 27.4 | 0.866 |
| solar-flare | PFoil | 73.73% | 85.2 | 403.4 | 0.202 |
| solar-flare | PFossil | 74.02% | 14.5 | 62.5 | 0.050 |
| solar-flare | PyLORD | 73.45% | 209.2 | 1377.7 | 3.184 |
| solar-flare | Pypper | 74.77% | 4.9 | 8.3 | 0.104 |
| solar-flare | Weka:JRip | 71.21% | 6.5 | 16.9 | 0.579 |
| splice | CN2 | 92.07% | 120.9 | 489.4 | 1.444 |
| splice | JavaLORD | 94.70% | 255.3 | 1077.6 | 83.904 |
| splice | Opus(WRAcc) | 75.77% | 11.8 | 33.8 | 117.367 |
| splice | PFoil | 92.76% | 78.2 | 379.4 | 0.422 |
| splice | PFossil | 94.23% | 23.7 | 106.8 | 0.132 |
| splice | PyLORD | 94.61% | 254.4 | 1073.4 | 8.642 |
| splice | Pypper | 93.82% | 11.4 | 49.6 | 0.302 |
| splice | Weka:JRip | 94.80% | 9.7 | 39.4 | 3.350 |
| tic-tac-toe | CN2 | 98.33% | 28.6 | 98.8 | 0.141 |
| tic-tac-toe | JavaLORD | 98.44% | 32.6 | 119.6 | 1.239 |
| tic-tac-toe | Opus(WRAcc) | 77.04% | 10.9 | 28.9 | 1.213 |
| tic-tac-toe | PFoil | 99.06% | 23.2 | 83.9 | 0.042 |
| tic-tac-toe | PFossil | 97.71% | 14.7 | 52.5 | 0.031 |
| tic-tac-toe | PyLORD | 95.93% | 53.9 | 222.2 | 1.289 |
| tic-tac-toe | Pypper | 96.25% | 7.5 | 23.1 | 0.056 |
| tic-tac-toe | Weka:JRip | 98.12% | 8.4 | 26.6 | 0.571 |
| vote | CN2 | 94.49% | 26.3 | 84.9 | 0.107 |
| vote | JavaLORD | 92.87% | 33.7 | 102.0 | 0.282 |
| vote | Opus(WRAcc) | 94.74% | 8.3 | 16.2 | 0.050 |
| vote | PFoil | 94.71% | 23.3 | 67.8 | 0.032 |
| vote | PFossil | 95.18% | 10.2 | 29.2 | 0.016 |
| vote | PyLORD | 93.79% | 38.7 | 121.4 | 0.352 |
| vote | Pypper | 95.18% | 1.7 | 3.3 | 0.020 |
| vote | Weka:JRip | 95.42% | 3.1 | 7.6 | 0.461 |
| zoo | CN2 | 96.00% | 8.9 | 19.8 | 0.027 |
| zoo | JavaLORD | 94.00% | 9.1 | 20.3 | 0.136 |
| zoo | Opus(WRAcc) | 95.00% | 7.0 | 14.0 | 0.038 |
| zoo | PFoil | 96.00% | 9.5 | 20.3 | 0.011 |
| zoo | PFossil | 95.00% | 8.9 | 21.0 | 0.011 |
| zoo | PyLORD | 95.00% | 9.1 | 20.3 | 0.067 |
| zoo | Pypper | 89.09% | 5.3 | 7.8 | 0.040 |
| zoo | Weka:JRip | 89.00% | 6.6 | 11.1 | 0.353 |

