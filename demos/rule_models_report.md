# Rule models: rule sets, cascades, decision lists

## Setup

**Full run.** Quick sanity check instead: `python demos/rule_models.py`.

A rule learner's covering loop can build different kinds of models from
the same search. This demo compares them, for CN2, PFossil and Pypper as
base learners:

- **rule set** -- one-vs-rest: a rule set per class, each learned against
  all other classes; a conflict between covering rules of different
  classes resolved by **best rule** (the covering rule with the best
  Laplace estimate, i.e. the rules sorted by it, an ordered rule set),
  **voting** (one vote per covering rule), **micro distribution voting**
  (their class counts summed, CN2's own) or **macro distribution voting**
  (their class proportions summed, every rule weighing the same) -- four
  ways of predicting with the same rules.
- **cascade** -- ordered one-vs-rest: the classes one after the other,
  each against those not yet handled, from the **rarest** class (Ripper's
  order) or the **most frequent** one; the last class is the default.
- **decision list** -- one covering loop over all classes, first match
  wins, the default the majority of what no rule covers. The class of a
  rule is chosen **per refinement** (by the heuristic, at every added
  condition; CN2's ordered mode), **per rule** (the best of one complete
  search per class) or **per seed example** (the class of a random
  uncovered example, whose generalizations the search explores; AQ's seed
  covering).

On two-class data (the second part), a cascade is a concept for one class
with the other as the default; the configurations there are a concept for
the **minority** class and one for the **majority** class, the **rule set**
of both classes with its four combiners, and the three **decision lists** of rules of either class.

Multi-class datasets, with their numbers of classes: `balance-scale` (3), `cmc` (3), `dna` (3), `hayes-roth` (3), `iris` (3), `splice` (3), `wine` (3), `car` (4), `lymph` (4), `vehicle` (4), `anneal` (5), `glass` (6), `solar-flare` (6), `segment` (7), `zoo` (7), `ecoli` (8), `led24` (10), `yeast` (10), `optdigits` (10), `texture` (11), `vowel` (11), `kropt` (18), `soybean` (19), `primary-tumor` (21), `audiology` (24), `letter` (26). Two-class datasets: `SPECT`, `Titanic`, `banknote-authentication`, `blood-transfusion-service-center`, `breast-cancer`, `breast-w`, `churn`, `climate-model-simulation-crashes`, `colic`, `compas-two-years`, `credit-approval`, `credit-g`, `cylinder-bands`, `diabetes`, `dresses-sales`, `heart-c`, `heart-h`, `heart-statlog`, `heloc`, `hepatitis`, `ilpd`, `ionosphere`, `kc1`, `kc2`, `kr-vs-kp`, `mofn-3-7-10`, `molecular-biology_promoters`, `monks-problems-1`, `monks-problems-2`, `monks-problems-3`, `mushroom`, `ozone-level-8hr`, `pc1`, `pc3`, `pc4`, `phoneme`, `qsar-biodeg`, `sick`, `sonar`, `spambase`, `tic-tac-toe`, `vote`, `wdbc`, `wilt`. Protocol:
10-fold stratified cross-validation, one `DataSpec` per training fold
(`TreeDiscretizer(max_intervals=8)`). Each fit is capped at
900s (1800s for the large `kropt` and `letter`); a time-out fails the whole cross-validation of that
learner on that dataset. Training time is measured inside the worker process.

## Multi-class datasets

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a model's rules together; the four rule-set variants share their rules and training time.

### CN2

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| rule set (micro distribution voting) | 80.86% | 3.94 | 265.7 | 1657.7 | 5.21 | 0 |
| decision list (per rule) | 80.91% | 3.98 | 98.4 | 618.3 | 26.39 | 0 |
| decision list (per refinement) | 80.62% | 4.17 | 108.0 | 629.7 | 9.68 | 0 |
| rule set (best rule) | 80.89% | 4.33 | 265.7 | 1657.7 | 5.21 | 0 |
| rule set (macro distribution voting) | 80.58% | 5.00 | 265.7 | 1657.7 | 5.21 | 0 |
| decision list (per seed example) | 79.48% | 5.13 | 83.5 | 579.0 | 2.36 | 0 |
| cascade (most frequent first) | 79.22% | 5.85 | 168.2 | 961.4 | 2.88 | 0 |
| rule set (voting) | 79.74% | 5.92 | 265.7 | 1657.7 | 5.21 | 0 |
| cascade (rarest first) | 78.31% | 6.67 | 156.7 | 859.6 | 2.73 | 0 |

### PFossil

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| rule set (best rule) | 78.99% | 4.00 | 37.3 | 194.4 | 0.20 | 0 |
| rule set (macro distribution voting) | 78.85% | 4.02 | 37.3 | 194.4 | 0.20 | 0 |
| decision list (per rule) | 78.59% | 4.42 | 33.6 | 162.4 | 1.03 | 0 |
| decision list (per refinement) | 78.32% | 4.48 | 23.3 | 81.6 | 0.22 | 0 |
| cascade (rarest first) | 78.01% | 4.69 | 32.0 | 153.2 | 0.16 | 0 |
| cascade (most frequent first) | 76.29% | 5.48 | 32.0 | 142.2 | 0.13 | 0 |
| rule set (micro distribution voting) | 77.54% | 5.62 | 37.3 | 194.4 | 0.20 | 0 |
| rule set (voting) | 77.68% | 5.67 | 37.3 | 194.4 | 0.20 | 0 |
| decision list (per seed example) | 75.79% | 6.62 | 28.8 | 149.4 | 0.30 | 0 |

### Pypper

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| cascade (rarest first) | 79.48% | 3.90 | 42.5 | 203.5 | 1.27 | 0 |
| cascade (most frequent first) | 78.23% | 4.10 | 41.6 | 215.1 | 1.13 | 0 |
| rule set (macro distribution voting) | 77.65% | 4.35 | 44.8 | 254.5 | 1.61 | 0 |
| rule set (best rule) | 77.66% | 4.46 | 44.8 | 254.5 | 1.61 | 0 |
| decision list (per seed example) | 78.17% | 5.35 | 81.5 | 678.2 | 1.59 | 0 |
| decision list (per rule) | 76.51% | 5.37 | 19.3 | 70.7 | 1.74 | 0 |
| decision list (per refinement) | 76.38% | 5.60 | 26.1 | 116.4 | 0.57 | 0 |
| rule set (micro distribution voting) | 76.85% | 5.88 | 44.8 | 254.5 | 1.61 | 0 |
| rule set (voting) | 76.62% | 6.00 | 44.8 | 254.5 | 1.61 | 0 |

### All base learners together

Each kind of model's mean rank, averaged over the 3 base learners (ranked within each base learner on each dataset, as in the tables above), and its accuracy averaged over them.

| model | accuracy | mean rank |
|---|--:|--:|
| rule set (best rule) | 79.18% | 4.26 |
| rule set (macro distribution voting) | 79.03% | 4.46 |
| decision list (per rule) | 78.67% | 4.59 |
| decision list (per refinement) | 78.44% | 4.75 |
| cascade (rarest first) | 78.60% | 5.09 |
| cascade (most frequent first) | 77.91% | 5.14 |
| rule set (micro distribution voting) | 78.42% | 5.15 |
| decision list (per seed example) | 77.81% | 5.70 |
| rule set (voting) | 78.01% | 5.87 |

Friedman test over the 78 (dataset, base learner) pairs: p = 0.0016 (9 kinds of model). Nemenyi critical difference at 5%: 1.36 -- two kinds whose mean ranks differ by less are not significantly different; the diagram joins them by a bar. The pairs of one dataset aren't independent of each other, so the test is on the optimistic side.

![critical-difference diagram, multiclass](rule_models_plots/rule_models_cd_multiclass.png)

## Two-class datasets

A concept for one class, the other the default; the rule set of both classes; the decision lists of rules of either class.

### CN2

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| rule set (best rule) | 85.84% | 3.19 | 100.0 | 330.7 | 2.79 | 0 |
| rule set (macro distribution voting) | 85.67% | 3.49 | 100.0 | 330.7 | 2.79 | 0 |
| rule set (micro distribution voting) | 85.55% | 3.65 | 100.0 | 330.7 | 2.79 | 0 |
| rule set (voting) | 85.17% | 4.51 | 100.0 | 330.7 | 2.79 | 0 |
| decision list (per rule) | 85.14% | 4.52 | 40.2 | 110.3 | 0.75 | 0 |
| decision list (per refinement) | 84.59% | 5.40 | 34.2 | 88.8 | 0.50 | 0 |
| decision list (per seed example) | 83.23% | 6.16 | 44.9 | 165.4 | 1.78 | 0 |
| concept (majority class) | 83.11% | 6.55 | 42.3 | 128.1 | 1.25 | 0 |
| concept (minority class) | 79.77% | 7.53 | 57.7 | 202.6 | 1.55 | 0 |

### PFossil

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| rule set (best rule) | 83.33% | 3.66 | 9.1 | 34.3 | 0.02 | 0 |
| rule set (macro distribution voting) | 83.17% | 3.76 | 9.1 | 34.3 | 0.02 | 0 |
| rule set (voting) | 82.65% | 4.45 | 9.1 | 34.3 | 0.02 | 0 |
| rule set (micro distribution voting) | 82.47% | 4.82 | 9.1 | 34.3 | 0.02 | 0 |
| decision list (per rule) | 82.28% | 5.43 | 4.2 | 18.7 | 0.02 | 0 |
| decision list (per refinement) | 82.00% | 5.48 | 3.9 | 15.9 | 0.02 | 0 |
| concept (minority class) | 81.58% | 5.49 | 4.3 | 17.5 | 0.01 | 0 |
| decision list (per seed example) | 81.07% | 5.75 | 6.3 | 24.7 | 0.50 | 0 |
| concept (majority class) | 75.52% | 6.16 | 4.8 | 16.8 | 0.01 | 0 |

### Pypper

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| rule set (macro distribution voting) | 85.39% | 3.66 | 9.1 | 28.7 | 0.17 | 0 |
| rule set (best rule) | 85.38% | 3.67 | 9.1 | 28.7 | 0.17 | 0 |
| rule set (voting) | 85.15% | 4.43 | 9.1 | 28.7 | 0.17 | 0 |
| rule set (micro distribution voting) | 85.14% | 4.60 | 9.1 | 28.7 | 0.17 | 0 |
| decision list (per rule) | 85.15% | 5.14 | 10.1 | 29.5 | 0.08 | 0 |
| concept (minority class) | 85.03% | 5.30 | 4.4 | 15.3 | 0.08 | 0 |
| decision list (per refinement) | 84.83% | 5.56 | 7.4 | 28.1 | 0.07 | 0 |
| concept (majority class) | 82.85% | 6.18 | 4.6 | 13.4 | 0.09 | 0 |
| decision list (per seed example) | 84.00% | 6.47 | 12.6 | 42.1 | 0.07 | 0 |

### All base learners together

Each kind of model's mean rank, averaged over the 3 base learners (ranked within each base learner on each dataset, as in the tables above), and its accuracy averaged over them.

| model | accuracy | mean rank |
|---|--:|--:|
| rule set (best rule) | 84.85% | 3.51 |
| rule set (macro distribution voting) | 84.74% | 3.64 |
| rule set (micro distribution voting) | 84.39% | 4.36 |
| rule set (voting) | 84.32% | 4.47 |
| decision list (per rule) | 84.19% | 5.03 |
| decision list (per refinement) | 83.81% | 5.48 |
| concept (minority class) | 82.12% | 6.11 |
| decision list (per seed example) | 82.77% | 6.12 |
| concept (majority class) | 80.50% | 6.30 |

Friedman test over the 132 (dataset, base learner) pairs: p = 5.8e-33 (9 kinds of model). Nemenyi critical difference at 5%: 1.05 -- two kinds whose mean ranks differ by less are not significantly different; the diagram joins them by a bar. The pairs of one dataset aren't independent of each other, so the test is on the optimistic side.

![critical-difference diagram, twoclass](rule_models_plots/rule_models_cd_twoclass.png)

## Decision lists against rule sets, over the number of classes

Per dataset and base learner, the accuracy of a decision list minus that of the rule set resolved by the best rule, in percentage points (positive: the decision list is better), and of the best decision list minus the best rule set, averaged over the datasets with the given numbers of classes and the three base learners.

| classes | pairs | list (per refinement) | list (per rule) | list (per seed example) | best list - best set |
|---|--:|--:|--:|--:|--:|
| 2 | 132 | -1.05 | -0.66 | -2.09 | -0.13 |
| 3 | 21 | -1.08 | -0.33 | -3.03 | -0.11 |
| 4-5 | 12 | +0.52 | +0.31 | -0.67 | +1.77 |
| 6-9 | 15 | +0.26 | -0.55 | -1.62 | +0.34 |
| 10-26 | 30 | -1.50 | -0.94 | -0.35 | +1.24 |

Share of the pairs on which the decision list is better, and Spearman's rank correlation of the difference with the number of classes:

| | ahead, two classes | ahead, more classes | correlation, all | correlation, more classes |
|---|--:|--:|--:|--:|
| decision list (per refinement) | 25% | 45% | +0.10 (p = 0.15) | -0.04 (p = 0.74) |
| decision list (per rule) | 33% | 49% | +0.11 (p = 0.11) | -0.04 (p = 0.73) |
| decision list (per seed example) | 21% | 40% | +0.09 (p = 0.19) | +0.15 (p = 0.2) |
| best decision list - best rule set | 39% | 58% | +0.18 (p = 0.01) | +0.07 (p = 0.55) |

## Per dataset (mean across folds)

| dataset | learner | accuracy | n_rules | train_time |
|---|---|---|---|---|
| SPECT | CN2:concept (majority class) | 76.77% | 10.2 | 0.035 |
| SPECT | CN2:concept (minority class) | 69.62% | 16.6 | 0.180 |
| SPECT | CN2:decision list (per refinement) | 75.60% | 11.0 | 0.057 |
| SPECT | CN2:decision list (per rule) | 76.74% | 13.8 | 0.119 |
| SPECT | CN2:decision list (per seed example) | 78.65% | 14.6 | 0.401 |
| SPECT | CN2:rule set (best rule) | 79.00% | 26.8 | 0.216 |
| SPECT | CN2:rule set (macro distribution voting) | 78.62% | 26.8 | 0.216 |
| SPECT | CN2:rule set (micro distribution voting) | 79.74% | 26.8 | 0.216 |
| SPECT | CN2:rule set (voting) | 77.85% | 26.8 | 0.216 |
| SPECT | PFossil:concept (majority class) | 70.81% | 3.5 | 0.004 |
| SPECT | PFossil:concept (minority class) | 73.76% | 6.2 | 0.010 |
| SPECT | PFossil:decision list (per refinement) | 74.50% | 4.5 | 0.014 |
| SPECT | PFossil:decision list (per rule) | 73.76% | 6.2 | 0.014 |
| SPECT | PFossil:decision list (per seed example) | 78.29% | 6.4 | 0.015 |
| SPECT | PFossil:rule set (best rule) | 78.68% | 9.7 | 0.014 |
| SPECT | PFossil:rule set (macro distribution voting) | 77.55% | 9.7 | 0.014 |
| SPECT | PFossil:rule set (micro distribution voting) | 78.68% | 9.7 | 0.014 |
| SPECT | PFossil:rule set (voting) | 77.55% | 9.7 | 0.014 |
| SPECT | Pypper:concept (majority class) | 77.51% | 3.4 | 0.037 |
| SPECT | Pypper:concept (minority class) | 80.11% | 1.0 | 0.024 |
| SPECT | Pypper:decision list (per refinement) | 81.58% | 1.7 | 0.018 |
| SPECT | Pypper:decision list (per rule) | 82.74% | 2.6 | 0.018 |
| SPECT | Pypper:decision list (per seed example) | 82.02% | 7.9 | 0.027 |
| SPECT | Pypper:rule set (best rule) | 80.85% | 4.4 | 0.051 |
| SPECT | Pypper:rule set (macro distribution voting) | 80.85% | 4.4 | 0.051 |
| SPECT | Pypper:rule set (micro distribution voting) | 80.85% | 4.4 | 0.051 |
| SPECT | Pypper:rule set (voting) | 80.85% | 4.4 | 0.051 |
| Titanic | CN2:concept (majority class) | 71.96% | 38.6 | 0.507 |
| Titanic | CN2:concept (minority class) | 66.09% | 38.2 | 0.452 |
| Titanic | CN2:decision list (per refinement) | 78.68% | 29.4 | 0.522 |
| Titanic | CN2:decision list (per rule) | 77.31% | 33.7 | 0.636 |
| Titanic | CN2:decision list (per seed example) | 74.18% | 24.9 | 1.722 |
| Titanic | CN2:rule set (best rule) | 79.52% | 76.8 | 0.932 |
| Titanic | CN2:rule set (macro distribution voting) | 79.07% | 76.8 | 0.932 |
| Titanic | CN2:rule set (micro distribution voting) | 77.31% | 76.8 | 0.932 |
| Titanic | CN2:rule set (voting) | 78.61% | 76.8 | 0.932 |
| Titanic | PFossil:concept (majority class) | 76.93% | 3.1 | 0.006 |
| Titanic | PFossil:concept (minority class) | 78.46% | 2.0 | 0.005 |
| Titanic | PFossil:decision list (per refinement) | 78.46% | 2.0 | 0.009 |
| Titanic | PFossil:decision list (per rule) | 77.54% | 1.9 | 0.008 |
| Titanic | PFossil:decision list (per seed example) | 70.73% | 6.7 | 0.112 |
| Titanic | PFossil:rule set (best rule) | 77.77% | 5.1 | 0.014 |
| Titanic | PFossil:rule set (macro distribution voting) | 77.85% | 5.1 | 0.014 |
| Titanic | PFossil:rule set (micro distribution voting) | 76.78% | 5.1 | 0.014 |
| Titanic | PFossil:rule set (voting) | 78.15% | 5.1 | 0.014 |
| Titanic | Pypper:concept (majority class) | 77.31% | 4.5 | 0.071 |
| Titanic | Pypper:concept (minority class) | 79.76% | 2.6 | 0.049 |
| Titanic | Pypper:decision list (per refinement) | 78.31% | 7.9 | 0.041 |
| Titanic | Pypper:decision list (per rule) | 77.77% | 11.7 | 0.064 |
| Titanic | Pypper:decision list (per seed example) | 74.56% | 15.4 | 0.075 |
| Titanic | Pypper:rule set (best rule) | 79.68% | 7.1 | 0.114 |
| Titanic | Pypper:rule set (macro distribution voting) | 79.68% | 7.1 | 0.114 |
| Titanic | Pypper:rule set (micro distribution voting) | 79.37% | 7.1 | 0.114 |
| Titanic | Pypper:rule set (voting) | 79.14% | 7.1 | 0.114 |
| anneal | CN2:cascade (most frequent first) | 99.22% | 15.1 | 0.037 |
| anneal | CN2:cascade (rarest first) | 96.21% | 22.4 | 0.191 |
| anneal | CN2:decision list (per refinement) | 98.44% | 13.4 | 0.078 |
| anneal | CN2:decision list (per rule) | 98.66% | 14.3 | 0.142 |
| anneal | CN2:decision list (per seed example) | 99.10% | 12.2 | 0.060 |
| anneal | CN2:rule set (best rule) | 98.77% | 33.9 | 0.218 |
| anneal | CN2:rule set (macro distribution voting) | 98.77% | 33.9 | 0.218 |
| anneal | CN2:rule set (micro distribution voting) | 98.77% | 33.9 | 0.218 |
| anneal | CN2:rule set (voting) | 98.66% | 33.9 | 0.218 |
| anneal | PFossil:cascade (most frequent first) | 99.67% | 11.8 | 0.014 |
| anneal | PFossil:cascade (rarest first) | 93.99% | 10.9 | 0.016 |
| anneal | PFossil:decision list (per refinement) | 99.33% | 10.7 | 0.021 |
| anneal | PFossil:decision list (per rule) | 93.65% | 10.0 | 0.033 |
| anneal | PFossil:decision list (per seed example) | 94.89% | 8.0 | 0.011 |
| anneal | PFossil:rule set (best rule) | 99.22% | 17.8 | 0.023 |
| anneal | PFossil:rule set (macro distribution voting) | 99.66% | 17.8 | 0.023 |
| anneal | PFossil:rule set (micro distribution voting) | 98.88% | 17.8 | 0.023 |
| anneal | PFossil:rule set (voting) | 99.66% | 17.8 | 0.023 |
| anneal | Pypper:cascade (most frequent first) | 98.66% | 9.5 | 0.052 |
| anneal | Pypper:cascade (rarest first) | 94.32% | 8.9 | 0.069 |
| anneal | Pypper:decision list (per refinement) | 98.88% | 8.8 | 0.031 |
| anneal | Pypper:decision list (per rule) | 98.88% | 9.8 | 0.066 |
| anneal | Pypper:decision list (per seed example) | 97.99% | 10.7 | 0.029 |
| anneal | Pypper:rule set (best rule) | 95.44% | 14.3 | 0.103 |
| anneal | Pypper:rule set (macro distribution voting) | 95.66% | 14.3 | 0.103 |
| anneal | Pypper:rule set (micro distribution voting) | 95.10% | 14.3 | 0.103 |
| anneal | Pypper:rule set (voting) | 95.44% | 14.3 | 0.103 |
| audiology | CN2:cascade (most frequent first) | 70.40% | 23.2 | 0.366 |
| audiology | CN2:cascade (rarest first) | 84.11% | 25.4 | 0.182 |
| audiology | CN2:decision list (per refinement) | 75.69% | 20.4 | 0.812 |
| audiology | CN2:decision list (per rule) | 76.11% | 20.2 | 1.659 |
| audiology | CN2:decision list (per seed example) | 74.31% | 22.5 | 0.277 |
| audiology | CN2:rule set (best rule) | 76.07% | 31.2 | 0.481 |
| audiology | CN2:rule set (macro distribution voting) | 75.63% | 31.2 | 0.481 |
| audiology | CN2:rule set (micro distribution voting) | 72.53% | 31.2 | 0.481 |
| audiology | CN2:rule set (voting) | 71.19% | 31.2 | 0.481 |
| audiology | PFossil:cascade (most frequent first) | 74.37% | 36.3 | 0.033 |
| audiology | PFossil:cascade (rarest first) | 84.05% | 34.0 | 0.035 |
| audiology | PFossil:decision list (per refinement) | 79.68% | 28.0 | 0.067 |
| audiology | PFossil:decision list (per rule) | 84.09% | 23.6 | 0.248 |
| audiology | PFossil:decision list (per seed example) | 73.46% | 28.6 | 0.025 |
| audiology | PFossil:rule set (best rule) | 77.89% | 43.3 | 0.050 |
| audiology | PFossil:rule set (macro distribution voting) | 77.00% | 43.3 | 0.050 |
| audiology | PFossil:rule set (micro distribution voting) | 74.78% | 43.3 | 0.050 |
| audiology | PFossil:rule set (voting) | 75.26% | 43.3 | 0.050 |
| audiology | Pypper:cascade (most frequent first) | 70.32% | 11.1 | 0.097 |
| audiology | Pypper:cascade (rarest first) | 75.71% | 11.1 | 0.089 |
| audiology | Pypper:decision list (per refinement) | 69.11% | 11.2 | 0.053 |
| audiology | Pypper:decision list (per rule) | 71.26% | 13.7 | 0.313 |
| audiology | Pypper:decision list (per seed example) | 70.45% | 14.9 | 0.046 |
| audiology | Pypper:rule set (best rule) | 66.84% | 13.8 | 0.113 |
| audiology | Pypper:rule set (macro distribution voting) | 67.73% | 13.8 | 0.113 |
| audiology | Pypper:rule set (micro distribution voting) | 65.51% | 13.8 | 0.113 |
| audiology | Pypper:rule set (voting) | 63.77% | 13.8 | 0.113 |
| balance-scale | CN2:cascade (most frequent first) | 75.66% | 64.3 | 0.272 |
| balance-scale | CN2:cascade (rarest first) | 79.03% | 80.8 | 0.526 |
| balance-scale | CN2:decision list (per refinement) | 79.36% | 48.9 | 0.434 |
| balance-scale | CN2:decision list (per rule) | 81.25% | 53.4 | 0.930 |
| balance-scale | CN2:decision list (per seed example) | 80.00% | 45.6 | 0.314 |
| balance-scale | CN2:rule set (best rule) | 78.06% | 136.3 | 0.726 |
| balance-scale | CN2:rule set (macro distribution voting) | 78.55% | 136.3 | 0.726 |
| balance-scale | CN2:rule set (micro distribution voting) | 78.22% | 136.3 | 0.726 |
| balance-scale | CN2:rule set (voting) | 78.70% | 136.3 | 0.726 |
| balance-scale | PFossil:cascade (most frequent first) | 44.16% | 12.4 | 0.017 |
| balance-scale | PFossil:cascade (rarest first) | 78.88% | 13.3 | 0.018 |
| balance-scale | PFossil:decision list (per refinement) | 78.74% | 13.6 | 0.028 |
| balance-scale | PFossil:decision list (per rule) | 79.68% | 16.5 | 0.043 |
| balance-scale | PFossil:decision list (per seed example) | 76.81% | 11.7 | 0.015 |
| balance-scale | PFossil:rule set (best rule) | 81.60% | 26.0 | 0.037 |
| balance-scale | PFossil:rule set (macro distribution voting) | 82.08% | 26.0 | 0.037 |
| balance-scale | PFossil:rule set (micro distribution voting) | 80.47% | 26.0 | 0.037 |
| balance-scale | PFossil:rule set (voting) | 79.67% | 26.0 | 0.037 |
| balance-scale | Pypper:cascade (most frequent first) | 70.91% | 10.5 | 0.104 |
| balance-scale | Pypper:cascade (rarest first) | 78.38% | 7.9 | 0.075 |
| balance-scale | Pypper:decision list (per refinement) | 81.27% | 16.9 | 0.062 |
| balance-scale | Pypper:decision list (per rule) | 83.04% | 16.2 | 0.103 |
| balance-scale | Pypper:decision list (per seed example) | 79.99% | 18.5 | 0.057 |
| balance-scale | Pypper:rule set (best rule) | 81.75% | 15.8 | 0.137 |
| balance-scale | Pypper:rule set (macro distribution voting) | 81.75% | 15.8 | 0.137 |
| balance-scale | Pypper:rule set (micro distribution voting) | 81.26% | 15.8 | 0.137 |
| balance-scale | Pypper:rule set (voting) | 81.26% | 15.8 | 0.137 |
| banknote-authentication | CN2:concept (majority class) | 99.13% | 13.2 | 0.100 |
| banknote-authentication | CN2:concept (minority class) | 98.25% | 14.6 | 0.101 |
| banknote-authentication | CN2:decision list (per refinement) | 98.76% | 13.6 | 0.164 |
| banknote-authentication | CN2:decision list (per rule) | 98.76% | 12.7 | 0.164 |
| banknote-authentication | CN2:decision list (per seed example) | 97.67% | 11.8 | 0.135 |
| banknote-authentication | CN2:rule set (best rule) | 98.91% | 27.8 | 0.197 |
| banknote-authentication | CN2:rule set (macro distribution voting) | 99.05% | 27.8 | 0.197 |
| banknote-authentication | CN2:rule set (micro distribution voting) | 98.76% | 27.8 | 0.197 |
| banknote-authentication | CN2:rule set (voting) | 98.83% | 27.8 | 0.197 |
| banknote-authentication | PFossil:concept (majority class) | 95.99% | 6.8 | 0.011 |
| banknote-authentication | PFossil:concept (minority class) | 96.57% | 7.2 | 0.011 |
| banknote-authentication | PFossil:decision list (per refinement) | 95.99% | 6.8 | 0.017 |
| banknote-authentication | PFossil:decision list (per rule) | 94.31% | 3.9 | 0.008 |
| banknote-authentication | PFossil:decision list (per seed example) | 91.76% | 3.4 | 0.006 |
| banknote-authentication | PFossil:rule set (best rule) | 99.20% | 14.0 | 0.021 |
| banknote-authentication | PFossil:rule set (macro distribution voting) | 99.05% | 14.0 | 0.021 |
| banknote-authentication | PFossil:rule set (micro distribution voting) | 94.32% | 14.0 | 0.021 |
| banknote-authentication | PFossil:rule set (voting) | 97.89% | 14.0 | 0.021 |
| banknote-authentication | Pypper:concept (majority class) | 98.91% | 5.2 | 0.043 |
| banknote-authentication | Pypper:concept (minority class) | 98.47% | 5.2 | 0.038 |
| banknote-authentication | Pypper:decision list (per refinement) | 98.54% | 6.8 | 0.033 |
| banknote-authentication | Pypper:decision list (per rule) | 98.69% | 6.2 | 0.032 |
| banknote-authentication | Pypper:decision list (per seed example) | 95.26% | 8.2 | 0.048 |
| banknote-authentication | Pypper:rule set (best rule) | 98.61% | 10.4 | 0.077 |
| banknote-authentication | Pypper:rule set (macro distribution voting) | 98.69% | 10.4 | 0.077 |
| banknote-authentication | Pypper:rule set (micro distribution voting) | 98.61% | 10.4 | 0.077 |
| banknote-authentication | Pypper:rule set (voting) | 98.69% | 10.4 | 0.077 |
| blood-transfusion-service-center | CN2:concept (majority class) | 77.13% | 18.3 | 0.209 |
| blood-transfusion-service-center | CN2:concept (minority class) | 53.07% | 11.1 | 0.146 |
| blood-transfusion-service-center | CN2:decision list (per refinement) | 77.53% | 19.1 | 0.328 |
| blood-transfusion-service-center | CN2:decision list (per rule) | 77.66% | 19.6 | 0.383 |
| blood-transfusion-service-center | CN2:decision list (per seed example) | 71.65% | 8.5 | 3.056 |
| blood-transfusion-service-center | CN2:rule set (best rule) | 77.40% | 29.4 | 0.381 |
| blood-transfusion-service-center | CN2:rule set (macro distribution voting) | 77.53% | 29.4 | 0.381 |
| blood-transfusion-service-center | CN2:rule set (micro distribution voting) | 77.26% | 29.4 | 0.381 |
| blood-transfusion-service-center | CN2:rule set (voting) | 74.06% | 29.4 | 0.381 |
| blood-transfusion-service-center | PFossil:concept (majority class) | 58.56% | 0.9 | 0.002 |
| blood-transfusion-service-center | PFossil:concept (minority class) | 76.86% | 1.0 | 0.003 |
| blood-transfusion-service-center | PFossil:decision list (per refinement) | 77.80% | 1.8 | 0.005 |
| blood-transfusion-service-center | PFossil:decision list (per rule) | 76.86% | 1.0 | 0.004 |
| blood-transfusion-service-center | PFossil:decision list (per seed example) | 76.34% | 4.4 | 0.090 |
| blood-transfusion-service-center | PFossil:rule set (best rule) | 76.86% | 1.9 | 0.005 |
| blood-transfusion-service-center | PFossil:rule set (macro distribution voting) | 76.86% | 1.9 | 0.005 |
| blood-transfusion-service-center | PFossil:rule set (micro distribution voting) | 76.86% | 1.9 | 0.005 |
| blood-transfusion-service-center | PFossil:rule set (voting) | 76.86% | 1.9 | 0.005 |
| blood-transfusion-service-center | Pypper:concept (majority class) | 75.54% | 2.6 | 0.038 |
| blood-transfusion-service-center | Pypper:concept (minority class) | 78.33% | 2.0 | 0.024 |
| blood-transfusion-service-center | Pypper:decision list (per refinement) | 78.06% | 6.1 | 0.029 |
| blood-transfusion-service-center | Pypper:decision list (per rule) | 78.06% | 3.6 | 0.022 |
| blood-transfusion-service-center | Pypper:decision list (per seed example) | 76.07% | 6.7 | 0.035 |
| blood-transfusion-service-center | Pypper:rule set (best rule) | 78.47% | 4.6 | 0.063 |
| blood-transfusion-service-center | Pypper:rule set (macro distribution voting) | 78.60% | 4.6 | 0.063 |
| blood-transfusion-service-center | Pypper:rule set (micro distribution voting) | 78.34% | 4.6 | 0.063 |
| blood-transfusion-service-center | Pypper:rule set (voting) | 78.34% | 4.6 | 0.063 |
| breast-cancer | CN2:concept (majority class) | 63.60% | 16.2 | 0.081 |
| breast-cancer | CN2:concept (minority class) | 67.13% | 18.8 | 0.104 |
| breast-cancer | CN2:decision list (per refinement) | 68.95% | 13.3 | 0.086 |
| breast-cancer | CN2:decision list (per rule) | 73.02% | 21.9 | 0.202 |
| breast-cancer | CN2:decision list (per seed example) | 63.69% | 25.7 | 0.391 |
| breast-cancer | CN2:rule set (best rule) | 70.62% | 35.0 | 0.184 |
| breast-cancer | CN2:rule set (macro distribution voting) | 71.66% | 35.0 | 0.184 |
| breast-cancer | CN2:rule set (micro distribution voting) | 69.93% | 35.0 | 0.184 |
| breast-cancer | CN2:rule set (voting) | 70.26% | 35.0 | 0.184 |
| breast-cancer | PFossil:concept (majority class) | 67.13% | 6.2 | 0.009 |
| breast-cancer | PFossil:concept (minority class) | 69.24% | 1.2 | 0.004 |
| breast-cancer | PFossil:decision list (per refinement) | 69.24% | 1.4 | 0.005 |
| breast-cancer | PFossil:decision list (per rule) | 69.24% | 1.2 | 0.006 |
| breast-cancer | PFossil:decision list (per seed example) | 68.55% | 8.4 | 0.013 |
| breast-cancer | PFossil:rule set (best rule) | 69.26% | 7.4 | 0.012 |
| breast-cancer | PFossil:rule set (macro distribution voting) | 69.98% | 7.4 | 0.012 |
| breast-cancer | PFossil:rule set (micro distribution voting) | 68.56% | 7.4 | 0.012 |
| breast-cancer | PFossil:rule set (voting) | 68.92% | 7.4 | 0.012 |
| breast-cancer | Pypper:concept (majority class) | 70.58% | 1.4 | 0.022 |
| breast-cancer | Pypper:concept (minority class) | 74.84% | 1.0 | 0.021 |
| breast-cancer | Pypper:decision list (per refinement) | 74.80% | 3.3 | 0.012 |
| breast-cancer | Pypper:decision list (per rule) | 73.72% | 4.6 | 0.022 |
| breast-cancer | Pypper:decision list (per seed example) | 70.22% | 8.1 | 0.022 |
| breast-cancer | Pypper:rule set (best rule) | 74.84% | 2.4 | 0.042 |
| breast-cancer | Pypper:rule set (macro distribution voting) | 74.84% | 2.4 | 0.042 |
| breast-cancer | Pypper:rule set (micro distribution voting) | 74.84% | 2.4 | 0.042 |
| breast-cancer | Pypper:rule set (voting) | 74.84% | 2.4 | 0.042 |
| breast-w | CN2:concept (majority class) | 94.57% | 19.0 | 0.075 |
| breast-w | CN2:concept (minority class) | 94.71% | 18.1 | 0.045 |
| breast-w | CN2:decision list (per refinement) | 94.85% | 14.2 | 0.062 |
| breast-w | CN2:decision list (per rule) | 95.42% | 13.8 | 0.117 |
| breast-w | CN2:decision list (per seed example) | 94.56% | 12.1 | 0.065 |
| breast-w | CN2:rule set (best rule) | 94.42% | 37.1 | 0.121 |
| breast-w | CN2:rule set (macro distribution voting) | 94.42% | 37.1 | 0.121 |
| breast-w | CN2:rule set (micro distribution voting) | 94.42% | 37.1 | 0.121 |
| breast-w | CN2:rule set (voting) | 94.42% | 37.1 | 0.121 |
| breast-w | PFossil:concept (majority class) | 94.41% | 9.5 | 0.015 |
| breast-w | PFossil:concept (minority class) | 94.56% | 6.1 | 0.008 |
| breast-w | PFossil:decision list (per refinement) | 94.85% | 5.6 | 0.020 |
| breast-w | PFossil:decision list (per rule) | 94.70% | 6.1 | 0.016 |
| breast-w | PFossil:decision list (per seed example) | 95.13% | 4.6 | 0.010 |
| breast-w | PFossil:rule set (best rule) | 94.13% | 15.6 | 0.023 |
| breast-w | PFossil:rule set (macro distribution voting) | 94.99% | 15.6 | 0.023 |
| breast-w | PFossil:rule set (micro distribution voting) | 94.27% | 15.6 | 0.023 |
| breast-w | PFossil:rule set (voting) | 94.99% | 15.6 | 0.023 |
| breast-w | Pypper:concept (majority class) | 95.27% | 2.5 | 0.032 |
| breast-w | Pypper:concept (minority class) | 95.57% | 3.9 | 0.033 |
| breast-w | Pypper:decision list (per refinement) | 95.14% | 4.3 | 0.023 |
| breast-w | Pypper:decision list (per rule) | 95.14% | 4.6 | 0.023 |
| breast-w | Pypper:decision list (per seed example) | 94.99% | 6.4 | 0.023 |
| breast-w | Pypper:rule set (best rule) | 94.57% | 6.4 | 0.061 |
| breast-w | Pypper:rule set (macro distribution voting) | 95.00% | 6.4 | 0.061 |
| breast-w | Pypper:rule set (micro distribution voting) | 94.71% | 6.4 | 0.061 |
| breast-w | Pypper:rule set (voting) | 95.00% | 6.4 | 0.061 |
| car | CN2:cascade (most frequent first) | 98.90% | 31.8 | 0.123 |
| car | CN2:cascade (rarest first) | 94.21% | 48.0 | 0.272 |
| car | CN2:decision list (per refinement) | 96.76% | 47.0 | 0.458 |
| car | CN2:decision list (per rule) | 97.74% | 42.7 | 0.663 |
| car | CN2:decision list (per seed example) | 95.83% | 35.5 | 0.215 |
| car | CN2:rule set (best rule) | 95.02% | 82.1 | 0.449 |
| car | CN2:rule set (macro distribution voting) | 94.73% | 82.1 | 0.449 |
| car | CN2:rule set (micro distribution voting) | 94.91% | 82.1 | 0.449 |
| car | CN2:rule set (voting) | 94.56% | 82.1 | 0.449 |
| car | PFossil:cascade (most frequent first) | 95.95% | 19.8 | 0.027 |
| car | PFossil:cascade (rarest first) | 82.87% | 5.4 | 0.015 |
| car | PFossil:decision list (per refinement) | 94.10% | 14.4 | 0.041 |
| car | PFossil:decision list (per rule) | 81.31% | 4.7 | 0.020 |
| car | PFossil:decision list (per seed example) | 78.82% | 2.1 | 0.158 |
| car | PFossil:rule set (best rule) | 93.87% | 16.2 | 0.033 |
| car | PFossil:rule set (macro distribution voting) | 93.87% | 16.2 | 0.033 |
| car | PFossil:rule set (micro distribution voting) | 84.90% | 16.2 | 0.033 |
| car | PFossil:rule set (voting) | 90.51% | 16.2 | 0.033 |
| car | Pypper:cascade (most frequent first) | 97.69% | 23.5 | 0.120 |
| car | Pypper:cascade (rarest first) | 94.56% | 23.2 | 0.242 |
| car | Pypper:decision list (per refinement) | 95.78% | 22.1 | 0.093 |
| car | Pypper:decision list (per rule) | 96.30% | 20.2 | 0.169 |
| car | Pypper:decision list (per seed example) | 93.75% | 26.5 | 0.080 |
| car | Pypper:rule set (best rule) | 92.36% | 29.7 | 0.288 |
| car | Pypper:rule set (macro distribution voting) | 92.24% | 29.7 | 0.288 |
| car | Pypper:rule set (micro distribution voting) | 90.91% | 29.7 | 0.288 |
| car | Pypper:rule set (voting) | 91.49% | 29.7 | 0.288 |
| churn | CN2:concept (majority class) | 88.66% | 97.0 | 0.881 |
| churn | CN2:concept (minority class) | 88.30% | 219.9 | 2.948 |
| churn | CN2:decision list (per refinement) | 89.46% | 100.8 | 1.409 |
| churn | CN2:decision list (per rule) | 89.72% | 124.7 | 1.497 |
| churn | CN2:decision list (per seed example) | 89.96% | 153.1 | 2.298 |
| churn | CN2:rule set (best rule) | 92.52% | 316.9 | 3.812 |
| churn | CN2:rule set (macro distribution voting) | 90.50% | 316.9 | 3.812 |
| churn | CN2:rule set (micro distribution voting) | 89.20% | 316.9 | 3.812 |
| churn | CN2:rule set (voting) | 90.08% | 316.9 | 3.812 |
| churn | PFossil:concept (majority class) | 91.46% | 6.7 | 0.033 |
| churn | PFossil:concept (minority class) | 90.36% | 2.5 | 0.012 |
| churn | PFossil:decision list (per refinement) | 92.80% | 8.1 | 0.079 |
| churn | PFossil:decision list (per rule) | 92.02% | 6.6 | 0.045 |
| churn | PFossil:decision list (per seed example) | 91.76% | 9.1 | 0.039 |
| churn | PFossil:rule set (best rule) | 90.00% | 9.2 | 0.053 |
| churn | PFossil:rule set (macro distribution voting) | 89.98% | 9.2 | 0.053 |
| churn | PFossil:rule set (micro distribution voting) | 89.18% | 9.2 | 0.053 |
| churn | PFossil:rule set (voting) | 89.00% | 9.2 | 0.053 |
| churn | Pypper:concept (majority class) | 93.02% | 7.6 | 0.236 |
| churn | Pypper:concept (minority class) | 93.92% | 10.6 | 0.202 |
| churn | Pypper:decision list (per refinement) | 93.52% | 14.2 | 0.164 |
| churn | Pypper:decision list (per rule) | 93.00% | 20.0 | 0.177 |
| churn | Pypper:decision list (per seed example) | 92.06% | 20.9 | 0.142 |
| churn | Pypper:rule set (best rule) | 93.66% | 18.2 | 0.439 |
| churn | Pypper:rule set (macro distribution voting) | 93.66% | 18.2 | 0.439 |
| churn | Pypper:rule set (micro distribution voting) | 93.52% | 18.2 | 0.439 |
| churn | Pypper:rule set (voting) | 93.46% | 18.2 | 0.439 |
| climate-model-simulation-crashes | CN2:concept (majority class) | 91.48% | 14.3 | 0.073 |
| climate-model-simulation-crashes | CN2:concept (minority class) | 87.78% | 24.4 | 0.090 |
| climate-model-simulation-crashes | CN2:decision list (per refinement) | 93.15% | 12.8 | 0.064 |
| climate-model-simulation-crashes | CN2:decision list (per rule) | 92.41% | 12.2 | 0.091 |
| climate-model-simulation-crashes | CN2:decision list (per seed example) | 92.04% | 13.1 | 0.059 |
| climate-model-simulation-crashes | CN2:rule set (best rule) | 92.41% | 38.7 | 0.159 |
| climate-model-simulation-crashes | CN2:rule set (macro distribution voting) | 92.59% | 38.7 | 0.159 |
| climate-model-simulation-crashes | CN2:rule set (micro distribution voting) | 92.41% | 38.7 | 0.159 |
| climate-model-simulation-crashes | CN2:rule set (voting) | 92.59% | 38.7 | 0.159 |
| climate-model-simulation-crashes | PFossil:concept (majority class) | 92.22% | 6.4 | 0.009 |
| climate-model-simulation-crashes | PFossil:concept (minority class) | 93.52% | 2.5 | 0.005 |
| climate-model-simulation-crashes | PFossil:decision list (per refinement) | 93.15% | 2.8 | 0.011 |
| climate-model-simulation-crashes | PFossil:decision list (per rule) | 93.33% | 3.1 | 0.010 |
| climate-model-simulation-crashes | PFossil:decision list (per seed example) | 91.85% | 5.6 | 0.009 |
| climate-model-simulation-crashes | PFossil:rule set (best rule) | 92.78% | 8.9 | 0.016 |
| climate-model-simulation-crashes | PFossil:rule set (macro distribution voting) | 92.96% | 8.9 | 0.016 |
| climate-model-simulation-crashes | PFossil:rule set (micro distribution voting) | 92.78% | 8.9 | 0.016 |
| climate-model-simulation-crashes | PFossil:rule set (voting) | 92.78% | 8.9 | 0.016 |
| climate-model-simulation-crashes | Pypper:concept (majority class) | 92.78% | 3.7 | 0.030 |
| climate-model-simulation-crashes | Pypper:concept (minority class) | 94.07% | 2.9 | 0.027 |
| climate-model-simulation-crashes | Pypper:decision list (per refinement) | 94.63% | 3.4 | 0.018 |
| climate-model-simulation-crashes | Pypper:decision list (per rule) | 94.63% | 4.0 | 0.021 |
| climate-model-simulation-crashes | Pypper:decision list (per seed example) | 94.07% | 6.0 | 0.018 |
| climate-model-simulation-crashes | Pypper:rule set (best rule) | 95.19% | 6.6 | 0.054 |
| climate-model-simulation-crashes | Pypper:rule set (macro distribution voting) | 94.81% | 6.6 | 0.054 |
| climate-model-simulation-crashes | Pypper:rule set (micro distribution voting) | 95.19% | 6.6 | 0.054 |
| climate-model-simulation-crashes | Pypper:rule set (voting) | 95.19% | 6.6 | 0.054 |
| cmc | CN2:cascade (most frequent first) | 49.16% | 90.4 | 0.824 |
| cmc | CN2:cascade (rarest first) | 40.33% | 131.2 | 1.388 |
| cmc | CN2:decision list (per refinement) | 50.44% | 78.7 | 1.203 |
| cmc | CN2:decision list (per rule) | 49.97% | 107.9 | 2.917 |
| cmc | CN2:decision list (per seed example) | 49.63% | 88.3 | 1.975 |
| cmc | CN2:rule set (best rule) | 51.87% | 183.2 | 1.855 |
| cmc | CN2:rule set (macro distribution voting) | 53.84% | 183.2 | 1.855 |
| cmc | CN2:rule set (micro distribution voting) | 54.11% | 183.2 | 1.855 |
| cmc | CN2:rule set (voting) | 52.01% | 183.2 | 1.855 |
| cmc | PFossil:cascade (most frequent first) | 36.66% | 1.4 | 0.007 |
| cmc | PFossil:cascade (rarest first) | 51.53% | 4.8 | 0.013 |
| cmc | PFossil:decision list (per refinement) | 45.02% | 0.7 | 0.005 |
| cmc | PFossil:decision list (per rule) | 51.46% | 10.9 | 0.037 |
| cmc | PFossil:decision list (per seed example) | 50.51% | 10.8 | 0.069 |
| cmc | PFossil:rule set (best rule) | 52.61% | 2.1 | 0.013 |
| cmc | PFossil:rule set (macro distribution voting) | 52.27% | 2.1 | 0.013 |
| cmc | PFossil:rule set (micro distribution voting) | 51.73% | 2.1 | 0.013 |
| cmc | PFossil:rule set (voting) | 51.73% | 2.1 | 0.013 |
| cmc | Pypper:cascade (most frequent first) | 50.92% | 5.9 | 0.105 |
| cmc | Pypper:cascade (rarest first) | 54.52% | 4.2 | 0.104 |
| cmc | Pypper:decision list (per refinement) | 49.28% | 10.1 | 0.057 |
| cmc | Pypper:decision list (per rule) | 50.50% | 6.6 | 0.073 |
| cmc | Pypper:decision list (per seed example) | 51.06% | 41.5 | 0.291 |
| cmc | Pypper:rule set (best rule) | 52.21% | 6.5 | 0.140 |
| cmc | Pypper:rule set (macro distribution voting) | 52.35% | 6.5 | 0.140 |
| cmc | Pypper:rule set (micro distribution voting) | 52.35% | 6.5 | 0.140 |
| cmc | Pypper:rule set (voting) | 52.35% | 6.5 | 0.140 |
| colic | CN2:concept (majority class) | 80.46% | 32.6 | 0.101 |
| colic | CN2:concept (minority class) | 78.82% | 32.4 | 0.116 |
| colic | CN2:decision list (per refinement) | 76.91% | 24.8 | 0.142 |
| colic | CN2:decision list (per rule) | 77.17% | 27.0 | 0.222 |
| colic | CN2:decision list (per seed example) | 72.04% | 23.5 | 0.119 |
| colic | CN2:rule set (best rule) | 82.09% | 65.0 | 0.219 |
| colic | CN2:rule set (macro distribution voting) | 81.55% | 65.0 | 0.219 |
| colic | CN2:rule set (micro distribution voting) | 82.91% | 65.0 | 0.219 |
| colic | CN2:rule set (voting) | 82.36% | 65.0 | 0.219 |
| colic | PFossil:concept (majority class) | 81.53% | 9.6 | 0.010 |
| colic | PFossil:concept (minority class) | 82.91% | 10.2 | 0.012 |
| colic | PFossil:decision list (per refinement) | 83.70% | 6.3 | 0.014 |
| colic | PFossil:decision list (per rule) | 82.08% | 11.2 | 0.023 |
| colic | PFossil:decision list (per seed example) | 84.28% | 9.1 | 0.013 |
| colic | PFossil:rule set (best rule) | 82.89% | 19.8 | 0.022 |
| colic | PFossil:rule set (macro distribution voting) | 81.79% | 19.8 | 0.022 |
| colic | PFossil:rule set (micro distribution voting) | 86.15% | 19.8 | 0.022 |
| colic | PFossil:rule set (voting) | 82.88% | 19.8 | 0.022 |
| colic | Pypper:concept (majority class) | 84.80% | 2.6 | 0.037 |
| colic | Pypper:concept (minority class) | 83.72% | 3.0 | 0.034 |
| colic | Pypper:decision list (per refinement) | 83.19% | 5.8 | 0.021 |
| colic | Pypper:decision list (per rule) | 81.86% | 7.1 | 0.030 |
| colic | Pypper:decision list (per seed example) | 84.26% | 7.8 | 0.021 |
| colic | Pypper:rule set (best rule) | 85.08% | 5.6 | 0.065 |
| colic | Pypper:rule set (macro distribution voting) | 84.80% | 5.6 | 0.065 |
| colic | Pypper:rule set (micro distribution voting) | 85.89% | 5.6 | 0.065 |
| colic | Pypper:rule set (voting) | 85.35% | 5.6 | 0.065 |
| compas-two-years | CN2:concept (majority class) | 57.35% | 55.8 | 0.792 |
| compas-two-years | CN2:concept (minority class) | 55.89% | 80.1 | 1.325 |
| compas-two-years | CN2:decision list (per refinement) | 67.20% | 62.6 | 1.414 |
| compas-two-years | CN2:decision list (per rule) | 66.71% | 78.8 | 2.058 |
| compas-two-years | CN2:decision list (per seed example) | 63.07% | 36.1 | 8.603 |
| compas-two-years | CN2:rule set (best rule) | 67.64% | 135.9 | 2.085 |
| compas-two-years | CN2:rule set (macro distribution voting) | 67.89% | 135.9 | 2.085 |
| compas-two-years | CN2:rule set (micro distribution voting) | 67.28% | 135.9 | 2.085 |
| compas-two-years | CN2:rule set (voting) | 66.99% | 135.9 | 2.085 |
| compas-two-years | PFossil:concept (majority class) | 64.78% | 1.0 | 0.007 |
| compas-two-years | PFossil:concept (minority class) | 52.96% | 0.0 | 0.004 |
| compas-two-years | PFossil:decision list (per refinement) | 52.96% | 0.0 | 0.011 |
| compas-two-years | PFossil:decision list (per rule) | 64.78% | 1.0 | 0.012 |
| compas-two-years | PFossil:decision list (per seed example) | 61.63% | 5.4 | 2.811 |
| compas-two-years | PFossil:rule set (best rule) | 52.96% | 1.0 | 0.018 |
| compas-two-years | PFossil:rule set (macro distribution voting) | 52.96% | 1.0 | 0.018 |
| compas-two-years | PFossil:rule set (micro distribution voting) | 52.96% | 1.0 | 0.018 |
| compas-two-years | PFossil:rule set (voting) | 52.96% | 1.0 | 0.018 |
| compas-two-years | Pypper:concept (majority class) | 67.32% | 3.5 | 0.250 |
| compas-two-years | Pypper:concept (minority class) | 67.03% | 4.8 | 0.205 |
| compas-two-years | Pypper:decision list (per refinement) | 66.80% | 15.0 | 0.191 |
| compas-two-years | Pypper:decision list (per rule) | 67.41% | 26.9 | 0.314 |
| compas-two-years | Pypper:decision list (per seed example) | 65.97% | 17.9 | 0.214 |
| compas-two-years | Pypper:rule set (best rule) | 67.03% | 8.3 | 0.460 |
| compas-two-years | Pypper:rule set (macro distribution voting) | 67.03% | 8.3 | 0.460 |
| compas-two-years | Pypper:rule set (micro distribution voting) | 66.62% | 8.3 | 0.460 |
| compas-two-years | Pypper:rule set (voting) | 66.62% | 8.3 | 0.460 |
| credit-approval | CN2:concept (majority class) | 76.96% | 47.1 | 0.318 |
| credit-approval | CN2:concept (minority class) | 82.46% | 45.0 | 0.264 |
| credit-approval | CN2:decision list (per refinement) | 80.58% | 25.5 | 0.167 |
| credit-approval | CN2:decision list (per rule) | 81.59% | 32.9 | 0.275 |
| credit-approval | CN2:decision list (per seed example) | 80.58% | 37.3 | 0.456 |
| credit-approval | CN2:rule set (best rule) | 85.80% | 92.1 | 0.570 |
| credit-approval | CN2:rule set (macro distribution voting) | 84.49% | 92.1 | 0.570 |
| credit-approval | CN2:rule set (micro distribution voting) | 86.23% | 92.1 | 0.570 |
| credit-approval | CN2:rule set (voting) | 84.35% | 92.1 | 0.570 |
| credit-approval | PFossil:concept (majority class) | 83.04% | 6.3 | 0.013 |
| credit-approval | PFossil:concept (minority class) | 84.20% | 2.5 | 0.005 |
| credit-approval | PFossil:decision list (per refinement) | 84.35% | 3.7 | 0.012 |
| credit-approval | PFossil:decision list (per rule) | 82.46% | 7.2 | 0.020 |
| credit-approval | PFossil:decision list (per seed example) | 84.49% | 4.6 | 0.007 |
| credit-approval | PFossil:rule set (best rule) | 82.17% | 8.8 | 0.018 |
| credit-approval | PFossil:rule set (macro distribution voting) | 82.03% | 8.8 | 0.018 |
| credit-approval | PFossil:rule set (micro distribution voting) | 85.07% | 8.8 | 0.018 |
| credit-approval | PFossil:rule set (voting) | 82.46% | 8.8 | 0.018 |
| credit-approval | Pypper:concept (majority class) | 85.94% | 2.1 | 0.037 |
| credit-approval | Pypper:concept (minority class) | 85.36% | 2.7 | 0.035 |
| credit-approval | Pypper:decision list (per refinement) | 83.62% | 5.8 | 0.025 |
| credit-approval | Pypper:decision list (per rule) | 85.22% | 6.9 | 0.032 |
| credit-approval | Pypper:decision list (per seed example) | 84.35% | 10.5 | 0.031 |
| credit-approval | Pypper:rule set (best rule) | 86.09% | 4.8 | 0.068 |
| credit-approval | Pypper:rule set (macro distribution voting) | 86.09% | 4.8 | 0.068 |
| credit-approval | Pypper:rule set (micro distribution voting) | 85.51% | 4.8 | 0.068 |
| credit-approval | Pypper:rule set (voting) | 86.09% | 4.8 | 0.068 |
| credit-g | CN2:concept (majority class) | 73.10% | 85.4 | 0.501 |
| credit-g | CN2:concept (minority class) | 65.60% | 102.1 | 0.495 |
| credit-g | CN2:decision list (per refinement) | 70.60% | 51.3 | 0.314 |
| credit-g | CN2:decision list (per rule) | 69.60% | 65.0 | 0.587 |
| credit-g | CN2:decision list (per seed example) | 70.80% | 75.1 | 0.433 |
| credit-g | CN2:rule set (best rule) | 74.50% | 187.5 | 0.992 |
| credit-g | CN2:rule set (macro distribution voting) | 73.50% | 187.5 | 0.992 |
| credit-g | CN2:rule set (micro distribution voting) | 73.70% | 187.5 | 0.992 |
| credit-g | CN2:rule set (voting) | 73.10% | 187.5 | 0.992 |
| credit-g | PFossil:concept (majority class) | 57.50% | 1.4 | 0.006 |
| credit-g | PFossil:concept (minority class) | 72.70% | 2.0 | 0.009 |
| credit-g | PFossil:decision list (per refinement) | 73.90% | 2.0 | 0.015 |
| credit-g | PFossil:decision list (per rule) | 72.70% | 2.0 | 0.015 |
| credit-g | PFossil:decision list (per seed example) | 69.30% | 12.4 | 0.117 |
| credit-g | PFossil:rule set (best rule) | 73.20% | 3.4 | 0.018 |
| credit-g | PFossil:rule set (macro distribution voting) | 73.20% | 3.4 | 0.018 |
| credit-g | PFossil:rule set (micro distribution voting) | 73.30% | 3.4 | 0.018 |
| credit-g | PFossil:rule set (voting) | 73.20% | 3.4 | 0.018 |
| credit-g | Pypper:concept (majority class) | 66.40% | 3.5 | 0.076 |
| credit-g | Pypper:concept (minority class) | 70.40% | 3.4 | 0.063 |
| credit-g | Pypper:decision list (per refinement) | 71.10% | 7.4 | 0.041 |
| credit-g | Pypper:decision list (per rule) | 72.60% | 13.8 | 0.069 |
| credit-g | Pypper:decision list (per seed example) | 72.30% | 15.5 | 0.053 |
| credit-g | Pypper:rule set (best rule) | 71.50% | 6.9 | 0.137 |
| credit-g | Pypper:rule set (macro distribution voting) | 71.50% | 6.9 | 0.137 |
| credit-g | Pypper:rule set (micro distribution voting) | 71.50% | 6.9 | 0.137 |
| credit-g | Pypper:rule set (voting) | 71.60% | 6.9 | 0.137 |
| cylinder-bands | CN2:concept (majority class) | 70.56% | 40.4 | 0.149 |
| cylinder-bands | CN2:concept (minority class) | 67.22% | 47.9 | 0.281 |
| cylinder-bands | CN2:decision list (per refinement) | 70.74% | 39.2 | 0.171 |
| cylinder-bands | CN2:decision list (per rule) | 71.48% | 40.9 | 0.325 |
| cylinder-bands | CN2:decision list (per seed example) | 75.19% | 39.0 | 0.271 |
| cylinder-bands | CN2:rule set (best rule) | 74.07% | 88.3 | 0.426 |
| cylinder-bands | CN2:rule set (macro distribution voting) | 73.52% | 88.3 | 0.426 |
| cylinder-bands | CN2:rule set (micro distribution voting) | 71.67% | 88.3 | 0.426 |
| cylinder-bands | CN2:rule set (voting) | 75.74% | 88.3 | 0.426 |
| cylinder-bands | PFossil:concept (majority class) | 74.44% | 7.6 | 0.017 |
| cylinder-bands | PFossil:concept (minority class) | 68.70% | 1.8 | 0.013 |
| cylinder-bands | PFossil:decision list (per refinement) | 74.44% | 6.9 | 0.032 |
| cylinder-bands | PFossil:decision list (per rule) | 74.07% | 4.5 | 0.027 |
| cylinder-bands | PFossil:decision list (per seed example) | 71.85% | 4.6 | 0.018 |
| cylinder-bands | PFossil:rule set (best rule) | 70.19% | 9.4 | 0.031 |
| cylinder-bands | PFossil:rule set (macro distribution voting) | 70.56% | 9.4 | 0.031 |
| cylinder-bands | PFossil:rule set (micro distribution voting) | 68.15% | 9.4 | 0.031 |
| cylinder-bands | PFossil:rule set (voting) | 67.96% | 9.4 | 0.031 |
| cylinder-bands | Pypper:concept (majority class) | 71.67% | 4.3 | 0.055 |
| cylinder-bands | Pypper:concept (minority class) | 66.48% | 3.7 | 0.064 |
| cylinder-bands | Pypper:decision list (per refinement) | 73.33% | 6.4 | 0.035 |
| cylinder-bands | Pypper:decision list (per rule) | 70.37% | 7.5 | 0.045 |
| cylinder-bands | Pypper:decision list (per seed example) | 69.07% | 8.9 | 0.040 |
| cylinder-bands | Pypper:rule set (best rule) | 67.59% | 8.0 | 0.120 |
| cylinder-bands | Pypper:rule set (macro distribution voting) | 69.07% | 8.0 | 0.120 |
| cylinder-bands | Pypper:rule set (micro distribution voting) | 66.67% | 8.0 | 0.120 |
| cylinder-bands | Pypper:rule set (voting) | 67.78% | 8.0 | 0.120 |
| diabetes | CN2:concept (majority class) | 71.48% | 44.7 | 0.228 |
| diabetes | CN2:concept (minority class) | 65.87% | 61.9 | 0.600 |
| diabetes | CN2:decision list (per refinement) | 73.06% | 37.4 | 0.239 |
| diabetes | CN2:decision list (per rule) | 73.30% | 42.8 | 0.383 |
| diabetes | CN2:decision list (per seed example) | 71.35% | 55.5 | 0.842 |
| diabetes | CN2:rule set (best rule) | 75.13% | 106.6 | 0.825 |
| diabetes | CN2:rule set (macro distribution voting) | 74.47% | 106.6 | 0.825 |
| diabetes | CN2:rule set (micro distribution voting) | 73.57% | 106.6 | 0.825 |
| diabetes | CN2:rule set (voting) | 73.56% | 106.6 | 0.825 |
| diabetes | PFossil:concept (majority class) | 74.09% | 6.1 | 0.010 |
| diabetes | PFossil:concept (minority class) | 75.26% | 3.0 | 0.007 |
| diabetes | PFossil:decision list (per refinement) | 73.05% | 3.4 | 0.011 |
| diabetes | PFossil:decision list (per rule) | 74.87% | 4.3 | 0.013 |
| diabetes | PFossil:decision list (per seed example) | 73.71% | 5.7 | 0.021 |
| diabetes | PFossil:rule set (best rule) | 74.61% | 9.1 | 0.018 |
| diabetes | PFossil:rule set (macro distribution voting) | 73.83% | 9.1 | 0.018 |
| diabetes | PFossil:rule set (micro distribution voting) | 75.13% | 9.1 | 0.018 |
| diabetes | PFossil:rule set (voting) | 74.09% | 9.1 | 0.018 |
| diabetes | Pypper:concept (majority class) | 71.35% | 3.9 | 0.062 |
| diabetes | Pypper:concept (minority class) | 74.22% | 3.8 | 0.050 |
| diabetes | Pypper:decision list (per refinement) | 74.35% | 6.7 | 0.033 |
| diabetes | Pypper:decision list (per rule) | 73.43% | 10.9 | 0.058 |
| diabetes | Pypper:decision list (per seed example) | 73.83% | 13.2 | 0.041 |
| diabetes | Pypper:rule set (best rule) | 74.61% | 7.7 | 0.111 |
| diabetes | Pypper:rule set (macro distribution voting) | 74.48% | 7.7 | 0.111 |
| diabetes | Pypper:rule set (micro distribution voting) | 74.48% | 7.7 | 0.111 |
| diabetes | Pypper:rule set (voting) | 74.35% | 7.7 | 0.111 |
| dna | CN2:cascade (most frequent first) | 91.43% | 90.1 | 0.742 |
| dna | CN2:cascade (rarest first) | 84.15% | 82.1 | 1.110 |
| dna | CN2:decision list (per refinement) | 88.48% | 84.7 | 0.930 |
| dna | CN2:decision list (per rule) | 87.64% | 79.8 | 2.584 |
| dna | CN2:decision list (per seed example) | 88.39% | 77.0 | 0.894 |
| dna | CN2:rule set (best rule) | 89.14% | 153.6 | 1.894 |
| dna | CN2:rule set (macro distribution voting) | 88.36% | 153.6 | 1.894 |
| dna | CN2:rule set (micro distribution voting) | 89.11% | 153.6 | 1.894 |
| dna | CN2:rule set (voting) | 88.36% | 153.6 | 1.894 |
| dna | PFossil:cascade (most frequent first) | 88.26% | 18.1 | 0.051 |
| dna | PFossil:cascade (rarest first) | 89.99% | 15.7 | 0.058 |
| dna | PFossil:decision list (per refinement) | 89.64% | 12.2 | 0.058 |
| dna | PFossil:decision list (per rule) | 93.53% | 12.3 | 0.075 |
| dna | PFossil:decision list (per seed example) | 91.05% | 11.5 | 0.030 |
| dna | PFossil:rule set (best rule) | 93.69% | 27.8 | 0.100 |
| dna | PFossil:rule set (macro distribution voting) | 93.53% | 27.8 | 0.100 |
| dna | PFossil:rule set (micro distribution voting) | 89.89% | 27.8 | 0.100 |
| dna | PFossil:rule set (voting) | 90.65% | 27.8 | 0.100 |
| dna | Pypper:cascade (most frequent first) | 91.12% | 23.1 | 0.285 |
| dna | Pypper:cascade (rarest first) | 92.12% | 15.8 | 0.323 |
| dna | Pypper:decision list (per refinement) | 91.12% | 22.5 | 0.157 |
| dna | Pypper:decision list (per rule) | 92.97% | 19.2 | 0.210 |
| dna | Pypper:decision list (per seed example) | 92.09% | 23.4 | 0.108 |
| dna | Pypper:rule set (best rule) | 93.13% | 29.0 | 0.537 |
| dna | Pypper:rule set (macro distribution voting) | 93.13% | 29.0 | 0.537 |
| dna | Pypper:rule set (micro distribution voting) | 93.16% | 29.0 | 0.537 |
| dna | Pypper:rule set (voting) | 93.22% | 29.0 | 0.537 |
| dresses-sales | CN2:concept (majority class) | 53.60% | 13.0 | 0.065 |
| dresses-sales | CN2:concept (minority class) | 57.20% | 11.2 | 0.052 |
| dresses-sales | CN2:decision list (per refinement) | 57.60% | 15.1 | 0.082 |
| dresses-sales | CN2:decision list (per rule) | 60.20% | 18.9 | 0.197 |
| dresses-sales | CN2:decision list (per seed example) | 58.00% | 50.7 | 1.097 |
| dresses-sales | CN2:rule set (best rule) | 62.60% | 24.2 | 0.112 |
| dresses-sales | CN2:rule set (macro distribution voting) | 62.40% | 24.2 | 0.112 |
| dresses-sales | CN2:rule set (micro distribution voting) | 62.40% | 24.2 | 0.112 |
| dresses-sales | CN2:rule set (voting) | 61.40% | 24.2 | 0.112 |
| dresses-sales | PFossil:concept (majority class) | 53.20% | 1.5 | 0.007 |
| dresses-sales | PFossil:concept (minority class) | 59.40% | 0.7 | 0.006 |
| dresses-sales | PFossil:decision list (per refinement) | 57.80% | 1.4 | 0.012 |
| dresses-sales | PFossil:decision list (per rule) | 59.20% | 2.4 | 0.017 |
| dresses-sales | PFossil:decision list (per seed example) | 59.40% | 8.0 | 0.052 |
| dresses-sales | PFossil:rule set (best rule) | 59.40% | 2.2 | 0.013 |
| dresses-sales | PFossil:rule set (macro distribution voting) | 59.40% | 2.2 | 0.013 |
| dresses-sales | PFossil:rule set (micro distribution voting) | 59.20% | 2.2 | 0.013 |
| dresses-sales | PFossil:rule set (voting) | 59.20% | 2.2 | 0.013 |
| dresses-sales | Pypper:concept (majority class) | 53.40% | 0.9 | 0.040 |
| dresses-sales | Pypper:concept (minority class) | 61.00% | 0.9 | 0.023 |
| dresses-sales | Pypper:decision list (per refinement) | 59.40% | 8.0 | 0.033 |
| dresses-sales | Pypper:decision list (per rule) | 58.20% | 8.4 | 0.043 |
| dresses-sales | Pypper:decision list (per seed example) | 58.40% | 10.5 | 0.034 |
| dresses-sales | Pypper:rule set (best rule) | 61.00% | 1.8 | 0.061 |
| dresses-sales | Pypper:rule set (macro distribution voting) | 61.00% | 1.8 | 0.061 |
| dresses-sales | Pypper:rule set (micro distribution voting) | 60.80% | 1.8 | 0.061 |
| dresses-sales | Pypper:rule set (voting) | 60.80% | 1.8 | 0.061 |
| ecoli | CN2:cascade (most frequent first) | 82.76% | 19.7 | 0.193 |
| ecoli | CN2:cascade (rarest first) | 75.96% | 21.0 | 0.209 |
| ecoli | CN2:decision list (per refinement) | 82.82% | 14.4 | 0.376 |
| ecoli | CN2:decision list (per rule) | 83.71% | 14.9 | 1.086 |
| ecoli | CN2:decision list (per seed example) | 82.79% | 14.7 | 0.155 |
| ecoli | CN2:rule set (best rule) | 82.49% | 33.0 | 0.351 |
| ecoli | CN2:rule set (macro distribution voting) | 82.48% | 33.0 | 0.351 |
| ecoli | CN2:rule set (micro distribution voting) | 83.97% | 33.0 | 0.351 |
| ecoli | CN2:rule set (voting) | 84.55% | 33.0 | 0.351 |
| ecoli | PFossil:cascade (most frequent first) | 82.79% | 20.2 | 0.019 |
| ecoli | PFossil:cascade (rarest first) | 75.64% | 23.0 | 0.027 |
| ecoli | PFossil:decision list (per refinement) | 85.19% | 19.8 | 0.035 |
| ecoli | PFossil:decision list (per rule) | 81.02% | 16.2 | 0.064 |
| ecoli | PFossil:decision list (per seed example) | 81.33% | 13.1 | 0.013 |
| ecoli | PFossil:rule set (best rule) | 81.92% | 30.0 | 0.037 |
| ecoli | PFossil:rule set (macro distribution voting) | 82.22% | 30.0 | 0.037 |
| ecoli | PFossil:rule set (micro distribution voting) | 84.29% | 30.0 | 0.037 |
| ecoli | PFossil:rule set (voting) | 83.98% | 30.0 | 0.037 |
| ecoli | Pypper:cascade (most frequent first) | 81.61% | 8.0 | 0.058 |
| ecoli | Pypper:cascade (rarest first) | 81.01% | 7.6 | 0.053 |
| ecoli | Pypper:decision list (per refinement) | 82.18% | 10.9 | 0.033 |
| ecoli | Pypper:decision list (per rule) | 81.31% | 11.2 | 0.115 |
| ecoli | Pypper:decision list (per seed example) | 82.75% | 10.6 | 0.033 |
| ecoli | Pypper:rule set (best rule) | 82.22% | 10.7 | 0.081 |
| ecoli | Pypper:rule set (macro distribution voting) | 81.91% | 10.7 | 0.081 |
| ecoli | Pypper:rule set (micro distribution voting) | 81.60% | 10.7 | 0.081 |
| ecoli | Pypper:rule set (voting) | 81.60% | 10.7 | 0.081 |
| glass | CN2:cascade (most frequent first) | 70.61% | 21.4 | 0.249 |
| glass | CN2:cascade (rarest first) | 68.79% | 22.5 | 0.204 |
| glass | CN2:decision list (per refinement) | 70.11% | 17.2 | 0.643 |
| glass | CN2:decision list (per rule) | 72.49% | 16.2 | 1.167 |
| glass | CN2:decision list (per seed example) | 67.92% | 15.7 | 0.304 |
| glass | CN2:rule set (best rule) | 72.03% | 38.0 | 0.399 |
| glass | CN2:rule set (macro distribution voting) | 70.22% | 38.0 | 0.399 |
| glass | CN2:rule set (micro distribution voting) | 71.08% | 38.0 | 0.399 |
| glass | CN2:rule set (voting) | 70.63% | 38.0 | 0.399 |
| glass | PFossil:cascade (most frequent first) | 68.29% | 16.8 | 0.018 |
| glass | PFossil:cascade (rarest first) | 63.55% | 15.6 | 0.017 |
| glass | PFossil:decision list (per refinement) | 68.77% | 17.1 | 0.029 |
| glass | PFossil:decision list (per rule) | 63.57% | 14.1 | 0.048 |
| glass | PFossil:decision list (per seed example) | 64.07% | 10.0 | 0.011 |
| glass | PFossil:rule set (best rule) | 71.97% | 31.2 | 0.036 |
| glass | PFossil:rule set (macro distribution voting) | 70.58% | 31.2 | 0.036 |
| glass | PFossil:rule set (micro distribution voting) | 70.11% | 31.2 | 0.036 |
| glass | PFossil:rule set (voting) | 69.68% | 31.2 | 0.036 |
| glass | Pypper:cascade (most frequent first) | 68.16% | 8.9 | 0.058 |
| glass | Pypper:cascade (rarest first) | 67.38% | 6.2 | 0.049 |
| glass | Pypper:decision list (per refinement) | 65.89% | 9.0 | 0.026 |
| glass | Pypper:decision list (per rule) | 67.73% | 7.6 | 0.071 |
| glass | Pypper:decision list (per seed example) | 65.95% | 11.5 | 0.032 |
| glass | Pypper:rule set (best rule) | 67.77% | 9.6 | 0.085 |
| glass | Pypper:rule set (macro distribution voting) | 65.91% | 9.6 | 0.085 |
| glass | Pypper:rule set (micro distribution voting) | 67.32% | 9.6 | 0.085 |
| glass | Pypper:rule set (voting) | 67.27% | 9.6 | 0.085 |
| hayes-roth | CN2:cascade (most frequent first) | 80.62% | 11.4 | 0.057 |
| hayes-roth | CN2:cascade (rarest first) | 82.50% | 10.7 | 0.046 |
| hayes-roth | CN2:decision list (per refinement) | 82.50% | 9.8 | 0.062 |
| hayes-roth | CN2:decision list (per rule) | 82.50% | 10.4 | 0.133 |
| hayes-roth | CN2:decision list (per seed example) | 74.38% | 9.6 | 0.139 |
| hayes-roth | CN2:rule set (best rule) | 81.88% | 23.8 | 0.106 |
| hayes-roth | CN2:rule set (macro distribution voting) | 80.00% | 23.8 | 0.106 |
| hayes-roth | CN2:rule set (micro distribution voting) | 80.62% | 23.8 | 0.106 |
| hayes-roth | CN2:rule set (voting) | 79.38% | 23.8 | 0.106 |
| hayes-roth | PFossil:cascade (most frequent first) | 81.25% | 5.9 | 0.006 |
| hayes-roth | PFossil:cascade (rarest first) | 85.62% | 7.6 | 0.006 |
| hayes-roth | PFossil:decision list (per refinement) | 84.38% | 8.7 | 0.010 |
| hayes-roth | PFossil:decision list (per rule) | 82.50% | 7.3 | 0.013 |
| hayes-roth | PFossil:decision list (per seed example) | 66.25% | 5.3 | 0.009 |
| hayes-roth | PFossil:rule set (best rule) | 82.50% | 13.5 | 0.013 |
| hayes-roth | PFossil:rule set (macro distribution voting) | 81.88% | 13.5 | 0.013 |
| hayes-roth | PFossil:rule set (micro distribution voting) | 81.88% | 13.5 | 0.013 |
| hayes-roth | PFossil:rule set (voting) | 81.25% | 13.5 | 0.013 |
| hayes-roth | Pypper:cascade (most frequent first) | 76.25% | 4.8 | 0.027 |
| hayes-roth | Pypper:cascade (rarest first) | 84.38% | 6.1 | 0.028 |
| hayes-roth | Pypper:decision list (per refinement) | 78.12% | 7.8 | 0.018 |
| hayes-roth | Pypper:decision list (per rule) | 81.88% | 9.0 | 0.033 |
| hayes-roth | Pypper:decision list (per seed example) | 70.62% | 9.1 | 0.023 |
| hayes-roth | Pypper:rule set (best rule) | 80.62% | 10.3 | 0.054 |
| hayes-roth | Pypper:rule set (macro distribution voting) | 79.38% | 10.3 | 0.054 |
| hayes-roth | Pypper:rule set (micro distribution voting) | 74.38% | 10.3 | 0.054 |
| hayes-roth | Pypper:rule set (voting) | 73.12% | 10.3 | 0.054 |
| heart-c | CN2:concept (majority class) | 75.56% | 25.1 | 0.099 |
| heart-c | CN2:concept (minority class) | 77.95% | 26.6 | 0.124 |
| heart-c | CN2:decision list (per refinement) | 76.90% | 19.7 | 0.109 |
| heart-c | CN2:decision list (per rule) | 80.55% | 21.7 | 0.174 |
| heart-c | CN2:decision list (per seed example) | 75.91% | 21.0 | 0.133 |
| heart-c | CN2:rule set (best rule) | 82.20% | 51.7 | 0.223 |
| heart-c | CN2:rule set (macro distribution voting) | 80.56% | 51.7 | 0.223 |
| heart-c | CN2:rule set (micro distribution voting) | 82.22% | 51.7 | 0.223 |
| heart-c | CN2:rule set (voting) | 80.23% | 51.7 | 0.223 |
| heart-c | PFossil:concept (majority class) | 78.27% | 7.9 | 0.010 |
| heart-c | PFossil:concept (minority class) | 72.95% | 10.3 | 0.015 |
| heart-c | PFossil:decision list (per refinement) | 75.59% | 6.7 | 0.015 |
| heart-c | PFossil:decision list (per rule) | 75.92% | 9.3 | 0.020 |
| heart-c | PFossil:decision list (per seed example) | 75.29% | 6.7 | 0.008 |
| heart-c | PFossil:rule set (best rule) | 79.54% | 18.2 | 0.024 |
| heart-c | PFossil:rule set (macro distribution voting) | 79.88% | 18.2 | 0.024 |
| heart-c | PFossil:rule set (micro distribution voting) | 77.24% | 18.2 | 0.024 |
| heart-c | PFossil:rule set (voting) | 78.24% | 18.2 | 0.024 |
| heart-c | Pypper:concept (majority class) | 82.60% | 2.7 | 0.031 |
| heart-c | Pypper:concept (minority class) | 79.23% | 3.4 | 0.034 |
| heart-c | Pypper:decision list (per refinement) | 80.87% | 5.4 | 0.017 |
| heart-c | Pypper:decision list (per rule) | 82.91% | 5.2 | 0.022 |
| heart-c | Pypper:decision list (per seed example) | 83.53% | 7.8 | 0.020 |
| heart-c | Pypper:rule set (best rule) | 80.22% | 6.1 | 0.060 |
| heart-c | Pypper:rule set (macro distribution voting) | 79.88% | 6.1 | 0.060 |
| heart-c | Pypper:rule set (micro distribution voting) | 82.22% | 6.1 | 0.060 |
| heart-c | Pypper:rule set (voting) | 81.89% | 6.1 | 0.060 |
| heart-h | CN2:concept (majority class) | 78.60% | 22.4 | 0.105 |
| heart-h | CN2:concept (minority class) | 76.20% | 27.7 | 0.178 |
| heart-h | CN2:decision list (per refinement) | 77.60% | 15.7 | 0.105 |
| heart-h | CN2:decision list (per rule) | 77.61% | 20.1 | 0.208 |
| heart-h | CN2:decision list (per seed example) | 79.61% | 20.5 | 0.334 |
| heart-h | CN2:rule set (best rule) | 78.92% | 50.1 | 0.280 |
| heart-h | CN2:rule set (macro distribution voting) | 79.94% | 50.1 | 0.280 |
| heart-h | CN2:rule set (micro distribution voting) | 79.25% | 50.1 | 0.280 |
| heart-h | CN2:rule set (voting) | 79.94% | 50.1 | 0.280 |
| heart-h | PFossil:concept (majority class) | 81.28% | 9.4 | 0.010 |
| heart-h | PFossil:concept (minority class) | 75.15% | 7.8 | 0.010 |
| heart-h | PFossil:decision list (per refinement) | 75.17% | 7.0 | 0.015 |
| heart-h | PFossil:decision list (per rule) | 76.89% | 4.6 | 0.011 |
| heart-h | PFossil:decision list (per seed example) | 75.87% | 5.2 | 0.008 |
| heart-h | PFossil:rule set (best rule) | 81.28% | 17.2 | 0.020 |
| heart-h | PFossil:rule set (macro distribution voting) | 79.92% | 17.2 | 0.020 |
| heart-h | PFossil:rule set (micro distribution voting) | 78.93% | 17.2 | 0.020 |
| heart-h | PFossil:rule set (voting) | 80.94% | 17.2 | 0.020 |
| heart-h | Pypper:concept (majority class) | 76.14% | 3.1 | 0.028 |
| heart-h | Pypper:concept (minority class) | 76.82% | 1.8 | 0.019 |
| heart-h | Pypper:decision list (per refinement) | 79.90% | 3.4 | 0.013 |
| heart-h | Pypper:decision list (per rule) | 78.23% | 5.7 | 0.023 |
| heart-h | Pypper:decision list (per seed example) | 77.18% | 6.3 | 0.021 |
| heart-h | Pypper:rule set (best rule) | 77.17% | 4.9 | 0.049 |
| heart-h | Pypper:rule set (macro distribution voting) | 77.17% | 4.9 | 0.049 |
| heart-h | Pypper:rule set (micro distribution voting) | 76.83% | 4.9 | 0.049 |
| heart-h | Pypper:rule set (voting) | 76.83% | 4.9 | 0.049 |
| heart-statlog | CN2:concept (majority class) | 72.96% | 23.1 | 0.098 |
| heart-statlog | CN2:concept (minority class) | 74.44% | 24.6 | 0.116 |
| heart-statlog | CN2:decision list (per refinement) | 75.19% | 16.2 | 0.085 |
| heart-statlog | CN2:decision list (per rule) | 79.63% | 19.4 | 0.152 |
| heart-statlog | CN2:decision list (per seed example) | 75.56% | 17.4 | 0.111 |
| heart-statlog | CN2:rule set (best rule) | 77.78% | 47.7 | 0.214 |
| heart-statlog | CN2:rule set (macro distribution voting) | 78.52% | 47.7 | 0.214 |
| heart-statlog | CN2:rule set (micro distribution voting) | 78.52% | 47.7 | 0.214 |
| heart-statlog | CN2:rule set (voting) | 77.78% | 47.7 | 0.214 |
| heart-statlog | PFossil:concept (majority class) | 78.52% | 8.1 | 0.010 |
| heart-statlog | PFossil:concept (minority class) | 75.19% | 9.5 | 0.014 |
| heart-statlog | PFossil:decision list (per refinement) | 77.78% | 7.1 | 0.017 |
| heart-statlog | PFossil:decision list (per rule) | 78.15% | 7.3 | 0.015 |
| heart-statlog | PFossil:decision list (per seed example) | 80.37% | 7.0 | 0.009 |
| heart-statlog | PFossil:rule set (best rule) | 80.74% | 17.6 | 0.023 |
| heart-statlog | PFossil:rule set (macro distribution voting) | 79.26% | 17.6 | 0.023 |
| heart-statlog | PFossil:rule set (micro distribution voting) | 80.00% | 17.6 | 0.023 |
| heart-statlog | PFossil:rule set (voting) | 79.26% | 17.6 | 0.023 |
| heart-statlog | Pypper:concept (majority class) | 83.33% | 2.6 | 0.026 |
| heart-statlog | Pypper:concept (minority class) | 81.48% | 2.9 | 0.026 |
| heart-statlog | Pypper:decision list (per refinement) | 79.26% | 4.9 | 0.016 |
| heart-statlog | Pypper:decision list (per rule) | 80.00% | 5.0 | 0.021 |
| heart-statlog | Pypper:decision list (per seed example) | 80.37% | 7.6 | 0.021 |
| heart-statlog | Pypper:rule set (best rule) | 81.85% | 5.5 | 0.052 |
| heart-statlog | Pypper:rule set (macro distribution voting) | 82.22% | 5.5 | 0.052 |
| heart-statlog | Pypper:rule set (micro distribution voting) | 82.59% | 5.5 | 0.052 |
| heart-statlog | Pypper:rule set (voting) | 82.59% | 5.5 | 0.052 |
| heloc | CN2:concept (majority class) | 58.33% | 292.7 | 4.440 |
| heloc | CN2:concept (minority class) | 61.86% | 467.9 | 8.216 |
| heloc | CN2:decision list (per refinement) | 69.67% | 175.6 | 3.769 |
| heloc | CN2:decision list (per rule) | 70.05% | 275.7 | 5.776 |
| heloc | CN2:decision list (per seed example) | 64.81% | 576.7 | 10.537 |
| heloc | CN2:rule set (best rule) | 71.37% | 760.6 | 12.680 |
| heloc | CN2:rule set (macro distribution voting) | 71.64% | 760.6 | 12.680 |
| heloc | CN2:rule set (micro distribution voting) | 70.78% | 760.6 | 12.680 |
| heloc | CN2:rule set (voting) | 71.55% | 760.6 | 12.680 |
| heloc | PFossil:concept (majority class) | 68.83% | 1.0 | 0.020 |
| heloc | PFossil:concept (minority class) | 69.17% | 1.0 | 0.014 |
| heloc | PFossil:decision list (per refinement) | 68.83% | 1.0 | 0.066 |
| heloc | PFossil:decision list (per rule) | 68.83% | 1.0 | 0.031 |
| heloc | PFossil:decision list (per seed example) | 65.42% | 10.0 | 14.681 |
| heloc | PFossil:rule set (best rule) | 68.83% | 2.0 | 0.044 |
| heloc | PFossil:rule set (macro distribution voting) | 68.83% | 2.0 | 0.044 |
| heloc | PFossil:rule set (micro distribution voting) | 68.83% | 2.0 | 0.044 |
| heloc | PFossil:rule set (voting) | 68.83% | 2.0 | 0.044 |
| heloc | Pypper:concept (majority class) | 69.80% | 7.2 | 0.552 |
| heloc | Pypper:concept (minority class) | 70.13% | 8.5 | 0.568 |
| heloc | Pypper:decision list (per refinement) | 70.09% | 24.8 | 0.601 |
| heloc | Pypper:decision list (per rule) | 70.26% | 33.5 | 0.618 |
| heloc | Pypper:decision list (per seed example) | 69.62% | 44.7 | 0.519 |
| heloc | Pypper:rule set (best rule) | 69.81% | 15.7 | 1.128 |
| heloc | Pypper:rule set (macro distribution voting) | 69.87% | 15.7 | 1.128 |
| heloc | Pypper:rule set (micro distribution voting) | 69.88% | 15.7 | 1.128 |
| heloc | Pypper:rule set (voting) | 69.75% | 15.7 | 1.128 |
| hepatitis | CN2:concept (majority class) | 76.79% | 10.7 | 0.035 |
| hepatitis | CN2:concept (minority class) | 74.42% | 11.5 | 0.039 |
| hepatitis | CN2:decision list (per refinement) | 82.08% | 9.3 | 0.039 |
| hepatitis | CN2:decision list (per rule) | 77.62% | 10.1 | 0.060 |
| hepatitis | CN2:decision list (per seed example) | 78.17% | 9.2 | 0.044 |
| hepatitis | CN2:rule set (best rule) | 82.04% | 22.2 | 0.076 |
| hepatitis | CN2:rule set (macro distribution voting) | 81.38% | 22.2 | 0.076 |
| hepatitis | CN2:rule set (micro distribution voting) | 82.67% | 22.2 | 0.076 |
| hepatitis | CN2:rule set (voting) | 82.00% | 22.2 | 0.076 |
| hepatitis | PFossil:concept (majority class) | 76.67% | 8.2 | 0.007 |
| hepatitis | PFossil:concept (minority class) | 73.79% | 6.8 | 0.007 |
| hepatitis | PFossil:decision list (per refinement) | 75.50% | 6.0 | 0.009 |
| hepatitis | PFossil:decision list (per rule) | 76.83% | 6.2 | 0.010 |
| hepatitis | PFossil:decision list (per seed example) | 75.58% | 5.5 | 0.006 |
| hepatitis | PFossil:rule set (best rule) | 78.75% | 15.0 | 0.015 |
| hepatitis | PFossil:rule set (macro distribution voting) | 77.54% | 15.0 | 0.015 |
| hepatitis | PFossil:rule set (micro distribution voting) | 80.04% | 15.0 | 0.015 |
| hepatitis | PFossil:rule set (voting) | 78.75% | 15.0 | 0.015 |
| hepatitis | Pypper:concept (majority class) | 72.17% | 2.1 | 0.016 |
| hepatitis | Pypper:concept (minority class) | 80.75% | 2.1 | 0.015 |
| hepatitis | Pypper:decision list (per refinement) | 78.25% | 3.4 | 0.010 |
| hepatitis | Pypper:decision list (per rule) | 77.54% | 4.8 | 0.016 |
| hepatitis | Pypper:decision list (per seed example) | 80.08% | 5.7 | 0.014 |
| hepatitis | Pypper:rule set (best rule) | 80.71% | 4.2 | 0.031 |
| hepatitis | Pypper:rule set (macro distribution voting) | 82.00% | 4.2 | 0.031 |
| hepatitis | Pypper:rule set (micro distribution voting) | 80.04% | 4.2 | 0.031 |
| hepatitis | Pypper:rule set (voting) | 80.67% | 4.2 | 0.031 |
| ilpd | CN2:concept (majority class) | 69.12% | 25.5 | 0.195 |
| ilpd | CN2:concept (minority class) | 61.22% | 54.5 | 0.482 |
| ilpd | CN2:decision list (per refinement) | 69.63% | 20.1 | 0.137 |
| ilpd | CN2:decision list (per rule) | 69.28% | 30.9 | 0.312 |
| ilpd | CN2:decision list (per seed example) | 69.29% | 47.0 | 0.799 |
| ilpd | CN2:rule set (best rule) | 68.61% | 80.0 | 0.661 |
| ilpd | CN2:rule set (macro distribution voting) | 69.13% | 80.0 | 0.661 |
| ilpd | CN2:rule set (micro distribution voting) | 69.30% | 80.0 | 0.661 |
| ilpd | CN2:rule set (voting) | 68.95% | 80.0 | 0.661 |
| ilpd | PFossil:concept (majority class) | 54.19% | 1.2 | 0.003 |
| ilpd | PFossil:concept (minority class) | 63.79% | 1.6 | 0.005 |
| ilpd | PFossil:decision list (per refinement) | 71.36% | 1.2 | 0.006 |
| ilpd | PFossil:decision list (per rule) | 63.62% | 2.2 | 0.008 |
| ilpd | PFossil:decision list (per seed example) | 64.66% | 4.1 | 0.008 |
| ilpd | PFossil:rule set (best rule) | 63.28% | 2.8 | 0.009 |
| ilpd | PFossil:rule set (macro distribution voting) | 63.28% | 2.8 | 0.009 |
| ilpd | PFossil:rule set (micro distribution voting) | 63.28% | 2.8 | 0.009 |
| ilpd | PFossil:rule set (voting) | 63.28% | 2.8 | 0.009 |
| ilpd | Pypper:concept (majority class) | 59.35% | 1.7 | 0.042 |
| ilpd | Pypper:concept (minority class) | 68.45% | 1.2 | 0.023 |
| ilpd | Pypper:decision list (per refinement) | 70.68% | 2.6 | 0.016 |
| ilpd | Pypper:decision list (per rule) | 69.28% | 10.1 | 0.047 |
| ilpd | Pypper:decision list (per seed example) | 70.32% | 13.5 | 0.040 |
| ilpd | Pypper:rule set (best rule) | 69.14% | 2.9 | 0.064 |
| ilpd | Pypper:rule set (macro distribution voting) | 69.14% | 2.9 | 0.064 |
| ilpd | Pypper:rule set (micro distribution voting) | 69.14% | 2.9 | 0.064 |
| ilpd | Pypper:rule set (voting) | 69.14% | 2.9 | 0.064 |
| ionosphere | CN2:concept (majority class) | 84.60% | 13.2 | 0.111 |
| ionosphere | CN2:concept (minority class) | 91.17% | 13.9 | 0.053 |
| ionosphere | CN2:decision list (per refinement) | 89.17% | 13.5 | 0.058 |
| ionosphere | CN2:decision list (per rule) | 87.73% | 13.1 | 0.118 |
| ionosphere | CN2:decision list (per seed example) | 87.46% | 12.6 | 0.055 |
| ionosphere | CN2:rule set (best rule) | 90.59% | 27.1 | 0.161 |
| ionosphere | CN2:rule set (macro distribution voting) | 90.02% | 27.1 | 0.161 |
| ionosphere | CN2:rule set (micro distribution voting) | 91.17% | 27.1 | 0.161 |
| ionosphere | CN2:rule set (voting) | 90.02% | 27.1 | 0.161 |
| ionosphere | PFossil:concept (majority class) | 88.60% | 3.5 | 0.009 |
| ionosphere | PFossil:concept (minority class) | 87.47% | 10.1 | 0.011 |
| ionosphere | PFossil:decision list (per refinement) | 88.03% | 3.6 | 0.016 |
| ionosphere | PFossil:decision list (per rule) | 88.60% | 3.5 | 0.012 |
| ionosphere | PFossil:decision list (per seed example) | 89.17% | 5.4 | 0.011 |
| ionosphere | PFossil:rule set (best rule) | 89.75% | 13.6 | 0.020 |
| ionosphere | PFossil:rule set (macro distribution voting) | 90.03% | 13.6 | 0.020 |
| ionosphere | PFossil:rule set (micro distribution voting) | 89.75% | 13.6 | 0.020 |
| ionosphere | PFossil:rule set (voting) | 90.03% | 13.6 | 0.020 |
| ionosphere | Pypper:concept (majority class) | 90.60% | 2.2 | 0.028 |
| ionosphere | Pypper:concept (minority class) | 89.17% | 4.4 | 0.028 |
| ionosphere | Pypper:decision list (per refinement) | 88.89% | 5.3 | 0.018 |
| ionosphere | Pypper:decision list (per rule) | 89.20% | 3.9 | 0.019 |
| ionosphere | Pypper:decision list (per seed example) | 87.18% | 6.4 | 0.020 |
| ionosphere | Pypper:rule set (best rule) | 90.60% | 6.6 | 0.054 |
| ionosphere | Pypper:rule set (macro distribution voting) | 89.75% | 6.6 | 0.054 |
| ionosphere | Pypper:rule set (micro distribution voting) | 90.03% | 6.6 | 0.054 |
| ionosphere | Pypper:rule set (voting) | 90.32% | 6.6 | 0.054 |
| iris | CN2:cascade (most frequent first) | 92.67% | 4.6 | 0.013 |
| iris | CN2:cascade (rarest first) | 92.67% | 4.6 | 0.013 |
| iris | CN2:decision list (per refinement) | 94.67% | 4.0 | 0.026 |
| iris | CN2:decision list (per rule) | 94.00% | 4.2 | 0.042 |
| iris | CN2:decision list (per seed example) | 88.00% | 4.2 | 0.026 |
| iris | CN2:rule set (best rule) | 92.67% | 8.7 | 0.027 |
| iris | CN2:rule set (macro distribution voting) | 92.67% | 8.7 | 0.027 |
| iris | CN2:rule set (micro distribution voting) | 92.67% | 8.7 | 0.027 |
| iris | CN2:rule set (voting) | 92.00% | 8.7 | 0.027 |
| iris | PFossil:cascade (most frequent first) | 92.67% | 3.7 | 0.004 |
| iris | PFossil:cascade (rarest first) | 92.67% | 3.7 | 0.004 |
| iris | PFossil:decision list (per refinement) | 92.67% | 4.0 | 0.005 |
| iris | PFossil:decision list (per rule) | 92.67% | 3.7 | 0.006 |
| iris | PFossil:decision list (per seed example) | 92.67% | 3.2 | 0.003 |
| iris | PFossil:rule set (best rule) | 92.00% | 5.8 | 0.006 |
| iris | PFossil:rule set (macro distribution voting) | 92.00% | 5.8 | 0.006 |
| iris | PFossil:rule set (micro distribution voting) | 92.00% | 5.8 | 0.006 |
| iris | PFossil:rule set (voting) | 91.33% | 5.8 | 0.006 |
| iris | Pypper:cascade (most frequent first) | 93.33% | 2.0 | 0.013 |
| iris | Pypper:cascade (rarest first) | 93.33% | 2.0 | 0.011 |
| iris | Pypper:decision list (per refinement) | 92.67% | 2.9 | 0.007 |
| iris | Pypper:decision list (per rule) | 92.67% | 2.7 | 0.011 |
| iris | Pypper:decision list (per seed example) | 92.00% | 2.5 | 0.023 |
| iris | Pypper:rule set (best rule) | 92.00% | 3.6 | 0.022 |
| iris | Pypper:rule set (macro distribution voting) | 92.00% | 3.6 | 0.022 |
| iris | Pypper:rule set (micro distribution voting) | 92.00% | 3.6 | 0.022 |
| iris | Pypper:rule set (voting) | 92.00% | 3.6 | 0.022 |
| kc1 | CN2:concept (majority class) | 83.88% | 49.0 | 1.773 |
| kc1 | CN2:concept (minority class) | 65.39% | 110.5 | 10.513 |
| kc1 | CN2:decision list (per refinement) | 84.35% | 52.1 | 2.284 |
| kc1 | CN2:decision list (per rule) | 84.78% | 63.0 | 5.827 |
| kc1 | CN2:decision list (per seed example) | 81.51% | 56.2 | 10.342 |
| kc1 | CN2:rule set (best rule) | 84.64% | 159.5 | 12.251 |
| kc1 | CN2:rule set (macro distribution voting) | 84.40% | 159.5 | 12.251 |
| kc1 | CN2:rule set (micro distribution voting) | 84.73% | 159.5 | 12.251 |
| kc1 | CN2:rule set (voting) | 82.36% | 159.5 | 12.251 |
| kc1 | PFossil:concept (majority class) | 80.99% | 2.5 | 0.011 |
| kc1 | PFossil:concept (minority class) | 80.51% | 1.0 | 0.006 |
| kc1 | PFossil:decision list (per refinement) | 80.51% | 1.0 | 0.013 |
| kc1 | PFossil:decision list (per rule) | 85.02% | 2.7 | 0.017 |
| kc1 | PFossil:decision list (per seed example) | 84.87% | 8.2 | 0.087 |
| kc1 | PFossil:rule set (best rule) | 83.31% | 3.5 | 0.018 |
| kc1 | PFossil:rule set (macro distribution voting) | 83.31% | 3.5 | 0.018 |
| kc1 | PFossil:rule set (micro distribution voting) | 83.31% | 3.5 | 0.018 |
| kc1 | PFossil:rule set (voting) | 83.31% | 3.5 | 0.018 |
| kc1 | Pypper:concept (majority class) | 69.90% | 1.5 | 0.096 |
| kc1 | Pypper:concept (minority class) | 84.07% | 2.6 | 0.051 |
| kc1 | Pypper:decision list (per refinement) | 84.21% | 4.6 | 0.043 |
| kc1 | Pypper:decision list (per rule) | 84.87% | 14.2 | 0.098 |
| kc1 | Pypper:decision list (per seed example) | 85.11% | 15.9 | 0.108 |
| kc1 | Pypper:rule set (best rule) | 84.49% | 4.1 | 0.147 |
| kc1 | Pypper:rule set (macro distribution voting) | 84.49% | 4.1 | 0.147 |
| kc1 | Pypper:rule set (micro distribution voting) | 84.45% | 4.1 | 0.147 |
| kc1 | Pypper:rule set (voting) | 84.49% | 4.1 | 0.147 |
| kc2 | CN2:concept (majority class) | 78.72% | 25.0 | 0.526 |
| kc2 | CN2:concept (minority class) | 80.82% | 29.5 | 1.951 |
| kc2 | CN2:decision list (per refinement) | 81.41% | 17.3 | 0.518 |
| kc2 | CN2:decision list (per rule) | 79.68% | 24.9 | 0.935 |
| kc2 | CN2:decision list (per seed example) | 82.36% | 24.8 | 0.970 |
| kc2 | CN2:rule set (best rule) | 82.16% | 54.5 | 2.467 |
| kc2 | CN2:rule set (macro distribution voting) | 82.74% | 54.5 | 2.467 |
| kc2 | CN2:rule set (micro distribution voting) | 82.35% | 54.5 | 2.467 |
| kc2 | CN2:rule set (voting) | 81.77% | 54.5 | 2.467 |
| kc2 | PFossil:concept (majority class) | 82.35% | 4.8 | 0.008 |
| kc2 | PFossil:concept (minority class) | 77.38% | 2.6 | 0.007 |
| kc2 | PFossil:decision list (per refinement) | 77.38% | 2.6 | 0.011 |
| kc2 | PFossil:decision list (per rule) | 79.67% | 5.1 | 0.014 |
| kc2 | PFossil:decision list (per seed example) | 81.79% | 5.6 | 0.025 |
| kc2 | PFossil:rule set (best rule) | 82.74% | 7.4 | 0.016 |
| kc2 | PFossil:rule set (macro distribution voting) | 82.35% | 7.4 | 0.016 |
| kc2 | PFossil:rule set (micro distribution voting) | 81.79% | 7.4 | 0.016 |
| kc2 | PFossil:rule set (voting) | 82.55% | 7.4 | 0.016 |
| kc2 | Pypper:concept (majority class) | 79.09% | 1.5 | 0.037 |
| kc2 | Pypper:concept (minority class) | 83.50% | 2.2 | 0.030 |
| kc2 | Pypper:decision list (per refinement) | 82.37% | 3.5 | 0.019 |
| kc2 | Pypper:decision list (per rule) | 81.20% | 7.3 | 0.035 |
| kc2 | Pypper:decision list (per seed example) | 83.31% | 8.6 | 0.033 |
| kc2 | Pypper:rule set (best rule) | 83.69% | 3.7 | 0.068 |
| kc2 | Pypper:rule set (macro distribution voting) | 83.69% | 3.7 | 0.068 |
| kc2 | Pypper:rule set (micro distribution voting) | 83.69% | 3.7 | 0.068 |
| kc2 | Pypper:rule set (voting) | 83.69% | 3.7 | 0.068 |
| kr-vs-kp | CN2:concept (majority class) | 98.47% | 37.3 | 0.239 |
| kr-vs-kp | CN2:concept (minority class) | 99.37% | 31.3 | 0.140 |
| kr-vs-kp | CN2:decision list (per refinement) | 99.09% | 30.8 | 0.213 |
| kr-vs-kp | CN2:decision list (per rule) | 99.34% | 30.0 | 0.179 |
| kr-vs-kp | CN2:decision list (per seed example) | 98.62% | 24.5 | 0.180 |
| kr-vs-kp | CN2:rule set (best rule) | 99.34% | 68.6 | 0.382 |
| kr-vs-kp | CN2:rule set (macro distribution voting) | 98.87% | 68.6 | 0.382 |
| kr-vs-kp | CN2:rule set (micro distribution voting) | 99.34% | 68.6 | 0.382 |
| kr-vs-kp | CN2:rule set (voting) | 98.87% | 68.6 | 0.382 |
| kr-vs-kp | PFossil:concept (majority class) | 95.68% | 4.8 | 0.016 |
| kr-vs-kp | PFossil:concept (minority class) | 98.25% | 9.7 | 0.022 |
| kr-vs-kp | PFossil:decision list (per refinement) | 94.40% | 4.5 | 0.022 |
| kr-vs-kp | PFossil:decision list (per rule) | 97.25% | 5.5 | 0.020 |
| kr-vs-kp | PFossil:decision list (per seed example) | 87.55% | 5.9 | 0.072 |
| kr-vs-kp | PFossil:rule set (best rule) | 99.37% | 14.5 | 0.041 |
| kr-vs-kp | PFossil:rule set (macro distribution voting) | 99.41% | 14.5 | 0.041 |
| kr-vs-kp | PFossil:rule set (micro distribution voting) | 94.31% | 14.5 | 0.041 |
| kr-vs-kp | PFossil:rule set (voting) | 96.43% | 14.5 | 0.041 |
| kr-vs-kp | Pypper:concept (majority class) | 98.84% | 9.2 | 0.092 |
| kr-vs-kp | Pypper:concept (minority class) | 99.12% | 14.3 | 0.130 |
| kr-vs-kp | Pypper:decision list (per refinement) | 99.28% | 17.6 | 0.094 |
| kr-vs-kp | Pypper:decision list (per rule) | 98.78% | 13.3 | 0.090 |
| kr-vs-kp | Pypper:decision list (per seed example) | 98.28% | 17.3 | 0.075 |
| kr-vs-kp | Pypper:rule set (best rule) | 99.25% | 23.5 | 0.219 |
| kr-vs-kp | Pypper:rule set (macro distribution voting) | 99.22% | 23.5 | 0.219 |
| kr-vs-kp | Pypper:rule set (micro distribution voting) | 99.16% | 23.5 | 0.219 |
| kr-vs-kp | Pypper:rule set (voting) | 99.00% | 23.5 | 0.219 |
| kropt | CN2:cascade (most frequent first) | 69.90% | 2072.4 | 34.611 |
| kropt | CN2:cascade (rarest first) | 80.35% | 1602.4 | 27.594 |
| kropt | CN2:decision list (per refinement) | 68.94% | 1257.7 | 117.237 |
| kropt | CN2:decision list (per rule) | 71.44% | 1124.2 | 226.836 |
| kropt | CN2:decision list (per seed example) | 61.73% | 874.7 | 21.822 |
| kropt | CN2:rule set (best rule) | 79.79% | 3018.3 | 62.828 |
| kropt | CN2:rule set (macro distribution voting) | 77.83% | 3018.3 | 62.828 |
| kropt | CN2:rule set (micro distribution voting) | 77.28% | 3018.3 | 62.828 |
| kropt | CN2:rule set (voting) | 73.22% | 3018.3 | 62.828 |
| kropt | PFossil:cascade (most frequent first) | 35.70% | 51.8 | 0.337 |
| kropt | PFossil:cascade (rarest first) | 34.75% | 63.3 | 1.136 |
| kropt | PFossil:decision list (per refinement) | 33.13% | 21.0 | 0.816 |
| kropt | PFossil:decision list (per rule) | 37.85% | 81.7 | 4.404 |
| kropt | PFossil:decision list (per seed example) | 37.92% | 126.7 | 4.470 |
| kropt | PFossil:rule set (best rule) | 24.68% | 46.4 | 0.980 |
| kropt | PFossil:rule set (macro distribution voting) | 24.54% | 46.4 | 0.980 |
| kropt | PFossil:rule set (micro distribution voting) | 22.30% | 46.4 | 0.980 |
| kropt | PFossil:rule set (voting) | 22.58% | 46.4 | 0.980 |
| kropt | Pypper:cascade (most frequent first) | 52.46% | 328.9 | 14.170 |
| kropt | Pypper:cascade (rarest first) | 60.35% | 398.6 | 18.215 |
| kropt | Pypper:decision list (per refinement) | 22.32% | 50.2 | 3.470 |
| kropt | Pypper:decision list (per rule) | 20.89% | 18.7 | 6.852 |
| kropt | Pypper:decision list (per seed example) | 60.28% | 1041.1 | 30.381 |
| kropt | Pypper:rule set (best rule) | 37.85% | 229.9 | 14.983 |
| kropt | Pypper:rule set (macro distribution voting) | 37.84% | 229.9 | 14.983 |
| kropt | Pypper:rule set (micro distribution voting) | 37.54% | 229.9 | 14.983 |
| kropt | Pypper:rule set (voting) | 37.58% | 229.9 | 14.983 |
| led24 | CN2:cascade (most frequent first) | 60.50% | 200.3 | 1.755 |
| led24 | CN2:cascade (rarest first) | 61.03% | 205.7 | 1.820 |
| led24 | CN2:decision list (per refinement) | 66.41% | 141.8 | 3.076 |
| led24 | CN2:decision list (per rule) | 64.50% | 140.1 | 9.375 |
| led24 | CN2:decision list (per seed example) | 65.06% | 123.1 | 0.945 |
| led24 | CN2:rule set (best rule) | 62.69% | 321.4 | 3.227 |
| led24 | CN2:rule set (macro distribution voting) | 61.16% | 321.4 | 3.227 |
| led24 | CN2:rule set (micro distribution voting) | 64.66% | 321.4 | 3.227 |
| led24 | CN2:rule set (voting) | 62.03% | 321.4 | 3.227 |
| led24 | PFossil:cascade (most frequent first) | 68.97% | 44.3 | 0.106 |
| led24 | PFossil:cascade (rarest first) | 70.03% | 48.5 | 0.117 |
| led24 | PFossil:decision list (per refinement) | 71.34% | 33.4 | 0.143 |
| led24 | PFossil:decision list (per rule) | 70.59% | 47.7 | 0.359 |
| led24 | PFossil:decision list (per seed example) | 65.50% | 29.5 | 0.066 |
| led24 | PFossil:rule set (best rule) | 70.00% | 45.4 | 0.155 |
| led24 | PFossil:rule set (macro distribution voting) | 70.28% | 45.4 | 0.155 |
| led24 | PFossil:rule set (micro distribution voting) | 70.56% | 45.4 | 0.155 |
| led24 | PFossil:rule set (voting) | 70.34% | 45.4 | 0.155 |
| led24 | Pypper:cascade (most frequent first) | 71.16% | 29.3 | 0.403 |
| led24 | Pypper:cascade (rarest first) | 70.88% | 30.9 | 0.399 |
| led24 | Pypper:decision list (per refinement) | 67.56% | 24.7 | 0.223 |
| led24 | Pypper:decision list (per rule) | 64.44% | 10.0 | 0.399 |
| led24 | Pypper:decision list (per seed example) | 67.66% | 74.2 | 0.370 |
| led24 | Pypper:rule set (best rule) | 68.25% | 33.9 | 0.597 |
| led24 | Pypper:rule set (macro distribution voting) | 68.22% | 33.9 | 0.597 |
| led24 | Pypper:rule set (micro distribution voting) | 68.28% | 33.9 | 0.597 |
| led24 | Pypper:rule set (voting) | 68.12% | 33.9 | 0.597 |
| letter | CN2:cascade (most frequent first) | 86.54% | 827.2 | 21.228 |
| letter | CN2:cascade (rarest first) | 86.27% | 897.2 | 22.317 |
| letter | CN2:decision list (per refinement) | 87.75% | 472.9 | 91.254 |
| letter | CN2:decision list (per rule) | 87.76% | 386.8 | 332.558 |
| letter | CN2:decision list (per seed example) | 85.52% | 345.8 | 17.711 |
| letter | CN2:rule set (best rule) | 87.56% | 1318.7 | 37.979 |
| letter | CN2:rule set (macro distribution voting) | 87.39% | 1318.7 | 37.979 |
| letter | CN2:rule set (micro distribution voting) | 87.47% | 1318.7 | 37.979 |
| letter | CN2:rule set (voting) | 87.26% | 1318.7 | 37.979 |
| letter | PFossil:cascade (most frequent first) | 78.96% | 180.4 | 1.593 |
| letter | PFossil:cascade (rarest first) | 77.04% | 180.0 | 1.583 |
| letter | PFossil:decision list (per refinement) | 72.04% | 77.4 | 2.614 |
| letter | PFossil:decision list (per rule) | 81.45% | 288.3 | 17.015 |
| letter | PFossil:decision list (per seed example) | 75.39% | 203.6 | 2.171 |
| letter | PFossil:rule set (best rule) | 76.22% | 142.3 | 2.162 |
| letter | PFossil:rule set (macro distribution voting) | 76.44% | 142.3 | 2.162 |
| letter | PFossil:rule set (micro distribution voting) | 74.39% | 142.3 | 2.162 |
| letter | PFossil:rule set (voting) | 75.29% | 142.3 | 2.162 |
| letter | Pypper:cascade (most frequent first) | 83.81% | 330.0 | 10.205 |
| letter | Pypper:cascade (rarest first) | 83.72% | 308.2 | 9.721 |
| letter | Pypper:decision list (per refinement) | 73.75% | 204.9 | 8.190 |
| letter | Pypper:decision list (per rule) | 70.96% | 111.2 | 30.323 |
| letter | Pypper:decision list (per seed example) | 83.46% | 455.4 | 7.912 |
| letter | Pypper:rule set (best rule) | 81.02% | 378.0 | 18.419 |
| letter | Pypper:rule set (macro distribution voting) | 81.26% | 378.0 | 18.419 |
| letter | Pypper:rule set (micro distribution voting) | 80.67% | 378.0 | 18.419 |
| letter | Pypper:rule set (voting) | 80.88% | 378.0 | 18.419 |
| lymph | CN2:cascade (most frequent first) | 84.52% | 8.6 | 0.029 |
| lymph | CN2:cascade (rarest first) | 77.14% | 10.3 | 0.028 |
| lymph | CN2:decision list (per refinement) | 83.90% | 8.5 | 0.070 |
| lymph | CN2:decision list (per rule) | 87.24% | 8.2 | 0.084 |
| lymph | CN2:decision list (per seed example) | 86.52% | 8.6 | 0.037 |
| lymph | CN2:rule set (best rule) | 83.95% | 18.5 | 0.060 |
| lymph | CN2:rule set (macro distribution voting) | 83.29% | 18.5 | 0.060 |
| lymph | CN2:rule set (micro distribution voting) | 83.95% | 18.5 | 0.060 |
| lymph | CN2:rule set (voting) | 83.29% | 18.5 | 0.060 |
| lymph | PFossil:cascade (most frequent first) | 79.76% | 9.6 | 0.009 |
| lymph | PFossil:cascade (rarest first) | 86.52% | 8.5 | 0.008 |
| lymph | PFossil:decision list (per refinement) | 80.48% | 9.2 | 0.014 |
| lymph | PFossil:decision list (per rule) | 86.48% | 7.0 | 0.017 |
| lymph | PFossil:decision list (per seed example) | 81.86% | 6.6 | 0.007 |
| lymph | PFossil:rule set (best rule) | 81.71% | 16.2 | 0.016 |
| lymph | PFossil:rule set (macro distribution voting) | 79.71% | 16.2 | 0.016 |
| lymph | PFossil:rule set (micro distribution voting) | 81.05% | 16.2 | 0.016 |
| lymph | PFossil:rule set (voting) | 81.10% | 16.2 | 0.016 |
| lymph | Pypper:cascade (most frequent first) | 77.67% | 4.0 | 0.024 |
| lymph | Pypper:cascade (rarest first) | 77.05% | 3.1 | 0.023 |
| lymph | Pypper:decision list (per refinement) | 75.71% | 4.5 | 0.012 |
| lymph | Pypper:decision list (per rule) | 80.43% | 5.3 | 0.028 |
| lymph | Pypper:decision list (per seed example) | 81.81% | 6.0 | 0.014 |
| lymph | Pypper:rule set (best rule) | 79.81% | 5.7 | 0.038 |
| lymph | Pypper:rule set (macro distribution voting) | 81.81% | 5.7 | 0.038 |
| lymph | Pypper:rule set (micro distribution voting) | 77.10% | 5.7 | 0.038 |
| lymph | Pypper:rule set (voting) | 77.81% | 5.7 | 0.038 |
| mofn-3-7-10 | CN2:concept (majority class) | 100.00% | 35.0 | 0.097 |
| mofn-3-7-10 | CN2:concept (minority class) | 100.00% | 21.0 | 0.090 |
| mofn-3-7-10 | CN2:decision list (per refinement) | 100.00% | 29.0 | 0.144 |
| mofn-3-7-10 | CN2:decision list (per rule) | 100.00% | 20.8 | 0.102 |
| mofn-3-7-10 | CN2:decision list (per seed example) | 98.64% | 20.7 | 0.076 |
| mofn-3-7-10 | CN2:rule set (best rule) | 100.00% | 56.0 | 0.196 |
| mofn-3-7-10 | CN2:rule set (macro distribution voting) | 100.00% | 56.0 | 0.196 |
| mofn-3-7-10 | CN2:rule set (micro distribution voting) | 100.00% | 56.0 | 0.196 |
| mofn-3-7-10 | CN2:rule set (voting) | 100.00% | 56.0 | 0.196 |
| mofn-3-7-10 | PFossil:concept (majority class) | 28.37% | 0.4 | 0.002 |
| mofn-3-7-10 | PFossil:concept (minority class) | 82.78% | 6.0 | 0.010 |
| mofn-3-7-10 | PFossil:decision list (per refinement) | 82.78% | 6.0 | 0.018 |
| mofn-3-7-10 | PFossil:decision list (per rule) | 82.78% | 6.0 | 0.014 |
| mofn-3-7-10 | PFossil:decision list (per seed example) | 84.89% | 8.2 | 0.048 |
| mofn-3-7-10 | PFossil:rule set (best rule) | 83.68% | 6.4 | 0.014 |
| mofn-3-7-10 | PFossil:rule set (macro distribution voting) | 83.76% | 6.4 | 0.014 |
| mofn-3-7-10 | PFossil:rule set (micro distribution voting) | 83.31% | 6.4 | 0.014 |
| mofn-3-7-10 | PFossil:rule set (voting) | 83.61% | 6.4 | 0.014 |
| mofn-3-7-10 | Pypper:concept (majority class) | 93.35% | 17.0 | 0.125 |
| mofn-3-7-10 | Pypper:concept (minority class) | 95.16% | 14.6 | 0.128 |
| mofn-3-7-10 | Pypper:decision list (per refinement) | 95.16% | 15.0 | 0.069 |
| mofn-3-7-10 | Pypper:decision list (per rule) | 98.11% | 14.6 | 0.067 |
| mofn-3-7-10 | Pypper:decision list (per seed example) | 96.83% | 20.3 | 0.063 |
| mofn-3-7-10 | Pypper:rule set (best rule) | 97.21% | 31.6 | 0.251 |
| mofn-3-7-10 | Pypper:rule set (macro distribution voting) | 97.43% | 31.6 | 0.251 |
| mofn-3-7-10 | Pypper:rule set (micro distribution voting) | 93.35% | 31.6 | 0.251 |
| mofn-3-7-10 | Pypper:rule set (voting) | 93.58% | 31.6 | 0.251 |
| molecular-biology_promoters | CN2:concept (majority class) | 75.55% | 5.8 | 0.020 |
| molecular-biology_promoters | CN2:concept (minority class) | 84.09% | 6.1 | 0.025 |
| molecular-biology_promoters | CN2:decision list (per refinement) | 80.36% | 7.3 | 0.023 |
| molecular-biology_promoters | CN2:decision list (per rule) | 83.09% | 6.4 | 0.032 |
| molecular-biology_promoters | CN2:decision list (per seed example) | 83.82% | 5.5 | 0.015 |
| molecular-biology_promoters | CN2:rule set (best rule) | 82.09% | 11.9 | 0.045 |
| molecular-biology_promoters | CN2:rule set (macro distribution voting) | 82.09% | 11.9 | 0.045 |
| molecular-biology_promoters | CN2:rule set (micro distribution voting) | 83.00% | 11.9 | 0.045 |
| molecular-biology_promoters | CN2:rule set (voting) | 82.09% | 11.9 | 0.045 |
| molecular-biology_promoters | PFossil:concept (majority class) | 83.18% | 4.2 | 0.005 |
| molecular-biology_promoters | PFossil:concept (minority class) | 81.27% | 3.2 | 0.004 |
| molecular-biology_promoters | PFossil:decision list (per refinement) | 80.36% | 3.0 | 0.008 |
| molecular-biology_promoters | PFossil:decision list (per rule) | 80.36% | 3.0 | 0.007 |
| molecular-biology_promoters | PFossil:decision list (per seed example) | 80.00% | 4.4 | 0.006 |
| molecular-biology_promoters | PFossil:rule set (best rule) | 82.27% | 7.4 | 0.009 |
| molecular-biology_promoters | PFossil:rule set (macro distribution voting) | 82.18% | 7.4 | 0.009 |
| molecular-biology_promoters | PFossil:rule set (micro distribution voting) | 81.36% | 7.4 | 0.009 |
| molecular-biology_promoters | PFossil:rule set (voting) | 81.27% | 7.4 | 0.009 |
| molecular-biology_promoters | Pypper:concept (majority class) | 83.18% | 2.2 | 0.014 |
| molecular-biology_promoters | Pypper:concept (minority class) | 83.18% | 2.1 | 0.015 |
| molecular-biology_promoters | Pypper:decision list (per refinement) | 81.09% | 2.3 | 0.009 |
| molecular-biology_promoters | Pypper:decision list (per rule) | 83.18% | 3.2 | 0.013 |
| molecular-biology_promoters | Pypper:decision list (per seed example) | 77.82% | 3.3 | 0.012 |
| molecular-biology_promoters | Pypper:rule set (best rule) | 81.36% | 4.3 | 0.028 |
| molecular-biology_promoters | Pypper:rule set (macro distribution voting) | 81.36% | 4.3 | 0.028 |
| molecular-biology_promoters | Pypper:rule set (micro distribution voting) | 81.18% | 4.3 | 0.028 |
| molecular-biology_promoters | Pypper:rule set (voting) | 80.27% | 4.3 | 0.028 |
| monks-problems-1 | CN2:concept (majority class) | 100.00% | 4.0 | 0.010 |
| monks-problems-1 | CN2:concept (minority class) | 100.00% | 3.9 | 0.012 |
| monks-problems-1 | CN2:decision list (per refinement) | 100.00% | 5.6 | 0.019 |
| monks-problems-1 | CN2:decision list (per rule) | 100.00% | 5.0 | 0.020 |
| monks-problems-1 | CN2:decision list (per seed example) | 83.52% | 5.1 | 0.037 |
| monks-problems-1 | CN2:rule set (best rule) | 100.00% | 7.9 | 0.020 |
| monks-problems-1 | CN2:rule set (macro distribution voting) | 100.00% | 7.9 | 0.020 |
| monks-problems-1 | CN2:rule set (micro distribution voting) | 100.00% | 7.9 | 0.020 |
| monks-problems-1 | CN2:rule set (voting) | 100.00% | 7.9 | 0.020 |
| monks-problems-1 | PFossil:concept (majority class) | 81.62% | 2.6 | 0.003 |
| monks-problems-1 | PFossil:concept (minority class) | 77.00% | 1.4 | 0.002 |
| monks-problems-1 | PFossil:decision list (per refinement) | 82.02% | 3.1 | 0.005 |
| monks-problems-1 | PFossil:decision list (per rule) | 74.64% | 1.0 | 0.002 |
| monks-problems-1 | PFossil:decision list (per seed example) | 55.27% | 1.0 | 0.030 |
| monks-problems-1 | PFossil:rule set (best rule) | 80.41% | 4.0 | 0.005 |
| monks-problems-1 | PFossil:rule set (macro distribution voting) | 80.41% | 4.0 | 0.005 |
| monks-problems-1 | PFossil:rule set (micro distribution voting) | 74.64% | 4.0 | 0.005 |
| monks-problems-1 | PFossil:rule set (voting) | 75.91% | 4.0 | 0.005 |
| monks-problems-1 | Pypper:concept (majority class) | 93.34% | 3.7 | 0.020 |
| monks-problems-1 | Pypper:concept (minority class) | 95.30% | 3.6 | 0.019 |
| monks-problems-1 | Pypper:decision list (per refinement) | 91.34% | 4.0 | 0.014 |
| monks-problems-1 | Pypper:decision list (per rule) | 90.80% | 4.3 | 0.022 |
| monks-problems-1 | Pypper:decision list (per seed example) | 79.95% | 6.7 | 0.075 |
| monks-problems-1 | Pypper:rule set (best rule) | 96.76% | 7.3 | 0.042 |
| monks-problems-1 | Pypper:rule set (macro distribution voting) | 96.76% | 7.3 | 0.042 |
| monks-problems-1 | Pypper:rule set (micro distribution voting) | 92.06% | 7.3 | 0.042 |
| monks-problems-1 | Pypper:rule set (voting) | 94.06% | 7.3 | 0.042 |
| monks-problems-2 | CN2:concept (majority class) | 97.34% | 36.2 | 0.135 |
| monks-problems-2 | CN2:concept (minority class) | 81.05% | 48.4 | 0.271 |
| monks-problems-2 | CN2:decision list (per refinement) | 98.33% | 33.5 | 0.167 |
| monks-problems-2 | CN2:decision list (per rule) | 99.50% | 32.3 | 0.192 |
| monks-problems-2 | CN2:decision list (per seed example) | 89.52% | 26.4 | 0.128 |
| monks-problems-2 | CN2:rule set (best rule) | 90.02% | 84.6 | 0.401 |
| monks-problems-2 | CN2:rule set (macro distribution voting) | 88.86% | 84.6 | 0.401 |
| monks-problems-2 | CN2:rule set (micro distribution voting) | 90.02% | 84.6 | 0.401 |
| monks-problems-2 | CN2:rule set (voting) | 88.86% | 84.6 | 0.401 |
| monks-problems-2 | PFossil:concept (majority class) | 34.28% | 0.0 | 0.001 |
| monks-problems-2 | PFossil:concept (minority class) | 66.72% | 0.4 | 0.002 |
| monks-problems-2 | PFossil:decision list (per refinement) | 65.72% | 0.0 | 0.002 |
| monks-problems-2 | PFossil:decision list (per rule) | 66.72% | 0.8 | 0.004 |
| monks-problems-2 | PFossil:decision list (per seed example) | 72.39% | 10.6 | 0.133 |
| monks-problems-2 | PFossil:rule set (best rule) | 66.72% | 0.4 | 0.004 |
| monks-problems-2 | PFossil:rule set (macro distribution voting) | 66.72% | 0.4 | 0.004 |
| monks-problems-2 | PFossil:rule set (micro distribution voting) | 66.72% | 0.4 | 0.004 |
| monks-problems-2 | PFossil:rule set (voting) | 66.72% | 0.4 | 0.004 |
| monks-problems-2 | Pypper:concept (majority class) | 77.36% | 10.1 | 0.074 |
| monks-problems-2 | Pypper:concept (minority class) | 72.22% | 4.4 | 0.052 |
| monks-problems-2 | Pypper:decision list (per refinement) | 66.72% | 6.8 | 0.027 |
| monks-problems-2 | Pypper:decision list (per rule) | 88.17% | 16.4 | 0.069 |
| monks-problems-2 | Pypper:decision list (per seed example) | 71.74% | 16.6 | 0.044 |
| monks-problems-2 | Pypper:rule set (best rule) | 74.03% | 14.5 | 0.113 |
| monks-problems-2 | Pypper:rule set (macro distribution voting) | 74.03% | 14.5 | 0.113 |
| monks-problems-2 | Pypper:rule set (micro distribution voting) | 74.36% | 14.5 | 0.113 |
| monks-problems-2 | Pypper:rule set (voting) | 74.20% | 14.5 | 0.113 |
| monks-problems-3 | CN2:concept (majority class) | 97.48% | 13.0 | 0.054 |
| monks-problems-3 | CN2:concept (minority class) | 96.57% | 10.0 | 0.045 |
| monks-problems-3 | CN2:decision list (per refinement) | 98.38% | 10.1 | 0.053 |
| monks-problems-3 | CN2:decision list (per rule) | 98.19% | 9.8 | 0.061 |
| monks-problems-3 | CN2:decision list (per seed example) | 96.75% | 10.7 | 0.090 |
| monks-problems-3 | CN2:rule set (best rule) | 97.84% | 23.0 | 0.101 |
| monks-problems-3 | CN2:rule set (macro distribution voting) | 97.84% | 23.0 | 0.101 |
| monks-problems-3 | CN2:rule set (micro distribution voting) | 97.84% | 23.0 | 0.101 |
| monks-problems-3 | CN2:rule set (voting) | 97.48% | 23.0 | 0.101 |
| monks-problems-3 | PFossil:concept (majority class) | 98.56% | 2.9 | 0.004 |
| monks-problems-3 | PFossil:concept (minority class) | 93.33% | 3.9 | 0.005 |
| monks-problems-3 | PFossil:decision list (per refinement) | 93.69% | 3.7 | 0.008 |
| monks-problems-3 | PFossil:decision list (per rule) | 98.56% | 2.9 | 0.006 |
| monks-problems-3 | PFossil:decision list (per seed example) | 96.22% | 3.2 | 0.007 |
| monks-problems-3 | PFossil:rule set (best rule) | 98.92% | 6.8 | 0.010 |
| monks-problems-3 | PFossil:rule set (macro distribution voting) | 98.92% | 6.8 | 0.010 |
| monks-problems-3 | PFossil:rule set (micro distribution voting) | 96.40% | 6.8 | 0.010 |
| monks-problems-3 | PFossil:rule set (voting) | 98.38% | 6.8 | 0.010 |
| monks-problems-3 | Pypper:concept (majority class) | 98.92% | 2.0 | 0.014 |
| monks-problems-3 | Pypper:concept (minority class) | 98.92% | 3.0 | 0.016 |
| monks-problems-3 | Pypper:decision list (per refinement) | 98.92% | 3.8 | 0.012 |
| monks-problems-3 | Pypper:decision list (per rule) | 98.92% | 2.9 | 0.012 |
| monks-problems-3 | Pypper:decision list (per seed example) | 96.22% | 3.5 | 0.022 |
| monks-problems-3 | Pypper:rule set (best rule) | 98.92% | 5.0 | 0.029 |
| monks-problems-3 | Pypper:rule set (macro distribution voting) | 98.92% | 5.0 | 0.029 |
| monks-problems-3 | Pypper:rule set (micro distribution voting) | 98.92% | 5.0 | 0.029 |
| monks-problems-3 | Pypper:rule set (voting) | 98.92% | 5.0 | 0.029 |
| mushroom | CN2:concept (majority class) | 100.00% | 14.0 | 0.072 |
| mushroom | CN2:concept (minority class) | 100.00% | 9.0 | 0.048 |
| mushroom | CN2:decision list (per refinement) | 100.00% | 16.1 | 0.133 |
| mushroom | CN2:decision list (per rule) | 100.00% | 16.1 | 0.097 |
| mushroom | CN2:decision list (per seed example) | 98.20% | 9.0 | 0.088 |
| mushroom | CN2:rule set (best rule) | 100.00% | 23.0 | 0.127 |
| mushroom | CN2:rule set (macro distribution voting) | 100.00% | 23.0 | 0.127 |
| mushroom | CN2:rule set (micro distribution voting) | 100.00% | 23.0 | 0.127 |
| mushroom | CN2:rule set (voting) | 100.00% | 23.0 | 0.127 |
| mushroom | PFossil:concept (majority class) | 99.95% | 5.4 | 0.026 |
| mushroom | PFossil:concept (minority class) | 100.00% | 6.9 | 0.031 |
| mushroom | PFossil:decision list (per refinement) | 99.95% | 5.4 | 0.061 |
| mushroom | PFossil:decision list (per rule) | 99.75% | 2.0 | 0.018 |
| mushroom | PFossil:decision list (per seed example) | 96.78% | 2.7 | 0.092 |
| mushroom | PFossil:rule set (best rule) | 99.95% | 12.3 | 0.068 |
| mushroom | PFossil:rule set (macro distribution voting) | 100.00% | 12.3 | 0.068 |
| mushroom | PFossil:rule set (micro distribution voting) | 99.95% | 12.3 | 0.068 |
| mushroom | PFossil:rule set (voting) | 99.95% | 12.3 | 0.068 |
| mushroom | Pypper:concept (majority class) | 99.98% | 4.5 | 0.078 |
| mushroom | Pypper:concept (minority class) | 99.96% | 5.2 | 0.079 |
| mushroom | Pypper:decision list (per refinement) | 99.98% | 5.5 | 0.083 |
| mushroom | Pypper:decision list (per rule) | 99.98% | 6.3 | 0.082 |
| mushroom | Pypper:decision list (per seed example) | 99.19% | 6.1 | 0.118 |
| mushroom | Pypper:rule set (best rule) | 99.94% | 9.7 | 0.164 |
| mushroom | Pypper:rule set (macro distribution voting) | 99.94% | 9.7 | 0.164 |
| mushroom | Pypper:rule set (micro distribution voting) | 99.94% | 9.7 | 0.164 |
| mushroom | Pypper:rule set (voting) | 99.94% | 9.7 | 0.164 |
| optdigits | CN2:cascade (most frequent first) | 90.89% | 160.9 | 2.534 |
| optdigits | CN2:cascade (rarest first) | 88.99% | 180.8 | 2.853 |
| optdigits | CN2:decision list (per refinement) | 92.81% | 98.3 | 3.331 |
| optdigits | CN2:decision list (per rule) | 93.63% | 83.0 | 11.783 |
| optdigits | CN2:decision list (per seed example) | 92.62% | 84.6 | 1.507 |
| optdigits | CN2:rule set (best rule) | 90.50% | 285.0 | 4.989 |
| optdigits | CN2:rule set (macro distribution voting) | 90.85% | 285.0 | 4.989 |
| optdigits | CN2:rule set (micro distribution voting) | 90.87% | 285.0 | 4.989 |
| optdigits | CN2:rule set (voting) | 90.87% | 285.0 | 4.989 |
| optdigits | PFossil:cascade (most frequent first) | 91.25% | 69.8 | 0.363 |
| optdigits | PFossil:cascade (rarest first) | 90.71% | 66.4 | 0.350 |
| optdigits | PFossil:decision list (per refinement) | 92.44% | 64.0 | 0.786 |
| optdigits | PFossil:decision list (per rule) | 92.60% | 50.8 | 1.622 |
| optdigits | PFossil:decision list (per seed example) | 92.38% | 53.8 | 0.279 |
| optdigits | PFossil:rule set (best rule) | 90.30% | 81.3 | 0.542 |
| optdigits | PFossil:rule set (macro distribution voting) | 90.39% | 81.3 | 0.542 |
| optdigits | PFossil:rule set (micro distribution voting) | 90.11% | 81.3 | 0.542 |
| optdigits | PFossil:rule set (voting) | 90.27% | 81.3 | 0.542 |
| optdigits | Pypper:cascade (most frequent first) | 92.24% | 56.9 | 1.139 |
| optdigits | Pypper:cascade (rarest first) | 91.09% | 61.2 | 1.249 |
| optdigits | Pypper:decision list (per refinement) | 91.07% | 57.9 | 0.827 |
| optdigits | Pypper:decision list (per rule) | 91.41% | 49.0 | 2.314 |
| optdigits | Pypper:decision list (per seed example) | 91.73% | 62.3 | 0.434 |
| optdigits | Pypper:rule set (best rule) | 91.39% | 81.2 | 2.088 |
| optdigits | Pypper:rule set (macro distribution voting) | 91.80% | 81.2 | 2.088 |
| optdigits | Pypper:rule set (micro distribution voting) | 91.51% | 81.2 | 2.088 |
| optdigits | Pypper:rule set (voting) | 91.53% | 81.2 | 2.088 |
| ozone-level-8hr | CN2:concept (majority class) | 92.62% | 57.2 | 0.709 |
| ozone-level-8hr | CN2:concept (minority class) | 89.50% | 92.7 | 1.003 |
| ozone-level-8hr | CN2:decision list (per refinement) | 92.23% | 55.8 | 0.503 |
| ozone-level-8hr | CN2:decision list (per rule) | 91.79% | 62.6 | 0.913 |
| ozone-level-8hr | CN2:decision list (per seed example) | 92.34% | 51.4 | 0.529 |
| ozone-level-8hr | CN2:rule set (best rule) | 93.57% | 149.9 | 1.707 |
| ozone-level-8hr | CN2:rule set (macro distribution voting) | 93.73% | 149.9 | 1.707 |
| ozone-level-8hr | CN2:rule set (micro distribution voting) | 93.57% | 149.9 | 1.707 |
| ozone-level-8hr | CN2:rule set (voting) | 93.80% | 149.9 | 1.707 |
| ozone-level-8hr | PFossil:concept (majority class) | 22.72% | 0.9 | 0.010 |
| ozone-level-8hr | PFossil:concept (minority class) | 93.17% | 2.0 | 0.020 |
| ozone-level-8hr | PFossil:decision list (per refinement) | 93.25% | 1.5 | 0.047 |
| ozone-level-8hr | PFossil:decision list (per rule) | 93.09% | 2.7 | 0.037 |
| ozone-level-8hr | PFossil:decision list (per seed example) | 91.60% | 10.3 | 1.174 |
| ozone-level-8hr | PFossil:rule set (best rule) | 93.33% | 2.9 | 0.032 |
| ozone-level-8hr | PFossil:rule set (macro distribution voting) | 93.33% | 2.9 | 0.032 |
| ozone-level-8hr | PFossil:rule set (micro distribution voting) | 93.33% | 2.9 | 0.032 |
| ozone-level-8hr | PFossil:rule set (voting) | 93.33% | 2.9 | 0.032 |
| ozone-level-8hr | Pypper:concept (majority class) | 84.96% | 3.3 | 0.107 |
| ozone-level-8hr | Pypper:concept (minority class) | 93.02% | 4.0 | 0.095 |
| ozone-level-8hr | Pypper:decision list (per refinement) | 92.98% | 5.4 | 0.086 |
| ozone-level-8hr | Pypper:decision list (per rule) | 93.01% | 10.8 | 0.100 |
| ozone-level-8hr | Pypper:decision list (per seed example) | 93.02% | 13.7 | 0.063 |
| ozone-level-8hr | Pypper:rule set (best rule) | 93.37% | 7.3 | 0.201 |
| ozone-level-8hr | Pypper:rule set (macro distribution voting) | 93.37% | 7.3 | 0.201 |
| ozone-level-8hr | Pypper:rule set (micro distribution voting) | 93.37% | 7.3 | 0.201 |
| ozone-level-8hr | Pypper:rule set (voting) | 93.37% | 7.3 | 0.201 |
| pc1 | CN2:concept (majority class) | 89.81% | 22.1 | 0.727 |
| pc1 | CN2:concept (minority class) | 83.31% | 36.7 | 1.736 |
| pc1 | CN2:decision list (per refinement) | 92.25% | 22.7 | 0.606 |
| pc1 | CN2:decision list (per rule) | 92.70% | 25.0 | 1.300 |
| pc1 | CN2:decision list (per seed example) | 90.08% | 26.2 | 5.026 |
| pc1 | CN2:rule set (best rule) | 91.16% | 58.8 | 2.467 |
| pc1 | CN2:rule set (macro distribution voting) | 90.35% | 58.8 | 2.467 |
| pc1 | CN2:rule set (micro distribution voting) | 91.07% | 58.8 | 2.467 |
| pc1 | CN2:rule set (voting) | 90.44% | 58.8 | 2.467 |
| pc1 | PFossil:concept (majority class) | 92.79% | 4.3 | 0.007 |
| pc1 | PFossil:concept (minority class) | 90.80% | 2.9 | 0.008 |
| pc1 | PFossil:decision list (per refinement) | 90.80% | 2.9 | 0.016 |
| pc1 | PFossil:decision list (per rule) | 92.88% | 3.2 | 0.011 |
| pc1 | PFossil:decision list (per seed example) | 93.06% | 5.0 | 0.008 |
| pc1 | PFossil:rule set (best rule) | 92.88% | 7.2 | 0.018 |
| pc1 | PFossil:rule set (macro distribution voting) | 92.79% | 7.2 | 0.018 |
| pc1 | PFossil:rule set (micro distribution voting) | 92.97% | 7.2 | 0.018 |
| pc1 | PFossil:rule set (voting) | 92.88% | 7.2 | 0.018 |
| pc1 | Pypper:concept (majority class) | 79.79% | 2.5 | 0.052 |
| pc1 | Pypper:concept (minority class) | 92.88% | 1.8 | 0.023 |
| pc1 | Pypper:decision list (per refinement) | 92.24% | 3.1 | 0.024 |
| pc1 | Pypper:decision list (per rule) | 92.25% | 6.4 | 0.038 |
| pc1 | Pypper:decision list (per seed example) | 92.61% | 9.5 | 0.052 |
| pc1 | Pypper:rule set (best rule) | 93.15% | 4.3 | 0.075 |
| pc1 | Pypper:rule set (macro distribution voting) | 93.15% | 4.3 | 0.075 |
| pc1 | Pypper:rule set (micro distribution voting) | 93.15% | 4.3 | 0.075 |
| pc1 | Pypper:rule set (voting) | 93.15% | 4.3 | 0.075 |
| pc3 | CN2:concept (majority class) | 88.10% | 39.1 | 1.041 |
| pc3 | CN2:concept (minority class) | 81.06% | 79.8 | 9.923 |
| pc3 | CN2:decision list (per refinement) | 88.48% | 46.3 | 0.929 |
| pc3 | CN2:decision list (per rule) | 87.84% | 52.3 | 1.265 |
| pc3 | CN2:decision list (per seed example) | 86.11% | 62.2 | 8.423 |
| pc3 | CN2:rule set (best rule) | 89.06% | 118.9 | 10.965 |
| pc3 | CN2:rule set (macro distribution voting) | 88.99% | 118.9 | 10.965 |
| pc3 | CN2:rule set (micro distribution voting) | 89.38% | 118.9 | 10.965 |
| pc3 | CN2:rule set (voting) | 88.74% | 118.9 | 10.965 |
| pc3 | PFossil:concept (majority class) | 77.60% | 1.0 | 0.005 |
| pc3 | PFossil:concept (minority class) | 85.09% | 1.1 | 0.008 |
| pc3 | PFossil:decision list (per refinement) | 85.09% | 1.1 | 0.015 |
| pc3 | PFossil:decision list (per rule) | 85.10% | 2.8 | 0.016 |
| pc3 | PFossil:decision list (per seed example) | 84.19% | 7.9 | 0.137 |
| pc3 | PFossil:rule set (best rule) | 85.09% | 2.1 | 0.017 |
| pc3 | PFossil:rule set (macro distribution voting) | 85.09% | 2.1 | 0.017 |
| pc3 | PFossil:rule set (micro distribution voting) | 85.09% | 2.1 | 0.017 |
| pc3 | PFossil:rule set (voting) | 85.09% | 2.1 | 0.017 |
| pc3 | Pypper:concept (majority class) | 74.14% | 1.8 | 0.083 |
| pc3 | Pypper:concept (minority class) | 88.74% | 2.1 | 0.046 |
| pc3 | Pypper:decision list (per refinement) | 89.12% | 3.6 | 0.036 |
| pc3 | Pypper:decision list (per rule) | 88.99% | 5.2 | 0.043 |
| pc3 | Pypper:decision list (per seed example) | 88.48% | 12.1 | 0.053 |
| pc3 | Pypper:rule set (best rule) | 88.61% | 3.9 | 0.125 |
| pc3 | Pypper:rule set (macro distribution voting) | 88.61% | 3.9 | 0.125 |
| pc3 | Pypper:rule set (micro distribution voting) | 88.61% | 3.9 | 0.125 |
| pc3 | Pypper:rule set (voting) | 88.61% | 3.9 | 0.125 |
| pc4 | CN2:concept (majority class) | 88.54% | 31.7 | 0.663 |
| pc4 | CN2:concept (minority class) | 86.41% | 72.4 | 0.791 |
| pc4 | CN2:decision list (per refinement) | 88.48% | 33.4 | 0.246 |
| pc4 | CN2:decision list (per rule) | 89.02% | 40.4 | 0.596 |
| pc4 | CN2:decision list (per seed example) | 90.05% | 45.4 | 2.038 |
| pc4 | CN2:rule set (best rule) | 88.96% | 104.1 | 1.448 |
| pc4 | CN2:rule set (macro distribution voting) | 89.64% | 104.1 | 1.448 |
| pc4 | CN2:rule set (micro distribution voting) | 88.89% | 104.1 | 1.448 |
| pc4 | CN2:rule set (voting) | 89.16% | 104.1 | 1.448 |
| pc4 | PFossil:concept (majority class) | 89.78% | 8.0 | 0.015 |
| pc4 | PFossil:concept (minority class) | 86.01% | 5.8 | 0.020 |
| pc4 | PFossil:decision list (per refinement) | 85.87% | 3.6 | 0.031 |
| pc4 | PFossil:decision list (per rule) | 86.01% | 5.8 | 0.030 |
| pc4 | PFossil:decision list (per seed example) | 88.62% | 5.4 | 0.018 |
| pc4 | PFossil:rule set (best rule) | 89.51% | 13.8 | 0.036 |
| pc4 | PFossil:rule set (macro distribution voting) | 89.71% | 13.8 | 0.036 |
| pc4 | PFossil:rule set (micro distribution voting) | 89.58% | 13.8 | 0.036 |
| pc4 | PFossil:rule set (voting) | 89.78% | 13.8 | 0.036 |
| pc4 | Pypper:concept (majority class) | 83.81% | 4.1 | 0.070 |
| pc4 | Pypper:concept (minority class) | 88.00% | 4.1 | 0.070 |
| pc4 | Pypper:decision list (per refinement) | 88.06% | 5.6 | 0.048 |
| pc4 | Pypper:decision list (per rule) | 87.79% | 5.6 | 0.042 |
| pc4 | Pypper:decision list (per seed example) | 88.55% | 15.5 | 0.057 |
| pc4 | Pypper:rule set (best rule) | 88.55% | 8.2 | 0.139 |
| pc4 | Pypper:rule set (macro distribution voting) | 88.55% | 8.2 | 0.139 |
| pc4 | Pypper:rule set (micro distribution voting) | 88.55% | 8.2 | 0.139 |
| pc4 | Pypper:rule set (voting) | 88.55% | 8.2 | 0.139 |
| phoneme | CN2:concept (majority class) | 76.42% | 131.6 | 1.768 |
| phoneme | CN2:concept (minority class) | 63.67% | 194.9 | 2.611 |
| phoneme | CN2:decision list (per refinement) | 83.46% | 119.9 | 2.681 |
| phoneme | CN2:decision list (per rule) | 83.94% | 139.4 | 3.141 |
| phoneme | CN2:decision list (per seed example) | 82.31% | 74.7 | 3.224 |
| phoneme | CN2:rule set (best rule) | 84.59% | 326.5 | 4.381 |
| phoneme | CN2:rule set (macro distribution voting) | 84.57% | 326.5 | 4.381 |
| phoneme | CN2:rule set (micro distribution voting) | 80.39% | 326.5 | 4.381 |
| phoneme | CN2:rule set (voting) | 82.98% | 326.5 | 4.381 |
| phoneme | PFossil:concept (majority class) | 73.56% | 1.0 | 0.006 |
| phoneme | PFossil:concept (minority class) | 77.11% | 3.4 | 0.015 |
| phoneme | PFossil:decision list (per refinement) | 77.92% | 2.9 | 0.026 |
| phoneme | PFossil:decision list (per rule) | 76.98% | 3.8 | 0.025 |
| phoneme | PFossil:decision list (per seed example) | 77.83% | 10.9 | 0.204 |
| phoneme | PFossil:rule set (best rule) | 78.35% | 4.4 | 0.026 |
| phoneme | PFossil:rule set (macro distribution voting) | 78.39% | 4.4 | 0.026 |
| phoneme | PFossil:rule set (micro distribution voting) | 78.35% | 4.4 | 0.026 |
| phoneme | PFossil:rule set (voting) | 78.46% | 4.4 | 0.026 |
| phoneme | Pypper:concept (majority class) | 82.98% | 15.1 | 0.444 |
| phoneme | Pypper:concept (minority class) | 83.18% | 15.9 | 0.499 |
| phoneme | Pypper:decision list (per refinement) | 83.27% | 29.3 | 0.343 |
| phoneme | Pypper:decision list (per rule) | 83.68% | 42.0 | 0.413 |
| phoneme | Pypper:decision list (per seed example) | 81.68% | 43.2 | 0.312 |
| phoneme | Pypper:rule set (best rule) | 83.44% | 31.0 | 0.955 |
| phoneme | Pypper:rule set (macro distribution voting) | 83.46% | 31.0 | 0.955 |
| phoneme | Pypper:rule set (micro distribution voting) | 83.40% | 31.0 | 0.955 |
| phoneme | Pypper:rule set (voting) | 83.49% | 31.0 | 0.955 |
| primary-tumor | CN2:cascade (most frequent first) | 31.85% | 55.6 | 0.547 |
| primary-tumor | CN2:cascade (rarest first) | 29.79% | 61.3 | 0.585 |
| primary-tumor | CN2:decision list (per refinement) | 37.77% | 25.9 | 0.830 |
| primary-tumor | CN2:decision list (per rule) | 38.65% | 25.6 | 3.310 |
| primary-tumor | CN2:decision list (per seed example) | 40.70% | 25.1 | 0.243 |
| primary-tumor | CN2:rule set (best rule) | 40.41% | 85.8 | 0.869 |
| primary-tumor | CN2:rule set (macro distribution voting) | 39.80% | 85.8 | 0.869 |
| primary-tumor | CN2:rule set (micro distribution voting) | 41.89% | 85.8 | 0.869 |
| primary-tumor | CN2:rule set (voting) | 40.42% | 85.8 | 0.869 |
| primary-tumor | PFossil:cascade (most frequent first) | 36.00% | 55.3 | 0.063 |
| primary-tumor | PFossil:cascade (rarest first) | 32.44% | 70.4 | 0.097 |
| primary-tumor | PFossil:decision list (per refinement) | 37.76% | 49.1 | 0.125 |
| primary-tumor | PFossil:decision list (per rule) | 38.34% | 58.3 | 0.589 |
| primary-tumor | PFossil:decision list (per seed example) | 38.05% | 35.8 | 0.043 |
| primary-tumor | PFossil:rule set (best rule) | 43.38% | 66.3 | 0.106 |
| primary-tumor | PFossil:rule set (macro distribution voting) | 41.61% | 66.3 | 0.106 |
| primary-tumor | PFossil:rule set (micro distribution voting) | 41.01% | 66.3 | 0.106 |
| primary-tumor | PFossil:rule set (voting) | 41.01% | 66.3 | 0.106 |
| primary-tumor | Pypper:cascade (most frequent first) | 35.70% | 11.5 | 0.130 |
| primary-tumor | Pypper:cascade (rarest first) | 38.66% | 8.6 | 0.109 |
| primary-tumor | Pypper:decision list (per refinement) | 35.98% | 7.6 | 0.046 |
| primary-tumor | Pypper:decision list (per rule) | 36.27% | 5.9 | 0.208 |
| primary-tumor | Pypper:decision list (per seed example) | 38.02% | 25.2 | 0.173 |
| primary-tumor | Pypper:rule set (best rule) | 37.47% | 9.1 | 0.123 |
| primary-tumor | Pypper:rule set (macro distribution voting) | 36.88% | 9.1 | 0.123 |
| primary-tumor | Pypper:rule set (micro distribution voting) | 38.07% | 9.1 | 0.123 |
| primary-tumor | Pypper:rule set (voting) | 36.59% | 9.1 | 0.123 |
| qsar-biodeg | CN2:concept (majority class) | 83.90% | 66.7 | 3.050 |
| qsar-biodeg | CN2:concept (minority class) | 77.18% | 67.0 | 0.798 |
| qsar-biodeg | CN2:decision list (per refinement) | 79.62% | 46.8 | 0.296 |
| qsar-biodeg | CN2:decision list (per rule) | 83.61% | 49.4 | 0.653 |
| qsar-biodeg | CN2:decision list (per seed example) | 82.09% | 52.4 | 2.104 |
| qsar-biodeg | CN2:rule set (best rule) | 83.42% | 133.7 | 3.854 |
| qsar-biodeg | CN2:rule set (macro distribution voting) | 83.99% | 133.7 | 3.854 |
| qsar-biodeg | CN2:rule set (micro distribution voting) | 81.72% | 133.7 | 3.854 |
| qsar-biodeg | CN2:rule set (voting) | 83.42% | 133.7 | 3.854 |
| qsar-biodeg | PFossil:concept (majority class) | 80.77% | 8.4 | 0.021 |
| qsar-biodeg | PFossil:concept (minority class) | 85.22% | 4.3 | 0.018 |
| qsar-biodeg | PFossil:decision list (per refinement) | 82.38% | 8.8 | 0.041 |
| qsar-biodeg | PFossil:decision list (per rule) | 85.22% | 5.8 | 0.032 |
| qsar-biodeg | PFossil:decision list (per seed example) | 83.89% | 5.6 | 0.018 |
| qsar-biodeg | PFossil:rule set (best rule) | 84.65% | 12.7 | 0.041 |
| qsar-biodeg | PFossil:rule set (macro distribution voting) | 84.74% | 12.7 | 0.041 |
| qsar-biodeg | PFossil:rule set (micro distribution voting) | 84.17% | 12.7 | 0.041 |
| qsar-biodeg | PFossil:rule set (voting) | 83.60% | 12.7 | 0.041 |
| qsar-biodeg | Pypper:concept (majority class) | 83.04% | 8.1 | 0.105 |
| qsar-biodeg | Pypper:concept (minority class) | 83.22% | 5.2 | 0.083 |
| qsar-biodeg | Pypper:decision list (per refinement) | 83.89% | 7.4 | 0.051 |
| qsar-biodeg | Pypper:decision list (per rule) | 80.76% | 10.8 | 0.066 |
| qsar-biodeg | Pypper:decision list (per seed example) | 81.99% | 17.2 | 0.060 |
| qsar-biodeg | Pypper:rule set (best rule) | 83.70% | 13.3 | 0.183 |
| qsar-biodeg | Pypper:rule set (macro distribution voting) | 83.32% | 13.3 | 0.183 |
| qsar-biodeg | Pypper:rule set (micro distribution voting) | 83.12% | 13.3 | 0.183 |
| qsar-biodeg | Pypper:rule set (voting) | 83.03% | 13.3 | 0.183 |
| segment | CN2:cascade (most frequent first) | 93.42% | 57.9 | 0.739 |
| segment | CN2:cascade (rarest first) | 93.42% | 57.9 | 0.741 |
| segment | CN2:decision list (per refinement) | 95.19% | 34.9 | 1.533 |
| segment | CN2:decision list (per rule) | 95.19% | 33.5 | 6.012 |
| segment | CN2:decision list (per seed example) | 95.50% | 29.9 | 0.901 |
| segment | CN2:rule set (best rule) | 94.63% | 96.3 | 1.488 |
| segment | CN2:rule set (macro distribution voting) | 94.29% | 96.3 | 1.488 |
| segment | CN2:rule set (micro distribution voting) | 94.29% | 96.3 | 1.488 |
| segment | CN2:rule set (voting) | 92.99% | 96.3 | 1.488 |
| segment | PFossil:cascade (most frequent first) | 94.85% | 21.4 | 0.053 |
| segment | PFossil:cascade (rarest first) | 94.85% | 21.4 | 0.053 |
| segment | PFossil:decision list (per refinement) | 95.11% | 16.5 | 0.067 |
| segment | PFossil:decision list (per rule) | 94.72% | 23.1 | 0.133 |
| segment | PFossil:decision list (per seed example) | 93.46% | 15.8 | 0.032 |
| segment | PFossil:rule set (best rule) | 93.33% | 30.5 | 0.084 |
| segment | PFossil:rule set (macro distribution voting) | 93.16% | 30.5 | 0.084 |
| segment | PFossil:rule set (micro distribution voting) | 93.12% | 30.5 | 0.084 |
| segment | PFossil:rule set (voting) | 92.73% | 30.5 | 0.084 |
| segment | Pypper:cascade (most frequent first) | 94.81% | 16.1 | 0.165 |
| segment | Pypper:cascade (rarest first) | 94.81% | 16.1 | 0.169 |
| segment | Pypper:decision list (per refinement) | 94.94% | 19.7 | 0.108 |
| segment | Pypper:decision list (per rule) | 94.63% | 18.0 | 0.251 |
| segment | Pypper:decision list (per seed example) | 94.46% | 23.9 | 0.094 |
| segment | Pypper:rule set (best rule) | 94.03% | 24.8 | 0.310 |
| segment | Pypper:rule set (macro distribution voting) | 93.90% | 24.8 | 0.310 |
| segment | Pypper:rule set (micro distribution voting) | 93.72% | 24.8 | 0.310 |
| segment | Pypper:rule set (voting) | 93.68% | 24.8 | 0.310 |
| sick | CN2:concept (majority class) | 97.77% | 41.3 | 0.433 |
| sick | CN2:concept (minority class) | 96.55% | 42.5 | 0.458 |
| sick | CN2:decision list (per refinement) | 97.51% | 29.2 | 0.494 |
| sick | CN2:decision list (per rule) | 97.75% | 30.3 | 0.599 |
| sick | CN2:decision list (per seed example) | 97.83% | 29.3 | 0.763 |
| sick | CN2:rule set (best rule) | 97.88% | 83.8 | 0.892 |
| sick | CN2:rule set (macro distribution voting) | 97.83% | 83.8 | 0.892 |
| sick | CN2:rule set (micro distribution voting) | 97.91% | 83.8 | 0.892 |
| sick | CN2:rule set (voting) | 97.77% | 83.8 | 0.892 |
| sick | PFossil:concept (majority class) | 98.25% | 10.2 | 0.028 |
| sick | PFossil:concept (minority class) | 97.67% | 6.2 | 0.022 |
| sick | PFossil:decision list (per refinement) | 97.67% | 6.2 | 0.042 |
| sick | PFossil:decision list (per rule) | 97.67% | 6.2 | 0.033 |
| sick | PFossil:decision list (per seed example) | 97.96% | 6.8 | 0.019 |
| sick | PFossil:rule set (best rule) | 98.30% | 16.4 | 0.053 |
| sick | PFossil:rule set (macro distribution voting) | 98.25% | 16.4 | 0.053 |
| sick | PFossil:rule set (micro distribution voting) | 98.20% | 16.4 | 0.053 |
| sick | PFossil:rule set (voting) | 98.20% | 16.4 | 0.053 |
| sick | Pypper:concept (majority class) | 97.96% | 6.3 | 0.091 |
| sick | Pypper:concept (minority class) | 98.20% | 4.2 | 0.060 |
| sick | Pypper:decision list (per refinement) | 98.14% | 5.4 | 0.052 |
| sick | Pypper:decision list (per rule) | 98.04% | 6.2 | 0.065 |
| sick | Pypper:decision list (per seed example) | 97.93% | 12.4 | 0.058 |
| sick | Pypper:rule set (best rule) | 98.30% | 10.5 | 0.152 |
| sick | Pypper:rule set (macro distribution voting) | 98.30% | 10.5 | 0.152 |
| sick | Pypper:rule set (micro distribution voting) | 98.30% | 10.5 | 0.152 |
| sick | Pypper:rule set (voting) | 98.33% | 10.5 | 0.152 |
| solar-flare | CN2:cascade (most frequent first) | 67.16% | 53.4 | 0.868 |
| solar-flare | CN2:cascade (rarest first) | 61.63% | 67.9 | 0.863 |
| solar-flare | CN2:decision list (per refinement) | 72.42% | 30.9 | 1.628 |
| solar-flare | CN2:decision list (per rule) | 72.61% | 33.0 | 3.876 |
| solar-flare | CN2:decision list (per seed example) | 69.99% | 24.1 | 1.156 |
| solar-flare | CN2:rule set (best rule) | 73.64% | 112.1 | 1.462 |
| solar-flare | CN2:rule set (macro distribution voting) | 73.36% | 112.1 | 1.462 |
| solar-flare | CN2:rule set (micro distribution voting) | 73.82% | 112.1 | 1.462 |
| solar-flare | CN2:rule set (voting) | 67.16% | 112.1 | 1.462 |
| solar-flare | PFossil:cascade (most frequent first) | 67.92% | 11.5 | 0.018 |
| solar-flare | PFossil:cascade (rarest first) | 70.64% | 12.6 | 0.027 |
| solar-flare | PFossil:decision list (per refinement) | 72.99% | 10.1 | 0.027 |
| solar-flare | PFossil:decision list (per rule) | 72.24% | 15.0 | 0.071 |
| solar-flare | PFossil:decision list (per seed example) | 69.51% | 7.8 | 0.012 |
| solar-flare | PFossil:rule set (best rule) | 74.02% | 14.5 | 0.035 |
| solar-flare | PFossil:rule set (macro distribution voting) | 74.77% | 14.5 | 0.035 |
| solar-flare | PFossil:rule set (micro distribution voting) | 74.30% | 14.5 | 0.035 |
| solar-flare | PFossil:rule set (voting) | 70.07% | 14.5 | 0.035 |
| solar-flare | Pypper:cascade (most frequent first) | 72.42% | 8.3 | 0.092 |
| solar-flare | Pypper:cascade (rarest first) | 74.39% | 5.5 | 0.069 |
| solar-flare | Pypper:decision list (per refinement) | 73.93% | 7.8 | 0.043 |
| solar-flare | Pypper:decision list (per rule) | 72.23% | 5.8 | 0.073 |
| solar-flare | Pypper:decision list (per seed example) | 74.30% | 19.7 | 0.135 |
| solar-flare | Pypper:rule set (best rule) | 69.52% | 7.5 | 0.111 |
| solar-flare | Pypper:rule set (macro distribution voting) | 69.43% | 7.5 | 0.111 |
| solar-flare | Pypper:rule set (micro distribution voting) | 69.43% | 7.5 | 0.111 |
| solar-flare | Pypper:rule set (voting) | 69.24% | 7.5 | 0.111 |
| sonar | CN2:concept (majority class) | 69.67% | 18.7 | 0.092 |
| sonar | CN2:concept (minority class) | 69.71% | 17.9 | 0.075 |
| sonar | CN2:decision list (per refinement) | 71.62% | 14.1 | 0.057 |
| sonar | CN2:decision list (per rule) | 74.98% | 13.1 | 0.081 |
| sonar | CN2:decision list (per seed example) | 74.60% | 10.7 | 0.046 |
| sonar | CN2:rule set (best rule) | 74.60% | 36.6 | 0.165 |
| sonar | CN2:rule set (macro distribution voting) | 74.52% | 36.6 | 0.165 |
| sonar | CN2:rule set (micro distribution voting) | 74.55% | 36.6 | 0.165 |
| sonar | CN2:rule set (voting) | 72.10% | 36.6 | 0.165 |
| sonar | PFossil:concept (majority class) | 71.12% | 8.4 | 0.014 |
| sonar | PFossil:concept (minority class) | 72.64% | 8.3 | 0.013 |
| sonar | PFossil:decision list (per refinement) | 73.52% | 6.0 | 0.021 |
| sonar | PFossil:decision list (per rule) | 72.02% | 4.2 | 0.018 |
| sonar | PFossil:decision list (per seed example) | 68.69% | 4.3 | 0.011 |
| sonar | PFossil:rule set (best rule) | 73.55% | 16.7 | 0.026 |
| sonar | PFossil:rule set (macro distribution voting) | 75.07% | 16.7 | 0.026 |
| sonar | PFossil:rule set (micro distribution voting) | 74.02% | 16.7 | 0.026 |
| sonar | PFossil:rule set (voting) | 75.52% | 16.7 | 0.026 |
| sonar | Pypper:concept (majority class) | 69.79% | 3.6 | 0.037 |
| sonar | Pypper:concept (minority class) | 75.05% | 3.0 | 0.031 |
| sonar | Pypper:decision list (per refinement) | 74.12% | 4.0 | 0.017 |
| sonar | Pypper:decision list (per rule) | 69.21% | 4.4 | 0.024 |
| sonar | Pypper:decision list (per seed example) | 73.69% | 5.4 | 0.027 |
| sonar | Pypper:rule set (best rule) | 73.57% | 6.6 | 0.066 |
| sonar | Pypper:rule set (macro distribution voting) | 72.62% | 6.6 | 0.066 |
| sonar | Pypper:rule set (micro distribution voting) | 74.05% | 6.6 | 0.066 |
| sonar | Pypper:rule set (voting) | 71.21% | 6.6 | 0.066 |
| soybean | CN2:cascade (most frequent first) | 89.90% | 31.6 | 0.255 |
| soybean | CN2:cascade (rarest first) | 90.20% | 29.1 | 0.251 |
| soybean | CN2:decision list (per refinement) | 90.64% | 26.6 | 0.699 |
| soybean | CN2:decision list (per rule) | 91.37% | 25.8 | 3.128 |
| soybean | CN2:decision list (per seed example) | 92.24% | 25.4 | 0.232 |
| soybean | CN2:rule set (best rule) | 93.12% | 43.0 | 0.410 |
| soybean | CN2:rule set (macro distribution voting) | 91.66% | 43.0 | 0.410 |
| soybean | CN2:rule set (micro distribution voting) | 92.39% | 43.0 | 0.410 |
| soybean | CN2:rule set (voting) | 89.90% | 43.0 | 0.410 |
| soybean | PFossil:cascade (most frequent first) | 90.48% | 31.1 | 0.039 |
| soybean | PFossil:cascade (rarest first) | 92.98% | 32.5 | 0.042 |
| soybean | PFossil:decision list (per refinement) | 93.27% | 35.8 | 0.082 |
| soybean | PFossil:decision list (per rule) | 94.29% | 22.8 | 0.164 |
| soybean | PFossil:decision list (per seed example) | 92.68% | 25.1 | 0.027 |
| soybean | PFossil:rule set (best rule) | 92.24% | 42.1 | 0.064 |
| soybean | PFossil:rule set (macro distribution voting) | 91.36% | 42.1 | 0.064 |
| soybean | PFossil:rule set (micro distribution voting) | 91.66% | 42.1 | 0.064 |
| soybean | PFossil:rule set (voting) | 90.34% | 42.1 | 0.064 |
| soybean | Pypper:cascade (most frequent first) | 88.16% | 26.9 | 0.153 |
| soybean | Pypper:cascade (rarest first) | 91.07% | 25.8 | 0.156 |
| soybean | Pypper:decision list (per refinement) | 91.08% | 28.5 | 0.110 |
| soybean | Pypper:decision list (per rule) | 90.20% | 28.5 | 0.500 |
| soybean | Pypper:decision list (per seed example) | 87.27% | 28.0 | 0.065 |
| soybean | Pypper:rule set (best rule) | 90.50% | 28.2 | 0.205 |
| soybean | Pypper:rule set (macro distribution voting) | 90.50% | 28.2 | 0.205 |
| soybean | Pypper:rule set (micro distribution voting) | 88.15% | 28.2 | 0.205 |
| soybean | Pypper:rule set (voting) | 87.71% | 28.2 | 0.205 |
| spambase | CN2:concept (majority class) | 90.94% | 195.5 | 33.539 |
| spambase | CN2:concept (minority class) | 85.74% | 213.8 | 20.578 |
| spambase | CN2:decision list (per refinement) | 93.02% | 101.9 | 1.545 |
| spambase | CN2:decision list (per rule) | 93.50% | 111.4 | 1.591 |
| spambase | CN2:decision list (per seed example) | 93.33% | 100.1 | 8.270 |
| spambase | CN2:rule set (best rule) | 94.07% | 409.3 | 53.984 |
| spambase | CN2:rule set (macro distribution voting) | 93.96% | 409.3 | 53.984 |
| spambase | CN2:rule set (micro distribution voting) | 92.96% | 409.3 | 53.984 |
| spambase | CN2:rule set (voting) | 93.70% | 409.3 | 53.984 |
| spambase | PFossil:concept (majority class) | 92.05% | 6.0 | 0.037 |
| spambase | PFossil:concept (minority class) | 92.72% | 6.1 | 0.052 |
| spambase | PFossil:decision list (per refinement) | 92.55% | 5.4 | 0.123 |
| spambase | PFossil:decision list (per rule) | 91.74% | 5.8 | 0.058 |
| spambase | PFossil:decision list (per seed example) | 92.07% | 7.2 | 0.041 |
| spambase | PFossil:rule set (best rule) | 92.81% | 12.1 | 0.094 |
| spambase | PFossil:rule set (macro distribution voting) | 92.76% | 12.1 | 0.094 |
| spambase | PFossil:rule set (micro distribution voting) | 91.81% | 12.1 | 0.094 |
| spambase | PFossil:rule set (voting) | 92.41% | 12.1 | 0.094 |
| spambase | Pypper:concept (majority class) | 92.37% | 15.2 | 0.460 |
| spambase | Pypper:concept (minority class) | 92.59% | 11.2 | 0.318 |
| spambase | Pypper:decision list (per refinement) | 92.76% | 17.8 | 0.295 |
| spambase | Pypper:decision list (per rule) | 93.09% | 30.8 | 0.319 |
| spambase | Pypper:decision list (per seed example) | 92.83% | 29.3 | 0.205 |
| spambase | Pypper:rule set (best rule) | 93.11% | 26.4 | 0.780 |
| spambase | Pypper:rule set (macro distribution voting) | 93.00% | 26.4 | 0.780 |
| spambase | Pypper:rule set (micro distribution voting) | 92.59% | 26.4 | 0.780 |
| spambase | Pypper:rule set (voting) | 92.83% | 26.4 | 0.780 |
| splice | CN2:cascade (most frequent first) | 91.76% | 80.2 | 0.703 |
| splice | CN2:cascade (rarest first) | 90.16% | 50.8 | 0.743 |
| splice | CN2:decision list (per refinement) | 91.10% | 69.1 | 0.817 |
| splice | CN2:decision list (per rule) | 90.63% | 60.9 | 1.724 |
| splice | CN2:decision list (per seed example) | 91.10% | 57.5 | 0.706 |
| splice | CN2:rule set (best rule) | 92.07% | 120.9 | 1.512 |
| splice | CN2:rule set (macro distribution voting) | 91.79% | 120.9 | 1.512 |
| splice | CN2:rule set (micro distribution voting) | 92.07% | 120.9 | 1.512 |
| splice | CN2:rule set (voting) | 91.79% | 120.9 | 1.512 |
| splice | PFossil:cascade (most frequent first) | 89.50% | 19.0 | 0.064 |
| splice | PFossil:cascade (rarest first) | 94.33% | 8.8 | 0.037 |
| splice | PFossil:decision list (per refinement) | 89.66% | 13.7 | 0.075 |
| splice | PFossil:decision list (per rule) | 93.26% | 12.3 | 0.076 |
| splice | PFossil:decision list (per seed example) | 89.31% | 10.6 | 0.033 |
| splice | PFossil:rule set (best rule) | 94.23% | 23.7 | 0.092 |
| splice | PFossil:rule set (macro distribution voting) | 94.29% | 23.7 | 0.092 |
| splice | PFossil:rule set (micro distribution voting) | 90.38% | 23.7 | 0.092 |
| splice | PFossil:rule set (voting) | 89.09% | 23.7 | 0.092 |
| splice | Pypper:cascade (most frequent first) | 93.39% | 17.6 | 0.269 |
| splice | Pypper:cascade (rarest first) | 93.10% | 12.1 | 0.227 |
| splice | Pypper:decision list (per refinement) | 93.89% | 16.5 | 0.133 |
| splice | Pypper:decision list (per rule) | 93.07% | 17.0 | 0.196 |
| splice | Pypper:decision list (per seed example) | 92.60% | 22.3 | 0.101 |
| splice | Pypper:rule set (best rule) | 93.20% | 26.8 | 0.502 |
| splice | Pypper:rule set (macro distribution voting) | 93.39% | 26.8 | 0.502 |
| splice | Pypper:rule set (micro distribution voting) | 93.39% | 26.8 | 0.502 |
| splice | Pypper:rule set (voting) | 93.26% | 26.8 | 0.502 |
| texture | CN2:cascade (most frequent first) | 92.00% | 131.4 | 5.013 |
| texture | CN2:cascade (rarest first) | 92.00% | 131.4 | 4.970 |
| texture | CN2:decision list (per refinement) | 94.31% | 87.4 | 18.936 |
| texture | CN2:decision list (per rule) | 93.62% | 78.4 | 56.440 |
| texture | CN2:decision list (per seed example) | 94.13% | 78.0 | 7.944 |
| texture | CN2:rule set (best rule) | 93.55% | 210.5 | 7.359 |
| texture | CN2:rule set (macro distribution voting) | 93.78% | 210.5 | 7.359 |
| texture | CN2:rule set (micro distribution voting) | 93.89% | 210.5 | 7.359 |
| texture | CN2:rule set (voting) | 93.75% | 210.5 | 7.359 |
| texture | PFossil:cascade (most frequent first) | 90.45% | 52.1 | 0.237 |
| texture | PFossil:cascade (rarest first) | 90.45% | 52.1 | 0.239 |
| texture | PFossil:decision list (per refinement) | 91.04% | 48.9 | 0.436 |
| texture | PFossil:decision list (per rule) | 90.29% | 41.9 | 0.748 |
| texture | PFossil:decision list (per seed example) | 89.44% | 36.4 | 0.152 |
| texture | PFossil:rule set (best rule) | 92.22% | 77.9 | 0.400 |
| texture | PFossil:rule set (macro distribution voting) | 92.45% | 77.9 | 0.400 |
| texture | PFossil:rule set (micro distribution voting) | 90.29% | 77.9 | 0.400 |
| texture | PFossil:rule set (voting) | 91.65% | 77.9 | 0.400 |
| texture | Pypper:cascade (most frequent first) | 92.33% | 56.0 | 0.966 |
| texture | Pypper:cascade (rarest first) | 92.33% | 56.0 | 0.966 |
| texture | Pypper:decision list (per refinement) | 91.33% | 54.6 | 0.642 |
| texture | Pypper:decision list (per rule) | 91.73% | 45.2 | 1.667 |
| texture | Pypper:decision list (per seed example) | 92.16% | 63.4 | 0.390 |
| texture | Pypper:rule set (best rule) | 92.40% | 81.3 | 1.654 |
| texture | Pypper:rule set (macro distribution voting) | 92.82% | 81.3 | 1.654 |
| texture | Pypper:rule set (micro distribution voting) | 92.05% | 81.3 | 1.654 |
| texture | Pypper:rule set (voting) | 92.38% | 81.3 | 1.654 |
| tic-tac-toe | CN2:concept (majority class) | 99.69% | 10.3 | 0.034 |
| tic-tac-toe | CN2:concept (minority class) | 96.56% | 18.3 | 0.077 |
| tic-tac-toe | CN2:decision list (per refinement) | 94.36% | 33.6 | 0.175 |
| tic-tac-toe | CN2:decision list (per rule) | 97.39% | 30.4 | 0.171 |
| tic-tac-toe | CN2:decision list (per seed example) | 98.23% | 15.7 | 0.049 |
| tic-tac-toe | CN2:rule set (best rule) | 98.33% | 28.6 | 0.106 |
| tic-tac-toe | CN2:rule set (macro distribution voting) | 98.02% | 28.6 | 0.106 |
| tic-tac-toe | CN2:rule set (micro distribution voting) | 98.33% | 28.6 | 0.106 |
| tic-tac-toe | CN2:rule set (voting) | 98.02% | 28.6 | 0.106 |
| tic-tac-toe | PFossil:concept (majority class) | 83.93% | 7.2 | 0.010 |
| tic-tac-toe | PFossil:concept (minority class) | 79.85% | 7.5 | 0.012 |
| tic-tac-toe | PFossil:decision list (per refinement) | 76.20% | 3.3 | 0.009 |
| tic-tac-toe | PFossil:decision list (per rule) | 79.85% | 7.5 | 0.017 |
| tic-tac-toe | PFossil:decision list (per seed example) | 73.89% | 8.7 | 0.037 |
| tic-tac-toe | PFossil:rule set (best rule) | 97.71% | 14.7 | 0.023 |
| tic-tac-toe | PFossil:rule set (macro distribution voting) | 95.10% | 14.7 | 0.023 |
| tic-tac-toe | PFossil:rule set (micro distribution voting) | 83.72% | 14.7 | 0.023 |
| tic-tac-toe | PFossil:rule set (voting) | 85.29% | 14.7 | 0.023 |
| tic-tac-toe | Pypper:concept (majority class) | 99.69% | 8.0 | 0.059 |
| tic-tac-toe | Pypper:concept (minority class) | 98.02% | 8.6 | 0.058 |
| tic-tac-toe | Pypper:decision list (per refinement) | 97.81% | 9.4 | 0.036 |
| tic-tac-toe | Pypper:decision list (per rule) | 97.71% | 13.4 | 0.054 |
| tic-tac-toe | Pypper:decision list (per seed example) | 93.01% | 17.2 | 0.047 |
| tic-tac-toe | Pypper:rule set (best rule) | 98.33% | 16.6 | 0.105 |
| tic-tac-toe | Pypper:rule set (macro distribution voting) | 98.33% | 16.6 | 0.105 |
| tic-tac-toe | Pypper:rule set (micro distribution voting) | 98.33% | 16.6 | 0.105 |
| tic-tac-toe | Pypper:rule set (voting) | 98.33% | 16.6 | 0.105 |
| vehicle | CN2:cascade (most frequent first) | 68.92% | 80.5 | 0.842 |
| vehicle | CN2:cascade (rarest first) | 69.75% | 78.0 | 0.787 |
| vehicle | CN2:decision list (per refinement) | 71.39% | 72.6 | 2.046 |
| vehicle | CN2:decision list (per rule) | 69.98% | 63.7 | 4.204 |
| vehicle | CN2:decision list (per seed example) | 71.05% | 51.8 | 1.897 |
| vehicle | CN2:rule set (best rule) | 68.69% | 153.1 | 1.894 |
| vehicle | CN2:rule set (macro distribution voting) | 67.74% | 153.1 | 1.894 |
| vehicle | CN2:rule set (micro distribution voting) | 68.80% | 153.1 | 1.894 |
| vehicle | CN2:rule set (voting) | 67.26% | 153.1 | 1.894 |
| vehicle | PFossil:cascade (most frequent first) | 70.10% | 20.6 | 0.035 |
| vehicle | PFossil:cascade (rarest first) | 70.22% | 22.4 | 0.037 |
| vehicle | PFossil:decision list (per refinement) | 66.66% | 8.2 | 0.028 |
| vehicle | PFossil:decision list (per rule) | 67.73% | 13.5 | 0.061 |
| vehicle | PFossil:decision list (per seed example) | 65.13% | 13.4 | 0.024 |
| vehicle | PFossil:rule set (best rule) | 71.64% | 26.1 | 0.054 |
| vehicle | PFossil:rule set (macro distribution voting) | 72.82% | 26.1 | 0.054 |
| vehicle | PFossil:rule set (micro distribution voting) | 70.93% | 26.1 | 0.054 |
| vehicle | PFossil:rule set (voting) | 71.53% | 26.1 | 0.054 |
| vehicle | Pypper:cascade (most frequent first) | 71.53% | 12.2 | 0.114 |
| vehicle | Pypper:cascade (rarest first) | 67.74% | 14.0 | 0.127 |
| vehicle | Pypper:decision list (per refinement) | 69.28% | 13.5 | 0.057 |
| vehicle | Pypper:decision list (per rule) | 69.76% | 16.5 | 0.147 |
| vehicle | Pypper:decision list (per seed example) | 69.63% | 21.5 | 0.068 |
| vehicle | Pypper:rule set (best rule) | 63.95% | 17.4 | 0.206 |
| vehicle | Pypper:rule set (macro distribution voting) | 64.19% | 17.4 | 0.206 |
| vehicle | Pypper:rule set (micro distribution voting) | 62.65% | 17.4 | 0.206 |
| vehicle | Pypper:rule set (voting) | 62.90% | 17.4 | 0.206 |
| vote | CN2:concept (majority class) | 94.25% | 14.3 | 0.043 |
| vote | CN2:concept (minority class) | 92.65% | 12.0 | 0.046 |
| vote | CN2:decision list (per refinement) | 93.56% | 10.4 | 0.054 |
| vote | CN2:decision list (per rule) | 93.79% | 10.3 | 0.064 |
| vote | CN2:decision list (per seed example) | 93.57% | 11.4 | 0.057 |
| vote | CN2:rule set (best rule) | 94.27% | 26.3 | 0.090 |
| vote | CN2:rule set (macro distribution voting) | 94.03% | 26.3 | 0.090 |
| vote | CN2:rule set (micro distribution voting) | 94.49% | 26.3 | 0.090 |
| vote | CN2:rule set (voting) | 94.03% | 26.3 | 0.090 |
| vote | PFossil:concept (majority class) | 93.56% | 6.8 | 0.007 |
| vote | PFossil:concept (minority class) | 93.81% | 3.4 | 0.004 |
| vote | PFossil:decision list (per refinement) | 95.42% | 4.7 | 0.007 |
| vote | PFossil:decision list (per rule) | 93.35% | 6.5 | 0.008 |
| vote | PFossil:decision list (per seed example) | 94.49% | 6.1 | 0.005 |
| vote | PFossil:rule set (best rule) | 95.18% | 10.2 | 0.012 |
| vote | PFossil:rule set (macro distribution voting) | 93.57% | 10.2 | 0.012 |
| vote | PFossil:rule set (micro distribution voting) | 95.42% | 10.2 | 0.012 |
| vote | PFossil:rule set (voting) | 94.03% | 10.2 | 0.012 |
| vote | Pypper:concept (majority class) | 96.09% | 3.0 | 0.017 |
| vote | Pypper:concept (minority class) | 95.18% | 1.8 | 0.016 |
| vote | Pypper:decision list (per refinement) | 94.49% | 6.1 | 0.015 |
| vote | Pypper:decision list (per rule) | 95.16% | 5.5 | 0.019 |
| vote | Pypper:decision list (per seed example) | 95.40% | 5.5 | 0.012 |
| vote | Pypper:rule set (best rule) | 96.09% | 4.8 | 0.033 |
| vote | Pypper:rule set (macro distribution voting) | 95.86% | 4.8 | 0.033 |
| vote | Pypper:rule set (micro distribution voting) | 95.18% | 4.8 | 0.033 |
| vote | Pypper:rule set (voting) | 95.86% | 4.8 | 0.033 |
| vowel | CN2:cascade (most frequent first) | 81.92% | 88.1 | 0.548 |
| vowel | CN2:cascade (rarest first) | 81.92% | 88.1 | 0.558 |
| vowel | CN2:decision list (per refinement) | 79.29% | 61.7 | 1.644 |
| vowel | CN2:decision list (per rule) | 80.51% | 52.0 | 5.433 |
| vowel | CN2:decision list (per seed example) | 79.09% | 53.9 | 0.823 |
| vowel | CN2:rule set (best rule) | 80.20% | 134.8 | 0.937 |
| vowel | CN2:rule set (macro distribution voting) | 82.22% | 134.8 | 0.937 |
| vowel | CN2:rule set (micro distribution voting) | 80.20% | 134.8 | 0.937 |
| vowel | CN2:rule set (voting) | 81.52% | 134.8 | 0.937 |
| vowel | PFossil:cascade (most frequent first) | 80.40% | 69.7 | 0.101 |
| vowel | PFossil:cascade (rarest first) | 80.40% | 69.7 | 0.100 |
| vowel | PFossil:decision list (per refinement) | 75.96% | 57.4 | 0.170 |
| vowel | PFossil:decision list (per rule) | 75.25% | 46.2 | 0.443 |
| vowel | PFossil:decision list (per seed example) | 71.41% | 42.4 | 0.072 |
| vowel | PFossil:rule set (best rule) | 77.27% | 94.9 | 0.165 |
| vowel | PFossil:rule set (macro distribution voting) | 79.80% | 94.9 | 0.165 |
| vowel | PFossil:rule set (micro distribution voting) | 74.24% | 94.9 | 0.165 |
| vowel | PFossil:rule set (voting) | 78.89% | 94.9 | 0.165 |
| vowel | Pypper:cascade (most frequent first) | 74.85% | 48.1 | 0.370 |
| vowel | Pypper:cascade (rarest first) | 74.85% | 48.1 | 0.371 |
| vowel | Pypper:decision list (per refinement) | 66.16% | 40.0 | 0.194 |
| vowel | Pypper:decision list (per rule) | 61.82% | 30.7 | 0.699 |
| vowel | Pypper:decision list (per seed example) | 72.22% | 49.8 | 0.159 |
| vowel | Pypper:rule set (best rule) | 73.84% | 68.9 | 0.584 |
| vowel | Pypper:rule set (macro distribution voting) | 74.85% | 68.9 | 0.584 |
| vowel | Pypper:rule set (micro distribution voting) | 72.02% | 68.9 | 0.584 |
| vowel | Pypper:rule set (voting) | 72.02% | 68.9 | 0.584 |
| wdbc | CN2:concept (majority class) | 94.37% | 12.5 | 0.049 |
| wdbc | CN2:concept (minority class) | 93.85% | 13.1 | 0.058 |
| wdbc | CN2:decision list (per refinement) | 94.55% | 11.0 | 0.061 |
| wdbc | CN2:decision list (per rule) | 94.55% | 10.5 | 0.082 |
| wdbc | CN2:decision list (per seed example) | 92.62% | 9.9 | 0.046 |
| wdbc | CN2:rule set (best rule) | 96.13% | 25.6 | 0.107 |
| wdbc | CN2:rule set (macro distribution voting) | 95.60% | 25.6 | 0.107 |
| wdbc | CN2:rule set (micro distribution voting) | 96.13% | 25.6 | 0.107 |
| wdbc | CN2:rule set (voting) | 95.60% | 25.6 | 0.107 |
| wdbc | PFossil:concept (majority class) | 94.02% | 4.1 | 0.006 |
| wdbc | PFossil:concept (minority class) | 94.02% | 7.6 | 0.012 |
| wdbc | PFossil:decision list (per refinement) | 94.02% | 7.3 | 0.020 |
| wdbc | PFossil:decision list (per rule) | 94.03% | 3.9 | 0.010 |
| wdbc | PFossil:decision list (per seed example) | 94.37% | 4.4 | 0.007 |
| wdbc | PFossil:rule set (best rule) | 94.56% | 11.7 | 0.018 |
| wdbc | PFossil:rule set (macro distribution voting) | 94.91% | 11.7 | 0.018 |
| wdbc | PFossil:rule set (micro distribution voting) | 93.85% | 11.7 | 0.018 |
| wdbc | PFossil:rule set (voting) | 94.56% | 11.7 | 0.018 |
| wdbc | Pypper:concept (majority class) | 93.83% | 3.4 | 0.031 |
| wdbc | Pypper:concept (minority class) | 95.08% | 3.8 | 0.029 |
| wdbc | Pypper:decision list (per refinement) | 95.08% | 4.8 | 0.020 |
| wdbc | Pypper:decision list (per rule) | 95.26% | 5.5 | 0.027 |
| wdbc | Pypper:decision list (per seed example) | 95.44% | 5.7 | 0.017 |
| wdbc | Pypper:rule set (best rule) | 95.44% | 7.2 | 0.059 |
| wdbc | Pypper:rule set (macro distribution voting) | 95.44% | 7.2 | 0.059 |
| wdbc | Pypper:rule set (micro distribution voting) | 95.43% | 7.2 | 0.059 |
| wdbc | Pypper:rule set (voting) | 95.08% | 7.2 | 0.059 |
| wilt | CN2:concept (majority class) | 96.63% | 43.4 | 0.635 |
| wilt | CN2:concept (minority class) | 71.01% | 38.0 | 0.626 |
| wilt | CN2:decision list (per refinement) | 97.02% | 38.5 | 1.120 |
| wilt | CN2:decision list (per rule) | 97.09% | 36.8 | 1.060 |
| wilt | CN2:decision list (per seed example) | 91.59% | 16.9 | 3.882 |
| wilt | CN2:rule set (best rule) | 96.90% | 81.4 | 1.268 |
| wilt | CN2:rule set (macro distribution voting) | 96.96% | 81.4 | 1.268 |
| wilt | CN2:rule set (micro distribution voting) | 96.63% | 81.4 | 1.268 |
| wilt | CN2:rule set (voting) | 91.78% | 81.4 | 1.268 |
| wilt | PFossil:concept (majority class) | 5.39% | 0.0 | 0.003 |
| wilt | PFossil:concept (minority class) | 97.02% | 1.0 | 0.006 |
| wilt | PFossil:decision list (per refinement) | 97.02% | 1.0 | 0.012 |
| wilt | PFossil:decision list (per rule) | 97.02% | 1.0 | 0.010 |
| wilt | PFossil:decision list (per seed example) | 91.20% | 4.5 | 1.891 |
| wilt | PFossil:rule set (best rule) | 97.02% | 1.0 | 0.014 |
| wilt | PFossil:rule set (macro distribution voting) | 97.02% | 1.0 | 0.014 |
| wilt | PFossil:rule set (micro distribution voting) | 97.02% | 1.0 | 0.014 |
| wilt | PFossil:rule set (voting) | 97.02% | 1.0 | 0.014 |
| wilt | Pypper:concept (majority class) | 97.33% | 4.5 | 0.079 |
| wilt | Pypper:concept (minority class) | 97.31% | 2.0 | 0.046 |
| wilt | Pypper:decision list (per refinement) | 97.25% | 6.8 | 0.072 |
| wilt | Pypper:decision list (per rule) | 97.29% | 6.6 | 0.071 |
| wilt | Pypper:decision list (per seed example) | 97.31% | 12.2 | 0.088 |
| wilt | Pypper:rule set (best rule) | 97.36% | 6.5 | 0.132 |
| wilt | Pypper:rule set (macro distribution voting) | 97.38% | 6.5 | 0.132 |
| wilt | Pypper:rule set (micro distribution voting) | 97.36% | 6.5 | 0.132 |
| wilt | Pypper:rule set (voting) | 97.38% | 6.5 | 0.132 |
| wine | CN2:cascade (most frequent first) | 92.68% | 5.8 | 0.011 |
| wine | CN2:cascade (rarest first) | 94.38% | 4.2 | 0.011 |
| wine | CN2:decision list (per refinement) | 93.86% | 5.5 | 0.018 |
| wine | CN2:decision list (per rule) | 92.09% | 4.9 | 0.025 |
| wine | CN2:decision list (per seed example) | 92.19% | 5.2 | 0.013 |
| wine | CN2:rule set (best rule) | 91.54% | 10.4 | 0.027 |
| wine | CN2:rule set (macro distribution voting) | 93.24% | 10.4 | 0.027 |
| wine | CN2:rule set (micro distribution voting) | 93.24% | 10.4 | 0.027 |
| wine | CN2:rule set (voting) | 93.24% | 10.4 | 0.027 |
| wine | PFossil:cascade (most frequent first) | 95.49% | 5.4 | 0.004 |
| wine | PFossil:cascade (rarest first) | 94.97% | 4.3 | 0.004 |
| wine | PFossil:decision list (per refinement) | 92.12% | 4.2 | 0.007 |
| wine | PFossil:decision list (per rule) | 91.05% | 4.0 | 0.008 |
| wine | PFossil:decision list (per seed example) | 93.27% | 5.3 | 0.005 |
| wine | PFossil:rule set (best rule) | 92.75% | 9.4 | 0.008 |
| wine | PFossil:rule set (macro distribution voting) | 93.82% | 9.4 | 0.008 |
| wine | PFossil:rule set (micro distribution voting) | 93.24% | 9.4 | 0.008 |
| wine | PFossil:rule set (voting) | 93.79% | 9.4 | 0.008 |
| wine | Pypper:cascade (most frequent first) | 88.20% | 3.5 | 0.018 |
| wine | Pypper:cascade (rarest first) | 92.09% | 2.8 | 0.016 |
| wine | Pypper:decision list (per refinement) | 89.87% | 3.0 | 0.008 |
| wine | Pypper:decision list (per rule) | 88.24% | 3.6 | 0.017 |
| wine | Pypper:decision list (per seed example) | 86.01% | 4.4 | 0.012 |
| wine | Pypper:rule set (best rule) | 92.09% | 5.3 | 0.029 |
| wine | Pypper:rule set (macro distribution voting) | 90.98% | 5.3 | 0.029 |
| wine | Pypper:rule set (micro distribution voting) | 90.92% | 5.3 | 0.029 |
| wine | Pypper:rule set (voting) | 88.69% | 5.3 | 0.029 |
| yeast | CN2:cascade (most frequent first) | 50.41% | 141.1 | 2.291 |
| yeast | CN2:cascade (rarest first) | 46.09% | 152.5 | 2.663 |
| yeast | CN2:decision list (per refinement) | 55.18% | 69.3 | 3.432 |
| yeast | CN2:decision list (per rule) | 57.41% | 67.6 | 9.875 |
| yeast | CN2:decision list (per seed example) | 55.53% | 51.9 | 1.011 |
| yeast | CN2:rule set (best rule) | 55.73% | 247.2 | 3.998 |
| yeast | CN2:rule set (macro distribution voting) | 54.45% | 247.2 | 3.998 |
| yeast | CN2:rule set (micro distribution voting) | 55.66% | 247.2 | 3.998 |
| yeast | CN2:rule set (voting) | 52.56% | 247.2 | 3.998 |
| yeast | PFossil:cascade (most frequent first) | 53.57% | 36.3 | 0.055 |
| yeast | PFossil:cascade (rarest first) | 56.06% | 29.4 | 0.076 |
| yeast | PFossil:decision list (per refinement) | 58.76% | 20.6 | 0.079 |
| yeast | PFossil:decision list (per rule) | 56.80% | 35.3 | 0.381 |
| yeast | PFossil:decision list (per seed example) | 54.24% | 24.2 | 0.042 |
| yeast | PFossil:rule set (best rule) | 57.48% | 31.2 | 0.089 |
| yeast | PFossil:rule set (macro distribution voting) | 57.54% | 31.2 | 0.089 |
| yeast | PFossil:rule set (micro distribution voting) | 55.52% | 31.2 | 0.089 |
| yeast | PFossil:rule set (voting) | 54.31% | 31.2 | 0.089 |
| yeast | Pypper:cascade (most frequent first) | 53.30% | 19.8 | 0.258 |
| yeast | Pypper:cascade (rarest first) | 59.57% | 15.6 | 0.227 |
| yeast | Pypper:decision list (per refinement) | 52.77% | 16.5 | 0.117 |
| yeast | Pypper:decision list (per rule) | 56.67% | 14.9 | 0.376 |
| yeast | Pypper:decision list (per seed example) | 55.06% | 45.6 | 0.297 |
| yeast | Pypper:rule set (best rule) | 59.43% | 18.5 | 0.313 |
| yeast | Pypper:rule set (macro distribution voting) | 59.02% | 18.5 | 0.313 |
| yeast | Pypper:rule set (micro distribution voting) | 57.88% | 18.5 | 0.313 |
| yeast | Pypper:rule set (voting) | 58.15% | 18.5 | 0.313 |
| zoo | CN2:cascade (most frequent first) | 96.00% | 6.7 | 0.028 |
| zoo | CN2:cascade (rarest first) | 95.00% | 7.1 | 0.015 |
| zoo | CN2:decision list (per refinement) | 96.00% | 6.3 | 0.031 |
| zoo | CN2:decision list (per rule) | 93.00% | 6.0 | 0.103 |
| zoo | CN2:decision list (per seed example) | 93.09% | 6.3 | 0.010 |
| zoo | CN2:rule set (best rule) | 97.00% | 8.9 | 0.025 |
| zoo | CN2:rule set (macro distribution voting) | 97.00% | 8.9 | 0.025 |
| zoo | CN2:rule set (micro distribution voting) | 96.00% | 8.9 | 0.025 |
| zoo | CN2:rule set (voting) | 96.00% | 8.9 | 0.025 |
| zoo | PFossil:cascade (most frequent first) | 96.00% | 8.3 | 0.006 |
| zoo | PFossil:cascade (rarest first) | 93.00% | 7.7 | 0.007 |
| zoo | PFossil:decision list (per refinement) | 96.00% | 6.8 | 0.008 |
| zoo | PFossil:decision list (per rule) | 97.00% | 6.0 | 0.015 |
| zoo | PFossil:decision list (per seed example) | 91.09% | 6.2 | 0.005 |
| zoo | PFossil:rule set (best rule) | 95.00% | 8.9 | 0.008 |
| zoo | PFossil:rule set (macro distribution voting) | 92.00% | 8.9 | 0.008 |
| zoo | PFossil:rule set (micro distribution voting) | 94.00% | 8.9 | 0.008 |
| zoo | PFossil:rule set (voting) | 93.00% | 8.9 | 0.008 |
| zoo | Pypper:cascade (most frequent first) | 93.09% | 5.2 | 0.025 |
| zoo | Pypper:cascade (rarest first) | 89.09% | 5.4 | 0.030 |
| zoo | Pypper:decision list (per refinement) | 92.00% | 6.0 | 0.012 |
| zoo | Pypper:decision list (per rule) | 90.00% | 6.1 | 0.043 |
| zoo | Pypper:decision list (per seed example) | 89.09% | 6.3 | 0.013 |
| zoo | Pypper:rule set (best rule) | 90.09% | 6.1 | 0.038 |
| zoo | Pypper:rule set (macro distribution voting) | 90.09% | 6.1 | 0.038 |
| zoo | Pypper:rule set (micro distribution voting) | 91.09% | 6.1 | 0.038 |
| zoo | Pypper:rule set (voting) | 90.09% | 6.1 | 0.038 |

