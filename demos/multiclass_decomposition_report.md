# Multi-class decomposition: one-vs-rest vs. pairwise

## Setup

**Full run.** Quick sanity check instead: `python demos/multiclass_decomposition.py`.

A rule learner that learns rules for one class against the rest has to
decompose a problem with several classes into two-class problems. This
demo compares the ways of doing that in pyrulearn, with Pypper
(pyrulearn's RIPPER), PFossil and PFoil as base learners -- in essence the
experiments of Fürnkranz, *Round Robin Classification* (JMLR 2002), with
Pypper standing in for Ripper. The paper's argument for pairwise
classification is its cost: a pairwise decomposition trains on about
(c - 1) times the examples in all, one-vs-rest on c times as many, but in many
small problems -- a large gain for a learner whose cost grows faster than
linearly with the examples. Of the three, PFossil grows at most linearly with
the examples, PFoil clearly faster (on `adult`, from 2,000 to 16,000 examples,
about n^1.6, with a rule count growing almost in proportion), Pypper in
between, depending on the data (n^1.0-1.8). CN2 would be another superlinear
learner, but its Laplace heuristic takes the number of classes of the whole
problem into every sub-problem (26 in a pair of `letter`'s classes), which
changes its searches and its results; it is left out until that is settled
(see ROADMAP.md):

- **ovr** -- one-vs-rest: one model per class, "this class vs. all
  others", every model trained on all examples; conflicts resolved by the
  best rule.
- **ordered** -- ordered one-vs-rest: classes from least to most frequent,
  each learned against the classes not yet handled; the most frequent
  class is the default. This is Ripper's (and Pypper's) own strategy.
- **pw_smaller / pw_larger** -- pairwise (round robin): one model per pair
  of classes, trained on those two classes' examples only, with the
  pair's smaller (larger) class as the target.
- **pw_both** -- the double round robin: for every pair, a model for each
  of its two classes.

A pairwise model is scored with three vote schemes on the same fitted
model: **vote** (each pair model votes for one class), **weighted_vote**
(the vote is split by the reliability of the rule that decided it) and
**accuracy_vote** (the vote is split by the pair model's accuracy on its
own two classes). Only prediction changes, so the three share their
training.

**Why training time is interesting.** Pairwise learns many more models --
c(c-1)/2 for c classes -- but each on few examples: every example is used
in c - 1 pair models, so pairwise learns from (c - 1) * n examples in
total, against c * n for one-vs-rest. For a learner whose time grows
linearly with the examples, one-vs-rest therefore takes about c / (c - 1)
times as long; for one that grows faster, pairwise gains more with every
class. Prediction goes the other way: every pair model has to be asked.

Datasets, with their numbers of classes: `letter` (26), `audiology` (24), `primary-tumor` (21), `soybean` (19), `kropt` (18), `texture` (11), `vowel` (11), `led24` (10), `yeast` (10), `optdigits` (10), `ecoli` (8), `segment` (7), `zoo` (7), `glass` (6), `solar-flare` (6), `anneal` (5), `car` (4), `lymph` (4), `vehicle` (4), `balance-scale` (3), `cmc` (3), `dna` (3), `hayes-roth` (3), `iris` (3), `splice` (3), `wine` (3). Protocol: 10-fold stratified cross-validation (5-fold for the large kropt, letter), one
`DataSpec` per training fold (`build_dataspec(discretizer=TreeDiscretizer(max_intervals=8))`).
Each fit is capped at 300s; a time-out counts as a failure, and it fails the whole
cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are
not run, and all its folds count as failures.
Training time is measured inside the worker process, so it excludes
shipping data and models between processes. Every learner runs single-threaded.
Fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

## Pypper

| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| pw_both:accuracy_vote | 81.79% | 2.96 | 250.5 | 670.9 | 2.51 | 0.202 | 0 |
| pw_both:vote | 81.19% | 3.50 | 250.5 | 670.9 | 2.51 | 0.155 | 0 |
| pw_smaller:vote | 80.78% | 4.90 | 123.3 | 332.1 | 1.24 | 0.079 | 0 |
| pw_smaller:accuracy_vote | 80.75% | 4.90 | 123.3 | 332.1 | 1.24 | 0.101 | 0 |
| pw_larger:accuracy_vote | 80.51% | 5.25 | 126.9 | 339.7 | 1.29 | 0.101 | 0 |
| pw_both:weighted_vote | 76.03% | 6.04 | 250.5 | 670.9 | 2.51 | 1.501 | 0 |
| pw_larger:vote | 74.55% | 6.04 | 126.9 | 339.7 | 1.29 | 0.078 | 0 |
| ordered | 79.73% | 6.58 | 29.2 | 127.1 | 0.68 | 0.000 | 0 |
| ovr | 78.36% | 7.31 | 34.0 | 170.1 | 0.92 | 0.005 | 0 |
| pw_smaller:weighted_vote | 66.76% | 8.06 | 123.3 | 332.1 | 1.24 | 0.684 | 0 |
| pw_larger:weighted_vote | 49.27% | 10.46 | 126.9 | 339.7 | 1.29 | 0.822 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## PFossil

| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| pw_both:accuracy_vote | 81.77% | 3.21 | 293.9 | 709.9 | 0.47 | 0.190 | 0 |
| pw_both:vote | 80.30% | 4.50 | 293.9 | 709.9 | 0.47 | 0.145 | 0 |
| pw_larger:accuracy_vote | 80.34% | 5.33 | 147.1 | 341.0 | 0.25 | 0.097 | 0 |
| pw_smaller:accuracy_vote | 79.58% | 5.33 | 145.4 | 368.4 | 0.25 | 0.095 | 0 |
| ovr | 80.13% | 5.37 | 34.9 | 166.8 | 0.14 | 0.003 | 0 |
| pw_smaller:vote | 79.42% | 5.62 | 145.4 | 368.4 | 0.25 | 0.071 | 0 |
| pw_both:weighted_vote | 75.34% | 5.79 | 293.9 | 709.9 | 0.47 | 1.436 | 0 |
| ordered | 78.91% | 6.27 | 28.1 | 122.6 | 0.11 | 0.000 | 0 |
| pw_larger:vote | 76.76% | 6.46 | 147.1 | 341.0 | 0.25 | 0.072 | 0 |
| pw_smaller:weighted_vote | 65.88% | 7.52 | 145.4 | 368.4 | 0.25 | 0.688 | 0 |
| pw_larger:weighted_vote | 48.22% | 10.62 | 147.1 | 341.0 | 0.25 | 0.768 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## PFoil

| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| pw_both:accuracy_vote | 81.60% | 3.10 | 581.1 | 2051.3 | 1.24 | 0.272 | 0 |
| pw_both:vote | 80.76% | 3.54 | 581.1 | 2051.3 | 1.24 | 0.224 | 0 |
| ovr | 81.26% | 3.85 | 187.8 | 1233.1 | 1.46 | 0.050 | 0 |
| pw_both:weighted_vote | 77.78% | 4.37 | 581.1 | 2051.3 | 1.24 | 1.730 | 0 |
| pw_larger:accuracy_vote | 79.46% | 4.92 | 296.6 | 1045.6 | 0.64 | 0.137 | 0 |
| pw_larger:vote | 78.30% | 5.54 | 296.6 | 1045.6 | 0.64 | 0.114 | 0 |
| pw_smaller:vote | 76.45% | 6.92 | 283.5 | 1005.3 | 0.63 | 0.117 | 0 |
| pw_smaller:accuracy_vote | 76.29% | 7.06 | 283.5 | 1005.3 | 0.63 | 0.134 | 0 |
| pw_smaller:weighted_vote | 66.15% | 7.62 | 283.5 | 1005.3 | 0.63 | 0.799 | 0 |
| ordered | 69.23% | 8.85 | 121.8 | 665.4 | 0.69 | 0.001 | 0 |
| pw_larger:weighted_vote | 49.30% | 10.25 | 296.6 | 1045.6 | 0.64 | 0.974 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## Against one-vs-rest, over the number of classes

Per dataset and base learner, the accuracy of a decomposition minus that of one-vs-rest, in percentage points (positive: the decomposition is better), averaged over the datasets with the given numbers of classes and the three base learners.

| classes | pairs | ordered | pw_smaller:vote | pw_smaller:accuracy_vote | pw_both:vote | pw_both:accuracy_vote |
|---|--:|--:|--:|--:|--:|--:|
| 3 | 21 | -2.37 | -0.92 | -1.39 | +0.07 | +0.79 |
| 4-5 | 12 | -2.67 | -1.91 | -1.92 | -0.13 | +1.92 |
| 6-9 | 15 | -6.44 | -3.53 | -3.44 | -0.05 | -0.20 |
| 10-26 | 30 | -3.92 | +1.62 | +1.89 | +3.45 | +4.63 |

Share of the pairs on which the decomposition is better than one-vs-rest, and Spearman's rank correlation of the difference with the number of classes:

| decomposition | ahead | correlation |
|---|--:|--:|
| ordered | 38% | -0.02 (p = 0.84) |
| pw_smaller:vote | 46% | +0.21 (p = 0.065) |
| pw_smaller:accuracy_vote | 47% | +0.22 (p = 0.048) |
| pw_both:vote | 69% | +0.33 (p = 0.0028) |
| pw_both:accuracy_vote | 68% | +0.35 (p = 0.0016) |

## Training and prediction time over the number of classes

One point per dataset and base learner: the mean one-vs-rest time divided by the mean pairwise time. Above 1, pairwise is faster. The dashed line is what a learner whose time grows linearly with the number of examples would give.

![training time ratio, one-vs-rest / pairwise](multiclass_decomposition_plots/multiclass_decomposition_train_ratio.png)

![training time ratio, one-vs-rest / double round robin](multiclass_decomposition_plots/multiclass_decomposition_train_ratio_double.png)

![prediction time ratio, one-vs-rest / pairwise](multiclass_decomposition_plots/multiclass_decomposition_predict_ratio.png)

## Growth of the base learners and the pairwise speed-up

The argument for pairwise classification is about cost: if a learner's training time grows like n^b with the number of examples n, one-vs-rest costs c n^b for c classes, pairwise the sum over the pairs of (n_i + n_j)^b -- about (c - 1) n examples in all, but in many small problems, so the larger b, the more pairwise gains. Here b is measured for each base learner on each dataset with at least 1,000 training rows: one-vs-rest is fitted on nested, stratified subsamples of the first training fold (25%, 50%, 100% of it, outside the cross-validation, each with its own discretization; a learner that times out on one is not tried on the larger ones), and b is the least-squares slope of log time over log rows. The predicted ratio uses the fold's real class counts; the measured one is the cross-validation's one-vs-rest / pairwise (`pw_smaller`) training time. Pairwise also has costs per pair the model leaves out -- building each pair's data, evaluating its model for the vote weights -- which weigh most with many classes and few examples.

Run times of rule learners are noisy in a way their results are not: one rule more or less can change a fit's time a lot and its quality hardly at all, so b from three subsamples, and the ratios, scatter. The points follow the line only roughly; a controlled experiment, varying the training-set size on a single dataset, would separate the growth from the noise better.

![pairwise speed-up, measured vs. predicted](multiclass_decomposition_plots/multiclass_decomposition_growth.png)

| dataset | classes | training rows | base | b | ovr / pairwise, measured | predicted (b) | predicted (b = 1) |
|---|--:|--:|---|--:|--:|--:|--:|
| cmc | 3 | 1,325 | PFoil | 1.19 | 1.35 | 1.62 | 1.50 |
| cmc | 3 | 1,325 | PFossil | 0.17 | 0.49 | 1.07 | 1.50 |
| cmc | 3 | 1,325 | Pypper | 0.13 | 0.84 | 1.05 | 1.50 |
| yeast | 10 | 1,335 | PFoil | 1.27 | 1.11 | 1.57 | 1.11 |
| yeast | 10 | 1,335 | PFossil | 0.39 | 0.42 | 0.45 | 1.11 |
| yeast | 10 | 1,335 | Pypper | 0.73 | 0.58 | 0.76 | 1.11 |
| car | 4 | 1,555 | PFoil | 0.90 | 1.68 | 1.27 | 1.33 |
| car | 4 | 1,555 | PFossil | 0.13 | 1.71 | 0.75 | 1.33 |
| car | 4 | 1,555 | Pypper | 0.69 | 1.15 | 1.13 | 1.33 |
| segment | 7 | 2,079 | PFoil | 0.81 | 1.93 | 0.92 | 1.17 |
| segment | 7 | 2,079 | PFossil | 0.45 | 1.12 | 0.58 | 1.17 |
| segment | 7 | 2,079 | Pypper | 0.70 | 1.27 | 0.80 | 1.17 |
| dna | 3 | 2,867 | PFoil | 1.05 | 1.93 | 1.53 | 1.50 |
| dna | 3 | 2,867 | PFossil | 0.51 | 1.39 | 1.23 | 1.50 |
| dna | 3 | 2,867 | Pypper | 0.59 | 1.94 | 1.28 | 1.50 |
| splice | 3 | 2,871 | PFoil | 1.01 | 1.84 | 1.51 | 1.50 |
| splice | 3 | 2,871 | PFossil | 0.33 | 1.71 | 1.15 | 1.50 |
| splice | 3 | 2,871 | Pypper | 0.87 | 1.68 | 1.43 | 1.50 |
| led24 | 10 | 2,880 | PFoil | 1.30 | 1.53 | 1.81 | 1.11 |
| led24 | 10 | 2,880 | PFossil | -0.07 | 0.47 | 0.20 | 1.11 |
| led24 | 10 | 2,880 | Pypper | 0.63 | 0.64 | 0.62 | 1.11 |
| texture | 11 | 4,950 | PFoil | 1.11 | 1.77 | 1.32 | 1.10 |
| texture | 11 | 4,950 | PFossil | 0.70 | 1.15 | 0.66 | 1.10 |
| texture | 11 | 4,950 | Pypper | 0.96 | 1.43 | 1.02 | 1.10 |
| optdigits | 10 | 5,058 | PFoil | 1.08 | 1.41 | 1.25 | 1.11 |
| optdigits | 10 | 5,058 | PFossil | 0.65 | 0.89 | 0.63 | 1.11 |
| optdigits | 10 | 5,058 | Pypper | 1.01 | 1.25 | 1.12 | 1.11 |
| letter | 26 | 16,000 | PFoil | 1.39 | 1.90 | 2.83 | 1.04 |
| letter | 26 | 16,000 | PFossil | 0.48 | 0.46 | 0.28 | 1.04 |
| letter | 26 | 16,000 | Pypper | 1.27 | 1.06 | 2.09 | 1.04 |
| kropt | 18 | 22,444 | PFoil | 1.46 | 2.94 | 2.57 | 1.06 |
| kropt | 18 | 22,444 | PFossil | 0.74 | 0.35 | 0.63 | 1.06 |
| kropt | 18 | 22,444 | Pypper | 1.42 | 0.48 | 2.36 | 1.06 |

Median b: PFoil 1.11, PFossil 0.45, Pypper 0.73.

## Per dataset (mean across folds)

| dataset | learner | accuracy | n_conditions | train_time | predict_time |
|---|---|---|---|---|---|
| anneal | PFoil:ordered | 92.87% | 54.1 | 0.027 | 0.000 |
| anneal | PFoil:ovr | 97.21% | 68.7 | 0.036 | 0.001 |
| anneal | PFoil:pw_both:accuracy_vote | 99.89% | 95.6 | 0.058 | 0.004 |
| anneal | PFoil:pw_both:vote | 99.11% | 95.6 | 0.058 | 0.003 |
| anneal | PFoil:pw_both:weighted_vote | 76.06% | 95.6 | 0.058 | 0.022 |
| anneal | PFoil:pw_larger:accuracy_vote | 99.33% | 31.7 | 0.026 | 0.002 |
| anneal | PFoil:pw_larger:vote | 99.33% | 31.7 | 0.026 | 0.002 |
| anneal | PFoil:pw_larger:weighted_vote | 76.06% | 31.7 | 0.026 | 0.013 |
| anneal | PFoil:pw_smaller:accuracy_vote | 93.20% | 63.9 | 0.035 | 0.003 |
| anneal | PFoil:pw_smaller:vote | 93.20% | 63.9 | 0.035 | 0.002 |
| anneal | PFoil:pw_smaller:weighted_vote | 96.88% | 63.9 | 0.035 | 0.010 |
| anneal | PFossil:ordered | 93.99% | 21.0 | 0.017 | 0.000 |
| anneal | PFossil:ovr | 99.22% | 30.5 | 0.023 | 0.000 |
| anneal | PFossil:pw_both:accuracy_vote | 99.11% | 63.1 | 0.046 | 0.004 |
| anneal | PFossil:pw_both:vote | 99.00% | 63.1 | 0.046 | 0.003 |
| anneal | PFossil:pw_both:weighted_vote | 76.17% | 63.1 | 0.046 | 0.029 |
| anneal | PFossil:pw_larger:accuracy_vote | 98.89% | 32.4 | 0.024 | 0.002 |
| anneal | PFossil:pw_larger:vote | 98.89% | 32.4 | 0.024 | 0.002 |
| anneal | PFossil:pw_larger:weighted_vote | 76.17% | 32.4 | 0.024 | 0.015 |
| anneal | PFossil:pw_smaller:accuracy_vote | 94.21% | 30.7 | 0.024 | 0.002 |
| anneal | PFossil:pw_smaller:vote | 94.21% | 30.7 | 0.024 | 0.002 |
| anneal | PFossil:pw_smaller:weighted_vote | 90.76% | 30.7 | 0.024 | 0.011 |
| anneal | Pypper:ordered | 94.32% | 22.5 | 0.070 | 0.000 |
| anneal | Pypper:ovr | 95.44% | 32.9 | 0.104 | 0.000 |
| anneal | Pypper:pw_both:accuracy_vote | 97.89% | 54.1 | 0.183 | 0.004 |
| anneal | Pypper:pw_both:vote | 95.99% | 54.1 | 0.183 | 0.003 |
| anneal | Pypper:pw_both:weighted_vote | 76.17% | 54.1 | 0.183 | 0.023 |
| anneal | Pypper:pw_larger:accuracy_vote | 98.11% | 24.1 | 0.095 | 0.002 |
| anneal | Pypper:pw_larger:vote | 98.33% | 24.1 | 0.095 | 0.002 |
| anneal | Pypper:pw_larger:weighted_vote | 76.17% | 24.1 | 0.095 | 0.015 |
| anneal | Pypper:pw_smaller:accuracy_vote | 94.65% | 30.0 | 0.104 | 0.002 |
| anneal | Pypper:pw_smaller:vote | 94.54% | 30.0 | 0.104 | 0.002 |
| anneal | Pypper:pw_smaller:weighted_vote | 93.76% | 30.0 | 0.104 | 0.012 |
| audiology | PFoil:ordered | 62.91% | 57.1 | 0.035 | 0.000 |
| audiology | PFoil:ovr | 77.04% | 94.3 | 0.045 | 0.001 |
| audiology | PFoil:pw_both:accuracy_vote | 67.27% | 284.8 | 0.314 | 0.021 |
| audiology | PFoil:pw_both:vote | 61.52% | 284.8 | 0.314 | 0.015 |
| audiology | PFoil:pw_both:weighted_vote | 46.46% | 284.8 | 0.314 | 0.112 |
| audiology | PFoil:pw_larger:accuracy_vote | 70.42% | 139.0 | 0.166 | 0.011 |
| audiology | PFoil:pw_larger:vote | 58.46% | 139.0 | 0.166 | 0.008 |
| audiology | PFoil:pw_larger:weighted_vote | 46.46% | 139.0 | 0.166 | 0.082 |
| audiology | PFoil:pw_smaller:accuracy_vote | 65.08% | 145.8 | 0.165 | 0.014 |
| audiology | PFoil:pw_smaller:vote | 71.23% | 145.8 | 0.165 | 0.009 |
| audiology | PFoil:pw_smaller:weighted_vote | 54.80% | 145.8 | 0.165 | 0.065 |
| audiology | PFossil:ordered | 84.05% | 72.5 | 0.037 | 0.000 |
| audiology | PFossil:ovr | 77.89% | 115.2 | 0.049 | 0.001 |
| audiology | PFossil:pw_both:accuracy_vote | 81.40% | 586.0 | 0.367 | 0.026 |
| audiology | PFossil:pw_both:vote | 81.40% | 586.0 | 0.367 | 0.020 |
| audiology | PFossil:pw_both:weighted_vote | 46.46% | 586.0 | 0.367 | 0.161 |
| audiology | PFossil:pw_larger:accuracy_vote | 80.53% | 292.0 | 0.194 | 0.013 |
| audiology | PFossil:pw_larger:vote | 81.40% | 292.0 | 0.194 | 0.010 |
| audiology | PFossil:pw_larger:weighted_vote | 46.46% | 292.0 | 0.194 | 0.111 |
| audiology | PFossil:pw_smaller:accuracy_vote | 80.95% | 294.0 | 0.192 | 0.013 |
| audiology | PFossil:pw_smaller:vote | 81.38% | 294.0 | 0.192 | 0.010 |
| audiology | PFossil:pw_smaller:weighted_vote | 54.86% | 294.0 | 0.192 | 0.068 |
| audiology | Pypper:ordered | 75.71% | 17.3 | 0.090 | 0.000 |
| audiology | Pypper:ovr | 66.84% | 32.4 | 0.117 | 0.000 |
| audiology | Pypper:pw_both:accuracy_vote | 68.18% | 143.6 | 1.116 | 0.022 |
| audiology | Pypper:pw_both:vote | 56.60% | 143.6 | 1.116 | 0.015 |
| audiology | Pypper:pw_both:weighted_vote | 46.46% | 143.6 | 1.116 | 0.120 |
| audiology | Pypper:pw_larger:accuracy_vote | 69.51% | 87.7 | 0.637 | 0.010 |
| audiology | Pypper:pw_larger:vote | 7.92% | 87.7 | 0.637 | 0.007 |
| audiology | Pypper:pw_larger:weighted_vote | 46.46% | 87.7 | 0.637 | 0.063 |
| audiology | Pypper:pw_smaller:accuracy_vote | 65.08% | 56.1 | 0.488 | 0.011 |
| audiology | Pypper:pw_smaller:vote | 69.96% | 56.1 | 0.488 | 0.007 |
| audiology | Pypper:pw_smaller:weighted_vote | 49.51% | 56.1 | 0.488 | 0.052 |
| balance-scale | PFoil:ordered | 47.83% | 145.9 | 0.049 | 0.000 |
| balance-scale | PFoil:ovr | 82.87% | 272.8 | 0.092 | 0.002 |
| balance-scale | PFoil:pw_both:accuracy_vote | 80.00% | 374.9 | 0.128 | 0.002 |
| balance-scale | PFoil:pw_both:vote | 79.52% | 374.9 | 0.128 | 0.002 |
| balance-scale | PFoil:pw_both:weighted_vote | 83.20% | 374.9 | 0.128 | 0.012 |
| balance-scale | PFoil:pw_larger:accuracy_vote | 81.91% | 194.4 | 0.064 | 0.001 |
| balance-scale | PFoil:pw_larger:vote | 80.30% | 194.4 | 0.064 | 0.001 |
| balance-scale | PFoil:pw_larger:weighted_vote | 79.50% | 194.4 | 0.064 | 0.006 |
| balance-scale | PFoil:pw_smaller:accuracy_vote | 64.95% | 180.6 | 0.064 | 0.001 |
| balance-scale | PFoil:pw_smaller:vote | 66.88% | 180.6 | 0.064 | 0.001 |
| balance-scale | PFoil:pw_smaller:weighted_vote | 73.27% | 180.6 | 0.064 | 0.006 |
| balance-scale | PFossil:ordered | 78.88% | 39.7 | 0.019 | 0.000 |
| balance-scale | PFossil:ovr | 81.60% | 84.5 | 0.035 | 0.001 |
| balance-scale | PFossil:pw_both:accuracy_vote | 80.15% | 75.3 | 0.036 | 0.001 |
| balance-scale | PFossil:pw_both:vote | 76.63% | 75.3 | 0.036 | 0.001 |
| balance-scale | PFossil:pw_both:weighted_vote | 81.60% | 75.3 | 0.036 | 0.006 |
| balance-scale | PFossil:pw_larger:accuracy_vote | 77.43% | 35.6 | 0.018 | 0.001 |
| balance-scale | PFossil:pw_larger:vote | 7.84% | 35.6 | 0.018 | 0.001 |
| balance-scale | PFossil:pw_larger:weighted_vote | 65.82% | 35.6 | 0.018 | 0.003 |
| balance-scale | PFossil:pw_smaller:accuracy_vote | 78.88% | 40.4 | 0.020 | 0.001 |
| balance-scale | PFossil:pw_smaller:vote | 79.04% | 40.4 | 0.020 | 0.001 |
| balance-scale | PFossil:pw_smaller:weighted_vote | 78.72% | 40.4 | 0.020 | 0.003 |
| balance-scale | Pypper:ordered | 78.38% | 23.8 | 0.083 | 0.000 |
| balance-scale | Pypper:ovr | 81.75% | 48.9 | 0.130 | 0.000 |
| balance-scale | Pypper:pw_both:accuracy_vote | 81.42% | 62.9 | 0.205 | 0.001 |
| balance-scale | Pypper:pw_both:vote | 81.43% | 62.9 | 0.205 | 0.001 |
| balance-scale | Pypper:pw_both:weighted_vote | 80.94% | 62.9 | 0.205 | 0.005 |
| balance-scale | Pypper:pw_larger:accuracy_vote | 70.37% | 39.4 | 0.142 | 0.001 |
| balance-scale | Pypper:pw_larger:vote | 59.69% | 39.4 | 0.142 | 0.001 |
| balance-scale | Pypper:pw_larger:weighted_vote | 77.57% | 39.4 | 0.142 | 0.003 |
| balance-scale | Pypper:pw_smaller:accuracy_vote | 78.38% | 25.0 | 0.082 | 0.001 |
| balance-scale | Pypper:pw_smaller:vote | 78.38% | 25.0 | 0.082 | 0.001 |
| balance-scale | Pypper:pw_smaller:weighted_vote | 78.38% | 25.0 | 0.082 | 0.004 |
| car | PFoil:ordered | 92.36% | 269.8 | 0.095 | 0.000 |
| car | PFoil:ovr | 96.70% | 449.1 | 0.156 | 0.002 |
| car | PFoil:pw_both:accuracy_vote | 98.44% | 360.5 | 0.146 | 0.006 |
| car | PFoil:pw_both:vote | 98.44% | 360.5 | 0.146 | 0.005 |
| car | PFoil:pw_both:weighted_vote | 93.17% | 360.5 | 0.146 | 0.038 |
| car | PFoil:pw_larger:accuracy_vote | 99.07% | 98.8 | 0.056 | 0.003 |
| car | PFoil:pw_larger:vote | 98.96% | 98.8 | 0.056 | 0.004 |
| car | PFoil:pw_larger:weighted_vote | 70.02% | 98.8 | 0.056 | 0.022 |
| car | PFoil:pw_smaller:accuracy_vote | 94.97% | 261.7 | 0.093 | 0.004 |
| car | PFoil:pw_smaller:vote | 95.20% | 261.7 | 0.093 | 0.003 |
| car | PFoil:pw_smaller:weighted_vote | 95.83% | 261.7 | 0.093 | 0.015 |
| car | PFossil:ordered | 82.87% | 24.4 | 0.015 | 0.000 |
| car | PFossil:ovr | 93.87% | 59.1 | 0.035 | 0.001 |
| car | PFossil:pw_both:accuracy_vote | 96.76% | 109.1 | 0.062 | 0.005 |
| car | PFossil:pw_both:vote | 83.28% | 109.1 | 0.062 | 0.004 |
| car | PFossil:pw_both:weighted_vote | 75.75% | 109.1 | 0.062 | 0.040 |
| car | PFossil:pw_larger:accuracy_vote | 94.50% | 73.7 | 0.043 | 0.003 |
| car | PFossil:pw_larger:vote | 93.40% | 73.7 | 0.043 | 0.003 |
| car | PFossil:pw_larger:weighted_vote | 70.02% | 73.7 | 0.043 | 0.019 |
| car | PFossil:pw_smaller:accuracy_vote | 82.93% | 35.4 | 0.020 | 0.003 |
| car | PFossil:pw_smaller:vote | 82.93% | 35.4 | 0.020 | 0.002 |
| car | PFossil:pw_smaller:weighted_vote | 82.93% | 35.4 | 0.020 | 0.011 |
| car | Pypper:ordered | 94.56% | 118.9 | 0.245 | 0.000 |
| car | Pypper:ovr | 92.36% | 138.7 | 0.292 | 0.001 |
| car | Pypper:pw_both:accuracy_vote | 98.09% | 203.0 | 0.424 | 0.006 |
| car | Pypper:pw_both:vote | 97.69% | 203.0 | 0.424 | 0.004 |
| car | Pypper:pw_both:weighted_vote | 92.36% | 203.0 | 0.424 | 0.036 |
| car | Pypper:pw_larger:accuracy_vote | 98.44% | 77.8 | 0.170 | 0.003 |
| car | Pypper:pw_larger:vote | 98.67% | 77.8 | 0.170 | 0.003 |
| car | Pypper:pw_larger:weighted_vote | 70.02% | 77.8 | 0.170 | 0.021 |
| car | Pypper:pw_smaller:accuracy_vote | 94.85% | 125.2 | 0.255 | 0.003 |
| car | Pypper:pw_smaller:vote | 95.14% | 125.2 | 0.255 | 0.002 |
| car | Pypper:pw_smaller:weighted_vote | 94.44% | 125.2 | 0.255 | 0.013 |
| cmc | PFoil:ordered | 35.71% | 781.6 | 0.258 | 0.001 |
| cmc | PFoil:ovr | 51.40% | 1266.9 | 0.444 | 0.008 |
| cmc | PFoil:pw_both:accuracy_vote | 47.26% | 2015.3 | 0.669 | 0.009 |
| cmc | PFoil:pw_both:vote | 45.56% | 2015.3 | 0.669 | 0.009 |
| cmc | PFoil:pw_both:weighted_vote | 52.56% | 2015.3 | 0.669 | 0.033 |
| cmc | PFoil:pw_larger:accuracy_vote | 43.45% | 1017.1 | 0.349 | 0.006 |
| cmc | PFoil:pw_larger:vote | 42.77% | 1017.1 | 0.349 | 0.004 |
| cmc | PFoil:pw_larger:weighted_vote | 45.76% | 1017.1 | 0.349 | 0.019 |
| cmc | PFoil:pw_smaller:accuracy_vote | 39.24% | 998.2 | 0.329 | 0.005 |
| cmc | PFoil:pw_smaller:vote | 42.23% | 998.2 | 0.329 | 0.004 |
| cmc | PFoil:pw_smaller:weighted_vote | 47.18% | 998.2 | 0.329 | 0.016 |
| cmc | PFossil:ordered | 51.53% | 22.4 | 0.014 | 0.000 |
| cmc | PFossil:ovr | 52.61% | 9.2 | 0.012 | 0.000 |
| cmc | PFossil:pw_both:accuracy_vote | 52.34% | 57.9 | 0.033 | 0.003 |
| cmc | PFossil:pw_both:vote | 48.27% | 57.9 | 0.033 | 0.002 |
| cmc | PFossil:pw_both:weighted_vote | 50.44% | 57.9 | 0.033 | 0.011 |
| cmc | PFossil:pw_larger:accuracy_vote | 47.38% | 9.2 | 0.013 | 0.001 |
| cmc | PFossil:pw_larger:vote | 48.40% | 9.2 | 0.013 | 0.002 |
| cmc | PFossil:pw_larger:weighted_vote | 43.59% | 9.2 | 0.013 | 0.005 |
| cmc | PFossil:pw_smaller:accuracy_vote | 47.39% | 48.7 | 0.024 | 0.002 |
| cmc | PFossil:pw_smaller:vote | 48.13% | 48.7 | 0.024 | 0.001 |
| cmc | PFossil:pw_smaller:weighted_vote | 47.86% | 48.7 | 0.024 | 0.006 |
| cmc | Pypper:ordered | 54.52% | 13.5 | 0.103 | 0.000 |
| cmc | Pypper:ovr | 52.21% | 20.7 | 0.139 | 0.000 |
| cmc | Pypper:pw_both:accuracy_vote | 54.65% | 44.6 | 0.309 | 0.003 |
| cmc | Pypper:pw_both:vote | 54.05% | 44.6 | 0.309 | 0.002 |
| cmc | Pypper:pw_both:weighted_vote | 53.50% | 44.6 | 0.309 | 0.013 |
| cmc | Pypper:pw_larger:accuracy_vote | 53.02% | 19.7 | 0.151 | 0.002 |
| cmc | Pypper:pw_larger:vote | 53.50% | 19.7 | 0.151 | 0.001 |
| cmc | Pypper:pw_larger:weighted_vote | 45.49% | 19.7 | 0.151 | 0.007 |
| cmc | Pypper:pw_smaller:accuracy_vote | 53.37% | 24.9 | 0.164 | 0.001 |
| cmc | Pypper:pw_smaller:vote | 53.10% | 24.9 | 0.164 | 0.001 |
| cmc | Pypper:pw_smaller:weighted_vote | 52.96% | 24.9 | 0.164 | 0.006 |
| dna | PFoil:ordered | 86.75% | 263.8 | 0.154 | 0.000 |
| dna | PFoil:ovr | 92.50% | 469.4 | 0.298 | 0.004 |
| dna | PFoil:pw_both:accuracy_vote | 92.75% | 464.5 | 0.301 | 0.008 |
| dna | PFoil:pw_both:vote | 92.78% | 464.5 | 0.301 | 0.008 |
| dna | PFoil:pw_both:weighted_vote | 92.66% | 464.5 | 0.301 | 0.044 |
| dna | PFoil:pw_larger:accuracy_vote | 91.40% | 211.4 | 0.156 | 0.005 |
| dna | PFoil:pw_larger:vote | 90.71% | 211.4 | 0.156 | 0.004 |
| dna | PFoil:pw_larger:weighted_vote | 51.91% | 211.4 | 0.156 | 0.027 |
| dna | PFoil:pw_smaller:accuracy_vote | 88.29% | 253.1 | 0.154 | 0.005 |
| dna | PFoil:pw_smaller:vote | 89.80% | 253.1 | 0.154 | 0.004 |
| dna | PFoil:pw_smaller:weighted_vote | 88.73% | 253.1 | 0.154 | 0.022 |
| dna | PFossil:ordered | 89.99% | 88.1 | 0.059 | 0.000 |
| dna | PFossil:ovr | 93.69% | 137.4 | 0.100 | 0.002 |
| dna | PFossil:pw_both:accuracy_vote | 93.60% | 195.2 | 0.151 | 0.006 |
| dna | PFossil:pw_both:vote | 92.91% | 195.2 | 0.151 | 0.007 |
| dna | PFossil:pw_both:weighted_vote | 92.53% | 195.2 | 0.151 | 0.044 |
| dna | PFossil:pw_larger:accuracy_vote | 92.75% | 93.3 | 0.086 | 0.004 |
| dna | PFossil:pw_larger:vote | 92.40% | 93.3 | 0.086 | 0.003 |
| dna | PFossil:pw_larger:weighted_vote | 51.91% | 93.3 | 0.086 | 0.027 |
| dna | PFossil:pw_smaller:accuracy_vote | 93.16% | 101.9 | 0.072 | 0.003 |
| dna | PFossil:pw_smaller:vote | 93.16% | 101.9 | 0.072 | 0.003 |
| dna | PFossil:pw_smaller:weighted_vote | 92.44% | 101.9 | 0.072 | 0.017 |
| dna | Pypper:ordered | 92.12% | 82.6 | 0.315 | 0.000 |
| dna | Pypper:ovr | 93.13% | 125.8 | 0.506 | 0.001 |
| dna | Pypper:pw_both:accuracy_vote | 94.57% | 119.0 | 0.492 | 0.006 |
| dna | Pypper:pw_both:vote | 94.70% | 119.0 | 0.492 | 0.005 |
| dna | Pypper:pw_both:weighted_vote | 94.04% | 119.0 | 0.492 | 0.037 |
| dna | Pypper:pw_larger:accuracy_vote | 94.16% | 59.4 | 0.241 | 0.004 |
| dna | Pypper:pw_larger:vote | 92.94% | 59.4 | 0.241 | 0.003 |
| dna | Pypper:pw_larger:weighted_vote | 51.91% | 59.4 | 0.241 | 0.022 |
| dna | Pypper:pw_smaller:accuracy_vote | 93.35% | 59.6 | 0.261 | 0.003 |
| dna | Pypper:pw_smaller:vote | 93.91% | 59.6 | 0.261 | 0.003 |
| dna | Pypper:pw_smaller:weighted_vote | 92.87% | 59.6 | 0.261 | 0.020 |
| ecoli | PFoil:ordered | 55.64% | 58.5 | 0.024 | 0.000 |
| ecoli | PFoil:ovr | 82.51% | 139.2 | 0.047 | 0.001 |
| ecoli | PFoil:pw_both:accuracy_vote | 86.36% | 173.0 | 0.081 | 0.005 |
| ecoli | PFoil:pw_both:vote | 85.47% | 173.0 | 0.081 | 0.004 |
| ecoli | PFoil:pw_both:weighted_vote | 75.64% | 173.0 | 0.081 | 0.025 |
| ecoli | PFoil:pw_larger:accuracy_vote | 83.69% | 94.0 | 0.045 | 0.002 |
| ecoli | PFoil:pw_larger:vote | 82.19% | 94.0 | 0.045 | 0.002 |
| ecoli | PFoil:pw_larger:weighted_vote | 48.52% | 94.0 | 0.045 | 0.014 |
| ecoli | PFoil:pw_smaller:accuracy_vote | 75.61% | 79.0 | 0.039 | 0.002 |
| ecoli | PFoil:pw_smaller:vote | 69.66% | 79.0 | 0.039 | 0.002 |
| ecoli | PFoil:pw_smaller:weighted_vote | 79.50% | 79.0 | 0.039 | 0.012 |
| ecoli | PFossil:ordered | 75.64% | 73.0 | 0.027 | 0.000 |
| ecoli | PFossil:ovr | 81.92% | 107.5 | 0.038 | 0.000 |
| ecoli | PFossil:pw_both:accuracy_vote | 81.00% | 173.4 | 0.077 | 0.004 |
| ecoli | PFossil:pw_both:vote | 82.79% | 173.4 | 0.077 | 0.004 |
| ecoli | PFossil:pw_both:weighted_vote | 69.07% | 173.4 | 0.077 | 0.031 |
| ecoli | PFossil:pw_larger:accuracy_vote | 83.67% | 68.9 | 0.036 | 0.003 |
| ecoli | PFossil:pw_larger:vote | 81.31% | 68.9 | 0.036 | 0.002 |
| ecoli | PFossil:pw_larger:weighted_vote | 46.72% | 68.9 | 0.036 | 0.014 |
| ecoli | PFossil:pw_smaller:accuracy_vote | 76.52% | 104.5 | 0.044 | 0.002 |
| ecoli | PFossil:pw_smaller:vote | 75.34% | 104.5 | 0.044 | 0.002 |
| ecoli | PFossil:pw_smaller:weighted_vote | 78.93% | 104.5 | 0.044 | 0.011 |
| ecoli | Pypper:ordered | 81.01% | 15.2 | 0.051 | 0.000 |
| ecoli | Pypper:ovr | 82.22% | 27.4 | 0.081 | 0.000 |
| ecoli | Pypper:pw_both:accuracy_vote | 82.76% | 47.3 | 0.237 | 0.004 |
| ecoli | Pypper:pw_both:vote | 82.47% | 47.3 | 0.237 | 0.003 |
| ecoli | Pypper:pw_both:weighted_vote | 69.05% | 47.3 | 0.237 | 0.030 |
| ecoli | Pypper:pw_larger:accuracy_vote | 80.97% | 24.4 | 0.138 | 0.002 |
| ecoli | Pypper:pw_larger:vote | 49.02% | 24.4 | 0.138 | 0.002 |
| ecoli | Pypper:pw_larger:weighted_vote | 43.46% | 24.4 | 0.138 | 0.011 |
| ecoli | Pypper:pw_smaller:accuracy_vote | 82.19% | 22.9 | 0.103 | 0.002 |
| ecoli | Pypper:pw_smaller:vote | 81.89% | 22.9 | 0.103 | 0.001 |
| ecoli | Pypper:pw_smaller:weighted_vote | 81.01% | 22.9 | 0.103 | 0.009 |
| glass | PFoil:ordered | 49.13% | 51.3 | 0.020 | 0.000 |
| glass | PFoil:ovr | 72.94% | 104.6 | 0.037 | 0.001 |
| glass | PFoil:pw_both:accuracy_vote | 66.80% | 138.4 | 0.060 | 0.002 |
| glass | PFoil:pw_both:vote | 67.81% | 138.4 | 0.060 | 0.001 |
| glass | PFoil:pw_both:weighted_vote | 67.86% | 138.4 | 0.060 | 0.010 |
| glass | PFoil:pw_larger:accuracy_vote | 65.91% | 72.1 | 0.033 | 0.001 |
| glass | PFoil:pw_larger:vote | 64.44% | 72.1 | 0.033 | 0.001 |
| glass | PFoil:pw_larger:weighted_vote | 44.96% | 72.1 | 0.033 | 0.006 |
| glass | PFoil:pw_smaller:accuracy_vote | 56.97% | 66.3 | 0.029 | 0.001 |
| glass | PFoil:pw_smaller:vote | 62.60% | 66.3 | 0.029 | 0.001 |
| glass | PFoil:pw_smaller:weighted_vote | 67.73% | 66.3 | 0.029 | 0.005 |
| glass | PFossil:ordered | 63.55% | 45.5 | 0.018 | 0.000 |
| glass | PFossil:ovr | 71.97% | 104.5 | 0.036 | 0.000 |
| glass | PFossil:pw_both:accuracy_vote | 69.68% | 138.5 | 0.059 | 0.002 |
| glass | PFossil:pw_both:vote | 70.65% | 138.5 | 0.059 | 0.001 |
| glass | PFossil:pw_both:weighted_vote | 63.14% | 138.5 | 0.059 | 0.010 |
| glass | PFossil:pw_larger:accuracy_vote | 71.15% | 62.0 | 0.029 | 0.001 |
| glass | PFossil:pw_larger:vote | 70.69% | 62.0 | 0.029 | 0.001 |
| glass | PFossil:pw_larger:weighted_vote | 48.61% | 62.0 | 0.029 | 0.006 |
| glass | PFossil:pw_smaller:accuracy_vote | 65.93% | 76.5 | 0.031 | 0.001 |
| glass | PFossil:pw_smaller:vote | 68.25% | 76.5 | 0.031 | 0.001 |
| glass | PFossil:pw_smaller:weighted_vote | 67.29% | 76.5 | 0.031 | 0.005 |
| glass | Pypper:ordered | 67.38% | 14.3 | 0.050 | 0.000 |
| glass | Pypper:ovr | 67.77% | 24.7 | 0.094 | 0.000 |
| glass | Pypper:pw_both:accuracy_vote | 67.32% | 55.9 | 0.207 | 0.002 |
| glass | Pypper:pw_both:vote | 72.45% | 55.9 | 0.207 | 0.001 |
| glass | Pypper:pw_both:weighted_vote | 66.41% | 55.9 | 0.207 | 0.008 |
| glass | Pypper:pw_larger:accuracy_vote | 68.20% | 28.8 | 0.113 | 0.001 |
| glass | Pypper:pw_larger:vote | 68.74% | 28.8 | 0.113 | 0.001 |
| glass | Pypper:pw_larger:weighted_vote | 45.80% | 28.8 | 0.113 | 0.005 |
| glass | Pypper:pw_smaller:accuracy_vote | 68.77% | 27.1 | 0.099 | 0.001 |
| glass | Pypper:pw_smaller:vote | 68.77% | 27.1 | 0.099 | 0.001 |
| glass | Pypper:pw_smaller:weighted_vote | 68.31% | 27.1 | 0.099 | 0.004 |
| hayes-roth | PFoil:ordered | 86.25% | 28.1 | 0.010 | 0.000 |
| hayes-roth | PFoil:ovr | 77.50% | 68.4 | 0.021 | 0.000 |
| hayes-roth | PFoil:pw_both:accuracy_vote | 85.62% | 63.5 | 0.021 | 0.000 |
| hayes-roth | PFoil:pw_both:vote | 86.88% | 63.5 | 0.021 | 0.000 |
| hayes-roth | PFoil:pw_both:weighted_vote | 84.38% | 63.5 | 0.021 | 0.002 |
| hayes-roth | PFoil:pw_larger:accuracy_vote | 86.25% | 32.1 | 0.011 | 0.000 |
| hayes-roth | PFoil:pw_larger:vote | 86.25% | 32.1 | 0.011 | 0.000 |
| hayes-roth | PFoil:pw_larger:weighted_vote | 54.37% | 32.1 | 0.011 | 0.001 |
| hayes-roth | PFoil:pw_smaller:accuracy_vote | 86.25% | 31.1 | 0.011 | 0.000 |
| hayes-roth | PFoil:pw_smaller:vote | 86.25% | 31.1 | 0.011 | 0.000 |
| hayes-roth | PFoil:pw_smaller:weighted_vote | 86.25% | 31.1 | 0.011 | 0.001 |
| hayes-roth | PFossil:ordered | 85.62% | 12.6 | 0.006 | 0.000 |
| hayes-roth | PFossil:ovr | 82.50% | 38.8 | 0.013 | 0.000 |
| hayes-roth | PFossil:pw_both:accuracy_vote | 83.75% | 31.5 | 0.013 | 0.000 |
| hayes-roth | PFossil:pw_both:vote | 83.75% | 31.5 | 0.013 | 0.000 |
| hayes-roth | PFossil:pw_both:weighted_vote | 83.75% | 31.5 | 0.013 | 0.002 |
| hayes-roth | PFossil:pw_larger:accuracy_vote | 83.12% | 14.5 | 0.006 | 0.000 |
| hayes-roth | PFossil:pw_larger:vote | 83.12% | 14.5 | 0.006 | 0.000 |
| hayes-roth | PFossil:pw_larger:weighted_vote | 52.50% | 14.5 | 0.006 | 0.001 |
| hayes-roth | PFossil:pw_smaller:accuracy_vote | 85.62% | 15.6 | 0.007 | 0.000 |
| hayes-roth | PFossil:pw_smaller:vote | 85.62% | 15.6 | 0.007 | 0.000 |
| hayes-roth | PFossil:pw_smaller:weighted_vote | 85.62% | 15.6 | 0.007 | 0.001 |
| hayes-roth | Pypper:ordered | 84.38% | 9.6 | 0.029 | 0.000 |
| hayes-roth | Pypper:ovr | 80.62% | 19.1 | 0.052 | 0.000 |
| hayes-roth | Pypper:pw_both:accuracy_vote | 84.38% | 24.8 | 0.064 | 0.000 |
| hayes-roth | Pypper:pw_both:vote | 83.75% | 24.8 | 0.064 | 0.000 |
| hayes-roth | Pypper:pw_both:weighted_vote | 86.88% | 24.8 | 0.064 | 0.001 |
| hayes-roth | Pypper:pw_larger:accuracy_vote | 82.50% | 12.3 | 0.029 | 0.000 |
| hayes-roth | Pypper:pw_larger:vote | 82.50% | 12.3 | 0.029 | 0.000 |
| hayes-roth | Pypper:pw_larger:weighted_vote | 51.25% | 12.3 | 0.029 | 0.001 |
| hayes-roth | Pypper:pw_smaller:accuracy_vote | 84.38% | 12.6 | 0.036 | 0.000 |
| hayes-roth | Pypper:pw_smaller:vote | 84.38% | 12.6 | 0.036 | 0.000 |
| hayes-roth | Pypper:pw_smaller:weighted_vote | 84.38% | 12.6 | 0.036 | 0.001 |
| iris | PFoil:ordered | 90.00% | 6.8 | 0.003 | 0.000 |
| iris | PFoil:ovr | 92.67% | 20.4 | 0.007 | 0.000 |
| iris | PFoil:pw_both:accuracy_vote | 92.00% | 17.1 | 0.008 | 0.000 |
| iris | PFoil:pw_both:vote | 89.33% | 17.1 | 0.008 | 0.000 |
| iris | PFoil:pw_both:weighted_vote | 92.67% | 17.1 | 0.008 | 0.001 |
| iris | PFoil:pw_larger:accuracy_vote | 90.00% | 7.8 | 0.004 | 0.000 |
| iris | PFoil:pw_larger:vote | 90.00% | 7.8 | 0.004 | 0.000 |
| iris | PFoil:pw_larger:weighted_vote | 90.00% | 7.8 | 0.004 | 0.001 |
| iris | PFoil:pw_smaller:accuracy_vote | 90.00% | 7.8 | 0.004 | 0.000 |
| iris | PFoil:pw_smaller:vote | 90.00% | 7.8 | 0.004 | 0.000 |
| iris | PFoil:pw_smaller:weighted_vote | 90.00% | 7.8 | 0.004 | 0.001 |
| iris | PFossil:ordered | 92.67% | 8.1 | 0.004 | 0.000 |
| iris | PFossil:ovr | 92.00% | 11.8 | 0.006 | 0.000 |
| iris | PFossil:pw_both:accuracy_vote | 92.00% | 15.4 | 0.008 | 0.000 |
| iris | PFossil:pw_both:vote | 92.67% | 15.4 | 0.008 | 0.000 |
| iris | PFossil:pw_both:weighted_vote | 93.33% | 15.4 | 0.008 | 0.001 |
| iris | PFossil:pw_larger:accuracy_vote | 92.67% | 9.1 | 0.004 | 0.000 |
| iris | PFossil:pw_larger:vote | 92.67% | 9.1 | 0.004 | 0.000 |
| iris | PFossil:pw_larger:weighted_vote | 92.67% | 9.1 | 0.004 | 0.001 |
| iris | PFossil:pw_smaller:accuracy_vote | 92.67% | 9.1 | 0.004 | 0.000 |
| iris | PFossil:pw_smaller:vote | 92.67% | 9.1 | 0.004 | 0.000 |
| iris | PFossil:pw_smaller:weighted_vote | 92.67% | 9.1 | 0.004 | 0.001 |
| iris | Pypper:ordered | 93.33% | 2.8 | 0.012 | 0.000 |
| iris | Pypper:ovr | 92.00% | 5.4 | 0.020 | 0.000 |
| iris | Pypper:pw_both:accuracy_vote | 93.33% | 7.5 | 0.030 | 0.000 |
| iris | Pypper:pw_both:vote | 92.67% | 7.5 | 0.030 | 0.000 |
| iris | Pypper:pw_both:weighted_vote | 92.67% | 7.5 | 0.030 | 0.001 |
| iris | Pypper:pw_larger:accuracy_vote | 93.33% | 3.8 | 0.016 | 0.000 |
| iris | Pypper:pw_larger:vote | 93.33% | 3.8 | 0.016 | 0.000 |
| iris | Pypper:pw_larger:weighted_vote | 93.33% | 3.8 | 0.016 | 0.001 |
| iris | Pypper:pw_smaller:accuracy_vote | 93.33% | 3.8 | 0.016 | 0.000 |
| iris | Pypper:pw_smaller:vote | 93.33% | 3.8 | 0.016 | 0.000 |
| iris | Pypper:pw_smaller:weighted_vote | 93.33% | 3.8 | 0.016 | 0.001 |
| kropt | PFoil:ordered | 69.80% | 15119.2 | 22.704 | 0.029 |
| kropt | PFoil:ovr | 75.41% | 31233.6 | 50.521 | 1.938 |
| kropt | PFoil:pw_both:accuracy_vote | 82.10% | 47380.2 | 34.107 | 6.764 |
| kropt | PFoil:pw_both:vote | 81.88% | 47380.2 | 34.107 | 5.894 |
| kropt | PFoil:pw_both:weighted_vote | 64.03% | 47380.2 | 34.107 | 34.605 |
| kropt | PFoil:pw_larger:accuracy_vote | 75.46% | 24408.4 | 17.192 | 3.443 |
| kropt | PFoil:pw_larger:vote | 75.29% | 24408.4 | 17.192 | 3.035 |
| kropt | PFoil:pw_larger:weighted_vote | 32.70% | 24408.4 | 17.192 | 23.003 |
| kropt | PFoil:pw_smaller:accuracy_vote | 75.78% | 22971.8 | 17.185 | 3.304 |
| kropt | PFoil:pw_smaller:vote | 78.50% | 22971.8 | 17.185 | 3.194 |
| kropt | PFoil:pw_smaller:weighted_vote | 72.94% | 22971.8 | 17.185 | 13.270 |
| kropt | PFossil:ordered | 35.48% | 420.6 | 1.036 | 0.001 |
| kropt | PFossil:ovr | 24.99% | 342.6 | 0.893 | 0.030 |
| kropt | PFossil:pw_both:accuracy_vote | 40.70% | 5670.4 | 4.797 | 3.441 |
| kropt | PFossil:pw_both:vote | 39.41% | 5670.4 | 4.797 | 2.578 |
| kropt | PFossil:pw_both:weighted_vote | 32.21% | 5670.4 | 4.797 | 24.693 |
| kropt | PFossil:pw_larger:accuracy_vote | 36.74% | 2446.2 | 2.441 | 1.755 |
| kropt | PFossil:pw_larger:vote | 34.82% | 2446.2 | 2.441 | 1.288 |
| kropt | PFossil:pw_larger:weighted_vote | 22.80% | 2446.2 | 2.441 | 14.590 |
| kropt | PFossil:pw_smaller:accuracy_vote | 39.58% | 3224.2 | 2.565 | 1.627 |
| kropt | PFossil:pw_smaller:vote | 36.35% | 3224.2 | 2.565 | 1.246 |
| kropt | PFossil:pw_smaller:weighted_vote | 39.95% | 3224.2 | 2.565 | 10.223 |
| kropt | Pypper:ordered | 58.91% | 1908.4 | 15.219 | 0.007 |
| kropt | Pypper:ovr | 37.11% | 1813.2 | 14.098 | 0.100 |
| kropt | Pypper:pw_both:accuracy_vote | 70.93% | 14687.8 | 61.076 | 4.186 |
| kropt | Pypper:pw_both:vote | 71.07% | 14687.8 | 61.076 | 3.368 |
| kropt | Pypper:pw_both:weighted_vote | 53.92% | 14687.8 | 61.076 | 29.612 |
| kropt | Pypper:pw_larger:accuracy_vote | 65.76% | 7569.6 | 31.295 | 2.134 |
| kropt | Pypper:pw_larger:vote | 66.06% | 7569.6 | 31.295 | 1.742 |
| kropt | Pypper:pw_larger:weighted_vote | 30.97% | 7569.6 | 31.295 | 18.492 |
| kropt | Pypper:pw_smaller:accuracy_vote | 67.32% | 7118.2 | 29.679 | 2.093 |
| kropt | Pypper:pw_smaller:vote | 67.89% | 7118.2 | 29.679 | 1.768 |
| kropt | Pypper:pw_smaller:weighted_vote | 63.93% | 7118.2 | 29.679 | 10.806 |
| led24 | PFoil:ordered | 47.06% | 2042.6 | 0.703 | 0.003 |
| led24 | PFoil:ovr | 61.81% | 3508.9 | 1.335 | 0.018 |
| led24 | PFoil:pw_both:accuracy_vote | 64.12% | 5456.2 | 1.735 | 0.091 |
| led24 | PFoil:pw_both:vote | 63.56% | 5456.2 | 1.735 | 0.074 |
| led24 | PFoil:pw_both:weighted_vote | 69.28% | 5456.2 | 1.735 | 0.674 |
| led24 | PFoil:pw_larger:accuracy_vote | 58.09% | 2730.4 | 0.878 | 0.046 |
| led24 | PFoil:pw_larger:vote | 55.44% | 2730.4 | 0.878 | 0.039 |
| led24 | PFoil:pw_larger:weighted_vote | 29.87% | 2730.4 | 0.878 | 0.308 |
| led24 | PFoil:pw_smaller:accuracy_vote | 54.34% | 2723.0 | 0.873 | 0.044 |
| led24 | PFoil:pw_smaller:vote | 53.88% | 2723.0 | 0.873 | 0.038 |
| led24 | PFoil:pw_smaller:weighted_vote | 30.91% | 2723.0 | 0.873 | 0.317 |
| led24 | PFossil:ordered | 70.03% | 250.2 | 0.120 | 0.000 |
| led24 | PFossil:ovr | 70.00% | 272.8 | 0.156 | 0.003 |
| led24 | PFossil:pw_both:accuracy_vote | 70.88% | 1698.1 | 0.672 | 0.059 |
| led24 | PFossil:pw_both:vote | 71.47% | 1698.1 | 0.672 | 0.046 |
| led24 | PFossil:pw_both:weighted_vote | 71.41% | 1698.1 | 0.672 | 0.477 |
| led24 | PFossil:pw_larger:accuracy_vote | 66.44% | 891.4 | 0.352 | 0.061 |
| led24 | PFossil:pw_larger:vote | 66.84% | 891.4 | 0.352 | 0.025 |
| led24 | PFossil:pw_larger:weighted_vote | 28.56% | 891.4 | 0.352 | 0.287 |
| led24 | PFossil:pw_smaller:accuracy_vote | 67.31% | 812.8 | 0.335 | 0.031 |
| led24 | PFossil:pw_smaller:vote | 67.34% | 812.8 | 0.335 | 0.025 |
| led24 | PFossil:pw_smaller:weighted_vote | 30.75% | 812.8 | 0.335 | 0.231 |
| led24 | Pypper:ordered | 70.88% | 125.4 | 0.395 | 0.000 |
| led24 | Pypper:ovr | 68.25% | 180.2 | 0.589 | 0.002 |
| led24 | Pypper:pw_both:accuracy_vote | 72.25% | 556.7 | 1.813 | 0.051 |
| led24 | Pypper:pw_both:vote | 72.16% | 556.7 | 1.813 | 0.038 |
| led24 | Pypper:pw_both:weighted_vote | 72.38% | 556.7 | 1.813 | 0.386 |
| led24 | Pypper:pw_larger:accuracy_vote | 71.56% | 277.4 | 0.902 | 0.027 |
| led24 | Pypper:pw_larger:vote | 71.41% | 277.4 | 0.902 | 0.020 |
| led24 | Pypper:pw_larger:weighted_vote | 30.69% | 277.4 | 0.902 | 0.216 |
| led24 | Pypper:pw_smaller:accuracy_vote | 72.28% | 280.1 | 0.914 | 0.027 |
| led24 | Pypper:pw_smaller:vote | 72.09% | 280.1 | 0.914 | 0.020 |
| led24 | Pypper:pw_smaller:weighted_vote | 34.47% | 280.1 | 0.914 | 0.188 |
| letter | PFoil:ordered | 59.28% | 4562.0 | 5.430 | 0.008 |
| letter | PFoil:ovr | 88.14% | 7100.6 | 10.884 | 0.408 |
| letter | PFoil:pw_both:accuracy_vote | 91.88% | 18456.8 | 11.021 | 5.757 |
| letter | PFoil:pw_both:vote | 92.01% | 18456.8 | 11.021 | 4.473 |
| letter | PFoil:pw_both:weighted_vote | 92.34% | 18456.8 | 11.021 | 44.299 |
| letter | PFoil:pw_larger:accuracy_vote | 89.43% | 9317.6 | 5.754 | 2.869 |
| letter | PFoil:pw_larger:vote | 89.75% | 9317.6 | 5.754 | 2.218 |
| letter | PFoil:pw_larger:weighted_vote | 26.68% | 9317.6 | 5.754 | 21.702 |
| letter | PFoil:pw_smaller:accuracy_vote | 89.16% | 9141.8 | 5.732 | 2.853 |
| letter | PFoil:pw_smaller:vote | 89.12% | 9141.8 | 5.732 | 2.222 |
| letter | PFoil:pw_smaller:weighted_vote | 19.82% | 9141.8 | 5.732 | 22.981 |
| letter | PFossil:ordered | 77.29% | 1170.6 | 1.402 | 0.002 |
| letter | PFossil:ovr | 75.67% | 1249.6 | 1.925 | 0.050 |
| letter | PFossil:pw_both:accuracy_vote | 86.04% | 11414.6 | 7.926 | 5.119 |
| letter | PFossil:pw_both:vote | 85.99% | 11414.6 | 7.926 | 3.950 |
| letter | PFossil:pw_both:weighted_vote | 86.30% | 11414.6 | 7.926 | 40.057 |
| letter | PFossil:pw_larger:accuracy_vote | 84.05% | 5755.2 | 4.183 | 2.533 |
| letter | PFossil:pw_larger:vote | 83.32% | 5755.2 | 4.183 | 1.918 |
| letter | PFossil:pw_larger:weighted_vote | 26.63% | 5755.2 | 4.183 | 20.006 |
| letter | PFossil:pw_smaller:accuracy_vote | 83.11% | 5675.0 | 4.203 | 2.549 |
| letter | PFossil:pw_smaller:vote | 82.84% | 5675.0 | 4.203 | 1.932 |
| letter | PFossil:pw_smaller:weighted_vote | 20.05% | 5675.0 | 4.203 | 20.859 |
| letter | Pypper:ordered | 82.80% | 1644.6 | 8.189 | 0.003 |
| letter | Pypper:ovr | 80.34% | 2349.2 | 15.144 | 0.114 |
| letter | Pypper:pw_both:accuracy_vote | 90.73% | 10007.0 | 28.351 | 4.968 |
| letter | Pypper:pw_both:vote | 90.71% | 10007.0 | 28.351 | 3.710 |
| letter | Pypper:pw_both:weighted_vote | 90.44% | 10007.0 | 28.351 | 39.174 |
| letter | Pypper:pw_larger:accuracy_vote | 89.04% | 5033.0 | 14.540 | 2.458 |
| letter | Pypper:pw_larger:vote | 89.06% | 5033.0 | 14.540 | 1.841 |
| letter | Pypper:pw_larger:weighted_vote | 27.48% | 5033.0 | 14.540 | 19.279 |
| letter | Pypper:pw_smaller:accuracy_vote | 88.89% | 4970.4 | 14.219 | 2.465 |
| letter | Pypper:pw_smaller:vote | 88.89% | 4970.4 | 14.219 | 1.866 |
| letter | Pypper:pw_smaller:weighted_vote | 21.62% | 4970.4 | 14.219 | 20.358 |
| lymph | PFoil:ordered | 79.81% | 18.6 | 0.008 | 0.000 |
| lymph | PFoil:ovr | 82.57% | 40.5 | 0.015 | 0.000 |
| lymph | PFoil:pw_both:accuracy_vote | 82.43% | 44.6 | 0.020 | 0.001 |
| lymph | PFoil:pw_both:vote | 83.81% | 44.6 | 0.020 | 0.001 |
| lymph | PFoil:pw_both:weighted_vote | 82.57% | 44.6 | 0.020 | 0.002 |
| lymph | PFoil:pw_larger:accuracy_vote | 81.76% | 23.3 | 0.011 | 0.000 |
| lymph | PFoil:pw_larger:vote | 83.14% | 23.3 | 0.011 | 0.000 |
| lymph | PFoil:pw_larger:weighted_vote | 54.76% | 23.3 | 0.011 | 0.001 |
| lymph | PFoil:pw_smaller:accuracy_vote | 79.14% | 21.3 | 0.010 | 0.000 |
| lymph | PFoil:pw_smaller:vote | 79.81% | 21.3 | 0.010 | 0.000 |
| lymph | PFoil:pw_smaller:weighted_vote | 78.52% | 21.3 | 0.010 | 0.001 |
| lymph | PFossil:ordered | 86.52% | 21.3 | 0.009 | 0.000 |
| lymph | PFossil:ovr | 81.71% | 45.0 | 0.016 | 0.000 |
| lymph | PFossil:pw_both:accuracy_vote | 84.52% | 48.3 | 0.021 | 0.001 |
| lymph | PFossil:pw_both:vote | 83.14% | 48.3 | 0.021 | 0.001 |
| lymph | PFossil:pw_both:weighted_vote | 80.48% | 48.3 | 0.021 | 0.003 |
| lymph | PFossil:pw_larger:accuracy_vote | 81.14% | 24.1 | 0.012 | 0.000 |
| lymph | PFossil:pw_larger:vote | 81.14% | 24.1 | 0.012 | 0.000 |
| lymph | PFossil:pw_larger:weighted_vote | 54.76% | 24.1 | 0.012 | 0.002 |
| lymph | PFossil:pw_smaller:accuracy_vote | 86.52% | 24.2 | 0.011 | 0.000 |
| lymph | PFossil:pw_smaller:vote | 86.52% | 24.2 | 0.011 | 0.000 |
| lymph | PFossil:pw_smaller:weighted_vote | 86.57% | 24.2 | 0.011 | 0.001 |
| lymph | Pypper:ordered | 77.05% | 6.8 | 0.026 | 0.000 |
| lymph | Pypper:ovr | 79.81% | 10.4 | 0.041 | 0.000 |
| lymph | Pypper:pw_both:accuracy_vote | 75.76% | 16.9 | 0.067 | 0.000 |
| lymph | Pypper:pw_both:vote | 78.48% | 16.9 | 0.067 | 0.000 |
| lymph | Pypper:pw_both:weighted_vote | 75.05% | 16.9 | 0.067 | 0.002 |
| lymph | Pypper:pw_larger:accuracy_vote | 73.76% | 9.1 | 0.039 | 0.000 |
| lymph | Pypper:pw_larger:vote | 76.48% | 9.1 | 0.039 | 0.000 |
| lymph | Pypper:pw_larger:weighted_vote | 54.76% | 9.1 | 0.039 | 0.001 |
| lymph | Pypper:pw_smaller:accuracy_vote | 77.05% | 7.8 | 0.030 | 0.000 |
| lymph | Pypper:pw_smaller:vote | 77.71% | 7.8 | 0.030 | 0.000 |
| lymph | Pypper:pw_smaller:weighted_vote | 77.10% | 7.8 | 0.030 | 0.001 |
| optdigits | PFoil:ordered | 88.26% | 420.3 | 0.482 | 0.001 |
| optdigits | PFoil:ovr | 93.35% | 692.5 | 0.841 | 0.011 |
| optdigits | PFoil:pw_both:accuracy_vote | 96.62% | 1076.2 | 1.077 | 0.102 |
| optdigits | PFoil:pw_both:vote | 96.55% | 1076.2 | 1.077 | 0.075 |
| optdigits | PFoil:pw_both:weighted_vote | 96.80% | 1076.2 | 1.077 | 0.818 |
| optdigits | PFoil:pw_larger:accuracy_vote | 94.57% | 537.7 | 0.601 | 0.051 |
| optdigits | PFoil:pw_larger:vote | 94.59% | 537.7 | 0.601 | 0.039 |
| optdigits | PFoil:pw_larger:weighted_vote | 32.03% | 537.7 | 0.601 | 0.415 |
| optdigits | PFoil:pw_smaller:accuracy_vote | 94.15% | 539.4 | 0.596 | 0.051 |
| optdigits | PFoil:pw_smaller:vote | 94.38% | 539.4 | 0.596 | 0.039 |
| optdigits | PFoil:pw_smaller:weighted_vote | 43.43% | 539.4 | 0.596 | 0.402 |
| optdigits | PFossil:ordered | 90.71% | 299.6 | 0.356 | 0.001 |
| optdigits | PFossil:ovr | 90.30% | 457.4 | 0.552 | 0.005 |
| optdigits | PFossil:pw_both:accuracy_vote | 95.64% | 1096.9 | 1.118 | 0.100 |
| optdigits | PFossil:pw_both:vote | 95.48% | 1096.9 | 1.118 | 0.079 |
| optdigits | PFossil:pw_both:weighted_vote | 95.82% | 1096.9 | 1.118 | 0.780 |
| optdigits | PFossil:pw_larger:accuracy_vote | 93.95% | 540.3 | 0.615 | 0.059 |
| optdigits | PFossil:pw_larger:vote | 93.84% | 540.3 | 0.615 | 0.041 |
| optdigits | PFossil:pw_larger:weighted_vote | 33.04% | 540.3 | 0.615 | 0.397 |
| optdigits | PFossil:pw_smaller:accuracy_vote | 93.90% | 558.7 | 0.624 | 0.053 |
| optdigits | PFossil:pw_smaller:vote | 94.04% | 558.7 | 0.624 | 0.039 |
| optdigits | PFossil:pw_smaller:weighted_vote | 40.18% | 558.7 | 0.624 | 0.416 |
| optdigits | Pypper:ordered | 91.09% | 256.4 | 1.228 | 0.001 |
| optdigits | Pypper:ovr | 91.39% | 391.3 | 2.075 | 0.006 |
| optdigits | Pypper:pw_both:accuracy_vote | 96.09% | 680.6 | 3.168 | 0.092 |
| optdigits | Pypper:pw_both:vote | 96.00% | 680.6 | 3.168 | 0.070 |
| optdigits | Pypper:pw_both:weighted_vote | 95.91% | 680.6 | 3.168 | 0.751 |
| optdigits | Pypper:pw_larger:accuracy_vote | 94.40% | 334.3 | 1.649 | 0.052 |
| optdigits | Pypper:pw_larger:vote | 94.29% | 334.3 | 1.649 | 0.035 |
| optdigits | Pypper:pw_larger:weighted_vote | 32.85% | 334.3 | 1.649 | 0.405 |
| optdigits | Pypper:pw_smaller:accuracy_vote | 94.75% | 347.0 | 1.655 | 0.047 |
| optdigits | Pypper:pw_smaller:vote | 94.80% | 347.0 | 1.655 | 0.035 |
| optdigits | Pypper:pw_smaller:weighted_vote | 43.27% | 347.0 | 1.655 | 0.395 |
| primary-tumor | PFoil:ordered | 9.45% | 223.8 | 0.078 | 0.001 |
| primary-tumor | PFoil:ovr | 43.35% | 329.0 | 0.111 | 0.002 |
| primary-tumor | PFoil:pw_both:accuracy_vote | 42.79% | 1259.7 | 0.555 | 0.031 |
| primary-tumor | PFoil:pw_both:vote | 44.25% | 1259.7 | 0.555 | 0.022 |
| primary-tumor | PFoil:pw_both:weighted_vote | 33.62% | 1259.7 | 0.555 | 0.186 |
| primary-tumor | PFoil:pw_larger:accuracy_vote | 41.32% | 637.6 | 0.292 | 0.015 |
| primary-tumor | PFoil:pw_larger:vote | 35.12% | 637.6 | 0.292 | 0.012 |
| primary-tumor | PFoil:pw_larger:weighted_vote | 24.78% | 637.6 | 0.292 | 0.130 |
| primary-tumor | PFoil:pw_smaller:accuracy_vote | 42.17% | 620.8 | 0.268 | 0.014 |
| primary-tumor | PFoil:pw_smaller:vote | 36.28% | 620.8 | 0.268 | 0.011 |
| primary-tumor | PFoil:pw_smaller:weighted_vote | 41.86% | 620.8 | 0.268 | 0.102 |
| primary-tumor | PFossil:ordered | 32.44% | 330.7 | 0.098 | 0.001 |
| primary-tumor | PFossil:ovr | 43.38% | 341.1 | 0.105 | 0.001 |
| primary-tumor | PFossil:pw_both:accuracy_vote | 46.02% | 1464.9 | 0.565 | 0.035 |
| primary-tumor | PFossil:pw_both:vote | 37.16% | 1464.9 | 0.565 | 0.027 |
| primary-tumor | PFossil:pw_both:weighted_vote | 35.08% | 1464.9 | 0.565 | 0.228 |
| primary-tumor | PFossil:pw_larger:accuracy_vote | 39.84% | 652.7 | 0.282 | 0.017 |
| primary-tumor | PFossil:pw_larger:vote | 28.61% | 652.7 | 0.282 | 0.012 |
| primary-tumor | PFossil:pw_larger:weighted_vote | 24.78% | 652.7 | 0.282 | 0.112 |
| primary-tumor | PFossil:pw_smaller:accuracy_vote | 39.22% | 813.1 | 0.299 | 0.030 |
| primary-tumor | PFossil:pw_smaller:vote | 38.62% | 813.1 | 0.299 | 0.012 |
| primary-tumor | PFossil:pw_smaller:weighted_vote | 41.59% | 813.1 | 0.299 | 0.086 |
| primary-tumor | Pypper:ordered | 38.66% | 24.8 | 0.110 | 0.000 |
| primary-tumor | Pypper:ovr | 37.47% | 27.6 | 0.122 | 0.000 |
| primary-tumor | Pypper:pw_both:accuracy_vote | 48.08% | 263.5 | 1.692 | 0.044 |
| primary-tumor | Pypper:pw_both:vote | 45.14% | 263.5 | 1.692 | 0.021 |
| primary-tumor | Pypper:pw_both:weighted_vote | 27.73% | 263.5 | 1.692 | 0.140 |
| primary-tumor | Pypper:pw_larger:accuracy_vote | 44.85% | 138.3 | 0.999 | 0.013 |
| primary-tumor | Pypper:pw_larger:vote | 0.00% | 138.3 | 0.999 | 0.009 |
| primary-tumor | Pypper:pw_larger:weighted_vote | 24.78% | 138.3 | 0.999 | 0.070 |
| primary-tumor | Pypper:pw_smaller:accuracy_vote | 44.85% | 125.8 | 0.705 | 0.012 |
| primary-tumor | Pypper:pw_smaller:vote | 41.88% | 125.8 | 0.705 | 0.008 |
| primary-tumor | Pypper:pw_smaller:weighted_vote | 33.35% | 125.8 | 0.705 | 0.065 |
| segment | PFoil:ordered | 83.81% | 133.7 | 0.082 | 0.000 |
| segment | PFoil:ovr | 95.19% | 291.3 | 0.173 | 0.003 |
| segment | PFoil:pw_both:accuracy_vote | 93.90% | 293.9 | 0.182 | 0.020 |
| segment | PFoil:pw_both:vote | 93.20% | 293.9 | 0.182 | 0.015 |
| segment | PFoil:pw_both:weighted_vote | 96.32% | 293.9 | 0.182 | 0.126 |
| segment | PFoil:pw_larger:accuracy_vote | 90.39% | 133.6 | 0.089 | 0.010 |
| segment | PFoil:pw_larger:vote | 89.26% | 133.6 | 0.089 | 0.008 |
| segment | PFoil:pw_larger:weighted_vote | 44.16% | 133.6 | 0.089 | 0.065 |
| segment | PFoil:pw_smaller:accuracy_vote | 90.39% | 133.6 | 0.090 | 0.010 |
| segment | PFoil:pw_smaller:vote | 89.26% | 133.6 | 0.090 | 0.008 |
| segment | PFoil:pw_smaller:weighted_vote | 44.16% | 133.6 | 0.090 | 0.083 |
| segment | PFossil:ordered | 94.85% | 75.5 | 0.054 | 0.000 |
| segment | PFossil:ovr | 93.33% | 121.8 | 0.085 | 0.001 |
| segment | PFossil:pw_both:accuracy_vote | 95.15% | 180.8 | 0.136 | 0.018 |
| segment | PFossil:pw_both:vote | 95.32% | 180.8 | 0.136 | 0.015 |
| segment | PFossil:pw_both:weighted_vote | 95.54% | 180.8 | 0.136 | 0.129 |
| segment | PFossil:pw_larger:accuracy_vote | 94.81% | 103.4 | 0.076 | 0.010 |
| segment | PFossil:pw_larger:vote | 94.46% | 103.4 | 0.076 | 0.007 |
| segment | PFossil:pw_larger:weighted_vote | 47.06% | 103.4 | 0.076 | 0.061 |
| segment | PFossil:pw_smaller:accuracy_vote | 94.81% | 103.4 | 0.076 | 0.011 |
| segment | PFossil:pw_smaller:vote | 94.46% | 103.4 | 0.076 | 0.008 |
| segment | PFossil:pw_smaller:weighted_vote | 47.06% | 103.4 | 0.076 | 0.062 |
| segment | Pypper:ordered | 94.81% | 51.3 | 0.162 | 0.000 |
| segment | Pypper:ovr | 94.03% | 90.3 | 0.304 | 0.001 |
| segment | Pypper:pw_both:accuracy_vote | 95.54% | 135.9 | 0.491 | 0.018 |
| segment | Pypper:pw_both:vote | 95.32% | 135.9 | 0.491 | 0.013 |
| segment | Pypper:pw_both:weighted_vote | 95.28% | 135.9 | 0.491 | 0.116 |
| segment | Pypper:pw_larger:accuracy_vote | 95.19% | 68.0 | 0.240 | 0.010 |
| segment | Pypper:pw_larger:vote | 95.24% | 68.0 | 0.240 | 0.007 |
| segment | Pypper:pw_larger:weighted_vote | 51.86% | 68.0 | 0.240 | 0.057 |
| segment | Pypper:pw_smaller:accuracy_vote | 95.19% | 68.0 | 0.240 | 0.009 |
| segment | Pypper:pw_smaller:vote | 95.24% | 68.0 | 0.240 | 0.007 |
| segment | Pypper:pw_smaller:weighted_vote | 51.86% | 68.0 | 0.240 | 0.059 |
| solar-flare | PFoil:ordered | 58.72% | 241.1 | 0.093 | 0.000 |
| solar-flare | PFoil:ovr | 73.73% | 403.4 | 0.146 | 0.003 |
| solar-flare | PFoil:pw_both:accuracy_vote | 72.51% | 694.9 | 0.243 | 0.009 |
| solar-flare | PFoil:pw_both:vote | 68.01% | 694.9 | 0.243 | 0.008 |
| solar-flare | PFoil:pw_both:weighted_vote | 73.45% | 694.9 | 0.243 | 0.062 |
| solar-flare | PFoil:pw_larger:accuracy_vote | 68.11% | 411.2 | 0.142 | 0.005 |
| solar-flare | PFoil:pw_larger:vote | 67.64% | 411.2 | 0.142 | 0.004 |
| solar-flare | PFoil:pw_larger:weighted_vote | 56.66% | 411.2 | 0.142 | 0.036 |
| solar-flare | PFoil:pw_smaller:accuracy_vote | 59.19% | 283.7 | 0.104 | 0.004 |
| solar-flare | PFoil:pw_smaller:vote | 60.60% | 283.7 | 0.104 | 0.003 |
| solar-flare | PFoil:pw_smaller:weighted_vote | 62.11% | 283.7 | 0.104 | 0.024 |
| solar-flare | PFossil:ordered | 70.64% | 50.0 | 0.027 | 0.000 |
| solar-flare | PFossil:ovr | 74.02% | 62.5 | 0.034 | 0.001 |
| solar-flare | PFossil:pw_both:accuracy_vote | 73.83% | 161.9 | 0.081 | 0.007 |
| solar-flare | PFossil:pw_both:vote | 74.30% | 161.9 | 0.081 | 0.005 |
| solar-flare | PFossil:pw_both:weighted_vote | 74.77% | 161.9 | 0.081 | 0.039 |
| solar-flare | PFossil:pw_larger:accuracy_vote | 74.77% | 73.0 | 0.038 | 0.004 |
| solar-flare | PFossil:pw_larger:vote | 73.93% | 73.0 | 0.038 | 0.003 |
| solar-flare | PFossil:pw_larger:weighted_vote | 58.82% | 73.0 | 0.038 | 0.023 |
| solar-flare | PFossil:pw_smaller:accuracy_vote | 72.23% | 88.9 | 0.046 | 0.003 |
| solar-flare | PFossil:pw_smaller:vote | 72.33% | 88.9 | 0.046 | 0.003 |
| solar-flare | PFossil:pw_smaller:weighted_vote | 71.39% | 88.9 | 0.046 | 0.020 |
| solar-flare | Pypper:ordered | 74.39% | 10.7 | 0.070 | 0.000 |
| solar-flare | Pypper:ovr | 69.52% | 20.5 | 0.112 | 0.000 |
| solar-flare | Pypper:pw_both:accuracy_vote | 74.48% | 55.6 | 0.328 | 0.006 |
| solar-flare | Pypper:pw_both:vote | 74.77% | 55.6 | 0.328 | 0.004 |
| solar-flare | Pypper:pw_both:weighted_vote | 74.49% | 55.6 | 0.328 | 0.038 |
| solar-flare | Pypper:pw_larger:accuracy_vote | 73.35% | 26.9 | 0.185 | 0.003 |
| solar-flare | Pypper:pw_larger:vote | 72.80% | 26.9 | 0.185 | 0.002 |
| solar-flare | Pypper:pw_larger:weighted_vote | 55.52% | 26.9 | 0.185 | 0.023 |
| solar-flare | Pypper:pw_smaller:accuracy_vote | 74.68% | 28.7 | 0.148 | 0.003 |
| solar-flare | Pypper:pw_smaller:vote | 74.77% | 28.7 | 0.148 | 0.002 |
| solar-flare | Pypper:pw_smaller:weighted_vote | 71.02% | 28.7 | 0.148 | 0.018 |
| soybean | PFoil:ordered | 85.65% | 83.5 | 0.046 | 0.000 |
| soybean | PFoil:ovr | 90.92% | 153.8 | 0.074 | 0.001 |
| soybean | PFoil:pw_both:accuracy_vote | 92.97% | 446.2 | 0.281 | 0.040 |
| soybean | PFoil:pw_both:vote | 93.41% | 446.2 | 0.281 | 0.028 |
| soybean | PFoil:pw_both:weighted_vote | 62.22% | 446.2 | 0.281 | 0.257 |
| soybean | PFoil:pw_larger:accuracy_vote | 92.68% | 223.6 | 0.156 | 0.020 |
| soybean | PFoil:pw_larger:vote | 92.83% | 223.6 | 0.156 | 0.015 |
| soybean | PFoil:pw_larger:weighted_vote | 40.12% | 223.6 | 0.156 | 0.145 |
| soybean | PFoil:pw_smaller:accuracy_vote | 91.80% | 230.8 | 0.150 | 0.019 |
| soybean | PFoil:pw_smaller:vote | 90.92% | 230.8 | 0.150 | 0.015 |
| soybean | PFoil:pw_smaller:weighted_vote | 74.81% | 230.8 | 0.150 | 0.137 |
| soybean | PFossil:ordered | 92.98% | 70.2 | 0.042 | 0.000 |
| soybean | PFossil:ovr | 92.24% | 135.4 | 0.066 | 0.001 |
| soybean | PFossil:pw_both:accuracy_vote | 92.39% | 418.9 | 0.287 | 0.041 |
| soybean | PFossil:pw_both:vote | 93.12% | 418.9 | 0.287 | 0.028 |
| soybean | PFossil:pw_both:weighted_vote | 61.65% | 418.9 | 0.287 | 0.289 |
| soybean | PFossil:pw_larger:accuracy_vote | 93.26% | 207.0 | 0.147 | 0.020 |
| soybean | PFossil:pw_larger:vote | 93.70% | 207.0 | 0.147 | 0.015 |
| soybean | PFossil:pw_larger:weighted_vote | 37.02% | 207.0 | 0.147 | 0.134 |
| soybean | PFossil:pw_smaller:accuracy_vote | 93.41% | 201.5 | 0.144 | 0.034 |
| soybean | PFossil:pw_smaller:vote | 93.27% | 201.5 | 0.144 | 0.015 |
| soybean | PFossil:pw_smaller:weighted_vote | 75.25% | 201.5 | 0.144 | 0.119 |
| soybean | Pypper:ordered | 91.07% | 49.2 | 0.154 | 0.000 |
| soybean | Pypper:ovr | 90.50% | 65.1 | 0.201 | 0.000 |
| soybean | Pypper:pw_both:accuracy_vote | 93.12% | 376.1 | 1.458 | 0.042 |
| soybean | Pypper:pw_both:vote | 92.83% | 376.1 | 1.458 | 0.032 |
| soybean | Pypper:pw_both:weighted_vote | 61.64% | 376.1 | 1.458 | 0.257 |
| soybean | Pypper:pw_larger:accuracy_vote | 92.83% | 184.8 | 0.738 | 0.021 |
| soybean | Pypper:pw_larger:vote | 90.79% | 184.8 | 0.738 | 0.015 |
| soybean | Pypper:pw_larger:weighted_vote | 40.85% | 184.8 | 0.738 | 0.130 |
| soybean | Pypper:pw_smaller:accuracy_vote | 92.24% | 190.5 | 0.737 | 0.030 |
| soybean | Pypper:pw_smaller:vote | 91.95% | 190.5 | 0.737 | 0.014 |
| soybean | Pypper:pw_smaller:weighted_vote | 74.38% | 190.5 | 0.737 | 0.125 |
| splice | PFoil:ordered | 87.96% | 192.8 | 0.142 | 0.000 |
| splice | PFoil:ovr | 92.76% | 379.4 | 0.287 | 0.004 |
| splice | PFoil:pw_both:accuracy_vote | 92.73% | 373.9 | 0.292 | 0.008 |
| splice | PFoil:pw_both:vote | 90.60% | 373.9 | 0.292 | 0.006 |
| splice | PFoil:pw_both:weighted_vote | 92.88% | 373.9 | 0.292 | 0.044 |
| splice | PFoil:pw_larger:accuracy_vote | 89.09% | 167.8 | 0.151 | 0.004 |
| splice | PFoil:pw_larger:vote | 88.40% | 167.8 | 0.151 | 0.004 |
| splice | PFoil:pw_larger:weighted_vote | 51.88% | 167.8 | 0.151 | 0.028 |
| splice | PFoil:pw_smaller:accuracy_vote | 90.44% | 209.0 | 0.156 | 0.004 |
| splice | PFoil:pw_smaller:vote | 90.34% | 209.0 | 0.156 | 0.004 |
| splice | PFoil:pw_smaller:weighted_vote | 90.94% | 209.0 | 0.156 | 0.019 |
| splice | PFossil:ordered | 94.33% | 40.5 | 0.039 | 0.000 |
| splice | PFossil:ovr | 94.23% | 106.8 | 0.093 | 0.002 |
| splice | PFossil:pw_both:accuracy_vote | 94.70% | 153.4 | 0.143 | 0.006 |
| splice | PFossil:pw_both:vote | 94.26% | 153.4 | 0.143 | 0.006 |
| splice | PFossil:pw_both:weighted_vote | 94.29% | 153.4 | 0.143 | 0.039 |
| splice | PFossil:pw_larger:accuracy_vote | 94.33% | 94.4 | 0.096 | 0.004 |
| splice | PFossil:pw_larger:vote | 93.98% | 94.4 | 0.096 | 0.004 |
| splice | PFossil:pw_larger:weighted_vote | 51.88% | 94.4 | 0.096 | 0.024 |
| splice | PFossil:pw_smaller:accuracy_vote | 92.01% | 55.1 | 0.054 | 0.003 |
| splice | PFossil:pw_smaller:vote | 92.32% | 55.1 | 0.054 | 0.003 |
| splice | PFossil:pw_smaller:weighted_vote | 91.94% | 55.1 | 0.054 | 0.015 |
| splice | Pypper:ordered | 93.10% | 51.7 | 0.232 | 0.000 |
| splice | Pypper:ovr | 93.20% | 111.8 | 0.501 | 0.001 |
| splice | Pypper:pw_both:accuracy_vote | 94.26% | 116.3 | 0.493 | 0.006 |
| splice | Pypper:pw_both:vote | 94.14% | 116.3 | 0.493 | 0.005 |
| splice | Pypper:pw_both:weighted_vote | 93.98% | 116.3 | 0.493 | 0.053 |
| splice | Pypper:pw_larger:accuracy_vote | 94.23% | 44.4 | 0.220 | 0.003 |
| splice | Pypper:pw_larger:vote | 94.29% | 44.4 | 0.220 | 0.003 |
| splice | Pypper:pw_larger:weighted_vote | 51.88% | 44.4 | 0.220 | 0.021 |
| splice | Pypper:pw_smaller:accuracy_vote | 93.76% | 74.3 | 0.298 | 0.003 |
| splice | Pypper:pw_smaller:vote | 93.67% | 74.3 | 0.298 | 0.003 |
| splice | Pypper:pw_smaller:weighted_vote | 92.73% | 74.3 | 0.298 | 0.016 |
| texture | PFoil:ordered | 84.27% | 469.9 | 0.444 | 0.001 |
| texture | PFoil:ovr | 94.51% | 777.3 | 0.763 | 0.011 |
| texture | PFoil:pw_both:accuracy_vote | 94.95% | 903.5 | 0.782 | 0.115 |
| texture | PFoil:pw_both:vote | 94.87% | 903.5 | 0.782 | 0.083 |
| texture | PFoil:pw_both:weighted_vote | 96.04% | 903.5 | 0.782 | 0.842 |
| texture | PFoil:pw_larger:accuracy_vote | 92.85% | 460.8 | 0.431 | 0.057 |
| texture | PFoil:pw_larger:vote | 91.95% | 460.8 | 0.431 | 0.043 |
| texture | PFoil:pw_larger:weighted_vote | 31.51% | 460.8 | 0.431 | 0.436 |
| texture | PFoil:pw_smaller:accuracy_vote | 92.85% | 460.8 | 0.431 | 0.057 |
| texture | PFoil:pw_smaller:vote | 91.95% | 460.8 | 0.431 | 0.042 |
| texture | PFoil:pw_smaller:weighted_vote | 31.51% | 460.8 | 0.431 | 0.420 |
| texture | PFossil:ordered | 90.45% | 219.7 | 0.242 | 0.001 |
| texture | PFossil:ovr | 92.22% | 385.6 | 0.408 | 0.005 |
| texture | PFossil:pw_both:accuracy_vote | 94.05% | 674.9 | 0.660 | 0.108 |
| texture | PFossil:pw_both:vote | 93.58% | 674.9 | 0.660 | 0.077 |
| texture | PFossil:pw_both:weighted_vote | 94.58% | 674.9 | 0.660 | 0.779 |
| texture | PFossil:pw_larger:accuracy_vote | 92.35% | 315.6 | 0.355 | 0.058 |
| texture | PFossil:pw_larger:vote | 91.71% | 315.6 | 0.355 | 0.041 |
| texture | PFossil:pw_larger:weighted_vote | 30.64% | 315.6 | 0.355 | 0.437 |
| texture | PFossil:pw_smaller:accuracy_vote | 92.35% | 315.6 | 0.354 | 0.055 |
| texture | PFossil:pw_smaller:vote | 91.71% | 315.6 | 0.354 | 0.040 |
| texture | PFossil:pw_smaller:weighted_vote | 30.64% | 315.6 | 0.354 | 0.398 |
| texture | Pypper:ordered | 92.33% | 237.1 | 0.978 | 0.001 |
| texture | Pypper:ovr | 92.40% | 378.4 | 1.652 | 0.005 |
| texture | Pypper:pw_both:accuracy_vote | 95.55% | 531.0 | 2.167 | 0.106 |
| texture | Pypper:pw_both:vote | 95.44% | 531.0 | 2.167 | 0.076 |
| texture | Pypper:pw_both:weighted_vote | 95.60% | 531.0 | 2.167 | 0.768 |
| texture | Pypper:pw_larger:accuracy_vote | 94.76% | 271.9 | 1.146 | 0.055 |
| texture | Pypper:pw_larger:vote | 94.69% | 271.9 | 1.146 | 0.039 |
| texture | Pypper:pw_larger:weighted_vote | 32.75% | 271.9 | 1.146 | 0.419 |
| texture | Pypper:pw_smaller:accuracy_vote | 94.76% | 271.9 | 1.151 | 0.054 |
| texture | Pypper:pw_smaller:vote | 94.69% | 271.9 | 1.151 | 0.039 |
| texture | Pypper:pw_smaller:weighted_vote | 32.75% | 271.9 | 1.151 | 0.378 |
| vehicle | PFoil:ordered | 61.96% | 204.4 | 0.082 | 0.000 |
| vehicle | PFoil:ovr | 71.87% | 382.2 | 0.156 | 0.002 |
| vehicle | PFoil:pw_both:accuracy_vote | 72.35% | 425.5 | 0.176 | 0.003 |
| vehicle | PFoil:pw_both:vote | 68.08% | 425.5 | 0.176 | 0.004 |
| vehicle | PFoil:pw_both:weighted_vote | 73.89% | 425.5 | 0.176 | 0.020 |
| vehicle | PFoil:pw_larger:accuracy_vote | 63.94% | 210.1 | 0.087 | 0.002 |
| vehicle | PFoil:pw_larger:vote | 62.63% | 210.1 | 0.087 | 0.002 |
| vehicle | PFoil:pw_larger:weighted_vote | 48.93% | 210.1 | 0.087 | 0.010 |
| vehicle | PFoil:pw_smaller:accuracy_vote | 67.03% | 216.6 | 0.091 | 0.002 |
| vehicle | PFoil:pw_smaller:vote | 67.85% | 216.6 | 0.091 | 0.002 |
| vehicle | PFoil:pw_smaller:weighted_vote | 54.63% | 216.6 | 0.091 | 0.010 |
| vehicle | PFossil:ordered | 70.22% | 81.7 | 0.037 | 0.000 |
| vehicle | PFossil:ovr | 71.64% | 124.9 | 0.055 | 0.001 |
| vehicle | PFossil:pw_both:accuracy_vote | 72.36% | 157.4 | 0.079 | 0.004 |
| vehicle | PFossil:pw_both:vote | 67.86% | 157.4 | 0.079 | 0.002 |
| vehicle | PFossil:pw_both:weighted_vote | 70.46% | 157.4 | 0.079 | 0.029 |
| vehicle | PFossil:pw_larger:accuracy_vote | 69.40% | 93.3 | 0.046 | 0.002 |
| vehicle | PFossil:pw_larger:vote | 67.51% | 93.3 | 0.046 | 0.001 |
| vehicle | PFossil:pw_larger:weighted_vote | 45.99% | 93.3 | 0.046 | 0.008 |
| vehicle | PFossil:pw_smaller:accuracy_vote | 68.43% | 63.3 | 0.034 | 0.001 |
| vehicle | PFossil:pw_smaller:vote | 66.31% | 63.3 | 0.034 | 0.001 |
| vehicle | PFossil:pw_smaller:weighted_vote | 55.68% | 63.3 | 0.034 | 0.007 |
| vehicle | Pypper:ordered | 67.74% | 39.3 | 0.126 | 0.000 |
| vehicle | Pypper:ovr | 63.95% | 57.7 | 0.204 | 0.000 |
| vehicle | Pypper:pw_both:accuracy_vote | 71.76% | 97.1 | 0.303 | 0.003 |
| vehicle | Pypper:pw_both:vote | 69.98% | 97.1 | 0.303 | 0.002 |
| vehicle | Pypper:pw_both:weighted_vote | 70.45% | 97.1 | 0.303 | 0.015 |
| vehicle | Pypper:pw_larger:accuracy_vote | 69.98% | 46.4 | 0.155 | 0.001 |
| vehicle | Pypper:pw_larger:vote | 70.69% | 46.4 | 0.155 | 0.001 |
| vehicle | Pypper:pw_larger:weighted_vote | 45.99% | 46.4 | 0.155 | 0.007 |
| vehicle | Pypper:pw_smaller:accuracy_vote | 70.34% | 51.5 | 0.157 | 0.001 |
| vehicle | Pypper:pw_smaller:vote | 69.98% | 51.5 | 0.157 | 0.001 |
| vehicle | Pypper:pw_smaller:weighted_vote | 58.65% | 51.5 | 0.157 | 0.007 |
| vowel | PFoil:ordered | 68.99% | 278.7 | 0.113 | 0.001 |
| vowel | PFoil:ovr | 81.82% | 465.9 | 0.191 | 0.002 |
| vowel | PFoil:pw_both:accuracy_vote | 87.27% | 790.4 | 0.346 | 0.023 |
| vowel | PFoil:pw_both:vote | 86.57% | 790.4 | 0.346 | 0.017 |
| vowel | PFoil:pw_both:weighted_vote | 87.47% | 790.4 | 0.346 | 0.180 |
| vowel | PFoil:pw_larger:accuracy_vote | 81.01% | 390.2 | 0.174 | 0.011 |
| vowel | PFoil:pw_larger:vote | 81.41% | 390.2 | 0.174 | 0.009 |
| vowel | PFoil:pw_larger:weighted_vote | 38.89% | 390.2 | 0.174 | 0.076 |
| vowel | PFoil:pw_smaller:accuracy_vote | 81.01% | 390.2 | 0.177 | 0.011 |
| vowel | PFoil:pw_smaller:vote | 81.41% | 390.2 | 0.177 | 0.009 |
| vowel | PFoil:pw_smaller:weighted_vote | 38.89% | 390.2 | 0.177 | 0.085 |
| vowel | PFossil:ordered | 80.40% | 247.7 | 0.104 | 0.001 |
| vowel | PFossil:ovr | 77.27% | 401.3 | 0.170 | 0.002 |
| vowel | PFossil:pw_both:accuracy_vote | 83.54% | 744.6 | 0.342 | 0.023 |
| vowel | PFossil:pw_both:vote | 81.82% | 744.6 | 0.342 | 0.017 |
| vowel | PFossil:pw_both:weighted_vote | 82.22% | 744.6 | 0.342 | 0.164 |
| vowel | PFossil:pw_larger:accuracy_vote | 82.73% | 372.1 | 0.172 | 0.012 |
| vowel | PFossil:pw_larger:vote | 81.31% | 372.1 | 0.172 | 0.009 |
| vowel | PFossil:pw_larger:weighted_vote | 35.96% | 372.1 | 0.172 | 0.089 |
| vowel | PFossil:pw_smaller:accuracy_vote | 82.73% | 372.1 | 0.172 | 0.012 |
| vowel | PFossil:pw_smaller:vote | 81.31% | 372.1 | 0.172 | 0.008 |
| vowel | PFossil:pw_smaller:weighted_vote | 35.96% | 372.1 | 0.172 | 0.074 |
| vowel | Pypper:ordered | 74.85% | 173.6 | 0.380 | 0.000 |
| vowel | Pypper:ovr | 73.84% | 272.2 | 0.588 | 0.001 |
| vowel | Pypper:pw_both:accuracy_vote | 82.32% | 539.9 | 1.326 | 0.021 |
| vowel | Pypper:pw_both:vote | 80.91% | 539.9 | 1.326 | 0.017 |
| vowel | Pypper:pw_both:weighted_vote | 81.21% | 539.9 | 1.326 | 0.156 |
| vowel | Pypper:pw_larger:accuracy_vote | 80.40% | 272.5 | 0.652 | 0.011 |
| vowel | Pypper:pw_larger:vote | 80.20% | 272.5 | 0.652 | 0.009 |
| vowel | Pypper:pw_larger:weighted_vote | 37.07% | 272.5 | 0.652 | 0.073 |
| vowel | Pypper:pw_smaller:accuracy_vote | 80.40% | 272.5 | 0.664 | 0.012 |
| vowel | Pypper:pw_smaller:vote | 80.20% | 272.5 | 0.664 | 0.008 |
| vowel | Pypper:pw_smaller:weighted_vote | 37.07% | 272.5 | 0.664 | 0.074 |
| wine | PFoil:ordered | 94.97% | 7.9 | 0.004 | 0.000 |
| wine | PFoil:ovr | 90.98% | 16.6 | 0.007 | 0.000 |
| wine | PFoil:pw_both:accuracy_vote | 96.08% | 18.7 | 0.010 | 0.000 |
| wine | PFoil:pw_both:vote | 97.19% | 18.7 | 0.010 | 0.000 |
| wine | PFoil:pw_both:weighted_vote | 97.19% | 18.7 | 0.010 | 0.002 |
| wine | PFoil:pw_larger:accuracy_vote | 96.63% | 9.7 | 0.006 | 0.000 |
| wine | PFoil:pw_larger:vote | 96.63% | 9.7 | 0.006 | 0.000 |
| wine | PFoil:pw_larger:weighted_vote | 44.93% | 9.7 | 0.006 | 0.001 |
| wine | PFoil:pw_smaller:accuracy_vote | 93.27% | 9.0 | 0.005 | 0.000 |
| wine | PFoil:pw_smaller:vote | 94.38% | 9.0 | 0.005 | 0.000 |
| wine | PFoil:pw_smaller:weighted_vote | 87.65% | 9.0 | 0.005 | 0.001 |
| wine | PFossil:ordered | 94.97% | 8.6 | 0.004 | 0.000 |
| wine | PFossil:ovr | 92.75% | 18.3 | 0.008 | 0.000 |
| wine | PFossil:pw_both:accuracy_vote | 94.41% | 16.6 | 0.009 | 0.000 |
| wine | PFossil:pw_both:vote | 93.30% | 16.6 | 0.009 | 0.000 |
| wine | PFossil:pw_both:weighted_vote | 93.86% | 16.6 | 0.009 | 0.002 |
| wine | PFossil:pw_larger:accuracy_vote | 92.75% | 7.0 | 0.005 | 0.000 |
| wine | PFossil:pw_larger:vote | 93.30% | 7.0 | 0.005 | 0.000 |
| wine | PFossil:pw_larger:weighted_vote | 42.16% | 7.0 | 0.005 | 0.001 |
| wine | PFossil:pw_smaller:accuracy_vote | 95.52% | 9.6 | 0.005 | 0.000 |
| wine | PFossil:pw_smaller:vote | 95.52% | 9.6 | 0.005 | 0.000 |
| wine | PFossil:pw_smaller:weighted_vote | 86.54% | 9.6 | 0.005 | 0.001 |
| wine | Pypper:ordered | 92.09% | 4.3 | 0.017 | 0.000 |
| wine | Pypper:ovr | 92.09% | 9.1 | 0.030 | 0.000 |
| wine | Pypper:pw_both:accuracy_vote | 91.67% | 11.2 | 0.041 | 0.000 |
| wine | Pypper:pw_both:vote | 92.12% | 11.2 | 0.041 | 0.000 |
| wine | Pypper:pw_both:weighted_vote | 92.12% | 11.2 | 0.041 | 0.002 |
| wine | Pypper:pw_larger:accuracy_vote | 92.75% | 5.5 | 0.021 | 0.000 |
| wine | Pypper:pw_larger:vote | 92.19% | 5.5 | 0.021 | 0.000 |
| wine | Pypper:pw_larger:weighted_vote | 46.63% | 5.5 | 0.021 | 0.001 |
| wine | Pypper:pw_smaller:accuracy_vote | 91.60% | 5.7 | 0.022 | 0.000 |
| wine | Pypper:pw_smaller:vote | 92.71% | 5.7 | 0.022 | 0.000 |
| wine | Pypper:pw_smaller:weighted_vote | 83.14% | 5.7 | 0.022 | 0.001 |
| yeast | PFoil:ordered | 23.79% | 744.7 | 0.287 | 0.001 |
| yeast | PFoil:ovr | 57.41% | 1245.0 | 0.467 | 0.010 |
| yeast | PFoil:pw_both:accuracy_vote | 54.85% | 2548.4 | 0.932 | 0.037 |
| yeast | PFoil:pw_both:vote | 52.63% | 2548.4 | 0.932 | 0.027 |
| yeast | PFoil:pw_both:weighted_vote | 51.82% | 2548.4 | 0.932 | 0.275 |
| yeast | PFoil:pw_larger:accuracy_vote | 48.05% | 1421.4 | 0.519 | 0.018 |
| yeast | PFoil:pw_larger:vote | 47.44% | 1421.4 | 0.519 | 0.016 |
| yeast | PFoil:pw_larger:weighted_vote | 36.25% | 1421.4 | 0.519 | 0.162 |
| yeast | PFoil:pw_smaller:accuracy_vote | 43.40% | 1127.0 | 0.423 | 0.016 |
| yeast | PFoil:pw_smaller:vote | 38.14% | 1127.0 | 0.423 | 0.015 |
| yeast | PFoil:pw_smaller:weighted_vote | 56.81% | 1127.0 | 0.423 | 0.096 |
| yeast | PFossil:ordered | 56.06% | 148.8 | 0.077 | 0.000 |
| yeast | PFossil:ovr | 57.48% | 180.5 | 0.091 | 0.001 |
| yeast | PFossil:pw_both:accuracy_vote | 59.57% | 898.5 | 0.369 | 0.026 |
| yeast | PFossil:pw_both:vote | 59.77% | 898.5 | 0.369 | 0.020 |
| yeast | PFossil:pw_both:weighted_vote | 51.76% | 898.5 | 0.369 | 0.211 |
| yeast | PFossil:pw_larger:accuracy_vote | 57.28% | 337.3 | 0.160 | 0.015 |
| yeast | PFossil:pw_larger:vote | 56.54% | 337.3 | 0.160 | 0.011 |
| yeast | PFossil:pw_larger:weighted_vote | 35.37% | 337.3 | 0.160 | 0.128 |
| yeast | PFossil:pw_smaller:accuracy_vote | 57.34% | 561.2 | 0.218 | 0.013 |
| yeast | PFossil:pw_smaller:vote | 57.48% | 561.2 | 0.218 | 0.011 |
| yeast | PFossil:pw_smaller:weighted_vote | 58.49% | 561.2 | 0.218 | 0.090 |
| yeast | Pypper:ordered | 59.57% | 42.0 | 0.230 | 0.000 |
| yeast | Pypper:ovr | 59.43% | 69.7 | 0.320 | 0.001 |
| yeast | Pypper:pw_both:accuracy_vote | 59.43% | 241.1 | 1.161 | 0.024 |
| yeast | Pypper:pw_both:vote | 58.90% | 241.1 | 1.161 | 0.018 |
| yeast | Pypper:pw_both:weighted_vote | 50.00% | 241.1 | 1.161 | 0.173 |
| yeast | Pypper:pw_larger:accuracy_vote | 57.68% | 113.8 | 0.613 | 0.012 |
| yeast | Pypper:pw_larger:vote | 58.49% | 113.8 | 0.613 | 0.009 |
| yeast | Pypper:pw_larger:weighted_vote | 35.18% | 113.8 | 0.613 | 0.091 |
| yeast | Pypper:pw_smaller:accuracy_vote | 59.30% | 127.3 | 0.554 | 0.012 |
| yeast | Pypper:pw_smaller:vote | 58.90% | 127.3 | 0.554 | 0.008 |
| yeast | Pypper:pw_smaller:weighted_vote | 59.51% | 127.3 | 0.554 | 0.072 |
| zoo | PFoil:ordered | 92.00% | 15.8 | 0.007 | 0.000 |
| zoo | PFoil:ovr | 96.00% | 20.3 | 0.008 | 0.000 |
| zoo | PFoil:pw_both:accuracy_vote | 93.00% | 43.7 | 0.026 | 0.001 |
| zoo | PFoil:pw_both:vote | 93.00% | 43.7 | 0.026 | 0.001 |
| zoo | PFoil:pw_both:weighted_vote | 88.00% | 43.7 | 0.026 | 0.006 |
| zoo | PFoil:pw_larger:accuracy_vote | 94.00% | 21.9 | 0.014 | 0.001 |
| zoo | PFoil:pw_larger:vote | 95.00% | 21.9 | 0.014 | 0.001 |
| zoo | PFoil:pw_larger:weighted_vote | 60.36% | 21.9 | 0.014 | 0.003 |
| zoo | PFoil:pw_smaller:accuracy_vote | 91.09% | 21.8 | 0.014 | 0.001 |
| zoo | PFoil:pw_smaller:vote | 91.09% | 21.8 | 0.014 | 0.001 |
| zoo | PFoil:pw_smaller:weighted_vote | 91.09% | 21.8 | 0.014 | 0.003 |
| zoo | PFossil:ordered | 93.00% | 16.7 | 0.007 | 0.000 |
| zoo | PFossil:ovr | 95.00% | 21.0 | 0.008 | 0.000 |
| zoo | PFossil:pw_both:accuracy_vote | 94.00% | 45.6 | 0.027 | 0.001 |
| zoo | PFossil:pw_both:vote | 93.00% | 45.6 | 0.027 | 0.001 |
| zoo | PFossil:pw_both:weighted_vote | 90.00% | 45.6 | 0.027 | 0.007 |
| zoo | PFossil:pw_larger:accuracy_vote | 93.00% | 22.7 | 0.014 | 0.001 |
| zoo | PFossil:pw_larger:vote | 93.00% | 22.7 | 0.014 | 0.001 |
| zoo | PFossil:pw_larger:weighted_vote | 60.36% | 22.7 | 0.014 | 0.003 |
| zoo | PFossil:pw_smaller:accuracy_vote | 94.00% | 22.9 | 0.014 | 0.001 |
| zoo | PFossil:pw_smaller:vote | 94.00% | 22.9 | 0.014 | 0.001 |
| zoo | PFossil:pw_smaller:weighted_vote | 93.00% | 22.9 | 0.014 | 0.003 |
| zoo | Pypper:ordered | 89.09% | 8.1 | 0.029 | 0.000 |
| zoo | Pypper:ovr | 90.09% | 10.3 | 0.035 | 0.000 |
| zoo | Pypper:pw_both:accuracy_vote | 91.00% | 40.0 | 0.162 | 0.001 |
| zoo | Pypper:pw_both:vote | 91.00% | 40.0 | 0.162 | 0.001 |
| zoo | Pypper:pw_both:weighted_vote | 84.18% | 40.0 | 0.162 | 0.006 |
| zoo | Pypper:pw_larger:accuracy_vote | 91.00% | 20.0 | 0.083 | 0.001 |
| zoo | Pypper:pw_larger:vote | 90.09% | 20.0 | 0.083 | 0.001 |
| zoo | Pypper:pw_larger:weighted_vote | 60.36% | 20.0 | 0.083 | 0.003 |
| zoo | Pypper:pw_smaller:accuracy_vote | 91.00% | 20.0 | 0.080 | 0.001 |
| zoo | Pypper:pw_smaller:vote | 89.09% | 20.0 | 0.080 | 0.000 |
| zoo | Pypper:pw_smaller:weighted_vote | 88.09% | 20.0 | 0.080 | 0.003 |

## Why the weighted vote does poorly

The weighted vote is clearly worse than the plain vote here, and much slower to predict. Both have the same cause in how it works: a pair model's vote is weighted by the Laplace estimate of the rule that decided it, and split -- weight `p` to the class it predicts, `1 - p` to the other class of its pair. Prediction is slower because it has to find, for every example and every pair model, the deciding rule among all covering rules, instead of just taking each model's prediction.

Most of a pair model's decisions are made by its default rule: of the c(c-1)/2 pair models, only c - 1 have the example's own class as one of theirs, so most of them see an example of neither of their classes, and their rules rarely fire. pyrulearn annotates every rule with the training examples its body covers, independently of the other rules -- so a rule means the same in every model it is part of, and the counts don't change when a rule set is reordered or turned into a decision list. The default rule's body is empty ("true"), so it is counted over all of its pair's examples, and its Laplace estimate is simply its class's share of the pair. For a balanced pair that is about 0.5: the default votes are almost neutral, while a rule that fires -- rightly or wrongly -- votes with a weight near 1. With the larger class as the pair's target, the default is the smaller class and its share is below 0.5, so the split gives most of the default's vote to the *other* class, against the model's own prediction.

To check that this is the cause, the default rules' counts are swapped, for prediction only, for counts over the examples each default actually decides (those no rule covers), on 4 small datasets (`primary-tumor`, `soybean`, `vowel`, `segment`; the first two with very unbalanced, the last two with balanced classes), 3-fold. Mean accuracy:

| base learner | pair target | vote | accuracy-weighted | weighted (default over all rows) | weighted (default over the rows it decides) |
|---|---|--:|--:|--:|--:|
| PFoil | larger | 74.07% | 75.50% | 37.95% | 76.69% |
| PFoil | smaller | 74.47% | 74.68% | 51.64% | 77.28% |
| PFossil | larger | 73.94% | 76.25% | 36.04% | 76.02% |
| PFossil | smaller | 76.66% | 77.54% | 50.08% | 77.32% |
| Pypper | larger | 63.77% | 75.50% | 35.09% | 74.41% |
| Pypper | smaller | 75.01% | 74.91% | 46.82% | 72.83% |

Counted over the examples it decides, the default rule gets a weight near its real reliability, and the weighted vote comes up to the level of the other two -- but not above it. The two ways of counting the default are two different meanings of a rule's counts (a statement in its own right vs. its role in this particular model); pyrulearn keeps the first, so the weighted vote stays as it is. The result: when most decisions come from default rules, rule-based vote weights add little, and the plain or the accuracy-weighted vote is the better choice.

Per dataset:

| dataset | base learner | pair target | vote | accuracy-weighted | weighted (default over all rows) | weighted (default over the rows it decides) |
|---|---|---|--:|--:|--:|--:|
| primary-tumor | PFoil | larger | 36.28% | 40.71% | 24.78% | 41.59% |
| primary-tumor | PFoil | smaller | 36.87% | 35.99% | 44.25% | 46.90% |
| primary-tumor | PFossil | larger | 30.09% | 37.46% | 24.78% | 38.35% |
| primary-tumor | PFossil | smaller | 40.41% | 42.48% | 40.71% | 44.25% |
| primary-tumor | Pypper | larger | 0.29% | 44.54% | 24.78% | 40.71% |
| primary-tumor | Pypper | smaller | 41.89% | 42.48% | 32.15% | 38.05% |
| segment | PFoil | larger | 89.61% | 90.30% | 48.27% | 93.46% |
| segment | PFoil | smaller | 89.61% | 90.30% | 48.27% | 93.46% |
| segment | PFossil | larger | 92.60% | 92.81% | 44.50% | 93.42% |
| segment | PFossil | smaller | 92.60% | 92.81% | 44.50% | 93.42% |
| segment | Pypper | larger | 94.94% | 94.59% | 45.02% | 94.72% |
| segment | Pypper | smaller | 94.94% | 94.59% | 45.02% | 94.72% |
| soybean | PFoil | larger | 90.49% | 91.07% | 39.97% | 90.78% |
| soybean | PFoil | smaller | 91.51% | 92.53% | 75.25% | 87.85% |
| soybean | PFossil | larger | 91.95% | 92.39% | 37.19% | 91.22% |
| soybean | PFossil | smaller | 92.53% | 92.54% | 77.45% | 90.48% |
| soybean | Pypper | larger | 88.15% | 91.95% | 35.73% | 91.22% |
| soybean | Pypper | smaller | 91.51% | 91.66% | 75.26% | 87.56% |
| vowel | PFoil | larger | 79.90% | 79.90% | 38.79% | 80.91% |
| vowel | PFoil | smaller | 79.90% | 79.90% | 38.79% | 80.91% |
| vowel | PFossil | larger | 81.11% | 82.32% | 37.68% | 81.11% |
| vowel | PFossil | smaller | 81.11% | 82.32% | 37.68% | 81.11% |
| vowel | Pypper | larger | 71.72% | 70.91% | 34.85% | 71.01% |
| vowel | Pypper | smaller | 71.72% | 70.91% | 34.85% | 71.01% |

