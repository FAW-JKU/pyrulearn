# Base against composite models: bagged rule sets, combined at two levels or one

## Setup

**Full run.** Quick sanity check instead: `python demos/base_vs_composite.py`.

Bagging with rule sets as members: 10, 25 and 50 concept sets -- one
concept description per class, learned one-vs-rest by Pypper, PFossil, PFoil -- each from
a bootstrap sample of the training fold; the smaller ensembles are the first
members of the largest. The same members are combined in two
ways, each with one of four conflict resolutions (**best rule**, the
covering rule with the best Laplace estimate; **voting**, one vote per
covering rule; **micro** / **macro distribution voting**, the covering
rules' class counts / class proportions summed):

- **two-level** -- every member resolves its own covering rules, and the
  members' predictions are combined by a plain vote;
- **one-level** -- the union of all members' rules as one rule set, its
  conflicts resolved directly.

A member whose rules don't cover an example abstains at both levels (the
members' defaults are ignored); examples no member covers go to the
majority class. **single** is the baseline: one concept set learned from the
whole training fold, with the same conflict resolution.

Datasets, with their numbers of classes: `balance-scale` (3), `cmc` (3), `dna` (3), `hayes-roth` (3), `iris` (3), `splice` (3), `wine` (3), `car` (4), `lymph` (4), `vehicle` (4), `anneal` (5), `glass` (6), `solar-flare` (6), `segment` (7), `zoo` (7), `ecoli` (8), `led24` (10), `yeast` (10), `optdigits` (10), `texture` (11), `vowel` (11), `soybean` (19), `primary-tumor` (21), `audiology` (24), `letter` (26). Two-class datasets: `SPECT`, `Titanic`, `banknote-authentication`, `blood-transfusion-service-center`, `breast-cancer`, `breast-w`, `churn`, `climate-model-simulation-crashes`, `colic`, `compas-two-years`, `credit-approval`, `credit-g`, `cylinder-bands`, `diabetes`, `dresses-sales`, `heart-c`, `heart-h`, `heart-statlog`, `heloc`, `hepatitis`, `ilpd`, `ionosphere`, `kc1`, `kc2`, `kr-vs-kp`, `mofn-3-7-10`, `molecular-biology_promoters`, `monks-problems-1`, `monks-problems-2`, `monks-problems-3`, `mushroom`, `ozone-level-8hr`, `pc1`, `pc3`, `pc4`, `phoneme`, `qsar-biodeg`, `sick`, `sonar`, `spambase`, `tic-tac-toe`, `vote`, `wdbc`, `wilt`. Protocol:
10-fold stratified cross-validation, one `DataSpec` per training fold
(`TreeDiscretizer(max_intervals=8)`). Each fit -- the members and the
single concept set together -- is capped at 1800s (3600s for
`letter`).

## All base learners

Accuracy averaged over the datasets and the three base learners.

| conflict resolution | single | two-level @10 | one-level @10 | two-level @25 | one-level @25 | two-level @50 | one-level @50 |
|---|--:|--:|--:|--:|--:|--:|--:|
| best rule | 83.36% | 85.14% | 83.41% | 85.49% | 82.96% | 85.57% | 82.62% |
| voting | 82.15% | 84.41% | 85.07% | 84.85% | 85.32% | 84.97% | 85.43% |
| micro distribution voting | 82.27% | 84.37% | 83.73% | 84.75% | 83.89% | 84.89% | 84.01% |
| macro distribution voting | 83.29% | 85.17% | 85.53% | 85.55% | 85.78% | 85.63% | 85.89% |

## Pypper

Rules of the single concept set: 19.4; of the ensembles: 187.7 (10 members), 470.3 (25 members), 940.4 (50 members).

| conflict resolution | single | two-level @10 | one-level @10 | two-level @25 | one-level @25 | two-level @50 | one-level @50 |
|---|--:|--:|--:|--:|--:|--:|--:|
| best rule | 83.16% | 85.08% (3.98) | 83.91% (6.14) | 85.43% (3.96) | 83.51% (6.34) | 85.54% (3.90) | 83.31% (6.38) |
| voting | 82.63% | 84.67% (4.62) | 85.26% (3.66) | 85.07% (4.58) | 85.49% (3.45) | 85.19% (4.43) | 85.52% (3.78) |
| micro distribution voting | 82.71% | 84.60% (5.04) | 83.85% (6.28) | 84.95% (5.25) | 83.95% (6.22) | 85.07% (5.01) | 84.17% (6.37) |
| macro distribution voting | 83.17% | 85.24% (3.56) | 85.53% (2.73) | 85.60% (3.33) | 85.63% (2.87) | 85.69% (3.30) | 85.79% (2.83) |

Accuracy, and in parentheses the mean rank among the eight ensemble variants of the same size (four conflict resolutions at two levels).

## PFossil

Rules of the single concept set: 19.2; of the ensembles: 201.4 (10 members), 503.3 (25 members), 1006.3 (50 members).

| conflict resolution | single | two-level @10 | one-level @10 | two-level @25 | one-level @25 | two-level @50 | one-level @50 |
|---|--:|--:|--:|--:|--:|--:|--:|
| best rule | 82.55% | 84.32% (3.86) | 82.63% (5.76) | 84.66% (3.86) | 82.17% (5.84) | 84.75% (3.91) | 81.79% (5.88) |
| voting | 81.65% | 83.42% (5.31) | 84.10% (3.96) | 83.76% (5.30) | 84.39% (4.12) | 83.87% (4.88) | 84.60% (4.06) |
| micro distribution voting | 81.48% | 83.13% (5.18) | 82.99% (5.46) | 83.50% (5.30) | 83.14% (5.43) | 83.67% (5.18) | 83.26% (5.67) |
| macro distribution voting | 82.39% | 84.34% (3.73) | 84.80% (2.75) | 84.66% (3.62) | 85.24% (2.52) | 84.76% (3.65) | 85.42% (2.75) |

Accuracy, and in parentheses the mean rank among the eight ensemble variants of the same size (four conflict resolutions at two levels).

## PFoil

Rules of the single concept set: 99.0; of the ensembles: 858.1 (10 members), 2144.0 (25 members), 4287.9 (50 members).

| conflict resolution | single | two-level @10 | one-level @10 | two-level @25 | one-level @25 | two-level @50 | one-level @50 |
|---|--:|--:|--:|--:|--:|--:|--:|
| best rule | 84.38% | 86.01% (3.75) | 83.70% (6.12) | 86.39% (3.57) | 83.20% (6.22) | 86.42% (3.65) | 82.75% (6.36) |
| voting | 82.17% | 85.15% (4.99) | 85.84% (3.90) | 85.71% (4.80) | 86.08% (4.14) | 85.87% (4.47) | 86.17% (4.09) |
| micro distribution voting | 82.62% | 85.39% (5.17) | 84.34% (5.84) | 85.79% (4.84) | 84.57% (6.03) | 85.91% (4.86) | 84.59% (6.04) |
| macro distribution voting | 84.31% | 85.94% (3.47) | 86.26% (2.76) | 86.37% (3.27) | 86.49% (3.12) | 86.44% (3.01) | 86.46% (3.50) |

Accuracy, and in parentheses the mean rank among the eight ensemble variants of the same size (four conflict resolutions at two levels).

## Two-level against one-level

Per (dataset, base learner) pair, the accuracy of the two-level combination minus that of the one-level one, in percentage points, and the number of pairs on which either is ahead.

| conflict resolution | members | mean difference | two-level ahead | one-level ahead | ties |
|---|--:|--:|--:|--:|--:|
| best rule | 10 | +1.72 | 156 | 44 | 7 |
| best rule | 25 | +2.54 | 162 | 38 | 7 |
| best rule | 50 | +2.95 | 160 | 40 | 7 |
| voting | 10 | -0.65 | 61 | 131 | 15 |
| voting | 25 | -0.47 | 61 | 126 | 20 |
| voting | 50 | -0.46 | 74 | 114 | 19 |
| micro distribution voting | 10 | +0.65 | 134 | 60 | 13 |
| micro distribution voting | 25 | +0.86 | 140 | 54 | 13 |
| micro distribution voting | 50 | +0.88 | 145 | 45 | 17 |
| macro distribution voting | 10 | -0.36 | 61 | 126 | 20 |
| macro distribution voting | 25 | -0.24 | 68 | 110 | 29 |
| macro distribution voting | 50 | -0.26 | 84 | 97 | 26 |

## Per dataset (mean across folds)

| dataset | learner | accuracy | n_rules |
|---|---|---|---|
| SPECT | PFoil:one-level@10 (best rule) | 84.66% | 229.5 |
| SPECT | PFoil:one-level@10 (macro distribution voting) | 84.25% | 229.5 |
| SPECT | PFoil:one-level@10 (micro distribution voting) | 79.42% | 229.5 |
| SPECT | PFoil:one-level@10 (voting) | 83.46% | 229.5 |
| SPECT | PFoil:one-level@25 (best rule) | 84.29% | 575.4 |
| SPECT | PFoil:one-level@25 (macro distribution voting) | 83.87% | 575.4 |
| SPECT | PFoil:one-level@25 (micro distribution voting) | 79.42% | 575.4 |
| SPECT | PFoil:one-level@25 (voting) | 82.72% | 575.4 |
| SPECT | PFoil:one-level@50 (best rule) | 84.29% | 1140.2 |
| SPECT | PFoil:one-level@50 (macro distribution voting) | 83.89% | 1140.2 |
| SPECT | PFoil:one-level@50 (micro distribution voting) | 79.42% | 1140.2 |
| SPECT | PFoil:one-level@50 (voting) | 82.35% | 1140.2 |
| SPECT | PFoil:single (best rule) | 82.05% | 25.8 |
| SPECT | PFoil:single (macro distribution voting) | 80.94% | 25.8 |
| SPECT | PFoil:single (micro distribution voting) | 76.81% | 25.8 |
| SPECT | PFoil:single (voting) | 79.77% | 25.8 |
| SPECT | PFoil:two-level@10 (best rule) | 85.00% | 229.5 |
| SPECT | PFoil:two-level@10 (macro distribution voting) | 84.25% | 229.5 |
| SPECT | PFoil:two-level@10 (micro distribution voting) | 79.40% | 229.5 |
| SPECT | PFoil:two-level@10 (voting) | 84.22% | 229.5 |
| SPECT | PFoil:two-level@25 (best rule) | 84.62% | 575.4 |
| SPECT | PFoil:two-level@25 (macro distribution voting) | 83.87% | 575.4 |
| SPECT | PFoil:two-level@25 (micro distribution voting) | 79.79% | 575.4 |
| SPECT | PFoil:two-level@25 (voting) | 83.11% | 575.4 |
| SPECT | PFoil:two-level@50 (best rule) | 85.00% | 1140.2 |
| SPECT | PFoil:two-level@50 (macro distribution voting) | 84.63% | 1140.2 |
| SPECT | PFoil:two-level@50 (micro distribution voting) | 79.40% | 1140.2 |
| SPECT | PFoil:two-level@50 (voting) | 84.23% | 1140.2 |
| SPECT | PFossil:one-level@10 (best rule) | 84.66% | 111.6 |
| SPECT | PFossil:one-level@10 (macro distribution voting) | 80.88% | 111.6 |
| SPECT | PFossil:one-level@10 (micro distribution voting) | 82.74% | 111.6 |
| SPECT | PFossil:one-level@10 (voting) | 79.03% | 111.6 |
| SPECT | PFossil:one-level@25 (best rule) | 84.29% | 281.3 |
| SPECT | PFossil:one-level@25 (macro distribution voting) | 81.62% | 281.3 |
| SPECT | PFossil:one-level@25 (micro distribution voting) | 83.87% | 281.3 |
| SPECT | PFossil:one-level@25 (voting) | 80.93% | 281.3 |
| SPECT | PFossil:one-level@50 (best rule) | 84.29% | 547.6 |
| SPECT | PFossil:one-level@50 (macro distribution voting) | 82.74% | 547.6 |
| SPECT | PFossil:one-level@50 (micro distribution voting) | 83.87% | 547.6 |
| SPECT | PFossil:one-level@50 (voting) | 80.54% | 547.6 |
| SPECT | PFossil:single (best rule) | 78.68% | 9.7 |
| SPECT | PFossil:single (macro distribution voting) | 77.55% | 9.7 |
| SPECT | PFossil:single (micro distribution voting) | 78.68% | 9.7 |
| SPECT | PFossil:single (voting) | 77.55% | 9.7 |
| SPECT | PFossil:two-level@10 (best rule) | 79.79% | 111.6 |
| SPECT | PFossil:two-level@10 (macro distribution voting) | 79.03% | 111.6 |
| SPECT | PFossil:two-level@10 (micro distribution voting) | 79.42% | 111.6 |
| SPECT | PFossil:two-level@10 (voting) | 79.42% | 111.6 |
| SPECT | PFossil:two-level@25 (best rule) | 80.90% | 281.3 |
| SPECT | PFossil:two-level@25 (macro distribution voting) | 80.53% | 281.3 |
| SPECT | PFossil:two-level@25 (micro distribution voting) | 80.90% | 281.3 |
| SPECT | PFossil:two-level@25 (voting) | 80.90% | 281.3 |
| SPECT | PFossil:two-level@50 (best rule) | 80.90% | 547.6 |
| SPECT | PFossil:two-level@50 (macro distribution voting) | 80.90% | 547.6 |
| SPECT | PFossil:two-level@50 (micro distribution voting) | 81.27% | 547.6 |
| SPECT | PFossil:two-level@50 (voting) | 81.27% | 547.6 |
| SPECT | Pypper:one-level@10 (best rule) | 83.89% | 54.6 |
| SPECT | Pypper:one-level@10 (macro distribution voting) | 83.50% | 54.6 |
| SPECT | Pypper:one-level@10 (micro distribution voting) | 82.76% | 54.6 |
| SPECT | Pypper:one-level@10 (voting) | 80.87% | 54.6 |
| SPECT | Pypper:one-level@25 (best rule) | 84.29% | 138.3 |
| SPECT | Pypper:one-level@25 (macro distribution voting) | 83.50% | 138.3 |
| SPECT | Pypper:one-level@25 (micro distribution voting) | 83.87% | 138.3 |
| SPECT | Pypper:one-level@25 (voting) | 82.76% | 138.3 |
| SPECT | Pypper:one-level@50 (best rule) | 84.29% | 269.6 |
| SPECT | Pypper:one-level@50 (macro distribution voting) | 83.87% | 269.6 |
| SPECT | Pypper:one-level@50 (micro distribution voting) | 83.52% | 269.6 |
| SPECT | Pypper:one-level@50 (voting) | 81.65% | 269.6 |
| SPECT | Pypper:single (best rule) | 80.85% | 4.4 |
| SPECT | Pypper:single (macro distribution voting) | 80.85% | 4.4 |
| SPECT | Pypper:single (micro distribution voting) | 80.85% | 4.4 |
| SPECT | Pypper:single (voting) | 80.85% | 4.4 |
| SPECT | Pypper:two-level@10 (best rule) | 81.64% | 54.6 |
| SPECT | Pypper:two-level@10 (macro distribution voting) | 81.64% | 54.6 |
| SPECT | Pypper:two-level@10 (micro distribution voting) | 81.64% | 54.6 |
| SPECT | Pypper:two-level@10 (voting) | 81.64% | 54.6 |
| SPECT | Pypper:two-level@25 (best rule) | 83.12% | 138.3 |
| SPECT | Pypper:two-level@25 (macro distribution voting) | 83.12% | 138.3 |
| SPECT | Pypper:two-level@25 (micro distribution voting) | 83.12% | 138.3 |
| SPECT | Pypper:two-level@25 (voting) | 83.12% | 138.3 |
| SPECT | Pypper:two-level@50 (best rule) | 81.28% | 269.6 |
| SPECT | Pypper:two-level@50 (macro distribution voting) | 81.28% | 269.6 |
| SPECT | Pypper:two-level@50 (micro distribution voting) | 81.28% | 269.6 |
| SPECT | Pypper:two-level@50 (voting) | 81.28% | 269.6 |
| Titanic | PFoil:one-level@10 (best rule) | 78.23% | 1000.1 |
| Titanic | PFoil:one-level@10 (macro distribution voting) | 79.14% | 1000.1 |
| Titanic | PFoil:one-level@10 (micro distribution voting) | 75.94% | 1000.1 |
| Titanic | PFoil:one-level@10 (voting) | 78.53% | 1000.1 |
| Titanic | PFoil:one-level@25 (best rule) | 78.38% | 2488.6 |
| Titanic | PFoil:one-level@25 (macro distribution voting) | 79.37% | 2488.6 |
| Titanic | PFoil:one-level@25 (micro distribution voting) | 76.09% | 2488.6 |
| Titanic | PFoil:one-level@25 (voting) | 78.84% | 2488.6 |
| Titanic | PFoil:one-level@50 (best rule) | 77.92% | 4969.9 |
| Titanic | PFoil:one-level@50 (macro distribution voting) | 79.07% | 4969.9 |
| Titanic | PFoil:one-level@50 (micro distribution voting) | 75.79% | 4969.9 |
| Titanic | PFoil:one-level@50 (voting) | 78.92% | 4969.9 |
| Titanic | PFoil:single (best rule) | 78.76% | 107.7 |
| Titanic | PFoil:single (macro distribution voting) | 77.92% | 107.7 |
| Titanic | PFoil:single (micro distribution voting) | 75.55% | 107.7 |
| Titanic | PFoil:single (voting) | 76.78% | 107.7 |
| Titanic | PFoil:two-level@10 (best rule) | 78.91% | 1000.1 |
| Titanic | PFoil:two-level@10 (macro distribution voting) | 78.68% | 1000.1 |
| Titanic | PFoil:two-level@10 (micro distribution voting) | 77.24% | 1000.1 |
| Titanic | PFoil:two-level@10 (voting) | 79.22% | 1000.1 |
| Titanic | PFoil:two-level@25 (best rule) | 78.84% | 2488.6 |
| Titanic | PFoil:two-level@25 (macro distribution voting) | 79.07% | 2488.6 |
| Titanic | PFoil:two-level@25 (micro distribution voting) | 78.00% | 2488.6 |
| Titanic | PFoil:two-level@25 (voting) | 79.52% | 2488.6 |
| Titanic | PFoil:two-level@50 (best rule) | 79.07% | 4969.9 |
| Titanic | PFoil:two-level@50 (macro distribution voting) | 79.14% | 4969.9 |
| Titanic | PFoil:two-level@50 (micro distribution voting) | 77.69% | 4969.9 |
| Titanic | PFoil:two-level@50 (voting) | 79.91% | 4969.9 |
| Titanic | PFossil:one-level@10 (best rule) | 78.53% | 64.9 |
| Titanic | PFossil:one-level@10 (macro distribution voting) | 78.84% | 64.9 |
| Titanic | PFossil:one-level@10 (micro distribution voting) | 78.53% | 64.9 |
| Titanic | PFossil:one-level@10 (voting) | 78.68% | 64.9 |
| Titanic | PFossil:one-level@25 (best rule) | 79.22% | 156.7 |
| Titanic | PFossil:one-level@25 (macro distribution voting) | 78.99% | 156.7 |
| Titanic | PFossil:one-level@25 (micro distribution voting) | 78.38% | 156.7 |
| Titanic | PFossil:one-level@25 (voting) | 78.84% | 156.7 |
| Titanic | PFossil:one-level@50 (best rule) | 79.37% | 314.8 |
| Titanic | PFossil:one-level@50 (macro distribution voting) | 78.76% | 314.8 |
| Titanic | PFossil:one-level@50 (micro distribution voting) | 78.23% | 314.8 |
| Titanic | PFossil:one-level@50 (voting) | 78.61% | 314.8 |
| Titanic | PFossil:single (best rule) | 77.77% | 5.1 |
| Titanic | PFossil:single (macro distribution voting) | 77.85% | 5.1 |
| Titanic | PFossil:single (micro distribution voting) | 76.78% | 5.1 |
| Titanic | PFossil:single (voting) | 78.15% | 5.1 |
| Titanic | PFossil:two-level@10 (best rule) | 78.91% | 64.9 |
| Titanic | PFossil:two-level@10 (macro distribution voting) | 78.61% | 64.9 |
| Titanic | PFossil:two-level@10 (micro distribution voting) | 77.84% | 64.9 |
| Titanic | PFossil:two-level@10 (voting) | 77.08% | 64.9 |
| Titanic | PFossil:two-level@25 (best rule) | 78.53% | 156.7 |
| Titanic | PFossil:two-level@25 (macro distribution voting) | 78.68% | 156.7 |
| Titanic | PFossil:two-level@25 (micro distribution voting) | 78.07% | 156.7 |
| Titanic | PFossil:two-level@25 (voting) | 77.62% | 156.7 |
| Titanic | PFossil:two-level@50 (best rule) | 78.61% | 314.8 |
| Titanic | PFossil:two-level@50 (macro distribution voting) | 78.68% | 314.8 |
| Titanic | PFossil:two-level@50 (micro distribution voting) | 78.38% | 314.8 |
| Titanic | PFossil:two-level@50 (voting) | 78.15% | 314.8 |
| Titanic | Pypper:one-level@10 (best rule) | 79.68% | 74.5 |
| Titanic | Pypper:one-level@10 (macro distribution voting) | 78.46% | 74.5 |
| Titanic | Pypper:one-level@10 (micro distribution voting) | 78.99% | 74.5 |
| Titanic | Pypper:one-level@10 (voting) | 78.61% | 74.5 |
| Titanic | Pypper:one-level@25 (best rule) | 79.52% | 182.8 |
| Titanic | Pypper:one-level@25 (macro distribution voting) | 78.99% | 182.8 |
| Titanic | Pypper:one-level@25 (micro distribution voting) | 78.61% | 182.8 |
| Titanic | Pypper:one-level@25 (voting) | 78.84% | 182.8 |
| Titanic | Pypper:one-level@50 (best rule) | 79.22% | 364.7 |
| Titanic | Pypper:one-level@50 (macro distribution voting) | 79.14% | 364.7 |
| Titanic | Pypper:one-level@50 (micro distribution voting) | 79.07% | 364.7 |
| Titanic | Pypper:one-level@50 (voting) | 79.07% | 364.7 |
| Titanic | Pypper:single (best rule) | 79.68% | 7.1 |
| Titanic | Pypper:single (macro distribution voting) | 79.68% | 7.1 |
| Titanic | Pypper:single (micro distribution voting) | 79.37% | 7.1 |
| Titanic | Pypper:single (voting) | 79.14% | 7.1 |
| Titanic | Pypper:two-level@10 (best rule) | 78.69% | 74.5 |
| Titanic | Pypper:two-level@10 (macro distribution voting) | 78.69% | 74.5 |
| Titanic | Pypper:two-level@10 (micro distribution voting) | 78.92% | 74.5 |
| Titanic | Pypper:two-level@10 (voting) | 78.76% | 74.5 |
| Titanic | Pypper:two-level@25 (best rule) | 78.76% | 182.8 |
| Titanic | Pypper:two-level@25 (macro distribution voting) | 78.76% | 182.8 |
| Titanic | Pypper:two-level@25 (micro distribution voting) | 78.91% | 182.8 |
| Titanic | Pypper:two-level@25 (voting) | 78.99% | 182.8 |
| Titanic | Pypper:two-level@50 (best rule) | 79.07% | 364.7 |
| Titanic | Pypper:two-level@50 (macro distribution voting) | 79.07% | 364.7 |
| Titanic | Pypper:two-level@50 (micro distribution voting) | 79.14% | 364.7 |
| Titanic | Pypper:two-level@50 (voting) | 79.14% | 364.7 |
| anneal | PFoil:one-level@10 (best rule) | 96.99% | 240.8 |
| anneal | PFoil:one-level@10 (macro distribution voting) | 98.33% | 240.8 |
| anneal | PFoil:one-level@10 (micro distribution voting) | 93.21% | 240.8 |
| anneal | PFoil:one-level@10 (voting) | 98.44% | 240.8 |
| anneal | PFoil:one-level@25 (best rule) | 96.99% | 604.1 |
| anneal | PFoil:one-level@25 (macro distribution voting) | 98.33% | 604.1 |
| anneal | PFoil:one-level@25 (micro distribution voting) | 92.87% | 604.1 |
| anneal | PFoil:one-level@25 (voting) | 98.33% | 604.1 |
| anneal | PFoil:one-level@50 (best rule) | 96.88% | 1207.6 |
| anneal | PFoil:one-level@50 (macro distribution voting) | 98.10% | 1207.6 |
| anneal | PFoil:one-level@50 (micro distribution voting) | 93.20% | 1207.6 |
| anneal | PFoil:one-level@50 (voting) | 98.88% | 1207.6 |
| anneal | PFoil:single (best rule) | 97.21% | 25.9 |
| anneal | PFoil:single (macro distribution voting) | 97.66% | 25.9 |
| anneal | PFoil:single (micro distribution voting) | 93.43% | 25.9 |
| anneal | PFoil:single (voting) | 94.88% | 25.9 |
| anneal | PFoil:two-level@10 (best rule) | 98.88% | 240.8 |
| anneal | PFoil:two-level@10 (macro distribution voting) | 99.00% | 240.8 |
| anneal | PFoil:two-level@10 (micro distribution voting) | 98.11% | 240.8 |
| anneal | PFoil:two-level@10 (voting) | 98.55% | 240.8 |
| anneal | PFoil:two-level@25 (best rule) | 99.00% | 604.1 |
| anneal | PFoil:two-level@25 (macro distribution voting) | 99.11% | 604.1 |
| anneal | PFoil:two-level@25 (micro distribution voting) | 98.66% | 604.1 |
| anneal | PFoil:two-level@25 (voting) | 98.66% | 604.1 |
| anneal | PFoil:two-level@50 (best rule) | 99.11% | 1207.6 |
| anneal | PFoil:two-level@50 (macro distribution voting) | 99.44% | 1207.6 |
| anneal | PFoil:two-level@50 (micro distribution voting) | 99.11% | 1207.6 |
| anneal | PFoil:two-level@50 (voting) | 99.11% | 1207.6 |
| anneal | PFossil:one-level@10 (best rule) | 98.22% | 181.0 |
| anneal | PFossil:one-level@10 (macro distribution voting) | 99.44% | 181.0 |
| anneal | PFossil:one-level@10 (micro distribution voting) | 98.33% | 181.0 |
| anneal | PFossil:one-level@10 (voting) | 98.33% | 181.0 |
| anneal | PFossil:one-level@25 (best rule) | 98.22% | 450.1 |
| anneal | PFossil:one-level@25 (macro distribution voting) | 99.44% | 450.1 |
| anneal | PFossil:one-level@25 (micro distribution voting) | 98.33% | 450.1 |
| anneal | PFossil:one-level@25 (voting) | 97.99% | 450.1 |
| anneal | PFossil:one-level@50 (best rule) | 97.99% | 886.7 |
| anneal | PFossil:one-level@50 (macro distribution voting) | 99.33% | 886.7 |
| anneal | PFossil:one-level@50 (micro distribution voting) | 98.44% | 886.7 |
| anneal | PFossil:one-level@50 (voting) | 98.00% | 886.7 |
| anneal | PFossil:single (best rule) | 99.22% | 17.8 |
| anneal | PFossil:single (macro distribution voting) | 99.66% | 17.8 |
| anneal | PFossil:single (micro distribution voting) | 98.88% | 17.8 |
| anneal | PFossil:single (voting) | 99.66% | 17.8 |
| anneal | PFossil:two-level@10 (best rule) | 99.33% | 181.0 |
| anneal | PFossil:two-level@10 (macro distribution voting) | 99.55% | 181.0 |
| anneal | PFossil:two-level@10 (micro distribution voting) | 99.22% | 181.0 |
| anneal | PFossil:two-level@10 (voting) | 99.55% | 181.0 |
| anneal | PFossil:two-level@25 (best rule) | 99.22% | 450.1 |
| anneal | PFossil:two-level@25 (macro distribution voting) | 99.55% | 450.1 |
| anneal | PFossil:two-level@25 (micro distribution voting) | 99.22% | 450.1 |
| anneal | PFossil:two-level@25 (voting) | 99.55% | 450.1 |
| anneal | PFossil:two-level@50 (best rule) | 99.22% | 886.7 |
| anneal | PFossil:two-level@50 (macro distribution voting) | 99.55% | 886.7 |
| anneal | PFossil:two-level@50 (micro distribution voting) | 99.11% | 886.7 |
| anneal | PFossil:two-level@50 (voting) | 99.55% | 886.7 |
| anneal | Pypper:one-level@10 (best rule) | 97.21% | 145.0 |
| anneal | Pypper:one-level@10 (macro distribution voting) | 98.66% | 145.0 |
| anneal | Pypper:one-level@10 (micro distribution voting) | 97.10% | 145.0 |
| anneal | Pypper:one-level@10 (voting) | 98.22% | 145.0 |
| anneal | Pypper:one-level@25 (best rule) | 96.88% | 361.3 |
| anneal | Pypper:one-level@25 (macro distribution voting) | 98.66% | 361.3 |
| anneal | Pypper:one-level@25 (micro distribution voting) | 97.10% | 361.3 |
| anneal | Pypper:one-level@25 (voting) | 98.22% | 361.3 |
| anneal | Pypper:one-level@50 (best rule) | 96.21% | 725.2 |
| anneal | Pypper:one-level@50 (macro distribution voting) | 98.66% | 725.2 |
| anneal | Pypper:one-level@50 (micro distribution voting) | 97.32% | 725.2 |
| anneal | Pypper:one-level@50 (voting) | 98.66% | 725.2 |
| anneal | Pypper:single (best rule) | 95.44% | 14.3 |
| anneal | Pypper:single (macro distribution voting) | 95.66% | 14.3 |
| anneal | Pypper:single (micro distribution voting) | 95.10% | 14.3 |
| anneal | Pypper:single (voting) | 95.44% | 14.3 |
| anneal | Pypper:two-level@10 (best rule) | 98.22% | 145.0 |
| anneal | Pypper:two-level@10 (macro distribution voting) | 98.33% | 145.0 |
| anneal | Pypper:two-level@10 (micro distribution voting) | 98.10% | 145.0 |
| anneal | Pypper:two-level@10 (voting) | 98.22% | 145.0 |
| anneal | Pypper:two-level@25 (best rule) | 98.33% | 361.3 |
| anneal | Pypper:two-level@25 (macro distribution voting) | 98.55% | 361.3 |
| anneal | Pypper:two-level@25 (micro distribution voting) | 98.22% | 361.3 |
| anneal | Pypper:two-level@25 (voting) | 98.44% | 361.3 |
| anneal | Pypper:two-level@50 (best rule) | 98.66% | 725.2 |
| anneal | Pypper:two-level@50 (macro distribution voting) | 98.89% | 725.2 |
| anneal | Pypper:two-level@50 (micro distribution voting) | 98.66% | 725.2 |
| anneal | Pypper:two-level@50 (voting) | 98.89% | 725.2 |
| audiology | PFoil:one-level@10 (best rule) | 70.40% | 374.1 |
| audiology | PFoil:one-level@10 (macro distribution voting) | 82.37% | 374.1 |
| audiology | PFoil:one-level@10 (micro distribution voting) | 71.32% | 374.1 |
| audiology | PFoil:one-level@10 (voting) | 77.94% | 374.1 |
| audiology | PFoil:one-level@25 (best rule) | 68.64% | 935.3 |
| audiology | PFoil:one-level@25 (macro distribution voting) | 82.35% | 935.3 |
| audiology | PFoil:one-level@25 (micro distribution voting) | 72.63% | 935.3 |
| audiology | PFoil:one-level@25 (voting) | 79.25% | 935.3 |
| audiology | PFoil:one-level@50 (best rule) | 67.25% | 1876.5 |
| audiology | PFoil:one-level@50 (macro distribution voting) | 83.70% | 1876.5 |
| audiology | PFoil:one-level@50 (micro distribution voting) | 72.61% | 1876.5 |
| audiology | PFoil:one-level@50 (voting) | 80.10% | 1876.5 |
| audiology | PFoil:single (best rule) | 77.04% | 42.3 |
| audiology | PFoil:single (macro distribution voting) | 78.81% | 42.3 |
| audiology | PFoil:single (micro distribution voting) | 61.09% | 42.3 |
| audiology | PFoil:single (voting) | 62.41% | 42.3 |
| audiology | PFoil:two-level@10 (best rule) | 79.70% | 374.1 |
| audiology | PFoil:two-level@10 (macro distribution voting) | 79.72% | 374.1 |
| audiology | PFoil:two-level@10 (micro distribution voting) | 77.51% | 374.1 |
| audiology | PFoil:two-level@10 (voting) | 73.54% | 374.1 |
| audiology | PFoil:two-level@25 (best rule) | 79.70% | 935.3 |
| audiology | PFoil:two-level@25 (macro distribution voting) | 79.70% | 935.3 |
| audiology | PFoil:two-level@25 (micro distribution voting) | 78.38% | 935.3 |
| audiology | PFoil:two-level@25 (voting) | 73.52% | 935.3 |
| audiology | PFoil:two-level@50 (best rule) | 78.79% | 1876.5 |
| audiology | PFoil:two-level@50 (macro distribution voting) | 81.44% | 1876.5 |
| audiology | PFoil:two-level@50 (micro distribution voting) | 77.89% | 1876.5 |
| audiology | PFoil:two-level@50 (voting) | 74.41% | 1876.5 |
| audiology | PFossil:one-level@10 (best rule) | 71.19% | 355.4 |
| audiology | PFossil:one-level@10 (macro distribution voting) | 80.57% | 355.4 |
| audiology | PFossil:one-level@10 (micro distribution voting) | 73.87% | 355.4 |
| audiology | PFossil:one-level@10 (voting) | 81.48% | 355.4 |
| audiology | PFossil:one-level@25 (best rule) | 66.38% | 904.5 |
| audiology | PFossil:one-level@25 (macro distribution voting) | 81.50% | 904.5 |
| audiology | PFossil:one-level@25 (micro distribution voting) | 73.91% | 904.5 |
| audiology | PFossil:one-level@25 (voting) | 82.37% | 904.5 |
| audiology | PFossil:one-level@50 (best rule) | 65.47% | 1817.6 |
| audiology | PFossil:one-level@50 (macro distribution voting) | 82.79% | 1817.6 |
| audiology | PFossil:one-level@50 (micro distribution voting) | 72.57% | 1817.6 |
| audiology | PFossil:one-level@50 (voting) | 82.79% | 1817.6 |
| audiology | PFossil:single (best rule) | 77.89% | 43.3 |
| audiology | PFossil:single (macro distribution voting) | 77.00% | 43.3 |
| audiology | PFossil:single (micro distribution voting) | 74.78% | 43.3 |
| audiology | PFossil:single (voting) | 75.26% | 43.3 |
| audiology | PFossil:two-level@10 (best rule) | 80.55% | 355.4 |
| audiology | PFossil:two-level@10 (macro distribution voting) | 79.68% | 355.4 |
| audiology | PFossil:two-level@10 (micro distribution voting) | 78.34% | 355.4 |
| audiology | PFossil:two-level@10 (voting) | 77.47% | 355.4 |
| audiology | PFossil:two-level@25 (best rule) | 81.46% | 904.5 |
| audiology | PFossil:two-level@25 (macro distribution voting) | 80.59% | 904.5 |
| audiology | PFossil:two-level@25 (micro distribution voting) | 79.68% | 904.5 |
| audiology | PFossil:two-level@25 (voting) | 81.05% | 904.5 |
| audiology | PFossil:two-level@50 (best rule) | 81.90% | 1817.6 |
| audiology | PFossil:two-level@50 (macro distribution voting) | 81.50% | 1817.6 |
| audiology | PFossil:two-level@50 (micro distribution voting) | 80.12% | 1817.6 |
| audiology | PFossil:two-level@50 (voting) | 80.18% | 1817.6 |
| audiology | Pypper:one-level@10 (best rule) | 72.65% | 156.8 |
| audiology | Pypper:one-level@10 (macro distribution voting) | 76.58% | 156.8 |
| audiology | Pypper:one-level@10 (micro distribution voting) | 72.65% | 156.8 |
| audiology | Pypper:one-level@10 (voting) | 75.69% | 156.8 |
| audiology | Pypper:one-level@25 (best rule) | 69.94% | 391.7 |
| audiology | Pypper:one-level@25 (macro distribution voting) | 76.17% | 391.7 |
| audiology | Pypper:one-level@25 (micro distribution voting) | 70.81% | 391.7 |
| audiology | Pypper:one-level@25 (voting) | 74.82% | 391.7 |
| audiology | Pypper:one-level@50 (best rule) | 69.07% | 791.6 |
| audiology | Pypper:one-level@50 (macro distribution voting) | 76.60% | 791.6 |
| audiology | Pypper:one-level@50 (micro distribution voting) | 72.61% | 791.6 |
| audiology | Pypper:one-level@50 (voting) | 74.82% | 791.6 |
| audiology | Pypper:single (best rule) | 66.84% | 13.8 |
| audiology | Pypper:single (macro distribution voting) | 67.73% | 13.8 |
| audiology | Pypper:single (micro distribution voting) | 65.51% | 13.8 |
| audiology | Pypper:single (voting) | 63.77% | 13.8 |
| audiology | Pypper:two-level@10 (best rule) | 76.58% | 156.8 |
| audiology | Pypper:two-level@10 (macro distribution voting) | 77.47% | 156.8 |
| audiology | Pypper:two-level@10 (micro distribution voting) | 75.71% | 156.8 |
| audiology | Pypper:two-level@10 (voting) | 74.41% | 156.8 |
| audiology | Pypper:two-level@25 (best rule) | 75.69% | 391.7 |
| audiology | Pypper:two-level@25 (macro distribution voting) | 76.15% | 391.7 |
| audiology | Pypper:two-level@25 (micro distribution voting) | 73.91% | 391.7 |
| audiology | Pypper:two-level@25 (voting) | 73.02% | 391.7 |
| audiology | Pypper:two-level@50 (best rule) | 76.17% | 791.6 |
| audiology | Pypper:two-level@50 (macro distribution voting) | 77.49% | 791.6 |
| audiology | Pypper:two-level@50 (micro distribution voting) | 73.91% | 791.6 |
| audiology | Pypper:two-level@50 (voting) | 73.02% | 791.6 |
| balance-scale | PFoil:one-level@10 (best rule) | 85.60% | 600.8 |
| balance-scale | PFoil:one-level@10 (macro distribution voting) | 88.16% | 600.8 |
| balance-scale | PFoil:one-level@10 (micro distribution voting) | 88.48% | 600.8 |
| balance-scale | PFoil:one-level@10 (voting) | 84.47% | 600.8 |
| balance-scale | PFoil:one-level@25 (best rule) | 86.24% | 1511.8 |
| balance-scale | PFoil:one-level@25 (macro distribution voting) | 87.52% | 1511.8 |
| balance-scale | PFoil:one-level@25 (micro distribution voting) | 88.00% | 1511.8 |
| balance-scale | PFoil:one-level@25 (voting) | 83.51% | 1511.8 |
| balance-scale | PFoil:one-level@50 (best rule) | 84.95% | 3040.5 |
| balance-scale | PFoil:one-level@50 (macro distribution voting) | 88.80% | 3040.5 |
| balance-scale | PFoil:one-level@50 (micro distribution voting) | 88.01% | 3040.5 |
| balance-scale | PFoil:one-level@50 (voting) | 84.64% | 3040.5 |
| balance-scale | PFoil:single (best rule) | 82.87% | 72.7 |
| balance-scale | PFoil:single (macro distribution voting) | 82.54% | 72.7 |
| balance-scale | PFoil:single (micro distribution voting) | 82.39% | 72.7 |
| balance-scale | PFoil:single (voting) | 81.11% | 72.7 |
| balance-scale | PFoil:two-level@10 (best rule) | 86.72% | 600.8 |
| balance-scale | PFoil:two-level@10 (macro distribution voting) | 85.91% | 600.8 |
| balance-scale | PFoil:two-level@10 (micro distribution voting) | 86.88% | 600.8 |
| balance-scale | PFoil:two-level@10 (voting) | 84.80% | 600.8 |
| balance-scale | PFoil:two-level@25 (best rule) | 86.40% | 1511.8 |
| balance-scale | PFoil:two-level@25 (macro distribution voting) | 85.28% | 1511.8 |
| balance-scale | PFoil:two-level@25 (micro distribution voting) | 86.88% | 1511.8 |
| balance-scale | PFoil:two-level@25 (voting) | 85.44% | 1511.8 |
| balance-scale | PFoil:two-level@50 (best rule) | 86.40% | 3040.5 |
| balance-scale | PFoil:two-level@50 (macro distribution voting) | 85.44% | 3040.5 |
| balance-scale | PFoil:two-level@50 (micro distribution voting) | 86.56% | 3040.5 |
| balance-scale | PFoil:two-level@50 (voting) | 85.76% | 3040.5 |
| balance-scale | PFossil:one-level@10 (best rule) | 83.34% | 240.3 |
| balance-scale | PFossil:one-level@10 (macro distribution voting) | 87.03% | 240.3 |
| balance-scale | PFossil:one-level@10 (micro distribution voting) | 86.23% | 240.3 |
| balance-scale | PFossil:one-level@10 (voting) | 87.68% | 240.3 |
| balance-scale | PFossil:one-level@25 (best rule) | 85.27% | 604.0 |
| balance-scale | PFossil:one-level@25 (macro distribution voting) | 87.04% | 604.0 |
| balance-scale | PFossil:one-level@25 (micro distribution voting) | 86.08% | 604.0 |
| balance-scale | PFossil:one-level@25 (voting) | 86.88% | 604.0 |
| balance-scale | PFossil:one-level@50 (best rule) | 85.44% | 1222.8 |
| balance-scale | PFossil:one-level@50 (macro distribution voting) | 87.68% | 1222.8 |
| balance-scale | PFossil:one-level@50 (micro distribution voting) | 87.37% | 1222.8 |
| balance-scale | PFossil:one-level@50 (voting) | 87.68% | 1222.8 |
| balance-scale | PFossil:single (best rule) | 81.60% | 26.0 |
| balance-scale | PFossil:single (macro distribution voting) | 82.08% | 26.0 |
| balance-scale | PFossil:single (micro distribution voting) | 80.47% | 26.0 |
| balance-scale | PFossil:single (voting) | 79.67% | 26.0 |
| balance-scale | PFossil:two-level@10 (best rule) | 86.23% | 240.3 |
| balance-scale | PFossil:two-level@10 (macro distribution voting) | 86.07% | 240.3 |
| balance-scale | PFossil:two-level@10 (micro distribution voting) | 86.23% | 240.3 |
| balance-scale | PFossil:two-level@10 (voting) | 86.07% | 240.3 |
| balance-scale | PFossil:two-level@25 (best rule) | 86.72% | 604.0 |
| balance-scale | PFossil:two-level@25 (macro distribution voting) | 87.20% | 604.0 |
| balance-scale | PFossil:two-level@25 (micro distribution voting) | 85.76% | 604.0 |
| balance-scale | PFossil:two-level@25 (voting) | 86.39% | 604.0 |
| balance-scale | PFossil:two-level@50 (best rule) | 87.69% | 1222.8 |
| balance-scale | PFossil:two-level@50 (macro distribution voting) | 87.20% | 1222.8 |
| balance-scale | PFossil:two-level@50 (micro distribution voting) | 87.36% | 1222.8 |
| balance-scale | PFossil:two-level@50 (voting) | 87.85% | 1222.8 |
| balance-scale | Pypper:one-level@10 (best rule) | 85.11% | 157.2 |
| balance-scale | Pypper:one-level@10 (macro distribution voting) | 87.52% | 157.2 |
| balance-scale | Pypper:one-level@10 (micro distribution voting) | 86.88% | 157.2 |
| balance-scale | Pypper:one-level@10 (voting) | 87.68% | 157.2 |
| balance-scale | Pypper:one-level@25 (best rule) | 84.16% | 393.7 |
| balance-scale | Pypper:one-level@25 (macro distribution voting) | 88.01% | 393.7 |
| balance-scale | Pypper:one-level@25 (micro distribution voting) | 87.53% | 393.7 |
| balance-scale | Pypper:one-level@25 (voting) | 88.01% | 393.7 |
| balance-scale | Pypper:one-level@50 (best rule) | 84.96% | 776.5 |
| balance-scale | Pypper:one-level@50 (macro distribution voting) | 88.65% | 776.5 |
| balance-scale | Pypper:one-level@50 (micro distribution voting) | 87.69% | 776.5 |
| balance-scale | Pypper:one-level@50 (voting) | 88.49% | 776.5 |
| balance-scale | Pypper:single (best rule) | 81.75% | 15.8 |
| balance-scale | Pypper:single (macro distribution voting) | 81.75% | 15.8 |
| balance-scale | Pypper:single (micro distribution voting) | 81.26% | 15.8 |
| balance-scale | Pypper:single (voting) | 81.26% | 15.8 |
| balance-scale | Pypper:two-level@10 (best rule) | 87.04% | 157.2 |
| balance-scale | Pypper:two-level@10 (macro distribution voting) | 86.72% | 157.2 |
| balance-scale | Pypper:two-level@10 (micro distribution voting) | 85.92% | 157.2 |
| balance-scale | Pypper:two-level@10 (voting) | 86.56% | 157.2 |
| balance-scale | Pypper:two-level@25 (best rule) | 87.85% | 393.7 |
| balance-scale | Pypper:two-level@25 (macro distribution voting) | 87.69% | 393.7 |
| balance-scale | Pypper:two-level@25 (micro distribution voting) | 87.84% | 393.7 |
| balance-scale | Pypper:two-level@25 (voting) | 88.16% | 393.7 |
| balance-scale | Pypper:two-level@50 (best rule) | 87.69% | 776.5 |
| balance-scale | Pypper:two-level@50 (macro distribution voting) | 87.69% | 776.5 |
| balance-scale | Pypper:two-level@50 (micro distribution voting) | 87.37% | 776.5 |
| balance-scale | Pypper:two-level@50 (voting) | 88.33% | 776.5 |
| banknote-authentication | PFoil:one-level@10 (best rule) | 98.25% | 207.6 |
| banknote-authentication | PFoil:one-level@10 (macro distribution voting) | 98.91% | 207.6 |
| banknote-authentication | PFoil:one-level@10 (micro distribution voting) | 98.62% | 207.6 |
| banknote-authentication | PFoil:one-level@10 (voting) | 98.91% | 207.6 |
| banknote-authentication | PFoil:one-level@25 (best rule) | 98.11% | 524.7 |
| banknote-authentication | PFoil:one-level@25 (macro distribution voting) | 98.91% | 524.7 |
| banknote-authentication | PFoil:one-level@25 (micro distribution voting) | 98.18% | 524.7 |
| banknote-authentication | PFoil:one-level@25 (voting) | 98.91% | 524.7 |
| banknote-authentication | PFoil:one-level@50 (best rule) | 97.81% | 1045.3 |
| banknote-authentication | PFoil:one-level@50 (macro distribution voting) | 98.98% | 1045.3 |
| banknote-authentication | PFoil:one-level@50 (micro distribution voting) | 98.54% | 1045.3 |
| banknote-authentication | PFoil:one-level@50 (voting) | 99.05% | 1045.3 |
| banknote-authentication | PFoil:single (best rule) | 98.40% | 25.7 |
| banknote-authentication | PFoil:single (macro distribution voting) | 98.76% | 25.7 |
| banknote-authentication | PFoil:single (micro distribution voting) | 98.18% | 25.7 |
| banknote-authentication | PFoil:single (voting) | 98.91% | 25.7 |
| banknote-authentication | PFoil:two-level@10 (best rule) | 98.91% | 207.6 |
| banknote-authentication | PFoil:two-level@10 (macro distribution voting) | 99.05% | 207.6 |
| banknote-authentication | PFoil:two-level@10 (micro distribution voting) | 98.91% | 207.6 |
| banknote-authentication | PFoil:two-level@10 (voting) | 99.05% | 207.6 |
| banknote-authentication | PFoil:two-level@25 (best rule) | 99.05% | 524.7 |
| banknote-authentication | PFoil:two-level@25 (macro distribution voting) | 99.13% | 524.7 |
| banknote-authentication | PFoil:two-level@25 (micro distribution voting) | 99.05% | 524.7 |
| banknote-authentication | PFoil:two-level@25 (voting) | 99.13% | 524.7 |
| banknote-authentication | PFoil:two-level@50 (best rule) | 99.05% | 1045.3 |
| banknote-authentication | PFoil:two-level@50 (macro distribution voting) | 99.20% | 1045.3 |
| banknote-authentication | PFoil:two-level@50 (micro distribution voting) | 99.05% | 1045.3 |
| banknote-authentication | PFoil:two-level@50 (voting) | 99.20% | 1045.3 |
| banknote-authentication | PFossil:one-level@10 (best rule) | 98.62% | 116.1 |
| banknote-authentication | PFossil:one-level@10 (macro distribution voting) | 98.69% | 116.1 |
| banknote-authentication | PFossil:one-level@10 (micro distribution voting) | 94.46% | 116.1 |
| banknote-authentication | PFossil:one-level@10 (voting) | 98.18% | 116.1 |
| banknote-authentication | PFossil:one-level@25 (best rule) | 98.83% | 296.6 |
| banknote-authentication | PFossil:one-level@25 (macro distribution voting) | 98.54% | 296.6 |
| banknote-authentication | PFossil:one-level@25 (micro distribution voting) | 94.39% | 296.6 |
| banknote-authentication | PFossil:one-level@25 (voting) | 98.18% | 296.6 |
| banknote-authentication | PFossil:one-level@50 (best rule) | 98.62% | 586.5 |
| banknote-authentication | PFossil:one-level@50 (macro distribution voting) | 98.62% | 586.5 |
| banknote-authentication | PFossil:one-level@50 (micro distribution voting) | 94.46% | 586.5 |
| banknote-authentication | PFossil:one-level@50 (voting) | 98.18% | 586.5 |
| banknote-authentication | PFossil:single (best rule) | 99.20% | 14.0 |
| banknote-authentication | PFossil:single (macro distribution voting) | 99.05% | 14.0 |
| banknote-authentication | PFossil:single (micro distribution voting) | 94.32% | 14.0 |
| banknote-authentication | PFossil:single (voting) | 97.89% | 14.0 |
| banknote-authentication | PFossil:two-level@10 (best rule) | 99.13% | 116.1 |
| banknote-authentication | PFossil:two-level@10 (macro distribution voting) | 99.05% | 116.1 |
| banknote-authentication | PFossil:two-level@10 (micro distribution voting) | 94.32% | 116.1 |
| banknote-authentication | PFossil:two-level@10 (voting) | 97.74% | 116.1 |
| banknote-authentication | PFossil:two-level@25 (best rule) | 99.13% | 296.6 |
| banknote-authentication | PFossil:two-level@25 (macro distribution voting) | 99.13% | 296.6 |
| banknote-authentication | PFossil:two-level@25 (micro distribution voting) | 94.32% | 296.6 |
| banknote-authentication | PFossil:two-level@25 (voting) | 97.96% | 296.6 |
| banknote-authentication | PFossil:two-level@50 (best rule) | 99.20% | 586.5 |
| banknote-authentication | PFossil:two-level@50 (macro distribution voting) | 99.20% | 586.5 |
| banknote-authentication | PFossil:two-level@50 (micro distribution voting) | 94.32% | 586.5 |
| banknote-authentication | PFossil:two-level@50 (voting) | 97.89% | 586.5 |
| banknote-authentication | Pypper:one-level@10 (best rule) | 98.25% | 103.2 |
| banknote-authentication | Pypper:one-level@10 (macro distribution voting) | 98.61% | 103.2 |
| banknote-authentication | Pypper:one-level@10 (micro distribution voting) | 98.18% | 103.2 |
| banknote-authentication | Pypper:one-level@10 (voting) | 98.69% | 103.2 |
| banknote-authentication | Pypper:one-level@25 (best rule) | 97.96% | 260.2 |
| banknote-authentication | Pypper:one-level@25 (macro distribution voting) | 98.69% | 260.2 |
| banknote-authentication | Pypper:one-level@25 (micro distribution voting) | 98.25% | 260.2 |
| banknote-authentication | Pypper:one-level@25 (voting) | 98.69% | 260.2 |
| banknote-authentication | Pypper:one-level@50 (best rule) | 97.59% | 517.2 |
| banknote-authentication | Pypper:one-level@50 (macro distribution voting) | 98.76% | 517.2 |
| banknote-authentication | Pypper:one-level@50 (micro distribution voting) | 98.25% | 517.2 |
| banknote-authentication | Pypper:one-level@50 (voting) | 98.76% | 517.2 |
| banknote-authentication | Pypper:single (best rule) | 98.61% | 10.4 |
| banknote-authentication | Pypper:single (macro distribution voting) | 98.69% | 10.4 |
| banknote-authentication | Pypper:single (micro distribution voting) | 98.61% | 10.4 |
| banknote-authentication | Pypper:single (voting) | 98.69% | 10.4 |
| banknote-authentication | Pypper:two-level@10 (best rule) | 98.69% | 103.2 |
| banknote-authentication | Pypper:two-level@10 (macro distribution voting) | 98.98% | 103.2 |
| banknote-authentication | Pypper:two-level@10 (micro distribution voting) | 98.62% | 103.2 |
| banknote-authentication | Pypper:two-level@10 (voting) | 98.91% | 103.2 |
| banknote-authentication | Pypper:two-level@25 (best rule) | 98.61% | 260.2 |
| banknote-authentication | Pypper:two-level@25 (macro distribution voting) | 98.83% | 260.2 |
| banknote-authentication | Pypper:two-level@25 (micro distribution voting) | 98.54% | 260.2 |
| banknote-authentication | Pypper:two-level@25 (voting) | 98.83% | 260.2 |
| banknote-authentication | Pypper:two-level@50 (best rule) | 98.61% | 517.2 |
| banknote-authentication | Pypper:two-level@50 (macro distribution voting) | 98.76% | 517.2 |
| banknote-authentication | Pypper:two-level@50 (micro distribution voting) | 98.61% | 517.2 |
| banknote-authentication | Pypper:two-level@50 (voting) | 98.83% | 517.2 |
| blood-transfusion-service-center | PFoil:one-level@10 (best rule) | 76.47% | 536.1 |
| blood-transfusion-service-center | PFoil:one-level@10 (macro distribution voting) | 76.61% | 536.1 |
| blood-transfusion-service-center | PFoil:one-level@10 (micro distribution voting) | 76.07% | 536.1 |
| blood-transfusion-service-center | PFoil:one-level@10 (voting) | 73.40% | 536.1 |
| blood-transfusion-service-center | PFoil:one-level@25 (best rule) | 77.00% | 1320.9 |
| blood-transfusion-service-center | PFoil:one-level@25 (macro distribution voting) | 76.47% | 1320.9 |
| blood-transfusion-service-center | PFoil:one-level@25 (micro distribution voting) | 76.21% | 1320.9 |
| blood-transfusion-service-center | PFoil:one-level@25 (voting) | 72.20% | 1320.9 |
| blood-transfusion-service-center | PFoil:one-level@50 (best rule) | 77.00% | 2635.5 |
| blood-transfusion-service-center | PFoil:one-level@50 (macro distribution voting) | 76.47% | 2635.5 |
| blood-transfusion-service-center | PFoil:one-level@50 (micro distribution voting) | 76.21% | 2635.5 |
| blood-transfusion-service-center | PFoil:one-level@50 (voting) | 73.26% | 2635.5 |
| blood-transfusion-service-center | PFoil:single (best rule) | 76.07% | 57.4 |
| blood-transfusion-service-center | PFoil:single (macro distribution voting) | 75.94% | 57.4 |
| blood-transfusion-service-center | PFoil:single (micro distribution voting) | 76.07% | 57.4 |
| blood-transfusion-service-center | PFoil:single (voting) | 68.05% | 57.4 |
| blood-transfusion-service-center | PFoil:two-level@10 (best rule) | 77.14% | 536.1 |
| blood-transfusion-service-center | PFoil:two-level@10 (macro distribution voting) | 76.47% | 536.1 |
| blood-transfusion-service-center | PFoil:two-level@10 (micro distribution voting) | 76.34% | 536.1 |
| blood-transfusion-service-center | PFoil:two-level@10 (voting) | 77.00% | 536.1 |
| blood-transfusion-service-center | PFoil:two-level@25 (best rule) | 77.27% | 1320.9 |
| blood-transfusion-service-center | PFoil:two-level@25 (macro distribution voting) | 76.47% | 1320.9 |
| blood-transfusion-service-center | PFoil:two-level@25 (micro distribution voting) | 76.74% | 1320.9 |
| blood-transfusion-service-center | PFoil:two-level@25 (voting) | 75.14% | 1320.9 |
| blood-transfusion-service-center | PFoil:two-level@50 (best rule) | 76.74% | 2635.5 |
| blood-transfusion-service-center | PFoil:two-level@50 (macro distribution voting) | 76.61% | 2635.5 |
| blood-transfusion-service-center | PFoil:two-level@50 (micro distribution voting) | 76.61% | 2635.5 |
| blood-transfusion-service-center | PFoil:two-level@50 (voting) | 75.93% | 2635.5 |
| blood-transfusion-service-center | PFossil:one-level@10 (best rule) | 73.12% | 18.9 |
| blood-transfusion-service-center | PFossil:one-level@10 (macro distribution voting) | 73.52% | 18.9 |
| blood-transfusion-service-center | PFossil:one-level@10 (micro distribution voting) | 73.26% | 18.9 |
| blood-transfusion-service-center | PFossil:one-level@10 (voting) | 68.46% | 18.9 |
| blood-transfusion-service-center | PFossil:one-level@25 (best rule) | 76.07% | 53.8 |
| blood-transfusion-service-center | PFossil:one-level@25 (macro distribution voting) | 77.00% | 53.8 |
| blood-transfusion-service-center | PFossil:one-level@25 (micro distribution voting) | 77.14% | 53.8 |
| blood-transfusion-service-center | PFossil:one-level@25 (voting) | 69.12% | 53.8 |
| blood-transfusion-service-center | PFossil:one-level@50 (best rule) | 77.94% | 105.1 |
| blood-transfusion-service-center | PFossil:one-level@50 (macro distribution voting) | 79.40% | 105.1 |
| blood-transfusion-service-center | PFossil:one-level@50 (micro distribution voting) | 78.74% | 105.1 |
| blood-transfusion-service-center | PFossil:one-level@50 (voting) | 68.99% | 105.1 |
| blood-transfusion-service-center | PFossil:single (best rule) | 76.86% | 1.9 |
| blood-transfusion-service-center | PFossil:single (macro distribution voting) | 76.86% | 1.9 |
| blood-transfusion-service-center | PFossil:single (micro distribution voting) | 76.86% | 1.9 |
| blood-transfusion-service-center | PFossil:single (voting) | 76.86% | 1.9 |
| blood-transfusion-service-center | PFossil:two-level@10 (best rule) | 68.46% | 18.9 |
| blood-transfusion-service-center | PFossil:two-level@10 (macro distribution voting) | 68.46% | 18.9 |
| blood-transfusion-service-center | PFossil:two-level@10 (micro distribution voting) | 68.46% | 18.9 |
| blood-transfusion-service-center | PFossil:two-level@10 (voting) | 68.46% | 18.9 |
| blood-transfusion-service-center | PFossil:two-level@25 (best rule) | 70.05% | 53.8 |
| blood-transfusion-service-center | PFossil:two-level@25 (macro distribution voting) | 70.05% | 53.8 |
| blood-transfusion-service-center | PFossil:two-level@25 (micro distribution voting) | 70.05% | 53.8 |
| blood-transfusion-service-center | PFossil:two-level@25 (voting) | 70.05% | 53.8 |
| blood-transfusion-service-center | PFossil:two-level@50 (best rule) | 68.99% | 105.1 |
| blood-transfusion-service-center | PFossil:two-level@50 (macro distribution voting) | 68.99% | 105.1 |
| blood-transfusion-service-center | PFossil:two-level@50 (micro distribution voting) | 68.99% | 105.1 |
| blood-transfusion-service-center | PFossil:two-level@50 (voting) | 68.99% | 105.1 |
| blood-transfusion-service-center | Pypper:one-level@10 (best rule) | 78.34% | 44.2 |
| blood-transfusion-service-center | Pypper:one-level@10 (macro distribution voting) | 78.74% | 44.2 |
| blood-transfusion-service-center | Pypper:one-level@10 (micro distribution voting) | 78.07% | 44.2 |
| blood-transfusion-service-center | Pypper:one-level@10 (voting) | 79.13% | 44.2 |
| blood-transfusion-service-center | Pypper:one-level@25 (best rule) | 77.80% | 111.7 |
| blood-transfusion-service-center | Pypper:one-level@25 (macro distribution voting) | 78.60% | 111.7 |
| blood-transfusion-service-center | Pypper:one-level@25 (micro distribution voting) | 78.07% | 111.7 |
| blood-transfusion-service-center | Pypper:one-level@25 (voting) | 79.27% | 111.7 |
| blood-transfusion-service-center | Pypper:one-level@50 (best rule) | 77.00% | 221.6 |
| blood-transfusion-service-center | Pypper:one-level@50 (macro distribution voting) | 78.60% | 221.6 |
| blood-transfusion-service-center | Pypper:one-level@50 (micro distribution voting) | 78.47% | 221.6 |
| blood-transfusion-service-center | Pypper:one-level@50 (voting) | 79.27% | 221.6 |
| blood-transfusion-service-center | Pypper:single (best rule) | 78.47% | 4.6 |
| blood-transfusion-service-center | Pypper:single (macro distribution voting) | 78.60% | 4.6 |
| blood-transfusion-service-center | Pypper:single (micro distribution voting) | 78.34% | 4.6 |
| blood-transfusion-service-center | Pypper:single (voting) | 78.34% | 4.6 |
| blood-transfusion-service-center | Pypper:two-level@10 (best rule) | 79.13% | 44.2 |
| blood-transfusion-service-center | Pypper:two-level@10 (macro distribution voting) | 79.13% | 44.2 |
| blood-transfusion-service-center | Pypper:two-level@10 (micro distribution voting) | 79.13% | 44.2 |
| blood-transfusion-service-center | Pypper:two-level@10 (voting) | 79.13% | 44.2 |
| blood-transfusion-service-center | Pypper:two-level@25 (best rule) | 79.27% | 111.7 |
| blood-transfusion-service-center | Pypper:two-level@25 (macro distribution voting) | 79.27% | 111.7 |
| blood-transfusion-service-center | Pypper:two-level@25 (micro distribution voting) | 79.27% | 111.7 |
| blood-transfusion-service-center | Pypper:two-level@25 (voting) | 79.27% | 111.7 |
| blood-transfusion-service-center | Pypper:two-level@50 (best rule) | 79.27% | 221.6 |
| blood-transfusion-service-center | Pypper:two-level@50 (macro distribution voting) | 79.27% | 221.6 |
| blood-transfusion-service-center | Pypper:two-level@50 (micro distribution voting) | 79.27% | 221.6 |
| blood-transfusion-service-center | Pypper:two-level@50 (voting) | 79.27% | 221.6 |
| breast-cancer | PFoil:one-level@10 (best rule) | 75.20% | 357.7 |
| breast-cancer | PFoil:one-level@10 (macro distribution voting) | 74.11% | 357.7 |
| breast-cancer | PFoil:one-level@10 (micro distribution voting) | 73.45% | 357.7 |
| breast-cancer | PFoil:one-level@10 (voting) | 74.82% | 357.7 |
| breast-cancer | PFoil:one-level@25 (best rule) | 73.08% | 909.7 |
| breast-cancer | PFoil:one-level@25 (macro distribution voting) | 74.46% | 909.7 |
| breast-cancer | PFoil:one-level@25 (micro distribution voting) | 73.09% | 909.7 |
| breast-cancer | PFoil:one-level@25 (voting) | 73.79% | 909.7 |
| breast-cancer | PFoil:one-level@50 (best rule) | 72.04% | 1824.4 |
| breast-cancer | PFoil:one-level@50 (macro distribution voting) | 74.11% | 1824.4 |
| breast-cancer | PFoil:one-level@50 (micro distribution voting) | 73.45% | 1824.4 |
| breast-cancer | PFoil:one-level@50 (voting) | 74.80% | 1824.4 |
| breast-cancer | PFoil:single (best rule) | 72.02% | 43.2 |
| breast-cancer | PFoil:single (macro distribution voting) | 72.73% | 43.2 |
| breast-cancer | PFoil:single (micro distribution voting) | 72.71% | 43.2 |
| breast-cancer | PFoil:single (voting) | 70.94% | 43.2 |
| breast-cancer | PFoil:two-level@10 (best rule) | 74.47% | 357.7 |
| breast-cancer | PFoil:two-level@10 (macro distribution voting) | 74.13% | 357.7 |
| breast-cancer | PFoil:two-level@10 (micro distribution voting) | 72.38% | 357.7 |
| breast-cancer | PFoil:two-level@10 (voting) | 72.72% | 357.7 |
| breast-cancer | PFoil:two-level@25 (best rule) | 72.72% | 909.7 |
| breast-cancer | PFoil:two-level@25 (macro distribution voting) | 74.11% | 909.7 |
| breast-cancer | PFoil:two-level@25 (micro distribution voting) | 73.45% | 909.7 |
| breast-cancer | PFoil:two-level@25 (voting) | 72.36% | 909.7 |
| breast-cancer | PFoil:two-level@50 (best rule) | 72.73% | 1824.4 |
| breast-cancer | PFoil:two-level@50 (macro distribution voting) | 74.11% | 1824.4 |
| breast-cancer | PFoil:two-level@50 (micro distribution voting) | 74.14% | 1824.4 |
| breast-cancer | PFoil:two-level@50 (voting) | 72.03% | 1824.4 |
| breast-cancer | PFossil:one-level@10 (best rule) | 74.50% | 114.8 |
| breast-cancer | PFossil:one-level@10 (macro distribution voting) | 75.16% | 114.8 |
| breast-cancer | PFossil:one-level@10 (micro distribution voting) | 74.50% | 114.8 |
| breast-cancer | PFossil:one-level@10 (voting) | 73.41% | 114.8 |
| breast-cancer | PFossil:one-level@25 (best rule) | 71.70% | 283.9 |
| breast-cancer | PFossil:one-level@25 (macro distribution voting) | 75.50% | 283.9 |
| breast-cancer | PFossil:one-level@25 (micro distribution voting) | 73.44% | 283.9 |
| breast-cancer | PFossil:one-level@25 (voting) | 73.76% | 283.9 |
| breast-cancer | PFossil:one-level@50 (best rule) | 72.35% | 590.1 |
| breast-cancer | PFossil:one-level@50 (macro distribution voting) | 75.52% | 590.1 |
| breast-cancer | PFossil:one-level@50 (micro distribution voting) | 73.78% | 590.1 |
| breast-cancer | PFossil:one-level@50 (voting) | 73.77% | 590.1 |
| breast-cancer | PFossil:single (best rule) | 69.26% | 7.4 |
| breast-cancer | PFossil:single (macro distribution voting) | 69.98% | 7.4 |
| breast-cancer | PFossil:single (micro distribution voting) | 68.56% | 7.4 |
| breast-cancer | PFossil:single (voting) | 68.92% | 7.4 |
| breast-cancer | PFossil:two-level@10 (best rule) | 75.16% | 114.8 |
| breast-cancer | PFossil:two-level@10 (macro distribution voting) | 74.47% | 114.8 |
| breast-cancer | PFossil:two-level@10 (micro distribution voting) | 74.82% | 114.8 |
| breast-cancer | PFossil:two-level@10 (voting) | 74.14% | 114.8 |
| breast-cancer | PFossil:two-level@25 (best rule) | 74.11% | 283.9 |
| breast-cancer | PFossil:two-level@25 (macro distribution voting) | 75.16% | 283.9 |
| breast-cancer | PFossil:two-level@25 (micro distribution voting) | 74.11% | 283.9 |
| breast-cancer | PFossil:two-level@25 (voting) | 73.77% | 283.9 |
| breast-cancer | PFossil:two-level@50 (best rule) | 74.48% | 590.1 |
| breast-cancer | PFossil:two-level@50 (macro distribution voting) | 74.47% | 590.1 |
| breast-cancer | PFossil:two-level@50 (micro distribution voting) | 74.48% | 590.1 |
| breast-cancer | PFossil:two-level@50 (voting) | 74.11% | 590.1 |
| breast-cancer | Pypper:one-level@10 (best rule) | 70.94% | 47.0 |
| breast-cancer | Pypper:one-level@10 (macro distribution voting) | 74.45% | 47.0 |
| breast-cancer | Pypper:one-level@10 (micro distribution voting) | 73.77% | 47.0 |
| breast-cancer | Pypper:one-level@10 (voting) | 73.40% | 47.0 |
| breast-cancer | Pypper:one-level@25 (best rule) | 72.33% | 118.4 |
| breast-cancer | Pypper:one-level@25 (macro distribution voting) | 74.11% | 118.4 |
| breast-cancer | Pypper:one-level@25 (micro distribution voting) | 72.75% | 118.4 |
| breast-cancer | Pypper:one-level@25 (voting) | 74.13% | 118.4 |
| breast-cancer | Pypper:one-level@50 (best rule) | 71.65% | 235.3 |
| breast-cancer | Pypper:one-level@50 (macro distribution voting) | 75.17% | 235.3 |
| breast-cancer | Pypper:one-level@50 (micro distribution voting) | 73.08% | 235.3 |
| breast-cancer | Pypper:one-level@50 (voting) | 74.14% | 235.3 |
| breast-cancer | Pypper:single (best rule) | 74.84% | 2.4 |
| breast-cancer | Pypper:single (macro distribution voting) | 74.84% | 2.4 |
| breast-cancer | Pypper:single (micro distribution voting) | 74.84% | 2.4 |
| breast-cancer | Pypper:single (voting) | 74.84% | 2.4 |
| breast-cancer | Pypper:two-level@10 (best rule) | 72.35% | 47.0 |
| breast-cancer | Pypper:two-level@10 (macro distribution voting) | 72.35% | 47.0 |
| breast-cancer | Pypper:two-level@10 (micro distribution voting) | 73.08% | 47.0 |
| breast-cancer | Pypper:two-level@10 (voting) | 72.72% | 47.0 |
| breast-cancer | Pypper:two-level@25 (best rule) | 74.11% | 118.4 |
| breast-cancer | Pypper:two-level@25 (macro distribution voting) | 74.11% | 118.4 |
| breast-cancer | Pypper:two-level@25 (micro distribution voting) | 73.44% | 118.4 |
| breast-cancer | Pypper:two-level@25 (voting) | 73.44% | 118.4 |
| breast-cancer | Pypper:two-level@50 (best rule) | 74.83% | 235.3 |
| breast-cancer | Pypper:two-level@50 (macro distribution voting) | 74.48% | 235.3 |
| breast-cancer | Pypper:two-level@50 (micro distribution voting) | 74.48% | 235.3 |
| breast-cancer | Pypper:two-level@50 (voting) | 74.83% | 235.3 |
| breast-w | PFoil:one-level@10 (best rule) | 91.56% | 144.6 |
| breast-w | PFoil:one-level@10 (macro distribution voting) | 96.28% | 144.6 |
| breast-w | PFoil:one-level@10 (micro distribution voting) | 94.71% | 144.6 |
| breast-w | PFoil:one-level@10 (voting) | 96.14% | 144.6 |
| breast-w | PFoil:one-level@25 (best rule) | 91.13% | 365.0 |
| breast-w | PFoil:one-level@25 (macro distribution voting) | 96.71% | 365.0 |
| breast-w | PFoil:one-level@25 (micro distribution voting) | 94.71% | 365.0 |
| breast-w | PFoil:one-level@25 (voting) | 96.57% | 365.0 |
| breast-w | PFoil:one-level@50 (best rule) | 90.42% | 734.6 |
| breast-w | PFoil:one-level@50 (macro distribution voting) | 96.42% | 734.6 |
| breast-w | PFoil:one-level@50 (micro distribution voting) | 95.14% | 734.6 |
| breast-w | PFoil:one-level@50 (voting) | 96.57% | 734.6 |
| breast-w | PFoil:single (best rule) | 95.85% | 17.1 |
| breast-w | PFoil:single (macro distribution voting) | 95.99% | 17.1 |
| breast-w | PFoil:single (micro distribution voting) | 95.42% | 17.1 |
| breast-w | PFoil:single (voting) | 94.70% | 17.1 |
| breast-w | PFoil:two-level@10 (best rule) | 96.14% | 144.6 |
| breast-w | PFoil:two-level@10 (macro distribution voting) | 96.14% | 144.6 |
| breast-w | PFoil:two-level@10 (micro distribution voting) | 95.85% | 144.6 |
| breast-w | PFoil:two-level@10 (voting) | 95.71% | 144.6 |
| breast-w | PFoil:two-level@25 (best rule) | 95.99% | 365.0 |
| breast-w | PFoil:two-level@25 (macro distribution voting) | 96.14% | 365.0 |
| breast-w | PFoil:two-level@25 (micro distribution voting) | 95.42% | 365.0 |
| breast-w | PFoil:two-level@25 (voting) | 95.85% | 365.0 |
| breast-w | PFoil:two-level@50 (best rule) | 96.14% | 734.6 |
| breast-w | PFoil:two-level@50 (macro distribution voting) | 96.42% | 734.6 |
| breast-w | PFoil:two-level@50 (micro distribution voting) | 95.99% | 734.6 |
| breast-w | PFoil:two-level@50 (voting) | 96.00% | 734.6 |
| breast-w | PFossil:one-level@10 (best rule) | 93.42% | 120.0 |
| breast-w | PFossil:one-level@10 (macro distribution voting) | 96.14% | 120.0 |
| breast-w | PFossil:one-level@10 (micro distribution voting) | 95.56% | 120.0 |
| breast-w | PFossil:one-level@10 (voting) | 96.14% | 120.0 |
| breast-w | PFossil:one-level@25 (best rule) | 92.28% | 303.2 |
| breast-w | PFossil:one-level@25 (macro distribution voting) | 96.14% | 303.2 |
| breast-w | PFossil:one-level@25 (micro distribution voting) | 95.56% | 303.2 |
| breast-w | PFossil:one-level@25 (voting) | 96.42% | 303.2 |
| breast-w | PFossil:one-level@50 (best rule) | 91.42% | 608.5 |
| breast-w | PFossil:one-level@50 (macro distribution voting) | 96.42% | 608.5 |
| breast-w | PFossil:one-level@50 (micro distribution voting) | 95.71% | 608.5 |
| breast-w | PFossil:one-level@50 (voting) | 96.28% | 608.5 |
| breast-w | PFossil:single (best rule) | 94.13% | 15.6 |
| breast-w | PFossil:single (macro distribution voting) | 94.99% | 15.6 |
| breast-w | PFossil:single (micro distribution voting) | 94.27% | 15.6 |
| breast-w | PFossil:single (voting) | 94.99% | 15.6 |
| breast-w | PFossil:two-level@10 (best rule) | 95.56% | 120.0 |
| breast-w | PFossil:two-level@10 (macro distribution voting) | 95.42% | 120.0 |
| breast-w | PFossil:two-level@10 (micro distribution voting) | 95.71% | 120.0 |
| breast-w | PFossil:two-level@10 (voting) | 95.56% | 120.0 |
| breast-w | PFossil:two-level@25 (best rule) | 95.56% | 303.2 |
| breast-w | PFossil:two-level@25 (macro distribution voting) | 95.99% | 303.2 |
| breast-w | PFossil:two-level@25 (micro distribution voting) | 95.85% | 303.2 |
| breast-w | PFossil:two-level@25 (voting) | 95.99% | 303.2 |
| breast-w | PFossil:two-level@50 (best rule) | 95.85% | 608.5 |
| breast-w | PFossil:two-level@50 (macro distribution voting) | 95.99% | 608.5 |
| breast-w | PFossil:two-level@50 (micro distribution voting) | 95.85% | 608.5 |
| breast-w | PFossil:two-level@50 (voting) | 95.99% | 608.5 |
| breast-w | Pypper:one-level@10 (best rule) | 93.85% | 68.5 |
| breast-w | Pypper:one-level@10 (macro distribution voting) | 96.85% | 68.5 |
| breast-w | Pypper:one-level@10 (micro distribution voting) | 95.85% | 68.5 |
| breast-w | Pypper:one-level@10 (voting) | 96.85% | 68.5 |
| breast-w | Pypper:one-level@25 (best rule) | 92.84% | 170.7 |
| breast-w | Pypper:one-level@25 (macro distribution voting) | 96.57% | 170.7 |
| breast-w | Pypper:one-level@25 (micro distribution voting) | 95.28% | 170.7 |
| breast-w | Pypper:one-level@25 (voting) | 96.57% | 170.7 |
| breast-w | Pypper:one-level@50 (best rule) | 91.56% | 343.2 |
| breast-w | Pypper:one-level@50 (macro distribution voting) | 96.57% | 343.2 |
| breast-w | Pypper:one-level@50 (micro distribution voting) | 95.42% | 343.2 |
| breast-w | Pypper:one-level@50 (voting) | 96.57% | 343.2 |
| breast-w | Pypper:single (best rule) | 94.57% | 6.4 |
| breast-w | Pypper:single (macro distribution voting) | 95.00% | 6.4 |
| breast-w | Pypper:single (micro distribution voting) | 94.71% | 6.4 |
| breast-w | Pypper:single (voting) | 95.00% | 6.4 |
| breast-w | Pypper:two-level@10 (best rule) | 96.42% | 68.5 |
| breast-w | Pypper:two-level@10 (macro distribution voting) | 96.42% | 68.5 |
| breast-w | Pypper:two-level@10 (micro distribution voting) | 96.57% | 68.5 |
| breast-w | Pypper:two-level@10 (voting) | 96.28% | 68.5 |
| breast-w | Pypper:two-level@25 (best rule) | 95.71% | 170.7 |
| breast-w | Pypper:two-level@25 (macro distribution voting) | 96.42% | 170.7 |
| breast-w | Pypper:two-level@25 (micro distribution voting) | 95.56% | 170.7 |
| breast-w | Pypper:two-level@25 (voting) | 96.28% | 170.7 |
| breast-w | Pypper:two-level@50 (best rule) | 95.99% | 343.2 |
| breast-w | Pypper:two-level@50 (macro distribution voting) | 96.28% | 343.2 |
| breast-w | Pypper:two-level@50 (micro distribution voting) | 95.71% | 343.2 |
| breast-w | Pypper:two-level@50 (voting) | 96.28% | 343.2 |
| car | PFoil:one-level@10 (best rule) | 95.49% | 682.0 |
| car | PFoil:one-level@10 (macro distribution voting) | 96.87% | 682.0 |
| car | PFoil:one-level@10 (micro distribution voting) | 94.85% | 682.0 |
| car | PFoil:one-level@10 (voting) | 96.53% | 682.0 |
| car | PFoil:one-level@25 (best rule) | 94.68% | 1722.6 |
| car | PFoil:one-level@25 (macro distribution voting) | 96.87% | 1722.6 |
| car | PFoil:one-level@25 (micro distribution voting) | 93.75% | 1722.6 |
| car | PFoil:one-level@25 (voting) | 96.35% | 1722.6 |
| car | PFoil:one-level@50 (best rule) | 94.50% | 3447.5 |
| car | PFoil:one-level@50 (macro distribution voting) | 96.93% | 3447.5 |
| car | PFoil:one-level@50 (micro distribution voting) | 93.29% | 3447.5 |
| car | PFoil:one-level@50 (voting) | 96.24% | 3447.5 |
| car | PFoil:single (best rule) | 96.70% | 75.4 |
| car | PFoil:single (macro distribution voting) | 96.35% | 75.4 |
| car | PFoil:single (micro distribution voting) | 95.60% | 75.4 |
| car | PFoil:single (voting) | 94.73% | 75.4 |
| car | PFoil:two-level@10 (best rule) | 97.22% | 682.0 |
| car | PFoil:two-level@10 (macro distribution voting) | 97.28% | 682.0 |
| car | PFoil:two-level@10 (micro distribution voting) | 96.41% | 682.0 |
| car | PFoil:two-level@10 (voting) | 96.12% | 682.0 |
| car | PFoil:two-level@25 (best rule) | 97.28% | 1722.6 |
| car | PFoil:two-level@25 (macro distribution voting) | 97.16% | 1722.6 |
| car | PFoil:two-level@25 (micro distribution voting) | 96.47% | 1722.6 |
| car | PFoil:two-level@25 (voting) | 96.41% | 1722.6 |
| car | PFoil:two-level@50 (best rule) | 97.34% | 3447.5 |
| car | PFoil:two-level@50 (macro distribution voting) | 97.22% | 3447.5 |
| car | PFoil:two-level@50 (micro distribution voting) | 96.93% | 3447.5 |
| car | PFoil:two-level@50 (voting) | 96.76% | 3447.5 |
| car | PFossil:one-level@10 (best rule) | 91.49% | 182.2 |
| car | PFossil:one-level@10 (macro distribution voting) | 93.46% | 182.2 |
| car | PFossil:one-level@10 (micro distribution voting) | 86.05% | 182.2 |
| car | PFossil:one-level@10 (voting) | 91.72% | 182.2 |
| car | PFossil:one-level@25 (best rule) | 89.01% | 462.1 |
| car | PFossil:one-level@25 (macro distribution voting) | 92.77% | 462.1 |
| car | PFossil:one-level@25 (micro distribution voting) | 87.09% | 462.1 |
| car | PFossil:one-level@25 (voting) | 91.49% | 462.1 |
| car | PFossil:one-level@50 (best rule) | 87.04% | 917.2 |
| car | PFossil:one-level@50 (macro distribution voting) | 92.48% | 917.2 |
| car | PFossil:one-level@50 (micro distribution voting) | 87.09% | 917.2 |
| car | PFossil:one-level@50 (voting) | 91.95% | 917.2 |
| car | PFossil:single (best rule) | 93.87% | 16.2 |
| car | PFossil:single (macro distribution voting) | 93.87% | 16.2 |
| car | PFossil:single (micro distribution voting) | 84.90% | 16.2 |
| car | PFossil:single (voting) | 90.51% | 16.2 |
| car | PFossil:two-level@10 (best rule) | 94.39% | 182.2 |
| car | PFossil:two-level@10 (macro distribution voting) | 93.75% | 182.2 |
| car | PFossil:two-level@10 (micro distribution voting) | 85.47% | 182.2 |
| car | PFossil:two-level@10 (voting) | 90.68% | 182.2 |
| car | PFossil:two-level@25 (best rule) | 95.08% | 462.1 |
| car | PFossil:two-level@25 (macro distribution voting) | 94.16% | 462.1 |
| car | PFossil:two-level@25 (micro distribution voting) | 86.17% | 462.1 |
| car | PFossil:two-level@25 (voting) | 90.86% | 462.1 |
| car | PFossil:two-level@50 (best rule) | 95.08% | 917.2 |
| car | PFossil:two-level@50 (macro distribution voting) | 94.16% | 917.2 |
| car | PFossil:two-level@50 (micro distribution voting) | 85.59% | 917.2 |
| car | PFossil:two-level@50 (voting) | 90.80% | 917.2 |
| car | Pypper:one-level@10 (best rule) | 94.79% | 250.3 |
| car | Pypper:one-level@10 (macro distribution voting) | 95.83% | 250.3 |
| car | Pypper:one-level@10 (micro distribution voting) | 91.84% | 250.3 |
| car | Pypper:one-level@10 (voting) | 94.44% | 250.3 |
| car | Pypper:one-level@25 (best rule) | 95.20% | 634.2 |
| car | Pypper:one-level@25 (macro distribution voting) | 95.77% | 634.2 |
| car | Pypper:one-level@25 (micro distribution voting) | 91.32% | 634.2 |
| car | Pypper:one-level@25 (voting) | 95.20% | 634.2 |
| car | Pypper:one-level@50 (best rule) | 94.50% | 1255.0 |
| car | Pypper:one-level@50 (macro distribution voting) | 96.12% | 1255.0 |
| car | Pypper:one-level@50 (micro distribution voting) | 91.20% | 1255.0 |
| car | Pypper:one-level@50 (voting) | 95.08% | 1255.0 |
| car | Pypper:single (best rule) | 92.36% | 29.7 |
| car | Pypper:single (macro distribution voting) | 92.24% | 29.7 |
| car | Pypper:single (micro distribution voting) | 90.91% | 29.7 |
| car | Pypper:single (voting) | 91.49% | 29.7 |
| car | Pypper:two-level@10 (best rule) | 95.66% | 250.3 |
| car | Pypper:two-level@10 (macro distribution voting) | 95.26% | 250.3 |
| car | Pypper:two-level@10 (micro distribution voting) | 93.57% | 250.3 |
| car | Pypper:two-level@10 (voting) | 94.73% | 250.3 |
| car | Pypper:two-level@25 (best rule) | 96.24% | 634.2 |
| car | Pypper:two-level@25 (macro distribution voting) | 96.06% | 634.2 |
| car | Pypper:two-level@25 (micro distribution voting) | 93.92% | 634.2 |
| car | Pypper:two-level@25 (voting) | 95.43% | 634.2 |
| car | Pypper:two-level@50 (best rule) | 96.47% | 1255.0 |
| car | Pypper:two-level@50 (macro distribution voting) | 96.35% | 1255.0 |
| car | Pypper:two-level@50 (micro distribution voting) | 94.62% | 1255.0 |
| car | Pypper:two-level@50 (voting) | 95.54% | 1255.0 |
| churn | PFoil:one-level@10 (best rule) | 90.88% | 1440.8 |
| churn | PFoil:one-level@10 (macro distribution voting) | 92.62% | 1440.8 |
| churn | PFoil:one-level@10 (micro distribution voting) | 86.20% | 1440.8 |
| churn | PFoil:one-level@10 (voting) | 93.50% | 1440.8 |
| churn | PFoil:one-level@25 (best rule) | 89.92% | 3601.5 |
| churn | PFoil:one-level@25 (macro distribution voting) | 92.66% | 3601.5 |
| churn | PFoil:one-level@25 (micro distribution voting) | 85.98% | 3601.5 |
| churn | PFoil:one-level@25 (voting) | 93.96% | 3601.5 |
| churn | PFoil:one-level@50 (best rule) | 89.28% | 7208.5 |
| churn | PFoil:one-level@50 (macro distribution voting) | 93.00% | 7208.5 |
| churn | PFoil:one-level@50 (micro distribution voting) | 85.88% | 7208.5 |
| churn | PFoil:one-level@50 (voting) | 94.02% | 7208.5 |
| churn | PFoil:single (best rule) | 92.36% | 178.2 |
| churn | PFoil:single (macro distribution voting) | 91.58% | 178.2 |
| churn | PFoil:single (micro distribution voting) | 89.94% | 178.2 |
| churn | PFoil:single (voting) | 89.86% | 178.2 |
| churn | PFoil:two-level@10 (best rule) | 93.28% | 1440.8 |
| churn | PFoil:two-level@10 (macro distribution voting) | 92.98% | 1440.8 |
| churn | PFoil:two-level@10 (micro distribution voting) | 91.32% | 1440.8 |
| churn | PFoil:two-level@10 (voting) | 93.34% | 1440.8 |
| churn | PFoil:two-level@25 (best rule) | 93.72% | 3601.5 |
| churn | PFoil:two-level@25 (macro distribution voting) | 93.40% | 3601.5 |
| churn | PFoil:two-level@25 (micro distribution voting) | 92.08% | 3601.5 |
| churn | PFoil:two-level@25 (voting) | 93.76% | 3601.5 |
| churn | PFoil:two-level@50 (best rule) | 93.62% | 7208.5 |
| churn | PFoil:two-level@50 (macro distribution voting) | 93.52% | 7208.5 |
| churn | PFoil:two-level@50 (micro distribution voting) | 92.20% | 7208.5 |
| churn | PFoil:two-level@50 (voting) | 93.88% | 7208.5 |
| churn | PFossil:one-level@10 (best rule) | 91.00% | 104.3 |
| churn | PFossil:one-level@10 (macro distribution voting) | 91.22% | 104.3 |
| churn | PFossil:one-level@10 (micro distribution voting) | 88.52% | 104.3 |
| churn | PFossil:one-level@10 (voting) | 91.10% | 104.3 |
| churn | PFossil:one-level@25 (best rule) | 91.70% | 266.2 |
| churn | PFossil:one-level@25 (macro distribution voting) | 91.82% | 266.2 |
| churn | PFossil:one-level@25 (micro distribution voting) | 88.94% | 266.2 |
| churn | PFossil:one-level@25 (voting) | 91.68% | 266.2 |
| churn | PFossil:one-level@50 (best rule) | 92.18% | 557.7 |
| churn | PFossil:one-level@50 (macro distribution voting) | 91.88% | 557.7 |
| churn | PFossil:one-level@50 (micro distribution voting) | 88.54% | 557.7 |
| churn | PFossil:one-level@50 (voting) | 91.74% | 557.7 |
| churn | PFossil:single (best rule) | 90.00% | 9.2 |
| churn | PFossil:single (macro distribution voting) | 89.98% | 9.2 |
| churn | PFossil:single (micro distribution voting) | 89.18% | 9.2 |
| churn | PFossil:single (voting) | 89.00% | 9.2 |
| churn | PFossil:two-level@10 (best rule) | 91.12% | 104.3 |
| churn | PFossil:two-level@10 (macro distribution voting) | 91.16% | 104.3 |
| churn | PFossil:two-level@10 (micro distribution voting) | 90.28% | 104.3 |
| churn | PFossil:two-level@10 (voting) | 90.20% | 104.3 |
| churn | PFossil:two-level@25 (best rule) | 91.60% | 266.2 |
| churn | PFossil:two-level@25 (macro distribution voting) | 91.64% | 266.2 |
| churn | PFossil:two-level@25 (micro distribution voting) | 90.88% | 266.2 |
| churn | PFossil:two-level@25 (voting) | 90.84% | 266.2 |
| churn | PFossil:two-level@50 (best rule) | 91.70% | 557.7 |
| churn | PFossil:two-level@50 (macro distribution voting) | 91.74% | 557.7 |
| churn | PFossil:two-level@50 (micro distribution voting) | 90.68% | 557.7 |
| churn | PFossil:two-level@50 (voting) | 90.76% | 557.7 |
| churn | Pypper:one-level@10 (best rule) | 92.94% | 187.2 |
| churn | Pypper:one-level@10 (macro distribution voting) | 93.62% | 187.2 |
| churn | Pypper:one-level@10 (micro distribution voting) | 92.22% | 187.2 |
| churn | Pypper:one-level@10 (voting) | 93.80% | 187.2 |
| churn | Pypper:one-level@25 (best rule) | 92.74% | 468.5 |
| churn | Pypper:one-level@25 (macro distribution voting) | 93.76% | 468.5 |
| churn | Pypper:one-level@25 (micro distribution voting) | 91.08% | 468.5 |
| churn | Pypper:one-level@25 (voting) | 93.82% | 468.5 |
| churn | Pypper:one-level@50 (best rule) | 93.10% | 927.0 |
| churn | Pypper:one-level@50 (macro distribution voting) | 93.72% | 927.0 |
| churn | Pypper:one-level@50 (micro distribution voting) | 91.14% | 927.0 |
| churn | Pypper:one-level@50 (voting) | 93.68% | 927.0 |
| churn | Pypper:single (best rule) | 93.66% | 18.2 |
| churn | Pypper:single (macro distribution voting) | 93.66% | 18.2 |
| churn | Pypper:single (micro distribution voting) | 93.52% | 18.2 |
| churn | Pypper:single (voting) | 93.46% | 18.2 |
| churn | Pypper:two-level@10 (best rule) | 93.72% | 187.2 |
| churn | Pypper:two-level@10 (macro distribution voting) | 93.74% | 187.2 |
| churn | Pypper:two-level@10 (micro distribution voting) | 93.70% | 187.2 |
| churn | Pypper:two-level@10 (voting) | 93.58% | 187.2 |
| churn | Pypper:two-level@25 (best rule) | 93.70% | 468.5 |
| churn | Pypper:two-level@25 (macro distribution voting) | 93.70% | 468.5 |
| churn | Pypper:two-level@25 (micro distribution voting) | 93.64% | 468.5 |
| churn | Pypper:two-level@25 (voting) | 93.66% | 468.5 |
| churn | Pypper:two-level@50 (best rule) | 93.72% | 927.0 |
| churn | Pypper:two-level@50 (macro distribution voting) | 93.76% | 927.0 |
| churn | Pypper:two-level@50 (micro distribution voting) | 93.70% | 927.0 |
| churn | Pypper:two-level@50 (voting) | 93.64% | 927.0 |
| climate-model-simulation-crashes | PFoil:one-level@10 (best rule) | 91.85% | 153.7 |
| climate-model-simulation-crashes | PFoil:one-level@10 (macro distribution voting) | 93.52% | 153.7 |
| climate-model-simulation-crashes | PFoil:one-level@10 (micro distribution voting) | 91.85% | 153.7 |
| climate-model-simulation-crashes | PFoil:one-level@10 (voting) | 94.07% | 153.7 |
| climate-model-simulation-crashes | PFoil:one-level@25 (best rule) | 91.48% | 384.5 |
| climate-model-simulation-crashes | PFoil:one-level@25 (macro distribution voting) | 93.70% | 384.5 |
| climate-model-simulation-crashes | PFoil:one-level@25 (micro distribution voting) | 91.67% | 384.5 |
| climate-model-simulation-crashes | PFoil:one-level@25 (voting) | 93.52% | 384.5 |
| climate-model-simulation-crashes | PFoil:one-level@50 (best rule) | 91.48% | 774.0 |
| climate-model-simulation-crashes | PFoil:one-level@50 (macro distribution voting) | 93.70% | 774.0 |
| climate-model-simulation-crashes | PFoil:one-level@50 (micro distribution voting) | 91.67% | 774.0 |
| climate-model-simulation-crashes | PFoil:one-level@50 (voting) | 93.70% | 774.0 |
| climate-model-simulation-crashes | PFoil:single (best rule) | 93.33% | 19.2 |
| climate-model-simulation-crashes | PFoil:single (macro distribution voting) | 93.89% | 19.2 |
| climate-model-simulation-crashes | PFoil:single (micro distribution voting) | 93.33% | 19.2 |
| climate-model-simulation-crashes | PFoil:single (voting) | 93.52% | 19.2 |
| climate-model-simulation-crashes | PFoil:two-level@10 (best rule) | 93.15% | 153.7 |
| climate-model-simulation-crashes | PFoil:two-level@10 (macro distribution voting) | 93.70% | 153.7 |
| climate-model-simulation-crashes | PFoil:two-level@10 (micro distribution voting) | 92.96% | 153.7 |
| climate-model-simulation-crashes | PFoil:two-level@10 (voting) | 93.33% | 153.7 |
| climate-model-simulation-crashes | PFoil:two-level@25 (best rule) | 93.15% | 384.5 |
| climate-model-simulation-crashes | PFoil:two-level@25 (macro distribution voting) | 93.70% | 384.5 |
| climate-model-simulation-crashes | PFoil:two-level@25 (micro distribution voting) | 92.78% | 384.5 |
| climate-model-simulation-crashes | PFoil:two-level@25 (voting) | 93.33% | 384.5 |
| climate-model-simulation-crashes | PFoil:two-level@50 (best rule) | 93.15% | 774.0 |
| climate-model-simulation-crashes | PFoil:two-level@50 (macro distribution voting) | 94.07% | 774.0 |
| climate-model-simulation-crashes | PFoil:two-level@50 (micro distribution voting) | 92.96% | 774.0 |
| climate-model-simulation-crashes | PFoil:two-level@50 (voting) | 93.33% | 774.0 |
| climate-model-simulation-crashes | PFossil:one-level@10 (best rule) | 91.67% | 110.2 |
| climate-model-simulation-crashes | PFossil:one-level@10 (macro distribution voting) | 93.52% | 110.2 |
| climate-model-simulation-crashes | PFossil:one-level@10 (micro distribution voting) | 91.48% | 110.2 |
| climate-model-simulation-crashes | PFossil:one-level@10 (voting) | 93.15% | 110.2 |
| climate-model-simulation-crashes | PFossil:one-level@25 (best rule) | 91.48% | 272.6 |
| climate-model-simulation-crashes | PFossil:one-level@25 (macro distribution voting) | 93.52% | 272.6 |
| climate-model-simulation-crashes | PFossil:one-level@25 (micro distribution voting) | 91.48% | 272.6 |
| climate-model-simulation-crashes | PFossil:one-level@25 (voting) | 93.52% | 272.6 |
| climate-model-simulation-crashes | PFossil:one-level@50 (best rule) | 91.48% | 555.6 |
| climate-model-simulation-crashes | PFossil:one-level@50 (macro distribution voting) | 93.33% | 555.6 |
| climate-model-simulation-crashes | PFossil:one-level@50 (micro distribution voting) | 91.48% | 555.6 |
| climate-model-simulation-crashes | PFossil:one-level@50 (voting) | 93.33% | 555.6 |
| climate-model-simulation-crashes | PFossil:single (best rule) | 92.78% | 8.9 |
| climate-model-simulation-crashes | PFossil:single (macro distribution voting) | 92.96% | 8.9 |
| climate-model-simulation-crashes | PFossil:single (micro distribution voting) | 92.78% | 8.9 |
| climate-model-simulation-crashes | PFossil:single (voting) | 92.78% | 8.9 |
| climate-model-simulation-crashes | PFossil:two-level@10 (best rule) | 92.41% | 110.2 |
| climate-model-simulation-crashes | PFossil:two-level@10 (macro distribution voting) | 93.52% | 110.2 |
| climate-model-simulation-crashes | PFossil:two-level@10 (micro distribution voting) | 92.41% | 110.2 |
| climate-model-simulation-crashes | PFossil:two-level@10 (voting) | 92.41% | 110.2 |
| climate-model-simulation-crashes | PFossil:two-level@25 (best rule) | 92.96% | 272.6 |
| climate-model-simulation-crashes | PFossil:two-level@25 (macro distribution voting) | 93.52% | 272.6 |
| climate-model-simulation-crashes | PFossil:two-level@25 (micro distribution voting) | 92.78% | 272.6 |
| climate-model-simulation-crashes | PFossil:two-level@25 (voting) | 92.78% | 272.6 |
| climate-model-simulation-crashes | PFossil:two-level@50 (best rule) | 92.78% | 555.6 |
| climate-model-simulation-crashes | PFossil:two-level@50 (macro distribution voting) | 93.33% | 555.6 |
| climate-model-simulation-crashes | PFossil:two-level@50 (micro distribution voting) | 92.59% | 555.6 |
| climate-model-simulation-crashes | PFossil:two-level@50 (voting) | 92.78% | 555.6 |
| climate-model-simulation-crashes | Pypper:one-level@10 (best rule) | 92.96% | 66.9 |
| climate-model-simulation-crashes | Pypper:one-level@10 (macro distribution voting) | 94.81% | 66.9 |
| climate-model-simulation-crashes | Pypper:one-level@10 (micro distribution voting) | 93.33% | 66.9 |
| climate-model-simulation-crashes | Pypper:one-level@10 (voting) | 94.81% | 66.9 |
| climate-model-simulation-crashes | Pypper:one-level@25 (best rule) | 92.22% | 166.9 |
| climate-model-simulation-crashes | Pypper:one-level@25 (macro distribution voting) | 95.00% | 166.9 |
| climate-model-simulation-crashes | Pypper:one-level@25 (micro distribution voting) | 92.78% | 166.9 |
| climate-model-simulation-crashes | Pypper:one-level@25 (voting) | 95.00% | 166.9 |
| climate-model-simulation-crashes | Pypper:one-level@50 (best rule) | 91.67% | 334.2 |
| climate-model-simulation-crashes | Pypper:one-level@50 (macro distribution voting) | 95.19% | 334.2 |
| climate-model-simulation-crashes | Pypper:one-level@50 (micro distribution voting) | 92.96% | 334.2 |
| climate-model-simulation-crashes | Pypper:one-level@50 (voting) | 95.00% | 334.2 |
| climate-model-simulation-crashes | Pypper:single (best rule) | 95.19% | 6.6 |
| climate-model-simulation-crashes | Pypper:single (macro distribution voting) | 94.81% | 6.6 |
| climate-model-simulation-crashes | Pypper:single (micro distribution voting) | 95.19% | 6.6 |
| climate-model-simulation-crashes | Pypper:single (voting) | 95.19% | 6.6 |
| climate-model-simulation-crashes | Pypper:two-level@10 (best rule) | 94.07% | 66.9 |
| climate-model-simulation-crashes | Pypper:two-level@10 (macro distribution voting) | 93.89% | 66.9 |
| climate-model-simulation-crashes | Pypper:two-level@10 (micro distribution voting) | 94.07% | 66.9 |
| climate-model-simulation-crashes | Pypper:two-level@10 (voting) | 94.07% | 66.9 |
| climate-model-simulation-crashes | Pypper:two-level@25 (best rule) | 94.63% | 166.9 |
| climate-model-simulation-crashes | Pypper:two-level@25 (macro distribution voting) | 94.63% | 166.9 |
| climate-model-simulation-crashes | Pypper:two-level@25 (micro distribution voting) | 94.63% | 166.9 |
| climate-model-simulation-crashes | Pypper:two-level@25 (voting) | 94.81% | 166.9 |
| climate-model-simulation-crashes | Pypper:two-level@50 (best rule) | 95.00% | 334.2 |
| climate-model-simulation-crashes | Pypper:two-level@50 (macro distribution voting) | 95.37% | 334.2 |
| climate-model-simulation-crashes | Pypper:two-level@50 (micro distribution voting) | 95.00% | 334.2 |
| climate-model-simulation-crashes | Pypper:two-level@50 (voting) | 95.00% | 334.2 |
| cmc | PFoil:one-level@10 (best rule) | 53.64% | 2078.5 |
| cmc | PFoil:one-level@10 (macro distribution voting) | 53.78% | 2078.5 |
| cmc | PFoil:one-level@10 (micro distribution voting) | 53.10% | 2078.5 |
| cmc | PFoil:one-level@10 (voting) | 51.94% | 2078.5 |
| cmc | PFoil:one-level@25 (best rule) | 53.51% | 5179.2 |
| cmc | PFoil:one-level@25 (macro distribution voting) | 53.78% | 5179.2 |
| cmc | PFoil:one-level@25 (micro distribution voting) | 53.71% | 5179.2 |
| cmc | PFoil:one-level@25 (voting) | 52.76% | 5179.2 |
| cmc | PFoil:one-level@50 (best rule) | 53.51% | 10326.3 |
| cmc | PFoil:one-level@50 (macro distribution voting) | 53.57% | 10326.3 |
| cmc | PFoil:one-level@50 (micro distribution voting) | 54.05% | 10326.3 |
| cmc | PFoil:one-level@50 (voting) | 53.03% | 10326.3 |
| cmc | PFoil:single (best rule) | 51.40% | 211.0 |
| cmc | PFoil:single (macro distribution voting) | 51.40% | 211.0 |
| cmc | PFoil:single (micro distribution voting) | 52.96% | 211.0 |
| cmc | PFoil:single (voting) | 50.31% | 211.0 |
| cmc | PFoil:two-level@10 (best rule) | 52.42% | 2078.5 |
| cmc | PFoil:two-level@10 (macro distribution voting) | 52.49% | 2078.5 |
| cmc | PFoil:two-level@10 (micro distribution voting) | 52.35% | 2078.5 |
| cmc | PFoil:two-level@10 (voting) | 51.40% | 2078.5 |
| cmc | PFoil:two-level@25 (best rule) | 53.37% | 5179.2 |
| cmc | PFoil:two-level@25 (macro distribution voting) | 53.37% | 5179.2 |
| cmc | PFoil:two-level@25 (micro distribution voting) | 53.57% | 5179.2 |
| cmc | PFoil:two-level@25 (voting) | 51.06% | 5179.2 |
| cmc | PFoil:two-level@50 (best rule) | 53.17% | 10326.3 |
| cmc | PFoil:two-level@50 (macro distribution voting) | 52.83% | 10326.3 |
| cmc | PFoil:two-level@50 (micro distribution voting) | 53.78% | 10326.3 |
| cmc | PFoil:two-level@50 (voting) | 51.81% | 10326.3 |
| cmc | PFossil:one-level@10 (best rule) | 53.02% | 40.9 |
| cmc | PFossil:one-level@10 (macro distribution voting) | 52.82% | 40.9 |
| cmc | PFossil:one-level@10 (micro distribution voting) | 51.74% | 40.9 |
| cmc | PFossil:one-level@10 (voting) | 51.60% | 40.9 |
| cmc | PFossil:one-level@25 (best rule) | 52.75% | 106.1 |
| cmc | PFossil:one-level@25 (macro distribution voting) | 53.23% | 106.1 |
| cmc | PFossil:one-level@25 (micro distribution voting) | 51.06% | 106.1 |
| cmc | PFossil:one-level@25 (voting) | 51.87% | 106.1 |
| cmc | PFossil:one-level@50 (best rule) | 53.23% | 192.0 |
| cmc | PFossil:one-level@50 (macro distribution voting) | 52.95% | 192.0 |
| cmc | PFossil:one-level@50 (micro distribution voting) | 51.46% | 192.0 |
| cmc | PFossil:one-level@50 (voting) | 51.53% | 192.0 |
| cmc | PFossil:single (best rule) | 52.61% | 2.1 |
| cmc | PFossil:single (macro distribution voting) | 52.27% | 2.1 |
| cmc | PFossil:single (micro distribution voting) | 51.73% | 2.1 |
| cmc | PFossil:single (voting) | 51.73% | 2.1 |
| cmc | PFossil:two-level@10 (best rule) | 52.68% | 40.9 |
| cmc | PFossil:two-level@10 (macro distribution voting) | 52.48% | 40.9 |
| cmc | PFossil:two-level@10 (micro distribution voting) | 51.67% | 40.9 |
| cmc | PFossil:two-level@10 (voting) | 51.94% | 40.9 |
| cmc | PFossil:two-level@25 (best rule) | 52.55% | 106.1 |
| cmc | PFossil:two-level@25 (macro distribution voting) | 52.14% | 106.1 |
| cmc | PFossil:two-level@25 (micro distribution voting) | 51.67% | 106.1 |
| cmc | PFossil:two-level@25 (voting) | 51.53% | 106.1 |
| cmc | PFossil:two-level@50 (best rule) | 52.21% | 192.0 |
| cmc | PFossil:two-level@50 (macro distribution voting) | 51.94% | 192.0 |
| cmc | PFossil:two-level@50 (micro distribution voting) | 51.94% | 192.0 |
| cmc | PFossil:two-level@50 (voting) | 51.67% | 192.0 |
| cmc | Pypper:one-level@10 (best rule) | 53.77% | 91.0 |
| cmc | Pypper:one-level@10 (macro distribution voting) | 54.18% | 91.0 |
| cmc | Pypper:one-level@10 (micro distribution voting) | 54.59% | 91.0 |
| cmc | Pypper:one-level@10 (voting) | 54.99% | 91.0 |
| cmc | Pypper:one-level@25 (best rule) | 53.36% | 230.9 |
| cmc | Pypper:one-level@25 (macro distribution voting) | 55.40% | 230.9 |
| cmc | Pypper:one-level@25 (micro distribution voting) | 56.15% | 230.9 |
| cmc | Pypper:one-level@25 (voting) | 55.74% | 230.9 |
| cmc | Pypper:one-level@50 (best rule) | 53.16% | 467.0 |
| cmc | Pypper:one-level@50 (macro distribution voting) | 55.47% | 467.0 |
| cmc | Pypper:one-level@50 (micro distribution voting) | 55.54% | 467.0 |
| cmc | Pypper:one-level@50 (voting) | 56.29% | 467.0 |
| cmc | Pypper:single (best rule) | 52.21% | 6.5 |
| cmc | Pypper:single (macro distribution voting) | 52.35% | 6.5 |
| cmc | Pypper:single (micro distribution voting) | 52.35% | 6.5 |
| cmc | Pypper:single (voting) | 52.35% | 6.5 |
| cmc | Pypper:two-level@10 (best rule) | 55.33% | 91.0 |
| cmc | Pypper:two-level@10 (macro distribution voting) | 55.33% | 91.0 |
| cmc | Pypper:two-level@10 (micro distribution voting) | 55.40% | 91.0 |
| cmc | Pypper:two-level@10 (voting) | 55.33% | 91.0 |
| cmc | Pypper:two-level@25 (best rule) | 55.81% | 230.9 |
| cmc | Pypper:two-level@25 (macro distribution voting) | 55.40% | 230.9 |
| cmc | Pypper:two-level@25 (micro distribution voting) | 55.60% | 230.9 |
| cmc | Pypper:two-level@25 (voting) | 55.40% | 230.9 |
| cmc | Pypper:two-level@50 (best rule) | 56.08% | 467.0 |
| cmc | Pypper:two-level@50 (macro distribution voting) | 55.94% | 467.0 |
| cmc | Pypper:two-level@50 (micro distribution voting) | 55.95% | 467.0 |
| cmc | Pypper:two-level@50 (voting) | 55.74% | 467.0 |
| colic | PFoil:one-level@10 (best rule) | 85.35% | 313.4 |
| colic | PFoil:one-level@10 (macro distribution voting) | 84.80% | 313.4 |
| colic | PFoil:one-level@10 (micro distribution voting) | 85.34% | 313.4 |
| colic | PFoil:one-level@10 (voting) | 85.34% | 313.4 |
| colic | PFoil:one-level@25 (best rule) | 85.07% | 779.5 |
| colic | PFoil:one-level@25 (macro distribution voting) | 85.07% | 779.5 |
| colic | PFoil:one-level@25 (micro distribution voting) | 84.26% | 779.5 |
| colic | PFoil:one-level@25 (voting) | 85.08% | 779.5 |
| colic | PFoil:one-level@50 (best rule) | 85.07% | 1555.4 |
| colic | PFoil:one-level@50 (macro distribution voting) | 85.07% | 1555.4 |
| colic | PFoil:one-level@50 (micro distribution voting) | 84.80% | 1555.4 |
| colic | PFoil:one-level@50 (voting) | 85.62% | 1555.4 |
| colic | PFoil:single (best rule) | 84.52% | 38.5 |
| colic | PFoil:single (macro distribution voting) | 83.98% | 38.5 |
| colic | PFoil:single (micro distribution voting) | 84.52% | 38.5 |
| colic | PFoil:single (voting) | 83.15% | 38.5 |
| colic | PFoil:two-level@10 (best rule) | 85.62% | 313.4 |
| colic | PFoil:two-level@10 (macro distribution voting) | 85.35% | 313.4 |
| colic | PFoil:two-level@10 (micro distribution voting) | 85.62% | 313.4 |
| colic | PFoil:two-level@10 (voting) | 85.35% | 313.4 |
| colic | PFoil:two-level@25 (best rule) | 86.43% | 779.5 |
| colic | PFoil:two-level@25 (macro distribution voting) | 85.62% | 779.5 |
| colic | PFoil:two-level@25 (micro distribution voting) | 85.62% | 779.5 |
| colic | PFoil:two-level@25 (voting) | 85.89% | 779.5 |
| colic | PFoil:two-level@50 (best rule) | 86.43% | 1555.4 |
| colic | PFoil:two-level@50 (macro distribution voting) | 86.43% | 1555.4 |
| colic | PFoil:two-level@50 (micro distribution voting) | 86.16% | 1555.4 |
| colic | PFoil:two-level@50 (voting) | 86.16% | 1555.4 |
| colic | PFossil:one-level@10 (best rule) | 81.01% | 168.4 |
| colic | PFossil:one-level@10 (macro distribution voting) | 85.89% | 168.4 |
| colic | PFossil:one-level@10 (micro distribution voting) | 86.43% | 168.4 |
| colic | PFossil:one-level@10 (voting) | 86.70% | 168.4 |
| colic | PFossil:one-level@25 (best rule) | 80.73% | 436.5 |
| colic | PFossil:one-level@25 (macro distribution voting) | 85.89% | 436.5 |
| colic | PFossil:one-level@25 (micro distribution voting) | 86.42% | 436.5 |
| colic | PFossil:one-level@25 (voting) | 86.43% | 436.5 |
| colic | PFossil:one-level@50 (best rule) | 82.36% | 859.2 |
| colic | PFossil:one-level@50 (macro distribution voting) | 85.89% | 859.2 |
| colic | PFossil:one-level@50 (micro distribution voting) | 85.88% | 859.2 |
| colic | PFossil:one-level@50 (voting) | 86.97% | 859.2 |
| colic | PFossil:single (best rule) | 82.89% | 19.8 |
| colic | PFossil:single (macro distribution voting) | 81.79% | 19.8 |
| colic | PFossil:single (micro distribution voting) | 86.15% | 19.8 |
| colic | PFossil:single (voting) | 82.88% | 19.8 |
| colic | PFossil:two-level@10 (best rule) | 85.89% | 168.4 |
| colic | PFossil:two-level@10 (macro distribution voting) | 85.89% | 168.4 |
| colic | PFossil:two-level@10 (micro distribution voting) | 86.70% | 168.4 |
| colic | PFossil:two-level@10 (voting) | 85.61% | 168.4 |
| colic | PFossil:two-level@25 (best rule) | 85.89% | 436.5 |
| colic | PFossil:two-level@25 (macro distribution voting) | 85.62% | 436.5 |
| colic | PFossil:two-level@25 (micro distribution voting) | 86.97% | 436.5 |
| colic | PFossil:two-level@25 (voting) | 86.70% | 436.5 |
| colic | PFossil:two-level@50 (best rule) | 85.35% | 859.2 |
| colic | PFossil:two-level@50 (macro distribution voting) | 85.62% | 859.2 |
| colic | PFossil:two-level@50 (micro distribution voting) | 86.43% | 859.2 |
| colic | PFossil:two-level@50 (voting) | 85.89% | 859.2 |
| colic | Pypper:one-level@10 (best rule) | 83.16% | 72.5 |
| colic | Pypper:one-level@10 (macro distribution voting) | 85.34% | 72.5 |
| colic | Pypper:one-level@10 (micro distribution voting) | 85.61% | 72.5 |
| colic | Pypper:one-level@10 (voting) | 85.34% | 72.5 |
| colic | Pypper:one-level@25 (best rule) | 84.26% | 175.7 |
| colic | Pypper:one-level@25 (macro distribution voting) | 85.88% | 175.7 |
| colic | Pypper:one-level@25 (micro distribution voting) | 85.88% | 175.7 |
| colic | Pypper:one-level@25 (voting) | 85.88% | 175.7 |
| colic | Pypper:one-level@50 (best rule) | 83.16% | 348.1 |
| colic | Pypper:one-level@50 (macro distribution voting) | 86.15% | 348.1 |
| colic | Pypper:one-level@50 (micro distribution voting) | 85.88% | 348.1 |
| colic | Pypper:one-level@50 (voting) | 86.15% | 348.1 |
| colic | Pypper:single (best rule) | 85.08% | 5.6 |
| colic | Pypper:single (macro distribution voting) | 84.80% | 5.6 |
| colic | Pypper:single (micro distribution voting) | 85.89% | 5.6 |
| colic | Pypper:single (voting) | 85.35% | 5.6 |
| colic | Pypper:two-level@10 (best rule) | 86.15% | 72.5 |
| colic | Pypper:two-level@10 (macro distribution voting) | 86.15% | 72.5 |
| colic | Pypper:two-level@10 (micro distribution voting) | 86.15% | 72.5 |
| colic | Pypper:two-level@10 (voting) | 85.34% | 72.5 |
| colic | Pypper:two-level@25 (best rule) | 85.61% | 175.7 |
| colic | Pypper:two-level@25 (macro distribution voting) | 85.61% | 175.7 |
| colic | Pypper:two-level@25 (micro distribution voting) | 85.88% | 175.7 |
| colic | Pypper:two-level@25 (voting) | 85.61% | 175.7 |
| colic | Pypper:two-level@50 (best rule) | 85.61% | 348.1 |
| colic | Pypper:two-level@50 (macro distribution voting) | 85.88% | 348.1 |
| colic | Pypper:two-level@50 (micro distribution voting) | 85.88% | 348.1 |
| colic | Pypper:two-level@50 (voting) | 85.61% | 348.1 |
| compas-two-years | PFoil:one-level@10 (best rule) | 67.09% | 3583.1 |
| compas-two-years | PFoil:one-level@10 (macro distribution voting) | 67.18% | 3583.1 |
| compas-two-years | PFoil:one-level@10 (micro distribution voting) | 63.57% | 3583.1 |
| compas-two-years | PFoil:one-level@10 (voting) | 65.92% | 3583.1 |
| compas-two-years | PFoil:one-level@25 (best rule) | 67.26% | 8930.2 |
| compas-two-years | PFoil:one-level@25 (macro distribution voting) | 67.16% | 8930.2 |
| compas-two-years | PFoil:one-level@25 (micro distribution voting) | 63.64% | 8930.2 |
| compas-two-years | PFoil:one-level@25 (voting) | 66.48% | 8930.2 |
| compas-two-years | PFoil:one-level@50 (best rule) | 67.58% | 17858.5 |
| compas-two-years | PFoil:one-level@50 (macro distribution voting) | 67.24% | 17858.5 |
| compas-two-years | PFoil:one-level@50 (micro distribution voting) | 63.62% | 17858.5 |
| compas-two-years | PFoil:one-level@50 (voting) | 66.46% | 17858.5 |
| compas-two-years | PFoil:single (best rule) | 66.65% | 330.8 |
| compas-two-years | PFoil:single (macro distribution voting) | 66.24% | 330.8 |
| compas-two-years | PFoil:single (micro distribution voting) | 62.26% | 330.8 |
| compas-two-years | PFoil:single (voting) | 63.94% | 330.8 |
| compas-two-years | PFoil:two-level@10 (best rule) | 66.79% | 3583.1 |
| compas-two-years | PFoil:two-level@10 (macro distribution voting) | 66.41% | 3583.1 |
| compas-two-years | PFoil:two-level@10 (micro distribution voting) | 64.15% | 3583.1 |
| compas-two-years | PFoil:two-level@10 (voting) | 64.63% | 3583.1 |
| compas-two-years | PFoil:two-level@25 (best rule) | 67.09% | 8930.2 |
| compas-two-years | PFoil:two-level@25 (macro distribution voting) | 66.94% | 8930.2 |
| compas-two-years | PFoil:two-level@25 (micro distribution voting) | 64.84% | 8930.2 |
| compas-two-years | PFoil:two-level@25 (voting) | 65.69% | 8930.2 |
| compas-two-years | PFoil:two-level@50 (best rule) | 67.11% | 17858.5 |
| compas-two-years | PFoil:two-level@50 (macro distribution voting) | 66.80% | 17858.5 |
| compas-two-years | PFoil:two-level@50 (micro distribution voting) | 64.61% | 17858.5 |
| compas-two-years | PFoil:two-level@50 (voting) | 65.82% | 17858.5 |
| compas-two-years | PFossil:one-level@10 (best rule) | 64.65% | 12.1 |
| compas-two-years | PFossil:one-level@10 (macro distribution voting) | 64.85% | 12.1 |
| compas-two-years | PFossil:one-level@10 (micro distribution voting) | 64.95% | 12.1 |
| compas-two-years | PFossil:one-level@10 (voting) | 64.95% | 12.1 |
| compas-two-years | PFossil:one-level@25 (best rule) | 65.40% | 31.3 |
| compas-two-years | PFossil:one-level@25 (macro distribution voting) | 64.57% | 31.3 |
| compas-two-years | PFossil:one-level@25 (micro distribution voting) | 64.53% | 31.3 |
| compas-two-years | PFossil:one-level@25 (voting) | 64.53% | 31.3 |
| compas-two-years | PFossil:one-level@50 (best rule) | 66.22% | 63.6 |
| compas-two-years | PFossil:one-level@50 (macro distribution voting) | 64.80% | 63.6 |
| compas-two-years | PFossil:one-level@50 (micro distribution voting) | 64.70% | 63.6 |
| compas-two-years | PFossil:one-level@50 (voting) | 64.68% | 63.6 |
| compas-two-years | PFossil:single (best rule) | 52.96% | 1.0 |
| compas-two-years | PFossil:single (macro distribution voting) | 52.96% | 1.0 |
| compas-two-years | PFossil:single (micro distribution voting) | 52.96% | 1.0 |
| compas-two-years | PFossil:single (voting) | 52.96% | 1.0 |
| compas-two-years | PFossil:two-level@10 (best rule) | 64.95% | 12.1 |
| compas-two-years | PFossil:two-level@10 (macro distribution voting) | 64.95% | 12.1 |
| compas-two-years | PFossil:two-level@10 (micro distribution voting) | 64.95% | 12.1 |
| compas-two-years | PFossil:two-level@10 (voting) | 64.95% | 12.1 |
| compas-two-years | PFossil:two-level@25 (best rule) | 64.53% | 31.3 |
| compas-two-years | PFossil:two-level@25 (macro distribution voting) | 64.53% | 31.3 |
| compas-two-years | PFossil:two-level@25 (micro distribution voting) | 64.53% | 31.3 |
| compas-two-years | PFossil:two-level@25 (voting) | 64.53% | 31.3 |
| compas-two-years | PFossil:two-level@50 (best rule) | 64.68% | 63.6 |
| compas-two-years | PFossil:two-level@50 (macro distribution voting) | 64.68% | 63.6 |
| compas-two-years | PFossil:two-level@50 (micro distribution voting) | 64.68% | 63.6 |
| compas-two-years | PFossil:two-level@50 (voting) | 64.68% | 63.6 |
| compas-two-years | Pypper:one-level@10 (best rule) | 68.15% | 91.0 |
| compas-two-years | Pypper:one-level@10 (macro distribution voting) | 67.92% | 91.0 |
| compas-two-years | Pypper:one-level@10 (micro distribution voting) | 67.22% | 91.0 |
| compas-two-years | Pypper:one-level@10 (voting) | 67.89% | 91.0 |
| compas-two-years | Pypper:one-level@25 (best rule) | 68.07% | 224.2 |
| compas-two-years | Pypper:one-level@25 (macro distribution voting) | 67.85% | 224.2 |
| compas-two-years | Pypper:one-level@25 (micro distribution voting) | 67.20% | 224.2 |
| compas-two-years | Pypper:one-level@25 (voting) | 67.92% | 224.2 |
| compas-two-years | Pypper:one-level@50 (best rule) | 68.00% | 450.0 |
| compas-two-years | Pypper:one-level@50 (macro distribution voting) | 67.81% | 450.0 |
| compas-two-years | Pypper:one-level@50 (micro distribution voting) | 67.28% | 450.0 |
| compas-two-years | Pypper:one-level@50 (voting) | 67.85% | 450.0 |
| compas-two-years | Pypper:single (best rule) | 67.03% | 8.3 |
| compas-two-years | Pypper:single (macro distribution voting) | 67.03% | 8.3 |
| compas-two-years | Pypper:single (micro distribution voting) | 66.62% | 8.3 |
| compas-two-years | Pypper:single (voting) | 66.62% | 8.3 |
| compas-two-years | Pypper:two-level@10 (best rule) | 67.77% | 91.0 |
| compas-two-years | Pypper:two-level@10 (macro distribution voting) | 67.79% | 91.0 |
| compas-two-years | Pypper:two-level@10 (micro distribution voting) | 67.70% | 91.0 |
| compas-two-years | Pypper:two-level@10 (voting) | 67.58% | 91.0 |
| compas-two-years | Pypper:two-level@25 (best rule) | 67.73% | 224.2 |
| compas-two-years | Pypper:two-level@25 (macro distribution voting) | 67.81% | 224.2 |
| compas-two-years | Pypper:two-level@25 (micro distribution voting) | 67.77% | 224.2 |
| compas-two-years | Pypper:two-level@25 (voting) | 67.62% | 224.2 |
| compas-two-years | Pypper:two-level@50 (best rule) | 67.70% | 450.0 |
| compas-two-years | Pypper:two-level@50 (macro distribution voting) | 67.70% | 450.0 |
| compas-two-years | Pypper:two-level@50 (micro distribution voting) | 67.77% | 450.0 |
| compas-two-years | Pypper:two-level@50 (voting) | 67.66% | 450.0 |
| credit-approval | PFoil:one-level@10 (best rule) | 86.38% | 460.0 |
| credit-approval | PFoil:one-level@10 (macro distribution voting) | 85.94% | 460.0 |
| credit-approval | PFoil:one-level@10 (micro distribution voting) | 87.39% | 460.0 |
| credit-approval | PFoil:one-level@10 (voting) | 85.80% | 460.0 |
| credit-approval | PFoil:one-level@25 (best rule) | 87.10% | 1143.0 |
| credit-approval | PFoil:one-level@25 (macro distribution voting) | 86.52% | 1143.0 |
| credit-approval | PFoil:one-level@25 (micro distribution voting) | 86.96% | 1143.0 |
| credit-approval | PFoil:one-level@25 (voting) | 87.10% | 1143.0 |
| credit-approval | PFoil:one-level@50 (best rule) | 86.52% | 2278.9 |
| credit-approval | PFoil:one-level@50 (macro distribution voting) | 85.80% | 2278.9 |
| credit-approval | PFoil:one-level@50 (micro distribution voting) | 86.67% | 2278.9 |
| credit-approval | PFoil:one-level@50 (voting) | 86.09% | 2278.9 |
| credit-approval | PFoil:single (best rule) | 84.49% | 57.5 |
| credit-approval | PFoil:single (macro distribution voting) | 85.07% | 57.5 |
| credit-approval | PFoil:single (micro distribution voting) | 83.77% | 57.5 |
| credit-approval | PFoil:single (voting) | 83.33% | 57.5 |
| credit-approval | PFoil:two-level@10 (best rule) | 85.51% | 460.0 |
| credit-approval | PFoil:two-level@10 (macro distribution voting) | 85.65% | 460.0 |
| credit-approval | PFoil:two-level@10 (micro distribution voting) | 85.22% | 460.0 |
| credit-approval | PFoil:two-level@10 (voting) | 84.78% | 460.0 |
| credit-approval | PFoil:two-level@25 (best rule) | 86.38% | 1143.0 |
| credit-approval | PFoil:two-level@25 (macro distribution voting) | 85.94% | 1143.0 |
| credit-approval | PFoil:two-level@25 (micro distribution voting) | 86.52% | 1143.0 |
| credit-approval | PFoil:two-level@25 (voting) | 86.38% | 1143.0 |
| credit-approval | PFoil:two-level@50 (best rule) | 85.51% | 2278.9 |
| credit-approval | PFoil:two-level@50 (macro distribution voting) | 85.80% | 2278.9 |
| credit-approval | PFoil:two-level@50 (micro distribution voting) | 86.52% | 2278.9 |
| credit-approval | PFoil:two-level@50 (voting) | 85.80% | 2278.9 |
| credit-approval | PFossil:one-level@10 (best rule) | 83.77% | 149.4 |
| credit-approval | PFossil:one-level@10 (macro distribution voting) | 84.78% | 149.4 |
| credit-approval | PFossil:one-level@10 (micro distribution voting) | 85.65% | 149.4 |
| credit-approval | PFossil:one-level@10 (voting) | 85.07% | 149.4 |
| credit-approval | PFossil:one-level@25 (best rule) | 82.32% | 363.0 |
| credit-approval | PFossil:one-level@25 (macro distribution voting) | 85.22% | 363.0 |
| credit-approval | PFossil:one-level@25 (micro distribution voting) | 85.36% | 363.0 |
| credit-approval | PFossil:one-level@25 (voting) | 85.22% | 363.0 |
| credit-approval | PFossil:one-level@50 (best rule) | 81.01% | 709.6 |
| credit-approval | PFossil:one-level@50 (macro distribution voting) | 85.22% | 709.6 |
| credit-approval | PFossil:one-level@50 (micro distribution voting) | 85.51% | 709.6 |
| credit-approval | PFossil:one-level@50 (voting) | 85.65% | 709.6 |
| credit-approval | PFossil:single (best rule) | 82.17% | 8.8 |
| credit-approval | PFossil:single (macro distribution voting) | 82.03% | 8.8 |
| credit-approval | PFossil:single (micro distribution voting) | 85.07% | 8.8 |
| credit-approval | PFossil:single (voting) | 82.46% | 8.8 |
| credit-approval | PFossil:two-level@10 (best rule) | 85.07% | 149.4 |
| credit-approval | PFossil:two-level@10 (macro distribution voting) | 85.22% | 149.4 |
| credit-approval | PFossil:two-level@10 (micro distribution voting) | 85.36% | 149.4 |
| credit-approval | PFossil:two-level@10 (voting) | 84.78% | 149.4 |
| credit-approval | PFossil:two-level@25 (best rule) | 84.49% | 363.0 |
| credit-approval | PFossil:two-level@25 (macro distribution voting) | 85.07% | 363.0 |
| credit-approval | PFossil:two-level@25 (micro distribution voting) | 85.80% | 363.0 |
| credit-approval | PFossil:two-level@25 (voting) | 85.65% | 363.0 |
| credit-approval | PFossil:two-level@50 (best rule) | 85.07% | 709.6 |
| credit-approval | PFossil:two-level@50 (macro distribution voting) | 85.51% | 709.6 |
| credit-approval | PFossil:two-level@50 (micro distribution voting) | 85.65% | 709.6 |
| credit-approval | PFossil:two-level@50 (voting) | 85.65% | 709.6 |
| credit-approval | Pypper:one-level@10 (best rule) | 85.80% | 72.8 |
| credit-approval | Pypper:one-level@10 (macro distribution voting) | 86.23% | 72.8 |
| credit-approval | Pypper:one-level@10 (micro distribution voting) | 85.51% | 72.8 |
| credit-approval | Pypper:one-level@10 (voting) | 86.09% | 72.8 |
| credit-approval | Pypper:one-level@25 (best rule) | 86.23% | 184.3 |
| credit-approval | Pypper:one-level@25 (macro distribution voting) | 85.94% | 184.3 |
| credit-approval | Pypper:one-level@25 (micro distribution voting) | 85.36% | 184.3 |
| credit-approval | Pypper:one-level@25 (voting) | 85.65% | 184.3 |
| credit-approval | Pypper:one-level@50 (best rule) | 86.23% | 359.7 |
| credit-approval | Pypper:one-level@50 (macro distribution voting) | 85.80% | 359.7 |
| credit-approval | Pypper:one-level@50 (micro distribution voting) | 85.36% | 359.7 |
| credit-approval | Pypper:one-level@50 (voting) | 85.65% | 359.7 |
| credit-approval | Pypper:single (best rule) | 86.09% | 4.8 |
| credit-approval | Pypper:single (macro distribution voting) | 86.09% | 4.8 |
| credit-approval | Pypper:single (micro distribution voting) | 85.51% | 4.8 |
| credit-approval | Pypper:single (voting) | 86.09% | 4.8 |
| credit-approval | Pypper:two-level@10 (best rule) | 86.52% | 72.8 |
| credit-approval | Pypper:two-level@10 (macro distribution voting) | 86.81% | 72.8 |
| credit-approval | Pypper:two-level@10 (micro distribution voting) | 85.65% | 72.8 |
| credit-approval | Pypper:two-level@10 (voting) | 85.94% | 72.8 |
| credit-approval | Pypper:two-level@25 (best rule) | 85.80% | 184.3 |
| credit-approval | Pypper:two-level@25 (macro distribution voting) | 85.94% | 184.3 |
| credit-approval | Pypper:two-level@25 (micro distribution voting) | 85.22% | 184.3 |
| credit-approval | Pypper:two-level@25 (voting) | 85.80% | 184.3 |
| credit-approval | Pypper:two-level@50 (best rule) | 85.51% | 359.7 |
| credit-approval | Pypper:two-level@50 (macro distribution voting) | 85.51% | 359.7 |
| credit-approval | Pypper:two-level@50 (micro distribution voting) | 85.07% | 359.7 |
| credit-approval | Pypper:two-level@50 (voting) | 85.80% | 359.7 |
| credit-g | PFoil:one-level@10 (best rule) | 72.70% | 748.9 |
| credit-g | PFoil:one-level@10 (macro distribution voting) | 76.40% | 748.9 |
| credit-g | PFoil:one-level@10 (micro distribution voting) | 72.90% | 748.9 |
| credit-g | PFoil:one-level@10 (voting) | 77.00% | 748.9 |
| credit-g | PFoil:one-level@25 (best rule) | 72.30% | 1868.9 |
| credit-g | PFoil:one-level@25 (macro distribution voting) | 77.40% | 1868.9 |
| credit-g | PFoil:one-level@25 (micro distribution voting) | 72.20% | 1868.9 |
| credit-g | PFoil:one-level@25 (voting) | 76.90% | 1868.9 |
| credit-g | PFoil:one-level@50 (best rule) | 71.70% | 3743.0 |
| credit-g | PFoil:one-level@50 (macro distribution voting) | 76.90% | 3743.0 |
| credit-g | PFoil:one-level@50 (micro distribution voting) | 72.40% | 3743.0 |
| credit-g | PFoil:one-level@50 (voting) | 76.30% | 3743.0 |
| credit-g | PFoil:single (best rule) | 73.10% | 94.4 |
| credit-g | PFoil:single (macro distribution voting) | 73.50% | 94.4 |
| credit-g | PFoil:single (micro distribution voting) | 72.30% | 94.4 |
| credit-g | PFoil:single (voting) | 72.30% | 94.4 |
| credit-g | PFoil:two-level@10 (best rule) | 76.00% | 748.9 |
| credit-g | PFoil:two-level@10 (macro distribution voting) | 76.00% | 748.9 |
| credit-g | PFoil:two-level@10 (micro distribution voting) | 74.40% | 748.9 |
| credit-g | PFoil:two-level@10 (voting) | 76.10% | 748.9 |
| credit-g | PFoil:two-level@25 (best rule) | 76.80% | 1868.9 |
| credit-g | PFoil:two-level@25 (macro distribution voting) | 76.60% | 1868.9 |
| credit-g | PFoil:two-level@25 (micro distribution voting) | 74.00% | 1868.9 |
| credit-g | PFoil:two-level@25 (voting) | 76.30% | 1868.9 |
| credit-g | PFoil:two-level@50 (best rule) | 76.90% | 3743.0 |
| credit-g | PFoil:two-level@50 (macro distribution voting) | 77.10% | 3743.0 |
| credit-g | PFoil:two-level@50 (micro distribution voting) | 75.40% | 3743.0 |
| credit-g | PFoil:two-level@50 (voting) | 76.40% | 3743.0 |
| credit-g | PFossil:one-level@10 (best rule) | 73.80% | 81.7 |
| credit-g | PFossil:one-level@10 (macro distribution voting) | 74.60% | 81.7 |
| credit-g | PFossil:one-level@10 (micro distribution voting) | 75.80% | 81.7 |
| credit-g | PFossil:one-level@10 (voting) | 72.90% | 81.7 |
| credit-g | PFossil:one-level@25 (best rule) | 73.30% | 206.0 |
| credit-g | PFossil:one-level@25 (macro distribution voting) | 75.50% | 206.0 |
| credit-g | PFossil:one-level@25 (micro distribution voting) | 75.30% | 206.0 |
| credit-g | PFossil:one-level@25 (voting) | 72.90% | 206.0 |
| credit-g | PFossil:one-level@50 (best rule) | 73.30% | 401.7 |
| credit-g | PFossil:one-level@50 (macro distribution voting) | 75.40% | 401.7 |
| credit-g | PFossil:one-level@50 (micro distribution voting) | 75.00% | 401.7 |
| credit-g | PFossil:one-level@50 (voting) | 74.00% | 401.7 |
| credit-g | PFossil:single (best rule) | 73.20% | 3.4 |
| credit-g | PFossil:single (macro distribution voting) | 73.20% | 3.4 |
| credit-g | PFossil:single (micro distribution voting) | 73.30% | 3.4 |
| credit-g | PFossil:single (voting) | 73.20% | 3.4 |
| credit-g | PFossil:two-level@10 (best rule) | 73.50% | 81.7 |
| credit-g | PFossil:two-level@10 (macro distribution voting) | 73.50% | 81.7 |
| credit-g | PFossil:two-level@10 (micro distribution voting) | 73.40% | 81.7 |
| credit-g | PFossil:two-level@10 (voting) | 73.20% | 81.7 |
| credit-g | PFossil:two-level@25 (best rule) | 74.70% | 206.0 |
| credit-g | PFossil:two-level@25 (macro distribution voting) | 74.90% | 206.0 |
| credit-g | PFossil:two-level@25 (micro distribution voting) | 74.50% | 206.0 |
| credit-g | PFossil:two-level@25 (voting) | 75.00% | 206.0 |
| credit-g | PFossil:two-level@50 (best rule) | 74.50% | 401.7 |
| credit-g | PFossil:two-level@50 (macro distribution voting) | 74.90% | 401.7 |
| credit-g | PFossil:two-level@50 (micro distribution voting) | 74.40% | 401.7 |
| credit-g | PFossil:two-level@50 (voting) | 74.50% | 401.7 |
| credit-g | Pypper:one-level@10 (best rule) | 73.10% | 98.5 |
| credit-g | Pypper:one-level@10 (macro distribution voting) | 74.10% | 98.5 |
| credit-g | Pypper:one-level@10 (micro distribution voting) | 72.40% | 98.5 |
| credit-g | Pypper:one-level@10 (voting) | 73.10% | 98.5 |
| credit-g | Pypper:one-level@25 (best rule) | 74.60% | 240.5 |
| credit-g | Pypper:one-level@25 (macro distribution voting) | 74.40% | 240.5 |
| credit-g | Pypper:one-level@25 (micro distribution voting) | 72.00% | 240.5 |
| credit-g | Pypper:one-level@25 (voting) | 74.10% | 240.5 |
| credit-g | Pypper:one-level@50 (best rule) | 74.00% | 467.4 |
| credit-g | Pypper:one-level@50 (macro distribution voting) | 74.10% | 467.4 |
| credit-g | Pypper:one-level@50 (micro distribution voting) | 73.00% | 467.4 |
| credit-g | Pypper:one-level@50 (voting) | 73.50% | 467.4 |
| credit-g | Pypper:single (best rule) | 71.50% | 6.9 |
| credit-g | Pypper:single (macro distribution voting) | 71.50% | 6.9 |
| credit-g | Pypper:single (micro distribution voting) | 71.50% | 6.9 |
| credit-g | Pypper:single (voting) | 71.60% | 6.9 |
| credit-g | Pypper:two-level@10 (best rule) | 73.90% | 98.5 |
| credit-g | Pypper:two-level@10 (macro distribution voting) | 73.70% | 98.5 |
| credit-g | Pypper:two-level@10 (micro distribution voting) | 73.40% | 98.5 |
| credit-g | Pypper:two-level@10 (voting) | 73.40% | 98.5 |
| credit-g | Pypper:two-level@25 (best rule) | 74.40% | 240.5 |
| credit-g | Pypper:two-level@25 (macro distribution voting) | 75.10% | 240.5 |
| credit-g | Pypper:two-level@25 (micro distribution voting) | 74.30% | 240.5 |
| credit-g | Pypper:two-level@25 (voting) | 74.60% | 240.5 |
| credit-g | Pypper:two-level@50 (best rule) | 73.90% | 467.4 |
| credit-g | Pypper:two-level@50 (macro distribution voting) | 73.90% | 467.4 |
| credit-g | Pypper:two-level@50 (micro distribution voting) | 74.10% | 467.4 |
| credit-g | Pypper:two-level@50 (voting) | 73.80% | 467.4 |
| cylinder-bands | PFoil:one-level@10 (best rule) | 73.52% | 407.2 |
| cylinder-bands | PFoil:one-level@10 (macro distribution voting) | 81.48% | 407.2 |
| cylinder-bands | PFoil:one-level@10 (micro distribution voting) | 77.59% | 407.2 |
| cylinder-bands | PFoil:one-level@10 (voting) | 81.48% | 407.2 |
| cylinder-bands | PFoil:one-level@25 (best rule) | 72.04% | 1016.2 |
| cylinder-bands | PFoil:one-level@25 (macro distribution voting) | 82.96% | 1016.2 |
| cylinder-bands | PFoil:one-level@25 (micro distribution voting) | 78.33% | 1016.2 |
| cylinder-bands | PFoil:one-level@25 (voting) | 82.78% | 1016.2 |
| cylinder-bands | PFoil:one-level@50 (best rule) | 70.19% | 2045.2 |
| cylinder-bands | PFoil:one-level@50 (macro distribution voting) | 84.44% | 2045.2 |
| cylinder-bands | PFoil:one-level@50 (micro distribution voting) | 78.89% | 2045.2 |
| cylinder-bands | PFoil:one-level@50 (voting) | 83.70% | 2045.2 |
| cylinder-bands | PFoil:single (best rule) | 75.00% | 48.8 |
| cylinder-bands | PFoil:single (macro distribution voting) | 76.30% | 48.8 |
| cylinder-bands | PFoil:single (micro distribution voting) | 73.70% | 48.8 |
| cylinder-bands | PFoil:single (voting) | 74.07% | 48.8 |
| cylinder-bands | PFoil:two-level@10 (best rule) | 78.89% | 407.2 |
| cylinder-bands | PFoil:two-level@10 (macro distribution voting) | 79.44% | 407.2 |
| cylinder-bands | PFoil:two-level@10 (micro distribution voting) | 77.96% | 407.2 |
| cylinder-bands | PFoil:two-level@10 (voting) | 79.07% | 407.2 |
| cylinder-bands | PFoil:two-level@25 (best rule) | 81.48% | 1016.2 |
| cylinder-bands | PFoil:two-level@25 (macro distribution voting) | 82.04% | 1016.2 |
| cylinder-bands | PFoil:two-level@25 (micro distribution voting) | 80.56% | 1016.2 |
| cylinder-bands | PFoil:two-level@25 (voting) | 81.30% | 1016.2 |
| cylinder-bands | PFoil:two-level@50 (best rule) | 81.85% | 2045.2 |
| cylinder-bands | PFoil:two-level@50 (macro distribution voting) | 82.22% | 2045.2 |
| cylinder-bands | PFoil:two-level@50 (micro distribution voting) | 81.11% | 2045.2 |
| cylinder-bands | PFoil:two-level@50 (voting) | 82.22% | 2045.2 |
| cylinder-bands | PFossil:one-level@10 (best rule) | 75.37% | 103.5 |
| cylinder-bands | PFossil:one-level@10 (macro distribution voting) | 74.81% | 103.5 |
| cylinder-bands | PFossil:one-level@10 (micro distribution voting) | 72.41% | 103.5 |
| cylinder-bands | PFossil:one-level@10 (voting) | 74.44% | 103.5 |
| cylinder-bands | PFossil:one-level@25 (best rule) | 72.59% | 268.3 |
| cylinder-bands | PFossil:one-level@25 (macro distribution voting) | 75.93% | 268.3 |
| cylinder-bands | PFossil:one-level@25 (micro distribution voting) | 72.22% | 268.3 |
| cylinder-bands | PFossil:one-level@25 (voting) | 75.19% | 268.3 |
| cylinder-bands | PFossil:one-level@50 (best rule) | 73.33% | 526.1 |
| cylinder-bands | PFossil:one-level@50 (macro distribution voting) | 75.74% | 526.1 |
| cylinder-bands | PFossil:one-level@50 (micro distribution voting) | 71.67% | 526.1 |
| cylinder-bands | PFossil:one-level@50 (voting) | 75.37% | 526.1 |
| cylinder-bands | PFossil:single (best rule) | 70.19% | 9.4 |
| cylinder-bands | PFossil:single (macro distribution voting) | 70.56% | 9.4 |
| cylinder-bands | PFossil:single (micro distribution voting) | 68.15% | 9.4 |
| cylinder-bands | PFossil:single (voting) | 67.96% | 9.4 |
| cylinder-bands | PFossil:two-level@10 (best rule) | 74.44% | 103.5 |
| cylinder-bands | PFossil:two-level@10 (macro distribution voting) | 74.26% | 103.5 |
| cylinder-bands | PFossil:two-level@10 (micro distribution voting) | 72.96% | 103.5 |
| cylinder-bands | PFossil:two-level@10 (voting) | 72.41% | 103.5 |
| cylinder-bands | PFossil:two-level@25 (best rule) | 74.44% | 268.3 |
| cylinder-bands | PFossil:two-level@25 (macro distribution voting) | 74.07% | 268.3 |
| cylinder-bands | PFossil:two-level@25 (micro distribution voting) | 73.33% | 268.3 |
| cylinder-bands | PFossil:two-level@25 (voting) | 71.85% | 268.3 |
| cylinder-bands | PFossil:two-level@50 (best rule) | 74.63% | 526.1 |
| cylinder-bands | PFossil:two-level@50 (macro distribution voting) | 74.63% | 526.1 |
| cylinder-bands | PFossil:two-level@50 (micro distribution voting) | 72.59% | 526.1 |
| cylinder-bands | PFossil:two-level@50 (voting) | 71.67% | 526.1 |
| cylinder-bands | Pypper:one-level@10 (best rule) | 75.00% | 99.4 |
| cylinder-bands | Pypper:one-level@10 (macro distribution voting) | 77.22% | 99.4 |
| cylinder-bands | Pypper:one-level@10 (micro distribution voting) | 70.19% | 99.4 |
| cylinder-bands | Pypper:one-level@10 (voting) | 77.04% | 99.4 |
| cylinder-bands | Pypper:one-level@25 (best rule) | 71.67% | 253.6 |
| cylinder-bands | Pypper:one-level@25 (macro distribution voting) | 78.52% | 253.6 |
| cylinder-bands | Pypper:one-level@25 (micro distribution voting) | 71.30% | 253.6 |
| cylinder-bands | Pypper:one-level@25 (voting) | 77.96% | 253.6 |
| cylinder-bands | Pypper:one-level@50 (best rule) | 74.81% | 513.3 |
| cylinder-bands | Pypper:one-level@50 (macro distribution voting) | 78.70% | 513.3 |
| cylinder-bands | Pypper:one-level@50 (micro distribution voting) | 70.93% | 513.3 |
| cylinder-bands | Pypper:one-level@50 (voting) | 78.70% | 513.3 |
| cylinder-bands | Pypper:single (best rule) | 67.59% | 8.0 |
| cylinder-bands | Pypper:single (macro distribution voting) | 69.07% | 8.0 |
| cylinder-bands | Pypper:single (micro distribution voting) | 66.67% | 8.0 |
| cylinder-bands | Pypper:single (voting) | 67.78% | 8.0 |
| cylinder-bands | Pypper:two-level@10 (best rule) | 76.30% | 99.4 |
| cylinder-bands | Pypper:two-level@10 (macro distribution voting) | 76.85% | 99.4 |
| cylinder-bands | Pypper:two-level@10 (micro distribution voting) | 73.33% | 99.4 |
| cylinder-bands | Pypper:two-level@10 (voting) | 73.89% | 99.4 |
| cylinder-bands | Pypper:two-level@25 (best rule) | 78.33% | 253.6 |
| cylinder-bands | Pypper:two-level@25 (macro distribution voting) | 78.33% | 253.6 |
| cylinder-bands | Pypper:two-level@25 (micro distribution voting) | 73.89% | 253.6 |
| cylinder-bands | Pypper:two-level@25 (voting) | 74.26% | 253.6 |
| cylinder-bands | Pypper:two-level@50 (best rule) | 78.33% | 513.3 |
| cylinder-bands | Pypper:two-level@50 (macro distribution voting) | 77.78% | 513.3 |
| cylinder-bands | Pypper:two-level@50 (micro distribution voting) | 73.15% | 513.3 |
| cylinder-bands | Pypper:two-level@50 (voting) | 75.00% | 513.3 |
| diabetes | PFoil:one-level@10 (best rule) | 73.82% | 709.9 |
| diabetes | PFoil:one-level@10 (macro distribution voting) | 75.39% | 709.9 |
| diabetes | PFoil:one-level@10 (micro distribution voting) | 74.47% | 709.9 |
| diabetes | PFoil:one-level@10 (voting) | 77.60% | 709.9 |
| diabetes | PFoil:one-level@25 (best rule) | 73.68% | 1758.1 |
| diabetes | PFoil:one-level@25 (macro distribution voting) | 76.43% | 1758.1 |
| diabetes | PFoil:one-level@25 (micro distribution voting) | 74.60% | 1758.1 |
| diabetes | PFoil:one-level@25 (voting) | 77.61% | 1758.1 |
| diabetes | PFoil:one-level@50 (best rule) | 74.21% | 3522.2 |
| diabetes | PFoil:one-level@50 (macro distribution voting) | 76.30% | 3522.2 |
| diabetes | PFoil:one-level@50 (micro distribution voting) | 75.51% | 3522.2 |
| diabetes | PFoil:one-level@50 (voting) | 76.82% | 3522.2 |
| diabetes | PFoil:single (best rule) | 75.77% | 81.1 |
| diabetes | PFoil:single (macro distribution voting) | 75.12% | 81.1 |
| diabetes | PFoil:single (micro distribution voting) | 73.44% | 81.1 |
| diabetes | PFoil:single (voting) | 75.40% | 81.1 |
| diabetes | PFoil:two-level@10 (best rule) | 75.65% | 709.9 |
| diabetes | PFoil:two-level@10 (macro distribution voting) | 75.65% | 709.9 |
| diabetes | PFoil:two-level@10 (micro distribution voting) | 75.78% | 709.9 |
| diabetes | PFoil:two-level@10 (voting) | 74.60% | 709.9 |
| diabetes | PFoil:two-level@25 (best rule) | 77.34% | 1758.1 |
| diabetes | PFoil:two-level@25 (macro distribution voting) | 76.69% | 1758.1 |
| diabetes | PFoil:two-level@25 (micro distribution voting) | 76.31% | 1758.1 |
| diabetes | PFoil:two-level@25 (voting) | 75.91% | 1758.1 |
| diabetes | PFoil:two-level@50 (best rule) | 77.34% | 3522.2 |
| diabetes | PFoil:two-level@50 (macro distribution voting) | 77.08% | 3522.2 |
| diabetes | PFoil:two-level@50 (micro distribution voting) | 75.13% | 3522.2 |
| diabetes | PFoil:two-level@50 (voting) | 76.30% | 3522.2 |
| diabetes | PFossil:one-level@10 (best rule) | 72.79% | 127.4 |
| diabetes | PFossil:one-level@10 (macro distribution voting) | 74.34% | 127.4 |
| diabetes | PFossil:one-level@10 (micro distribution voting) | 75.00% | 127.4 |
| diabetes | PFossil:one-level@10 (voting) | 76.04% | 127.4 |
| diabetes | PFossil:one-level@25 (best rule) | 74.08% | 316.1 |
| diabetes | PFossil:one-level@25 (macro distribution voting) | 75.91% | 316.1 |
| diabetes | PFossil:one-level@25 (micro distribution voting) | 76.04% | 316.1 |
| diabetes | PFossil:one-level@25 (voting) | 75.39% | 316.1 |
| diabetes | PFossil:one-level@50 (best rule) | 73.69% | 661.5 |
| diabetes | PFossil:one-level@50 (macro distribution voting) | 75.52% | 661.5 |
| diabetes | PFossil:one-level@50 (micro distribution voting) | 75.91% | 661.5 |
| diabetes | PFossil:one-level@50 (voting) | 75.12% | 661.5 |
| diabetes | PFossil:single (best rule) | 74.61% | 9.1 |
| diabetes | PFossil:single (macro distribution voting) | 73.83% | 9.1 |
| diabetes | PFossil:single (micro distribution voting) | 75.13% | 9.1 |
| diabetes | PFossil:single (voting) | 74.09% | 9.1 |
| diabetes | PFossil:two-level@10 (best rule) | 75.39% | 127.4 |
| diabetes | PFossil:two-level@10 (macro distribution voting) | 74.74% | 127.4 |
| diabetes | PFossil:two-level@10 (micro distribution voting) | 75.26% | 127.4 |
| diabetes | PFossil:two-level@10 (voting) | 74.99% | 127.4 |
| diabetes | PFossil:two-level@25 (best rule) | 75.26% | 316.1 |
| diabetes | PFossil:two-level@25 (macro distribution voting) | 75.91% | 316.1 |
| diabetes | PFossil:two-level@25 (micro distribution voting) | 76.44% | 316.1 |
| diabetes | PFossil:two-level@25 (voting) | 75.13% | 316.1 |
| diabetes | PFossil:two-level@50 (best rule) | 76.17% | 661.5 |
| diabetes | PFossil:two-level@50 (macro distribution voting) | 75.65% | 661.5 |
| diabetes | PFossil:two-level@50 (micro distribution voting) | 76.05% | 661.5 |
| diabetes | PFossil:two-level@50 (voting) | 75.52% | 661.5 |
| diabetes | Pypper:one-level@10 (best rule) | 74.98% | 88.8 |
| diabetes | Pypper:one-level@10 (macro distribution voting) | 74.22% | 88.8 |
| diabetes | Pypper:one-level@10 (micro distribution voting) | 75.65% | 88.8 |
| diabetes | Pypper:one-level@10 (voting) | 74.87% | 88.8 |
| diabetes | Pypper:one-level@25 (best rule) | 74.73% | 210.3 |
| diabetes | Pypper:one-level@25 (macro distribution voting) | 74.61% | 210.3 |
| diabetes | Pypper:one-level@25 (micro distribution voting) | 74.61% | 210.3 |
| diabetes | Pypper:one-level@25 (voting) | 74.73% | 210.3 |
| diabetes | Pypper:one-level@50 (best rule) | 74.73% | 418.5 |
| diabetes | Pypper:one-level@50 (macro distribution voting) | 75.00% | 418.5 |
| diabetes | Pypper:one-level@50 (micro distribution voting) | 74.60% | 418.5 |
| diabetes | Pypper:one-level@50 (voting) | 74.35% | 418.5 |
| diabetes | Pypper:single (best rule) | 74.61% | 7.7 |
| diabetes | Pypper:single (macro distribution voting) | 74.48% | 7.7 |
| diabetes | Pypper:single (micro distribution voting) | 74.48% | 7.7 |
| diabetes | Pypper:single (voting) | 74.35% | 7.7 |
| diabetes | Pypper:two-level@10 (best rule) | 74.61% | 88.8 |
| diabetes | Pypper:two-level@10 (macro distribution voting) | 74.74% | 88.8 |
| diabetes | Pypper:two-level@10 (micro distribution voting) | 74.48% | 88.8 |
| diabetes | Pypper:two-level@10 (voting) | 74.87% | 88.8 |
| diabetes | Pypper:two-level@25 (best rule) | 74.73% | 210.3 |
| diabetes | Pypper:two-level@25 (macro distribution voting) | 74.48% | 210.3 |
| diabetes | Pypper:two-level@25 (micro distribution voting) | 74.34% | 210.3 |
| diabetes | Pypper:two-level@25 (voting) | 74.35% | 210.3 |
| diabetes | Pypper:two-level@50 (best rule) | 74.21% | 418.5 |
| diabetes | Pypper:two-level@50 (macro distribution voting) | 74.48% | 418.5 |
| diabetes | Pypper:two-level@50 (micro distribution voting) | 74.08% | 418.5 |
| diabetes | Pypper:two-level@50 (voting) | 74.48% | 418.5 |
| dna | PFoil:one-level@10 (best rule) | 92.91% | 699.4 |
| dna | PFoil:one-level@10 (macro distribution voting) | 95.13% | 699.4 |
| dna | PFoil:one-level@10 (micro distribution voting) | 94.76% | 699.4 |
| dna | PFoil:one-level@10 (voting) | 95.01% | 699.4 |
| dna | PFoil:one-level@25 (best rule) | 92.88% | 1752.6 |
| dna | PFoil:one-level@25 (macro distribution voting) | 95.39% | 1752.6 |
| dna | PFoil:one-level@25 (micro distribution voting) | 94.82% | 1752.6 |
| dna | PFoil:one-level@25 (voting) | 95.29% | 1752.6 |
| dna | PFoil:one-level@50 (best rule) | 91.97% | 3508.6 |
| dna | PFoil:one-level@50 (macro distribution voting) | 95.32% | 3508.6 |
| dna | PFoil:one-level@50 (micro distribution voting) | 95.10% | 3508.6 |
| dna | PFoil:one-level@50 (voting) | 95.29% | 3508.6 |
| dna | PFoil:single (best rule) | 92.50% | 89.0 |
| dna | PFoil:single (macro distribution voting) | 91.97% | 89.0 |
| dna | PFoil:single (micro distribution voting) | 92.00% | 89.0 |
| dna | PFoil:single (voting) | 91.05% | 89.0 |
| dna | PFoil:two-level@10 (best rule) | 94.73% | 699.4 |
| dna | PFoil:two-level@10 (macro distribution voting) | 94.95% | 699.4 |
| dna | PFoil:two-level@10 (micro distribution voting) | 94.69% | 699.4 |
| dna | PFoil:two-level@10 (voting) | 95.04% | 699.4 |
| dna | PFoil:two-level@25 (best rule) | 95.01% | 1752.6 |
| dna | PFoil:two-level@25 (macro distribution voting) | 95.45% | 1752.6 |
| dna | PFoil:two-level@25 (micro distribution voting) | 95.07% | 1752.6 |
| dna | PFoil:two-level@25 (voting) | 95.42% | 1752.6 |
| dna | PFoil:two-level@50 (best rule) | 95.29% | 3508.6 |
| dna | PFoil:two-level@50 (macro distribution voting) | 95.64% | 3508.6 |
| dna | PFoil:two-level@50 (micro distribution voting) | 95.48% | 3508.6 |
| dna | PFoil:two-level@50 (voting) | 95.61% | 3508.6 |
| dna | PFossil:one-level@10 (best rule) | 90.58% | 327.8 |
| dna | PFossil:one-level@10 (macro distribution voting) | 95.01% | 327.8 |
| dna | PFossil:one-level@10 (micro distribution voting) | 89.96% | 327.8 |
| dna | PFossil:one-level@10 (voting) | 92.78% | 327.8 |
| dna | PFossil:one-level@25 (best rule) | 89.08% | 824.6 |
| dna | PFossil:one-level@25 (macro distribution voting) | 95.54% | 824.6 |
| dna | PFossil:one-level@25 (micro distribution voting) | 90.11% | 824.6 |
| dna | PFossil:one-level@25 (voting) | 93.94% | 824.6 |
| dna | PFossil:one-level@50 (best rule) | 87.95% | 1657.7 |
| dna | PFossil:one-level@50 (macro distribution voting) | 95.57% | 1657.7 |
| dna | PFossil:one-level@50 (micro distribution voting) | 90.05% | 1657.7 |
| dna | PFossil:one-level@50 (voting) | 94.51% | 1657.7 |
| dna | PFossil:single (best rule) | 93.69% | 27.8 |
| dna | PFossil:single (macro distribution voting) | 93.53% | 27.8 |
| dna | PFossil:single (micro distribution voting) | 89.89% | 27.8 |
| dna | PFossil:single (voting) | 90.65% | 27.8 |
| dna | PFossil:two-level@10 (best rule) | 94.91% | 327.8 |
| dna | PFossil:two-level@10 (macro distribution voting) | 94.85% | 327.8 |
| dna | PFossil:two-level@10 (micro distribution voting) | 90.08% | 327.8 |
| dna | PFossil:two-level@10 (voting) | 90.43% | 327.8 |
| dna | PFossil:two-level@25 (best rule) | 95.45% | 824.6 |
| dna | PFossil:two-level@25 (macro distribution voting) | 95.48% | 824.6 |
| dna | PFossil:two-level@25 (micro distribution voting) | 90.55% | 824.6 |
| dna | PFossil:two-level@25 (voting) | 90.99% | 824.6 |
| dna | PFossil:two-level@50 (best rule) | 95.42% | 1657.7 |
| dna | PFossil:two-level@50 (macro distribution voting) | 95.57% | 1657.7 |
| dna | PFossil:two-level@50 (micro distribution voting) | 90.58% | 1657.7 |
| dna | PFossil:two-level@50 (voting) | 91.18% | 1657.7 |
| dna | Pypper:one-level@10 (best rule) | 92.50% | 266.0 |
| dna | Pypper:one-level@10 (macro distribution voting) | 93.69% | 266.0 |
| dna | Pypper:one-level@10 (micro distribution voting) | 91.90% | 266.0 |
| dna | Pypper:one-level@10 (voting) | 93.53% | 266.0 |
| dna | Pypper:one-level@25 (best rule) | 92.72% | 653.9 |
| dna | Pypper:one-level@25 (macro distribution voting) | 94.07% | 653.9 |
| dna | Pypper:one-level@25 (micro distribution voting) | 92.25% | 653.9 |
| dna | Pypper:one-level@25 (voting) | 93.88% | 653.9 |
| dna | Pypper:one-level@50 (best rule) | 93.03% | 1325.0 |
| dna | Pypper:one-level@50 (macro distribution voting) | 94.13% | 1325.0 |
| dna | Pypper:one-level@50 (micro distribution voting) | 92.34% | 1325.0 |
| dna | Pypper:one-level@50 (voting) | 93.97% | 1325.0 |
| dna | Pypper:single (best rule) | 93.13% | 29.0 |
| dna | Pypper:single (macro distribution voting) | 93.13% | 29.0 |
| dna | Pypper:single (micro distribution voting) | 93.16% | 29.0 |
| dna | Pypper:single (voting) | 93.22% | 29.0 |
| dna | Pypper:two-level@10 (best rule) | 93.82% | 266.0 |
| dna | Pypper:two-level@10 (macro distribution voting) | 93.60% | 266.0 |
| dna | Pypper:two-level@10 (micro distribution voting) | 92.69% | 266.0 |
| dna | Pypper:two-level@10 (voting) | 93.60% | 266.0 |
| dna | Pypper:two-level@25 (best rule) | 94.22% | 653.9 |
| dna | Pypper:two-level@25 (macro distribution voting) | 94.23% | 653.9 |
| dna | Pypper:two-level@25 (micro distribution voting) | 92.91% | 653.9 |
| dna | Pypper:two-level@25 (voting) | 93.97% | 653.9 |
| dna | Pypper:two-level@50 (best rule) | 94.19% | 1325.0 |
| dna | Pypper:two-level@50 (macro distribution voting) | 94.10% | 1325.0 |
| dna | Pypper:two-level@50 (micro distribution voting) | 93.13% | 1325.0 |
| dna | Pypper:two-level@50 (voting) | 94.07% | 1325.0 |
| dresses-sales | PFoil:one-level@10 (best rule) | 56.60% | 598.7 |
| dresses-sales | PFoil:one-level@10 (macro distribution voting) | 58.60% | 598.7 |
| dresses-sales | PFoil:one-level@10 (micro distribution voting) | 61.00% | 598.7 |
| dresses-sales | PFoil:one-level@10 (voting) | 59.80% | 598.7 |
| dresses-sales | PFoil:one-level@25 (best rule) | 59.40% | 1491.2 |
| dresses-sales | PFoil:one-level@25 (macro distribution voting) | 59.40% | 1491.2 |
| dresses-sales | PFoil:one-level@25 (micro distribution voting) | 61.00% | 1491.2 |
| dresses-sales | PFoil:one-level@25 (voting) | 58.80% | 1491.2 |
| dresses-sales | PFoil:one-level@50 (best rule) | 58.80% | 2981.6 |
| dresses-sales | PFoil:one-level@50 (macro distribution voting) | 60.20% | 2981.6 |
| dresses-sales | PFoil:one-level@50 (micro distribution voting) | 60.00% | 2981.6 |
| dresses-sales | PFoil:one-level@50 (voting) | 60.00% | 2981.6 |
| dresses-sales | PFoil:single (best rule) | 58.20% | 67.1 |
| dresses-sales | PFoil:single (macro distribution voting) | 59.00% | 67.1 |
| dresses-sales | PFoil:single (micro distribution voting) | 60.00% | 67.1 |
| dresses-sales | PFoil:single (voting) | 56.80% | 67.1 |
| dresses-sales | PFoil:two-level@10 (best rule) | 58.60% | 598.7 |
| dresses-sales | PFoil:two-level@10 (macro distribution voting) | 60.80% | 598.7 |
| dresses-sales | PFoil:two-level@10 (micro distribution voting) | 60.80% | 598.7 |
| dresses-sales | PFoil:two-level@10 (voting) | 58.00% | 598.7 |
| dresses-sales | PFoil:two-level@25 (best rule) | 62.80% | 1491.2 |
| dresses-sales | PFoil:two-level@25 (macro distribution voting) | 62.80% | 1491.2 |
| dresses-sales | PFoil:two-level@25 (micro distribution voting) | 59.40% | 1491.2 |
| dresses-sales | PFoil:two-level@25 (voting) | 59.20% | 1491.2 |
| dresses-sales | PFoil:two-level@50 (best rule) | 61.80% | 2981.6 |
| dresses-sales | PFoil:two-level@50 (macro distribution voting) | 60.40% | 2981.6 |
| dresses-sales | PFoil:two-level@50 (micro distribution voting) | 59.80% | 2981.6 |
| dresses-sales | PFoil:two-level@50 (voting) | 59.20% | 2981.6 |
| dresses-sales | PFossil:one-level@10 (best rule) | 59.60% | 46.4 |
| dresses-sales | PFossil:one-level@10 (macro distribution voting) | 61.40% | 46.4 |
| dresses-sales | PFossil:one-level@10 (micro distribution voting) | 59.00% | 46.4 |
| dresses-sales | PFossil:one-level@10 (voting) | 60.20% | 46.4 |
| dresses-sales | PFossil:one-level@25 (best rule) | 61.20% | 118.3 |
| dresses-sales | PFossil:one-level@25 (macro distribution voting) | 62.00% | 118.3 |
| dresses-sales | PFossil:one-level@25 (micro distribution voting) | 59.40% | 118.3 |
| dresses-sales | PFossil:one-level@25 (voting) | 61.00% | 118.3 |
| dresses-sales | PFossil:one-level@50 (best rule) | 58.60% | 250.9 |
| dresses-sales | PFossil:one-level@50 (macro distribution voting) | 63.20% | 250.9 |
| dresses-sales | PFossil:one-level@50 (micro distribution voting) | 60.00% | 250.9 |
| dresses-sales | PFossil:one-level@50 (voting) | 62.20% | 250.9 |
| dresses-sales | PFossil:single (best rule) | 59.40% | 2.2 |
| dresses-sales | PFossil:single (macro distribution voting) | 59.40% | 2.2 |
| dresses-sales | PFossil:single (micro distribution voting) | 59.20% | 2.2 |
| dresses-sales | PFossil:single (voting) | 59.20% | 2.2 |
| dresses-sales | PFossil:two-level@10 (best rule) | 59.40% | 46.4 |
| dresses-sales | PFossil:two-level@10 (macro distribution voting) | 59.60% | 46.4 |
| dresses-sales | PFossil:two-level@10 (micro distribution voting) | 60.20% | 46.4 |
| dresses-sales | PFossil:two-level@10 (voting) | 60.60% | 46.4 |
| dresses-sales | PFossil:two-level@25 (best rule) | 61.00% | 118.3 |
| dresses-sales | PFossil:two-level@25 (macro distribution voting) | 61.40% | 118.3 |
| dresses-sales | PFossil:two-level@25 (micro distribution voting) | 60.60% | 118.3 |
| dresses-sales | PFossil:two-level@25 (voting) | 60.40% | 118.3 |
| dresses-sales | PFossil:two-level@50 (best rule) | 62.00% | 250.9 |
| dresses-sales | PFossil:two-level@50 (macro distribution voting) | 61.40% | 250.9 |
| dresses-sales | PFossil:two-level@50 (micro distribution voting) | 61.40% | 250.9 |
| dresses-sales | PFossil:two-level@50 (voting) | 61.40% | 250.9 |
| dresses-sales | Pypper:one-level@10 (best rule) | 59.60% | 47.4 |
| dresses-sales | Pypper:one-level@10 (macro distribution voting) | 61.60% | 47.4 |
| dresses-sales | Pypper:one-level@10 (micro distribution voting) | 59.40% | 47.4 |
| dresses-sales | Pypper:one-level@10 (voting) | 62.60% | 47.4 |
| dresses-sales | Pypper:one-level@25 (best rule) | 58.80% | 120.3 |
| dresses-sales | Pypper:one-level@25 (macro distribution voting) | 61.80% | 120.3 |
| dresses-sales | Pypper:one-level@25 (micro distribution voting) | 61.00% | 120.3 |
| dresses-sales | Pypper:one-level@25 (voting) | 62.20% | 120.3 |
| dresses-sales | Pypper:one-level@50 (best rule) | 60.00% | 237.6 |
| dresses-sales | Pypper:one-level@50 (macro distribution voting) | 63.00% | 237.6 |
| dresses-sales | Pypper:one-level@50 (micro distribution voting) | 60.40% | 237.6 |
| dresses-sales | Pypper:one-level@50 (voting) | 61.40% | 237.6 |
| dresses-sales | Pypper:single (best rule) | 61.00% | 1.8 |
| dresses-sales | Pypper:single (macro distribution voting) | 61.00% | 1.8 |
| dresses-sales | Pypper:single (micro distribution voting) | 60.80% | 1.8 |
| dresses-sales | Pypper:single (voting) | 60.80% | 1.8 |
| dresses-sales | Pypper:two-level@10 (best rule) | 62.40% | 47.4 |
| dresses-sales | Pypper:two-level@10 (macro distribution voting) | 62.80% | 47.4 |
| dresses-sales | Pypper:two-level@10 (micro distribution voting) | 62.80% | 47.4 |
| dresses-sales | Pypper:two-level@10 (voting) | 61.60% | 47.4 |
| dresses-sales | Pypper:two-level@25 (best rule) | 62.00% | 120.3 |
| dresses-sales | Pypper:two-level@25 (macro distribution voting) | 62.60% | 120.3 |
| dresses-sales | Pypper:two-level@25 (micro distribution voting) | 61.40% | 120.3 |
| dresses-sales | Pypper:two-level@25 (voting) | 60.60% | 120.3 |
| dresses-sales | Pypper:two-level@50 (best rule) | 62.80% | 237.6 |
| dresses-sales | Pypper:two-level@50 (macro distribution voting) | 62.40% | 237.6 |
| dresses-sales | Pypper:two-level@50 (micro distribution voting) | 60.80% | 237.6 |
| dresses-sales | Pypper:two-level@50 (voting) | 60.80% | 237.6 |
| ecoli | PFoil:one-level@10 (best rule) | 82.47% | 342.2 |
| ecoli | PFoil:one-level@10 (macro distribution voting) | 86.65% | 342.2 |
| ecoli | PFoil:one-level@10 (micro distribution voting) | 85.16% | 342.2 |
| ecoli | PFoil:one-level@10 (voting) | 84.57% | 342.2 |
| ecoli | PFoil:one-level@25 (best rule) | 81.26% | 849.4 |
| ecoli | PFoil:one-level@25 (macro distribution voting) | 86.35% | 849.4 |
| ecoli | PFoil:one-level@25 (micro distribution voting) | 84.56% | 849.4 |
| ecoli | PFoil:one-level@25 (voting) | 84.57% | 849.4 |
| ecoli | PFoil:one-level@50 (best rule) | 80.96% | 1705.1 |
| ecoli | PFoil:one-level@50 (macro distribution voting) | 86.35% | 1705.1 |
| ecoli | PFoil:one-level@50 (micro distribution voting) | 84.86% | 1705.1 |
| ecoli | PFoil:one-level@50 (voting) | 84.88% | 1705.1 |
| ecoli | PFoil:single (best rule) | 82.51% | 42.6 |
| ecoli | PFoil:single (macro distribution voting) | 83.10% | 42.6 |
| ecoli | PFoil:single (micro distribution voting) | 79.22% | 42.6 |
| ecoli | PFoil:single (voting) | 79.80% | 42.6 |
| ecoli | PFoil:two-level@10 (best rule) | 84.57% | 342.2 |
| ecoli | PFoil:two-level@10 (macro distribution voting) | 85.16% | 342.2 |
| ecoli | PFoil:two-level@10 (micro distribution voting) | 84.55% | 342.2 |
| ecoli | PFoil:two-level@10 (voting) | 85.47% | 342.2 |
| ecoli | PFoil:two-level@25 (best rule) | 85.46% | 849.4 |
| ecoli | PFoil:two-level@25 (macro distribution voting) | 86.05% | 849.4 |
| ecoli | PFoil:two-level@25 (micro distribution voting) | 84.55% | 849.4 |
| ecoli | PFoil:two-level@25 (voting) | 86.35% | 849.4 |
| ecoli | PFoil:two-level@50 (best rule) | 85.46% | 1705.1 |
| ecoli | PFoil:two-level@50 (macro distribution voting) | 86.05% | 1705.1 |
| ecoli | PFoil:two-level@50 (micro distribution voting) | 84.87% | 1705.1 |
| ecoli | PFoil:two-level@50 (voting) | 85.46% | 1705.1 |
| ecoli | PFossil:one-level@10 (best rule) | 80.99% | 240.8 |
| ecoli | PFossil:one-level@10 (macro distribution voting) | 86.38% | 240.8 |
| ecoli | PFossil:one-level@10 (micro distribution voting) | 85.18% | 240.8 |
| ecoli | PFossil:one-level@10 (voting) | 86.97% | 240.8 |
| ecoli | PFossil:one-level@25 (best rule) | 80.69% | 594.0 |
| ecoli | PFossil:one-level@25 (macro distribution voting) | 86.66% | 594.0 |
| ecoli | PFossil:one-level@25 (micro distribution voting) | 85.18% | 594.0 |
| ecoli | PFossil:one-level@25 (voting) | 86.66% | 594.0 |
| ecoli | PFossil:one-level@50 (best rule) | 80.39% | 1183.1 |
| ecoli | PFossil:one-level@50 (macro distribution voting) | 86.36% | 1183.1 |
| ecoli | PFossil:one-level@50 (micro distribution voting) | 85.17% | 1183.1 |
| ecoli | PFossil:one-level@50 (voting) | 86.67% | 1183.1 |
| ecoli | PFossil:single (best rule) | 81.92% | 30.0 |
| ecoli | PFossil:single (macro distribution voting) | 82.22% | 30.0 |
| ecoli | PFossil:single (micro distribution voting) | 84.29% | 30.0 |
| ecoli | PFossil:single (voting) | 83.98% | 30.0 |
| ecoli | PFossil:two-level@10 (best rule) | 85.19% | 240.8 |
| ecoli | PFossil:two-level@10 (macro distribution voting) | 86.08% | 240.8 |
| ecoli | PFossil:two-level@10 (micro distribution voting) | 86.08% | 240.8 |
| ecoli | PFossil:two-level@10 (voting) | 86.07% | 240.8 |
| ecoli | PFossil:two-level@25 (best rule) | 86.07% | 594.0 |
| ecoli | PFossil:two-level@25 (macro distribution voting) | 85.78% | 594.0 |
| ecoli | PFossil:two-level@25 (micro distribution voting) | 86.07% | 594.0 |
| ecoli | PFossil:two-level@25 (voting) | 86.35% | 594.0 |
| ecoli | PFossil:two-level@50 (best rule) | 85.18% | 1183.1 |
| ecoli | PFossil:two-level@50 (macro distribution voting) | 86.07% | 1183.1 |
| ecoli | PFossil:two-level@50 (micro distribution voting) | 85.18% | 1183.1 |
| ecoli | PFossil:two-level@50 (voting) | 86.35% | 1183.1 |
| ecoli | Pypper:one-level@10 (best rule) | 83.07% | 109.5 |
| ecoli | Pypper:one-level@10 (macro distribution voting) | 86.07% | 109.5 |
| ecoli | Pypper:one-level@10 (micro distribution voting) | 83.66% | 109.5 |
| ecoli | Pypper:one-level@10 (voting) | 86.67% | 109.5 |
| ecoli | Pypper:one-level@25 (best rule) | 82.77% | 272.9 |
| ecoli | Pypper:one-level@25 (macro distribution voting) | 86.68% | 272.9 |
| ecoli | Pypper:one-level@25 (micro distribution voting) | 82.44% | 272.9 |
| ecoli | Pypper:one-level@25 (voting) | 86.38% | 272.9 |
| ecoli | Pypper:one-level@50 (best rule) | 83.98% | 542.3 |
| ecoli | Pypper:one-level@50 (macro distribution voting) | 86.37% | 542.3 |
| ecoli | Pypper:one-level@50 (micro distribution voting) | 83.38% | 542.3 |
| ecoli | Pypper:one-level@50 (voting) | 85.47% | 542.3 |
| ecoli | Pypper:single (best rule) | 82.22% | 10.7 |
| ecoli | Pypper:single (macro distribution voting) | 81.91% | 10.7 |
| ecoli | Pypper:single (micro distribution voting) | 81.60% | 10.7 |
| ecoli | Pypper:single (voting) | 81.60% | 10.7 |
| ecoli | Pypper:two-level@10 (best rule) | 85.77% | 109.5 |
| ecoli | Pypper:two-level@10 (macro distribution voting) | 86.08% | 109.5 |
| ecoli | Pypper:two-level@10 (micro distribution voting) | 84.55% | 109.5 |
| ecoli | Pypper:two-level@10 (voting) | 85.46% | 109.5 |
| ecoli | Pypper:two-level@25 (best rule) | 86.37% | 272.9 |
| ecoli | Pypper:two-level@25 (macro distribution voting) | 86.68% | 272.9 |
| ecoli | Pypper:two-level@25 (micro distribution voting) | 85.77% | 272.9 |
| ecoli | Pypper:two-level@25 (voting) | 86.37% | 272.9 |
| ecoli | Pypper:two-level@50 (best rule) | 86.67% | 542.3 |
| ecoli | Pypper:two-level@50 (macro distribution voting) | 86.07% | 542.3 |
| ecoli | Pypper:two-level@50 (micro distribution voting) | 85.76% | 542.3 |
| ecoli | Pypper:two-level@50 (voting) | 86.95% | 542.3 |
| glass | PFoil:one-level@10 (best rule) | 70.67% | 279.3 |
| glass | PFoil:one-level@10 (macro distribution voting) | 74.81% | 279.3 |
| glass | PFoil:one-level@10 (micro distribution voting) | 69.26% | 279.3 |
| glass | PFoil:one-level@10 (voting) | 74.78% | 279.3 |
| glass | PFoil:one-level@25 (best rule) | 68.33% | 697.4 |
| glass | PFoil:one-level@25 (macro distribution voting) | 75.74% | 697.4 |
| glass | PFoil:one-level@25 (micro distribution voting) | 70.67% | 697.4 |
| glass | PFoil:one-level@25 (voting) | 76.19% | 697.4 |
| glass | PFoil:one-level@50 (best rule) | 65.09% | 1404.5 |
| glass | PFoil:one-level@50 (macro distribution voting) | 75.28% | 1404.5 |
| glass | PFoil:one-level@50 (micro distribution voting) | 69.26% | 1404.5 |
| glass | PFoil:one-level@50 (voting) | 74.76% | 1404.5 |
| glass | PFoil:single (best rule) | 72.94% | 34.6 |
| glass | PFoil:single (macro distribution voting) | 72.47% | 34.6 |
| glass | PFoil:single (micro distribution voting) | 66.00% | 34.6 |
| glass | PFoil:single (voting) | 67.38% | 34.6 |
| glass | PFoil:two-level@10 (best rule) | 75.24% | 279.3 |
| glass | PFoil:two-level@10 (macro distribution voting) | 73.90% | 279.3 |
| glass | PFoil:two-level@10 (micro distribution voting) | 72.53% | 279.3 |
| glass | PFoil:two-level@10 (voting) | 73.01% | 279.3 |
| glass | PFoil:two-level@25 (best rule) | 75.74% | 697.4 |
| glass | PFoil:two-level@25 (macro distribution voting) | 76.23% | 697.4 |
| glass | PFoil:two-level@25 (micro distribution voting) | 73.46% | 697.4 |
| glass | PFoil:two-level@25 (voting) | 73.92% | 697.4 |
| glass | PFoil:two-level@50 (best rule) | 75.74% | 1404.5 |
| glass | PFoil:two-level@50 (macro distribution voting) | 75.28% | 1404.5 |
| glass | PFoil:two-level@50 (micro distribution voting) | 73.48% | 1404.5 |
| glass | PFoil:two-level@50 (voting) | 72.53% | 1404.5 |
| glass | PFossil:one-level@10 (best rule) | 66.36% | 238.9 |
| glass | PFossil:one-level@10 (macro distribution voting) | 73.38% | 238.9 |
| glass | PFossil:one-level@10 (micro distribution voting) | 72.03% | 238.9 |
| glass | PFossil:one-level@10 (voting) | 73.38% | 238.9 |
| glass | PFossil:one-level@25 (best rule) | 70.15% | 593.8 |
| glass | PFossil:one-level@25 (macro distribution voting) | 72.94% | 593.8 |
| glass | PFossil:one-level@25 (micro distribution voting) | 72.92% | 593.8 |
| glass | PFossil:one-level@25 (voting) | 72.47% | 593.8 |
| glass | PFossil:one-level@50 (best rule) | 65.87% | 1211.7 |
| glass | PFossil:one-level@50 (macro distribution voting) | 71.54% | 1211.7 |
| glass | PFossil:one-level@50 (micro distribution voting) | 70.56% | 1211.7 |
| glass | PFossil:one-level@50 (voting) | 73.38% | 1211.7 |
| glass | PFossil:single (best rule) | 71.97% | 31.2 |
| glass | PFossil:single (macro distribution voting) | 70.58% | 31.2 |
| glass | PFossil:single (micro distribution voting) | 70.11% | 31.2 |
| glass | PFossil:single (voting) | 69.68% | 31.2 |
| glass | PFossil:two-level@10 (best rule) | 72.92% | 238.9 |
| glass | PFossil:two-level@10 (macro distribution voting) | 73.85% | 238.9 |
| glass | PFossil:two-level@10 (micro distribution voting) | 74.81% | 238.9 |
| glass | PFossil:two-level@10 (voting) | 72.90% | 238.9 |
| glass | PFossil:two-level@25 (best rule) | 73.38% | 593.8 |
| glass | PFossil:two-level@25 (macro distribution voting) | 72.42% | 593.8 |
| glass | PFossil:two-level@25 (micro distribution voting) | 74.33% | 593.8 |
| glass | PFossil:two-level@25 (voting) | 71.52% | 593.8 |
| glass | PFossil:two-level@50 (best rule) | 72.47% | 1211.7 |
| glass | PFossil:two-level@50 (macro distribution voting) | 72.01% | 1211.7 |
| glass | PFossil:two-level@50 (micro distribution voting) | 73.85% | 1211.7 |
| glass | PFossil:two-level@50 (voting) | 72.45% | 1211.7 |
| glass | Pypper:one-level@10 (best rule) | 69.70% | 106.7 |
| glass | Pypper:one-level@10 (macro distribution voting) | 72.01% | 106.7 |
| glass | Pypper:one-level@10 (micro distribution voting) | 66.43% | 106.7 |
| glass | Pypper:one-level@10 (voting) | 72.49% | 106.7 |
| glass | Pypper:one-level@25 (best rule) | 67.84% | 279.9 |
| glass | Pypper:one-level@25 (macro distribution voting) | 74.39% | 279.9 |
| glass | Pypper:one-level@25 (micro distribution voting) | 69.26% | 279.9 |
| glass | Pypper:one-level@25 (voting) | 75.78% | 279.9 |
| glass | Pypper:one-level@50 (best rule) | 67.38% | 553.1 |
| glass | Pypper:one-level@50 (macro distribution voting) | 75.32% | 553.1 |
| glass | Pypper:one-level@50 (micro distribution voting) | 69.70% | 553.1 |
| glass | Pypper:one-level@50 (voting) | 76.23% | 553.1 |
| glass | Pypper:single (best rule) | 67.77% | 9.6 |
| glass | Pypper:single (macro distribution voting) | 65.91% | 9.6 |
| glass | Pypper:single (micro distribution voting) | 67.32% | 9.6 |
| glass | Pypper:single (voting) | 67.27% | 9.6 |
| glass | Pypper:two-level@10 (best rule) | 71.58% | 106.7 |
| glass | Pypper:two-level@10 (macro distribution voting) | 71.58% | 106.7 |
| glass | Pypper:two-level@10 (micro distribution voting) | 68.83% | 106.7 |
| glass | Pypper:two-level@10 (voting) | 70.17% | 106.7 |
| glass | Pypper:two-level@25 (best rule) | 73.44% | 279.9 |
| glass | Pypper:two-level@25 (macro distribution voting) | 73.92% | 279.9 |
| glass | Pypper:two-level@25 (micro distribution voting) | 72.53% | 279.9 |
| glass | Pypper:two-level@25 (voting) | 72.55% | 279.9 |
| glass | Pypper:two-level@50 (best rule) | 73.46% | 553.1 |
| glass | Pypper:two-level@50 (macro distribution voting) | 74.87% | 553.1 |
| glass | Pypper:two-level@50 (micro distribution voting) | 72.49% | 553.1 |
| glass | Pypper:two-level@50 (voting) | 73.94% | 553.1 |
| hayes-roth | PFoil:one-level@10 (best rule) | 77.50% | 229.1 |
| hayes-roth | PFoil:one-level@10 (macro distribution voting) | 78.75% | 229.1 |
| hayes-roth | PFoil:one-level@10 (micro distribution voting) | 79.38% | 229.1 |
| hayes-roth | PFoil:one-level@10 (voting) | 80.00% | 229.1 |
| hayes-roth | PFoil:one-level@25 (best rule) | 76.25% | 565.3 |
| hayes-roth | PFoil:one-level@25 (macro distribution voting) | 80.00% | 565.3 |
| hayes-roth | PFoil:one-level@25 (micro distribution voting) | 80.00% | 565.3 |
| hayes-roth | PFoil:one-level@25 (voting) | 78.12% | 565.3 |
| hayes-roth | PFoil:one-level@50 (best rule) | 76.25% | 1131.7 |
| hayes-roth | PFoil:one-level@50 (macro distribution voting) | 76.88% | 1131.7 |
| hayes-roth | PFoil:one-level@50 (micro distribution voting) | 78.12% | 1131.7 |
| hayes-roth | PFoil:one-level@50 (voting) | 75.62% | 1131.7 |
| hayes-roth | PFoil:single (best rule) | 77.50% | 25.8 |
| hayes-roth | PFoil:single (macro distribution voting) | 72.50% | 25.8 |
| hayes-roth | PFoil:single (micro distribution voting) | 76.88% | 25.8 |
| hayes-roth | PFoil:single (voting) | 70.62% | 25.8 |
| hayes-roth | PFoil:two-level@10 (best rule) | 81.88% | 229.1 |
| hayes-roth | PFoil:two-level@10 (macro distribution voting) | 76.88% | 229.1 |
| hayes-roth | PFoil:two-level@10 (micro distribution voting) | 81.88% | 229.1 |
| hayes-roth | PFoil:two-level@10 (voting) | 78.12% | 229.1 |
| hayes-roth | PFoil:two-level@25 (best rule) | 81.25% | 565.3 |
| hayes-roth | PFoil:two-level@25 (macro distribution voting) | 76.88% | 565.3 |
| hayes-roth | PFoil:two-level@25 (micro distribution voting) | 81.25% | 565.3 |
| hayes-roth | PFoil:two-level@25 (voting) | 75.62% | 565.3 |
| hayes-roth | PFoil:two-level@50 (best rule) | 79.38% | 1131.7 |
| hayes-roth | PFoil:two-level@50 (macro distribution voting) | 75.62% | 1131.7 |
| hayes-roth | PFoil:two-level@50 (micro distribution voting) | 78.12% | 1131.7 |
| hayes-roth | PFoil:two-level@50 (voting) | 74.38% | 1131.7 |
| hayes-roth | PFossil:one-level@10 (best rule) | 78.75% | 139.2 |
| hayes-roth | PFossil:one-level@10 (macro distribution voting) | 85.00% | 139.2 |
| hayes-roth | PFossil:one-level@10 (micro distribution voting) | 80.00% | 139.2 |
| hayes-roth | PFossil:one-level@10 (voting) | 84.38% | 139.2 |
| hayes-roth | PFossil:one-level@25 (best rule) | 78.12% | 337.0 |
| hayes-roth | PFossil:one-level@25 (macro distribution voting) | 83.75% | 337.0 |
| hayes-roth | PFossil:one-level@25 (micro distribution voting) | 80.00% | 337.0 |
| hayes-roth | PFossil:one-level@25 (voting) | 83.75% | 337.0 |
| hayes-roth | PFossil:one-level@50 (best rule) | 79.38% | 675.3 |
| hayes-roth | PFossil:one-level@50 (macro distribution voting) | 84.38% | 675.3 |
| hayes-roth | PFossil:one-level@50 (micro distribution voting) | 81.25% | 675.3 |
| hayes-roth | PFossil:one-level@50 (voting) | 85.00% | 675.3 |
| hayes-roth | PFossil:single (best rule) | 82.50% | 13.5 |
| hayes-roth | PFossil:single (macro distribution voting) | 81.88% | 13.5 |
| hayes-roth | PFossil:single (micro distribution voting) | 81.88% | 13.5 |
| hayes-roth | PFossil:single (voting) | 81.25% | 13.5 |
| hayes-roth | PFossil:two-level@10 (best rule) | 85.62% | 139.2 |
| hayes-roth | PFossil:two-level@10 (macro distribution voting) | 81.88% | 139.2 |
| hayes-roth | PFossil:two-level@10 (micro distribution voting) | 80.00% | 139.2 |
| hayes-roth | PFossil:two-level@10 (voting) | 78.12% | 139.2 |
| hayes-roth | PFossil:two-level@25 (best rule) | 84.38% | 337.0 |
| hayes-roth | PFossil:two-level@25 (macro distribution voting) | 80.62% | 337.0 |
| hayes-roth | PFossil:two-level@25 (micro distribution voting) | 83.75% | 337.0 |
| hayes-roth | PFossil:two-level@25 (voting) | 76.88% | 337.0 |
| hayes-roth | PFossil:two-level@50 (best rule) | 84.38% | 675.3 |
| hayes-roth | PFossil:two-level@50 (macro distribution voting) | 81.25% | 675.3 |
| hayes-roth | PFossil:two-level@50 (micro distribution voting) | 83.12% | 675.3 |
| hayes-roth | PFossil:two-level@50 (voting) | 76.88% | 675.3 |
| hayes-roth | Pypper:one-level@10 (best rule) | 80.62% | 98.3 |
| hayes-roth | Pypper:one-level@10 (macro distribution voting) | 85.00% | 98.3 |
| hayes-roth | Pypper:one-level@10 (micro distribution voting) | 77.50% | 98.3 |
| hayes-roth | Pypper:one-level@10 (voting) | 83.75% | 98.3 |
| hayes-roth | Pypper:one-level@25 (best rule) | 81.25% | 237.7 |
| hayes-roth | Pypper:one-level@25 (macro distribution voting) | 85.00% | 237.7 |
| hayes-roth | Pypper:one-level@25 (micro distribution voting) | 75.00% | 237.7 |
| hayes-roth | Pypper:one-level@25 (voting) | 81.88% | 237.7 |
| hayes-roth | Pypper:one-level@50 (best rule) | 80.62% | 471.2 |
| hayes-roth | Pypper:one-level@50 (macro distribution voting) | 81.25% | 471.2 |
| hayes-roth | Pypper:one-level@50 (micro distribution voting) | 76.25% | 471.2 |
| hayes-roth | Pypper:one-level@50 (voting) | 80.62% | 471.2 |
| hayes-roth | Pypper:single (best rule) | 80.62% | 10.3 |
| hayes-roth | Pypper:single (macro distribution voting) | 79.38% | 10.3 |
| hayes-roth | Pypper:single (micro distribution voting) | 74.38% | 10.3 |
| hayes-roth | Pypper:single (voting) | 73.12% | 10.3 |
| hayes-roth | Pypper:two-level@10 (best rule) | 83.75% | 98.3 |
| hayes-roth | Pypper:two-level@10 (macro distribution voting) | 83.12% | 98.3 |
| hayes-roth | Pypper:two-level@10 (micro distribution voting) | 79.38% | 98.3 |
| hayes-roth | Pypper:two-level@10 (voting) | 78.12% | 98.3 |
| hayes-roth | Pypper:two-level@25 (best rule) | 83.12% | 237.7 |
| hayes-roth | Pypper:two-level@25 (macro distribution voting) | 81.88% | 237.7 |
| hayes-roth | Pypper:two-level@25 (micro distribution voting) | 79.38% | 237.7 |
| hayes-roth | Pypper:two-level@25 (voting) | 77.50% | 237.7 |
| hayes-roth | Pypper:two-level@50 (best rule) | 82.50% | 471.2 |
| hayes-roth | Pypper:two-level@50 (macro distribution voting) | 80.62% | 471.2 |
| hayes-roth | Pypper:two-level@50 (micro distribution voting) | 80.62% | 471.2 |
| hayes-roth | Pypper:two-level@50 (voting) | 75.62% | 471.2 |
| heart-c | PFoil:one-level@10 (best rule) | 79.58% | 231.0 |
| heart-c | PFoil:one-level@10 (macro distribution voting) | 83.52% | 231.0 |
| heart-c | PFoil:one-level@10 (micro distribution voting) | 84.19% | 231.0 |
| heart-c | PFoil:one-level@10 (voting) | 82.54% | 231.0 |
| heart-c | PFoil:one-level@25 (best rule) | 80.87% | 580.0 |
| heart-c | PFoil:one-level@25 (macro distribution voting) | 83.52% | 580.0 |
| heart-c | PFoil:one-level@25 (micro distribution voting) | 83.55% | 580.0 |
| heart-c | PFoil:one-level@25 (voting) | 82.54% | 580.0 |
| heart-c | PFoil:one-level@50 (best rule) | 81.56% | 1167.5 |
| heart-c | PFoil:one-level@50 (macro distribution voting) | 83.85% | 1167.5 |
| heart-c | PFoil:one-level@50 (micro distribution voting) | 83.54% | 1167.5 |
| heart-c | PFoil:one-level@50 (voting) | 82.86% | 1167.5 |
| heart-c | PFoil:single (best rule) | 78.92% | 29.6 |
| heart-c | PFoil:single (macro distribution voting) | 78.59% | 29.6 |
| heart-c | PFoil:single (micro distribution voting) | 78.62% | 29.6 |
| heart-c | PFoil:single (voting) | 77.61% | 29.6 |
| heart-c | PFoil:two-level@10 (best rule) | 83.23% | 231.0 |
| heart-c | PFoil:two-level@10 (macro distribution voting) | 81.89% | 231.0 |
| heart-c | PFoil:two-level@10 (micro distribution voting) | 83.84% | 231.0 |
| heart-c | PFoil:two-level@10 (voting) | 80.58% | 231.0 |
| heart-c | PFoil:two-level@25 (best rule) | 82.56% | 580.0 |
| heart-c | PFoil:two-level@25 (macro distribution voting) | 82.89% | 580.0 |
| heart-c | PFoil:two-level@25 (micro distribution voting) | 82.87% | 580.0 |
| heart-c | PFoil:two-level@25 (voting) | 82.24% | 580.0 |
| heart-c | PFoil:two-level@50 (best rule) | 83.19% | 1167.5 |
| heart-c | PFoil:two-level@50 (macro distribution voting) | 83.53% | 1167.5 |
| heart-c | PFoil:two-level@50 (micro distribution voting) | 83.18% | 1167.5 |
| heart-c | PFoil:two-level@50 (voting) | 81.89% | 1167.5 |
| heart-c | PFossil:one-level@10 (best rule) | 76.92% | 153.4 |
| heart-c | PFossil:one-level@10 (macro distribution voting) | 83.53% | 153.4 |
| heart-c | PFossil:one-level@10 (micro distribution voting) | 80.89% | 153.4 |
| heart-c | PFossil:one-level@10 (voting) | 82.86% | 153.4 |
| heart-c | PFossil:one-level@25 (best rule) | 77.91% | 390.6 |
| heart-c | PFossil:one-level@25 (macro distribution voting) | 82.87% | 390.6 |
| heart-c | PFossil:one-level@25 (micro distribution voting) | 80.24% | 390.6 |
| heart-c | PFossil:one-level@25 (voting) | 82.22% | 390.6 |
| heart-c | PFossil:one-level@50 (best rule) | 77.91% | 786.6 |
| heart-c | PFossil:one-level@50 (macro distribution voting) | 82.23% | 786.6 |
| heart-c | PFossil:one-level@50 (micro distribution voting) | 81.22% | 786.6 |
| heart-c | PFossil:one-level@50 (voting) | 81.89% | 786.6 |
| heart-c | PFossil:single (best rule) | 79.54% | 18.2 |
| heart-c | PFossil:single (macro distribution voting) | 79.88% | 18.2 |
| heart-c | PFossil:single (micro distribution voting) | 77.24% | 18.2 |
| heart-c | PFossil:single (voting) | 78.24% | 18.2 |
| heart-c | PFossil:two-level@10 (best rule) | 81.54% | 153.4 |
| heart-c | PFossil:two-level@10 (macro distribution voting) | 81.87% | 153.4 |
| heart-c | PFossil:two-level@10 (micro distribution voting) | 79.89% | 153.4 |
| heart-c | PFossil:two-level@10 (voting) | 82.22% | 153.4 |
| heart-c | PFossil:two-level@25 (best rule) | 82.22% | 390.6 |
| heart-c | PFossil:two-level@25 (macro distribution voting) | 81.56% | 390.6 |
| heart-c | PFossil:two-level@25 (micro distribution voting) | 79.90% | 390.6 |
| heart-c | PFossil:two-level@25 (voting) | 80.25% | 390.6 |
| heart-c | PFossil:two-level@50 (best rule) | 84.19% | 786.6 |
| heart-c | PFossil:two-level@50 (macro distribution voting) | 83.87% | 786.6 |
| heart-c | PFossil:two-level@50 (micro distribution voting) | 81.22% | 786.6 |
| heart-c | PFossil:two-level@50 (voting) | 81.22% | 786.6 |
| heart-c | Pypper:one-level@10 (best rule) | 79.25% | 70.1 |
| heart-c | Pypper:one-level@10 (macro distribution voting) | 83.52% | 70.1 |
| heart-c | Pypper:one-level@10 (micro distribution voting) | 81.55% | 70.1 |
| heart-c | Pypper:one-level@10 (voting) | 81.55% | 70.1 |
| heart-c | Pypper:one-level@25 (best rule) | 78.91% | 178.0 |
| heart-c | Pypper:one-level@25 (macro distribution voting) | 82.88% | 178.0 |
| heart-c | Pypper:one-level@25 (micro distribution voting) | 82.56% | 178.0 |
| heart-c | Pypper:one-level@25 (voting) | 82.55% | 178.0 |
| heart-c | Pypper:one-level@50 (best rule) | 79.23% | 357.7 |
| heart-c | Pypper:one-level@50 (macro distribution voting) | 84.18% | 357.7 |
| heart-c | Pypper:one-level@50 (micro distribution voting) | 82.86% | 357.7 |
| heart-c | Pypper:one-level@50 (voting) | 83.53% | 357.7 |
| heart-c | Pypper:single (best rule) | 80.22% | 6.1 |
| heart-c | Pypper:single (macro distribution voting) | 79.88% | 6.1 |
| heart-c | Pypper:single (micro distribution voting) | 82.22% | 6.1 |
| heart-c | Pypper:single (voting) | 81.89% | 6.1 |
| heart-c | Pypper:two-level@10 (best rule) | 82.20% | 70.1 |
| heart-c | Pypper:two-level@10 (macro distribution voting) | 82.53% | 70.1 |
| heart-c | Pypper:two-level@10 (micro distribution voting) | 82.23% | 70.1 |
| heart-c | Pypper:two-level@10 (voting) | 80.90% | 70.1 |
| heart-c | Pypper:two-level@25 (best rule) | 83.22% | 178.0 |
| heart-c | Pypper:two-level@25 (macro distribution voting) | 83.20% | 178.0 |
| heart-c | Pypper:two-level@25 (micro distribution voting) | 81.56% | 178.0 |
| heart-c | Pypper:two-level@25 (voting) | 82.23% | 178.0 |
| heart-c | Pypper:two-level@50 (best rule) | 83.53% | 357.7 |
| heart-c | Pypper:two-level@50 (macro distribution voting) | 83.53% | 357.7 |
| heart-c | Pypper:two-level@50 (micro distribution voting) | 83.19% | 357.7 |
| heart-c | Pypper:two-level@50 (voting) | 83.86% | 357.7 |
| heart-h | PFoil:one-level@10 (best rule) | 78.90% | 264.1 |
| heart-h | PFoil:one-level@10 (macro distribution voting) | 80.28% | 264.1 |
| heart-h | PFoil:one-level@10 (micro distribution voting) | 79.57% | 264.1 |
| heart-h | PFoil:one-level@10 (voting) | 80.62% | 264.1 |
| heart-h | PFoil:one-level@25 (best rule) | 80.59% | 651.8 |
| heart-h | PFoil:one-level@25 (macro distribution voting) | 80.95% | 651.8 |
| heart-h | PFoil:one-level@25 (micro distribution voting) | 80.59% | 651.8 |
| heart-h | PFoil:one-level@25 (voting) | 81.30% | 651.8 |
| heart-h | PFoil:one-level@50 (best rule) | 78.59% | 1291.2 |
| heart-h | PFoil:one-level@50 (macro distribution voting) | 80.95% | 1291.2 |
| heart-h | PFoil:one-level@50 (micro distribution voting) | 79.59% | 1291.2 |
| heart-h | PFoil:one-level@50 (voting) | 81.63% | 1291.2 |
| heart-h | PFoil:single (best rule) | 78.93% | 30.4 |
| heart-h | PFoil:single (macro distribution voting) | 79.25% | 30.4 |
| heart-h | PFoil:single (micro distribution voting) | 79.29% | 30.4 |
| heart-h | PFoil:single (voting) | 78.56% | 30.4 |
| heart-h | PFoil:two-level@10 (best rule) | 79.61% | 264.1 |
| heart-h | PFoil:two-level@10 (macro distribution voting) | 79.61% | 264.1 |
| heart-h | PFoil:two-level@10 (micro distribution voting) | 79.92% | 264.1 |
| heart-h | PFoil:two-level@10 (voting) | 80.29% | 264.1 |
| heart-h | PFoil:two-level@25 (best rule) | 80.63% | 651.8 |
| heart-h | PFoil:two-level@25 (macro distribution voting) | 79.94% | 651.8 |
| heart-h | PFoil:two-level@25 (micro distribution voting) | 79.92% | 651.8 |
| heart-h | PFoil:two-level@25 (voting) | 79.29% | 651.8 |
| heart-h | PFoil:two-level@50 (best rule) | 80.62% | 1291.2 |
| heart-h | PFoil:two-level@50 (macro distribution voting) | 80.28% | 1291.2 |
| heart-h | PFoil:two-level@50 (micro distribution voting) | 80.62% | 1291.2 |
| heart-h | PFoil:two-level@50 (voting) | 80.63% | 1291.2 |
| heart-h | PFossil:one-level@10 (best rule) | 80.92% | 144.5 |
| heart-h | PFossil:one-level@10 (macro distribution voting) | 80.94% | 144.5 |
| heart-h | PFossil:one-level@10 (micro distribution voting) | 80.60% | 144.5 |
| heart-h | PFossil:one-level@10 (voting) | 81.29% | 144.5 |
| heart-h | PFossil:one-level@25 (best rule) | 80.57% | 364.5 |
| heart-h | PFossil:one-level@25 (macro distribution voting) | 80.28% | 364.5 |
| heart-h | PFossil:one-level@25 (micro distribution voting) | 79.23% | 364.5 |
| heart-h | PFossil:one-level@25 (voting) | 79.93% | 364.5 |
| heart-h | PFossil:one-level@50 (best rule) | 82.28% | 736.2 |
| heart-h | PFossil:one-level@50 (macro distribution voting) | 80.62% | 736.2 |
| heart-h | PFossil:one-level@50 (micro distribution voting) | 80.61% | 736.2 |
| heart-h | PFossil:one-level@50 (voting) | 80.97% | 736.2 |
| heart-h | PFossil:single (best rule) | 81.28% | 17.2 |
| heart-h | PFossil:single (macro distribution voting) | 79.92% | 17.2 |
| heart-h | PFossil:single (micro distribution voting) | 78.93% | 17.2 |
| heart-h | PFossil:single (voting) | 80.94% | 17.2 |
| heart-h | PFossil:two-level@10 (best rule) | 80.60% | 144.5 |
| heart-h | PFossil:two-level@10 (macro distribution voting) | 80.94% | 144.5 |
| heart-h | PFossil:two-level@10 (micro distribution voting) | 81.99% | 144.5 |
| heart-h | PFossil:two-level@10 (voting) | 79.94% | 144.5 |
| heart-h | PFossil:two-level@25 (best rule) | 80.61% | 364.5 |
| heart-h | PFossil:two-level@25 (macro distribution voting) | 79.93% | 364.5 |
| heart-h | PFossil:two-level@25 (micro distribution voting) | 80.62% | 364.5 |
| heart-h | PFossil:two-level@25 (voting) | 79.57% | 364.5 |
| heart-h | PFossil:two-level@50 (best rule) | 80.26% | 736.2 |
| heart-h | PFossil:two-level@50 (macro distribution voting) | 80.29% | 736.2 |
| heart-h | PFossil:two-level@50 (micro distribution voting) | 81.30% | 736.2 |
| heart-h | PFossil:two-level@50 (voting) | 78.91% | 736.2 |
| heart-h | Pypper:one-level@10 (best rule) | 81.94% | 61.3 |
| heart-h | Pypper:one-level@10 (macro distribution voting) | 82.31% | 61.3 |
| heart-h | Pypper:one-level@10 (micro distribution voting) | 81.95% | 61.3 |
| heart-h | Pypper:one-level@10 (voting) | 82.31% | 61.3 |
| heart-h | Pypper:one-level@25 (best rule) | 80.93% | 148.0 |
| heart-h | Pypper:one-level@25 (macro distribution voting) | 81.31% | 148.0 |
| heart-h | Pypper:one-level@25 (micro distribution voting) | 79.95% | 148.0 |
| heart-h | Pypper:one-level@25 (voting) | 81.66% | 148.0 |
| heart-h | Pypper:one-level@50 (best rule) | 80.93% | 298.4 |
| heart-h | Pypper:one-level@50 (macro distribution voting) | 81.66% | 298.4 |
| heart-h | Pypper:one-level@50 (micro distribution voting) | 79.62% | 298.4 |
| heart-h | Pypper:one-level@50 (voting) | 81.31% | 298.4 |
| heart-h | Pypper:single (best rule) | 77.17% | 4.9 |
| heart-h | Pypper:single (macro distribution voting) | 77.17% | 4.9 |
| heart-h | Pypper:single (micro distribution voting) | 76.83% | 4.9 |
| heart-h | Pypper:single (voting) | 76.83% | 4.9 |
| heart-h | Pypper:two-level@10 (best rule) | 81.29% | 61.3 |
| heart-h | Pypper:two-level@10 (macro distribution voting) | 82.31% | 61.3 |
| heart-h | Pypper:two-level@10 (micro distribution voting) | 81.95% | 61.3 |
| heart-h | Pypper:two-level@10 (voting) | 82.64% | 61.3 |
| heart-h | Pypper:two-level@25 (best rule) | 80.28% | 148.0 |
| heart-h | Pypper:two-level@25 (macro distribution voting) | 80.28% | 148.0 |
| heart-h | Pypper:two-level@25 (micro distribution voting) | 79.61% | 148.0 |
| heart-h | Pypper:two-level@25 (voting) | 80.62% | 148.0 |
| heart-h | Pypper:two-level@50 (best rule) | 79.94% | 298.4 |
| heart-h | Pypper:two-level@50 (macro distribution voting) | 80.29% | 298.4 |
| heart-h | Pypper:two-level@50 (micro distribution voting) | 80.31% | 298.4 |
| heart-h | Pypper:two-level@50 (voting) | 80.63% | 298.4 |
| heart-statlog | PFoil:one-level@10 (best rule) | 83.33% | 214.4 |
| heart-statlog | PFoil:one-level@10 (macro distribution voting) | 83.70% | 214.4 |
| heart-statlog | PFoil:one-level@10 (micro distribution voting) | 82.96% | 214.4 |
| heart-statlog | PFoil:one-level@10 (voting) | 80.74% | 214.4 |
| heart-statlog | PFoil:one-level@25 (best rule) | 82.22% | 535.5 |
| heart-statlog | PFoil:one-level@25 (macro distribution voting) | 82.96% | 535.5 |
| heart-statlog | PFoil:one-level@25 (micro distribution voting) | 83.33% | 535.5 |
| heart-statlog | PFoil:one-level@25 (voting) | 81.85% | 535.5 |
| heart-statlog | PFoil:one-level@50 (best rule) | 80.00% | 1074.3 |
| heart-statlog | PFoil:one-level@50 (macro distribution voting) | 82.59% | 1074.3 |
| heart-statlog | PFoil:one-level@50 (micro distribution voting) | 83.70% | 1074.3 |
| heart-statlog | PFoil:one-level@50 (voting) | 81.48% | 1074.3 |
| heart-statlog | PFoil:single (best rule) | 77.04% | 26.5 |
| heart-statlog | PFoil:single (macro distribution voting) | 76.30% | 26.5 |
| heart-statlog | PFoil:single (micro distribution voting) | 77.04% | 26.5 |
| heart-statlog | PFoil:single (voting) | 75.19% | 26.5 |
| heart-statlog | PFoil:two-level@10 (best rule) | 82.59% | 214.4 |
| heart-statlog | PFoil:two-level@10 (macro distribution voting) | 81.48% | 214.4 |
| heart-statlog | PFoil:two-level@10 (micro distribution voting) | 81.48% | 214.4 |
| heart-statlog | PFoil:two-level@10 (voting) | 78.52% | 214.4 |
| heart-statlog | PFoil:two-level@25 (best rule) | 81.85% | 535.5 |
| heart-statlog | PFoil:two-level@25 (macro distribution voting) | 81.48% | 535.5 |
| heart-statlog | PFoil:two-level@25 (micro distribution voting) | 81.85% | 535.5 |
| heart-statlog | PFoil:two-level@25 (voting) | 80.00% | 535.5 |
| heart-statlog | PFoil:two-level@50 (best rule) | 81.11% | 1074.3 |
| heart-statlog | PFoil:two-level@50 (macro distribution voting) | 80.37% | 1074.3 |
| heart-statlog | PFoil:two-level@50 (micro distribution voting) | 82.59% | 1074.3 |
| heart-statlog | PFoil:two-level@50 (voting) | 81.11% | 1074.3 |
| heart-statlog | PFossil:one-level@10 (best rule) | 80.37% | 151.2 |
| heart-statlog | PFossil:one-level@10 (macro distribution voting) | 80.74% | 151.2 |
| heart-statlog | PFossil:one-level@10 (micro distribution voting) | 81.85% | 151.2 |
| heart-statlog | PFossil:one-level@10 (voting) | 80.74% | 151.2 |
| heart-statlog | PFossil:one-level@25 (best rule) | 80.37% | 374.5 |
| heart-statlog | PFossil:one-level@25 (macro distribution voting) | 82.22% | 374.5 |
| heart-statlog | PFossil:one-level@25 (micro distribution voting) | 82.22% | 374.5 |
| heart-statlog | PFossil:one-level@25 (voting) | 82.22% | 374.5 |
| heart-statlog | PFossil:one-level@50 (best rule) | 79.63% | 750.4 |
| heart-statlog | PFossil:one-level@50 (macro distribution voting) | 82.59% | 750.4 |
| heart-statlog | PFossil:one-level@50 (micro distribution voting) | 82.59% | 750.4 |
| heart-statlog | PFossil:one-level@50 (voting) | 81.48% | 750.4 |
| heart-statlog | PFossil:single (best rule) | 80.74% | 17.6 |
| heart-statlog | PFossil:single (macro distribution voting) | 79.26% | 17.6 |
| heart-statlog | PFossil:single (micro distribution voting) | 80.00% | 17.6 |
| heart-statlog | PFossil:single (voting) | 79.26% | 17.6 |
| heart-statlog | PFossil:two-level@10 (best rule) | 82.22% | 151.2 |
| heart-statlog | PFossil:two-level@10 (macro distribution voting) | 79.63% | 151.2 |
| heart-statlog | PFossil:two-level@10 (micro distribution voting) | 80.74% | 151.2 |
| heart-statlog | PFossil:two-level@10 (voting) | 81.11% | 151.2 |
| heart-statlog | PFossil:two-level@25 (best rule) | 81.85% | 374.5 |
| heart-statlog | PFossil:two-level@25 (macro distribution voting) | 81.48% | 374.5 |
| heart-statlog | PFossil:two-level@25 (micro distribution voting) | 81.48% | 374.5 |
| heart-statlog | PFossil:two-level@25 (voting) | 81.11% | 374.5 |
| heart-statlog | PFossil:two-level@50 (best rule) | 83.33% | 750.4 |
| heart-statlog | PFossil:two-level@50 (macro distribution voting) | 81.85% | 750.4 |
| heart-statlog | PFossil:two-level@50 (micro distribution voting) | 82.59% | 750.4 |
| heart-statlog | PFossil:two-level@50 (voting) | 81.11% | 750.4 |
| heart-statlog | Pypper:one-level@10 (best rule) | 80.00% | 72.0 |
| heart-statlog | Pypper:one-level@10 (macro distribution voting) | 84.81% | 72.0 |
| heart-statlog | Pypper:one-level@10 (micro distribution voting) | 81.11% | 72.0 |
| heart-statlog | Pypper:one-level@10 (voting) | 83.70% | 72.0 |
| heart-statlog | Pypper:one-level@25 (best rule) | 81.48% | 169.5 |
| heart-statlog | Pypper:one-level@25 (macro distribution voting) | 84.81% | 169.5 |
| heart-statlog | Pypper:one-level@25 (micro distribution voting) | 83.33% | 169.5 |
| heart-statlog | Pypper:one-level@25 (voting) | 84.81% | 169.5 |
| heart-statlog | Pypper:one-level@50 (best rule) | 80.74% | 339.5 |
| heart-statlog | Pypper:one-level@50 (macro distribution voting) | 84.07% | 339.5 |
| heart-statlog | Pypper:one-level@50 (micro distribution voting) | 82.22% | 339.5 |
| heart-statlog | Pypper:one-level@50 (voting) | 84.07% | 339.5 |
| heart-statlog | Pypper:single (best rule) | 81.85% | 5.5 |
| heart-statlog | Pypper:single (macro distribution voting) | 82.22% | 5.5 |
| heart-statlog | Pypper:single (micro distribution voting) | 82.59% | 5.5 |
| heart-statlog | Pypper:single (voting) | 82.59% | 5.5 |
| heart-statlog | Pypper:two-level@10 (best rule) | 82.22% | 72.0 |
| heart-statlog | Pypper:two-level@10 (macro distribution voting) | 82.96% | 72.0 |
| heart-statlog | Pypper:two-level@10 (micro distribution voting) | 81.85% | 72.0 |
| heart-statlog | Pypper:two-level@10 (voting) | 82.22% | 72.0 |
| heart-statlog | Pypper:two-level@25 (best rule) | 84.07% | 169.5 |
| heart-statlog | Pypper:two-level@25 (macro distribution voting) | 84.07% | 169.5 |
| heart-statlog | Pypper:two-level@25 (micro distribution voting) | 84.07% | 169.5 |
| heart-statlog | Pypper:two-level@25 (voting) | 83.70% | 169.5 |
| heart-statlog | Pypper:two-level@50 (best rule) | 84.44% | 339.5 |
| heart-statlog | Pypper:two-level@50 (macro distribution voting) | 84.44% | 339.5 |
| heart-statlog | Pypper:two-level@50 (micro distribution voting) | 83.33% | 339.5 |
| heart-statlog | Pypper:two-level@50 (voting) | 82.59% | 339.5 |
| heloc | PFoil:one-level@10 (best rule) | 71.16% | 5792.0 |
| heloc | PFoil:one-level@10 (macro distribution voting) | 71.24% | 5792.0 |
| heloc | PFoil:one-level@10 (micro distribution voting) | 70.13% | 5792.0 |
| heloc | PFoil:one-level@10 (voting) | 70.13% | 5792.0 |
| heloc | PFoil:one-level@25 (best rule) | 71.37% | 14450.0 |
| heloc | PFoil:one-level@25 (macro distribution voting) | 71.64% | 14450.0 |
| heloc | PFoil:one-level@25 (micro distribution voting) | 70.61% | 14450.0 |
| heloc | PFoil:one-level@25 (voting) | 70.49% | 14450.0 |
| heloc | PFoil:one-level@50 (best rule) | 71.81% | 28910.0 |
| heloc | PFoil:one-level@50 (macro distribution voting) | 71.64% | 28910.0 |
| heloc | PFoil:one-level@50 (micro distribution voting) | 70.87% | 28910.0 |
| heloc | PFoil:one-level@50 (voting) | 70.99% | 28910.0 |
| heloc | PFoil:single (best rule) | 67.84% | 739.2 |
| heloc | PFoil:single (macro distribution voting) | 68.04% | 739.2 |
| heloc | PFoil:single (micro distribution voting) | 66.05% | 739.2 |
| heloc | PFoil:single (voting) | 62.21% | 739.2 |
| heloc | PFoil:two-level@10 (best rule) | 71.06% | 5792.0 |
| heloc | PFoil:two-level@10 (macro distribution voting) | 71.24% | 5792.0 |
| heloc | PFoil:two-level@10 (micro distribution voting) | 70.81% | 5792.0 |
| heloc | PFoil:two-level@10 (voting) | 69.86% | 5792.0 |
| heloc | PFoil:two-level@25 (best rule) | 71.60% | 14450.0 |
| heloc | PFoil:two-level@25 (macro distribution voting) | 71.30% | 14450.0 |
| heloc | PFoil:two-level@25 (micro distribution voting) | 71.54% | 14450.0 |
| heloc | PFoil:two-level@25 (voting) | 70.39% | 14450.0 |
| heloc | PFoil:two-level@50 (best rule) | 71.71% | 28910.0 |
| heloc | PFoil:two-level@50 (macro distribution voting) | 71.78% | 28910.0 |
| heloc | PFoil:two-level@50 (micro distribution voting) | 71.65% | 28910.0 |
| heloc | PFoil:two-level@50 (voting) | 70.65% | 28910.0 |
| heloc | PFossil:one-level@10 (best rule) | 69.77% | 20.3 |
| heloc | PFossil:one-level@10 (macro distribution voting) | 69.40% | 20.3 |
| heloc | PFossil:one-level@10 (micro distribution voting) | 69.23% | 20.3 |
| heloc | PFossil:one-level@10 (voting) | 69.26% | 20.3 |
| heloc | PFossil:one-level@25 (best rule) | 70.26% | 50.8 |
| heloc | PFossil:one-level@25 (macro distribution voting) | 69.49% | 50.8 |
| heloc | PFossil:one-level@25 (micro distribution voting) | 69.49% | 50.8 |
| heloc | PFossil:one-level@25 (voting) | 69.47% | 50.8 |
| heloc | PFossil:one-level@50 (best rule) | 69.99% | 101.4 |
| heloc | PFossil:one-level@50 (macro distribution voting) | 69.38% | 101.4 |
| heloc | PFossil:one-level@50 (micro distribution voting) | 69.34% | 101.4 |
| heloc | PFossil:one-level@50 (voting) | 69.33% | 101.4 |
| heloc | PFossil:single (best rule) | 68.83% | 2.0 |
| heloc | PFossil:single (macro distribution voting) | 68.83% | 2.0 |
| heloc | PFossil:single (micro distribution voting) | 68.83% | 2.0 |
| heloc | PFossil:single (voting) | 68.83% | 2.0 |
| heloc | PFossil:two-level@10 (best rule) | 69.26% | 20.3 |
| heloc | PFossil:two-level@10 (macro distribution voting) | 69.26% | 20.3 |
| heloc | PFossil:two-level@10 (micro distribution voting) | 69.26% | 20.3 |
| heloc | PFossil:two-level@10 (voting) | 69.26% | 20.3 |
| heloc | PFossil:two-level@25 (best rule) | 69.48% | 50.8 |
| heloc | PFossil:two-level@25 (macro distribution voting) | 69.48% | 50.8 |
| heloc | PFossil:two-level@25 (micro distribution voting) | 69.48% | 50.8 |
| heloc | PFossil:two-level@25 (voting) | 69.47% | 50.8 |
| heloc | PFossil:two-level@50 (best rule) | 69.33% | 101.4 |
| heloc | PFossil:two-level@50 (macro distribution voting) | 69.33% | 101.4 |
| heloc | PFossil:two-level@50 (micro distribution voting) | 69.33% | 101.4 |
| heloc | PFossil:two-level@50 (voting) | 69.33% | 101.4 |
| heloc | Pypper:one-level@10 (best rule) | 71.31% | 163.6 |
| heloc | Pypper:one-level@10 (macro distribution voting) | 71.46% | 163.6 |
| heloc | Pypper:one-level@10 (micro distribution voting) | 71.03% | 163.6 |
| heloc | Pypper:one-level@10 (voting) | 71.28% | 163.6 |
| heloc | Pypper:one-level@25 (best rule) | 71.49% | 414.9 |
| heloc | Pypper:one-level@25 (macro distribution voting) | 71.50% | 414.9 |
| heloc | Pypper:one-level@25 (micro distribution voting) | 71.00% | 414.9 |
| heloc | Pypper:one-level@25 (voting) | 71.38% | 414.9 |
| heloc | Pypper:one-level@50 (best rule) | 71.63% | 839.6 |
| heloc | Pypper:one-level@50 (macro distribution voting) | 71.42% | 839.6 |
| heloc | Pypper:one-level@50 (micro distribution voting) | 71.05% | 839.6 |
| heloc | Pypper:one-level@50 (voting) | 71.41% | 839.6 |
| heloc | Pypper:single (best rule) | 69.81% | 15.7 |
| heloc | Pypper:single (macro distribution voting) | 69.87% | 15.7 |
| heloc | Pypper:single (micro distribution voting) | 69.88% | 15.7 |
| heloc | Pypper:single (voting) | 69.75% | 15.7 |
| heloc | Pypper:two-level@10 (best rule) | 71.26% | 163.6 |
| heloc | Pypper:two-level@10 (macro distribution voting) | 71.26% | 163.6 |
| heloc | Pypper:two-level@10 (micro distribution voting) | 71.10% | 163.6 |
| heloc | Pypper:two-level@10 (voting) | 71.32% | 163.6 |
| heloc | Pypper:two-level@25 (best rule) | 71.25% | 414.9 |
| heloc | Pypper:two-level@25 (macro distribution voting) | 71.31% | 414.9 |
| heloc | Pypper:two-level@25 (micro distribution voting) | 71.24% | 414.9 |
| heloc | Pypper:two-level@25 (voting) | 71.30% | 414.9 |
| heloc | Pypper:two-level@50 (best rule) | 71.33% | 839.6 |
| heloc | Pypper:two-level@50 (macro distribution voting) | 71.42% | 839.6 |
| heloc | Pypper:two-level@50 (micro distribution voting) | 71.26% | 839.6 |
| heloc | Pypper:two-level@50 (voting) | 71.58% | 839.6 |
| hepatitis | PFoil:one-level@10 (best rule) | 82.62% | 131.6 |
| hepatitis | PFoil:one-level@10 (macro distribution voting) | 84.04% | 131.6 |
| hepatitis | PFoil:one-level@10 (micro distribution voting) | 82.00% | 131.6 |
| hepatitis | PFoil:one-level@10 (voting) | 84.75% | 131.6 |
| hepatitis | PFoil:one-level@25 (best rule) | 80.04% | 331.6 |
| hepatitis | PFoil:one-level@25 (macro distribution voting) | 80.83% | 331.6 |
| hepatitis | PFoil:one-level@25 (micro distribution voting) | 81.33% | 331.6 |
| hepatitis | PFoil:one-level@25 (voting) | 82.79% | 331.6 |
| hepatitis | PFoil:one-level@50 (best rule) | 80.04% | 657.2 |
| hepatitis | PFoil:one-level@50 (macro distribution voting) | 81.42% | 657.2 |
| hepatitis | PFoil:one-level@50 (micro distribution voting) | 81.33% | 657.2 |
| hepatitis | PFoil:one-level@50 (voting) | 83.46% | 657.2 |
| hepatitis | PFoil:single (best rule) | 77.58% | 17.1 |
| hepatitis | PFoil:single (macro distribution voting) | 77.54% | 17.1 |
| hepatitis | PFoil:single (micro distribution voting) | 79.46% | 17.1 |
| hepatitis | PFoil:single (voting) | 76.29% | 17.1 |
| hepatitis | PFoil:two-level@10 (best rule) | 84.04% | 131.6 |
| hepatitis | PFoil:two-level@10 (macro distribution voting) | 82.79% | 131.6 |
| hepatitis | PFoil:two-level@10 (micro distribution voting) | 82.71% | 131.6 |
| hepatitis | PFoil:two-level@10 (voting) | 85.29% | 131.6 |
| hepatitis | PFoil:two-level@25 (best rule) | 82.04% | 331.6 |
| hepatitis | PFoil:two-level@25 (macro distribution voting) | 80.83% | 331.6 |
| hepatitis | PFoil:two-level@25 (micro distribution voting) | 80.71% | 331.6 |
| hepatitis | PFoil:two-level@25 (voting) | 83.38% | 331.6 |
| hepatitis | PFoil:two-level@50 (best rule) | 83.96% | 657.2 |
| hepatitis | PFoil:two-level@50 (macro distribution voting) | 82.79% | 657.2 |
| hepatitis | PFoil:two-level@50 (micro distribution voting) | 83.29% | 657.2 |
| hepatitis | PFoil:two-level@50 (voting) | 84.04% | 657.2 |
| hepatitis | PFossil:one-level@10 (best rule) | 80.08% | 112.1 |
| hepatitis | PFossil:one-level@10 (macro distribution voting) | 81.46% | 112.1 |
| hepatitis | PFossil:one-level@10 (micro distribution voting) | 81.96% | 112.1 |
| hepatitis | PFossil:one-level@10 (voting) | 81.42% | 112.1 |
| hepatitis | PFossil:one-level@25 (best rule) | 80.67% | 281.9 |
| hepatitis | PFossil:one-level@25 (macro distribution voting) | 81.42% | 281.9 |
| hepatitis | PFossil:one-level@25 (micro distribution voting) | 81.96% | 281.9 |
| hepatitis | PFossil:one-level@25 (voting) | 81.42% | 281.9 |
| hepatitis | PFossil:one-level@50 (best rule) | 80.67% | 565.8 |
| hepatitis | PFossil:one-level@50 (macro distribution voting) | 81.42% | 565.8 |
| hepatitis | PFossil:one-level@50 (micro distribution voting) | 81.33% | 565.8 |
| hepatitis | PFossil:one-level@50 (voting) | 81.42% | 565.8 |
| hepatitis | PFossil:single (best rule) | 78.75% | 15.0 |
| hepatitis | PFossil:single (macro distribution voting) | 77.54% | 15.0 |
| hepatitis | PFossil:single (micro distribution voting) | 80.04% | 15.0 |
| hepatitis | PFossil:single (voting) | 78.75% | 15.0 |
| hepatitis | PFossil:two-level@10 (best rule) | 82.67% | 112.1 |
| hepatitis | PFossil:two-level@10 (macro distribution voting) | 80.83% | 112.1 |
| hepatitis | PFossil:two-level@10 (micro distribution voting) | 82.67% | 112.1 |
| hepatitis | PFossil:two-level@10 (voting) | 81.42% | 112.1 |
| hepatitis | PFossil:two-level@25 (best rule) | 81.42% | 281.9 |
| hepatitis | PFossil:two-level@25 (macro distribution voting) | 80.83% | 281.9 |
| hepatitis | PFossil:two-level@25 (micro distribution voting) | 80.79% | 281.9 |
| hepatitis | PFossil:two-level@25 (voting) | 80.79% | 281.9 |
| hepatitis | PFossil:two-level@50 (best rule) | 81.42% | 565.8 |
| hepatitis | PFossil:two-level@50 (macro distribution voting) | 80.17% | 565.8 |
| hepatitis | PFossil:two-level@50 (micro distribution voting) | 82.04% | 565.8 |
| hepatitis | PFossil:two-level@50 (voting) | 82.67% | 565.8 |
| hepatitis | Pypper:one-level@10 (best rule) | 80.17% | 47.6 |
| hepatitis | Pypper:one-level@10 (macro distribution voting) | 80.17% | 47.6 |
| hepatitis | Pypper:one-level@10 (micro distribution voting) | 79.46% | 47.6 |
| hepatitis | Pypper:one-level@10 (voting) | 82.12% | 47.6 |
| hepatitis | Pypper:one-level@25 (best rule) | 79.42% | 114.6 |
| hepatitis | Pypper:one-level@25 (macro distribution voting) | 78.92% | 114.6 |
| hepatitis | Pypper:one-level@25 (micro distribution voting) | 80.62% | 114.6 |
| hepatitis | Pypper:one-level@25 (voting) | 80.83% | 114.6 |
| hepatitis | Pypper:one-level@50 (best rule) | 80.00% | 228.0 |
| hepatitis | Pypper:one-level@50 (macro distribution voting) | 82.17% | 228.0 |
| hepatitis | Pypper:one-level@50 (micro distribution voting) | 80.00% | 228.0 |
| hepatitis | Pypper:one-level@50 (voting) | 81.50% | 228.0 |
| hepatitis | Pypper:single (best rule) | 80.71% | 4.2 |
| hepatitis | Pypper:single (macro distribution voting) | 82.00% | 4.2 |
| hepatitis | Pypper:single (micro distribution voting) | 80.04% | 4.2 |
| hepatitis | Pypper:single (voting) | 80.67% | 4.2 |
| hepatitis | Pypper:two-level@10 (best rule) | 79.54% | 47.6 |
| hepatitis | Pypper:two-level@10 (macro distribution voting) | 80.83% | 47.6 |
| hepatitis | Pypper:two-level@10 (micro distribution voting) | 80.79% | 47.6 |
| hepatitis | Pypper:two-level@10 (voting) | 80.17% | 47.6 |
| hepatitis | Pypper:two-level@25 (best rule) | 80.17% | 114.6 |
| hepatitis | Pypper:two-level@25 (macro distribution voting) | 80.83% | 114.6 |
| hepatitis | Pypper:two-level@25 (micro distribution voting) | 80.17% | 114.6 |
| hepatitis | Pypper:two-level@25 (voting) | 78.92% | 114.6 |
| hepatitis | Pypper:two-level@50 (best rule) | 80.25% | 228.0 |
| hepatitis | Pypper:two-level@50 (macro distribution voting) | 82.17% | 228.0 |
| hepatitis | Pypper:two-level@50 (micro distribution voting) | 80.21% | 228.0 |
| hepatitis | Pypper:two-level@50 (voting) | 80.25% | 228.0 |
| ilpd | PFoil:one-level@10 (best rule) | 71.19% | 520.5 |
| ilpd | PFoil:one-level@10 (macro distribution voting) | 72.03% | 520.5 |
| ilpd | PFoil:one-level@10 (micro distribution voting) | 71.01% | 520.5 |
| ilpd | PFoil:one-level@10 (voting) | 71.50% | 520.5 |
| ilpd | PFoil:one-level@25 (best rule) | 71.69% | 1313.3 |
| ilpd | PFoil:one-level@25 (macro distribution voting) | 72.20% | 1313.3 |
| ilpd | PFoil:one-level@25 (micro distribution voting) | 71.52% | 1313.3 |
| ilpd | PFoil:one-level@25 (voting) | 71.85% | 1313.3 |
| ilpd | PFoil:one-level@50 (best rule) | 72.55% | 2638.7 |
| ilpd | PFoil:one-level@50 (macro distribution voting) | 70.83% | 2638.7 |
| ilpd | PFoil:one-level@50 (micro distribution voting) | 71.69% | 2638.7 |
| ilpd | PFoil:one-level@50 (voting) | 71.17% | 2638.7 |
| ilpd | PFoil:single (best rule) | 68.94% | 64.2 |
| ilpd | PFoil:single (macro distribution voting) | 69.62% | 64.2 |
| ilpd | PFoil:single (micro distribution voting) | 69.12% | 64.2 |
| ilpd | PFoil:single (voting) | 68.42% | 64.2 |
| ilpd | PFoil:two-level@10 (best rule) | 72.36% | 520.5 |
| ilpd | PFoil:two-level@10 (macro distribution voting) | 72.19% | 520.5 |
| ilpd | PFoil:two-level@10 (micro distribution voting) | 71.85% | 520.5 |
| ilpd | PFoil:two-level@10 (voting) | 70.99% | 520.5 |
| ilpd | PFoil:two-level@25 (best rule) | 70.48% | 1313.3 |
| ilpd | PFoil:two-level@25 (macro distribution voting) | 72.19% | 1313.3 |
| ilpd | PFoil:two-level@25 (micro distribution voting) | 71.52% | 1313.3 |
| ilpd | PFoil:two-level@25 (voting) | 71.68% | 1313.3 |
| ilpd | PFoil:two-level@50 (best rule) | 70.31% | 2638.7 |
| ilpd | PFoil:two-level@50 (macro distribution voting) | 72.19% | 2638.7 |
| ilpd | PFoil:two-level@50 (micro distribution voting) | 70.84% | 2638.7 |
| ilpd | PFoil:two-level@50 (voting) | 71.69% | 2638.7 |
| ilpd | PFossil:one-level@10 (best rule) | 70.33% | 83.8 |
| ilpd | PFossil:one-level@10 (macro distribution voting) | 68.96% | 83.8 |
| ilpd | PFossil:one-level@10 (micro distribution voting) | 70.32% | 83.8 |
| ilpd | PFossil:one-level@10 (voting) | 66.18% | 83.8 |
| ilpd | PFossil:one-level@25 (best rule) | 71.20% | 199.8 |
| ilpd | PFossil:one-level@25 (macro distribution voting) | 71.01% | 199.8 |
| ilpd | PFossil:one-level@25 (micro distribution voting) | 70.15% | 199.8 |
| ilpd | PFossil:one-level@25 (voting) | 65.51% | 199.8 |
| ilpd | PFossil:one-level@50 (best rule) | 72.05% | 382.9 |
| ilpd | PFossil:one-level@50 (macro distribution voting) | 71.18% | 382.9 |
| ilpd | PFossil:one-level@50 (micro distribution voting) | 71.01% | 382.9 |
| ilpd | PFossil:one-level@50 (voting) | 66.36% | 382.9 |
| ilpd | PFossil:single (best rule) | 63.28% | 2.8 |
| ilpd | PFossil:single (macro distribution voting) | 63.28% | 2.8 |
| ilpd | PFossil:single (micro distribution voting) | 63.28% | 2.8 |
| ilpd | PFossil:single (voting) | 63.28% | 2.8 |
| ilpd | PFossil:two-level@10 (best rule) | 66.35% | 83.8 |
| ilpd | PFossil:two-level@10 (macro distribution voting) | 66.35% | 83.8 |
| ilpd | PFossil:two-level@10 (micro distribution voting) | 66.69% | 83.8 |
| ilpd | PFossil:two-level@10 (voting) | 66.01% | 83.8 |
| ilpd | PFossil:two-level@25 (best rule) | 66.88% | 199.8 |
| ilpd | PFossil:two-level@25 (macro distribution voting) | 66.88% | 199.8 |
| ilpd | PFossil:two-level@25 (micro distribution voting) | 66.71% | 199.8 |
| ilpd | PFossil:two-level@25 (voting) | 66.88% | 199.8 |
| ilpd | PFossil:two-level@50 (best rule) | 67.91% | 382.9 |
| ilpd | PFossil:two-level@50 (macro distribution voting) | 67.74% | 382.9 |
| ilpd | PFossil:two-level@50 (micro distribution voting) | 68.42% | 382.9 |
| ilpd | PFossil:two-level@50 (voting) | 67.57% | 382.9 |
| ilpd | Pypper:one-level@10 (best rule) | 68.79% | 58.5 |
| ilpd | Pypper:one-level@10 (macro distribution voting) | 69.99% | 58.5 |
| ilpd | Pypper:one-level@10 (micro distribution voting) | 69.30% | 58.5 |
| ilpd | Pypper:one-level@10 (voting) | 69.47% | 58.5 |
| ilpd | Pypper:one-level@25 (best rule) | 69.99% | 148.1 |
| ilpd | Pypper:one-level@25 (macro distribution voting) | 70.50% | 148.1 |
| ilpd | Pypper:one-level@25 (micro distribution voting) | 69.98% | 148.1 |
| ilpd | Pypper:one-level@25 (voting) | 69.81% | 148.1 |
| ilpd | Pypper:one-level@50 (best rule) | 70.49% | 291.5 |
| ilpd | Pypper:one-level@50 (macro distribution voting) | 69.82% | 291.5 |
| ilpd | Pypper:one-level@50 (micro distribution voting) | 69.82% | 291.5 |
| ilpd | Pypper:one-level@50 (voting) | 68.27% | 291.5 |
| ilpd | Pypper:single (best rule) | 69.14% | 2.9 |
| ilpd | Pypper:single (macro distribution voting) | 69.14% | 2.9 |
| ilpd | Pypper:single (micro distribution voting) | 69.14% | 2.9 |
| ilpd | Pypper:single (voting) | 69.14% | 2.9 |
| ilpd | Pypper:two-level@10 (best rule) | 68.60% | 58.5 |
| ilpd | Pypper:two-level@10 (macro distribution voting) | 68.60% | 58.5 |
| ilpd | Pypper:two-level@10 (micro distribution voting) | 68.61% | 58.5 |
| ilpd | Pypper:two-level@10 (voting) | 69.47% | 58.5 |
| ilpd | Pypper:two-level@25 (best rule) | 69.12% | 148.1 |
| ilpd | Pypper:two-level@25 (macro distribution voting) | 69.30% | 148.1 |
| ilpd | Pypper:two-level@25 (micro distribution voting) | 69.47% | 148.1 |
| ilpd | Pypper:two-level@25 (voting) | 69.30% | 148.1 |
| ilpd | Pypper:two-level@50 (best rule) | 68.78% | 291.5 |
| ilpd | Pypper:two-level@50 (macro distribution voting) | 68.78% | 291.5 |
| ilpd | Pypper:two-level@50 (micro distribution voting) | 68.45% | 291.5 |
| ilpd | Pypper:two-level@50 (voting) | 68.10% | 291.5 |
| ionosphere | PFoil:one-level@10 (best rule) | 80.06% | 140.7 |
| ionosphere | PFoil:one-level@10 (macro distribution voting) | 90.03% | 140.7 |
| ionosphere | PFoil:one-level@10 (micro distribution voting) | 90.59% | 140.7 |
| ionosphere | PFoil:one-level@10 (voting) | 90.32% | 140.7 |
| ionosphere | PFoil:one-level@25 (best rule) | 78.33% | 351.9 |
| ionosphere | PFoil:one-level@25 (macro distribution voting) | 91.73% | 351.9 |
| ionosphere | PFoil:one-level@25 (micro distribution voting) | 91.44% | 351.9 |
| ionosphere | PFoil:one-level@25 (voting) | 91.44% | 351.9 |
| ionosphere | PFoil:one-level@50 (best rule) | 75.20% | 696.4 |
| ionosphere | PFoil:one-level@50 (macro distribution voting) | 92.01% | 696.4 |
| ionosphere | PFoil:one-level@50 (micro distribution voting) | 91.73% | 696.4 |
| ionosphere | PFoil:one-level@50 (voting) | 92.59% | 696.4 |
| ionosphere | PFoil:single (best rule) | 90.87% | 18.1 |
| ionosphere | PFoil:single (macro distribution voting) | 92.02% | 18.1 |
| ionosphere | PFoil:single (micro distribution voting) | 91.17% | 18.1 |
| ionosphere | PFoil:single (voting) | 92.30% | 18.1 |
| ionosphere | PFoil:two-level@10 (best rule) | 90.31% | 140.7 |
| ionosphere | PFoil:two-level@10 (macro distribution voting) | 90.31% | 140.7 |
| ionosphere | PFoil:two-level@10 (micro distribution voting) | 89.75% | 140.7 |
| ionosphere | PFoil:two-level@10 (voting) | 90.60% | 140.7 |
| ionosphere | PFoil:two-level@25 (best rule) | 91.46% | 351.9 |
| ionosphere | PFoil:two-level@25 (macro distribution voting) | 90.89% | 351.9 |
| ionosphere | PFoil:two-level@25 (micro distribution voting) | 91.73% | 351.9 |
| ionosphere | PFoil:two-level@25 (voting) | 92.02% | 351.9 |
| ionosphere | PFoil:two-level@50 (best rule) | 92.30% | 696.4 |
| ionosphere | PFoil:two-level@50 (macro distribution voting) | 92.30% | 696.4 |
| ionosphere | PFoil:two-level@50 (micro distribution voting) | 92.58% | 696.4 |
| ionosphere | PFoil:two-level@50 (voting) | 92.59% | 696.4 |
| ionosphere | PFossil:one-level@10 (best rule) | 89.44% | 115.2 |
| ionosphere | PFossil:one-level@10 (macro distribution voting) | 90.88% | 115.2 |
| ionosphere | PFossil:one-level@10 (micro distribution voting) | 92.30% | 115.2 |
| ionosphere | PFossil:one-level@10 (voting) | 90.60% | 115.2 |
| ionosphere | PFossil:one-level@25 (best rule) | 86.88% | 281.3 |
| ionosphere | PFossil:one-level@25 (macro distribution voting) | 91.74% | 281.3 |
| ionosphere | PFossil:one-level@25 (micro distribution voting) | 92.87% | 281.3 |
| ionosphere | PFossil:one-level@25 (voting) | 90.88% | 281.3 |
| ionosphere | PFossil:one-level@50 (best rule) | 85.74% | 556.7 |
| ionosphere | PFossil:one-level@50 (macro distribution voting) | 92.31% | 556.7 |
| ionosphere | PFossil:one-level@50 (micro distribution voting) | 92.87% | 556.7 |
| ionosphere | PFossil:one-level@50 (voting) | 91.74% | 556.7 |
| ionosphere | PFossil:single (best rule) | 89.75% | 13.6 |
| ionosphere | PFossil:single (macro distribution voting) | 90.03% | 13.6 |
| ionosphere | PFossil:single (micro distribution voting) | 89.75% | 13.6 |
| ionosphere | PFossil:single (voting) | 90.03% | 13.6 |
| ionosphere | PFossil:two-level@10 (best rule) | 92.58% | 115.2 |
| ionosphere | PFossil:two-level@10 (macro distribution voting) | 92.58% | 115.2 |
| ionosphere | PFossil:two-level@10 (micro distribution voting) | 91.17% | 115.2 |
| ionosphere | PFossil:two-level@10 (voting) | 92.87% | 115.2 |
| ionosphere | PFossil:two-level@25 (best rule) | 92.58% | 281.3 |
| ionosphere | PFossil:two-level@25 (macro distribution voting) | 92.30% | 281.3 |
| ionosphere | PFossil:two-level@25 (micro distribution voting) | 91.73% | 281.3 |
| ionosphere | PFossil:two-level@25 (voting) | 92.29% | 281.3 |
| ionosphere | PFossil:two-level@50 (best rule) | 92.58% | 556.7 |
| ionosphere | PFossil:two-level@50 (macro distribution voting) | 93.16% | 556.7 |
| ionosphere | PFossil:two-level@50 (micro distribution voting) | 91.45% | 556.7 |
| ionosphere | PFossil:two-level@50 (voting) | 92.87% | 556.7 |
| ionosphere | Pypper:one-level@10 (best rule) | 88.60% | 65.6 |
| ionosphere | Pypper:one-level@10 (macro distribution voting) | 90.60% | 65.6 |
| ionosphere | Pypper:one-level@10 (micro distribution voting) | 88.88% | 65.6 |
| ionosphere | Pypper:one-level@10 (voting) | 90.32% | 65.6 |
| ionosphere | Pypper:one-level@25 (best rule) | 87.46% | 165.8 |
| ionosphere | Pypper:one-level@25 (macro distribution voting) | 90.03% | 165.8 |
| ionosphere | Pypper:one-level@25 (micro distribution voting) | 90.02% | 165.8 |
| ionosphere | Pypper:one-level@25 (voting) | 90.03% | 165.8 |
| ionosphere | Pypper:one-level@50 (best rule) | 85.17% | 333.1 |
| ionosphere | Pypper:one-level@50 (macro distribution voting) | 90.32% | 333.1 |
| ionosphere | Pypper:one-level@50 (micro distribution voting) | 90.32% | 333.1 |
| ionosphere | Pypper:one-level@50 (voting) | 90.03% | 333.1 |
| ionosphere | Pypper:single (best rule) | 90.60% | 6.6 |
| ionosphere | Pypper:single (macro distribution voting) | 89.75% | 6.6 |
| ionosphere | Pypper:single (micro distribution voting) | 90.03% | 6.6 |
| ionosphere | Pypper:single (voting) | 90.32% | 6.6 |
| ionosphere | Pypper:two-level@10 (best rule) | 91.16% | 65.6 |
| ionosphere | Pypper:two-level@10 (macro distribution voting) | 91.16% | 65.6 |
| ionosphere | Pypper:two-level@10 (micro distribution voting) | 89.17% | 65.6 |
| ionosphere | Pypper:two-level@10 (voting) | 90.60% | 65.6 |
| ionosphere | Pypper:two-level@25 (best rule) | 91.17% | 165.8 |
| ionosphere | Pypper:two-level@25 (macro distribution voting) | 90.88% | 165.8 |
| ionosphere | Pypper:two-level@25 (micro distribution voting) | 89.46% | 165.8 |
| ionosphere | Pypper:two-level@25 (voting) | 91.46% | 165.8 |
| ionosphere | Pypper:two-level@50 (best rule) | 90.60% | 333.1 |
| ionosphere | Pypper:two-level@50 (macro distribution voting) | 91.17% | 333.1 |
| ionosphere | Pypper:two-level@50 (micro distribution voting) | 90.32% | 333.1 |
| ionosphere | Pypper:two-level@50 (voting) | 91.75% | 333.1 |
| iris | PFoil:one-level@10 (best rule) | 94.00% | 69.4 |
| iris | PFoil:one-level@10 (macro distribution voting) | 92.00% | 69.4 |
| iris | PFoil:one-level@10 (micro distribution voting) | 93.33% | 69.4 |
| iris | PFoil:one-level@10 (voting) | 92.00% | 69.4 |
| iris | PFoil:one-level@25 (best rule) | 93.33% | 169.0 |
| iris | PFoil:one-level@25 (macro distribution voting) | 92.67% | 169.0 |
| iris | PFoil:one-level@25 (micro distribution voting) | 92.67% | 169.0 |
| iris | PFoil:one-level@25 (voting) | 92.67% | 169.0 |
| iris | PFoil:one-level@50 (best rule) | 94.00% | 333.2 |
| iris | PFoil:one-level@50 (macro distribution voting) | 92.67% | 333.2 |
| iris | PFoil:one-level@50 (micro distribution voting) | 93.33% | 333.2 |
| iris | PFoil:one-level@50 (voting) | 92.67% | 333.2 |
| iris | PFoil:single (best rule) | 92.67% | 8.9 |
| iris | PFoil:single (macro distribution voting) | 92.67% | 8.9 |
| iris | PFoil:single (micro distribution voting) | 92.67% | 8.9 |
| iris | PFoil:single (voting) | 90.67% | 8.9 |
| iris | PFoil:two-level@10 (best rule) | 92.67% | 69.4 |
| iris | PFoil:two-level@10 (macro distribution voting) | 92.00% | 69.4 |
| iris | PFoil:two-level@10 (micro distribution voting) | 92.67% | 69.4 |
| iris | PFoil:two-level@10 (voting) | 92.00% | 69.4 |
| iris | PFoil:two-level@25 (best rule) | 92.67% | 169.0 |
| iris | PFoil:two-level@25 (macro distribution voting) | 92.67% | 169.0 |
| iris | PFoil:two-level@25 (micro distribution voting) | 92.67% | 169.0 |
| iris | PFoil:two-level@25 (voting) | 92.67% | 169.0 |
| iris | PFoil:two-level@50 (best rule) | 92.67% | 333.2 |
| iris | PFoil:two-level@50 (macro distribution voting) | 92.67% | 333.2 |
| iris | PFoil:two-level@50 (micro distribution voting) | 92.67% | 333.2 |
| iris | PFoil:two-level@50 (voting) | 93.33% | 333.2 |
| iris | PFossil:one-level@10 (best rule) | 92.67% | 54.4 |
| iris | PFossil:one-level@10 (macro distribution voting) | 94.00% | 54.4 |
| iris | PFossil:one-level@10 (micro distribution voting) | 93.33% | 54.4 |
| iris | PFossil:one-level@10 (voting) | 93.33% | 54.4 |
| iris | PFossil:one-level@25 (best rule) | 93.33% | 133.4 |
| iris | PFossil:one-level@25 (macro distribution voting) | 92.67% | 133.4 |
| iris | PFossil:one-level@25 (micro distribution voting) | 93.33% | 133.4 |
| iris | PFossil:one-level@25 (voting) | 92.00% | 133.4 |
| iris | PFossil:one-level@50 (best rule) | 94.00% | 263.0 |
| iris | PFossil:one-level@50 (macro distribution voting) | 92.67% | 263.0 |
| iris | PFossil:one-level@50 (micro distribution voting) | 93.33% | 263.0 |
| iris | PFossil:one-level@50 (voting) | 92.67% | 263.0 |
| iris | PFossil:single (best rule) | 92.00% | 5.8 |
| iris | PFossil:single (macro distribution voting) | 92.00% | 5.8 |
| iris | PFossil:single (micro distribution voting) | 92.00% | 5.8 |
| iris | PFossil:single (voting) | 91.33% | 5.8 |
| iris | PFossil:two-level@10 (best rule) | 93.33% | 54.4 |
| iris | PFossil:two-level@10 (macro distribution voting) | 94.00% | 54.4 |
| iris | PFossil:two-level@10 (micro distribution voting) | 93.33% | 54.4 |
| iris | PFossil:two-level@10 (voting) | 93.33% | 54.4 |
| iris | PFossil:two-level@25 (best rule) | 92.67% | 133.4 |
| iris | PFossil:two-level@25 (macro distribution voting) | 92.67% | 133.4 |
| iris | PFossil:two-level@25 (micro distribution voting) | 92.67% | 133.4 |
| iris | PFossil:two-level@25 (voting) | 92.67% | 133.4 |
| iris | PFossil:two-level@50 (best rule) | 92.67% | 263.0 |
| iris | PFossil:two-level@50 (macro distribution voting) | 92.00% | 263.0 |
| iris | PFossil:two-level@50 (micro distribution voting) | 93.33% | 263.0 |
| iris | PFossil:two-level@50 (voting) | 92.67% | 263.0 |
| iris | Pypper:one-level@10 (best rule) | 94.67% | 37.0 |
| iris | Pypper:one-level@10 (macro distribution voting) | 94.67% | 37.0 |
| iris | Pypper:one-level@10 (micro distribution voting) | 94.00% | 37.0 |
| iris | Pypper:one-level@10 (voting) | 94.00% | 37.0 |
| iris | Pypper:one-level@25 (best rule) | 91.33% | 91.8 |
| iris | Pypper:one-level@25 (macro distribution voting) | 93.33% | 91.8 |
| iris | Pypper:one-level@25 (micro distribution voting) | 93.33% | 91.8 |
| iris | Pypper:one-level@25 (voting) | 93.33% | 91.8 |
| iris | Pypper:one-level@50 (best rule) | 95.33% | 184.5 |
| iris | Pypper:one-level@50 (macro distribution voting) | 93.33% | 184.5 |
| iris | Pypper:one-level@50 (micro distribution voting) | 93.33% | 184.5 |
| iris | Pypper:one-level@50 (voting) | 93.33% | 184.5 |
| iris | Pypper:single (best rule) | 92.00% | 3.6 |
| iris | Pypper:single (macro distribution voting) | 92.00% | 3.6 |
| iris | Pypper:single (micro distribution voting) | 92.00% | 3.6 |
| iris | Pypper:single (voting) | 92.00% | 3.6 |
| iris | Pypper:two-level@10 (best rule) | 94.67% | 37.0 |
| iris | Pypper:two-level@10 (macro distribution voting) | 94.00% | 37.0 |
| iris | Pypper:two-level@10 (micro distribution voting) | 94.67% | 37.0 |
| iris | Pypper:two-level@10 (voting) | 94.67% | 37.0 |
| iris | Pypper:two-level@25 (best rule) | 93.33% | 91.8 |
| iris | Pypper:two-level@25 (macro distribution voting) | 93.33% | 91.8 |
| iris | Pypper:two-level@25 (micro distribution voting) | 93.33% | 91.8 |
| iris | Pypper:two-level@25 (voting) | 93.33% | 91.8 |
| iris | Pypper:two-level@50 (best rule) | 93.33% | 184.5 |
| iris | Pypper:two-level@50 (macro distribution voting) | 93.33% | 184.5 |
| iris | Pypper:two-level@50 (micro distribution voting) | 93.33% | 184.5 |
| iris | Pypper:two-level@50 (voting) | 93.33% | 184.5 |
| kc1 | PFoil:one-level@10 (best rule) | 85.63% | 1143.6 |
| kc1 | PFoil:one-level@10 (macro distribution voting) | 85.97% | 1143.6 |
| kc1 | PFoil:one-level@10 (micro distribution voting) | 84.83% | 1143.6 |
| kc1 | PFoil:one-level@10 (voting) | 84.45% | 1143.6 |
| kc1 | PFoil:one-level@25 (best rule) | 85.25% | 2861.5 |
| kc1 | PFoil:one-level@25 (macro distribution voting) | 85.73% | 2861.5 |
| kc1 | PFoil:one-level@25 (micro distribution voting) | 84.87% | 2861.5 |
| kc1 | PFoil:one-level@25 (voting) | 84.87% | 2861.5 |
| kc1 | PFoil:one-level@50 (best rule) | 85.59% | 5708.8 |
| kc1 | PFoil:one-level@50 (macro distribution voting) | 85.97% | 5708.8 |
| kc1 | PFoil:one-level@50 (micro distribution voting) | 84.87% | 5708.8 |
| kc1 | PFoil:one-level@50 (voting) | 85.54% | 5708.8 |
| kc1 | PFoil:single (best rule) | 85.87% | 130.9 |
| kc1 | PFoil:single (macro distribution voting) | 86.06% | 130.9 |
| kc1 | PFoil:single (micro distribution voting) | 85.87% | 130.9 |
| kc1 | PFoil:single (voting) | 83.45% | 130.9 |
| kc1 | PFoil:two-level@10 (best rule) | 85.87% | 1143.6 |
| kc1 | PFoil:two-level@10 (macro distribution voting) | 85.82% | 1143.6 |
| kc1 | PFoil:two-level@10 (micro distribution voting) | 84.97% | 1143.6 |
| kc1 | PFoil:two-level@10 (voting) | 85.25% | 1143.6 |
| kc1 | PFoil:two-level@25 (best rule) | 86.11% | 2861.5 |
| kc1 | PFoil:two-level@25 (macro distribution voting) | 86.49% | 2861.5 |
| kc1 | PFoil:two-level@25 (micro distribution voting) | 85.73% | 2861.5 |
| kc1 | PFoil:two-level@25 (voting) | 85.35% | 2861.5 |
| kc1 | PFoil:two-level@50 (best rule) | 86.34% | 5708.8 |
| kc1 | PFoil:two-level@50 (macro distribution voting) | 86.20% | 5708.8 |
| kc1 | PFoil:two-level@50 (micro distribution voting) | 85.40% | 5708.8 |
| kc1 | PFoil:two-level@50 (voting) | 85.78% | 5708.8 |
| kc1 | PFossil:one-level@10 (best rule) | 85.16% | 50.3 |
| kc1 | PFossil:one-level@10 (macro distribution voting) | 84.92% | 50.3 |
| kc1 | PFossil:one-level@10 (micro distribution voting) | 84.92% | 50.3 |
| kc1 | PFossil:one-level@10 (voting) | 82.27% | 50.3 |
| kc1 | PFossil:one-level@25 (best rule) | 84.83% | 132.8 |
| kc1 | PFossil:one-level@25 (macro distribution voting) | 84.78% | 132.8 |
| kc1 | PFossil:one-level@25 (micro distribution voting) | 84.78% | 132.8 |
| kc1 | PFossil:one-level@25 (voting) | 82.32% | 132.8 |
| kc1 | PFossil:one-level@50 (best rule) | 84.83% | 271.4 |
| kc1 | PFossil:one-level@50 (macro distribution voting) | 84.64% | 271.4 |
| kc1 | PFossil:one-level@50 (micro distribution voting) | 84.64% | 271.4 |
| kc1 | PFossil:one-level@50 (voting) | 82.60% | 271.4 |
| kc1 | PFossil:single (best rule) | 83.31% | 3.5 |
| kc1 | PFossil:single (macro distribution voting) | 83.31% | 3.5 |
| kc1 | PFossil:single (micro distribution voting) | 83.31% | 3.5 |
| kc1 | PFossil:single (voting) | 83.31% | 3.5 |
| kc1 | PFossil:two-level@10 (best rule) | 84.97% | 50.3 |
| kc1 | PFossil:two-level@10 (macro distribution voting) | 84.97% | 50.3 |
| kc1 | PFossil:two-level@10 (micro distribution voting) | 84.97% | 50.3 |
| kc1 | PFossil:two-level@10 (voting) | 84.97% | 50.3 |
| kc1 | PFossil:two-level@25 (best rule) | 85.35% | 132.8 |
| kc1 | PFossil:two-level@25 (macro distribution voting) | 85.35% | 132.8 |
| kc1 | PFossil:two-level@25 (micro distribution voting) | 85.35% | 132.8 |
| kc1 | PFossil:two-level@25 (voting) | 85.25% | 132.8 |
| kc1 | PFossil:two-level@50 (best rule) | 85.54% | 271.4 |
| kc1 | PFossil:two-level@50 (macro distribution voting) | 85.54% | 271.4 |
| kc1 | PFossil:two-level@50 (micro distribution voting) | 85.54% | 271.4 |
| kc1 | PFossil:two-level@50 (voting) | 85.59% | 271.4 |
| kc1 | Pypper:one-level@10 (best rule) | 85.11% | 74.7 |
| kc1 | Pypper:one-level@10 (macro distribution voting) | 85.63% | 74.7 |
| kc1 | Pypper:one-level@10 (micro distribution voting) | 85.25% | 74.7 |
| kc1 | Pypper:one-level@10 (voting) | 85.11% | 74.7 |
| kc1 | Pypper:one-level@25 (best rule) | 85.49% | 167.8 |
| kc1 | Pypper:one-level@25 (macro distribution voting) | 85.40% | 167.8 |
| kc1 | Pypper:one-level@25 (micro distribution voting) | 85.11% | 167.8 |
| kc1 | Pypper:one-level@25 (voting) | 84.92% | 167.8 |
| kc1 | Pypper:one-level@50 (best rule) | 85.44% | 330.8 |
| kc1 | Pypper:one-level@50 (macro distribution voting) | 85.82% | 330.8 |
| kc1 | Pypper:one-level@50 (micro distribution voting) | 85.49% | 330.8 |
| kc1 | Pypper:one-level@50 (voting) | 84.87% | 330.8 |
| kc1 | Pypper:single (best rule) | 84.49% | 4.1 |
| kc1 | Pypper:single (macro distribution voting) | 84.49% | 4.1 |
| kc1 | Pypper:single (micro distribution voting) | 84.45% | 4.1 |
| kc1 | Pypper:single (voting) | 84.49% | 4.1 |
| kc1 | Pypper:two-level@10 (best rule) | 85.40% | 74.7 |
| kc1 | Pypper:two-level@10 (macro distribution voting) | 85.44% | 74.7 |
| kc1 | Pypper:two-level@10 (micro distribution voting) | 85.21% | 74.7 |
| kc1 | Pypper:two-level@10 (voting) | 85.35% | 74.7 |
| kc1 | Pypper:two-level@25 (best rule) | 84.78% | 167.8 |
| kc1 | Pypper:two-level@25 (macro distribution voting) | 84.78% | 167.8 |
| kc1 | Pypper:two-level@25 (micro distribution voting) | 84.78% | 167.8 |
| kc1 | Pypper:two-level@25 (voting) | 84.78% | 167.8 |
| kc1 | Pypper:two-level@50 (best rule) | 85.02% | 330.8 |
| kc1 | Pypper:two-level@50 (macro distribution voting) | 84.97% | 330.8 |
| kc1 | Pypper:two-level@50 (micro distribution voting) | 84.87% | 330.8 |
| kc1 | Pypper:two-level@50 (voting) | 84.87% | 330.8 |
| kc2 | PFoil:one-level@10 (best rule) | 84.26% | 346.2 |
| kc2 | PFoil:one-level@10 (macro distribution voting) | 84.66% | 346.2 |
| kc2 | PFoil:one-level@10 (micro distribution voting) | 83.70% | 346.2 |
| kc2 | PFoil:one-level@10 (voting) | 83.70% | 346.2 |
| kc2 | PFoil:one-level@25 (best rule) | 84.46% | 869.4 |
| kc2 | PFoil:one-level@25 (macro distribution voting) | 84.27% | 869.4 |
| kc2 | PFoil:one-level@25 (micro distribution voting) | 83.52% | 869.4 |
| kc2 | PFoil:one-level@25 (voting) | 83.50% | 869.4 |
| kc2 | PFoil:one-level@50 (best rule) | 85.03% | 1733.1 |
| kc2 | PFoil:one-level@50 (macro distribution voting) | 84.08% | 1733.1 |
| kc2 | PFoil:one-level@50 (micro distribution voting) | 84.09% | 1733.1 |
| kc2 | PFoil:one-level@50 (voting) | 82.54% | 1733.1 |
| kc2 | PFoil:single (best rule) | 82.55% | 41.0 |
| kc2 | PFoil:single (macro distribution voting) | 81.98% | 41.0 |
| kc2 | PFoil:single (micro distribution voting) | 81.98% | 41.0 |
| kc2 | PFoil:single (voting) | 81.41% | 41.0 |
| kc2 | PFoil:two-level@10 (best rule) | 84.46% | 346.2 |
| kc2 | PFoil:two-level@10 (macro distribution voting) | 84.47% | 346.2 |
| kc2 | PFoil:two-level@10 (micro distribution voting) | 84.08% | 346.2 |
| kc2 | PFoil:two-level@10 (voting) | 84.27% | 346.2 |
| kc2 | PFoil:two-level@25 (best rule) | 84.27% | 869.4 |
| kc2 | PFoil:two-level@25 (macro distribution voting) | 84.27% | 869.4 |
| kc2 | PFoil:two-level@25 (micro distribution voting) | 84.46% | 869.4 |
| kc2 | PFoil:two-level@25 (voting) | 84.84% | 869.4 |
| kc2 | PFoil:two-level@50 (best rule) | 84.27% | 1733.1 |
| kc2 | PFoil:two-level@50 (macro distribution voting) | 84.08% | 1733.1 |
| kc2 | PFoil:two-level@50 (micro distribution voting) | 83.88% | 1733.1 |
| kc2 | PFoil:two-level@50 (voting) | 84.46% | 1733.1 |
| kc2 | PFossil:one-level@10 (best rule) | 82.74% | 102.5 |
| kc2 | PFossil:one-level@10 (macro distribution voting) | 83.89% | 102.5 |
| kc2 | PFossil:one-level@10 (micro distribution voting) | 83.89% | 102.5 |
| kc2 | PFossil:one-level@10 (voting) | 83.70% | 102.5 |
| kc2 | PFossil:one-level@25 (best rule) | 82.74% | 253.9 |
| kc2 | PFossil:one-level@25 (macro distribution voting) | 83.32% | 253.9 |
| kc2 | PFossil:one-level@25 (micro distribution voting) | 83.89% | 253.9 |
| kc2 | PFossil:one-level@25 (voting) | 84.07% | 253.9 |
| kc2 | PFossil:one-level@50 (best rule) | 83.51% | 506.2 |
| kc2 | PFossil:one-level@50 (macro distribution voting) | 83.89% | 506.2 |
| kc2 | PFossil:one-level@50 (micro distribution voting) | 84.27% | 506.2 |
| kc2 | PFossil:one-level@50 (voting) | 83.88% | 506.2 |
| kc2 | PFossil:single (best rule) | 82.74% | 7.4 |
| kc2 | PFossil:single (macro distribution voting) | 82.35% | 7.4 |
| kc2 | PFossil:single (micro distribution voting) | 81.79% | 7.4 |
| kc2 | PFossil:single (voting) | 82.55% | 7.4 |
| kc2 | PFossil:two-level@10 (best rule) | 83.12% | 102.5 |
| kc2 | PFossil:two-level@10 (macro distribution voting) | 82.93% | 102.5 |
| kc2 | PFossil:two-level@10 (micro distribution voting) | 82.94% | 102.5 |
| kc2 | PFossil:two-level@10 (voting) | 83.32% | 102.5 |
| kc2 | PFossil:two-level@25 (best rule) | 83.69% | 253.9 |
| kc2 | PFossil:two-level@25 (macro distribution voting) | 83.31% | 253.9 |
| kc2 | PFossil:two-level@25 (micro distribution voting) | 82.92% | 253.9 |
| kc2 | PFossil:two-level@25 (voting) | 83.31% | 253.9 |
| kc2 | PFossil:two-level@50 (best rule) | 83.69% | 506.2 |
| kc2 | PFossil:two-level@50 (macro distribution voting) | 83.12% | 506.2 |
| kc2 | PFossil:two-level@50 (micro distribution voting) | 83.11% | 506.2 |
| kc2 | PFossil:two-level@50 (voting) | 83.32% | 506.2 |
| kc2 | Pypper:one-level@10 (best rule) | 83.49% | 57.2 |
| kc2 | Pypper:one-level@10 (macro distribution voting) | 84.27% | 57.2 |
| kc2 | Pypper:one-level@10 (micro distribution voting) | 84.85% | 57.2 |
| kc2 | Pypper:one-level@10 (voting) | 84.46% | 57.2 |
| kc2 | Pypper:one-level@25 (best rule) | 82.35% | 129.8 |
| kc2 | Pypper:one-level@25 (macro distribution voting) | 84.07% | 129.8 |
| kc2 | Pypper:one-level@25 (micro distribution voting) | 84.09% | 129.8 |
| kc2 | Pypper:one-level@25 (voting) | 83.88% | 129.8 |
| kc2 | Pypper:one-level@50 (best rule) | 82.93% | 252.5 |
| kc2 | Pypper:one-level@50 (macro distribution voting) | 84.84% | 252.5 |
| kc2 | Pypper:one-level@50 (micro distribution voting) | 83.71% | 252.5 |
| kc2 | Pypper:one-level@50 (voting) | 83.88% | 252.5 |
| kc2 | Pypper:single (best rule) | 83.69% | 3.7 |
| kc2 | Pypper:single (macro distribution voting) | 83.69% | 3.7 |
| kc2 | Pypper:single (micro distribution voting) | 83.69% | 3.7 |
| kc2 | Pypper:single (voting) | 83.69% | 3.7 |
| kc2 | Pypper:two-level@10 (best rule) | 83.88% | 57.2 |
| kc2 | Pypper:two-level@10 (macro distribution voting) | 84.07% | 57.2 |
| kc2 | Pypper:two-level@10 (micro distribution voting) | 83.69% | 57.2 |
| kc2 | Pypper:two-level@10 (voting) | 83.88% | 57.2 |
| kc2 | Pypper:two-level@25 (best rule) | 84.07% | 129.8 |
| kc2 | Pypper:two-level@25 (macro distribution voting) | 83.50% | 129.8 |
| kc2 | Pypper:two-level@25 (micro distribution voting) | 83.88% | 129.8 |
| kc2 | Pypper:two-level@25 (voting) | 83.69% | 129.8 |
| kc2 | Pypper:two-level@50 (best rule) | 83.69% | 252.5 |
| kc2 | Pypper:two-level@50 (macro distribution voting) | 83.88% | 252.5 |
| kc2 | Pypper:two-level@50 (micro distribution voting) | 83.31% | 252.5 |
| kc2 | Pypper:two-level@50 (voting) | 83.88% | 252.5 |
| kr-vs-kp | PFoil:one-level@10 (best rule) | 99.09% | 367.5 |
| kr-vs-kp | PFoil:one-level@10 (macro distribution voting) | 99.53% | 367.5 |
| kr-vs-kp | PFoil:one-level@10 (micro distribution voting) | 98.87% | 367.5 |
| kr-vs-kp | PFoil:one-level@10 (voting) | 99.47% | 367.5 |
| kr-vs-kp | PFoil:one-level@25 (best rule) | 98.72% | 919.3 |
| kr-vs-kp | PFoil:one-level@25 (macro distribution voting) | 99.47% | 919.3 |
| kr-vs-kp | PFoil:one-level@25 (micro distribution voting) | 99.19% | 919.3 |
| kr-vs-kp | PFoil:one-level@25 (voting) | 99.44% | 919.3 |
| kr-vs-kp | PFoil:one-level@50 (best rule) | 98.40% | 1835.2 |
| kr-vs-kp | PFoil:one-level@50 (macro distribution voting) | 99.47% | 1835.2 |
| kr-vs-kp | PFoil:one-level@50 (micro distribution voting) | 99.12% | 1835.2 |
| kr-vs-kp | PFoil:one-level@50 (voting) | 99.44% | 1835.2 |
| kr-vs-kp | PFoil:single (best rule) | 99.34% | 40.8 |
| kr-vs-kp | PFoil:single (macro distribution voting) | 99.22% | 40.8 |
| kr-vs-kp | PFoil:single (micro distribution voting) | 99.09% | 40.8 |
| kr-vs-kp | PFoil:single (voting) | 98.84% | 40.8 |
| kr-vs-kp | PFoil:two-level@10 (best rule) | 99.47% | 367.5 |
| kr-vs-kp | PFoil:two-level@10 (macro distribution voting) | 99.47% | 367.5 |
| kr-vs-kp | PFoil:two-level@10 (micro distribution voting) | 99.34% | 367.5 |
| kr-vs-kp | PFoil:two-level@10 (voting) | 99.28% | 367.5 |
| kr-vs-kp | PFoil:two-level@25 (best rule) | 99.50% | 919.3 |
| kr-vs-kp | PFoil:two-level@25 (macro distribution voting) | 99.50% | 919.3 |
| kr-vs-kp | PFoil:two-level@25 (micro distribution voting) | 99.41% | 919.3 |
| kr-vs-kp | PFoil:two-level@25 (voting) | 99.44% | 919.3 |
| kr-vs-kp | PFoil:two-level@50 (best rule) | 99.44% | 1835.2 |
| kr-vs-kp | PFoil:two-level@50 (macro distribution voting) | 99.44% | 1835.2 |
| kr-vs-kp | PFoil:two-level@50 (micro distribution voting) | 99.34% | 1835.2 |
| kr-vs-kp | PFoil:two-level@50 (voting) | 99.50% | 1835.2 |
| kr-vs-kp | PFossil:one-level@10 (best rule) | 99.12% | 141.0 |
| kr-vs-kp | PFossil:one-level@10 (macro distribution voting) | 98.87% | 141.0 |
| kr-vs-kp | PFossil:one-level@10 (micro distribution voting) | 94.31% | 141.0 |
| kr-vs-kp | PFossil:one-level@10 (voting) | 96.21% | 141.0 |
| kr-vs-kp | PFossil:one-level@25 (best rule) | 98.65% | 350.7 |
| kr-vs-kp | PFossil:one-level@25 (macro distribution voting) | 99.09% | 350.7 |
| kr-vs-kp | PFossil:one-level@25 (micro distribution voting) | 94.31% | 350.7 |
| kr-vs-kp | PFossil:one-level@25 (voting) | 96.50% | 350.7 |
| kr-vs-kp | PFossil:one-level@50 (best rule) | 98.56% | 698.2 |
| kr-vs-kp | PFossil:one-level@50 (macro distribution voting) | 99.22% | 698.2 |
| kr-vs-kp | PFossil:one-level@50 (micro distribution voting) | 94.21% | 698.2 |
| kr-vs-kp | PFossil:one-level@50 (voting) | 96.62% | 698.2 |
| kr-vs-kp | PFossil:single (best rule) | 99.37% | 14.5 |
| kr-vs-kp | PFossil:single (macro distribution voting) | 99.41% | 14.5 |
| kr-vs-kp | PFossil:single (micro distribution voting) | 94.31% | 14.5 |
| kr-vs-kp | PFossil:single (voting) | 96.43% | 14.5 |
| kr-vs-kp | PFossil:two-level@10 (best rule) | 99.31% | 141.0 |
| kr-vs-kp | PFossil:two-level@10 (macro distribution voting) | 99.41% | 141.0 |
| kr-vs-kp | PFossil:two-level@10 (micro distribution voting) | 94.31% | 141.0 |
| kr-vs-kp | PFossil:two-level@10 (voting) | 96.46% | 141.0 |
| kr-vs-kp | PFossil:two-level@25 (best rule) | 99.31% | 350.7 |
| kr-vs-kp | PFossil:two-level@25 (macro distribution voting) | 99.34% | 350.7 |
| kr-vs-kp | PFossil:two-level@25 (micro distribution voting) | 94.31% | 350.7 |
| kr-vs-kp | PFossil:two-level@25 (voting) | 96.40% | 350.7 |
| kr-vs-kp | PFossil:two-level@50 (best rule) | 99.34% | 698.2 |
| kr-vs-kp | PFossil:two-level@50 (macro distribution voting) | 99.37% | 698.2 |
| kr-vs-kp | PFossil:two-level@50 (micro distribution voting) | 94.31% | 698.2 |
| kr-vs-kp | PFossil:two-level@50 (voting) | 96.40% | 698.2 |
| kr-vs-kp | Pypper:one-level@10 (best rule) | 98.75% | 223.3 |
| kr-vs-kp | Pypper:one-level@10 (macro distribution voting) | 99.31% | 223.3 |
| kr-vs-kp | Pypper:one-level@10 (micro distribution voting) | 98.50% | 223.3 |
| kr-vs-kp | Pypper:one-level@10 (voting) | 99.37% | 223.3 |
| kr-vs-kp | Pypper:one-level@25 (best rule) | 98.56% | 552.4 |
| kr-vs-kp | Pypper:one-level@25 (macro distribution voting) | 99.44% | 552.4 |
| kr-vs-kp | Pypper:one-level@25 (micro distribution voting) | 98.56% | 552.4 |
| kr-vs-kp | Pypper:one-level@25 (voting) | 99.37% | 552.4 |
| kr-vs-kp | Pypper:one-level@50 (best rule) | 98.53% | 1111.4 |
| kr-vs-kp | Pypper:one-level@50 (macro distribution voting) | 99.37% | 1111.4 |
| kr-vs-kp | Pypper:one-level@50 (micro distribution voting) | 98.62% | 1111.4 |
| kr-vs-kp | Pypper:one-level@50 (voting) | 99.37% | 1111.4 |
| kr-vs-kp | Pypper:single (best rule) | 99.25% | 23.5 |
| kr-vs-kp | Pypper:single (macro distribution voting) | 99.22% | 23.5 |
| kr-vs-kp | Pypper:single (micro distribution voting) | 99.16% | 23.5 |
| kr-vs-kp | Pypper:single (voting) | 99.00% | 23.5 |
| kr-vs-kp | Pypper:two-level@10 (best rule) | 99.37% | 223.3 |
| kr-vs-kp | Pypper:two-level@10 (macro distribution voting) | 99.34% | 223.3 |
| kr-vs-kp | Pypper:two-level@10 (micro distribution voting) | 99.25% | 223.3 |
| kr-vs-kp | Pypper:two-level@10 (voting) | 99.22% | 223.3 |
| kr-vs-kp | Pypper:two-level@25 (best rule) | 99.37% | 552.4 |
| kr-vs-kp | Pypper:two-level@25 (macro distribution voting) | 99.44% | 552.4 |
| kr-vs-kp | Pypper:two-level@25 (micro distribution voting) | 99.28% | 552.4 |
| kr-vs-kp | Pypper:two-level@25 (voting) | 99.31% | 552.4 |
| kr-vs-kp | Pypper:two-level@50 (best rule) | 99.41% | 1111.4 |
| kr-vs-kp | Pypper:two-level@50 (macro distribution voting) | 99.34% | 1111.4 |
| kr-vs-kp | Pypper:two-level@50 (micro distribution voting) | 99.31% | 1111.4 |
| kr-vs-kp | Pypper:two-level@50 (voting) | 99.31% | 1111.4 |
| led24 | PFoil:one-level@10 (best rule) | 70.97% | 3316.9 |
| led24 | PFoil:one-level@10 (macro distribution voting) | 71.16% | 3316.9 |
| led24 | PFoil:one-level@10 (micro distribution voting) | 71.50% | 3316.9 |
| led24 | PFoil:one-level@10 (voting) | 69.91% | 3316.9 |
| led24 | PFoil:one-level@25 (best rule) | 71.62% | 8286.4 |
| led24 | PFoil:one-level@25 (macro distribution voting) | 71.66% | 8286.4 |
| led24 | PFoil:one-level@25 (micro distribution voting) | 72.41% | 8286.4 |
| led24 | PFoil:one-level@25 (voting) | 70.91% | 8286.4 |
| led24 | PFoil:one-level@50 (best rule) | 72.00% | 16575.1 |
| led24 | PFoil:one-level@50 (macro distribution voting) | 72.12% | 16575.1 |
| led24 | PFoil:one-level@50 (micro distribution voting) | 72.69% | 16575.1 |
| led24 | PFoil:one-level@50 (voting) | 71.72% | 16575.1 |
| led24 | PFoil:single (best rule) | 61.81% | 444.2 |
| led24 | PFoil:single (macro distribution voting) | 62.34% | 444.2 |
| led24 | PFoil:single (micro distribution voting) | 61.97% | 444.2 |
| led24 | PFoil:single (voting) | 56.84% | 444.2 |
| led24 | PFoil:two-level@10 (best rule) | 70.47% | 3316.9 |
| led24 | PFoil:two-level@10 (macro distribution voting) | 70.28% | 3316.9 |
| led24 | PFoil:two-level@10 (micro distribution voting) | 70.31% | 3316.9 |
| led24 | PFoil:two-level@10 (voting) | 68.66% | 3316.9 |
| led24 | PFoil:two-level@25 (best rule) | 71.88% | 8286.4 |
| led24 | PFoil:two-level@25 (macro distribution voting) | 71.44% | 8286.4 |
| led24 | PFoil:two-level@25 (micro distribution voting) | 71.75% | 8286.4 |
| led24 | PFoil:two-level@25 (voting) | 70.81% | 8286.4 |
| led24 | PFoil:two-level@50 (best rule) | 71.78% | 16575.1 |
| led24 | PFoil:two-level@50 (macro distribution voting) | 72.00% | 16575.1 |
| led24 | PFoil:two-level@50 (micro distribution voting) | 72.19% | 16575.1 |
| led24 | PFoil:two-level@50 (voting) | 71.59% | 16575.1 |
| led24 | PFossil:one-level@10 (best rule) | 60.53% | 742.2 |
| led24 | PFossil:one-level@10 (macro distribution voting) | 71.78% | 742.2 |
| led24 | PFossil:one-level@10 (micro distribution voting) | 71.09% | 742.2 |
| led24 | PFossil:one-level@10 (voting) | 71.75% | 742.2 |
| led24 | PFossil:one-level@25 (best rule) | 57.03% | 1856.1 |
| led24 | PFossil:one-level@25 (macro distribution voting) | 71.44% | 1856.1 |
| led24 | PFossil:one-level@25 (micro distribution voting) | 71.44% | 1856.1 |
| led24 | PFossil:one-level@25 (voting) | 71.47% | 1856.1 |
| led24 | PFossil:one-level@50 (best rule) | 54.37% | 3696.4 |
| led24 | PFossil:one-level@50 (macro distribution voting) | 71.44% | 3696.4 |
| led24 | PFossil:one-level@50 (micro distribution voting) | 71.41% | 3696.4 |
| led24 | PFossil:one-level@50 (voting) | 71.81% | 3696.4 |
| led24 | PFossil:single (best rule) | 70.00% | 45.4 |
| led24 | PFossil:single (macro distribution voting) | 70.28% | 45.4 |
| led24 | PFossil:single (micro distribution voting) | 70.56% | 45.4 |
| led24 | PFossil:single (voting) | 70.34% | 45.4 |
| led24 | PFossil:two-level@10 (best rule) | 71.34% | 742.2 |
| led24 | PFossil:two-level@10 (macro distribution voting) | 71.41% | 742.2 |
| led24 | PFossil:two-level@10 (micro distribution voting) | 70.97% | 742.2 |
| led24 | PFossil:two-level@10 (voting) | 70.94% | 742.2 |
| led24 | PFossil:two-level@25 (best rule) | 71.44% | 1856.1 |
| led24 | PFossil:two-level@25 (macro distribution voting) | 71.62% | 1856.1 |
| led24 | PFossil:two-level@25 (micro distribution voting) | 71.66% | 1856.1 |
| led24 | PFossil:two-level@25 (voting) | 71.22% | 1856.1 |
| led24 | PFossil:two-level@50 (best rule) | 71.28% | 3696.4 |
| led24 | PFossil:two-level@50 (macro distribution voting) | 71.44% | 3696.4 |
| led24 | PFossil:two-level@50 (micro distribution voting) | 71.50% | 3696.4 |
| led24 | PFossil:two-level@50 (voting) | 71.47% | 3696.4 |
| led24 | Pypper:one-level@10 (best rule) | 67.31% | 375.5 |
| led24 | Pypper:one-level@10 (macro distribution voting) | 71.66% | 375.5 |
| led24 | Pypper:one-level@10 (micro distribution voting) | 72.34% | 375.5 |
| led24 | Pypper:one-level@10 (voting) | 71.56% | 375.5 |
| led24 | Pypper:one-level@25 (best rule) | 66.62% | 956.0 |
| led24 | Pypper:one-level@25 (macro distribution voting) | 72.12% | 956.0 |
| led24 | Pypper:one-level@25 (micro distribution voting) | 72.78% | 956.0 |
| led24 | Pypper:one-level@25 (voting) | 72.06% | 956.0 |
| led24 | Pypper:one-level@50 (best rule) | 66.47% | 1929.7 |
| led24 | Pypper:one-level@50 (macro distribution voting) | 72.09% | 1929.7 |
| led24 | Pypper:one-level@50 (micro distribution voting) | 72.59% | 1929.7 |
| led24 | Pypper:one-level@50 (voting) | 72.22% | 1929.7 |
| led24 | Pypper:single (best rule) | 68.25% | 33.9 |
| led24 | Pypper:single (macro distribution voting) | 68.22% | 33.9 |
| led24 | Pypper:single (micro distribution voting) | 68.28% | 33.9 |
| led24 | Pypper:single (voting) | 68.12% | 33.9 |
| led24 | Pypper:two-level@10 (best rule) | 71.41% | 375.5 |
| led24 | Pypper:two-level@10 (macro distribution voting) | 71.41% | 375.5 |
| led24 | Pypper:two-level@10 (micro distribution voting) | 71.97% | 375.5 |
| led24 | Pypper:two-level@10 (voting) | 71.28% | 375.5 |
| led24 | Pypper:two-level@25 (best rule) | 71.69% | 956.0 |
| led24 | Pypper:two-level@25 (macro distribution voting) | 71.91% | 956.0 |
| led24 | Pypper:two-level@25 (micro distribution voting) | 72.31% | 956.0 |
| led24 | Pypper:two-level@25 (voting) | 72.06% | 956.0 |
| led24 | Pypper:two-level@50 (best rule) | 72.28% | 1929.7 |
| led24 | Pypper:two-level@50 (macro distribution voting) | 72.31% | 1929.7 |
| led24 | Pypper:two-level@50 (micro distribution voting) | 72.66% | 1929.7 |
| led24 | Pypper:two-level@50 (voting) | 72.47% | 1929.7 |
| letter | PFoil:one-level@10 (best rule) | 87.52% | 9440.8 |
| letter | PFoil:one-level@10 (macro distribution voting) | 93.27% | 9440.8 |
| letter | PFoil:one-level@10 (micro distribution voting) | 81.04% | 9440.8 |
| letter | PFoil:one-level@10 (voting) | 92.35% | 9440.8 |
| letter | PFoil:one-level@25 (best rule) | 85.76% | 23662.2 |
| letter | PFoil:one-level@25 (macro distribution voting) | 93.66% | 23662.2 |
| letter | PFoil:one-level@25 (micro distribution voting) | 82.21% | 23662.2 |
| letter | PFoil:one-level@25 (voting) | 93.16% | 23662.2 |
| letter | PFoil:one-level@50 (best rule) | 84.25% | 47344.3 |
| letter | PFoil:one-level@50 (macro distribution voting) | 93.89% | 47344.3 |
| letter | PFoil:one-level@50 (micro distribution voting) | 82.65% | 47344.3 |
| letter | PFoil:one-level@50 (voting) | 93.38% | 47344.3 |
| letter | PFoil:single (best rule) | 88.78% | 1083.6 |
| letter | PFoil:single (macro distribution voting) | 89.12% | 1083.6 |
| letter | PFoil:single (micro distribution voting) | 72.10% | 1083.6 |
| letter | PFoil:single (voting) | 80.75% | 1083.6 |
| letter | PFoil:two-level@10 (best rule) | 92.56% | 9440.8 |
| letter | PFoil:two-level@10 (macro distribution voting) | 92.89% | 9440.8 |
| letter | PFoil:two-level@10 (micro distribution voting) | 89.34% | 9440.8 |
| letter | PFoil:two-level@10 (voting) | 90.38% | 9440.8 |
| letter | PFoil:two-level@25 (best rule) | 93.28% | 23662.2 |
| letter | PFoil:two-level@25 (macro distribution voting) | 93.71% | 23662.2 |
| letter | PFoil:two-level@25 (micro distribution voting) | 90.95% | 23662.2 |
| letter | PFoil:two-level@25 (voting) | 91.96% | 23662.2 |
| letter | PFoil:two-level@50 (best rule) | 93.55% | 47344.3 |
| letter | PFoil:two-level@50 (macro distribution voting) | 93.93% | 47344.3 |
| letter | PFoil:two-level@50 (micro distribution voting) | 91.48% | 47344.3 |
| letter | PFoil:two-level@50 (voting) | 92.69% | 47344.3 |
| letter | PFossil:one-level@10 (best rule) | 85.49% | 1583.4 |
| letter | PFossil:one-level@10 (macro distribution voting) | 86.37% | 1583.4 |
| letter | PFossil:one-level@10 (micro distribution voting) | 81.27% | 1583.4 |
| letter | PFossil:one-level@10 (voting) | 85.15% | 1583.4 |
| letter | PFossil:one-level@25 (best rule) | 86.24% | 3911.1 |
| letter | PFossil:one-level@25 (macro distribution voting) | 87.09% | 3911.1 |
| letter | PFossil:one-level@25 (micro distribution voting) | 81.86% | 3911.1 |
| letter | PFossil:one-level@25 (voting) | 85.97% | 3911.1 |
| letter | PFossil:one-level@50 (best rule) | 86.00% | 7834.6 |
| letter | PFossil:one-level@50 (macro distribution voting) | 87.33% | 7834.6 |
| letter | PFossil:one-level@50 (micro distribution voting) | 82.23% | 7834.6 |
| letter | PFossil:one-level@50 (voting) | 86.45% | 7834.6 |
| letter | PFossil:single (best rule) | 76.22% | 142.3 |
| letter | PFossil:single (macro distribution voting) | 76.44% | 142.3 |
| letter | PFossil:single (micro distribution voting) | 74.39% | 142.3 |
| letter | PFossil:single (voting) | 75.29% | 142.3 |
| letter | PFossil:two-level@10 (best rule) | 85.63% | 1583.4 |
| letter | PFossil:two-level@10 (macro distribution voting) | 85.99% | 1583.4 |
| letter | PFossil:two-level@10 (micro distribution voting) | 83.31% | 1583.4 |
| letter | PFossil:two-level@10 (voting) | 84.73% | 1583.4 |
| letter | PFossil:two-level@25 (best rule) | 86.84% | 3911.1 |
| letter | PFossil:two-level@25 (macro distribution voting) | 87.07% | 3911.1 |
| letter | PFossil:two-level@25 (micro distribution voting) | 84.14% | 3911.1 |
| letter | PFossil:two-level@25 (voting) | 85.93% | 3911.1 |
| letter | PFossil:two-level@50 (best rule) | 87.28% | 7834.6 |
| letter | PFossil:two-level@50 (macro distribution voting) | 87.41% | 7834.6 |
| letter | PFossil:two-level@50 (micro distribution voting) | 84.54% | 7834.6 |
| letter | PFossil:two-level@50 (voting) | 86.39% | 7834.6 |
| letter | Pypper:one-level@10 (best rule) | 87.00% | 3369.1 |
| letter | Pypper:one-level@10 (macro distribution voting) | 90.44% | 3369.1 |
| letter | Pypper:one-level@10 (micro distribution voting) | 87.29% | 3369.1 |
| letter | Pypper:one-level@10 (voting) | 90.20% | 3369.1 |
| letter | Pypper:one-level@25 (best rule) | 86.45% | 8496.2 |
| letter | Pypper:one-level@25 (macro distribution voting) | 91.33% | 8496.2 |
| letter | Pypper:one-level@25 (micro distribution voting) | 88.06% | 8496.2 |
| letter | Pypper:one-level@25 (voting) | 91.19% | 8496.2 |
| letter | Pypper:one-level@50 (best rule) | 85.70% | 17014.7 |
| letter | Pypper:one-level@50 (macro distribution voting) | 91.64% | 17014.7 |
| letter | Pypper:one-level@50 (micro distribution voting) | 88.29% | 17014.7 |
| letter | Pypper:one-level@50 (voting) | 91.59% | 17014.7 |
| letter | Pypper:single (best rule) | 81.02% | 378.0 |
| letter | Pypper:single (macro distribution voting) | 81.26% | 378.0 |
| letter | Pypper:single (micro distribution voting) | 80.67% | 378.0 |
| letter | Pypper:single (voting) | 80.88% | 378.0 |
| letter | Pypper:two-level@10 (best rule) | 89.89% | 3369.1 |
| letter | Pypper:two-level@10 (macro distribution voting) | 90.14% | 3369.1 |
| letter | Pypper:two-level@10 (micro distribution voting) | 89.28% | 3369.1 |
| letter | Pypper:two-level@10 (voting) | 89.65% | 3369.1 |
| letter | Pypper:two-level@25 (best rule) | 90.89% | 8496.2 |
| letter | Pypper:two-level@25 (macro distribution voting) | 91.21% | 8496.2 |
| letter | Pypper:two-level@25 (micro distribution voting) | 90.37% | 8496.2 |
| letter | Pypper:two-level@25 (voting) | 90.84% | 8496.2 |
| letter | Pypper:two-level@50 (best rule) | 91.26% | 17014.7 |
| letter | Pypper:two-level@50 (macro distribution voting) | 91.60% | 17014.7 |
| letter | Pypper:two-level@50 (micro distribution voting) | 90.82% | 17014.7 |
| letter | Pypper:two-level@50 (voting) | 91.44% | 17014.7 |
| lymph | PFoil:one-level@10 (best rule) | 78.52% | 140.2 |
| lymph | PFoil:one-level@10 (macro distribution voting) | 82.57% | 140.2 |
| lymph | PFoil:one-level@10 (micro distribution voting) | 80.52% | 140.2 |
| lymph | PFoil:one-level@10 (voting) | 81.24% | 140.2 |
| lymph | PFoil:one-level@25 (best rule) | 79.14% | 345.9 |
| lymph | PFoil:one-level@25 (macro distribution voting) | 85.24% | 345.9 |
| lymph | PFoil:one-level@25 (micro distribution voting) | 81.86% | 345.9 |
| lymph | PFoil:one-level@25 (voting) | 83.90% | 345.9 |
| lymph | PFoil:one-level@50 (best rule) | 77.76% | 693.6 |
| lymph | PFoil:one-level@50 (macro distribution voting) | 82.57% | 693.6 |
| lymph | PFoil:one-level@50 (micro distribution voting) | 81.86% | 693.6 |
| lymph | PFoil:one-level@50 (voting) | 82.57% | 693.6 |
| lymph | PFoil:single (best rule) | 82.57% | 17.1 |
| lymph | PFoil:single (macro distribution voting) | 81.19% | 17.1 |
| lymph | PFoil:single (micro distribution voting) | 77.86% | 17.1 |
| lymph | PFoil:single (voting) | 78.48% | 17.1 |
| lymph | PFoil:two-level@10 (best rule) | 82.52% | 140.2 |
| lymph | PFoil:two-level@10 (macro distribution voting) | 82.52% | 140.2 |
| lymph | PFoil:two-level@10 (micro distribution voting) | 81.86% | 140.2 |
| lymph | PFoil:two-level@10 (voting) | 82.52% | 140.2 |
| lymph | PFoil:two-level@25 (best rule) | 84.57% | 345.9 |
| lymph | PFoil:two-level@25 (macro distribution voting) | 84.57% | 345.9 |
| lymph | PFoil:two-level@25 (micro distribution voting) | 82.52% | 345.9 |
| lymph | PFoil:two-level@25 (voting) | 82.57% | 345.9 |
| lymph | PFoil:two-level@50 (best rule) | 83.90% | 693.6 |
| lymph | PFoil:two-level@50 (macro distribution voting) | 82.57% | 693.6 |
| lymph | PFoil:two-level@50 (micro distribution voting) | 82.52% | 693.6 |
| lymph | PFoil:two-level@50 (voting) | 83.24% | 693.6 |
| lymph | PFossil:one-level@10 (best rule) | 78.52% | 124.5 |
| lymph | PFossil:one-level@10 (macro distribution voting) | 83.81% | 124.5 |
| lymph | PFossil:one-level@10 (micro distribution voting) | 78.52% | 124.5 |
| lymph | PFossil:one-level@10 (voting) | 82.48% | 124.5 |
| lymph | PFossil:one-level@25 (best rule) | 81.19% | 303.8 |
| lymph | PFossil:one-level@25 (macro distribution voting) | 83.81% | 303.8 |
| lymph | PFossil:one-level@25 (micro distribution voting) | 79.86% | 303.8 |
| lymph | PFossil:one-level@25 (voting) | 83.86% | 303.8 |
| lymph | PFossil:one-level@50 (best rule) | 77.19% | 602.2 |
| lymph | PFossil:one-level@50 (macro distribution voting) | 83.81% | 602.2 |
| lymph | PFossil:one-level@50 (micro distribution voting) | 79.86% | 602.2 |
| lymph | PFossil:one-level@50 (voting) | 82.43% | 602.2 |
| lymph | PFossil:single (best rule) | 81.71% | 16.2 |
| lymph | PFossil:single (macro distribution voting) | 79.71% | 16.2 |
| lymph | PFossil:single (micro distribution voting) | 81.05% | 16.2 |
| lymph | PFossil:single (voting) | 81.10% | 16.2 |
| lymph | PFossil:two-level@10 (best rule) | 80.48% | 124.5 |
| lymph | PFossil:two-level@10 (macro distribution voting) | 82.43% | 124.5 |
| lymph | PFossil:two-level@10 (micro distribution voting) | 81.19% | 124.5 |
| lymph | PFossil:two-level@10 (voting) | 81.19% | 124.5 |
| lymph | PFossil:two-level@25 (best rule) | 83.86% | 303.8 |
| lymph | PFossil:two-level@25 (macro distribution voting) | 83.76% | 303.8 |
| lymph | PFossil:two-level@25 (micro distribution voting) | 81.14% | 303.8 |
| lymph | PFossil:two-level@25 (voting) | 81.10% | 303.8 |
| lymph | PFossil:two-level@50 (best rule) | 82.48% | 602.2 |
| lymph | PFossil:two-level@50 (macro distribution voting) | 83.14% | 602.2 |
| lymph | PFossil:two-level@50 (micro distribution voting) | 80.43% | 602.2 |
| lymph | PFossil:two-level@50 (voting) | 81.76% | 602.2 |
| lymph | Pypper:one-level@10 (best rule) | 81.86% | 57.5 |
| lymph | Pypper:one-level@10 (macro distribution voting) | 81.19% | 57.5 |
| lymph | Pypper:one-level@10 (micro distribution voting) | 80.48% | 57.5 |
| lymph | Pypper:one-level@10 (voting) | 80.52% | 57.5 |
| lymph | Pypper:one-level@25 (best rule) | 81.14% | 145.1 |
| lymph | Pypper:one-level@25 (macro distribution voting) | 80.52% | 145.1 |
| lymph | Pypper:one-level@25 (micro distribution voting) | 79.14% | 145.1 |
| lymph | Pypper:one-level@25 (voting) | 79.86% | 145.1 |
| lymph | Pypper:one-level@50 (best rule) | 78.48% | 294.3 |
| lymph | Pypper:one-level@50 (macro distribution voting) | 80.52% | 294.3 |
| lymph | Pypper:one-level@50 (micro distribution voting) | 79.14% | 294.3 |
| lymph | Pypper:one-level@50 (voting) | 79.86% | 294.3 |
| lymph | Pypper:single (best rule) | 79.81% | 5.7 |
| lymph | Pypper:single (macro distribution voting) | 81.81% | 5.7 |
| lymph | Pypper:single (micro distribution voting) | 77.10% | 5.7 |
| lymph | Pypper:single (voting) | 77.81% | 5.7 |
| lymph | Pypper:two-level@10 (best rule) | 80.52% | 57.5 |
| lymph | Pypper:two-level@10 (macro distribution voting) | 80.52% | 57.5 |
| lymph | Pypper:two-level@10 (micro distribution voting) | 79.14% | 57.5 |
| lymph | Pypper:two-level@10 (voting) | 77.81% | 57.5 |
| lymph | Pypper:two-level@25 (best rule) | 79.86% | 145.1 |
| lymph | Pypper:two-level@25 (macro distribution voting) | 81.19% | 145.1 |
| lymph | Pypper:two-level@25 (micro distribution voting) | 80.52% | 145.1 |
| lymph | Pypper:two-level@25 (voting) | 78.43% | 145.1 |
| lymph | Pypper:two-level@50 (best rule) | 78.48% | 294.3 |
| lymph | Pypper:two-level@50 (macro distribution voting) | 80.52% | 294.3 |
| lymph | Pypper:two-level@50 (micro distribution voting) | 79.81% | 294.3 |
| lymph | Pypper:two-level@50 (voting) | 77.76% | 294.3 |
| mofn-3-7-10 | PFoil:one-level@10 (best rule) | 89.80% | 615.6 |
| mofn-3-7-10 | PFoil:one-level@10 (macro distribution voting) | 100.00% | 615.6 |
| mofn-3-7-10 | PFoil:one-level@10 (micro distribution voting) | 98.26% | 615.6 |
| mofn-3-7-10 | PFoil:one-level@10 (voting) | 100.00% | 615.6 |
| mofn-3-7-10 | PFoil:one-level@25 (best rule) | 85.87% | 1533.2 |
| mofn-3-7-10 | PFoil:one-level@25 (macro distribution voting) | 100.00% | 1533.2 |
| mofn-3-7-10 | PFoil:one-level@25 (micro distribution voting) | 98.71% | 1533.2 |
| mofn-3-7-10 | PFoil:one-level@25 (voting) | 100.00% | 1533.2 |
| mofn-3-7-10 | PFoil:one-level@50 (best rule) | 84.67% | 3067.1 |
| mofn-3-7-10 | PFoil:one-level@50 (macro distribution voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:one-level@50 (micro distribution voting) | 98.87% | 3067.1 |
| mofn-3-7-10 | PFoil:one-level@50 (voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:single (best rule) | 100.00% | 56.0 |
| mofn-3-7-10 | PFoil:single (macro distribution voting) | 100.00% | 56.0 |
| mofn-3-7-10 | PFoil:single (micro distribution voting) | 100.00% | 56.0 |
| mofn-3-7-10 | PFoil:single (voting) | 100.00% | 56.0 |
| mofn-3-7-10 | PFoil:two-level@10 (best rule) | 99.77% | 615.6 |
| mofn-3-7-10 | PFoil:two-level@10 (macro distribution voting) | 99.77% | 615.6 |
| mofn-3-7-10 | PFoil:two-level@10 (micro distribution voting) | 99.77% | 615.6 |
| mofn-3-7-10 | PFoil:two-level@10 (voting) | 99.77% | 615.6 |
| mofn-3-7-10 | PFoil:two-level@25 (best rule) | 100.00% | 1533.2 |
| mofn-3-7-10 | PFoil:two-level@25 (macro distribution voting) | 100.00% | 1533.2 |
| mofn-3-7-10 | PFoil:two-level@25 (micro distribution voting) | 100.00% | 1533.2 |
| mofn-3-7-10 | PFoil:two-level@25 (voting) | 100.00% | 1533.2 |
| mofn-3-7-10 | PFoil:two-level@50 (best rule) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:two-level@50 (macro distribution voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:two-level@50 (micro distribution voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:two-level@50 (voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFossil:one-level@10 (best rule) | 89.50% | 155.9 |
| mofn-3-7-10 | PFossil:one-level@10 (macro distribution voting) | 97.96% | 155.9 |
| mofn-3-7-10 | PFossil:one-level@10 (micro distribution voting) | 83.68% | 155.9 |
| mofn-3-7-10 | PFossil:one-level@10 (voting) | 99.62% | 155.9 |
| mofn-3-7-10 | PFossil:one-level@25 (best rule) | 86.78% | 380.7 |
| mofn-3-7-10 | PFossil:one-level@25 (macro distribution voting) | 99.55% | 380.7 |
| mofn-3-7-10 | PFossil:one-level@25 (micro distribution voting) | 83.76% | 380.7 |
| mofn-3-7-10 | PFossil:one-level@25 (voting) | 100.00% | 380.7 |
| mofn-3-7-10 | PFossil:one-level@50 (best rule) | 85.65% | 767.9 |
| mofn-3-7-10 | PFossil:one-level@50 (macro distribution voting) | 99.85% | 767.9 |
| mofn-3-7-10 | PFossil:one-level@50 (micro distribution voting) | 83.61% | 767.9 |
| mofn-3-7-10 | PFossil:one-level@50 (voting) | 100.00% | 767.9 |
| mofn-3-7-10 | PFossil:single (best rule) | 83.68% | 6.4 |
| mofn-3-7-10 | PFossil:single (macro distribution voting) | 83.76% | 6.4 |
| mofn-3-7-10 | PFossil:single (micro distribution voting) | 83.31% | 6.4 |
| mofn-3-7-10 | PFossil:single (voting) | 83.61% | 6.4 |
| mofn-3-7-10 | PFossil:two-level@10 (best rule) | 95.16% | 155.9 |
| mofn-3-7-10 | PFossil:two-level@10 (macro distribution voting) | 96.45% | 155.9 |
| mofn-3-7-10 | PFossil:two-level@10 (micro distribution voting) | 89.19% | 155.9 |
| mofn-3-7-10 | PFossil:two-level@10 (voting) | 91.61% | 155.9 |
| mofn-3-7-10 | PFossil:two-level@25 (best rule) | 97.35% | 380.7 |
| mofn-3-7-10 | PFossil:two-level@25 (macro distribution voting) | 99.32% | 380.7 |
| mofn-3-7-10 | PFossil:two-level@25 (micro distribution voting) | 91.91% | 380.7 |
| mofn-3-7-10 | PFossil:two-level@25 (voting) | 94.03% | 380.7 |
| mofn-3-7-10 | PFossil:two-level@50 (best rule) | 97.73% | 767.9 |
| mofn-3-7-10 | PFossil:two-level@50 (macro distribution voting) | 99.47% | 767.9 |
| mofn-3-7-10 | PFossil:two-level@50 (micro distribution voting) | 90.48% | 767.9 |
| mofn-3-7-10 | PFossil:two-level@50 (voting) | 93.65% | 767.9 |
| mofn-3-7-10 | Pypper:one-level@10 (best rule) | 86.63% | 242.4 |
| mofn-3-7-10 | Pypper:one-level@10 (macro distribution voting) | 99.85% | 242.4 |
| mofn-3-7-10 | Pypper:one-level@10 (micro distribution voting) | 85.57% | 242.4 |
| mofn-3-7-10 | Pypper:one-level@10 (voting) | 100.00% | 242.4 |
| mofn-3-7-10 | Pypper:one-level@25 (best rule) | 85.72% | 614.5 |
| mofn-3-7-10 | Pypper:one-level@25 (macro distribution voting) | 100.00% | 614.5 |
| mofn-3-7-10 | Pypper:one-level@25 (micro distribution voting) | 84.74% | 614.5 |
| mofn-3-7-10 | Pypper:one-level@25 (voting) | 100.00% | 614.5 |
| mofn-3-7-10 | Pypper:one-level@50 (best rule) | 84.59% | 1223.2 |
| mofn-3-7-10 | Pypper:one-level@50 (macro distribution voting) | 100.00% | 1223.2 |
| mofn-3-7-10 | Pypper:one-level@50 (micro distribution voting) | 84.52% | 1223.2 |
| mofn-3-7-10 | Pypper:one-level@50 (voting) | 100.00% | 1223.2 |
| mofn-3-7-10 | Pypper:single (best rule) | 97.21% | 31.6 |
| mofn-3-7-10 | Pypper:single (macro distribution voting) | 97.43% | 31.6 |
| mofn-3-7-10 | Pypper:single (micro distribution voting) | 93.35% | 31.6 |
| mofn-3-7-10 | Pypper:single (voting) | 93.58% | 31.6 |
| mofn-3-7-10 | Pypper:two-level@10 (best rule) | 96.00% | 242.4 |
| mofn-3-7-10 | Pypper:two-level@10 (macro distribution voting) | 98.57% | 242.4 |
| mofn-3-7-10 | Pypper:two-level@10 (micro distribution voting) | 93.88% | 242.4 |
| mofn-3-7-10 | Pypper:two-level@10 (voting) | 94.33% | 242.4 |
| mofn-3-7-10 | Pypper:two-level@25 (best rule) | 97.51% | 614.5 |
| mofn-3-7-10 | Pypper:two-level@25 (macro distribution voting) | 99.85% | 614.5 |
| mofn-3-7-10 | Pypper:two-level@25 (micro distribution voting) | 96.38% | 614.5 |
| mofn-3-7-10 | Pypper:two-level@25 (voting) | 97.13% | 614.5 |
| mofn-3-7-10 | Pypper:two-level@50 (best rule) | 98.79% | 1223.2 |
| mofn-3-7-10 | Pypper:two-level@50 (macro distribution voting) | 100.00% | 1223.2 |
| mofn-3-7-10 | Pypper:two-level@50 (micro distribution voting) | 97.58% | 1223.2 |
| mofn-3-7-10 | Pypper:two-level@50 (voting) | 98.19% | 1223.2 |
| molecular-biology_promoters | PFoil:one-level@10 (best rule) | 82.09% | 69.6 |
| molecular-biology_promoters | PFoil:one-level@10 (macro distribution voting) | 90.64% | 69.6 |
| molecular-biology_promoters | PFoil:one-level@10 (micro distribution voting) | 87.82% | 69.6 |
| molecular-biology_promoters | PFoil:one-level@10 (voting) | 90.64% | 69.6 |
| molecular-biology_promoters | PFoil:one-level@25 (best rule) | 78.91% | 175.4 |
| molecular-biology_promoters | PFoil:one-level@25 (macro distribution voting) | 90.64% | 175.4 |
| molecular-biology_promoters | PFoil:one-level@25 (micro distribution voting) | 89.73% | 175.4 |
| molecular-biology_promoters | PFoil:one-level@25 (voting) | 90.64% | 175.4 |
| molecular-biology_promoters | PFoil:one-level@50 (best rule) | 85.73% | 347.1 |
| molecular-biology_promoters | PFoil:one-level@50 (macro distribution voting) | 90.55% | 347.1 |
| molecular-biology_promoters | PFoil:one-level@50 (micro distribution voting) | 89.64% | 347.1 |
| molecular-biology_promoters | PFoil:one-level@50 (voting) | 93.45% | 347.1 |
| molecular-biology_promoters | PFoil:single (best rule) | 86.09% | 9.0 |
| molecular-biology_promoters | PFoil:single (macro distribution voting) | 85.18% | 9.0 |
| molecular-biology_promoters | PFoil:single (micro distribution voting) | 86.09% | 9.0 |
| molecular-biology_promoters | PFoil:single (voting) | 83.36% | 9.0 |
| molecular-biology_promoters | PFoil:two-level@10 (best rule) | 86.73% | 69.6 |
| molecular-biology_promoters | PFoil:two-level@10 (macro distribution voting) | 87.82% | 69.6 |
| molecular-biology_promoters | PFoil:two-level@10 (micro distribution voting) | 86.73% | 69.6 |
| molecular-biology_promoters | PFoil:two-level@10 (voting) | 85.91% | 69.6 |
| molecular-biology_promoters | PFoil:two-level@25 (best rule) | 88.73% | 175.4 |
| molecular-biology_promoters | PFoil:two-level@25 (macro distribution voting) | 90.55% | 175.4 |
| molecular-biology_promoters | PFoil:two-level@25 (micro distribution voting) | 88.73% | 175.4 |
| molecular-biology_promoters | PFoil:two-level@25 (voting) | 91.55% | 175.4 |
| molecular-biology_promoters | PFoil:two-level@50 (best rule) | 90.64% | 347.1 |
| molecular-biology_promoters | PFoil:two-level@50 (macro distribution voting) | 91.55% | 347.1 |
| molecular-biology_promoters | PFoil:two-level@50 (micro distribution voting) | 90.64% | 347.1 |
| molecular-biology_promoters | PFoil:two-level@50 (voting) | 92.45% | 347.1 |
| molecular-biology_promoters | PFossil:one-level@10 (best rule) | 80.09% | 67.6 |
| molecular-biology_promoters | PFossil:one-level@10 (macro distribution voting) | 89.64% | 67.6 |
| molecular-biology_promoters | PFossil:one-level@10 (micro distribution voting) | 89.64% | 67.6 |
| molecular-biology_promoters | PFossil:one-level@10 (voting) | 87.82% | 67.6 |
| molecular-biology_promoters | PFossil:one-level@25 (best rule) | 76.18% | 168.2 |
| molecular-biology_promoters | PFossil:one-level@25 (macro distribution voting) | 91.64% | 168.2 |
| molecular-biology_promoters | PFossil:one-level@25 (micro distribution voting) | 90.73% | 168.2 |
| molecular-biology_promoters | PFossil:one-level@25 (voting) | 90.64% | 168.2 |
| molecular-biology_promoters | PFossil:one-level@50 (best rule) | 75.27% | 330.2 |
| molecular-biology_promoters | PFossil:one-level@50 (macro distribution voting) | 92.55% | 330.2 |
| molecular-biology_promoters | PFossil:one-level@50 (micro distribution voting) | 90.73% | 330.2 |
| molecular-biology_promoters | PFossil:one-level@50 (voting) | 92.55% | 330.2 |
| molecular-biology_promoters | PFossil:single (best rule) | 82.27% | 7.4 |
| molecular-biology_promoters | PFossil:single (macro distribution voting) | 82.18% | 7.4 |
| molecular-biology_promoters | PFossil:single (micro distribution voting) | 81.36% | 7.4 |
| molecular-biology_promoters | PFossil:single (voting) | 81.27% | 7.4 |
| molecular-biology_promoters | PFossil:two-level@10 (best rule) | 87.82% | 67.6 |
| molecular-biology_promoters | PFossil:two-level@10 (macro distribution voting) | 89.64% | 67.6 |
| molecular-biology_promoters | PFossil:two-level@10 (micro distribution voting) | 83.91% | 67.6 |
| molecular-biology_promoters | PFossil:two-level@10 (voting) | 86.00% | 67.6 |
| molecular-biology_promoters | PFossil:two-level@25 (best rule) | 90.73% | 168.2 |
| molecular-biology_promoters | PFossil:two-level@25 (macro distribution voting) | 93.45% | 168.2 |
| molecular-biology_promoters | PFossil:two-level@25 (micro distribution voting) | 87.73% | 168.2 |
| molecular-biology_promoters | PFossil:two-level@25 (voting) | 90.64% | 168.2 |
| molecular-biology_promoters | PFossil:two-level@50 (best rule) | 92.55% | 330.2 |
| molecular-biology_promoters | PFossil:two-level@50 (macro distribution voting) | 93.45% | 330.2 |
| molecular-biology_promoters | PFossil:two-level@50 (micro distribution voting) | 92.55% | 330.2 |
| molecular-biology_promoters | PFossil:two-level@50 (voting) | 92.45% | 330.2 |
| molecular-biology_promoters | Pypper:one-level@10 (best rule) | 86.00% | 40.0 |
| molecular-biology_promoters | Pypper:one-level@10 (macro distribution voting) | 84.73% | 40.0 |
| molecular-biology_promoters | Pypper:one-level@10 (micro distribution voting) | 84.73% | 40.0 |
| molecular-biology_promoters | Pypper:one-level@10 (voting) | 84.73% | 40.0 |
| molecular-biology_promoters | Pypper:one-level@25 (best rule) | 81.09% | 100.7 |
| molecular-biology_promoters | Pypper:one-level@25 (macro distribution voting) | 85.82% | 100.7 |
| molecular-biology_promoters | Pypper:one-level@25 (micro distribution voting) | 87.64% | 100.7 |
| molecular-biology_promoters | Pypper:one-level@25 (voting) | 86.73% | 100.7 |
| molecular-biology_promoters | Pypper:one-level@50 (best rule) | 81.00% | 200.3 |
| molecular-biology_promoters | Pypper:one-level@50 (macro distribution voting) | 88.73% | 200.3 |
| molecular-biology_promoters | Pypper:one-level@50 (micro distribution voting) | 87.82% | 200.3 |
| molecular-biology_promoters | Pypper:one-level@50 (voting) | 87.73% | 200.3 |
| molecular-biology_promoters | Pypper:single (best rule) | 81.36% | 4.3 |
| molecular-biology_promoters | Pypper:single (macro distribution voting) | 81.36% | 4.3 |
| molecular-biology_promoters | Pypper:single (micro distribution voting) | 81.18% | 4.3 |
| molecular-biology_promoters | Pypper:single (voting) | 80.27% | 4.3 |
| molecular-biology_promoters | Pypper:two-level@10 (best rule) | 82.91% | 40.0 |
| molecular-biology_promoters | Pypper:two-level@10 (macro distribution voting) | 82.91% | 40.0 |
| molecular-biology_promoters | Pypper:two-level@10 (micro distribution voting) | 83.91% | 40.0 |
| molecular-biology_promoters | Pypper:two-level@10 (voting) | 83.91% | 40.0 |
| molecular-biology_promoters | Pypper:two-level@25 (best rule) | 86.64% | 100.7 |
| molecular-biology_promoters | Pypper:two-level@25 (macro distribution voting) | 86.64% | 100.7 |
| molecular-biology_promoters | Pypper:two-level@25 (micro distribution voting) | 86.73% | 100.7 |
| molecular-biology_promoters | Pypper:two-level@25 (voting) | 87.64% | 100.7 |
| molecular-biology_promoters | Pypper:two-level@50 (best rule) | 86.82% | 200.3 |
| molecular-biology_promoters | Pypper:two-level@50 (macro distribution voting) | 87.73% | 200.3 |
| molecular-biology_promoters | Pypper:two-level@50 (micro distribution voting) | 86.82% | 200.3 |
| molecular-biology_promoters | Pypper:two-level@50 (voting) | 88.64% | 200.3 |
| monks-problems-1 | PFoil:one-level@10 (best rule) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:one-level@10 (macro distribution voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:one-level@10 (micro distribution voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:one-level@10 (voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:one-level@25 (best rule) | 100.00% | 316.7 |
| monks-problems-1 | PFoil:one-level@25 (macro distribution voting) | 100.00% | 316.7 |
| monks-problems-1 | PFoil:one-level@25 (micro distribution voting) | 100.00% | 316.7 |
| monks-problems-1 | PFoil:one-level@25 (voting) | 100.00% | 316.7 |
| monks-problems-1 | PFoil:one-level@50 (best rule) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:one-level@50 (macro distribution voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:one-level@50 (micro distribution voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:one-level@50 (voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:single (best rule) | 100.00% | 12.0 |
| monks-problems-1 | PFoil:single (macro distribution voting) | 100.00% | 12.0 |
| monks-problems-1 | PFoil:single (micro distribution voting) | 100.00% | 12.0 |
| monks-problems-1 | PFoil:single (voting) | 100.00% | 12.0 |
| monks-problems-1 | PFoil:two-level@10 (best rule) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:two-level@10 (macro distribution voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:two-level@10 (micro distribution voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:two-level@10 (voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:two-level@25 (best rule) | 100.00% | 316.7 |
| monks-problems-1 | PFoil:two-level@25 (macro distribution voting) | 100.00% | 316.7 |
| monks-problems-1 | PFoil:two-level@25 (micro distribution voting) | 100.00% | 316.7 |
| monks-problems-1 | PFoil:two-level@25 (voting) | 100.00% | 316.7 |
| monks-problems-1 | PFoil:two-level@50 (best rule) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:two-level@50 (macro distribution voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:two-level@50 (micro distribution voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:two-level@50 (voting) | 100.00% | 638.9 |
| monks-problems-1 | PFossil:one-level@10 (best rule) | 86.36% | 32.5 |
| monks-problems-1 | PFossil:one-level@10 (macro distribution voting) | 83.81% | 32.5 |
| monks-problems-1 | PFossil:one-level@10 (micro distribution voting) | 74.64% | 32.5 |
| monks-problems-1 | PFossil:one-level@10 (voting) | 74.64% | 32.5 |
| monks-problems-1 | PFossil:one-level@25 (best rule) | 87.07% | 89.3 |
| monks-problems-1 | PFossil:one-level@25 (macro distribution voting) | 87.75% | 89.3 |
| monks-problems-1 | PFossil:one-level@25 (micro distribution voting) | 74.64% | 89.3 |
| monks-problems-1 | PFossil:one-level@25 (voting) | 74.64% | 89.3 |
| monks-problems-1 | PFossil:one-level@50 (best rule) | 80.22% | 179.2 |
| monks-problems-1 | PFossil:one-level@50 (macro distribution voting) | 88.31% | 179.2 |
| monks-problems-1 | PFossil:one-level@50 (micro distribution voting) | 74.64% | 179.2 |
| monks-problems-1 | PFossil:one-level@50 (voting) | 74.64% | 179.2 |
| monks-problems-1 | PFossil:single (best rule) | 80.41% | 4.0 |
| monks-problems-1 | PFossil:single (macro distribution voting) | 80.41% | 4.0 |
| monks-problems-1 | PFossil:single (micro distribution voting) | 74.64% | 4.0 |
| monks-problems-1 | PFossil:single (voting) | 75.91% | 4.0 |
| monks-problems-1 | PFossil:two-level@10 (best rule) | 79.49% | 32.5 |
| monks-problems-1 | PFossil:two-level@10 (macro distribution voting) | 79.85% | 32.5 |
| monks-problems-1 | PFossil:two-level@10 (micro distribution voting) | 74.64% | 32.5 |
| monks-problems-1 | PFossil:two-level@10 (voting) | 80.05% | 32.5 |
| monks-problems-1 | PFossil:two-level@25 (best rule) | 78.78% | 89.3 |
| monks-problems-1 | PFossil:two-level@25 (macro distribution voting) | 78.78% | 89.3 |
| monks-problems-1 | PFossil:two-level@25 (micro distribution voting) | 74.64% | 89.3 |
| monks-problems-1 | PFossil:two-level@25 (voting) | 76.96% | 89.3 |
| monks-problems-1 | PFossil:two-level@50 (best rule) | 75.36% | 179.2 |
| monks-problems-1 | PFossil:two-level@50 (macro distribution voting) | 75.91% | 179.2 |
| monks-problems-1 | PFossil:two-level@50 (micro distribution voting) | 74.64% | 179.2 |
| monks-problems-1 | PFossil:two-level@50 (voting) | 74.64% | 179.2 |
| monks-problems-1 | Pypper:one-level@10 (best rule) | 100.00% | 66.9 |
| monks-problems-1 | Pypper:one-level@10 (macro distribution voting) | 100.00% | 66.9 |
| monks-problems-1 | Pypper:one-level@10 (micro distribution voting) | 94.26% | 66.9 |
| monks-problems-1 | Pypper:one-level@10 (voting) | 100.00% | 66.9 |
| monks-problems-1 | Pypper:one-level@25 (best rule) | 100.00% | 170.0 |
| monks-problems-1 | Pypper:one-level@25 (macro distribution voting) | 100.00% | 170.0 |
| monks-problems-1 | Pypper:one-level@25 (micro distribution voting) | 94.99% | 170.0 |
| monks-problems-1 | Pypper:one-level@25 (voting) | 100.00% | 170.0 |
| monks-problems-1 | Pypper:one-level@50 (best rule) | 100.00% | 345.9 |
| monks-problems-1 | Pypper:one-level@50 (macro distribution voting) | 100.00% | 345.9 |
| monks-problems-1 | Pypper:one-level@50 (micro distribution voting) | 99.27% | 345.9 |
| monks-problems-1 | Pypper:one-level@50 (voting) | 100.00% | 345.9 |
| monks-problems-1 | Pypper:single (best rule) | 96.76% | 7.3 |
| monks-problems-1 | Pypper:single (macro distribution voting) | 96.76% | 7.3 |
| monks-problems-1 | Pypper:single (micro distribution voting) | 92.06% | 7.3 |
| monks-problems-1 | Pypper:single (voting) | 94.06% | 7.3 |
| monks-problems-1 | Pypper:two-level@10 (best rule) | 100.00% | 66.9 |
| monks-problems-1 | Pypper:two-level@10 (macro distribution voting) | 100.00% | 66.9 |
| monks-problems-1 | Pypper:two-level@10 (micro distribution voting) | 98.55% | 66.9 |
| monks-problems-1 | Pypper:two-level@10 (voting) | 100.00% | 66.9 |
| monks-problems-1 | Pypper:two-level@25 (best rule) | 100.00% | 170.0 |
| monks-problems-1 | Pypper:two-level@25 (macro distribution voting) | 100.00% | 170.0 |
| monks-problems-1 | Pypper:two-level@25 (micro distribution voting) | 100.00% | 170.0 |
| monks-problems-1 | Pypper:two-level@25 (voting) | 100.00% | 170.0 |
| monks-problems-1 | Pypper:two-level@50 (best rule) | 100.00% | 345.9 |
| monks-problems-1 | Pypper:two-level@50 (macro distribution voting) | 100.00% | 345.9 |
| monks-problems-1 | Pypper:two-level@50 (micro distribution voting) | 100.00% | 345.9 |
| monks-problems-1 | Pypper:two-level@50 (voting) | 100.00% | 345.9 |
| monks-problems-2 | PFoil:one-level@10 (best rule) | 81.86% | 616.5 |
| monks-problems-2 | PFoil:one-level@10 (macro distribution voting) | 90.51% | 616.5 |
| monks-problems-2 | PFoil:one-level@10 (micro distribution voting) | 80.86% | 616.5 |
| monks-problems-2 | PFoil:one-level@10 (voting) | 92.17% | 616.5 |
| monks-problems-2 | PFoil:one-level@25 (best rule) | 77.53% | 1556.9 |
| monks-problems-2 | PFoil:one-level@25 (macro distribution voting) | 90.68% | 1556.9 |
| monks-problems-2 | PFoil:one-level@25 (micro distribution voting) | 81.36% | 1556.9 |
| monks-problems-2 | PFoil:one-level@25 (voting) | 93.51% | 1556.9 |
| monks-problems-2 | PFoil:one-level@50 (best rule) | 73.71% | 3080.6 |
| monks-problems-2 | PFoil:one-level@50 (macro distribution voting) | 90.68% | 3080.6 |
| monks-problems-2 | PFoil:one-level@50 (micro distribution voting) | 81.19% | 3080.6 |
| monks-problems-2 | PFoil:one-level@50 (voting) | 93.67% | 3080.6 |
| monks-problems-2 | PFoil:single (best rule) | 97.50% | 49.3 |
| monks-problems-2 | PFoil:single (macro distribution voting) | 97.17% | 49.3 |
| monks-problems-2 | PFoil:single (micro distribution voting) | 97.50% | 49.3 |
| monks-problems-2 | PFoil:single (voting) | 97.34% | 49.3 |
| monks-problems-2 | PFoil:two-level@10 (best rule) | 95.51% | 616.5 |
| monks-problems-2 | PFoil:two-level@10 (macro distribution voting) | 94.68% | 616.5 |
| monks-problems-2 | PFoil:two-level@10 (micro distribution voting) | 93.35% | 616.5 |
| monks-problems-2 | PFoil:two-level@10 (voting) | 91.84% | 616.5 |
| monks-problems-2 | PFoil:two-level@25 (best rule) | 95.34% | 1556.9 |
| monks-problems-2 | PFoil:two-level@25 (macro distribution voting) | 93.84% | 1556.9 |
| monks-problems-2 | PFoil:two-level@25 (micro distribution voting) | 93.84% | 1556.9 |
| monks-problems-2 | PFoil:two-level@25 (voting) | 93.51% | 1556.9 |
| monks-problems-2 | PFoil:two-level@50 (best rule) | 95.84% | 3080.6 |
| monks-problems-2 | PFoil:two-level@50 (macro distribution voting) | 95.01% | 3080.6 |
| monks-problems-2 | PFoil:two-level@50 (micro distribution voting) | 93.68% | 3080.6 |
| monks-problems-2 | PFoil:two-level@50 (voting) | 93.84% | 3080.6 |
| monks-problems-2 | PFossil:one-level@10 (best rule) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:one-level@10 (macro distribution voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:one-level@10 (micro distribution voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:one-level@10 (voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:one-level@25 (best rule) | 66.06% | 4.6 |
| monks-problems-2 | PFossil:one-level@25 (macro distribution voting) | 66.06% | 4.6 |
| monks-problems-2 | PFossil:one-level@25 (micro distribution voting) | 66.06% | 4.6 |
| monks-problems-2 | PFossil:one-level@25 (voting) | 66.06% | 4.6 |
| monks-problems-2 | PFossil:one-level@50 (best rule) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:one-level@50 (macro distribution voting) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:one-level@50 (micro distribution voting) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:one-level@50 (voting) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:single (best rule) | 66.72% | 0.4 |
| monks-problems-2 | PFossil:single (macro distribution voting) | 66.72% | 0.4 |
| monks-problems-2 | PFossil:single (micro distribution voting) | 66.72% | 0.4 |
| monks-problems-2 | PFossil:single (voting) | 66.72% | 0.4 |
| monks-problems-2 | PFossil:two-level@10 (best rule) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:two-level@10 (macro distribution voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:two-level@10 (micro distribution voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:two-level@10 (voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:two-level@25 (best rule) | 66.06% | 4.6 |
| monks-problems-2 | PFossil:two-level@25 (macro distribution voting) | 66.06% | 4.6 |
| monks-problems-2 | PFossil:two-level@25 (micro distribution voting) | 66.06% | 4.6 |
| monks-problems-2 | PFossil:two-level@25 (voting) | 66.06% | 4.6 |
| monks-problems-2 | PFossil:two-level@50 (best rule) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:two-level@50 (macro distribution voting) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:two-level@50 (micro distribution voting) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:two-level@50 (voting) | 69.54% | 9.4 |
| monks-problems-2 | Pypper:one-level@10 (best rule) | 77.21% | 87.8 |
| monks-problems-2 | Pypper:one-level@10 (macro distribution voting) | 71.72% | 87.8 |
| monks-problems-2 | Pypper:one-level@10 (micro distribution voting) | 67.39% | 87.8 |
| monks-problems-2 | Pypper:one-level@10 (voting) | 69.06% | 87.8 |
| monks-problems-2 | Pypper:one-level@25 (best rule) | 77.54% | 225.9 |
| monks-problems-2 | Pypper:one-level@25 (macro distribution voting) | 70.05% | 225.9 |
| monks-problems-2 | Pypper:one-level@25 (micro distribution voting) | 65.72% | 225.9 |
| monks-problems-2 | Pypper:one-level@25 (voting) | 68.72% | 225.9 |
| monks-problems-2 | Pypper:one-level@50 (best rule) | 82.03% | 447.1 |
| monks-problems-2 | Pypper:one-level@50 (macro distribution voting) | 69.88% | 447.1 |
| monks-problems-2 | Pypper:one-level@50 (micro distribution voting) | 65.72% | 447.1 |
| monks-problems-2 | Pypper:one-level@50 (voting) | 69.22% | 447.1 |
| monks-problems-2 | Pypper:single (best rule) | 74.03% | 14.5 |
| monks-problems-2 | Pypper:single (macro distribution voting) | 74.03% | 14.5 |
| monks-problems-2 | Pypper:single (micro distribution voting) | 74.36% | 14.5 |
| monks-problems-2 | Pypper:single (voting) | 74.20% | 14.5 |
| monks-problems-2 | Pypper:two-level@10 (best rule) | 70.89% | 87.8 |
| monks-problems-2 | Pypper:two-level@10 (macro distribution voting) | 70.89% | 87.8 |
| monks-problems-2 | Pypper:two-level@10 (micro distribution voting) | 69.89% | 87.8 |
| monks-problems-2 | Pypper:two-level@10 (voting) | 69.89% | 87.8 |
| monks-problems-2 | Pypper:two-level@25 (best rule) | 70.89% | 225.9 |
| monks-problems-2 | Pypper:two-level@25 (macro distribution voting) | 70.72% | 225.9 |
| monks-problems-2 | Pypper:two-level@25 (micro distribution voting) | 70.39% | 225.9 |
| monks-problems-2 | Pypper:two-level@25 (voting) | 70.22% | 225.9 |
| monks-problems-2 | Pypper:two-level@50 (best rule) | 72.38% | 447.1 |
| monks-problems-2 | Pypper:two-level@50 (macro distribution voting) | 72.04% | 447.1 |
| monks-problems-2 | Pypper:two-level@50 (micro distribution voting) | 71.38% | 447.1 |
| monks-problems-2 | Pypper:two-level@50 (voting) | 71.22% | 447.1 |
| monks-problems-3 | PFoil:one-level@10 (best rule) | 98.92% | 162.4 |
| monks-problems-3 | PFoil:one-level@10 (macro distribution voting) | 98.92% | 162.4 |
| monks-problems-3 | PFoil:one-level@10 (micro distribution voting) | 98.92% | 162.4 |
| monks-problems-3 | PFoil:one-level@10 (voting) | 98.20% | 162.4 |
| monks-problems-3 | PFoil:one-level@25 (best rule) | 98.92% | 403.7 |
| monks-problems-3 | PFoil:one-level@25 (macro distribution voting) | 98.74% | 403.7 |
| monks-problems-3 | PFoil:one-level@25 (micro distribution voting) | 98.92% | 403.7 |
| monks-problems-3 | PFoil:one-level@25 (voting) | 98.38% | 403.7 |
| monks-problems-3 | PFoil:one-level@50 (best rule) | 98.92% | 796.1 |
| monks-problems-3 | PFoil:one-level@50 (macro distribution voting) | 98.92% | 796.1 |
| monks-problems-3 | PFoil:one-level@50 (micro distribution voting) | 98.92% | 796.1 |
| monks-problems-3 | PFoil:one-level@50 (voting) | 98.56% | 796.1 |
| monks-problems-3 | PFoil:single (best rule) | 97.66% | 20.8 |
| monks-problems-3 | PFoil:single (macro distribution voting) | 97.66% | 20.8 |
| monks-problems-3 | PFoil:single (micro distribution voting) | 97.66% | 20.8 |
| monks-problems-3 | PFoil:single (voting) | 96.76% | 20.8 |
| monks-problems-3 | PFoil:two-level@10 (best rule) | 98.20% | 162.4 |
| monks-problems-3 | PFoil:two-level@10 (macro distribution voting) | 98.02% | 162.4 |
| monks-problems-3 | PFoil:two-level@10 (micro distribution voting) | 98.20% | 162.4 |
| monks-problems-3 | PFoil:two-level@10 (voting) | 97.84% | 162.4 |
| monks-problems-3 | PFoil:two-level@25 (best rule) | 98.02% | 403.7 |
| monks-problems-3 | PFoil:two-level@25 (macro distribution voting) | 98.02% | 403.7 |
| monks-problems-3 | PFoil:two-level@25 (micro distribution voting) | 98.02% | 403.7 |
| monks-problems-3 | PFoil:two-level@25 (voting) | 97.66% | 403.7 |
| monks-problems-3 | PFoil:two-level@50 (best rule) | 98.02% | 796.1 |
| monks-problems-3 | PFoil:two-level@50 (macro distribution voting) | 98.02% | 796.1 |
| monks-problems-3 | PFoil:two-level@50 (micro distribution voting) | 98.02% | 796.1 |
| monks-problems-3 | PFoil:two-level@50 (voting) | 97.84% | 796.1 |
| monks-problems-3 | PFossil:one-level@10 (best rule) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:one-level@10 (macro distribution voting) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:one-level@10 (micro distribution voting) | 96.40% | 64.8 |
| monks-problems-3 | PFossil:one-level@10 (voting) | 97.84% | 64.8 |
| monks-problems-3 | PFossil:one-level@25 (best rule) | 98.92% | 163.0 |
| monks-problems-3 | PFossil:one-level@25 (macro distribution voting) | 98.92% | 163.0 |
| monks-problems-3 | PFossil:one-level@25 (micro distribution voting) | 96.40% | 163.0 |
| monks-problems-3 | PFossil:one-level@25 (voting) | 98.20% | 163.0 |
| monks-problems-3 | PFossil:one-level@50 (best rule) | 98.92% | 321.7 |
| monks-problems-3 | PFossil:one-level@50 (macro distribution voting) | 98.92% | 321.7 |
| monks-problems-3 | PFossil:one-level@50 (micro distribution voting) | 96.40% | 321.7 |
| monks-problems-3 | PFossil:one-level@50 (voting) | 98.02% | 321.7 |
| monks-problems-3 | PFossil:single (best rule) | 98.92% | 6.8 |
| monks-problems-3 | PFossil:single (macro distribution voting) | 98.92% | 6.8 |
| monks-problems-3 | PFossil:single (micro distribution voting) | 96.40% | 6.8 |
| monks-problems-3 | PFossil:single (voting) | 98.38% | 6.8 |
| monks-problems-3 | PFossil:two-level@10 (best rule) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:two-level@10 (macro distribution voting) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:two-level@10 (micro distribution voting) | 96.40% | 64.8 |
| monks-problems-3 | PFossil:two-level@10 (voting) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:two-level@25 (best rule) | 98.92% | 163.0 |
| monks-problems-3 | PFossil:two-level@25 (macro distribution voting) | 98.92% | 163.0 |
| monks-problems-3 | PFossil:two-level@25 (micro distribution voting) | 96.40% | 163.0 |
| monks-problems-3 | PFossil:two-level@25 (voting) | 98.92% | 163.0 |
| monks-problems-3 | PFossil:two-level@50 (best rule) | 98.92% | 321.7 |
| monks-problems-3 | PFossil:two-level@50 (macro distribution voting) | 98.92% | 321.7 |
| monks-problems-3 | PFossil:two-level@50 (micro distribution voting) | 96.40% | 321.7 |
| monks-problems-3 | PFossil:two-level@50 (voting) | 98.92% | 321.7 |
| monks-problems-3 | Pypper:one-level@10 (best rule) | 98.92% | 51.5 |
| monks-problems-3 | Pypper:one-level@10 (macro distribution voting) | 98.92% | 51.5 |
| monks-problems-3 | Pypper:one-level@10 (micro distribution voting) | 98.56% | 51.5 |
| monks-problems-3 | Pypper:one-level@10 (voting) | 98.92% | 51.5 |
| monks-problems-3 | Pypper:one-level@25 (best rule) | 98.92% | 126.9 |
| monks-problems-3 | Pypper:one-level@25 (macro distribution voting) | 98.92% | 126.9 |
| monks-problems-3 | Pypper:one-level@25 (micro distribution voting) | 98.20% | 126.9 |
| monks-problems-3 | Pypper:one-level@25 (voting) | 98.92% | 126.9 |
| monks-problems-3 | Pypper:one-level@50 (best rule) | 98.92% | 255.8 |
| monks-problems-3 | Pypper:one-level@50 (macro distribution voting) | 98.92% | 255.8 |
| monks-problems-3 | Pypper:one-level@50 (micro distribution voting) | 98.92% | 255.8 |
| monks-problems-3 | Pypper:one-level@50 (voting) | 98.92% | 255.8 |
| monks-problems-3 | Pypper:single (best rule) | 98.92% | 5.0 |
| monks-problems-3 | Pypper:single (macro distribution voting) | 98.92% | 5.0 |
| monks-problems-3 | Pypper:single (micro distribution voting) | 98.92% | 5.0 |
| monks-problems-3 | Pypper:single (voting) | 98.92% | 5.0 |
| monks-problems-3 | Pypper:two-level@10 (best rule) | 98.92% | 51.5 |
| monks-problems-3 | Pypper:two-level@10 (macro distribution voting) | 98.92% | 51.5 |
| monks-problems-3 | Pypper:two-level@10 (micro distribution voting) | 98.92% | 51.5 |
| monks-problems-3 | Pypper:two-level@10 (voting) | 98.92% | 51.5 |
| monks-problems-3 | Pypper:two-level@25 (best rule) | 98.92% | 126.9 |
| monks-problems-3 | Pypper:two-level@25 (macro distribution voting) | 98.92% | 126.9 |
| monks-problems-3 | Pypper:two-level@25 (micro distribution voting) | 98.92% | 126.9 |
| monks-problems-3 | Pypper:two-level@25 (voting) | 98.92% | 126.9 |
| monks-problems-3 | Pypper:two-level@50 (best rule) | 98.92% | 255.8 |
| monks-problems-3 | Pypper:two-level@50 (macro distribution voting) | 98.92% | 255.8 |
| monks-problems-3 | Pypper:two-level@50 (micro distribution voting) | 98.92% | 255.8 |
| monks-problems-3 | Pypper:two-level@50 (voting) | 98.92% | 255.8 |
| mushroom | PFoil:one-level@10 (best rule) | 99.96% | 121.6 |
| mushroom | PFoil:one-level@10 (macro distribution voting) | 100.00% | 121.6 |
| mushroom | PFoil:one-level@10 (micro distribution voting) | 99.96% | 121.6 |
| mushroom | PFoil:one-level@10 (voting) | 100.00% | 121.6 |
| mushroom | PFoil:one-level@25 (best rule) | 99.93% | 304.8 |
| mushroom | PFoil:one-level@25 (macro distribution voting) | 100.00% | 304.8 |
| mushroom | PFoil:one-level@25 (micro distribution voting) | 99.93% | 304.8 |
| mushroom | PFoil:one-level@25 (voting) | 100.00% | 304.8 |
| mushroom | PFoil:one-level@50 (best rule) | 99.93% | 609.6 |
| mushroom | PFoil:one-level@50 (macro distribution voting) | 100.00% | 609.6 |
| mushroom | PFoil:one-level@50 (micro distribution voting) | 99.93% | 609.6 |
| mushroom | PFoil:one-level@50 (voting) | 100.00% | 609.6 |
| mushroom | PFoil:single (best rule) | 100.00% | 11.5 |
| mushroom | PFoil:single (macro distribution voting) | 100.00% | 11.5 |
| mushroom | PFoil:single (micro distribution voting) | 100.00% | 11.5 |
| mushroom | PFoil:single (voting) | 100.00% | 11.5 |
| mushroom | PFoil:two-level@10 (best rule) | 100.00% | 121.6 |
| mushroom | PFoil:two-level@10 (macro distribution voting) | 100.00% | 121.6 |
| mushroom | PFoil:two-level@10 (micro distribution voting) | 100.00% | 121.6 |
| mushroom | PFoil:two-level@10 (voting) | 100.00% | 121.6 |
| mushroom | PFoil:two-level@25 (best rule) | 100.00% | 304.8 |
| mushroom | PFoil:two-level@25 (macro distribution voting) | 100.00% | 304.8 |
| mushroom | PFoil:two-level@25 (micro distribution voting) | 100.00% | 304.8 |
| mushroom | PFoil:two-level@25 (voting) | 100.00% | 304.8 |
| mushroom | PFoil:two-level@50 (best rule) | 100.00% | 609.6 |
| mushroom | PFoil:two-level@50 (macro distribution voting) | 100.00% | 609.6 |
| mushroom | PFoil:two-level@50 (micro distribution voting) | 100.00% | 609.6 |
| mushroom | PFoil:two-level@50 (voting) | 100.00% | 609.6 |
| mushroom | PFossil:one-level@10 (best rule) | 99.84% | 117.3 |
| mushroom | PFossil:one-level@10 (macro distribution voting) | 99.95% | 117.3 |
| mushroom | PFossil:one-level@10 (micro distribution voting) | 99.83% | 117.3 |
| mushroom | PFossil:one-level@10 (voting) | 99.95% | 117.3 |
| mushroom | PFossil:one-level@25 (best rule) | 99.82% | 290.9 |
| mushroom | PFossil:one-level@25 (macro distribution voting) | 99.95% | 290.9 |
| mushroom | PFossil:one-level@25 (micro distribution voting) | 99.82% | 290.9 |
| mushroom | PFossil:one-level@25 (voting) | 99.95% | 290.9 |
| mushroom | PFossil:one-level@50 (best rule) | 99.82% | 582.8 |
| mushroom | PFossil:one-level@50 (macro distribution voting) | 99.95% | 582.8 |
| mushroom | PFossil:one-level@50 (micro distribution voting) | 99.80% | 582.8 |
| mushroom | PFossil:one-level@50 (voting) | 99.95% | 582.8 |
| mushroom | PFossil:single (best rule) | 99.95% | 12.3 |
| mushroom | PFossil:single (macro distribution voting) | 100.00% | 12.3 |
| mushroom | PFossil:single (micro distribution voting) | 99.95% | 12.3 |
| mushroom | PFossil:single (voting) | 99.95% | 12.3 |
| mushroom | PFossil:two-level@10 (best rule) | 99.95% | 117.3 |
| mushroom | PFossil:two-level@10 (macro distribution voting) | 99.99% | 117.3 |
| mushroom | PFossil:two-level@10 (micro distribution voting) | 99.95% | 117.3 |
| mushroom | PFossil:two-level@10 (voting) | 99.95% | 117.3 |
| mushroom | PFossil:two-level@25 (best rule) | 99.95% | 290.9 |
| mushroom | PFossil:two-level@25 (macro distribution voting) | 100.00% | 290.9 |
| mushroom | PFossil:two-level@25 (micro distribution voting) | 99.95% | 290.9 |
| mushroom | PFossil:two-level@25 (voting) | 99.95% | 290.9 |
| mushroom | PFossil:two-level@50 (best rule) | 99.95% | 582.8 |
| mushroom | PFossil:two-level@50 (macro distribution voting) | 100.00% | 582.8 |
| mushroom | PFossil:two-level@50 (micro distribution voting) | 99.95% | 582.8 |
| mushroom | PFossil:two-level@50 (voting) | 99.95% | 582.8 |
| mushroom | Pypper:one-level@10 (best rule) | 99.91% | 107.6 |
| mushroom | Pypper:one-level@10 (macro distribution voting) | 99.99% | 107.6 |
| mushroom | Pypper:one-level@10 (micro distribution voting) | 99.91% | 107.6 |
| mushroom | Pypper:one-level@10 (voting) | 99.98% | 107.6 |
| mushroom | Pypper:one-level@25 (best rule) | 99.85% | 269.8 |
| mushroom | Pypper:one-level@25 (macro distribution voting) | 99.98% | 269.8 |
| mushroom | Pypper:one-level@25 (micro distribution voting) | 99.83% | 269.8 |
| mushroom | Pypper:one-level@25 (voting) | 99.98% | 269.8 |
| mushroom | Pypper:one-level@50 (best rule) | 99.86% | 538.9 |
| mushroom | Pypper:one-level@50 (macro distribution voting) | 99.96% | 538.9 |
| mushroom | Pypper:one-level@50 (micro distribution voting) | 99.83% | 538.9 |
| mushroom | Pypper:one-level@50 (voting) | 99.96% | 538.9 |
| mushroom | Pypper:single (best rule) | 99.94% | 9.7 |
| mushroom | Pypper:single (macro distribution voting) | 99.94% | 9.7 |
| mushroom | Pypper:single (micro distribution voting) | 99.94% | 9.7 |
| mushroom | Pypper:single (voting) | 99.94% | 9.7 |
| mushroom | Pypper:two-level@10 (best rule) | 100.00% | 107.6 |
| mushroom | Pypper:two-level@10 (macro distribution voting) | 100.00% | 107.6 |
| mushroom | Pypper:two-level@10 (micro distribution voting) | 100.00% | 107.6 |
| mushroom | Pypper:two-level@10 (voting) | 100.00% | 107.6 |
| mushroom | Pypper:two-level@25 (best rule) | 99.98% | 269.8 |
| mushroom | Pypper:two-level@25 (macro distribution voting) | 99.98% | 269.8 |
| mushroom | Pypper:two-level@25 (micro distribution voting) | 99.98% | 269.8 |
| mushroom | Pypper:two-level@25 (voting) | 99.98% | 269.8 |
| mushroom | Pypper:two-level@50 (best rule) | 99.98% | 538.9 |
| mushroom | Pypper:two-level@50 (macro distribution voting) | 99.98% | 538.9 |
| mushroom | Pypper:two-level@50 (micro distribution voting) | 99.98% | 538.9 |
| mushroom | Pypper:two-level@50 (voting) | 99.98% | 538.9 |
| optdigits | PFoil:one-level@10 (best rule) | 93.04% | 1171.8 |
| optdigits | PFoil:one-level@10 (macro distribution voting) | 97.58% | 1171.8 |
| optdigits | PFoil:one-level@10 (micro distribution voting) | 96.69% | 1171.8 |
| optdigits | PFoil:one-level@10 (voting) | 97.58% | 1171.8 |
| optdigits | PFoil:one-level@25 (best rule) | 90.96% | 2916.1 |
| optdigits | PFoil:one-level@25 (macro distribution voting) | 97.86% | 2916.1 |
| optdigits | PFoil:one-level@25 (micro distribution voting) | 96.99% | 2916.1 |
| optdigits | PFoil:one-level@25 (voting) | 97.90% | 2916.1 |
| optdigits | PFoil:one-level@50 (best rule) | 89.45% | 5824.9 |
| optdigits | PFoil:one-level@50 (macro distribution voting) | 97.90% | 5824.9 |
| optdigits | PFoil:one-level@50 (micro distribution voting) | 97.05% | 5824.9 |
| optdigits | PFoil:one-level@50 (voting) | 97.94% | 5824.9 |
| optdigits | PFoil:single (best rule) | 93.35% | 138.8 |
| optdigits | PFoil:single (macro distribution voting) | 94.02% | 138.8 |
| optdigits | PFoil:single (micro distribution voting) | 92.44% | 138.8 |
| optdigits | PFoil:single (voting) | 92.88% | 138.8 |
| optdigits | PFoil:two-level@10 (best rule) | 97.06% | 1171.8 |
| optdigits | PFoil:two-level@10 (macro distribution voting) | 97.53% | 1171.8 |
| optdigits | PFoil:two-level@10 (micro distribution voting) | 97.03% | 1171.8 |
| optdigits | PFoil:two-level@10 (voting) | 97.28% | 1171.8 |
| optdigits | PFoil:two-level@25 (best rule) | 97.63% | 2916.1 |
| optdigits | PFoil:two-level@25 (macro distribution voting) | 97.92% | 2916.1 |
| optdigits | PFoil:two-level@25 (micro distribution voting) | 97.47% | 2916.1 |
| optdigits | PFoil:two-level@25 (voting) | 97.74% | 2916.1 |
| optdigits | PFoil:two-level@50 (best rule) | 97.65% | 5824.9 |
| optdigits | PFoil:two-level@50 (macro distribution voting) | 97.92% | 5824.9 |
| optdigits | PFoil:two-level@50 (micro distribution voting) | 97.56% | 5824.9 |
| optdigits | PFoil:two-level@50 (voting) | 97.86% | 5824.9 |
| optdigits | PFossil:one-level@10 (best rule) | 92.79% | 722.9 |
| optdigits | PFossil:one-level@10 (macro distribution voting) | 96.48% | 722.9 |
| optdigits | PFossil:one-level@10 (micro distribution voting) | 95.39% | 722.9 |
| optdigits | PFossil:one-level@10 (voting) | 96.37% | 722.9 |
| optdigits | PFossil:one-level@25 (best rule) | 91.19% | 1831.9 |
| optdigits | PFossil:one-level@25 (macro distribution voting) | 96.81% | 1831.9 |
| optdigits | PFossil:one-level@25 (micro distribution voting) | 95.75% | 1831.9 |
| optdigits | PFossil:one-level@25 (voting) | 96.80% | 1831.9 |
| optdigits | PFossil:one-level@50 (best rule) | 89.79% | 3682.1 |
| optdigits | PFossil:one-level@50 (macro distribution voting) | 97.01% | 3682.1 |
| optdigits | PFossil:one-level@50 (micro distribution voting) | 95.82% | 3682.1 |
| optdigits | PFossil:one-level@50 (voting) | 96.87% | 3682.1 |
| optdigits | PFossil:single (best rule) | 90.30% | 81.3 |
| optdigits | PFossil:single (macro distribution voting) | 90.39% | 81.3 |
| optdigits | PFossil:single (micro distribution voting) | 90.11% | 81.3 |
| optdigits | PFossil:single (voting) | 90.27% | 81.3 |
| optdigits | PFossil:two-level@10 (best rule) | 96.01% | 722.9 |
| optdigits | PFossil:two-level@10 (macro distribution voting) | 96.44% | 722.9 |
| optdigits | PFossil:two-level@10 (micro distribution voting) | 95.77% | 722.9 |
| optdigits | PFossil:two-level@10 (voting) | 96.12% | 722.9 |
| optdigits | PFossil:two-level@25 (best rule) | 96.64% | 1831.9 |
| optdigits | PFossil:two-level@25 (macro distribution voting) | 96.96% | 1831.9 |
| optdigits | PFossil:two-level@25 (micro distribution voting) | 96.23% | 1831.9 |
| optdigits | PFossil:two-level@25 (voting) | 96.64% | 1831.9 |
| optdigits | PFossil:two-level@50 (best rule) | 96.74% | 3682.1 |
| optdigits | PFossil:two-level@50 (macro distribution voting) | 97.10% | 3682.1 |
| optdigits | PFossil:two-level@50 (micro distribution voting) | 96.33% | 3682.1 |
| optdigits | PFossil:two-level@50 (voting) | 96.71% | 3682.1 |
| optdigits | Pypper:one-level@10 (best rule) | 93.11% | 709.2 |
| optdigits | Pypper:one-level@10 (macro distribution voting) | 96.87% | 709.2 |
| optdigits | Pypper:one-level@10 (micro distribution voting) | 96.00% | 709.2 |
| optdigits | Pypper:one-level@10 (voting) | 96.83% | 709.2 |
| optdigits | Pypper:one-level@25 (best rule) | 92.08% | 1795.3 |
| optdigits | Pypper:one-level@25 (macro distribution voting) | 97.22% | 1795.3 |
| optdigits | Pypper:one-level@25 (micro distribution voting) | 96.39% | 1795.3 |
| optdigits | Pypper:one-level@25 (voting) | 97.17% | 1795.3 |
| optdigits | Pypper:one-level@50 (best rule) | 90.93% | 3582.1 |
| optdigits | Pypper:one-level@50 (macro distribution voting) | 97.33% | 3582.1 |
| optdigits | Pypper:one-level@50 (micro distribution voting) | 96.37% | 3582.1 |
| optdigits | Pypper:one-level@50 (voting) | 97.31% | 3582.1 |
| optdigits | Pypper:single (best rule) | 91.39% | 81.2 |
| optdigits | Pypper:single (macro distribution voting) | 91.80% | 81.2 |
| optdigits | Pypper:single (micro distribution voting) | 91.51% | 81.2 |
| optdigits | Pypper:single (voting) | 91.53% | 81.2 |
| optdigits | Pypper:two-level@10 (best rule) | 96.42% | 709.2 |
| optdigits | Pypper:two-level@10 (macro distribution voting) | 96.76% | 709.2 |
| optdigits | Pypper:two-level@10 (micro distribution voting) | 96.23% | 709.2 |
| optdigits | Pypper:two-level@10 (voting) | 96.55% | 709.2 |
| optdigits | Pypper:two-level@25 (best rule) | 97.10% | 1795.3 |
| optdigits | Pypper:two-level@25 (macro distribution voting) | 97.17% | 1795.3 |
| optdigits | Pypper:two-level@25 (micro distribution voting) | 96.74% | 1795.3 |
| optdigits | Pypper:two-level@25 (voting) | 97.01% | 1795.3 |
| optdigits | Pypper:two-level@50 (best rule) | 97.17% | 3582.1 |
| optdigits | Pypper:two-level@50 (macro distribution voting) | 97.30% | 3582.1 |
| optdigits | Pypper:two-level@50 (micro distribution voting) | 96.85% | 3582.1 |
| optdigits | Pypper:two-level@50 (voting) | 97.17% | 3582.1 |
| ozone-level-8hr | PFoil:one-level@10 (best rule) | 93.69% | 433.9 |
| ozone-level-8hr | PFoil:one-level@10 (macro distribution voting) | 94.12% | 433.9 |
| ozone-level-8hr | PFoil:one-level@10 (micro distribution voting) | 93.69% | 433.9 |
| ozone-level-8hr | PFoil:one-level@10 (voting) | 94.28% | 433.9 |
| ozone-level-8hr | PFoil:one-level@25 (best rule) | 93.69% | 1095.5 |
| ozone-level-8hr | PFoil:one-level@25 (macro distribution voting) | 94.08% | 1095.5 |
| ozone-level-8hr | PFoil:one-level@25 (micro distribution voting) | 93.69% | 1095.5 |
| ozone-level-8hr | PFoil:one-level@25 (voting) | 94.32% | 1095.5 |
| ozone-level-8hr | PFoil:one-level@50 (best rule) | 93.69% | 2185.7 |
| ozone-level-8hr | PFoil:one-level@50 (macro distribution voting) | 93.92% | 2185.7 |
| ozone-level-8hr | PFoil:one-level@50 (micro distribution voting) | 93.69% | 2185.7 |
| ozone-level-8hr | PFoil:one-level@50 (voting) | 94.20% | 2185.7 |
| ozone-level-8hr | PFoil:single (best rule) | 93.29% | 56.6 |
| ozone-level-8hr | PFoil:single (macro distribution voting) | 93.53% | 56.6 |
| ozone-level-8hr | PFoil:single (micro distribution voting) | 93.37% | 56.6 |
| ozone-level-8hr | PFoil:single (voting) | 93.09% | 56.6 |
| ozone-level-8hr | PFoil:two-level@10 (best rule) | 94.04% | 433.9 |
| ozone-level-8hr | PFoil:two-level@10 (macro distribution voting) | 94.24% | 433.9 |
| ozone-level-8hr | PFoil:two-level@10 (micro distribution voting) | 94.04% | 433.9 |
| ozone-level-8hr | PFoil:two-level@10 (voting) | 94.20% | 433.9 |
| ozone-level-8hr | PFoil:two-level@25 (best rule) | 94.04% | 1095.5 |
| ozone-level-8hr | PFoil:two-level@25 (macro distribution voting) | 94.08% | 1095.5 |
| ozone-level-8hr | PFoil:two-level@25 (micro distribution voting) | 93.96% | 1095.5 |
| ozone-level-8hr | PFoil:two-level@25 (voting) | 94.16% | 1095.5 |
| ozone-level-8hr | PFoil:two-level@50 (best rule) | 93.92% | 2185.7 |
| ozone-level-8hr | PFoil:two-level@50 (macro distribution voting) | 94.16% | 2185.7 |
| ozone-level-8hr | PFoil:two-level@50 (micro distribution voting) | 93.88% | 2185.7 |
| ozone-level-8hr | PFoil:two-level@50 (voting) | 94.00% | 2185.7 |
| ozone-level-8hr | PFossil:one-level@10 (best rule) | 93.84% | 108.4 |
| ozone-level-8hr | PFossil:one-level@10 (macro distribution voting) | 93.84% | 108.4 |
| ozone-level-8hr | PFossil:one-level@10 (micro distribution voting) | 93.65% | 108.4 |
| ozone-level-8hr | PFossil:one-level@10 (voting) | 93.88% | 108.4 |
| ozone-level-8hr | PFossil:one-level@25 (best rule) | 93.73% | 269.3 |
| ozone-level-8hr | PFossil:one-level@25 (macro distribution voting) | 93.80% | 269.3 |
| ozone-level-8hr | PFossil:one-level@25 (micro distribution voting) | 93.65% | 269.3 |
| ozone-level-8hr | PFossil:one-level@25 (voting) | 93.88% | 269.3 |
| ozone-level-8hr | PFossil:one-level@50 (best rule) | 93.69% | 536.2 |
| ozone-level-8hr | PFossil:one-level@50 (macro distribution voting) | 93.80% | 536.2 |
| ozone-level-8hr | PFossil:one-level@50 (micro distribution voting) | 93.69% | 536.2 |
| ozone-level-8hr | PFossil:one-level@50 (voting) | 93.88% | 536.2 |
| ozone-level-8hr | PFossil:single (best rule) | 93.33% | 2.9 |
| ozone-level-8hr | PFossil:single (macro distribution voting) | 93.33% | 2.9 |
| ozone-level-8hr | PFossil:single (micro distribution voting) | 93.33% | 2.9 |
| ozone-level-8hr | PFossil:single (voting) | 93.33% | 2.9 |
| ozone-level-8hr | PFossil:two-level@10 (best rule) | 93.61% | 108.4 |
| ozone-level-8hr | PFossil:two-level@10 (macro distribution voting) | 93.69% | 108.4 |
| ozone-level-8hr | PFossil:two-level@10 (micro distribution voting) | 93.65% | 108.4 |
| ozone-level-8hr | PFossil:two-level@10 (voting) | 93.61% | 108.4 |
| ozone-level-8hr | PFossil:two-level@25 (best rule) | 93.88% | 269.3 |
| ozone-level-8hr | PFossil:two-level@25 (macro distribution voting) | 93.88% | 269.3 |
| ozone-level-8hr | PFossil:two-level@25 (micro distribution voting) | 93.84% | 269.3 |
| ozone-level-8hr | PFossil:two-level@25 (voting) | 93.80% | 269.3 |
| ozone-level-8hr | PFossil:two-level@50 (best rule) | 93.77% | 536.2 |
| ozone-level-8hr | PFossil:two-level@50 (macro distribution voting) | 93.84% | 536.2 |
| ozone-level-8hr | PFossil:two-level@50 (micro distribution voting) | 93.73% | 536.2 |
| ozone-level-8hr | PFossil:two-level@50 (voting) | 93.73% | 536.2 |
| ozone-level-8hr | Pypper:one-level@10 (best rule) | 93.73% | 103.0 |
| ozone-level-8hr | Pypper:one-level@10 (macro distribution voting) | 94.32% | 103.0 |
| ozone-level-8hr | Pypper:one-level@10 (micro distribution voting) | 93.76% | 103.0 |
| ozone-level-8hr | Pypper:one-level@10 (voting) | 94.08% | 103.0 |
| ozone-level-8hr | Pypper:one-level@25 (best rule) | 93.73% | 254.0 |
| ozone-level-8hr | Pypper:one-level@25 (macro distribution voting) | 94.00% | 254.0 |
| ozone-level-8hr | Pypper:one-level@25 (micro distribution voting) | 93.69% | 254.0 |
| ozone-level-8hr | Pypper:one-level@25 (voting) | 94.24% | 254.0 |
| ozone-level-8hr | Pypper:one-level@50 (best rule) | 93.73% | 490.0 |
| ozone-level-8hr | Pypper:one-level@50 (macro distribution voting) | 94.12% | 490.0 |
| ozone-level-8hr | Pypper:one-level@50 (micro distribution voting) | 93.73% | 490.0 |
| ozone-level-8hr | Pypper:one-level@50 (voting) | 94.16% | 490.0 |
| ozone-level-8hr | Pypper:single (best rule) | 93.37% | 7.3 |
| ozone-level-8hr | Pypper:single (macro distribution voting) | 93.37% | 7.3 |
| ozone-level-8hr | Pypper:single (micro distribution voting) | 93.37% | 7.3 |
| ozone-level-8hr | Pypper:single (voting) | 93.37% | 7.3 |
| ozone-level-8hr | Pypper:two-level@10 (best rule) | 94.24% | 103.0 |
| ozone-level-8hr | Pypper:two-level@10 (macro distribution voting) | 94.32% | 103.0 |
| ozone-level-8hr | Pypper:two-level@10 (micro distribution voting) | 94.24% | 103.0 |
| ozone-level-8hr | Pypper:two-level@10 (voting) | 94.24% | 103.0 |
| ozone-level-8hr | Pypper:two-level@25 (best rule) | 94.12% | 254.0 |
| ozone-level-8hr | Pypper:two-level@25 (macro distribution voting) | 94.12% | 254.0 |
| ozone-level-8hr | Pypper:two-level@25 (micro distribution voting) | 94.12% | 254.0 |
| ozone-level-8hr | Pypper:two-level@25 (voting) | 94.20% | 254.0 |
| ozone-level-8hr | Pypper:two-level@50 (best rule) | 94.16% | 490.0 |
| ozone-level-8hr | Pypper:two-level@50 (macro distribution voting) | 94.24% | 490.0 |
| ozone-level-8hr | Pypper:two-level@50 (micro distribution voting) | 94.16% | 490.0 |
| ozone-level-8hr | Pypper:two-level@50 (voting) | 94.20% | 490.0 |
| pc1 | PFoil:one-level@10 (best rule) | 93.15% | 417.6 |
| pc1 | PFoil:one-level@10 (macro distribution voting) | 92.61% | 417.6 |
| pc1 | PFoil:one-level@10 (micro distribution voting) | 93.06% | 417.6 |
| pc1 | PFoil:one-level@10 (voting) | 91.88% | 417.6 |
| pc1 | PFoil:one-level@25 (best rule) | 93.15% | 1039.9 |
| pc1 | PFoil:one-level@25 (macro distribution voting) | 92.97% | 1039.9 |
| pc1 | PFoil:one-level@25 (micro distribution voting) | 93.06% | 1039.9 |
| pc1 | PFoil:one-level@25 (voting) | 92.24% | 1039.9 |
| pc1 | PFoil:one-level@50 (best rule) | 93.06% | 2056.1 |
| pc1 | PFoil:one-level@50 (macro distribution voting) | 93.15% | 2056.1 |
| pc1 | PFoil:one-level@50 (micro distribution voting) | 93.06% | 2056.1 |
| pc1 | PFoil:one-level@50 (voting) | 92.70% | 2056.1 |
| pc1 | PFoil:single (best rule) | 93.06% | 48.8 |
| pc1 | PFoil:single (macro distribution voting) | 93.06% | 48.8 |
| pc1 | PFoil:single (micro distribution voting) | 92.97% | 48.8 |
| pc1 | PFoil:single (voting) | 91.52% | 48.8 |
| pc1 | PFoil:two-level@10 (best rule) | 92.97% | 417.6 |
| pc1 | PFoil:two-level@10 (macro distribution voting) | 92.79% | 417.6 |
| pc1 | PFoil:two-level@10 (micro distribution voting) | 92.79% | 417.6 |
| pc1 | PFoil:two-level@10 (voting) | 92.61% | 417.6 |
| pc1 | PFoil:two-level@25 (best rule) | 93.33% | 1039.9 |
| pc1 | PFoil:two-level@25 (macro distribution voting) | 93.15% | 1039.9 |
| pc1 | PFoil:two-level@25 (micro distribution voting) | 93.15% | 1039.9 |
| pc1 | PFoil:two-level@25 (voting) | 92.79% | 1039.9 |
| pc1 | PFoil:two-level@50 (best rule) | 93.15% | 2056.1 |
| pc1 | PFoil:two-level@50 (macro distribution voting) | 93.15% | 2056.1 |
| pc1 | PFoil:two-level@50 (micro distribution voting) | 93.24% | 2056.1 |
| pc1 | PFoil:two-level@50 (voting) | 92.97% | 2056.1 |
| pc1 | PFossil:one-level@10 (best rule) | 93.33% | 69.2 |
| pc1 | PFossil:one-level@10 (macro distribution voting) | 92.97% | 69.2 |
| pc1 | PFossil:one-level@10 (micro distribution voting) | 93.15% | 69.2 |
| pc1 | PFossil:one-level@10 (voting) | 92.97% | 69.2 |
| pc1 | PFossil:one-level@25 (best rule) | 93.24% | 182.1 |
| pc1 | PFossil:one-level@25 (macro distribution voting) | 92.88% | 182.1 |
| pc1 | PFossil:one-level@25 (micro distribution voting) | 93.24% | 182.1 |
| pc1 | PFossil:one-level@25 (voting) | 92.79% | 182.1 |
| pc1 | PFossil:one-level@50 (best rule) | 93.06% | 363.4 |
| pc1 | PFossil:one-level@50 (macro distribution voting) | 92.70% | 363.4 |
| pc1 | PFossil:one-level@50 (micro distribution voting) | 93.15% | 363.4 |
| pc1 | PFossil:one-level@50 (voting) | 92.79% | 363.4 |
| pc1 | PFossil:single (best rule) | 92.88% | 7.2 |
| pc1 | PFossil:single (macro distribution voting) | 92.79% | 7.2 |
| pc1 | PFossil:single (micro distribution voting) | 92.97% | 7.2 |
| pc1 | PFossil:single (voting) | 92.88% | 7.2 |
| pc1 | PFossil:two-level@10 (best rule) | 92.88% | 69.2 |
| pc1 | PFossil:two-level@10 (macro distribution voting) | 92.97% | 69.2 |
| pc1 | PFossil:two-level@10 (micro distribution voting) | 93.06% | 69.2 |
| pc1 | PFossil:two-level@10 (voting) | 93.06% | 69.2 |
| pc1 | PFossil:two-level@25 (best rule) | 92.79% | 182.1 |
| pc1 | PFossil:two-level@25 (macro distribution voting) | 92.61% | 182.1 |
| pc1 | PFossil:two-level@25 (micro distribution voting) | 92.88% | 182.1 |
| pc1 | PFossil:two-level@25 (voting) | 92.79% | 182.1 |
| pc1 | PFossil:two-level@50 (best rule) | 92.97% | 363.4 |
| pc1 | PFossil:two-level@50 (macro distribution voting) | 92.70% | 363.4 |
| pc1 | PFossil:two-level@50 (micro distribution voting) | 92.97% | 363.4 |
| pc1 | PFossil:two-level@50 (voting) | 92.88% | 363.4 |
| pc1 | Pypper:one-level@10 (best rule) | 93.06% | 49.4 |
| pc1 | Pypper:one-level@10 (macro distribution voting) | 92.88% | 49.4 |
| pc1 | Pypper:one-level@10 (micro distribution voting) | 93.06% | 49.4 |
| pc1 | Pypper:one-level@10 (voting) | 93.06% | 49.4 |
| pc1 | Pypper:one-level@25 (best rule) | 93.06% | 130.1 |
| pc1 | Pypper:one-level@25 (macro distribution voting) | 93.24% | 130.1 |
| pc1 | Pypper:one-level@25 (micro distribution voting) | 93.15% | 130.1 |
| pc1 | Pypper:one-level@25 (voting) | 93.51% | 130.1 |
| pc1 | Pypper:one-level@50 (best rule) | 93.06% | 265.7 |
| pc1 | Pypper:one-level@50 (macro distribution voting) | 93.33% | 265.7 |
| pc1 | Pypper:one-level@50 (micro distribution voting) | 93.06% | 265.7 |
| pc1 | Pypper:one-level@50 (voting) | 93.42% | 265.7 |
| pc1 | Pypper:single (best rule) | 93.15% | 4.3 |
| pc1 | Pypper:single (macro distribution voting) | 93.15% | 4.3 |
| pc1 | Pypper:single (micro distribution voting) | 93.15% | 4.3 |
| pc1 | Pypper:single (voting) | 93.15% | 4.3 |
| pc1 | Pypper:two-level@10 (best rule) | 93.24% | 49.4 |
| pc1 | Pypper:two-level@10 (macro distribution voting) | 93.24% | 49.4 |
| pc1 | Pypper:two-level@10 (micro distribution voting) | 93.24% | 49.4 |
| pc1 | Pypper:two-level@10 (voting) | 93.24% | 49.4 |
| pc1 | Pypper:two-level@25 (best rule) | 93.60% | 130.1 |
| pc1 | Pypper:two-level@25 (macro distribution voting) | 93.51% | 130.1 |
| pc1 | Pypper:two-level@25 (micro distribution voting) | 93.51% | 130.1 |
| pc1 | Pypper:two-level@25 (voting) | 93.42% | 130.1 |
| pc1 | Pypper:two-level@50 (best rule) | 93.42% | 265.7 |
| pc1 | Pypper:two-level@50 (macro distribution voting) | 93.33% | 265.7 |
| pc1 | Pypper:two-level@50 (micro distribution voting) | 93.42% | 265.7 |
| pc1 | Pypper:two-level@50 (voting) | 93.24% | 265.7 |
| pc3 | PFoil:one-level@10 (best rule) | 89.76% | 592.6 |
| pc3 | PFoil:one-level@10 (macro distribution voting) | 89.76% | 592.6 |
| pc3 | PFoil:one-level@10 (micro distribution voting) | 89.76% | 592.6 |
| pc3 | PFoil:one-level@10 (voting) | 89.12% | 592.6 |
| pc3 | PFoil:one-level@25 (best rule) | 89.83% | 1464.7 |
| pc3 | PFoil:one-level@25 (macro distribution voting) | 89.83% | 1464.7 |
| pc3 | PFoil:one-level@25 (micro distribution voting) | 89.76% | 1464.7 |
| pc3 | PFoil:one-level@25 (voting) | 89.57% | 1464.7 |
| pc3 | PFoil:one-level@50 (best rule) | 89.76% | 2927.1 |
| pc3 | PFoil:one-level@50 (macro distribution voting) | 89.76% | 2927.1 |
| pc3 | PFoil:one-level@50 (micro distribution voting) | 89.76% | 2927.1 |
| pc3 | PFoil:one-level@50 (voting) | 89.70% | 2927.1 |
| pc3 | PFoil:single (best rule) | 87.78% | 72.1 |
| pc3 | PFoil:single (macro distribution voting) | 87.84% | 72.1 |
| pc3 | PFoil:single (micro distribution voting) | 88.16% | 72.1 |
| pc3 | PFoil:single (voting) | 86.88% | 72.1 |
| pc3 | PFoil:two-level@10 (best rule) | 89.70% | 592.6 |
| pc3 | PFoil:two-level@10 (macro distribution voting) | 89.70% | 592.6 |
| pc3 | PFoil:two-level@10 (micro distribution voting) | 89.57% | 592.6 |
| pc3 | PFoil:two-level@10 (voting) | 89.25% | 592.6 |
| pc3 | PFoil:two-level@25 (best rule) | 90.08% | 1464.7 |
| pc3 | PFoil:two-level@25 (macro distribution voting) | 89.95% | 1464.7 |
| pc3 | PFoil:two-level@25 (micro distribution voting) | 89.76% | 1464.7 |
| pc3 | PFoil:two-level@25 (voting) | 89.57% | 1464.7 |
| pc3 | PFoil:two-level@50 (best rule) | 90.08% | 2927.1 |
| pc3 | PFoil:two-level@50 (macro distribution voting) | 90.08% | 2927.1 |
| pc3 | PFoil:two-level@50 (micro distribution voting) | 89.70% | 2927.1 |
| pc3 | PFoil:two-level@50 (voting) | 89.76% | 2927.1 |
| pc3 | PFossil:one-level@10 (best rule) | 89.89% | 66.6 |
| pc3 | PFossil:one-level@10 (macro distribution voting) | 89.51% | 66.6 |
| pc3 | PFossil:one-level@10 (micro distribution voting) | 90.02% | 66.6 |
| pc3 | PFossil:one-level@10 (voting) | 86.88% | 66.6 |
| pc3 | PFossil:one-level@25 (best rule) | 90.02% | 172.5 |
| pc3 | PFossil:one-level@25 (macro distribution voting) | 89.89% | 172.5 |
| pc3 | PFossil:one-level@25 (micro distribution voting) | 89.76% | 172.5 |
| pc3 | PFossil:one-level@25 (voting) | 87.14% | 172.5 |
| pc3 | PFossil:one-level@50 (best rule) | 89.83% | 347.7 |
| pc3 | PFossil:one-level@50 (macro distribution voting) | 89.76% | 347.7 |
| pc3 | PFossil:one-level@50 (micro distribution voting) | 89.76% | 347.7 |
| pc3 | PFossil:one-level@50 (voting) | 86.69% | 347.7 |
| pc3 | PFossil:single (best rule) | 85.09% | 2.1 |
| pc3 | PFossil:single (macro distribution voting) | 85.09% | 2.1 |
| pc3 | PFossil:single (micro distribution voting) | 85.09% | 2.1 |
| pc3 | PFossil:single (voting) | 85.09% | 2.1 |
| pc3 | PFossil:two-level@10 (best rule) | 87.39% | 66.6 |
| pc3 | PFossil:two-level@10 (macro distribution voting) | 87.46% | 66.6 |
| pc3 | PFossil:two-level@10 (micro distribution voting) | 87.33% | 66.6 |
| pc3 | PFossil:two-level@10 (voting) | 87.33% | 66.6 |
| pc3 | PFossil:two-level@25 (best rule) | 88.10% | 172.5 |
| pc3 | PFossil:two-level@25 (macro distribution voting) | 88.23% | 172.5 |
| pc3 | PFossil:two-level@25 (micro distribution voting) | 88.10% | 172.5 |
| pc3 | PFossil:two-level@25 (voting) | 87.91% | 172.5 |
| pc3 | PFossil:two-level@50 (best rule) | 88.10% | 347.7 |
| pc3 | PFossil:two-level@50 (macro distribution voting) | 88.16% | 347.7 |
| pc3 | PFossil:two-level@50 (micro distribution voting) | 88.10% | 347.7 |
| pc3 | PFossil:two-level@50 (voting) | 87.97% | 347.7 |
| pc3 | Pypper:one-level@10 (best rule) | 90.15% | 57.2 |
| pc3 | Pypper:one-level@10 (macro distribution voting) | 89.70% | 57.2 |
| pc3 | Pypper:one-level@10 (micro distribution voting) | 90.02% | 57.2 |
| pc3 | Pypper:one-level@10 (voting) | 88.74% | 57.2 |
| pc3 | Pypper:one-level@25 (best rule) | 89.89% | 141.0 |
| pc3 | Pypper:one-level@25 (macro distribution voting) | 89.83% | 141.0 |
| pc3 | Pypper:one-level@25 (micro distribution voting) | 89.89% | 141.0 |
| pc3 | Pypper:one-level@25 (voting) | 88.48% | 141.0 |
| pc3 | Pypper:one-level@50 (best rule) | 89.76% | 298.8 |
| pc3 | Pypper:one-level@50 (macro distribution voting) | 89.70% | 298.8 |
| pc3 | Pypper:one-level@50 (micro distribution voting) | 89.83% | 298.8 |
| pc3 | Pypper:one-level@50 (voting) | 89.19% | 298.8 |
| pc3 | Pypper:single (best rule) | 88.61% | 3.9 |
| pc3 | Pypper:single (macro distribution voting) | 88.61% | 3.9 |
| pc3 | Pypper:single (micro distribution voting) | 88.61% | 3.9 |
| pc3 | Pypper:single (voting) | 88.61% | 3.9 |
| pc3 | Pypper:two-level@10 (best rule) | 88.93% | 57.2 |
| pc3 | Pypper:two-level@10 (macro distribution voting) | 88.87% | 57.2 |
| pc3 | Pypper:two-level@10 (micro distribution voting) | 88.99% | 57.2 |
| pc3 | Pypper:two-level@10 (voting) | 88.80% | 57.2 |
| pc3 | Pypper:two-level@25 (best rule) | 89.12% | 141.0 |
| pc3 | Pypper:two-level@25 (macro distribution voting) | 88.99% | 141.0 |
| pc3 | Pypper:two-level@25 (micro distribution voting) | 89.12% | 141.0 |
| pc3 | Pypper:two-level@25 (voting) | 88.99% | 141.0 |
| pc3 | Pypper:two-level@50 (best rule) | 89.12% | 298.8 |
| pc3 | Pypper:two-level@50 (macro distribution voting) | 89.12% | 298.8 |
| pc3 | Pypper:two-level@50 (micro distribution voting) | 89.12% | 298.8 |
| pc3 | Pypper:two-level@50 (voting) | 89.06% | 298.8 |
| pc4 | PFoil:one-level@10 (best rule) | 89.37% | 437.3 |
| pc4 | PFoil:one-level@10 (macro distribution voting) | 90.40% | 437.3 |
| pc4 | PFoil:one-level@10 (micro distribution voting) | 88.34% | 437.3 |
| pc4 | PFoil:one-level@10 (voting) | 90.67% | 437.3 |
| pc4 | PFoil:one-level@25 (best rule) | 88.55% | 1094.7 |
| pc4 | PFoil:one-level@25 (macro distribution voting) | 90.33% | 1094.7 |
| pc4 | PFoil:one-level@25 (micro distribution voting) | 88.48% | 1094.7 |
| pc4 | PFoil:one-level@25 (voting) | 90.54% | 1094.7 |
| pc4 | PFoil:one-level@50 (best rule) | 88.27% | 2187.8 |
| pc4 | PFoil:one-level@50 (macro distribution voting) | 90.33% | 2187.8 |
| pc4 | PFoil:one-level@50 (micro distribution voting) | 88.13% | 2187.8 |
| pc4 | PFoil:one-level@50 (voting) | 90.60% | 2187.8 |
| pc4 | PFoil:single (best rule) | 89.85% | 52.3 |
| pc4 | PFoil:single (macro distribution voting) | 90.19% | 52.3 |
| pc4 | PFoil:single (micro distribution voting) | 89.23% | 52.3 |
| pc4 | PFoil:single (voting) | 88.96% | 52.3 |
| pc4 | PFoil:two-level@10 (best rule) | 89.85% | 437.3 |
| pc4 | PFoil:two-level@10 (macro distribution voting) | 90.19% | 437.3 |
| pc4 | PFoil:two-level@10 (micro distribution voting) | 89.51% | 437.3 |
| pc4 | PFoil:two-level@10 (voting) | 90.12% | 437.3 |
| pc4 | PFoil:two-level@25 (best rule) | 90.05% | 1094.7 |
| pc4 | PFoil:two-level@25 (macro distribution voting) | 90.05% | 1094.7 |
| pc4 | PFoil:two-level@25 (micro distribution voting) | 89.37% | 1094.7 |
| pc4 | PFoil:two-level@25 (voting) | 90.40% | 1094.7 |
| pc4 | PFoil:two-level@50 (best rule) | 89.98% | 2187.8 |
| pc4 | PFoil:two-level@50 (macro distribution voting) | 90.26% | 2187.8 |
| pc4 | PFoil:two-level@50 (micro distribution voting) | 89.85% | 2187.8 |
| pc4 | PFoil:two-level@50 (voting) | 90.05% | 2187.8 |
| pc4 | PFossil:one-level@10 (best rule) | 89.10% | 142.2 |
| pc4 | PFossil:one-level@10 (macro distribution voting) | 90.05% | 142.2 |
| pc4 | PFossil:one-level@10 (micro distribution voting) | 89.30% | 142.2 |
| pc4 | PFossil:one-level@10 (voting) | 90.40% | 142.2 |
| pc4 | PFossil:one-level@25 (best rule) | 88.96% | 364.9 |
| pc4 | PFossil:one-level@25 (macro distribution voting) | 89.99% | 364.9 |
| pc4 | PFossil:one-level@25 (micro distribution voting) | 89.30% | 364.9 |
| pc4 | PFossil:one-level@25 (voting) | 89.92% | 364.9 |
| pc4 | PFossil:one-level@50 (best rule) | 88.82% | 726.2 |
| pc4 | PFossil:one-level@50 (macro distribution voting) | 89.99% | 726.2 |
| pc4 | PFossil:one-level@50 (micro distribution voting) | 89.58% | 726.2 |
| pc4 | PFossil:one-level@50 (voting) | 89.71% | 726.2 |
| pc4 | PFossil:single (best rule) | 89.51% | 13.8 |
| pc4 | PFossil:single (macro distribution voting) | 89.71% | 13.8 |
| pc4 | PFossil:single (micro distribution voting) | 89.58% | 13.8 |
| pc4 | PFossil:single (voting) | 89.78% | 13.8 |
| pc4 | PFossil:two-level@10 (best rule) | 89.99% | 142.2 |
| pc4 | PFossil:two-level@10 (macro distribution voting) | 90.19% | 142.2 |
| pc4 | PFossil:two-level@10 (micro distribution voting) | 90.05% | 142.2 |
| pc4 | PFossil:two-level@10 (voting) | 90.19% | 142.2 |
| pc4 | PFossil:two-level@25 (best rule) | 89.99% | 364.9 |
| pc4 | PFossil:two-level@25 (macro distribution voting) | 90.26% | 364.9 |
| pc4 | PFossil:two-level@25 (micro distribution voting) | 89.85% | 364.9 |
| pc4 | PFossil:two-level@25 (voting) | 90.26% | 364.9 |
| pc4 | PFossil:two-level@50 (best rule) | 89.78% | 726.2 |
| pc4 | PFossil:two-level@50 (macro distribution voting) | 90.12% | 726.2 |
| pc4 | PFossil:two-level@50 (micro distribution voting) | 89.64% | 726.2 |
| pc4 | PFossil:two-level@50 (voting) | 89.92% | 726.2 |
| pc4 | Pypper:one-level@10 (best rule) | 88.55% | 84.7 |
| pc4 | Pypper:one-level@10 (macro distribution voting) | 88.68% | 84.7 |
| pc4 | Pypper:one-level@10 (micro distribution voting) | 88.48% | 84.7 |
| pc4 | Pypper:one-level@10 (voting) | 88.61% | 84.7 |
| pc4 | Pypper:one-level@25 (best rule) | 89.16% | 207.0 |
| pc4 | Pypper:one-level@25 (macro distribution voting) | 89.44% | 207.0 |
| pc4 | Pypper:one-level@25 (micro distribution voting) | 89.23% | 207.0 |
| pc4 | Pypper:one-level@25 (voting) | 89.44% | 207.0 |
| pc4 | Pypper:one-level@50 (best rule) | 88.82% | 408.8 |
| pc4 | Pypper:one-level@50 (macro distribution voting) | 88.96% | 408.8 |
| pc4 | Pypper:one-level@50 (micro distribution voting) | 89.23% | 408.8 |
| pc4 | Pypper:one-level@50 (voting) | 89.30% | 408.8 |
| pc4 | Pypper:single (best rule) | 88.55% | 8.2 |
| pc4 | Pypper:single (macro distribution voting) | 88.55% | 8.2 |
| pc4 | Pypper:single (micro distribution voting) | 88.55% | 8.2 |
| pc4 | Pypper:single (voting) | 88.55% | 8.2 |
| pc4 | Pypper:two-level@10 (best rule) | 88.07% | 84.7 |
| pc4 | Pypper:two-level@10 (macro distribution voting) | 88.07% | 84.7 |
| pc4 | Pypper:two-level@10 (micro distribution voting) | 88.13% | 84.7 |
| pc4 | Pypper:two-level@10 (voting) | 88.27% | 84.7 |
| pc4 | Pypper:two-level@25 (best rule) | 89.30% | 207.0 |
| pc4 | Pypper:two-level@25 (macro distribution voting) | 89.37% | 207.0 |
| pc4 | Pypper:two-level@25 (micro distribution voting) | 89.23% | 207.0 |
| pc4 | Pypper:two-level@25 (voting) | 89.30% | 207.0 |
| pc4 | Pypper:two-level@50 (best rule) | 89.85% | 408.8 |
| pc4 | Pypper:two-level@50 (macro distribution voting) | 89.85% | 408.8 |
| pc4 | Pypper:two-level@50 (micro distribution voting) | 89.78% | 408.8 |
| pc4 | Pypper:two-level@50 (voting) | 89.71% | 408.8 |
| phoneme | PFoil:one-level@10 (best rule) | 84.94% | 3538.4 |
| phoneme | PFoil:one-level@10 (macro distribution voting) | 85.90% | 3538.4 |
| phoneme | PFoil:one-level@10 (micro distribution voting) | 82.42% | 3538.4 |
| phoneme | PFoil:one-level@10 (voting) | 86.31% | 3538.4 |
| phoneme | PFoil:one-level@25 (best rule) | 84.84% | 8839.7 |
| phoneme | PFoil:one-level@25 (macro distribution voting) | 85.83% | 8839.7 |
| phoneme | PFoil:one-level@25 (micro distribution voting) | 82.38% | 8839.7 |
| phoneme | PFoil:one-level@25 (voting) | 86.31% | 8839.7 |
| phoneme | PFoil:one-level@50 (best rule) | 84.46% | 17689.9 |
| phoneme | PFoil:one-level@50 (macro distribution voting) | 85.86% | 17689.9 |
| phoneme | PFoil:one-level@50 (micro distribution voting) | 82.29% | 17689.9 |
| phoneme | PFoil:one-level@50 (voting) | 86.29% | 17689.9 |
| phoneme | PFoil:single (best rule) | 85.33% | 394.7 |
| phoneme | PFoil:single (macro distribution voting) | 85.44% | 394.7 |
| phoneme | PFoil:single (micro distribution voting) | 81.36% | 394.7 |
| phoneme | PFoil:single (voting) | 84.18% | 394.7 |
| phoneme | PFoil:two-level@10 (best rule) | 85.81% | 3538.4 |
| phoneme | PFoil:two-level@10 (macro distribution voting) | 86.21% | 3538.4 |
| phoneme | PFoil:two-level@10 (micro distribution voting) | 84.44% | 3538.4 |
| phoneme | PFoil:two-level@10 (voting) | 85.01% | 3538.4 |
| phoneme | PFoil:two-level@25 (best rule) | 86.05% | 8839.7 |
| phoneme | PFoil:two-level@25 (macro distribution voting) | 86.01% | 8839.7 |
| phoneme | PFoil:two-level@25 (micro distribution voting) | 84.79% | 8839.7 |
| phoneme | PFoil:two-level@25 (voting) | 85.62% | 8839.7 |
| phoneme | PFoil:two-level@50 (best rule) | 86.03% | 17689.9 |
| phoneme | PFoil:two-level@50 (macro distribution voting) | 85.90% | 17689.9 |
| phoneme | PFoil:two-level@50 (micro distribution voting) | 84.86% | 17689.9 |
| phoneme | PFoil:two-level@50 (voting) | 85.60% | 17689.9 |
| phoneme | PFossil:one-level@10 (best rule) | 78.63% | 50.8 |
| phoneme | PFossil:one-level@10 (macro distribution voting) | 78.55% | 50.8 |
| phoneme | PFossil:one-level@10 (micro distribution voting) | 78.18% | 50.8 |
| phoneme | PFossil:one-level@10 (voting) | 78.35% | 50.8 |
| phoneme | PFossil:one-level@25 (best rule) | 78.98% | 127.2 |
| phoneme | PFossil:one-level@25 (macro distribution voting) | 78.55% | 127.2 |
| phoneme | PFossil:one-level@25 (micro distribution voting) | 78.81% | 127.2 |
| phoneme | PFossil:one-level@25 (voting) | 78.11% | 127.2 |
| phoneme | PFossil:one-level@50 (best rule) | 79.01% | 254.1 |
| phoneme | PFossil:one-level@50 (macro distribution voting) | 78.48% | 254.1 |
| phoneme | PFossil:one-level@50 (micro distribution voting) | 78.65% | 254.1 |
| phoneme | PFossil:one-level@50 (voting) | 77.98% | 254.1 |
| phoneme | PFossil:single (best rule) | 78.35% | 4.4 |
| phoneme | PFossil:single (macro distribution voting) | 78.39% | 4.4 |
| phoneme | PFossil:single (micro distribution voting) | 78.35% | 4.4 |
| phoneme | PFossil:single (voting) | 78.46% | 4.4 |
| phoneme | PFossil:two-level@10 (best rule) | 77.76% | 50.8 |
| phoneme | PFossil:two-level@10 (macro distribution voting) | 77.78% | 50.8 |
| phoneme | PFossil:two-level@10 (micro distribution voting) | 77.76% | 50.8 |
| phoneme | PFossil:two-level@10 (voting) | 77.87% | 50.8 |
| phoneme | PFossil:two-level@25 (best rule) | 77.50% | 127.2 |
| phoneme | PFossil:two-level@25 (macro distribution voting) | 77.53% | 127.2 |
| phoneme | PFossil:two-level@25 (micro distribution voting) | 77.50% | 127.2 |
| phoneme | PFossil:two-level@25 (voting) | 77.63% | 127.2 |
| phoneme | PFossil:two-level@50 (best rule) | 77.46% | 254.1 |
| phoneme | PFossil:two-level@50 (macro distribution voting) | 77.46% | 254.1 |
| phoneme | PFossil:two-level@50 (micro distribution voting) | 77.46% | 254.1 |
| phoneme | PFossil:two-level@50 (voting) | 77.53% | 254.1 |
| phoneme | Pypper:one-level@10 (best rule) | 84.31% | 266.6 |
| phoneme | Pypper:one-level@10 (macro distribution voting) | 84.88% | 266.6 |
| phoneme | Pypper:one-level@10 (micro distribution voting) | 83.60% | 266.6 |
| phoneme | Pypper:one-level@10 (voting) | 84.68% | 266.6 |
| phoneme | Pypper:one-level@25 (best rule) | 83.72% | 654.5 |
| phoneme | Pypper:one-level@25 (macro distribution voting) | 84.60% | 654.5 |
| phoneme | Pypper:one-level@25 (micro distribution voting) | 83.35% | 654.5 |
| phoneme | Pypper:one-level@25 (voting) | 84.40% | 654.5 |
| phoneme | Pypper:one-level@50 (best rule) | 83.51% | 1309.5 |
| phoneme | Pypper:one-level@50 (macro distribution voting) | 84.86% | 1309.5 |
| phoneme | Pypper:one-level@50 (micro distribution voting) | 83.64% | 1309.5 |
| phoneme | Pypper:one-level@50 (voting) | 84.38% | 1309.5 |
| phoneme | Pypper:single (best rule) | 83.44% | 31.0 |
| phoneme | Pypper:single (macro distribution voting) | 83.46% | 31.0 |
| phoneme | Pypper:single (micro distribution voting) | 83.40% | 31.0 |
| phoneme | Pypper:single (voting) | 83.49% | 31.0 |
| phoneme | Pypper:two-level@10 (best rule) | 84.83% | 266.6 |
| phoneme | Pypper:two-level@10 (macro distribution voting) | 84.72% | 266.6 |
| phoneme | Pypper:two-level@10 (micro distribution voting) | 84.66% | 266.6 |
| phoneme | Pypper:two-level@10 (voting) | 84.73% | 266.6 |
| phoneme | Pypper:two-level@25 (best rule) | 84.68% | 654.5 |
| phoneme | Pypper:two-level@25 (macro distribution voting) | 84.59% | 654.5 |
| phoneme | Pypper:two-level@25 (micro distribution voting) | 84.49% | 654.5 |
| phoneme | Pypper:two-level@25 (voting) | 84.59% | 654.5 |
| phoneme | Pypper:two-level@50 (best rule) | 84.88% | 1309.5 |
| phoneme | Pypper:two-level@50 (macro distribution voting) | 84.84% | 1309.5 |
| phoneme | Pypper:two-level@50 (micro distribution voting) | 84.57% | 1309.5 |
| phoneme | Pypper:two-level@50 (voting) | 84.79% | 1309.5 |
| primary-tumor | PFoil:one-level@10 (best rule) | 43.07% | 864.2 |
| primary-tumor | PFoil:one-level@10 (macro distribution voting) | 45.73% | 864.2 |
| primary-tumor | PFoil:one-level@10 (micro distribution voting) | 44.27% | 864.2 |
| primary-tumor | PFoil:one-level@10 (voting) | 42.49% | 864.2 |
| primary-tumor | PFoil:one-level@25 (best rule) | 43.97% | 2172.1 |
| primary-tumor | PFoil:one-level@25 (macro distribution voting) | 46.88% | 2172.1 |
| primary-tumor | PFoil:one-level@25 (micro distribution voting) | 44.55% | 2172.1 |
| primary-tumor | PFoil:one-level@25 (voting) | 44.25% | 2172.1 |
| primary-tumor | PFoil:one-level@50 (best rule) | 43.96% | 4370.6 |
| primary-tumor | PFoil:one-level@50 (macro distribution voting) | 46.90% | 4370.6 |
| primary-tumor | PFoil:one-level@50 (micro distribution voting) | 43.67% | 4370.6 |
| primary-tumor | PFoil:one-level@50 (voting) | 45.13% | 4370.6 |
| primary-tumor | PFoil:single (best rule) | 43.35% | 99.1 |
| primary-tumor | PFoil:single (macro distribution voting) | 43.65% | 99.1 |
| primary-tumor | PFoil:single (micro distribution voting) | 37.75% | 99.1 |
| primary-tumor | PFoil:single (voting) | 36.28% | 99.1 |
| primary-tumor | PFoil:two-level@10 (best rule) | 43.06% | 864.2 |
| primary-tumor | PFoil:two-level@10 (macro distribution voting) | 43.65% | 864.2 |
| primary-tumor | PFoil:two-level@10 (micro distribution voting) | 43.07% | 864.2 |
| primary-tumor | PFoil:two-level@10 (voting) | 40.73% | 864.2 |
| primary-tumor | PFoil:two-level@25 (best rule) | 46.01% | 2172.1 |
| primary-tumor | PFoil:two-level@25 (macro distribution voting) | 44.82% | 2172.1 |
| primary-tumor | PFoil:two-level@25 (micro distribution voting) | 44.82% | 2172.1 |
| primary-tumor | PFoil:two-level@25 (voting) | 42.50% | 2172.1 |
| primary-tumor | PFoil:two-level@50 (best rule) | 46.60% | 4370.6 |
| primary-tumor | PFoil:two-level@50 (macro distribution voting) | 44.83% | 4370.6 |
| primary-tumor | PFoil:two-level@50 (micro distribution voting) | 45.71% | 4370.6 |
| primary-tumor | PFoil:two-level@50 (voting) | 42.80% | 4370.6 |
| primary-tumor | PFossil:one-level@10 (best rule) | 42.49% | 676.3 |
| primary-tumor | PFossil:one-level@10 (macro distribution voting) | 43.07% | 676.3 |
| primary-tumor | PFossil:one-level@10 (micro distribution voting) | 45.44% | 676.3 |
| primary-tumor | PFossil:one-level@10 (voting) | 43.68% | 676.3 |
| primary-tumor | PFossil:one-level@25 (best rule) | 43.65% | 1679.6 |
| primary-tumor | PFossil:one-level@25 (macro distribution voting) | 44.25% | 1679.6 |
| primary-tumor | PFossil:one-level@25 (micro distribution voting) | 44.24% | 1679.6 |
| primary-tumor | PFossil:one-level@25 (voting) | 43.36% | 1679.6 |
| primary-tumor | PFossil:one-level@50 (best rule) | 42.17% | 3367.6 |
| primary-tumor | PFossil:one-level@50 (macro distribution voting) | 44.84% | 3367.6 |
| primary-tumor | PFossil:one-level@50 (micro distribution voting) | 46.02% | 3367.6 |
| primary-tumor | PFossil:one-level@50 (voting) | 43.96% | 3367.6 |
| primary-tumor | PFossil:single (best rule) | 43.38% | 66.3 |
| primary-tumor | PFossil:single (macro distribution voting) | 41.61% | 66.3 |
| primary-tumor | PFossil:single (micro distribution voting) | 41.01% | 66.3 |
| primary-tumor | PFossil:single (voting) | 41.01% | 66.3 |
| primary-tumor | PFossil:two-level@10 (best rule) | 41.60% | 676.3 |
| primary-tumor | PFossil:two-level@10 (macro distribution voting) | 44.56% | 676.3 |
| primary-tumor | PFossil:two-level@10 (micro distribution voting) | 42.19% | 676.3 |
| primary-tumor | PFossil:two-level@10 (voting) | 43.08% | 676.3 |
| primary-tumor | PFossil:two-level@25 (best rule) | 43.38% | 1679.6 |
| primary-tumor | PFossil:two-level@25 (macro distribution voting) | 43.96% | 1679.6 |
| primary-tumor | PFossil:two-level@25 (micro distribution voting) | 42.47% | 1679.6 |
| primary-tumor | PFossil:two-level@25 (voting) | 41.89% | 1679.6 |
| primary-tumor | PFossil:two-level@50 (best rule) | 44.55% | 3367.6 |
| primary-tumor | PFossil:two-level@50 (macro distribution voting) | 44.55% | 3367.6 |
| primary-tumor | PFossil:two-level@50 (micro distribution voting) | 43.95% | 3367.6 |
| primary-tumor | PFossil:two-level@50 (voting) | 44.53% | 3367.6 |
| primary-tumor | Pypper:one-level@10 (best rule) | 43.66% | 149.5 |
| primary-tumor | Pypper:one-level@10 (macro distribution voting) | 44.25% | 149.5 |
| primary-tumor | Pypper:one-level@10 (micro distribution voting) | 43.36% | 149.5 |
| primary-tumor | Pypper:one-level@10 (voting) | 45.14% | 149.5 |
| primary-tumor | Pypper:one-level@25 (best rule) | 44.54% | 369.8 |
| primary-tumor | Pypper:one-level@25 (macro distribution voting) | 45.12% | 369.8 |
| primary-tumor | Pypper:one-level@25 (micro distribution voting) | 44.24% | 369.8 |
| primary-tumor | Pypper:one-level@25 (voting) | 45.42% | 369.8 |
| primary-tumor | Pypper:one-level@50 (best rule) | 42.77% | 749.0 |
| primary-tumor | Pypper:one-level@50 (macro distribution voting) | 45.42% | 749.0 |
| primary-tumor | Pypper:one-level@50 (micro distribution voting) | 46.31% | 749.0 |
| primary-tumor | Pypper:one-level@50 (voting) | 46.60% | 749.0 |
| primary-tumor | Pypper:single (best rule) | 37.47% | 9.1 |
| primary-tumor | Pypper:single (macro distribution voting) | 36.88% | 9.1 |
| primary-tumor | Pypper:single (micro distribution voting) | 38.07% | 9.1 |
| primary-tumor | Pypper:single (voting) | 36.59% | 9.1 |
| primary-tumor | Pypper:two-level@10 (best rule) | 44.25% | 149.5 |
| primary-tumor | Pypper:two-level@10 (macro distribution voting) | 44.25% | 149.5 |
| primary-tumor | Pypper:two-level@10 (micro distribution voting) | 44.84% | 149.5 |
| primary-tumor | Pypper:two-level@10 (voting) | 43.96% | 149.5 |
| primary-tumor | Pypper:two-level@25 (best rule) | 45.12% | 369.8 |
| primary-tumor | Pypper:two-level@25 (macro distribution voting) | 45.12% | 369.8 |
| primary-tumor | Pypper:two-level@25 (micro distribution voting) | 44.23% | 369.8 |
| primary-tumor | Pypper:two-level@25 (voting) | 44.53% | 369.8 |
| primary-tumor | Pypper:two-level@50 (best rule) | 46.90% | 749.0 |
| primary-tumor | Pypper:two-level@50 (macro distribution voting) | 46.31% | 749.0 |
| primary-tumor | Pypper:two-level@50 (micro distribution voting) | 46.02% | 749.0 |
| primary-tumor | Pypper:two-level@50 (voting) | 44.54% | 749.0 |
| qsar-biodeg | PFoil:one-level@10 (best rule) | 84.17% | 515.4 |
| qsar-biodeg | PFoil:one-level@10 (macro distribution voting) | 86.45% | 515.4 |
| qsar-biodeg | PFoil:one-level@10 (micro distribution voting) | 83.32% | 515.4 |
| qsar-biodeg | PFoil:one-level@10 (voting) | 86.36% | 515.4 |
| qsar-biodeg | PFoil:one-level@25 (best rule) | 84.74% | 1284.6 |
| qsar-biodeg | PFoil:one-level@25 (macro distribution voting) | 86.45% | 1284.6 |
| qsar-biodeg | PFoil:one-level@25 (micro distribution voting) | 84.36% | 1284.6 |
| qsar-biodeg | PFoil:one-level@25 (voting) | 85.89% | 1284.6 |
| qsar-biodeg | PFoil:one-level@50 (best rule) | 84.36% | 2573.5 |
| qsar-biodeg | PFoil:one-level@50 (macro distribution voting) | 86.73% | 2573.5 |
| qsar-biodeg | PFoil:one-level@50 (micro distribution voting) | 84.07% | 2573.5 |
| qsar-biodeg | PFoil:one-level@50 (voting) | 86.07% | 2573.5 |
| qsar-biodeg | PFoil:single (best rule) | 84.55% | 62.7 |
| qsar-biodeg | PFoil:single (macro distribution voting) | 84.17% | 62.7 |
| qsar-biodeg | PFoil:single (micro distribution voting) | 81.23% | 62.7 |
| qsar-biodeg | PFoil:single (voting) | 83.89% | 62.7 |
| qsar-biodeg | PFoil:two-level@10 (best rule) | 86.35% | 515.4 |
| qsar-biodeg | PFoil:two-level@10 (macro distribution voting) | 86.26% | 515.4 |
| qsar-biodeg | PFoil:two-level@10 (micro distribution voting) | 84.74% | 515.4 |
| qsar-biodeg | PFoil:two-level@10 (voting) | 85.59% | 515.4 |
| qsar-biodeg | PFoil:two-level@25 (best rule) | 86.54% | 1284.6 |
| qsar-biodeg | PFoil:two-level@25 (macro distribution voting) | 86.45% | 1284.6 |
| qsar-biodeg | PFoil:two-level@25 (micro distribution voting) | 85.02% | 1284.6 |
| qsar-biodeg | PFoil:two-level@25 (voting) | 86.54% | 1284.6 |
| qsar-biodeg | PFoil:two-level@50 (best rule) | 86.73% | 2573.5 |
| qsar-biodeg | PFoil:two-level@50 (macro distribution voting) | 86.83% | 2573.5 |
| qsar-biodeg | PFoil:two-level@50 (micro distribution voting) | 85.50% | 2573.5 |
| qsar-biodeg | PFoil:two-level@50 (voting) | 86.73% | 2573.5 |
| qsar-biodeg | PFossil:one-level@10 (best rule) | 82.94% | 135.5 |
| qsar-biodeg | PFossil:one-level@10 (macro distribution voting) | 85.41% | 135.5 |
| qsar-biodeg | PFossil:one-level@10 (micro distribution voting) | 83.89% | 135.5 |
| qsar-biodeg | PFossil:one-level@10 (voting) | 85.31% | 135.5 |
| qsar-biodeg | PFossil:one-level@25 (best rule) | 83.12% | 342.4 |
| qsar-biodeg | PFossil:one-level@25 (macro distribution voting) | 85.88% | 342.4 |
| qsar-biodeg | PFossil:one-level@25 (micro distribution voting) | 84.08% | 342.4 |
| qsar-biodeg | PFossil:one-level@25 (voting) | 85.98% | 342.4 |
| qsar-biodeg | PFossil:one-level@50 (best rule) | 83.31% | 674.0 |
| qsar-biodeg | PFossil:one-level@50 (macro distribution voting) | 86.17% | 674.0 |
| qsar-biodeg | PFossil:one-level@50 (micro distribution voting) | 83.60% | 674.0 |
| qsar-biodeg | PFossil:one-level@50 (voting) | 85.79% | 674.0 |
| qsar-biodeg | PFossil:single (best rule) | 84.65% | 12.7 |
| qsar-biodeg | PFossil:single (macro distribution voting) | 84.74% | 12.7 |
| qsar-biodeg | PFossil:single (micro distribution voting) | 84.17% | 12.7 |
| qsar-biodeg | PFossil:single (voting) | 83.60% | 12.7 |
| qsar-biodeg | PFossil:two-level@10 (best rule) | 85.69% | 135.5 |
| qsar-biodeg | PFossil:two-level@10 (macro distribution voting) | 85.60% | 135.5 |
| qsar-biodeg | PFossil:two-level@10 (micro distribution voting) | 84.18% | 135.5 |
| qsar-biodeg | PFossil:two-level@10 (voting) | 84.45% | 135.5 |
| qsar-biodeg | PFossil:two-level@25 (best rule) | 85.98% | 342.4 |
| qsar-biodeg | PFossil:two-level@25 (macro distribution voting) | 85.79% | 342.4 |
| qsar-biodeg | PFossil:two-level@25 (micro distribution voting) | 84.55% | 342.4 |
| qsar-biodeg | PFossil:two-level@25 (voting) | 84.55% | 342.4 |
| qsar-biodeg | PFossil:two-level@50 (best rule) | 85.69% | 674.0 |
| qsar-biodeg | PFossil:two-level@50 (macro distribution voting) | 85.98% | 674.0 |
| qsar-biodeg | PFossil:two-level@50 (micro distribution voting) | 84.56% | 674.0 |
| qsar-biodeg | PFossil:two-level@50 (voting) | 84.93% | 674.0 |
| qsar-biodeg | Pypper:one-level@10 (best rule) | 83.98% | 136.8 |
| qsar-biodeg | Pypper:one-level@10 (macro distribution voting) | 84.93% | 136.8 |
| qsar-biodeg | Pypper:one-level@10 (micro distribution voting) | 83.59% | 136.8 |
| qsar-biodeg | Pypper:one-level@10 (voting) | 84.84% | 136.8 |
| qsar-biodeg | Pypper:one-level@25 (best rule) | 84.55% | 345.8 |
| qsar-biodeg | Pypper:one-level@25 (macro distribution voting) | 84.74% | 345.8 |
| qsar-biodeg | Pypper:one-level@25 (micro distribution voting) | 83.31% | 345.8 |
| qsar-biodeg | Pypper:one-level@25 (voting) | 84.65% | 345.8 |
| qsar-biodeg | Pypper:one-level@50 (best rule) | 85.03% | 699.5 |
| qsar-biodeg | Pypper:one-level@50 (macro distribution voting) | 84.65% | 699.5 |
| qsar-biodeg | Pypper:one-level@50 (micro distribution voting) | 83.79% | 699.5 |
| qsar-biodeg | Pypper:one-level@50 (voting) | 84.46% | 699.5 |
| qsar-biodeg | Pypper:single (best rule) | 83.70% | 13.3 |
| qsar-biodeg | Pypper:single (macro distribution voting) | 83.32% | 13.3 |
| qsar-biodeg | Pypper:single (micro distribution voting) | 83.12% | 13.3 |
| qsar-biodeg | Pypper:single (voting) | 83.03% | 13.3 |
| qsar-biodeg | Pypper:two-level@10 (best rule) | 84.83% | 136.8 |
| qsar-biodeg | Pypper:two-level@10 (macro distribution voting) | 84.74% | 136.8 |
| qsar-biodeg | Pypper:two-level@10 (micro distribution voting) | 84.36% | 136.8 |
| qsar-biodeg | Pypper:two-level@10 (voting) | 84.55% | 136.8 |
| qsar-biodeg | Pypper:two-level@25 (best rule) | 84.55% | 345.8 |
| qsar-biodeg | Pypper:two-level@25 (macro distribution voting) | 85.12% | 345.8 |
| qsar-biodeg | Pypper:two-level@25 (micro distribution voting) | 84.17% | 345.8 |
| qsar-biodeg | Pypper:two-level@25 (voting) | 84.74% | 345.8 |
| qsar-biodeg | Pypper:two-level@50 (best rule) | 84.93% | 699.5 |
| qsar-biodeg | Pypper:two-level@50 (macro distribution voting) | 84.84% | 699.5 |
| qsar-biodeg | Pypper:two-level@50 (micro distribution voting) | 85.21% | 699.5 |
| qsar-biodeg | Pypper:two-level@50 (voting) | 85.12% | 699.5 |
| segment | PFoil:one-level@10 (best rule) | 94.76% | 609.0 |
| segment | PFoil:one-level@10 (macro distribution voting) | 96.36% | 609.0 |
| segment | PFoil:one-level@10 (micro distribution voting) | 95.24% | 609.0 |
| segment | PFoil:one-level@10 (voting) | 96.41% | 609.0 |
| segment | PFoil:one-level@25 (best rule) | 94.72% | 1530.7 |
| segment | PFoil:one-level@25 (macro distribution voting) | 96.49% | 1530.7 |
| segment | PFoil:one-level@25 (micro distribution voting) | 95.50% | 1530.7 |
| segment | PFoil:one-level@25 (voting) | 96.75% | 1530.7 |
| segment | PFoil:one-level@50 (best rule) | 94.11% | 3060.6 |
| segment | PFoil:one-level@50 (macro distribution voting) | 96.75% | 3060.6 |
| segment | PFoil:one-level@50 (micro distribution voting) | 95.58% | 3060.6 |
| segment | PFoil:one-level@50 (voting) | 96.84% | 3060.6 |
| segment | PFoil:single (best rule) | 95.19% | 68.8 |
| segment | PFoil:single (macro distribution voting) | 95.19% | 68.8 |
| segment | PFoil:single (micro distribution voting) | 93.03% | 68.8 |
| segment | PFoil:single (voting) | 90.91% | 68.8 |
| segment | PFoil:two-level@10 (best rule) | 96.23% | 609.0 |
| segment | PFoil:two-level@10 (macro distribution voting) | 96.45% | 609.0 |
| segment | PFoil:two-level@10 (micro distribution voting) | 96.36% | 609.0 |
| segment | PFoil:two-level@10 (voting) | 96.45% | 609.0 |
| segment | PFoil:two-level@25 (best rule) | 96.71% | 1530.7 |
| segment | PFoil:two-level@25 (macro distribution voting) | 96.93% | 1530.7 |
| segment | PFoil:two-level@25 (micro distribution voting) | 96.58% | 1530.7 |
| segment | PFoil:two-level@25 (voting) | 96.67% | 1530.7 |
| segment | PFoil:two-level@50 (best rule) | 96.71% | 3060.6 |
| segment | PFoil:two-level@50 (macro distribution voting) | 96.93% | 3060.6 |
| segment | PFoil:two-level@50 (micro distribution voting) | 96.75% | 3060.6 |
| segment | PFoil:two-level@50 (voting) | 96.84% | 3060.6 |
| segment | PFossil:one-level@10 (best rule) | 95.06% | 319.3 |
| segment | PFossil:one-level@10 (macro distribution voting) | 95.71% | 319.3 |
| segment | PFossil:one-level@10 (micro distribution voting) | 94.81% | 319.3 |
| segment | PFossil:one-level@10 (voting) | 95.58% | 319.3 |
| segment | PFossil:one-level@25 (best rule) | 94.89% | 790.0 |
| segment | PFossil:one-level@25 (macro distribution voting) | 95.76% | 790.0 |
| segment | PFossil:one-level@25 (micro distribution voting) | 94.94% | 790.0 |
| segment | PFossil:one-level@25 (voting) | 95.63% | 790.0 |
| segment | PFossil:one-level@50 (best rule) | 95.15% | 1586.1 |
| segment | PFossil:one-level@50 (macro distribution voting) | 95.63% | 1586.1 |
| segment | PFossil:one-level@50 (micro distribution voting) | 94.94% | 1586.1 |
| segment | PFossil:one-level@50 (voting) | 95.58% | 1586.1 |
| segment | PFossil:single (best rule) | 93.33% | 30.5 |
| segment | PFossil:single (macro distribution voting) | 93.16% | 30.5 |
| segment | PFossil:single (micro distribution voting) | 93.12% | 30.5 |
| segment | PFossil:single (voting) | 92.73% | 30.5 |
| segment | PFossil:two-level@10 (best rule) | 95.32% | 319.3 |
| segment | PFossil:two-level@10 (macro distribution voting) | 95.45% | 319.3 |
| segment | PFossil:two-level@10 (micro distribution voting) | 95.19% | 319.3 |
| segment | PFossil:two-level@10 (voting) | 94.94% | 319.3 |
| segment | PFossil:two-level@25 (best rule) | 95.41% | 790.0 |
| segment | PFossil:two-level@25 (macro distribution voting) | 95.71% | 790.0 |
| segment | PFossil:two-level@25 (micro distribution voting) | 95.28% | 790.0 |
| segment | PFossil:two-level@25 (voting) | 95.37% | 790.0 |
| segment | PFossil:two-level@50 (best rule) | 95.41% | 1586.1 |
| segment | PFossil:two-level@50 (macro distribution voting) | 95.89% | 1586.1 |
| segment | PFossil:two-level@50 (micro distribution voting) | 95.24% | 1586.1 |
| segment | PFossil:two-level@50 (voting) | 95.89% | 1586.1 |
| segment | Pypper:one-level@10 (best rule) | 94.63% | 250.3 |
| segment | Pypper:one-level@10 (macro distribution voting) | 95.54% | 250.3 |
| segment | Pypper:one-level@10 (micro distribution voting) | 94.72% | 250.3 |
| segment | Pypper:one-level@10 (voting) | 95.54% | 250.3 |
| segment | Pypper:one-level@25 (best rule) | 94.16% | 625.6 |
| segment | Pypper:one-level@25 (macro distribution voting) | 95.97% | 625.6 |
| segment | Pypper:one-level@25 (micro distribution voting) | 94.76% | 625.6 |
| segment | Pypper:one-level@25 (voting) | 95.89% | 625.6 |
| segment | Pypper:one-level@50 (best rule) | 93.81% | 1245.5 |
| segment | Pypper:one-level@50 (macro distribution voting) | 95.76% | 1245.5 |
| segment | Pypper:one-level@50 (micro distribution voting) | 95.02% | 1245.5 |
| segment | Pypper:one-level@50 (voting) | 95.80% | 1245.5 |
| segment | Pypper:single (best rule) | 94.03% | 24.8 |
| segment | Pypper:single (macro distribution voting) | 93.90% | 24.8 |
| segment | Pypper:single (micro distribution voting) | 93.72% | 24.8 |
| segment | Pypper:single (voting) | 93.68% | 24.8 |
| segment | Pypper:two-level@10 (best rule) | 95.24% | 250.3 |
| segment | Pypper:two-level@10 (macro distribution voting) | 95.45% | 250.3 |
| segment | Pypper:two-level@10 (micro distribution voting) | 95.15% | 250.3 |
| segment | Pypper:two-level@10 (voting) | 95.32% | 250.3 |
| segment | Pypper:two-level@25 (best rule) | 95.63% | 625.6 |
| segment | Pypper:two-level@25 (macro distribution voting) | 95.84% | 625.6 |
| segment | Pypper:two-level@25 (micro distribution voting) | 95.58% | 625.6 |
| segment | Pypper:two-level@25 (voting) | 95.97% | 625.6 |
| segment | Pypper:two-level@50 (best rule) | 95.67% | 1245.5 |
| segment | Pypper:two-level@50 (macro distribution voting) | 95.63% | 1245.5 |
| segment | Pypper:two-level@50 (micro distribution voting) | 95.63% | 1245.5 |
| segment | Pypper:two-level@50 (voting) | 95.76% | 1245.5 |
| sick | PFoil:one-level@10 (best rule) | 97.56% | 502.3 |
| sick | PFoil:one-level@10 (macro distribution voting) | 98.33% | 502.3 |
| sick | PFoil:one-level@10 (micro distribution voting) | 96.37% | 502.3 |
| sick | PFoil:one-level@10 (voting) | 98.25% | 502.3 |
| sick | PFoil:one-level@25 (best rule) | 97.38% | 1267.4 |
| sick | PFoil:one-level@25 (macro distribution voting) | 98.30% | 1267.4 |
| sick | PFoil:one-level@25 (micro distribution voting) | 96.58% | 1267.4 |
| sick | PFoil:one-level@25 (voting) | 98.36% | 1267.4 |
| sick | PFoil:one-level@50 (best rule) | 97.16% | 2521.9 |
| sick | PFoil:one-level@50 (macro distribution voting) | 98.38% | 2521.9 |
| sick | PFoil:one-level@50 (micro distribution voting) | 96.42% | 2521.9 |
| sick | PFoil:one-level@50 (voting) | 98.36% | 2521.9 |
| sick | PFoil:single (best rule) | 97.99% | 63.5 |
| sick | PFoil:single (macro distribution voting) | 98.14% | 63.5 |
| sick | PFoil:single (micro distribution voting) | 96.95% | 63.5 |
| sick | PFoil:single (voting) | 97.51% | 63.5 |
| sick | PFoil:two-level@10 (best rule) | 98.12% | 502.3 |
| sick | PFoil:two-level@10 (macro distribution voting) | 98.36% | 502.3 |
| sick | PFoil:two-level@10 (micro distribution voting) | 97.69% | 502.3 |
| sick | PFoil:two-level@10 (voting) | 98.14% | 502.3 |
| sick | PFoil:two-level@25 (best rule) | 98.25% | 1267.4 |
| sick | PFoil:two-level@25 (macro distribution voting) | 98.36% | 1267.4 |
| sick | PFoil:two-level@25 (micro distribution voting) | 97.93% | 1267.4 |
| sick | PFoil:two-level@25 (voting) | 98.14% | 1267.4 |
| sick | PFoil:two-level@50 (best rule) | 98.28% | 2521.9 |
| sick | PFoil:two-level@50 (macro distribution voting) | 98.36% | 2521.9 |
| sick | PFoil:two-level@50 (micro distribution voting) | 97.99% | 2521.9 |
| sick | PFoil:two-level@50 (voting) | 98.20% | 2521.9 |
| sick | PFossil:one-level@10 (best rule) | 97.48% | 166.5 |
| sick | PFossil:one-level@10 (macro distribution voting) | 98.14% | 166.5 |
| sick | PFossil:one-level@10 (micro distribution voting) | 97.83% | 166.5 |
| sick | PFossil:one-level@10 (voting) | 98.25% | 166.5 |
| sick | PFossil:one-level@25 (best rule) | 97.03% | 403.3 |
| sick | PFossil:one-level@25 (macro distribution voting) | 98.30% | 403.3 |
| sick | PFossil:one-level@25 (micro distribution voting) | 97.93% | 403.3 |
| sick | PFossil:one-level@25 (voting) | 98.25% | 403.3 |
| sick | PFossil:one-level@50 (best rule) | 96.85% | 806.0 |
| sick | PFossil:one-level@50 (macro distribution voting) | 98.28% | 806.0 |
| sick | PFossil:one-level@50 (micro distribution voting) | 98.01% | 806.0 |
| sick | PFossil:one-level@50 (voting) | 98.36% | 806.0 |
| sick | PFossil:single (best rule) | 98.30% | 16.4 |
| sick | PFossil:single (macro distribution voting) | 98.25% | 16.4 |
| sick | PFossil:single (micro distribution voting) | 98.20% | 16.4 |
| sick | PFossil:single (voting) | 98.20% | 16.4 |
| sick | PFossil:two-level@10 (best rule) | 98.28% | 166.5 |
| sick | PFossil:two-level@10 (macro distribution voting) | 98.22% | 166.5 |
| sick | PFossil:two-level@10 (micro distribution voting) | 98.20% | 166.5 |
| sick | PFossil:two-level@10 (voting) | 98.20% | 166.5 |
| sick | PFossil:two-level@25 (best rule) | 98.25% | 403.3 |
| sick | PFossil:two-level@25 (macro distribution voting) | 98.28% | 403.3 |
| sick | PFossil:two-level@25 (micro distribution voting) | 98.17% | 403.3 |
| sick | PFossil:two-level@25 (voting) | 98.25% | 403.3 |
| sick | PFossil:two-level@50 (best rule) | 98.30% | 806.0 |
| sick | PFossil:two-level@50 (macro distribution voting) | 98.38% | 806.0 |
| sick | PFossil:two-level@50 (micro distribution voting) | 98.25% | 806.0 |
| sick | PFossil:two-level@50 (voting) | 98.36% | 806.0 |
| sick | Pypper:one-level@10 (best rule) | 98.06% | 103.0 |
| sick | Pypper:one-level@10 (macro distribution voting) | 98.20% | 103.0 |
| sick | Pypper:one-level@10 (micro distribution voting) | 98.07% | 103.0 |
| sick | Pypper:one-level@10 (voting) | 98.14% | 103.0 |
| sick | Pypper:one-level@25 (best rule) | 97.69% | 254.5 |
| sick | Pypper:one-level@25 (macro distribution voting) | 98.33% | 254.5 |
| sick | Pypper:one-level@25 (micro distribution voting) | 97.99% | 254.5 |
| sick | Pypper:one-level@25 (voting) | 98.25% | 254.5 |
| sick | Pypper:one-level@50 (best rule) | 97.67% | 512.7 |
| sick | Pypper:one-level@50 (macro distribution voting) | 98.33% | 512.7 |
| sick | Pypper:one-level@50 (micro distribution voting) | 98.06% | 512.7 |
| sick | Pypper:one-level@50 (voting) | 98.36% | 512.7 |
| sick | Pypper:single (best rule) | 98.30% | 10.5 |
| sick | Pypper:single (macro distribution voting) | 98.30% | 10.5 |
| sick | Pypper:single (micro distribution voting) | 98.30% | 10.5 |
| sick | Pypper:single (voting) | 98.33% | 10.5 |
| sick | Pypper:two-level@10 (best rule) | 98.28% | 103.0 |
| sick | Pypper:two-level@10 (macro distribution voting) | 98.28% | 103.0 |
| sick | Pypper:two-level@10 (micro distribution voting) | 98.17% | 103.0 |
| sick | Pypper:two-level@10 (voting) | 98.30% | 103.0 |
| sick | Pypper:two-level@25 (best rule) | 98.25% | 254.5 |
| sick | Pypper:two-level@25 (macro distribution voting) | 98.25% | 254.5 |
| sick | Pypper:two-level@25 (micro distribution voting) | 98.20% | 254.5 |
| sick | Pypper:two-level@25 (voting) | 98.28% | 254.5 |
| sick | Pypper:two-level@50 (best rule) | 98.33% | 512.7 |
| sick | Pypper:two-level@50 (macro distribution voting) | 98.28% | 512.7 |
| sick | Pypper:two-level@50 (micro distribution voting) | 98.28% | 512.7 |
| sick | Pypper:two-level@50 (voting) | 98.28% | 512.7 |
| solar-flare | PFoil:one-level@10 (best rule) | 73.36% | 875.2 |
| solar-flare | PFoil:one-level@10 (macro distribution voting) | 74.21% | 875.2 |
| solar-flare | PFoil:one-level@10 (micro distribution voting) | 73.92% | 875.2 |
| solar-flare | PFoil:one-level@10 (voting) | 73.17% | 875.2 |
| solar-flare | PFoil:one-level@25 (best rule) | 73.74% | 2166.3 |
| solar-flare | PFoil:one-level@25 (macro distribution voting) | 74.49% | 2166.3 |
| solar-flare | PFoil:one-level@25 (micro distribution voting) | 74.21% | 2166.3 |
| solar-flare | PFoil:one-level@25 (voting) | 73.45% | 2166.3 |
| solar-flare | PFoil:one-level@50 (best rule) | 73.65% | 4341.4 |
| solar-flare | PFoil:one-level@50 (macro distribution voting) | 74.86% | 4341.4 |
| solar-flare | PFoil:one-level@50 (micro distribution voting) | 74.11% | 4341.4 |
| solar-flare | PFoil:one-level@50 (voting) | 72.33% | 4341.4 |
| solar-flare | PFoil:single (best rule) | 73.73% | 85.2 |
| solar-flare | PFoil:single (macro distribution voting) | 74.39% | 85.2 |
| solar-flare | PFoil:single (micro distribution voting) | 74.02% | 85.2 |
| solar-flare | PFoil:single (voting) | 71.30% | 85.2 |
| solar-flare | PFoil:two-level@10 (best rule) | 73.36% | 875.2 |
| solar-flare | PFoil:two-level@10 (macro distribution voting) | 74.30% | 875.2 |
| solar-flare | PFoil:two-level@10 (micro distribution voting) | 75.24% | 875.2 |
| solar-flare | PFoil:two-level@10 (voting) | 71.67% | 875.2 |
| solar-flare | PFoil:two-level@25 (best rule) | 73.73% | 2166.3 |
| solar-flare | PFoil:two-level@25 (macro distribution voting) | 74.49% | 2166.3 |
| solar-flare | PFoil:two-level@25 (micro distribution voting) | 74.96% | 2166.3 |
| solar-flare | PFoil:two-level@25 (voting) | 71.11% | 2166.3 |
| solar-flare | PFoil:two-level@50 (best rule) | 73.55% | 4341.4 |
| solar-flare | PFoil:two-level@50 (macro distribution voting) | 74.48% | 4341.4 |
| solar-flare | PFoil:two-level@50 (micro distribution voting) | 75.24% | 4341.4 |
| solar-flare | PFoil:two-level@50 (voting) | 71.58% | 4341.4 |
| solar-flare | PFossil:one-level@10 (best rule) | 72.42% | 183.8 |
| solar-flare | PFossil:one-level@10 (macro distribution voting) | 74.21% | 183.8 |
| solar-flare | PFossil:one-level@10 (micro distribution voting) | 74.96% | 183.8 |
| solar-flare | PFossil:one-level@10 (voting) | 73.64% | 183.8 |
| solar-flare | PFossil:one-level@25 (best rule) | 71.67% | 457.0 |
| solar-flare | PFossil:one-level@25 (macro distribution voting) | 74.96% | 457.0 |
| solar-flare | PFossil:one-level@25 (micro distribution voting) | 74.96% | 457.0 |
| solar-flare | PFossil:one-level@25 (voting) | 74.01% | 457.0 |
| solar-flare | PFossil:one-level@50 (best rule) | 72.04% | 911.9 |
| solar-flare | PFossil:one-level@50 (macro distribution voting) | 75.14% | 911.9 |
| solar-flare | PFossil:one-level@50 (micro distribution voting) | 74.96% | 911.9 |
| solar-flare | PFossil:one-level@50 (voting) | 74.20% | 911.9 |
| solar-flare | PFossil:single (best rule) | 74.02% | 14.5 |
| solar-flare | PFossil:single (macro distribution voting) | 74.77% | 14.5 |
| solar-flare | PFossil:single (micro distribution voting) | 74.30% | 14.5 |
| solar-flare | PFossil:single (voting) | 70.07% | 14.5 |
| solar-flare | PFossil:two-level@10 (best rule) | 73.64% | 183.8 |
| solar-flare | PFossil:two-level@10 (macro distribution voting) | 73.83% | 183.8 |
| solar-flare | PFossil:two-level@10 (micro distribution voting) | 74.77% | 183.8 |
| solar-flare | PFossil:two-level@10 (voting) | 69.14% | 183.8 |
| solar-flare | PFossil:two-level@25 (best rule) | 73.83% | 457.0 |
| solar-flare | PFossil:two-level@25 (macro distribution voting) | 74.58% | 457.0 |
| solar-flare | PFossil:two-level@25 (micro distribution voting) | 74.86% | 457.0 |
| solar-flare | PFossil:two-level@25 (voting) | 69.80% | 457.0 |
| solar-flare | PFossil:two-level@50 (best rule) | 73.83% | 911.9 |
| solar-flare | PFossil:two-level@50 (macro distribution voting) | 74.20% | 911.9 |
| solar-flare | PFossil:two-level@50 (micro distribution voting) | 75.14% | 911.9 |
| solar-flare | PFossil:two-level@50 (voting) | 69.14% | 911.9 |
| solar-flare | Pypper:one-level@10 (best rule) | 73.36% | 107.7 |
| solar-flare | Pypper:one-level@10 (macro distribution voting) | 74.78% | 107.7 |
| solar-flare | Pypper:one-level@10 (micro distribution voting) | 74.87% | 107.7 |
| solar-flare | Pypper:one-level@10 (voting) | 74.21% | 107.7 |
| solar-flare | Pypper:one-level@25 (best rule) | 73.93% | 257.4 |
| solar-flare | Pypper:one-level@25 (macro distribution voting) | 75.05% | 257.4 |
| solar-flare | Pypper:one-level@25 (micro distribution voting) | 74.77% | 257.4 |
| solar-flare | Pypper:one-level@25 (voting) | 74.86% | 257.4 |
| solar-flare | Pypper:one-level@50 (best rule) | 72.80% | 522.3 |
| solar-flare | Pypper:one-level@50 (macro distribution voting) | 75.15% | 522.3 |
| solar-flare | Pypper:one-level@50 (micro distribution voting) | 74.68% | 522.3 |
| solar-flare | Pypper:one-level@50 (voting) | 75.43% | 522.3 |
| solar-flare | Pypper:single (best rule) | 69.52% | 7.5 |
| solar-flare | Pypper:single (macro distribution voting) | 69.43% | 7.5 |
| solar-flare | Pypper:single (micro distribution voting) | 69.43% | 7.5 |
| solar-flare | Pypper:single (voting) | 69.24% | 7.5 |
| solar-flare | Pypper:two-level@10 (best rule) | 74.21% | 107.7 |
| solar-flare | Pypper:two-level@10 (macro distribution voting) | 74.59% | 107.7 |
| solar-flare | Pypper:two-level@10 (micro distribution voting) | 75.06% | 107.7 |
| solar-flare | Pypper:two-level@10 (voting) | 75.15% | 107.7 |
| solar-flare | Pypper:two-level@25 (best rule) | 74.96% | 257.4 |
| solar-flare | Pypper:two-level@25 (macro distribution voting) | 74.68% | 257.4 |
| solar-flare | Pypper:two-level@25 (micro distribution voting) | 75.05% | 257.4 |
| solar-flare | Pypper:two-level@25 (voting) | 74.87% | 257.4 |
| solar-flare | Pypper:two-level@50 (best rule) | 74.96% | 522.3 |
| solar-flare | Pypper:two-level@50 (macro distribution voting) | 75.24% | 522.3 |
| solar-flare | Pypper:two-level@50 (micro distribution voting) | 74.96% | 522.3 |
| solar-flare | Pypper:two-level@50 (voting) | 74.96% | 522.3 |
| sonar | PFoil:one-level@10 (best rule) | 71.24% | 119.2 |
| sonar | PFoil:one-level@10 (macro distribution voting) | 82.26% | 119.2 |
| sonar | PFoil:one-level@10 (micro distribution voting) | 79.88% | 119.2 |
| sonar | PFoil:one-level@10 (voting) | 81.79% | 119.2 |
| sonar | PFoil:one-level@25 (best rule) | 71.19% | 301.4 |
| sonar | PFoil:one-level@25 (macro distribution voting) | 81.29% | 301.4 |
| sonar | PFoil:one-level@25 (micro distribution voting) | 78.90% | 301.4 |
| sonar | PFoil:one-level@25 (voting) | 81.29% | 301.4 |
| sonar | PFoil:one-level@50 (best rule) | 71.62% | 601.7 |
| sonar | PFoil:one-level@50 (macro distribution voting) | 80.81% | 601.7 |
| sonar | PFoil:one-level@50 (micro distribution voting) | 79.38% | 601.7 |
| sonar | PFoil:one-level@50 (voting) | 80.33% | 601.7 |
| sonar | PFoil:single (best rule) | 77.90% | 14.8 |
| sonar | PFoil:single (macro distribution voting) | 75.07% | 14.8 |
| sonar | PFoil:single (micro distribution voting) | 76.00% | 14.8 |
| sonar | PFoil:single (voting) | 75.07% | 14.8 |
| sonar | PFoil:two-level@10 (best rule) | 80.33% | 119.2 |
| sonar | PFoil:two-level@10 (macro distribution voting) | 81.31% | 119.2 |
| sonar | PFoil:two-level@10 (micro distribution voting) | 80.81% | 119.2 |
| sonar | PFoil:two-level@10 (voting) | 77.93% | 119.2 |
| sonar | PFoil:two-level@25 (best rule) | 80.36% | 301.4 |
| sonar | PFoil:two-level@25 (macro distribution voting) | 82.26% | 301.4 |
| sonar | PFoil:two-level@25 (micro distribution voting) | 80.83% | 301.4 |
| sonar | PFoil:two-level@25 (voting) | 80.81% | 301.4 |
| sonar | PFoil:two-level@50 (best rule) | 79.86% | 601.7 |
| sonar | PFoil:two-level@50 (macro distribution voting) | 80.33% | 601.7 |
| sonar | PFoil:two-level@50 (micro distribution voting) | 79.86% | 601.7 |
| sonar | PFoil:two-level@50 (voting) | 79.38% | 601.7 |
| sonar | PFossil:one-level@10 (best rule) | 73.60% | 130.0 |
| sonar | PFossil:one-level@10 (macro distribution voting) | 76.95% | 130.0 |
| sonar | PFossil:one-level@10 (micro distribution voting) | 75.48% | 130.0 |
| sonar | PFossil:one-level@10 (voting) | 76.98% | 130.0 |
| sonar | PFossil:one-level@25 (best rule) | 70.24% | 330.0 |
| sonar | PFossil:one-level@25 (macro distribution voting) | 77.93% | 330.0 |
| sonar | PFossil:one-level@25 (micro distribution voting) | 75.02% | 330.0 |
| sonar | PFossil:one-level@25 (voting) | 77.93% | 330.0 |
| sonar | PFossil:one-level@50 (best rule) | 68.29% | 664.2 |
| sonar | PFossil:one-level@50 (macro distribution voting) | 78.43% | 664.2 |
| sonar | PFossil:one-level@50 (micro distribution voting) | 72.64% | 664.2 |
| sonar | PFossil:one-level@50 (voting) | 78.90% | 664.2 |
| sonar | PFossil:single (best rule) | 73.55% | 16.7 |
| sonar | PFossil:single (macro distribution voting) | 75.07% | 16.7 |
| sonar | PFossil:single (micro distribution voting) | 74.02% | 16.7 |
| sonar | PFossil:single (voting) | 75.52% | 16.7 |
| sonar | PFossil:two-level@10 (best rule) | 78.40% | 130.0 |
| sonar | PFossil:two-level@10 (macro distribution voting) | 76.98% | 130.0 |
| sonar | PFossil:two-level@10 (micro distribution voting) | 76.95% | 130.0 |
| sonar | PFossil:two-level@10 (voting) | 77.93% | 130.0 |
| sonar | PFossil:two-level@25 (best rule) | 77.43% | 330.0 |
| sonar | PFossil:two-level@25 (macro distribution voting) | 74.05% | 330.0 |
| sonar | PFossil:two-level@25 (micro distribution voting) | 77.43% | 330.0 |
| sonar | PFossil:two-level@25 (voting) | 75.98% | 330.0 |
| sonar | PFossil:two-level@50 (best rule) | 76.02% | 664.2 |
| sonar | PFossil:two-level@50 (macro distribution voting) | 75.52% | 664.2 |
| sonar | PFossil:two-level@50 (micro distribution voting) | 76.52% | 664.2 |
| sonar | PFossil:two-level@50 (voting) | 75.07% | 664.2 |
| sonar | Pypper:one-level@10 (best rule) | 72.62% | 64.6 |
| sonar | Pypper:one-level@10 (macro distribution voting) | 78.90% | 64.6 |
| sonar | Pypper:one-level@10 (micro distribution voting) | 74.55% | 64.6 |
| sonar | Pypper:one-level@10 (voting) | 76.48% | 64.6 |
| sonar | Pypper:one-level@25 (best rule) | 73.57% | 159.2 |
| sonar | Pypper:one-level@25 (macro distribution voting) | 77.00% | 159.2 |
| sonar | Pypper:one-level@25 (micro distribution voting) | 76.02% | 159.2 |
| sonar | Pypper:one-level@25 (voting) | 77.98% | 159.2 |
| sonar | Pypper:one-level@50 (best rule) | 71.64% | 320.1 |
| sonar | Pypper:one-level@50 (macro distribution voting) | 78.93% | 320.1 |
| sonar | Pypper:one-level@50 (micro distribution voting) | 76.02% | 320.1 |
| sonar | Pypper:one-level@50 (voting) | 78.43% | 320.1 |
| sonar | Pypper:single (best rule) | 73.57% | 6.6 |
| sonar | Pypper:single (macro distribution voting) | 72.62% | 6.6 |
| sonar | Pypper:single (micro distribution voting) | 74.05% | 6.6 |
| sonar | Pypper:single (voting) | 71.21% | 6.6 |
| sonar | Pypper:two-level@10 (best rule) | 76.48% | 64.6 |
| sonar | Pypper:two-level@10 (macro distribution voting) | 77.45% | 64.6 |
| sonar | Pypper:two-level@10 (micro distribution voting) | 74.50% | 64.6 |
| sonar | Pypper:two-level@10 (voting) | 75.00% | 64.6 |
| sonar | Pypper:two-level@25 (best rule) | 75.98% | 159.2 |
| sonar | Pypper:two-level@25 (macro distribution voting) | 78.43% | 159.2 |
| sonar | Pypper:two-level@25 (micro distribution voting) | 76.50% | 159.2 |
| sonar | Pypper:two-level@25 (voting) | 77.95% | 159.2 |
| sonar | Pypper:two-level@50 (best rule) | 77.98% | 320.1 |
| sonar | Pypper:two-level@50 (macro distribution voting) | 78.43% | 320.1 |
| sonar | Pypper:two-level@50 (micro distribution voting) | 77.98% | 320.1 |
| sonar | Pypper:two-level@50 (voting) | 77.02% | 320.1 |
| soybean | PFoil:one-level@10 (best rule) | 88.44% | 476.0 |
| soybean | PFoil:one-level@10 (macro distribution voting) | 93.42% | 476.0 |
| soybean | PFoil:one-level@10 (micro distribution voting) | 90.05% | 476.0 |
| soybean | PFoil:one-level@10 (voting) | 91.81% | 476.0 |
| soybean | PFoil:one-level@25 (best rule) | 85.81% | 1189.0 |
| soybean | PFoil:one-level@25 (macro distribution voting) | 92.69% | 1189.0 |
| soybean | PFoil:one-level@25 (micro distribution voting) | 89.91% | 1189.0 |
| soybean | PFoil:one-level@25 (voting) | 91.95% | 1189.0 |
| soybean | PFoil:one-level@50 (best rule) | 83.61% | 2382.2 |
| soybean | PFoil:one-level@50 (macro distribution voting) | 92.83% | 2382.2 |
| soybean | PFoil:one-level@50 (micro distribution voting) | 90.06% | 2382.2 |
| soybean | PFoil:one-level@50 (voting) | 92.54% | 2382.2 |
| soybean | PFoil:single (best rule) | 90.92% | 53.1 |
| soybean | PFoil:single (macro distribution voting) | 90.35% | 53.1 |
| soybean | PFoil:single (micro distribution voting) | 86.98% | 53.1 |
| soybean | PFoil:single (voting) | 84.78% | 53.1 |
| soybean | PFoil:two-level@10 (best rule) | 92.98% | 476.0 |
| soybean | PFoil:two-level@10 (macro distribution voting) | 91.52% | 476.0 |
| soybean | PFoil:two-level@10 (micro distribution voting) | 91.66% | 476.0 |
| soybean | PFoil:two-level@10 (voting) | 86.69% | 476.0 |
| soybean | PFoil:two-level@25 (best rule) | 92.98% | 1189.0 |
| soybean | PFoil:two-level@25 (macro distribution voting) | 91.66% | 1189.0 |
| soybean | PFoil:two-level@25 (micro distribution voting) | 91.37% | 1189.0 |
| soybean | PFoil:two-level@25 (voting) | 88.88% | 1189.0 |
| soybean | PFoil:two-level@50 (best rule) | 92.40% | 2382.2 |
| soybean | PFoil:two-level@50 (macro distribution voting) | 91.37% | 2382.2 |
| soybean | PFoil:two-level@50 (micro distribution voting) | 91.23% | 2382.2 |
| soybean | PFoil:two-level@50 (voting) | 88.58% | 2382.2 |
| soybean | PFossil:one-level@10 (best rule) | 88.30% | 400.9 |
| soybean | PFossil:one-level@10 (macro distribution voting) | 93.42% | 400.9 |
| soybean | PFossil:one-level@10 (micro distribution voting) | 91.52% | 400.9 |
| soybean | PFossil:one-level@10 (voting) | 92.98% | 400.9 |
| soybean | PFossil:one-level@25 (best rule) | 86.10% | 1002.0 |
| soybean | PFossil:one-level@25 (macro distribution voting) | 94.29% | 1002.0 |
| soybean | PFossil:one-level@25 (micro distribution voting) | 90.93% | 1002.0 |
| soybean | PFossil:one-level@25 (voting) | 93.56% | 1002.0 |
| soybean | PFossil:one-level@50 (best rule) | 84.64% | 2005.2 |
| soybean | PFossil:one-level@50 (macro distribution voting) | 93.71% | 2005.2 |
| soybean | PFossil:one-level@50 (micro distribution voting) | 91.52% | 2005.2 |
| soybean | PFossil:one-level@50 (voting) | 93.12% | 2005.2 |
| soybean | PFossil:single (best rule) | 92.24% | 42.1 |
| soybean | PFossil:single (macro distribution voting) | 91.36% | 42.1 |
| soybean | PFossil:single (micro distribution voting) | 91.66% | 42.1 |
| soybean | PFossil:single (voting) | 90.34% | 42.1 |
| soybean | PFossil:two-level@10 (best rule) | 93.27% | 400.9 |
| soybean | PFossil:two-level@10 (macro distribution voting) | 92.39% | 400.9 |
| soybean | PFossil:two-level@10 (micro distribution voting) | 92.39% | 400.9 |
| soybean | PFossil:two-level@10 (voting) | 91.51% | 400.9 |
| soybean | PFossil:two-level@25 (best rule) | 93.12% | 1002.0 |
| soybean | PFossil:two-level@25 (macro distribution voting) | 93.27% | 1002.0 |
| soybean | PFossil:two-level@25 (micro distribution voting) | 92.24% | 1002.0 |
| soybean | PFossil:two-level@25 (voting) | 92.24% | 1002.0 |
| soybean | PFossil:two-level@50 (best rule) | 93.12% | 2005.2 |
| soybean | PFossil:two-level@50 (macro distribution voting) | 92.83% | 2005.2 |
| soybean | PFossil:two-level@50 (micro distribution voting) | 92.39% | 2005.2 |
| soybean | PFossil:two-level@50 (voting) | 92.10% | 2005.2 |
| soybean | Pypper:one-level@10 (best rule) | 90.78% | 292.9 |
| soybean | Pypper:one-level@10 (macro distribution voting) | 93.56% | 292.9 |
| soybean | Pypper:one-level@10 (micro distribution voting) | 89.02% | 292.9 |
| soybean | Pypper:one-level@10 (voting) | 92.83% | 292.9 |
| soybean | Pypper:one-level@25 (best rule) | 88.44% | 733.1 |
| soybean | Pypper:one-level@25 (macro distribution voting) | 93.42% | 733.1 |
| soybean | Pypper:one-level@25 (micro distribution voting) | 90.49% | 733.1 |
| soybean | Pypper:one-level@25 (voting) | 91.96% | 733.1 |
| soybean | Pypper:one-level@50 (best rule) | 86.40% | 1467.5 |
| soybean | Pypper:one-level@50 (macro distribution voting) | 93.42% | 1467.5 |
| soybean | Pypper:one-level@50 (micro distribution voting) | 90.49% | 1467.5 |
| soybean | Pypper:one-level@50 (voting) | 91.95% | 1467.5 |
| soybean | Pypper:single (best rule) | 90.50% | 28.2 |
| soybean | Pypper:single (macro distribution voting) | 90.50% | 28.2 |
| soybean | Pypper:single (micro distribution voting) | 88.15% | 28.2 |
| soybean | Pypper:single (voting) | 87.71% | 28.2 |
| soybean | Pypper:two-level@10 (best rule) | 92.69% | 292.9 |
| soybean | Pypper:two-level@10 (macro distribution voting) | 92.84% | 292.9 |
| soybean | Pypper:two-level@10 (micro distribution voting) | 90.63% | 292.9 |
| soybean | Pypper:two-level@10 (voting) | 89.76% | 292.9 |
| soybean | Pypper:two-level@25 (best rule) | 92.84% | 733.1 |
| soybean | Pypper:two-level@25 (macro distribution voting) | 92.84% | 733.1 |
| soybean | Pypper:two-level@25 (micro distribution voting) | 90.78% | 733.1 |
| soybean | Pypper:two-level@25 (voting) | 89.76% | 733.1 |
| soybean | Pypper:two-level@50 (best rule) | 92.69% | 1467.5 |
| soybean | Pypper:two-level@50 (macro distribution voting) | 92.69% | 1467.5 |
| soybean | Pypper:two-level@50 (micro distribution voting) | 90.20% | 1467.5 |
| soybean | Pypper:two-level@50 (voting) | 89.32% | 1467.5 |
| spambase | PFoil:one-level@10 (best rule) | 94.37% | 1110.8 |
| spambase | PFoil:one-level@10 (macro distribution voting) | 94.83% | 1110.8 |
| spambase | PFoil:one-level@10 (micro distribution voting) | 93.24% | 1110.8 |
| spambase | PFoil:one-level@10 (voting) | 94.96% | 1110.8 |
| spambase | PFoil:one-level@25 (best rule) | 93.87% | 2754.0 |
| spambase | PFoil:one-level@25 (macro distribution voting) | 94.68% | 2754.0 |
| spambase | PFoil:one-level@25 (micro distribution voting) | 93.31% | 2754.0 |
| spambase | PFoil:one-level@25 (voting) | 94.87% | 2754.0 |
| spambase | PFoil:one-level@50 (best rule) | 93.70% | 5496.1 |
| spambase | PFoil:one-level@50 (macro distribution voting) | 94.76% | 5496.1 |
| spambase | PFoil:one-level@50 (micro distribution voting) | 93.54% | 5496.1 |
| spambase | PFoil:one-level@50 (voting) | 94.83% | 5496.1 |
| spambase | PFoil:single (best rule) | 93.87% | 126.1 |
| spambase | PFoil:single (macro distribution voting) | 94.02% | 126.1 |
| spambase | PFoil:single (micro distribution voting) | 90.50% | 126.1 |
| spambase | PFoil:single (voting) | 93.59% | 126.1 |
| spambase | PFoil:two-level@10 (best rule) | 94.98% | 1110.8 |
| spambase | PFoil:two-level@10 (macro distribution voting) | 94.81% | 1110.8 |
| spambase | PFoil:two-level@10 (micro distribution voting) | 94.04% | 1110.8 |
| spambase | PFoil:two-level@10 (voting) | 94.91% | 1110.8 |
| spambase | PFoil:two-level@25 (best rule) | 94.96% | 2754.0 |
| spambase | PFoil:two-level@25 (macro distribution voting) | 94.96% | 2754.0 |
| spambase | PFoil:two-level@25 (micro distribution voting) | 94.18% | 2754.0 |
| spambase | PFoil:two-level@25 (voting) | 95.02% | 2754.0 |
| spambase | PFoil:two-level@50 (best rule) | 95.04% | 5496.1 |
| spambase | PFoil:two-level@50 (macro distribution voting) | 94.94% | 5496.1 |
| spambase | PFoil:two-level@50 (micro distribution voting) | 94.28% | 5496.1 |
| spambase | PFoil:two-level@50 (voting) | 95.02% | 5496.1 |
| spambase | PFossil:one-level@10 (best rule) | 93.13% | 114.0 |
| spambase | PFossil:one-level@10 (macro distribution voting) | 93.52% | 114.0 |
| spambase | PFossil:one-level@10 (micro distribution voting) | 92.13% | 114.0 |
| spambase | PFossil:one-level@10 (voting) | 93.46% | 114.0 |
| spambase | PFossil:one-level@25 (best rule) | 93.24% | 288.6 |
| spambase | PFossil:one-level@25 (macro distribution voting) | 93.50% | 288.6 |
| spambase | PFossil:one-level@25 (micro distribution voting) | 92.37% | 288.6 |
| spambase | PFossil:one-level@25 (voting) | 93.48% | 288.6 |
| spambase | PFossil:one-level@50 (best rule) | 93.57% | 574.3 |
| spambase | PFossil:one-level@50 (macro distribution voting) | 93.50% | 574.3 |
| spambase | PFossil:one-level@50 (micro distribution voting) | 92.37% | 574.3 |
| spambase | PFossil:one-level@50 (voting) | 93.48% | 574.3 |
| spambase | PFossil:single (best rule) | 92.81% | 12.1 |
| spambase | PFossil:single (macro distribution voting) | 92.76% | 12.1 |
| spambase | PFossil:single (micro distribution voting) | 91.81% | 12.1 |
| spambase | PFossil:single (voting) | 92.41% | 12.1 |
| spambase | PFossil:two-level@10 (best rule) | 93.44% | 114.0 |
| spambase | PFossil:two-level@10 (macro distribution voting) | 93.50% | 114.0 |
| spambase | PFossil:two-level@10 (micro distribution voting) | 92.76% | 114.0 |
| spambase | PFossil:two-level@10 (voting) | 93.05% | 114.0 |
| spambase | PFossil:two-level@25 (best rule) | 93.35% | 288.6 |
| spambase | PFossil:two-level@25 (macro distribution voting) | 93.37% | 288.6 |
| spambase | PFossil:two-level@25 (micro distribution voting) | 92.68% | 288.6 |
| spambase | PFossil:two-level@25 (voting) | 93.07% | 288.6 |
| spambase | PFossil:two-level@50 (best rule) | 93.44% | 574.3 |
| spambase | PFossil:two-level@50 (macro distribution voting) | 93.46% | 574.3 |
| spambase | PFossil:two-level@50 (micro distribution voting) | 92.89% | 574.3 |
| spambase | PFossil:two-level@50 (voting) | 93.09% | 574.3 |
| spambase | Pypper:one-level@10 (best rule) | 94.09% | 257.7 |
| spambase | Pypper:one-level@10 (macro distribution voting) | 94.02% | 257.7 |
| spambase | Pypper:one-level@10 (micro distribution voting) | 93.18% | 257.7 |
| spambase | Pypper:one-level@10 (voting) | 93.91% | 257.7 |
| spambase | Pypper:one-level@25 (best rule) | 93.91% | 655.0 |
| spambase | Pypper:one-level@25 (macro distribution voting) | 94.13% | 655.0 |
| spambase | Pypper:one-level@25 (micro distribution voting) | 93.46% | 655.0 |
| spambase | Pypper:one-level@25 (voting) | 94.13% | 655.0 |
| spambase | Pypper:one-level@50 (best rule) | 93.68% | 1304.2 |
| spambase | Pypper:one-level@50 (macro distribution voting) | 94.20% | 1304.2 |
| spambase | Pypper:one-level@50 (micro distribution voting) | 93.44% | 1304.2 |
| spambase | Pypper:one-level@50 (voting) | 94.18% | 1304.2 |
| spambase | Pypper:single (best rule) | 93.11% | 26.4 |
| spambase | Pypper:single (macro distribution voting) | 93.00% | 26.4 |
| spambase | Pypper:single (micro distribution voting) | 92.59% | 26.4 |
| spambase | Pypper:single (voting) | 92.83% | 26.4 |
| spambase | Pypper:two-level@10 (best rule) | 94.04% | 257.7 |
| spambase | Pypper:two-level@10 (macro distribution voting) | 93.98% | 257.7 |
| spambase | Pypper:two-level@10 (micro distribution voting) | 93.63% | 257.7 |
| spambase | Pypper:two-level@10 (voting) | 93.81% | 257.7 |
| spambase | Pypper:two-level@25 (best rule) | 94.35% | 655.0 |
| spambase | Pypper:two-level@25 (macro distribution voting) | 94.26% | 655.0 |
| spambase | Pypper:two-level@25 (micro distribution voting) | 93.72% | 655.0 |
| spambase | Pypper:two-level@25 (voting) | 94.00% | 655.0 |
| spambase | Pypper:two-level@50 (best rule) | 94.15% | 1304.2 |
| spambase | Pypper:two-level@50 (macro distribution voting) | 94.22% | 1304.2 |
| spambase | Pypper:two-level@50 (micro distribution voting) | 93.89% | 1304.2 |
| spambase | Pypper:two-level@50 (voting) | 94.04% | 1304.2 |
| splice | PFoil:one-level@10 (best rule) | 94.45% | 598.6 |
| splice | PFoil:one-level@10 (macro distribution voting) | 95.99% | 598.6 |
| splice | PFoil:one-level@10 (micro distribution voting) | 95.89% | 598.6 |
| splice | PFoil:one-level@10 (voting) | 95.89% | 598.6 |
| splice | PFoil:one-level@25 (best rule) | 94.45% | 1487.0 |
| splice | PFoil:one-level@25 (macro distribution voting) | 96.08% | 1487.0 |
| splice | PFoil:one-level@25 (micro distribution voting) | 95.99% | 1487.0 |
| splice | PFoil:one-level@25 (voting) | 96.11% | 1487.0 |
| splice | PFoil:one-level@50 (best rule) | 94.33% | 2978.9 |
| splice | PFoil:one-level@50 (macro distribution voting) | 96.14% | 2978.9 |
| splice | PFoil:one-level@50 (micro distribution voting) | 95.96% | 2978.9 |
| splice | PFoil:one-level@50 (voting) | 96.05% | 2978.9 |
| splice | PFoil:single (best rule) | 92.76% | 78.2 |
| splice | PFoil:single (macro distribution voting) | 92.82% | 78.2 |
| splice | PFoil:single (micro distribution voting) | 92.79% | 78.2 |
| splice | PFoil:single (voting) | 92.41% | 78.2 |
| splice | PFoil:two-level@10 (best rule) | 95.89% | 598.6 |
| splice | PFoil:two-level@10 (macro distribution voting) | 96.14% | 598.6 |
| splice | PFoil:two-level@10 (micro distribution voting) | 95.86% | 598.6 |
| splice | PFoil:two-level@10 (voting) | 95.92% | 598.6 |
| splice | PFoil:two-level@25 (best rule) | 96.21% | 1487.0 |
| splice | PFoil:two-level@25 (macro distribution voting) | 96.18% | 1487.0 |
| splice | PFoil:two-level@25 (micro distribution voting) | 96.24% | 1487.0 |
| splice | PFoil:two-level@25 (voting) | 96.08% | 1487.0 |
| splice | PFoil:two-level@50 (best rule) | 96.18% | 2978.9 |
| splice | PFoil:two-level@50 (macro distribution voting) | 96.27% | 2978.9 |
| splice | PFoil:two-level@50 (micro distribution voting) | 96.14% | 2978.9 |
| splice | PFoil:two-level@50 (voting) | 96.27% | 2978.9 |
| splice | PFossil:one-level@10 (best rule) | 90.72% | 288.6 |
| splice | PFossil:one-level@10 (macro distribution voting) | 95.33% | 288.6 |
| splice | PFossil:one-level@10 (micro distribution voting) | 90.78% | 288.6 |
| splice | PFossil:one-level@10 (voting) | 94.08% | 288.6 |
| splice | PFossil:one-level@25 (best rule) | 89.62% | 729.9 |
| splice | PFossil:one-level@25 (macro distribution voting) | 95.49% | 729.9 |
| splice | PFossil:one-level@25 (micro distribution voting) | 90.63% | 729.9 |
| splice | PFossil:one-level@25 (voting) | 94.80% | 729.9 |
| splice | PFossil:one-level@50 (best rule) | 89.22% | 1442.8 |
| splice | PFossil:one-level@50 (macro distribution voting) | 95.64% | 1442.8 |
| splice | PFossil:one-level@50 (micro distribution voting) | 90.66% | 1442.8 |
| splice | PFossil:one-level@50 (voting) | 95.20% | 1442.8 |
| splice | PFossil:single (best rule) | 94.23% | 23.7 |
| splice | PFossil:single (macro distribution voting) | 94.29% | 23.7 |
| splice | PFossil:single (micro distribution voting) | 90.38% | 23.7 |
| splice | PFossil:single (voting) | 89.09% | 23.7 |
| splice | PFossil:two-level@10 (best rule) | 95.30% | 288.6 |
| splice | PFossil:two-level@10 (macro distribution voting) | 95.49% | 288.6 |
| splice | PFossil:two-level@10 (micro distribution voting) | 90.91% | 288.6 |
| splice | PFossil:two-level@10 (voting) | 90.60% | 288.6 |
| splice | PFossil:two-level@25 (best rule) | 95.33% | 729.9 |
| splice | PFossil:two-level@25 (macro distribution voting) | 95.45% | 729.9 |
| splice | PFossil:two-level@25 (micro distribution voting) | 90.72% | 729.9 |
| splice | PFossil:two-level@25 (voting) | 90.94% | 729.9 |
| splice | PFossil:two-level@50 (best rule) | 95.30% | 1442.8 |
| splice | PFossil:two-level@50 (macro distribution voting) | 95.49% | 1442.8 |
| splice | PFossil:two-level@50 (micro distribution voting) | 90.78% | 1442.8 |
| splice | PFossil:two-level@50 (voting) | 91.00% | 1442.8 |
| splice | Pypper:one-level@10 (best rule) | 94.64% | 238.6 |
| splice | Pypper:one-level@10 (macro distribution voting) | 95.52% | 238.6 |
| splice | Pypper:one-level@10 (micro distribution voting) | 95.14% | 238.6 |
| splice | Pypper:one-level@10 (voting) | 95.42% | 238.6 |
| splice | Pypper:one-level@25 (best rule) | 94.86% | 593.5 |
| splice | Pypper:one-level@25 (macro distribution voting) | 95.55% | 593.5 |
| splice | Pypper:one-level@25 (micro distribution voting) | 95.24% | 593.5 |
| splice | Pypper:one-level@25 (voting) | 95.55% | 593.5 |
| splice | Pypper:one-level@50 (best rule) | 94.95% | 1191.7 |
| splice | Pypper:one-level@50 (macro distribution voting) | 95.55% | 1191.7 |
| splice | Pypper:one-level@50 (micro distribution voting) | 95.05% | 1191.7 |
| splice | Pypper:one-level@50 (voting) | 95.61% | 1191.7 |
| splice | Pypper:single (best rule) | 93.20% | 26.8 |
| splice | Pypper:single (macro distribution voting) | 93.39% | 26.8 |
| splice | Pypper:single (micro distribution voting) | 93.39% | 26.8 |
| splice | Pypper:single (voting) | 93.26% | 26.8 |
| splice | Pypper:two-level@10 (best rule) | 95.30% | 238.6 |
| splice | Pypper:two-level@10 (macro distribution voting) | 95.45% | 238.6 |
| splice | Pypper:two-level@10 (micro distribution voting) | 95.11% | 238.6 |
| splice | Pypper:two-level@10 (voting) | 95.61% | 238.6 |
| splice | Pypper:two-level@25 (best rule) | 95.55% | 593.5 |
| splice | Pypper:two-level@25 (macro distribution voting) | 95.61% | 593.5 |
| splice | Pypper:two-level@25 (micro distribution voting) | 95.33% | 593.5 |
| splice | Pypper:two-level@25 (voting) | 95.67% | 593.5 |
| splice | Pypper:two-level@50 (best rule) | 95.52% | 1191.7 |
| splice | Pypper:two-level@50 (macro distribution voting) | 95.64% | 1191.7 |
| splice | Pypper:two-level@50 (micro distribution voting) | 95.42% | 1191.7 |
| splice | Pypper:two-level@50 (voting) | 95.64% | 1191.7 |
| texture | PFoil:one-level@10 (best rule) | 93.07% | 1317.8 |
| texture | PFoil:one-level@10 (macro distribution voting) | 97.35% | 1317.8 |
| texture | PFoil:one-level@10 (micro distribution voting) | 95.62% | 1317.8 |
| texture | PFoil:one-level@10 (voting) | 97.20% | 1317.8 |
| texture | PFoil:one-level@25 (best rule) | 91.87% | 3299.0 |
| texture | PFoil:one-level@25 (macro distribution voting) | 97.60% | 3299.0 |
| texture | PFoil:one-level@25 (micro distribution voting) | 95.69% | 3299.0 |
| texture | PFoil:one-level@25 (voting) | 97.47% | 3299.0 |
| texture | PFoil:one-level@50 (best rule) | 91.15% | 6610.9 |
| texture | PFoil:one-level@50 (macro distribution voting) | 97.51% | 6610.9 |
| texture | PFoil:one-level@50 (micro distribution voting) | 95.78% | 6610.9 |
| texture | PFoil:one-level@50 (voting) | 97.47% | 6610.9 |
| texture | PFoil:single (best rule) | 94.51% | 156.3 |
| texture | PFoil:single (macro distribution voting) | 95.02% | 156.3 |
| texture | PFoil:single (micro distribution voting) | 91.93% | 156.3 |
| texture | PFoil:single (voting) | 93.25% | 156.3 |
| texture | PFoil:two-level@10 (best rule) | 96.96% | 1317.8 |
| texture | PFoil:two-level@10 (macro distribution voting) | 97.38% | 1317.8 |
| texture | PFoil:two-level@10 (micro distribution voting) | 96.51% | 1317.8 |
| texture | PFoil:two-level@10 (voting) | 97.02% | 1317.8 |
| texture | PFoil:two-level@25 (best rule) | 97.25% | 3299.0 |
| texture | PFoil:two-level@25 (macro distribution voting) | 97.69% | 3299.0 |
| texture | PFoil:two-level@25 (micro distribution voting) | 96.76% | 3299.0 |
| texture | PFoil:two-level@25 (voting) | 97.44% | 3299.0 |
| texture | PFoil:two-level@50 (best rule) | 97.18% | 6610.9 |
| texture | PFoil:two-level@50 (macro distribution voting) | 97.53% | 6610.9 |
| texture | PFoil:two-level@50 (micro distribution voting) | 96.75% | 6610.9 |
| texture | PFoil:two-level@50 (voting) | 97.51% | 6610.9 |
| texture | PFossil:one-level@10 (best rule) | 93.07% | 835.4 |
| texture | PFossil:one-level@10 (macro distribution voting) | 95.96% | 835.4 |
| texture | PFossil:one-level@10 (micro distribution voting) | 92.45% | 835.4 |
| texture | PFossil:one-level@10 (voting) | 95.71% | 835.4 |
| texture | PFossil:one-level@25 (best rule) | 92.71% | 2064.1 |
| texture | PFossil:one-level@25 (macro distribution voting) | 96.20% | 2064.1 |
| texture | PFossil:one-level@25 (micro distribution voting) | 92.87% | 2064.1 |
| texture | PFossil:one-level@25 (voting) | 95.95% | 2064.1 |
| texture | PFossil:one-level@50 (best rule) | 92.16% | 4124.7 |
| texture | PFossil:one-level@50 (macro distribution voting) | 96.27% | 4124.7 |
| texture | PFossil:one-level@50 (micro distribution voting) | 92.58% | 4124.7 |
| texture | PFossil:one-level@50 (voting) | 96.02% | 4124.7 |
| texture | PFossil:single (best rule) | 92.22% | 77.9 |
| texture | PFossil:single (macro distribution voting) | 92.45% | 77.9 |
| texture | PFossil:single (micro distribution voting) | 90.29% | 77.9 |
| texture | PFossil:single (voting) | 91.65% | 77.9 |
| texture | PFossil:two-level@10 (best rule) | 95.73% | 835.4 |
| texture | PFossil:two-level@10 (macro distribution voting) | 95.84% | 835.4 |
| texture | PFossil:two-level@10 (micro distribution voting) | 93.24% | 835.4 |
| texture | PFossil:two-level@10 (voting) | 95.20% | 835.4 |
| texture | PFossil:two-level@25 (best rule) | 96.20% | 2064.1 |
| texture | PFossil:two-level@25 (macro distribution voting) | 96.38% | 2064.1 |
| texture | PFossil:two-level@25 (micro distribution voting) | 93.55% | 2064.1 |
| texture | PFossil:two-level@25 (voting) | 95.73% | 2064.1 |
| texture | PFossil:two-level@50 (best rule) | 96.24% | 4124.7 |
| texture | PFossil:two-level@50 (macro distribution voting) | 96.45% | 4124.7 |
| texture | PFossil:two-level@50 (micro distribution voting) | 93.64% | 4124.7 |
| texture | PFossil:two-level@50 (voting) | 95.93% | 4124.7 |
| texture | Pypper:one-level@10 (best rule) | 93.40% | 746.2 |
| texture | Pypper:one-level@10 (macro distribution voting) | 96.60% | 746.2 |
| texture | Pypper:one-level@10 (micro distribution voting) | 95.07% | 746.2 |
| texture | Pypper:one-level@10 (voting) | 96.58% | 746.2 |
| texture | Pypper:one-level@25 (best rule) | 92.58% | 1862.2 |
| texture | Pypper:one-level@25 (macro distribution voting) | 96.96% | 1862.2 |
| texture | Pypper:one-level@25 (micro distribution voting) | 95.42% | 1862.2 |
| texture | Pypper:one-level@25 (voting) | 96.93% | 1862.2 |
| texture | Pypper:one-level@50 (best rule) | 92.02% | 3720.4 |
| texture | Pypper:one-level@50 (macro distribution voting) | 96.98% | 3720.4 |
| texture | Pypper:one-level@50 (micro distribution voting) | 95.56% | 3720.4 |
| texture | Pypper:one-level@50 (voting) | 97.04% | 3720.4 |
| texture | Pypper:single (best rule) | 92.40% | 81.3 |
| texture | Pypper:single (macro distribution voting) | 92.82% | 81.3 |
| texture | Pypper:single (micro distribution voting) | 92.05% | 81.3 |
| texture | Pypper:single (voting) | 92.38% | 81.3 |
| texture | Pypper:two-level@10 (best rule) | 96.45% | 746.2 |
| texture | Pypper:two-level@10 (macro distribution voting) | 96.69% | 746.2 |
| texture | Pypper:two-level@10 (micro distribution voting) | 96.04% | 746.2 |
| texture | Pypper:two-level@10 (voting) | 96.44% | 746.2 |
| texture | Pypper:two-level@25 (best rule) | 96.75% | 1862.2 |
| texture | Pypper:two-level@25 (macro distribution voting) | 96.96% | 1862.2 |
| texture | Pypper:two-level@25 (micro distribution voting) | 96.33% | 1862.2 |
| texture | Pypper:two-level@25 (voting) | 96.85% | 1862.2 |
| texture | Pypper:two-level@50 (best rule) | 96.73% | 3720.4 |
| texture | Pypper:two-level@50 (macro distribution voting) | 97.02% | 3720.4 |
| texture | Pypper:two-level@50 (micro distribution voting) | 96.35% | 3720.4 |
| texture | Pypper:two-level@50 (voting) | 96.87% | 3720.4 |
| tic-tac-toe | PFoil:one-level@10 (best rule) | 97.18% | 238.4 |
| tic-tac-toe | PFoil:one-level@10 (macro distribution voting) | 99.27% | 238.4 |
| tic-tac-toe | PFoil:one-level@10 (micro distribution voting) | 98.85% | 238.4 |
| tic-tac-toe | PFoil:one-level@10 (voting) | 99.58% | 238.4 |
| tic-tac-toe | PFoil:one-level@25 (best rule) | 96.55% | 597.0 |
| tic-tac-toe | PFoil:one-level@25 (macro distribution voting) | 99.38% | 597.0 |
| tic-tac-toe | PFoil:one-level@25 (micro distribution voting) | 98.44% | 597.0 |
| tic-tac-toe | PFoil:one-level@25 (voting) | 99.69% | 597.0 |
| tic-tac-toe | PFoil:one-level@50 (best rule) | 94.57% | 1187.2 |
| tic-tac-toe | PFoil:one-level@50 (macro distribution voting) | 99.38% | 1187.2 |
| tic-tac-toe | PFoil:one-level@50 (micro distribution voting) | 98.33% | 1187.2 |
| tic-tac-toe | PFoil:one-level@50 (voting) | 99.79% | 1187.2 |
| tic-tac-toe | PFoil:single (best rule) | 99.06% | 23.2 |
| tic-tac-toe | PFoil:single (macro distribution voting) | 98.85% | 23.2 |
| tic-tac-toe | PFoil:single (micro distribution voting) | 99.06% | 23.2 |
| tic-tac-toe | PFoil:single (voting) | 98.43% | 23.2 |
| tic-tac-toe | PFoil:two-level@10 (best rule) | 99.48% | 238.4 |
| tic-tac-toe | PFoil:two-level@10 (macro distribution voting) | 99.48% | 238.4 |
| tic-tac-toe | PFoil:two-level@10 (micro distribution voting) | 99.48% | 238.4 |
| tic-tac-toe | PFoil:two-level@10 (voting) | 99.48% | 238.4 |
| tic-tac-toe | PFoil:two-level@25 (best rule) | 99.69% | 597.0 |
| tic-tac-toe | PFoil:two-level@25 (macro distribution voting) | 99.69% | 597.0 |
| tic-tac-toe | PFoil:two-level@25 (micro distribution voting) | 99.69% | 597.0 |
| tic-tac-toe | PFoil:two-level@25 (voting) | 99.69% | 597.0 |
| tic-tac-toe | PFoil:two-level@50 (best rule) | 99.69% | 1187.2 |
| tic-tac-toe | PFoil:two-level@50 (macro distribution voting) | 99.69% | 1187.2 |
| tic-tac-toe | PFoil:two-level@50 (micro distribution voting) | 99.69% | 1187.2 |
| tic-tac-toe | PFoil:two-level@50 (voting) | 99.69% | 1187.2 |
| tic-tac-toe | PFossil:one-level@10 (best rule) | 97.49% | 180.1 |
| tic-tac-toe | PFossil:one-level@10 (macro distribution voting) | 97.80% | 180.1 |
| tic-tac-toe | PFossil:one-level@10 (micro distribution voting) | 84.66% | 180.1 |
| tic-tac-toe | PFossil:one-level@10 (voting) | 93.74% | 180.1 |
| tic-tac-toe | PFossil:one-level@25 (best rule) | 97.18% | 450.5 |
| tic-tac-toe | PFossil:one-level@25 (macro distribution voting) | 98.85% | 450.5 |
| tic-tac-toe | PFossil:one-level@25 (micro distribution voting) | 84.97% | 450.5 |
| tic-tac-toe | PFossil:one-level@25 (voting) | 96.66% | 450.5 |
| tic-tac-toe | PFossil:one-level@50 (best rule) | 96.45% | 883.0 |
| tic-tac-toe | PFossil:one-level@50 (macro distribution voting) | 98.85% | 883.0 |
| tic-tac-toe | PFossil:one-level@50 (micro distribution voting) | 85.29% | 883.0 |
| tic-tac-toe | PFossil:one-level@50 (voting) | 96.98% | 883.0 |
| tic-tac-toe | PFossil:single (best rule) | 97.71% | 14.7 |
| tic-tac-toe | PFossil:single (macro distribution voting) | 95.10% | 14.7 |
| tic-tac-toe | PFossil:single (micro distribution voting) | 83.72% | 14.7 |
| tic-tac-toe | PFossil:single (voting) | 85.29% | 14.7 |
| tic-tac-toe | PFossil:two-level@10 (best rule) | 98.85% | 180.1 |
| tic-tac-toe | PFossil:two-level@10 (macro distribution voting) | 98.23% | 180.1 |
| tic-tac-toe | PFossil:two-level@10 (micro distribution voting) | 87.79% | 180.1 |
| tic-tac-toe | PFossil:two-level@10 (voting) | 89.77% | 180.1 |
| tic-tac-toe | PFossil:two-level@25 (best rule) | 99.17% | 450.5 |
| tic-tac-toe | PFossil:two-level@25 (macro distribution voting) | 99.17% | 450.5 |
| tic-tac-toe | PFossil:two-level@25 (micro distribution voting) | 89.88% | 450.5 |
| tic-tac-toe | PFossil:two-level@25 (voting) | 93.01% | 450.5 |
| tic-tac-toe | PFossil:two-level@50 (best rule) | 99.17% | 883.0 |
| tic-tac-toe | PFossil:two-level@50 (macro distribution voting) | 99.17% | 883.0 |
| tic-tac-toe | PFossil:two-level@50 (micro distribution voting) | 89.36% | 883.0 |
| tic-tac-toe | PFossil:two-level@50 (voting) | 90.62% | 883.0 |
| tic-tac-toe | Pypper:one-level@10 (best rule) | 97.60% | 176.7 |
| tic-tac-toe | Pypper:one-level@10 (macro distribution voting) | 98.75% | 176.7 |
| tic-tac-toe | Pypper:one-level@10 (micro distribution voting) | 98.54% | 176.7 |
| tic-tac-toe | Pypper:one-level@10 (voting) | 98.75% | 176.7 |
| tic-tac-toe | Pypper:one-level@25 (best rule) | 97.18% | 438.5 |
| tic-tac-toe | Pypper:one-level@25 (macro distribution voting) | 98.75% | 438.5 |
| tic-tac-toe | Pypper:one-level@25 (micro distribution voting) | 98.33% | 438.5 |
| tic-tac-toe | Pypper:one-level@25 (voting) | 98.75% | 438.5 |
| tic-tac-toe | Pypper:one-level@50 (best rule) | 96.66% | 870.0 |
| tic-tac-toe | Pypper:one-level@50 (macro distribution voting) | 98.96% | 870.0 |
| tic-tac-toe | Pypper:one-level@50 (micro distribution voting) | 98.33% | 870.0 |
| tic-tac-toe | Pypper:one-level@50 (voting) | 98.96% | 870.0 |
| tic-tac-toe | Pypper:single (best rule) | 98.33% | 16.6 |
| tic-tac-toe | Pypper:single (macro distribution voting) | 98.33% | 16.6 |
| tic-tac-toe | Pypper:single (micro distribution voting) | 98.33% | 16.6 |
| tic-tac-toe | Pypper:single (voting) | 98.33% | 16.6 |
| tic-tac-toe | Pypper:two-level@10 (best rule) | 98.64% | 176.7 |
| tic-tac-toe | Pypper:two-level@10 (macro distribution voting) | 98.64% | 176.7 |
| tic-tac-toe | Pypper:two-level@10 (micro distribution voting) | 98.64% | 176.7 |
| tic-tac-toe | Pypper:two-level@10 (voting) | 98.64% | 176.7 |
| tic-tac-toe | Pypper:two-level@25 (best rule) | 98.75% | 438.5 |
| tic-tac-toe | Pypper:two-level@25 (macro distribution voting) | 98.75% | 438.5 |
| tic-tac-toe | Pypper:two-level@25 (micro distribution voting) | 98.75% | 438.5 |
| tic-tac-toe | Pypper:two-level@25 (voting) | 98.75% | 438.5 |
| tic-tac-toe | Pypper:two-level@50 (best rule) | 98.85% | 870.0 |
| tic-tac-toe | Pypper:two-level@50 (macro distribution voting) | 98.85% | 870.0 |
| tic-tac-toe | Pypper:two-level@50 (micro distribution voting) | 98.85% | 870.0 |
| tic-tac-toe | Pypper:two-level@50 (voting) | 98.85% | 870.0 |
| vehicle | PFoil:one-level@10 (best rule) | 69.99% | 740.8 |
| vehicle | PFoil:one-level@10 (macro distribution voting) | 74.48% | 740.8 |
| vehicle | PFoil:one-level@10 (micro distribution voting) | 72.12% | 740.8 |
| vehicle | PFoil:one-level@10 (voting) | 74.60% | 740.8 |
| vehicle | PFoil:one-level@25 (best rule) | 67.86% | 1861.7 |
| vehicle | PFoil:one-level@25 (macro distribution voting) | 74.24% | 1861.7 |
| vehicle | PFoil:one-level@25 (micro distribution voting) | 72.82% | 1861.7 |
| vehicle | PFoil:one-level@25 (voting) | 75.54% | 1861.7 |
| vehicle | PFoil:one-level@50 (best rule) | 67.98% | 3722.8 |
| vehicle | PFoil:one-level@50 (macro distribution voting) | 74.83% | 3722.8 |
| vehicle | PFoil:one-level@50 (micro distribution voting) | 72.12% | 3722.8 |
| vehicle | PFoil:one-level@50 (voting) | 75.19% | 3722.8 |
| vehicle | PFoil:single (best rule) | 71.87% | 87.8 |
| vehicle | PFoil:single (macro distribution voting) | 71.41% | 87.8 |
| vehicle | PFoil:single (micro distribution voting) | 71.28% | 87.8 |
| vehicle | PFoil:single (voting) | 68.92% | 87.8 |
| vehicle | PFoil:two-level@10 (best rule) | 74.12% | 740.8 |
| vehicle | PFoil:two-level@10 (macro distribution voting) | 73.77% | 740.8 |
| vehicle | PFoil:two-level@10 (micro distribution voting) | 73.77% | 740.8 |
| vehicle | PFoil:two-level@10 (voting) | 73.41% | 740.8 |
| vehicle | PFoil:two-level@25 (best rule) | 74.24% | 1861.7 |
| vehicle | PFoil:two-level@25 (macro distribution voting) | 75.42% | 1861.7 |
| vehicle | PFoil:two-level@25 (micro distribution voting) | 74.00% | 1861.7 |
| vehicle | PFoil:two-level@25 (voting) | 75.31% | 1861.7 |
| vehicle | PFoil:two-level@50 (best rule) | 74.72% | 3722.8 |
| vehicle | PFoil:two-level@50 (macro distribution voting) | 75.66% | 3722.8 |
| vehicle | PFoil:two-level@50 (micro distribution voting) | 74.13% | 3722.8 |
| vehicle | PFoil:two-level@50 (voting) | 74.60% | 3722.8 |
| vehicle | PFossil:one-level@10 (best rule) | 71.04% | 236.3 |
| vehicle | PFossil:one-level@10 (macro distribution voting) | 74.71% | 236.3 |
| vehicle | PFossil:one-level@10 (micro distribution voting) | 72.94% | 236.3 |
| vehicle | PFossil:one-level@10 (voting) | 74.00% | 236.3 |
| vehicle | PFossil:one-level@25 (best rule) | 67.50% | 595.5 |
| vehicle | PFossil:one-level@25 (macro distribution voting) | 75.53% | 595.5 |
| vehicle | PFossil:one-level@25 (micro distribution voting) | 72.11% | 595.5 |
| vehicle | PFossil:one-level@25 (voting) | 74.23% | 595.5 |
| vehicle | PFossil:one-level@50 (best rule) | 66.91% | 1189.4 |
| vehicle | PFossil:one-level@50 (macro distribution voting) | 74.59% | 1189.4 |
| vehicle | PFossil:one-level@50 (micro distribution voting) | 72.47% | 1189.4 |
| vehicle | PFossil:one-level@50 (voting) | 73.88% | 1189.4 |
| vehicle | PFossil:single (best rule) | 71.64% | 26.1 |
| vehicle | PFossil:single (macro distribution voting) | 72.82% | 26.1 |
| vehicle | PFossil:single (micro distribution voting) | 70.93% | 26.1 |
| vehicle | PFossil:single (voting) | 71.53% | 26.1 |
| vehicle | PFossil:two-level@10 (best rule) | 73.77% | 236.3 |
| vehicle | PFossil:two-level@10 (macro distribution voting) | 74.36% | 236.3 |
| vehicle | PFossil:two-level@10 (micro distribution voting) | 73.29% | 236.3 |
| vehicle | PFossil:two-level@10 (voting) | 74.47% | 236.3 |
| vehicle | PFossil:two-level@25 (best rule) | 73.06% | 595.5 |
| vehicle | PFossil:two-level@25 (macro distribution voting) | 74.36% | 595.5 |
| vehicle | PFossil:two-level@25 (micro distribution voting) | 72.82% | 595.5 |
| vehicle | PFossil:two-level@25 (voting) | 75.30% | 595.5 |
| vehicle | PFossil:two-level@50 (best rule) | 72.47% | 1189.4 |
| vehicle | PFossil:two-level@50 (macro distribution voting) | 73.88% | 1189.4 |
| vehicle | PFossil:two-level@50 (micro distribution voting) | 72.35% | 1189.4 |
| vehicle | PFossil:two-level@50 (voting) | 74.82% | 1189.4 |
| vehicle | Pypper:one-level@10 (best rule) | 70.70% | 172.9 |
| vehicle | Pypper:one-level@10 (macro distribution voting) | 73.89% | 172.9 |
| vehicle | Pypper:one-level@10 (micro distribution voting) | 72.59% | 172.9 |
| vehicle | Pypper:one-level@10 (voting) | 73.88% | 172.9 |
| vehicle | Pypper:one-level@25 (best rule) | 69.05% | 441.2 |
| vehicle | Pypper:one-level@25 (macro distribution voting) | 74.71% | 441.2 |
| vehicle | Pypper:one-level@25 (micro distribution voting) | 71.99% | 441.2 |
| vehicle | Pypper:one-level@25 (voting) | 74.71% | 441.2 |
| vehicle | Pypper:one-level@50 (best rule) | 68.10% | 874.1 |
| vehicle | Pypper:one-level@50 (macro distribution voting) | 73.64% | 874.1 |
| vehicle | Pypper:one-level@50 (micro distribution voting) | 71.75% | 874.1 |
| vehicle | Pypper:one-level@50 (voting) | 73.29% | 874.1 |
| vehicle | Pypper:single (best rule) | 63.95% | 17.4 |
| vehicle | Pypper:single (macro distribution voting) | 64.19% | 17.4 |
| vehicle | Pypper:single (micro distribution voting) | 62.65% | 17.4 |
| vehicle | Pypper:single (voting) | 62.90% | 17.4 |
| vehicle | Pypper:two-level@10 (best rule) | 73.29% | 172.9 |
| vehicle | Pypper:two-level@10 (macro distribution voting) | 73.77% | 172.9 |
| vehicle | Pypper:two-level@10 (micro distribution voting) | 73.29% | 172.9 |
| vehicle | Pypper:two-level@10 (voting) | 74.00% | 172.9 |
| vehicle | Pypper:two-level@25 (best rule) | 74.47% | 441.2 |
| vehicle | Pypper:two-level@25 (macro distribution voting) | 74.83% | 441.2 |
| vehicle | Pypper:two-level@25 (micro distribution voting) | 73.88% | 441.2 |
| vehicle | Pypper:two-level@25 (voting) | 73.88% | 441.2 |
| vehicle | Pypper:two-level@50 (best rule) | 73.06% | 874.1 |
| vehicle | Pypper:two-level@50 (macro distribution voting) | 73.88% | 874.1 |
| vehicle | Pypper:two-level@50 (micro distribution voting) | 72.70% | 874.1 |
| vehicle | Pypper:two-level@50 (voting) | 73.88% | 874.1 |
| vote | PFoil:one-level@10 (best rule) | 94.49% | 176.1 |
| vote | PFoil:one-level@10 (macro distribution voting) | 95.87% | 176.1 |
| vote | PFoil:one-level@10 (micro distribution voting) | 95.18% | 176.1 |
| vote | PFoil:one-level@10 (voting) | 95.64% | 176.1 |
| vote | PFoil:one-level@25 (best rule) | 95.18% | 440.2 |
| vote | PFoil:one-level@25 (macro distribution voting) | 96.10% | 440.2 |
| vote | PFoil:one-level@25 (micro distribution voting) | 95.18% | 440.2 |
| vote | PFoil:one-level@25 (voting) | 95.64% | 440.2 |
| vote | PFoil:one-level@50 (best rule) | 94.73% | 871.4 |
| vote | PFoil:one-level@50 (macro distribution voting) | 96.34% | 871.4 |
| vote | PFoil:one-level@50 (micro distribution voting) | 95.42% | 871.4 |
| vote | PFoil:one-level@50 (voting) | 96.10% | 871.4 |
| vote | PFoil:single (best rule) | 94.71% | 23.3 |
| vote | PFoil:single (macro distribution voting) | 94.71% | 23.3 |
| vote | PFoil:single (micro distribution voting) | 94.03% | 23.3 |
| vote | PFoil:single (voting) | 95.17% | 23.3 |
| vote | PFoil:two-level@10 (best rule) | 95.41% | 176.1 |
| vote | PFoil:two-level@10 (macro distribution voting) | 96.33% | 176.1 |
| vote | PFoil:two-level@10 (micro distribution voting) | 95.19% | 176.1 |
| vote | PFoil:two-level@10 (voting) | 95.86% | 176.1 |
| vote | PFoil:two-level@25 (best rule) | 95.18% | 440.2 |
| vote | PFoil:two-level@25 (macro distribution voting) | 95.87% | 440.2 |
| vote | PFoil:two-level@25 (micro distribution voting) | 95.18% | 440.2 |
| vote | PFoil:two-level@25 (voting) | 95.86% | 440.2 |
| vote | PFoil:two-level@50 (best rule) | 95.41% | 871.4 |
| vote | PFoil:two-level@50 (macro distribution voting) | 95.87% | 871.4 |
| vote | PFoil:two-level@50 (micro distribution voting) | 95.18% | 871.4 |
| vote | PFoil:two-level@50 (voting) | 95.87% | 871.4 |
| vote | PFossil:one-level@10 (best rule) | 93.55% | 74.0 |
| vote | PFossil:one-level@10 (macro distribution voting) | 95.65% | 74.0 |
| vote | PFossil:one-level@10 (micro distribution voting) | 95.19% | 74.0 |
| vote | PFossil:one-level@10 (voting) | 95.42% | 74.0 |
| vote | PFossil:one-level@25 (best rule) | 92.65% | 184.7 |
| vote | PFossil:one-level@25 (macro distribution voting) | 95.87% | 184.7 |
| vote | PFossil:one-level@25 (micro distribution voting) | 95.65% | 184.7 |
| vote | PFossil:one-level@25 (voting) | 95.64% | 184.7 |
| vote | PFossil:one-level@50 (best rule) | 92.18% | 372.3 |
| vote | PFossil:one-level@50 (macro distribution voting) | 96.10% | 372.3 |
| vote | PFossil:one-level@50 (micro distribution voting) | 95.65% | 372.3 |
| vote | PFossil:one-level@50 (voting) | 96.34% | 372.3 |
| vote | PFossil:single (best rule) | 95.18% | 10.2 |
| vote | PFossil:single (macro distribution voting) | 93.57% | 10.2 |
| vote | PFossil:single (micro distribution voting) | 95.42% | 10.2 |
| vote | PFossil:single (voting) | 94.03% | 10.2 |
| vote | PFossil:two-level@10 (best rule) | 95.63% | 74.0 |
| vote | PFossil:two-level@10 (macro distribution voting) | 95.17% | 74.0 |
| vote | PFossil:two-level@10 (micro distribution voting) | 95.19% | 74.0 |
| vote | PFossil:two-level@10 (voting) | 94.02% | 74.0 |
| vote | PFossil:two-level@25 (best rule) | 95.64% | 184.7 |
| vote | PFossil:two-level@25 (macro distribution voting) | 95.64% | 184.7 |
| vote | PFossil:two-level@25 (micro distribution voting) | 95.65% | 184.7 |
| vote | PFossil:two-level@25 (voting) | 95.62% | 184.7 |
| vote | PFossil:two-level@50 (best rule) | 95.87% | 372.3 |
| vote | PFossil:two-level@50 (macro distribution voting) | 95.41% | 372.3 |
| vote | PFossil:two-level@50 (micro distribution voting) | 95.65% | 372.3 |
| vote | PFossil:two-level@50 (voting) | 94.94% | 372.3 |
| vote | Pypper:one-level@10 (best rule) | 94.26% | 39.8 |
| vote | Pypper:one-level@10 (macro distribution voting) | 95.65% | 39.8 |
| vote | Pypper:one-level@10 (micro distribution voting) | 95.65% | 39.8 |
| vote | Pypper:one-level@10 (voting) | 95.65% | 39.8 |
| vote | Pypper:one-level@25 (best rule) | 95.18% | 106.5 |
| vote | Pypper:one-level@25 (macro distribution voting) | 95.65% | 106.5 |
| vote | Pypper:one-level@25 (micro distribution voting) | 95.65% | 106.5 |
| vote | Pypper:one-level@25 (voting) | 95.65% | 106.5 |
| vote | Pypper:one-level@50 (best rule) | 94.95% | 219.8 |
| vote | Pypper:one-level@50 (macro distribution voting) | 95.65% | 219.8 |
| vote | Pypper:one-level@50 (micro distribution voting) | 95.65% | 219.8 |
| vote | Pypper:one-level@50 (voting) | 95.65% | 219.8 |
| vote | Pypper:single (best rule) | 96.09% | 4.8 |
| vote | Pypper:single (macro distribution voting) | 95.86% | 4.8 |
| vote | Pypper:single (micro distribution voting) | 95.18% | 4.8 |
| vote | Pypper:single (voting) | 95.86% | 4.8 |
| vote | Pypper:two-level@10 (best rule) | 95.42% | 39.8 |
| vote | Pypper:two-level@10 (macro distribution voting) | 95.42% | 39.8 |
| vote | Pypper:two-level@10 (micro distribution voting) | 95.65% | 39.8 |
| vote | Pypper:two-level@10 (voting) | 95.42% | 39.8 |
| vote | Pypper:two-level@25 (best rule) | 95.64% | 106.5 |
| vote | Pypper:two-level@25 (macro distribution voting) | 95.64% | 106.5 |
| vote | Pypper:two-level@25 (micro distribution voting) | 95.65% | 106.5 |
| vote | Pypper:two-level@25 (voting) | 95.64% | 106.5 |
| vote | Pypper:two-level@50 (best rule) | 95.64% | 219.8 |
| vote | Pypper:two-level@50 (macro distribution voting) | 95.64% | 219.8 |
| vote | Pypper:two-level@50 (micro distribution voting) | 95.65% | 219.8 |
| vote | Pypper:two-level@50 (voting) | 95.64% | 219.8 |
| vowel | PFoil:one-level@10 (best rule) | 73.64% | 1017.7 |
| vowel | PFoil:one-level@10 (macro distribution voting) | 88.69% | 1017.7 |
| vowel | PFoil:one-level@10 (micro distribution voting) | 82.02% | 1017.7 |
| vowel | PFoil:one-level@10 (voting) | 88.79% | 1017.7 |
| vowel | PFoil:one-level@25 (best rule) | 71.72% | 2556.4 |
| vowel | PFoil:one-level@25 (macro distribution voting) | 90.00% | 2556.4 |
| vowel | PFoil:one-level@25 (micro distribution voting) | 84.95% | 2556.4 |
| vowel | PFoil:one-level@25 (voting) | 90.71% | 2556.4 |
| vowel | PFoil:one-level@50 (best rule) | 68.79% | 5132.3 |
| vowel | PFoil:one-level@50 (macro distribution voting) | 89.90% | 5132.3 |
| vowel | PFoil:one-level@50 (micro distribution voting) | 85.96% | 5132.3 |
| vowel | PFoil:one-level@50 (voting) | 90.30% | 5132.3 |
| vowel | PFoil:single (best rule) | 81.82% | 115.8 |
| vowel | PFoil:single (macro distribution voting) | 82.73% | 115.8 |
| vowel | PFoil:single (micro distribution voting) | 73.54% | 115.8 |
| vowel | PFoil:single (voting) | 72.73% | 115.8 |
| vowel | PFoil:two-level@10 (best rule) | 87.98% | 1017.7 |
| vowel | PFoil:two-level@10 (macro distribution voting) | 87.68% | 1017.7 |
| vowel | PFoil:two-level@10 (micro distribution voting) | 87.17% | 1017.7 |
| vowel | PFoil:two-level@10 (voting) | 87.37% | 1017.7 |
| vowel | PFoil:two-level@25 (best rule) | 89.29% | 2556.4 |
| vowel | PFoil:two-level@25 (macro distribution voting) | 90.40% | 2556.4 |
| vowel | PFoil:two-level@25 (micro distribution voting) | 89.19% | 2556.4 |
| vowel | PFoil:two-level@25 (voting) | 89.70% | 2556.4 |
| vowel | PFoil:two-level@50 (best rule) | 90.10% | 5132.3 |
| vowel | PFoil:two-level@50 (macro distribution voting) | 90.71% | 5132.3 |
| vowel | PFoil:two-level@50 (micro distribution voting) | 89.70% | 5132.3 |
| vowel | PFoil:two-level@50 (voting) | 90.00% | 5132.3 |
| vowel | PFossil:one-level@10 (best rule) | 76.87% | 871.1 |
| vowel | PFossil:one-level@10 (macro distribution voting) | 87.37% | 871.1 |
| vowel | PFossil:one-level@10 (micro distribution voting) | 78.28% | 871.1 |
| vowel | PFossil:one-level@10 (voting) | 87.17% | 871.1 |
| vowel | PFossil:one-level@25 (best rule) | 73.94% | 2179.6 |
| vowel | PFossil:one-level@25 (macro distribution voting) | 88.59% | 2179.6 |
| vowel | PFossil:one-level@25 (micro distribution voting) | 80.00% | 2179.6 |
| vowel | PFossil:one-level@25 (voting) | 88.59% | 2179.6 |
| vowel | PFossil:one-level@50 (best rule) | 72.83% | 4356.5 |
| vowel | PFossil:one-level@50 (macro distribution voting) | 89.60% | 4356.5 |
| vowel | PFossil:one-level@50 (micro distribution voting) | 80.61% | 4356.5 |
| vowel | PFossil:one-level@50 (voting) | 88.99% | 4356.5 |
| vowel | PFossil:single (best rule) | 77.27% | 94.9 |
| vowel | PFossil:single (macro distribution voting) | 79.80% | 94.9 |
| vowel | PFossil:single (micro distribution voting) | 74.24% | 94.9 |
| vowel | PFossil:single (voting) | 78.89% | 94.9 |
| vowel | PFossil:two-level@10 (best rule) | 85.56% | 871.1 |
| vowel | PFossil:two-level@10 (macro distribution voting) | 87.37% | 871.1 |
| vowel | PFossil:two-level@10 (micro distribution voting) | 83.43% | 871.1 |
| vowel | PFossil:two-level@10 (voting) | 86.57% | 871.1 |
| vowel | PFossil:two-level@25 (best rule) | 87.88% | 2179.6 |
| vowel | PFossil:two-level@25 (macro distribution voting) | 89.29% | 2179.6 |
| vowel | PFossil:two-level@25 (micro distribution voting) | 84.55% | 2179.6 |
| vowel | PFossil:two-level@25 (voting) | 87.78% | 2179.6 |
| vowel | PFossil:two-level@50 (best rule) | 87.58% | 4356.5 |
| vowel | PFossil:two-level@50 (macro distribution voting) | 89.70% | 4356.5 |
| vowel | PFossil:two-level@50 (micro distribution voting) | 84.34% | 4356.5 |
| vowel | PFossil:two-level@50 (voting) | 88.18% | 4356.5 |
| vowel | Pypper:one-level@10 (best rule) | 74.55% | 541.2 |
| vowel | Pypper:one-level@10 (macro distribution voting) | 83.23% | 541.2 |
| vowel | Pypper:one-level@10 (micro distribution voting) | 74.85% | 541.2 |
| vowel | Pypper:one-level@10 (voting) | 82.42% | 541.2 |
| vowel | Pypper:one-level@25 (best rule) | 72.73% | 1366.3 |
| vowel | Pypper:one-level@25 (macro distribution voting) | 85.76% | 1366.3 |
| vowel | Pypper:one-level@25 (micro distribution voting) | 76.97% | 1366.3 |
| vowel | Pypper:one-level@25 (voting) | 84.14% | 1366.3 |
| vowel | Pypper:one-level@50 (best rule) | 70.40% | 2738.3 |
| vowel | Pypper:one-level@50 (macro distribution voting) | 85.56% | 2738.3 |
| vowel | Pypper:one-level@50 (micro distribution voting) | 77.78% | 2738.3 |
| vowel | Pypper:one-level@50 (voting) | 84.55% | 2738.3 |
| vowel | Pypper:single (best rule) | 73.84% | 68.9 |
| vowel | Pypper:single (macro distribution voting) | 74.85% | 68.9 |
| vowel | Pypper:single (micro distribution voting) | 72.02% | 68.9 |
| vowel | Pypper:single (voting) | 72.02% | 68.9 |
| vowel | Pypper:two-level@10 (best rule) | 81.52% | 541.2 |
| vowel | Pypper:two-level@10 (macro distribution voting) | 83.33% | 541.2 |
| vowel | Pypper:two-level@10 (micro distribution voting) | 79.70% | 541.2 |
| vowel | Pypper:two-level@10 (voting) | 80.71% | 541.2 |
| vowel | Pypper:two-level@25 (best rule) | 85.56% | 1366.3 |
| vowel | Pypper:two-level@25 (macro distribution voting) | 86.26% | 1366.3 |
| vowel | Pypper:two-level@25 (micro distribution voting) | 82.42% | 1366.3 |
| vowel | Pypper:two-level@25 (voting) | 84.44% | 1366.3 |
| vowel | Pypper:two-level@50 (best rule) | 84.95% | 2738.3 |
| vowel | Pypper:two-level@50 (macro distribution voting) | 86.57% | 2738.3 |
| vowel | Pypper:two-level@50 (micro distribution voting) | 81.92% | 2738.3 |
| vowel | Pypper:two-level@50 (voting) | 85.05% | 2738.3 |
| wdbc | PFoil:one-level@10 (best rule) | 93.33% | 124.4 |
| wdbc | PFoil:one-level@10 (macro distribution voting) | 96.66% | 124.4 |
| wdbc | PFoil:one-level@10 (micro distribution voting) | 96.48% | 124.4 |
| wdbc | PFoil:one-level@10 (voting) | 97.01% | 124.4 |
| wdbc | PFoil:one-level@25 (best rule) | 92.27% | 306.9 |
| wdbc | PFoil:one-level@25 (macro distribution voting) | 96.66% | 306.9 |
| wdbc | PFoil:one-level@25 (micro distribution voting) | 96.13% | 306.9 |
| wdbc | PFoil:one-level@25 (voting) | 97.19% | 306.9 |
| wdbc | PFoil:one-level@50 (best rule) | 91.57% | 605.0 |
| wdbc | PFoil:one-level@50 (macro distribution voting) | 96.48% | 605.0 |
| wdbc | PFoil:one-level@50 (micro distribution voting) | 96.13% | 605.0 |
| wdbc | PFoil:one-level@50 (voting) | 97.18% | 605.0 |
| wdbc | PFoil:single (best rule) | 94.38% | 15.4 |
| wdbc | PFoil:single (macro distribution voting) | 94.38% | 15.4 |
| wdbc | PFoil:single (micro distribution voting) | 93.67% | 15.4 |
| wdbc | PFoil:single (voting) | 93.85% | 15.4 |
| wdbc | PFoil:two-level@10 (best rule) | 95.96% | 124.4 |
| wdbc | PFoil:two-level@10 (macro distribution voting) | 96.48% | 124.4 |
| wdbc | PFoil:two-level@10 (micro distribution voting) | 95.60% | 124.4 |
| wdbc | PFoil:two-level@10 (voting) | 96.31% | 124.4 |
| wdbc | PFoil:two-level@25 (best rule) | 96.31% | 306.9 |
| wdbc | PFoil:two-level@25 (macro distribution voting) | 96.13% | 306.9 |
| wdbc | PFoil:two-level@25 (micro distribution voting) | 96.48% | 306.9 |
| wdbc | PFoil:two-level@25 (voting) | 95.78% | 306.9 |
| wdbc | PFoil:two-level@50 (best rule) | 96.13% | 605.0 |
| wdbc | PFoil:two-level@50 (macro distribution voting) | 96.31% | 605.0 |
| wdbc | PFoil:two-level@50 (micro distribution voting) | 96.13% | 605.0 |
| wdbc | PFoil:two-level@50 (voting) | 96.13% | 605.0 |
| wdbc | PFossil:one-level@10 (best rule) | 95.44% | 106.1 |
| wdbc | PFossil:one-level@10 (macro distribution voting) | 95.78% | 106.1 |
| wdbc | PFossil:one-level@10 (micro distribution voting) | 94.55% | 106.1 |
| wdbc | PFossil:one-level@10 (voting) | 95.60% | 106.1 |
| wdbc | PFossil:one-level@25 (best rule) | 94.91% | 262.7 |
| wdbc | PFossil:one-level@25 (macro distribution voting) | 95.78% | 262.7 |
| wdbc | PFossil:one-level@25 (micro distribution voting) | 95.08% | 262.7 |
| wdbc | PFossil:one-level@25 (voting) | 95.78% | 262.7 |
| wdbc | PFossil:one-level@50 (best rule) | 95.43% | 517.1 |
| wdbc | PFossil:one-level@50 (macro distribution voting) | 95.60% | 517.1 |
| wdbc | PFossil:one-level@50 (micro distribution voting) | 94.55% | 517.1 |
| wdbc | PFossil:one-level@50 (voting) | 95.60% | 517.1 |
| wdbc | PFossil:single (best rule) | 94.56% | 11.7 |
| wdbc | PFossil:single (macro distribution voting) | 94.91% | 11.7 |
| wdbc | PFossil:single (micro distribution voting) | 93.85% | 11.7 |
| wdbc | PFossil:single (voting) | 94.56% | 11.7 |
| wdbc | PFossil:two-level@10 (best rule) | 95.43% | 106.1 |
| wdbc | PFossil:two-level@10 (macro distribution voting) | 95.43% | 106.1 |
| wdbc | PFossil:two-level@10 (micro distribution voting) | 94.37% | 106.1 |
| wdbc | PFossil:two-level@10 (voting) | 95.08% | 106.1 |
| wdbc | PFossil:two-level@25 (best rule) | 95.60% | 262.7 |
| wdbc | PFossil:two-level@25 (macro distribution voting) | 95.43% | 262.7 |
| wdbc | PFossil:two-level@25 (micro distribution voting) | 94.55% | 262.7 |
| wdbc | PFossil:two-level@25 (voting) | 95.60% | 262.7 |
| wdbc | PFossil:two-level@50 (best rule) | 96.13% | 517.1 |
| wdbc | PFossil:two-level@50 (macro distribution voting) | 95.78% | 517.1 |
| wdbc | PFossil:two-level@50 (micro distribution voting) | 94.73% | 517.1 |
| wdbc | PFossil:two-level@50 (voting) | 95.61% | 517.1 |
| wdbc | Pypper:one-level@10 (best rule) | 95.43% | 65.3 |
| wdbc | Pypper:one-level@10 (macro distribution voting) | 95.96% | 65.3 |
| wdbc | Pypper:one-level@10 (micro distribution voting) | 94.56% | 65.3 |
| wdbc | Pypper:one-level@10 (voting) | 95.96% | 65.3 |
| wdbc | Pypper:one-level@25 (best rule) | 94.03% | 162.5 |
| wdbc | Pypper:one-level@25 (macro distribution voting) | 96.13% | 162.5 |
| wdbc | Pypper:one-level@25 (micro distribution voting) | 94.73% | 162.5 |
| wdbc | Pypper:one-level@25 (voting) | 96.13% | 162.5 |
| wdbc | Pypper:one-level@50 (best rule) | 95.25% | 322.1 |
| wdbc | Pypper:one-level@50 (macro distribution voting) | 95.96% | 322.1 |
| wdbc | Pypper:one-level@50 (micro distribution voting) | 94.90% | 322.1 |
| wdbc | Pypper:one-level@50 (voting) | 95.96% | 322.1 |
| wdbc | Pypper:single (best rule) | 95.44% | 7.2 |
| wdbc | Pypper:single (macro distribution voting) | 95.44% | 7.2 |
| wdbc | Pypper:single (micro distribution voting) | 95.43% | 7.2 |
| wdbc | Pypper:single (voting) | 95.08% | 7.2 |
| wdbc | Pypper:two-level@10 (best rule) | 95.60% | 65.3 |
| wdbc | Pypper:two-level@10 (macro distribution voting) | 95.60% | 65.3 |
| wdbc | Pypper:two-level@10 (micro distribution voting) | 95.08% | 65.3 |
| wdbc | Pypper:two-level@10 (voting) | 95.26% | 65.3 |
| wdbc | Pypper:two-level@25 (best rule) | 95.96% | 162.5 |
| wdbc | Pypper:two-level@25 (macro distribution voting) | 96.13% | 162.5 |
| wdbc | Pypper:two-level@25 (micro distribution voting) | 95.26% | 162.5 |
| wdbc | Pypper:two-level@25 (voting) | 95.08% | 162.5 |
| wdbc | Pypper:two-level@50 (best rule) | 96.13% | 322.1 |
| wdbc | Pypper:two-level@50 (macro distribution voting) | 96.31% | 322.1 |
| wdbc | Pypper:two-level@50 (micro distribution voting) | 95.43% | 322.1 |
| wdbc | Pypper:two-level@50 (voting) | 95.43% | 322.1 |
| wilt | PFoil:one-level@10 (best rule) | 96.73% | 889.6 |
| wilt | PFoil:one-level@10 (macro distribution voting) | 96.94% | 889.6 |
| wilt | PFoil:one-level@10 (micro distribution voting) | 95.31% | 889.6 |
| wilt | PFoil:one-level@10 (voting) | 94.88% | 889.6 |
| wilt | PFoil:one-level@25 (best rule) | 96.82% | 2198.1 |
| wilt | PFoil:one-level@25 (macro distribution voting) | 97.05% | 2198.1 |
| wilt | PFoil:one-level@25 (micro distribution voting) | 95.29% | 2198.1 |
| wilt | PFoil:one-level@25 (voting) | 94.90% | 2198.1 |
| wilt | PFoil:one-level@50 (best rule) | 96.80% | 4390.2 |
| wilt | PFoil:one-level@50 (macro distribution voting) | 97.00% | 4390.2 |
| wilt | PFoil:one-level@50 (micro distribution voting) | 95.16% | 4390.2 |
| wilt | PFoil:one-level@50 (voting) | 95.02% | 4390.2 |
| wilt | PFoil:single (best rule) | 96.55% | 101.2 |
| wilt | PFoil:single (macro distribution voting) | 96.98% | 101.2 |
| wilt | PFoil:single (micro distribution voting) | 95.85% | 101.2 |
| wilt | PFoil:single (voting) | 93.82% | 101.2 |
| wilt | PFoil:two-level@10 (best rule) | 96.88% | 889.6 |
| wilt | PFoil:two-level@10 (macro distribution voting) | 96.92% | 889.6 |
| wilt | PFoil:two-level@10 (micro distribution voting) | 96.32% | 889.6 |
| wilt | PFoil:two-level@10 (voting) | 96.01% | 889.6 |
| wilt | PFoil:two-level@25 (best rule) | 96.86% | 2198.1 |
| wilt | PFoil:two-level@25 (macro distribution voting) | 97.15% | 2198.1 |
| wilt | PFoil:two-level@25 (micro distribution voting) | 96.20% | 2198.1 |
| wilt | PFoil:two-level@25 (voting) | 95.93% | 2198.1 |
| wilt | PFoil:two-level@50 (best rule) | 96.92% | 4390.2 |
| wilt | PFoil:two-level@50 (macro distribution voting) | 97.17% | 4390.2 |
| wilt | PFoil:two-level@50 (micro distribution voting) | 96.28% | 4390.2 |
| wilt | PFoil:two-level@50 (voting) | 95.54% | 4390.2 |
| wilt | PFossil:one-level@10 (best rule) | 97.02% | 10.3 |
| wilt | PFossil:one-level@10 (macro distribution voting) | 97.02% | 10.3 |
| wilt | PFossil:one-level@10 (micro distribution voting) | 97.02% | 10.3 |
| wilt | PFossil:one-level@10 (voting) | 97.02% | 10.3 |
| wilt | PFossil:one-level@25 (best rule) | 96.92% | 25.5 |
| wilt | PFossil:one-level@25 (macro distribution voting) | 96.92% | 25.5 |
| wilt | PFossil:one-level@25 (micro distribution voting) | 96.92% | 25.5 |
| wilt | PFossil:one-level@25 (voting) | 96.92% | 25.5 |
| wilt | PFossil:one-level@50 (best rule) | 96.40% | 51.6 |
| wilt | PFossil:one-level@50 (macro distribution voting) | 96.40% | 51.6 |
| wilt | PFossil:one-level@50 (micro distribution voting) | 96.40% | 51.6 |
| wilt | PFossil:one-level@50 (voting) | 96.40% | 51.6 |
| wilt | PFossil:single (best rule) | 97.02% | 1.0 |
| wilt | PFossil:single (macro distribution voting) | 97.02% | 1.0 |
| wilt | PFossil:single (micro distribution voting) | 97.02% | 1.0 |
| wilt | PFossil:single (voting) | 97.02% | 1.0 |
| wilt | PFossil:two-level@10 (best rule) | 97.02% | 10.3 |
| wilt | PFossil:two-level@10 (macro distribution voting) | 97.02% | 10.3 |
| wilt | PFossil:two-level@10 (micro distribution voting) | 97.02% | 10.3 |
| wilt | PFossil:two-level@10 (voting) | 97.02% | 10.3 |
| wilt | PFossil:two-level@25 (best rule) | 96.92% | 25.5 |
| wilt | PFossil:two-level@25 (macro distribution voting) | 96.92% | 25.5 |
| wilt | PFossil:two-level@25 (micro distribution voting) | 96.92% | 25.5 |
| wilt | PFossil:two-level@25 (voting) | 96.92% | 25.5 |
| wilt | PFossil:two-level@50 (best rule) | 96.40% | 51.6 |
| wilt | PFossil:two-level@50 (macro distribution voting) | 96.40% | 51.6 |
| wilt | PFossil:two-level@50 (micro distribution voting) | 96.40% | 51.6 |
| wilt | PFossil:two-level@50 (voting) | 96.40% | 51.6 |
| wilt | Pypper:one-level@10 (best rule) | 97.23% | 63.1 |
| wilt | Pypper:one-level@10 (macro distribution voting) | 97.44% | 63.1 |
| wilt | Pypper:one-level@10 (micro distribution voting) | 97.31% | 63.1 |
| wilt | Pypper:one-level@10 (voting) | 97.38% | 63.1 |
| wilt | Pypper:one-level@25 (best rule) | 97.00% | 160.8 |
| wilt | Pypper:one-level@25 (macro distribution voting) | 97.42% | 160.8 |
| wilt | Pypper:one-level@25 (micro distribution voting) | 97.29% | 160.8 |
| wilt | Pypper:one-level@25 (voting) | 97.42% | 160.8 |
| wilt | Pypper:one-level@50 (best rule) | 96.98% | 326.6 |
| wilt | Pypper:one-level@50 (macro distribution voting) | 97.44% | 326.6 |
| wilt | Pypper:one-level@50 (micro distribution voting) | 97.33% | 326.6 |
| wilt | Pypper:one-level@50 (voting) | 97.42% | 326.6 |
| wilt | Pypper:single (best rule) | 97.36% | 6.5 |
| wilt | Pypper:single (macro distribution voting) | 97.38% | 6.5 |
| wilt | Pypper:single (micro distribution voting) | 97.36% | 6.5 |
| wilt | Pypper:single (voting) | 97.38% | 6.5 |
| wilt | Pypper:two-level@10 (best rule) | 97.48% | 63.1 |
| wilt | Pypper:two-level@10 (macro distribution voting) | 97.46% | 63.1 |
| wilt | Pypper:two-level@10 (micro distribution voting) | 97.48% | 63.1 |
| wilt | Pypper:two-level@10 (voting) | 97.48% | 63.1 |
| wilt | Pypper:two-level@25 (best rule) | 97.44% | 160.8 |
| wilt | Pypper:two-level@25 (macro distribution voting) | 97.44% | 160.8 |
| wilt | Pypper:two-level@25 (micro distribution voting) | 97.44% | 160.8 |
| wilt | Pypper:two-level@25 (voting) | 97.44% | 160.8 |
| wilt | Pypper:two-level@50 (best rule) | 97.44% | 326.6 |
| wilt | Pypper:two-level@50 (macro distribution voting) | 97.44% | 326.6 |
| wilt | Pypper:two-level@50 (micro distribution voting) | 97.44% | 326.6 |
| wilt | Pypper:two-level@50 (voting) | 97.44% | 326.6 |
| wine | PFoil:one-level@10 (best rule) | 92.65% | 74.5 |
| wine | PFoil:one-level@10 (macro distribution voting) | 96.63% | 74.5 |
| wine | PFoil:one-level@10 (micro distribution voting) | 96.05% | 74.5 |
| wine | PFoil:one-level@10 (voting) | 96.63% | 74.5 |
| wine | PFoil:one-level@25 (best rule) | 91.47% | 187.9 |
| wine | PFoil:one-level@25 (macro distribution voting) | 97.75% | 187.9 |
| wine | PFoil:one-level@25 (micro distribution voting) | 97.19% | 187.9 |
| wine | PFoil:one-level@25 (voting) | 96.63% | 187.9 |
| wine | PFoil:one-level@50 (best rule) | 89.22% | 377.7 |
| wine | PFoil:one-level@50 (macro distribution voting) | 97.78% | 377.7 |
| wine | PFoil:one-level@50 (micro distribution voting) | 97.75% | 377.7 |
| wine | PFoil:one-level@50 (voting) | 97.78% | 377.7 |
| wine | PFoil:single (best rule) | 90.98% | 8.6 |
| wine | PFoil:single (macro distribution voting) | 91.54% | 8.6 |
| wine | PFoil:single (micro distribution voting) | 90.98% | 8.6 |
| wine | PFoil:single (voting) | 90.42% | 8.6 |
| wine | PFoil:two-level@10 (best rule) | 95.46% | 74.5 |
| wine | PFoil:two-level@10 (macro distribution voting) | 96.60% | 74.5 |
| wine | PFoil:two-level@10 (micro distribution voting) | 96.05% | 74.5 |
| wine | PFoil:two-level@10 (voting) | 96.60% | 74.5 |
| wine | PFoil:two-level@25 (best rule) | 96.60% | 187.9 |
| wine | PFoil:two-level@25 (macro distribution voting) | 97.75% | 187.9 |
| wine | PFoil:two-level@25 (micro distribution voting) | 96.05% | 187.9 |
| wine | PFoil:two-level@25 (voting) | 97.16% | 187.9 |
| wine | PFoil:two-level@50 (best rule) | 96.05% | 377.7 |
| wine | PFoil:two-level@50 (macro distribution voting) | 96.60% | 377.7 |
| wine | PFoil:two-level@50 (micro distribution voting) | 96.08% | 377.7 |
| wine | PFoil:two-level@50 (voting) | 96.60% | 377.7 |
| wine | PFossil:one-level@10 (best rule) | 90.98% | 77.0 |
| wine | PFossil:one-level@10 (macro distribution voting) | 96.63% | 77.0 |
| wine | PFossil:one-level@10 (micro distribution voting) | 96.05% | 77.0 |
| wine | PFossil:one-level@10 (voting) | 96.63% | 77.0 |
| wine | PFossil:one-level@25 (best rule) | 89.25% | 197.4 |
| wine | PFossil:one-level@25 (macro distribution voting) | 97.75% | 197.4 |
| wine | PFossil:one-level@25 (micro distribution voting) | 97.19% | 197.4 |
| wine | PFossil:one-level@25 (voting) | 97.75% | 197.4 |
| wine | PFossil:one-level@50 (best rule) | 90.36% | 397.4 |
| wine | PFossil:one-level@50 (macro distribution voting) | 97.19% | 397.4 |
| wine | PFossil:one-level@50 (micro distribution voting) | 97.16% | 397.4 |
| wine | PFossil:one-level@50 (voting) | 97.19% | 397.4 |
| wine | PFossil:single (best rule) | 92.75% | 9.4 |
| wine | PFossil:single (macro distribution voting) | 93.82% | 9.4 |
| wine | PFossil:single (micro distribution voting) | 93.24% | 9.4 |
| wine | PFossil:single (voting) | 93.79% | 9.4 |
| wine | PFossil:two-level@10 (best rule) | 96.60% | 77.0 |
| wine | PFossil:two-level@10 (macro distribution voting) | 96.60% | 77.0 |
| wine | PFossil:two-level@10 (micro distribution voting) | 96.60% | 77.0 |
| wine | PFossil:two-level@10 (voting) | 96.05% | 77.0 |
| wine | PFossil:two-level@25 (best rule) | 96.05% | 197.4 |
| wine | PFossil:two-level@25 (macro distribution voting) | 97.75% | 197.4 |
| wine | PFossil:two-level@25 (micro distribution voting) | 96.60% | 197.4 |
| wine | PFossil:two-level@25 (voting) | 97.16% | 197.4 |
| wine | PFossil:two-level@50 (best rule) | 97.19% | 397.4 |
| wine | PFossil:two-level@50 (macro distribution voting) | 97.75% | 397.4 |
| wine | PFossil:two-level@50 (micro distribution voting) | 96.60% | 397.4 |
| wine | PFossil:two-level@50 (voting) | 97.16% | 397.4 |
| wine | Pypper:one-level@10 (best rule) | 91.47% | 52.3 |
| wine | Pypper:one-level@10 (macro distribution voting) | 97.75% | 52.3 |
| wine | Pypper:one-level@10 (micro distribution voting) | 96.60% | 52.3 |
| wine | Pypper:one-level@10 (voting) | 97.75% | 52.3 |
| wine | Pypper:one-level@25 (best rule) | 92.03% | 130.8 |
| wine | Pypper:one-level@25 (macro distribution voting) | 97.19% | 130.8 |
| wine | Pypper:one-level@25 (micro distribution voting) | 97.19% | 130.8 |
| wine | Pypper:one-level@25 (voting) | 97.19% | 130.8 |
| wine | Pypper:one-level@50 (best rule) | 90.36% | 262.2 |
| wine | Pypper:one-level@50 (macro distribution voting) | 97.75% | 262.2 |
| wine | Pypper:one-level@50 (micro distribution voting) | 96.60% | 262.2 |
| wine | Pypper:one-level@50 (voting) | 97.75% | 262.2 |
| wine | Pypper:single (best rule) | 92.09% | 5.3 |
| wine | Pypper:single (macro distribution voting) | 90.98% | 5.3 |
| wine | Pypper:single (micro distribution voting) | 90.92% | 5.3 |
| wine | Pypper:single (voting) | 88.69% | 5.3 |
| wine | Pypper:two-level@10 (best rule) | 96.60% | 52.3 |
| wine | Pypper:two-level@10 (macro distribution voting) | 97.16% | 52.3 |
| wine | Pypper:two-level@10 (micro distribution voting) | 95.42% | 52.3 |
| wine | Pypper:two-level@10 (voting) | 94.87% | 52.3 |
| wine | Pypper:two-level@25 (best rule) | 96.63% | 130.8 |
| wine | Pypper:two-level@25 (macro distribution voting) | 97.19% | 130.8 |
| wine | Pypper:two-level@25 (micro distribution voting) | 96.05% | 130.8 |
| wine | Pypper:two-level@25 (voting) | 96.60% | 130.8 |
| wine | Pypper:two-level@50 (best rule) | 96.63% | 262.2 |
| wine | Pypper:two-level@50 (macro distribution voting) | 97.19% | 262.2 |
| wine | Pypper:two-level@50 (micro distribution voting) | 96.05% | 262.2 |
| wine | Pypper:two-level@50 (voting) | 96.57% | 262.2 |
| yeast | PFoil:one-level@10 (best rule) | 58.16% | 2029.6 |
| yeast | PFoil:one-level@10 (macro distribution voting) | 59.03% | 2029.6 |
| yeast | PFoil:one-level@10 (micro distribution voting) | 49.47% | 2029.6 |
| yeast | PFoil:one-level@10 (voting) | 58.22% | 2029.6 |
| yeast | PFoil:one-level@25 (best rule) | 59.24% | 5079.1 |
| yeast | PFoil:one-level@25 (macro distribution voting) | 59.57% | 5079.1 |
| yeast | PFoil:one-level@25 (micro distribution voting) | 50.81% | 5079.1 |
| yeast | PFoil:one-level@25 (voting) | 57.82% | 5079.1 |
| yeast | PFoil:one-level@50 (best rule) | 60.04% | 10179.0 |
| yeast | PFoil:one-level@50 (macro distribution voting) | 59.64% | 10179.0 |
| yeast | PFoil:one-level@50 (micro distribution voting) | 51.22% | 10179.0 |
| yeast | PFoil:one-level@50 (voting) | 58.42% | 10179.0 |
| yeast | PFoil:single (best rule) | 57.41% | 216.0 |
| yeast | PFoil:single (macro distribution voting) | 56.88% | 216.0 |
| yeast | PFoil:single (micro distribution voting) | 47.64% | 216.0 |
| yeast | PFoil:single (voting) | 52.29% | 216.0 |
| yeast | PFoil:two-level@10 (best rule) | 59.77% | 2029.6 |
| yeast | PFoil:two-level@10 (macro distribution voting) | 59.43% | 2029.6 |
| yeast | PFoil:two-level@10 (micro distribution voting) | 55.80% | 2029.6 |
| yeast | PFoil:two-level@10 (voting) | 55.46% | 2029.6 |
| yeast | PFoil:two-level@25 (best rule) | 59.50% | 5079.1 |
| yeast | PFoil:two-level@25 (macro distribution voting) | 59.91% | 5079.1 |
| yeast | PFoil:two-level@25 (micro distribution voting) | 57.35% | 5079.1 |
| yeast | PFoil:two-level@25 (voting) | 56.40% | 5079.1 |
| yeast | PFoil:two-level@50 (best rule) | 60.11% | 10179.0 |
| yeast | PFoil:two-level@50 (macro distribution voting) | 59.77% | 10179.0 |
| yeast | PFoil:two-level@50 (micro distribution voting) | 56.74% | 10179.0 |
| yeast | PFoil:two-level@50 (voting) | 56.67% | 10179.0 |
| yeast | PFossil:one-level@10 (best rule) | 55.86% | 435.4 |
| yeast | PFossil:one-level@10 (macro distribution voting) | 60.31% | 435.4 |
| yeast | PFossil:one-level@10 (micro distribution voting) | 57.14% | 435.4 |
| yeast | PFossil:one-level@10 (voting) | 60.17% | 435.4 |
| yeast | PFossil:one-level@25 (best rule) | 54.24% | 1069.4 |
| yeast | PFossil:one-level@25 (macro distribution voting) | 60.64% | 1069.4 |
| yeast | PFossil:one-level@25 (micro distribution voting) | 57.21% | 1069.4 |
| yeast | PFossil:one-level@25 (voting) | 59.84% | 1069.4 |
| yeast | PFossil:one-level@50 (best rule) | 52.29% | 2135.2 |
| yeast | PFossil:one-level@50 (macro distribution voting) | 60.91% | 2135.2 |
| yeast | PFossil:one-level@50 (micro distribution voting) | 57.68% | 2135.2 |
| yeast | PFossil:one-level@50 (voting) | 60.51% | 2135.2 |
| yeast | PFossil:single (best rule) | 57.48% | 31.2 |
| yeast | PFossil:single (macro distribution voting) | 57.54% | 31.2 |
| yeast | PFossil:single (micro distribution voting) | 55.52% | 31.2 |
| yeast | PFossil:single (voting) | 54.31% | 31.2 |
| yeast | PFossil:two-level@10 (best rule) | 60.31% | 435.4 |
| yeast | PFossil:two-level@10 (macro distribution voting) | 60.24% | 435.4 |
| yeast | PFossil:two-level@10 (micro distribution voting) | 58.22% | 435.4 |
| yeast | PFossil:two-level@10 (voting) | 57.61% | 435.4 |
| yeast | PFossil:two-level@25 (best rule) | 60.58% | 1069.4 |
| yeast | PFossil:two-level@25 (macro distribution voting) | 60.44% | 1069.4 |
| yeast | PFossil:two-level@25 (micro distribution voting) | 58.09% | 1069.4 |
| yeast | PFossil:two-level@25 (voting) | 57.82% | 1069.4 |
| yeast | PFossil:two-level@50 (best rule) | 60.38% | 2135.2 |
| yeast | PFossil:two-level@50 (macro distribution voting) | 60.37% | 2135.2 |
| yeast | PFossil:two-level@50 (micro distribution voting) | 58.09% | 2135.2 |
| yeast | PFossil:two-level@50 (voting) | 57.68% | 2135.2 |
| yeast | Pypper:one-level@10 (best rule) | 58.69% | 236.6 |
| yeast | Pypper:one-level@10 (macro distribution voting) | 61.66% | 236.6 |
| yeast | Pypper:one-level@10 (micro distribution voting) | 58.90% | 236.6 |
| yeast | Pypper:one-level@10 (voting) | 60.85% | 236.6 |
| yeast | Pypper:one-level@25 (best rule) | 59.77% | 583.5 |
| yeast | Pypper:one-level@25 (macro distribution voting) | 61.86% | 583.5 |
| yeast | Pypper:one-level@25 (micro distribution voting) | 58.76% | 583.5 |
| yeast | Pypper:one-level@25 (voting) | 61.45% | 583.5 |
| yeast | Pypper:one-level@50 (best rule) | 58.90% | 1153.7 |
| yeast | Pypper:one-level@50 (macro distribution voting) | 61.32% | 1153.7 |
| yeast | Pypper:one-level@50 (micro distribution voting) | 59.70% | 1153.7 |
| yeast | Pypper:one-level@50 (voting) | 61.12% | 1153.7 |
| yeast | Pypper:single (best rule) | 59.43% | 18.5 |
| yeast | Pypper:single (macro distribution voting) | 59.02% | 18.5 |
| yeast | Pypper:single (micro distribution voting) | 57.88% | 18.5 |
| yeast | Pypper:single (voting) | 58.15% | 18.5 |
| yeast | Pypper:two-level@10 (best rule) | 62.00% | 236.6 |
| yeast | Pypper:two-level@10 (macro distribution voting) | 61.46% | 236.6 |
| yeast | Pypper:two-level@10 (micro distribution voting) | 60.78% | 236.6 |
| yeast | Pypper:two-level@10 (voting) | 60.51% | 236.6 |
| yeast | Pypper:two-level@25 (best rule) | 61.79% | 583.5 |
| yeast | Pypper:two-level@25 (macro distribution voting) | 61.93% | 583.5 |
| yeast | Pypper:two-level@25 (micro distribution voting) | 60.58% | 583.5 |
| yeast | Pypper:two-level@25 (voting) | 60.18% | 583.5 |
| yeast | Pypper:two-level@50 (best rule) | 61.99% | 1153.7 |
| yeast | Pypper:two-level@50 (macro distribution voting) | 61.73% | 1153.7 |
| yeast | Pypper:two-level@50 (micro distribution voting) | 60.51% | 1153.7 |
| yeast | Pypper:two-level@50 (voting) | 60.38% | 1153.7 |
| zoo | PFoil:one-level@10 (best rule) | 93.00% | 85.5 |
| zoo | PFoil:one-level@10 (macro distribution voting) | 95.00% | 85.5 |
| zoo | PFoil:one-level@10 (micro distribution voting) | 94.00% | 85.5 |
| zoo | PFoil:one-level@10 (voting) | 95.00% | 85.5 |
| zoo | PFoil:one-level@25 (best rule) | 93.09% | 216.2 |
| zoo | PFoil:one-level@25 (macro distribution voting) | 97.00% | 216.2 |
| zoo | PFoil:one-level@25 (micro distribution voting) | 96.00% | 216.2 |
| zoo | PFoil:one-level@25 (voting) | 97.00% | 216.2 |
| zoo | PFoil:one-level@50 (best rule) | 94.09% | 424.6 |
| zoo | PFoil:one-level@50 (macro distribution voting) | 97.00% | 424.6 |
| zoo | PFoil:one-level@50 (micro distribution voting) | 96.00% | 424.6 |
| zoo | PFoil:one-level@50 (voting) | 96.00% | 424.6 |
| zoo | PFoil:single (best rule) | 96.00% | 9.5 |
| zoo | PFoil:single (macro distribution voting) | 95.00% | 9.5 |
| zoo | PFoil:single (micro distribution voting) | 95.00% | 9.5 |
| zoo | PFoil:single (voting) | 95.00% | 9.5 |
| zoo | PFoil:two-level@10 (best rule) | 95.00% | 85.5 |
| zoo | PFoil:two-level@10 (macro distribution voting) | 95.00% | 85.5 |
| zoo | PFoil:two-level@10 (micro distribution voting) | 95.00% | 85.5 |
| zoo | PFoil:two-level@10 (voting) | 94.00% | 85.5 |
| zoo | PFoil:two-level@25 (best rule) | 97.00% | 216.2 |
| zoo | PFoil:two-level@25 (macro distribution voting) | 96.00% | 216.2 |
| zoo | PFoil:two-level@25 (micro distribution voting) | 97.00% | 216.2 |
| zoo | PFoil:two-level@25 (voting) | 96.00% | 216.2 |
| zoo | PFoil:two-level@50 (best rule) | 98.00% | 424.6 |
| zoo | PFoil:two-level@50 (macro distribution voting) | 98.00% | 424.6 |
| zoo | PFoil:two-level@50 (micro distribution voting) | 98.00% | 424.6 |
| zoo | PFoil:two-level@50 (voting) | 97.00% | 424.6 |
| zoo | PFossil:one-level@10 (best rule) | 93.00% | 83.6 |
| zoo | PFossil:one-level@10 (macro distribution voting) | 95.00% | 83.6 |
| zoo | PFossil:one-level@10 (micro distribution voting) | 95.00% | 83.6 |
| zoo | PFossil:one-level@10 (voting) | 95.00% | 83.6 |
| zoo | PFossil:one-level@25 (best rule) | 93.09% | 212.4 |
| zoo | PFossil:one-level@25 (macro distribution voting) | 97.00% | 212.4 |
| zoo | PFossil:one-level@25 (micro distribution voting) | 95.00% | 212.4 |
| zoo | PFossil:one-level@25 (voting) | 98.00% | 212.4 |
| zoo | PFossil:one-level@50 (best rule) | 94.09% | 420.1 |
| zoo | PFossil:one-level@50 (macro distribution voting) | 98.00% | 420.1 |
| zoo | PFossil:one-level@50 (micro distribution voting) | 96.00% | 420.1 |
| zoo | PFossil:one-level@50 (voting) | 98.00% | 420.1 |
| zoo | PFossil:single (best rule) | 95.00% | 8.9 |
| zoo | PFossil:single (macro distribution voting) | 92.00% | 8.9 |
| zoo | PFossil:single (micro distribution voting) | 94.00% | 8.9 |
| zoo | PFossil:single (voting) | 93.00% | 8.9 |
| zoo | PFossil:two-level@10 (best rule) | 95.00% | 83.6 |
| zoo | PFossil:two-level@10 (macro distribution voting) | 95.00% | 83.6 |
| zoo | PFossil:two-level@10 (micro distribution voting) | 94.00% | 83.6 |
| zoo | PFossil:two-level@10 (voting) | 94.00% | 83.6 |
| zoo | PFossil:two-level@25 (best rule) | 98.00% | 212.4 |
| zoo | PFossil:two-level@25 (macro distribution voting) | 96.00% | 212.4 |
| zoo | PFossil:two-level@25 (micro distribution voting) | 96.00% | 212.4 |
| zoo | PFossil:two-level@25 (voting) | 96.00% | 212.4 |
| zoo | PFossil:two-level@50 (best rule) | 98.00% | 420.1 |
| zoo | PFossil:two-level@50 (macro distribution voting) | 97.00% | 420.1 |
| zoo | PFossil:two-level@50 (micro distribution voting) | 97.00% | 420.1 |
| zoo | PFossil:two-level@50 (voting) | 96.00% | 420.1 |
| zoo | Pypper:one-level@10 (best rule) | 91.00% | 62.1 |
| zoo | Pypper:one-level@10 (macro distribution voting) | 93.00% | 62.1 |
| zoo | Pypper:one-level@10 (micro distribution voting) | 92.00% | 62.1 |
| zoo | Pypper:one-level@10 (voting) | 92.00% | 62.1 |
| zoo | Pypper:one-level@25 (best rule) | 88.09% | 154.2 |
| zoo | Pypper:one-level@25 (macro distribution voting) | 93.00% | 154.2 |
| zoo | Pypper:one-level@25 (micro distribution voting) | 91.00% | 154.2 |
| zoo | Pypper:one-level@25 (voting) | 93.00% | 154.2 |
| zoo | Pypper:one-level@50 (best rule) | 89.09% | 300.4 |
| zoo | Pypper:one-level@50 (macro distribution voting) | 94.00% | 300.4 |
| zoo | Pypper:one-level@50 (micro distribution voting) | 93.00% | 300.4 |
| zoo | Pypper:one-level@50 (voting) | 94.00% | 300.4 |
| zoo | Pypper:single (best rule) | 90.09% | 6.1 |
| zoo | Pypper:single (macro distribution voting) | 90.09% | 6.1 |
| zoo | Pypper:single (micro distribution voting) | 91.09% | 6.1 |
| zoo | Pypper:single (voting) | 90.09% | 6.1 |
| zoo | Pypper:two-level@10 (best rule) | 92.00% | 62.1 |
| zoo | Pypper:two-level@10 (macro distribution voting) | 92.00% | 62.1 |
| zoo | Pypper:two-level@10 (micro distribution voting) | 91.00% | 62.1 |
| zoo | Pypper:two-level@10 (voting) | 91.00% | 62.1 |
| zoo | Pypper:two-level@25 (best rule) | 92.00% | 154.2 |
| zoo | Pypper:two-level@25 (macro distribution voting) | 93.00% | 154.2 |
| zoo | Pypper:two-level@25 (micro distribution voting) | 92.00% | 154.2 |
| zoo | Pypper:two-level@25 (voting) | 92.00% | 154.2 |
| zoo | Pypper:two-level@50 (best rule) | 94.00% | 300.4 |
| zoo | Pypper:two-level@50 (macro distribution voting) | 94.00% | 300.4 |
| zoo | Pypper:two-level@50 (micro distribution voting) | 93.00% | 300.4 |
| zoo | Pypper:two-level@50 (voting) | 93.00% | 300.4 |

