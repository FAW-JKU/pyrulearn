# Refinement heuristics: static and adaptive, regular and inverted

## Setup

**Full run** (69 datasets, 10-fold).

The covering learner of the
[heuristic comparison](heuristic_comparison_report.md) (after Janssen &
Fürnkranz, 2010), on its datasets: 69 datasets of the catalog (44 binary, 25 multi-class), those of the [separate-and-conquer comparison](seco_learners_comparison_report.md), not the paper's: greedy top-down refinement until no refinement is left
(`BeamSearch(beam_width=1)`, in section 4 also width 5), the
best rule on the refinement path by a separate selection heuristic, the
m-estimate (`SingleRuleLearner(h, selection=MEstimate(m_s))`), no
stopping criterion and no pruning; covering stops once a new rule covers
no more positives than negatives; ordered class binarization. Numeric
attributes are discretized per training fold
(`TreeDiscretizer(max_intervals=8)`). Each fit is capped at 600s (3600s for `letter`), and a fold
that times out fails the learner's whole cross-validation on that dataset.

A heuristic `h` refines in one of four frames: **static** (`h`, judged
from the origin of the coverage space of the whole data), **adaptive**
(`Adaptive(h)`: among the examples of the rule being refined, its class
distribution as the prior), **static inverted** (`Inverted(h)`:
`h(N - n, P - p)`, judged by the examples a rule excludes, from the upper
right corner of the whole coverage space), and **adaptive inverted**
(`Adaptive(Inverted(h))`: `h(N' - n, P' - p)` among the parent's `P'`
positives and `N'` negatives -- the inverted heuristics of Stecher,
Janssen & Fürnkranz, ECML PKDD 2014). Scores relative to the parent have
no common scale across parents: the selection, always static, picks the
rule that is returned.

Measures: test accuracy, number of rules and conditions; means across
every dataset and fold.

## m-estimates for refinement and for selection

`MEstimate(m_r)` refines and `MEstimate(m_s)` selects the rule on the refinement path, for every pair of m in {0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32}, in each of the four frames; the diagonal of the regular (static) matrix is the plain m-estimate. Rows: m for refinement, columns: m for selection.

![the four matrices, accuracy](refinement_heuristics_plots/matrices_accuracy.png)

![the four matrices, conditions per rule](refinement_heuristics_plots/matrices_conds_per_rule.png)

### Accuracy

Regular m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 81.12% | 81.08% | 81.14% | 80.79% | 80.00% | 79.93% | 79.52% | 78.44% | 77.70% |
| m=0.2 | 81.14% | 81.13% | 81.19% | 80.79% | 80.10% | 79.97% | 79.52% | 78.43% | 77.69% |
| m=0.5 | 81.24% | 81.25% | 81.28% | 80.93% | 80.11% | 80.00% | 79.62% | 78.52% | 77.72% |
| m=1 | 81.45% | 81.45% | 81.49% | 81.50% | 80.59% | 80.44% | 79.87% | 78.70% | 77.85% |
| m=2 | 81.85% | 81.85% | 81.88% | 81.91% | 81.96% | 81.39% | 80.26% | 79.18% | 78.29% |
| m=4 | 82.28% | 82.27% | 82.30% | 82.31% | 82.29% | 82.27% | 81.17% | 80.07% | 78.88% |
| m=8 | 82.37% | 82.38% | 82.39% | 82.44% | 82.30% | 82.68% | 82.66% | 81.57% | 80.20% |
| m=16 | 82.71% | 82.71% | 82.68% | 82.66% | 82.59% | 82.84% | 82.97% | 82.96% | 81.86% |
| m=32 | 82.62% | 82.62% | 82.59% | 82.69% | 82.49% | 82.80% | 82.91% | 83.01% | 82.94% |

Adaptive inverted m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 82.48% | 82.49% | 82.49% | 82.52% | 82.63% | 82.86% | 82.81% | 82.72% | 82.30% |
| m=0.2 | 82.48% | 82.49% | 82.51% | 82.53% | 82.65% | 82.91% | 82.79% | 82.73% | 82.33% |
| m=0.5 | 82.31% | 82.33% | 82.34% | 82.39% | 82.56% | 82.78% | 82.90% | 82.76% | 82.38% |
| m=1 | 82.42% | 82.42% | 82.48% | 82.51% | 82.63% | 82.77% | 82.70% | 82.57% | 82.43% |
| m=2 | 82.41% | 82.41% | 82.44% | 82.46% | 82.54% | 82.64% | 82.66% | 82.71% | 82.46% |
| m=4 | 82.57% | 82.57% | 82.58% | 82.66% | 82.68% | 82.89% | 82.87% | 82.82% | 82.56% |
| m=8 | 82.72% | 82.73% | 82.72% | 82.78% | 82.81% | 82.82% | 82.92% | 82.87% | 82.64% |
| m=16 | 82.86% | 82.87% | 82.89% | 82.89% | 82.91% | 83.16% | 83.33% | 83.32% | 82.99% |
| m=32 | 82.87% | 82.88% | 82.87% | 82.92% | 82.97% | 83.25% | 83.37% | 83.17% | 82.83% |

Adaptive m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 81.16% | 81.12% | 81.14% | 80.76% | 80.01% | 79.92% | 79.50% | 78.43% | 77.69% |
| m=0.2 | 81.19% | 81.18% | 81.14% | 80.77% | 80.08% | 79.97% | 79.52% | 78.44% | 77.70% |
| m=0.5 | 81.20% | 81.22% | 81.33% | 80.97% | 80.16% | 80.06% | 79.58% | 78.51% | 77.71% |
| m=1 | 81.17% | 81.22% | 81.36% | 81.27% | 80.46% | 80.27% | 79.79% | 78.66% | 77.82% |
| m=2 | 81.52% | 81.52% | 81.55% | 81.70% | 81.77% | 80.99% | 80.09% | 79.04% | 78.17% |
| m=4 | 81.99% | 82.00% | 81.96% | 82.04% | 82.19% | 82.29% | 80.83% | 79.67% | 78.70% |
| m=8 | 82.31% | 82.31% | 82.37% | 82.42% | 82.46% | 82.78% | 82.30% | 81.09% | 79.79% |
| m=16 | 82.34% | 82.34% | 82.32% | 82.40% | 82.40% | 82.55% | 82.53% | 82.31% | 81.23% |
| m=32 | 82.47% | 82.48% | 82.47% | 82.43% | 82.60% | 82.91% | 83.13% | 82.72% | 82.54% |

Static inverted m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 81.76% | 81.75% | 81.84% | 81.92% | 81.98% | 82.14% | 82.24% | 82.28% | 81.99% |
| m=0.2 | 81.76% | 81.76% | 81.84% | 81.92% | 81.98% | 82.14% | 82.24% | 82.28% | 81.99% |
| m=0.5 | 81.75% | 81.74% | 81.82% | 81.91% | 81.95% | 82.13% | 82.22% | 82.29% | 82.03% |
| m=1 | 81.79% | 81.78% | 81.88% | 81.95% | 81.98% | 82.16% | 82.28% | 82.27% | 82.01% |
| m=2 | 81.69% | 81.67% | 81.78% | 81.88% | 81.98% | 82.17% | 82.33% | 82.28% | 82.00% |
| m=4 | 81.68% | 81.67% | 81.78% | 81.92% | 81.97% | 82.21% | 82.20% | 82.19% | 82.00% |
| m=8 | 81.78% | 81.77% | 81.83% | 81.98% | 81.95% | 82.21% | 82.33% | 82.20% | 81.94% |
| m=16 | 81.62% | 81.63% | 81.72% | 81.77% | 81.93% | 82.35% | 82.33% | 82.25% | 82.01% |
| m=32 | 81.69% | 81.68% | 81.76% | 81.72% | 82.00% | 82.20% | 82.13% | 82.25% | 82.02% |

### Number of rules

Regular m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 139.5 | 135.9 | 120.9 | 104.5 | 48.9 | 24.0 | 14.8 | 10.2 | 7.2 |
| m=0.2 | 137.8 | 137.7 | 119.9 | 104.1 | 48.9 | 24.4 | 14.9 | 10.2 | 7.2 |
| m=0.5 | 129.0 | 129.0 | 128.7 | 105.0 | 52.2 | 25.5 | 15.1 | 10.5 | 7.3 |
| m=1 | 117.4 | 117.3 | 116.8 | 116.2 | 53.3 | 27.2 | 15.9 | 10.8 | 7.5 |
| m=2 | 101.5 | 101.3 | 100.9 | 99.8 | 99.3 | 34.0 | 18.6 | 11.8 | 8.0 |
| m=4 | 86.6 | 86.5 | 85.7 | 84.6 | 82.9 | 78.5 | 26.7 | 14.5 | 9.0 |
| m=8 | 74.7 | 74.6 | 73.8 | 72.6 | 70.7 | 64.6 | 60.4 | 21.4 | 11.5 |
| m=16 | 66.5 | 66.3 | 65.5 | 64.0 | 61.5 | 54.6 | 47.9 | 42.6 | 18.2 |
| m=32 | 61.2 | 61.1 | 60.3 | 58.6 | 55.4 | 47.9 | 40.4 | 33.5 | 29.4 |

Adaptive inverted m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 53.2 | 53.0 | 52.3 | 50.5 | 47.0 | 38.2 | 28.9 | 19.6 | 13.3 |
| m=0.2 | 53.3 | 53.2 | 52.4 | 50.6 | 46.9 | 38.3 | 28.8 | 19.5 | 13.2 |
| m=0.5 | 53.2 | 53.1 | 52.3 | 50.5 | 46.8 | 38.3 | 28.6 | 19.7 | 13.2 |
| m=1 | 53.1 | 53.1 | 52.2 | 50.7 | 46.9 | 38.4 | 28.5 | 19.6 | 13.1 |
| m=2 | 52.9 | 52.8 | 52.0 | 50.3 | 46.8 | 38.4 | 28.5 | 19.4 | 13.1 |
| m=4 | 53.1 | 53.0 | 52.4 | 50.7 | 47.2 | 38.5 | 28.5 | 19.5 | 13.2 |
| m=8 | 53.8 | 53.6 | 52.8 | 51.1 | 47.6 | 38.4 | 28.6 | 19.2 | 13.2 |
| m=16 | 54.7 | 54.5 | 53.6 | 52.2 | 48.1 | 38.7 | 28.3 | 18.9 | 13.0 |
| m=32 | 55.5 | 55.3 | 54.6 | 53.0 | 49.1 | 39.0 | 28.7 | 19.1 | 12.8 |

Adaptive m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 139.6 | 134.8 | 121.3 | 104.8 | 48.7 | 24.0 | 14.7 | 10.2 | 7.2 |
| m=0.2 | 139.3 | 136.4 | 121.6 | 103.7 | 49.1 | 24.2 | 14.8 | 10.2 | 7.2 |
| m=0.5 | 135.8 | 134.6 | 122.9 | 103.9 | 50.4 | 25.1 | 15.1 | 10.4 | 7.3 |
| m=1 | 129.8 | 129.0 | 122.1 | 107.2 | 52.1 | 26.1 | 15.7 | 10.7 | 7.5 |
| m=2 | 119.0 | 118.7 | 115.5 | 105.8 | 61.2 | 29.0 | 17.0 | 11.3 | 7.9 |
| m=4 | 107.2 | 107.1 | 105.9 | 100.6 | 74.7 | 41.1 | 20.7 | 13.0 | 8.6 |
| m=8 | 95.6 | 95.5 | 94.6 | 92.0 | 81.1 | 51.6 | 30.1 | 15.9 | 9.9 |
| m=16 | 85.6 | 85.6 | 84.8 | 82.9 | 76.7 | 56.2 | 37.5 | 22.3 | 12.6 |
| m=32 | 78.0 | 77.9 | 77.1 | 75.5 | 70.7 | 55.3 | 39.5 | 25.6 | 16.3 |

Static inverted m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 64.6 | 64.5 | 63.0 | 60.3 | 54.4 | 42.5 | 28.2 | 17.7 | 11.6 |
| m=0.2 | 64.6 | 64.5 | 63.0 | 60.3 | 54.4 | 42.5 | 28.2 | 17.7 | 11.6 |
| m=0.5 | 64.6 | 64.5 | 63.1 | 60.3 | 54.4 | 42.6 | 28.3 | 17.7 | 11.6 |
| m=1 | 64.5 | 64.5 | 63.1 | 60.3 | 54.4 | 42.5 | 28.3 | 17.7 | 11.7 |
| m=2 | 64.5 | 64.5 | 63.0 | 60.4 | 54.5 | 42.5 | 28.4 | 17.7 | 11.7 |
| m=4 | 64.6 | 64.4 | 63.0 | 60.4 | 54.4 | 42.5 | 28.1 | 17.7 | 11.7 |
| m=8 | 64.5 | 64.4 | 63.0 | 60.4 | 54.3 | 42.2 | 28.2 | 17.8 | 11.6 |
| m=16 | 64.5 | 64.2 | 63.0 | 60.4 | 54.5 | 42.4 | 28.2 | 17.7 | 11.6 |
| m=32 | 64.3 | 64.1 | 62.5 | 60.4 | 54.2 | 42.5 | 28.2 | 17.7 | 11.6 |

### Conditions per rule

Regular m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 3.2 | 3.1 | 3.1 | 3.0 | 2.6 | 2.1 | 1.9 | 1.7 | 1.6 |
| m=0.2 | 3.2 | 3.2 | 3.1 | 3.0 | 2.6 | 2.2 | 1.9 | 1.7 | 1.6 |
| m=0.5 | 3.2 | 3.2 | 3.2 | 3.0 | 2.7 | 2.2 | 1.9 | 1.8 | 1.7 |
| m=1 | 3.3 | 3.3 | 3.3 | 3.2 | 2.8 | 2.4 | 2.0 | 1.9 | 1.7 |
| m=2 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | 2.8 | 2.4 | 2.2 | 1.9 |
| m=4 | 3.7 | 3.7 | 3.7 | 3.7 | 3.7 | 3.5 | 2.9 | 2.5 | 2.2 |
| m=8 | 4.5 | 4.5 | 4.5 | 4.5 | 4.4 | 4.2 | 3.8 | 3.2 | 2.6 |
| m=16 | 6.3 | 6.3 | 6.3 | 6.2 | 6.2 | 5.8 | 5.2 | 4.1 | 3.5 |
| m=32 | 9.2 | 9.2 | 9.2 | 9.2 | 9.2 | 9.0 | 8.1 | 6.5 | 4.7 |

Adaptive inverted m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 15.5 | 15.5 | 15.5 | 15.5 | 15.5 | 15.5 | 15.1 | 14.4 | 13.5 |
| m=0.2 | 15.5 | 15.5 | 15.4 | 15.4 | 15.4 | 15.4 | 15.0 | 14.3 | 13.4 |
| m=0.5 | 15.1 | 15.1 | 15.1 | 15.1 | 15.1 | 15.1 | 14.6 | 14.0 | 13.0 |
| m=1 | 14.6 | 14.6 | 14.5 | 14.5 | 14.5 | 14.5 | 14.0 | 13.3 | 12.5 |
| m=2 | 13.5 | 13.5 | 13.4 | 13.4 | 13.4 | 13.4 | 12.9 | 12.3 | 11.5 |
| m=4 | 12.1 | 12.1 | 12.1 | 12.1 | 12.1 | 11.9 | 11.5 | 10.9 | 10.2 |
| m=8 | 10.6 | 10.6 | 10.6 | 10.6 | 10.6 | 10.5 | 10.1 | 9.6 | 8.9 |
| m=16 | 9.3 | 9.3 | 9.3 | 9.3 | 9.3 | 9.1 | 8.8 | 8.2 | 7.6 |
| m=32 | 8.2 | 8.2 | 8.2 | 8.2 | 8.1 | 7.9 | 7.6 | 7.1 | 6.6 |

Adaptive m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 3.1 | 3.1 | 3.1 | 3.0 | 2.6 | 2.1 | 1.9 | 1.7 | 1.6 |
| m=0.2 | 3.2 | 3.1 | 3.1 | 3.0 | 2.6 | 2.1 | 1.9 | 1.7 | 1.6 |
| m=0.5 | 3.2 | 3.2 | 3.1 | 3.0 | 2.6 | 2.2 | 1.9 | 1.8 | 1.6 |
| m=1 | 3.2 | 3.2 | 3.1 | 3.0 | 2.7 | 2.3 | 2.0 | 1.8 | 1.7 |
| m=2 | 3.3 | 3.3 | 3.2 | 3.1 | 2.9 | 2.5 | 2.2 | 2.0 | 1.8 |
| m=4 | 3.3 | 3.3 | 3.3 | 3.3 | 3.1 | 2.8 | 2.4 | 2.2 | 2.0 |
| m=8 | 3.5 | 3.5 | 3.5 | 3.4 | 3.3 | 3.1 | 2.8 | 2.5 | 2.2 |
| m=16 | 3.6 | 3.6 | 3.6 | 3.6 | 3.5 | 3.3 | 3.1 | 2.8 | 2.5 |
| m=32 | 3.8 | 3.8 | 3.8 | 3.8 | 3.7 | 3.6 | 3.4 | 3.1 | 2.8 |

Static inverted m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 28.9 | 28.9 | 28.9 | 28.9 | 28.9 | 28.7 | 27.3 | 25.2 | 22.8 |
| m=0.2 | 28.9 | 28.9 | 28.9 | 28.9 | 28.9 | 28.7 | 27.3 | 25.3 | 22.8 |
| m=0.5 | 28.9 | 28.9 | 28.9 | 28.9 | 28.9 | 28.7 | 27.3 | 25.2 | 22.8 |
| m=1 | 28.9 | 28.9 | 28.9 | 28.9 | 28.9 | 28.7 | 27.3 | 25.2 | 22.8 |
| m=2 | 28.8 | 28.8 | 28.9 | 28.8 | 28.9 | 28.7 | 27.2 | 25.2 | 22.7 |
| m=4 | 28.8 | 28.8 | 28.9 | 28.8 | 28.8 | 28.6 | 27.2 | 25.3 | 22.7 |
| m=8 | 28.8 | 28.8 | 28.9 | 28.8 | 28.8 | 28.5 | 27.3 | 25.2 | 22.6 |
| m=16 | 28.6 | 28.7 | 28.7 | 28.6 | 28.7 | 28.5 | 27.1 | 25.1 | 22.5 |
| m=32 | 28.5 | 28.5 | 28.6 | 28.5 | 28.5 | 28.1 | 26.8 | 24.8 | 22.2 |

### Conditions in the rule set

Regular m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 590.5 | 578.4 | 503.3 | 425.1 | 164.2 | 67.5 | 36.0 | 22.3 | 14.1 |
| m=0.2 | 586.9 | 586.3 | 497.8 | 423.1 | 162.9 | 69.5 | 36.4 | 22.5 | 14.2 |
| m=0.5 | 554.4 | 554.0 | 552.5 | 431.8 | 182.0 | 74.8 | 37.9 | 23.5 | 14.5 |
| m=1 | 509.6 | 508.7 | 506.2 | 502.1 | 193.3 | 84.9 | 42.0 | 25.2 | 15.5 |
| m=2 | 456.3 | 455.6 | 453.0 | 446.2 | 441.7 | 122.2 | 56.1 | 30.5 | 17.7 |
| m=4 | 417.1 | 416.4 | 411.6 | 404.1 | 391.2 | 358.9 | 96.7 | 42.7 | 22.2 |
| m=8 | 419.2 | 418.5 | 413.1 | 403.5 | 385.9 | 336.2 | 290.8 | 79.8 | 34.0 |
| m=16 | 506.3 | 505.5 | 498.1 | 483.3 | 458.2 | 383.1 | 294.7 | 218.4 | 73.1 |
| m=32 | 700.7 | 700.3 | 693.3 | 675.6 | 635.1 | 532.0 | 399.3 | 256.1 | 162.6 |

Adaptive inverted m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 1269.2 | 1266.4 | 1255.6 | 1219.1 | 1149.1 | 958.6 | 723.5 | 466.6 | 284.3 |
| m=0.2 | 1272.5 | 1270.7 | 1254.6 | 1219.2 | 1145.5 | 956.9 | 710.7 | 459.7 | 281.5 |
| m=0.5 | 1249.1 | 1248.3 | 1231.7 | 1197.7 | 1126.2 | 942.5 | 692.0 | 456.4 | 271.3 |
| m=1 | 1216.6 | 1216.1 | 1199.0 | 1168.0 | 1093.5 | 914.2 | 665.1 | 436.1 | 256.7 |
| m=2 | 1148.8 | 1149.5 | 1134.9 | 1102.5 | 1035.5 | 860.7 | 624.5 | 401.8 | 241.2 |
| m=4 | 1067.5 | 1065.8 | 1056.4 | 1029.0 | 967.6 | 797.8 | 574.7 | 368.2 | 217.9 |
| m=8 | 978.8 | 976.9 | 963.2 | 937.0 | 881.0 | 713.4 | 514.6 | 320.0 | 192.5 |
| m=16 | 881.7 | 878.9 | 868.7 | 846.1 | 784.0 | 630.0 | 439.4 | 272.9 | 163.5 |
| m=32 | 785.6 | 783.8 | 775.6 | 754.4 | 699.5 | 552.5 | 389.7 | 236.1 | 139.0 |

Adaptive m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 590.3 | 570.1 | 505.3 | 427.3 | 161.9 | 67.2 | 35.8 | 22.3 | 14.0 |
| m=0.2 | 590.7 | 577.8 | 507.0 | 420.1 | 163.7 | 68.0 | 36.0 | 22.4 | 14.1 |
| m=0.5 | 579.2 | 573.5 | 512.2 | 422.1 | 169.6 | 71.9 | 37.4 | 23.0 | 14.5 |
| m=1 | 560.2 | 555.4 | 517.4 | 437.1 | 179.7 | 76.4 | 39.5 | 24.3 | 15.2 |
| m=2 | 520.7 | 518.6 | 499.7 | 445.2 | 217.2 | 90.5 | 45.2 | 27.0 | 16.6 |
| m=4 | 478.1 | 477.4 | 470.1 | 438.7 | 292.7 | 138.2 | 60.1 | 33.2 | 19.5 |
| m=8 | 438.7 | 438.4 | 432.9 | 417.2 | 353.5 | 197.1 | 98.0 | 44.9 | 24.3 |
| m=16 | 407.2 | 406.9 | 403.0 | 391.5 | 355.5 | 237.8 | 142.5 | 71.3 | 34.7 |
| m=32 | 388.7 | 388.4 | 383.2 | 373.9 | 345.7 | 254.9 | 165.7 | 94.1 | 51.6 |

Static inverted m-estimate for refinement:

| refine \ select | m=0.1 | m=0.2 | m=0.5 | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m=0.1 | 2965.6 | 2962.7 | 2893.8 | 2777.7 | 2524.7 | 2005.4 | 1292.0 | 750.1 | 422.0 |
| m=0.2 | 2965.7 | 2962.6 | 2893.8 | 2776.0 | 2524.7 | 2005.6 | 1292.0 | 750.2 | 422.0 |
| m=0.5 | 2962.4 | 2964.7 | 2895.2 | 2771.9 | 2523.8 | 2008.4 | 1300.8 | 748.6 | 422.6 |
| m=1 | 2959.7 | 2963.7 | 2896.0 | 2769.9 | 2521.5 | 2003.3 | 1301.1 | 749.4 | 422.8 |
| m=2 | 2958.2 | 2964.3 | 2892.0 | 2773.4 | 2524.3 | 1998.7 | 1306.7 | 750.0 | 422.9 |
| m=4 | 2962.9 | 2961.3 | 2893.5 | 2772.0 | 2514.1 | 1995.8 | 1282.2 | 749.9 | 423.9 |
| m=8 | 2953.3 | 2958.7 | 2897.4 | 2769.1 | 2504.7 | 1975.5 | 1300.6 | 751.6 | 421.2 |
| m=16 | 2952.8 | 2944.9 | 2886.8 | 2757.5 | 2518.8 | 1984.1 | 1297.8 | 748.6 | 420.3 |
| m=32 | 2941.3 | 2934.9 | 2861.6 | 2754.8 | 2489.2 | 1979.9 | 1290.0 | 739.6 | 415.0 |

## Static vs. adaptive, regular vs. inverted

Each heuristic refines in every frame where it differs, with the m-estimate selecting (m_s in {0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32}); shown is the best selection, with its m_s. Inverting a symmetric heuristic (WRA, correlation) changes nothing, and adapting one that doesn't depend on the frame (Laplace) neither -- unless it is inverted, as the inverted Laplace estimate depends on it. The static Foil gain is the gain over the universal rule (`WeightedGain(CoveredPositives(), InformationContent())`), the adaptive one Foil's own (`FoilGain()`). The m-estimate (m_r = 16) is from section 1.

![static to adaptive](refinement_heuristics_plots/frames.png)

| heuristic | static | adaptive | static inverted | adaptive inverted |
|---|--:|--:|--:|--:|
| m-estimate (m=16) | 82.97% (294.7; m_s=8) | 82.55% (237.8; m_s=4) | 82.35% (1984.1; m_s=4) | 83.33% (439.4; m_s=8) |
| WRA | 82.70% (1730.9; m_s=4) | 83.50% (191.4; m_s=8) | -- | -- |
| relative cost (c_r=0.342) | 82.68% (1003.5; m_s=4) | 82.81% (129.8; m_s=4) | 82.25% (648.8; m_s=16) | 83.16% (598.2; m_s=8) |
| F-measure (beta=0.5) | 82.75% (252.6; m_s=16) | 82.80% (364.3; m_s=4) | 81.53% (168.3; m_s=16) | 81.76% (366.8; m_s=4) |
| Correlation | 83.44% (550.1; m_s=8) | 83.73% (204.3; m_s=4) | -- | -- |
| Foil gain | 82.83% (1448.7; m_s=4) | 83.37% (167.9; m_s=8) | 82.81% (1160.7; m_s=8) | 83.35% (288.3; m_s=4) |
| Laplace | 81.23% (538.1; m_s=0.5) | -- | 82.24% (1967.3; m_s=4) | 82.89% (787.5; m_s=4) |

Cells: accuracy (conditions in the rule set; the best m for selection). --: the same as the frame to its left.

### Fixed selection (m_s = 8)

As above, with the m-estimate selecting with m_s = 8 for every heuristic and frame, so that only the refinement heuristics differ.

| heuristic | static | adaptive | static inverted | adaptive inverted |
|---|--:|--:|--:|--:|
| m-estimate (m=16) | 82.97% (294.7) | 82.53% (142.5) | 82.33% (1297.8) | 83.33% (439.4) |
| WRA | 82.66% (1185.8) | 83.50% (191.4) | -- | -- |
| relative cost (c_r=0.342) | 82.49% (739.3) | 82.61% (79.8) | 82.20% (1098.2) | 83.16% (598.2) |
| F-measure (beta=0.5) | 82.75% (535.8) | 82.72% (241.0) | 81.30% (314.9) | 81.55% (107.7) |
| Correlation | 83.44% (550.1) | 83.64% (128.4) | -- | -- |
| Foil gain | 82.69% (989.1) | 83.37% (167.9) | 82.81% (1160.7) | 83.23% (199.7) |
| Laplace | 79.49% (35.9) | -- | 82.19% (1313.7) | 82.66% (589.6) |

Cells: accuracy (conditions in the rule set). --: the same as the frame to its left.

Foil's own search, climbing while the gain is positive and returning the rule it ends at (`GainAscentHillClimbing`, no selection): Foil gain: 82.95% with 423.7 conditions; Foil gain over the root: 81.18% with 32.6 conditions.

## Adaptive heuristics and m

The best selection for each m for refinement, in each frame; m -> infinity is WRA (static: WRA, adaptive: adaptive WRA; inverting it changes nothing).

![best accuracy per m for refinement](refinement_heuristics_plots/m_refine.png)

| m for refinement | static | adaptive | static inverted | adaptive inverted |
|---|--:|--:|--:|--:|
| 0.1 | 81.14% (503.3) | 81.16% (590.3) | 82.28% (750.1) | 82.86% (958.6) |
| 0.2 | 81.19% (497.8) | 81.19% (590.7) | 82.28% (750.2) | 82.91% (956.9) |
| 0.5 | 81.28% (552.5) | 81.33% (512.2) | 82.29% (748.6) | 82.90% (692.0) |
| 1 | 81.50% (502.1) | 81.36% (517.4) | 82.28% (1301.1) | 82.77% (914.2) |
| 2 | 81.96% (441.7) | 81.77% (217.2) | 82.33% (1306.7) | 82.71% (401.8) |
| 4 | 82.31% (404.1) | 82.29% (138.2) | 82.21% (1995.8) | 82.89% (797.8) |
| 8 | 82.68% (336.2) | 82.78% (197.1) | 82.33% (1300.6) | 82.92% (514.6) |
| 16 | 82.97% (294.7) | 82.55% (237.8) | 82.35% (1984.1) | 83.33% (439.4) |
| 32 | 83.01% (256.1) | 83.13% (165.7) | 82.25% (739.6) | 83.37% (389.7) |
| inf | 82.70% (1730.9) | 83.50% (191.4) | 82.70% (1730.9) | 83.50% (191.4) |

## Beam search

Beam width 1 vs. 5, the m-estimate selecting (m_s in {4, 8, 16}). A beam ranks the refinements of all its rules together: scores relative to different parents are compared, which have no common scale. `Gain(...)` ranks by the improvement over the parent instead of the level; with width 1 the two rank the refinements of a rule alike.

| refinement | m_s | width 1 | width 5 |
|---|--:|--:|--:|
| m=16 | 4 | 82.84% (383.1) | 82.90% (330.5) |
| m=16 | 8 | 82.97% (294.7) | 83.06% (262.2) |
| m=16 | 16 | 82.96% (218.4) | 83.12% (194.9) |
| Adaptive(m=16) | 4 | 82.55% (237.8) | 82.71% (158.8) |
| Adaptive(m=16) | 8 | 82.53% (142.5) | 82.83% (88.8) |
| Adaptive(m=16) | 16 | 82.31% (71.3) | 82.49% (54.1) |
| Adaptive(Inverted(m=16)) | 4 | 83.16% (630.0) | 82.90% (734.7) |
| Adaptive(Inverted(m=16)) | 8 | 83.33% (439.4) | 83.09% (531.6) |
| Adaptive(Inverted(m=16)) | 16 | 83.32% (272.9) | 83.14% (339.6) |
| Adaptive(Gain(m=16)) | 4 | 82.55% (237.8) | 82.75% (255.6) |
| Adaptive(Gain(m=16)) | 8 | 82.53% (142.5) | 83.00% (175.3) |
| Adaptive(Gain(m=16)) | 16 | 82.31% (71.3) | 83.20% (99.8) |
| Adaptive(Gain(Inverted(m=16))) | 4 | 83.16% (630.0) | 82.87% (559.9) |
| Adaptive(Gain(Inverted(m=16))) | 8 | 83.33% (439.4) | 83.06% (422.8) |
| Adaptive(Gain(Inverted(m=16))) | 16 | 83.32% (272.9) | 83.03% (277.0) |
| WRA | 4 | 82.70% (1730.9) | 82.93% (1743.8) |
| WRA | 8 | 82.66% (1185.8) | 82.79% (1232.0) |
| WRA | 16 | 82.59% (635.2) | 82.49% (733.0) |
| Adaptive(WRA) | 4 | 83.25% (281.5) | 83.15% (260.5) |
| Adaptive(WRA) | 8 | 83.50% (191.4) | 83.31% (187.8) |
| Adaptive(WRA) | 16 | 83.20% (115.3) | 83.80% (119.3) |

## Weighted gains

`WeightedGain(weight, base)`: ``weight(r) * (base(r) - base(r0))``, the gain over the universal rule ``r0`` of the frame -- over the root (static) or over the parent (adaptive). The weights are Klösgen's family ``c^a``, with ``c`` the covered examples (`p + n`) or positives (`p`); the base is linear (precision) or logarithmic (information content). `p + n` and precision is WRA, `p` and information content Foil gain, `sqrt(p + n)` and precision Klösgen's quality function, weight 1 ranks as the base itself. Best selection (m_s in {4, 8, 16}).

| weight | base | static | adaptive |
|---|---|--:|--:|
| 1 | precision | 79.84% (65.8; m_s=4) | 79.84% (65.8; m_s=4) |
| 1 | information content | 79.84% (65.8; m_s=4) | 79.84% (65.8; m_s=4) |
| sqrt(p) | precision | 83.13% (371.4; m_s=4) | 82.82% (110.2; m_s=4) |
| sqrt(p) | information content | 83.45% (385.8; m_s=16) | 83.34% (148.8; m_s=4) |
| p | precision | 83.37% (330.8; m_s=16) | 83.41% (207.8; m_s=4) |
| p | information content | 82.83% (1448.7; m_s=4) | 83.37% (167.9; m_s=8) |
| p^2 | precision | 82.89% (1392.6; m_s=4) | 83.43% (220.0; m_s=8) |
| p^2 | information content | 82.63% (609.2; m_s=16) | 83.21% (243.5; m_s=8) |
| sqrt(p + n) | precision | 83.37% (264.2; m_s=16) | 83.27% (141.5; m_s=4) |
| sqrt(p + n) | information content | 82.75% (724.1; m_s=16) | 83.28% (212.6; m_s=4) |
| p + n | precision | 82.68% (1730.0; m_s=4) | 83.50% (189.1; m_s=8) |
| p + n | information content | 81.80% (2315.5; m_s=4) | 83.24% (139.7; m_s=16) |
| (p + n)^2 | precision | 81.70% (1503.5; m_s=8) | 83.12% (196.8; m_s=16) |
| (p + n)^2 | information content | 81.32% (2362.9; m_s=4) | 82.78% (466.2; m_s=4) |

Cells: accuracy (conditions in the rule set; the best m for selection).

### Fixed selection (m_s = 8)

As above, with the m-estimate selecting with m_s = 8 for every gain.

| weight | base | static | adaptive |
|---|---|--:|--:|
| 1 | precision | 79.49% (34.8) | 79.49% (34.8) |
| 1 | information content | 79.49% (34.8) | 79.49% (34.8) |
| sqrt(p) | precision | 83.12% (255.5) | 82.50% (68.4) |
| sqrt(p) | information content | 83.38% (641.2) | 83.08% (94.2) |
| p | precision | 83.33% (586.2) | 83.41% (132.1) |
| p | information content | 82.69% (989.1) | 83.37% (167.9) |
| p^2 | precision | 82.73% (893.0) | 83.43% (220.0) |
| p^2 | information content | 82.44% (1102.2) | 83.21% (243.5) |
| sqrt(p + n) | precision | 83.29% (447.2) | 83.04% (88.5) |
| sqrt(p + n) | information content | 82.62% (1224.3) | 83.24% (126.1) |
| p + n | precision | 82.66% (1175.9) | 83.50% (189.1) |
| p + n | information content | 81.71% (1420.6) | 82.97% (225.3) |
| (p + n)^2 | precision | 81.70% (1503.5) | 83.04% (313.6) |
| (p + n)^2 | information content | 80.92% (1376.4) | 82.66% (335.7) |

### Paired comparisons

Each gain with its best selection (as above); per dataset the mean accuracy across folds; wins/losses of the second over the first, the mean difference, and the two-sided Wilcoxon signed-rank test (p) and sign test (sign p) across datasets (not corrected for the 20 tests).

| comparison | first | second | datasets | wins/losses | difference | p | sign p |
|---|---|---|--:|--:|--:|--:|--:|
| WRA vs. a weaker weight | refine gain of precision weighted by p + n, select m=4 | refine gain of precision weighted by sqrt(p + n), select m=16 | 69 | 39/28 | +0.69 | 0.051 | 0.222 |
| WRA vs. weight p | refine gain of precision weighted by p + n, select m=4 | refine gain of precision weighted by p, select m=16 | 69 | 39/27 | +0.69 | 0.018 | 0.175 |
| WRA vs. adaptive WRA | refine gain of precision weighted by p + n, select m=4 | refine Adaptive(gain of precision weighted by p + n), select m=8 | 69 | 49/19 | +0.82 | 0.001 | 0.000 |
| adaptive WRA vs. weight p^2 | refine Adaptive(gain of precision weighted by p + n), select m=8 | refine Adaptive(gain of precision weighted by p^2), select m=8 | 69 | 29/37 | -0.07 | 0.504 | 0.389 |
| Foil gain vs. weight p + n | refine Adaptive(gain of information content weighted by p), select m=8 | refine Adaptive(gain of information content weighted by p + n), select m=16 | 69 | 30/38 | -0.13 | 0.642 | 0.396 |
| Foil gain vs. adaptive WRA | refine Adaptive(gain of information content weighted by p), select m=8 | refine Adaptive(gain of precision weighted by p + n), select m=8 | 69 | 37/24 | +0.13 | 0.195 | 0.124 |
| static Foil gain vs. weight sqrt(p) | refine gain of information content weighted by p, select m=4 | refine gain of information content weighted by sqrt(p), select m=16 | 69 | 42/23 | +0.61 | 0.034 | 0.025 |
| static vs. adaptive Foil gain | refine gain of information content weighted by p, select m=4 | refine Adaptive(gain of information content weighted by p), select m=8 | 69 | 40/24 | +0.54 | 0.057 | 0.060 |
| static sqrt(p): precision vs. information content | refine gain of precision weighted by sqrt(p), select m=4 | refine gain of information content weighted by sqrt(p), select m=16 | 69 | 43/22 | +0.32 | 0.085 | 0.013 |
| static p: precision vs. information content | refine gain of precision weighted by p, select m=16 | refine gain of information content weighted by p, select m=4 | 69 | 29/38 | -0.54 | 0.024 | 0.328 |
| static p^2: precision vs. information content | refine gain of precision weighted by p^2, select m=4 | refine gain of information content weighted by p^2, select m=16 | 69 | 32/34 | -0.25 | 0.926 | 0.902 |
| static sqrt(p + n): precision vs. information content | refine gain of precision weighted by sqrt(p + n), select m=16 | refine gain of information content weighted by sqrt(p + n), select m=16 | 69 | 27/40 | -0.61 | 0.040 | 0.142 |
| static p + n: precision vs. information content | refine gain of precision weighted by p + n, select m=4 | refine gain of information content weighted by p + n, select m=4 | 69 | 25/42 | -0.88 | 0.021 | 0.050 |
| static (p + n)^2: precision vs. information content | refine gain of precision weighted by (p + n)^2, select m=8 | refine gain of information content weighted by (p + n)^2, select m=4 | 69 | 24/43 | -0.38 | 0.073 | 0.027 |
| adaptive sqrt(p): precision vs. information content | refine Adaptive(gain of precision weighted by sqrt(p)), select m=4 | refine Adaptive(gain of information content weighted by sqrt(p)), select m=4 | 69 | 38/25 | +0.52 | 0.096 | 0.130 |
| adaptive p: precision vs. information content | refine Adaptive(gain of precision weighted by p), select m=4 | refine Adaptive(gain of information content weighted by p), select m=8 | 69 | 34/31 | -0.04 | 0.766 | 0.804 |
| adaptive p^2: precision vs. information content | refine Adaptive(gain of precision weighted by p^2), select m=8 | refine Adaptive(gain of information content weighted by p^2), select m=8 | 69 | 27/38 | -0.22 | 0.055 | 0.215 |
| adaptive sqrt(p + n): precision vs. information content | refine Adaptive(gain of precision weighted by sqrt(p + n)), select m=4 | refine Adaptive(gain of information content weighted by sqrt(p + n)), select m=4 | 69 | 33/28 | +0.01 | 0.777 | 0.609 |
| adaptive p + n: precision vs. information content | refine Adaptive(gain of precision weighted by p + n), select m=8 | refine Adaptive(gain of information content weighted by p + n), select m=16 | 69 | 31/37 | -0.26 | 0.253 | 0.545 |
| adaptive (p + n)^2: precision vs. information content | refine Adaptive(gain of precision weighted by (p + n)^2), select m=16 | refine Adaptive(gain of information content weighted by (p + n)^2), select m=4 | 69 | 31/34 | -0.34 | 0.310 | 0.804 |

### Paired comparisons, fixed selection (m_s = 8)

Each gain with the selection m_s = 8; per dataset the mean accuracy across folds; wins/losses of the second over the first, the mean difference, and the two-sided Wilcoxon signed-rank test (p) and sign test (sign p) across datasets (not corrected for the 20 tests).

| comparison | first | second | datasets | wins/losses | difference | p | sign p |
|---|---|---|--:|--:|--:|--:|--:|
| WRA vs. a weaker weight | refine gain of precision weighted by p + n, select m=8 | refine gain of precision weighted by sqrt(p + n), select m=8 | 69 | 42/24 | +0.63 | 0.038 | 0.036 |
| WRA vs. weight p | refine gain of precision weighted by p + n, select m=8 | refine gain of precision weighted by p, select m=8 | 69 | 42/24 | +0.67 | 0.013 | 0.036 |
| WRA vs. adaptive WRA | refine gain of precision weighted by p + n, select m=8 | refine Adaptive(gain of precision weighted by p + n), select m=8 | 69 | 45/20 | +0.84 | 0.005 | 0.003 |
| adaptive WRA vs. weight p^2 | refine Adaptive(gain of precision weighted by p + n), select m=8 | refine Adaptive(gain of precision weighted by p^2), select m=8 | 69 | 29/37 | -0.07 | 0.504 | 0.389 |
| Foil gain vs. weight p + n | refine Adaptive(gain of information content weighted by p), select m=8 | refine Adaptive(gain of information content weighted by p + n), select m=8 | 69 | 24/41 | -0.40 | 0.030 | 0.046 |
| Foil gain vs. adaptive WRA | refine Adaptive(gain of information content weighted by p), select m=8 | refine Adaptive(gain of precision weighted by p + n), select m=8 | 69 | 37/24 | +0.13 | 0.195 | 0.124 |
| static Foil gain vs. weight sqrt(p) | refine gain of information content weighted by p, select m=8 | refine gain of information content weighted by sqrt(p), select m=8 | 69 | 47/19 | +0.69 | 0.000 | 0.001 |
| static vs. adaptive Foil gain | refine gain of information content weighted by p, select m=8 | refine Adaptive(gain of information content weighted by p), select m=8 | 69 | 44/20 | +0.68 | 0.003 | 0.004 |
| static sqrt(p): precision vs. information content | refine gain of precision weighted by sqrt(p), select m=8 | refine gain of information content weighted by sqrt(p), select m=8 | 69 | 35/28 | +0.26 | 0.701 | 0.450 |
| static p: precision vs. information content | refine gain of precision weighted by p, select m=8 | refine gain of information content weighted by p, select m=8 | 69 | 20/45 | -0.64 | 0.001 | 0.003 |
| static p^2: precision vs. information content | refine gain of precision weighted by p^2, select m=8 | refine gain of information content weighted by p^2, select m=8 | 69 | 29/34 | -0.29 | 0.314 | 0.615 |
| static sqrt(p + n): precision vs. information content | refine gain of precision weighted by sqrt(p + n), select m=8 | refine gain of information content weighted by sqrt(p + n), select m=8 | 69 | 20/45 | -0.66 | 0.001 | 0.003 |
| static p + n: precision vs. information content | refine gain of precision weighted by p + n, select m=8 | refine gain of information content weighted by p + n, select m=8 | 69 | 24/44 | -0.95 | 0.003 | 0.021 |
| static (p + n)^2: precision vs. information content | refine gain of precision weighted by (p + n)^2, select m=8 | refine gain of information content weighted by (p + n)^2, select m=8 | 69 | 22/42 | -0.78 | 0.009 | 0.017 |
| adaptive sqrt(p): precision vs. information content | refine Adaptive(gain of precision weighted by sqrt(p)), select m=8 | refine Adaptive(gain of information content weighted by sqrt(p)), select m=8 | 69 | 35/27 | +0.58 | 0.072 | 0.374 |
| adaptive p: precision vs. information content | refine Adaptive(gain of precision weighted by p), select m=8 | refine Adaptive(gain of information content weighted by p), select m=8 | 69 | 30/36 | -0.03 | 0.816 | 0.539 |
| adaptive p^2: precision vs. information content | refine Adaptive(gain of precision weighted by p^2), select m=8 | refine Adaptive(gain of information content weighted by p^2), select m=8 | 69 | 27/38 | -0.22 | 0.055 | 0.215 |
| adaptive sqrt(p + n): precision vs. information content | refine Adaptive(gain of precision weighted by sqrt(p + n)), select m=8 | refine Adaptive(gain of information content weighted by sqrt(p + n)), select m=8 | 69 | 29/37 | +0.20 | 0.742 | 0.389 |
| adaptive p + n: precision vs. information content | refine Adaptive(gain of precision weighted by p + n), select m=8 | refine Adaptive(gain of information content weighted by p + n), select m=8 | 69 | 23/43 | -0.53 | 0.004 | 0.019 |
| adaptive (p + n)^2: precision vs. information content | refine Adaptive(gain of precision weighted by (p + n)^2), select m=8 | refine Adaptive(gain of information content weighted by (p + n)^2), select m=8 | 69 | 23/37 | -0.38 | 0.030 | 0.092 |

## Ranks

Win counts and mean accuracy ranks across datasets among the best selection of each frame of section 2 and of the m-estimate.

| learner | accuracy | wins | mean rank |
|---|--:|--:|--:|
| refine Adaptive(Correlation), select m=4 | 83.73% | 7.0 | 8.93 |
| refine Adaptive(WRA), select m=8 | 83.50% | 4.7 | 9.23 |
| refine Adaptive(Foil gain), select m=8 | 83.37% | 2.7 | 9.91 |
| refine Correlation, select m=8 | 83.44% | 2.1 | 10.09 |
| refine Adaptive(relative cost (c_r=0.342)), select m=4 | 82.81% | 8.0 | 10.36 |
| refine Adaptive(Inverted(Foil gain)), select m=4 | 83.35% | 5.1 | 10.59 |
| refine Adaptive(Inverted(m=16)), select m=8 | 83.33% | 1.5 | 10.66 |
| refine Adaptive(Inverted(relative cost (c_r=0.342))), select m=8 | 83.16% | 1.5 | 10.67 |
| refine m=16, select m=8 | 82.97% | 3.4 | 11.44 |
| refine Inverted(Foil gain), select m=8 | 82.81% | 2.5 | 11.86 |
| refine Adaptive(Inverted(Laplace)), select m=4 | 82.89% | 4.4 | 11.92 |
| refine F-measure (beta=0.5), select m=16 | 82.75% | 4.1 | 12.38 |
| refine Adaptive(m=16), select m=4 | 82.55% | 1.4 | 12.51 |
| refine Adaptive(F-measure (beta=0.5)), select m=4 | 82.80% | 3.1 | 12.58 |
| refine Inverted(F-measure (beta=0.5)), select m=16 | 81.53% | 4.0 | 12.67 |
| refine Inverted(relative cost (c_r=0.342)), select m=16 | 82.25% | 6.4 | 12.84 |
| refine Foil gain, select m=4 | 82.83% | 1.2 | 12.84 |
| refine relative cost (c_r=0.342), select m=4 | 82.68% | 0.2 | 12.86 |
| refine WRA, select m=4 | 82.70% | 0.9 | 13.24 |
| refine Inverted(m=16), select m=4 | 82.35% | 1.0 | 13.97 |
| refine Adaptive(Inverted(F-measure (beta=0.5))), select m=4 | 81.76% | 2.2 | 14.20 |
| refine Inverted(Laplace), select m=4 | 82.24% | 1.0 | 14.30 |
| refine Laplace, select m=0.5 | 81.23% | 0.5 | 15.92 |

Failed folds: 0.
