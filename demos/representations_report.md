# Data representations: same rules, different speed

**Full run.** Quick sanity check instead: `python demos/representations.py`.

pyrulearn's learners never look at the data directly. They ask it two things -- which examples a rule covers (`coverage(rule)`) and which features an example has (`features_of(row)`) -- so the data can be stored in any structure that answers these questions. Four such representations are implemented:

- **HBitmap** (horizontal bitmap) -- a bit matrix packed by example, one bit per example and feature. Covering a rule checks every example's row against the rule's features, whatever the density.
- **VBitmap** (vertical bitmap) -- the same bit matrix packed by feature: one bit-vector over the examples per feature. Covering a rule ANDs its features' bit-vectors, and counting is a popcount; same memory as HBitmap.
- **NList** -- LORD's vertical index (Huynh, Fürnkranz & Beck, 2023): examples are inserted into a prefix tree of their true features, and each feature keeps the list of tree nodes it occurs in (its N-list). Covering a rule intersects these lists; the work grows with how often the features are true, not with the number of examples.
- **Sparse** -- `scipy` sparse matrices: per-feature example lists without the prefix tree.

All four must lead to exactly the same rules, so the only thing that differs is time. This demo measures that time for five learners -- CN2, PFossil, Pypper, PyLORD and ENDER, each with its default settings -- and checks the rules and the test-set predictions against HBitmap at every point. Each fit runs in a separate process and is capped at 300s, PyLORD's at 600s, CN2's at 1800s; after a time-out, the larger points of that curve are skipped. The learners ran side by side, one worker process each, every learner's fits one after another; building the representations, timing `coverage()` and measuring memory came first, with nothing else running. The numerical libraries were limited to one thread (`OMP_NUM_THREADS=1` and its relatives), so each fit used one core.

**Equality check:** at 148 points where HBitmap and at least one other representation finished, 148 gave identical rules and predictions for every representation.

## 1. Training-set size

Training sets of doubling size, drawn from one shuffled pool of the `adult` dataset (48,842 examples, 14 attributes, 8 of them nominal), each evaluated on the same 20% test set. Numeric attributes are discretized per training set (`TreeDiscretizer(max_intervals=8)`). Two encodings: **with negation** (pyrulearn's default -- every test comes with its negation, so exactly half the features are true in every example) and **without negation** (no `x != v` features; numeric attributes keep both `x >= t` and `x < t`, so it is sparser only in the nominal attributes). The first is the bitmaps' best case and the list-based representations' worst; the second is where those should gain.

At the largest training set: with negation 278 binary features, density 0.493; without negation 182 binary features, density 0.274 (density = share of features that are true).

One pair of plots per learner: left with a logarithmic time axis (ratios -- a constant factor between two representations is a constant gap), right with a linear one (absolute seconds saved).

<img src="representations_plots/size_cn2.png" alt="CN2 fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_cn2_linear.png" alt="CN2 fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pfossil.png" alt="PFossil fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pfossil_linear.png" alt="PFossil fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pypper.png" alt="Pypper fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pypper_linear.png" alt="Pypper fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pylord.png" alt="PyLORD fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pylord_linear.png" alt="PyLORD fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_ender.png" alt="ENDER fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_ender_linear.png" alt="ENDER fit time vs. training-set size, linear scale" width="49%">

Fit time in seconds, with negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 2.04 | 8.37 | 10.44 | 20.63 | 64.27 | 97.62 | 260.81 |
| CN2 | VBitmap | 1.83 (90%) | 7.80 (93%) | 9.46 (91%) | 17.51 (85%) | 53.96 (84%) | 67.53 (69%) | 151.28 (58%) |
| CN2 | NList | 2.01 (98%) | 8.27 (99%) | 10.17 (97%) | 19.89 (96%) | 58.94 (92%) | 78.98 (81%) | 185.91 (71%) |
| CN2 | Sparse | 2.17 (106%) | 8.75 (104%) | 10.82 (104%) | 21.45 (104%) | 63.26 (98%) | 89.75 (92%) | 234.42 (90%) |
| PFossil | HBitmap | 0.03 | 0.03 | 0.04 | 0.09 | 0.13 | 0.28 | 0.55 |
| PFossil | VBitmap | 0.02 (68%) | 0.02 (67%) | 0.02 (59%) | 0.04 (52%) | 0.07 (52%) | 0.14 (49%) | 0.27 (49%) |
| PFossil | NList | 0.03 (99%) | 0.03 (100%) | 0.04 (99%) | 0.08 (91%) | 0.12 (93%) | 0.24 (84%) | 0.45 (81%) |
| PFossil | Sparse | 0.03 (101%) | 0.03 (98%) | 0.04 (96%) | 0.08 (93%) | 0.12 (88%) | 0.25 (86%) | 0.49 (88%) |
| Pypper | HBitmap | 0.12 | 0.28 | 0.19 | 0.64 | 1.86 | 2.72 | 25.31 |
| Pypper | VBitmap | 0.04 (35%) | 0.09 (34%) | 0.06 (34%) | 0.15 (23%) | 0.28 (15%) | 0.39 (14%) | 1.64 (6%) |
| Pypper | NList | 0.08 (65%) | 0.23 (82%) | 0.14 (72%) | 0.35 (54%) | 0.72 (39%) | 0.88 (32%) | 5.17 (20%) |
| Pypper | Sparse | 0.06 (52%) | 0.16 (59%) | 0.12 (63%) | 0.40 (63%) | 1.01 (54%) | 1.26 (46%) | 13.48 (53%) |
| PyLORD | HBitmap | 1.12 | 3.03 | 10.52 | 34.22 | 111.09 | 414.91 | time-out |
| PyLORD | VBitmap | 0.77 (69%) | 1.99 (66%) | 6.12 (58%) | 16.90 (49%) | 44.34 (40%) | 142.57 (34%) | 505.01 |
| PyLORD | NList | 1.10 (99%) | 2.96 (97%) | 9.66 (92%) | 29.16 (85%) | 87.18 (78%) | 294.03 (71%) | time-out |
| PyLORD | Sparse | 1.16 (104%) | 3.22 (106%) | 10.75 (102%) | 36.61 (107%) | 120.82 (109%) | 456.45 (110%) | time-out |
| ENDER | HBitmap | 1.03 | 1.53 | 2.38 | 4.00 | 6.95 | 12.59 | 23.34 |
| ENDER | VBitmap | 0.84 (82%) | 1.30 (85%) | 2.14 (90%) | 3.81 (95%) | 6.83 (98%) | 12.46 (99%) | 23.03 (99%) |
| ENDER | NList | 1.12 (109%) | 1.53 (100%) | 2.31 (97%) | 3.69 (92%) | 6.26 (90%) | 11.28 (90%) | 20.55 (88%) |
| ENDER | Sparse | 1.27 (124%) | 1.77 (116%) | 2.61 (109%) | 4.04 (101%) | 6.64 (96%) | 11.71 (93%) | 21.16 (91%) |

Without negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 0.79 | 2.35 | 3.73 | 9.03 | 18.03 | 37.66 | 152.55 |
| CN2 | VBitmap | 0.70 (89%) | 2.01 (86%) | 3.04 (82%) | 6.87 (76%) | 12.34 (68%) | 23.67 (63%) | 77.46 (51%) |
| CN2 | NList | 0.79 (99%) | 2.28 (97%) | 3.51 (94%) | 8.18 (91%) | 15.41 (85%) | 28.89 (77%) | 101.68 (67%) |
| CN2 | Sparse | 0.85 (108%) | 2.49 (106%) | 3.94 (105%) | 9.08 (101%) | 16.70 (93%) | 32.01 (85%) | 116.28 (76%) |
| PFossil | HBitmap | 0.04 | 0.02 | 0.03 | 0.09 | 0.11 | 0.19 | 0.42 |
| PFossil | VBitmap | 0.03 (71%) | 0.01 (68%) | 0.02 (63%) | 0.06 (65%) | 0.06 (58%) | 0.11 (58%) | 0.24 (57%) |
| PFossil | NList | 0.03 (93%) | 0.02 (114%) | 0.03 (97%) | 0.08 (89%) | 0.10 (90%) | 0.16 (85%) | 0.34 (81%) |
| PFossil | Sparse | 0.04 (99%) | 0.02 (95%) | 0.02 (90%) | 0.08 (85%) | 0.09 (82%) | 0.15 (77%) | 0.33 (80%) |
| Pypper | HBitmap | 0.03 | 0.06 | 0.31 | 0.23 | 0.91 | 5.45 | 39.49 |
| Pypper | VBitmap | 0.02 (63%) | 0.03 (51%) | 0.08 (27%) | 0.07 (31%) | 0.18 (19%) | 0.55 (10%) | 2.06 (5%) |
| Pypper | NList | 0.03 (96%) | 0.05 (89%) | 0.19 (62%) | 0.14 (60%) | 0.38 (41%) | 1.39 (25%) | 5.91 (15%) |
| Pypper | Sparse | 0.02 (89%) | 0.05 (81%) | 0.16 (51%) | 0.14 (59%) | 0.43 (48%) | 2.41 (44%) | 17.72 (45%) |
| PyLORD | HBitmap | 0.81 | 2.17 | 6.46 | 21.05 | 67.40 | 248.04 | time-out |
| PyLORD | VBitmap | 0.55 (69%) | 1.37 (63%) | 3.63 (56%) | 10.09 (48%) | 27.36 (41%) | 84.18 (34%) | 290.39 |
| PyLORD | NList | 0.82 (101%) | 2.13 (98%) | 6.10 (94%) | 18.29 (87%) | 55.44 (82%) | 182.41 (74%) | time-out |
| PyLORD | Sparse | 0.84 (104%) | 2.14 (99%) | 6.32 (98%) | 19.55 (93%) | 58.04 (86%) | 197.68 (80%) | time-out |
| ENDER | HBitmap | 0.73 | 1.02 | 1.61 | 2.76 | 4.91 | 9.16 | 17.62 |
| ENDER | VBitmap | 0.60 (82%) | 0.88 (86%) | 1.47 (91%) | 2.58 (93%) | 4.61 (94%) | 8.70 (95%) | 16.78 (95%) |
| ENDER | NList | 0.74 (102%) | 1.06 (103%) | 1.58 (98%) | 2.61 (94%) | 4.57 (93%) | 8.41 (92%) | 16.03 (91%) |
| ENDER | Sparse | 0.93 (127%) | 1.13 (110%) | 1.67 (104%) | 2.66 (96%) | 4.55 (93%) | 8.30 (91%) | 15.74 (89%) |

**Growth.** How fast each learner's fit time grows with the training-set size: the exponent *b* of time ~ n^*b*, fitted by least squares to log time over log n from n=2,000 on (below that, a fit's fixed costs still dominate). 1 is linear, 2 quadratic; a curve cut short by a time-out is fitted on the points that finished. This is what decides, for example, how much a learner gains from splitting a problem into smaller ones (`demos/multiclass_decomposition.py`):

| learner | representation | with negation | without negation |
|---|---|--:|--:|
| CN2 | HBitmap | 1.15 | 1.28 |
| CN2 | VBitmap | 0.99 | 1.11 |
| CN2 | NList | 1.04 | 1.15 |
| CN2 | Sparse | 1.09 | 1.16 |
| PFossil | HBitmap | 0.94 | 0.89 |
| PFossil | VBitmap | 0.88 | 0.85 |
| PFossil | NList | 0.87 | 0.83 |
| PFossil | Sparse | 0.91 | 0.84 |
| Pypper | HBitmap | 1.62 | 1.85 |
| Pypper | VBitmap | 1.08 | 1.21 |
| Pypper | NList | 1.18 | 1.32 |
| Pypper | Sparse | 1.53 | 1.77 |
| PyLORD | HBitmap | 1.76 (4 points) | 1.75 (4 points) |
| PyLORD | VBitmap | 1.58 | 1.57 |
| PyLORD | NList | 1.64 (4 points) | 1.63 (4 points) |
| PyLORD | Sparse | 1.79 (4 points) | 1.65 (4 points) |
| ENDER | HBitmap | 0.82 | 0.86 |
| ENDER | VBitmap | 0.86 | 0.88 |
| ENDER | NList | 0.79 | 0.84 |
| ENDER | Sparse | 0.76 | 0.81 |

Building a vertical index is paid once per fit, on top of the fit itself:

![index build time vs. training-set size](representations_plots/size_index_build.png)

**Memory.** What each representation holds once built, measured with `tracemalloc` (the input matrix and the HBitmap matrix the others are built from don't count), and in parentheses its share of HBitmap's memory; the last row is the unpacked `n x k` bool matrix that `data.X` builds on demand, which any learner or external library working on a plain array pays on top.

![memory vs. training-set size](representations_plots/size_memory.png)

With negation:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 14 KiB | 31 KiB | 65 KiB | 133 KiB | 274 KiB | 547 KiB | 1.1 MiB |
| VBitmap | 15 KiB (105%) | 31 KiB (103%) | 66 KiB (102%) | 135 KiB (102%) | 271 KiB (99%) | 542 KiB (99%) | 1.1 MiB (100%) |
| NList | 828 KiB (5869%) | 1.5 MiB (4858%) | 3.2 MiB (5109%) | 6.0 MiB (4592%) | 12.0 MiB (4505%) | 21.5 MiB (4020%) | 37.7 MiB (3528%) |
| Sparse | 539 KiB (3823%) | 1.2 MiB (3858%) | 2.5 MiB (3887%) | 5.1 MiB (3953%) | 10.4 MiB (3896%) | 20.8 MiB (3897%) | 41.9 MiB (3925%) |
| unpacked `X` | 108 KiB | 238 KiB | 508 KiB | 1.0 MiB | 2.1 MiB | 4.2 MiB | 8.5 MiB |

Without negation:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 10 KiB | 21 KiB | 43 KiB | 90 KiB | 180 KiB | 360 KiB | 719 KiB |
| VBitmap | 10 KiB (108%) | 21 KiB (103%) | 44 KiB (102%) | 89 KiB (99%) | 178 KiB (99%) | 356 KiB (99%) | 716 KiB (100%) |
| NList | 230 KiB (2414%) | 410 KiB (1978%) | 742 KiB (1717%) | 1.3 MiB (1509%) | 2.9 MiB (1629%) | 5.0 MiB (1410%) | 8.2 MiB (1169%) |
| Sparse | 238 KiB (2502%) | 483 KiB (2330%) | 984 KiB (2278%) | 1.9 MiB (2183%) | 3.8 MiB (2184%) | 7.7 MiB (2185%) | 15.3 MiB (2185%) |
| unpacked `X` | 74 KiB | 160 KiB | 338 KiB | 699 KiB | 1.4 MiB | 2.8 MiB | 5.6 MiB |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass), with negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 576 KiB | 1.0 MiB | 1.3 MiB | 1.6 MiB | 2.0 MiB | 3.2 MiB | 6.2 MiB |
| CN2 | VBitmap | 393 KiB (68%) | 463 KiB (43%) | 586 KiB (43%) | 946 KiB (58%) | 1.5 MiB (75%) | 2.8 MiB (88%) | 5.5 MiB (88%) |
| CN2 | NList | 2.0 MiB (350%) | 3.8 MiB (368%) | 6.2 MiB (465%) | 10.8 MiB (681%) | 21.8 MiB (1097%) | 39.9 MiB (1236%) | 72.1 MiB (1159%) |
| CN2 | Sparse | 1.3 MiB (237%) | 2.8 MiB (268%) | 5.9 MiB (437%) | 12.2 MiB (768%) | 24.5 MiB (1233%) | 49.1 MiB (1522%) | 98.7 MiB (1586%) |
| PFossil | HBitmap | 519 KiB | 1019 KiB | 1.2 MiB | 1.3 MiB | 1.6 MiB | 2.0 MiB | 3.8 MiB |
| PFossil | VBitmap | 136 KiB (26%) | 216 KiB (21%) | 384 KiB (31%) | 636 KiB (47%) | 1.1 MiB (71%) | 2.1 MiB (103%) | 4.0 MiB (106%) |
| PFossil | NList | 1.9 MiB (380%) | 3.8 MiB (382%) | 6.2 MiB (514%) | 10.8 MiB (815%) | 21.8 MiB (1407%) | 39.9 MiB (1987%) | 72.1 MiB (1900%) |
| PFossil | Sparse | 1.3 MiB (251%) | 2.7 MiB (276%) | 5.8 MiB (482%) | 12.0 MiB (909%) | 24.4 MiB (1571%) | 48.7 MiB (2426%) | 98.0 MiB (2582%) |
| Pypper | HBitmap | 431 KiB | 764 KiB | 1.2 MiB | 1.3 MiB | 1.5 MiB | 2.2 MiB | 4.3 MiB |
| Pypper | VBitmap | 148 KiB (34%) | 210 KiB (27%) | 368 KiB (31%) | 627 KiB (48%) | 1.1 MiB (73%) | 2.1 MiB (97%) | 4.2 MiB (98%) |
| Pypper | NList | 1.7 MiB (402%) | 3.1 MiB (420%) | 6.1 MiB (531%) | 10.8 MiB (838%) | 21.8 MiB (1425%) | 39.7 MiB (1846%) | 71.9 MiB (1690%) |
| Pypper | Sparse | 1.1 MiB (254%) | 2.3 MiB (304%) | 4.7 MiB (409%) | 9.8 MiB (763%) | 19.8 MiB (1298%) | 39.6 MiB (1837%) | 79.6 MiB (1873%) |
| PyLORD | HBitmap | 459 KiB | 1.1 MiB | 4.0 MiB | 13.8 MiB | 54.0 MiB |  |  |
| PyLORD | VBitmap | 353 KiB (77%) | 1.1 MiB (98%) | 3.7 MiB (93%) | 13.8 MiB (100%) | 54.0 MiB (100%) |  |  |
| PyLORD | NList | 2.0 MiB (446%) | 4.0 MiB (368%) | 7.4 MiB (184%) | 20.6 MiB (149%) | 67.7 MiB (125%) |  |  |
| PyLORD | Sparse | 1.4 MiB (305%) | 2.9 MiB (272%) | 6.2 MiB (155%) | 18.8 MiB (136%) |  |  |  |
| ENDER | HBitmap | 908 KiB | 1.0 MiB | 1.6 MiB | 3.2 MiB | 3.8 MiB | 5.0 MiB | 7.5 MiB |
| ENDER | VBitmap | 881 KiB (97%) | 1002 KiB (95%) | 1.5 MiB (92%) | 3.0 MiB (94%) | 6.1 MiB (159%) | 11.6 MiB (231%) | 14.1 MiB (188%) |
| ENDER | NList | 1.8 MiB (202%) | 2.8 MiB (267%) | 5.8 MiB (353%) | 11.1 MiB (345%) | 22.4 MiB (588%) | 41.0 MiB (816%) | 74.5 MiB (993%) |
| ENDER | Sparse | 1.4 MiB (157%) | 2.1 MiB (207%) | 3.6 MiB (220%) | 7.4 MiB (230%) | 14.9 MiB (392%) | 29.8 MiB (592%) | 59.9 MiB (799%) |

Without negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 429 KiB | 762 KiB | 1.2 MiB | 1.5 MiB | 1.9 MiB | 2.5 MiB | 5.2 MiB |
| CN2 | VBitmap | 245 KiB (57%) | 335 KiB (44%) | 451 KiB (37%) | 734 KiB (47%) | 1.2 MiB (63%) | 2.1 MiB (85%) | 4.5 MiB (87%) |
| CN2 | NList | 1.0 MiB (250%) | 2.0 MiB (275%) | 3.6 MiB (299%) | 4.7 MiB (310%) | 7.4 MiB (393%) | 12.4 MiB (501%) | 22.1 MiB (428%) |
| CN2 | Sparse | 653 KiB (152%) | 1.2 MiB (163%) | 2.4 MiB (202%) | 4.8 MiB (313%) | 9.5 MiB (504%) | 19.0 MiB (767%) | 37.6 MiB (727%) |
| PFossil | HBitmap | 387 KiB | 711 KiB | 1.1 MiB | 1.2 MiB | 1.4 MiB | 1.8 MiB | 2.7 MiB |
| PFossil | VBitmap | 108 KiB (28%) | 151 KiB (21%) | 270 KiB (23%) | 468 KiB (37%) | 812 KiB (55%) | 1.5 MiB (82%) | 2.9 MiB (106%) |
| PFossil | NList | 1.0 MiB (268%) | 2.0 MiB (289%) | 3.5 MiB (311%) | 4.6 MiB (370%) | 7.1 MiB (495%) | 12.4 MiB (689%) | 22.1 MiB (822%) |
| PFossil | Sparse | 606 KiB (157%) | 1.2 MiB (167%) | 2.3 MiB (207%) | 4.7 MiB (374%) | 9.2 MiB (642%) | 18.4 MiB (1025%) | 36.8 MiB (1368%) |
| Pypper | HBitmap | 301 KiB | 533 KiB | 1014 KiB | 1.2 MiB | 1.4 MiB | 1.9 MiB | 3.8 MiB |
| Pypper | VBitmap | 101 KiB (34%) | 152 KiB (29%) | 274 KiB (27%) | 510 KiB (41%) | 992 KiB (68%) | 1.9 MiB (99%) | 3.8 MiB (100%) |
| Pypper | NList | 845 KiB (280%) | 1.6 MiB (300%) | 3.1 MiB (311%) | 4.5 MiB (375%) | 7.0 MiB (492%) | 12.3 MiB (636%) | 21.9 MiB (573%) |
| Pypper | Sparse | 495 KiB (164%) | 978 KiB (184%) | 1.9 MiB (194%) | 3.8 MiB (314%) | 7.5 MiB (531%) | 15.0 MiB (779%) | 30.0 MiB (786%) |
| PyLORD | HBitmap | 362 KiB | 1.1 MiB | 4.0 MiB | 13.6 MiB | 52.2 MiB |  |  |
| PyLORD | VBitmap | 342 KiB (95%) | 1.1 MiB (98%) | 3.7 MiB (92%) | 13.5 MiB (100%) | 52.1 MiB (100%) |  |  |
| PyLORD | NList | 1.1 MiB (304%) | 2.2 MiB (202%) | 4.8 MiB (117%) | 15.4 MiB (114%) | 56.1 MiB (107%) |  |  |
| PyLORD | Sparse | 695 KiB (192%) | 1.6 MiB (145%) | 4.7 MiB (116%) | 15.4 MiB (114%) | 55.8 MiB (107%) |  |  |
| ENDER | HBitmap | 771 KiB | 849 KiB | 1.2 MiB | 2.4 MiB | 3.7 MiB | 4.8 MiB | 7.1 MiB |
| ENDER | VBitmap | 755 KiB (98%) | 819 KiB (96%) | 1.1 MiB (92%) | 2.2 MiB (92%) | 4.3 MiB (117%) | 8.6 MiB (178%) | 13.8 MiB (193%) |
| ENDER | NList | 1.0 MiB (136%) | 1.3 MiB (161%) | 2.1 MiB (177%) | 4.0 MiB (171%) | 7.5 MiB (204%) | 13.6 MiB (281%) | 24.5 MiB (344%) |
| ENDER | Sparse | 1.0 MiB (137%) | 1.3 MiB (153%) | 1.9 MiB (158%) | 3.1 MiB (132%) | 6.2 MiB (168%) | 12.4 MiB (256%) | 24.6 MiB (346%) |

Peak memory while building (with negation) -- the vertical indexes need more than they keep:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 19 KiB | 36 KiB | 70 KiB | 138 KiB | 279 KiB | 552 KiB | 1.1 MiB |
| VBitmap | 143 KiB (742%) | 306 KiB (856%) | 645 KiB (924%) | 1.3 MiB (970%) | 2.6 MiB (972%) | 5.2 MiB (972%) | 9.5 MiB (889%) |
| NList | 4.4 MiB (23384%) | 8.5 MiB (24321%) | 17.7 MiB (26016%) | 34.5 MiB (25571%) | 69.8 MiB (25629%) | 130.9 MiB (24267%) | 255.7 MiB (23823%) |
| Sparse | 698 KiB (3625%) | 1.2 MiB (3304%) | 2.5 MiB (3601%) | 5.1 MiB (3806%) | 10.4 MiB (3825%) | 20.8 MiB (3860%) | 41.9 MiB (3907%) |
| unpacked `X` | 108 KiB | 238 KiB | 508 KiB | 1.0 MiB | 2.1 MiB | 4.2 MiB | 8.5 MiB |

## 2. Density

Synthetic data, 1000 training and 1000 test examples with 500 features: 4 of them carry a fixed learnable concept (true with probability 0.4, class = f0 and not f1, 5% label noise); every other feature is noise, true with the probability on the x-axis. Size and feature count stay fixed, so this isolates density.

![CN2 fit time vs. density](representations_plots/density_cn2.png)

![PFossil fit time vs. density](representations_plots/density_pfossil.png)

![Pypper fit time vs. density](representations_plots/density_pypper.png)

![PyLORD fit time vs. density](representations_plots/density_pylord.png)

![ENDER fit time vs. density](representations_plots/density_ender.png)

The pure data operation, 800 `coverage()` calls for random rules of 1-4 conditions, without any search around it:

![coverage() time vs. density](representations_plots/density_coverage.png)

Fit time in seconds:

| learner | representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 1.06 | 0.98 | 1.16 | 1.59 | 0.94 | 0.82 | 0.17 | 0.10 |
| CN2 | VBitmap | 0.89 (84%) | 0.82 (83%) | 0.95 (82%) | 1.31 (83%) | 0.72 (77%) | 0.57 (70%) | 0.12 (67%) | 0.06 (65%) |
| CN2 | NList | 1.03 (97%) | 0.95 (97%) | 1.10 (94%) | 1.50 (95%) | 0.89 (95%) | 0.80 (98%) | 0.17 (95%) | 0.09 (89%) |
| CN2 | Sparse | 1.06 (100%) | 0.95 (96%) | 1.07 (93%) | 1.45 (91%) | 0.90 (95%) | 0.72 (88%) | 0.15 (86%) | 0.08 (81%) |
| PFossil | HBitmap | 0.03 | 0.03 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| PFossil | VBitmap | 0.02 (62%) | 0.02 (63%) | 0.01 (59%) | 0.01 (58%) | 0.01 (58%) | 0.01 (58%) | 0.01 (57%) | 0.01 (55%) |
| PFossil | NList | 0.04 (110%) | 0.03 (101%) | 0.01 (111%) | 0.01 (99%) | 0.01 (99%) | 0.01 (90%) | 0.01 (93%) | 0.01 (79%) |
| PFossil | Sparse | 0.03 (92%) | 0.03 (85%) | 0.01 (72%) | 0.01 (68%) | 0.01 (65%) | 0.01 (66%) | 0.01 (65%) | 0.01 (63%) |
| Pypper | HBitmap | 0.10 | 0.02 | 0.04 | 0.02 | 0.07 | 0.02 | 0.05 | 0.02 |
| Pypper | VBitmap | 0.04 (42%) | 0.01 (64%) | 0.02 (52%) | 0.01 (65%) | 0.03 (41%) | 0.01 (57%) | 0.02 (45%) | 0.01 (57%) |
| Pypper | NList | 0.09 (89%) | 0.02 (114%) | 0.03 (87%) | 0.02 (101%) | 0.05 (68%) | 0.02 (84%) | 0.04 (70%) | 0.02 (83%) |
| Pypper | Sparse | 0.07 (67%) | 0.02 (84%) | 0.02 (63%) | 0.01 (74%) | 0.03 (50%) | 0.02 (67%) | 0.03 (54%) | 0.02 (68%) |
| PyLORD | HBitmap | 3.60 | 2.52 | 1.75 | 1.60 | 1.54 | 1.43 | 1.30 | 1.10 |
| PyLORD | VBitmap | 2.07 (57%) | 1.36 (54%) | 0.93 (53%) | 0.96 (60%) | 0.97 (63%) | 0.92 (64%) | 0.86 (66%) | 0.74 (67%) |
| PyLORD | NList | 3.50 (97%) | 2.57 (102%) | 1.91 (109%) | 1.93 (120%) | 1.91 (124%) | 1.42 (99%) | 1.31 (100%) | 1.11 (101%) |
| PyLORD | Sparse | 3.54 (98%) | 2.27 (90%) | 1.45 (83%) | 1.42 (89%) | 1.38 (90%) | 1.26 (88%) | 1.17 (90%) | 1.00 (91%) |
| ENDER | HBitmap | 0.93 | 0.76 | 0.69 | 0.66 | 0.63 | 0.62 | 0.63 | 0.61 |
| ENDER | VBitmap | 0.83 (88%) | 0.68 (89%) | 0.62 (91%) | 0.59 (90%) | 0.56 (89%) | 0.55 (88%) | 0.55 (87%) | 0.55 (89%) |
| ENDER | NList | 0.86 (92%) | 0.69 (91%) | 0.61 (90%) | 0.59 (89%) | 0.55 (87%) | 0.56 (89%) | 0.55 (88%) | 0.55 (89%) |
| ENDER | Sparse | 0.89 (96%) | 0.71 (93%) | 0.60 (88%) | 0.57 (86%) | 0.53 (84%) | 0.53 (85%) | 0.51 (82%) | 0.52 (85%) |

Memory held by each representation (share of HBitmap in parentheses; the last row is the unpacked bool matrix):

![memory vs. density](representations_plots/density_memory.png)

| representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB |
| VBitmap | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) |
| NList | 16.9 MiB (27943%) | 10.1 MiB (16816%) | 3.4 MiB (5637%) | 1.7 MiB (2809%) | 709 KiB (1148%) | 376 KiB (609%) | 208 KiB (337%) | 111 KiB (180%) |
| Sparse | 2.4 MiB (3962%) | 1.4 MiB (2390%) | 509 KiB (824%) | 264 KiB (428%) | 119 KiB (193%) | 72 KiB (116%) | 48 KiB (77%) | 33 KiB (54%) |
| unpacked `X` | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass):

| learner | representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 1.6 MiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | 1.3 MiB | 1.3 MiB | 1.2 MiB | 1.2 MiB |
| CN2 | VBitmap | 1.2 MiB (72%) | 1.2 MiB (87%) | 1.2 MiB (85%) | 1.1 MiB (79%) | 796 KiB (59%) | 754 KiB (56%) | 654 KiB (51%) | 470 KiB (38%) |
| CN2 | NList | 20.3 MiB (1234%) | 13.1 MiB (962%) | 6.4 MiB (471%) | 4.8 MiB (333%) | 3.7 MiB (276%) | 1.2 MiB (88%) | 866 KiB (68%) | 485 KiB (39%) |
| CN2 | Sparse | 5.7 MiB (346%) | 3.5 MiB (257%) | 1.6 MiB (120%) | 1.4 MiB (95%) | 858 KiB (63%) | 784 KiB (58%) | 645 KiB (51%) | 373 KiB (30%) |
| PFossil | HBitmap | 1.2 MiB | 1.2 MiB | 1.2 MiB | 1.2 MiB | 1.2 MiB | 1.2 MiB | 1.2 MiB | 1.2 MiB |
| PFossil | VBitmap | 405 KiB (32%) | 403 KiB (32%) | 402 KiB (33%) | 402 KiB (33%) | 402 KiB (33%) | 400 KiB (33%) | 395 KiB (32%) | 357 KiB (29%) |
| PFossil | NList | 19.7 MiB (1620%) | 13.0 MiB (1069%) | 6.3 MiB (526%) | 4.6 MiB (382%) | 3.6 MiB (299%) | 701 KiB (57%) | 481 KiB (39%) | 369 KiB (30%) |
| PFossil | Sparse | 5.6 MiB (459%) | 3.4 MiB (281%) | 1.2 MiB (104%) | 713 KiB (58%) | 380 KiB (31%) | 271 KiB (22%) | 243 KiB (20%) | 207 KiB (17%) |
| Pypper | HBitmap | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB |
| Pypper | VBitmap | 412 KiB (36%) | 408 KiB (37%) | 410 KiB (36%) | 408 KiB (37%) | 408 KiB (35%) | 379 KiB (33%) | 352 KiB (31%) | 351 KiB (31%) |
| Pypper | NList | 19.7 MiB (1743%) | 13.0 MiB (1194%) | 6.3 MiB (563%) | 4.6 MiB (419%) | 3.6 MiB (314%) | 662 KiB (58%) | 476 KiB (41%) | 363 KiB (32%) |
| Pypper | Sparse | 4.6 MiB (404%) | 2.8 MiB (256%) | 1.0 MiB (93%) | 597 KiB (53%) | 333 KiB (29%) | 257 KiB (22%) | 215 KiB (19%) | 209 KiB (18%) |
| PyLORD | HBitmap | 1.1 MiB | 1.1 MiB | 1.3 MiB | 1.4 MiB | 1.6 MiB | 1.6 MiB | 1.5 MiB | 1.2 MiB |
| PyLORD | VBitmap | 1.1 MiB (99%) | 1.1 MiB (99%) | 1.1 MiB (86%) | 1.4 MiB (100%) | 1.6 MiB (99%) | 1.6 MiB (99%) | 1.5 MiB (99%) | 1.2 MiB (98%) |
| PyLORD | NList | 20.0 MiB (1793%) | 13.2 MiB (1184%) | 6.4 MiB (490%) | 4.8 MiB (340%) | 3.8 MiB (241%) | 2.0 MiB (128%) | 1.7 MiB (115%) | 1.3 MiB (108%) |
| PyLORD | Sparse | 5.9 MiB (525%) | 3.6 MiB (325%) | 1.6 MiB (123%) | 1.7 MiB (118%) | 1.7 MiB (107%) | 1.6 MiB (104%) | 1.5 MiB (101%) | 1.2 MiB (99%) |
| ENDER | HBitmap | 1.5 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB |
| ENDER | VBitmap | 1.4 MiB (91%) | 1.4 MiB (91%) | 1.4 MiB (91%) | 1.4 MiB (91%) | 1.4 MiB (91%) | 1.4 MiB (91%) | 1.4 MiB (91%) | 1.4 MiB (91%) |
| ENDER | NList | 19.4 MiB (1307%) | 11.9 MiB (796%) | 5.2 MiB (346%) | 3.5 MiB (232%) | 2.5 MiB (166%) | 810 KiB (53%) | 601 KiB (39%) | 476 KiB (31%) |
| ENDER | Sparse | 3.4 MiB (230%) | 2.2 MiB (143%) | 1015 KiB (66%) | 702 KiB (46%) | 493 KiB (32%) | 410 KiB (27%) | 384 KiB (25%) | 363 KiB (24%) |

## 3. Width

Synthetic data again, 1000 training and 1000 test examples, the same concept on 4 signal features -- but now the number of *true* features per example stays fixed (on average 16 noise features, plus the signal ones) while the number of features grows. That is how sparse data usually looks: one-hot encodings of attributes with many values, or words in a text. The bitmaps grow with the number of features (one bit per cell); a sparse representation with the number of true cells, which stays the same. `Sparse` keeps two copies of a 4-byte index plus a 1-byte value per true cell, about 10 bytes against 1/8 byte per cell for the bitmaps, so it should take less memory below a density of about 1/80 -- here from about 1,408 features on.

![memory vs. width](representations_plots/width_memory.png)

Memory held by each representation (share of HBitmap in parentheses; the last row is the unpacked bool matrix):

| representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 13 KiB | 25 KiB | 62 KiB | 122 KiB | 244 KiB | 611 KiB | 1.2 MiB | 2.4 MiB |
| VBitmap | 13 KiB (103%) | 26 KiB (105%) | 63 KiB (102%) | 126 KiB (103%) | 251 KiB (103%) | 626 KiB (102%) | 1.2 MiB (102%) | 2.4 MiB (102%) |
| NList | 387 KiB (2997%) | 637 KiB (2586%) | 1.1 MiB (1847%) | 2.1 MiB (1754%) | 4.1 MiB (1703%) | 9.9 MiB (1652%) | 19.5 MiB (1634%) | 38.7 MiB (1624%) |
| Sparse | 179 KiB (1383%) | 179 KiB (727%) | 180 KiB (292%) | 182 KiB (149%) | 186 KiB (76%) | 198 KiB (32%) | 218 KiB (18%) | 257 KiB (11%) |
| unpacked `X` | 98 KiB | 195 KiB | 488 KiB | 977 KiB | 1.9 MiB | 4.8 MiB | 9.5 MiB | 19.1 MiB |

800 `coverage()` calls for random rules of 1-4 conditions:

![coverage() time vs. width](representations_plots/width_coverage.png)

![fit time vs. width](representations_plots/width_fit.png)

Fit time in seconds:

| learner | representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 0.98 | 1.06 | 1.28 | 1.39 | 1.79 | 3.95 | 6.14 | 14.22 |
| CN2 | VBitmap | 0.74 (76%) | 0.84 (79%) | 1.05 (82%) | 1.02 (73%) | 1.04 (58%) | 1.57 (40%) | 2.37 (39%) | 3.74 (26%) |
| CN2 | NList | 1.01 (104%) | 1.02 (96%) | 1.23 (96%) | 1.25 (90%) | 1.25 (70%) | 1.55 (39%) | 2.30 (38%) | 3.46 (24%) |
| CN2 | Sparse | 1.08 (110%) | 1.03 (97%) | 1.21 (94%) | 1.16 (83%) | 1.12 (63%) | 1.47 (37%) | 2.24 (36%) | 3.37 (24%) |
| PFossil | HBitmap | 0.01 | 0.01 | 0.01 | 0.02 | 0.03 | 0.06 | 0.11 | 0.27 |
| PFossil | VBitmap | 0.01 (72%) | 0.01 (67%) | 0.01 (60%) | 0.01 (45%) | 0.01 (35%) | 0.01 (24%) | 0.02 (18%) | 0.03 (12%) |
| PFossil | NList | 0.01 (99%) | 0.01 (101%) | 0.01 (101%) | 0.01 (76%) | 0.02 (61%) | 0.03 (51%) | 0.05 (45%) | 0.09 (33%) |
| PFossil | Sparse | 0.01 (92%) | 0.01 (84%) | 0.01 (66%) | 0.01 (48%) | 0.01 (36%) | 0.01 (20%) | 0.02 (14%) | 0.02 (8%) |
| Pypper | HBitmap | 0.05 | 0.02 | 0.02 | 0.04 | 0.03 | 0.51 | 0.50 | 0.89 |
| Pypper | VBitmap | 0.02 (44%) | 0.01 (69%) | 0.01 (59%) | 0.02 (43%) | 0.02 (48%) | 0.12 (24%) | 0.11 (21%) | 0.15 (17%) |
| Pypper | NList | 0.04 (79%) | 0.02 (101%) | 0.02 (92%) | 0.03 (66%) | 0.02 (74%) | 0.15 (29%) | 0.13 (26%) | 0.18 (20%) |
| Pypper | Sparse | 0.03 (62%) | 0.01 (84%) | 0.01 (67%) | 0.02 (47%) | 0.01 (46%) | 0.10 (20%) | 0.08 (16%) | 0.10 (11%) |
| PyLORD | HBitmap | 1.72 | 1.48 | 1.59 | 1.50 | 1.27 | 1.30 | 1.57 | 2.10 |
| PyLORD | VBitmap | 1.09 (63%) | 0.93 (63%) | 0.97 (61%) | 0.90 (60%) | 0.76 (60%) | 0.74 (57%) | 0.81 (52%) | 0.96 (45%) |
| PyLORD | NList | 1.67 (97%) | 1.51 (102%) | 1.91 (120%) | 1.46 (97%) | 1.17 (92%) | 1.06 (82%) | 1.11 (71%) | 1.20 (57%) |
| PyLORD | Sparse | 1.64 (95%) | 1.36 (92%) | 1.39 (88%) | 1.27 (84%) | 1.03 (81%) | 0.92 (71%) | 0.94 (60%) | 1.02 (48%) |
| ENDER | HBitmap | 0.48 | 0.53 | 0.65 | 0.87 | 1.28 | 2.39 | 4.52 | 9.93 |
| ENDER | VBitmap | 0.44 (91%) | 0.48 (91%) | 0.57 (88%) | 0.77 (88%) | 1.13 (88%) | 2.24 (94%) | 4.10 (91%) | 7.84 (79%) |
| ENDER | NList | 0.47 (99%) | 0.50 (96%) | 0.57 (87%) | 0.62 (71%) | 0.67 (52%) | 0.81 (34%) | 1.06 (23%) | 1.58 (16%) |
| ENDER | Sparse | 0.52 (109%) | 0.54 (102%) | 0.56 (85%) | 0.58 (66%) | 0.62 (48%) | 0.75 (31%) | 0.98 (22%) | 1.48 (15%) |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass):

| learner | representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 623 KiB | 992 KiB | 1.4 MiB | 1.8 MiB | 3.2 MiB | 7.3 MiB | 12.1 MiB | 23.9 MiB |
| CN2 | VBitmap | 404 KiB (65%) | 567 KiB (57%) | 986 KiB (69%) | 1.4 MiB (80%) | 3.2 MiB (100%) | 7.4 MiB (100%) | 12.1 MiB (100%) | 23.9 MiB (100%) |
| CN2 | NList | 1.6 MiB (262%) | 2.8 MiB (290%) | 4.1 MiB (299%) | 3.6 MiB (206%) | 7.2 MiB (228%) | 16.8 MiB (229%) | 30.6 MiB (253%) | 60.5 MiB (253%) |
| CN2 | Sparse | 630 KiB (101%) | 756 KiB (76%) | 1.1 MiB (80%) | 1.5 MiB (84%) | 3.1 MiB (98%) | 6.9 MiB (94%) | 11.1 MiB (92%) | 21.7 MiB (91%) |
| PFossil | HBitmap | 469 KiB | 823 KiB | 1.2 MiB | 1.4 MiB | 1.8 MiB | 3.2 MiB | 4.8 MiB | 8.9 MiB |
| PFossil | VBitmap | 129 KiB (28%) | 174 KiB (21%) | 402 KiB (33%) | 688 KiB (48%) | 1.3 MiB (73%) | 3.5 MiB (107%) | 5.7 MiB (120%) | 11.1 MiB (125%) |
| PFossil | NList | 1.4 MiB (316%) | 2.6 MiB (329%) | 4.0 MiB (335%) | 2.7 MiB (192%) | 5.0 MiB (276%) | 11.9 MiB (368%) | 22.8 MiB (478%) | 45.3 MiB (509%) |
| PFossil | Sparse | 475 KiB (101%) | 479 KiB (58%) | 521 KiB (43%) | 554 KiB (39%) | 923 KiB (50%) | 2.0 MiB (62%) | 3.2 MiB (68%) | 6.6 MiB (75%) |
| Pypper | HBitmap | 362 KiB | 599 KiB | 1.1 MiB | 1.3 MiB | 1.6 MiB | 2.7 MiB | 4.4 MiB | 9.8 MiB |
| Pypper | VBitmap | 135 KiB (37%) | 179 KiB (30%) | 409 KiB (37%) | 684 KiB (51%) | 1.2 MiB (76%) | 2.9 MiB (105%) | 5.1 MiB (115%) | 11.1 MiB (113%) |
| Pypper | NList | 1.1 MiB (325%) | 2.1 MiB (351%) | 4.0 MiB (366%) | 2.6 MiB (200%) | 4.9 MiB (305%) | 11.7 MiB (430%) | 22.4 MiB (504%) | 45.7 MiB (464%) |
| Pypper | Sparse | 401 KiB (111%) | 399 KiB (67%) | 442 KiB (40%) | 487 KiB (36%) | 803 KiB (49%) | 1.8 MiB (68%) | 2.9 MiB (65%) | 7.0 MiB (71%) |
| PyLORD | HBitmap | 1.4 MiB | 1.2 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 2.7 MiB | 4.7 MiB | 8.7 MiB |
| PyLORD | VBitmap | 1.4 MiB (100%) | 1.2 MiB (99%) | 1.5 MiB (100%) | 1.5 MiB (100%) | 1.5 MiB (99%) | 2.0 MiB (76%) | 2.9 MiB (62%) | 4.6 MiB (53%) |
| PyLORD | NList | 1.8 MiB (136%) | 2.8 MiB (232%) | 4.2 MiB (278%) | 3.7 MiB (252%) | 5.6 MiB (365%) | 11.5 MiB (427%) | 21.4 MiB (455%) | 41.1 MiB (474%) |
| PyLORD | Sparse | 1.6 MiB (115%) | 1.4 MiB (116%) | 1.7 MiB (111%) | 1.6 MiB (108%) | 1.5 MiB (98%) | 1.7 MiB (61%) | 1.9 MiB (41%) | 2.4 MiB (28%) |
| ENDER | HBitmap | 443 KiB | 709 KiB | 1.5 MiB | 2.8 MiB | 3.2 MiB | 4.6 MiB | 6.1 MiB | 10.6 MiB |
| ENDER | VBitmap | 414 KiB (93%) | 654 KiB (92%) | 1.4 MiB (91%) | 2.5 MiB (91%) | 5.0 MiB (155%) | 10.7 MiB (231%) | 12.1 MiB (197%) | 25.0 MiB (235%) |
| ENDER | NList | 902 KiB (204%) | 1.4 MiB (208%) | 2.9 MiB (194%) | 2.7 MiB (97%) | 4.9 MiB (150%) | 11.6 MiB (250%) | 22.0 MiB (359%) | 44.4 MiB (418%) |
| ENDER | Sparse | 542 KiB (122%) | 557 KiB (79%) | 590 KiB (39%) | 592 KiB (21%) | 792 KiB (24%) | 1.7 MiB (36%) | 2.5 MiB (41%) | 5.7 MiB (54%) |

