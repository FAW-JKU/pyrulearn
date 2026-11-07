# Data representations: same rules, different speed

**Full run.** Quick sanity check instead: `python demos/representations.py`.

pyrulearn's learners never look at the data directly. They ask it two things -- which examples a rule covers (`coverage(rule)`) and which features an example has (`features_of(row)`) -- so the data can be stored in any structure that answers these questions. Five such representations are implemented:

- **HBitmap** (horizontal bitmap) -- a bit matrix packed by example, one bit per example and feature. Covering a rule checks every example's row against the rule's features, whatever the density.
- **VBitmap** (vertical bitmap) -- the same bit matrix packed by feature: one bit-vector over the examples per feature. Covering a rule ANDs its features' bit-vectors, and counting is a popcount; same memory as HBitmap.
- **NList** -- LORD's vertical index (Huynh, Fürnkranz & Beck, 2023): examples are inserted into a prefix tree of their true features, and each feature keeps the list of tree nodes it occurs in (its N-list). Covering a rule intersects these lists; the work grows with how often the features are true, not with the number of examples. Each node keeps the bitmask of the features on its path, so that a refinement is one vectorized test.
- **PrePost** -- the same tree without these bitmasks, as in PrePost (Deng et al., 2012): a node is compared with another by their pre/post codes, and an example's features are read back by walking up the tree. Less memory, and binary searches instead of the bitmask test.
- **Sparse** -- `scipy` sparse matrices: per-feature example lists without the prefix tree.

All five must lead to exactly the same rules, so the only thing that differs is time. This demo measures that time for five learners -- CN2, PFossil, Pypper, PyLORD and ENDER, each with its default settings -- and checks the rules and the test-set predictions against HBitmap at every point. Each fit runs in a separate process and is capped at 300s, PyLORD's at 600s, CN2's at 1800s; after a time-out, the larger points of that curve are skipped. The learners ran side by side, one worker process each, every learner's fits one after another; building the representations, timing `coverage()` and measuring memory came first, with nothing else running. The numerical libraries were limited to one thread (`OMP_NUM_THREADS=1` and its relatives), so each fit used one core.

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
| CN2 | HBitmap | 0.87 | 4.15 | 4.01 | 13.12 | 44.76 | 77.63 | 233.44 |
| CN2 | VBitmap | 0.69 (79%) | 3.73 (90%) | 3.23 (81%) | 10.45 (80%) | 35.85 (80%) | 48.68 (63%) | 125.50 (54%) |
| CN2 | NList | 0.82 (94%) | 4.07 (98%) | 3.73 (93%) | 12.08 (92%) | 40.29 (90%) | 59.98 (77%) | 164.29 (70%) |
| CN2 | PrePost | 0.83 (95%) | 4.15 (100%) | 3.82 (95%) | 12.51 (95%) | 41.44 (93%) | 59.75 (77%) | 160.27 (69%) |
| CN2 | Sparse | 0.94 (107%) | 4.42 (106%) | 4.29 (107%) | 13.38 (102%) | 44.09 (98%) | 70.63 (91%) | 206.56 (88%) |
| PFossil | HBitmap | 0.03 | 0.04 | 0.04 | 0.09 | 0.14 | 0.30 | 0.60 |
| PFossil | VBitmap | 0.02 (69%) | 0.03 (66%) | 0.03 (79%) | 0.05 (52%) | 0.07 (51%) | 0.15 (49%) | 0.28 (46%) |
| PFossil | NList | 0.03 (99%) | 0.04 (104%) | 0.05 (113%) | 0.09 (97%) | 0.14 (94%) | 0.24 (81%) | 0.47 (78%) |
| PFossil | PrePost | 0.03 (119%) | 0.05 (138%) | 0.05 (125%) | 0.11 (116%) | 0.18 (123%) | 0.34 (114%) | 0.64 (107%) |
| PFossil | Sparse | 0.03 (106%) | 0.04 (98%) | 0.04 (94%) | 0.08 (89%) | 0.13 (88%) | 0.26 (88%) | 0.51 (86%) |
| Pypper | HBitmap | 0.13 | 0.06 | 0.41 | 0.67 | 2.37 | 2.82 | 34.17 |
| Pypper | VBitmap | 0.04 (35%) | 0.03 (54%) | 0.10 (26%) | 0.15 (22%) | 0.33 (14%) | 0.41 (15%) | 2.01 (6%) |
| Pypper | NList | 0.08 (67%) | 0.06 (95%) | 0.26 (64%) | 0.35 (52%) | 0.87 (37%) | 0.94 (33%) | 6.27 (18%) |
| Pypper | PrePost | 0.09 (73%) | 0.06 (104%) | 0.30 (74%) | 0.43 (64%) | 1.06 (45%) | 1.14 (41%) | 8.83 (26%) |
| Pypper | Sparse | 0.07 (52%) | 0.05 (83%) | 0.22 (55%) | 0.39 (58%) | 1.22 (52%) | 1.38 (49%) | 18.58 (54%) |
| PyLORD | HBitmap | 1.10 | 3.12 | 10.71 | 35.34 | 115.71 | 436.72 | time-out |
| PyLORD | VBitmap | 0.82 (75%) | 2.03 (65%) | 6.12 (57%) | 17.44 (49%) | 47.33 (41%) | 149.61 (34%) | 529.64 |
| PyLORD | NList | 1.13 (103%) | 2.99 (96%) | 9.81 (92%) | 30.29 (86%) | 91.85 (79%) | 308.25 (71%) | time-out |
| PyLORD | PrePost | 1.87 (170%) | 4.75 (152%) | 13.57 (127%) | 39.69 (112%) | 115.34 (100%) | 357.69 (82%) | time-out |
| PyLORD | Sparse | 1.20 (110%) | 3.26 (104%) | 11.10 (104%) | 38.28 (108%) | 127.84 (110%) | 487.43 (112%) | time-out |
| ENDER | HBitmap | 1.07 | 1.61 | 2.48 | 4.17 | 7.40 | 13.44 | 24.11 |
| ENDER | VBitmap | 0.88 (82%) | 1.38 (85%) | 2.26 (91%) | 3.97 (95%) | 7.10 (96%) | 14.08 (105%) | 24.42 (101%) |
| ENDER | NList | 1.22 (114%) | 1.59 (99%) | 2.42 (98%) | 3.90 (94%) | 6.56 (89%) | 12.14 (90%) | 21.47 (89%) |
| ENDER | PrePost | 1.13 (106%) | 1.66 (103%) | 2.45 (99%) | 3.86 (93%) | 6.92 (94%) | 12.26 (91%) | 21.80 (90%) |
| ENDER | Sparse | 1.35 (126%) | 1.89 (117%) | 2.70 (109%) | 4.21 (101%) | 7.26 (98%) | 12.63 (94%) | 22.23 (92%) |

Without negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 0.29 | 1.02 | 2.32 | 6.26 | 15.30 | 27.70 | 134.48 |
| CN2 | VBitmap | 0.23 (80%) | 0.81 (80%) | 1.76 (76%) | 4.42 (71%) | 9.96 (65%) | 16.44 (59%) | 66.19 (49%) |
| CN2 | NList | 0.29 (99%) | 0.99 (97%) | 2.16 (93%) | 5.55 (89%) | 12.70 (83%) | 21.06 (76%) | 89.69 (67%) |
| CN2 | PrePost | 0.29 (99%) | 0.99 (97%) | 2.16 (93%) | 5.58 (89%) | 12.73 (83%) | 20.96 (76%) | 89.29 (66%) |
| CN2 | Sparse | 0.31 (107%) | 1.09 (107%) | 2.36 (102%) | 6.20 (99%) | 13.77 (90%) | 23.09 (83%) | 102.37 (76%) |
| PFossil | HBitmap | 0.04 | 0.02 | 0.03 | 0.10 | 0.11 | 0.20 | 0.43 |
| PFossil | VBitmap | 0.03 (74%) | 0.01 (70%) | 0.02 (64%) | 0.07 (66%) | 0.06 (55%) | 0.11 (55%) | 0.24 (57%) |
| PFossil | NList | 0.04 (99%) | 0.02 (102%) | 0.03 (93%) | 0.09 (86%) | 0.10 (84%) | 0.17 (84%) | 0.34 (81%) |
| PFossil | PrePost | 0.04 (96%) | 0.02 (110%) | 0.03 (113%) | 0.09 (95%) | 0.11 (98%) | 0.19 (97%) | 0.39 (92%) |
| PFossil | Sparse | 0.04 (101%) | 0.02 (93%) | 0.03 (90%) | 0.08 (84%) | 0.09 (77%) | 0.15 (75%) | 0.32 (75%) |
| Pypper | HBitmap | 0.09 | 0.03 | 0.43 | 0.48 | 0.85 | 3.74 | 28.48 |
| Pypper | VBitmap | 0.04 (47%) | 0.02 (63%) | 0.11 (25%) | 0.12 (25%) | 0.17 (20%) | 0.47 (13%) | 1.79 (6%) |
| Pypper | NList | 0.08 (91%) | 0.03 (97%) | 0.26 (62%) | 0.25 (52%) | 0.36 (42%) | 1.08 (29%) | 4.59 (16%) |
| Pypper | PrePost | 0.08 (97%) | 0.03 (103%) | 0.30 (70%) | 0.30 (62%) | 0.41 (48%) | 1.31 (35%) | 6.01 (21%) |
| Pypper | Sparse | 0.06 (74%) | 0.03 (87%) | 0.22 (51%) | 0.27 (55%) | 0.41 (49%) | 1.88 (50%) | 12.95 (45%) |
| PyLORD | HBitmap | 0.80 | 2.13 | 6.54 | 21.18 | 68.32 | 249.32 | time-out |
| PyLORD | VBitmap | 0.55 (68%) | 1.37 (64%) | 3.65 (56%) | 10.19 (48%) | 27.74 (41%) | 88.05 (35%) | 293.64 |
| PyLORD | NList | 0.82 (102%) | 2.10 (99%) | 6.15 (94%) | 18.54 (88%) | 54.72 (80%) | 184.30 (74%) | time-out |
| PyLORD | PrePost | 1.13 (140%) | 2.75 (129%) | 7.56 (116%) | 21.59 (102%) | 61.24 (90%) | 197.14 (79%) | time-out |
| PyLORD | Sparse | 0.84 (105%) | 2.17 (102%) | 6.33 (97%) | 20.11 (95%) | 58.13 (85%) | 200.00 (80%) | time-out |
| ENDER | HBitmap | 0.75 | 1.08 | 1.68 | 2.86 | 5.09 | 9.58 | 18.58 |
| ENDER | VBitmap | 0.62 (82%) | 0.92 (86%) | 1.52 (90%) | 2.66 (93%) | 4.79 (94%) | 9.14 (95%) | 17.91 (96%) |
| ENDER | NList | 0.77 (102%) | 1.06 (98%) | 1.64 (98%) | 2.70 (94%) | 4.77 (94%) | 8.78 (92%) | 16.82 (91%) |
| ENDER | PrePost | 0.78 (103%) | 1.07 (99%) | 1.66 (99%) | 2.72 (95%) | 4.78 (94%) | 9.02 (94%) | 16.59 (89%) |
| ENDER | Sparse | 0.90 (119%) | 1.19 (110%) | 1.76 (105%) | 2.80 (98%) | 4.82 (95%) | 8.90 (93%) | 16.64 (90%) |

**Growth.** How fast each learner's fit time grows with the training-set size: the exponent *b* of time ~ n^*b*, fitted by least squares to log time over log n from n=2,000 on (below that, a fit's fixed costs still dominate). 1 is linear, 2 quadratic; a curve cut short by a time-out is fitted on the points that finished. This is what decides, for example, how much a learner gains from splitting a problem into smaller ones (`demos/multiclass_decomposition.py`):

| learner | representation | with negation | without negation |
|---|---|--:|--:|
| CN2 | HBitmap | 1.43 | 1.39 |
| CN2 | VBitmap | 1.28 | 1.24 |
| CN2 | NList | 1.32 | 1.27 |
| CN2 | PrePost | 1.30 | 1.27 |
| CN2 | Sparse | 1.36 | 1.28 |
| PFossil | HBitmap | 0.93 | 0.88 |
| PFossil | VBitmap | 0.77 | 0.82 |
| PFossil | NList | 0.80 | 0.84 |
| PFossil | PrePost | 0.88 | 0.82 |
| PFossil | Sparse | 0.90 | 0.81 |
| Pypper | HBitmap | 1.49 | 1.51 |
| Pypper | VBitmap | 1.00 | 1.01 |
| Pypper | NList | 1.06 | 1.03 |
| Pypper | PrePost | 1.12 | 1.08 |
| Pypper | Sparse | 1.46 | 1.46 |
| PyLORD | HBitmap | 1.78 (4 points) | 1.75 (4 points) |
| PyLORD | VBitmap | 1.60 | 1.58 |
| PyLORD | NList | 1.65 (4 points) | 1.63 (4 points) |
| PyLORD | PrePost | 1.57 (4 points) | 1.56 (4 points) |
| PyLORD | Sparse | 1.81 (4 points) | 1.65 (4 points) |
| ENDER | HBitmap | 0.83 | 0.87 |
| ENDER | VBitmap | 0.87 | 0.89 |
| ENDER | NList | 0.79 | 0.84 |
| ENDER | PrePost | 0.80 | 0.84 |
| ENDER | Sparse | 0.77 | 0.81 |

Building a vertical index is paid once per fit, on top of the fit itself:

![index build time vs. training-set size](representations_plots/size_index_build.png)

**Memory.** What each representation holds once built, measured with `tracemalloc` (the input matrix and the HBitmap matrix the others are built from don't count), and in parentheses its share of HBitmap's memory; the last row is the unpacked `n x k` bool matrix that `data.X` builds on demand, which any learner or external library working on a plain array pays on top.

![memory vs. training-set size](representations_plots/size_memory.png)

With negation:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 14 KiB | 31 KiB | 65 KiB | 133 KiB | 274 KiB | 547 KiB | 1.1 MiB |
| VBitmap | 15 KiB (105%) | 31 KiB (103%) | 66 KiB (102%) | 135 KiB (102%) | 271 KiB (99%) | 542 KiB (99%) | 1.1 MiB (100%) |
| NList | 828 KiB (5869%) | 1.5 MiB (4862%) | 3.2 MiB (5109%) | 6.0 MiB (4592%) | 12.0 MiB (4505%) | 21.5 MiB (4020%) | 37.7 MiB (3528%) |
| PrePost | 425 KiB (3012%) | 762 KiB (2493%) | 1.4 MiB (2191%) | 2.6 MiB (1971%) | 5.2 MiB (1932%) | 9.2 MiB (1729%) | 16.3 MiB (1524%) |
| Sparse | 539 KiB (3823%) | 1.2 MiB (3860%) | 2.5 MiB (3887%) | 5.1 MiB (3953%) | 10.4 MiB (3896%) | 20.8 MiB (3897%) | 41.9 MiB (3925%) |
| unpacked `X` | 108 KiB | 238 KiB | 508 KiB | 1.0 MiB | 2.1 MiB | 4.2 MiB | 8.5 MiB |

Without negation:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 10 KiB | 21 KiB | 43 KiB | 90 KiB | 180 KiB | 360 KiB | 719 KiB |
| VBitmap | 10 KiB (108%) | 21 KiB (103%) | 44 KiB (102%) | 89 KiB (99%) | 178 KiB (99%) | 356 KiB (99%) | 716 KiB (100%) |
| NList | 230 KiB (2416%) | 410 KiB (1979%) | 742 KiB (1718%) | 1.3 MiB (1508%) | 2.9 MiB (1629%) | 5.0 MiB (1410%) | 8.2 MiB (1169%) |
| PrePost | 151 KiB (1593%) | 270 KiB (1303%) | 489 KiB (1133%) | 898 KiB (997%) | 1.9 MiB (1072%) | 3.3 MiB (934%) | 5.5 MiB (783%) |
| Sparse | 238 KiB (2502%) | 483 KiB (2331%) | 984 KiB (2278%) | 1.9 MiB (2183%) | 3.8 MiB (2184%) | 7.7 MiB (2185%) | 15.3 MiB (2185%) |
| unpacked `X` | 74 KiB | 160 KiB | 338 KiB | 699 KiB | 1.4 MiB | 2.8 MiB | 5.6 MiB |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass), with negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 571 KiB | 1.0 MiB | 1.3 MiB | 1.6 MiB | 2.0 MiB | 3.2 MiB | 6.2 MiB |
| CN2 | VBitmap | 393 KiB (69%) | 463 KiB (43%) | 585 KiB (43%) | 945 KiB (58%) | 1.5 MiB (75%) | 2.8 MiB (88%) | 5.5 MiB (88%) |
| CN2 | NList | 2.0 MiB (354%) | 3.8 MiB (369%) | 6.2 MiB (465%) | 10.8 MiB (681%) | 21.8 MiB (1097%) | 39.9 MiB (1236%) | 72.1 MiB (1159%) |
| CN2 | PrePost | 1.6 MiB (283%) | 3.1 MiB (301%) | 4.4 MiB (327%) | 7.4 MiB (466%) | 14.9 MiB (751%) | 27.6 MiB (857%) | 50.7 MiB (815%) |
| CN2 | Sparse | 1.3 MiB (238%) | 2.8 MiB (269%) | 5.8 MiB (436%) | 12.2 MiB (767%) | 24.5 MiB (1234%) | 49.1 MiB (1522%) | 98.6 MiB (1586%) |
| PFossil | HBitmap | 520 KiB | 1019 KiB | 1.2 MiB | 1.3 MiB | 1.6 MiB | 2.0 MiB | 3.8 MiB |
| PFossil | VBitmap | 137 KiB (26%) | 216 KiB (21%) | 385 KiB (31%) | 637 KiB (47%) | 1.1 MiB (71%) | 2.1 MiB (103%) | 4.0 MiB (106%) |
| PFossil | NList | 1.9 MiB (380%) | 3.8 MiB (382%) | 6.2 MiB (514%) | 10.8 MiB (815%) | 21.8 MiB (1406%) | 39.9 MiB (1985%) | 72.1 MiB (1900%) |
| PFossil | PrePost | 1.5 MiB (302%) | 3.1 MiB (311%) | 4.3 MiB (360%) | 7.4 MiB (558%) | 14.9 MiB (963%) | 27.6 MiB (1376%) | 50.7 MiB (1336%) |
| PFossil | Sparse | 1.3 MiB (251%) | 2.7 MiB (276%) | 5.8 MiB (482%) | 12.0 MiB (908%) | 24.4 MiB (1570%) | 48.7 MiB (2425%) | 98.0 MiB (2582%) |
| Pypper | HBitmap | 433 KiB | 745 KiB | 1.2 MiB | 1.3 MiB | 1.5 MiB | 2.2 MiB | 4.3 MiB |
| Pypper | VBitmap | 150 KiB (35%) | 205 KiB (28%) | 372 KiB (31%) | 629 KiB (48%) | 1.1 MiB (72%) | 2.1 MiB (97%) | 4.2 MiB (98%) |
| Pypper | NList | 1.7 MiB (400%) | 3.1 MiB (430%) | 6.1 MiB (524%) | 10.8 MiB (834%) | 21.8 MiB (1416%) | 39.7 MiB (1845%) | 71.9 MiB (1686%) |
| Pypper | PrePost | 1.3 MiB (307%) | 2.4 MiB (333%) | 4.3 MiB (366%) | 7.4 MiB (570%) | 14.9 MiB (968%) | 27.5 MiB (1276%) | 50.4 MiB (1183%) |
| Pypper | Sparse | 1.1 MiB (253%) | 2.2 MiB (307%) | 4.7 MiB (404%) | 9.8 MiB (759%) | 19.8 MiB (1289%) | 39.6 MiB (1836%) | 79.6 MiB (1868%) |
| PyLORD | HBitmap | 459 KiB | 1.1 MiB | 4.1 MiB | 13.8 MiB | 54.0 MiB |  |  |
| PyLORD | VBitmap | 353 KiB (77%) | 1.1 MiB (97%) | 3.7 MiB (92%) | 13.8 MiB (100%) | 54.0 MiB (100%) |  |  |
| PyLORD | NList | 2.0 MiB (446%) | 4.0 MiB (368%) | 7.4 MiB (182%) | 20.6 MiB (149%) | 67.7 MiB (125%) |  |  |
| PyLORD | PrePost | 1.6 MiB (358%) | 3.3 MiB (302%) | 5.6 MiB (137%) | 17.4 MiB (126%) | 61.0 MiB (113%) |  |  |
| PyLORD | Sparse | 1.4 MiB (305%) | 2.9 MiB (272%) | 6.2 MiB (153%) | 18.8 MiB (136%) | 64.2 MiB (119%) |  |  |
| ENDER | HBitmap | 990 KiB | 1.1 MiB | 1.7 MiB | 3.3 MiB | 3.9 MiB | 5.1 MiB | 7.6 MiB |
| ENDER | VBitmap | 944 KiB (95%) | 1.0 MiB (95%) | 1.6 MiB (91%) | 3.1 MiB (95%) | 6.1 MiB (159%) | 11.7 MiB (229%) | 14.2 MiB (187%) |
| ENDER | NList | 1.9 MiB (192%) | 2.8 MiB (252%) | 5.8 MiB (338%) | 11.1 MiB (339%) | 22.4 MiB (580%) | 41.0 MiB (805%) | 74.5 MiB (983%) |
| ENDER | PrePost | 1.5 MiB (151%) | 2.0 MiB (187%) | 3.9 MiB (230%) | 7.7 MiB (235%) | 15.5 MiB (402%) | 28.8 MiB (565%) | 53.1 MiB (700%) |
| ENDER | Sparse | 1.5 MiB (153%) | 2.2 MiB (201%) | 3.7 MiB (216%) | 7.5 MiB (230%) | 15.0 MiB (389%) | 29.9 MiB (586%) | 60.0 MiB (791%) |

Without negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 431 KiB | 760 KiB | 1.2 MiB | 1.5 MiB | 1.9 MiB | 2.5 MiB | 5.3 MiB |
| CN2 | VBitmap | 241 KiB (56%) | 335 KiB (44%) | 451 KiB (37%) | 734 KiB (47%) | 1.2 MiB (63%) | 2.1 MiB (85%) | 4.5 MiB (85%) |
| CN2 | NList | 1.0 MiB (248%) | 2.0 MiB (276%) | 3.6 MiB (299%) | 4.8 MiB (308%) | 7.4 MiB (392%) | 12.4 MiB (500%) | 22.1 MiB (420%) |
| CN2 | PrePost | 999 KiB (231%) | 1.9 MiB (258%) | 3.3 MiB (279%) | 4.3 MiB (279%) | 6.4 MiB (340%) | 10.7 MiB (433%) | 19.4 MiB (369%) |
| CN2 | Sparse | 650 KiB (151%) | 1.2 MiB (163%) | 2.4 MiB (202%) | 4.8 MiB (311%) | 9.5 MiB (503%) | 18.9 MiB (762%) | 37.6 MiB (714%) |
| PFossil | HBitmap | 388 KiB | 712 KiB | 1.1 MiB | 1.2 MiB | 1.4 MiB | 1.8 MiB | 2.7 MiB |
| PFossil | VBitmap | 109 KiB (28%) | 152 KiB (21%) | 271 KiB (23%) | 469 KiB (37%) | 813 KiB (55%) | 1.5 MiB (82%) | 2.9 MiB (106%) |
| PFossil | NList | 1.0 MiB (267%) | 2.0 MiB (289%) | 3.5 MiB (310%) | 4.6 MiB (370%) | 7.1 MiB (495%) | 12.4 MiB (688%) | 22.1 MiB (821%) |
| PFossil | PrePost | 960 KiB (247%) | 1.9 MiB (269%) | 3.3 MiB (289%) | 4.2 MiB (334%) | 6.1 MiB (426%) | 10.7 MiB (596%) | 19.4 MiB (721%) |
| PFossil | Sparse | 605 KiB (156%) | 1.2 MiB (167%) | 2.3 MiB (207%) | 4.7 MiB (374%) | 9.2 MiB (642%) | 18.4 MiB (1024%) | 36.8 MiB (1367%) |
| Pypper | HBitmap | 307 KiB | 531 KiB | 1016 KiB | 1.2 MiB | 1.4 MiB | 1.9 MiB | 3.8 MiB |
| Pypper | VBitmap | 106 KiB (34%) | 152 KiB (29%) | 276 KiB (27%) | 516 KiB (42%) | 993 KiB (68%) | 1.9 MiB (99%) | 3.8 MiB (100%) |
| Pypper | NList | 846 KiB (275%) | 1.6 MiB (300%) | 3.1 MiB (311%) | 4.5 MiB (374%) | 7.0 MiB (492%) | 12.3 MiB (638%) | 21.9 MiB (574%) |
| Pypper | PrePost | 768 KiB (250%) | 1.4 MiB (273%) | 2.8 MiB (286%) | 4.1 MiB (337%) | 6.0 MiB (423%) | 10.6 MiB (551%) | 19.2 MiB (503%) |
| Pypper | Sparse | 503 KiB (164%) | 974 KiB (183%) | 1.9 MiB (194%) | 3.8 MiB (313%) | 7.5 MiB (531%) | 15.0 MiB (780%) | 30.0 MiB (786%) |
| PyLORD | HBitmap | 360 KiB | 1.1 MiB | 3.8 MiB | 13.6 MiB | 52.1 MiB |  |  |
| PyLORD | VBitmap | 342 KiB (95%) | 1.1 MiB (98%) | 3.7 MiB (99%) | 13.5 MiB (100%) | 52.1 MiB (100%) | 196.8 MiB |  |
| PyLORD | NList | 1.1 MiB (306%) | 2.2 MiB (202%) | 4.7 MiB (126%) | 15.4 MiB (114%) | 56.2 MiB (108%) | 204.0 MiB |  |
| PyLORD | PrePost | 1022 KiB (284%) | 2.0 MiB (189%) | 4.5 MiB (120%) | 15.0 MiB (110%) | 55.1 MiB (106%) |  |  |
| PyLORD | Sparse | 698 KiB (194%) | 1.6 MiB (144%) | 4.7 MiB (125%) | 15.5 MiB (114%) | 55.8 MiB (107%) |  |  |
| ENDER | HBitmap | 843 KiB | 908 KiB | 1.2 MiB | 2.4 MiB | 3.8 MiB | 4.9 MiB | 7.2 MiB |
| ENDER | VBitmap | 815 KiB (97%) | 878 KiB (97%) | 1.1 MiB (92%) | 2.2 MiB (92%) | 4.4 MiB (117%) | 8.7 MiB (177%) | 13.8 MiB (192%) |
| ENDER | NList | 1.1 MiB (134%) | 1.4 MiB (159%) | 2.2 MiB (174%) | 4.1 MiB (169%) | 7.6 MiB (202%) | 13.6 MiB (278%) | 24.5 MiB (341%) |
| ENDER | PrePost | 1.0 MiB (126%) | 1.3 MiB (145%) | 1.9 MiB (154%) | 3.6 MiB (150%) | 6.6 MiB (176%) | 11.9 MiB (243%) | 21.8 MiB (304%) |
| ENDER | Sparse | 1.1 MiB (129%) | 1.4 MiB (153%) | 1.9 MiB (157%) | 3.2 MiB (133%) | 6.3 MiB (168%) | 12.4 MiB (254%) | 24.7 MiB (344%) |

Peak memory while building (with negation) -- the vertical indexes need more than they keep:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 19 KiB | 36 KiB | 70 KiB | 138 KiB | 279 KiB | 552 KiB | 1.1 MiB |
| VBitmap | 143 KiB (742%) | 306 KiB (857%) | 645 KiB (924%) | 1.3 MiB (970%) | 2.6 MiB (972%) | 5.2 MiB (972%) | 9.5 MiB (889%) |
| NList | 3.9 MiB (20885%) | 7.6 MiB (21910%) | 16.2 MiB (23703%) | 31.6 MiB (23410%) | 63.9 MiB (23465%) | 127.0 MiB (23545%) | 255.7 MiB (23823%) |
| PrePost | 3.4 MiB (18339%) | 7.1 MiB (20297%) | 15.0 MiB (22027%) | 31.4 MiB (23241%) | 63.5 MiB (23336%) | 127.0 MiB (23545%) | 255.7 MiB (23823%) |
| Sparse | 698 KiB (3625%) | 1.2 MiB (3306%) | 2.5 MiB (3601%) | 5.1 MiB (3806%) | 10.4 MiB (3825%) | 20.8 MiB (3860%) | 41.9 MiB (3907%) |
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
| CN2 | HBitmap | 1.14 | 1.04 | 1.24 | 1.72 | 0.95 | 0.84 | 0.18 | 0.11 |
| CN2 | VBitmap | 0.95 (84%) | 0.87 (83%) | 1.02 (82%) | 1.43 (83%) | 0.73 (78%) | 0.60 (72%) | 0.13 (70%) | 0.07 (67%) |
| CN2 | NList | 1.09 (96%) | 1.00 (96%) | 1.17 (94%) | 1.62 (94%) | 0.88 (93%) | 0.78 (94%) | 0.17 (92%) | 0.09 (86%) |
| CN2 | PrePost | 1.09 (96%) | 1.00 (96%) | 1.15 (93%) | 1.58 (92%) | 0.86 (91%) | 0.76 (91%) | 0.16 (89%) | 0.09 (83%) |
| CN2 | Sparse | 1.12 (98%) | 1.01 (96%) | 1.16 (93%) | 1.60 (93%) | 0.85 (90%) | 0.71 (85%) | 0.16 (85%) | 0.08 (79%) |
| PFossil | HBitmap | 0.03 | 0.03 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| PFossil | VBitmap | 0.02 (65%) | 0.02 (64%) | 0.01 (61%) | 0.01 (61%) | 0.01 (57%) | 0.01 (63%) | 0.01 (54%) | 0.01 (54%) |
| PFossil | NList | 0.04 (107%) | 0.03 (99%) | 0.01 (106%) | 0.01 (100%) | 0.01 (99%) | 0.01 (97%) | 0.01 (89%) | 0.01 (79%) |
| PFossil | PrePost | 0.06 (168%) | 0.05 (137%) | 0.02 (148%) | 0.02 (137%) | 0.01 (107%) | 0.01 (100%) | 0.01 (86%) | 0.01 (79%) |
| PFossil | Sparse | 0.03 (92%) | 0.03 (85%) | 0.01 (72%) | 0.01 (72%) | 0.01 (65%) | 0.01 (68%) | 0.01 (61%) | 0.01 (62%) |
| Pypper | HBitmap | 0.10 | 0.06 | 0.04 | 0.02 | 0.09 | 0.03 | 0.06 | 0.03 |
| Pypper | VBitmap | 0.04 (42%) | 0.03 (47%) | 0.02 (53%) | 0.01 (65%) | 0.03 (37%) | 0.01 (56%) | 0.02 (45%) | 0.01 (55%) |
| Pypper | NList | 0.09 (88%) | 0.05 (88%) | 0.03 (87%) | 0.02 (99%) | 0.06 (66%) | 0.02 (83%) | 0.04 (71%) | 0.02 (81%) |
| Pypper | PrePost | 0.11 (105%) | 0.06 (110%) | 0.03 (97%) | 0.02 (112%) | 0.06 (63%) | 0.02 (84%) | 0.04 (67%) | 0.02 (80%) |
| Pypper | Sparse | 0.07 (67%) | 0.04 (66%) | 0.02 (65%) | 0.01 (75%) | 0.04 (46%) | 0.02 (67%) | 0.03 (54%) | 0.02 (67%) |
| PyLORD | HBitmap | 3.64 | 2.52 | 1.71 | 1.63 | 1.55 | 1.43 | 1.32 | 1.12 |
| PyLORD | VBitmap | 2.10 (58%) | 1.39 (55%) | 0.95 (55%) | 0.97 (60%) | 0.98 (63%) | 0.93 (65%) | 0.89 (67%) | 0.76 (68%) |
| PyLORD | NList | 3.55 (97%) | 2.52 (100%) | 1.91 (112%) | 1.94 (119%) | 1.93 (125%) | 1.44 (101%) | 1.32 (100%) | 1.14 (102%) |
| PyLORD | PrePost | 6.72 (184%) | 4.50 (179%) | 2.53 (148%) | 2.23 (137%) | 2.02 (130%) | 1.49 (104%) | 1.34 (102%) | 1.12 (100%) |
| PyLORD | Sparse | 3.57 (98%) | 2.29 (91%) | 1.45 (85%) | 1.42 (87%) | 1.37 (89%) | 1.27 (89%) | 1.19 (90%) | 1.03 (91%) |
| ENDER | HBitmap | 0.95 | 0.78 | 0.70 | 0.66 | 0.64 | 0.63 | 0.63 | 0.62 |
| ENDER | VBitmap | 0.85 (89%) | 0.70 (89%) | 0.63 (91%) | 0.60 (90%) | 0.57 (89%) | 0.56 (90%) | 0.56 (89%) | 0.56 (90%) |
| ENDER | NList | 0.89 (94%) | 0.72 (92%) | 0.63 (91%) | 0.59 (89%) | 0.56 (88%) | 0.59 (94%) | 0.57 (92%) | 0.57 (92%) |
| ENDER | PrePost | 0.89 (94%) | 0.72 (91%) | 0.62 (90%) | 0.58 (88%) | 0.55 (87%) | 0.58 (92%) | 0.57 (91%) | 0.56 (91%) |
| ENDER | Sparse | 0.93 (98%) | 0.73 (93%) | 0.63 (90%) | 0.58 (88%) | 0.55 (86%) | 0.54 (86%) | 0.54 (86%) | 0.53 (86%) |

Memory held by each representation (share of HBitmap in parentheses; the last row is the unpacked bool matrix):

![memory vs. density](representations_plots/density_memory.png)

| representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB |
| VBitmap | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) |
| NList | 16.9 MiB (27943%) | 10.1 MiB (16816%) | 3.4 MiB (5637%) | 1.7 MiB (2809%) | 709 KiB (1147%) | 376 KiB (608%) | 208 KiB (337%) | 111 KiB (180%) |
| PrePost | 4.7 MiB (7811%) | 2.8 MiB (4720%) | 997 KiB (1615%) | 512 KiB (829%) | 227 KiB (368%) | 135 KiB (218%) | 88 KiB (143%) | 61 KiB (99%) |
| Sparse | 2.4 MiB (3962%) | 1.4 MiB (2390%) | 509 KiB (824%) | 264 KiB (428%) | 119 KiB (193%) | 72 KiB (117%) | 48 KiB (77%) | 33 KiB (54%) |
| unpacked `X` | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass):

| learner | representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 1.6 MiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | 1.3 MiB | 1.3 MiB | 1.2 MiB | 1.2 MiB |
| CN2 | VBitmap | 1.2 MiB (72%) | 1.2 MiB (87%) | 1.2 MiB (85%) | 1.1 MiB (79%) | 796 KiB (59%) | 753 KiB (56%) | 654 KiB (51%) | 471 KiB (38%) |
| CN2 | NList | 20.3 MiB (1234%) | 13.1 MiB (961%) | 6.4 MiB (470%) | 4.8 MiB (332%) | 3.7 MiB (276%) | 1.2 MiB (88%) | 865 KiB (68%) | 485 KiB (39%) |
| CN2 | PrePost | 8.1 MiB (495%) | 5.8 MiB (426%) | 4.0 MiB (293%) | 3.6 MiB (250%) | 3.2 MiB (241%) | 937 KiB (70%) | 743 KiB (58%) | 433 KiB (35%) |
| CN2 | Sparse | 5.7 MiB (345%) | 3.5 MiB (257%) | 1.6 MiB (120%) | 1.4 MiB (95%) | 857 KiB (63%) | 779 KiB (58%) | 644 KiB (51%) | 374 KiB (30%) |
| PFossil | HBitmap | 1.2 MiB | 1.2 MiB | 1.2 MiB | 1.2 MiB | 1.2 MiB | 1.2 MiB | 1.2 MiB | 1.2 MiB |
| PFossil | VBitmap | 406 KiB (32%) | 404 KiB (32%) | 403 KiB (33%) | 403 KiB (33%) | 403 KiB (33%) | 401 KiB (33%) | 396 KiB (32%) | 358 KiB (29%) |
| PFossil | NList | 19.7 MiB (1618%) | 13.0 MiB (1068%) | 6.3 MiB (526%) | 4.6 MiB (382%) | 3.6 MiB (299%) | 702 KiB (57%) | 482 KiB (39%) | 370 KiB (30%) |
| PFossil | PrePost | 7.6 MiB (623%) | 5.7 MiB (470%) | 3.9 MiB (323%) | 3.4 MiB (283%) | 3.1 MiB (260%) | 461 KiB (38%) | 359 KiB (29%) | 277 KiB (23%) |
| PFossil | Sparse | 5.6 MiB (459%) | 3.4 MiB (281%) | 1.2 MiB (104%) | 714 KiB (58%) | 381 KiB (31%) | 272 KiB (22%) | 244 KiB (20%) | 208 KiB (17%) |
| Pypper | HBitmap | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB |
| Pypper | VBitmap | 415 KiB (36%) | 413 KiB (37%) | 412 KiB (37%) | 409 KiB (37%) | 409 KiB (35%) | 380 KiB (33%) | 353 KiB (31%) | 352 KiB (31%) |
| Pypper | NList | 19.7 MiB (1740%) | 13.0 MiB (1189%) | 6.3 MiB (572%) | 4.6 MiB (418%) | 3.6 MiB (312%) | 664 KiB (58%) | 477 KiB (41%) | 364 KiB (32%) |
| Pypper | PrePost | 7.6 MiB (669%) | 5.7 MiB (522%) | 3.8 MiB (350%) | 3.4 MiB (309%) | 3.1 MiB (271%) | 422 KiB (37%) | 326 KiB (28%) | 281 KiB (24%) |
| Pypper | Sparse | 4.6 MiB (403%) | 2.8 MiB (256%) | 1.0 MiB (94%) | 598 KiB (54%) | 336 KiB (29%) | 258 KiB (22%) | 215 KiB (19%) | 210 KiB (18%) |
| PyLORD | HBitmap | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.4 MiB | 1.6 MiB | 1.6 MiB | 1.5 MiB | 1.2 MiB |
| PyLORD | VBitmap | 1.1 MiB (98%) | 1.1 MiB (99%) | 1.1 MiB (99%) | 1.4 MiB (99%) | 1.6 MiB (99%) | 1.6 MiB (99%) | 1.5 MiB (99%) | 1.2 MiB (98%) |
| PyLORD | NList | 20.0 MiB (1791%) | 13.2 MiB (1183%) | 6.4 MiB (565%) | 4.8 MiB (339%) | 3.8 MiB (241%) | 2.0 MiB (128%) | 1.8 MiB (116%) | 1.3 MiB (109%) |
| PyLORD | PrePost | 7.9 MiB (704%) | 6.1 MiB (542%) | 4.0 MiB (352%) | 3.6 MiB (255%) | 3.3 MiB (211%) | 1.8 MiB (113%) | 1.6 MiB (108%) | 1.3 MiB (105%) |
| PyLORD | Sparse | 5.9 MiB (525%) | 3.6 MiB (325%) | 1.6 MiB (142%) | 1.7 MiB (118%) | 1.7 MiB (107%) | 1.6 MiB (104%) | 1.5 MiB (101%) | 1.2 MiB (99%) |
| ENDER | HBitmap | 1.6 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB |
| ENDER | VBitmap | 1.4 MiB (91%) | 1.4 MiB (92%) | 1.4 MiB (91%) | 1.4 MiB (91%) | 1.4 MiB (91%) | 1.4 MiB (91%) | 1.4 MiB (91%) | 1.4 MiB (91%) |
| ENDER | NList | 19.4 MiB (1244%) | 12.0 MiB (767%) | 5.3 MiB (334%) | 3.6 MiB (226%) | 2.6 MiB (162%) | 916 KiB (57%) | 695 KiB (43%) | 567 KiB (35%) |
| ENDER | PrePost | 7.3 MiB (467%) | 4.7 MiB (300%) | 2.8 MiB (180%) | 2.4 MiB (150%) | 2.1 MiB (132%) | 669 KiB (42%) | 584 KiB (36%) | 520 KiB (32%) |
| ENDER | Sparse | 3.5 MiB (225%) | 2.2 MiB (143%) | 1.1 MiB (70%) | 798 KiB (49%) | 595 KiB (37%) | 507 KiB (32%) | 478 KiB (30%) | 446 KiB (28%) |

## 3. Width

Synthetic data again, 1000 training and 1000 test examples, the same concept on 4 signal features -- but now the number of *true* features per example stays fixed (on average 16 noise features, plus the signal ones) while the number of features grows. That is how sparse data usually looks: one-hot encodings of attributes with many values, or words in a text. The bitmaps grow with the number of features (one bit per cell); a sparse representation with the number of true cells, which stays the same. `Sparse` keeps two copies of a 4-byte index plus a 1-byte value per true cell, about 10 bytes against 1/8 byte per cell for the bitmaps, so it should take less memory below a density of about 1/80 -- here from about 1,408 features on.

![memory vs. width](representations_plots/width_memory.png)

Memory held by each representation (share of HBitmap in parentheses; the last row is the unpacked bool matrix):

| representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 13 KiB | 25 KiB | 62 KiB | 122 KiB | 244 KiB | 611 KiB | 1.2 MiB | 2.4 MiB |
| VBitmap | 13 KiB (103%) | 26 KiB (105%) | 63 KiB (102%) | 126 KiB (103%) | 251 KiB (103%) | 626 KiB (102%) | 1.2 MiB (102%) | 2.4 MiB (102%) |
| NList | 387 KiB (2996%) | 637 KiB (2585%) | 1.1 MiB (1846%) | 2.1 MiB (1754%) | 4.1 MiB (1703%) | 9.9 MiB (1652%) | 19.5 MiB (1634%) | 38.7 MiB (1624%) |
| PrePost | 329 KiB (2543%) | 337 KiB (1369%) | 347 KiB (562%) | 358 KiB (293%) | 376 KiB (154%) | 424 KiB (69%) | 503 KiB (41%) | 660 KiB (27%) |
| Sparse | 179 KiB (1383%) | 179 KiB (727%) | 181 KiB (292%) | 183 KiB (149%) | 187 KiB (76%) | 198 KiB (32%) | 218 KiB (18%) | 257 KiB (11%) |
| unpacked `X` | 98 KiB | 195 KiB | 488 KiB | 977 KiB | 1.9 MiB | 4.8 MiB | 9.5 MiB | 19.1 MiB |

800 `coverage()` calls for random rules of 1-4 conditions:

![coverage() time vs. width](representations_plots/width_coverage.png)

![fit time vs. width](representations_plots/width_fit.png)

Fit time in seconds:

| learner | representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 1.10 | 1.16 | 1.30 | 1.33 | 1.74 | 3.93 | 6.08 | 14.24 |
| CN2 | VBitmap | 0.87 (79%) | 0.92 (79%) | 1.04 (80%) | 0.99 (74%) | 0.99 (57%) | 1.60 (41%) | 2.30 (38%) | 3.71 (26%) |
| CN2 | NList | 1.11 (101%) | 1.10 (95%) | 1.21 (93%) | 1.13 (85%) | 1.10 (63%) | 1.57 (40%) | 2.19 (36%) | 3.43 (24%) |
| CN2 | PrePost | 1.09 (99%) | 1.09 (94%) | 1.20 (92%) | 1.12 (84%) | 1.09 (62%) | 1.56 (40%) | 2.19 (36%) | 3.39 (24%) |
| CN2 | Sparse | 1.17 (107%) | 1.13 (98%) | 1.17 (90%) | 1.07 (81%) | 1.04 (60%) | 1.52 (39%) | 2.13 (35%) | 3.32 (23%) |
| PFossil | HBitmap | 0.01 | 0.01 | 0.01 | 0.02 | 0.03 | 0.06 | 0.12 | 0.28 |
| PFossil | VBitmap | 0.01 (73%) | 0.01 (68%) | 0.01 (59%) | 0.01 (46%) | 0.01 (36%) | 0.02 (25%) | 0.02 (19%) | 0.04 (13%) |
| PFossil | NList | 0.01 (103%) | 0.01 (99%) | 0.01 (107%) | 0.01 (78%) | 0.02 (61%) | 0.03 (50%) | 0.05 (45%) | 0.09 (33%) |
| PFossil | PrePost | 0.01 (110%) | 0.01 (106%) | 0.01 (119%) | 0.02 (93%) | 0.03 (78%) | 0.04 (72%) | 0.08 (67%) | 0.14 (50%) |
| PFossil | Sparse | 0.01 (97%) | 0.01 (85%) | 0.01 (70%) | 0.01 (48%) | 0.01 (34%) | 0.01 (21%) | 0.02 (14%) | 0.03 (10%) |
| Pypper | HBitmap | 0.06 | 0.02 | 0.04 | 0.04 | 0.03 | 0.54 | 0.52 | 0.95 |
| Pypper | VBitmap | 0.03 (44%) | 0.01 (66%) | 0.02 (46%) | 0.02 (45%) | 0.02 (45%) | 0.13 (24%) | 0.12 (23%) | 0.16 (17%) |
| Pypper | NList | 0.05 (79%) | 0.02 (93%) | 0.04 (78%) | 0.03 (68%) | 0.02 (72%) | 0.15 (29%) | 0.13 (26%) | 0.19 (20%) |
| Pypper | PrePost | 0.04 (74%) | 0.02 (94%) | 0.04 (82%) | 0.03 (73%) | 0.03 (87%) | 0.16 (29%) | 0.16 (30%) | 0.23 (24%) |
| Pypper | Sparse | 0.04 (61%) | 0.02 (80%) | 0.03 (57%) | 0.02 (50%) | 0.02 (45%) | 0.11 (20%) | 0.09 (18%) | 0.11 (11%) |
| PyLORD | HBitmap | 1.72 | 1.48 | 1.60 | 1.49 | 1.28 | 1.31 | 1.62 | 2.09 |
| PyLORD | VBitmap | 1.08 (62%) | 0.91 (62%) | 0.99 (62%) | 0.91 (61%) | 0.77 (60%) | 0.75 (57%) | 0.82 (51%) | 0.95 (45%) |
| PyLORD | NList | 1.72 (100%) | 1.51 (102%) | 1.95 (122%) | 1.45 (97%) | 1.18 (92%) | 1.07 (82%) | 1.11 (69%) | 1.23 (59%) |
| PyLORD | PrePost | 1.88 (109%) | 1.75 (118%) | 2.15 (135%) | 1.67 (112%) | 1.37 (107%) | 1.26 (96%) | 1.31 (81%) | 1.44 (69%) |
| PyLORD | Sparse | 1.66 (96%) | 1.36 (92%) | 1.41 (88%) | 1.27 (85%) | 1.04 (81%) | 0.93 (71%) | 0.96 (59%) | 1.03 (49%) |
| ENDER | HBitmap | 0.49 | 0.54 | 0.65 | 0.87 | 1.28 | 2.43 | 4.64 | 10.20 |
| ENDER | VBitmap | 0.45 (91%) | 0.48 (90%) | 0.59 (90%) | 0.79 (90%) | 1.16 (91%) | 2.28 (94%) | 4.24 (92%) | 8.19 (80%) |
| ENDER | NList | 0.49 (100%) | 0.52 (96%) | 0.58 (88%) | 0.64 (74%) | 0.69 (54%) | 0.84 (34%) | 1.10 (24%) | 1.65 (16%) |
| ENDER | PrePost | 0.48 (98%) | 0.51 (94%) | 0.57 (88%) | 0.63 (73%) | 0.69 (54%) | 0.83 (34%) | 1.11 (24%) | 1.70 (17%) |
| ENDER | Sparse | 0.54 (110%) | 0.55 (103%) | 0.57 (88%) | 0.59 (68%) | 0.64 (50%) | 0.78 (32%) | 1.02 (22%) | 1.54 (15%) |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass):

| learner | representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 622 KiB | 990 KiB | 1.4 MiB | 1.8 MiB | 3.2 MiB | 7.3 MiB | 12.1 MiB | 23.9 MiB |
| CN2 | VBitmap | 404 KiB (65%) | 567 KiB (57%) | 986 KiB (69%) | 1.4 MiB (80%) | 3.2 MiB (100%) | 7.4 MiB (100%) | 12.1 MiB (100%) | 23.9 MiB (100%) |
| CN2 | NList | 1.6 MiB (263%) | 2.8 MiB (291%) | 4.2 MiB (299%) | 3.6 MiB (206%) | 7.2 MiB (228%) | 16.8 MiB (229%) | 30.6 MiB (253%) | 60.5 MiB (253%) |
| CN2 | PrePost | 1.5 MiB (254%) | 2.5 MiB (261%) | 3.4 MiB (244%) | 1.9 MiB (106%) | 3.5 MiB (111%) | 7.4 MiB (100%) | 11.6 MiB (96%) | 22.4 MiB (94%) |
| CN2 | Sparse | 643 KiB (103%) | 745 KiB (75%) | 1.1 MiB (80%) | 1.5 MiB (84%) | 3.1 MiB (98%) | 6.9 MiB (94%) | 11.1 MiB (92%) | 21.7 MiB (91%) |
| PFossil | HBitmap | 470 KiB | 824 KiB | 1.2 MiB | 1.4 MiB | 1.8 MiB | 3.2 MiB | 4.8 MiB | 8.9 MiB |
| PFossil | VBitmap | 129 KiB (28%) | 175 KiB (21%) | 403 KiB (33%) | 689 KiB (48%) | 1.3 MiB (73%) | 3.5 MiB (107%) | 5.7 MiB (120%) | 11.1 MiB (125%) |
| PFossil | NList | 1.4 MiB (316%) | 2.6 MiB (329%) | 4.0 MiB (335%) | 2.7 MiB (192%) | 5.0 MiB (276%) | 11.9 MiB (368%) | 22.7 MiB (478%) | 45.3 MiB (509%) |
| PFossil | PrePost | 1.4 MiB (303%) | 2.4 MiB (293%) | 3.2 MiB (270%) | 976 KiB (68%) | 1.3 MiB (72%) | 2.5 MiB (76%) | 3.7 MiB (79%) | 7.3 MiB (81%) |
| PFossil | Sparse | 477 KiB (101%) | 480 KiB (58%) | 522 KiB (43%) | 555 KiB (39%) | 924 KiB (50%) | 2.0 MiB (62%) | 3.2 MiB (68%) | 6.6 MiB (75%) |
| Pypper | HBitmap | 364 KiB | 601 KiB | 1.1 MiB | 1.3 MiB | 1.6 MiB | 2.7 MiB | 4.4 MiB | 9.8 MiB |
| Pypper | VBitmap | 136 KiB (37%) | 181 KiB (30%) | 412 KiB (37%) | 685 KiB (51%) | 1.2 MiB (76%) | 2.9 MiB (105%) | 5.1 MiB (115%) | 11.1 MiB (113%) |
| Pypper | NList | 1.2 MiB (324%) | 2.1 MiB (351%) | 4.0 MiB (365%) | 2.6 MiB (200%) | 4.9 MiB (305%) | 11.7 MiB (430%) | 22.4 MiB (504%) | 45.7 MiB (464%) |
| Pypper | PrePost | 1.1 MiB (307%) | 1.8 MiB (301%) | 3.2 MiB (294%) | 893 KiB (67%) | 1.2 MiB (75%) | 2.3 MiB (84%) | 3.4 MiB (77%) | 7.6 MiB (77%) |
| Pypper | Sparse | 402 KiB (110%) | 401 KiB (67%) | 447 KiB (40%) | 505 KiB (38%) | 805 KiB (49%) | 1.8 MiB (67%) | 2.9 MiB (65%) | 7.0 MiB (71%) |
| PyLORD | HBitmap | 1.4 MiB | 1.2 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 2.7 MiB | 4.8 MiB | 8.7 MiB |
| PyLORD | VBitmap | 1.4 MiB (99%) | 1.2 MiB (100%) | 1.5 MiB (99%) | 1.5 MiB (100%) | 1.5 MiB (99%) | 2.0 MiB (76%) | 2.9 MiB (60%) | 4.6 MiB (53%) |
| PyLORD | NList | 1.9 MiB (140%) | 2.8 MiB (234%) | 4.2 MiB (279%) | 3.7 MiB (252%) | 5.6 MiB (365%) | 11.5 MiB (427%) | 21.4 MiB (443%) | 41.1 MiB (474%) |
| PyLORD | PrePost | 1.8 MiB (130%) | 2.5 MiB (209%) | 3.4 MiB (227%) | 2.0 MiB (134%) | 1.9 MiB (124%) | 2.1 MiB (78%) | 2.4 MiB (50%) | 3.0 MiB (35%) |
| PyLORD | Sparse | 1.6 MiB (114%) | 1.4 MiB (117%) | 1.7 MiB (111%) | 1.6 MiB (108%) | 1.5 MiB (98%) | 1.7 MiB (61%) | 1.9 MiB (40%) | 2.4 MiB (28%) |
| ENDER | HBitmap | 530 KiB | 786 KiB | 1.6 MiB | 2.9 MiB | 3.3 MiB | 4.7 MiB | 6.2 MiB | 10.7 MiB |
| ENDER | VBitmap | 486 KiB (92%) | 728 KiB (93%) | 1.4 MiB (91%) | 2.6 MiB (91%) | 5.1 MiB (153%) | 10.8 MiB (229%) | 12.1 MiB (195%) | 25.0 MiB (234%) |
| ENDER | NList | 978 KiB (184%) | 1.5 MiB (198%) | 3.0 MiB (189%) | 2.8 MiB (98%) | 4.9 MiB (149%) | 11.6 MiB (248%) | 22.1 MiB (356%) | 44.5 MiB (416%) |
| ENDER | PrePost | 921 KiB (174%) | 1.2 MiB (160%) | 2.2 MiB (139%) | 1.1 MiB (37%) | 1.3 MiB (38%) | 2.2 MiB (47%) | 3.1 MiB (50%) | 6.4 MiB (60%) |
| ENDER | Sparse | 632 KiB (119%) | 656 KiB (84%) | 689 KiB (43%) | 696 KiB (24%) | 883 KiB (26%) | 1.8 MiB (38%) | 2.6 MiB (42%) | 5.8 MiB (54%) |

