"""
Native negation check: does handing a learner the negation features matter?
============================================================================

`build_dataspec` gives every attribute its negation features by default
(``x != v``, ``not f``). Learners differ in whether they need them:

- **Pypper** (and every native SeCo learner) only tests a feature for being
  true -- without the negation features, it cannot express a negation at
  all, so its rules should differ.
- **Weka:JRip** and **Witt:RIPPER** test every Boolean column for both
  values -- with the negation features they see each condition twice (as
  the negation feature, and as its partner tested for ``False``). With
  `native_negation=True` they get only the features that are not
  negations and provide those themselves. The conditions they can use are
  the same either way; whether the models are too is what this checks.
  (For JRip they need not be: Ripper's description lengths, which decide
  when to stop and how to optimize, count the possible conditions, and
  those double with the negation features.)

For each dataset and fold (built as `run_cv` builds them), each learner is
fitted with and without the negation features -- Pypper on the data
without them, JRip and Witt:RIPPER with `native_negation=True` -- and the
script prints how many test predictions differ, both accuracies and both
fit times, and a summary per learner. A quick check, nothing is written.

Run: `python demos/native_negation_check.py [datasets] [--folds K]`
(default: vote, breast-cancer, tic-tac-toe, credit-approval; 3 folds).
Needs Weka (see `ripper_comparison.py`'s `WEKA_JAR`/`WEKA_JAVA`) and the
`wittgenstein` extra.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ripper_comparison import MAX_INTERVALS, RANDOM_STATE, WEKA_JAR, WEKA_JAVA  # noqa: E402

from pyrulearn.data import TreeDiscretizer, default_representation  # noqa: E402
from pyrulearn.data.io import DEFAULT_MAX_ORDINAL_VALUES, build_dataspec, encode  # noqa: E402
from pyrulearn.experiments.catalog import Catalog  # noqa: E402
from pyrulearn.experiments.runner import _stratified_folds  # noqa: E402
from pyrulearn.interfaces.weka import WekaJRip  # noqa: E402
from pyrulearn.interfaces.wittgenstein import WittRIPPER  # noqa: E402
from pyrulearn.learners.seco import Pypper  # noqa: E402
from pyrulearn.models import ConceptModel  # noqa: E402

DEFAULT_DATASETS = "vote,breast-cancer,tic-tac-toe,credit-approval"


class _WittRIPPER(WittRIPPER):
    """Binary datasets only: `fit(data)` -> a `ConceptModel`."""

    def _default_model(self, data) -> type:
        return ConceptModel


#: name -> (learner with the negation features, learner without them,
#: whether "without" means the data without them (a native learner) or the
#: learner's own `native_negation`)
LEARNERS = {
    "Weka:JRip": (lambda: WekaJRip(jar=WEKA_JAR, java=WEKA_JAVA),
                  lambda: WekaJRip(jar=WEKA_JAR, java=WEKA_JAVA, native_negation=True), False),
    "Witt:RIPPER": (lambda: _WittRIPPER(random_state=RANDOM_STATE),
                    lambda: _WittRIPPER(random_state=RANDOM_STATE, native_negation=True), False),
    "Pypper": (lambda: Pypper(random_state=RANDOM_STATE), lambda: Pypper(random_state=RANDOM_STATE), True),
}


def main(names, n_folds: int) -> None:
    rep_cls = default_representation()
    totals = {name: [0, 0, 0] for name in LEARNERS}  # differing predictions, predictions, folds that differ
    for entry in Catalog.default().select(names=names):
        df, target = entry.load()
        for fold, (tr, te) in enumerate(_stratified_folds(df[target].to_numpy(), n_folds, RANDOM_STATE)):
            train_df = df.iloc[tr].reset_index(drop=True)
            test_df = df.iloc[te].reset_index(drop=True)
            spec = build_dataspec(train_df, target=target, discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS),
                                  max_ordinal_values=DEFAULT_MAX_ORDINAL_VALUES, skip_unusable=True).build()
            train = encode(spec, train_df, train_df[target].to_numpy(), representation=rep_cls)
            test = encode(spec, test_df, test_df[target].to_numpy(), representation=rep_cls)
            train_without, test_without = train.without_negations(), test.without_negations()
            y = np.asarray(test.y)
            for name, (make_with, make_without, reduced) in LEARNERS.items():
                t0 = time.perf_counter()
                with_neg = make_with().fit(train)
                t_with = time.perf_counter() - t0
                t0 = time.perf_counter()
                without = make_without().fit(train_without if reduced else train)
                t_without = time.perf_counter() - t0
                p_with = np.asarray(with_neg.predict(test))
                p_without = np.asarray(without.predict(test_without if reduced else test))
                diff = int((p_with != p_without).sum())
                totals[name][0] += diff
                totals[name][1] += len(y)
                totals[name][2] += diff > 0
                print(f"{entry.name:16s} fold {fold}  {name:12s} features {train.spec.n_features:4d} / "
                      f"{train_without.spec.n_features:4d}  differing predictions {diff:4d}/{len(y):<5d} "
                      f"accuracy {np.mean(p_with == y):.2%} / {np.mean(p_without == y):.2%}  "
                      f"fit {t_with:.2f}s / {t_without:.2f}s", flush=True)
    print("\nwith / without the negation features, over all datasets and folds:")
    for name, (diff, n, folds) in totals.items():
        print(f"  {name:12s} {diff} of {n} test predictions differ ({diff / n:.2%}), in {folds} folds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("datasets", nargs="?", default=DEFAULT_DATASETS,
                        help=f"comma-separated catalog names (default: {DEFAULT_DATASETS})")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    main(args.datasets.split(","), args.folds)
