"""
demos/noise_robustness.py
=========================

How rule learners cope with class noise, on four two-class datasets whose
concepts are free or nearly free of noise (`FULL_DATASETS`). A share of the
training labels (`NOISE_LEVELS`) is replaced by the other class; the test
folds keep their labels.

The question behind it: PFoil, whose only pruning is FOIL's encoding-length
restriction, is as accurate as the learners that prune much more (Pypper,
PFossil). The restriction allows a rule as many literals as it takes to
encode its covered positives -- for all but the rules covering a handful of
examples more than any rule needs -- so PFoil learns nearly consistent
rules, which suits noise-free data. With noise, it should learn ever more
rules and lose accuracy faster than the others. Next to the
separate-and-conquer learners, with and without their stopping criteria,
the demo has boosted rules (known to suffer from class noise, the
exponential loss more than the logistic one), bagged and boosted PFoil
concept sets, and PyLORD, which learns one rule per training example.

Learners: `LEARNERS`, grouped by `GROUPS`.

Run: `python demos/noise_robustness.py` with no arguments is the **quick**
default (`QUICK_DATASETS`, `QUICK_NOISE_LEVELS`, `QUICK_FOLDS`-fold);
`--full` runs `FULL_DATASETS` at all `NOISE_LEVELS`, `N_FOLDS`-fold, and
writes ``noise_robustness_report.md`` (plots in ``noise_robustness_plots/``).
Needs the `experiments` extra.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import format_accuracy, format_mean_count, render_results_table, \
    render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.learners.adaboost import AdaBoost
from pyrulearn.learners.bagging import Bagging
from pyrulearn.learners.base import NativeRuleLearner
from pyrulearn.learners.boosting import ENDER, Boomer, ExponentialLoss, Slipper
from pyrulearn.learners.multiclass import OneVsRest
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.learners.seco import CN2, PFoil, PFossil, Pypper

RANDOM_STATE = 0
N_FOLDS = 10
MAX_INTERVALS = 8
FIT_TIMEOUT = 1800.0
#: members of the bagged and boosted ensembles
N_MEMBERS = 10

#: name -> the learner (default settings unless the name says otherwise)
LEARNERS = {
    "CN2": lambda: CN2(),
    "CN2 (no stopping)": lambda: CN2(significance_threshold=None),
    "PFoil": lambda: PFoil(),
    "PFoil (no stopping)": lambda: PFoil(mdl_stopping=False),
    "PFossil": lambda: PFossil(),
    "Pypper": lambda: Pypper(),
    "PyLORD": lambda: PyLORD(random_state=RANDOM_STATE),
    "Slipper": lambda: Slipper(random_state=RANDOM_STATE),
    "ENDER": lambda: ENDER(random_state=RANDOM_STATE),
    "ENDER (exponential loss)": lambda: ENDER(loss=ExponentialLoss(), random_state=RANDOM_STATE),
    "Boomer": lambda: Boomer(random_state=RANDOM_STATE),
    "Bagging:PFoil": lambda: Bagging(OneVsRest(PFoil()), n_estimators=N_MEMBERS, random_state=RANDOM_STATE),
    "AdaBoost:PFoil": lambda: AdaBoost(OneVsRest(PFoil()), n_estimators=N_MEMBERS, early_stopping=False,
                                       random_state=RANDOM_STATE),
}
#: group -> its learners, in the order of the report
GROUPS = {
    "separate-and-conquer": ["CN2", "CN2 (no stopping)", "PFoil", "PFoil (no stopping)", "PFossil", "Pypper"],
    "local rules": ["PyLORD"],
    "boosted rules": ["Slipper", "ENDER", "ENDER (exponential loss)", "Boomer"],
    "ensembles of concept sets": ["Bagging:PFoil", "AdaBoost:PFoil"],
}
#: drawn in grey in the plots of the other groups: the two poles of separate-and-conquer
REFERENCES = ["PFoil", "Pypper"]
#: shares of the training labels replaced by the other class
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]
QUICK_NOISE_LEVELS = [0.0, 0.1, 0.3]

# concepts free or nearly free of noise: a board game, an endgame, a field guide's species,
# votes along party lines
FULL_DATASETS = ["tic-tac-toe", "vote", "kr-vs-kp", "mushroom"]
QUICK_DATASETS = ["tic-tac-toe", "vote"]
QUICK_FOLDS = 3

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "noise_robustness"
CACHE_DIR = os.path.join(HERE, f"_{NAME}_cache")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")

# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


def add_class_noise(y: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """`y` with a share `rate` of its labels (rounded to whole examples)
    replaced by a different class, chosen uniformly."""
    y = np.asarray(y)
    classes = np.unique(y)
    noisy = y.copy()
    for i in rng.choice(len(y), size=int(round(rate * len(y))), replace=False):
        noisy[i] = rng.choice(classes[classes != y[i]])
    return noisy


class Noisy(NativeRuleLearner):
    """The learner named `name` (`LEARNERS`), fit on the training data with
    a share `rate` of its labels replaced. The noise depends only on the
    training data and `RANDOM_STATE`, so every learner of a fold sees the
    same noisy labels."""

    def __init__(self, name: str, rate: float):
        self.name = name
        self.rate = rate

    @property
    def display_name(self) -> str:
        return f"{self.name}@{self.rate:.0%}"

    def _provenance_params(self):
        params = {"name": self.name, "rate": self.rate, "random_state": RANDOM_STATE}
        return {**params, "members": N_MEMBERS} if self.name.startswith(("Bagging", "AdaBoost")) else params

    def fit(self, data):
        if self.rate > 0:
            data = data.relabel(add_class_noise(data.y, self.rate, np.random.default_rng(RANDOM_STATE)))
        return LEARNERS[self.name]().fit(data)


def build_learners(levels):
    return [Noisy(name, rate) for rate in levels for name in LEARNERS]


def _split(results: pd.DataFrame) -> pd.DataFrame:
    """`name` and `noise` columns from the learner's display name."""
    parts = results["learner"].str.rsplit("@", n=1, expand=True)
    return results.assign(name=parts[0], noise=parts[1].str.rstrip("%").astype(float) / 100)


def _description(datasets, quick: bool, n_folds: int, levels) -> str:
    mode = ("**Quick run** -- a fast sanity check, the default with no arguments. Full run: "
            f"`python demos/{NAME}.py --full`.\n\n" if quick else
            f"**Full run.** Quick sanity check instead: `python demos/{NAME}.py`.\n\n")
    names = ", ".join(f"`{d.name}`" for d in datasets)
    return f"""\
{mode}Rule learners under class noise: in every training fold, a share of the
labels ({', '.join(f'{r:.0%}' for r in levels)}) is replaced by the other
class; the test folds keep their labels, so the accuracy is that of the true
concept. Every learner of a fold sees the same noisy labels.

Separate-and-conquer:

- **CN2** -- beam search with the Laplace estimate, its significance test
  as the stopping criterion; rules for every class. **CN2 (no stopping)**:
  without the test.
- **PFoil** -- FOIL gain, greedy search, Quinlan's encoding-length
  restriction as the stopping criterion; rules for every class.
  **PFoil (no stopping)**: without the restriction, rules grow until they
  cover no negatives.
- **PFossil** -- correlation, rules below a correlation of 0.3 dropped.
- **Pypper** -- RIPPER: grow and prune (IREP*), then optimize the rule set.

Local rules: **PyLORD** -- one locally optimal rule per training example.

Boosted rules: **Slipper** (AdaBoost's reweighting after every rule),
**ENDER** (logistic loss), **ENDER (exponential loss)** (AdaBoost's loss,
which weights the misclassified examples exponentially) and **Boomer**
(logistic loss, regularized Newton steps), each with its default number of
rules.

Ensembles of concept sets: **Bagging:PFoil** and **AdaBoost:PFoil**, {N_MEMBERS}
PFoil concept sets each, from bootstrap samples or by SAMME (restarting from
a bootstrap sample where SAMME would stop); members abstain where their
rules don't cover an example.

Datasets, two-class, with concepts free or nearly free of noise: {names}.
Protocol: {n_folds}-fold stratified cross-validation, one `DataSpec` per
training fold (`TreeDiscretizer(max_intervals={MAX_INTERVALS})`), each fit capped
at {FIT_TIMEOUT:.0f}s.
"""


def _by_noise(res: pd.DataFrame, names, column: str, title: str, fmt) -> list:
    """Mean `column` over the datasets, per learner of `names` and noise
    level."""
    levels = sorted(res["noise"].unique())
    means = res.groupby(["name", "noise"])[column].mean()
    L = [f"{title}:\n\n| learner | " + " | ".join(f"{r:.0%}" for r in levels) + " |\n|---|"
         + "--:|" * len(levels) + "\n"]
    for name in names:
        L.append(f"| {name} | " + " | ".join(fmt(means.get((name, r), np.nan)) for r in levels) + " |\n")
    L.append("\n")
    return L


def _seconds(v) -> str:
    return "--" if v != v else f"{v:.2f}"


def _slug(group: str) -> str:
    return group.replace(" ", "_").replace("-", "_")


def _plot(res: pd.DataFrame, group: str, quick: bool) -> list:
    """Accuracy, number of rules and fit time against the noise level for
    the learners of `group`, with `REFERENCES` from the other groups as thin
    grey lines."""
    path = os.path.join(PLOTS_DIR, f"{NAME}{'_quick' if quick else ''}_{_slug(group)}.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter

        means = res.groupby(["name", "noise"])[["accuracy", "n_rules", "fit_time"]].mean()
        present = set(means.index.get_level_values(0))
        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        columns = ["accuracy", "n_rules", "fit_time"]
        references = [n for n in REFERENCES if n not in GROUPS[group] and n in present]
        for name, style in zip(references, ["-", "--"]):
            m = means.loc[name]
            for ax, column in zip(axes, columns):
                ax.plot(m.index, m[column], style, color="0.6", lw=1, label=f"{name} (reference)")
        for name, marker in zip(GROUPS[group], "osD^vP"):
            if name in present:
                m = means.loc[name]
                for ax, column in zip(axes, columns):
                    ax.plot(m.index, m[column], "-", marker=marker, markersize=4, label=name)
        for ax in axes:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_xlabel("class noise in the training data")
            ax.grid(alpha=0.3, which="both")
        axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[0].set_ylabel("accuracy (mean)")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("rules (mean, log scale)")
        axes[2].set_yscale("log")
        axes[2].set_ylabel("fit time in s (mean, log scale)")
        for ax in axes[1:]:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"Plot   -> {path}")
        return [f"![{group}: accuracy, rules and fit time against class noise]({os.path.basename(PLOTS_DIR)}/"
                f"{os.path.basename(path)})\n\n"]
    except Exception as e:  # noqa: BLE001 -- a plot failure shouldn't sink the rest of the report
        print(f"Plot skipped: {type(e).__name__}: {e}")
        return []


def write_report(results, datasets, quick: bool, n_folds: int, levels, report_path: str) -> None:
    res = _split(results)
    L = ["# Rule learners under class noise\n\n"]
    L.append(render_setup_section(_description(datasets, quick, n_folds, levels)))
    failed = res[res["error"].notna()].drop_duplicates(["dataset", "learner"])
    if len(failed):
        L.append("Left out, as a fold failed (the whole cross-validation of that learner is then counted as "
                 "failed): " + ", ".join(f"`{r.dataset}` ({r.learner}: {r.error})" for r in failed.itertuples())
                 + ".\n\n")
    done = res[~res["dataset"].isin(failed["dataset"])]
    if len(failed):
        L.append(f"The summaries below are on the {done['dataset'].nunique()} datasets on which every learner "
                 "completed.\n\n")
    L.append("Means over the datasets. In the plots of the other groups, "
             + " and ".join(REFERENCES) + " are drawn as thin grey lines for reference.\n\n")
    for group, names in GROUPS.items():
        L.append(f"## {group[0].upper()}{group[1:]}\n\n")
        L += _plot(done, group, quick)
        L += _by_noise(done, names, "accuracy", "Accuracy", format_accuracy)
        L += _by_noise(done, names, "n_rules", "Rules", format_mean_count)
        L += _by_noise(done, names, "n_conditions", "Conditions", format_mean_count)
        L += _by_noise(done, names, "fit_time", "Fit time (seconds)", _seconds)
    L.append("## Per dataset (mean across folds)\n\n")
    L.append(render_results_table(res, ["accuracy", "n_rules", "n_conditions", "fit_time"],
                                  group_by=["dataset", "learner"], include_overall=False))
    L.append("\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def main(quick: bool = True) -> None:
    datasets = Catalog.default().select(names=QUICK_DATASETS if quick else FULL_DATASETS)
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    levels = QUICK_NOISE_LEVELS if quick else NOISE_LEVELS
    results = run_cv(build_learners(levels), datasets, n_folds=n_folds, fit_timeout=FIT_TIMEOUT,
                     discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS), random_state=RANDOM_STATE,
                     cache_dir=CACHE_DIR, n_jobs=_n_jobs())
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(results, datasets, quick, n_folds, levels, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"{len(FULL_DATASETS)} datasets at {len(NOISE_LEVELS)} noise levels, {N_FOLDS}-fold")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(quick=not args.full)
