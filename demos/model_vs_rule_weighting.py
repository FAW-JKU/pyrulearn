"""
demos/model_vs_rule_weighting.py
================================

Weighting models against weighting rules: two ways to learn a weighted
ensemble of rules, compared by accuracy against the number of rules.
Cross-validated through `pyrulearn.experiments.runner.run_cv` on the
datasets of `demos/base_vs_composite.py`.

- **Model weighting** -- `pyrulearn.learners.boosting.AdaBoost` (SAMME,
  the multi-class AdaBoost) over concept sets, one concept description per
  class learned one-vs-rest (`OneVsRest`) by `Pypper`, `PFossil` or
  `PFoil`: each member is a whole rule set, learned from the examples
  reweighted by the errors of its predecessors, and the members vote with
  weights. Any learner can be boosted this way. Each fit is scored with two
  votes (`DEFAULTS`). As in `demos/base_vs_composite.py`, a member whose
  rules don't cover an example abstains (`abstain`, the default): the members' own
  defaults are ignored, and examples no member covers go to the majority
  class. Or the defaults vote too, with the member's weight.
- **Rule weighting** -- `Slipper`, `ENDER` and `Boomer`: each boosting
  round adds a single rule with a weight, and a prediction sums the
  weights of the covering rules. This is unique to rule learning: a rule,
  unlike a whole model, abstains where it doesn't cover an example, so
  that a sum of rules is a model of its own.

`Slipper` reweights the examples like AdaBoost, after each rule instead of
after each rule set; `ENDER` (here with its default, the logistic loss)
and `Boomer` (the logistic loss with Newton steps) fit each rule's weight
to the gradient of a loss.

Configurations (`LEARNERS`): each learner fit once per fold at its largest
size, and scored at every size of `STAGES` from its first rounds
(`EnsembleModel.staged`, `LinearRuleModel.staged`) -- boosting never
changes earlier rounds, so these are the models a fit with fewer rounds
builds. A round is a member for AdaBoost, a rule for `ENDER` and `Boomer`,
and a rule per class for `Slipper` (one boosting run per class). AdaBoost
doesn't stop early (`early_stopping=False`): where SAMME would stop -- at a
member no better than random guessing, which is left out, or one without
training errors, which is kept -- the example weights are reset to those of
a bootstrap sample, and boosting goes on until it has all its members.

Measures: accuracy and number of rules at every size, mean rank across
datasets.

Run: `python demos/model_vs_rule_weighting.py` with no arguments is the
**quick** default (`QUICK_DATASETS`, `QUICK_FOLDS`-fold); its report goes to
``model_vs_rule_weighting_quick_report.md`` and is not checked in. `--full`
runs the multi-class and two-class datasets of `demos/base_vs_composite.py`,
`N_FOLDS`-fold, and writes ``model_vs_rule_weighting_report.md``. Needs the
`experiments` extra.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from base_vs_composite import BINARY_DATASETS, FULL_DATASETS, QUICK_DATASETS
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import format_accuracy, format_mean_count, render_results_table, \
    render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import mean_rank
from pyrulearn.learners.boosting import AdaBoost
from pyrulearn.learners.base import NativeRuleLearner
from pyrulearn.learners.boosting import ENDER, Boomer, Slipper
from pyrulearn.learners.multiclass import OneVsRest
from pyrulearn.learners.seco import PFoil, PFossil, Pypper

RANDOM_STATE = 0
N_FOLDS = 10
MAX_INTERVALS = 8
FIT_TIMEOUT = 1800.0
#: the largest dataset, with a longer cap per fit
LONG_DATASETS = ["letter"]
LONG_FIT_TIMEOUT = 3600.0

def _boosted(base, n):
    return AdaBoost(OneVsRest(base()), n_estimators=n, early_stopping=False)


#: name -> (strategy, the learner at `n` rounds)
LEARNERS = {
    "AdaBoost:Pypper": ("model weighting", lambda n: _boosted(Pypper, n)),
    "AdaBoost:PFossil": ("model weighting", lambda n: _boosted(PFossil, n)),
    "AdaBoost:PFoil": ("model weighting", lambda n: _boosted(PFoil, n)),
    "Slipper": ("rule weighting", lambda n: Slipper(n_rounds=n, random_state=RANDOM_STATE)),
    "ENDER": ("rule weighting", lambda n: ENDER(n_rules=n, random_state=RANDOM_STATE)),
    "Boomer": ("rule weighting", lambda n: Boomer(n_rules=n, random_state=RANDOM_STATE)),
}
#: name -> the numbers of rounds it is scored at; the largest is fit
STAGES = {
    "AdaBoost:Pypper": [1, 5, 10, 25],
    "AdaBoost:PFossil": [1, 5, 10, 25],
    "AdaBoost:PFoil": [1, 5, 10, 25],
    "Slipper": [5, 10, 25, 50, 100],
    "ENDER": [10, 25, 50, 100, 250, 500],
    "Boomer": [10, 25, 50, 100, 250, 500],
}
#: marks the curve of an AdaBoost fit scored with the members' defaults voting
DEFAULTS = " (defaults vote)"
#: name in the report -> (the fit it is scored from, strategy)
CURVES = {}
for name, (strategy, _) in LEARNERS.items():
    CURVES[name] = (name, strategy)
    if strategy == "model weighting":
        CURVES[name + DEFAULTS] = (name, strategy)

QUICK_FOLDS = 3

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "model_vs_rule_weighting"
CACHE_DIR = os.path.join(HERE, f"_{NAME}_cache")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")

# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


class Staged(NativeRuleLearner):
    """The learner named `name` (`LEARNERS`), fit at the largest of its
    `STAGES`; the stages ride along as `stages`, for `measure`."""

    def __init__(self, name: str):
        self.name = name

    @property
    def display_name(self) -> str:
        return self.name

    def _provenance_params(self):
        # its cached results hold one accuracy per stage: other stages mean fitting it again
        params = {"name": self.name, "stages": STAGES[self.name], "random_state": RANDOM_STATE,
                  "early_stopping": False}
        return {**params, "votes": "both"} if LEARNERS[self.name][0] == "model weighting" else params

    def fit(self, data):
        model = LEARNERS[self.name][1](max(STAGES[self.name])).fit(data)
        model.stages_scored = STAGES[self.name]
        return model


def build_learners():
    return [Staged(name) for name in LEARNERS]


def _prefix(model, n):
    """The model after `n` rounds -- for AdaBoost, which may have stopped
    early, at most its whole ensemble."""
    if hasattr(model, "members"):
        return model.staged(min(n, len(model.members)))
    return model.staged(n)


def measure(model, test_rep) -> dict:
    """Accuracy and number of rules after each number of rounds the model
    is scored at; for an ensemble, also with the members' defaults voting."""
    y = np.asarray(test_rep.y)
    out = {}
    for n in model.stages_scored:
        prefix = _prefix(model, n)
        out[f"accuracy@{n}"] = float(np.mean(np.asarray(prefix.predict(test_rep)) == y))
        out[f"rules@{n}"] = len(prefix.rules)
        out[f"members@{n}"] = len(prefix.members) if hasattr(prefix, "members") else np.nan
        if hasattr(prefix, "members"):
            prefix.abstain = False
            out[f"accuracy_defaults@{n}"] = float(np.mean(np.asarray(prefix.predict(test_rep)) == y))
    return out


def variants(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, fold, curve, number of rounds)."""
    rows = []
    for _, r in results.iterrows():
        for name, (fitted, strategy) in CURVES.items():
            if fitted != r["learner"]:
                continue
            key = "accuracy_defaults" if name.endswith(DEFAULTS) else "accuracy"
            for n in STAGES[fitted]:
                rows.append({"dataset": r["dataset"], "fold": r["fold"], "name": name,
                             "strategy": strategy, "rounds": n, "learner": f"{name}@{n}",
                             "accuracy": r.get(f"{key}@{n}"), "n_rules": r.get(f"rules@{n}"),
                             "members": r.get(f"members@{n}"),
                             "error": r["error"]})
    return pd.DataFrame(rows)


def _description(datasets, binary, quick: bool, n_folds: int) -> str:
    mode = ("**Quick run** -- a fast sanity check, the default with no arguments. Full run: "
            f"`python demos/{NAME}.py --full`.\n\n" if quick else
            f"**Full run.** Quick sanity check instead: `python demos/{NAME}.py`.\n\n")
    names = ", ".join(f"`{d.name}` ({d.n_classes})" for d in sorted(datasets, key=lambda d: d.n_classes))
    control = (f" Two-class datasets: {', '.join(f'`{d.name}`' for d in binary)}." if binary else "")
    stages = "; ".join(f"{name}: {', '.join(map(str, s))}" for name, s in STAGES.items())
    return f"""\
{mode}Two strategies for a weighted ensemble of rules:

- **model weighting** -- AdaBoost (SAMME) over concept sets learned
  one-vs-rest by Pypper, PFossil or PFoil: every member is a whole rule set,
  learned from the examples reweighted by its predecessors' errors, and the
  members vote with weights. This works with any learner. Every fit is
  scored with two votes. **AdaBoost:...**: as in
  `demos/base_vs_composite.py`, the concept sets vote without their defaults:
  a member whose rules don't cover an example abstains, and examples no
  member covers go to the majority class. **AdaBoost:...{DEFAULTS}**: such a
  member votes for its default class, with its full weight. The members are
  the same; a member's training error counts its default in both.
- **rule weighting** -- Slipper, ENDER and Boomer: every round adds one rule
  with a weight, and the weights of the covering rules are summed. This is
  unique to rule learning, as a rule abstains where it doesn't cover an
  example.

Slipper reweights the examples as AdaBoost does, but after every rule;
ENDER (logistic loss) and Boomer (logistic loss, Newton steps) fit every
rule's weight to the gradient of the loss. The two strategies don't learn
the same rules, so they are compared by accuracy against the number of
rules.

Each learner is fit once per fold at its largest size and scored after each
number of rounds (its first rounds are the model a fit with fewer rounds
builds): {stages}. A round is a member for AdaBoost, a rule for ENDER and
Boomer, and a rule per class for Slipper. Where SAMME would stop boosting --
at a member no better than random guessing (left out) or one without training
errors (kept) -- the example weights are reset to those of a bootstrap
sample, and boosting goes on.

Datasets, with their numbers of classes: {names}.{control} Protocol:
{n_folds}-fold stratified cross-validation, one `DataSpec` per training fold
(`TreeDiscretizer(max_intervals={MAX_INTERVALS})`). Each fit is capped at
{FIT_TIMEOUT:.0f}s ({LONG_FIT_TIMEOUT:.0f}s for {', '.join(f'`{n}`' for n in LONG_DATASETS)}).
"""


def _per_learner(var: pd.DataFrame) -> list:
    """Per learner and number of rounds: members kept (AdaBoost), rules
    and accuracy."""
    means = var.groupby(["name", "rounds"])[["members", "n_rules", "accuracy"]].mean()
    L = ["## Accuracy against the number of rules\n\n"
         "| strategy | learner | rounds | members | rules | accuracy |\n|---|---|--:|--:|--:|--:|\n"]
    for name, (fitted, strategy) in CURVES.items():
        for n in STAGES[fitted]:
            if (name, n) in means.index:
                m = means.loc[(name, n)]
                members = format_mean_count(m["members"]) if m["members"] == m["members"] else ""
                L.append(f"| {strategy} | {name} | {n} | {members} | {format_mean_count(m['n_rules'])} | "
                         f"{format_accuracy(m['accuracy'])} |\n")
    L.append("\nMembers: the AdaBoost members, on average -- fewer than the rounds only where even "
             "twice as many rounds didn't give members better than random guessing.\n\n")
    return L


def _largest(var: pd.DataFrame) -> list:
    """Every learner at its largest size: mean rank across datasets."""
    top = var[var["rounds"] == var["name"].map(lambda n: max(STAGES[CURVES[n][0]]))]
    ranks = mean_rank(top.assign(learner=top["name"]), "accuracy")
    means = top.groupby("name")[["n_rules", "accuracy"]].mean()
    L = ["## At the largest size\n\nEvery learner after its last round, sorted by mean rank (accuracy rank "
         "across datasets).\n\n| learner | strategy | rules | accuracy | mean rank |\n|---|---|--:|--:|--:|\n"]
    for name in ranks.index:
        L.append(f"| {name} | {CURVES[name][1]} | {format_mean_count(means.loc[name, 'n_rules'])} | "
                 f"{format_accuracy(means.loc[name, 'accuracy'])} | {ranks[name]:.2f} |\n")
    L.append("\n")
    return L


def _plot(var: pd.DataFrame, quick: bool) -> list:
    """Accuracy against the mean number of rules, one line per learner."""
    path = os.path.join(PLOTS_DIR, f"{NAME}{'_quick' if quick else ''}_accuracy_vs_rules.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        means = var.groupby(["name", "rounds"])[["n_rules", "accuracy"]].mean()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        # the two curves of an AdaBoost fit share its colour
        colors = dict(zip(LEARNERS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
        for name, (fitted, strategy) in CURVES.items():
            m = means.loc[name] if name in means.index.get_level_values(0) else None
            if m is None:
                continue
            ax.plot(m["n_rules"], m["accuracy"], ":^" if name.endswith(DEFAULTS) else
                    "--o" if strategy == "model weighting" else "-s",
                    color=colors[fitted], markersize=4, label=f"{name} ({strategy})")
        ax.set_xscale("log")
        ax.set_xlabel("rules (log scale)")
        ax.set_ylabel("accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"Plot   -> {path}")
        return [f"![accuracy against the number of rules]({os.path.basename(PLOTS_DIR)}/{os.path.basename(path)})\n\n"]
    except Exception as e:  # noqa: BLE001 -- a plot failure shouldn't sink the rest of the report
        print(f"Plot skipped: {type(e).__name__}: {e}")
        return []


def write_report(var, datasets, binary, quick: bool, n_folds: int, report_path: str) -> None:
    L = ["# Weighting models against weighting rules\n\n"]
    L.append(render_setup_section(_description(datasets, binary, quick, n_folds)))
    failed = var[var["error"].notna()].drop_duplicates(["dataset", "name"])
    if len(failed):
        L.append("Left out, as a fold failed (the whole cross-validation of that learner is then counted as "
                 "failed): " + ", ".join(f"`{r.dataset}` ({r.name}: {r.error})" for r in failed.itertuples())
                 + ".\n\n")
    done = var[~var["dataset"].isin(failed["dataset"])]
    if len(failed):
        L.append(f"The summaries below are on the {done['dataset'].nunique()} datasets on which every learner "
                 "completed.\n\n")
    L += _per_learner(done)
    L += _plot(done, quick)
    L += _largest(done)
    L.append("## Per dataset (mean across folds)\n\n")
    L.append(render_results_table(var, ["accuracy", "n_rules"], group_by=["dataset", "learner"],
                                  include_overall=False))
    L.append("\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def main(quick: bool = True) -> None:
    catalog = Catalog.default()
    datasets = [d for d in catalog.select(names=QUICK_DATASETS if quick else FULL_DATASETS) if d.n_classes > 2]
    binary = [d for d in catalog.select(names=QUICK_DATASETS if quick else BINARY_DATASETS) if d.n_classes == 2]
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    common = dict(n_folds=n_folds, discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS),
                  random_state=RANDOM_STATE, cache_dir=CACHE_DIR, measure_fn=measure, n_jobs=_n_jobs())
    everything = datasets + binary
    runs = []
    for group, timeout in (([d for d in everything if d.name not in LONG_DATASETS], FIT_TIMEOUT),
                           ([d for d in everything if d.name in LONG_DATASETS], LONG_FIT_TIMEOUT)):
        if group:
            runs.append(run_cv(build_learners(), group, fit_timeout=timeout, **common))
    results = pd.concat(runs, ignore_index=True)
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(variants(results), datasets, binary, quick, n_folds, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"{len(FULL_DATASETS)} multi-class and {len(BINARY_DATASETS)} two-class "
                             f"datasets, {N_FOLDS}-fold")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(quick=not args.full)
