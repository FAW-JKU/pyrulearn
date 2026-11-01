"""
demos/rule_models.py
====================

The same rule learners building different kinds of rule models: for each
base learner -- CN2, PFossil and Pypper -- the model types pyrulearn can
build from its covering loop, cross-validated through
`pyrulearn.experiments.runner.run_cv` on multi-class datasets of
`pyrulearn.experiments.catalog`, and in a second part on two-class ones.

Configurations (`CONFIGURATIONS`), each a `fit(data, model=...)` of the
base learner:

- **rule set** -- one-vs-rest (`ConceptSet`): a rule set per class, learned
  against all others; its conflicts resolved four ways on the same fitted
  model (`COMBINERS`): **best rule** (the covering rule with the best
  Laplace estimate -- the same as sorting the rules by it, an ordered rule
  set), **voting** (one vote per covering rule), **micro distribution
  voting** (the covering rules' class counts summed, CN2's own) and **macro
  distribution voting** (their class proportions summed, every rule
  weighing the same).
- **cascade (rarest first) / cascade (most frequent first)** -- ordered
  one-vs-rest (`ConceptCascade`): the classes learned one after the other,
  each against those not yet handled, the last one the default.
- **decision list (per refinement)** -- one covering loop over all
  classes, the class of each rule chosen by the heuristic at every
  refinement, so that it can change with each added condition
  (`target_class=HeadPerRefinement()`; CN2's ordered mode).
- **decision list (per rule)** -- the same loop, the class chosen once per
  rule: the best of one complete search per class
  (`target_class=HeadPerRule()`).
- **decision list (per seed example)** -- AQ's seed covering: the class of
  a random uncovered example, whose generalizations the search explores
  (`seed=RandomSeed()`).

(The configurations keep their earlier names internally -- "any class",
"each class", "seed" -- so that cached results stay valid; `LABELS` maps
them to these.)

Pairwise models are compared in `demos/multiclass_decomposition.py`.

A second part compares the models on two-class data (`BINARY_DATASETS`,
every small and medium two-class dataset of the catalog,
`BINARY_CONFIGURATIONS`): a concept for the **minority** class (the
majority the default, Ripper's choice), a concept for the **majority**
class (the minority the default), the **rule set** of both classes with
its four combiners, and the three **decision lists** of rules of either
class (per refinement, per rule, per seed example).

Measures: accuracy, numbers of rules and conditions, training time, mean
rank across datasets.

Run: `python demos/rule_models.py` with no arguments is the **quick**
default (`QUICK_DATASETS` and `QUICK_BINARY`, `QUICK_FOLDS`-fold); its report goes to
``rule_models_quick_report.md`` and is not checked in. `--full` runs
`FULL_DATASETS`, the large `LARGE_DATASETS` (with `LARGE_FIT_TIMEOUT`) and
`BINARY_DATASETS` (`N_FOLDS`-fold) and writes
``rule_models_report.md``. Needs the `experiments` extra.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from pyrulearn.combiners import CountVoteCombiner, HeuristicMaxCombiner, MacroVoteCombiner, MicroVoteCombiner
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import format_accuracy, format_mean_count, render_results_table, \
    render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import critical_difference, critical_difference_diagram, friedman_test, mean_rank
from pyrulearn.learners.base import NativeRuleLearner
from pyrulearn.learners.seco import (
    CN2, HeadPerRefinement, HeadPerRule, PFossil, Pypper, RandomSeed,
)
from pyrulearn.models import ConceptCascade, ConceptModel, ConceptSet, DecisionList

RANDOM_STATE = 0
N_FOLDS = 10
#: the two large multi-class datasets, with a longer cap per fit (one search per class and rule
#: makes "decision list (per rule)" expensive with many classes)
LARGE_DATASETS = ["kropt", "letter"]
LARGE_FIT_TIMEOUT = 1800.0
MAX_INTERVALS = 8
FIT_TIMEOUT = 900.0

BASES = {"CN2": CN2, "PFossil": PFossil, "Pypper": Pypper}

#: name -> (constructor arguments, fit arguments)
CONFIGURATIONS = {
    "rule set": ({}, {"model": ConceptSet}),
    "cascade (rarest first)": ({}, {"model": ConceptCascade, "order": "least_frequent"}),
    "cascade (most frequent first)": ({}, {"model": ConceptCascade, "order": "most_frequent"}),
    "decision list (any class)": ({}, {"model": DecisionList, "target_class": HeadPerRefinement()}),
    "decision list (each class)": ({}, {"model": DecisionList, "target_class": HeadPerRule()}),
    "decision list (seed)": ({}, {"model": DecisionList, "seed": RandomSeed()}),
}
#: the second part, on two-class data: a concept's class ("minority" or
#: "majority") is resolved at fit time, the other class its default
BINARY_CONFIGURATIONS = {
    "concept (minority class)": ({}, {"model": ConceptModel, "label": "minority"}),
    "concept (majority class)": ({}, {"model": ConceptModel, "label": "majority"}),
    "rule set": ({}, {"model": ConceptSet}),
    **{name: CONFIGURATIONS[name] for name in ("decision list (any class)", "decision list (each class)",
                                               "decision list (seed)")},
}
#: configurations a base learner can't be set up for
UNSUPPORTED = set()

#: the report's (and the paper's) names of configurations whose internal name -- part of the
#: cache key -- differs: the class of a decision list's rule chosen per refinement, per rule or per
#: seed example
LABELS = {"decision list (any class)": "decision list (per refinement)",
          "decision list (each class)": "decision list (per rule)",
          "decision list (seed)": "decision list (per seed example)"}

#: a rule set's conflict resolution, all on the same fitted model (named as in the paper)
COMBINERS = {"best rule": HeuristicMaxCombiner, "voting": CountVoteCombiner,
             "micro distribution voting": MicroVoteCombiner, "macro distribution voting": MacroVoteCombiner}

# small multi-class datasets
QUICK_DATASETS = ["car", "vehicle", "zoo", "soybean", "primary-tumor"]
QUICK_FOLDS = 3
# every small and medium multi-class dataset that is mostly symbolic or has
# few numeric attributes, from 3 to 24 classes
FULL_DATASETS = ["balance-scale", "cmc", "dna", "hayes-roth", "iris", "splice", "wine", "car", "lymph",
                 "vehicle", "anneal", "glass", "solar-flare", "ecoli", "segment", "zoo", "led24", "yeast",
                 "optdigits", "texture", "vowel", "soybean", "primary-tumor", "audiology"]
# every small and medium two-class dataset of the catalog
BINARY_DATASETS = ["SPECT", "Titanic", "banknote-authentication", "blood-transfusion-service-center",
                   "breast-cancer", "breast-w", "churn", "climate-model-simulation-crashes", "colic",
                   "compas-two-years", "credit-approval", "credit-g", "cylinder-bands", "diabetes",
                   "dresses-sales", "heart-c", "heart-h", "heart-statlog", "heloc", "hepatitis", "ilpd",
                   "ionosphere", "kc1", "kc2", "kr-vs-kp", "mofn-3-7-10", "molecular-biology_promoters",
                   "monks-problems-1", "monks-problems-2", "monks-problems-3", "mushroom", "ozone-level-8hr",
                   "pc1", "pc3", "pc4", "phoneme", "qsar-biodeg", "sick", "sonar", "spambase", "tic-tac-toe",
                   "vote", "wdbc", "wilt"]
QUICK_BINARY = ["breast-cancer", "credit-g", "tic-tac-toe", "vote"]

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "rule_models"
CACHE_DIR = os.path.join(HERE, "_rule_models_cache")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")

# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


class Configuration(NativeRuleLearner):
    """`base` building the model of `configuration`, named
    ``"<base>:<configuration>"``; records the time of its own fit on the
    returned model (`fit_seconds`)."""

    def __init__(self, base: str, configuration: str, random_state: int = RANDOM_STATE):
        self.base = base
        self.configuration = configuration
        self.random_state = random_state

    @property
    def display_name(self) -> str:
        return f"{self.base}:{self.configuration}"

    def _provenance_params(self):
        params = {"base": self.base, "configuration": self.configuration, "random_state": self.random_state}
        if self.configuration == "rule set":
            # its cached results hold one accuracy per combiner: a new combiner means fitting it again
            params["combiners"] = list(COMBINERS)
        return params

    def fit(self, data):
        construct, fit = {**CONFIGURATIONS, **BINARY_CONFIGURATIONS}[self.configuration]
        learner = BASES[self.base](random_state=self.random_state, **construct)
        if fit.get("label") in ("minority", "majority"):
            labels, counts = np.unique(np.asarray(data.y), return_counts=True)
            order = labels[np.argsort(counts, kind="stable")]   # minority first
            label, other = (order[0], order[-1]) if fit["label"] == "minority" else (order[-1], order[0])
            fit = {**fit, "label": label, "fallback": other}
        t0 = time.perf_counter()
        model = learner.fit(data, **fit)
        model.fit_seconds = time.perf_counter() - t0
        return model


def build_learners(configurations=CONFIGURATIONS):
    return [Configuration(base, name) for base in BASES for name in configurations
            if (base, name) not in UNSUPPORTED]


def measure(model, test_rep) -> dict:
    """Training time from inside the worker, and the accuracy of a rule set
    under each combiner (`COMBINERS`)."""
    y = np.asarray(test_rep.y)
    out = {"train_time": getattr(model, "fit_seconds", np.nan)}
    if isinstance(model, ConceptSet):
        for name, combiner in COMBINERS.items():
            model.combiner = combiner()
            out[f"accuracy_{name}"] = float(np.mean(np.asarray(model.predict(test_rep)) == y))
    return out


def variants(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, fold, variant): a rule set expanded into one
    row per combiner, with that combiner's accuracy."""
    rows = []
    for _, r in results.iterrows():
        base, configuration = r["learner"].split(":", 1)
        common = {"dataset": r["dataset"], "fold": r["fold"], "base": base, "n_rules": r["n_rules"],
                  "n_conditions": r["n_conditions"], "train_time": r.get("train_time"), "error": r["error"]}
        if configuration == "rule set":
            for name in COMBINERS:
                variant = f"rule set ({name})"
                rows.append({**common, "learner": f"{base}:{variant}", "variant": variant,
                             "accuracy": r.get(f"accuracy_{name}")})
        else:
            variant = LABELS.get(configuration, configuration)
            rows.append({**common, "learner": f"{base}:{variant}", "variant": variant, "accuracy": r["accuracy"]})
    return pd.DataFrame(rows)


def _description(datasets, binary, quick: bool, n_folds: int) -> str:
    mode = ("**Quick run** -- a fast sanity check, the default with no arguments. Full run: "
            f"`python demos/{NAME}.py --full`.\n\n" if quick else
            f"**Full run.** Quick sanity check instead: `python demos/{NAME}.py`.\n\n")
    names = ", ".join(f"`{d.name}` ({d.n_classes})" for d in sorted(datasets, key=lambda d: d.n_classes))
    control = (f" Two-class datasets: {', '.join(f'`{d.name}`' for d in binary)}." if binary else "")
    big = [d.name for d in datasets if d.name in LARGE_DATASETS]
    large_cap = (f" ({LARGE_FIT_TIMEOUT:.0f}s for the large {' and '.join(f'`{n}`' for n in big)})"
                 if big else "")
    return f"""\
{mode}A rule learner's covering loop can build different kinds of models from
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

Multi-class datasets, with their numbers of classes: {names}.{control} Protocol:
{n_folds}-fold stratified cross-validation, one `DataSpec` per training fold
(`TreeDiscretizer(max_intervals={MAX_INTERVALS})`). Each fit is capped at
{FIT_TIMEOUT:.0f}s{large_cap}; a time-out fails the whole cross-validation of that
learner on that dataset. Training time is measured inside the worker process.
"""


def _summary(var: pd.DataFrame, bases) -> list:
    """Per base learner: the variants sorted by mean accuracy rank."""
    L = []
    for base in bases:
        v = var[var["base"] == base]
        if not len(v):
            continue
        ranks = mean_rank(v, "accuracy")
        means = v.groupby("learner")[["accuracy", "n_rules", "n_conditions", "train_time"]].mean(numeric_only=True)
        failures = v.groupby("learner")["error"].apply(lambda e: e.notna().sum())
        L.append(f"### {base}\n\n| model | accuracy | mean rank | rules | conditions | train (s) | failures |\n"
                 "|---|--:|--:|--:|--:|--:|--:|\n")
        for learner in ranks.sort_values().index:
            m = means.loc[learner]
            L.append(f"| {learner.split(':', 1)[1]} | {format_accuracy(m['accuracy'])} | {ranks[learner]:.2f} | "
                     f"{format_mean_count(m['n_rules'])} | {format_mean_count(m['n_conditions'])} | "
                     f"{m['train_time']:.2f} | {int(failures.get(learner, 0))} |\n")
        L.append("\n")
    return L


def _overall(var: pd.DataFrame, group: str, quick: bool) -> list:
    """The kinds of model across the base learners: each model ranked within
    each base learner on each dataset, the ranks averaged over all
    (dataset, base learner) pairs -- the mean of the per-learner mean ranks,
    since every base learner sees the same datasets. Only the kinds every
    base learner has. Then the Friedman test over those pairs and the
    critical-difference diagram."""
    n_bases = var["base"].nunique()
    common = [v for v, b in var.groupby("variant")["base"].nunique().items() if b == n_bases]
    left_out = sorted(set(var["variant"]) - set(common))
    blocks = var[var["variant"].isin(common)].assign(
        dataset=lambda d: d["dataset"] + " / " + d["base"], learner=lambda d: d["variant"])
    ranks = mean_rank(blocks, "accuracy")
    means = blocks.groupby("learner")["accuracy"].mean()
    L = ["### All base learners together\n\n"
         f"Each kind of model's mean rank, averaged over the {n_bases} base learners (ranked within each "
         "base learner on each dataset, as in the tables above), and its accuracy averaged over them"
         + (f"; left out, since not every base learner has it: {', '.join(left_out)}" if left_out else "")
         + ".\n\n| model | accuracy | mean rank |\n|---|--:|--:|\n"]
    for learner in ranks.index:
        L.append(f"| {learner} | {format_accuracy(means[learner])} | {ranks[learner]:.2f} |\n")
    for variant in left_out:
        # ranked where it exists: among all kinds of model of the base learners that have it
        bases = sorted(var.loc[var["variant"] == variant, "base"].unique())
        own = var[var["base"].isin(bases)].assign(
            dataset=lambda d: d["dataset"] + " / " + d["base"], learner=lambda d: d["variant"])
        n_kinds = own["variant"].nunique()
        L.append(f"\n{variant[0].upper() + variant[1:]}, which only {' and '.join(bases)} have: mean rank "
                 f"{mean_rank(own, 'accuracy')[variant]:.2f} among their {n_kinds} kinds of model (the ranks "
                 f"above are among {len(common)}), accuracy "
                 f"{format_accuracy(own.loc[own['variant'] == variant, 'accuracy'].mean())}.\n")
    try:
        f = friedman_test(blocks, "accuracy")
        cd = critical_difference(f.n_learners, f.n_datasets)
        L.append(f"\nFriedman test over the {f.n_datasets} (dataset, base learner) pairs: p = {f.pvalue:.2g} "
                 f"({f.n_learners} kinds of model). Nemenyi critical difference at 5%: {cd:.2f} -- two kinds "
                 "whose mean ranks differ by less are not significantly different; the diagram joins them by "
                 "a bar. The pairs of one dataset aren't independent of each other, so the test is on the "
                 "optimistic side.\n\n")
    except ValueError as exc:
        L.append(f"\nNo Friedman test ({exc}).\n\n")
    path = os.path.join(PLOTS_DIR, f"{NAME}_{'quick_' if quick else ''}cd_{group}.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(PLOTS_DIR, exist_ok=True)
        ax = critical_difference_diagram(blocks, "accuracy")
        ax.figure.tight_layout()
        ax.figure.savefig(path, dpi=110)
        plt.close(ax.figure)
        L.append(f"![critical-difference diagram, {group}]({os.path.basename(PLOTS_DIR)}/{os.path.basename(path)})\n\n")
        print(f"Plot   -> {path}")
    except Exception as e:  # noqa: BLE001 -- a plot failure shouldn't sink the rest of the report
        print(f"Critical-difference diagram ({group}) skipped: {type(e).__name__}: {e}")
    return L


#: the classes groups of `_by_classes`: (label, lowest, highest)
CLASS_GROUPS = [("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-9", 6, 9), ("10-26", 10, 26)]


def _by_classes(var: pd.DataFrame, n_classes: dict) -> list:
    """Decision lists against rule sets over the number of classes: per
    dataset and base learner, the accuracy of each decision list minus that
    of the rule set resolved by the best rule, and of the best decision list
    minus the best rule set, averaged within groups of datasets with similar
    numbers of classes; then the share of pairs on which the decision list
    is ahead, and the rank correlation with the number of classes, over all
    datasets and over the multi-class ones alone."""
    from scipy.stats import spearmanr

    acc = var.groupby(["dataset", "base", "variant"])["accuracy"].mean().unstack("variant")
    if "rule set (best rule)" not in acc:
        return []
    lists = [c for c in acc.columns if c.startswith("decision list")]
    diff = pd.DataFrame({c: acc[c] - acc["rule set (best rule)"] for c in lists})
    diff["best decision list - best rule set"] = (
        acc[lists].max(axis=1) - acc[[c for c in acc.columns if c.startswith("rule set")]].max(axis=1))
    diff = 100 * diff.dropna()
    k = diff.index.get_level_values("dataset").map(n_classes).to_numpy()
    cols = list(diff.columns)
    L = ["## Decision lists against rule sets, over the number of classes\n\n"
         "Per dataset and base learner, the accuracy of a decision list minus that of the rule set resolved "
         "by the best rule, in percentage points (positive: the decision list is better), and of the best "
         "decision list minus the best rule set, averaged over the datasets with the given numbers of "
         "classes and the three base learners.\n\n"
         "| classes | pairs | "
         + " | ".join("best list - best set" if c.startswith("best") else c.replace("decision list ", "list ")
                      for c in cols) + " |\n"
         "|---|--:|" + "--:|" * len(cols) + "\n"]
    for label, lo, hi in CLASS_GROUPS:
        sel = (k >= lo) & (k <= hi)
        if sel.any():
            L.append(f"| {label} | {int(sel.sum())} | "
                     + " | ".join(f"{diff.loc[sel, c].mean():+.2f}" for c in cols) + " |\n")
    multi = k > 2
    L.append("\nShare of the pairs on which the decision list is better, and Spearman's rank correlation of "
             "the difference with the number of classes:\n\n"
             "| | ahead, two classes | ahead, more classes | correlation, all | correlation, more classes |\n"
             "|---|--:|--:|--:|--:|\n")
    for c in cols:
        ahead2 = (diff.loc[~multi, c] > 0).mean() if (~multi).any() else float("nan")
        aheadm = (diff.loc[multi, c] > 0).mean() if multi.any() else float("nan")
        rho_all = spearmanr(k, diff[c])
        rho_m = spearmanr(k[multi], diff.loc[multi, c]) if multi.sum() > 2 else None
        L.append(f"| {c} | {100 * ahead2:.0f}% | {100 * aheadm:.0f}% | {rho_all.statistic:+.2f} "
                 f"(p = {rho_all.pvalue:.2g}) | "
                 + (f"{rho_m.statistic:+.2f} (p = {rho_m.pvalue:.2g})" if rho_m is not None else "--") + " |\n")
    L.append("\n")
    return L


def write_report(var, datasets, binary, quick: bool, n_folds: int, report_path: str) -> None:
    L = ["# Rule models: rule sets, cascades, decision lists\n\n"]
    L.append(render_setup_section(_description(datasets, binary, quick, n_folds)))
    multi = var[var["dataset"].isin([d.name for d in datasets])]
    L.append("## Multi-class datasets\n\n"
             "Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all "
             "of a model's rules together; the four rule-set variants share their rules and training "
             "time.\n\n")
    L += _summary(multi, BASES)
    L += _overall(multi, "multiclass", quick)
    if binary:
        two = var[var["dataset"].isin([d.name for d in binary])]
        L.append("## Two-class datasets\n\n"
                 "A concept for one class, the other the default; the rule set of both classes; the "
                 "decision lists of rules of either class.\n\n")
        L += _summary(two, BASES)
        L += _overall(two, "twoclass", quick)
    L += _by_classes(var, {d.name: d.n_classes for d in list(datasets) + list(binary)})
    L.append("## Per dataset (mean across folds)\n\n")
    L.append(render_results_table(var, ["accuracy", "n_rules", "train_time"], group_by=["dataset", "learner"],
                                  include_overall=False))
    L.append("\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def main(quick: bool = True) -> None:
    catalog = Catalog.default()
    datasets = catalog.select(names=QUICK_DATASETS if quick else FULL_DATASETS)
    large = [] if quick else catalog.select(names=LARGE_DATASETS)
    binary = catalog.select(names=QUICK_BINARY if quick else BINARY_DATASETS)
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    common = dict(discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS), random_state=RANDOM_STATE,
                  cache_dir=CACHE_DIR, measure_fn=measure, n_jobs=_n_jobs())
    runs = [run_cv(build_learners(CONFIGURATIONS), datasets, n_folds=n_folds, fit_timeout=FIT_TIMEOUT, **common),
            run_cv(build_learners(BINARY_CONFIGURATIONS), binary, n_folds=n_folds, fit_timeout=FIT_TIMEOUT, **common)]
    if large:
        runs.append(run_cv(build_learners(CONFIGURATIONS), large, n_folds=n_folds,
                           fit_timeout=LARGE_FIT_TIMEOUT, **common))
    results = pd.concat(runs, ignore_index=True)
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(variants(results), list(datasets) + list(large), binary, quick, n_folds, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"{len(FULL_DATASETS)} multi-class and {len(BINARY_DATASETS)} two-class datasets, "
                             f"{N_FOLDS}-fold")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(quick=not args.full)
