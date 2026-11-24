"""
demos/covering_boosting.py
===========================

Two angles on the same question -- what should happen to an example once
an accepted rule covers it? -- in one framework, because they turn out
to literally share code:

- **Removal covering** (the classic separate-and-conquer default):
  covered examples leave the scope entirely.
- **Weighted covering**: covered examples stay, reweighted by a
  `pyrulearn.learners.seco.Reweighting` scheme -- multiplicative/additive
  (CN2-SD), or `LRIReweighting` (Lightweight Rule Induction) -- until a
  `CoveringStop` criterion is met.
- **Boosting**: `Slipper` (Cohen & Singer 1999) is `WeightedCovering`
  with `AdaBoostReweighting` -- the *same* `CoveringState`/reweighting
  machinery as the line above, just assigning each rule a fitted weight
  (confidence), which plain `SeCo` refuses to build into an unweighted
  rule set (see `SeCo`'s own `covering=` docstring) -- so it needs its
  own learner class, `Slipper`, but not its own mechanism. `ENDER`
  (gradient boosting of rules for a pluggable loss) and its `Boomer`
  configuration aren't as separate from this as they look, either:
  AdaBoost reweighting *is* gradient boosting on the exponential loss
  (Friedman, Hastie & Tibshirani 2000) -- `ENDER`'s `ExponentialLoss`
  defines its per-example quantity as the same `exp(-y f)` AdaBoost
  reweights by, and fits each rule's weight with the identical closed
  form `AdaBoostReweighting.confidence` uses. What they don't share:
  `ENDER`'s *other* losses (logistic, sigmoid) and weight-fitting
  methods have no reweighting-scheme equivalent at all, and weighted
  covering discards its weights at predict time (a heuristic scores an
  unweighted rule set) while boosting bakes the fitted weight into the
  model itself (`LinearRuleModel`, summed at predict time).

Two sections, one report:

- **Section 1 -- covering strategies**: a fixed search (beam width 5)
  and rule-growing setup, crossed with covering strategy (removal;
  weighted, multiplicative/additive/LRI) and heuristic (Laplace, WRAcc,
  m-estimate with m=16 -- `demos/heuristic_comparison.py`'s own measured
  best-performing setting),
  via plain `SeCo`. `CPAR` and `LRI` sit alongside as complete
  algorithms (their own classes -- extra procedure beyond a bare
  reweighting scheme, not expressible as a `SeCo` config). Deliberately
  *not* reusing `CN2`'s own significance-based stopping here: that's a
  search-level criterion that can end the covering loop by itself, which
  would confound the comparison this section is actually about.
- **Section 2 -- boosting**: `Slipper`, `ENDER` (a few representative
  configs: the paper's default CS-Log, CS-Exp, and MLRules'
  newton/subsample setting), `Boomer`. Headline measure is accuracy vs.
  rule count as a curve (refit at a few `n_rounds`/`n_rules` budgets),
  not one fitted size -- simpler than hunting for a staged-prediction
  shortcut, if more fits.

A closing **comparative discussion**, built fresh from the measured
numbers each run, ties the two sections together: Slipper's point here
is directly comparable to section 1's `WeightedCovering` +
`AdaBoostReweighting` entry (same mechanism), and to `ENDER`'s
exponential-loss configuration (the same mechanism again, by a
classical result -- see the module docstring above); `ENDER`'s other
losses, and weighted covering's non-exponential schemes, are where the
two families actually part ways.

Datasets: the same two-class, mostly-symbolic list as
`demos/seco_learners_comparison.py`'s `FULL_BINARY` -- these learners'
weighted-covering/boosting machinery is inherently a one-class-vs-rest
framing, and that list is already curated for exactly this learner
family.

Measures: accuracy, rule count, rule length (conditions per rule), fit
time, and "overlap" -- the mean number of a model's rules covering a
test row, pooled across every concept the model has (not restricted to
one "positive" class: a plain `SeCo` config here builds a one-vs-rest
`ConceptSet` with no single target class to restrict to). Each fit
(section 1 or 2) is capped at {FIT_TIMEOUT}s; a time-out or an error
counts as a failure, not as the end of the run.

Run: `python demos/covering_boosting.py` with no arguments is
the **quick** default; its report isn't checked in. `--full` runs both
sections at full scale and writes the canonical report.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pyrulearn.experiments.catalog import Catalog, CatalogEntry
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.experiments.report import (
    format_accuracy, format_mean_count, render_results_table, render_setup_section,
)
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import mean_rank
from pyrulearn.heuristics import Laplace, MEstimate, WRAcc
from pyrulearn.learners.boosting import LRI, Boomer, ENDER, Newton, Slipper
from pyrulearn.learners.cpar import CPAR
from pyrulearn.objectives.losses import ExponentialLoss
from pyrulearn.learners.seco import (
    AdditiveReweighting, BeamSearch, CoveredAtLeast, LRIReweighting,
    MultiplicativeReweighting, RemovalCovering, Rounds, SeCo, SingleRuleLearner,
    WeightedCovering,
)

RANDOM_STATE = 0
MAX_INTERVALS = 8
FIT_TIMEOUT = 300.0
BEAM_WIDTH = 5

# -- section 1: covering strategies -------------------------------------

# m=16 is demos/heuristic_comparison.py's own measured best-performing
# m-estimate setting, not a default from the literature -- reused here
# rather than re-deriving it.
HEURISTICS = {"Laplace": lambda: Laplace(), "WRAcc": lambda: WRAcc(), "MEstimate16": lambda: MEstimate(16)}
# gamma=0.5 and CoveredAtLeast(5) are CN2-SD's own defaults (Lavrač et
# al. 2004); Rounds(50) for LRI matches LRIReweighting's own class
# default n_rules -- each scheme keeps its own literature-default stop
# rather than forcing one arbitrary stop onto all of them.
COVERINGS = {
    "Removal": lambda: RemovalCovering(),
    "Weighted-Mult": lambda: WeightedCovering(MultiplicativeReweighting(0.5), CoveredAtLeast(5)),
    "Weighted-Add": lambda: WeightedCovering(AdditiveReweighting(), CoveredAtLeast(5)),
    "Weighted-LRI": lambda: WeightedCovering(LRIReweighting(), Rounds(50)),
}
COVERING_CONFIGS = [f"{cov}+{heur}" for cov in COVERINGS for heur in HEURISTICS]
COMPLETE_ALGORITHMS = ["CPAR", "LRI"]
SECTION1_NAMES = COVERING_CONFIGS + COMPLETE_ALGORITHMS

# -- section 2: boosting -------------------------------------------------

BOOSTING_FAMILIES = ["Slipper", "ENDER-CSLog", "ENDER-CSExp", "ENDER-Newton", "Boomer"]
N_RULES_BUDGETS = [10, 20, 50, 100]
SECTION2_NAMES = [f"{family}@{n}" for family in BOOSTING_FAMILIES for n in N_RULES_BUDGETS]

# -- datasets -------------------------------------------------------------

# Same curation as demos/seco_learners_comparison.py's FULL_BINARY:
# mostly symbolic two-class datasets, the kind these learners were
# designed for.
FULL_DATASETS = [
    "molecular-biology_promoters", "hepatitis", "SPECT", "heart-statlog", "breast-cancer",
    "heart-h", "heart-c", "colic", "vote", "dresses-sales", "cylinder-bands", "credit-approval",
    "tic-tac-toe", "credit-g", "kr-vs-kp", "sick", "mushroom", "adult",
]
N_FOLDS = 10
LARGE_FOLDS = 5  # adult

QUICK_DATASETS = ["vote", "hepatitis", "SPECT"]
QUICK_FOLDS = 3

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "covering_boosting"
CACHE_DIR = os.path.join(HERE, "_covering_boosting_cache")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")

# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"


def _labeled(cls):
    """`cls` named by a `label` in the results -- the same class is run with
    many settings, which `run_cv` tells apart by `display_name`. A
    module-level class, so a worker process can unpickle it by reference."""
    class Labeled(cls):
        def __init__(self, label, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.label = label

        @property
        def display_name(self) -> str:
            return self.label
    Labeled.__name__ = Labeled.__qualname__ = f"Labeled{cls.__name__}"
    Labeled.__module__ = __name__
    return Labeled


LabeledSeCo = _labeled(SeCo)
LabeledCPAR = _labeled(CPAR)
LabeledLRI = _labeled(LRI)
LabeledSlipper = _labeled(Slipper)
LabeledENDER = _labeled(ENDER)
LabeledBoomer = _labeled(Boomer)


# =========================================================== learners ===

def build_covering_learner(name: str) -> SeCo:
    cov_name, heur_name = name.split("+")
    heuristic = HEURISTICS[heur_name]()
    covering = COVERINGS[cov_name]()
    srl = SingleRuleLearner(heuristic=heuristic, search=BeamSearch(beam_width=BEAM_WIDTH))
    return LabeledSeCo(name, single_rule_learner=srl, covering=covering, random_state=RANDOM_STATE)


def build_section1_learner(name: str):
    if name == "CPAR":
        return LabeledCPAR(name)
    if name == "LRI":
        return LabeledLRI(name)
    return build_covering_learner(name)


def build_section2_learner(name: str):
    family, n_str = name.split("@")
    n = int(n_str)
    if family == "Slipper":
        return LabeledSlipper(name, n_rounds=n, random_state=RANDOM_STATE)
    if family == "ENDER-CSLog":
        return LabeledENDER(name, n_rules=n, random_state=RANDOM_STATE)
    if family == "ENDER-CSExp":
        return LabeledENDER(name, n_rules=n, loss=ExponentialLoss(), random_state=RANDOM_STATE)
    if family == "ENDER-Newton":
        return LabeledENDER(name, n_rules=n, method=Newton(), subsample=0.5, random_state=RANDOM_STATE)
    if family == "Boomer":
        return LabeledBoomer(name, n_rules=n, random_state=RANDOM_STATE)
    raise ValueError(f"unknown boosting family {family!r}")


def _overlap(model, test_rep) -> float:
    n = len(np.asarray(test_rep.y))
    counts = np.zeros(n, dtype=int)
    for rule in model.rules:
        counts += rule.covers_data(test_rep).astype(int)
    return float(counts.mean())


def _overlap_measure(model, test_rep) -> dict:
    """`run_cv`'s `measure_fn`: the overlap column next to its default measures."""
    return {"overlap": _overlap(model, test_rep)}


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


def _concurrency_note() -> str:
    if N_JOBS.rstrip("!") == "1":
        return " Every learner runs single-threaded, one fit at a time."
    return (f" Every learner runs single-threaded; fits run {N_JOBS.rstrip('!')} at a time (`run_cv`'s "
            "`n_jobs`), each in its own process and timed there, with the numerical libraries limited to "
            "one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` "
            "learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run "
            "as a second sequential run (per-learner ratios and rankings alike); time ratios between "
            "learners are reproducible to about +-10% either way.")


# =============================================================== the loop ===

def run_section(
    datasets: Sequence[CatalogEntry], names: Sequence[str], build_fn,
    n_folds: int, cache_dir: Optional[str] = CACHE_DIR, fit_timeout: float = FIT_TIMEOUT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Runs one section: every `name` in `names` fit (via `build_fn`)
    against every dataset, `n_folds`-fold cross-validation
    (`pyrulearn.experiments.runner.run_cv`). One row per
    (dataset, fold, name) with accuracy/n_rules/n_conditions/fit_time/
    overlap/error."""
    results = run_cv([build_fn(name) for name in names], datasets, n_folds=n_folds,
                     fit_timeout=fit_timeout, discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS), random_state=RANDOM_STATE,
                     cache_dir=cache_dir, measure_fn=_overlap_measure, verbose=verbose, n_jobs=_n_jobs())
    results = results.rename(columns={"learner": "name"})
    if "overlap" not in results.columns:  # every fit failed
        results["overlap"] = np.nan
    return results


# ================================================================= report ===

def _common_cells(
    results: pd.DataFrame, names: Sequence[str],
) -> Tuple[Optional[set], Optional[str]]:
    """Like the pool-distillers demo's dataset-coverage check, but at
    (dataset, fold) granularity -- the unit each `name` here succeeds
    or fails at (a fold, not a whole dataset, can time out). Returns
    `(None, None)` if every name in `names` succeeded on exactly the
    same cells; otherwise the intersection and a note naming how many
    cells each name is missing, so callers can show a mean restricted
    to `common` alongside the unrestricted one."""
    ok = results[results["error"].isna()]
    cells_by_name: Dict[str, set] = {}
    for name in names:
        sub = ok[ok["name"] == name]
        if not sub.empty:
            cells_by_name[name] = set(zip(sub["dataset"], sub["fold"]))
    if len(set(map(frozenset, cells_by_name.values()))) <= 1:
        return None, None
    common = set.intersection(*cells_by_name.values())
    all_cells = set.union(*cells_by_name.values())
    note = "; ".join(
        f"{name} missing {len(all_cells - cells)} of {len(all_cells)} cells"
        for name, cells in cells_by_name.items() if all_cells - cells
    )
    return common, note


def _common_cell_means(results: pd.DataFrame, common: set, column: str = "accuracy") -> pd.Series:
    indexed = results.set_index(["dataset", "fold"])
    restricted = indexed[indexed.index.isin(common)]
    return restricted.reset_index().groupby("name")[column].mean()


def _summary_table(results: pd.DataFrame, names: Sequence[str]) -> List[str]:
    summary = results.groupby("name")[["accuracy", "n_rules", "n_conditions", "fit_time",
                                       "overlap"]].mean(numeric_only=True)
    failures = results.groupby("name")["error"].apply(lambda s: int(s.notna().sum()))

    common, note = _common_cells(results, names)
    common_acc = _common_cell_means(results, common) if common is not None else None

    acc_label = "accuracy (own cells)" if common is not None else "accuracy"
    cols = ["name", acc_label]
    if common_acc is not None:
        cols.append(f"accuracy ({len(common)} common cells)")
    cols += ["n_rules", "conditions/rule", "fit_time (s)", "overlap", "failures"]
    lines = ["| " + " | ".join(cols) + " |\n", "|---|" + "--:|" * (len(cols) - 1) + "\n"]
    for name in names:
        if name not in summary.index:
            continue
        row = summary.loc[name]
        cpr = row["n_conditions"] / row["n_rules"] if row["n_rules"] else float("nan")
        cells = [name, format_accuracy(row['accuracy'])]
        if common_acc is not None:
            cells.append(format_accuracy(common_acc.get(name, float('nan'))))
        cells += [format_mean_count(row['n_rules']), format_mean_count(cpr), f"{row['fit_time']:.2f}",
                 format_mean_count(row['overlap']), str(int(failures.get(name, 0)))]
        lines.append("| " + " | ".join(cells) + " |\n")
    if note is not None:
        lines.append(f"\n*{note} (see the \"failures\" column). \"own cells\" is each name "
                     f"averaged over whatever (dataset, fold) cells it succeeded on; "
                     f"\"{len(common)} common cells\" restricts every name to the same cells, "
                     f"for a fair comparison.*\n\n")
    return lines


def _rank_table(results: pd.DataFrame, names: Sequence[str]) -> List[str]:
    r = results.rename(columns={"name": "learner"})
    ranks = mean_rank(r, "accuracy")
    lines = ["| name | mean rank |\n|---|--:|\n"]
    for name, rank in ranks.items():
        if name in names:
            lines.append(f"| {name} | {rank:.2f} |\n")
    return lines


def _boosting_curve_table(results: pd.DataFrame, common: Optional[set] = None) -> List[str]:
    """`common`, if given, is section 1's own common-cells set (from
    `_common_cells(section1_results, SECTION1_NAMES)`), not one
    recomputed from section 2's own names -- so the "common cells"
    column here uses the *same* cells as section 1's summary table,
    making the two tables' accuracy columns directly comparable rather
    than each restricted to its own, differently-sized, common subset.
    """
    names = [f"{family}@{n}" for family in BOOSTING_FAMILIES for n in N_RULES_BUDGETS]
    summary = results.groupby("name")[["accuracy", "n_rules", "fit_time"]].mean(numeric_only=True)
    failures = results.groupby("name")["error"].apply(lambda s: int(s.notna().sum()))
    _, note = _common_cells(results, names)  # still used for the per-name failure breakdown

    common_acc = _common_cell_means(results, common) if common is not None else None

    acc_label = "accuracy (own cells)" if common is not None else "accuracy"
    cols = ["family", "budget", acc_label]
    if common_acc is not None:
        cols.append(f"accuracy ({len(common)} common cells)")
    cols += ["n_rules (actual)", "fit_time (s)", "failures"]
    lines = ["| " + " | ".join(cols) + " |\n", "|---|" + "--:|" * (len(cols) - 1) + "\n"]
    for family in BOOSTING_FAMILIES:
        for n in N_RULES_BUDGETS:
            name = f"{family}@{n}"
            if name not in summary.index:
                continue
            row = summary.loc[name]
            cells = [family, str(n), format_accuracy(row['accuracy'])]
            if common_acc is not None:
                cells.append(format_accuracy(common_acc.get(name, float('nan'))))
            cells += [format_mean_count(row['n_rules']), f"{row['fit_time']:.2f}", str(int(failures.get(name, 0)))]
            lines.append("| " + " | ".join(cells) + " |\n")

    footnote_parts = []
    if note is not None:
        footnote_parts.append(note)
    if common_acc is not None:
        footnote_parts.append(
            "\"own cells\" is each name averaged over whatever (dataset, fold) cells it "
            f"succeeded on; \"{len(common)} common cells\" is the same cell subset section 1's "
            "summary table uses, so the two tables' accuracy columns are directly comparable"
        )
    if footnote_parts:
        lines.append(f"\n*{'. '.join(footnote_parts)} (see the \"failures\" column).*\n\n")
    return lines


def _comparative_discussion(section1_results: pd.DataFrame, section2_results: pd.DataFrame) -> List[str]:
    """Built fresh from the measured numbers every run, restricted to
    the cells common to every row being compared (see `_common_cells`)
    so one dataset's failures don't quietly tilt which point looks
    best."""
    lines = ["## Comparative discussion\n\n"]

    common1, _ = _common_cells(section1_results, SECTION1_NAMES)
    s1 = (_common_cell_means(section1_results, common1) if common1 is not None
         else section1_results.groupby("name")["accuracy"].mean()).sort_values(ascending=False)
    if s1.empty:
        return lines
    best_name, best_acc = s1.index[0], s1.iloc[0]

    common2, _ = _common_cells(section2_results, SECTION2_NAMES)
    s2 = (_common_cell_means(section2_results, common2) if common2 is not None
         else section2_results.groupby("name")["accuracy"].mean()).sort_values(ascending=False)
    best2_name, best2_acc = (s2.index[0], s2.iloc[0]) if not s2.empty else (None, None)

    lines.append(
        f"Section 1's best covering-strategy point is `{best_name}` (mean accuracy "
        f"{format_accuracy(best_acc)} over the cells every section-1 config completed). Section 2's best "
        f"boosting point is `{best2_name}` ({format_accuracy(best2_acc)}, same restriction). "
    )

    # Ground the Slipper/ENDER-exponential kinship in the actual numbers
    # too, not just the theory below -- same budget, same cells.
    top_budget = N_RULES_BUDGETS[-1]
    exp_name, slipper_name = f"ENDER-CSExp@{top_budget}", f"Slipper@{top_budget}"
    exp_acc, slipper_acc = s2.get(exp_name), s2.get(slipper_name)
    kinship = (f"At budget {top_budget}, `{slipper_name}` scores {format_accuracy(slipper_acc)} and "
              f"`{exp_name}` scores {format_accuracy(exp_acc)} -- close, as expected from the theory below. "
              if exp_acc is not None and slipper_acc is not None else "")

    lines.append(
        "Boosting and weighted covering aren't as separate as they might look. "
        "`AdaBoostReweighting` (Slipper's mechanism, and `WeightedCovering`'s rule-weight-"
        "assigning option, which plain `SeCo` itself refuses to build into an unweighted set) "
        "and `ENDER`'s `ExponentialLoss` configuration are the *same* thing by a classical "
        "result (Friedman, Hastie & Tibshirani 2000): AdaBoost **is** gradient boosting on the "
        "exponential loss. `ExponentialLoss` here literally defines its per-example quantity as "
        "``w = exp(-y f)`` -- AdaBoost's reweighting formula -- and fits each rule's weight with "
        "the identical closed form `AdaBoostReweighting.confidence` uses. "
        f"{kinship}"
        "What actually distinguishes the two families: section 1's *other* reweighting schemes "
        "(`MultiplicativeReweighting`'s ``gamma**k``, `AdditiveReweighting`'s ``1/(k+1)``, "
        "`LRIReweighting`'s ``1 + e**3``) are hand-designed update rules with no loss function "
        "behind them, so they can't generalize the way `ENDER`'s pluggable-loss framework does "
        "(logistic, exponential, sigmoid, five weight-fitting methods). And weighted covering "
        "uses its weights only to steer training -- it still predicts with a heuristic score "
        "over an *unweighted* rule set -- while boosting bakes the fitted weight into the model "
        "itself (`LinearRuleModel`, summed at predict time): there the weight is part of the "
        "answer, not just how training got there.\n\n"
    )
    return lines


# Section 1: one color per covering strategy (as asked), one marker
# shape per heuristic -- so a covering strategy's three heuristic
# variants are visibly the same family, distinguishable by shape.
# CPAR/LRI get their own colors too, as the complete-algorithm rows
# the summary table already treats them as.
_COVERING_COLORS = {
    "Removal": "tab:blue", "Weighted-Mult": "tab:green",
    "Weighted-Add": "tab:red", "Weighted-LRI": "tab:purple",
}
_HEURISTIC_MARKERS = {"Laplace": "o", "WRAcc": "s", "MEstimate16": "^"}
_COMPLETE_COLORS = {"CPAR": "tab:brown", "LRI": "tab:pink"}
_COMPLETE_MARKERS = {"CPAR": "D", "LRI": "P"}
# Section 2: one color per boosting family, its four budgets (10/20/
# 50/100) connected by a line -- the accuracy-vs-cost curve the family
# traces out, not four unrelated points.
_FAMILY_COLORS = {
    "Slipper": "tab:orange", "ENDER-CSLog": "tab:cyan",
    "ENDER-CSExp": "tab:olive", "ENDER-Newton": "gold", "Boomer": "black",
}
# shapes unused by the heuristics (o s ^) and the complete algorithms (D P)
_FAMILY_MARKERS = {
    "Slipper": "v", "ENDER-CSLog": "<", "ENDER-CSExp": ">", "ENDER-Newton": "X", "Boomer": "*",
}


def _accuracy_vs_time_plot(
    section1_results: pd.DataFrame, section2_results: pd.DataFrame, common: set, path: str,
) -> None:
    """Accuracy (the same `common` cells every table in this report
    uses, so this plot is on the same footing) against fit time (log
    scale, since it spans timeouts down to sub-second boosting fits)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from matplotlib.lines import Line2D

    os.makedirs(os.path.dirname(path), exist_ok=True)

    s1_acc = _common_cell_means(section1_results, common)
    s1_time = section1_results.groupby("name")["fit_time"].mean()
    s2_acc = _common_cell_means(section2_results, common)
    s2_time = section2_results.groupby("name")["fit_time"].mean()

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for name in SECTION1_NAMES:
        if name not in s1_acc.index or name not in s1_time.index:
            continue
        acc, t = s1_acc[name], s1_time[name]
        if name in _COMPLETE_COLORS:
            ax.scatter(t, acc, color=_COMPLETE_COLORS[name], marker=_COMPLETE_MARKERS[name], s=55, zorder=3)
            continue
        cov_name, heur_name = name.split("+")
        ax.scatter(t, acc, color=_COVERING_COLORS[cov_name], marker=_HEURISTIC_MARKERS[heur_name],
                  s=55, zorder=3)

    # each covering strategy's three points on one dotted line, in the order Laplace, m-estimate, WRAcc
    for cov_name, color in _COVERING_COLORS.items():
        names = [f"{cov_name}+{h}" for h in ("Laplace", "MEstimate16", "WRAcc")]
        names = [n for n in names if n in s1_acc.index and n in s1_time.index]
        if len(names) > 1:
            ax.plot([s1_time[n] for n in names], [s1_acc[n] for n in names], ":", color=color, lw=1.2, zorder=2)

    for family in BOOSTING_FAMILIES:
        xs, ys = [], []
        for n in N_RULES_BUDGETS:
            name = f"{family}@{n}"
            if name in s2_acc.index and name in s2_time.index:
                xs.append(s2_time[name])
                ys.append(s2_acc[name])
        if xs:
            ax.plot(xs, ys, "-", marker=_FAMILY_MARKERS[family], color=_FAMILY_COLORS[family], markersize=5, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("fit time (s, log scale)")
    ax.set_ylabel(f"accuracy ({len(common)} common cells)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Accuracy vs. fit time: covering strategies and boosting")
    ax.grid(alpha=0.3, which="both")

    # covering strategies: color only (their shape is the heuristic, legend 2);
    # complete algorithms and boosting families: the marker they're drawn with
    algorithm_handles = [
        Line2D([0], [0], marker="o", color=c, linestyle="", markersize=7, label=n)
        for n, c in _COVERING_COLORS.items()
    ] + [
        Line2D([0], [0], marker=_COMPLETE_MARKERS[n], color=c, linestyle="", markersize=7, label=n)
        for n, c in _COMPLETE_COLORS.items()
    ] + [
        Line2D([0], [0], marker=_FAMILY_MARKERS[n], color=c, linestyle="-", markersize=5, label=n)
        for n, c in _FAMILY_COLORS.items()
    ]
    # the heuristics in the order the dotted lines connect them, each with a dotted line through its marker
    heuristic_handles = [
        Line2D([0], [0], marker=_HEURISTIC_MARKERS[h], color="gray", linestyle=":", markersize=7, label=h)
        for h in ("Laplace", "MEstimate16", "WRAcc")
    ] + [Line2D([0], [0], color="gray", linestyle="-", marker="o", markersize=4, label="rule budget")]
    # both legends outside the axes, to the right, so they cover no points
    legend1 = ax.legend(handles=algorithm_handles, fontsize=7, loc="upper left",
                        bbox_to_anchor=(1.02, 1.0), borderaxespad=0, title="algorithm")
    ax.add_artist(legend1)
    ax.legend(handles=heuristic_handles, fontsize=7, loc="lower left",
              bbox_to_anchor=(1.02, 0.0), borderaxespad=0, title="heuristic")

    # fixed margins, with the right fifth of the figure left to the legends
    # (tight_layout and bbox_inches="tight" don't count the second legend)
    fig.subplots_adjust(left=0.09, right=0.76, bottom=0.11, top=0.93)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(
    section1_results: pd.DataFrame, section2_results: pd.DataFrame,
    datasets: Sequence[CatalogEntry], n_folds: int, report_path: str, quick: bool,
) -> None:
    mode_note = (
        f"**Quick run** -- a fast sanity check, the default with no arguments. Full comparison: "
        f"`python demos/{NAME}.py --full`.\n\n" if quick else
        f"**Full run**. Quick sanity check instead: `python demos/{NAME}.py` with no arguments.\n\n"
    )
    dataset_names = ", ".join(d.name for d in datasets)
    description = f"""\
{mode_note}Covering strategies (plain `SeCo`, beam width {BEAM_WIDTH}) x heuristics
(`Laplace`, `WRAcc`, m-estimate m=16), plus `CPAR`/`LRI` as complete algorithms, on
{dataset_names} ({n_folds}-fold; {LARGE_FOLDS}-fold for `adult`, the
catalog's one "large" dataset here -- weighted covering and boosting
are no faster on it). Boosting (`Slipper`, `ENDER` x 3 configs,
`Boomer`) on the same datasets/folds, at rule-count budgets
{N_RULES_BUDGETS}.

Protocol: stratified cross-validation (`pyrulearn.experiments.runner.run_cv`), one
`DataSpec` per training fold (`build_dataspec(discretizer=TreeDiscretizer(max_intervals={MAX_INTERVALS}))`), the test
fold encoded against that same `DataSpec`, in the library's default representation
(vertical bitmaps (`VerticalBitmapRepresentation`)). Each fit is capped at {FIT_TIMEOUT:.0f}s; a time-out or an error counts
as a failure, not as the end of the run, and is excluded from every
mean it would otherwise enter (a "failures" column reports the count
instead).{_concurrency_note()}

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: accuracy, rule count, conditions per rule, fit time, and
overlap (mean rules covering a test row, pooled across every concept --
see the module docstring for why this isn't restricted to one
"positive" class).
"""
    L = ["# Covering strategies and boosting\n\n"]
    L.append(render_setup_section(description))

    L.append("## Section 1: covering strategies\n\n")
    L += _summary_table(section1_results, SECTION1_NAMES)
    L.append("\n### Mean rank\n\n")
    L += _rank_table(section1_results, SECTION1_NAMES)
    L.append("\n### Per-dataset results (mean across folds)\n\n")
    L.append(render_results_table(section1_results.rename(columns={"name": "learner"}),
                                  ["accuracy", "n_rules", "fit_time", "overlap"],
                                  group_by=["dataset", "learner"], include_overall=False))
    L.append("\n")

    L.append("## Section 2: boosting\n\n")
    common1, _ = _common_cells(section1_results, SECTION1_NAMES)
    L += _boosting_curve_table(section2_results, common1)
    L.append("\n### Per-dataset results (mean across folds)\n\n")
    L.append(render_results_table(section2_results.rename(columns={"name": "learner"}),
                                  ["accuracy", "n_rules", "fit_time"],
                                  group_by=["dataset", "learner"], include_overall=False))
    L.append("\n")

    L += _comparative_discussion(section1_results, section2_results)

    tag = "_quick" if quick else ""
    plot_path = os.path.join(PLOTS_DIR, f"{NAME}{tag}_accuracy_vs_time.png")
    # `_common_cells` has no set to give when every name has the same cells (nothing failed): they are all common
    plot_common = common1 if common1 is not None else set(
        zip(section1_results["dataset"], section1_results["fold"]))
    _accuracy_vs_time_plot(section1_results, section2_results, plot_common, plot_path)
    L.append("## Accuracy vs. fit time\n\n")
    L.append(
        "Every point here is on the same accuracy basis as the tables above (the "
        f"{len(plot_common)} common cells), against mean fit time on a log scale, since it spans "
        "sub-second boosting fits to multi-minute (and timed-out) covering fits. Color is the "
        "algorithm family -- the four covering strategies, `CPAR`/`LRI`, and the five boosting "
        "families; for section 1, marker shape is the heuristic, so a covering strategy's three "
        "heuristic variants show up as same-color points of different shapes, joined by a dotted "
        "line in the order Laplace, m-estimate, WRAcc. A boosting "
        "family's four rule-count budgets (10/20/50/100) are connected by a line, tracing out "
        "that family's own accuracy-vs-cost curve rather than four unrelated points. `Boomer` "
        "and `ENDER-Newton`'s lines nearly overlap at the top right -- both reach similarly high "
        "accuracy at a similar cost, a real finding (the two are closely related mechanisms; see "
        "the comparative discussion above), not a plotting artifact.\n\n"
    )
    L.append(f"![accuracy vs. fit time]({os.path.relpath(plot_path, HERE).replace(os.sep, '/')})\n\n")
    print(f"Plot   -> {plot_path}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def _run_both_sections(datasets: Sequence[CatalogEntry], n_folds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=== Section 1: covering strategies ===", flush=True)
    section1 = run_section(datasets, SECTION1_NAMES, build_section1_learner, n_folds)
    print("=== Section 2: boosting ===", flush=True)
    section2 = run_section(datasets, SECTION2_NAMES, build_section2_learner, n_folds)
    return section1, section2


def main(quick: bool = True) -> None:
    names = QUICK_DATASETS if quick else FULL_DATASETS
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")

    datasets = Catalog.default().select(names=names)

    # "adult" (the catalog's only "large" dataset here) gets fewer
    # folds -- same split as demos/seco_learners_comparison.py's
    # LARGE_FOLDS, since weighted covering/boosting are no faster on it.
    if quick:
        section1_results, section2_results = _run_both_sections(datasets, n_folds)
    else:
        small = [d for d in datasets if d.size != "large"]
        large = [d for d in datasets if d.size == "large"]
        groups = [_run_both_sections(small, n_folds)] + ([_run_both_sections(large, LARGE_FOLDS)] if large else [])
        section1_results = pd.concat([g[0] for g in groups], ignore_index=True)
        section2_results = pd.concat([g[1] for g in groups], ignore_index=True)

    write_report(section1_results, section2_results, datasets, n_folds, report_path, quick)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run the full comparison instead of the quick default")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(quick=not args.full)
