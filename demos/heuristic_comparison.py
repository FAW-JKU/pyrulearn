"""
demos/heuristic_comparison.py
=============================

Rule learning heuristics inside a plain separate-and-conquer learner,
after Janssen & Fürnkranz, "On the Quest for Optimal Rule Learning
Heuristics" (Machine Learning, 2010).

The learner follows the paper's Algorithms 1 and 2:

- single rule: greedy top-down search that keeps adding the best
  condition until no refinement is left and returns the best rule
  encountered along the way (`BeamSearch(beam_width=1)`), no stopping
  criterion, no pruning;
- covering: stop once the newly learned rule covers no more positives
  than negatives (`stop_covering`: tp - fp <= 0);
- multi-class: ordered class binarization -- classes from least to most
  frequent, each learned against all larger classes, the largest one
  becoming the default rule (`ConceptCascade`, order "least_frequent").

The only thing that varies is the heuristic: the paper's standard
heuristics (Precision, Laplace, Accuracy, WRA, Correlation) and its
tuned parametrized ones (cost c=0.437, relative cost c_r=0.342,
F-measure beta=0.5, m-estimate m=22.466), plus an m-estimate sweep
over m in {2, 4, 8, 16, 32}. Weka's JRip (as in the paper) and our own
Pypper (both also with ordered class binarization) are included as
benchmarks. Kloesgen is left out.

A last section adds the inverted m-estimate of Stecher, Janssen &
Fürnkranz (ECML PKDD 2014) for refinement, `Adaptive(Inverted(MEstimate(
16)))`, with the m-estimate for every m in `MATRIX_M` selecting the rule
on the refinement path, as a reference curve. The full study of
refinement and selection heuristics -- every pair of m, static and
adaptive, regular and inverted -- is `refinement_heuristics`, which shares
this demo's result cache.

Differences from the paper: numeric attributes are discretized by
`build_dataspec` (tree-based, `MAX_INTERVALS` intervals per attribute)
rather than tested at every midpoint between adjacent values, ties are
not broken randomly, and the datasets are by default the 69 of
`seco_learners_comparison` (`FULL_DATASETS`), not the paper's. With
`--paper-datasets`, they are instead the subset of the paper's 57 that
are in `pyrulearn.experiments.catalog` (`PAPER_DATASETS`; their
tuning/validation split is not kept).

Run: `python demos/heuristic_comparison.py` with no arguments
is the **quick** default (`QUICK_DATASETS_SPEC`, `QUICK_FOLDS`); its
report/plots go to ``heuristic_comparison_quick_*`` and are not
checked in. `--full` runs all `FULL_DATASETS` with `N_FOLDS` folds and
writes the canonical ``heuristic_comparison_report.md`` (plots in
``heuristic_comparison_plots/``); `--full --paper-datasets` runs
`PAPER_DATASETS` instead and writes ``heuristic_comparison_paper_*``.
Needs Weka at `WEKA_JAR`/`WEKA_JAVA` below, plus the `experiments` extra.
"""

from __future__ import annotations

import os
from typing import Optional


import pandas as pd

from pyrulearn.experiments.catalog import Catalog
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.experiments.report import (
    format_accuracy, format_mean_count, render_results_table, render_setup_section,
)
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import mean_rank, win_counts
from pyrulearn.heuristics import (
    Accuracy, Correlation, CoverageDifference, FBeta, Inverted, Laplace, LinearCost, LinearCostRates,
    MEstimate, Precision, Adaptive, WRAcc,
)
from pyrulearn.interfaces.weka import WekaJRip
from pyrulearn.learners.seco import BeamSearch, Pypper, SeCo, SingleRuleLearner
from pyrulearn.models import ConceptCascade, ConceptModel
from pyrulearn.pruning import ThresholdPrePruning
from seco_learners_comparison import BINARY_DATASETS, MULTICLASS_DATASETS

# $WEKA_JAR / $WEKA_JAVA win; else this machine's Windows install, else `java` on the PATH
_WIN_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAVA = os.environ.get("WEKA_JAVA") or (_WIN_JAVA if os.path.exists(_WIN_JAVA) else "java")
WEKA_JAR = os.environ.get("WEKA_JAR") or r"C:\Program Files\Weka-3-8-7\weka.jar"

RANDOM_STATE = 0
N_FOLDS = 10
MAX_INTERVALS = 8
FIT_TIMEOUT = 600.0  # no stopping criterion -> Precision/Laplace/Accuracy grow large theories
#: the largest dataset, with a longer cap per fit (as in `seco_learners_comparison`)
LONG_DATASETS = ["letter"]
LONG_FIT_TIMEOUT = 3600.0

M_SWEEP = (2, 4, 8, 16, 32)
PAPER_M = 22.466
# m for refinement x m for selection (`refinement_heuristics`' matrices);
# here, the inverted m-estimate with INVERTED_M refines, each m selects
MATRIX_M = (0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32)
INVERTED_M = 16

# The default datasets: the 69 of `seco_learners_comparison` (44 two-class,
# 25 multi-class), shared with most other demos.
FULL_DATASETS = BINARY_DATASETS + MULTICLASS_DATASETS
FULL_DATASETS_SPEC = ",".join(FULL_DATASETS)

# --paper-datasets: the paper's datasets that are in the catalog (their names -> ours where
# they differ: cleveland-heart-disease -> heart-c, credit/credit-a ->
# credit-approval, horse-colic -> colic, krkp -> kr-vs-kp, lymphography ->
# lymph, monk1-3 -> monks-problems-1..3, house-votes-84 -> vote,
# promoters -> molecular-biology_promoters). sick (not the same data as
# their sick-euthyroid) and titanic (a different version) are left out.
PAPER_DATASETS = [
    # from their 27 tuning datasets
    "anneal", "audiology", "breast-cancer", "heart-c", "credit-approval", "glass", "hepatitis",
    "colic", "hypothyroid", "iris", "kr-vs-kp", "lymph", "monks-problems-1", "monks-problems-2",
    "monks-problems-3", "mushroom", "soybean", "tic-tac-toe", "vote", "vowel", "wine",
    # from their 30 validation datasets
    "balance-scale", "breast-w", "credit-g", "diabetes", "hayes-roth", "heart-h", "heart-statlog",
    "ionosphere", "primary-tumor", "molecular-biology_promoters", "segment", "solar-flare",
    "sonar", "vehicle", "zoo",
]
PAPER_DATASETS_SPEC = ",".join(PAPER_DATASETS)

# Quick run (the no-args default): ten small datasets from the list above,
# binary and multi-class, 3-fold.
QUICK_DATASETS_SPEC = ("breast-cancer,hepatitis,heart-c,vote,iris,wine,glass,tic-tac-toe,lymph,"
                       "hayes-roth")
QUICK_FOLDS = 3

HERE = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(HERE, "heuristic_comparison_plots")
CACHE_DIR = os.path.join(HERE, "_heuristic_comparison_cache")


# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


def _concurrency_note() -> str:
    if N_JOBS.rstrip("!") == "1":
        return "\n\nEvery learner runs single-threaded, one fit at a time."
    return (f"\n\nEvery learner runs single-threaded; fits run {N_JOBS.rstrip('!')} at a time (`run_cv`'s "
            "`n_jobs`), each in its own process and timed there, with the numerical libraries limited to "
            "one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` "
            "learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run "
            "as a second sequential run (per-learner ratios and rankings alike); time ratios between "
            "learners are reproducible to about +-10% either way.")


def _cost_ratio(c: float) -> float:
    """The paper's c*p - (1-c)*n ranks rules exactly like p - ((1-c)/c)*n."""
    return (1 - c) / c


def run_tag(quick: bool, paper: bool) -> str:
    """The suffix of a run's report and plot names: none for the canonical
    full run, ``_quick`` for the quick one, ``_paper`` for the paper's
    datasets."""
    return "_quick" if quick else "_paper" if paper else ""


def _paths(quick: bool, paper: bool):
    tag = run_tag(quick, paper)
    return (os.path.join(HERE, f"heuristic_comparison{tag}_report.md"),
            os.path.join(PLOTS_DIR, f"heuristic_comparison{tag}_tradeoff.png"),
            os.path.join(PLOTS_DIR, f"heuristic_comparison{tag}_m_sweep.png"))


def datasets_spec(quick: bool, paper: bool) -> str:
    return QUICK_DATASETS_SPEC if quick else PAPER_DATASETS_SPEC if paper else FULL_DATASETS_SPEC


def run(learners, datasets, n_folds: int, cache_dir: str, n_jobs):
    """`run_cv` with `FIT_TIMEOUT` per fit, `LONG_FIT_TIMEOUT` on
    `LONG_DATASETS`."""
    runs = []
    for group, timeout in (([d for d in datasets if d.name not in LONG_DATASETS], FIT_TIMEOUT),
                           ([d for d in datasets if d.name in LONG_DATASETS], LONG_FIT_TIMEOUT)):
        if group:
            runs.append(run_cv(learners, group, n_folds=n_folds, fit_timeout=timeout,
                               discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS),
                               random_state=RANDOM_STATE, cache_dir=cache_dir, n_jobs=n_jobs))
    return pd.concat([r for r in runs if len(r)], ignore_index=True)


def timeout_note(datasets) -> str:
    long_ = [d.name for d in datasets if d.name in LONG_DATASETS]
    return f"Each fit is capped at {FIT_TIMEOUT:.0f}s" + (
        f" ({LONG_FIT_TIMEOUT:.0f}s for {', '.join(f'`{n}`' for n in long_)})" if long_ else "")


def datasets_note(datasets, paper: bool, quick: bool = False) -> str:
    n_multiclass = sum(1 for d in datasets if d.task == "multiclass")
    counts = f"{len(datasets) - n_multiclass} binary, {n_multiclass} multi-class"
    if quick:
        return f"{len(datasets)} small datasets of the catalog ({counts})"
    if paper:
        return f"{len(datasets)} of the paper's datasets that are in `pyrulearn.experiments.catalog` ({counts})"
    return (f"{len(datasets)} datasets of the catalog ({counts}), those of the "
            "[separate-and-conquer comparison](seco_learners_comparison_report.md), not the paper's")


class HeuristicSeCo(SeCo):
    """The paper's covering learner with one fixed heuristic, named by
    `label` in the results; with a `selection`, the heuristic refines and
    the selection picks the rule on the refinement path."""

    def __init__(self, label: str, heuristic, random_state: Optional[int] = None, selection=None):
        # no selection: the learner's parameters (and so its cached results) as before
        extra = {} if selection is None else {"selection": selection}
        super().__init__(
            single_rule_learner=SingleRuleLearner(heuristic=heuristic, search=BeamSearch(beam_width=1), **extra),
            stop_covering=ThresholdPrePruning(CoverageDifference(), 0, operator="<="),
            random_state=random_state,
        )
        self.label = label

    @property
    def display_name(self) -> str:
        return self.label

    def _provenance_params(self):
        return {"label": self.label}

    def _default_model(self, data) -> type:
        return ConceptModel if self.target_class is not None else ConceptCascade


def _heuristic_specs():
    specs = [(f"m-estimate (m={m})", MEstimate(m=m)) for m in M_SWEEP]
    specs += [
        (f"m-estimate (m={PAPER_M})", MEstimate(m=PAPER_M)),
        ("relative cost (c_r=0.342)", LinearCostRates(cost_ratio=_cost_ratio(0.342))),
        ("cost (c=0.437)", LinearCost(cost_ratio=_cost_ratio(0.437))),
        ("F-measure (beta=0.5)", FBeta(beta=0.5)),
        ("Correlation", Correlation()),
        ("WRA", WRAcc()),
        ("Precision", Precision()),
        ("Laplace", Laplace()),
        ("Accuracy", Accuracy()),
    ]
    return specs


def build_heuristic_variants():
    return [HeuristicSeCo(label, h, random_state=RANDOM_STATE) for label, h in _heuristic_specs()]


def _m_text(m) -> str:
    return f"{m:g}"


def _matrix_label(inverted: bool, m_refine, m_select) -> str:
    """The learner refining by the (inverted) m-estimate with `m_refine`
    and selecting by the m-estimate with `m_select`. Without inversion
    and with the same m it is the plain m-estimate: the sweep's own
    learner where the sweep has that m."""
    if not inverted and m_refine == m_select and m_refine in M_SWEEP:
        return f"m-estimate (m={m_refine})"
    refine = f"Adaptive(Inverted(m={_m_text(m_refine)}))" if inverted else f"m={_m_text(m_refine)}"
    return f"refine {refine}, select m={_m_text(m_select)}"


def _is_matrix_only(label: str) -> bool:
    return label.startswith("refine ")


def build_matrix_variants():
    """The inverted m-estimate with `INVERTED_M` refining, the m-estimate
    with every m of `MATRIX_M` selecting (a row of `refinement_heuristics`'
    matrix, the same learners)."""
    return [HeuristicSeCo(_matrix_label(True, INVERTED_M, m_select), Adaptive(Inverted(MEstimate(m=INVERTED_M))),
                          random_state=RANDOM_STATE, selection=MEstimate(m=m_select))
            for m_select in MATRIX_M]


def build_learners():
    return build_heuristic_variants() + build_matrix_variants() + [
        WekaJRip(jar=WEKA_JAR, java=WEKA_JAVA),
        Pypper(random_state=RANDOM_STATE),
    ]


BENCHMARKS = ("Weka:JRip", "Pypper")


def _build_description(datasets, n_folds: int, quick: bool, paper: bool) -> str:
    mode_note = (
        f"**Quick run** ({len(datasets)} small datasets, {n_folds}-fold) -- a fast sanity "
        f"check, the default with no arguments. Full comparison ({len(FULL_DATASETS)} "
        f"datasets, {N_FOLDS}-fold): `python demos/heuristic_comparison.py --full`.\n\n"
        if quick else
        f"**Full run** ({len(datasets)} datasets, {n_folds}-fold). Quick sanity check "
        f"instead: `python demos/heuristic_comparison.py` with no arguments.\n\n"
    )
    return f"""\
{mode_note}Rule learning heuristics compared inside one plain separate-and-conquer
learner, after Janssen & Fürnkranz (Machine Learning, 2010). The learner
follows the paper's Algorithms 1 and 2: greedy top-down search that
refines until no refinement is left and returns the best rule
encountered (`BeamSearch(beam_width=1)`), no stopping criterion and no
pruning; covering stops once a new rule covers no more positives than
negatives; multi-class problems use ordered class binarization (least
frequent class first, the largest class as the default rule).

Heuristics: the paper's standard ones (Precision, Laplace, Accuracy,
WRA, Correlation), its tuned parametrized ones (cost c=0.437, relative
cost c_r=0.342, F-measure beta=0.5, m-estimate m={PAPER_M}), and an
m-estimate sweep over m in {{{', '.join(str(m) for m in M_SWEEP)}}}.
Benchmarks: Weka's JRip and pyrulearn's Pypper, both with their own
pruning, and both also use ordered class binarization. What each
heuristic prefers is visualized by its coverage-space isometrics in the
companion demo, [heuristic_isometrics](heuristic_isometrics_report.md).

Datasets: {datasets_note(datasets, paper, quick)}. Numeric attributes are discretized per
training fold (`build_dataspec(discretizer=TreeDiscretizer(max_intervals={MAX_INTERVALS}))`) instead
of the paper's tests at every midpoint. {timeout_note(datasets)}; a timeout counts as a failure.

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: test accuracy, number of rules and conditions, fit time.{_concurrency_note()}

The summary and the per-dataset results below are for the heuristics as
in the paper; the [last section](#the-inverted-m-estimate-for-refinement)
adds the inverted m-estimate for refinement, with the m-estimate
selecting.
"""


def write_report(results, datasets, n_folds: int, report_path: str,
                 tradeoff_plot_path: str, m_sweep_plot_path: str, quick: bool, paper: bool = False) -> None:
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]
    plots_dir = os.path.basename(os.path.dirname(tradeoff_plot_path))
    all_results = results
    results = results[~results["learner"].map(_is_matrix_only)]

    lines = ["# Rule learning heuristics in separate-and-conquer (after Janssen & Fürnkranz)\n\n"]
    lines.append(render_setup_section(_build_description(datasets, n_folds, quick, paper)))
    lines.append(f"![accuracy vs. theory size]({plots_dir}/{os.path.basename(tradeoff_plot_path)})\n\n")
    lines.append(f"![m-estimate sweep]({plots_dir}/{os.path.basename(m_sweep_plot_path)})\n\n")

    lines.append("## Summary by learner (mean across every dataset and fold)\n\n")
    ranks = mean_rank(results, "accuracy")
    wins = win_counts(results, "accuracy")
    summary = results.groupby("learner")[
        ["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time"]].mean(numeric_only=True)
    failures = results.groupby("learner")["error"].apply(lambda e: e.notna().sum())
    lines.append("| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | "
                 "wins | mean rank | failures |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    for learner in summary["accuracy"].sort_values(ascending=False).index:
        row = summary.loc[learner]
        lines.append(f"| {learner} | {format_accuracy(row['accuracy'])} | {format_mean_count(row['n_rules'])} | "
                     f"{format_mean_count(row['n_conditions'])} | {format_mean_count(row['conds_per_rule'])} | "
                     f"{row['fit_time']:.2f} | {wins.get(learner, 0.0):.1f} | "
                     f"{ranks[learner]:.2f} | {int(failures.get(learner, 0))} |\n")
    lines.append("\n`wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie "
                 "split evenly; `mean rank` -- average accuracy rank across datasets, failures "
                 "tied for last.\n\n")

    lines.append("## Per-dataset results, per learner (mean across folds)\n\n")
    lines.append(render_results_table(
        results, ["accuracy", "n_rules", "conds_per_rule", "n_conditions", "fit_time"],
        group_by=["dataset", "learner"], include_overall=False))
    lines.append("\n")

    lines += _inverted_section(all_results, run_tag(quick, paper))

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {report_path}")


def _inverted_section(results, tag: str = "") -> list:
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]
    means = results.groupby("learner")[["accuracy", "n_rules", "conds_per_rule", "n_conditions"]].mean(
        numeric_only=True)
    lines = [
        "## The inverted m-estimate for refinement\n\n",
        f"`Adaptive(Inverted(MEstimate({INVERTED_M})))` refines, after Stecher, Janssen & Fürnkranz "
        "(ECML PKDD 2014): it judges a refinement from the rule it refines (covering P̂ positives and "
        "N̂ negatives) by the examples it excludes, h(N̂ - n, P̂ - p), and prefers the refinement whose "
        "excluded examples are the purest in negatives (see the "
        "[isometrics](heuristic_isometrics_report.md#inverted-heuristics)). The regular m-estimate "
        "selects the rule on the refinement path, with every m below. Every pair of m, and the other "
        f"frames, are in [refinement_heuristics](refinement_heuristics{tag}_report.md).\n\n",
        "| m for selection | accuracy | n_rules | conds/rule | n_conditions |\n|--:|--:|--:|--:|--:|\n",
    ]
    for m_select in MATRIX_M:
        label = _matrix_label(True, INVERTED_M, m_select)
        if label not in means.index:
            continue
        row = means.loc[label]
        lines.append(f"| m={_m_text(m_select)} | {format_accuracy(row['accuracy'])} | "
                     f"{format_mean_count(row['n_rules'])} | {format_mean_count(row['conds_per_rule'])} | "
                     f"{format_mean_count(row['n_conditions'])} |\n")
    n_failed = int(results[results["learner"].map(_is_matrix_only)]["error"].notna().sum())
    lines.append(f"\nFailed folds: {n_failed}.\n\n")
    return lines


def _m_of(label: str) -> Optional[float]:
    if label.startswith("m-estimate (m="):
        return float(label[len("m-estimate (m="):-1])
    return None


def write_plots(results, tradeoff_plot_path: str, m_sweep_plot_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        return
    os.makedirs(os.path.dirname(tradeoff_plot_path), exist_ok=True)
    all_summary = results.groupby("learner")[["accuracy", "n_conditions"]].mean(numeric_only=True)
    summary = all_summary[[not _is_matrix_only(learner) for learner in all_summary.index]]

    fig, ax = plt.subplots(figsize=(8, 6))
    for learner, row in summary.iterrows():
        if learner in BENCHMARKS:
            color, marker = "tab:red", "s"
        elif _m_of(learner) is not None:
            color, marker = "tab:blue", "o"
        else:
            color, marker = "tab:gray", "^"
        ax.scatter(row["n_conditions"], row["accuracy"], color=color, marker=marker, s=50)
        ax.annotate(learner, (row["n_conditions"], row["accuracy"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean number of conditions (log)")
    ax.set_ylabel("mean test accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("accuracy vs. theory size per heuristic")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(tradeoff_plot_path, dpi=110)
    plt.close(fig)
    print(f"Plot   -> {tradeoff_plot_path}")

    sweep = sorted((m, learner) for learner in summary.index if (m := _m_of(learner)) is not None)
    ms = [m for m, _ in sweep]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(ms, [summary.loc[l, "accuracy"] for _, l in sweep], "o-", color="tab:blue")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("m (log scale)")
    ax1.set_ylabel("mean test accuracy", color="tab:blue")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.axvline(PAPER_M, color="gray", linestyle=":", label=f"paper's m={PAPER_M}")
    ax2 = ax1.twinx()
    ax2.plot(ms, [summary.loc[l, "n_conditions"] for _, l in sweep], "s--", color="tab:orange")
    ax2.set_ylabel("mean number of conditions", color="tab:orange")
    ax1.set_title("m-estimate: accuracy and theory size vs. m")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(m_sweep_plot_path, dpi=110)
    plt.close(fig)
    print(f"Plot   -> {m_sweep_plot_path}")


def main(spec: Optional[str] = None, full: bool = False, paper: bool = False) -> None:
    quick = not full and spec is None
    spec = spec or datasets_spec(quick, paper)
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path, tradeoff_plot_path, m_sweep_plot_path = _paths(quick, paper)

    datasets = Catalog.default().parse(spec, random_state=RANDOM_STATE)
    results = run(build_learners(), datasets, n_folds, CACHE_DIR, _n_jobs())
    write_report(results, datasets, n_folds, report_path, tradeoff_plot_path, m_sweep_plot_path, quick, paper)
    write_plots(results, tradeoff_plot_path, m_sweep_plot_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"run all {len(FULL_DATASETS)} datasets, {N_FOLDS}-fold, instead of "
                             "the quick default")
    parser.add_argument("--paper-datasets", action="store_true",
                        help=f"with --full: the {len(PAPER_DATASETS)} datasets of Janssen & Fuernkranz that are "
                             "in the catalog instead (report heuristic_comparison_paper_report.md)")
    parser.add_argument("--datasets", default=None,
                        help="Catalog.parse() spec for a full-scale run, e.g. 'vote,mushroom'")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(args.datasets, full=args.full, paper=args.paper_datasets)
