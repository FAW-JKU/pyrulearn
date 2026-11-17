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

A separate section separates the m-estimate's two jobs, after Stecher,
Janssen & Fürnkranz (ECML PKDD 2014): one m-estimate chooses the
conditions (refinement), another one the rule on the refinement path
(selection, `SingleRuleLearner(MEstimate(m_r), selection=MEstimate(m_s))`),
for every pair of m values in `MATRIX_M`; and the same with the inverted
m-estimate for refinement (`Inverted(MEstimate(m_r))`). The results are
two m x m matrices, drawn as heatmaps on one color scale.

Differences from the paper: numeric attributes are discretized by
`build_dataspec` (tree-based, `MAX_INTERVALS` intervals per attribute)
rather than tested at every midpoint between adjacent values, ties are
not broken randomly, and the datasets are the subset of the paper's 57
that are in `pyrulearn.experiments.catalog` (their tuning/validation
split is not kept).

Run: `python demos/heuristic_comparison.py` with no arguments
is the **quick** default (`QUICK_DATASETS_SPEC`, `QUICK_FOLDS`); its
report/plots go to ``heuristic_comparison_quick_*`` and are not
checked in. `--full` runs all `FULL_DATASETS` with `N_FOLDS` folds and
writes the canonical ``heuristic_comparison_report.md`` (plots in
``heuristic_comparison_plots/``). Needs Weka at
`WEKA_JAR`/`WEKA_JAVA` below, plus the `experiments` extra.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
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
    MEstimate, Precision, WRAcc,
)
from pyrulearn.interfaces.weka import WekaJRip
from pyrulearn.learners.seco import BeamSearch, Pypper, SeCo, SingleRuleLearner
from pyrulearn.models import ConceptCascade, ConceptModel
from pyrulearn.pruning import ThresholdPrePruning

# $WEKA_JAR / $WEKA_JAVA win; else this machine's Windows install, else `java` on the PATH
_WIN_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAVA = os.environ.get("WEKA_JAVA") or (_WIN_JAVA if os.path.exists(_WIN_JAVA) else "java")
WEKA_JAR = os.environ.get("WEKA_JAR") or r"C:\Program Files\Weka-3-8-7\weka.jar"

RANDOM_STATE = 0
N_FOLDS = 10
MAX_INTERVALS = 8
FIT_TIMEOUT = 600.0  # no stopping criterion -> Precision/Laplace/Accuracy grow large theories

M_SWEEP = (2, 4, 8, 16, 32)
PAPER_M = 22.466
# m for refinement x m for selection, regular and inverted refinement
MATRIX_M = (0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32)

# The paper's datasets that are in the catalog (their names -> ours where
# they differ: cleveland-heart-disease -> heart-c, credit/credit-a ->
# credit-approval, horse-colic -> colic, krkp -> kr-vs-kp, lymphography ->
# lymph, monk1-3 -> monks-problems-1..3, house-votes-84 -> vote,
# promoters -> molecular-biology_promoters). sick (not the same data as
# their sick-euthyroid) and titanic (a different version) are left out.
FULL_DATASETS = [
    # from their 27 tuning datasets
    "anneal", "audiology", "breast-cancer", "heart-c", "credit-approval", "glass", "hepatitis",
    "colic", "hypothyroid", "iris", "kr-vs-kp", "lymph", "monks-problems-1", "monks-problems-2",
    "monks-problems-3", "mushroom", "soybean", "tic-tac-toe", "vote", "vowel", "wine",
    # from their 30 validation datasets
    "balance-scale", "breast-w", "credit-g", "diabetes", "hayes-roth", "heart-h", "heart-statlog",
    "ionosphere", "primary-tumor", "molecular-biology_promoters", "segment", "solar-flare",
    "sonar", "vehicle", "zoo",
]
FULL_DATASETS_SPEC = ",".join(FULL_DATASETS)

# Quick run (the no-args default): ten small datasets from the list above,
# binary and multi-class, 3-fold.
QUICK_DATASETS_SPEC = ("breast-cancer,hepatitis,heart-c,vote,iris,wine,glass,tic-tac-toe,lymph,"
                       "hayes-roth")
QUICK_FOLDS = 3

HERE = os.path.dirname(__file__)
REPORT_PATH = os.path.join(HERE, "heuristic_comparison_report.md")
PLOTS_DIR = os.path.join(HERE, "heuristic_comparison_plots")
TRADEOFF_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_tradeoff.png")
M_SWEEP_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_m_sweep.png")
MATRIX_ACCURACY_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_m_matrix_accuracy.png")
MATRIX_LENGTH_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_m_matrix_rule_length.png")
QUICK_REPORT_PATH = os.path.join(HERE, "heuristic_comparison_quick_report.md")
QUICK_TRADEOFF_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_quick_tradeoff.png")
QUICK_M_SWEEP_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_quick_m_sweep.png")
QUICK_MATRIX_ACCURACY_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_quick_m_matrix_accuracy.png")
QUICK_MATRIX_LENGTH_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_quick_m_matrix_rule_length.png")
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
    refine = f"Inverted(m={_m_text(m_refine)})" if inverted else f"m={_m_text(m_refine)}"
    return f"refine {refine}, select m={_m_text(m_select)}"


def _is_matrix_only(label: str) -> bool:
    return label.startswith("refine ")


def build_matrix_variants():
    """Every (inverted?, m for refinement, m for selection) of `MATRIX_M`
    the sweep doesn't already have. The same m without inversion has no
    separate selection (the same learner, and the same cached results,
    as the plain m-estimate)."""
    learners = []
    for inverted in (False, True):
        for m_refine in MATRIX_M:
            for m_select in MATRIX_M:
                label = _matrix_label(inverted, m_refine, m_select)
                if not _is_matrix_only(label):
                    continue
                refine = Inverted(MEstimate(m=m_refine)) if inverted else MEstimate(m=m_refine)
                selection = None if not inverted and m_refine == m_select else MEstimate(m=m_select)
                learners.append(HeuristicSeCo(label, refine, random_state=RANDOM_STATE, selection=selection))
    return learners


def build_learners():
    return build_heuristic_variants() + build_matrix_variants() + [
        WekaJRip(jar=WEKA_JAR, java=WEKA_JAVA),
        Pypper(random_state=RANDOM_STATE),
    ]


BENCHMARKS = ("Weka:JRip", "Pypper")


def _build_description(datasets, n_folds: int, quick: bool) -> str:
    n_multiclass = sum(1 for d in datasets if d.task == "multiclass")
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

Datasets: {len(datasets)} of the paper's datasets that are in
`pyrulearn.experiments.catalog` ({len(datasets) - n_multiclass} binary,
{n_multiclass} multi-class). Numeric attributes are discretized per
training fold (`build_dataspec(discretizer=TreeDiscretizer(max_intervals={MAX_INTERVALS}))`) instead
of the paper's tests at every midpoint. Each fit is capped at
{FIT_TIMEOUT:.0f}s; a timeout counts as a failure.

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: test accuracy, number of rules and conditions, fit time.{_concurrency_note()}

The summary and the per-dataset results below are for the heuristics as
in the paper; the [last section](#m-estimates-for-refinement-and-for-selection)
varies the m-estimate's m separately for refinement and for selection,
with and without inversion.
"""


def write_report(results, datasets, n_folds: int, report_path: str,
                 tradeoff_plot_path: str, m_sweep_plot_path: str, matrix_plot_paths, quick: bool) -> None:
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]
    plots_dir = os.path.basename(os.path.dirname(tradeoff_plot_path))
    all_results = results
    results = results[~results["learner"].map(_is_matrix_only)]

    lines = ["# Rule learning heuristics in separate-and-conquer (after Janssen & Fürnkranz)\n\n"]
    lines.append(render_setup_section(_build_description(datasets, n_folds, quick)))
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

    lines += _matrix_section(all_results, plots_dir, matrix_plot_paths)

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {report_path}")


MATRIX_MEASURES = (("accuracy", "accuracy", format_accuracy),
                   ("n_rules", "number of rules", format_mean_count),
                   ("conds_per_rule", "conditions per rule", format_mean_count),
                   ("n_conditions", "conditions in the rule set", format_mean_count))


def _matrices(results):
    """{measure: (regular, inverted)}, each an m x m DataFrame (rows: m
    for refinement, columns: m for selection) of means across every
    dataset and fold; NaN where a learner has no result."""
    means = results.groupby("learner")[[m for m, _, _ in MATRIX_MEASURES]].mean(numeric_only=True)
    out = {}
    for measure, _, _ in MATRIX_MEASURES:
        pair = []
        for inverted in (False, True):
            pair.append(pd.DataFrame(
                [[means[measure].get(_matrix_label(inverted, r, c), np.nan) for c in MATRIX_M] for r in MATRIX_M],
                index=list(MATRIX_M), columns=list(MATRIX_M)))
        out[measure] = tuple(pair)
    return out


def _matrix_table(matrix, fmt) -> str:
    best = np.nanmax(matrix.to_numpy()) if np.isfinite(matrix.to_numpy()).any() else None
    head = "| refine \\ select | " + " | ".join(f"m={_m_text(c)}" for c in matrix.columns) + " |\n"
    rows = [head, "|---|" + "--:|" * len(matrix.columns) + "\n"]
    for r in matrix.index:
        cells = []
        for c in matrix.columns:
            v = matrix.loc[r, c]
            cells.append("n/a" if np.isnan(v) else fmt(v))
        rows.append(f"| m={_m_text(r)} | " + " | ".join(cells) + " |\n")
    return "".join(rows) + "\n"


def _matrix_section(results, plots_dir: str, matrix_plot_paths) -> list:
    accuracy_plot, length_plot = matrix_plot_paths
    lines = [
        "## m-estimates for refinement and for selection\n\n",
        "The m-estimate does two jobs in the learner: it chooses the condition to add "
        "(refinement), and it chooses the rule on the refinement path that is returned "
        "(selection). Here the two get an m of their own -- `SingleRuleLearner("
        "MEstimate(m_r), selection=MEstimate(m_s))` -- for every pair of m in "
        f"{{{', '.join(_m_text(m) for m in MATRIX_M)}}}; the diagonal (m_r = m_s) is the "
        "plain m-estimate. The second matrix refines by the inverted m-estimate instead, "
        "`Inverted(MEstimate(m_r))`, h(N - n, P - p), after Stecher, Janssen & Fürnkranz "
        "(ECML PKDD 2014): it judges a refinement from the universal rule, where the "
        "search starts, rather than from the empty rule, and prefers the refinement whose "
        "excluded examples are the purest in negatives (see the "
        "[isometrics](heuristic_isometrics_report.md#inverted-heuristics)). `P` and `N` are "
        "the examples left in the current covering step, and the selection is always the "
        "regular m-estimate.\n\n",
        "Rows: m for refinement, columns: m for selection; means across every dataset and "
        "fold. Both heatmaps of a figure share one color scale.\n\n",
        f"![m for refinement vs. m for selection, accuracy]({plots_dir}/{os.path.basename(accuracy_plot)})\n\n",
        f"![m for refinement vs. m for selection, rule length]({plots_dir}/{os.path.basename(length_plot)})\n\n",
    ]
    matrices = _matrices(results)
    for measure, title, fmt in MATRIX_MEASURES:
        regular, inverted = matrices[measure]
        lines.append(f"### {title[0].upper() + title[1:]}\n\n")
        lines.append("Regular m-estimate for refinement:\n\n" + _matrix_table(regular, fmt))
        lines.append("Inverted m-estimate for refinement:\n\n" + _matrix_table(inverted, fmt))
    matrix_results = results[results["learner"].map(_is_matrix_only)]
    n_failed = int(matrix_results["error"].notna().sum())
    lines.append(f"Failed folds in the matrices (beyond the sweep's own learners): {n_failed}.\n\n")
    return lines


def _m_of(label: str) -> Optional[float]:
    if label.startswith("m-estimate (m="):
        return float(label[len("m-estimate (m="):-1])
    return None


def write_plots(results, tradeoff_plot_path: str, m_sweep_plot_path: str, matrix_plot_paths) -> None:
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

    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]
    matrices = _matrices(results)
    accuracy_plot, length_plot = matrix_plot_paths
    _plot_matrix_pair(plt, matrices["accuracy"], "RdYlGn", "mean test accuracy", accuracy_plot,
                      lambda v: f"{100 * v:.2f}", PercentFormatter(1.0))
    _plot_matrix_pair(plt, matrices["conds_per_rule"], "YlOrBr", "mean conditions per rule", length_plot,
                      lambda v: f"{v:.1f}", None)


def _plot_matrix_pair(plt, pair, cmap, label: str, path: str, cell_text, formatter) -> None:
    """The regular and the inverted matrix side by side, on one color scale."""
    values = np.concatenate([m.to_numpy().ravel() for m in pair])
    if not np.isfinite(values).any():
        return
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
    ticks = [_m_text(m) for m in MATRIX_M]
    for ax, matrix, title in zip(axes, pair, ("m-estimate refines", "inverted m-estimate refines")):
        data = matrix.to_numpy()
        image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, cell_text(data[i, j]), ha="center", va="center", fontsize=7, color="black")
        ax.set_xticks(range(len(ticks)), ticks)
        ax.set_yticks(range(len(ticks)), ticks)
        ax.set_xlabel("m for selection (the rule on the refinement path)")
        ax.set_ylabel("m for refinement (the condition to add)")
        ax.set_title(title)
    colorbar = fig.colorbar(image, ax=list(axes), shrink=0.85)
    colorbar.set_label(label)
    if formatter is not None:
        colorbar.ax.yaxis.set_major_formatter(formatter)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Plot   -> {path}")


def main(datasets_spec: Optional[str] = None) -> None:
    quick = datasets_spec is None
    spec = QUICK_DATASETS_SPEC if quick else datasets_spec
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path = QUICK_REPORT_PATH if quick else REPORT_PATH
    tradeoff_plot_path = QUICK_TRADEOFF_PLOT_PATH if quick else TRADEOFF_PLOT_PATH
    m_sweep_plot_path = QUICK_M_SWEEP_PLOT_PATH if quick else M_SWEEP_PLOT_PATH
    matrix_plot_paths = ((QUICK_MATRIX_ACCURACY_PLOT_PATH, QUICK_MATRIX_LENGTH_PLOT_PATH) if quick else
                         (MATRIX_ACCURACY_PLOT_PATH, MATRIX_LENGTH_PLOT_PATH))

    datasets = Catalog.default().parse(spec, random_state=RANDOM_STATE)
    results = run_cv(build_learners(), datasets, n_folds=n_folds, fit_timeout=FIT_TIMEOUT,
                     discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS), random_state=RANDOM_STATE, cache_dir=CACHE_DIR,
                     n_jobs=_n_jobs())
    write_report(results, datasets, n_folds, report_path, tradeoff_plot_path, m_sweep_plot_path,
                 matrix_plot_paths, quick)
    write_plots(results, tradeoff_plot_path, m_sweep_plot_path, matrix_plot_paths)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"run all {len(FULL_DATASETS)} datasets, {N_FOLDS}-fold, instead of "
                             "the quick default")
    parser.add_argument("--datasets", default=None,
                        help="Catalog.parse() spec for a full-scale run, e.g. 'vote,mushroom'")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(args.datasets or (FULL_DATASETS_SPEC if args.full else None))
