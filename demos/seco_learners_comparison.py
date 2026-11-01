"""
demos/seco_learners_comparison.py
=================================

pyrulearn's separate-and-conquer (SeCo) rule learners side by side, with
Weka's JRip and the reference Java implementation of LORD as external
baselines, cross-validated through `pyrulearn.experiments.runner.run_cv`
on mostly symbolic binary and multi-class datasets of
`pyrulearn.experiments.catalog` (see `FULL_BINARY`/`FULL_MULTICLASS`
for which, and why). All learners
here learn a rule at a time and *remove* what it covers; learners that
instead reweight covered examples (CPAR, LRI) belong to a separate
covering vs. weighted-covering comparison.

Learners (each with its own default settings and its own default way of
handling several classes -- see `LEARNER_NOTES`):

- **CN2**      -- `pyrulearn.learners.seco.CN2` (Clark & Boswell 1991):
                  beam search (width 5) with Laplace, CN2's significance
                  test as a filter on the search's rules.
- **PFoil**    -- `PFoil`: FOIL's greedy gain-ascent search with FOIL
                  gain and Quinlan's (1990) encoding-length restriction.
- **PFossil**  -- `PFossil`: FOSSIL (Fürnkranz 1994), hill climbing on
                  correlation with a 0.3 correlation cutoff.
- **Pypper**   -- `Pypper`: pyrulearn's RIPPER (IREP* grow/prune plus
                  `ReplaceReviseOptimization`).
- **Opus(WRAcc)** -- `pyrulearn.learners.opus.Opus` (after Webb 1995):
                  the optimal rule by `WRAcc` in every covering step, found
                  by exhaustive branch-and-bound search
                  (`BranchAndBoundSearch`), at most 3 conditions per rule.
- **PyLORD**   -- `pyrulearn.learners.pylord.PyLORD`: a simplified
                  reimplementation of LORD (Huynh, Fürnkranz & Beck 2023),
                  one locally optimal rule per training example.
- **Weka:JRip** -- Weka's RIPPER, via `pyrulearn.interfaces.weka.WekaJRip`
                  (needs Weka, `WEKA_JAR`/`WEKA_JAVA` below).
- **JavaLORD** -- the reference LORD implementation, via
                  `pyrulearn.interfaces.lord.JavaLORD`. Only included when
                  found (`LORD_CLASSPATH` below, or `$LORD_CLASSPATH`).
                  Limited to one thread in this demo only (LORD's own
                  default is half the machine's cores).

**AQR** (Clark & Niblett 1989, the AQ baseline CN2 was designed to improve
on) and **Opus with the Laplace estimate** (Webb's own setting: the rule
CN2's beam search approximates, found exactly) are *not* in the main
comparison: `run_preliminary_aqr_check` fits them against CN2 on a few
small datasets first, and their measured results -- many narrow rules and
far higher fit times -- are the report's argument for leaving them out.

Run: `python demos/seco_learners_comparison.py` with no arguments
is the **quick** default (`QUICK_DATASETS_SPEC`, `QUICK_FOLDS`-fold); its
report/plots go to ``seco_learners_comparison_quick_*`` and are not
checked in. `--full` runs `FULL_DATASETS_SPEC`, `N_FOLDS`-fold
(`LARGE_FOLDS`-fold for the two large datasets), and writes the canonical
``seco_learners_comparison_report.md`` (plots in
``seco_learners_comparison_plots/``). Needs the `experiments` extra.
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
from pyrulearn.experiments.stats import critical_difference_diagram, mean_rank, win_counts
from pyrulearn.interfaces.lord import JavaLORD
from pyrulearn.interfaces.weka import WekaJRip
from pyrulearn.heuristics import WRAcc
from pyrulearn.learners.opus import Opus
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.learners.seco import AQR, CN2, PFoil, PFossil, Pypper

# $WEKA_JAR / $WEKA_JAVA win; else this machine's Windows install, else `java` on the PATH
_WIN_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAVA = os.environ.get("WEKA_JAVA") or (_WIN_JAVA if os.path.exists(_WIN_JAVA) else "java")
WEKA_JAR = os.environ.get("WEKA_JAR") or r"C:\Program Files\Weka-3-8-7\weka.jar"
# the reference LORD (github.com/vqphuynh/LORD): its compiled classes plus
# its bundled Weka; $LORD_CLASSPATH / $LORD_JAVA take precedence ($LORD_DIR moves
# the checkout)
LORD_DIR = os.environ.get("LORD_DIR") or r"C:\Users\juffi\Github\LORD"
LORD_CLASSPATH = os.environ.get("LORD_CLASSPATH") or os.pathsep.join(
    [os.path.join(LORD_DIR, "bin"), os.path.join(LORD_DIR, "libs", "weka_3.8_stable.jar")])
LORD_JAVA = os.environ.get("LORD_JAVA") or WEKA_JAVA

# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"

RANDOM_STATE = 0
N_FOLDS = 10
MAX_INTERVALS = 8
FIT_TIMEOUT = 900.0

# Datasets whose attributes are mostly symbolic (at least half nominal) --
# the kind these learners were designed for; numeric attributes and their
# discretization are the numeric_discretization demo's topic. Left out:
# artificial concepts (monks-problems-1/2/3, mofn-3-7-10, hayes-roth, led24),
# datasets with many classes (audiology, primary-tumor, soybean, kropt --
# for the multi-class demo), and large datasets other than adult and
# connect-4.
FULL_BINARY = [
    "molecular-biology_promoters", "hepatitis", "SPECT", "heart-statlog", "breast-cancer",
    "heart-h", "heart-c", "colic", "vote", "dresses-sales", "cylinder-bands", "credit-approval",
    "tic-tac-toe", "credit-g", "kr-vs-kp", "sick", "mushroom", "adult",
]
FULL_MULTICLASS = [
    "zoo", "lymph", "analcatdata_dmft", "anneal", "solar-flare", "cmc", "car", "dna", "splice",
    "hypothyroid", "connect-4",
]
FULL_DATASETS_SPEC = ",".join(FULL_BINARY + FULL_MULTICLASS)
LARGE_FOLDS = 5  # for the catalog's "large" datasets (adult, connect-4)

# ten small, mostly symbolic datasets, two of them multi-class
QUICK_DATASETS_SPEC = ("vote,tic-tac-toe,hepatitis,breast-cancer,heart-statlog,credit-approval,"
                       "colic,SPECT,zoo,lymph")
QUICK_FOLDS = 3

HERE = os.path.dirname(__file__)
NAME = "seco_learners_comparison"
REPORT_PATH = os.path.join(HERE, f"{NAME}_report.md")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")
QUICK_REPORT_PATH = os.path.join(HERE, f"{NAME}_quick_report.md")
CACHE_DIR = os.path.join(HERE, "_seco_learners_comparison_cache")


def _plot_paths(quick: bool):
    tag = "_quick" if quick else ""
    return {kind: os.path.join(PLOTS_DIR, f"{NAME}{tag}_{kind}.png")
            for kind in ("accuracy", "fit_time", "cd")}


def lord_available() -> bool:
    return os.path.exists(LORD_CLASSPATH.split(os.pathsep)[0])


# exhaustive search is exponential in the rule length: at most 3 conditions
# keeps every Opus fit on the quick datasets within seconds (WRAcc) or a
# few minutes (Laplace); 4 costs up to 10x more for about the same rules
OPUS_MAX_CONDITIONS = 3


class LabeledOpus(Opus):
    """`Opus` named by `label` in the results -- the main comparison and
    the preliminary check run it with different heuristics."""

    def __init__(self, label: str, heuristic=None, random_state: Optional[int] = None):
        super().__init__(heuristic=heuristic, max_conditions=OPUS_MAX_CONDITIONS, random_state=random_state)
        self.label = label

    @property
    def display_name(self) -> str:
        return self.label

    def _provenance_params(self):
        return {"label": self.label, "heuristic": repr(self.heuristic) if self.heuristic else "Laplace(n_classes=c)",
                "max_conditions": self.max_conditions}


def build_learners():
    learners = [
        CN2(random_state=RANDOM_STATE),
        PFoil(random_state=RANDOM_STATE),
        PFossil(random_state=RANDOM_STATE),
        Pypper(random_state=RANDOM_STATE),
        LabeledOpus("Opus(WRAcc)", heuristic=WRAcc(), random_state=RANDOM_STATE),
        PyLORD(random_state=RANDOM_STATE),
        WekaJRip(jar=WEKA_JAR, java=WEKA_JAVA),
    ]
    if lord_available():
        # one thread, for this comparison only: the reference LORD uses half the machine's cores by
        # default, every other learner here one (`JavaLORD` itself keeps LORD's default)
        learners.append(JavaLORD(classpath=LORD_CLASSPATH, java=LORD_JAVA, thread_count=1))
    return learners


# AQR and Opus(Laplace) vs. CN2 preliminary check -- see
# run_preliminary_aqr_check(): small datasets with class noise, one of them
# (credit-approval) with several numeric attributes, where AQR's and
# Opus(Laplace)'s fit times blow up
PRELIM_DATASETS = ["breast-cancer", "hepatitis", "heart-statlog", "credit-approval"]
PRELIM_FOLDS = 3


def run_preliminary_aqr_check():
    """Fits `AQR` -- the learner CN2 was designed to improve on -- and
    `Opus` with the Laplace estimate -- the rule CN2's beam search
    approximates, found exactly -- against `CN2` on `PRELIM_DATASETS`, and
    returns `(results, verdict)`: the `run_cv` table and sentences built
    from the measured numbers."""
    datasets = Catalog.default().select(names=PRELIM_DATASETS)
    learners = [CN2(random_state=RANDOM_STATE), AQR(random_state=RANDOM_STATE),
                LabeledOpus("Opus(Laplace)", random_state=RANDOM_STATE)]
    results = run_cv(learners, datasets,
                     n_folds=PRELIM_FOLDS, fit_timeout=FIT_TIMEOUT, discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS),
                     random_state=RANDOM_STATE, cache_dir=CACHE_DIR, n_jobs=_n_jobs())
    r = results.copy()
    r["conds_per_rule"] = r["n_conditions"] / r["n_rules"]
    m = r.groupby("learner")[["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time"]].mean()
    aqr, cn2, opus = m.loc["AQR"], m.loc["CN2"], m.loc["Opus(Laplace)"]
    verdict = (
        f"Across {len(datasets)} small datasets ({PRELIM_FOLDS}-fold), AQR was "
        f"{'less' if aqr['accuracy'] < cn2['accuracy'] else 'not less'} accurate than CN2 "
        f"(mean accuracy {format_accuracy(aqr['accuracy'])} vs. {format_accuracy(cn2['accuracy'])}), learned rules "
        f"{aqr['conds_per_rule'] / cn2['conds_per_rule']:.1f}x as long "
        f"({aqr['conds_per_rule']:.1f} vs. {cn2['conds_per_rule']:.1f} conditions per rule; "
        f"{format_mean_count(aqr['n_conditions'])} vs. {format_mean_count(cn2['n_conditions'])} conditions in total), and took "
        f"{aqr['fit_time'] / cn2['fit_time']:.0f}x as long per fit ({aqr['fit_time']:.1f}s vs. "
        f"{cn2['fit_time']:.2f}s, mean) -- requiring every rule to be consistent overfits noisy "
        f"data, and on larger datasets with numeric attributes the fit times would dominate the "
        f"whole comparison. Opus with the Laplace estimate (at most {OPUS_MAX_CONDITIONS} "
        f"conditions per rule), which finds the rule CN2's beam search only approximates, was "
        f"{'less' if opus['accuracy'] < cn2['accuracy'] else 'not less'} accurate than CN2 "
        f"(mean accuracy {format_accuracy(opus['accuracy'])}), learned "
        f"{opus['n_rules'] / cn2['n_rules']:.1f}x as many rules ({format_mean_count(opus['n_rules'])} vs. "
        f"{format_mean_count(cn2['n_rules'])}, mean) and took {opus['fit_time'] / cn2['fit_time']:.0f}x as long "
        f"per fit ({opus['fit_time']:.1f}s, mean). Webb (1993) found classifiers that optimize "
        f"the Laplace estimate exactly often less accurate than those of a heuristic search "
        f"that fails to -- the result behind Quinlan & Cameron-Jones's 'oversearching'. The "
        f"main comparison runs Opus with WRAcc instead."
    )
    return results, verdict


# what the report says about each learner, keyed by display_name
LEARNER_NOTES = {
    "CN2": "Beam search (width 5) with the Laplace estimate; CN2's likelihood-ratio "
           "significance test filters the rules a search can return. Learns rules for "
           "every class (one-vs-rest rule set).",
    "PFoil": "FOIL: greedy search that adds the condition with the highest FOIL gain; "
             "Quinlan's encoding-length restriction stops rules that cost more bits "
             "than the examples they explain. Rules for every class.",
    "PFossil": "FOSSIL: hill climbing on the correlation between rule and class; rules "
               "below a correlation of 0.3 are dropped. Rules for every class.",
    "Pypper": "pyrulearn's RIPPER: grow on two thirds of the data, prune on the rest "
              "(IREP*), then optimize the rule set (replace/revise). Rules for the "
              "classes from least to most frequent, the most frequent one as default.",
    "Opus(WRAcc)": "Opus: in every covering step the rule with the highest weighted relative "
                   f"accuracy among all conjunctions of up to {OPUS_MAX_CONDITIONS} conditions, "
                   "found by exhaustive branch-and-bound search (OPUS). Rules for every class.",
    "PyLORD": "A simplified LORD: for every training example, the best rule covering it "
              "(greedy m-estimate search, then pruning); a test example is classified by "
              "the best rule that covers it. Many overlapping rules.",
    "Weka:JRip": "Weka's RIPPER, run as a subprocess on the same binarized data. Fit time "
                 "includes the JVM start.",
    "JavaLORD": "The reference LORD implementation (Java), run as a subprocess on the same "
                "binarized data, and, in this demo only, on a single thread: by default it uses half "
                "the machine's cores, which no other learner here does.",
}


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


def _concurrency_note() -> str:
    threads = ("Every learner runs single-threaded; the reference Java LORD, which by default uses "
               "half the machine's cores, is limited to one thread here (`thread_count=1`, for this "
               "comparison only).")
    if N_JOBS.rstrip("!") == "1":
        return f"\n\n{threads} Fits run one at a time."
    return (f"\n\n{threads} Fits run {N_JOBS.rstrip('!')} at a time (`run_cv`'s `n_jobs`), each in its own "
            "process and timed there, with the numerical libraries limited to one thread "
            "(`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 "
            "datasets found the fit times of 5 concurrent fits as close to a sequential run as a second "
            "sequential run (per-learner ratios and rankings alike); time ratios between learners are "
            "reproducible to about +-10% either way.")


def _build_description(datasets, n_folds: int, quick: bool, learner_names,
                       aqr_verdict: str) -> str:
    mode_note = (
        f"**Quick run** ({len(datasets)} small datasets, {n_folds}-fold) -- a fast sanity "
        f"check, the default with no arguments. Full comparison ({len(FULL_BINARY)} binary and "
        f"{len(FULL_MULTICLASS)} multi-class datasets): `python demos/{NAME}.py --full`.\n\n"
        if quick else
        f"**Full run** ({len(datasets)} datasets). Quick sanity check "
        f"instead: `python demos/{NAME}.py` with no arguments.\n\n"
    )
    n_multiclass = sum(1 for d in datasets if d.task == "multiclass")
    large = [d.name for d in datasets if d.size == "large"]
    folds_note = (f"{n_folds}-fold stratified cross-validation" if quick or not large else
                  f"{n_folds}-fold stratified cross-validation ({LARGE_FOLDS}-fold for the "
                  f"large {', '.join(large)})")
    lord_note = ("" if lord_available() else
                 f"\nThe reference Java LORD is not included in this run (not found at "
                 f"`{LORD_CLASSPATH.split(os.pathsep)[0]}`; set `$LORD_CLASSPATH`).\n")
    learner_lines = "\n".join(f"- **{n}** -- {LEARNER_NOTES.get(n, '')}" for n in learner_names)
    return f"""\
{mode_note}pyrulearn's separate-and-conquer rule learners compared with each other
and with two external baselines, on {len(datasets) - n_multiclass} binary and
{n_multiclass} multi-class datasets from `pyrulearn.experiments.catalog`.
All of them learn one rule at a time and remove the examples it covers;
they differ in how a rule is searched for, when it stops, and whether and
how the rule set is pruned. Every learner runs with its default settings,
including its default way of handling several classes.

The datasets have mostly symbolic attributes (at least half nominal), the
kind these learners were designed for -- numeric attributes and their
discretization are the topic of the numeric-discretization demo. Artificial
concepts (the monks problems, m-of-n, Hayes-Roth, LED) and datasets with
many classes (left for the multi-class demo) are not included.

{learner_lines}
{lord_note}
AQR, the AQ baseline CN2 was designed to improve on, and Opus with the
Laplace estimate were checked separately and left out -- see "Why AQR and
Opus(Laplace) aren't in the main comparison".
{aqr_verdict}

Protocol: {folds_note}
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(discretizer=TreeDiscretizer(max_intervals={MAX_INTERVALS}))`), the test fold binarized
against that same `DataSpec` -- every learner sees the identical binary
features per fold (as vertical bitmaps, the library's default representation; the learners
that need the dense matrix convert them). Each fit is capped at {FIT_TIMEOUT:.0f}s; a
time-out or an error counts as a failure (and as last in the ranking),
not as the end of the run. The cap is deliberately kept for the large
datasets too: which learners scale to tens of thousands of examples is
part of the result.{_concurrency_note()}

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: test accuracy, number of rules and conditions, fit time.
"""


def _summary_table(results) -> list:
    """Per learner, sorted by mean rank (best first): accuracy, size, fit time,
    wins, mean rank and failures -- ranks and wins within `results`."""
    ranks = mean_rank(results, "accuracy")
    wins = win_counts(results, "accuracy")
    summary = results.groupby("learner")[
        ["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time"]].mean(numeric_only=True)
    failures = results.groupby("learner")["error"].apply(lambda e: e.notna().sum())
    lines = ["| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | "
             "wins | mean rank | failures |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|\n"]
    for learner in ranks.sort_values().index:
        row = summary.loc[learner]
        lines.append(f"| {learner} | {format_accuracy(row['accuracy'])} | {format_mean_count(row['n_rules'])} | "
                     f"{format_mean_count(row['n_conditions'])} | {format_mean_count(row['conds_per_rule'])} | "
                     f"{row['fit_time']:.2f} | {wins.get(learner, 0.0):.1f} | "
                     f"{ranks[learner]:.2f} | {int(failures.get(learner, 0))} |\n")
    return lines


def write_report(results, datasets, n_folds: int, report_path: str, plots: dict, quick: bool,
                 learner_names, aqr_results, aqr_verdict: str) -> None:
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]
    plots_dir = os.path.basename(PLOTS_DIR)
    large_names = {d.name for d in datasets if d.size == "large"}

    lines = ["# Separate-and-conquer rule learners compared\n\n"]
    lines.append(render_setup_section(_build_description(datasets, n_folds, quick, learner_names, aqr_verdict)))
    lines.append(f"![accuracy vs. rule-set complexity]({plots_dir}/{os.path.basename(plots['accuracy'])})\n\n")
    lines.append(f"![fit time per dataset]({plots_dir}/{os.path.basename(plots['fit_time'])})\n\n")
    lines.append(f"![critical-difference diagram (accuracy)]({plots_dir}/{os.path.basename(plots['cd'])})\n\n")

    lines.append("## Why AQR and Opus(Laplace) aren't in the main comparison\n\n")
    lines.append(f"{aqr_verdict}\n\n")
    aqr_results = aqr_results.copy()
    aqr_results["conds_per_rule"] = aqr_results["n_conditions"] / aqr_results["n_rules"]
    lines.append(render_results_table(
        aqr_results, ["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time"],
        group_by=["dataset", "learner"], include_overall=False))
    lines.append("\n")

    large = results[results["dataset"].isin(large_names)]
    results = results[~results["dataset"].isin(large_names)]
    n_main = results["dataset"].nunique()
    lines.append(f"## Summary by learner (mean across the {n_main} datasets and their folds)\n\n")
    if large_names:
        lines.append(f"Without the two large datasets ({', '.join(sorted(large_names))}), where most "
                     "learners hit the time cap -- see the section on them below. On these "
                     f"{n_main} datasets every learner has a result for (almost) every fold, so the "
                     "means compare like with like.\n\n")
    summary = results.groupby("learner")[
        ["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time"]].mean(numeric_only=True)
    lines += _summary_table(results)
    lines.append("\nSorted by mean rank. `wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie "
                 "split evenly; `mean rank` -- average accuracy rank across datasets, failures "
                 "tied for last. Means over successful fits only; `failures` counts the fits "
                 "that timed out or raised.\n\n")

    size = {d.name: d.size for d in datasets}
    by_size = results.assign(size=results["dataset"].map(size))
    for label, sizes, rows_note in (("small", "small", "fewer than 1,000 examples"),
                                    ("medium", "medium", "1,000 to 10,000 examples")):
        part = by_size[by_size["size"] == sizes].drop(columns="size")
        if part["dataset"].nunique() and len(set(size.values()) - {"large"}) > 1:
            names = sorted(part["dataset"].unique())
            lines.append(f"### {label.capitalize()} datasets only ({len(names)}, {rows_note})\n\n")
            lines.append(", ".join(f"`{n}`" for n in names) + "\n\n")
            lines += _summary_table(part)
            lines.append("\n")

    task = {d.name: d.task for d in datasets if d.name not in large_names}
    if len(set(task.values())) > 1:
        lines.append("## Mean accuracy by target type\n\n")
        lines.append("| learner | binary | multi-class |\n|---|--:|--:|\n")
        by_task = results.assign(task=results["dataset"].map(task)).groupby(
            ["learner", "task"])["accuracy"].mean()
        for learner in mean_rank(results, "accuracy").sort_values().index:
            b, m = by_task.get((learner, "binary")), by_task.get((learner, "multiclass"))
            lines.append(f"| {learner} | {format_accuracy(b)} | {format_accuracy(m)} |\n")
        lines.append("\n")

    if len(large):
        lines.append(f"## The large datasets: {', '.join(sorted(large_names))}\n\n")
        lines.append(
            f"{LARGE_FOLDS}-fold cross-validation, with the same {FIT_TIMEOUT:.0f}s cap per fit as "
            "everywhere else -- which learners get through tens of thousands of examples at all "
            "is the result here. `folds` counts the folds that finished; accuracy, size and fit "
            "time are means over those folds only.\n\n")
        lines.append("| dataset | learner | folds | accuracy | n_rules | n_conditions | fit_time (s) |\n"
                     "|---|---|--:|--:|--:|--:|--:|\n")
        for (ds, learner), g in large.groupby(["dataset", "learner"]):
            ok = g[g["error"].isna()]
            cells = ([format_accuracy(ok[c].mean()) if c == "accuracy" else
                      f"{ok[c].mean():.2f}" if c == "fit_time" else format_mean_count(ok[c].mean())
                      for c in ("accuracy", "n_rules", "n_conditions", "fit_time")]
                     if len(ok) else ["--"] * 4)
            lines.append(f"| {ds} | {learner} | {len(ok)}/{len(g)} | " + " | ".join(cells) + " |\n")
        lines.append("\n")

    lines.append("## Per-dataset results, per learner (mean across folds)\n\n")
    lines.append(render_results_table(
        results, ["accuracy", "n_rules", "n_conditions", "fit_time"],
        group_by=["dataset", "learner"], include_overall=False))
    lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {report_path}")


def write_plots(results, plots: dict, learner_names) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    colors = dict(zip(learner_names, plt.get_cmap("tab10").colors))
    markers = dict(zip(learner_names, ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]))
    per_pair = results.groupby(["dataset", "learner"], as_index=False).mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for learner in learner_names:
        sub = per_pair[per_pair["learner"] == learner].dropna(subset=["accuracy"])
        if len(sub):
            ax.scatter(sub["n_conditions"], sub["accuracy"], label=learner,
                       color=colors[learner], marker=markers[learner], alpha=0.7, s=40)
    ax.set_xscale("symlog")
    ax.set_xlabel("total conditions (rule set complexity)")
    ax.set_ylabel("test accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("accuracy vs. rule-set complexity, one point per dataset")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots["accuracy"], dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot   -> {plots['accuracy']}")

    names = sorted(per_pair["dataset"].unique())
    x = np.arange(len(names))
    w = 0.8 / len(learner_names)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(names) * len(learner_names) / 4), 5.5))
    for i, learner in enumerate(learner_names):
        sub = per_pair[per_pair["learner"] == learner].set_index("dataset")
        ts = [sub["fit_time"].get(n, np.nan) for n in names]
        ax.bar(x + (i - (len(learner_names) - 1) / 2) * w, ts, w, label=learner, color=colors[learner])
    ax.set_yscale("log")
    ax.set_ylabel("mean fit time (s, log)")
    ax.set_title("fit time per dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=len(learner_names))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots["fit_time"], dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot   -> {plots['fit_time']}")

    try:
        ax = critical_difference_diagram(results, "accuracy")
        ax.figure.tight_layout()
        ax.figure.savefig(plots["cd"], dpi=110)
        plt.close(ax.figure)
        print(f"Plot   -> {plots['cd']}")
    except Exception as e:  # noqa: BLE001 -- a plot failure shouldn't sink the report
        print(f"Critical-difference diagram skipped: {type(e).__name__}: {e}")


def main(datasets_spec: Optional[str] = None) -> None:
    quick = datasets_spec is None
    spec = QUICK_DATASETS_SPEC if quick else datasets_spec
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path = QUICK_REPORT_PATH if quick else REPORT_PATH
    plots = _plot_paths(quick)

    datasets = Catalog.default().parse(spec, random_state=RANDOM_STATE)
    learners = build_learners()
    learner_names = [l.display_name for l in learners]
    print("Preliminary check: AQR and Opus(Laplace) vs. CN2 ...")
    aqr_results, aqr_verdict = run_preliminary_aqr_check()
    print(aqr_verdict)

    groups = [(datasets, n_folds)] if quick else [
        ([d for d in datasets if d.size != "large"], n_folds),
        ([d for d in datasets if d.size == "large"], LARGE_FOLDS),
    ]
    results = pd.concat([
        run_cv(learners, group, n_folds=folds, fit_timeout=FIT_TIMEOUT, discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS),
               random_state=RANDOM_STATE, cache_dir=CACHE_DIR, n_jobs=_n_jobs())
        for group, folds in groups if group
    ], ignore_index=True)
    write_report(results, datasets, n_folds, report_path, plots, quick, learner_names,
                 aqr_results, aqr_verdict)
    large_names = {d.name for d in datasets if d.size == "large"}
    write_plots(results[~results["dataset"].isin(large_names)], plots, learner_names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"run the full comparison ({len(FULL_BINARY)} binary and "
                             f"{len(FULL_MULTICLASS)} multi-class datasets) instead of the quick default")
    parser.add_argument("--datasets", default=None,
                        help="Catalog.parse() spec for a full-scale run, e.g. 'vote,mushroom'")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(args.datasets or (FULL_DATASETS_SPEC if args.full else None))
