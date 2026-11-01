"""
demos/seco_learners_comparison.py
=================================

pyrulearn's separate-and-conquer (SeCo) rule learners side by side, with
Weka's JRip and the reference Java implementation of LORD as external
baselines, cross-validated through `pyrulearn.experiments.runner.run_cv`
on the 69 datasets of the other comparisons (44 two-class, 25
multi-class; `BINARY_DATASETS`, `MULTICLASS_DATASETS`). All learners
here but the two LORDs learn a rule at a time and *remove* what it
covers; learners that instead reweight covered examples (CPAR, LRI)
belong to a separate covering vs. weighted-covering comparison.

Learners (each with its own default settings and its own default way of
handling several classes -- see `LEARNER_NOTES`):

- **CN2**      -- `pyrulearn.learners.seco.CN2` (Clark & Boswell 1991):
                  beam search (width 5) with Laplace, CN2's significance
                  test as a filter on the search's rules, a rule set per
                  class. **CN2 (beam 1)**: the same with a beam of one.
- **CN2 (ordered)** -- Clark & Niblett's (1989) original: a decision
                  list, entropy, and the significance test over all
                  classes (`pyrulearn.pruning.LikelihoodRatioTest`).
                  **CN2 (ordered, beam 1)**: with a beam of one.
- **AQR**      -- `AQR` (Clark & Niblett 1989), their AQ: consistent rules
                  grown from seed examples.
- **PFoil**    -- `PFoil`: FOIL's greedy gain-ascent search with FOIL
                  gain and Quinlan's (1990) encoding-length restriction.
- **PFossil**  -- `PFossil`: FOSSIL (Fürnkranz 1994), hill climbing on
                  correlation with a 0.3 correlation cutoff.
- **Pypper**   -- `Pypper`: pyrulearn's RIPPER (IREP* grow/prune plus
                  `ReplaceReviseOptimization`).
- **Opus(Laplace)** -- `pyrulearn.learners.opus.Opus` (after Webb 1995):
                  the optimal rule by the Laplace estimate in every
                  covering step -- Webb's setting, the rule CN2's beam
                  search approximates -- found by exhaustive
                  branch-and-bound search (`BranchAndBoundSearch`), at most
                  `OPUS_MAX_CONDITIONS` conditions per rule.
                  **Opus(WRAcc)**: the same with `WRAcc`.
- **PyLORD**   -- `pyrulearn.learners.pylord.PyLORD`: a simplified
                  reimplementation of LORD (Huynh, Fürnkranz & Beck 2023),
                  one locally optimal rule per training example.
- **Weka:JRip** -- Weka's RIPPER, via `pyrulearn.interfaces.weka.WekaJRip`
                  (needs Weka, `WEKA_JAR`/`WEKA_JAVA` below). Also the unit
                  of the run times (`time_ratios`).
- **JavaLORD** -- the reference LORD implementation, via
                  `pyrulearn.interfaces.lord.JavaLORD`. Only included when
                  found (`LORD_CLASSPATH` below, or `$LORD_CLASSPATH`).
                  Limited to one thread in this demo only (LORD's own
                  default is half the machine's cores).

AQR and the two Opus are **optional** (`OPTIONAL`): they took 92% of the fit
time of a full run (about 8 of 9 hours on 10 cores), most of it time-outs. A
run without `--slow` doesn't start them, but its report and plots still show
what the cache holds for them, from the last run with `--slow`.

Run: `python demos/seco_learners_comparison.py` with no arguments
is the **quick** default (`QUICK_DATASETS`, `QUICK_FOLDS`-fold); its
report/plots go to ``seco_learners_comparison_quick_*`` and are not
checked in. `--full` runs all 69 datasets, `N_FOLDS`-fold, and writes the
canonical ``seco_learners_comparison_report.md`` (plots in
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
from pyrulearn.heuristics import Entropy, WRAcc
from pyrulearn.interfaces.lord import JavaLORD
from pyrulearn.interfaces.weka import WekaJRip
from pyrulearn.learners import NativeRuleLearner
from pyrulearn.learners.opus import Opus
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.learners.seco import AQR, CN2, PFoil, PFossil, Pypper
from pyrulearn.models import DecisionList
from pyrulearn.pruning import LikelihoodRatioTest

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
#: the largest dataset, with a longer cap per fit
LONG_DATASETS = ["letter"]
LONG_FIT_TIMEOUT = 3600.0

# the 69 datasets of demos/base_vs_composite.py and demos/model_vs_rule_weighting.py:
# every small and medium two-class dataset of the catalog, and the multi-class
# datasets of demos/rule_models.py without kropt
BINARY_DATASETS = ["SPECT", "Titanic", "banknote-authentication", "blood-transfusion-service-center",
                   "breast-cancer", "breast-w", "churn", "climate-model-simulation-crashes", "colic",
                   "compas-two-years", "credit-approval", "credit-g", "cylinder-bands", "diabetes",
                   "dresses-sales", "heart-c", "heart-h", "heart-statlog", "heloc", "hepatitis", "ilpd",
                   "ionosphere", "kc1", "kc2", "kr-vs-kp", "mofn-3-7-10", "molecular-biology_promoters",
                   "monks-problems-1", "monks-problems-2", "monks-problems-3", "mushroom", "ozone-level-8hr",
                   "pc1", "pc3", "pc4", "phoneme", "qsar-biodeg", "sick", "sonar", "spambase", "tic-tac-toe",
                   "vote", "wdbc", "wilt"]
MULTICLASS_DATASETS = ["balance-scale", "cmc", "dna", "hayes-roth", "iris", "splice", "wine", "car", "lymph",
                       "vehicle", "anneal", "glass", "solar-flare", "ecoli", "segment", "zoo", "led24", "yeast",
                       "optdigits", "texture", "vowel", "soybean", "primary-tumor", "audiology", "letter"]

# ten small datasets, two of them multi-class
QUICK_DATASETS = ["vote", "tic-tac-toe", "hepatitis", "breast-cancer", "heart-statlog", "credit-approval",
                  "colic", "SPECT", "zoo", "lymph"]
QUICK_FOLDS = 3

HERE = os.path.dirname(__file__)
NAME = "seco_learners_comparison"
REPORT_PATH = os.path.join(HERE, f"{NAME}_report.md")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")
QUICK_REPORT_PATH = os.path.join(HERE, f"{NAME}_quick_report.md")
CACHE_DIR = os.path.join(HERE, "_seco_learners_comparison_cache")

#: the learner whose fit time is the unit of every other's (`time_ratios`)
REFERENCE = "Weka:JRip"
#: not started without `--slow`; the report shows what the cache holds for them
OPTIONAL = ("AQR", "Opus(Laplace)", "Opus(WRAcc)")


def _plot_paths(quick: bool):
    tag = "_quick" if quick else ""
    return {kind: os.path.join(PLOTS_DIR, f"{NAME}{tag}_{kind}.png")
            for kind in ("accuracy", "time", "size", "cd")}


def lord_available() -> bool:
    return os.path.exists(LORD_CLASSPATH.split(os.pathsep)[0])


# exhaustive search is exponential in the rule length. On five small datasets (vote,
# tic-tac-toe, breast-cancer, heart-statlog, hepatitis; 3-fold), 4, 5 or unlimited
# conditions were no more accurate than 3 with either heuristic (WRAcc 80.40% / 80.58% vs.
# 81.20%, Laplace 83.78% vs. 83.96% at 4), at 3-7x the time per extra condition; Laplace with
# 5 and both without a limit timed out (300s) on breast-cancer
OPUS_MAX_CONDITIONS = 3

#: the learners configured here: name -> (learner, constructor arguments, fit arguments)
VARIANTS = {
    "CN2": (CN2, {}, {}),
    "CN2 (beam 1)": (CN2, {"beam_width": 1}, {}),
    "CN2 (ordered)": (CN2, {"heuristic": Entropy(), "filtering": LikelihoodRatioTest()}, {"model": DecisionList}),
    "CN2 (ordered, beam 1)": (CN2, {"heuristic": Entropy(), "filtering": LikelihoodRatioTest(), "beam_width": 1},
                              {"model": DecisionList}),
    "Opus(Laplace)": (Opus, {"max_conditions": OPUS_MAX_CONDITIONS}, {}),
    "Opus(WRAcc)": (Opus, {"heuristic": WRAcc(), "max_conditions": OPUS_MAX_CONDITIONS}, {}),
}


class Variant(NativeRuleLearner):
    """The learner `VARIANTS[name]` describes, named `name` in the results."""

    def __init__(self, name: str, random_state: int = RANDOM_STATE):
        self.name = name
        self.random_state = random_state

    @property
    def display_name(self) -> str:
        return self.name

    def _provenance_params(self):
        learner, construct, fit = VARIANTS[self.name]
        return {"name": self.name, "learner": learner.__name__, "random_state": self.random_state,
                **{k: repr(v) for k, v in {**construct, **fit}.items()}}

    def fit(self, data):
        learner, construct, fit = VARIANTS[self.name]
        return learner(random_state=self.random_state, **construct).fit(data, **fit)


def build_learners():
    learners = [
        Variant("CN2"), Variant("CN2 (beam 1)"), Variant("CN2 (ordered)"), Variant("CN2 (ordered, beam 1)"),
        AQR(random_state=RANDOM_STATE),
        PFoil(random_state=RANDOM_STATE),
        PFossil(random_state=RANDOM_STATE),
        Pypper(random_state=RANDOM_STATE),
        Variant("Opus(Laplace)"), Variant("Opus(WRAcc)"),
        PyLORD(random_state=RANDOM_STATE),
        WekaJRip(jar=WEKA_JAR, java=WEKA_JAVA),
    ]
    if lord_available():
        # one thread, for this comparison only: the reference LORD uses half the machine's cores by
        # default, every other learner here one (`JavaLORD` itself keeps LORD's default)
        learners.append(JavaLORD(classpath=LORD_CLASSPATH, java=LORD_JAVA, thread_count=1))
    return learners


# what the report says about each learner, keyed by display_name
LEARNER_NOTES = {
    "CN2": "Clark & Boswell's (1991) CN2: beam search (width 5) with the Laplace estimate; CN2's "
           "likelihood-ratio significance test, of the rule's class against the rest, filters the "
           "rules a search can return. Rules for every class (one-vs-rest rule set).",
    "CN2 (beam 1)": "The same with a beam of width 1.",
    "CN2 (ordered)": "Clark & Niblett's (1989) original CN2: one decision list over all classes, each "
                     "rule predicting the majority of the examples it covers; beam search (width 5) with "
                     "the entropy of the covered class distribution, and the significance test of that "
                     "distribution over all classes (`LikelihoodRatioTest`).",
    "CN2 (ordered, beam 1)": "The same with a beam of width 1.",
    "AQR": "Clark & Niblett's (1989) AQ: each rule grown from a seed example by beam search over the "
           "seed's generalizations, ranked by the positives covered (then the negatives, then the "
           "length); only consistent rules are accepted. Rules for every class.",
    "PFoil": "FOIL: greedy search that adds the condition with the highest FOIL gain; "
             "Quinlan's encoding-length restriction stops rules that cost more bits "
             "than the examples they explain. Rules for every class.",
    "PFossil": "FOSSIL: hill climbing on the correlation between rule and class; rules "
               "below a correlation of 0.3 are dropped. Rules for every class.",
    "Pypper": "pyrulearn's RIPPER: grow on two thirds of the data, prune on the rest "
              "(IREP*), then optimize the rule set (replace/revise). Rules for the "
              "classes from least to most frequent, the most frequent one as default.",
    "Opus(Laplace)": "Opus: in every covering step the rule with the highest Laplace estimate among all "
                     f"conjunctions of up to {OPUS_MAX_CONDITIONS} conditions, found by exhaustive "
                     "branch-and-bound search (OPUS) -- Webb's setting, the rule CN2's beam search "
                     "approximates. Rules for every class.",
    "Opus(WRAcc)": "The same with weighted relative accuracy, which prefers more general rules.",
    "PyLORD": "A simplified LORD: for every training example, the best rule covering it "
              "(greedy m-estimate search, then pruning); a test example is classified by "
              "the best rule that covers it. Many overlapping rules; not a covering algorithm.",
    "Weka:JRip": "Weka's RIPPER, run as a subprocess on the same binarized data. Fit time "
                 "includes the JVM start. The unit of the run times.",
    "JavaLORD": "The reference LORD implementation (Java), run as a subprocess on the same "
                "binarized data, and, in this demo only, on a single thread: by default it uses half "
                "the machine's cores, which no other learner here does.",
}


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


def _cap(dataset: str) -> float:
    return LONG_FIT_TIMEOUT if dataset in LONG_DATASETS else FIT_TIMEOUT


def time_ratios(results: pd.DataFrame) -> pd.DataFrame:
    """Per learner, the geometric mean over the datasets of its mean fit
    time divided by `REFERENCE`'s on the same dataset: ``ratio``, the
    number of datasets it rests on (``datasets``), and how many of them
    timed out (``capped``). A time-out counts at the cap -- the ratio is
    then a lower bound -- so that a learner failing on exactly the hard
    datasets doesn't look fast; datasets with another error, and those
    the reference didn't finish, are left out. With a geometric mean the
    ratio between two learners doesn't depend on the reference: it only
    sets the unit."""
    rows = []
    for (dataset, learner), g in results.groupby(["dataset", "learner"]):
        timed_out = (g["error"] == "timeout").any()
        ok = g[g["error"].isna()]
        time = _cap(dataset) if timed_out else (ok["fit_time"].mean() if len(ok) else np.nan)
        rows.append({"dataset": dataset, "learner": learner, "time": time, "capped": timed_out})
    t = pd.DataFrame(rows)
    ref = t[(t["learner"] == REFERENCE) & ~t["capped"]].set_index("dataset")["time"]
    t = t[t["dataset"].isin(ref.index) & t["time"].notna()].copy()
    t["log_ratio"] = np.log(t["time"] / t["dataset"].map(ref))
    return t.groupby("learner").agg(ratio=("log_ratio", lambda v: float(np.exp(v.mean()))),
                                    datasets=("log_ratio", "size"), capped=("capped", "sum"))


def _format_ratio(row) -> str:
    text = f"{row['ratio']:.3g}"
    return f"≥ {text}" if row["capped"] else text


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


def _build_description(datasets, n_folds: int, quick: bool, learner_names) -> str:
    n_full = len(BINARY_DATASETS) + len(MULTICLASS_DATASETS)
    mode_note = (
        f"**Quick run** ({len(datasets)} small datasets, {n_folds}-fold) -- a fast sanity "
        f"check, the default with no arguments. Full comparison ({n_full} datasets): "
        f"`python demos/{NAME}.py --full`.\n\n"
        if quick else
        f"**Full run** ({len(datasets)} datasets). Quick sanity check "
        f"instead: `python demos/{NAME}.py` with no arguments.\n\n"
    )
    n_multiclass = sum(1 for d in datasets if d.task == "multiclass")
    long_ = [d.name for d in datasets if d.name in LONG_DATASETS]
    cap_note = (f"Each fit is capped at {FIT_TIMEOUT:.0f}s" + (
        f" ({LONG_FIT_TIMEOUT:.0f}s for {', '.join(f'`{n}`' for n in long_)})" if long_ else ""))
    lord_note = ("" if lord_available() else
                 f"\nThe reference Java LORD is not included in this run (not found at "
                 f"`{LORD_CLASSPATH.split(os.pathsep)[0]}`; set `$LORD_CLASSPATH`).\n")
    learner_lines = "\n".join(f"- **{n}** -- {LEARNER_NOTES.get(n, '')}" for n in learner_names)
    optional = [n for n in OPTIONAL if n in learner_names]
    optional_note = (f"\n{', '.join(optional)} are optional and take most of a full run's time: a run "
                     f"reports them from the cache of the last run with `--slow`, the only one that "
                     f"starts them.\n" if optional else "")
    return f"""\
{mode_note}pyrulearn's separate-and-conquer rule learners compared with each other
and with two external baselines, on {len(datasets) - n_multiclass} two-class and
{n_multiclass} multi-class datasets from `pyrulearn.experiments.catalog` -- the
datasets of the other comparisons. All but the two LORDs learn one rule at a
time and remove the examples it covers; they differ in how a rule is searched
for, when it stops, and whether and how the rule set is pruned. The LORDs, which
learn a locally optimal rule per training example, are included as a reference
point. Every learner runs with its default settings, including its default way
of handling several classes; the variants of CN2 and Opus are named after
what differs.

{learner_lines}
{lord_note}{optional_note}
Protocol: {n_folds}-fold stratified cross-validation
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(discretizer=TreeDiscretizer(max_intervals={MAX_INTERVALS}))`), the test fold binarized
against that same `DataSpec` -- every learner sees the identical binary
features per fold (as vertical bitmaps, the library's default representation; the learners
that need the dense matrix convert them). {cap_note}; a
time-out or an error counts as a failure (and as last in the ranking),
not as the end of the run. Which learners get through the larger datasets
at all is part of the result.{_concurrency_note()}

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: test accuracy, number of rules and conditions, fit time. Fit
times are given relative to {REFERENCE}'s: per dataset, a learner's mean fit
time divided by {REFERENCE}'s, and the geometric mean of these ratios over
the datasets (`time_ratios`). With a geometric mean, the ratio between any
two learners doesn't depend on the reference, which only sets the unit
(note that {REFERENCE}'s times include starting the JVM, which dominates on
the small datasets). A dataset a learner timed out on counts at the cap,
and its ratio is then a lower bound (marked ≥).
"""


def _summary_table(results) -> list:
    """Per learner, sorted by mean rank (best first): accuracy, size, time
    ratio, wins, mean rank and failures -- all within `results`."""
    ranks = mean_rank(results, "accuracy")
    wins = win_counts(results, "accuracy")
    ratios = time_ratios(results)
    summary = results.groupby("learner")[
        ["accuracy", "n_rules", "n_conditions", "conds_per_rule"]].mean(numeric_only=True)
    failures = results.groupby("learner")["error"].apply(lambda e: e.notna().sum())
    lines = [f"| learner | accuracy | n_rules | n_conditions | conds/rule | time (x {REFERENCE}) | "
             "wins | mean rank | failures |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|\n"]
    for learner in ranks.sort_values().index:
        row = summary.loc[learner]
        ratio = _format_ratio(ratios.loc[learner]) if learner in ratios.index else "--"
        lines.append(f"| {learner} | {format_accuracy(row['accuracy'])} | {format_mean_count(row['n_rules'])} | "
                     f"{format_mean_count(row['n_conditions'])} | {format_mean_count(row['conds_per_rule'])} | "
                     f"{ratio} | {wins.get(learner, 0.0):.1f} | "
                     f"{ranks[learner]:.2f} | {int(failures.get(learner, 0))} |\n")
    return lines


def write_report(results, datasets, n_folds: int, report_path: str, plots: dict, quick: bool,
                 learner_names) -> None:
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]
    plots_dir = os.path.basename(PLOTS_DIR)

    lines = ["# Separate-and-conquer rule learners compared\n\n"]
    lines.append(render_setup_section(_build_description(datasets, n_folds, quick, learner_names)))
    lines.append(f"![accuracy vs. rule-set complexity]({plots_dir}/{os.path.basename(plots['accuracy'])})\n\n")
    lines.append(f"![mean rank vs. fit time]({plots_dir}/{os.path.basename(plots['time'])})\n\n")
    lines.append(f"![mean rank vs. rule-set size]({plots_dir}/{os.path.basename(plots['size'])})\n\n")
    lines.append(f"![critical-difference diagram (accuracy)]({plots_dir}/{os.path.basename(plots['cd'])})\n\n")

    n_main = results["dataset"].nunique()
    lines.append(f"## Summary by learner (mean across the {n_main} datasets and their folds)\n\n")
    lines += _summary_table(results)
    lines.append("\nSorted by mean rank. `wins` -- datasets where a learner's mean accuracy was (tied-for-)best, "
                 "a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures "
                 f"tied for last; `time (x {REFERENCE})` -- the geometric mean over the datasets of the "
                 f"fit time relative to {REFERENCE}'s, a lower bound (≥) if it includes time-outs, "
                 "counted at the cap. Accuracy and sizes are means over successful fits only; `failures` "
                 "counts the fits that timed out or raised.\n\n")

    ratios = time_ratios(results)
    capped = ratios[ratios["capped"] > 0]
    if len(capped):
        lines.append("Time-outs in the time ratios: " + ", ".join(
            f"{learner} on {int(row['capped'])} of {int(row['datasets'])} datasets"
            for learner, row in capped.iterrows()) + ".\n\n")

    size = {d.name: d.size for d in datasets}
    by_size = results.assign(size=results["dataset"].map(size))
    for label, rows_note in (("small", "fewer than 1,000 examples"), ("medium", "1,000 to 10,000 examples"),
                             ("large", "more than 10,000 examples")):
        part = by_size[by_size["size"] == label].drop(columns="size")
        if part["dataset"].nunique() and len(set(size.values())) > 1:
            names = sorted(part["dataset"].unique())
            lines.append(f"### {label.capitalize()} datasets only ({len(names)}, {rows_note})\n\n")
            lines.append(", ".join(f"`{n}`" for n in names) + "\n\n")
            lines += _summary_table(part)
            lines.append("\n")

    task = {d.name: d.task for d in datasets}
    if len(set(task.values())) > 1:
        lines.append("## Mean accuracy by target type\n\n")
        lines.append("| learner | two-class | multi-class |\n|---|--:|--:|\n")
        by_task = results.assign(task=results["dataset"].map(task)).groupby(
            ["learner", "task"])["accuracy"].mean()
        for learner in mean_rank(results, "accuracy").sort_values().index:
            b, m = by_task.get((learner, "binary")), by_task.get((learner, "multiclass"))
            lines.append(f"| {learner} | {format_accuracy(b)} | {format_accuracy(m)} |\n")
        lines.append("\n")

    lines.append("## Per-dataset results, per learner (mean across folds)\n\n")
    lines.append(render_results_table(
        results, ["accuracy", "n_rules", "n_conditions", "fit_time"],
        group_by=["dataset", "learner"], include_overall=False))
    lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {report_path}")


#: the color of pyrulearn's own learners in the rank plots, and of the external ones in all plots
NATIVE_COLOR = "#2a6fbb"
EXTERNAL_COLOR = "#7f7f7f"
#: `REFERENCE` and its line in the plot of the fit times
REFERENCE_COLOR = "#ff7f0e"
#: the external learners, by their pyrulearn counterpart: grey, with the counterpart's marker
EXTERNAL = {"Weka:JRip": "Pypper", "JavaLORD": "PyLORD"}
#: one color per pyrulearn learner, from tab20: close variants as the dark and the light version
#: of one hue (beam 5 and beam 1, Laplace and WRAcc), the other learners each a dark hue of their own;
#: tab20's grey is kept for the external learners
COLORS = {"CN2": "#1f77b4", "CN2 (beam 1)": "#aec7e8",
          "CN2 (ordered)": "#ff7f0e", "CN2 (ordered, beam 1)": "#ffbb78",
          "Opus(Laplace)": "#2ca02c", "Opus(WRAcc)": "#98df8a",
          "AQR": "#d62728", "PFoil": "#9467bd", "PFossil": "#8c564b", "Pypper": "#e377c2", "PyLORD": "#bcbd22"}
#: for a learner not in `COLORS`
SPARE_COLORS = ["#17becf", "#9edae5", "#ff9896", "#c5b0d5"]
#: and one marker per learner, so no two share both
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "p", "d"]


def _labeled_scatter(ax, x, y, labels, open_points=(), highlight=None):
    """One point per learner, named next to it -- with a dozen learners, a
    legend of colors would be unreadable. The external learners in grey,
    `highlight` in `REFERENCE_COLOR`."""
    for xi, yi, label in zip(x, y, labels):
        color = (REFERENCE_COLOR if label == highlight else
                 EXTERNAL_COLOR if label in EXTERNAL else NATIVE_COLOR)
        ax.scatter([xi], [yi], s=40, color=color, facecolors="none" if label in open_points else color, zorder=3)
        ax.annotate(label, (xi, yi), xytext=(6, 3), textcoords="offset points", fontsize=8)


def _styles(learner_names):
    """learner -> (color, marker), in legend order: pyrulearn's learners
    each with their own color (`COLORS`) and marker (`MARKERS`), the
    external learners (`EXTERNAL`) last, grey, with their counterpart's
    marker."""
    native = [n for n in learner_names if n not in EXTERNAL]
    spare = iter(SPARE_COLORS * len(native))
    styles = {n: (COLORS.get(n) or next(spare), MARKERS[i % len(MARKERS)]) for i, n in enumerate(native)}
    for learner in learner_names:
        if learner in EXTERNAL:
            counterpart = styles.get(EXTERNAL[learner])
            styles[learner] = (EXTERNAL_COLOR, counterpart[1] if counterpart else "o")
    return styles


def write_plots(results, plots: dict, learner_names) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    ranks = mean_rank(results, "accuracy")

    styles = _styles(learner_names)
    per_pair = results.groupby(["dataset", "learner"], as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for learner in styles:  # in legend order: by family, the external learners last
        sub = per_pair[per_pair["learner"] == learner].dropna(subset=["accuracy"])
        if len(sub):
            color, marker = styles[learner]
            ax.scatter(sub["n_conditions"], sub["accuracy"], label=learner, color=color, marker=marker,
                       alpha=0.7, s=36)
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

    with_timeouts = set(results.loc[results["error"] == "timeout", "learner"])
    ratios = time_ratios(results)
    names = [n for n in ranks.index if n in ratios.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=1.5, zorder=2)
    _labeled_scatter(ax, ratios.loc[names, "ratio"], ranks[names], names,
                     open_points=with_timeouts, highlight=REFERENCE)
    ax.set_xscale("log")
    ax.invert_yaxis()
    for x, text, side in ((0.9, f"\u2190 faster than {REFERENCE}", "right"), (1.1, "slower \u2192", "left")):
        ax.text(x, 0.03, text, transform=ax.get_xaxis_transform(), ha=side, fontsize=9, color=REFERENCE_COLOR)
    ax.set_xlabel(f"fit time relative to {REFERENCE} (geometric mean over the datasets, log)")
    ax.set_ylabel("mean accuracy rank (better up)")
    ax.set_title("accuracy vs. fit time, one point per learner\n(open: includes time-outs, counted at the cap)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots["time"], dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot   -> {plots['time']}")

    ok = results[results["error"].isna() & (results["n_conditions"] > 0)]
    per_pair = ok.groupby(["dataset", "learner"])["n_conditions"].mean()
    size = np.exp(np.log(per_pair).groupby("learner").mean())
    names = [n for n in ranks.index if n in size.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    _labeled_scatter(ax, size[names], ranks[names], names, open_points=with_timeouts)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("conditions in the model (geometric mean over the datasets, log)")
    ax.set_ylabel("mean accuracy rank (better up)")
    ax.set_title("accuracy vs. rule-set size, one point per learner\n(open: includes time-outs)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots["size"], dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot   -> {plots['size']}")

    try:
        ax = critical_difference_diagram(results, "accuracy")
        ax.figure.tight_layout()
        ax.figure.savefig(plots["cd"], dpi=110)
        plt.close(ax.figure)
        print(f"Plot   -> {plots['cd']}")
    except Exception as e:  # noqa: BLE001 -- a plot failure shouldn't sink the report
        print(f"Critical-difference diagram skipped: {type(e).__name__}: {e}")


def main(quick: bool = True, names: Optional[list] = None, slow: bool = False) -> None:
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path = QUICK_REPORT_PATH if quick else REPORT_PATH
    plots = _plot_paths(quick)

    names = names or (QUICK_DATASETS if quick else BINARY_DATASETS + MULTICLASS_DATASETS)
    datasets = Catalog.default().select(names=names)
    learners = build_learners()
    fitted = learners if slow else [l for l in learners if l.display_name not in OPTIONAL]
    optional = [l for l in learners if l not in fitted]

    runs = []
    for group, timeout in (([d for d in datasets if d.name not in LONG_DATASETS], FIT_TIMEOUT),
                           ([d for d in datasets if d.name in LONG_DATASETS], LONG_FIT_TIMEOUT)):
        if group:
            common = dict(n_folds=n_folds, fit_timeout=timeout, cache_dir=CACHE_DIR, random_state=RANDOM_STATE,
                          discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS))
            runs.append(run_cv(fitted, group, n_jobs=_n_jobs(), **common))
            if optional:
                runs.append(run_cv(optional, group, cached_only=True, **common))
    results = pd.concat([r for r in runs if len(r)], ignore_index=True)
    # an optional learner the cache doesn't hold in full would rank on other datasets than the rest
    rows = results.groupby("learner").size()
    incomplete = [l.display_name for l in optional if rows.get(l.display_name, 0) < len(datasets) * n_folds]
    if incomplete:
        print(f"Not in the cache for this run, so left out: {', '.join(incomplete)} (`--slow` runs them)")
        results = results[~results["learner"].isin(incomplete)]
    learner_names = [l.display_name for l in learners if l.display_name not in incomplete]
    write_report(results, datasets, n_folds, report_path, plots, quick, learner_names)
    write_plots(results, plots, learner_names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"run the full comparison ({len(BINARY_DATASETS)} two-class and "
                             f"{len(MULTICLASS_DATASETS)} multi-class datasets) instead of the quick default")
    parser.add_argument("--datasets", default=None,
                        help="comma-separated dataset names for a full-scale run, e.g. 'vote,mushroom'")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    parser.add_argument("--slow", action="store_true",
                        help=f"also run the optional learners ({', '.join(OPTIONAL)}), most of a full "
                             "run's time; without it, the report takes them from the cache")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(quick=not (args.full or args.datasets),
         names=args.datasets.split(",") if args.datasets else None, slow=args.slow)
