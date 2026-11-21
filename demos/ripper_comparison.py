"""
demos/ripper_comparison.py
==========================

Four RIPPER-family rule learners (Slipper included -- Cohen & Singer's
own confidence-rated-boosting follow-up to RIPPER, not a separate
family), cross-validated through `pyrulearn.experiments.runner.run_cv`
(shared per-fold binarization, per-fit timeout, uniform measurement,
fold caching) on datasets from `pyrulearn.experiments.catalog` -- this
demo picks the learners, the datasets, and what to report; everything
else is the shared infrastructure. Two run sizes (see "Run:" below):
**quick** (10 small datasets, the no-args default, output not
checked in) and **full** (`FULL_DATASETS_SPEC` -- every small/medium
binary dataset, 44 of them -- or any other `--datasets` spec, output to
the checked-in report/plots):

- **Weka:JRip**   -- Weka's `weka.classifiers.rules.JRip`, via
                     `pyrulearn.interfaces.weka.WekaJRip` (subprocess,
                     needs a local Weka install -- see `WEKA_JAR`/
                     `WEKA_JAVA` below).
- **Witt:RIPPER** -- the `wittgenstein` package's RIPPER, via
                     `pyrulearn.interfaces.wittgenstein.WittRIPPER`.
                     Natively binary; the local subclass below overrides
                     `_default_model` so `fit(data)` (as `run_cv` always
                     calls it, no `model=`) auto-dispatches to
                     `ConceptModel` for a binary target or `ConceptSet`
                     (one-vs-rest) for multi-class, so one learner
                     instance handles every dataset here.
- **Pypper**      -- pyrulearn's own RIPPER re-implementation
                     (`pyrulearn.learners.seco.Pypper`): IREP* growth/
                     pruning plus `ReplaceReviseOptimization`, per class
                     inside a least-frequent-first ordered decomposition.
- **Slipper**     -- pyrulearn's own SLIPPER (Cohen & Singer, 1999):
                     confidence-rated boosting of rules
                     (`pyrulearn.learners.boosting.Slipper`).

`IMod:Slipper` (`imodels`' `SlipperClassifier`, `pyrulearn.interfaces.
imodels.IModSlipper` -- despite the name, a different algorithm from the
native `Slipper` above, see its own docstring) is deliberately **not**
in that main comparison: `run_preliminary_slipper_check` fits it against
the native `Slipper` on `PRELIM_DATASETS` (a handful of small, few-
feature datasets, so this check itself stays quick) first, and its
result -- consistently much slower, for no accuracy gain (on a couple of
sampled larger datasets it was 20-60x slower than every other learner
here, once timing out outright at a 60s budget on a plain 1,000-row
dataset) -- is the report's own argument for excluding it, not an
unexplained absence. It's binary-only regardless (see `IModSlipper`'s
docstring), so it would have needed dropping from any multi-class run
anyway.

All four main-comparison learners see the identical per-fold binarized
data (`build_dataspec`/`binarize`, `MAX_INTERVALS` discretization), in the
library's default representation, vertical bitmaps (the Weka learners convert it to
the dense matrix they need) -- no learner here gets the raw, non-binarized
data.
That comparison (a learner's own intrinsic numeric handling vs. our
discretization) is a deliberately separate, later demo
(`examples/REVISION_PLAN.md`'s comparison (e)); this one used to also
run Weka JRip on raw data (`jrip_native`) as a preview of that, which is
why the two are related but no longer why they're the same script.

Run: `python demos/ripper_comparison.py` with no arguments is the
**quick** default -- `QUICK_DATASETS_SPEC` (10 small datasets),
`QUICK_FOLDS`-fold, well under a minute; its report/plots go to
``ripper_comparison_quick_*`` and are *not* checked in (see
`.gitignore`) -- rerun it any time for a fast sanity check. `--full` (or
an explicit `--datasets <spec>`) runs the full comparison instead
(`FULL_DATASETS_SPEC`, `N_FOLDS`-fold) and writes to the canonical
``ripper_comparison_report.md``/``.png`` paths, which *are* checked
in -- a sample from a real full run, not regenerated on every change to
this script. Either way needs Weka installed at `WEKA_JAR`/`WEKA_JAVA`
below, plus the `experiments`, `wittgenstein` and `imodels` extras.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from pyrulearn.experiments.catalog import Catalog
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.experiments.report import (
    format_accuracy, format_mean_count, render_results_table, render_setup_section,
)
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import critical_difference_diagram, mean_rank, win_counts
from pyrulearn.interfaces.imodels import IModSlipper
from pyrulearn.interfaces.weka import WekaJRip
from pyrulearn.interfaces.wittgenstein import WittRIPPER as _WittRIPPERBase
from pyrulearn.learners.boosting import Slipper
from pyrulearn.learners.seco import Pypper
from pyrulearn.models import ConceptModel, ConceptSet

# $WEKA_JAR / $WEKA_JAVA win; else this machine's Windows install, else `java` on the PATH
_WIN_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAVA = os.environ.get("WEKA_JAVA") or (_WIN_JAVA if os.path.exists(_WIN_JAVA) else "java")
WEKA_JAR = os.environ.get("WEKA_JAR") or r"C:\Program Files\Weka-3-8-7\weka.jar"

RANDOM_STATE = 0
N_FOLDS = 10             # cheap now that IMod:Slipper -- the one slow learner -- is out of the main run
MAX_INTERVALS = 6        # numeric-feature discretization (build_dataspec)
FIT_TIMEOUT = 180.0      # seconds, per (dataset, fold, learner) -- Weka's JVM startup included

# Slipper vs. IMod:Slipper preliminary check -- see run_preliminary_slipper_check().
# Small, low-feature-count datasets (feature count, not row count, is
# what made IMod:Slipper slow in testing -- e.g. it took ~19s/fit on the
# 208-row but 60-numeric-attribute `sonar`, vs. ~5s/fit on the larger but
# 19-attribute `hepatitis`), so this stays quick even though IMod:Slipper
# is in it.
PRELIM_DATASETS = ["breast-cancer", "heart-statlog", "hepatitis", "vote"]
PRELIM_FOLDS = 3
PRELIM_TIMEOUT = 90.0

HERE = os.path.dirname(__file__)
# The canonical, checked-in output -- reproduced by `--full` (or an
# explicit `--datasets`); see FULL_DATASETS_SPEC/QUICK_DATASETS_SPEC below.
REPORT_PATH = os.path.join(HERE, "ripper_comparison_report.md")
PLOTS_DIR = os.path.join(HERE, "ripper_comparison_plots")
ACCURACY_PLOT_PATH = os.path.join(PLOTS_DIR, "ripper_comparison_accuracy.png")
FIT_TIME_PLOT_PATH = os.path.join(PLOTS_DIR, "ripper_comparison_fit_time.png")
CD_PLOT_PATH = os.path.join(PLOTS_DIR, "ripper_comparison_cd.png")
# The quick, no-args default's output -- *not* checked in (see .gitignore's
# examples/*_quick_report.md / _quick*.png patterns): a fast sanity check
# shouldn't overwrite the committed full-run sample every time it's run.
QUICK_REPORT_PATH = os.path.join(HERE, "ripper_comparison_quick_report.md")
QUICK_ACCURACY_PLOT_PATH = os.path.join(PLOTS_DIR, "ripper_comparison_quick_accuracy.png")
QUICK_FIT_TIME_PLOT_PATH = os.path.join(PLOTS_DIR, "ripper_comparison_quick_fit_time.png")
QUICK_CD_PLOT_PATH = os.path.join(PLOTS_DIR, "ripper_comparison_quick_cd.png")
CACHE_DIR = os.path.join(HERE, "_ripper_comparison_cache")


def _time_cd_path(cd_plot_path: str) -> str:
    """The fit-time critical-difference diagram, next to the accuracy one."""
    return cd_plot_path.replace("_cd.png", "_cd_fit_time.png")


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

# Full run: the 44 small/medium (<=10,000 rows) binary datasets the
# catalog had when this comparison was set up (25 small + 19 medium),
# named explicitly: the catalog has since gained more binary ones (the
# deep-rule-network demo's artificial NNF datasets and car-binary), which
# "binary,small,medium" would now pull in too, changing the comparison.
# `--datasets` overrides this, e.g. '--datasets small,medium' (binary and
# multi-class both -- exercises Witt:RIPPER's multi-class dispatch) or
# '--datasets all' (includes >10,000-row datasets, an overnight-on-a-
# strong-machine run, not a quick one); `--full` is shorthand for this list.
FULL_DATASETS_SPEC = (
    "SPECT,Titanic,banknote-authentication,blood-transfusion-service-center,breast-cancer,"
    "breast-w,churn,climate-model-simulation-crashes,colic,compas-two-years,credit-approval,"
    "credit-g,cylinder-bands,diabetes,dresses-sales,heart-c,heart-h,heart-statlog,heloc,"
    "hepatitis,ilpd,ionosphere,kc1,kc2,kr-vs-kp,mofn-3-7-10,molecular-biology_promoters,"
    "monks-problems-1,monks-problems-2,monks-problems-3,mushroom,ozone-level-8hr,pc1,pc3,pc4,"
    "phoneme,qsar-biodeg,sick,sonar,spambase,tic-tac-toe,vote,wdbc,wilt"
)

# Quick run (the no-args default): 10 hand-picked small datasets, 3-fold
# instead of N_FOLDS -- a sanity check that runs in well under a minute
# (`sonar`'s 60 numeric attributes make it the slowest of the ten, still
# a few seconds), not a statistically rigorous comparison. Named
# explicitly (not e.g. "binary,small,10" picked at random) so it's the
# same ten -- and the same cache hits -- every time. No monks-problems-*
# (synthetic, and all three landing in one quick run skews it towards
# them).
QUICK_DATASETS_SPEC = ("vote,tic-tac-toe,hepatitis,breast-cancer,heart-statlog,"
                       "credit-approval,colic,diabetes,ionosphere,sonar")
QUICK_FOLDS = 3


def _build_description(datasets, n_folds: int, prelim_verdict: str, quick: bool) -> str:
    n_binary = sum(1 for d in datasets if d.task == "binary")
    n_multiclass = len(datasets) - n_binary
    dataset_count = (f"{len(datasets)} binary" if n_multiclass == 0
                    else f"{n_binary} binary and {n_multiclass} multi-class")
    mode_note = (
        f"**Quick run** ({dataset_count} datasets, a fast sanity check, not a "
        f"statistically rigorous comparison -- default with no arguments). For the "
        f"full comparison (44 binary datasets, {N_FOLDS}-fold): "
        f"`python demos/ripper_comparison.py --full` (or an explicit "
        f"`--datasets`); a sample from that run is committed at "
        f"`{os.path.basename(REPORT_PATH)}` (plus its plots) in this directory.\n\n"
        if quick else
        f"**Full run** ({dataset_count} datasets). For a quick sanity check instead "
        f"(10 small datasets, {QUICK_FOLDS}-fold, well under a minute): "
        f"`python demos/ripper_comparison.py` with no arguments -- its "
        f"output isn't checked in (see `{os.path.basename(QUICK_REPORT_PATH)}` "
        f"after running it).\n\n"
    )
    return f"""\
{mode_note}Four RIPPER-family rule learners -- Weka:JRip, Witt:RIPPER,
Pypper, Slipper -- compared on {dataset_count} datasets from
`pyrulearn.experiments.catalog`.

A fifth, IMod:Slipper (`imodels`' SlipperClassifier), is checked
separately first against the native Slipper on a handful of small,
low-feature-count datasets, then left out of the main comparison below --
see "Why IMod:Slipper isn't in the main comparison". {prelim_verdict}

Protocol: `{n_folds}`-fold stratified cross-validation
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(discretizer=TreeDiscretizer(max_intervals={MAX_INTERVALS}))`), the test fold binarized
against that same `DataSpec` -- every learner sees the identical
binary features per fold (as vertical bitmaps, the library's default representation;
the Weka learners convert them to the matrix they need), so differences
reflect the algorithms, not the data preparation. Each fit is capped at {FIT_TIMEOUT:.0f}s; a
timeout or an exception is recorded as a failure, not fatal to the run.

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

All three implementations of RIPPER learn their rules for the same class
on a binary dataset, the minority class, leaving the majority class to the
default rule: Weka:JRip and Pypper by their least-frequent-first ordering,
Witt:RIPPER by its `pos_class` default. And all three search the same
conditions: Weka:JRip and Witt:RIPPER test every Boolean column for both
values, so they get the features without their explicit negations
(`native_negation=True`) and provide those themselves by their `= False`
tests, while Pypper and Slipper, which only test a feature for being true,
get the negations as features.

Measures: test accuracy, rule count, total condition count, fit time, and,
for the external implementations, the run time of the external algorithm
itself (`Provenance.run_time`: Weka's own "Time taken to build model",
wittgenstein's `fit`); the rest of their fit time is preparation --
converting the data, writing and reading files, starting the JVM, importing
the rules. Witt:RIPPER auto-dispatches one-vs-rest for multi-class (see the
module docstring); every other learner handles multi-class
natively.{_concurrency_note()}
"""


class WittRIPPER(_WittRIPPERBase):
    """`WittRIPPER`, but `fit(data)` with no `model=` (as `run_cv` always
    calls it) picks `ConceptModel` for a binary target, `ConceptSet`
    (one-vs-rest) for multi-class -- see the module docstring."""

    def _default_model(self, data) -> type:
        return ConceptModel if len(np.unique(np.asarray(data.y))) <= 2 else ConceptSet


def build_learners():
    return [
        WekaJRip(jar=WEKA_JAR, java=WEKA_JAVA, native_negation=True),
        WittRIPPER(random_state=RANDOM_STATE, native_negation=True),
        Pypper(random_state=RANDOM_STATE),
        Slipper(random_state=RANDOM_STATE),
    ]


LEARNER_ORDER = ["Weka:JRip", "Witt:RIPPER", "Pypper", "Slipper"]
COLORS = dict(zip(LEARNER_ORDER, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
MARKERS = dict(zip(LEARNER_ORDER, ["o", "s", "^", "D"]))


def measure_run_time(model, test_rep) -> dict:
    """`run_cv`'s `measure_fn`: the external algorithm's own run time
    (`Provenance.run_time`), `NaN` for a native learner, whose whole fit
    time is the algorithm's."""
    rt = getattr(model.provenance, "run_time", None) if model.provenance is not None else None
    return {"run_time": np.nan if rt is None else float(rt)}


#: the learner the run times are reported relative to
TIME_REFERENCE = "Pypper"


def run_time_section(results) -> str:
    """The run times relative to `TIME_REFERENCE`: per learner, the mean
    fit time, run time and preparation time in seconds, and each of them
    relative to the reference's fit time (geometric mean over the datasets
    of the per-dataset means). A native learner's run time is its fit
    time, its preparation time zero."""
    per = results.groupby(["dataset", "learner"])[["fit_time", "run_time"]].mean()
    per["run_time"] = per["run_time"].fillna(per["fit_time"])
    per["prep_time"] = (per["fit_time"] - per["run_time"]).clip(lower=0.0)
    ref = per.xs(TIME_REFERENCE, level="learner")["fit_time"]

    def geo(values) -> float:
        values = values.dropna()
        return float(np.exp(np.log(values.clip(lower=1e-6)).mean())) if len(values) else float("nan")

    out = [f"## Run time relative to {TIME_REFERENCE}\n\n",
           f"Seconds per fit (mean over datasets and folds), and relative to {TIME_REFERENCE}'s fit "
           "time (geometric mean over the datasets): the fit time, the run time of the algorithm "
           "itself, and the preparation time (fit time minus run time -- converting the data, "
           "files, the JVM's start, importing the rules; zero for a native learner).\n\n",
           "| learner | fit (s) | run (s) | preparation (s) | fit ×ref | run ×ref | preparation ×ref |\n",
           "|---|--:|--:|--:|--:|--:|--:|\n"]
    for learner in LEARNER_ORDER:
        if learner not in per.index.get_level_values("learner"):
            continue
        sub = per.xs(learner, level="learner")
        rel = sub.div(ref, axis=0)
        native = (sub["prep_time"] == 0).all()
        prep_rel = "--" if native else f"{geo(rel['prep_time']):.3f}"
        out.append(f"| {learner} | {sub['fit_time'].mean():.3f} | {sub['run_time'].mean():.3f} | "
                   f"{sub['prep_time'].mean():.3f} | {geo(rel['fit_time']):.3f} | "
                   f"{geo(rel['run_time']):.3f} | {prep_rel} |\n")
    out.append("\n")
    return "".join(out)


def select_datasets(spec: str):
    return Catalog.default().parse(spec, random_state=RANDOM_STATE)


def run_preliminary_slipper_check():
    """Fits the native `Slipper` against `IMod:Slipper` on
    `PRELIM_DATASETS`, and returns `(results, verdict)` -- `results` the
    usual long-format `run_cv` table (just these two learners, these
    datasets), `verdict` a plain-English sentence built from the actual
    numbers (never a canned claim) comparing their mean accuracy and
    mean fit time, for both the report and the module `DESCRIPTION`.
    """
    datasets = Catalog.default().select(names=PRELIM_DATASETS)
    results = run_cv([Slipper(random_state=RANDOM_STATE), IModSlipper(random_state=RANDOM_STATE)],
                     datasets, n_folds=PRELIM_FOLDS, fit_timeout=PRELIM_TIMEOUT,
                     discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS), random_state=RANDOM_STATE, cache_dir=CACHE_DIR,
                     n_jobs=_n_jobs())
    by_learner = results.groupby("learner").agg(accuracy=("accuracy", "mean"), fit_time=("fit_time", "mean"))
    slipper, imod = by_learner.loc["Slipper"], by_learner.loc["IMod:Slipper"]
    ratio = imod["fit_time"] / slipper["fit_time"] if slipper["fit_time"] else float("inf")
    acc_cmp = ("no more accurate" if imod["accuracy"] <= slipper["accuracy"] else "more accurate")
    verdict = (
        f"Across {len(datasets)} datasets ({PRELIM_FOLDS}-fold), IMod:Slipper was {acc_cmp} than "
        f"Slipper (mean accuracy {format_accuracy(imod['accuracy'])} vs. {format_accuracy(slipper['accuracy'])}) while taking "
        f"{ratio:.0f}x as long per fit ({imod['fit_time']:.2f}s vs. {slipper['fit_time']:.2f}s, mean) "
        f"-- not worth its cost at the scale of the main comparison below."
    )
    return results, verdict


def write_report(results, datasets, n_folds: int, prelim_results, prelim_verdict: str,
                 report_path: str, accuracy_plot_path: str, fit_time_plot_path: str,
                 cd_plot_path: str, quick: bool) -> None:
    task_by_name = {d.name: d.task for d in datasets}
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]

    plots_dir = os.path.basename(os.path.dirname(accuracy_plot_path))
    lines = ["# RIPPER-family comparison: Weka:JRip / Witt:RIPPER / Pypper / Slipper\n\n"]
    lines.append(render_setup_section(_build_description(datasets, n_folds, prelim_verdict, quick)))
    lines.append(f"![accuracy vs. rule-set complexity]({plots_dir}/{os.path.basename(accuracy_plot_path)})\n\n")
    lines.append(f"![fit time per dataset]({plots_dir}/{os.path.basename(fit_time_plot_path)})\n\n")
    lines.append(f"![critical-difference diagram (accuracy)]({plots_dir}/{os.path.basename(cd_plot_path)})\n\n")
    lines.append("The same for fit time, rank 1 the fastest:\n\n")
    lines.append(f"![critical-difference diagram (fit time)]({plots_dir}/{os.path.basename(_time_cd_path(cd_plot_path))})\n\n")

    lines.append("## Why IMod:Slipper isn't in the main comparison\n\n")
    lines.append(f"{prelim_verdict}\n\n")
    lines.append(render_results_table(prelim_results, ["accuracy", "fit_time"], group_by="learner",
                                      include_overall=False))  # averaging across two different
    # algorithms isn't a meaningful "overall" -- see render_results_table's own docstring
    lines.append("\n")

    lines.append("## Summary by learner (mean across every dataset and fold)\n\n")
    ranks = mean_rank(results, "accuracy")
    wins = win_counts(results, "accuracy")
    summary = results.groupby("learner")[["accuracy", "n_rules", "conds_per_rule", "fit_time"]].mean(
        numeric_only=True)
    lines.append("| learner | accuracy | n_rules | conds/rule | fit_time (s) | wins | mean rank |\n"
                "|---|--:|--:|--:|--:|--:|--:|\n")
    for learner in ranks.index:  # ranks is already sorted best-first
        row = summary.loc[learner]
        lines.append(f"| {learner} | {format_accuracy(row['accuracy'])} | {format_mean_count(row['n_rules'])} | "
                     f"{format_mean_count(row['conds_per_rule'])} | {row['fit_time']:.2f} | "
                     f"{wins.get(learner, 0.0):.1f} | {ranks[learner]:.2f} |\n")
    lines.append("\n`wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie "
                "split evenly (`pyrulearn.experiments.stats.win_counts`); `mean rank` -- average "
                "accuracy rank across datasets, failures tied for last (`stats.mean_rank`).\n\n")

    lines.append(run_time_section(results))

    lines.append("## Per-dataset results, per learner (mean across folds)\n\n")
    lines.append(render_results_table(
        results, ["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time", "run_time"],
        group_by=["dataset", "learner"], include_overall=False))  # ditto -- see "Summary by
    # learner" above instead for a meaningful per-learner overall
    lines.append("\n")

    binary_names = {d.name for d in datasets if task_by_name.get(d.name) == "binary"}
    multiclass_names = {d.name for d in datasets if task_by_name.get(d.name) == "multiclass"}
    if binary_names and multiclass_names:  # only worth a section when the run actually mixes both
        lines.append("## Mean accuracy by target type\n\n")
        lines.append("| learner | binary | multi-class |\n|---|--:|--:|\n")
        for learner in LEARNER_ORDER:
            sub = results[results["learner"] == learner]
            bin_acc = sub[sub["dataset"].isin(binary_names)]["accuracy"].mean()
            multi_acc = sub[sub["dataset"].isin(multiclass_names)]["accuracy"].mean()
            lines.append(f"| {learner} | {format_accuracy(bin_acc)} | {format_accuracy(multi_acc)} |\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {report_path}")


def write_plots(results, accuracy_plot_path: str, fit_time_plot_path: str, cd_plot_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        return

    os.makedirs(os.path.dirname(accuracy_plot_path), exist_ok=True)
    per_pair = results.groupby(["dataset", "learner"], as_index=False).mean(numeric_only=True)

    fig1, ax1 = plt.subplots(figsize=(6.5, 5.5))
    for learner in LEARNER_ORDER:
        sub = per_pair[per_pair["learner"] == learner].dropna(subset=["accuracy"])
        if len(sub):
            ax1.scatter(sub["n_conditions"], sub["accuracy"], label=learner,
                       color=COLORS[learner], marker=MARKERS[learner], alpha=0.7, s=45)
    ax1.set_xlabel("total conditions (rule set complexity)")
    ax1.set_ylabel("test accuracy")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_title("accuracy vs. rule-set complexity")
    ax1.set_xscale("symlog")
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(accuracy_plot_path, dpi=110)
    plt.close(fig1)
    print(f"Plot   -> {accuracy_plot_path}")

    names = sorted(per_pair["dataset"].unique())
    x = np.arange(len(names))
    w = 0.8 / len(LEARNER_ORDER)
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    for i, learner in enumerate(LEARNER_ORDER):
        sub = per_pair[per_pair["learner"] == learner].set_index("dataset")
        ts = [sub["fit_time"].get(n, np.nan) for n in names]
        ax2.bar(x + (i - (len(LEARNER_ORDER) - 1) / 2) * w, ts, w, label=learner, color=COLORS[learner])
    ax2.set_yscale("log")
    ax2.set_ylabel("mean fit time (s, log)")
    ax2.set_title("fit time per dataset")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(fit_time_plot_path, dpi=110)
    plt.close(fig2)
    print(f"Plot   -> {fit_time_plot_path}")

    for measure, higher_is_better, path in (("accuracy", True, cd_plot_path),
                                            ("fit_time", False, _time_cd_path(cd_plot_path))):
        try:
            ax = critical_difference_diagram(results, measure, higher_is_better=higher_is_better)
            ax.figure.tight_layout()
            ax.figure.savefig(path, dpi=110)
            plt.close(ax.figure)
            print(f"Plot   -> {path}")
        except Exception as e:  # noqa: BLE001 -- a plot failure shouldn't sink the rest of the report
            print(f"Critical-difference diagram ({measure}) skipped: {type(e).__name__}: {e}")


def main(datasets_spec: Optional[str] = None) -> None:
    """No arguments -> the quick, no-args default (`QUICK_DATASETS_SPEC`,
    `QUICK_FOLDS`, output *not* checked in -- see the module docstring
    and `.gitignore`). Any `datasets_spec` -> a full-scale run
    (`N_FOLDS` folds, output to the canonical, checked-in paths) --
    pass `FULL_DATASETS_SPEC` itself to reproduce the committed sample,
    or any other `Catalog.parse()` spec for a custom full-scale run.
    """
    quick = datasets_spec is None
    spec = QUICK_DATASETS_SPEC if quick else datasets_spec
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path = QUICK_REPORT_PATH if quick else REPORT_PATH
    accuracy_plot_path = QUICK_ACCURACY_PLOT_PATH if quick else ACCURACY_PLOT_PATH
    fit_time_plot_path = QUICK_FIT_TIME_PLOT_PATH if quick else FIT_TIME_PLOT_PATH
    cd_plot_path = QUICK_CD_PLOT_PATH if quick else CD_PLOT_PATH

    print("Preliminary check: Slipper vs. IMod:Slipper ...")
    prelim_results, prelim_verdict = run_preliminary_slipper_check()
    print(prelim_verdict)

    datasets = select_datasets(spec)
    results = run_cv(build_learners(), datasets, n_folds=n_folds, fit_timeout=FIT_TIMEOUT,
                     discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS), random_state=RANDOM_STATE, cache_dir=CACHE_DIR,
                     n_jobs=_n_jobs(), measure_fn=measure_run_time)
    write_report(results, datasets, n_folds, prelim_results, prelim_verdict,
                report_path, accuracy_plot_path, fit_time_plot_path, cd_plot_path, quick)
    write_plots(results, accuracy_plot_path, fit_time_plot_path, cd_plot_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"run the full comparison (44 binary datasets, "
                             f"{N_FOLDS}-fold) instead of the quick default -- shorthand for "
                             f"--datasets {FULL_DATASETS_SPEC!r}")
    parser.add_argument("--datasets", default=None,
                        help=f"Catalog.parse() spec for a full-scale run (any spec here means "
                             f"'full mode': {N_FOLDS}-fold, output to the checked-in report/plots, "
                             f"not the quick ones) -- e.g. 'small,medium' (83, binary and "
                             "multi-class), 'all' (100 -- an overnight run) or 'vote,mushroom'. "
                             "With neither --full nor --datasets, runs the quick default instead "
                             f"({QUICK_DATASETS_SPEC.count(',') + 1} small datasets, "
                             f"{QUICK_FOLDS}-fold, output not checked in).")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(args.datasets or (FULL_DATASETS_SPEC if args.full else None))
