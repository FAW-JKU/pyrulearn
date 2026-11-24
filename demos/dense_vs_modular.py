"""
demos/dense_vs_modular.py
=========================

Dense vs. modular: the same rule-learning algorithm written twice -- once
specialized for speed on a dense matrix (`DenseCPAR`, `DenseENDER`,
`DenseBoomer`, `DenseORB`, `DenseLRI`), once built from the library's
exchangeable components (`CPAR`, `ENDER`, `Boomer`, `ORB`, `LRI`: a
`RuleSearch`, an `Objective`, a covering or boosting loop), so that it
runs on every data representation and takes any search. This demo asks
what the components cost -- or save -- and what they allow:

- **Pairs**: each dense learner next to its modular twin, on the same
  data: the dense one on the Boolean matrix it is written for, the modular
  one on vertical bitmaps, the library's default representation. They should learn
  the same models (up to ties and, for
  `LRI`, the conditions a term already implies), so the report counts how
  often they do and compares fit times.
- **Variants**: the modular learners take other settings than the dense
  ones -- `ORB` and `LRI` with a beam search of width 5 and 10 (the dense
  versions have their own greedy and exhaustive searches only). `CPAR`
  without branching (PRM) and `ENDER` with the exponential loss (CS-Exp)
  and with the Newton criterion (MLRules) are further settings, run as
  pairs too.
- **Horizontal bitmap**: the modular learners again on `HorizontalBitmapRepresentation`,
  to compare them with the vertical bitmaps they run on in the main comparison.

Exhaustive `ORB` (branch and bound over all conjunctions) is checked
separately on a few very small datasets, where it finishes at all, and
left out of the main comparison -- see "Why exhaustive ORB isn't in the
main comparison" in the report.

Learners (binary datasets only: `ORB` boosts a binary loss):

- **CPAR / DenseCPAR**, **CPAR PRM / DenseCPAR PRM** --
  `pyrulearn.learners.cpar` (Yin & Han 2003), and its setting without
  branching (`gain_similarity=1`).
- **ENDER / DenseENDER**, **ENDER CS-Exp / DenseENDER CS-Exp**,
  **ENDER MLRules / DenseENDER MLRules** -- the constant-step logit
  setting, the exponential loss, and the Newton criterion with half the
  examples per round, with the paper's defaults (500 rules).
- **Boomer / DenseBoomer** -- BOOMER's single-label case (1000 rules).
- **ORB(greedy) / DenseORB(greedy)**, **ORB(beam 5)**, **ORB(beam 10)** --
  optimal rule boosting (10 rules) with the greedy `HillClimbing`, and a
  `BeamSearch` for the modular one.
- **LRI / DenseLRI**, **LRI(beam 5)**, **LRI(beam 10)** -- Lightweight
  Rule Induction (50 rules per class).

Run: `python demos/dense_vs_modular.py` with no arguments is the **quick**
default (`QUICK_DATASETS_SPEC`, `QUICK_FOLDS`-fold); its report/plots go
to ``dense_vs_modular_quick_*`` and are not checked in. `--full` runs all
the small and medium binary datasets of the catalog (10-fold for the
small, 5-fold for the medium ones) and writes the canonical
``dense_vs_modular_report.md`` (plots in ``dense_vs_modular_plots/``).
Needs the `experiments` extra.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
import pandas as pd

from pyrulearn.data import HorizontalBitmapRepresentation, default_representation
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import format_accuracy, format_mean_count, render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import critical_difference_diagram, mean_rank, win_counts
from pyrulearn.learners.boosting import (
    ENDER, ORB, Boomer, DenseBoomer, DenseENDER, DenseORB, ExponentialLoss, Newton,
)
from pyrulearn.learners.cpar import CPAR, DenseCPAR
from pyrulearn.learners.boosting import LRI, DenseLRI
from pyrulearn.learners.seco import BeamSearch, BranchAndBoundSearch, HillClimbing

RANDOM_STATE = 0
N_FOLDS = 10
MEDIUM_FOLDS = 5          # for the catalog's "medium" datasets
MAX_INTERVALS = 8
FIT_TIMEOUT = 300.0

# every binary dataset of the catalog with fewer than 10,000 examples -- ORB boosts a
# binary loss, so the multi-class datasets are out; the artificial concepts stay in
FULL_DATASETS_SPEC = "binary,small,medium"

# six small datasets, a mix of symbolic and numeric attributes
QUICK_DATASETS_SPEC = "vote,tic-tac-toe,hepatitis,breast-cancer,heart-statlog,SPECT"
QUICK_FOLDS = 3
QUICK_MAX_INTERVALS = 4          # coarser discretization keeps the quick run and the probes fast

# the exhaustive-ORB check: datasets small enough for branch and bound over all
# conjunctions to finish within the cap (a few seconds on the first three, 70-85s
# on tic-tac-toe), and one where it doesn't (breast-cancer)
EXHAUSTIVE_DATASETS = ["monks-problems-3", "mofn-3-7-10", "vote", "tic-tac-toe", "breast-cancer"]
QUICK_EXHAUSTIVE_DATASETS = ["monks-problems-3", "vote"]
EXHAUSTIVE_FOLDS = 3
EXHAUSTIVE_TIMEOUT = 120.0

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "dense_vs_modular"
REPORT_PATH = os.path.join(HERE, f"{NAME}_report.md")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")
QUICK_REPORT_PATH = os.path.join(HERE, f"{NAME}_quick_report.md")
CACHE_DIR = os.path.join(HERE, "_dense_vs_modular_cache")


# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


def _concurrency_note() -> str:
    if N_JOBS.rstrip("!") == "1":
        return "Fits run one at a time."
    return (f"Fits run {N_JOBS.rstrip('!')} at a time (`run_cv`'s `n_jobs`), each in its own process and "
            "timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its "
            "relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 "
            "concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and "
            "rankings alike); time ratios between learners are reproducible to about +-10% either way.")


def _plot_paths(quick: bool):
    tag = "_quick" if quick else ""
    return {kind: os.path.join(PLOTS_DIR, f"{NAME}{tag}_{kind}.png")
            for kind in ("fit_time", "accuracy", "cd")}


# -- the learners ------------------------------------------------------------------

def _labeled(cls):
    """`cls` named by a `label` in the results (the same class is run with
    different settings). A module-level class, so a worker process can
    unpickle it by reference."""
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


LabeledCPAR = _labeled(CPAR)
LabeledDenseCPAR = _labeled(DenseCPAR)
LabeledENDER = _labeled(ENDER)
LabeledDenseENDER = _labeled(DenseENDER)
LabeledBoomer = _labeled(Boomer)
LabeledDenseBoomer = _labeled(DenseBoomer)
LabeledORB = _labeled(ORB)
LabeledDenseORB = _labeled(DenseORB)
LabeledLRI = _labeled(LRI)
LabeledDenseLRI = _labeled(DenseLRI)

# (family, dense learner, modular twin)
PAIRS = [
    ("CPAR", "DenseCPAR", "CPAR"),
    ("CPAR PRM", "DenseCPAR PRM", "CPAR PRM"),
    ("ENDER", "DenseENDER", "ENDER"),
    ("ENDER CS-Exp", "DenseENDER CS-Exp", "ENDER CS-Exp"),
    ("ENDER MLRules", "DenseENDER MLRules", "ENDER MLRules"),
    ("Boomer", "DenseBoomer", "Boomer"),
    ("ORB", "DenseORB(greedy)", "ORB(greedy)"),
    ("LRI", "DenseLRI", "LRI"),
]
DENSE = [d for _, d, _ in PAIRS]
MODULAR = [m for _, _, m in PAIRS] + ["ORB(beam 5)", "ORB(beam 10)", "LRI(beam 5)", "LRI(beam 10)"]
FAMILY_COLORS = {"CPAR": "tab:blue", "CPAR PRM": "tab:cyan", "ENDER": "tab:orange", "ENDER CS-Exp": "tab:green",
                 "ENDER MLRules": "tab:olive", "Boomer": "tab:red", "ORB": "tab:purple", "LRI": "tab:brown"}


def build_learners():
    # LRI grows a term for at most 5 conditions, with any search; its adaptive error
    # weight isn't comparable across steps, so a beam can't prune on its bound
    lri_beam = lambda w: BeamSearch(beam_width=w, max_conditions=5, optimistic_pruning=False)  # noqa: E731
    return [
        LabeledCPAR("CPAR"), LabeledDenseCPAR("DenseCPAR"),
        LabeledCPAR("CPAR PRM", gain_similarity=1.0), LabeledDenseCPAR("DenseCPAR PRM", gain_similarity=1.0),
        LabeledENDER("ENDER"), LabeledDenseENDER("DenseENDER"),
        LabeledENDER("ENDER CS-Exp", loss=ExponentialLoss()),
        LabeledDenseENDER("DenseENDER CS-Exp", loss=ExponentialLoss()),
        LabeledENDER("ENDER MLRules", method=Newton(), subsample=0.5),
        LabeledDenseENDER("DenseENDER MLRules", method=Newton(), subsample=0.5),
        LabeledBoomer("Boomer"), LabeledDenseBoomer("DenseBoomer"),
        LabeledORB("ORB(greedy)", search=HillClimbing()), LabeledDenseORB("DenseORB(greedy)", search="greedy"),
        LabeledORB("ORB(beam 5)", search=BeamSearch(beam_width=5)),
        LabeledORB("ORB(beam 10)", search=BeamSearch(beam_width=10)),
        LabeledLRI("LRI"), LabeledDenseLRI("DenseLRI"),
        LabeledLRI("LRI(beam 5)", search=lri_beam(5)), LabeledLRI("LRI(beam 10)", search=lri_beam(10)),
    ]


def build_exhaustive_learners():
    return [
        LabeledDenseORB("DenseORB(exhaustive)", search="exhaustive"),
        LabeledORB("ORB(exhaustive)", search=BranchAndBoundSearch()),
        LabeledDenseORB("DenseORB(greedy)", search="greedy"), LabeledORB("ORB(greedy)", search=HillClimbing()),
        LabeledORB("ORB(beam 5)", search=BeamSearch(beam_width=5)),
        LabeledORB("ORB(beam 10)", search=BeamSearch(beam_width=10)),
    ]


def digest(model, test_rep) -> dict:
    """What identifies a model up to the order of its rules, for counting how
    often a dense learner and its modular twin learn the same one."""
    text = "\n".join(sorted(str(model).splitlines()))
    return {"digest": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]}


def _is_dense(learner) -> bool:
    return learner.display_name.startswith("Dense")


def run_on_own_representation(learners, datasets, n_folds, **kwargs):
    """`run_cv` with each learner on the representation it is meant for: the
    dense learners on the Boolean matrix they are written for, the modular
    ones on the library's default (vertical bitmaps). (Handing a dense learner
    another representation would make it decode the matrix inside its fit and count that in its time.)"""
    parts = [run_cv(group, datasets, n_folds=n_folds, representation=representation, **kwargs)
             for group, representation in (([l for l in learners if _is_dense(l)], HorizontalBitmapRepresentation),
                                           ([l for l in learners if not _is_dense(l)], default_representation()))
             if group]
    return pd.concat(parts, ignore_index=True)


# the roster for the report, one entry per algorithm: (name, description, [(variant, what differs)]);
# the variants also have a dense twin where the description says so
LEARNER_GROUPS = [
    ("CPAR", "Classification based on Predictive Association Rules (Yin & Han 2003), built from "
             "`GainAscentHillClimbing` with branching and a `WeightedCovering`; `DenseCPAR` is the "
             "same learner with a hand-written dense search.",
     [("CPAR PRM", "`gain_similarity=1`: only exact ties branch, which is close to PRM, the variant of "
                   "the paper without the copies of a rule (`DenseCPAR PRM` on the dense search).")]),
    ("ENDER", "Gradient boosting of rules, constant-step criterion and logistic loss (CS-Log), 500 "
              "rules, grown through the representation's own counting; `DenseENDER` is the same "
              "learner on a dense float matrix, and each variant has a dense twin too.",
     [("ENDER CS-Exp", "the exponential loss -- the best-ranked setting of the paper."),
      ("ENDER MLRules", "the Newton criterion and half the examples per round -- MLRules "
                        "(Dembczynski et al. 2008).")]),
    ("Boomer", "BOOMER's single-label case (Newton criterion, L2-regularized, 1000 rules) as an "
               "ENDER setting; `DenseBoomer` is the same on a dense float matrix.", []),
    ("ORB", "Optimal rule boosting (10 rules), each rule found by a `RuleSearch` on the XGBoost-style "
            "gain. `ORB(greedy)` uses a greedy `HillClimbing`, and `DenseORB(greedy)` is the same "
            "learner with a hand-written dense greedy search. The dense version's exhaustive search "
            "is checked separately (see below).",
     [("ORB(beam 5)", "a `BeamSearch` of width 5."),
      ("ORB(beam 10)", "a `BeamSearch` of width 10 (`ORB`'s default).")]),
    ("LRI", "Lightweight Rule Induction (50 DNF rules per class): terms by a greedy `HillClimbing` "
            "without a local-optimum stop on the weighted error `FP + k*FN`; `DenseLRI` is the same "
            "learner with a hand-written dense term search.",
     [("LRI(beam 5)", "terms grown by a `BeamSearch` of width 5 (at most 5 conditions)."),
      ("LRI(beam 10)", "terms grown by a `BeamSearch` of width 10.")]),
]


# -- the exhaustive-ORB check ------------------------------------------------------------

_pct = format_accuracy


def run_exhaustive_check(quick: bool):
    """Fits exhaustive `ORB` (dense and modular) against its greedy and beam variants
    on a few very small datasets -- the only place it is run -- and returns
    `(results, verdict)`: the `run_cv` table and a sentence built from the numbers."""
    names = QUICK_EXHAUSTIVE_DATASETS if quick else EXHAUSTIVE_DATASETS
    datasets = Catalog.default().select(names=names)
    results = run_on_own_representation(
        build_exhaustive_learners(), datasets, EXHAUSTIVE_FOLDS, fit_timeout=EXHAUSTIVE_TIMEOUT,
        discretizer=TreeDiscretizer(max_intervals=QUICK_MAX_INTERVALS if quick else MAX_INTERVALS), random_state=RANDOM_STATE,
        cache_dir=CACHE_DIR, n_jobs=_n_jobs())
    ok = results[results["error"].isna()]
    m = ok.groupby("learner")[["accuracy", "n_conditions", "fit_time"]].mean()
    failed = results[results["error"].notna()].groupby("learner").size()
    ex, beam10, greedy = m.loc["ORB(exhaustive)"], m.loc["ORB(beam 10)"], m.loc["ORB(greedy)"]
    verdict = (
        f"On {len(names)} very small datasets ({', '.join(names)}), exhaustive `ORB` reached a mean accuracy of "
        f"{_pct(ex['accuracy'])} against {_pct(beam10['accuracy'])} for a beam search of width 10 and "
        f"{_pct(greedy['accuracy'])} for greedy search (over the fits that finished), in {ex['fit_time']:.1f}s "
        f"per fit against {beam10['fit_time']:.2f}s and {greedy['fit_time']:.2f}s -- "
        f"{ex['fit_time'] / beam10['fit_time']:.0f}x as long as the beam search"
        + (f"; {int(failed.get('ORB(exhaustive)', 0))} of its {len(results[results['learner'] == 'ORB(exhaustive)'])} "
           f"fits hit the {EXHAUSTIVE_TIMEOUT:.0f}s cap" if failed.get("ORB(exhaustive)", 0) else "")
        + ". Branch and bound is exponential in the number of conditions it may combine, so it doesn't "
          "scale to the feature spaces of most datasets (dozens to hundreds of binary features after "
          "discretizing numeric attributes) -- it is left out of the main comparison."
    )
    return results, verdict


# -- the report ------------------------------------------------------------------------------

def _build_description(datasets, quick: bool, exhaustive_verdict: str) -> str:
    n_small = sum(1 for d in datasets if d.size == "small")
    n_medium = sum(1 for d in datasets if d.size == "medium")
    mode_note = (
        f"**Quick run** ({len(datasets)} small datasets, {QUICK_FOLDS}-fold) -- a fast sanity check, "
        f"the default with no arguments. Full comparison (all small and medium binary datasets): "
        f"`python demos/{NAME}.py --full`.\n\n"
        if quick else
        f"**Full run** ({n_small} small" + (f" and {n_medium} medium" if n_medium else "")
        + " binary datasets" + ("" if n_medium else "; the medium ones were left out") + "). Quick sanity check "
        f"instead: `python demos/{NAME}.py` with no arguments.\n\n"
    )
    folds_note = (f"{QUICK_FOLDS}-fold stratified cross-validation" if quick else
                  f"{N_FOLDS}-fold stratified cross-validation"
                  + (f" ({MEDIUM_FOLDS}-fold for the medium datasets)" if n_medium else ""))
    learner_lines = "\n".join(
        f"- **{name}** -- {text}" + "".join(f"\n  - **{v}** -- {note}" for v, note in variants)
        for name, text, variants in LEARNER_GROUPS)
    return f"""\
{mode_note}The same rule-learning algorithms written twice: specialized for speed on a dense
matrix (`Dense...`), and built from the library's exchangeable components -- a
`RuleSearch`, an `Objective`, a covering or boosting loop -- so that they run on every
data representation and take any search. Each dense learner is paired with its
modular twin on the same data -- the dense one on the Boolean matrix it is written for, the
modular one on vertical bitmaps, the library's default representation; the modular learners are then
run again on the horizontal bitmap; and the modular `ORB` and `LRI` are run
with a beam search of width 5 and 10, which the dense versions don't have.

{learner_lines}

All learners run with their defaults (listed above) on {len(datasets)} binary datasets from
`pyrulearn.experiments.catalog` -- {'every one of the small and medium size classes in the full run' if n_medium else 'every one of the small size class'}, numeric and symbolic attributes alike; `ORB` boosts a binary loss, so no multi-class
datasets. Exhaustive `ORB` was checked separately and left out -- see "Why exhaustive ORB
isn't in the main comparison". {exhaustive_verdict}

Protocol: {folds_note}
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(discretizer=TreeDiscretizer(max_intervals={QUICK_MAX_INTERVALS if quick else MAX_INTERVALS}))`), the test fold encoded against that same
`DataSpec` -- every learner sees the identical binary features per fold, the dense learners as a
Boolean matrix and the modular ones as vertical bitmaps; the `[HBitmap]` rows are the modular learners on
the same data as horizontal bitmaps. Each fit is capped at
{FIT_TIMEOUT:.0f}s; a time-out or an error counts as a failure (and as last in the ranking), not as the end
of the run. Fit times are wall-clock seconds of one fit in a worker process (see below for
how many ran at once).

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: test accuracy, number of rules and conditions, fit time, and whether a dense
learner and its modular twin learned the same model (the same rules, whatever their order).

Every learner runs single-threaded. {_concurrency_note()}
"""


def _summary_table(results) -> list:
    ranks = mean_rank(results, "accuracy")
    wins = win_counts(results, "accuracy")
    summary = results[results["error"].isna()].groupby("learner")[
        ["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time"]].mean(numeric_only=True)
    failures = results.groupby("learner")["error"].apply(lambda e: e.notna().sum())
    lines = ["| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | "
             "wins | mean rank | failures |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|\n"]
    for learner in ranks.sort_values().index:
        row = summary.loc[learner]
        lines.append(f"| {learner} | {_pct(row['accuracy'])} | {format_mean_count(row['n_rules'])} | "
                     f"{format_mean_count(row['n_conditions'])} | {format_mean_count(row['conds_per_rule'])} | "
                     f"{row['fit_time']:.2f} | {wins.get(learner, 0.0):.1f} | "
                     f"{ranks[learner]:.2f} | {int(failures.get(learner, 0))} |\n")
    return lines


def _per_dataset(results, column: str, fmt) -> list:
    """One row per dataset, one column per learner (in roster order), the mean of `column`."""
    pivot = results.groupby(["dataset", "learner"])[column].mean().unstack("learner")
    order = [l for l in list(MODULAR) + list(DENSE) if l in pivot.columns] + \
            [l for l in pivot.columns if l not in MODULAR and l not in DENSE]
    pivot = pivot[order]
    lines = ["| dataset | " + " | ".join(order) + " |\n", "|---|" + "--:|" * len(order) + "\n"]
    for ds, row in pivot.iterrows():
        lines.append(f"| {ds} | " + " | ".join("" if pd.isna(v) else fmt(v) for v in row) + " |\n")
    return lines


def _pair_table(results, size_of: dict) -> list:
    """Dense vs. modular, per pair: identical models, accuracy, geometric-mean fit-time ratio."""
    ok = results[results["error"].isna()]
    lines = ["| pair | identical models | accuracy dense | accuracy modular | difference (points) | "
             "time modular / dense, small | medium | all |\n|---|--:|--:|--:|--:|--:|--:|--:|\n"]
    for family, dense, modular in PAIRS:
        d, m = ok[ok["learner"] == dense], ok[ok["learner"] == modular]
        joined = d.merge(m, on=["dataset", "fold"], suffixes=("_d", "_m"))
        if joined.empty:
            continue
        same = int((joined["digest_d"] == joined["digest_m"]).sum())
        acc_d, acc_m = joined["accuracy_d"].mean(), joined["accuracy_m"].mean()
        per_ds = joined.groupby("dataset")[["fit_time_d", "fit_time_m"]].mean()
        ratio = per_ds["fit_time_m"] / per_ds["fit_time_d"]

        def gmean(names):
            r = ratio[[n for n in ratio.index if n in names]]
            return f"{np.exp(np.log(r).mean()):.2f}" if len(r) else "--"
        small = {n for n, s in size_of.items() if s == "small"}
        medium = {n for n, s in size_of.items() if s == "medium"}
        lines.append(f"| {family} | {same}/{len(joined)} | {_pct(acc_d)} | {_pct(acc_m)} | "
                     f"{100 * (acc_m - acc_d):+.2f} | {gmean(small)} | {gmean(medium)} | "
                     f"{gmean(set(ratio.index))} |\n")
    return lines


def _boolean_table(results, boolean_results) -> list:
    """Each modular learner on the horizontal bitmap against the same learner on the vertical one."""
    ok = results[results["error"].isna()]
    bo = boolean_results[boolean_results["error"].isna()].copy()
    bo["learner"] = bo["learner"].str.replace(" [HBitmap]", "", regex=False)
    lines = ["| learner | accuracy, VBitmap | accuracy, HBitmap | time HBitmap / VBitmap (geometric mean over datasets) |\n"
             "|---|--:|--:|--:|\n"]
    for name in MODULAR:
        a, b = ok[ok["learner"] == name], bo[bo["learner"] == name]
        joined = a.merge(b, on=["dataset", "fold"], suffixes=("_n", "_b"))
        if joined.empty:
            continue
        per_ds = joined.groupby("dataset")[["fit_time_n", "fit_time_b"]].mean()
        ratio = np.exp(np.log(per_ds["fit_time_b"] / per_ds["fit_time_n"]).mean())
        lines.append(f"| {name} | {_pct(joined['accuracy_n'].mean())} | {_pct(joined['accuracy_b'].mean())} | "
                     f"{ratio:.2f} |\n")
    return lines


def write_report(results, boolean_results, datasets, report_path: str, plots: dict, quick: bool,
                 exhaustive_results, exhaustive_verdict: str) -> None:
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]
    plots_dir = os.path.basename(PLOTS_DIR)
    size_of = {d.name: d.size for d in datasets}

    lines = ["# Dense and modular rule learners compared\n\n"]
    lines.append(render_setup_section(_build_description(datasets, quick, exhaustive_verdict)))
    lines.append(f"![fit time, modular vs. dense]({plots_dir}/{os.path.basename(plots['fit_time'])})\n\n")
    lines.append(f"![accuracy, modular vs. dense]({plots_dir}/{os.path.basename(plots['accuracy'])})\n\n")
    lines.append(f"![critical-difference diagram (accuracy)]({plots_dir}/{os.path.basename(plots['cd'])})\n\n")

    lines.append("## Why exhaustive ORB isn't in the main comparison\n\n")
    lines.append(f"{exhaustive_verdict}\n\n")
    ex = exhaustive_results[exhaustive_results["error"].isna()]
    lines.append("Accuracy (mean over the folds that finished):\n\n")
    lines += _per_dataset(ex, "accuracy", _pct)
    lines.append("\nFit time in seconds (mean over the folds that finished; a blank cell: every fold "
                 "hit the cap):\n\n")
    lines += _per_dataset(ex, "fit_time", lambda v: f"{v:.2f}")
    lines.append("\n")

    lines.append("## Dense vs. modular\n\n")
    lines.append("For each pair, over the folds where both fits finished: how often the two learned the "
                 "same model (the same rules in any order), their mean accuracy, and the ratio of the "
                 "modular to the dense fit time -- the geometric mean over datasets, so a value below 1 "
                 "means the modular learner is faster. The dense learner on the Boolean matrix, the modular one on "
                 "vertical bitmaps (the default); the modular learners on the horizontal bitmap are compared with "
                 "them below.\n\n")
    lines += _pair_table(results, size_of)
    lines.append("\n")

    lines.append(f"## Summary by learner (mean across the {results['dataset'].nunique()} datasets and their folds)\n\n")
    lines += _summary_table(results)
    lines.append("\nSorted by mean rank. `wins` -- datasets where a learner's mean accuracy was (tied-for-)best, "
                 "a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied "
                 "for last. Means over successful fits only; `failures` counts the fits that timed out or "
                 "raised. A dense learner and its twin that learn the same models tie for rank, so the "
                 "ranking separates the algorithms and the searches, not dense from modular.\n\n")

    by_size = results.assign(size=results["dataset"].map(size_of))
    for label, rows_note in (("small", "fewer than 1,000 examples"), ("medium", "1,000 to 10,000 examples")):
        part = by_size[by_size["size"] == label].drop(columns="size")
        if part["dataset"].nunique() and len(set(size_of.values())) > 1:
            names = sorted(part["dataset"].unique())
            lines.append(f"### {label.capitalize()} datasets only ({len(names)}, {rows_note})\n\n")
            lines.append(", ".join(f"`{n}`" for n in names) + "\n\n")
            lines += _summary_table(part)
            lines.append("\n")

    lines.append("## Beam search for ORB and LRI\n\n")
    lines.append("The dense `ORB` and `LRI` come with their own greedy (and, for `ORB`, exhaustive) search; "
                 "the modular ones take any `RuleSearch`. Mean over all datasets:\n\n")
    ok = results[results["error"].isna()]
    lines.append("| learner | accuracy | n_conditions | fit_time (s) |\n|---|--:|--:|--:|\n")
    for name in ("ORB(greedy)", "ORB(beam 5)", "ORB(beam 10)", "LRI", "LRI(beam 5)", "LRI(beam 10)"):
        sub = ok[ok["learner"] == name]
        if len(sub):
            lines.append(f"| {name} | {_pct(sub['accuracy'].mean())} | {format_mean_count(sub['n_conditions'].mean())} | "
                         f"{sub['fit_time'].mean():.2f} |\n")
    lines.append("\n")

    if len(boolean_results):
        lines.append("## The modular learners on the horizontal bitmap\n\n")
        lines.append("The modular learners above run on `VerticalBitmapRepresentation`, the library's default. Here "
                     "they are run again on `HorizontalBitmapRepresentation`, the matrix the dense learners work "
                     "on, over the folds where both finished; a time ratio above 1 means the vertical bitmap is "
                     "faster:\n\n")
        lines += _boolean_table(results, boolean_results)
        lines.append("\n")

    lines.append("## Per-dataset results\n\nAccuracy (mean over the folds that finished):\n\n")
    lines += _per_dataset(ok, "accuracy", _pct)
    lines.append("\nFit time in seconds (mean over the folds that finished):\n\n")
    lines += _per_dataset(ok, "fit_time", lambda v: f"{v:.2f}")
    lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {report_path}")


def write_plots(results, plots: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    per = results[results["error"].isna()].groupby(["dataset", "learner"], as_index=False).mean(numeric_only=True)

    for measure, key, title, log in (("fit_time", "fit_time", "mean fit time per dataset (s)", True),
                                     ("accuracy", "accuracy", "mean test accuracy per dataset", False)):
        fig, ax = plt.subplots(figsize=(6, 6))
        lo, hi = np.inf, -np.inf
        for family, dense, modular in PAIRS:
            d = per[per["learner"] == dense].set_index("dataset")[measure]
            m = per[per["learner"] == modular].set_index("dataset")[measure]
            both = d.index.intersection(m.index)
            if not len(both):
                continue
            x, y = d[both].to_numpy(), m[both].to_numpy()
            if not log:
                x, y = 100 * x, 100 * y
            ax.scatter(x, y, label=family, color=FAMILY_COLORS[family], alpha=0.7, s=36)
            lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
        if np.isfinite(lo):
            ax.plot([lo, hi], [lo, hi], color="gray", lw=1, ls="--", label="equal")
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        suffix = "" if log else " (%)"
        ax.set_xlabel(f"dense, {title}{suffix}")
        ax.set_ylabel(f"modular, {title}{suffix}")
        ax.set_title(("below" if log else "on") + " the line: " + ("modular faster" if log else "same accuracy"))
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots[measure], dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot   -> {plots[measure]}")

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
    print("Preliminary check: exhaustive ORB ...")
    exhaustive_results, exhaustive_verdict = run_exhaustive_check(quick)
    print(exhaustive_verdict)

    learners = build_learners()
    modular = [l for l in learners if l.display_name in MODULAR]
    groups = [(datasets, n_folds)] if quick else [
        ([d for d in datasets if d.size != "medium"], n_folds),
        ([d for d in datasets if d.size == "medium"], MEDIUM_FOLDS),
    ]
    common = dict(fit_timeout=FIT_TIMEOUT, discretizer=TreeDiscretizer(max_intervals=QUICK_MAX_INTERVALS if quick else MAX_INTERVALS), random_state=RANDOM_STATE,
                  cache_dir=CACHE_DIR, measure_fn=digest, n_jobs=_n_jobs())
    results = pd.concat([
        run_on_own_representation(learners, group, folds, **common)
        for group, folds in groups if group
    ], ignore_index=True)
    boolean_results = pd.concat([
        run_cv(modular, group, n_folds=folds, representation=HorizontalBitmapRepresentation, **common)
        for group, folds in groups if group
    ], ignore_index=True)
    boolean_results["learner"] = boolean_results["learner"] + " [HBitmap]"

    write_report(results, boolean_results, datasets, report_path, plots, quick, exhaustive_results,
                 exhaustive_verdict)
    write_plots(results, plots)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="run all the small and medium binary datasets instead of the quick default")
    parser.add_argument("--datasets", default=None,
                        help="Catalog.parse() spec for a full-scale run, e.g. 'vote,mushroom'")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(args.datasets or (FULL_DATASETS_SPEC if args.full else None))
