"""
demos/deep_rule_networks.py
===========================

Deep rule networks -- `pyrulearn.models.DeepModel`s, formulas in negation
normal form learned as fully connected networks of alternating AND and OR
layers with 0/1 weights -- in two parts.

**Part 1, the SAT paper's artificial datasets.** Seip, Fürnkranz et al.
("A SAT-based approach to learning Boolean circuits in negation normal
form", the long version of their SYNASC 2025 paper) learn these networks
with a SAT solver and compare them with Beck & Fürnkranz's stochastic
hill climbing on 20 artificial datasets: the full truth tables of ten
Boolean features, labeled by random deep NNFs (the catalog's tag
``artificial``). Their architectures: NNF(1) ``[20]``, NNF(3) ``[32, 8,
2]``, NNF(5) ``[32, 16, 8, 4, 2]``. Here: `DeepSAT` (an exact fit, the
paper's method), `DeepLocalSearch` (Beck's hill climbing with our
consistent moves), `DeepSAT` with MaxSAT on NNF(1) (the smallest network
fitting the data, with a time limit -- the anytime linear search returns
the smallest found), and Pypper, against the paper's Table 1
(`PAPER_TABLE`).

**Part 2, the learners over time.** On a few datasets (`TRACE_DATASETS`),
the learners run with `trace=True`, so every new best network during a
fit is recorded; each is measured on the test fold, which gives test
accuracy (and network size) against seconds -- local search and the
genetic algorithm improving the training error, MaxSAT shrinking an exact
fit -- with Pypper as the reference. Local search also starts from
Pypper's rules (`RuleSetInit`).

Times are wall-clock, so they depend on the machine and its load.

Run: `python demos/deep_rule_networks.py` with no arguments is the
**quick** default (two artificial datasets and tic-tac-toe, few folds,
short time limits); `--full` runs everything (a few hours with
``--n-jobs 10``) and writes ``deep_rule_networks_report.md`` (plots in
``deep_rule_networks_plots/``). Needs the `experiments` and `sat` extras.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.learners.base import NativeRuleLearner
from pyrulearn.learners.deep import DeepGenetic, DeepLocalSearch, DeepSAT, RuleSetInit
from pyrulearn.learners.seco import Pypper

RANDOM_STATE = 0
FIT_TIMEOUT = 1800.0
NNF = {"NNF(1)": [20], "NNF(3)": [32, 8, 2], "NNF(5)": [32, 16, 8, 4, 2]}

# -- part 1 --------------------------------------------------------------------

SEIP_FOLDS, QUICK_SEIP_FOLDS = 10, 3
QUICK_ARTIFICIAL = ["artificial-nnf-5", "artificial-nnf-82"]
#: seconds the MaxSAT search on NNF(1) gets per fit
MAXSAT_SECONDS, QUICK_MAXSAT_SECONDS = 60, 5

#: the long paper's Table 1 (artificial datasets): its SAT and stochastic hill
#: climbing (SHS) accuracies per architecture, and Ripper's and CART's; None: time-out
PAPER_TABLE = pd.DataFrame(
    [[5, .9883, .9580, .9873, .9863, .9980, .9531, .9805, .9844],
     [16, .9765, .9639, .9804, .9707, .9902, .9629, .9766, .9551],
     [19, 1, 1, 1, .9902, 1, .9746, 1, 1],
     [24, .9883, .9053, .9883, .9043, .9932, .9160, .9463, .9404],
     [36, .9648, .8828, .9775, .9209, None, .9043, .8867, .9111],
     [44, .9825, .9629, .9785, .9551, .9902, .9326, .9482, .9697],
     [53, .9873, .9805, .9892, .9805, .9971, .9775, .9746, .9824],
     [57, .9961, .9824, .9932, .9736, .9961, .9639, .9951, .9902],
     [60, .9736, .9443, .9824, .9453, .9922, .9209, .958, .9883],
     [65, .9932, .9854, .9980, .9688, .9981, .9414, .9961, .9922],
     [68, .9805, .9248, .9717, .9443, .9912, .9619, .9688, .9355],
     [69, .9776, .9551, .9727, .9658, .9824, .9199, .9795, .9717],
     [70, .9815, .9014, .9766, .9062, .9912, .9229, .9111, .8984],
     [81, .9883, .9004, .9726, .9131, .9951, .8857, .9248, .9756],
     [82, .9990, .9941, .9990, .9980, .9990, .9717, 1, 1],
     [85, .9971, 1, .9990, .9980, 1, .9736, 1, 1],
     [89, .9902, .8926, .9922, .9434, .9980, .9629, .9502, .9395],
     [107, .9755, .8965, .9795, .8730, .9951, .8643, .9043, .9277],
     [112, .9824, .9346, .9834, .9248, .9863, .9189, .9082, .9561],
     [118, .9912, .9688, .9824, .9414, .9912, .9434, .9736, .9688]],
    columns=["seed", "SAT NNF(5)", "SHS NNF(5)", "SAT NNF(3)", "SHS NNF(3)", "SAT NNF(1)", "SHS NNF(1)",
             "Ripper", "CART"])

# -- part 2 --------------------------------------------------------------------

TRACE_DATASETS = ["tic-tac-toe", "monks-problems-2", "car-binary", "artificial-nnf-24", "kr-vs-kp"]
QUICK_TRACE_DATASETS = ["tic-tac-toe"]
TRACE_FOLDS, QUICK_TRACE_FOLDS = 5, 2
#: seconds the MaxSAT search gets per fit
TRACE_MAXSAT_SECONDS, QUICK_TRACE_MAXSAT_SECONDS = 300, 10
TRACE_ARCHITECTURES = ["NNF(1)", "NNF(3)"]
TRACE_LEARNERS = ["local search", "local search from Pypper", "genetic", "MaxSAT"]


def _variants(quick: bool) -> Dict[str, Tuple[str, Callable[[], NativeRuleLearner]]]:
    """name -> (a fixed description, which also keys the cache; the learner)."""
    seconds = QUICK_MAXSAT_SECONDS if quick else MAXSAT_SECONDS
    trace_seconds = QUICK_TRACE_MAXSAT_SECONDS if quick else TRACE_MAXSAT_SECONDS
    out: Dict[str, Tuple[str, Callable[[], NativeRuleLearner]]] = {
        "Pypper": ("Pypper()", lambda: Pypper()),
    }
    # part 1, as in the papers: the output learns the formula's class (True)
    for arch, layers in NNF.items():
        out[f"SAT {arch}"] = (f"DeepSAT(layers={layers}, target_class=True)",
                              lambda layers=layers: DeepSAT(layers=layers, target_class=True))
        out[f"local search {arch}"] = (f"DeepLocalSearch(layers={layers}, target_class=True)",
                                       lambda layers=layers: DeepLocalSearch(layers=layers, target_class=True))
        out[f"genetic {arch}"] = (f"DeepGenetic(layers={layers}, target_class=True)",
                                  lambda layers=layers: DeepGenetic(layers=layers, target_class=True))
    out["MaxSAT NNF(1)"] = (f"DeepSAT(layers=[20], minimize='weights', max_seconds={seconds}, target_class=True)",
                            lambda: DeepSAT(layers=[20], minimize="weights", max_seconds=seconds,
                                            target_class=True))
    for arch in TRACE_ARCHITECTURES:
        layers = NNF[arch]
        out[f"traced local search {arch}"] = (
            f"DeepLocalSearch(layers={layers}, trace=True)",
            lambda layers=layers: DeepLocalSearch(layers=layers, trace=True))
        out[f"traced local search from Pypper {arch}"] = (
            f"DeepLocalSearch(layers={layers}, init=RuleSetInit(Pypper()), trace=True)",
            lambda layers=layers: DeepLocalSearch(layers=layers, init=RuleSetInit(Pypper()), trace=True))
        out[f"traced genetic {arch}"] = (
            f"DeepGenetic(layers={layers}, trace=True)",
            lambda layers=layers: DeepGenetic(layers=layers, trace=True))
        out[f"traced MaxSAT {arch}"] = (
            f"DeepSAT(layers={layers}, minimize='weights', max_seconds={trace_seconds}, trace=True)",
            lambda layers=layers: DeepSAT(layers=layers, minimize="weights", max_seconds=trace_seconds,
                                          trace=True))
    return out


class Variant(NativeRuleLearner):
    """The learner `_variants(quick)[name]` describes, named `name` in the results."""

    def __init__(self, name: str, quick: bool):
        self.name = name
        self.quick = quick

    @property
    def display_name(self) -> str:
        return self.name

    def _provenance_params(self):
        return {"name": self.name, "learner": _variants(self.quick)[self.name][0]}

    def fit(self, data):
        return _variants(self.quick)[self.name][1]().fit(data)


def trace_curve(model, test_rep) -> dict:
    """`run_cv`'s extra measure: for a traced model, every point of its run
    -- seconds, effort, training error, the size it minimizes, the test
    accuracy and the backbone's edges of that point's network."""
    trace = getattr(model, "trace", None)
    if not trace:
        return {}
    points = []
    for i, p in enumerate(trace):
        net = model.trace_model(i)
        points.append([p.seconds, p.evaluations, p.train_error, p.size,
                       float((net.predict(test_rep) == test_rep.y).mean()), net.pruned().n_edges])
    return {"trace": points}


# -- running -------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "deep_rule_networks"
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")
CACHE_DIR = os.path.join(HERE, f"_{NAME}_cache")
N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


def part1(quick: bool) -> pd.DataFrame:
    cat = Catalog.default()
    datasets = cat.select(names=QUICK_ARTIFICIAL) if quick else cat.select(tags="artificial")
    names = (["Pypper"] + [f"SAT {a}" for a in NNF] + ["MaxSAT NNF(1)"] + [f"local search {a}" for a in NNF]
             + [f"genetic {a}" for a in NNF])
    return run_cv([Variant(n, quick) for n in names], datasets,
                  n_folds=QUICK_SEIP_FOLDS if quick else SEIP_FOLDS, fit_timeout=FIT_TIMEOUT,
                  random_state=RANDOM_STATE, cache_dir=CACHE_DIR, n_jobs=_n_jobs())


def part2(quick: bool) -> pd.DataFrame:
    datasets = Catalog.default().select(names=QUICK_TRACE_DATASETS if quick else TRACE_DATASETS)
    names = ["Pypper"] + [f"traced {learner} {arch}" for arch in TRACE_ARCHITECTURES for learner in TRACE_LEARNERS]
    return run_cv([Variant(n, quick) for n in names], datasets,
                  n_folds=QUICK_TRACE_FOLDS if quick else TRACE_FOLDS, fit_timeout=FIT_TIMEOUT,
                  random_state=RANDOM_STATE, cache_dir=CACHE_DIR, measure_fn=trace_curve, n_jobs=_n_jobs())


# -- the report ----------------------------------------------------------------

COLORS = {"local search": "#2a78d6", "local search from Pypper": "#1baf7a", "genetic": "#eb6834",
          "MaxSAT": "#6250d6"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _pct(v) -> str:
    return "time-out" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.2f}"


def _accuracy_by_dataset(res: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    """Mean accuracy per dataset and learner; NaN where a fold failed: the fit
    cap, or DeepSAT finding no network within its own time limit."""
    ok = res[res["error"].isna()]
    acc = ok.pivot_table(index="dataset", columns="learner", values="accuracy").reindex(columns=names)
    for ds, learner in _failures(res).index:
        if learner in acc.columns and ds in acc.index:
            acc.loc[ds, learner] = np.nan
    return acc


def _failures(res: pd.DataFrame) -> pd.Series:
    """The number of failed folds per (dataset, learner)."""
    return res[res["error"].notna()].groupby(["dataset", "learner"]).size()


def _failure_note(res: pd.DataFrame, rename=lambda n: n) -> List[str]:
    failed = _failures(res)
    if not len(failed):
        return []
    folds = res.groupby(["dataset", "learner"]).size()
    by_learner: Dict[str, List[str]] = {}
    for (ds, learner), k in failed.items():
        by_learner.setdefault(rename(learner), []).append(f"{ds} ({k}/{folds[(ds, learner)]})")
    return ["Time-outs (failed folds / folds; the fit cap of " + f"{FIT_TIMEOUT:.0f} s, or DeepSAT "
            "finding no network within its own limit): "
            + "; ".join(f"{n}: {', '.join(sorted(v))}" for n, v in sorted(by_learner.items())) + ".\n"]


def _part1_section(res: pd.DataFrame, quick: bool) -> List[str]:
    ours = (["Pypper"] + [f"SAT {a}" for a in NNF] + ["MaxSAT NNF(1)"] + [f"local search {a}" for a in NNF]
            + [f"genetic {a}" for a in NNF])
    acc = _accuracy_by_dataset(res, ours)
    times = res.pivot_table(index="dataset", columns="learner", values="fit_time").reindex(columns=ours)
    L = ["## Part 1: the SAT paper's artificial datasets\n",
         "Accuracy (%) per dataset, ours next to the paper's Table 1 (its SAT and stochastic hill "
         "climbing, SHS). Our folds are not the paper's (both stratified 10-fold), so single "
         "datasets can differ by chance; the means are the comparison.\n"]
    cols = []
    for arch in NNF:
        cols += [(f"SAT {arch}", "ours"), (f"SAT {arch}", "paper"), (f"local search {arch}", "ours"),
                 (f"SHS {arch}", "paper"), (f"genetic {arch}", "ours")]
    header = "| dataset | " + " | ".join(f"{name} ({who})" for name, who in cols) + " | MaxSAT NNF(1) | Pypper |"
    L += [header, "|---|" + "--:|" * (len(cols) + 2)]
    paper = PAPER_TABLE.set_index("seed")
    sums: Dict[str, List[float]] = {}
    for ds in acc.index:
        seed = int(ds.rsplit("-", 1)[1])
        cells = []
        for name, who in cols:
            v = acc.loc[ds, name] if who == "ours" else paper.loc[seed, name]
            v = None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
            cells.append(_pct(v))
            if v is not None:
                sums.setdefault(f"{name}|{who}", []).append(v)
        for name in ("MaxSAT NNF(1)", "Pypper"):
            v = acc.loc[ds, name]
            cells.append(_pct(v))
            if not np.isnan(v):
                sums.setdefault(f"{name}|ours", []).append(v)
        L.append(f"| {ds} | " + " | ".join(cells) + " |")
    mean = [_pct(np.mean(sums.get(f"{n}|{w}", [np.nan]))) for n, w in cols]
    mean += [_pct(np.mean(sums.get(f"{n}|ours", [np.nan]))) for n in ("MaxSAT NNF(1)", "Pypper")]
    L += ["| **mean** | " + " | ".join(mean) + " |", ""]
    if quick:
        L.append("(Quick run: the paper's means are over all 20 datasets, ours over the ones run.)\n")
    L += ["Mean fit time (s): " + ", ".join(f"{n} {times[n].mean():.1f}" for n in ours if n in times) + ".\n"]
    L += _failure_note(res)
    return L


def _curve(points: List[list], grid: np.ndarray, column: int) -> np.ndarray:
    """The best-so-far network's value at each time of `grid` (NaN before
    the first point)."""
    t = np.array([p[0] for p in points])
    v = np.array([np.nan if p[column] is None else p[column] for p in points], dtype=float)
    idx = np.searchsorted(t, grid, side="right") - 1
    return np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)


def _plot_traces(res: pd.DataFrame, quick: bool) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    datasets = list(dict.fromkeys(res["dataset"]))
    lines = []
    for ds in datasets:
        sub = res[res["dataset"] == ds]
        pypper = sub[(sub["learner"] == "Pypper") & sub["error"].isna()]["accuracy"].mean()
        fig, axes = plt.subplots(2, len(TRACE_ARCHITECTURES), figsize=(5.5 * len(TRACE_ARCHITECTURES), 7.2),
                                 sharex=True, facecolor=SURF, squeeze=False)
        t_max = max((pt[0] for tr in sub["trace"].dropna() for pt in tr), default=1.0)
        grid = np.geomspace(0.01, max(t_max, 0.02) * 1.05, 300)
        for c, arch in enumerate(TRACE_ARCHITECTURES):
            for r, (column, label) in enumerate(((4, "test accuracy"), (5, "edges (backbone)"))):
                ax = axes[r][c]
                ax.set_facecolor(SURF)
                for learner, color in COLORS.items():
                    rows = sub[sub["learner"] == f"traced {learner} {arch}"]
                    # a failed fold counts as one without a network at any time
                    curves = [_curve(tr, grid, column) if isinstance(tr, list) and tr and pd.isna(err)
                              else np.full(len(grid), np.nan)
                              for tr, err in zip(rows["trace"], rows["error"])]
                    if not curves or all(np.isnan(c).all() for c in curves):
                        continue
                    stack = np.vstack(curves)
                    enough = np.sum(~np.isnan(stack), axis=0) >= max(1, len(curves) // 2 + 1)
                    with np.errstate(all="ignore"):
                        mean = np.where(enough, np.nanmean(stack, axis=0), np.nan)
                    ax.step(grid, mean, where="post", color=color, lw=2, label=learner)
                if r == 0 and not np.isnan(pypper):
                    ax.axhline(pypper, color=INK2, lw=1)
                    ax.annotate(f"Pypper {pypper:.3f}", (grid[0], pypper), xytext=(2, 4),
                                textcoords="offset points", color=INK2, fontsize=8.5)
                ax.set_xscale("log")
                if r == 1:
                    ax.set_yscale("log")
                    ax.set_xlabel("seconds (log scale)", color=INK2, fontsize=9)
                ax.set_title(f"{arch}: {label}", color=INK, fontsize=10, loc="left")
                ax.grid(True, axis="y", color=GRID, lw=1)
                ax.set_axisbelow(True)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
                for s_ in ("left", "bottom"):
                    ax.spines[s_].set_color(GRID)
                ax.tick_params(colors=INK2, labelsize=8.5)
        axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower right")
        fig.suptitle(f"{ds}: the best network so far, mean over folds", x=0.01, ha="left", color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = os.path.join(PLOTS_DIR, f"{NAME}{'_quick' if quick else ''}_{ds}.png")
        fig.savefig(path, dpi=130, facecolor=SURF)
        plt.close(fig)
        lines.append(f"![{ds}]({os.path.basename(PLOTS_DIR)}/{os.path.basename(path)})\n")
    return lines


def _part2_section(res: pd.DataFrame, quick: bool) -> List[str]:
    names = ["Pypper"] + [f"traced {learner} {arch}" for arch in TRACE_ARCHITECTURES for learner in TRACE_LEARNERS]
    acc = _accuracy_by_dataset(res, names)
    times = res.pivot_table(index="dataset", columns="learner", values="fit_time").reindex(columns=names)
    L = ["## Part 2: the learners over time\n",
         "Each curve is the test accuracy (top) and the size (bottom: edges of the network's "
         "backbone, `DeepModel.pruned`) of the best network found so far, against the seconds "
         "of the fit, averaged over the folds where a network exists by then (shown once that is at least "
         "half of all folds; a failed fold never has one). Local search and the genetic algorithm "
         "improve the training error; MaxSAT "
         "starts from an exact fit and makes it smaller.\n"]
    L += _plot_traces(res, quick)
    L += ["Final test accuracy (%) and mean fit time (s):\n",
          "| dataset | " + " | ".join(n.replace("traced ", "") for n in names) + " |",
          "|---|" + "--:|" * len(names)]
    for ds in acc.index:
        L.append(f"| {ds} | " + " | ".join(f"{_pct(acc.loc[ds, n])} ({times.loc[ds, n]:.0f} s)" for n in names)
                 + " |")
    return L + [""] + _failure_note(res, lambda n: n.replace("traced ", ""))


DESCRIPTION = f"""
Part 1: the 20 artificial datasets of Beck & Fürnkranz (2021), as used by Seip, Fürnkranz et
al. (catalog tag `artificial`; all 1,024 combinations of ten Boolean features, labeled by a
random deep NNF), stratified {SEIP_FOLDS}-fold cross-validation. The Boolean columns become
`x` and `not x`, the features and negations the SAT paper's networks take. Architectures as
in the paper: NNF(1) `{NNF['NNF(1)']}`, NNF(3) `{NNF['NNF(3)']}`, NNF(5) `{NNF['NNF(5)']}`,
first layer AND, output OR, the output learning the formula's class (`target_class=True`,
as the papers' networks do). Learners: `DeepSAT` (plain SAT, an exact fit; Glucose 4),
`DeepLocalSearch` (Beck's stochastic hill climbing, consistent moves, random start),
`DeepGenetic` (the memetic algorithm, node-wise crossover, random start),
`DeepSAT(minimize="weights")` on NNF(1) with {MAXSAT_SECONDS} s (the smallest exact fit found
by then), Pypper.

Part 2: {", ".join(TRACE_DATASETS)}, stratified {TRACE_FOLDS}-fold. NNF(1) and NNF(3);
local search (random start, and started from Pypper's rules, `RuleSetInit(Pypper())`),
the genetic algorithm (`DeepGenetic`, node-wise crossover), MaxSAT (`DeepSAT(minimize=
"weights")`, {TRACE_MAXSAT_SECONDS} s), all with `trace=True`; Pypper as the reference.

Each fit is capped at {FIT_TIMEOUT:.0f} s; a time-out drops the learner's whole
cross-validation on that dataset. Times are wall-clock.
"""


def write_report(res1: pd.DataFrame, res2: pd.DataFrame, quick: bool, path: str) -> None:
    L = [f"# Deep rule networks{' (quick run)' if quick else ''}\n",
         render_setup_section(DESCRIPTION)]
    L += _part1_section(res1, quick)
    L += _part2_section(res2, quick)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Report -> {path}")


def main(quick: bool = True) -> None:
    res1 = part1(quick)
    res2 = part2(quick)
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(res1, res2, quick, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="every dataset and fold, the full time limits")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(quick=not args.full)
