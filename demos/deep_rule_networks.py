"""
demos/deep_rule_networks.py
===========================

Deep rule networks -- `pyrulearn.models.DeepModel`s, formulas in negation
normal form learned as fully connected networks of alternating AND and OR
layers with 0/1 weights -- in four parts.

**Part 1, the SAT paper's artificial datasets.** Seip, Fürnkranz et al.
("A SAT-based approach to learning Boolean circuits in negation normal
form", the long version of their SYNASC 2025 paper) learn these networks
with a SAT solver and compare them with Beck & Fürnkranz's stochastic
hill climbing on 20 artificial datasets: the full truth tables of ten
Boolean features, labeled by random deep NNFs (the catalog's tag
``artificial``). Their architectures: NNF(1) ``[20]``, NNF(3) ``[32, 8,
2]``, NNF(5) ``[32, 16, 8, 4, 2]``. Here: `DeepSAT` (an exact fit, the
paper's method), `DeepLocalSearch` (Beck's hill climbing with our
consistent moves), `DeepGenetic`, `DeepNeuroFuzzy`, `DeepSAT` with MaxSAT
on NNF(1) (the smallest network fitting the data, with a time limit), and
Pypper, against the paper's Table 1 (`PAPER_TABLE`).

**Part 2, the learners over time.** On a few datasets (`TRACE_DATASETS`),
the four learners run with `trace=True` and the same regularized
objective (`size_penalty=LAMBDA`: errors plus lambda per edge), each from
a random start and from Pypper's rules (`RuleSetInit`); every new best
network during a fit is measured on the test fold, which gives test
accuracy (and network size) against seconds, with Pypper as the
reference.

**Part 3, KRK illegality.** 1000 positions, the coordinates only
(`pyrulearn.datasets.KRKIllegal`): "same file", "adjacent" and so on have
to appear as hidden nodes, so a wide network ``[128, 16, 4]`` (the rules
need about ``[76, 5, 3]``) and a time budget per fit, to see how far the
anytime learners get -- against Pypper on the coordinates and Pypper with
the relations given (the upper bound).

**Part 4, FOIL as the start.** Parts 2 and 3 again, the learners started
from FOIL's rules (`PFoil`, for the class the network learns, at most as
many rules as the first layer has nodes: `FoilInit`) instead of
Pypper's: FOIL's rules are more and longer, a richer start than Pypper's
few general ones -- and from both (`PypperFoilInit`: Pypper's rules, then
FOIL's new ones). Compared with the runs from Pypper's rules.

Times are wall-clock, so they depend on the machine and its load.

Run: `python demos/deep_rule_networks.py` with no arguments is the
**quick** default (two artificial datasets and tic-tac-toe, few folds,
short time limits); `--full` runs everything (a few hours with
``--n-jobs 10``) and writes ``deep_rule_networks_report.md`` (plots in
``deep_rule_networks_plots/``). Needs the `experiments` and `sat` extras.
``--report-only`` fits nothing: it writes the report and plots from what
the cache holds (with ``--full``, of the full run).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from pyrulearn.datasets import KRKIllegal, Parity
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.learners.base import NativeRuleLearner
from pyrulearn.learners.deep import (DeepGenetic, DeepInit, DeepLocalSearch, DeepNeuroFuzzy, DeepSAT,
                                     RuleSetInit)
from pyrulearn.learners.seco import PFoil, Pypper

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

#: the size penalty of part 2 and 3: an edge has to be worth one misclassified example
#: (a pilot on five datasets, k = 1/lambda in 1, 3, 10, 30: 1 was best on the noisy ones)
LAMBDA = 1.0
TRACE_DATASETS = ["tic-tac-toe", "kr-vs-kp", "vote", "mushroom", "breast-cancer", "SPECT", Parity(4, 6)]
QUICK_TRACE_DATASETS = ["tic-tac-toe"]
#: datasets whose MaxSAT formula would be too large (millions of clauses in Python lists)
NO_MAXSAT = ["mushroom"]
TRACE_FOLDS, QUICK_TRACE_FOLDS = 5, 2
#: seconds the MaxSAT search gets per fit
TRACE_MAXSAT_SECONDS, QUICK_TRACE_MAXSAT_SECONDS = 300, 10
#: the other learners' time budget per fit in part 2 -- below the fit cap, so a long final
#: climb (mushroom, kr-vs-kp) returns its best network instead of losing the fit
TRACE_SECONDS, QUICK_TRACE_SECONDS = 900, 30
TRACE_ARCHITECTURES = ["NNF(1)", "NNF(3)"]
ALGORITHMS = ["local search", "genetic", "MaxSAT", "neuro-fuzzy"]
TRACE_LEARNERS = [f"{a}{s}" for a in ALGORITHMS for s in ("", " from Pypper")]

# -- part 3 --------------------------------------------------------------------

KRK, QUICK_KRK = KRKIllegal(1000), KRKIllegal(300)
KRK_ARCHITECTURE = [128, 16, 4]
#: seconds per fit; the learners get epochs and generations enough for the time to end them
KRK_SECONDS, QUICK_KRK_SECONDS = 1500, 20
KRK_ALGORITHMS = ["local search", "genetic", "neuro-fuzzy"]
KRK_LEARNERS = [f"{a}{s}" for a in KRK_ALGORITHMS for s in ("", " from Pypper")]

# -- part 4 --------------------------------------------------------------------

#: the starts compared in part 4 (the line style of their curves)
STARTS = {"Pypper": "--", "FOIL": ":", "Pypper+FOIL": "-."}


class FoilInit(DeepInit):
    """FOIL's rules as the start: `PFoil` for the class the network's
    (single) output learns, at most as many rules as the first layer has
    nodes (FOIL learns more rules than Pypper; the first ones cover
    most), put in by `RuleSetInit`."""

    def weights(self, data, labels, shapes, rng, first="and"):
        if len(labels) != 1:
            raise ValueError("FoilInit is for one output (two classes)")
        rules = PFoil(target_class=labels[0], max_rules=shapes[0][0]).fit(data)
        return RuleSetInit(rules).weights(data, labels, shapes, rng, first)


class PypperFoilInit(DeepInit):
    """Both rule sets as the start: Pypper's rules for the class the
    network's (single) output learns -- few, general --, then FOIL's
    (`PFoil`) that are not among them -- more, specific --, at most as
    many as the first layer has nodes, all put in by `RuleSetInit`: the
    union of the two DNFs."""

    def weights(self, data, labels, shapes, rng, first="and"):
        if len(labels) != 1:
            raise ValueError("PypperFoilInit is for one output (two classes)")
        rules, seen = [], set()
        for model in (Pypper().fit(data), PFoil(target_class=labels[0]).fit(data)):
            for r in model.rules:
                key = frozenset(r.dataspec.feature_name(c.feature) for c in r.conditions)
                if r.target == labels[0] and key not in seen:
                    seen.add(key)
                    rules.append(r)
        return RuleSetInit(SimpleNamespace(rules=rules[:shapes[0][0]])).weights(data, labels, shapes, rng, first)


class FoilRules(NativeRuleLearner):
    """The start on its own, for reference: `PFoil` for the less frequent
    class (the deep learners' output there), at most `max_rules` rules."""

    def __init__(self, max_rules: int):
        self.max_rules = max_rules

    def _provenance_params(self):
        return {"max_rules": self.max_rules}

    def fit(self, data):
        classes, counts = np.unique(np.asarray(data.y), return_counts=True)
        target = classes[int(np.argmin(counts))]
        return PFoil(target_class=target.item() if isinstance(target, np.generic) else target,
                     max_rules=self.max_rules).fit(data)


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
        out[f"neuro-fuzzy {arch}"] = (f"DeepNeuroFuzzy(layers={layers}, target_class=True)",
                                      lambda layers=layers: DeepNeuroFuzzy(layers=layers, target_class=True))
    out["MaxSAT NNF(1)"] = (f"DeepSAT(layers=[20], minimize='weights', max_seconds={seconds}, target_class=True)",
                            lambda: DeepSAT(layers=[20], minimize="weights", max_seconds=seconds,
                                            target_class=True))
    # parts 2 and 3: one objective (errors + LAMBDA per edge), a random start or Pypper's rules
    krk_seconds = QUICK_KRK_SECONDS if quick else KRK_SECONDS

    budget = QUICK_TRACE_SECONDS if quick else TRACE_SECONDS

    def make(algorithm, layers, start, seconds, unbounded=False):
        kw = dict(layers=layers, size_penalty=LAMBDA, trace=True)
        if start == "Pypper":
            kw["init"] = RuleSetInit(Pypper())
        elif start == "FOIL":
            kw["init"] = FoilInit()
        elif start == "Pypper+FOIL":
            kw["init"] = PypperFoilInit()
        if algorithm == "MaxSAT":
            return DeepSAT(minimize="weights", max_seconds=trace_seconds, **kw)
        kw["max_seconds"] = seconds  # unbounded: stopped by the time, not by the epochs or generations
        if algorithm == "local search":
            return DeepLocalSearch(n_epochs=10 ** 6 if unbounded else 5, **kw)
        if algorithm == "genetic":
            return DeepGenetic(generations=10 ** 6 if unbounded else 20, **kw)
        return DeepNeuroFuzzy(epochs=10 ** 6 if unbounded else 100, **kw)

    def describe(algorithm, layers, start, seconds, unbounded=False):
        start = {"": "", "Pypper": ", init=RuleSetInit(Pypper())",
                 "FOIL": ", init=FoilInit() (PFoil, the output's class, at most a rule per first-layer node)",
                 "Pypper+FOIL": ", init=PypperFoilInit() (Pypper's rules, then PFoil's new ones, at most a rule "
                                "per first-layer node)"}[start]
        if algorithm == "MaxSAT":
            return (f"DeepSAT(layers={layers}, minimize='weights', size_penalty={LAMBDA}, "
                    f"max_seconds={trace_seconds}{start}, trace=True)")
        name = {"local search": "DeepLocalSearch", "genetic": "DeepGenetic", "neuro-fuzzy": "DeepNeuroFuzzy"}
        extra = f", max_seconds={seconds}" + (", unbounded epochs" if unbounded else "")
        return f"{name[algorithm]}(layers={layers}, size_penalty={LAMBDA}{start}{extra}, trace=True)"

    for arch in TRACE_ARCHITECTURES:
        for algorithm in ALGORITHMS:
            for start in ("", *STARTS):
                name = f"traced {algorithm}{f' from {start}' if start else ''} {arch}"
                out[name] = (describe(algorithm, NNF[arch], start, budget),
                             lambda a=algorithm, l=NNF[arch], s=start: make(a, l, s, budget))
        out[f"FOIL start {arch}"] = (f"PFoil(target_class=minority, max_rules={NNF[arch][0]})",
                                     lambda n=NNF[arch][0]: FoilRules(n))
    for algorithm in KRK_ALGORITHMS:
        for start in ("", *STARTS):
            name = f"krk {algorithm}{f' from {start}' if start else ''}"
            out[name] = (describe(algorithm, KRK_ARCHITECTURE, start, krk_seconds, unbounded=True),
                         lambda a=algorithm, s=start: make(a, KRK_ARCHITECTURE, s, krk_seconds, unbounded=True))
    out["krk FOIL start"] = (f"PFoil(target_class=minority, max_rules={KRK_ARCHITECTURE[0]})",
                             lambda: FoilRules(KRK_ARCHITECTURE[0]))
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
#: `--report-only`: the tables from the cache, nothing fitted
CACHED_ONLY = False
#: (dataset, learner) -> (failed folds, folds run): what really ran, for the report -- with
#: `stop_on_timeout` a timed-out pair's finished folds also read as time-outs
RAN: Dict[Tuple[str, str], Tuple[int, int]] = {}


def _run_cv(learners, datasets, **kw) -> pd.DataFrame:
    """`run_cv` (only from the cache with `CACHED_ONLY`), noting in `RAN`
    how many folds of each (dataset, learner) ran and how many of them
    failed."""
    res = run_cv(learners, datasets, cached_only=CACHED_ONLY, **kw)
    raw = run_cv(learners, datasets, cached_only=True, stop_on_timeout=False,
                 **{k: v for k, v in kw.items() if k not in ("n_jobs", "measure_fn")})
    if len(raw):
        for (ds, learner), rows in raw.groupby(["dataset", "learner"]):
            RAN[(ds, learner)] = (int(rows["error"].notna().sum()), len(rows))
    return res


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


def part1(quick: bool) -> pd.DataFrame:
    cat = Catalog.default()
    datasets = cat.select(names=QUICK_ARTIFICIAL) if quick else cat.select(tags="artificial")
    names = (["Pypper"] + [f"SAT {a}" for a in NNF] + ["MaxSAT NNF(1)"] + [f"local search {a}" for a in NNF]
             + [f"genetic {a}" for a in NNF] + [f"neuro-fuzzy {a}" for a in NNF])
    return _run_cv([Variant(n, quick) for n in names], datasets,
                  n_folds=QUICK_SEIP_FOLDS if quick else SEIP_FOLDS, fit_timeout=FIT_TIMEOUT,
                  random_state=RANDOM_STATE, cache_dir=CACHE_DIR, n_jobs=_n_jobs())


def _datasets(names) -> list:
    """Catalog entries for names, generated datasets as they are."""
    catalog = Catalog.default()
    return [catalog.select(names=[d])[0] if isinstance(d, str) else d for d in names]


def part2(quick: bool) -> pd.DataFrame:
    datasets = _datasets(QUICK_TRACE_DATASETS if quick else TRACE_DATASETS)
    names = ["Pypper"] + [f"traced {learner} {arch}" for arch in TRACE_ARCHITECTURES for learner in TRACE_LEARNERS]
    without = [n for n in names if "MaxSAT" not in n]
    common = dict(n_folds=QUICK_TRACE_FOLDS if quick else TRACE_FOLDS, fit_timeout=FIT_TIMEOUT,
                  random_state=RANDOM_STATE, cache_dir=CACHE_DIR, measure_fn=trace_curve, n_jobs=_n_jobs())
    full = [d for d in datasets if d.name not in NO_MAXSAT]
    small = [d for d in datasets if d.name in NO_MAXSAT]
    parts = [_run_cv([Variant(n, quick) for n in names], full, **common)] if full else []
    if small:
        parts.append(_run_cv([Variant(n, quick) for n in without], small, **common))
    return pd.concat(parts, ignore_index=True)


def part3(quick: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The KRK runs, and Pypper on the positions with the relations given."""
    krk = QUICK_KRK if quick else KRK
    names = ["Pypper"] + [f"krk {learner}" for learner in KRK_LEARNERS]
    common = dict(n_folds=QUICK_TRACE_FOLDS if quick else TRACE_FOLDS, fit_timeout=FIT_TIMEOUT,
                  random_state=RANDOM_STATE, cache_dir=CACHE_DIR, n_jobs=_n_jobs())
    res = _run_cv([Variant(n, quick) for n in names], [krk], measure_fn=trace_curve, **common)
    relational = KRKIllegal(krk.n, krk.random_state, relations=True)
    upper = _run_cv([Variant("Pypper", quick)], [relational], **common)
    return res, upper


def part4(quick: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parts 2 and 3 started from FOIL's rules (and the FOIL starts on their own)."""
    datasets = _datasets(QUICK_TRACE_DATASETS if quick else TRACE_DATASETS)
    names = [f"FOIL start {arch}" for arch in TRACE_ARCHITECTURES] + [
        f"traced {a} from {start} {arch}" for arch in TRACE_ARCHITECTURES for a in ALGORITHMS
        for start in ("FOIL", "Pypper+FOIL")]
    without = [n for n in names if "MaxSAT" not in n]
    common = dict(n_folds=QUICK_TRACE_FOLDS if quick else TRACE_FOLDS, fit_timeout=FIT_TIMEOUT,
                  random_state=RANDOM_STATE, cache_dir=CACHE_DIR, measure_fn=trace_curve, n_jobs=_n_jobs())
    full = [d for d in datasets if d.name not in NO_MAXSAT]
    small = [d for d in datasets if d.name in NO_MAXSAT]
    parts = [_run_cv([Variant(n, quick) for n in names], full, **common)] if full else []
    if small:
        parts.append(_run_cv([Variant(n, quick) for n in without], small, **common))
    krk = QUICK_KRK if quick else KRK
    krk_names = ["krk FOIL start"] + [f"krk {a} from {start}" for a in KRK_ALGORITHMS
                                      for start in ("FOIL", "Pypper+FOIL")]
    return pd.concat(parts, ignore_index=True), _run_cv([Variant(n, quick) for n in krk_names], [krk], **common)


# -- the report ----------------------------------------------------------------

COLORS = {"local search": "#2a78d6", "genetic": "#eb6834", "MaxSAT": "#6250d6", "neuro-fuzzy": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _pct(v, dataset: str = None, learner: str = None) -> str:
    """An accuracy in %; a failure as ``time-out``, or ``time-out (k/n)``
    when only k of the n folds that ran failed (the others finished: with
    parallel fits, those already running when a fold timed out)."""
    if not (v is None or (isinstance(v, float) and np.isnan(v))):
        return f"{100 * v:.2f}"
    failed, ran = RAN.get((dataset, learner), (0, 0))
    return f"time-out ({failed}/{ran})" if 0 < failed < ran else "time-out"


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
        bad, ran = RAN.get((ds, learner), (k, folds[(ds, learner)]))
        skipped = folds[(ds, learner)] - ran
        by_learner.setdefault(rename(learner), []).append(
            f"{ds} ({bad}/{ran}" + (f", {skipped} skipped" if skipped else "") + ")")
    return ["Time-outs (failed folds / folds run; the fit cap of " + f"{FIT_TIMEOUT:.0f} s, or DeepSAT "
            "finding no network within its own limit; after a time-out the folds not yet started are "
            "skipped): " + "; ".join(f"{n}: {', '.join(sorted(v))}" for n, v in sorted(by_learner.items()))
            + ".\n"]


def _part1_section(res: pd.DataFrame, quick: bool) -> List[str]:
    ours = (["Pypper"] + [f"SAT {a}" for a in NNF] + ["MaxSAT NNF(1)"] + [f"local search {a}" for a in NNF]
            + [f"genetic {a}" for a in NNF] + [f"neuro-fuzzy {a}" for a in NNF])
    acc = _accuracy_by_dataset(res, ours)
    times = res.pivot_table(index="dataset", columns="learner", values="fit_time").reindex(columns=ours)
    L = ["## Part 1: the SAT paper's artificial datasets\n",
         "Accuracy (%) per dataset, ours next to the paper's Table 1 (its SAT and stochastic hill "
         "climbing, SHS). Our folds are not the paper's (both stratified 10-fold), so single "
         "datasets can differ by chance; the means are the comparison.\n"]
    cols = []
    for arch in NNF:
        cols += [(f"SAT {arch}", "ours"), (f"SAT {arch}", "paper"), (f"local search {arch}", "ours"),
                 (f"SHS {arch}", "paper"), (f"genetic {arch}", "ours"), (f"neuro-fuzzy {arch}", "ours")]
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
            cells.append(_pct(v, ds, name) if who == "ours" else _pct(v))
            if v is not None:
                sums.setdefault(f"{name}|{who}", []).append(v)
        for name in ("MaxSAT NNF(1)", "Pypper"):
            v = acc.loc[ds, name]
            cells.append(_pct(v, ds, name))
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


def _plot_traces(res: pd.DataFrame, quick: bool, panels: List[Tuple[str, str]], algorithms: List[str],
                 references: Dict[str, List[Tuple[str, float]]], prefix: str = "",
                 starts: Tuple[Tuple[str, str], ...] = (("", "-"), ("Pypper", "--")), tag: str = "") -> List[str]:
    """One figure per dataset: a column per (title, learner-name suffix) in
    `panels`, test accuracy above and backbone edges below; a color per
    algorithm, a line style per start in `starts` ((start, style); by
    default solid from a random start, dashed from Pypper); horizontal
    `references` (label, accuracy) per dataset. `tag` goes into the file
    names."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, SymmetricalLogLocator

    os.makedirs(PLOTS_DIR, exist_ok=True)
    datasets = list(dict.fromkeys(res["dataset"]))
    lines = []
    for ds in datasets:
        sub = res[res["dataset"] == ds]
        fig, axes = plt.subplots(2, len(panels), figsize=(5.5 * len(panels) + 1.5, 7.2), sharex=True,
                                 facecolor=SURF, squeeze=False)
        t_max = max((pt[0] for tr in sub["trace"].dropna() for pt in tr), default=1.0) if "trace" in sub else 1.0
        grid = np.geomspace(0.01, max(t_max, 0.02) * 1.05, 300)
        for c, (title, suffix) in enumerate(panels):
            for r, (column, label) in enumerate(((4, "test accuracy"), (5, "edges (backbone)"))):
                ax = axes[r][c]
                ax.set_facecolor(SURF)
                for algorithm in algorithms:
                    for start, style in starts:
                        name = f"{prefix}{algorithm}{f' from {start}' if start else ''}{suffix}"
                        rows = sub[sub["learner"] == name]
                        # a failed fold counts as one without a network at any time
                        curves = [_curve(tr, grid, column) if isinstance(tr, list) and tr and pd.isna(err)
                                  else np.full(len(grid), np.nan)
                                  for tr, err in zip(rows["trace"], rows["error"])] if len(rows) else []
                        if not curves or all(np.isnan(cv).all() for cv in curves):
                            continue
                        stack = np.vstack(curves)
                        enough = np.sum(~np.isnan(stack), axis=0) >= max(1, len(curves) // 2 + 1)
                        with np.errstate(all="ignore"):
                            mean = np.where(enough, np.nanmean(stack, axis=0), np.nan)
                        ax.step(grid, mean, where="post", color=COLORS[algorithm], lw=2, ls=style,
                                label=f"{algorithm}{f' from {start}' if start else ''}")
                if r == 0:
                    for k, (ref_label, value) in enumerate(references.get(ds, [])):
                        if value is None or np.isnan(value):
                            continue
                        ax.axhline(value, color=INK2, lw=1, ls=("-", ":")[k % 2])
                        ax.annotate(f"{ref_label} {value:.3f}", (0.0, value), xytext=(4, 4),
                                    xycoords=("axes fraction", "data"), textcoords="offset points",
                                    color=INK2, fontsize=8.5)
                ax.set_xscale("log")
                if r == 1:
                    ax.set_yscale("symlog", linthresh=1)  # an empty network (0 edges) stays visible
                    ax.yaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=1, subs=[1, 2, 3, 5, 7]))
                    ax.yaxis.set_major_formatter(ScalarFormatter())
                    ax.set_xlabel("seconds (log scale)", color=INK2, fontsize=9)
                ax.set_title(f"{title}: {label}", color=INK, fontsize=10, loc="left")
                ax.grid(True, axis="y", color=GRID, lw=1)
                ax.set_axisbelow(True)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
                for s_ in ("left", "bottom"):
                    ax.spines[s_].set_color(GRID)
                ax.tick_params(colors=INK2, labelsize=8.5)
        handles, labels = axes[0][0].get_legend_handles_labels()
        for row in axes:  # the legend from whichever panel has the most lines
            for ax in row:
                h, lab = ax.get_legend_handles_labels()
                if len(h) > len(handles):
                    handles, labels = h, lab
        fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="center right")
        styles = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dash-dotted"}
        legend = ", ".join(f"{styles[style]}: {f'from {start}' if start else 'random start'}" for start, style in starts)
        fig.suptitle(f"{ds}: the best network so far, mean over folds{chr(10) if len(panels) == 1 else ' '}({legend})",
                     x=0.01, ha="left",
                     color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1 - 1.5 / (5.5 * len(panels) + 1.5), 0.96))
        path = os.path.join(PLOTS_DIR, f"{NAME}{'_quick' if quick else ''}{tag}_{ds}.png")
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
         f"half of all folds; a failed fold never has one). All four learners minimize the training errors "
         f"plus {LAMBDA:g} per edge; solid from a random start, dashed from Pypper's rules.\n"]
    refs = {ds: [("Pypper", res[(res["dataset"] == ds) & (res["learner"] == "Pypper")]["accuracy"].mean())]
            for ds in res["dataset"].unique()}
    L += _plot_traces(res, quick, [(arch, f" {arch}") for arch in TRACE_ARCHITECTURES], ALGORITHMS, refs,
                      prefix="traced ")
    L += ["Final test accuracy (%) and mean fit time (s):\n",
          "| dataset | " + " | ".join(n.replace("traced ", "") for n in names) + " |",
          "|---|" + "--:|" * len(names)]
    for ds in acc.index:
        L.append(f"| {ds} | " + " | ".join(f"{_pct(acc.loc[ds, n], ds, n)} ({times.loc[ds, n]:.0f} s)" for n in names)
                 + " |")
    if NO_MAXSAT and not quick:
        L.append(f"MaxSAT is not run on {', '.join(NO_MAXSAT)}: its formula would have tens of millions of "
                 "clauses.\n")
    return L + [""] + _failure_note(res, lambda n: n.replace("traced ", ""))


def _part3_section(res: pd.DataFrame, upper: pd.DataFrame, quick: bool) -> List[str]:
    krk = (QUICK_KRK if quick else KRK).name
    names = ["Pypper"] + [f"krk {learner}" for learner in KRK_LEARNERS]
    acc = _accuracy_by_dataset(res, names)
    times = res.pivot_table(index="dataset", columns="learner", values="fit_time").reindex(columns=names)
    edges = {}
    for n in names[1:]:
        rows = res[(res["learner"] == n) & res["error"].isna()]
        last = [tr[-1][5] for tr in rows.get("trace", pd.Series(dtype=object)) if isinstance(tr, list) and tr]
        edges[n] = np.mean(last) if last else np.nan
    pypper = acc.loc[krk, "Pypper"] if krk in acc.index else np.nan
    bound = upper[upper["error"].isna()]["accuracy"].mean()
    L = ["## Part 3: KRK illegality\n",
         f"{krk}: the coordinates only, stratified {QUICK_TRACE_FOLDS if quick else TRACE_FOLDS}-fold; network "
         f"`{KRK_ARCHITECTURE}`, {QUICK_KRK_SECONDS if quick else KRK_SECONDS} s per fit, errors plus "
         f"{LAMBDA:g} per edge. Reference lines: Pypper on the coordinates, and Pypper with the 24 relations "
         "given (`relations=True`), what the background knowledge allows.\n"]
    L += _plot_traces(res, quick, [(f"{KRK_ARCHITECTURE}", "")], KRK_ALGORITHMS,
                      {krk: [("Pypper", pypper), ("Pypper with relations", bound)]}, prefix="krk ")
    L += ["Final test accuracy (%), backbone edges and mean fit time (s):\n",
          "| learner | accuracy | edges | seconds |", "|---|--:|--:|--:|"]
    for n in names:
        e = "" if n == "Pypper" or np.isnan(edges.get(n, np.nan)) else f"{edges[n]:.0f}"
        L.append(f"| {n.replace('krk ', '')} | {_pct(acc.loc[krk, n], krk, n) if krk in acc.index else '-'} | {e} | "
                 f"{times.loc[krk, n]:.0f} |" if krk in times.index else f"| {n} | - | | |")
    L.append(f"| Pypper with relations | {_pct(bound)} | | |")
    return L + [""] + _failure_note(res, lambda n: n.replace("krk ", ""))


def _part4_section(res2: pd.DataFrame, res3: pd.DataFrame, res4: pd.DataFrame, krk4: pd.DataFrame,
                   quick: bool) -> List[str]:
    """The starts compared: final test accuracy and edges per dataset,
    learner and architecture, from Pypper's rules, FOIL's and both, and
    the curves of all three."""
    title = "## Part 4: FOIL and Pypper+FOIL as the start\n"
    if res4.empty or "dataset" not in res4:
        return [title, "(No results yet.)\n"]
    both = pd.concat([res2, res4], ignore_index=True)
    krk = pd.concat([res3, krk4], ignore_index=True)
    starts = list(STARTS)

    def rows(res, ds, learner):
        return res[(res["dataset"] == ds) & (res["learner"] == learner)]

    def cell(res, ds, learner):
        r = rows(res, ds, learner)
        if not len(r):
            return "not run"
        ok = r[r["error"].isna()]
        if not len(ok):
            return _pct(np.nan, ds, learner)
        last = [tr[-1][5] for tr in ok.get("trace", pd.Series(dtype=object)) if isinstance(tr, list) and tr]
        return f"{100 * ok['accuracy'].mean():.2f}" + (f" ({np.mean(last):.0f})" if last else "")

    def start_cell(res, ds, learner):
        """The start itself: the first point of local search's trace (its start network)."""
        ok = rows(res, ds, learner)
        ok = ok[ok["error"].isna()] if len(ok) else ok
        first = [tr[0] for tr in ok.get("trace", pd.Series(dtype=object)) if isinstance(tr, list) and tr]
        if not first:
            return "not run"
        return f"{100 * np.mean([p[4] for p in first]):.2f} ({np.mean([p[5] for p in first]):.0f})"

    L = [title,
         "Parts 2 and 3 with the learners started from Pypper's rules, FOIL's (`PFoil` for the class the "
         "network learns, at most one rule per first-layer node: `FoilInit`) and both (`PypperFoilInit`: "
         "Pypper's rules, then FOIL's new ones, at most one rule per first-layer node; the union of the "
         "two DNFs). Final test accuracy (%) with the backbone's edges in brackets; *the start*: the start "
         "network itself (the first point of local search's trace).\n"]
    for arch in TRACE_ARCHITECTURES:
        L += [f"**{arch}** `{NNF[arch]}`:\n",
              "| dataset | learner | " + " | ".join(f"from {s_}" for s_ in starts) + " |",
              "|---|---|" + "--:|" * len(starts)]
        for ds in dict.fromkeys(res4["dataset"]):
            L.append(f"| {ds} | the start | " + " | ".join(
                start_cell(both, ds, f"traced local search from {s_} {arch}") for s_ in starts) + " |")
            for a in ALGORITHMS:
                L.append(f"| | {a} | " + " | ".join(cell(both, ds, f"traced {a} from {s_} {arch}") for s_ in starts)
                         + " |")
        L.append("")
    ds = (QUICK_KRK if quick else KRK).name
    L += [f"**KRK** (`{KRK_ARCHITECTURE}`, {QUICK_KRK_SECONDS if quick else KRK_SECONDS} s per fit):\n",
          "| learner | " + " | ".join(f"from {s_}" for s_ in starts) + " |", "|---|" + "--:|" * len(starts),
          "| the start | " + " | ".join(start_cell(krk, ds, f"krk local search from {s_}") for s_ in starts) + " |"]
    for a in KRK_ALGORITHMS:
        L.append(f"| {a} | " + " | ".join(cell(krk, ds, f"krk {a} from {s_}") for s_ in starts) + " |")
    L.append("")
    styles = tuple((s_, STARTS[s_]) for s_ in starts)

    def accuracy(res, d, learner):
        r = rows(res, d, learner)
        return r[r["error"].isna()]["accuracy"].mean() if len(r) else np.nan

    refs = {d: [("Pypper", accuracy(both, d, "Pypper")), ("FOIL (NNF(1))", accuracy(both, d, "FOIL start NNF(1)"))]
            for d in res4["dataset"].unique()}
    L += _plot_traces(both, quick, [(arch, f" {arch}") for arch in TRACE_ARCHITECTURES], ALGORITHMS, refs,
                      prefix="traced ", starts=styles, tag="_foil")
    L += _plot_traces(krk, quick, [(f"{KRK_ARCHITECTURE}", "")], KRK_ALGORITHMS,
                      {ds: [("Pypper", accuracy(krk, ds, "Pypper")), ("FOIL", accuracy(krk, ds, "krk FOIL start"))]},
                      prefix="krk ", starts=styles, tag="_foil")
    return L + _failure_note(res4, lambda n: n.replace("traced ", "")) + _failure_note(krk4, lambda n: n.replace("krk ", ""))


DESCRIPTION = f"""
Part 1: the 20 artificial datasets of Beck & Fürnkranz (2021), as used by Seip, Fürnkranz et
al. (catalog tag `artificial`; all 1,024 combinations of ten Boolean features, labeled by a
random deep NNF), stratified {SEIP_FOLDS}-fold cross-validation. The Boolean columns become
`x` and `not x`, the features and negations the SAT paper's networks take. Architectures as
in the paper: NNF(1) `{NNF['NNF(1)']}`, NNF(3) `{NNF['NNF(3)']}`, NNF(5) `{NNF['NNF(5)']}`,
first layer AND, output OR, the output learning the formula's class (`target_class=True`,
as the papers' networks do). Learners: `DeepSAT` (plain SAT, an exact fit; Glucose 4),
`DeepLocalSearch` (Beck's stochastic hill climbing, consistent moves, random start),
`DeepGenetic` (the memetic algorithm, node-wise crossover, random start), `DeepNeuroFuzzy`
(gradient descent on the fuzzy relaxation, random binarization, random start),
`DeepSAT(minimize="weights")` on NNF(1) with {MAXSAT_SECONDS} s (the smallest exact fit found
by then), Pypper.

Part 2: {", ".join(d if isinstance(d, str) else d.name for d in TRACE_DATASETS)}, stratified
{TRACE_FOLDS}-fold (parity-4-of-10: the parity of four of ten Boolean features, all 1,024 rows).
NNF(1) and NNF(3); local search, the genetic algorithm, MaxSAT (`DeepSAT(minimize="weights")`
with soft examples, {TRACE_MAXSAT_SECONDS} s) and `DeepNeuroFuzzy`, each from a random start and
from Pypper's rules (`RuleSetInit(Pypper())`), all minimizing the training errors plus
lambda = {LAMBDA:g} per edge (`size_penalty`), with `trace=True`; Pypper as the reference.
Local search, the genetic algorithm and `DeepNeuroFuzzy` stop after {TRACE_SECONDS} s at the
latest (`max_seconds`), with their best network so far.

Part 3: KRK illegality, {KRK.n} positions (coordinates only), stratified {TRACE_FOLDS}-fold,
network `{KRK_ARCHITECTURE}`; local search, the genetic algorithm and `DeepNeuroFuzzy`, random
and from Pypper, lambda = {LAMBDA:g}, each with {KRK_SECONDS} s (`max_seconds`, epochs and
generations unbounded). No MaxSAT: the formula would have millions of variables.

Part 4: parts 2 and 3 with the learners started from FOIL's rules instead of Pypper's
(`FoilInit`: `PFoil` for the output's class, at most one rule per first-layer node), and from
both (`PypperFoilInit`: Pypper's rules, then FOIL's new ones, the same limit).

Each fit is capped at {FIT_TIMEOUT:.0f} s; a time-out drops the learner's whole
cross-validation on that dataset (the folds not yet started are skipped; `time-out (k/n)`:
k of the n folds that ran failed, the others -- already running -- finished). Times are
wall-clock.
"""


def write_report(res1: pd.DataFrame, res2: pd.DataFrame, res3: pd.DataFrame, upper: pd.DataFrame,
                 res4: pd.DataFrame, krk4: pd.DataFrame, quick: bool, path: str) -> None:
    L = [f"# Deep rule networks{' (quick run)' if quick else ''}\n",
         render_setup_section(DESCRIPTION)]
    L += _part1_section(res1, quick)
    L += _part2_section(res2, quick)
    L += _part3_section(res3, upper, quick)
    L += _part4_section(res2, res3, res4, krk4, quick)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Report -> {path}")


def main(quick: bool = True) -> None:
    res1 = part1(quick)
    res2 = part2(quick)
    res3, upper = part3(quick)
    res4, krk4 = part4(quick)
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(res1, res2, res3, upper, res4, krk4, quick, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="every dataset and fold, the full time limits")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    parser.add_argument("--report-only", action="store_true",
                        help="fit nothing: the report and plots from what the cache holds")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    CACHED_ONLY = args.report_only
    main(quick=not args.full)
