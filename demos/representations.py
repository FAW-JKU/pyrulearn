"""
demos/representations.py
========================

Five encodings of the same Boolean data, all behind one
`DataRepresentation` interface:

- `HorizontalBitmapRepresentation` -- a `numpy.packbits` bit matrix,
  packed by example (the baseline).
- `VerticalBitmapRepresentation` -- the same bit matrix packed by
  feature: one bit-vector over the examples per feature.
- `NListRepresentation` -- LORD's PPC-tree / N-list vertical index:
  per-feature row lists, bucketed by shared row prefix.
- `PrePostNListRepresentation` -- the same tree without the per-node path
  bitmasks: nodes compared by their pre/post codes, as in PrePost.
- `SparseDataRepresentation` -- `scipy` CSR/CSC: per-feature row lists
  without the prefix tree.

A learner asks its data only two things -- `coverage(rule)` and
`features_of(row)` -- so the five are interchangeable and must give the
same rules; only the time differs. This demo measures that time and checks
the equality at every point:

1. **Training-set size** -- the `adult` dataset (48,842 rows), training
   sets of doubling size from one shuffled pool, one fixed test set; with
   and without negation features.
2. **Density** -- synthetic data with a fixed learnable concept and a
   fixed number of features, whose noise features get sparser and sparser.
3. **Width** -- the same concept, a fixed number of true features per
   example, and more and more features: how sparse data usually looks
   (one-hot encodings of many-valued attributes, text), and where the
   sparse representations should start to pay off in memory.

All three also record the **memory** each representation holds after it is
built (and its peak while building), measured with `tracemalloc` in a
separate, untimed pass -- next to the unpacked `n x k` bool matrix a
learner gets from `data.X` -- and what a fit costs in total: that, plus
the peak the fit allocates on top of it (caches a representation builds
while a learner runs included), again in a separate, untimed pass.

Learners: CN2, PFossil, Pypper, PyLORD and ENDER, each with its defaults.
Every fit runs in a worker process under a cap (`FIT_TIMEOUT`, longer
for the learners in `FIT_TIMEOUTS`); after a time-out the remaining,
larger points of that curve are skipped. A cached time-out under a
smaller cap than the current one is fitted again.
`--parallel` first prepares every point (representations, their memory,
the `coverage()` timings) with nothing else running, and then fits the
learners side by side, one worker process each, every learner's fits one
after another.

Run: `python demos/representations.py` is the **quick** default
(sizes up to 1,000, a short density sweep); its report/plots go to
``representations_quick_*`` and are not checked in. `--full` runs the
full curves and writes ``representations_report.md`` (plots in
``representations_plots/``). Every measurement is cached, so an
interrupted run continues where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import tracemalloc
from typing import Dict, List, Optional

import numpy as np

from pyrulearn import DataSpec, Rule
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.data import (
    HorizontalBitmapRepresentation,
    NListRepresentation,
    PrePostNListRepresentation,
    SparseDataRepresentation,
    VerticalBitmapRepresentation,
)
from pyrulearn.data.io import binarize, build_dataspec
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.runner import TimeoutRunner
from pyrulearn.learners.boosting import ENDER
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.learners.seco import CN2, PFossil, Pypper

RANDOM_STATE = 0
MAX_INTERVALS = 8
FIT_TIMEOUT = 300.0
#: learners with a longer cap than FIT_TIMEOUT: PyLORD's and CN2's larger points need it
FIT_TIMEOUTS = {"PyLORD": 600.0, "CN2": 1800.0}
REPRESENTATIONS = ("HBitmap", "VBitmap", "NList", "PrePost", "Sparse")
_VERTICAL = {"VBitmap": VerticalBitmapRepresentation, "NList": NListRepresentation,
             "PrePost": PrePostNListRepresentation, "Sparse": SparseDataRepresentation}
LEARNERS = ("CN2", "PFossil", "Pypper", "PyLORD", "ENDER")

SIZE_DATASET = "adult"
SIZE_TEST_FRACTION = 0.2
SIZES = [500, 1000, 2000, 4000, 8000, 16000, 32000]
QUICK_SIZES = [250, 500, 1000]

DENSITY_N = 1000
DENSITY_K = 500
DENSITY_N_SIGNAL = 4
DENSITY_SIGNAL_P = 0.4
DENSITIES = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
QUICK_DENSITIES = [0.5, 0.05, 0.005]
WIDTH_N = 1000
WIDTH_NOISE_PER_ROW = 16      # expected true noise features per example, whatever k
WIDTHS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
QUICK_WIDTHS = [100, 1000, 5000]
COVERAGE_TRIALS = 800

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "representations"
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")
CACHE_DIR = os.path.join(HERE, "_representations_cache")

STYLE = {"HBitmap": ("o", "tab:blue"), "VBitmap": ("v", "tab:purple"), "NList": ("s", "tab:orange"),
         "PrePost": ("^", "tab:green"), "Sparse": ("d", "tab:red")}


def make_learner(name: str):
    return {"CN2": CN2, "PFossil": PFossil, "Pypper": Pypper, "PyLORD": PyLORD, "ENDER": ENDER}[name](
        random_state=RANDOM_STATE)


def timed_fit(learner_name: str, rep):
    """Runs in the worker process: fit, and time only the fit itself (not
    shipping the data to the worker). Falls back to CPU time when wall time
    ran far ahead of it -- the computer slept during the fit."""
    w0, c0 = time.perf_counter(), time.process_time()
    model = make_learner(learner_name).fit(rep)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    return model, (cpu if wall - cpu > 30 else wall)


def measured_fit(learner_name: str, rep) -> int:
    """Runs in the worker process: the peak memory the fit allocates on top
    of `rep` (which arrived before tracing starts) -- the learner's own
    work and whatever the representation caches while it runs. Untimed:
    tracing slows allocation down."""
    tracemalloc.start()
    try:
        make_learner(learner_name).fit(rep)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _build(kind: str, spec, X, y):
    """The representation of kind `kind`, and the seconds spent building
    its index from the shared HBitmap matrix (0 for HBitmap itself)."""
    boolean = HorizontalBitmapRepresentation(spec, X, y)
    if kind == "HBitmap":
        return boolean, 0.0
    t = time.perf_counter()
    rep = _VERTICAL[kind].from_boolean(boolean)
    return rep, time.perf_counter() - t


def _measure_memory(kind: str, spec, X, y) -> dict:
    """Bytes the representation of kind `kind` holds once built, and the peak
    while building it (`tracemalloc`; the inputs `X` and `y` are allocated
    before tracing starts, so neither counts, nor does the HBitmap matrix a
    vertical index is built from). A separate pass from `_build`, which is
    timed: tracing slows allocation down."""
    boolean = None if kind == "HBitmap" else HorizontalBitmapRepresentation(spec, X, y)
    tracemalloc.start()
    try:
        rep = HorizontalBitmapRepresentation(spec, X, y) if kind == "HBitmap" else _VERTICAL[kind].from_boolean(boolean)
        held, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del rep, boolean
    return {"mem_bytes": held, "mem_peak_bytes": peak, "x_bytes": int(X.size * X.itemsize)}


class Cache:
    """One JSON line per finished measurement, keyed by what was measured."""

    def __init__(self, path: str):
        self.path = path
        self.rows: Dict[str, dict] = {}
        self._lock = threading.Lock()  # the learners' threads of a --parallel run share it
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.rows[row["key"]] = row

    def get(self, key: str) -> Optional[dict]:
        return self.rows.get(key)

    def put(self, row: dict) -> None:
        with self._lock:
            self.rows[row["key"]] = row
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


def fit_timeout(learner: str) -> float:
    """The cap on one fit of `learner`, in seconds."""
    return FIT_TIMEOUTS.get(learner, FIT_TIMEOUT)


def _fit_point(runner, cache, key_base: dict, learner: str, reps: dict, test_reps: dict,
               skip: set, label: str = "") -> List[dict]:
    """Fits `learner` on every representation in `reps` (unless its curve
    already timed out, see `skip`), checking rules and test predictions
    against HBitmap. A cached time-out is fitted again when it happened under
    a smaller cap than `fit_timeout(learner)` (a row without a recorded cap
    timed out under `FIT_TIMEOUT`)."""
    rows = []
    cap = fit_timeout(learner)
    for kind in REPRESENTATIONS:
        key = json.dumps({**key_base, "learner": learner, "rep": kind}, sort_keys=True)
        row = cache.get(key)
        if row is not None and row["error"] == "timeout" and row.get("timeout", FIT_TIMEOUT) < cap:
            row = None
        if row is None:
            if (learner, key_base.get("negation"), kind) in skip:
                continue
            (result, err) = runner.run(timed_fit, learner, reps[kind], timeout=cap)
            if err is None:
                model, seconds = result
                row = {"key": key, **key_base, "learner": learner, "rep": kind, "fit_time": seconds,
                       "error": None, "rules": _digest(str(model)),
                       "preds": _digest(",".join(map(str, model.predict(test_reps[kind])))),
                       "n_rules": model.evaluate(test_reps[kind]).n_rules}
            else:
                row = {"key": key, **key_base, "learner": learner, "rep": kind, "fit_time": None,
                       "error": err, "rules": None, "preds": None, "n_rules": None, "timeout": cap}
            cache.put(row)
            cached = ""
        else:
            cached = " (cached)"
        if row["error"] is None and "fit_peak" not in row:
            (peak, err) = runner.run(measured_fit, learner, reps[kind], timeout=cap)
            row = {**row, "fit_peak": peak if err is None else None}
            cache.put(row)
        status = f"{row['fit_time']:.2f}s" if row["error"] is None else row["error"]
        print(f"    {label}{learner:8s} {kind:12s} {status}{cached}", flush=True)
        if row["error"] is not None:
            skip.add((learner, key_base.get("negation"), kind))
        rows.append(row)
    return rows


# -- part 1: training-set size -------------------------------------------------------

def run_size_curve(runner, cache, sizes, defer: Optional[list] = None) -> List[dict]:
    entry = Catalog.default().select(names=[SIZE_DATASET])[0]
    df, target = entry.load()
    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(len(df))
    n_test = int(round(SIZE_TEST_FRACTION * len(df)))
    test_df, pool = df.iloc[order[:n_test]], df.iloc[order[n_test:]]
    rows, skip = [], set()
    for negation in (True, False):
        for n in sizes:
            train_df = pool.iloc[:n]
            spec = build_dataspec(train_df, target=target, discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS),
                                  include_negations=negation, skip_unusable=True).build()
            X, Xt = binarize(spec, train_df), binarize(spec, test_df)
            y, yt = train_df[target].astype(str).to_numpy(), test_df[target].astype(str).to_numpy()
            reps, test_reps = {}, {}
            base = {"part": "size", "negation": negation, "n": n}
            for kind in REPRESENTATIONS:
                reps[kind], build_s = _build(kind, spec, X, y)
                test_reps[kind] = _build(kind, spec, Xt, yt)[0]
                rows.append({**base, "learner": None, "rep": kind, "build_time": build_s,
                             "density": float(X.mean()), "n_features": spec.n_features,
                             **_measure_memory(kind, spec, X, y)})
            print(f"  {SIZE_DATASET}, {'with' if negation else 'without'} negation, n={n} "
                  f"({spec.n_features} features, density {X.mean():.3f})", flush=True)
            if defer is not None:  # --parallel: the fits come later, see _fit_points_parallel
                defer.append((base, f"n={n} {'neg' if negation else 'pos'} ", reps, test_reps))
                continue
            for learner in LEARNERS:
                rows += _fit_point(runner, cache, base, learner, reps, test_reps, skip)
    return rows


# -- part 2: density ---------------------------------------------------------------------

def _synthetic(noise_p: float, seed: int, n: int):
    rng = np.random.default_rng(seed)
    signal = rng.random((n, DENSITY_N_SIGNAL)) < DENSITY_SIGNAL_P
    X = np.concatenate([signal, rng.random((n, DENSITY_K - DENSITY_N_SIGNAL)) < noise_p], axis=1)
    y = np.where((signal[:, 0] & ~signal[:, 1]) ^ (rng.random(n) < 0.05), "pos", "neg")
    return X, y


def _synthetic_point(base: dict, X, y, Xt, yt):
    """One point of a synthetic sweep: build every representation, time
    `COVERAGE_TRIALS` `coverage()` calls for random rules of 1-4 conditions,
    measure the memory, and fit every learner."""
    k = X.shape[1]
    spec = DataSpec([f"f{i}" for i in range(k)])
    rrng = np.random.default_rng(0)
    probe = [Rule(sorted(int(f) for f in rrng.choice(k, size=int(rrng.integers(1, 5)), replace=False)),
                  target="pos", dataspec=spec)
             for _ in range(COVERAGE_TRIALS)]
    rows, reps, test_reps = [], {}, {}
    for kind in REPRESENTATIONS:
        reps[kind], build_s = _build(kind, spec, X, y)
        test_reps[kind] = _build(kind, spec, Xt, yt)[0]
        timings = []
        for _ in range(3):  # best of three: the first pass also pays warm-up costs
            t = time.perf_counter()
            for r in probe:
                reps[kind].coverage(r)
            timings.append(time.perf_counter() - t)
        rows.append({**base, "learner": None, "rep": kind, "build_time": build_s,
                     "coverage_ms": min(timings) * 1000, "density": float(X.mean()),
                     **_measure_memory(kind, spec, X, y)})
    return rows, reps, test_reps


def run_density_sweep(runner, cache, densities, defer: Optional[list] = None) -> List[dict]:
    rows, skip = [], set()
    for p in densities:
        X, y = _synthetic(p, seed=0, n=DENSITY_N)
        Xt, yt = _synthetic(p, seed=1, n=DENSITY_N)
        base = {"part": "density", "noise_density": p}
        point, reps, test_reps = _synthetic_point(base, X, y, Xt, yt)
        rows += point
        print(f"  noise density {p} (overall {X.mean():.3f})", flush=True)
        if defer is not None:
            defer.append((base, f"p={p} ", reps, test_reps))
            continue
        for learner in LEARNERS:
            rows += _fit_point(runner, cache, base, learner, reps, test_reps, skip)
    return rows


# -- part 3: width -------------------------------------------------------------------------

def _wide(k: int, seed: int, n: int):
    """The density sweep's concept on `DENSITY_N_SIGNAL` signal features, plus
    `k - DENSITY_N_SIGNAL` noise features of which each example has
    `WIDTH_NOISE_PER_ROW` true on average (a binomial count, at random
    positions) -- so the true features per example stay fixed as k grows."""
    rng = np.random.default_rng(seed)
    signal = rng.random((n, DENSITY_N_SIGNAL)) < DENSITY_SIGNAL_P
    m = k - DENSITY_N_SIGNAL
    noise = np.zeros((n, m), dtype=bool)
    for i, c in enumerate(rng.binomial(m, min(1.0, WIDTH_NOISE_PER_ROW / m), size=n)):
        noise[i, rng.choice(m, size=c, replace=False)] = True
    y = np.where((signal[:, 0] & ~signal[:, 1]) ^ (rng.random(n) < 0.05), "pos", "neg")
    return np.concatenate([signal, noise], axis=1), y


def run_width_sweep(runner, cache, widths, defer: Optional[list] = None) -> List[dict]:
    rows, skip = [], set()
    for k in widths:
        X, y = _wide(k, seed=0, n=WIDTH_N)
        Xt, yt = _wide(k, seed=1, n=WIDTH_N)
        base = {"part": "width", "k": k}
        point, reps, test_reps = _synthetic_point(base, X, y, Xt, yt)
        rows += point
        print(f"  k={k} (density {X.mean():.4f}, {X.sum(axis=1).mean():.1f} true per example)", flush=True)
        if defer is not None:
            defer.append((base, f"k={k} ", reps, test_reps))
            continue
        for learner in LEARNERS:
            rows += _fit_point(runner, cache, base, learner, reps, test_reps, skip)
    return rows


# -- --parallel: the learners side by side ---------------------------------------------------

def _fit_points_parallel(points, cache) -> List[dict]:
    """Fits every learner at every prepared point: one thread per learner,
    each with its own worker process, going through the points in order --
    so the learners run side by side, a learner's own fits one after
    another, and a time-out still skips the rest of that learner's curve."""
    runners = [TimeoutRunner(timeout=FIT_TIMEOUT) for _ in LEARNERS]  # started one after the other
    rows: Dict[str, List[dict]] = {learner: [] for learner in LEARNERS}

    def work(learner: str, runner) -> None:
        skips: Dict[str, set] = {}  # one per part, as in the sequential run
        for base, label, reps, test_reps in points:
            skip = skips.setdefault(base["part"], set())
            rows[learner] += _fit_point(runner, cache, base, learner, reps, test_reps, skip, label=label)

    threads = [threading.Thread(target=work, args=(learner, runner), daemon=True)
               for learner, runner in zip(LEARNERS, runners)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        for runner in runners:
            runner.close()
    return [r for learner in LEARNERS for r in rows[learner]]


# -- plots ---------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, name: str, quick: bool) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"{name}{'_quick' if quick else ''}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    _plt().close(fig)
    return path


def _plain_log_y(ax) -> None:
    """Log y axis with tick labels as plain numbers (0.6, 2, 40), not 6x10^-1;
    of the minor ticks only the 2s and 5s are labeled, so they don't overlap."""
    import math
    from matplotlib.ticker import FuncFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(
        lambda v, _: f"{v:g}" if v > 0 and round(v / 10 ** math.floor(math.log10(v)), 6) in (2, 5) else ""))


def write_plots(size_rows, density_rows, width_rows, quick: bool) -> Dict[str, str]:
    plt = _plt()
    paths = {}
    fits = [r for r in size_rows if r["learner"] and r["error"] is None]
    sizes = sorted({r["n"] for r in size_rows})

    def size_axes(ax, kinds, log_y: bool, ylabel: str, title: str) -> None:
        # x: log-spaced, labeled with the actual sizes; legend: colors for the
        # representations and line styles for the encoding, as two groups
        from matplotlib.lines import Line2D
        from matplotlib.ticker import NullLocator
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n:,}" for n in sizes])
        ax.xaxis.set_minor_locator(NullLocator())
        if log_y:
            _plain_log_y(ax)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel(f"training examples ({SIZE_DATASET})")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        handles = [Line2D([], [], color=STYLE[k][1], marker=STYLE[k][0], markersize=4, label=k) for k in kinds]
        handles += [Line2D([], [], color="black", linestyle="-", label="with negation"),
                    Line2D([], [], color="black", linestyle="--", label="without negation")]
        ax.legend(handles=handles, fontsize=7)
        ax.grid(alpha=0.3, which="both")

    def size_lines(ax, rows, value: str, kinds) -> None:
        for kind in kinds:
            marker, color = STYLE[kind]
            for negation, ls in ((True, "-"), (False, "--")):
                pts = sorted((r["n"], r[value]) for r in rows if r["rep"] == kind and r["negation"] == negation)
                if pts:
                    ax.plot(*zip(*pts), ls, marker=marker, color=color, markersize=4)

    for learner in LEARNERS:
        mine = [r for r in fits if r["learner"] == learner]
        for log_y, suffix in ((True, ""), (False, "_linear")):
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            size_lines(ax, mine, "fit_time", REPRESENTATIONS)
            size_axes(ax, REPRESENTATIONS, log_y, "fit time (s)" + ("" if log_y else ", linear scale"),
                      f"{learner}: fit time vs. training-set size")
            paths[f"size_{learner}{suffix}"] = _save(fig, f"size_{learner.lower()}{suffix}", quick)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    builds = [r for r in size_rows if r["learner"] is None and r["rep"] != "HBitmap"]
    size_lines(ax, builds, "build_time", REPRESENTATIONS[1:])
    size_axes(ax, REPRESENTATIONS[1:], True, "index build time (s)",
              "building the index from the HBitmap matrix")
    paths["size_build"] = _save(fig, "size_index_build", quick)

    dfits = [r for r in density_rows if r["learner"] and r["error"] is None]
    densities = sorted({r["noise_density"] for r in density_rows})

    def density_x(ax) -> None:
        from matplotlib.ticker import NullLocator
        ax.set_xscale("log")
        ax.set_xticks(densities)
        ax.set_xticklabels([f"{d:g}" for d in densities])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.invert_xaxis()

    for learner in LEARNERS:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for kind in REPRESENTATIONS:
            marker, color = STYLE[kind]
            pts = sorted((r["noise_density"], r["fit_time"]) for r in dfits
                         if r["learner"] == learner and r["rep"] == kind)
            if pts:
                ax.plot(*zip(*pts), "-", marker=marker, color=color, markersize=4, label=kind)
        _plain_log_y(ax)
        density_x(ax)
        ax.set_xlabel(f"density of the noise features (k={DENSITY_K}, n={DENSITY_N})")
        ax.set_ylabel("fit time (s)")
        ax.set_title(f"{learner}: fit time vs. density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        paths[f"density_{learner}"] = _save(fig, f"density_{learner.lower()}", quick)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for kind in REPRESENTATIONS:
        marker, color = STYLE[kind]
        pts = sorted((r["noise_density"], r["coverage_ms"]) for r in density_rows
                     if r["learner"] is None and r["rep"] == kind)
        ax.plot(*zip(*pts), "-", marker=marker, color=color, markersize=4, label=kind)
    _plain_log_y(ax)
    density_x(ax)
    ax.set_xlabel(f"density of the noise features (k={DENSITY_K}, n={DENSITY_N})")
    ax.set_ylabel(f"time for {COVERAGE_TRIALS} coverage() calls (ms)")
    ax.set_title("coverage() alone vs. density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    paths["density_coverage"] = _save(fig, "density_coverage", quick)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    mem = [r for r in size_rows if r["learner"] is None]
    size_lines(ax, [{**r, "mb": r["mem_bytes"] / 2**20} for r in mem], "mb", REPRESENTATIONS)
    size_axes(ax, REPRESENTATIONS, True, "memory held (MiB)", "memory of the representation vs. training-set size")
    paths["size_memory"] = _save(fig, "size_memory", quick)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for kind in REPRESENTATIONS:
        marker, color = STYLE[kind]
        pts = sorted((r["noise_density"], r["mem_bytes"] / 2**20) for r in density_rows
                     if r["learner"] is None and r["rep"] == kind)
        ax.plot(*zip(*pts), "-", marker=marker, color=color, markersize=4, label=kind)
    _plain_log_y(ax)
    density_x(ax)
    ax.set_xlabel(f"density of the noise features (k={DENSITY_K}, n={DENSITY_N})")
    ax.set_ylabel("memory held (MiB)")
    ax.set_title("memory of the representation vs. density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    paths["density_memory"] = _save(fig, "density_memory", quick)

    widths = sorted({r["k"] for r in width_rows})

    def width_x(ax) -> None:
        from matplotlib.ticker import NullLocator
        ax.set_xscale("log")
        ax.set_xticks(widths)
        ax.set_xticklabels([f"{k:,}" for k in widths], rotation=30)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(f"features k (n={WIDTH_N}, about {WIDTH_NOISE_PER_ROW + DENSITY_N_SIGNAL * DENSITY_SIGNAL_P:.0f} "
                      "true per example)")
        ax.grid(alpha=0.3, which="both")

    for value, ylabel, title, key in (
            ("mem_bytes", "memory held (MiB)", "memory of the representation vs. width", "width_memory"),
            ("coverage_ms", f"time for {COVERAGE_TRIALS} coverage() calls (ms)", "coverage() alone vs. width",
             "width_coverage")):
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        scale = 2 ** -20 if value == "mem_bytes" else 1
        for kind in REPRESENTATIONS:
            marker, color = STYLE[kind]
            pts = sorted((r["k"], r[value] * scale) for r in width_rows if r["learner"] is None and r["rep"] == kind)
            ax.plot(*zip(*pts), "-", marker=marker, color=color, markersize=4, label=kind)
        if value == "mem_bytes":
            pts = sorted((r["k"], r["x_bytes"] * scale) for r in width_rows
                         if r["learner"] is None and r["rep"] == "HBitmap")
            ax.plot(*zip(*pts), ":", color="gray", label="unpacked X")
        _plain_log_y(ax)
        width_x(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        paths[key] = _save(fig, key, quick)

    wfits = [r for r in width_rows if r["learner"] and r["error"] is None]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax in axes.ravel()[len(LEARNERS):]:
        ax.set_visible(False)
    for ax, learner in zip(axes.ravel(), LEARNERS):
        for kind in REPRESENTATIONS:
            marker, color = STYLE[kind]
            pts = sorted((r["k"], r["fit_time"]) for r in wfits if r["learner"] == learner and r["rep"] == kind)
            if pts:
                ax.plot(*zip(*pts), "-", marker=marker, color=color, markersize=4, label=kind)
        _plain_log_y(ax)
        width_x(ax)
        ax.set_ylabel("fit time (s)")
        ax.set_title(learner)
        ax.legend(fontsize=7)
    paths["width_fit"] = _save(fig, "width_fit", quick)
    return paths


# -- report ----------------------------------------------------------------------------------

def _agreement(rows) -> tuple:
    """(points compared, points where every representation matched HBitmap,
    list of mismatches)."""
    groups: Dict[str, Dict[str, dict]] = {}
    for r in rows:
        if r["learner"] and r["error"] is None:
            k = json.dumps({x: r[x] for x in r if x not in (
                "key", "rep", "fit_time", "error", "rules", "preds", "n_rules", "fit_peak")}, sort_keys=True)
            groups.setdefault(k, {})[r["rep"]] = r
    compared, agreed, bad = 0, 0, []
    for k, by_rep in groups.items():
        if "HBitmap" not in by_rep or len(by_rep) < 2:
            continue
        compared += 1
        ref = by_rep["HBitmap"]
        diff = [kind for kind, r in by_rep.items() if (r["rules"], r["preds"]) != (ref["rules"], ref["preds"])]
        if diff:
            bad.append((json.loads(k), diff))
        else:
            agreed += 1
    return compared, agreed, bad


def _fit_table(rows, x_field: str, x_label: str, negation=None) -> str:
    """One row per (learner, representation), fit time at each `x_field`
    value. Every non-HBitmap cell also carries, in parentheses, its ratio
    to HBitmap's fit time at the same point (``19.41 (37%)``) -- below
    100% is faster than HBitmap -- so a gain or a loss sits right next to
    the number it's relative to, rather than in a separate, unweighted
    summary that a few cheap points can dominate."""
    fits = [r for r in rows if r["learner"] and (negation is None or r["negation"] == negation)]
    xs = sorted({r[x_field] for r in fits}, reverse=(x_field == "noise_density"))
    lines = [f"| learner | representation | " + " | ".join(f"{x_label}={x}" for x in xs) + " |\n",
             "|---|---|" + "--:|" * len(xs) + "\n"]
    for learner in LEARNERS:
        boolean_at = {}
        for kind in REPRESENTATIONS:
            cells = []
            for x in xs:
                r = next((r for r in fits if r["learner"] == learner and r["rep"] == kind and r[x_field] == x), None)
                if r is None:
                    cells.append("")
                elif r["error"] == "timeout":
                    cells.append("time-out")
                elif r["error"]:
                    cells.append("error")
                elif kind == "HBitmap":
                    boolean_at[x] = r["fit_time"]
                    cells.append(f"{r['fit_time']:.2f}")
                else:
                    ref = boolean_at.get(x)
                    ratio = f" ({100 * r['fit_time'] / ref:.0f}%)" if ref else ""
                    cells.append(f"{r['fit_time']:.2f}{ratio}")
            lines.append(f"| {learner} | {kind} | " + " | ".join(cells) + " |\n")
    return "".join(lines)


#: the growth exponent is fitted from this training-set size on: below it a fit's fixed
#: costs (setting up the search, the first rule) still dominate the time
GROWTH_MIN_N = 2000


def _growth_table(rows) -> str:
    """How fit time grows with the training-set size: the exponent ``b`` of
    ``time ~ n^b``, the least-squares slope of log time over log n across the
    sizes from `GROWTH_MIN_N` on that finished (1: linear, 2: quadratic). A
    curve that a time-out cut short is fitted on fewer points, shown in
    parentheses; with fewer than three, no exponent."""
    fits = [r for r in rows if r["learner"] and r["error"] is None and r["n"] >= GROWTH_MIN_N]
    lines = ["| learner | representation | with negation | without negation |\n", "|---|---|--:|--:|\n"]
    sizes = sorted({r["n"] for r in rows if r["n"] >= GROWTH_MIN_N})
    for learner in [l for l in LEARNERS if any(r["learner"] == l for r in fits)]:
        for kind in REPRESENTATIONS:
            cells = []
            for negation in (True, False):
                pts = sorted((r["n"], r["fit_time"]) for r in fits
                             if r["learner"] == learner and r["rep"] == kind and r["negation"] == negation
                             and r["fit_time"] and r["fit_time"] > 0)
                if len(pts) < 3:
                    cells.append("--")
                    continue
                b = np.polyfit(np.log([n for n, _ in pts]), np.log([t for _, t in pts]), 1)[0]
                cells.append(f"{b:.2f}" + ("" if len(pts) == len(sizes) else f" ({len(pts)} points)"))
            lines.append(f"| {learner} | {kind} | " + " | ".join(cells) + " |\n")
    return "".join(lines)


def _bytes(n: float) -> str:
    return f"{n / 1024:.0f} KiB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.1f} MiB"


def _memory_table(rows, x_field: str, x_label: str, field: str = "mem_bytes", negation=None) -> str:
    """One row per representation, the memory at each `x_field` value, each
    non-HBitmap cell with its ratio to HBitmap in parentheses; the last row is
    the unpacked bool matrix `X` a learner gets from `data.X`, for scale."""
    rows = [r for r in rows if r["learner"] is None and (negation is None or r["negation"] == negation)]
    xs = sorted({r[x_field] for r in rows}, reverse=(x_field == "noise_density"))
    lines = ["| representation | " + " | ".join(f"{x_label}={x}" for x in xs) + " |\n",
             "|---|" + "--:|" * len(xs) + "\n"]
    at = lambda kind, x: next((r for r in rows if r["rep"] == kind and r[x_field] == x), None)  # noqa: E731
    for kind in REPRESENTATIONS:
        cells = []
        for x in xs:
            r, ref = at(kind, x), at("HBitmap", x)
            if r is None:
                cells.append("")
            elif kind == "HBitmap" or ref is None:
                cells.append(_bytes(r[field]))
            else:
                cells.append(f"{_bytes(r[field])} ({100 * r[field] / ref[field]:.0f}%)")
        lines.append(f"| {kind} | " + " | ".join(cells) + " |\n")
    lines.append("| unpacked `X` | " + " | ".join(
        _bytes(at("HBitmap", x)["x_bytes"]) if at("HBitmap", x) else "" for x in xs) + " |\n")
    return "".join(lines)


def _fit_memory_table(rows, x_field: str, x_label: str, negation=None) -> str:
    """One row per (learner, representation): the memory a fit costs in
    total at each `x_field` value -- what the representation holds plus the
    peak the fit allocates on top of it -- each non-HBitmap cell with its
    ratio to HBitmap in parentheses."""
    rows = [r for r in rows if negation is None or r.get("negation") == negation]
    held = {(r["rep"], r[x_field]): r["mem_bytes"] for r in rows if r["learner"] is None}
    fits = [r for r in rows if r["learner"] and r["error"] is None and r.get("fit_peak") is not None]
    xs = sorted({r[x_field] for r in rows if r["learner"]}, reverse=(x_field == "noise_density"))
    lines = [f"| learner | representation | " + " | ".join(f"{x_label}={x}" for x in xs) + " |\n",
             "|---|---|" + "--:|" * len(xs) + "\n"]
    for learner in LEARNERS:
        boolean_at = {}
        for kind in REPRESENTATIONS:
            cells = []
            for x in xs:
                r = next((r for r in fits if r["learner"] == learner and r["rep"] == kind and r[x_field] == x), None)
                if r is None or (kind, x) not in held:
                    cells.append("")
                    continue
                total = held[(kind, x)] + r["fit_peak"]
                if kind == "HBitmap":
                    boolean_at[x] = total
                    cells.append(_bytes(total))
                else:
                    ref = boolean_at.get(x)
                    cells.append(_bytes(total) + (f" ({100 * total / ref:.0f}%)" if ref else ""))
            lines.append(f"| {learner} | {kind} | " + " | ".join(cells) + " |\n")
    return "".join(lines)


def write_report(size_rows, density_rows, width_rows, paths, quick: bool, report_path: str,
                 parallel: bool = False) -> None:
    plots = os.path.basename(PLOTS_DIR)
    img = lambda key, alt: f"![{alt}]({plots}/{os.path.basename(paths[key])})\n\n"  # noqa: E731
    size_info = {(r["negation"], r["n"]): r for r in size_rows if r["learner"] is None and r["rep"] == "HBitmap"}
    compared, agreed, bad = _agreement(size_rows + density_rows + width_rows)
    L = ["# Data representations: same rules, different speed\n\n"]
    L.append(
        ("**Quick run** (training sets up to 1,000 examples, short density sweep) -- a fast sanity "
         "check, the default with no arguments. Full run: `python demos/representations.py "
         "--full`.\n\n" if quick else
         "**Full run.** Quick sanity check instead: `python demos/representations.py`.\n\n")
        + "pyrulearn's learners never look at the data directly. They ask it two things -- which "
        "examples a rule covers (`coverage(rule)`) and which features an example has "
        "(`features_of(row)`) -- so the data can be stored in any structure that answers these "
        "questions. Five such representations are implemented:\n\n"
        "- **HBitmap** (horizontal bitmap) -- a bit matrix packed by example, one bit per example "
        "and feature. Covering a rule checks every example's row against the rule's features, "
        "whatever the density.\n"
        "- **VBitmap** (vertical bitmap) -- the same bit matrix packed by feature: one bit-vector "
        "over the examples per feature. Covering a rule ANDs its features' bit-vectors, and "
        "counting is a popcount; same memory as HBitmap.\n"
        "- **NList** -- LORD's vertical index (Huynh, Fürnkranz & Beck, 2023): examples are "
        "inserted into a prefix tree of their true features, and each feature keeps the list of "
        "tree nodes it occurs in (its N-list). Covering a rule intersects these lists; the work "
        "grows with how often the features are true, not with the number of examples. Each "
        "node keeps the bitmask of the features on its path, so that a refinement is one "
        "vectorized test.\n"
        "- **PrePost** -- the same tree without these bitmasks, as in PrePost (Deng et al., "
        "2012): a node is compared with another by their pre/post codes, and an example's "
        "features are read back by walking up the tree. Less memory, and binary searches "
        "instead of the bitmask test.\n"
        "- **Sparse** -- `scipy` sparse matrices: per-feature example lists without the prefix "
        "tree.\n\n"
        "All five must lead to exactly the same rules, so the only thing that differs is time. "
        "This demo measures that time for five learners -- CN2, PFossil, Pypper, PyLORD and "
        "ENDER, each with its default settings -- and checks the rules and the test-set predictions "
        "against HBitmap at every point. Each fit runs in a separate process and is "
        f"capped at {FIT_TIMEOUT:.0f}s"
        + "".join(f", {name}'s at {t:.0f}s" for name, t in FIT_TIMEOUTS.items())
        + "; after a time-out, the larger points of that curve are "
        "skipped. "
        + ("The learners ran side by side, one worker process each, every learner's fits one after "
           "another; building the representations, timing `coverage()` and measuring memory came "
           "first, with nothing else running." if parallel else
           "Everything ran one at a time.")
        + (" The numerical libraries were limited to one thread (`OMP_NUM_THREADS=1` and its "
           "relatives), so each fit used one core.\n\n"
           if os.environ.get("OMP_NUM_THREADS") == "1" else "\n\n")
        + f"**Equality check:** at {compared} points where HBitmap and at least one other "
        f"representation finished, {agreed} gave identical rules and predictions for every "
        f"representation" + (".\n\n" if not bad else
                              f"; mismatches: {bad}.\n\n"))

    L.append("## 1. Training-set size\n\n")
    L.append(
        f"Training sets of doubling size, drawn from one shuffled pool of the `{SIZE_DATASET}` "
        f"dataset (48,842 examples, 14 attributes, 8 of them nominal), each evaluated on the "
        f"same {SIZE_TEST_FRACTION:.0%} test set. Numeric attributes are discretized per "
        f"training set (`TreeDiscretizer(max_intervals={MAX_INTERVALS})`). Two encodings: **with negation** "
        "(pyrulearn's default -- every test comes with its negation, so exactly half the "
        "features are true in every example) and **without negation** (no `x != v` features; "
        "numeric attributes keep both `x >= t` and `x < t`, so it is sparser only in the "
        "nominal attributes). The first is the bitmaps' best case and the list-based "
        "representations' worst; the second is where those should gain.\n\n")
    largest = {neg: size_info[(neg, max(n for (ng, n) in size_info if ng == neg))]
               for neg in (True, False) if any(ng == neg for (ng, _) in size_info)}
    if largest:
        L.append("At the largest training set: "
                 + "; ".join(f"{'with' if neg else 'without'} negation {r['n_features']} binary "
                             f"features, density {r['density']:.3f}" for neg, r in largest.items())
                 + " (density = share of features that are true).\n\n")
    L.append("One pair of plots per learner: left with a logarithmic time axis (ratios -- a "
             "constant factor between two representations is a constant gap), right with a "
             "linear one (absolute seconds saved).\n\n")
    for learner in LEARNERS:
        pair = [f'<img src="{plots}/{os.path.basename(paths[key])}" alt="{learner} fit time vs. '
                f'training-set size, {scale}" width="49%">'
                for key, scale in ((f"size_{learner}", "log scale"), (f"size_{learner}_linear", "linear scale"))]
        L.append(" ".join(pair) + "\n\n")
    L.append("Fit time in seconds, with negation:\n\n" + _fit_table(size_rows, "n", "n", True) + "\n")
    L.append("Without negation:\n\n" + _fit_table(size_rows, "n", "n", False) + "\n")
    L.append(f"**Growth.** How fast each learner's fit time grows with the training-set size: the exponent "
             f"*b* of time ~ n^*b*, fitted by least squares to log time over log n from n={GROWTH_MIN_N:,} on "
             "(below that, a fit's fixed costs still dominate). 1 is linear, 2 quadratic; a curve cut short "
             "by a time-out is fitted on the points that finished. This is what decides, for example, how much "
             "a learner gains from splitting a problem into smaller ones (`demos/multiclass_decomposition.py`):"
             "\n\n" + _growth_table(size_rows) + "\n")
    L.append("Building a vertical index is paid once per fit, on top of the fit itself:\n\n")
    L.append(img("size_build", "index build time vs. training-set size"))
    L.append(
        "**Memory.** What each representation holds once built, measured with `tracemalloc` (the "
        "input matrix and the HBitmap matrix the others are built from don't count), and in "
        "parentheses its share of HBitmap's memory; the last row is the "
        "unpacked `n x k` bool matrix that `data.X` builds on demand, which any learner or "
        "external library working on a plain array pays on top.\n\n")
    L.append(img("size_memory", "memory vs. training-set size"))
    L.append("With negation:\n\n" + _memory_table(size_rows, "n", "n", negation=True) + "\n")
    L.append("Without negation:\n\n" + _memory_table(size_rows, "n", "n", negation=False) + "\n")
    L.append("Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on "
             "top of it, caches the representation builds during the fit included (`tracemalloc`, a "
             "separate untimed pass), with negation:\n\n"
             + _fit_memory_table(size_rows, "n", "n", negation=True) + "\n")
    L.append("Without negation:\n\n" + _fit_memory_table(size_rows, "n", "n", negation=False) + "\n")
    L.append("Peak memory while building (with negation) -- the vertical indexes need more than "
             "they keep:\n\n" + _memory_table(size_rows, "n", "n", "mem_peak_bytes", negation=True) + "\n")

    L.append("## 2. Density\n\n")
    L.append(
        f"Synthetic data, {DENSITY_N} training and {DENSITY_N} test examples with "
        f"{DENSITY_K} features: {DENSITY_N_SIGNAL} of them carry a fixed learnable concept "
        f"(true with probability {DENSITY_SIGNAL_P}, class = f0 and not f1, 5% label noise); "
        "every other feature is noise, true with the probability on the x-axis. Size and "
        "feature count stay fixed, so this isolates density.\n\n")
    for learner in LEARNERS:
        L.append(img(f"density_{learner}", f"{learner} fit time vs. density"))
    L.append(
        f"The pure data operation, {COVERAGE_TRIALS} `coverage()` calls for random rules of 1-4 "
        "conditions, without any search around it:\n\n")
    L.append(img("density_coverage", "coverage() time vs. density"))
    L.append("Fit time in seconds:\n\n" + _fit_table(density_rows, "noise_density", "p") + "\n")
    L.append("Memory held by each representation (share of HBitmap in parentheses; the last row is the "
             "unpacked bool matrix):\n\n")
    L.append(img("density_memory", "memory vs. density"))
    L.append(_memory_table(density_rows, "noise_density", "p") + "\n")
    L.append("Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on "
             "top of it, caches the representation builds during the fit included (`tracemalloc`, a "
             "separate untimed pass):\n\n"
             + _fit_memory_table(density_rows, "noise_density", "p") + "\n")

    L.append("## 3. Width\n\n")
    L.append(
        f"Synthetic data again, {WIDTH_N} training and {WIDTH_N} test examples, the same concept on "
        f"{DENSITY_N_SIGNAL} signal features -- but now the number of *true* features per example stays "
        f"fixed (on average {WIDTH_NOISE_PER_ROW} noise features, plus the signal ones) while the number "
        "of features grows. That is how sparse data usually looks: one-hot encodings of attributes with "
        "many values, or words in a text. The bitmaps grow with the number of features "
        "(one bit per cell); a sparse representation with the number of true cells, which stays the same. "
        "`Sparse` keeps two copies of a 4-byte index plus a 1-byte value per true cell, about 10 bytes "
        "against 1/8 byte per cell for the bitmaps, so it should take less memory below a density of about "
        f"1/80 -- here from about {80 * (WIDTH_NOISE_PER_ROW + DENSITY_N_SIGNAL * DENSITY_SIGNAL_P):,.0f} "
        "features on.\n\n")
    L.append(img("width_memory", "memory vs. width"))
    L.append("Memory held by each representation (share of HBitmap in parentheses; the last row is the "
             "unpacked bool matrix):\n\n" + _memory_table(width_rows, "k", "k") + "\n")
    L.append(f"{COVERAGE_TRIALS} `coverage()` calls for random rules of 1-4 conditions:\n\n")
    L.append(img("width_coverage", "coverage() time vs. width"))
    L.append(img("width_fit", "fit time vs. width"))
    L.append("Fit time in seconds:\n\n" + _fit_table(width_rows, "k", "k") + "\n")
    L.append("Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on "
             "top of it, caches the representation builds during the fit included (`tracemalloc`, a "
             "separate untimed pass):\n\n"
             + _fit_memory_table(width_rows, "k", "k") + "\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def main(quick: bool = True, parallel: bool = False) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = Cache(os.path.join(CACHE_DIR, f"{'quick' if quick else 'full'}.jsonl"))
    points = [] if parallel else None
    with TimeoutRunner(timeout=FIT_TIMEOUT) as runner:
        print(f"Part 1: training-set size ({SIZE_DATASET})", flush=True)
        size_rows = run_size_curve(runner, cache, QUICK_SIZES if quick else SIZES, defer=points)
        print("Part 2: density", flush=True)
        density_rows = run_density_sweep(runner, cache, QUICK_DENSITIES if quick else DENSITIES, defer=points)
        print("Part 3: width", flush=True)
        width_rows = run_width_sweep(runner, cache, QUICK_WIDTHS if quick else WIDTHS, defer=points)
    if parallel:
        print(f"Fitting {len(LEARNERS)} learners side by side", flush=True)
        fits = _fit_points_parallel(points, cache)
        size_rows += [r for r in fits if r["part"] == "size"]
        density_rows += [r for r in fits if r["part"] == "density"]
        width_rows += [r for r in fits if r["part"] == "width"]
    paths = write_plots(size_rows, density_rows, width_rows, quick)
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(size_rows, density_rows, width_rows, paths, quick, report, parallel=parallel)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="full curves (training sets up to 32,000, the whole density sweep)")
    parser.add_argument("--parallel", action="store_true",
                        help="prepare every point first, then fit the learners side by side, one process each")
    args = parser.parse_args()
    main(quick=not args.full, parallel=args.parallel)
