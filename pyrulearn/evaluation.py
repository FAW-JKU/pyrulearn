"""
pyrulearn.evaluation
=====================

Measured statistics for rules and models (`RuleStats`, `ConfusionMatrix`,
`ModelStats`) and coverage-space plots -- a standard rule-learning
visualization (see e.g. Fürnkranz & Flach's ROC/coverage-space analyses)
where a point is ``(negatives covered, positives covered)``. (Named
`evaluation`, not `analysis` -- this module is specifically about
*measuring how a rule/model actually performed*, as opposed to
inspecting its logical structure.)

`RuleStats` -- the one-vs-rest confusion-matrix quartet (`tp`/`fp`/`fn`/
`tn`) a single rule/candidate produces against one positive class, plus
optional search `length` -- lives here (`pyrulearn.heuristics` re-exports
it for every existing importer) since it's a measured-statistics concept,
not a heuristic-search one; `pyrulearn.heuristics` just scores it.
`ConfusionMatrix` generalizes it to every label at once (a genuine
N-label confusion matrix, not just a designated positive class):
`ConfusionMatrix.rule_stats(label)` derives the one-vs-rest `RuleStats`
for any chosen label on demand -- two different labels give two
genuinely different `RuleStats` from the same matrix (tp/fp for one
label are the other's fn/tn mirrored, not a stored, retrievable pair),
so this is a computation, not a lookup.

`ModelStats` is what `pyrulearn.models.RuleModel.evaluate(data)`
returns (and what a rule's frozen training stats, `SingleRule.stats()`,
are): a `ConfusionMatrix` from a fitted model's own `predict(data)` vs
`data.y`, plus `n_rules`/`n_conditions` complexity counts -- a whole
model's (or one rule's) measured performance, as opposed to one
candidate rule's `RuleStats`. Lives in this
module (not `models.py`) alongside the `ConfusionMatrix` it wraps, kept
apart from `pyrulearn.models`'s own `RuleModel` hierarchy definitions.

Three related but distinct plots also live here:

- ``coverage_space_plot``: one point per rule. For a ``RuleSet`` (no
  inherent order) each point is that rule's own raw coverage, in
  isolation, optionally overlaid with ``build_refinement_graph`` edges
  connecting rules that differ by one literal, and/or (``show_convex_hull``)
  the boundary of the area under the ROC-convex-hull curve -- from
  ``(0, 0)`` to ``(N, 0)`` to ``(N, P)`` and back to ``(0, 0)`` via the
  hull itself (``_upper_hull``) -- whose enclosed area is exactly what
  ``coverage_space_auc`` integrates. For a ``RuleList`` (ordered) it's
  instead points connected by directional
  **arrows**, in order, starting at ``(0, 0)``, via
  ``RuleList.coverage_path`` -- each step is the *unique* (fired)
  coverage of trying that rule after all the earlier ones, i.e. the
  decision list's cumulative coverage as you'd apply it.
- ``rule_refinement_plot``: the specialization path of a single
  ``Rule``'s own conditions, via ``rule_refinement_path`` -- also drawn
  as directional arrows, starting at ``(N, P)`` (the empty rule, covering
  everyone) and adding one condition at a time, in the rule's own order
  (see ``Rule.order_by_precision`` for computing a reasonable order first
  if the rule doesn't have one).

Both are thin wrappers around ``CoverageSpace``, the reusable, composable
version of the same idea: a coverage-space (or, ``normalized=True``, ROC-
space) plot that owns its dimensions and ``Axes``, with ``plot_ruleset``/
``plot_rule_refinement`` methods that draw a layer into it -- so several
layers (a heuristic's isometrics via ``RuleHeuristic.plot_isometrics``, a
ruleset's coverage, a rule's refinement path) can share one plot instead
of each call starting a fresh figure.

Only imports matplotlib/networkx lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .data import DataRepresentation
from .models import FlatRuleSet, RuleList, RuleModel
from .rule import Rule


# -- how numbers are shown ------------------------------------------------------
# One convention for everything pyrulearn prints -- reprs, progress lines and the
# demos' reports: accuracies as percentages with two decimals, means of counts
# (rules, conditions, conditions per rule) with one decimal.

def format_accuracy(value: float) -> str:
    """An accuracy (a fraction) as a percentage with two decimals, ``86.31%``;
    NaN prints as ``n/a``."""
    return "n/a" if value != value else f"{100 * value:.2f}%"


def format_mean_count(value: float) -> str:
    """A mean of integers (number of rules, of conditions, conditions per rule)
    with one decimal, ``12.3``; NaN prints as ``n/a``."""
    return "n/a" if value != value else f"{value:.1f}"

if TYPE_CHECKING:
    from .heuristics import GainHeuristic, RuleHeuristic


def _sortkey(c: Any):
    """Total order for labels that keeps numbers numeric -- for
    deterministic, otherwise-arbitrary ordering of label lists. (Same
    rule `pyrulearn.models` uses for the same reason; duplicated rather
    than imported to keep this module free of any dependency on
    `models.py`.)"""
    return (0, c) if isinstance(c, (int, float)) else (1, str(c))


#: sentinel pseudo-label for a `None` (abstained) prediction in a
#: `ConfusionMatrix` -- distinct from any real label, including a real
#: `None` label if one ever occurs, since `is`/`in` identity is used to
#: place it, not equality. It prints as ``ABSTAIN``.
class _Abstain:
    __slots__ = ()

    def __repr__(self) -> str:
        return "ABSTAIN"

    def __reduce__(self):             # pickling keeps the singleton
        return "ABSTAIN"


ABSTAIN = _Abstain()


@dataclass(frozen=True)
class RuleStats:
    """The confusion-matrix quartet a `RuleHeuristic` scores against,
    plus optional search context.

    `tp`/`fp` are this rule's covered positives/negatives (treating
    "covers" as "predicts positive"); `fn`/`tn` are the positives/
    negatives it *doesn't* cover. `n_pos`/`n_neg` (total actual
    positives/negatives) are derived, not stored, since they're just
    `tp+fn`/`fp+tn`.

    The four counts are plain integers (rows), or floats -- sums of row
    weights -- when the data carries weights or a search is restricted
    to a weight vector (see `pyrulearn.data.DataRepresentation`). Every
    heuristic is plain arithmetic on them and works with either.

    `length` (rule size, e.g. `Rule.length()`) defaults to unused --
    only heuristics that specifically want it read past `tp`/`fp`/`fn`/
    `tn`. A gain-style heuristic's *parent* stats (the rule before its
    last refinement) aren't part of this bundle -- see `GainHeuristic`,
    which takes them as a separate, mandatory argument to `score`
    instead.
    """
    tp: Union[int, float]
    fp: Union[int, float]
    fn: Union[int, float]
    tn: Union[int, float]
    length: int = 0

    @property
    def n_pos(self) -> Union[int, float]:
        return self.tp + self.fn

    @property
    def n_neg(self) -> Union[int, float]:
        return self.fp + self.tn

    @classmethod
    def universal(cls, n_pos: int, n_neg: int) -> "RuleStats":
        """Stats for the *universal* rule -- the empty conjunction
        (vacuously true), which covers every example -- given a
        dataset's total `n_pos`/`n_neg`. The natural "no rule yet"
        baseline for gain-style heuristics like `FoilGain` (see
        `RuleHeuristic.plot_isometrics`'s `parent=` argument), or any
        other place that wants the stats of "predict positive always"
        without needing an actual `Rule`/`DataSpec` on hand to compute
        them from. Unlike a genuine `Rule`, `RuleStats` has no
        contradictory-literal invariant to satisfy, so this (and
        `empty`) are just the two degenerate confusion-matrix corners --
        no dedicated `Rule` object is needed for either, and there
        wouldn't be a natural one for `empty` anyway (an always-false
        `Rule` would need literally contradictory conditions, which
        `Rule.__init__` rejects by design).
        """
        return cls(tp=n_pos, fp=n_neg, fn=0, tn=0)

    @classmethod
    def empty(cls, n_pos: int, n_neg: int) -> "RuleStats":
        """Stats for the *empty* rule -- covers no example at all
        ("predict negative always") -- the opposite corner from
        `universal`. See `universal` for why this is a `RuleStats`
        classmethod rather than an actual `Rule`.
        """
        return cls(tp=0, fp=0, fn=n_pos, tn=n_neg)

    @classmethod
    def from_rule(
        cls,
        rule: Rule,
        data: DataRepresentation,
        positive_class: Optional[Any] = None,
        example_mask: Optional[np.ndarray] = None,
    ) -> "RuleStats":
        """Compute tp/fp/fn/tn from `rule`'s coverage over `data`
        (`positive_class` defaults to `rule.target`) and `length` from
        `rule.length()` -- the same `pos_mask`/covered-count pattern
        `RuleModel.coverage_space` and `Rule.order_by_precision`
        already use.

        A gain-style heuristic needing a *parent* rule's stats too (see
        `GainHeuristic`) just calls this again on the parent rule, under
        the same `example_mask` -- nothing here is cached or reused
        across calls, so that's always a clean, fresh recount, not
        something this method needs to do on a caller's behalf.

        Pass `example_mask` (a boolean array over `data`'s rows)
        to restrict every one of the four counts to just the masked-in
        examples -- a grow/prune split, a SeCo loop's "remaining"
        examples, etc. -- without constructing a smaller
        `DataRepresentation` for it. Works over any `DataRepresentation`:
        coverage dispatches to the representation's own `coverage`.
        `example_mask` may also be a weight vector, and `data` may carry
        row weights (`DataRepresentation.scope`); the counts are then
        sums of the effective weights.
        """
        if data.y is None:
            raise ValueError("data has no labels; RuleStats.from_rule needs data.y")
        pos_class = positive_class if positive_class is not None else rule.target
        if pos_class is None:
            raise ValueError("from_rule needs positive_class (or a Rule with a target)")
        cov = rule.covers_data_packed(data)
        pos_mask = data.y == pos_class
        example_mask, weights = data.scope(example_mask)
        if weights is not None:
            pw, nw = weights * pos_mask, weights * ~pos_mask
            tp, fp = float(pw[cov].sum()), float(nw[cov].sum())
            return cls(tp=tp, fp=fp, fn=float(pw.sum()) - tp, tn=float(nw.sum()) - fp,
                       length=rule.length())
        if example_mask is not None:
            tp = int(np.sum(cov & pos_mask & example_mask))
            fp = int(np.sum(cov & ~pos_mask & example_mask))
            fn = int(np.sum(~cov & pos_mask & example_mask))
            tn = int(np.sum(~cov & ~pos_mask & example_mask))
        else:
            tp = int(np.sum(cov & pos_mask))
            fp = int(np.sum(cov & ~pos_mask))
            fn = int(np.sum(~cov & pos_mask))
            tn = int(np.sum(~cov & ~pos_mask))
        return cls(tp=tp, fp=fp, fn=fn, tn=tn, length=rule.length())


def _rule_score_fn(
    by: Optional[Union["RuleHeuristic", Callable[[Rule], Any]]],
    data: Optional[DataRepresentation] = None,
) -> Callable[[Rule], Any]:
    """Resolve `by` to a per-rule scalar key (higher sorts first):

    - ``None``          -- `Laplace` scored against the rule's own
      *measured* `stats()` (the same default `pyrulearn.combiners.
      HeuristicMaxCombiner` uses for predict-time ranking -- ranking a
      ruleset for inspection and ranking it for prediction are the same
      operation, just at different times). Raises `ValueError` if a rule
      has no measured stats -- annotate it first (`pyrulearn.models.
      annotate_rules`, or a `fit()`/importer `data=` call, which already
      do). No `data` needed for this case -- it reads the cached stats.
    - a `RuleHeuristic` -- ``heuristic.score_rule(rule, data, positive_class=rule.target)``,
      computed *fresh* against `data` rather than from cached stats;
      needs `data` unless the heuristic's `needs_data` is `False` (only
      `MinimalLength`, scored from the rule alone). Must be a plain
      heuristic -- a `GainHeuristic` has no score without a parent rule.
    - a plain callable  -- used as the key directly

    `RuleHeuristic`/`GainHeuristic`/`Laplace` are imported here, locally,
    rather than at module level -- `pyrulearn.heuristics` imports from
    this module (for `RuleStats`), so a module-level import here would be
    a load-order cycle; by the time this function is actually *called*,
    every module involved has long finished loading, so the cycle never
    manifests here.
    """
    from .combiners import _heuristic_score
    from .heuristics import GainHeuristic, Laplace, RuleHeuristic

    if by is None:
        laplace = Laplace()
        return lambda r: _heuristic_score(r, laplace)
    if isinstance(by, RuleHeuristic):
        if isinstance(by, GainHeuristic):
            raise ValueError(
                f"cannot sort rules by {type(by).__name__}: a GainHeuristic scores a "
                "refinement against a parent rule, not a rule on its own"
            )
        if not by.needs_data:
            return lambda r: by.score(RuleStats(tp=0, fp=0, fn=0, tn=0, length=r.length()))
        if data is None:
            raise ValueError(f"sorting rules by {type(by).__name__} needs `data` to score them against")
        return lambda r: by.score_rule(r, data, positive_class=r.target)
    if callable(by):
        return by
    raise ValueError(f"`by` must be None, a RuleHeuristic, or a callable; got {by!r}")


def sort_rules(
    rules: Sequence[Rule],
    by: Optional[Union["RuleHeuristic", Callable[[Rule], Any]]] = None,
    data: Optional[DataRepresentation] = None,
    descending: bool = True,
) -> List[Rule]:
    """`rules` sorted by `by` (see `_rule_score_fn`) -- by descending
    `Laplace`-on-measured-stats when `by` is `None`, else by the given
    `RuleHeuristic` (e.g. `MinimalLength()` for shortest-first,
    `Precision()` with `data=`) or callable. Best first by default;
    stable (equal keys keep input order)."""
    return sorted(rules, key=_rule_score_fn(by, data), reverse=descending)


#: label sequence -> shared `(labels list, label->index dict)`; see `ConfusionMatrix.from_rule_counts`
_SHARED_LABELS: Dict[Tuple[Any, ...], Tuple[List[Any], Dict[Any, int]]] = {}


class ConfusionMatrix:
    """A genuine N-label confusion matrix: `counts[i, j]` is how many
    rows whose true label is `labels[i]` were predicted `labels[j]`.
    Generalizes `RuleStats` (a single designated-positive-class 2x2 view)
    to every label at once -- built once from a model's real predictions
    (`from_predictions`), then sliced per label via `rule_stats` on
    demand, as many times as wanted, for whichever labels are of
    interest.

    A `None` (abstained) prediction lands in a dedicated `ABSTAIN`
    pseudo-label column -- kept as its own bucket rather than silently
    folded into an existing label or dropped, so a model that predicts
    "label or abstain" is exactly representable, and a caller can decide
    for itself whether/how abstentions should count (e.g. `accuracy`
    below counts them as wrong; a different reading -- excluding them
    from the denominator entirely -- is a deliberate design point left
    open for later, not resolved by this class).
    """

    def __init__(self, labels: Sequence[Any], counts: np.ndarray):
        self.labels: List[Any] = list(labels)
        counts = np.asarray(counts)
        # row counts stay integers; sums of row weights stay floats
        self.counts: np.ndarray = counts.astype(float if counts.dtype.kind == "f" else np.int64)
        if self.counts.shape != (len(self.labels), len(self.labels)):
            raise ValueError(
                f"counts must be ({len(self.labels)}, {len(self.labels)}) for "
                f"{len(self.labels)} labels, got {self.counts.shape}"
            )
        self._index: Dict[Any, int] = {l: i for i, l in enumerate(self.labels)}

    @classmethod
    def from_predictions(
        cls, y_true: Sequence[Any], y_pred: Sequence[Any], labels: Optional[Sequence[Any]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> "ConfusionMatrix":
        """Build the matrix from parallel true/predicted arrays. `labels`
        fixes the label set (and its order) -- typically a model's own
        `RuleModel.labels` -- so every label the model *could* have
        predicted gets a row/column even if it happens to win zero of
        them on this particular data; a true or predicted value outside
        `labels` still gets folded in (added to the label set) rather
        than silently dropped, since a mismatch there usually means the
        caller's label set is stale, not that the data is wrong. `None`
        entries in `y_pred` (an abstained row) go to a trailing `ABSTAIN`
        pseudo-label column, added automatically iff at least one
        prediction actually abstained. With `weights` (one per row), each
        cell is a sum of row weights instead of a row count (floats).
        """
        y_true = list(y_true)
        y_pred = [ABSTAIN if p is None else p for p in y_pred]
        seen = set(y_true) | {p for p in y_pred if p is not ABSTAIN}
        if labels is None:
            all_labels = sorted(seen, key=_sortkey)
        else:
            all_labels = list(labels) + [l for l in sorted(seen, key=_sortkey) if l not in labels]
        if ABSTAIN in y_pred:
            all_labels = all_labels + [ABSTAIN]

        index = {l: i for i, l in enumerate(all_labels)}
        n = len(all_labels)
        if weights is None:
            counts = np.zeros((n, n), dtype=np.int64)
            for t, p in zip(y_true, y_pred):
                counts[index[t], index[p]] += 1
        else:
            counts = np.zeros((n, n), dtype=float)
            for t, p, w in zip(y_true, y_pred, weights):
                counts[index[t], index[p]] += w
        return cls(all_labels, counts)

    @classmethod
    def from_rule_counts(
        cls, target: Any, covered: Dict[Any, int], totals: Dict[Any, int],
    ) -> "ConfusionMatrix":
        """The matrix of a single rule that predicts `target` on the rows
        it covers and abstains elsewhere, built by arithmetic from counts
        instead of predicting over the data: `covered[c]` is how many of
        the rule's covered rows have true label `c`, `totals[c]` how many
        rows of the whole data do. Identical -- labels, order, counts,
        the trailing `ABSTAIN` column present iff some row is uncovered --
        to `from_predictions(y, rule_predictions, labels=[target])`, which
        is what `RuleModel.evaluate` produces for a `SingleRule` with no
        default prediction (checked by `tests/test_evaluation`-style
        equivalence tests); it just skips the coverage pass, for producers
        (a CAR miner, a decision-tree leaf) that already know the counts.
        """
        others = sorted((c for c in totals if c != target), key=_sortkey)
        labels: List[Any] = [target] + others
        n_rows = sum(totals.values())
        uncovered = n_rows - sum(covered.get(c, 0) for c in labels)
        if uncovered > 0:
            labels.append(ABSTAIN)
        # One shared (labels, index) pair per distinct label sequence: a pool of thousands of rules
        # has only a handful of them, and the per-rule copies were ~40% of a matrix's memory. Treated
        # as read-only everywhere (nothing in this package mutates `.labels`/`._index`).
        shared_labels, index = _SHARED_LABELS.setdefault(
            tuple(labels), (labels, {l: i for i, l in enumerate(labels)}))
        counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
        t = index[target]
        for c in [target] + others:
            counts[index[c], t] = covered.get(c, 0)
            if uncovered > 0:
                counts[index[c], index[ABSTAIN]] = totals.get(c, 0) - covered.get(c, 0)
        cm = cls.__new__(cls)
        cm.labels, cm._index, cm.counts = shared_labels, index, counts
        return cm

    def rule_stats(self, label: Any) -> RuleStats:
        """The one-vs-rest `RuleStats` view of this matrix with `label`
        as the designated positive class: `tp`/`fp` are rows predicted
        `label` (correctly/incorrectly), `fn`/`tn` are rows *not*
        predicted `label` (that should/shouldn't have been). Computed
        fresh each call -- a different `label` gives a genuinely
        different result (its own tp/fp are the complement's fn/tn),
        not a cached, retrievable pair."""
        i = self._index[label]
        num = self._num
        tp = num(self.counts[i, i])
        fp = num(self.counts[:, i].sum()) - tp
        fn = num(self.counts[i, :].sum()) - tp
        tn = num(self.counts.sum()) - tp - fp - fn
        return RuleStats(tp=tp, fp=fp, fn=fn, tn=tn)

    @property
    def _num(self):
        """`int` for a row-count matrix, `float` for a weighted one."""
        return float if self.counts.dtype.kind == "f" else int

    def predicted_as(self, label: Any) -> Dict[Any, int]:
        """The true-label breakdown of every row predicted `label`:
        `{true_label: count}` -- the `label` column of the matrix, read
        directly rather than rotated into a 2x2 view like `rule_stats`.

        For a `SingleRule`'s own `ConfusionMatrix` (it predicts only its
        own target, or abstains elsewhere -- see `pyrulearn.models.
        RuleModel.evaluate`), calling this with the rule's own target
        gives exactly a decision-tree leaf's classic per-class count
        breakdown, generalized to any rule -- what `pyrulearn.combiners.
        DistributionCombiner` reads: the true-label distribution among
        the rows this rule actually fired on.
        """
        j = self._index[label]
        num = self._num
        return {self.labels[i]: num(self.counts[i, j]) for i in range(len(self.labels))}

    @property
    def n_total(self) -> Union[int, float]:
        return self._num(self.counts.sum())

    @property
    def accuracy(self) -> float:
        """Fraction of rows correctly predicted -- an abstained row
        (the `ABSTAIN` column) always counts as incorrect here, never
        excluded from the denominator. Prefer `rule_stats(label)` for a
        per-label precision/recall-style reading."""
        total = self.n_total
        return float(np.trace(self.counts)) / total if total else 0.0

    def __repr__(self) -> str:
        return f"ConfusionMatrix(labels={self.labels!r}, n={self.n_total})"


class ModelStats:
    """A model's measured performance and complexity on some data --
    what `pyrulearn.models.RuleModel.evaluate` returns, and what a rule's
    frozen training stats (`SingleRule.stats()`) are. `confusion` (a
    `ConfusionMatrix` from the model's `predict(data)` vs `data.y`) is
    `None` when `data.y` wasn't available to score against -- `n_rows`/
    `n_rules`/`n_conditions` are still meaningful then.

    `n_rules`/`n_conditions` are the model's own rule-count/condition-
    count at measurement time -- a `CompositeModel`'s pooled `.rules` is
    already the recursive view, so this doesn't walk the tree twice.

    No per-rule breakdown here (deliberately -- that's not a "constituent
    model" in the recursive sense; call `.stats()` on an actual member/
    `SingleRule` for its own numbers instead) and no separate "unique
    coverage" count (droppable -- `RuleModel._unique_mask`/`coverage_
    matrix` are still there to recompute it directly if ever wanted).
    """

    __slots__ = ("n_rows", "confusion", "n_rules", "n_conditions")

    def __init__(self, n_rows: int, confusion: Optional[ConfusionMatrix],
                n_rules: int, n_conditions: int):
        self.n_rows = n_rows
        self.confusion = confusion
        self.n_rules = n_rules
        self.n_conditions = n_conditions

    def __repr__(self) -> str:
        acc = f", accuracy={format_accuracy(self.confusion.accuracy)}" if self.confusion is not None else ""
        return (f"ModelStats(n_rows={self.n_rows}, n_rules={self.n_rules}, "
                f"n_conditions={self.n_conditions}{acc})")


def _apply_aspect(ax, x_extent: float, y_extent: float, aspect: str, max_aspect_ratio: float) -> None:
    """Set `ax`'s physical box shape so it reflects the true x:y extent
    ratio of whatever's actually plotted -- `x_extent`/`y_extent` are
    `n_neg`/`n_pos` for a raw coverage space, or `1.0`/`1.0` for a
    normalized ROC space (always a perfect square, regardless of the
    underlying class balance -- that's what "ROC space" conventionally
    means). `aspect="square"` leaves the default (roughly square) box
    alone; `"rectangular"` uses the true ratio unclamped; `"auto"` (the
    default) uses the true ratio but clamped to `[1/max_aspect_ratio,
    max_aspect_ratio]`, so a very skewed dataset still gets a readable
    box rather than a sliver."""
    if aspect == "square":
        return
    if aspect not in ("auto", "rectangular"):
        raise ValueError(f"Unknown aspect {aspect!r}; choose from 'auto', 'rectangular', 'square'")
    ratio = (y_extent / x_extent) if x_extent else 1.0
    if aspect == "auto":
        ratio = min(max(ratio, 1.0 / max_aspect_ratio), max_aspect_ratio)
    ax.set_box_aspect(ratio)


def _plot_arrow_path(ax, pts: np.ndarray, color: str = "C0", zorder: int = 2) -> None:
    """Draw `pts` (an (n, 2) array) as points connected by directional
    arrows in sequence -- for plots where the order between consecutive
    points matters (`RuleList.coverage_path`, a single `Rule`'s
    refinement path), as opposed to a plain line/scatter."""
    for i in range(len(pts) - 1):
        ax.annotate(
            "", xy=tuple(pts[i + 1]), xytext=tuple(pts[i]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, shrinkA=0, shrinkB=0),
            zorder=zorder,
        )
    ax.scatter(pts[:, 0], pts[:, 1], color=color, zorder=zorder + 1)


def _upper_hull(points: np.ndarray) -> np.ndarray:
    """The upper-left boundary of the convex hull, in ascending-x order:
    for each x, the maximum achievable y. This is the "ROC convex hull"
    proper -- what `coverage_space_auc` integrates, and what
    `coverage_space_plot`'s `show_convex_hull` draws (extended down to
    the x-axis to show the AUC area, see there). A small self-contained
    monotone-chain implementation rather than a
    ``scipy.spatial.ConvexHull`` dependency, since this module otherwise
    only needs matplotlib/networkx.
    """
    pts = list(set(map(tuple, points)))
    if len(pts) <= 2:
        return np.array(sorted(pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    chain = []
    for p in sorted(pts, reverse=True):
        while len(chain) >= 2 and cross(chain[-2], chain[-1], p) <= 0:
            chain.pop()
        chain.append(p)
    return np.array(list(reversed(chain)))


def _same_target_points(ruleset: RuleModel, data: DataRepresentation, positive_class: Any) -> np.ndarray:
    """Raw (negatives, positives) coverage points for just the rules in
    `ruleset` whose own `target == positive_class` -- the subset the
    convex hull / AUC are computed over, since both are only meaningful
    for a single target class at a time (see `coverage_space_auc`)."""
    same_target = [r for r in ruleset.rules if r.target == positive_class]
    if not same_target:
        return np.zeros((0, 2), dtype=int)
    return FlatRuleSet(same_target).coverage_space(data, positive_class)


def coverage_space_auc(ruleset: RuleModel, data: DataRepresentation, positive_class: Any) -> float:
    """Area under the convex-hull ROC curve, restricted to the rules in
    `ruleset` whose own `target == positive_class` (other rules are
    ignored -- for a multi-class `RuleSet`, call this once per class,
    e.g. after `ruleset.filter(cls)`, rather than expecting one number to
    cover every class at once). Points are normalized to [0, 1] x [0, 1]
    (FPR, TPR) before integrating the upper hull boundary (`_upper_hull`)
    via the trapezoidal rule. A class with no matching rules gets the
    trivial ``(0, 0)``-``(N, P)`` hull, i.e. AUC 0.5 (no better than
    random).
    """
    if data.y is None:
        raise ValueError("data has no labels; coverage_space_auc needs data.y")
    y = np.asarray(data.y)
    n_neg = int(np.sum(y != positive_class))
    n_pos = int(np.sum(y == positive_class))
    if n_neg == 0 or n_pos == 0:
        raise ValueError("coverage_space_auc needs both classes present in data.y")

    pts = _same_target_points(ruleset, data, positive_class)
    anchors = np.array([[0, 0], [n_neg, n_pos]])
    hull = _upper_hull(np.vstack([pts, anchors]) if len(pts) else anchors)

    fpr = hull[:, 0] / n_neg
    tpr = hull[:, 1] / n_pos
    return float(np.trapezoid(tpr, fpr))


def build_refinement_graph(ruleset: RuleModel):
    """Build a directed graph (networkx.DiGraph) where an edge r1 -> r2
    means r2 specializes r1 by exactly one additional literal (i.e. r2's
    literal set is r1's plus one more feature). This is the natural
    "refinement step" edge for top-down rule learners (e.g. sequential
    covering / separate-and-conquer).

    Requires ``networkx``.
    """
    import networkx as nx

    g = nx.DiGraph()
    for i, r in enumerate(ruleset.rules):
        g.add_node(i, rule=r)

    for i, r1 in enumerate(ruleset.rules):
        lits1 = set(r1.pos)
        for j, r2 in enumerate(ruleset.rules):
            if i == j:
                continue
            lits2 = set(r2.pos)
            if lits1 < lits2 and len(lits2) == len(lits1) + 1:
                g.add_edge(i, j)
    return g


def isometrics_cmap():
    """The default colormap of isometrics (`CoverageSpace.contour`): from
    red for low scores over orange and yellow to green for high ones, as in
    a heatmap, but with a darker yellow than matplotlib's ``RdYlGn``, so
    that the middle lines stay visible on white."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "isometrics", ["#c62828", "#ef6c00", "#c9a400", "#7cb342", "#1b5e20"])


class CoverageSpace:
    """A coverage-space plot: x = negatives covered, y = positives
    covered -- or, `normalized=True`, the classic **ROC space**: FPR/TPR
    in `[0, 1] x [0, 1]`. Owns the space's dimensions (`n_neg`, `n_pos`)
    and the matplotlib `Axes`; other things draw *into* it as layers --
    `plot_ruleset`/`plot_rule_refinement` here, or
    `RuleHeuristic.plot_isometrics` (in `pyrulearn.heuristics`) -- so
    several can share one plot, e.g. to compare a heuristic's isometrics
    against where real rules actually land.

    `n_pos`/`n_neg` default to `100`/`160` (a fixed, nicely-proportioned
    pair, smaller value on the Pos axis) when *both* are omitted -- a
    dataset-agnostic space for visualizing a heuristic's isometrics in
    the abstract; pass real counts, or use `from_data`, to
    size the space to actual data. A normalized (`normalized=True`) space
    is always displayed as a perfect square (`[0, 1] x [0, 1]`, the
    conventional ROC-space shape) regardless of `n_pos`/`n_neg` or
    `aspect` -- those still affect how the heuristic itself is *scored*
    internally (see `RuleHeuristic.plot_isometrics`), just not the box
    shape.

    `show_labels` controls whether *concrete numbers* -- tick labels
    beyond the two endpoints, and isometric value labels (`ax.clabel`,
    in `RuleHeuristic.plot_isometrics`) -- are drawn at all. `None` (the
    default) means "yes, if `n_pos`/`n_neg` were actually given" and "no"
    otherwise, since the default space's ``100``/``160`` are arbitrary
    placeholders, not real counts, and printing them as if they meant
    something would be misleading. Pass `True`/`False` explicitly to
    override that default either way. Axis text (`"negatives covered"`/
    `"positives covered"`, or `"FPR"`/`"TPR"` when normalized) and the
    two endpoint ticks are always shown either way -- only their *labels*
    change: `"0"`/`"N"` and `"0"`/`"P"` (symbolic, since the real counts
    aren't meaningful) when `show_labels` is `False` and the space isn't
    normalized, literal `"0"`/`"1"` when it is (that endpoint is always
    true, regardless of the placeholder counts behind it), or the real
    numbers (with matplotlib's normal, denser default ticks) when
    `show_labels` is `True`. A light gray minor-tick grid is always drawn
    for visual reference, styled like the random-guess diagonal. The
    plot's `title` (if given) is unaffected either way.
    """

    #: Default dataset-agnostic space size (see `__init__`), used when
    #: both `n_pos`/`n_neg` are omitted.
    DEFAULT_N_POS = 100
    DEFAULT_N_NEG = 160

    def __init__(
        self,
        n_pos: Optional[int] = None,
        n_neg: Optional[int] = None,
        normalized: bool = False,
        ax=None,
        aspect: str = "auto",
        max_aspect_ratio: float = 3.0,
        title: Optional[str] = None,
        show_labels: Optional[bool] = None,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator

        dims_given = n_pos is not None or n_neg is not None
        if n_pos is None and n_neg is None:
            n_pos, n_neg = self.DEFAULT_N_POS, self.DEFAULT_N_NEG
        elif n_pos is None or n_neg is None:
            raise ValueError("pass both n_pos and n_neg, or neither (for the default space)")
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.normalized = normalized
        self.show_labels = dims_given if show_labels is None else show_labels

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        self.ax = ax

        x_extent = 1.0 if normalized else n_neg
        y_extent = 1.0 if normalized else n_pos
        pad_x, pad_y = (0.02, 0.02) if normalized else (0.5, 0.5)
        ax.plot([0, x_extent], [0, y_extent], linestyle="--", color="lightgray", zorder=0,
                label="random-guess line")
        _apply_aspect(ax, x_extent, y_extent, aspect, max_aspect_ratio)
        ax.set_xlim(-pad_x, x_extent + pad_x)
        ax.set_ylim(-pad_y, y_extent + pad_y)
        if self.show_labels:
            ax.set_xlabel("FPR" if normalized else f"negatives covered (of {n_neg})")
            ax.set_ylabel("TPR" if normalized else f"positives covered (of {n_pos})")
            x_minor, y_minor = 5, 5
        else:
            # no concrete (arbitrary/placeholder) numbers, but still
            # meaningful axis text and endpoint ticks -- symbolic "N"/"P"
            # for a raw space (the real totals aren't real), or literal
            # "1" for a normalized space (that endpoint is always true,
            # regardless of the placeholder counts behind it)
            ax.set_xlabel("FPR" if normalized else "negatives covered")
            ax.set_ylabel("TPR" if normalized else "positives covered")
            x_end = "1" if normalized else "N"
            y_end = "1" if normalized else "P"
            ax.set_xticks([0, x_extent])
            ax.set_xticklabels(["0", x_end])
            ax.set_yticks([0, y_extent])
            ax.set_yticklabels(["0", y_end])
            # exactly one major interval per axis here, so AutoMinorLocator(n)
            # directly controls the total grid-cell count on that axis --
            # scale by the true x:y extent ratio so cells come out square
            # (same physical size on both axes) instead of "5 either way"
            # regardless of a rectangular space's actual shape
            shorter = min(x_extent, y_extent)
            x_minor = max(1, round(5 * x_extent / shorter))
            y_minor = max(1, round(5 * y_extent / shorter))
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
        ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))
        ax.grid(True, which="both", color="lightgray", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_title(title or ("ROC space" if normalized else "Coverage space"))

    @classmethod
    def from_data(
        cls, data: DataRepresentation, positive_class: Any, **kwargs
    ) -> "CoverageSpace":
        """Build a `CoverageSpace` sized to `data`'s actual
        (negatives, positives) counts for `positive_class`. `**kwargs`
        are passed through to `__init__` (`normalized=`, `ax=`, ...)."""
        if data.y is None:
            raise ValueError("data has no labels; from_data needs data.y")
        y = np.asarray(data.y)
        n_neg = int(np.sum(y != positive_class))
        n_pos = int(np.sum(y == positive_class))
        return cls(n_pos=n_pos, n_neg=n_neg, **kwargs)

    # -- coordinate mapping ------------------------------------------------

    def _display(self, neg, pos):
        """Raw (negatives, positives) coordinates -> whatever's actually
        plotted: unchanged for a raw coverage space, or normalized to
        (FPR, TPR) if `self.normalized`."""
        if not self.normalized:
            return neg, pos
        return np.asarray(neg) / self.n_neg, np.asarray(pos) / self.n_pos

    # -- generic layers ------------------------------------------------------

    def scatter(self, points: np.ndarray, **kwargs):
        """Scatter an `(n, 2)` array of raw (negatives, positives) points."""
        x, y = self._display(points[:, 0], points[:, 1])
        return self.ax.scatter(x, y, **kwargs)

    def arrow_path(self, points: np.ndarray, color: str = "C0", zorder: int = 2) -> None:
        """Raw (negatives, positives) `points` connected by directional
        arrows, in order (see `_plot_arrow_path`)."""
        x, y = self._display(points[:, 0], points[:, 1])
        _plot_arrow_path(self.ax, np.column_stack([x, y]), color=color, zorder=zorder)

    def contour(
        self,
        score_fn,
        levels=10,
        resolution: int = 200,
        filled: bool = False,
        cmap: Any = None,
        shade: float = 0.3,
        **kwargs,
    ):
        """Draw isometrics of `score_fn(neg, pos) -> float` -- evaluated
        over a grid spanning this space's full `[0, n_neg] x [0, n_pos]`
        extent -- as contour lines (or, `filled=True`, filled bands).
        `levels` is passed straight to matplotlib's `contour`/`contourf`
        (an int for that many automatically chosen levels, or explicit
        level values). `cmap` defaults to `isometrics_cmap()`, from red
        (low scores) to green (high ones), as in a heatmap. Lines are
        drawn over bands of the same colors with opacity `shade` (0 for
        bare lines; skipped for fewer than two levels, which bound no
        band). Pass `colors=` (a single color, for a flat-colored
        set of isometrics -- handy when overlaying more than one
        heuristic's isometrics on the same space) instead of `cmap` to
        tell them apart by color rather than by shape alone; `cmap` and
        the shading are dropped automatically when `colors` is given,
        since matplotlib rejects both together. Returns the
        `QuadContourSet` matplotlib hands back, in case the caller wants
        to `ax.clabel(...)` it directly.
        """
        neg = np.linspace(0, self.n_neg, resolution)
        pos = np.linspace(0, self.n_pos, resolution)
        NEG, POS = np.meshgrid(neg, pos)
        Z = np.vectorize(score_fn, otypes=[float])(NEG, POS)
        x, y = self._display(NEG, POS)
        if "colors" in kwargs:
            cmap = None
        elif cmap is None:
            cmap = isometrics_cmap()
        if filled:
            return self.ax.contourf(x, y, Z, levels=levels, cmap=cmap, **kwargs)
        if cmap is not None and shade > 0 and (np.ndim(levels) == 0 or np.size(levels) >= 2):
            # under the grid and everything drawn on top (rules, paths, ...)
            bands = self.ax.contourf(x, y, Z, levels=levels, cmap=cmap, alpha=shade, zorder=0.5)
            levels = bands.levels
        return self.ax.contour(x, y, Z, levels=levels, cmap=cmap, **kwargs)

    # -- rule-learning-specific layers ---------------------------------------

    def plot_ruleset(
        self,
        ruleset: RuleModel,
        data: DataRepresentation,
        positive_class: Any,
        show_refinements: bool = True,
        annotate: bool = False,
        show_convex_hull: bool = False,
    ):
        """Draw `ruleset`'s coverage into this space -- see
        `coverage_space_plot` (the standalone-figure wrapper around this
        method) for the full semantics of every argument. Returns
        `self.ax`.
        """
        if isinstance(ruleset, RuleList):
            pts = ruleset.coverage_path(data, positive_class)
            self.arrow_path(pts, color="C0", zorder=2)
            if annotate:
                for i, (nx, ny) in enumerate(pts):
                    dx, dy = self._display(np.array([nx]), np.array([ny]))
                    self.ax.annotate(str(i), (dx[0], dy[0]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        else:
            pts = ruleset.coverage_space(data, positive_class)
            if show_refinements:
                g = build_refinement_graph(ruleset)
                for i, j in g.edges():
                    x, y = self._display(np.array([pts[i, 0], pts[j, 0]]), np.array([pts[i, 1], pts[j, 1]]))
                    self.ax.plot(x, y, color="gray", linewidth=0.8, alpha=0.6, zorder=1)
            if show_convex_hull:
                hull_pts = _same_target_points(ruleset, data, positive_class)
                anchors = np.array([[0, 0], [self.n_neg, self.n_pos]])
                upper = _upper_hull(np.vstack([hull_pts, anchors]) if len(hull_pts) else anchors)
                # AUC-area boundary: (0,0) -> (n_neg,0) -> (n_neg,n_pos) -> upper
                # hull back down to (0,0) -- not a bare (0,0)-(N,P) closing edge,
                # which would sit right on top of the random-guess diagonal
                boundary = np.vstack([[0, 0], [self.n_neg, 0], upper[::-1]])
                auc = coverage_space_auc(ruleset, data, positive_class)
                bx, by = self._display(boundary[:, 0], boundary[:, 1])
                self.ax.plot(bx, by, color="C2", linewidth=1.2, zorder=1, label=f"convex hull (AUC={auc:.3f})")
            self.scatter(pts, zorder=2)
            if annotate:
                for i in range(len(ruleset.rules)):
                    dx, dy = self._display(np.array([pts[i, 0]]), np.array([pts[i, 1]]))
                    self.ax.annotate(str(i), (dx[0], dy[0]), fontsize=8, xytext=(3, 3), textcoords="offset points")

        if self.show_labels:
            xlabel = "FPR" if self.normalized else f"not {positive_class!r} covered (of {self.n_neg})"
            ylabel = "TPR" if self.normalized else f"{positive_class!r} covered (of {self.n_pos})"
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
        self.ax.legend(loc="lower right", fontsize=8)
        return self.ax

    def plot_rule_refinement(
        self,
        rule: Rule,
        data: DataRepresentation,
        positive_class: Optional[Any] = None,
        annotate: bool = True,
        path: Optional[np.ndarray] = None,
    ):
        """Draw `rule`'s own specialization path into this space -- see
        `rule_refinement_plot` (the standalone-figure wrapper around
        this method) for the full semantics. `path`, if already
        computed (e.g. by that wrapper, to size the space beforehand),
        is reused as-is instead of recomputing it. Returns `self.ax`.
        """
        positive_class = _resolve_positive_class(rule, positive_class)
        if path is None:
            path = rule_refinement_path(rule, data, positive_class)
        conds = rule.conditions if rule.ordered else rule.canonical_conditions()

        self.arrow_path(path, color="C1", zorder=2)
        if annotate:
            for k, (nx, ny) in enumerate(path):
                label = "∅" if k == 0 else Rule([conds[k - 1]], dataspec=data.spec).to_string("conditions")
                dx, dy = self._display(np.array([nx]), np.array([ny]))
                # the start point sits in the top-right corner, where a
                # up-right offset would run into the title -- tuck it inside
                offset, ha = ((-4, -12), "right") if k == 0 else ((3, 3), "left")
                self.ax.annotate(label, (dx[0], dy[0]), fontsize=8, xytext=offset, textcoords="offset points", ha=ha)

        if self.show_labels:
            xlabel = "FPR" if self.normalized else f"not {positive_class!r} covered (of {self.n_neg})"
            ylabel = "TPR" if self.normalized else f"{positive_class!r} covered (of {self.n_pos})"
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
        self.ax.legend(loc="lower right", fontsize=8)
        return self.ax


def coverage_space_plot(
    ruleset: RuleModel,
    data: DataRepresentation,
    positive_class: Any,
    show_refinements: bool = True,
    ax=None,
    annotate: bool = False,
    aspect: str = "auto",
    max_aspect_ratio: float = 3.0,
    show_convex_hull: bool = False,
):
    """Plot a rule classifier in coverage space: x = negatives covered,
    y = positives covered. A standalone-figure convenience wrapper
    around `CoverageSpace.plot_ruleset` -- use `CoverageSpace` directly
    to layer this on top of (or under) other plots, e.g. a
    `RuleHeuristic`'s isometrics.

    For a `RuleList`, this plots its `coverage_path` instead of a
    scatter: points connected by directional **arrows**, in order,
    starting at ``(0, 0)``, one step per rule in list order, using each
    rule's *unique* (fired) coverage. `show_refinements`/`show_convex_hull`
    are ignored in this mode (the arrows already show the sequential
    structure); `annotate` still numbers each step if set. For any other
    `RuleModel` (e.g. a `FlatRuleSet`), each point is that rule's own raw
    coverage in isolation, as an unordered scatter (over *all* rules,
    regardless of target), optionally with `build_refinement_graph` edges
    overlaid and points numbered via `annotate`. `show_convex_hull=True`
    additionally draws the boundary of the area under the ROC-convex-hull
    curve -- ``(0, 0)`` to ``(N, 0)`` to ``(N, P)`` and back to ``(0, 0)``
    via the hull itself -- restricted to just the rules whose own
    `target == positive_class` (see `coverage_space_auc`, whose value is
    shown in the boundary's legend label, and is exactly the enclosed
    area normalized by ``N * P``): the hull/AUC are only meaningful for a
    single target class at a time, so a mixed-target `RuleSet` needs one
    call per class, not one hull mixing them.

    `aspect` controls the plot's physical box shape -- `"square"` (old
    behavior), `"rectangular"` (width:height = negatives:positives,
    unclamped), or `"auto"` (the default: same, but clamped to
    `[1/max_aspect_ratio, max_aspect_ratio]` so a very skewed dataset
    doesn't produce an unreadable sliver).

    Returns the matplotlib ``Axes`` used.
    """
    space = CoverageSpace.from_data(
        data, positive_class, ax=ax, aspect=aspect, max_aspect_ratio=max_aspect_ratio,
        title=f"Coverage space (target={positive_class!r})",
    )
    return space.plot_ruleset(
        ruleset, data, positive_class,
        show_refinements=show_refinements, annotate=annotate, show_convex_hull=show_convex_hull,
    )


def _resolve_positive_class(rule: Rule, positive_class: Optional[Any]) -> Any:
    """`positive_class` if given, else `rule.target` -- the "head of the
    rule" default requested for single-rule plots: a rule's own true
    positives are examples of the class *it* predicts, not some
    externally-fixed class. Raises if neither is available."""
    if positive_class is None:
        positive_class = rule.target
    if positive_class is None:
        raise ValueError("positive_class must be given for a rule with no target")
    # unwrap numpy scalars (e.g. np.str_) so axis labels show 'neg', not np.str_('neg')
    return positive_class.item() if isinstance(positive_class, np.generic) else positive_class


def rule_refinement_path(rule: Rule, data: DataRepresentation, positive_class: Optional[Any] = None) -> np.ndarray:
    """(negatives, positives) coverage of `rule` after each of its
    conditions is added, one at a time, in `rule`'s own condition order
    (its stored order if `rule.ordered`, else canonical ascending-feature
    order -- see `Rule.order_by_precision` for computing a more
    meaningful order first). Shape ``(len(rule.conditions) + 1, 2)``:
    point 0 is the *empty* rule (no conditions, covers everyone --
    ``(N, P)``); the last point is `rule`'s own full coverage.
    Monotonically non-increasing in both coordinates, since a conjunction
    can only stay the same size or shrink as literals are added.

    `positive_class` defaults to `rule.target` -- for a single rule,
    "positive" naturally means the class *it* predicts (its "head"), so
    an example only counts toward "positives covered" if it's actually
    of that class.
    """
    positive_class = _resolve_positive_class(rule, positive_class)
    if data.y is None:
        raise ValueError("data has no labels; rule_refinement_path needs data.y")
    spec = data.spec
    conds = rule.conditions if rule.ordered else rule.canonical_conditions()
    pos_mask = data.y == positive_class
    pts = np.zeros((len(conds) + 1, 2), dtype=int)
    for k in range(len(conds) + 1):
        prefix = Rule(conds[:k], target=rule.target, dataspec=spec, n_features=spec.n_features)
        cov = prefix.covers_data_packed(data)
        pts[k] = (int(np.sum(cov & ~pos_mask)), int(np.sum(cov & pos_mask)))
    return pts


def rule_refinement_plot(
    rule: Rule,
    data: DataRepresentation,
    positive_class: Optional[Any] = None,
    ax=None,
    annotate: bool = True,
    aspect: str = "auto",
    max_aspect_ratio: float = 3.0,
):
    """Plot one `Rule`'s own specialization path in coverage space (see
    `rule_refinement_path`): starts at ``(N, P)`` and steps toward the
    rule's own coverage as each condition is added, in the rule's own
    order. `annotate` (default `True`) labels each step with the
    condition that was just added (the start point is labeled ``"∅"``).
    `positive_class` defaults to `rule.target` (see `rule_refinement_path`).

    Returns the matplotlib ``Axes`` used.
    """
    positive_class = _resolve_positive_class(rule, positive_class)
    path = rule_refinement_path(rule, data, positive_class)
    n_neg, n_pos = (int(v) for v in path[0])
    space = CoverageSpace(
        n_pos=n_pos, n_neg=n_neg, ax=ax, aspect=aspect, max_aspect_ratio=max_aspect_ratio,
        title=f"Refinement path for {rule.to_string('logic')}",
    )
    return space.plot_rule_refinement(rule, data, positive_class=positive_class, annotate=annotate, path=path)


def rule_length_distribution(ruleset: RuleModel) -> np.ndarray:
    """Array of rule lengths (number of literals), one entry per rule --
    quick input to a histogram of rule complexity."""
    return np.array([r.length() for r in ruleset.rules])


def summarize(ruleset: RuleModel, data: Optional[DataRepresentation] = None) -> dict:
    """Small dict of aggregate stats: rule count, mean/median length,
    per-target rule counts, and (if `data` is given) each rule's mean raw
    coverage over it, computed fresh via `coverage_matrix` -- no prior
    annotation call needed."""
    lengths = rule_length_distribution(ruleset)
    targets: dict = {}
    for r in ruleset.rules:
        targets[r.target] = targets.get(r.target, 0) + 1
    out = {
        "n_rules": len(ruleset.rules),
        "mean_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        "median_length": float(np.median(lengths)) if len(lengths) else 0.0,
        "rules_per_target": targets,
    }
    if data is not None and ruleset.rules:
        out["mean_n_covered"] = float(ruleset.coverage_matrix(data).sum(axis=1).mean())
    return out
