"""
pyrulearn.heuristics
=======================

`RuleHeuristic`: pluggable rule-evaluation heuristics, scored from
`RuleStats` (defined in `pyrulearn.evaluation`, re-exported here for
every existing importer) -- the confusion-matrix quartet `tp`/`fp`/`fn`/
`tn` a rule produces against one positive class (covering ⇒ "predicts
positive"), plus optional `length`. `length` isn't coverage information
as such -- it's heuristic-search context -- which is why it lives on
`RuleStats`, not as part of `Rule` itself: a bare `Rule` has no inherent
notion of "its own refinement history", only a search process does.

**Static and adaptive heuristics.** A heuristic scores a rule within a
*frame*, a coverage space: by default the whole data (`n_pos`, `n_neg`),
in which every rule is compared with the universal rule. A heuristic
that is `relative_to_parent` takes the rule a refinement was made from
(its parent) as the frame instead: the parent's covered examples are the
whole data, its class distribution the prior, and so the parent is the
universal rule of the frame. Such a heuristic takes the parent's
statistics as an optional second argument of `score`/`batch_score`;
without it, it scores in the whole data, like a static heuristic.
`Adaptive(h)` makes any heuristic adaptive (adaptive WRA, a local
`MEstimate`; up to 0.7.1, it was called `Relative`, which still works
as a deprecated alias); `FoilGain` is adaptive by definition. Scores relative to
different parents have no common scale: a search compares them only
among the refinements of one rule, and picks the rule it returns by a
separate, static `selection` (or, by default, the last rule it
reached, `LastRule`).

Three wrappers build heuristics from others: `Adaptive(h)` (the frame),
`Inverted(h)` (judge a rule by the examples it excludes, after Stecher,
Janssen & Fürnkranz; their inverted heuristics are
`Adaptive(Inverted(h))`), and `WeightedGain(weight, base)` (``weight(r)
* (base(r) - base(r0))``, the gain over the universal rule ``r0`` of
the frame -- over the parent inside `Adaptive`). `WRAcc` and `FoilGain`
are fast shortcuts for two weighted gains; `Gain(h)` is the unweighted
one.

Used to rank/select candidate rules or refinements during search --
`Rule.order_by_precision`'s greedy criterion is one hardcoded instance
of this general idea -- or, later, as isometric overlays on
`pyrulearn.evaluation.coverage_space_plot`.

Every heuristic follows the same convention: **higher score = more
preferred**, so they're interchangeable as a ranking key regardless of
which one is plugged in.

Background: Fürnkranz & Flach, "ROC 'n' Rule Learning -- Towards a
Better Understanding of Covering Algorithms" (Machine Learning, 2005)
analyze these heuristics via their *isometrics* in coverage space (the
(fp, tp) plane) -- curves along which a heuristic scores constantly (see
`RuleHeuristic.plot_isometrics` to actually draw them). Their shape sorts
the heuristics into three groups -- pencil, parallel and curved. (In
Fürnkranz & Flach's terminology the isometrics of the first two are
*linear*, i.e. straight lines, and those of the third *non-linear*.)

- **Pencil heuristics** -- isometrics are a family of straight lines
  through one common pivot point (`Precision`: pivot at the origin;
  `Laplace`: pivot at (-1, -1); `MEstimate`: pivot slides along a line
  as its `m` parameter varies, with `Precision` as the m=0 limit;
  `GeneralizedMEstimate`: pivot can be anywhere on that same line, since
  its `cost` parameter frees the pivot from `MEstimate`'s dataset-tied
  prior; `GHeuristic`: pivot slides along the fp-axis, at `(-g, 0)`, as its `g`
  parameter varies; `FBeta`: pivot also slides along the fp-axis, at
  `(-beta**2*n_pos, 0)` -- the same sub-family as `GHeuristic`, but tied
  to the dataset's actual class balance rather than a free constant;
  `beta -> 0` recovers `Precision`'s pivot at the origin, `beta ->
  infinity` pushes it to `-infinity`, flattening the isometrics toward
  `Recall`'s horizontal lines). `Entropy` belongs here too -- same pivot as
  `Precision` (the origin), since it also depends only on the ratio
  `p = tp/(tp+fp)`, constant along any ray from the origin -- but
  *unlike* `Precision`, each score value corresponds to a symmetric
  *pair* of rays (`p` and `1-p` give the same entropy, since `H` is
  symmetric around `p=0.5`): plain entropy alone can't tell a rule
  that's 90% positive from one that's 90% *negative* apart, only how
  skewed the split is either way -- part of why CN2 assigns the covered
  majority class as the rule's actual prediction separately, rather
  than reading it off the entropy score. Because their isometrics are
  not parallel, these can prefer points *inside* coverage space's convex
  hull, not just on it -- part of why plain `Precision` is prone to
  overfitting: it can't distinguish a lucky single positive from a
  robust, larger-coverage rule.
- **Parallel-line heuristics** -- isometrics are parallel lines, not
  radiating from a point (`WRAcc`, `Accuracy`, `Support`, `Coverage`,
  `LinearCost`, `CoverageDifference`, and `YoudenJ`/`LinearCostRates` --
  the same family again, just with tp/fp each rescaled by their own
  class total instead of used raw). `CoveredPositives` (horizontal
  isometrics) and `CoveredNegatives` (vertical) are the two degenerate
  limits of this family -- zero weight on fp, or on tp, respectively
  (`UncoveredPositives`/`UncoveredNegatives` sit in the same two
  families too, just affine-shifted -- see `CoveredPositives`'s
  docstring for the full four-way "covered/uncovered x
  positives/negatives" quadrant naming);
  `Recall` is `CoveredPositives`'s own rate-normalized twin (`tp/n_pos`
  vs. raw `tp`), the same relationship `Support` has to `Coverage`.
  Since their isometrics are parallel, a
  rule is optimal for *some* such heuristic if and only if it lies on
  the convex hull.
- **Curved-isometric heuristics** -- `Correlation` (hyperbolic
  isometrics) and `LikelihoodRatio` (isometrics that bow away from the
  origin, since -- unlike the pencil group -- it depends on absolute
  covered counts, not just their ratio) belong to neither group above.
  `FoilGain` (with a fixed parent, see `plot_isometrics`'s `parent=`)
  has a straight zero isometric through the origin and the parent, and
  curves on either side of it: its `tp *` factor makes a low-tp point
  score near 0 whatever its precision.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from .evaluation import CoverageSpace

from .evaluation import RuleStats
from .data import DataRepresentation
from .rule import Rule

#: what RuleHeuristic.score returns -- a plain float for an ordinary
#: heuristic, or a tuple for a LEF (Michalski's Lexicographic Evaluation
#: Functional -- see the LEF class): Python tuples already compare
#: lexicographically, so every place a score gets compared (`>`,
#: `sorted(key=...)`) already does the right thing for either shape, no
#: special-casing needed.
Score = Union[float, Tuple[float, ...]]


def _counts(stats: RuleStats):
    """`stats`' tp, fp, fn, tn and length as float arrays broadcast to one
    shape -- what the built-in `batch_score`s compute on. Float, not the
    counts' own int dtype: the scalar `score`s multiply Python ints, which
    never overflow, but e.g. `Correlation`'s four-way product overflows
    int64 at around a million rows."""
    return np.broadcast_arrays(*(
        np.asarray(v, dtype=float) for v in (stats.tp, stats.fp, stats.fn, stats.tn, stats.length)
    ))


def _optimistic(stats: RuleStats) -> RuleStats:
    """`stats` projected to ``(tp, 0)``: every positive kept, every
    negative shed, with `fn`/`tn` shifted to hold the class totals."""
    fp = 0 if np.ndim(stats.fp) == 0 else np.zeros_like(stats.fp)
    return RuleStats(tp=stats.tp, fp=fp, fn=stats.fn, tn=stats.fp + stats.tn, length=stats.length)


def _div(num, den, default: float = 0.0):
    """``num / den`` where ``den > 0``, `default` elsewhere -- elementwise,
    without the divide-by-zero warnings `np.where` would raise by
    computing both branches."""
    num, den = np.broadcast_arrays(num, den)
    out = np.full(num.shape, default, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _loop(score_one, stats: RuleStats):
    """`batch_score`'s default: `score_one` (a scalar `score`) applied to
    every element of `stats`' arrays, as plain Python numbers. A tuple
    score (`LEF`-shaped) comes back as one row per element."""
    arrs = np.broadcast_arrays(*(np.asarray(v) for v in (stats.tp, stats.fp, stats.fn, stats.tn, stats.length)))
    shape = arrs[0].shape
    results = [
        score_one(RuleStats(tp=tp, fp=fp, fn=fn, tn=tn, length=length))
        for tp, fp, fn, tn, length in zip(*(a.ravel().tolist() for a in arrs))
    ]
    out = np.asarray(results, dtype=float)
    return out.reshape(shape + out.shape[1:])


def _class_axis(totals, covered: np.ndarray) -> np.ndarray:
    """`totals` (one count per class) shaped to broadcast against
    `covered` (``k`` or ``k x m``) along its first axis."""
    return np.asarray(totals, dtype=float).reshape((-1,) + (1,) * (covered.ndim - 1))


def head_stats(covered, totals, length: Any = 0) -> RuleStats:
    """The one-vs-rest `RuleStats` of every possible head: for class ``c``,
    the covered examples of ``c`` are the true positives and those of
    every other class the false positives, against the class totals of
    the scope (`totals`, one per class). `covered` is ``k`` (one rule)
    or ``k x m`` (``m`` rules); the result's arrays have its shape."""
    covered = np.asarray(covered, dtype=float)
    totals = _class_axis(totals, covered)
    n_covered = covered.sum(axis=0)
    fp = n_covered - covered
    fn = totals - covered
    return RuleStats(tp=covered, fp=fp, fn=fn, tn=(totals.sum() - totals) - fp, length=length)


def _best_of_heads(scores, covered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per rule, the best score over the heads (``k`` or ``k x m``
    `scores`) and its head: only heads the rule covers an example of
    count, and ties go to the head with more covered examples, then to
    the first. ``-inf`` (and head 0) for a rule covering nothing."""
    if isinstance(scores, tuple) or np.asarray(scores).ndim != covered.ndim:
        raise TypeError("choosing the head needs one score per rule -- a lexicographic (LEF) "
                        "score has no single maximum over the heads")
    scores = np.where(covered > 0, np.asarray(scores, dtype=float), -np.inf)
    best = np.full(covered.shape[1:], -np.inf)
    best_covered = np.full(covered.shape[1:], -1.0)
    head = np.zeros(covered.shape[1:], dtype=np.int64)
    for c in range(covered.shape[0]):
        better = (scores[c] > best) | ((scores[c] == best) & (covered[c] > best_covered))
        best = np.where(better, scores[c], best)
        best_covered = np.where(better, covered[c], best_covered)
        head = np.where(better, c, head)
    return best, head


class RuleHeuristic(ABC):
    """Base for rule-evaluation heuristics. Higher `score(...)` = more
    preferred, uniformly across every concrete heuristic. `score(...)`
    normally returns a plain float, but see `LEF` for the one exception
    (a tuple, for lexicographic tie-breaking) -- callers that compare
    scores via `>`/`sorted(key=...)` (as every heuristic consumer in this
    codebase does) don't need to care which shape they got.

    `needs_data` -- whether scoring a rule requires a
    `DataRepresentation` to count coverage against. `True` for almost
    every heuristic; `False` only for the purely structural ones
    (`MinimalLength`, which reads `stats.length` alone), letting callers
    like `pyrulearn.evaluation.sort_rules` score them from the `Rule`
    with no data on hand.

    **Two versions of the same formula.** `score` scores one rule
    (`stats` holding plain numbers) in plain Python; `batch_score` scores
    many at once (`stats` holding numpy arrays, e.g. every open child of
    a search node) and returns an array. They exist separately for
    speed: plain Python arithmetic is fast for one value and slow for a
    thousand, numpy the other way round. The default `batch_score` just
    calls `score` once per element -- correct for any heuristic, so a new
    heuristic only *needs* `score`; every built-in also overrides
    `batch_score` with the same formula in numpy. The two must agree
    (`tests/test_heuristics.py` checks every built-in on a grid that
    includes the zero-denominator cases); they can differ in the last
    bit only where numpy's `log2`/`log` rounds differently from `math`'s.
    A search must therefore compare `batch_score` values only with other
    `batch_score` values, never with a `score` -- otherwise a child with
    exactly its parent's stats could come out one bit "better".

    `relative_to_parent` -- whether a search scores a refinement relative
    to the rule it refines (see the module docstring): `score`/
    `batch_score` then take that rule's statistics as an optional second
    argument (`parent`), and without it score in the whole data, relative
    to the universal rule. `Adaptive`, `FoilGain`, and every wrapper
    around an adaptive heuristic are.
    """

    needs_data: bool = True
    relative_to_parent: bool = False

    @abstractmethod
    def score(self, stats: RuleStats) -> Score:
        raise NotImplementedError

    def batch_score(self, stats: RuleStats) -> Any:
        """`score` for every element of `stats`' arrays at once -- an
        array (a tuple of arrays for `LEF`), one entry per element. See
        the class docstring for why this exists next to `score`."""
        return _loop(self.score, stats)

    # -- bounding the refinements -----------------------------------------
    #
    # A refinement covers a subset of its rule's examples: at most `tp`
    # positives and `fp` negatives. An upper bound on its score lets a
    # search drop a rule none of whose refinements could beat the best
    # rule so far (optimistic pruning).

    def bound(self, stats: RuleStats) -> Score:
        """An upper bound on the score of every refinement of a rule with
        `stats`: by default the score at the ``(tp, 0)`` projection --
        every positive kept, every negative shed -- an upper bound for
        any heuristic that rises with `tp` and falls with `fp`. One that
        doesn't (`LikelihoodRatio`) overrides it. `length` stays the
        rule's own, so a length-penalized bound is no lower than any
        refinement's score."""
        return self.score(_optimistic(stats))

    def batch_bound(self, stats: RuleStats) -> Any:
        """`bound` for every element of `stats`' arrays at once."""
        return self.batch_score(_optimistic(stats))

    # -- choosing the head ------------------------------------------------
    #
    # A search for a rule of any class (a decision list's, `target_class`
    # None) knows a rule's covered examples per class, not its head. It
    # asks the heuristic for the best head: by default, the heuristic
    # scores the rule once per class, as that class against the rest
    # (`head_stats`), and the best of these scores wins. Every heuristic
    # gets this from its `batch_score` alone. A heuristic that always
    # prefers the covered majority can skip the other heads
    # (`MajorityHead`), and one defined on the whole class distribution
    # rather than on one class against the rest overrides it
    # (`ClassNeutral`).

    def head_scores(self, covered, totals, length: Any = 0, parent_covered=None) -> np.ndarray:
        """The score of each possible head: ``k`` scores for ``k`` covered
        counts (``k x m`` for ``k x m``), each class against the rest of
        the scope (`totals`). `parent_covered` (the parent's covered
        counts, ``k``) is the frame of an adaptive heuristic, which
        overrides this."""
        return np.asarray(self.batch_score(head_stats(covered, totals, length)))

    def best_head(self, covered, totals, length: Any = 0, parent_covered=None) -> Tuple[Any, Any]:
        """``(score, head)`` of a rule covering `covered` examples of each
        class (``k``, or ``k x m`` for ``m`` rules) in a scope holding
        `totals` of each: the best `head_scores` over the classes the rule
        covers, and the index of that class (ties: more covered examples,
        then the first class)."""
        covered = np.asarray(covered, dtype=float)
        return _best_of_heads(self.head_scores(covered, totals, length, parent_covered), covered)

    def head_bound(self, covered, totals, length: Any = 0) -> Any:
        """An upper bound on the score of every refinement of a rule
        covering `covered` of each class, over all heads: for each head
        `batch_bound` for each head, maximized over the heads."""
        covered = np.asarray(covered, dtype=float)
        return _best_of_heads(self.batch_bound(head_stats(covered, totals, length)), covered)[0]

    def __repr__(self) -> str:
        """The class name plus every constructor argument that differs
        from its default -- ``Laplace``, ``MEstimate(m=5)``,
        ``LEF(CoveredPositives, MinimalLength)`` -- what a printed model's
        conflict-resolution line names."""
        import inspect
        name = type(self).__name__
        try:
            params = inspect.signature(type(self).__init__).parameters
        except (TypeError, ValueError):
            return name
        parts = []
        for pname, param in params.items():
            if pname == "self" or not hasattr(self, pname):
                continue
            value = getattr(self, pname)
            if param.kind is param.VAR_POSITIONAL:
                parts.extend(repr(v) for v in value)
            elif param.default is param.empty or value != param.default:
                parts.append(f"{pname}={value!r}")
        return f"{name}({', '.join(parts)})" if parts else name

    def score_rule(
        self, rule: Rule, data: DataRepresentation, positive_class: Optional[Any] = None,
        parent_rule: Optional[Rule] = None,
    ) -> Score:
        """Convenience: `RuleStats.from_rule(...)` then `score(...)` in
        one call, so a `RuleHeuristic` can be used directly as a
        rule-level ranking key without the caller building stats by
        hand each time. `parent_rule` is the frame of an adaptive
        heuristic (ignored by a static one); without it, an adaptive
        heuristic scores relative to the universal rule."""
        stats = RuleStats.from_rule(rule, data, positive_class)
        if parent_rule is not None and self.relative_to_parent:
            return self.score(stats, RuleStats.from_rule(parent_rule, data, positive_class))
        return self.score(stats)

    def plot_isometrics(
        self,
        space: Optional[CoverageSpace] = None,
        levels: Any = 10,
        resolution: int = 200,
        filled: bool = False,
        cmap: Any = None,
        parent: Optional[RuleStats] = None,
        **kwargs,
    ) -> CoverageSpace:
        """Plot this heuristic's isometrics -- curves (or, `filled=True`,
        bands) of constant `score(...)` -- over a
        `pyrulearn.evaluation.CoverageSpace`. Inherited as-is by every
        concrete `RuleHeuristic`, adaptive ones included (see `parent=`
        below).

        `space` is built fresh (a dataset-agnostic, default-sized
        `CoverageSpace`) if omitted; pass one explicitly -- e.g. from
        `CoverageSpace.from_data` -- to size it to real data,
        or to layer these isometrics onto an existing plot (a ruleset's
        coverage, a rule's refinement path, ...). Returns the
        `CoverageSpace` used, so further layers can be added to it
        afterward: ``space = Precision().plot_isometrics();
        space.plot_ruleset(rs, rep, "pos")``.

        `tp`/`fp` vary continuously over the space's full `[0, n_neg] x
        [0, n_pos]` extent (not restricted to integers achievable by an
        actual rule) -- isometrics are a property of the heuristic's
        formula, not of any particular dataset's achievable points.
        `fn`/`tn` are derived to keep `n_pos`/`n_neg` fixed at the
        space's own dimensions. `stats.length` still has no meaningful
        value here (`LengthPenalized` will just ignore it, defaulting to
        0). `parent=` (a `RuleStats`, fixed across every grid point) is
        the frame of an adaptive heuristic -- e.g. `FoilGain` -- and
        ignored by a static one; without it, an adaptive heuristic is
        drawn relative to the universal rule. Points outside the
        parent's coverage (more positives or negatives than it covers)
        are no refinements of it, and their scores mean nothing.

        Isometric value labels (`ax.clabel`) are skipped when
        `space.show_labels` is `False` (the default for a dataset-agnostic
        space -- see `CoverageSpace`). By default, the isometrics run from
        red (low scores) to green (high ones) over lightly shaded bands of
        the same colors (`CoverageSpace.contour`'s `cmap` and `shade`,
        e.g. `shade=0` for bare lines). Color/linestyle/linewidth are
        fully configurable via `**kwargs`, forwarded through
        `CoverageSpace.contour` to matplotlib's `contour`/`contourf` --
        e.g. `colors="tab:blue", linestyles="dashed", linewidths=0.8` to
        keep one heuristic's isometrics visually distinct when layering
        several on the same space.
        """
        from .evaluation import CoverageSpace

        if space is None:
            space = CoverageSpace()

        def score_at(neg: float, pos: float) -> float:
            stats = RuleStats(tp=pos, fp=neg, fn=space.n_pos - pos, tn=space.n_neg - neg)
            if parent is not None and self.relative_to_parent:
                return self.score(stats, parent)
            return self.score(stats)

        cs = space.contour(score_at, levels=levels, resolution=resolution, filled=filled, cmap=cmap, **kwargs)
        if not filled and space.show_labels:
            space.ax.clabel(cs, inline=True, fontsize=7)
        return space

    def plot_isometric_through_point(
        self,
        neg: float,
        pos: float,
        space: Optional[CoverageSpace] = None,
        mark_point: bool = True,
        resolution: int = 200,
        cmap: Any = None,
        **kwargs,
    ) -> CoverageSpace:
        """Plot the single isometric of this heuristic that passes
        through the raw coverage point `(neg, pos)` -- every other
        point on the drawn curve scores exactly as well, under this
        heuristic, as this one point does. `space` is built fresh
        (default-sized) if omitted, same as `plot_isometrics`.
        `mark_point=True` (the default) also scatters the point itself,
        in whatever `colors=` was passed (if any), so it's clear which
        point the line actually passes through. See
        `plot_isometric_through_rule` for the common case of a specific
        `Rule`'s own coverage.
        """
        from .evaluation import CoverageSpace

        if space is None:
            space = CoverageSpace()
        level = self.score(RuleStats(tp=pos, fp=neg, fn=space.n_pos - pos, tn=space.n_neg - neg))
        self.plot_isometrics(space=space, levels=[level], resolution=resolution, cmap=cmap, **kwargs)
        if mark_point:
            marker_kwargs = {"color": kwargs["colors"]} if "colors" in kwargs else {}
            space.scatter(np.array([[neg, pos]]), zorder=3, **marker_kwargs)
        return space

    def plot_isometric_through_rule(
        self,
        rule: Rule,
        data: DataRepresentation,
        positive_class: Optional[Any] = None,
        space: Optional[CoverageSpace] = None,
        mark_point: bool = True,
        **kwargs,
    ) -> CoverageSpace:
        """Convenience wrapper around `plot_isometric_through_point`:
        the point is `rule`'s own raw (negatives, positives) coverage
        over `data` (`positive_class` defaults to
        `rule.target`, same resolution as `RuleStats.from_rule`).
        `space` defaults to `CoverageSpace.from_data(
        data, positive_class)` if omitted -- sized to the
        real data the rule was actually scored against, *not* the
        dataset-agnostic default (which would size the space to
        arbitrary placeholder counts, unrelated to this rule's real
        coverage).
        """
        from .evaluation import CoverageSpace

        resolved_class = positive_class if positive_class is not None else rule.target
        if resolved_class is None:
            raise ValueError("plot_isometric_through_rule needs positive_class (or a Rule with a target)")
        stats = RuleStats.from_rule(rule, data, resolved_class)
        if space is None:
            space = CoverageSpace.from_data(data, resolved_class)
        return self.plot_isometric_through_point(stats.fp, stats.tp, space=space, mark_point=mark_point, **kwargs)


class Adaptive(RuleHeuristic):
    """`base` relative to the parent rule: a refinement is scored among
    the parent's covered examples only -- they are the whole data (the
    frame), their class distribution is the prior, and so the parent is
    the universal rule. The frame thus adapts to the search, whereas that
    of the static `base` stays the whole data. `Adaptive(WRAcc())` is
    weighted relative accuracy with the parent's precision as the prior
    (adaptive WRA),
    `Adaptive(MEstimate(m))` an m-estimate whose `m` is weighed against
    the parent's examples. For a heuristic that doesn't depend on the
    frame (`Precision`, `Laplace`) it changes nothing.

    Scores relative to different parents have no common scale: a search
    compares them among the refinements of one rule, and the rule on the
    refinement path it returns is picked by a static `selection`, or
    by default the last one it reached (`LastRule`). Without a parent
    (`score(stats)`, e.g. for an isometrics plot or `sort_rules`) the
    frame is the whole data, and `Adaptive(h)` scores as `h`.

    `Adaptive` is idempotent: `Adaptive(h)` scores `base` *without* a
    parent, so an adaptive `base` (`FoilGain`, another `Adaptive`) is
    relative to the universal rule of the frame -- the parent again.
    Stecher, Janssen & Fürnkranz's inverted heuristics are
    `Adaptive(Inverted(h))` (`Inverted(Adaptive(h))` is the same).

    The bound (`bound`/`head_bound`) is relative to the rule itself: the
    best score any refinement of the rule can reach relative to it, by
    `base`'s own bound -- what a single lineage can prune by (a
    hill-climbing step that can't improve on the rule), not a bound
    against the rules of other parents."""

    relative_to_parent = True

    def __init__(self, base: RuleHeuristic):
        if not isinstance(base, RuleHeuristic):
            raise TypeError(f"Adaptive needs a RuleHeuristic, got {type(base).__name__}")
        self.base = base

    @property
    def needs_data(self) -> bool:
        return self.base.needs_data

    def score(self, stats: RuleStats, parent: Optional[RuleStats] = None) -> Score:
        return self.base.score(_within(stats, parent))

    def batch_score(self, stats: RuleStats, parent: Optional[RuleStats] = None) -> Any:
        return self.base.batch_score(_within(stats, parent))

    def bound(self, stats: RuleStats) -> Score:
        return self.base.bound(_within(stats, stats))

    def batch_bound(self, stats: RuleStats) -> Any:
        return self.base.batch_bound(_within(stats, stats))

    def head_scores(self, covered, totals, length: Any = 0, parent_covered=None) -> np.ndarray:
        """Each head's score with the parent's covered examples of each
        class (`parent_covered`) as the frame."""
        scope = totals if parent_covered is None else parent_covered
        return self.base.head_scores(covered, scope, length)

    def best_head(self, covered, totals, length: Any = 0, parent_covered=None) -> Tuple[Any, Any]:
        scope = totals if parent_covered is None else parent_covered
        return self.base.best_head(covered, scope, length)

    def head_bound(self, covered, totals, length: Any = 0) -> Any:
        return self.base.head_bound(covered, covered, length)

    def __repr__(self) -> str:
        return f"Adaptive({self.base!r})"


class Relative(Adaptive):
    """Deprecated alias of `Adaptive`, its name up to pyrulearn 0.7.1."""

    def __init__(self, base: RuleHeuristic):
        warnings.warn("Relative is deprecated and will be removed; use Adaptive, its new name",
                      DeprecationWarning, stacklevel=2)
        super().__init__(base)


class MajorityHead:
    """Mixin for a heuristic whose best head is always the covered
    majority -- one whose score, for a fixed number of covered examples,
    grows with the number of them that are positive, whatever the class
    totals (`Precision`, `Laplace`, ...). `best_head` then scores only
    that head instead of all of them, with the same result. Purely a
    speed-up: without it, the default `RuleHeuristic.best_head` finds the
    same head. Put it before the heuristic's base class."""

    def best_head(self, covered, totals, length: Any = 0, parent_covered=None) -> Tuple[Any, Any]:
        covered = np.asarray(covered, dtype=float)
        head = covered.argmax(axis=0)
        stats = head_stats(covered, totals, length)
        pick = lambda a: np.take_along_axis(a, head[None, ...], axis=0)[0]   # noqa: E731
        scores = np.asarray(self.batch_score(RuleStats(tp=pick(stats.tp), fp=pick(stats.fp), fn=pick(stats.fn),
                                                       tn=pick(stats.tn), length=length)), dtype=float)
        return np.where(covered.sum(axis=0) > 0, scores, -np.inf), head


class ClassNeutral:
    """Mixin for a heuristic defined on the whole class distribution of
    the covered examples rather than on one class against the rest --
    original CN2's entropy, say. Such a heuristic implements
    `score_distribution` (``k`` or ``k x m`` covered counts to one score
    per rule); its rule predicts the covered majority, and its bound is
    the best score of a rule covering one class only. `score(stats)` stays
    the two-class case, the positive class against the rest. Put it
    before the heuristic's base class."""

    @abstractmethod
    def score_distribution(self, covered: np.ndarray, length: Any = 0) -> np.ndarray:
        raise NotImplementedError

    def best_head(self, covered, totals, length: Any = 0, parent_covered=None) -> Tuple[Any, Any]:
        covered = np.asarray(covered, dtype=float)
        scores = np.asarray(self.score_distribution(covered, length), dtype=float)
        return np.where(covered.sum(axis=0) > 0, scores, -np.inf), covered.argmax(axis=0)

    def head_bound(self, covered, totals, length: Any = 0) -> Any:
        covered = np.asarray(covered, dtype=float)
        best = np.full(covered.shape[1:], -np.inf)
        for c in range(covered.shape[0]):
            pure = np.zeros_like(covered)
            pure[c] = covered[c]
            score = np.asarray(self.score_distribution(pure, length), dtype=float)
            best = np.where(covered[c] > 0, np.maximum(best, score), best)
        return best


class Precision(MajorityHead, RuleHeuristic):
    """h = tp / (tp+fp) -- also called Confidence in association-rule-
    mining terminology (same formula, different literature). Isometrics:
    a pencil of lines through the origin -- rewards purity regardless of
    coverage. See `Recall` for its natural counterpart, and `FBeta` for
    a parametrized combination of the two.
    """

    def score(self, stats: RuleStats) -> float:
        total = stats.tp + stats.fp
        return stats.tp / total if total > 0 else 0.0

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        return _div(tp, tp + fp)


class InformationContent(MajorityHead, RuleHeuristic):
    """h = log2(tp / (tp+fp)) -- Quinlan's information content of a rule,
    ``I = -log2(precision)``, the bits needed to signal that a covered
    example is positive; negated, as `CoveredNegatives` is, to keep this
    module's higher-is-better convention. It ranks rules exactly as
    `Precision` does (same isometrics); its use is as the base of a gain:
    `FoilGain` is ``Adaptive(WeightedGain(CoveredPositives(),
    InformationContent()))``. ``-inf`` for a rule covering no positive
    (precision 0: infinitely many bits)."""

    def score(self, stats: RuleStats) -> float:
        total = stats.tp + stats.fp
        return math.log2(stats.tp / total) if stats.tp > 0 else -math.inf

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        out = np.full(tp.shape, -np.inf)
        np.log2(_div(tp, tp + fp), out=out, where=tp > 0)
        return out


class Recall(RuleHeuristic):
    """h = tp / n_pos -- the true positive rate (a.k.a. sensitivity or
    hit rate; the same `tp/n_pos` term `YoudenJ` uses). Precision's
    natural counterpart, trading off "how complete is the coverage"
    against "how pure is it". Isometrics: parallel horizontal lines --
    `CoveredPositives`'s own rate-normalized twin (the relationship
    `Support` has to `Coverage`), so it ranks identically to raw
    `CoveredPositives` for a fixed dataset, just rescaled by the
    constant `n_pos`.
    """

    def score(self, stats: RuleStats) -> float:
        return stats.tp / stats.n_pos if stats.n_pos > 0 else 0.0

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, _, fn, _, _ = _counts(stats)
        return _div(tp, tp + fn)


class FBeta(RuleHeuristic):
    """The F-beta score, the weighted harmonic mean of `Precision` and
    `Recall`: ``h = (1+beta**2) * P*R / (beta**2*P + R)``. `beta` sets
    how many times more heavily recall is weighted than precision
    (`beta=1`, the default, is the standard F1 score, weighing them
    equally; `beta<1` favors precision, `beta>1` favors recall).

    Isometrics: a pencil of lines, like `Precision`/`Laplace`/
    `MEstimate`/`GHeuristic` -- not obvious from the P/R formula above,
    but clear once rewritten as ``h = (1+beta**2)*tp / (tp+fp+beta**2*
    n_pos)``, which pivots at `(-beta**2*n_pos, 0)`, on the fp-axis,
    the same sub-family `GHeuristic` belongs to (there, the pivot is a
    free parameter `-g`; here, it's `-beta**2*n_pos`, tying the pivot to
    the dataset's actual class balance rather than an arbitrary
    constant). `beta -> 0` collapses the pivot to the origin, recovering
    `Precision`; `beta -> infinity` pushes it to `-infinity`, flattening
    the isometrics toward `Recall`'s horizontal lines.
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def score(self, stats: RuleStats) -> float:
        p = Precision().score(stats)
        r = Recall().score(stats)
        b2 = self.beta ** 2
        denom = b2 * p + r
        return (1 + b2) * p * r / denom if denom > 0 else 0.0

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        p = Precision().batch_score(stats)
        r = Recall().batch_score(stats)
        b2 = self.beta ** 2
        return _div((1 + b2) * p * r, b2 * p + r)


class CoveredPositives(MajorityHead, RuleHeuristic):
    """h = tp -- maximizes covered positives alone; a rule's covered
    negatives don't affect its score at all. Isometrics: horizontal
    lines (tp constant), the degenerate limit of the parallel-line
    family with zero weight on fp -- so this alone can't distinguish a
    perfectly pure rule from one that also covers every negative, as
    long as both cover the same positives. Rarely useful by itself; a
    natural building block (e.g. the first term `WRAcc`/`LinearCost`-
    style heuristics trade off against fp) or an upper-bound reference.
    Not to be confused with `Recall` (`tp/n_pos`) -- this is the raw
    count, not a rate.

    One of the four confusion-matrix quadrant counts, alongside
    `CoveredNegatives`/`UncoveredPositives`/`UncoveredNegatives` --
    together they read "covered/uncovered" x "positives/negatives", and
    each one's sign follows the same rule: a *correct* quadrant (this
    one, and `UncoveredNegatives`) scores its raw count directly, an
    *error* quadrant (`CoveredNegatives`, `UncoveredPositives`) negates
    it, so more of it is never accidentally "better" under this
    module's higher-is-better convention.
    """

    def score(self, stats: RuleStats) -> float:
        return stats.tp

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        return _counts(stats)[0].copy()


class CoveredNegatives(RuleHeuristic):
    """h = -fp -- minimizes covered negatives alone (negated, to keep
    this module's higher-is-better convention); a rule's covered
    positives don't affect its score at all. Isometrics: vertical lines
    (fp constant) -- the complementary degenerate case to
    `CoveredPositives`. See its docstring for how this fits alongside
    `UncoveredPositives`/`UncoveredNegatives` as the four confusion-
    matrix quadrant counts.
    """

    def score(self, stats: RuleStats) -> float:
        return -stats.fp

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        return -_counts(stats)[1]


class UncoveredPositives(RuleHeuristic):
    """h = -fn -- positives this rule does *not* cover, negated (an
    error quadrant, like `CoveredNegatives` -- see `CoveredPositives`'s
    docstring). Isometrics: since `fn = n_pos - tp` for a fixed dataset,
    `-fn = tp - n_pos` -- the same horizontal-line family as
    `CoveredPositives` (zero weight on fp), just every line's label
    shifted by the constant `-n_pos`; the lines themselves sit in
    exactly the same places. What "uncovered" means in practice is
    entirely a property of whatever `stats` this is scored against --
    e.g. plugged into a `pyrulearn.pruning.ThresholdPrePruning` used as
    `pyrulearn.learners.seco.SeCo`'s own `stop_covering`, with `stats` computed
    against the loop's own `remaining` mask, this reads as "how many
    positives would still need explaining after this rule is added."
    """

    def score(self, stats: RuleStats) -> float:
        return -stats.fn

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        return -_counts(stats)[2]


class UncoveredNegatives(RuleHeuristic):
    """h = tn -- negatives this rule correctly leaves uncovered (a
    *correct* quadrant, like `CoveredPositives` -- see its docstring --
    so scored directly, no negation). Isometrics: since `tn = n_neg -
    fp` for a fixed dataset, this is the same vertical-line family as
    `CoveredNegatives` (zero weight on tp), just shifted by the constant
    `n_neg`.
    """

    def score(self, stats: RuleStats) -> float:
        return stats.tn

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        return _counts(stats)[3].copy()


class Laplace(MajorityHead, RuleHeuristic):
    """h = (tp+1) / (tp+fp+n_classes). Precision with its pivot moved to
    (-1, -(n_classes-1)) -- regularizes toward 1/n_classes, the accuracy
    expected of a rule on a problem with `n_classes` evenly distributed
    classes, for low-coverage rules.

    `n_classes` defaults to 2 (a rule's covered/not-covered split,
    toward 0.5) -- the usual form in rule learning. The Laplace estimate
    of CN2 (Clark & Boswell 1991) and of Webb's OPUS experiments counts
    the problem's classes, and gives more weight to coverage when there
    are more than two."""

    def __init__(self, n_classes: int = 2):
        if n_classes < 1:
            raise ValueError(f"n_classes must be at least 1, got {n_classes}")
        self.n_classes = n_classes

    def score(self, stats: RuleStats) -> float:
        return (stats.tp + 1) / (stats.tp + stats.fp + self.n_classes)

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        return (tp + 1) / (tp + fp + self.n_classes)


class MEstimate(RuleHeuristic):
    """h = (tp + m*p0) / (tp+fp+m), where p0 = n_pos/(n_pos+n_neg) is
    the prior positive rate. Generalizes `Precision` (m=0); as
    m -> infinity the isometrics straighten into parallel lines,
    approaching `WRAcc`-like behavior. `m` is this heuristic's knob
    between "trust purity" (small m) and "trust coverage" (large m).
    Isometrics: a pencil pivoting at `(-m*(1-p0), -m*p0)` -- see
    `GeneralizedMEstimate` for freeing that pivot from the dataset's
    own prior.

    Minus the constant p0, it is a weighted precision gain over the
    universal rule: ``h - p0 = (tp+fp)/(tp+fp+m) * (precision - p0)``, so
    it ranks rules as ``WeightedGain(w, Precision())`` with that weight
    ``w``. `Adaptive(MEstimate(m))` is the same gain over the parent
    rule, with `m` weighed against the parent's examples.
    """

    def __init__(self, m: float):
        self.m = m

    def score(self, stats: RuleStats) -> float:
        total = stats.n_pos + stats.n_neg
        p0 = stats.n_pos / total if total > 0 else 0.0
        return (stats.tp + self.m * p0) / (stats.tp + stats.fp + self.m)

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, fn, tn, _ = _counts(stats)
        p0 = _div(tp + fn, (tp + fn) + (fp + tn))
        with np.errstate(divide="ignore", invalid="ignore"):
            return (tp + self.m * p0) / (tp + fp + self.m)


class GeneralizedMEstimate(RuleHeuristic):
    """Generalizes `MEstimate` by replacing its prior positive rate p0
    with a free `cost` parameter: ``h = (tp + m*cost) / (tp+fp+m)``.
    `MEstimate(m)` is exactly `GeneralizedMEstimate(m, cost=p0)` for a
    given dataset's actual prior -- freeing `cost` from that constraint
    lets the pivot land anywhere on the line `fp+tp = -m`, at
    `(-m*(1-cost), -m*cost)`, rather than only at the single point
    `MEstimate`'s own prior-tied pivot occupies on that same line for a
    given `m`. Isometrics: same pencil family as `Precision`/`Laplace`/
    `MEstimate`.
    """

    def __init__(self, m: float, cost: float):
        self.m = m
        self.cost = cost

    def score(self, stats: RuleStats) -> float:
        return (stats.tp + self.m * self.cost) / (stats.tp + stats.fp + self.m)

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (tp + self.m * self.cost) / (tp + fp + self.m)


class GHeuristic(MajorityHead, RuleHeuristic):
    """The "g heuristic" from Gamberger & Lavrač's expert-guided subgroup
    discovery (as used in CN2-SD), ``h = tp / (fp + g)``, where the
    "generalization parameter" `g` trades off purity against coverage.
    Isometrics: a pencil of lines, like `Precision`/`Laplace`/
    `MEstimate`, but pivoting at `(-g, 0)` -- on the fp-axis itself,
    unlike `Laplace`'s fixed `(-1, -1)` or `MEstimate`'s prior-dependent
    sliding pivot. Larger `g` increasingly tolerates covered negatives
    in exchange for covering more positives, favoring more general
    rules (the same "trust purity vs. trust coverage" knob `MEstimate`'s
    `m` provides, via a differently-shaped family). `g` is meant to be a 
    positive constant (CN2-SD typically uses small integers, not 0).
    """

    def __init__(self, g: float):
        self.g = g

    def score(self, stats: RuleStats) -> float:
        denom = stats.fp + self.g
        return stats.tp / denom if denom > 0 else 0.0

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        return _div(tp, fp + self.g)


class WRAcc(RuleHeuristic):
    """Weighted relative accuracy: coverage-weighted improvement over
    the prior positive rate. Isometrics: parallel lines -- unlike the
    precision family, `WRAcc` can only ever prefer points on coverage
    space's convex hull. These lines are parallel to the diagonal, and on
    a fixed dataset `WRAcc` ranks rules exactly as `YoudenJ` (it is
    ``n_pos * n_neg / total**2`` times tpr - fpr).

    A fast shortcut for ``WeightedGain(Support(), Precision())`` -- the
    coverage-weighted precision gain over the universal rule, whose
    precision is the prior -- with the same scores. `Adaptive(WRAcc())`
    takes the gain over the parent rule instead ("adaptive" WRA)."""

    def score(self, stats: RuleStats) -> float:
        total = stats.n_pos + stats.n_neg
        covered = stats.tp + stats.fp
        if total == 0 or covered == 0:
            return 0.0
        p0 = stats.n_pos / total
        return (covered / total) * (stats.tp / covered - p0)

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, fn, tn, _ = _counts(stats)
        total = (tp + fn) + (fp + tn)
        covered = tp + fp
        with np.errstate(divide="ignore", invalid="ignore"):
            h = (covered / total) * (tp / covered - (tp + fn) / total)
        return np.where((total == 0) | (covered == 0), 0.0, h)


class YoudenJ(RuleHeuristic):
    """Youden's J statistic (Youden, 1950): h = tpr - fpr = tp/n_pos -
    fp/n_neg, also known as informedness (Powers, 2007) -- the natural
    quantity in ROC space, the vertical distance above the random-guess
    diagonal when the rule is plotted as (FPR, TPR); the relative cost
    measure with cost 1/2 (`LinearCostRates(cost_ratio=1.0)`).

    On a fixed dataset it ranks rules exactly as `WRAcc`, which is
    ``n_pos * n_neg / total**2`` times J: both have the same isometrics,
    parallel lines of slope n_pos/n_neg in coverage space, i.e., parallel
    to the diagonal. Only the scale differs, which matters only where
    scores from different data are compared or combined (e.g., the rules
    of later covering rounds, whose class totals have shrunk).
    """

    def score(self, stats: RuleStats) -> float:
        if stats.n_pos == 0 or stats.n_neg == 0:
            return 0.0
        return stats.tp / stats.n_pos - stats.fp / stats.n_neg

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, fn, tn, _ = _counts(stats)
        n_pos, n_neg = tp + fn, fp + tn
        with np.errstate(divide="ignore", invalid="ignore"):
            h = tp / n_pos - fp / n_neg
        return np.where((n_pos == 0) | (n_neg == 0), 0.0, h)


class LinearCostRates(RuleHeuristic):
    """h = tpr - cost_ratio*fpr -- the rate-space counterpart to
    `LinearCost`: trades off true/false positive *rates* rather than
    raw tp/fp counts, weighted by `cost_ratio` (how many points of FPR
    one point of TPR is "worth"). `YoudenJ` is exactly
    `LinearCostRates(cost_ratio=1.0)`, kept as its own zero-config class
    for the same discoverability reason `CoverageDifference` is kept
    alongside `LinearCost`.
    """

    def __init__(self, cost_ratio: float = 1.0):
        self.cost_ratio = cost_ratio

    def score(self, stats: RuleStats) -> float:
        if stats.n_pos == 0 or stats.n_neg == 0:
            return 0.0
        return stats.tp / stats.n_pos - self.cost_ratio * (stats.fp / stats.n_neg)

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, fn, tn, _ = _counts(stats)
        n_pos, n_neg = tp + fn, fp + tn
        with np.errstate(divide="ignore", invalid="ignore"):
            h = tp / n_pos - self.cost_ratio * (fp / n_neg)
        return np.where((n_pos == 0) | (n_neg == 0), 0.0, h)


class Accuracy(RuleHeuristic):
    """h = (tp+tn) / (n_pos+n_neg) -- fraction correctly classified if
    the rule predicts positive when it fires, negative otherwise.
    Isometrics: parallel lines of slope 1 (tp - fp constant), regardless
    of class distribution -- see `CoverageDifference` for the
    unnormalized, division-free heuristic with the *same* isometrics
    (same ranking), cheaper to compute when you only need relative
    order, not an actual accuracy percentage.
    """

    def score(self, stats: RuleStats) -> float:
        total = stats.n_pos + stats.n_neg
        return (stats.tp + stats.tn) / total if total > 0 else 0.0

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, fn, tn, _ = _counts(stats)
        return _div(tp + tn, (tp + fn) + (fp + tn))


class CoverageDifference(RuleHeuristic):
    """h = tp - fp. Since Accuracy = (tp - fp + n_neg)/(n_pos+n_neg) and
    both n_neg and n_pos+n_neg are constants for a fixed dataset,
    `CoverageDifference` ranks rules *identically* to `Accuracy` -- same
    isometrics -- just without the normalizing division, so prefer this
    one when only relative order matters (e.g. picking the best
    refinement) and `Accuracy` when the actual percentage matters (e.g.
    reporting/printing). Also exactly `LinearCost(cost_ratio=1.0)`,
    kept as its own zero-config class since "accuracy's fast cousin" is
    the more discoverable framing than requiring the caller to know
    `LinearCost`'s default.
    """

    def score(self, stats: RuleStats) -> float:
        return stats.tp - stats.fp

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        return tp - fp


class SlipperZ(RuleHeuristic):
    """h = sqrt(tp) - sqrt(fp): Slipper's rule-growing objective (Cohen &
    Singer 1999, their Z-tilde). With `tp`/`fp` the boosting weights of the
    covered positives/negatives, maximizing it minimizes the boosting
    loss bound ``Z`` a rule with its optimal confidence
    ``1/2 ln(tp/fp)`` achieves. Rewards coverage more than precision
    does: a rule covering more positives beats a slightly purer one."""

    def score(self, stats: RuleStats) -> float:
        return math.sqrt(max(stats.tp, 0)) - math.sqrt(max(stats.fp, 0))

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        return np.sqrt(np.maximum(tp, 0)) - np.sqrt(np.maximum(fp, 0))


class Support(RuleHeuristic):
    """h = (tp+fp) / (n_pos+n_neg) -- fraction of all examples covered,
    regardless of class. Isometrics: parallel anti-diagonal lines
    (tp+fp constant). Rarely useful alone (ignores purity entirely) but
    a natural building block/tie-breaker alongside a purity-based
    heuristic."""

    def score(self, stats: RuleStats) -> float:
        total = stats.n_pos + stats.n_neg
        return (stats.tp + stats.fp) / total if total > 0 else 0.0

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, fn, tn, _ = _counts(stats)
        return _div(tp + fp, (tp + fn) + (fp + tn))


class Coverage(RuleHeuristic):
    """h = tp + fp -- total examples covered, regardless of class; the
    unnormalized version of `Support` (`Support` = `Coverage` /
    (n_pos+n_neg)) -- the same relationship `CoverageDifference` has to
    `Accuracy`. Isometrics: parallel anti-diagonal lines, same family as
    `Support`, just without the normalizing division. Ignores purity
    entirely -- a natural building block/tie-breaker alongside a
    purity-based heuristic, not typically used alone.
    """

    def score(self, stats: RuleStats) -> float:
        return stats.tp + stats.fp

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        return tp + fp


class LinearCost(RuleHeuristic):
    """h = tp - cost_ratio*fp -- a direct linear trade-off between
    covered positives and negatives, weighted by `cost_ratio` (how many
    negatives one additional positive is "worth"). Isometrics: parallel
    lines of slope `cost_ratio` -- the general family `WRAcc`/`Accuracy`/
    `CoverageDifference` are specific instances of.
    """

    def __init__(self, cost_ratio: float = 1.0):
        self.cost_ratio = cost_ratio

    def score(self, stats: RuleStats) -> float:
        return stats.tp - self.cost_ratio * stats.fp

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        return tp - self.cost_ratio * fp


class LengthPenalized(RuleHeuristic):
    """Wraps another heuristic, subtracting a per-condition penalty --
    a simple Occam's-razor-style complexity penalty, and one of the
    heuristics here that read past `tp`/`fp`/`fn`/`tn` (`stats.length`).
    """

    def __init__(self, base: RuleHeuristic, penalty: float):
        self.base = base
        self.penalty = penalty

    @property
    def relative_to_parent(self) -> bool:
        return self.base.relative_to_parent

    def score(self, stats: RuleStats, *parent: RuleStats) -> float:
        parent = parent if self.relative_to_parent else ()
        return self.base.score(stats, *parent) - self.penalty * stats.length

    def batch_score(self, stats: RuleStats, *parent: RuleStats) -> np.ndarray:
        parent = parent if self.relative_to_parent else ()
        return self.base.batch_score(stats, *parent) - self.penalty * _counts(stats)[4]

    def best_head(self, covered, totals, length: Any = 0, parent_covered=None) -> Tuple[Any, Any]:
        score, head = self.base.best_head(covered, totals, length, parent_covered)
        return score - self.penalty * np.asarray(length, dtype=float), head

    def head_bound(self, covered, totals, length: Any = 0) -> Any:
        return self.base.head_bound(covered, totals, length) - self.penalty * np.asarray(length, dtype=float)

    def bound(self, stats: RuleStats) -> float:
        return self.base.bound(stats) - self.penalty * stats.length

    def batch_bound(self, stats: RuleStats) -> np.ndarray:
        return self.base.batch_bound(stats) - self.penalty * _counts(stats)[4]


class MinimalLength(RuleHeuristic):
    """h = -length -- shorter (fewer conditions) scores higher, i.e.
    generality measured purely as rule length. Michalski's "minimize the
    number of selectors", the second criterion of AQ's default `LEF`.
    Unlike `LengthPenalized` this isn't a wrapper and carries no
    trade-off weight: it's meant purely as a lexicographic tie-break
    *inside* a `LEF` (once earlier criteria have tied), not as a
    standalone objective -- on its own it just rewards the empty rule.
    Reads `stats.length` alone -- and nothing else -- so `needs_data` is
    `False`: `pyrulearn.evaluation.sort_rules(rules, by=MinimalLength())`
    ranks by rule size with no `data=` argument. Mid-search the field is
    populated by `RuleStats.from_rule` from `Rule.length()`.
    """

    needs_data = False

    def score(self, stats: RuleStats) -> float:
        return -stats.length

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        return -_counts(stats)[4]


class LastRule(RuleHeuristic):
    """h = length -- a *selection* heuristic: of the rules on a top-down
    refinement path, the last one is the longest, so selecting by length
    returns the rule a search ended at (as Foil does). The default
    selection of a search whose heuristic is relative to the parent rule
    (`RuleHeuristic.relative_to_parent`), whose scores along the path have
    no common scale to pick the best rule by. Among rules of the same
    length (a beam's), the search's own ranking decides. Its bound is
    ``+inf``: a refinement is always longer, so it never prunes."""

    needs_data = False

    def score(self, stats: RuleStats) -> float:
        return stats.length

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        return _counts(stats)[4].copy()

    def bound(self, stats: RuleStats) -> float:
        return math.inf

    def batch_bound(self, stats: RuleStats) -> np.ndarray:
        return np.full(_counts(stats)[4].shape, np.inf)

    def head_bound(self, covered, totals, length: Any = 0) -> Any:
        return np.full(np.asarray(covered).shape[1:], np.inf)


class Correlation(RuleHeuristic):
    """FOSSIL's search heuristic (Fürnkranz, 1994): the four-field
    (Matthews/phi) correlation coefficient between "rule covers this
    example" and "example is the positive class",
    ``(tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(fp+tn)(fn+tn))``. Ranges over
    [-1, 1] -- 1 for a rule that covers exactly the positives and
    nothing else, -1 for a rule whose coverage perfectly *anti*-predicts
    the class, 0 for coverage uncorrelated with class -- already matches
    this module's higher-is-better convention, no sign flip needed.
    Isometrics are hyperbolic, not straight lines (pencil or parallel),
    so `Correlation` doesn't fall into either family described above.

    The correlation is undefined, and scored 0, where a row or column of
    the table is empty. With fractional row weights (boosting), a count
    derived by subtraction (``fn = P - tp``) can come out a rounding error
    below 0 where it should be 0; the product under the root is then
    taken as 0 too.
    """

    def score(self, stats: RuleStats) -> float:
        product = (stats.tp + stats.fp) * (stats.tp + stats.fn) * (stats.fp + stats.tn) * (stats.fn + stats.tn)
        if product <= 0:
            return 0.0
        return (stats.tp * stats.tn - stats.fp * stats.fn) / math.sqrt(product)

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, fn, tn, _ = _counts(stats)
        denom = np.sqrt(np.maximum((tp + fp) * (tp + fn) * (fp + tn) * (fn + tn), 0.0))
        return _div(tp * tn - fp * fn, denom)


class ChiSquare(RuleHeuristic):
    """Pearson's chi-square statistic for the rule's own 2x2 contingency
    table ("rule covers this example" x "example is the positive
    class") -- CMAR's own significance test (Li, Han & Pei, 2001), used
    there (via `pyrulearn.pruning.ThresholdPrePruning`, the same
    significance-gate pattern `pyrulearn.learners.seco.CN2` already uses
    for its own `LikelihoodRatio`) to drop class association rules whose
    antecedent isn't actually correlated with the class, not merely
    confident.

    ``n * phi²`` where `phi` is the four-field correlation coefficient
    `Correlation` already computes -- ``phi = (tp*tn - fp*fn) /
    sqrt((tp+fp)(tp+fn)(fp+tn)(fn+tn))``, ``n = tp+fp+fn+tn`` -- so this
    is: ``n * (tp*tn - fp*fn)² / ((tp+fp)(tp+fn)(fp+tn)(fn+tn))``.
    Asymptotically chi-squared distributed with 1 degree of freedom under
    the null hypothesis of independence, same as `LikelihoodRatio` --
    higher = further from what independence alone would predict = more
    significant, matching this module's higher-is-better convention with
    no sign flip needed. Like every other heuristic here, this only
    *computes* the statistic -- thresholding for significance is the
    caller's job (`ThresholdPrePruning(ChiSquare(), critical_value, "<")`,
    `3.841` for alpha=0.05 at 1 degree of freedom being this codebase's
    own established default, `CN2`'s `significance_threshold`).

    `yates_correction` (default `True`, matching CMAR's own paper)
    applies Yates' continuity correction -- subtracting `n/2` from
    ``|tp*tn - fp*fn|`` before squaring -- which keeps the statistic from
    overstating significance on small counts, at the cost of being
    slightly conservative on large ones.
    """

    def __init__(self, yates_correction: bool = True):
        self.yates_correction = yates_correction

    def score(self, stats: RuleStats) -> float:
        n = stats.tp + stats.fp + stats.fn + stats.tn
        denom = (stats.tp + stats.fp) * (stats.tp + stats.fn) * (stats.fp + stats.tn) * (stats.fn + stats.tn)
        if n == 0 or denom <= 0:   # <= 0: a rounding error below 0, as for `Correlation`
            return 0.0
        diff = abs(stats.tp * stats.tn - stats.fp * stats.fn)
        if self.yates_correction:
            diff = max(0.0, diff - n / 2)
        return n * diff * diff / denom

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, fn, tn, _ = _counts(stats)
        n = tp + fp + fn + tn
        denom = (tp + fp) * (tp + fn) * (fp + tn) * (fn + tn)
        diff = np.abs(tp * tn - fp * fn)
        if self.yates_correction:
            diff = np.maximum(0.0, diff - n / 2)
        return np.where(n == 0, 0.0, _div(n * diff * diff, denom))


class Entropy(ClassNeutral, RuleHeuristic):
    """CN2's original search heuristic (Clark & Niblett, 1989): the
    *negated* binary entropy of the class distribution among covered
    examples, ``p = tp/(tp+fp)``, ``h = p*log2(p) + (1-p)*log2(1-p)``
    (i.e. ``-H(p)``). A pure rule (``p`` = 0 or 1) scores 0, the best
    possible; an even 50/50 split among covered positives/negatives
    scores -1 bit, the worst. CN2 itself *minimizes* plain entropy
    directly -- negated here purely so `Entropy` stays interchangeable
    with every other heuristic in this file under the shared
    higher-is-better convention.

    Isometrics are a pencil of lines through the origin, same pivot as
    `Precision` (since `h` depends only on the ratio `p`, constant along
    any ray from the origin) -- but each score value is a symmetric
    *pair* of rays, not a single one: `h(p) == h(1-p)`, so `Entropy`
    alone can't tell a 90%-positive rule from a 90%-*negative* one
    apart, only how skewed either split is (`RuleHeuristic.plot_isometrics`
    makes this pairing directly visible).

    Searching for a rule of any class (`best_head`), it is the negated
    entropy of the covered examples' whole class distribution, and the
    rule predicts their majority -- CN2's ordered mode. For two classes
    that is exactly `score`.
    """

    def score_distribution(self, covered: np.ndarray, length: Any = 0) -> np.ndarray:
        covered = np.asarray(covered, dtype=float)
        n = covered.sum(axis=0)
        p = covered / np.where(n > 0, n, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(p > 0, p * np.log2(p), 0.0).sum(axis=0)
        return np.where(n > 0, h, -math.log2(covered.shape[0]))

    def score(self, stats: RuleStats) -> float:
        covered = stats.tp + stats.fp
        if covered == 0:
            return -1.0  # no information at all: treat as the worst case (maximal entropy)
        p = stats.tp / covered
        if p == 0.0 or p == 1.0:
            return 0.0
        return p * math.log2(p) + (1 - p) * math.log2(1 - p)

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        covered = tp + fp
        p = _div(tp, covered)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = p * np.log2(p) + (1 - p) * np.log2(1 - p)
        h = np.where((p == 0.0) | (p == 1.0), 0.0, h)
        return np.where(covered == 0, -1.0, h)


class LikelihoodRatio(RuleHeuristic):
    """CN2's likelihood-ratio statistic (Clark & Niblett, 1989): the
    standard G-test / log-likelihood-ratio statistic comparing a rule's
    covered-example class counts (tp, fp) against the counts *expected*
    if coverage were independent of class -- ``e_tp = (tp+fp)*p0``,
    ``e_fp = (tp+fp)*(1-p0)``, where ``p0 = n_pos/(n_pos+n_neg)`` is the
    dataset's prior positive rate --
    ``2 * (tp*ln(tp/e_tp) + fp*ln(fp/e_fp))``. Higher = further from
    what the prior alone would predict = more significant/interesting,
    which already matches this module's higher-is-better convention
    directly (no sign flip, unlike `Entropy`). Asymptotically
    chi-squared distributed with 1 degree of freedom under the null
    hypothesis of independence -- CN2 itself thresholds this statistic
    for significance filtering; this class only computes it, no
    thresholding.
    """

    def score(self, stats: RuleStats) -> float:
        covered = stats.tp + stats.fp
        total = stats.n_pos + stats.n_neg
        if covered == 0 or total == 0:
            return 0.0
        p0 = stats.n_pos / total
        if p0 <= 0.0 or p0 >= 1.0:
            return 0.0
        e_tp = covered * p0
        e_fp = covered * (1 - p0)
        term_tp = stats.tp * math.log(stats.tp / e_tp) if stats.tp > 0 else 0.0
        term_fp = stats.fp * math.log(stats.fp / e_fp) if stats.fp > 0 else 0.0
        return 2 * (term_tp + term_fp)

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        tp, fp, fn, tn, _ = _counts(stats)
        covered = tp + fp
        total = (tp + fn) + (fp + tn)
        p0 = _div(tp + fn, total)
        e_tp = covered * p0
        e_fp = covered * (1 - p0)
        with np.errstate(divide="ignore", invalid="ignore"):
            term_tp = np.where(tp > 0, tp * np.log(tp / e_tp), 0.0)
            term_fp = np.where(fp > 0, fp * np.log(fp / e_fp), 0.0)
        valid = (covered != 0) & (total != 0) & (p0 > 0.0) & (p0 < 1.0)
        return np.where(valid, 2 * (term_tp + term_fp), 0.0)

    # Not monotone in tp and fp -- it rewards a covered distribution far
    # from the prior in either direction -- so the (tp, 0) projection is
    # no bound. The statistic is 2 (tp+fp) KL(covered || prior), jointly
    # convex in (tp, fp), so its maximum over the refinements' box
    # [1, tp] x [0, fp] (a refinement covering no positive is never a
    # search's result) lies at a corner; (1, 0) is below (tp, 0).

    def bound(self, stats: RuleStats) -> float:
        return float(self.batch_bound(stats))

    def batch_bound(self, stats: RuleStats) -> np.ndarray:
        tp, fp, fn, tn, length = _counts(stats)
        one = np.minimum(tp, 1.0)

        def at(t, f):
            return self.batch_score(RuleStats(tp=t, fp=f, fn=tp + fn - t, tn=fp + tn - f, length=length))

        return np.maximum.reduce([at(tp, 0.0 * fp), at(tp, fp), at(one, fp)])


class WeightedGain(RuleHeuristic):
    """``h(r) = weight(r) * (base(r) - base(r0))``: how much a rule
    improves `base` over the universal rule ``r0`` of the frame, weighted
    by `weight` (no weight: 1, `Gain`). In the whole data ``r0`` is the
    universal rule, so the gain is over the root of a top-down search;
    inside `Adaptive` the frame is the parent's covered examples, whose
    universal rule is the parent, so the gain is over the parent.

    `WRAcc` is ``WeightedGain(Support(), Precision())`` and Foil's gain
    ``Adaptive(WeightedGain(CoveredPositives(), InformationContent()))``
    (`FoilGain`); both have fast shortcuts with the same scores. With
    ``Support()`` raised to a power ``a`` it is Klösgen's family of
    quality functions; the m-estimate ranks as one, too (see `MEstimate`).

    A zero weight scores 0 whatever `base` is -- also where `base` is
    infinite (`InformationContent` of a rule covering no positive), as the
    limit ``p * log2(p/(p+n)) -> 0`` of Foil's gain is. A rule that
    doesn't change `base` (``base(r) == base(r0)``, infinities included)
    gains 0. `weight` and `base` must be static heuristics of single
    scores (wrap the `WeightedGain` in `Adaptive`, not its parts; a `LEF`
    can't be subtracted).

    The bound is the default ``(tp, 0)`` projection: sound for the gains
    above (keeping every positive and dropping every negative maximizes
    both the weight and the improvement), but not for any weight and
    base -- set ``optimistic_pruning=False`` for one where it isn't."""

    def __init__(self, weight: Optional[RuleHeuristic], base: RuleHeuristic):
        for part, name in ((weight, "weight"), (base, "base")):
            if part is None and name == "weight":
                continue
            if not isinstance(part, RuleHeuristic):
                raise TypeError(f"WeightedGain's {name} must be a RuleHeuristic, got {type(part).__name__}")
            if part.relative_to_parent:
                raise TypeError(f"WeightedGain's {name} must be static, got {part!r}: wrap the WeightedGain "
                                f"in Adaptive instead")
            if isinstance(part, LEF):
                raise TypeError(f"WeightedGain's {name} must score a single value, got the LEF {part!r}")
        self.weight = weight
        self.base = base

    @staticmethod
    def _root(stats: RuleStats) -> RuleStats:
        return RuleStats(tp=stats.tp + stats.fn, fp=stats.fp + stats.tn, fn=stats.fn * 0, tn=stats.tn * 0)

    def score(self, stats: RuleStats) -> float:
        w = 1.0 if self.weight is None else self.weight.score(stats)
        if w == 0:
            return 0.0
        b, b0 = self.base.score(stats), self.base.score(self._root(stats))
        return 0.0 if b == b0 else w * (b - b0)

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        b = np.asarray(self.base.batch_score(stats), dtype=float)
        b0 = np.asarray(self.base.batch_score(self._root(stats)), dtype=float)
        w = np.ones_like(b) if self.weight is None else np.asarray(self.weight.batch_score(stats), dtype=float)
        with np.errstate(invalid="ignore"):
            diff = np.where(b == b0, 0.0, b - b0)
            return np.where(w == 0, 0.0, w * diff)

    def __repr__(self) -> str:
        return f"WeightedGain({self.weight!r}, {self.base!r})"


class Gain(WeightedGain):
    """``h(r) = base(r) - base(r0)``: the unweighted `WeightedGain`, the
    improvement of `base` over the universal rule of the frame.
    `Adaptive(Gain(h))` is the improvement over the parent rule, `h` of
    the refinement minus `h` of the parent, both in the parent's frame.
    It ranks the refinements of one rule exactly as `Adaptive(h)` does
    (the parent's score is the same for all of them); it differs only
    where refinements of different parents are compared, as in a beam:
    by improvement, not by level."""

    def __init__(self, base: RuleHeuristic):
        super().__init__(None, base)

    def __repr__(self) -> str:
        return f"Gain({self.base!r})"


class FoilGain(RuleHeuristic):
    """Foil's information gain (Quinlan, 1990): ``tp * (log2(tp/(tp+fp))
    - log2(tp'/(tp'+fp')))``, the bits of information content a
    refinement saves per covered positive, over its parent rule (``tp'``,
    ``fp'``), times the positives it still covers. Adaptive to the parent
    rule by definition: a fast shortcut for
    ``Adaptive(WeightedGain(CoveredPositives(), InformationContent()))``,
    with the same scores (`Adaptive(FoilGain())` is `FoilGain()`).
    Without a parent (an isometrics plot, `sort_rules`), the gain is over
    the universal rule of the whole data, ``log2(n_pos/(n_pos+n_neg))``.

    A rule covering no positive gains 0 (the limit of ``p * log2(...)``),
    as much as one that doesn't improve on its parent; one below its
    parent's precision gains less than 0, and the more positives it
    covers, the less. Isometrics (with a fixed parent): the zero isometric
    is the line through the origin and the parent, with curves on either
    side -- the ``tp *`` factor makes a low-tp point score near 0 whatever
    its precision.

    Its bound is relative to the rule itself: the gain of the ``(tp, 0)``
    projection over the rule, ``-tp * log2(tp/(tp+fp))`` -- what
    `pyrulearn.learners.seco.GainAscentHillClimbing` stops by."""

    relative_to_parent = True

    @staticmethod
    def _prior(stats: RuleStats, parent: Optional[RuleStats]):
        if parent is None:
            return stats.tp + stats.fn, (stats.tp + stats.fn) + (stats.fp + stats.tn)
        return parent.tp, parent.tp + parent.fp

    def score(self, stats: RuleStats, parent: Optional[RuleStats] = None) -> float:
        ptp, ptotal = self._prior(stats, parent)
        if stats.tp == 0 or ptp == 0:
            return 0.0
        return stats.tp * (math.log2(stats.tp / (stats.tp + stats.fp)) - math.log2(ptp / ptotal))

    def batch_score(self, stats: RuleStats, parent: Optional[RuleStats] = None) -> np.ndarray:
        tp, fp, fn, tn, _ = _counts(stats)
        if parent is None:
            ptp, ptotal = tp + fn, (tp + fn) + (fp + tn)
        else:
            ptp, pfp, _, _, _ = _counts(parent)
            ptotal = ptp + pfp
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = tp * (np.log2(tp / (tp + fp)) - np.log2(ptp / ptotal))
        return np.where((tp == 0) | (ptp == 0), 0.0, gain)

    def bound(self, stats: RuleStats) -> float:
        return self.score(_optimistic(stats), stats)

    def batch_bound(self, stats: RuleStats) -> np.ndarray:
        return self.batch_score(_optimistic(stats), stats)

    def head_scores(self, covered, totals, length: Any = 0, parent_covered=None) -> np.ndarray:
        """Each head's gain over the parent's statistics for the same head
        (`parent_covered`, ``k``); without a parent, over the class's share
        of the scope (`totals`)."""
        covered = np.asarray(covered, dtype=float)
        stats = head_stats(covered, totals, length)
        if parent_covered is None:
            return np.asarray(self.batch_score(stats))
        parent = head_stats(parent_covered, totals)
        rows = []
        for c in range(covered.shape[0]):
            child = RuleStats(tp=stats.tp[c], fp=stats.fp[c], fn=stats.fn[c], tn=stats.tn[c], length=length)
            own = RuleStats(tp=float(parent.tp[c]), fp=float(parent.fp[c]), fn=float(parent.fn[c]),
                            tn=float(parent.tn[c]))
            rows.append(np.asarray(self.batch_score(child, own), dtype=float))
        return np.stack(rows)

    def head_bound(self, covered, totals, length: Any = 0) -> Any:
        covered = np.asarray(covered, dtype=float)
        return _best_of_heads(self.batch_bound(head_stats(covered, totals, length)), covered)[0]


class LEF(RuleHeuristic):
    """Michalski's Lexicographic Evaluation Functional (AQ): a list of
    heuristics compared in order -- the first one decides unless it ties,
    in which case the second one breaks the tie, and so on. `score(...)`
    returns the tuple of each constituent's own score rather than a
    single float; Python tuples already compare element-by-element and
    stop at the first difference, which *is* lexicographic comparison,
    so every existing consumer that ranks candidates via `>` or
    `sorted(key=...)` (every one in this codebase) gets LEF semantics
    with no changes of its own -- `LEF` only needs to produce the right
    kind of value, not be special-cased anywhere it's used.

    No lazy evaluation: every constituent heuristic is scored for every
    call, even when an earlier one alone would already have decided the
    comparison (true Michalski LEFs conventionally skip evaluating later
    ones once an earlier one has settled it). Not implemented that way
    here -- doing so would mean abandoning the plain `score(stats) ->
    value` shape for a pairwise comparator instead (`sorted(key=...)`
    needs a complete key up front; it can't lazily fill one in on
    demand), and it wouldn't actually save much: each constituent's own
    `score()` call is cheap scalar arithmetic over an already-computed
    `RuleStats` (a handful of ints), not another pass over the data, so
    `k` of them costs `k` cheap calls stacked on the one `RuleStats.
    from_rule` pass every candidate already pays regardless -- not
    worth the extra complexity to shave off.
    """

    def __init__(self, *heuristics: RuleHeuristic):
        self.heuristics = heuristics

    @property
    def relative_to_parent(self) -> bool:
        return any(h.relative_to_parent for h in self.heuristics)

    def score(self, stats: RuleStats, *parent: RuleStats) -> Tuple[float, ...]:
        return tuple(h.score(stats, *parent) if h.relative_to_parent else h.score(stats) for h in self.heuristics)

    def batch_score(self, stats: RuleStats, *parent: RuleStats) -> Tuple[np.ndarray, ...]:
        return tuple(h.batch_score(stats, *parent) if h.relative_to_parent else h.batch_score(stats)
                     for h in self.heuristics)


def _within(stats: RuleStats, parent: Optional[RuleStats]) -> RuleStats:
    """`stats` of a refinement counted among its parent's examples only:
    the parent's covered positives and negatives as the scope. `stats`
    itself without a parent (relative to the universal rule)."""
    if parent is None:
        return stats
    return RuleStats(tp=stats.tp, fp=stats.fp, fn=parent.tp - stats.tp, tn=parent.fp - stats.fp,
                     length=stats.length)


def _inverted_stats(stats: RuleStats) -> RuleStats:
    """`stats` seen from the upper right corner of the coverage space: the
    examples the rule *excludes*, with the classes exchanged -- its
    uncovered negatives (`tn`) as the covered positives, its uncovered
    positives (`fn`) as the covered negatives, and so on. Swapping twice
    gives back `stats`."""
    return RuleStats(tp=stats.tn, fp=stats.fn, fn=stats.fp, tn=stats.tp, length=stats.length)


def _projection_bound(heuristic: RuleHeuristic) -> bool:
    """Whether `heuristic`'s bound is the score at the ``(tp, 0)``
    projection (`RuleHeuristic.bound`), directly or through a wrapper --
    the bound `Inverted` can carry over."""
    if isinstance(heuristic, (Inverted, LengthPenalized, Adaptive)):
        return _projection_bound(heuristic.base)
    return type(heuristic).bound is RuleHeuristic.bound and type(heuristic).batch_bound is RuleHeuristic.batch_bound


def _ranks_like(a: RuleHeuristic, b: RuleHeuristic, n_pos: int = 20, n_neg: int = 30) -> bool:
    """Whether `a` and `b` order every point of a coverage space with
    `n_pos` positives and `n_neg` negatives the same way (equal scores
    included) -- e.g. a heuristic and its inversion, for a heuristic that
    is symmetric in that sense. False for lexicographic (tuple) scores,
    which this doesn't compare."""
    tp, fp = np.meshgrid(np.arange(n_pos + 1.0), np.arange(n_neg + 1.0))
    tp, fp = tp.ravel(), fp.ravel()
    stats = RuleStats(tp=tp, fp=fp, fn=n_pos - tp, tn=n_neg - fp)
    sa, sb = a.batch_score(stats), b.batch_score(stats)
    if isinstance(sa, tuple) or isinstance(sb, tuple):
        return False
    sa, sb = np.round(np.asarray(sa, dtype=float), 9), np.round(np.asarray(sb, dtype=float), 9)
    return bool(np.array_equal(np.sign(sa[:, None] - sa[None, :]), np.sign(sb[:, None] - sb[None, :])))


class Inverted(RuleHeuristic):
    """`base` evaluated from the upper right corner of the coverage space
    instead of from the origin: a rule is judged by the examples it
    *excludes*, with the classes exchanged -- ``h'(p, n) = h(N - n, P -
    p)``, the negatives it excludes taking the place of the covered
    positives, the positives it excludes that of the covered negatives.
    The statistics ``(tp, fp, fn, tn)`` become ``(tn, fn, fp, tp)``.
    `Inverted(Precision())`, say, is the share of negatives among the
    excluded examples, the negative predictive value; its isometrics are
    a pencil of lines through the upper right corner ``(N, P)``.

    On its own, an inverted heuristic is static, like any other: the
    corner is that of the whole data. Stecher, Janssen & Fürnkranz's
    inverted heuristics judge a refinement by what *it* excludes from its
    parent rule, so the corner moves with the search: they are
    `Adaptive(Inverted(h))`, ``h(N' - n, P' - p)`` among the parent's
    ``P'`` positives and ``N'`` negatives. An inverted adaptive heuristic
    is adaptive too, and inverts within the parent's frame:
    `Inverted(Adaptive(h))` is `Adaptive(Inverted(h))`, and
    `Inverted(FoilGain())` the inverted Foil gain (Mooney's CNF learner).

    Any heuristic of the statistics can be inverted, `LEF` and
    `LengthPenalized` included; `Inverted(Inverted(h))` scores as `h`. A
    heuristic that ranks every rule the same way inverted or not
    (`Accuracy`, `WRAcc`, `Correlation`, `ChiSquare`, `YoudenJ`: symmetric
    in the two classes and in covered/uncovered) isn't changed by it, and
    the constructor warns (adaptive, it is just `Adaptive(h)`). A
    class-neutral heuristic (`Entropy`) has an inverted two-class score
    (the purity of the *excluded* examples), but its score of the whole
    class distribution has no inverted form: a search for a rule of any
    class with it raises a ValueError.

    The bound is the default one, the score at the ``(tp, 0)`` projection:
    for a `base` that rises with `tp` and falls with `fp`, so does the
    inverted heuristic (keeping every positive and dropping every negative
    is still the best a refinement can do). A `base` with a bound of its
    own (`LikelihoodRatio`) has none here -- run the search with
    ``optimistic_pruning=False``.
    """

    def __init__(self, base: RuleHeuristic):
        import warnings
        if not isinstance(base, RuleHeuristic):
            raise TypeError(f"Inverted needs a RuleHeuristic, got {type(base).__name__}")
        self.base = base
        if not base.relative_to_parent and _ranks_like(base, self):
            warnings.warn(f"{base!r} ranks rules the same way inverted or not: Inverted({base!r}) scores as "
                          f"{base!r}", stacklevel=2)

    @property
    def relative_to_parent(self) -> bool:
        return self.base.relative_to_parent

    @property
    def needs_data(self) -> bool:
        return self.base.needs_data

    def _frame(self, stats: RuleStats, parent: Optional[RuleStats]) -> RuleStats:
        return _within(stats, parent) if self.relative_to_parent else stats

    def score(self, stats: RuleStats, parent: Optional[RuleStats] = None) -> Score:
        return self.base.score(_inverted_stats(self._frame(stats, parent)))

    def batch_score(self, stats: RuleStats, parent: Optional[RuleStats] = None) -> Any:
        return self.base.batch_score(_inverted_stats(self._frame(stats, parent)))

    def head_scores(self, covered, totals, length: Any = 0, parent_covered=None) -> np.ndarray:
        """Each head's inverted score, against the class totals of the
        scope (`totals`) -- for an adaptive `base`, of the parent rule
        (`parent_covered`)."""
        self._check_head()
        scope = parent_covered if parent_covered is not None and self.relative_to_parent else totals
        return np.asarray(self.base.batch_score(_inverted_stats(head_stats(covered, scope, length))))

    def _check_bound(self) -> None:
        if not _projection_bound(self.base):
            raise ValueError(f"{self!r} has no optimistic bound ({self.base!r} has a bound of its own, which "
                             f"doesn't carry over); search with optimistic_pruning=False")

    def bound(self, stats: RuleStats) -> Score:
        self._check_bound()
        if self.relative_to_parent:   # relative to the rule itself, as `Adaptive`'s
            return self.score(_optimistic(stats), stats)
        return super().bound(stats)

    def batch_bound(self, stats: RuleStats) -> Any:
        self._check_bound()
        if self.relative_to_parent:
            return self.batch_score(_optimistic(stats), stats)
        return super().batch_bound(stats)

    def _check_head(self) -> None:
        base = self.base
        while isinstance(base, (Inverted, LengthPenalized, Adaptive)):
            base = base.base
        if isinstance(base, ClassNeutral):
            raise ValueError(f"{self!r} has no inverted score of a class distribution: search for a rule of a "
                             f"fixed class, or use a heuristic of one class against the rest")

    def best_head(self, covered, totals, length: Any = 0, parent_covered=None) -> Tuple[Any, Any]:
        self._check_head()
        return super().best_head(covered, totals, length, parent_covered)

    def head_bound(self, covered, totals, length: Any = 0) -> Any:
        self._check_head()
        if self.relative_to_parent:
            covered = np.asarray(covered, dtype=float)
            return _best_of_heads(self.batch_score(_optimistic(head_stats(covered, covered, length))), covered)[0]
        return super().head_bound(covered, totals, length)

    def __repr__(self) -> str:
        return f"Inverted({self.base!r})"
