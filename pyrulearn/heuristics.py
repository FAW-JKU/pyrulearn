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

`GainHeuristic` is a separate category from plain `RuleHeuristic`: its
`score` takes a *second* mandatory argument, the parent rule's own
`RuleStats` (the rule's stats *before* its last refinement) -- unlike
every other heuristic here, a gain heuristic's score is only meaningful
as an improvement relative to that specific parent, not on a shared
absolute scale comparable across different parents/search rounds. That
used to be modeled as an optional `RuleStats.parent` field read
implicitly inside `score`; it's a mandatory second parameter instead
now, so a caller (a search algorithm, `plot_isometrics`, ...) is forced
to know which category of heuristic it's holding and supply the parent
explicitly -- calling `FoilGain().score(stats)` with one argument now
raises `TypeError` immediately rather than a heuristic-specific
"parent wasn't set" check. See `GainHeuristic`'s own docstring for why
this can't just be handled uniformly by the caller regardless of
category (e.g. `BeamSearch`'s "how does the overall-best-seen rule get
updated" logic genuinely differs between the two).

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
  `FoilGain`, with a fixed parent (see `plot_isometrics`'s `parent=`),
  isn't even a single smooth curve -- its `tp *` factor can make a
  low-tp, low-precision point score the same as a higher-tp, higher-
  precision one, producing a visibly non-convex, "hooked" isometric.
"""

from __future__ import annotations

import math
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

    See `GainHeuristic` for the one category that *isn't* a drop-in
    match for this base's single-argument `score` -- code that wants to
    call an arbitrary `RuleHeuristic` generically (as `score_rule` and
    `plot_isometrics` below do) needs to check for it explicitly.

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
    """

    needs_data: bool = True

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
        the scope (`totals`). `parent_covered` is for `GainHeuristic`."""
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

    def score_rule(self, rule: Rule, data: DataRepresentation, positive_class: Optional[Any] = None) -> Score:
        """Convenience: `RuleStats.from_rule(...)` then `score(...)` in
        one call, so a `RuleHeuristic` can be used directly as a
        rule-level ranking key without the caller building stats by
        hand each time. `GainHeuristic` overrides this with its own
        `parent_rule=`-taking version -- this one only works for a
        plain `RuleHeuristic`."""
        return self.score(RuleStats.from_rule(rule, data, positive_class))

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
        concrete `RuleHeuristic` -- including a `GainHeuristic`, whose
        two-argument `score` this method calls correctly via `parent=`
        below (see its own paragraph).

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
        required if `self` is a `GainHeuristic` -- e.g. `FoilGain` --
        and ignored otherwise; pass `RuleStats.universal(space.n_pos,
        space.n_neg)` for the standard "gain over the universal rule"
        reading (the same choice a gain heuristic needs to make when
        there's no specific prior refinement to compare against).

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
            if isinstance(self, GainHeuristic):
                if parent is None:
                    raise ValueError(f"{type(self).__name__}.plot_isometrics needs parent= (a GainHeuristic has no meaningful score without one)")
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


class GainHeuristic(RuleHeuristic):
    """Heuristics whose score is only meaningful as an *improvement*
    relative to one specific parent rule -- not, like every other
    heuristic in this module, on a shared absolute scale comparable
    across different parents or different points in a search. `FoilGain`
    is the only current example: its score is a function of *both* a
    rule's own stats and its immediate parent's, and two gain scores
    computed against two different parents simply aren't answering the
    same question, however close the numbers might land.

    That's why `score` takes `parent_stats` as a second, mandatory
    argument here rather than reading an optional `RuleStats.parent`
    field implicitly (the earlier design) -- a caller must know it's
    holding a `GainHeuristic` and supply the parent explicitly;
    `isinstance(heuristic, GainHeuristic)` is the intended way to find
    out.

    A search algorithm that can have more than one candidate in flight
    at once (`BeamSearch`, at any `beam_width` -- see its own docstring)
    can never safely rank a `GainHeuristic`'s scores against each other,
    since different candidates' scores may be deltas against different
    parents. Rather than carve out a special case for it there,
    `GainHeuristic` support lives entirely in `pyrulearn.learners.seco.
    GainAscentHillClimbing`, whose single-lineage structure (only ever
    one current rule, hence only ever one parent) makes every comparison
    it performs automatically valid, with no bookkeeping needed to keep
    it that way. `DeltaGain` wraps a plain `RuleHeuristic` into a
    `GainHeuristic` (its score minus its parent's); it predates
    `pyrulearn.learners.seco.HillClimbing` ranking a plain heuristic directly and
    is now redundant there, but still valid -- kept for the parametrized
    "any heuristic, gain-shaped" pattern and for explicitness.
    """

    @abstractmethod
    def score(self, stats: RuleStats, parent_stats: RuleStats) -> Score:
        raise NotImplementedError

    def batch_score(self, stats: RuleStats, parent_stats: RuleStats) -> Any:
        """`score` for every element of `stats`' arrays against the one
        `parent_stats` (plain numbers) -- see `RuleHeuristic.batch_score`."""
        return _loop(lambda s: self.score(s, parent_stats), stats)

    def head_scores(self, covered, totals, length: Any = 0, parent_covered=None) -> np.ndarray:
        """Each head's gain over the parent's statistics *for the same
        head* (`parent_covered`, ``k``) -- one `batch_score` per head, so
        that every gain heuristic gets it. With no parent (a search's
        root) every head gains 0."""
        covered = np.asarray(covered, dtype=float)
        if parent_covered is None:
            return np.zeros_like(covered)
        stats = head_stats(covered, totals, length)
        parent = head_stats(parent_covered, totals)
        rows = []
        for c in range(covered.shape[0]):
            child = RuleStats(tp=stats.tp[c], fp=stats.fp[c], fn=stats.fn[c], tn=stats.tn[c], length=length)
            own = RuleStats(tp=float(parent.tp[c]), fp=float(parent.fp[c]), fn=float(parent.fn[c]),
                            tn=float(parent.tn[c]))
            rows.append(np.asarray(self.batch_score(child, own), dtype=float))
        return np.stack(rows)

    def score_rule(
        self, rule: Rule, data: DataRepresentation,
        positive_class: Optional[Any] = None, parent_rule: Optional[Rule] = None,
    ) -> Score:
        """Same convenience as `RuleHeuristic.score_rule`, extended with
        the mandatory `parent_rule` this category needs -- computes both
        rules' stats and calls the two-argument `score`."""
        if parent_rule is None:
            raise ValueError(f"{type(self).__name__}.score_rule needs parent_rule")
        stats = RuleStats.from_rule(rule, data, positive_class)
        parent_stats = RuleStats.from_rule(parent_rule, data, positive_class)
        return self.score(stats, parent_stats)


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
    space's convex hull."""

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
    fp/n_neg -- an alternative to `WRAcc` that trades off *rates* rather
    than raw coverage-weighted counts: where `WRAcc` weights by
    `covered/total` (so a rule's score shrinks as its raw coverage
    shrinks, regardless of class balance), this normalizes tp and fp
    separately by their own class totals, making it the natural
    quantity in ROC space -- the vertical distance above the
    random-guess diagonal when the rule is plotted as (FPR, TPR); also
    known as informedness. Still linear in (tp, fp) for a fixed dataset
    (n_pos/n_neg are constants), so its isometrics are parallel lines
    too, just a different slope than `WRAcc`'s.
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

    def score(self, stats: RuleStats) -> float:
        return self.base.score(stats) - self.penalty * stats.length

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        return self.base.batch_score(stats) - self.penalty * _counts(stats)[4]

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


class FoilGain(GainHeuristic):
    """FOIL's information gain (Quinlan, 1990): rewards a refinement by
    how much it increases the (log) probability that a covered example
    is positive, weighted by how many positives survive the refinement
    -- ``tp * (log2(tp/(tp+fp)) - log2(tp0/(tp0+fp0)))``, where
    ``tp0``/``fp0`` are the *pre-refinement* rule's `parent_stats.tp`/
    `parent_stats.fp` (see `GainHeuristic` for why that's a separate,
    mandatory argument to `score` rather than an optional field read off
    `stats` itself -- see `RuleHeuristic.plot_isometrics`'s `parent=`
    argument to make this plottable anyway, e.g. against
    `RuleStats.universal(...)`). With a fixed parent, isometrics are
    neither a pencil nor parallel lines, nor even always a single
    connected curve like `Correlation`/`LikelihoodRatio` -- the `tp *`
    factor can make a low-tp, low-precision point score the same as a
    higher-tp, higher-precision one, producing a visibly non-convex,
    "hooked" isometric near the tp axis.
    """

    def score(self, stats: RuleStats, parent_stats: RuleStats) -> float:
        if stats.tp == 0 or parent_stats.tp == 0:
            return 0.0
        return stats.tp * (
            math.log2(stats.tp / (stats.tp + stats.fp))
            - math.log2(parent_stats.tp / (parent_stats.tp + parent_stats.fp))
        )

    def batch_score(self, stats: RuleStats, parent_stats: RuleStats) -> np.ndarray:
        tp, fp, _, _, _ = _counts(stats)
        ptp, pfp, _, _, _ = _counts(parent_stats)
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = tp * (np.log2(tp / (tp + fp)) - np.log2(ptp / (ptp + pfp)))
        return np.where((tp == 0) | (ptp == 0), 0.0, gain)


class DeltaGain(GainHeuristic):
    """Turns any ordinary `RuleHeuristic` into a `GainHeuristic`: scores
    a refinement by how much `base`'s own score improved relative to
    its parent, ``base.score(stats) - base.score(parent_stats)``. This
    doesn't give `base` a shared absolute scale across different
    parents -- nothing could, that's the whole reason `GainHeuristic`
    exists as its own category -- but it does make "is this refinement
    an improvement at all" (a positive delta) well-defined for *any*
    heuristic this way. `pyrulearn.learners.seco.HillClimbing` now ranks a plain
    `RuleHeuristic` directly (its single lineage makes the same-parent
    comparison sound without the wrapper), so `DeltaGain(Laplace())` and
    plain `Laplace` behave identically there -- the wrapper is redundant
    for that use, kept for explicitness and for anywhere a `GainHeuristic`
    is specifically expected (`pyrulearn.learners.seco.GainAscentHillClimbing`).

    `base` must return a plain float, not a `LEF`'s tuple -- subtracting
    two tuples isn't defined, and `LEF`'s own tie-breaking semantics
    wouldn't carry a clear meaning under subtraction anyway.
    """

    def __init__(self, base: RuleHeuristic):
        self.base = base

    def score(self, stats: RuleStats, parent_stats: RuleStats) -> float:
        return self.base.score(stats) - self.base.score(parent_stats)

    def batch_score(self, stats: RuleStats, parent_stats: RuleStats) -> np.ndarray:
        # the parent through batch_score too, so a child with its parent's
        # exact stats gains exactly 0
        return self.base.batch_score(stats) - self.base.batch_score(parent_stats)


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

    def score(self, stats: RuleStats) -> Tuple[float, ...]:
        return tuple(h.score(stats) for h in self.heuristics)

    def batch_score(self, stats: RuleStats) -> Tuple[np.ndarray, ...]:
        return tuple(h.batch_score(stats) for h in self.heuristics)
