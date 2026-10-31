"""
pyrulearn.learners.seco
===========================

A generic separate-and-conquer (SeCo) rule learner, built from five
composable building blocks (see Fürnkranz's SeCo survey for the general
shape this follows):

1. **search for a single rule** -- `RuleSearch`. `BeamSearch` (any
   `RuleHeuristic` except `GainHeuristic`, `beam_width` at least 1) is
   the general-purpose default; `HillClimbing` is its single-lineage
   greedy special case (`beam_width` 1 plus "stop at a local maximum of
   the heuristic"); `GainAscentHillClimbing` is a further subclass, the
   only search that can run a `GainHeuristic` (e.g. `FoilGain`) -- a
   gain score is only meaningful against its own immediate parent, and a
   single lineage is the one place "only one parent, ever" holds (see
   `BeamSearch`'s docstring for why it can't rank gain scores safely at
   any beam width).
2. **a search heuristic** -- already existed before this module:
   `pyrulearn.heuristics.RuleHeuristic`/`RuleStats`/`GainHeuristic`/
   `DeltaGain`, reused as-is. `BeamSearch`/`HillClimbing`/
   `GainAscentHillClimbing`'s own
   `filtering=`/`stopping=` criteria (`pyrulearn.pruning.
   PrePruningCriterion` and friends) live in their own module too,
   since they're equally reusable outside a SeCo-shaped loop.
3. **prepare for learning a single rule** -- `SingleRulePreparation`
   (`NoSplit` default; `GrowPruneSplit` for RIPPER/IREP-style behavior).
4. **post-process a single rule** -- `SingleRulePostProcessing`
   (`NoPostProcessing` default; `ReducedErrorPruning` for RIPPER/IREP-
   style behavior).
5. **initialize the search space** -- `SearchSpaceInit`
   (`EmptyRuleAllFeatures` default; `FeatureSubset` for a named subset;
   `SeedExample` for AQ's single-example seed -- generalize one still-
   uncovered positive rather than start from the fully-open empty rule).

These five compose into `SingleRuleLearner`. The outer `SeCo`
sequential-covering loop (subclasses `pyrulearn.learners.
NativeRuleLearner`) calls it repeatedly: learn one rule, remove every
example it covers (both classes -- see `SeCo`'s own docstring) from
further consideration, repeat until `stop_covering` (a
`pyrulearn.pruning.PrePruningCriterion`, same as `filtering=`/
`stopping=`, but consulted only via its bare `evaluate()` -- see
`SeCo`'s own docstring) says stop, the single-rule search returns
nothing (`filtering` rejected everything it reached), or no positive
examples remain uncovered. `RuleSet`-level
preparation/post-processing (e.g. RIPPER's whole-set optimization
phase, revisiting rules already added in light of each other) is a
separate, higher-level pair of hooks around that outer loop, distinct
from building blocks 3/4 above (which only ever see one rule in
isolation) -- not yet implemented; `SeCo` as built so far always
accepts every rule it finds (down to the covering criterion) without
ever reconsidering earlier ones.

Usually, `target_class` is a fixed parameter -- the building blocks find
and score rules for one predetermined class. Multi-class is then handled
*around* this machinery, via the `fit(data, model=...)` switcher (see
`pyrulearn.learners.RuleLearner.fit`) -- `fit(data)` with no
`target_class` builds `_MULTICLASS_DEFAULT` (`ConceptSet`, one-vs-rest,
for the SeCo family): a binary decomposition (`model=ConceptSet |
ConceptCascade | PairwiseModel` -- one-vs-rest / ordered peeling / round
robin, via `DecomposingLearner`; `pyrulearn.learners.multiclass` wraps
these as thin sugar) fits copies of itself with `target_class` set.

A search can also look for the best rule of *any* class: with
`target_class` None, `SingleRuleLearner` searches with a `HeadObjective`,
which counts each rule's covered examples per class and lets the
heuristic choose its head (`pyrulearn.heuristics.RuleHeuristic.
best_head`) -- the head may change from a rule to its refinement.
`model=DecisionList` (`SeCo._decision_list_fit`) is one covering loop
over all classes at once, rules kept in learn order, first match wins
(`model=FlatRuleSet` gives the same rules resolved by that order,
`ListCombiner()`). Each of its rules comes from three independent
choices (see "seeds and heads" below): a seed (`seed=`: `FirstSeed`,
`RandomSeed`, or none), the search space (`space_init=`; with a seed,
the seed's generalizations unless given) and the head (`target_class=`:
`HeadPerRefinement`, `HeadPerSeedExample`, `HeadPerRule`; with a seed, `HeadPerSeedExample`). AQ's
seed covering is a random seed with its generalizations and its class
(`AQR`'s own `_MULTICLASS_DEFAULT`); CN2's ordered mode is no seed, all
features and `HeadPerRefinement()`.

"Masks, not splits": every one of these building blocks that needs to
restrict which examples are in scope (a grow/prune split, "which
examples are still uncovered") does so via a boolean `example_mask` over
`data`'s rows, not by constructing a smaller
`HorizontalBitmapRepresentation`. This is already how `RuleStats.from_rule`
and `Rule.specialize` work; the pieces built here just thread that mask
through rather than reintroducing a splitting approach.

The `example_mask` may also be a non-negative *weight vector* -- a
weighted covering scope (`WeightedCovering`) -- and the data may carry
row weights of its own (`pyrulearn.data.DataRepresentation.weights`).
Everything that counts examples then sums their effective weights
(`DataRepresentation.scope`); everything that only needs to know which
rows are in scope (seed picking, the grow/prune split) takes the rows
with a positive weight. How the scope evolves from one rule to the next
is `SeCo`'s `covering` strategy: `RemovalCovering` (the default, the
classic "separate") or `WeightedCovering`.
"""

from __future__ import annotations

import heapq
import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..models import (
    ConceptCascade, ConceptModel, ConceptSet, DecisionList, FlatRuleSet, MajorityClass, SingleRule,
    annotate_default_rule, annotate_rules,
)
from ..heuristics import (
    Correlation, CoveredNegatives, CoveredPositives, FoilGain, GainHeuristic,
    Laplace, LEF, LikelihoodRatio, MinimalLength, Precision, RuleHeuristic, RuleStats, Score,
)
from .base import DecomposingLearner, NativeRuleLearner, produces
from ..combiners import ListCombiner, MicroVoteCombiner
from ..pruning import AnyOf, EncodingLengthRestriction, PrePruningCriterion, ThresholdPrePruning
from ..data import HorizontalBitmapRepresentation
from ..data.attributes import NumericGroup, ThresholdChain
from ..rule import Literal, Rule


class RuleSearch(ABC):
    """Base for "find the single best rule for `target_class`" search
    strategies -- building block 1. Concrete subclasses explore
    refinements of `initial_candidates` (each an already-constructed
    `(rule, open-feature-mask)` pair, typically from a future
    `SearchSpaceInit`), scoring candidates via `heuristic` -- a
    `RuleHeuristic`, or any `Objective` (see `Objective`; a heuristic is
    wrapped in a `HeuristicObjective`) -- against
    `data` restricted to `example_mask` (if given, e.g. a
    grow-set mask), and return the single best rule found -- or `None`
    if a `filtering` or `stopping` criterion is configured and no rule
    the search ever reached was in its acceptable region (the criterion
    is a hard gate on what may be returned, not just a tie-breaker; the
    caller decides what "found nothing" means -- `SeCo`, for one, ends
    its covering loop).

    `filtering` and `stopping` (both optional, independent, and usable
    together, on top of the always-on `tp == 0` floor and every search's
    `optimistic_pruning`) are the two
    different ways a `PrePruningCriterion` can be consumed -- both gate
    what may be returned; only `stopping` also halts the search -- see
    its own docstring, and each search's, for exactly
    how each applies
    both.
    """

    @abstractmethod
    def search(
        self,
        data: HorizontalBitmapRepresentation,
        target_class: Any,
        heuristic: RuleHeuristic,
        initial_candidates: Sequence[Tuple[Rule, FrozenSet[int]]],
        example_mask: Optional[np.ndarray] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
    ) -> Optional[Rule]:
        raise NotImplementedError

    def search_all(
        self,
        data: HorizontalBitmapRepresentation,
        target_class: Any,
        heuristic: RuleHeuristic,
        initial_candidates: Sequence[Tuple[Rule, FrozenSet[int]]],
        example_mask: Optional[np.ndarray] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
    ) -> List[Rule]:
        """Every rule one search yields -- for a search that can yield
        several (`GainAscentHillClimbing` with `branch_similarity`, CPAR's),
        in the order they were found; `SeCo`'s covering loop accepts each.
        Default: just `search`'s rule (none if it found nothing)."""
        rule = self.search(data, target_class, heuristic, initial_candidates,
                           example_mask=example_mask, filtering=filtering, stopping=stopping)
        return [] if rule is None else [rule]


def handle_for(data, rule: Rule, example_mask: Optional[np.ndarray]):
    """The `data` cover handle for `rule`: `initial_cover`
    (masking is the zeroth refinement), refined in through each of
    `rule`'s own conditions. In practice this is always zero iterations
    -- every `SearchSpaceInit` seeds the empty rule -- but staying
    correct for a hypothetical non-empty seed costs nothing."""
    handle = data.initial_cover(example_mask)
    for lit in rule.conditions:
        handle = data.refine_cover(handle, lit.feature)
    return handle


def stats_from_handle(data, target_class: Any, rule: Rule, handle) -> RuleStats:
    tp, fp, fn, tn = data.cover_counts(handle, target_class)
    return RuleStats(tp=tp, fp=fp, fn=fn, tn=tn, length=rule.length())


def numeric_chain_lookup(dataspec) -> Dict[int, Tuple[Tuple[int, ...], int]]:
    """feature -> ``(chain, position)`` for every feature that's one of a
    numeric attribute's monotonic thresholds -- a `NumericGroup`'s `ge`
    (ascending) *and* its `lt` reversed (``tuple(reversed(c.lt))``), or a
    standalone `ThresholdChain`'s `feature_indices`. Built once per
    search call from `dataspec.constraints` (static for the call, so
    cheap to redo rather than cache).

    `lt` is reversed deliberately: `ge` is True-from-the-bottom (the
    lowest threshold is satisfied most often), `lt` is True-from-the-top
    (the highest threshold's `<` test is satisfied most often) -- same
    monotonic shape, opposite end. Reversing `lt` makes both families
    "True iff the row satisfies at least `position + 1` of this chain",
    the one invariant `chain_cover_counts` relies on, so `score_children`
    can batch a `<t` chain through the exact same call as a `>=t` one.
    This only reorders the *lookup*, not the feature indices themselves,
    and doesn't assume `lt[k]`/`ge[k]` are complements -- `chain_cover_counts`
    always counts the real column, so missing values (where *neither*
    holds under `MissingStrategy.NEVER_COVERS`) need no special case.
    Used by `score_children`; see `ROADMAP.md`'s numeric-threshold-
    counting item.
    """
    chains: Dict[int, Tuple[Tuple[int, ...], int]] = {}
    if dataspec is None:
        return chains
    for c in dataspec.constraints:
        if isinstance(c, NumericGroup):
            groups: List[Tuple[int, ...]] = [c.ge]
            if c.lt:
                groups.append(tuple(reversed(c.lt)))
        elif isinstance(c, ThresholdChain):
            groups = [c.feature_indices]
        else:
            continue
        for g in groups:
            for pos, f in enumerate(g):
                chains[f] = (g, pos)
    return chains


def score_children(
    data,
    dataspec,
    target_class: Any,
    rule: Rule,
    mask: FrozenSet[int],
    handle,
    stats: RuleStats,
    chains: Dict[int, Tuple[Tuple[int, ...], int]],
) -> Tuple[List[Tuple[Rule, FrozenSet[int], Any, RuleStats]], Set[int]]:
    """`rule.specialize(dataspec, mask)`'s one-literal children, each
    with its handle and stats, all built eagerly. No search uses this
    any more -- they score from `count_open_children` and build only
    what they follow (`materialize_child`); kept until it's decided
    whether `chain_cover_counts` comes back (see `ROADMAP.md`). Every
    child whose added literal is one of at least two
    *open* thresholds of the same numeric attribute's chain (`chains`,
    from `numeric_chain_lookup` -- a `>=t` family or a reversed `<t`
    family) is scored together, via one `data.chain_cover_counts` pass
    over the attribute's whole open run instead of one `refine_cover`+
    `cover_counts` per threshold. Everything else (non-numeric features,
    a lone open threshold, or a representation with no
    `chain_cover_counts`) falls back to the original per-child
    `refine_cover`+`cover_counts`.

    `chain_cover_counts` only exists on representations that define it
    (currently `HorizontalBitmapRepresentation`); on any other representation
    `getattr` finds nothing and every child falls back, so this is a
    pure opt-in fast path, not a new required primitive.

    Returns `(children, dead)` exactly as the old inline loop did:
    `children` the non-degenerate (`tp > 0`) results in `rule.specialize`'s
    own order (so tie-breaking among equally-scored children is
    unaffected), `dead` the added features whose child covers no
    positive -- for the caller to mask out of every sibling's own child
    mask, same as before.
    """
    specialized = rule.specialize(dataspec, mask)
    if not specialized:
        return [], set()
    by_feature: Dict[int, Tuple[Rule, FrozenSet[int]]] = {}
    for child_rule, child_mask in specialized:
        by_feature[child_rule.conditions[-1].feature] = (child_rule, child_mask)

    n_pos_scope = stats.tp + stats.fn
    n_neg_scope = stats.fp + stats.tn
    chain_counts = getattr(data, "chain_cover_counts", None)
    tp_fp: Dict[int, Tuple[Any, Any]] = {}
    if chain_counts is not None:
        groups: Dict[int, List[int]] = {}
        for f in by_feature:
            info = chains.get(f)
            if info is not None:
                groups.setdefault(id(info[0]), []).append(f)
        for feats in groups.values():
            if len(feats) < 2:
                continue  # one open threshold: no pass to save by batching
            full = chains[feats[0]][0]
            ordered = [f for f in full if f in feats]
            tp_arr, fp_arr = chain_counts(handle, target_class, ordered)
            for f, tp, fp in zip(ordered, tp_arr, fp_arr):
                tp_fp[f] = (tp, fp)

    children: List[Tuple[Rule, FrozenSet[int], Any, RuleStats]] = []
    dead: Set[int] = set()
    for added, (child_rule, child_mask) in by_feature.items():
        if added not in tp_fp:
            child_handle = data.refine_cover(handle, added)
            child_stats = stats_from_handle(data, target_class, child_rule, child_handle)
            if child_stats.tp == 0:
                dead.add(added)
            else:
                children.append((child_rule, child_mask, child_handle, child_stats))
            continue
        tp, fp = tp_fp[added]
        if tp == 0:
            dead.add(added)
            continue
        child_handle = data.refine_cover(handle, added)
        fn, tn = n_pos_scope - tp, n_neg_scope - fp
        children.append((child_rule, child_mask, child_handle, RuleStats(tp=tp, fp=fp, fn=fn, tn=tn, length=child_rule.length())))
    return children, dead


def _score_children_matmul(
    data,
    dataspec,
    target_class: Any,
    rule: Rule,
    mask: FrozenSet[int],
    handle,
    stats: RuleStats,
    chains: Dict[int, Tuple[Tuple[int, ...], int]],
) -> Tuple[List[Tuple[Rule, FrozenSet[int], Any, RuleStats]], Set[int]]:
    """Experimental alternative to `score_children`: every open feature
    of `rule.specialize`'s children -- numeric or not, one attribute or
    several -- scored in a single `data.batch_cover_counts` matrix
    multiply, the ENDER-style "one matmul per round" rather than
    `score_children`'s "one pass per numeric attribute's chain".
    `chains` is accepted and ignored, only so this is a drop-in swap for
    `score_children` (same signature) in a benchmark. Superseded by
    `count_open_children`, which counts the same way without building
    every child first. See `batch_cover_counts` and `ROADMAP.md`.
    """
    specialized = rule.specialize(dataspec, mask)
    if not specialized:
        return [], set()
    by_feature: Dict[int, Tuple[Rule, FrozenSet[int]]] = {}
    for child_rule, child_mask in specialized:
        by_feature[child_rule.conditions[-1].feature] = (child_rule, child_mask)

    n_pos_scope = stats.tp + stats.fn
    n_neg_scope = stats.fp + stats.tn
    batch = getattr(data, "batch_cover_counts", None)
    children: List[Tuple[Rule, FrozenSet[int], Any, RuleStats]] = []
    dead: Set[int] = set()
    if batch is None:
        for added, (child_rule, child_mask) in by_feature.items():
            child_handle = data.refine_cover(handle, added)
            child_stats = stats_from_handle(data, target_class, child_rule, child_handle)
            if child_stats.tp == 0:
                dead.add(added)
            else:
                children.append((child_rule, child_mask, child_handle, child_stats))
        return children, dead

    features = list(by_feature.keys())
    tp_arr, fp_arr, _ = batch(handle, target_class, features)
    for f, tp, fp in zip(features, tp_arr, fp_arr):
        child_rule, child_mask = by_feature[f]
        if tp == 0:
            dead.add(f)
            continue
        child_handle = data.refine_cover(handle, f)
        fn, tn = n_pos_scope - tp, n_neg_scope - fp
        children.append((child_rule, child_mask, child_handle, RuleStats(tp=tp, fp=fp, fn=fn, tn=tn, length=child_rule.length())))
    return children, dead


def count_open_children(
    data, target_class: Any, mask: FrozenSet[int], handle, stats: RuleStats,
) -> Tuple[List[int], Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any], np.ndarray]:
    """``(features, tp, fp, fn, tn, same)`` for the one-literal child on every
    feature in `mask`, in sorted feature order (`Rule.specialize`'s own),
    *without* building any child -- no `Rule`, no closure, no mask, no
    handle. The lazy counterpart of `score_children`: a search that only
    ever moves to one (or a few) children per node scores them all from
    these counts, then builds just the ones it picks via
    `materialize_child`. Profiled on `sonar` (1080 features), building
    every child eagerly -- `extend_closure` and a mask copy per child --
    was ~95% of a single-lineage search's time, the counting itself ~2%.

    One `data.batch_cover_counts` matmul when the representation has it,
    otherwise one `refine_cover` + `cover_counts` per feature. Features
    `Rule.specialize` would drop as contradictory are counted too; the
    caller skips them when `materialize_child` returns None.

    `same` marks the children that cover exactly the rule's own rows
    (those with a nonzero weight): their counts are set to the rule's
    `stats` exactly, and a search must score them exactly as the rule
    itself -- never better. Their weighted counts are mathematically
    equal anyway, but summed over a different set of rows they can round
    differently, and a search would then add a condition that changes
    nothing. Known exactly from `batch_cover_counts`, and for unweighted
    data (integer counts) from any representation; weighted data on a
    representation without `batch_cover_counts` gets no marks yet.
    """
    features = sorted(mask)
    if not features:
        return features, [], [], [], [], np.zeros(0, dtype=bool)
    batch = getattr(data, "batch_cover_counts", None)
    if batch is not None:
        tp, fp, same = batch(handle, target_class, features)
        fn = (stats.tp + stats.fn) - tp
        tn = (stats.fp + stats.tn) - fp
    else:
        counts = [data.cover_counts(data.refine_cover(handle, f), target_class) for f in features]
        tp, fp, fn, tn = (np.asarray(col) for col in zip(*counts))
        exact = all(isinstance(v, (int, np.integer)) for v in (stats.tp, stats.fp)) and tp.dtype.kind in "iu"
        same = (tp + fp == stats.tp + stats.fp) if exact else np.zeros(len(features), dtype=bool)
    if same.any():
        tp, fp, fn, tn = (np.where(same, s, a).astype(np.result_type(a, s))
                          for a, s in ((tp, stats.tp), (fp, stats.fp), (fn, stats.fn), (tn, stats.tn)))
    return features, tp, fp, fn, tn, same


def live_open_children(
    data, target_class: Any, mask: FrozenSet[int], handle, stats: RuleStats,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Set[int]]:
    """`count_open_children` as arrays, without the dead children: the
    ``(features, tp, fp, fn, tn, same)`` of every open child covering at
    least one positive, plus the set of `dead` features (``tp == 0``). A
    dead feature stays dead in every descendant -- coverage only shrinks
    -- so a search masks it out of the child it moves to."""
    features, tps, fps, fns, tns, same = count_open_children(data, target_class, mask, handle, stats)
    features = np.asarray(features, dtype=np.int64)
    tps, fps, fns, tns = (np.asarray(a) for a in (tps, fps, fns, tns))
    alive = tps != 0
    dead = set(features[~alive].tolist())
    return (features[alive], tps[alive], fps[alive], fns[alive], tns[alive], np.asarray(same)[alive], dead)


def rank_best_first(scores: Any, *ties: Any) -> np.ndarray:
    """Indices ordering candidates best-first: by `scores` (a
    `batch_score` result) descending -- compared lexicographically when
    `LEF`-shaped (a tuple of arrays, or a 2-D array's columns) -- then by
    each of `ties` ascending, then in their original order (stable)."""
    if isinstance(scores, tuple):
        cols = [np.asarray(c, dtype=float) for c in scores]
    else:
        s = np.asarray(scores, dtype=float)
        cols = [s] if s.ndim == 1 else list(s.T)
    # np.lexsort's *last* key is the primary one
    return np.lexsort([np.asarray(t) for t in reversed(ties)] + [-c for c in reversed(cols)])


def set_score(scores: Any, i: int, score: Score) -> None:
    """Set candidate `i`'s score in a `batch_score` result, in place --
    for a child that keeps all its parent's rows (`count_open_children`'s
    `same`), which must score exactly as the parent."""
    if isinstance(scores, tuple):
        for column, s in zip(scores, score):
            column[i] = s
    else:
        scores[i] = score


def score_at(scores: Any, i: int) -> Score:
    """Candidate `i`'s score out of a `batch_score` result, as a plain
    float (a tuple of floats when `LEF`-shaped)."""
    if isinstance(scores, tuple):
        return tuple(float(c[i]) for c in scores)
    s = np.asarray(scores)
    return float(s[i]) if s.ndim == 1 else tuple(float(x) for x in s[i])


def batch_score_one(heuristic: RuleHeuristic, stats: RuleStats, *parent_stats: RuleStats) -> Score:
    """`heuristic`'s score for the one rule `stats`, computed through
    `batch_score` -- for anything a search compares against `batch_score`
    values (a parent's own score, the running best), so both sides are
    computed by the same code (see `RuleHeuristic`'s docstring)."""
    v = heuristic.batch_score(stats, *parent_stats)
    if isinstance(v, tuple):
        return tuple(float(np.asarray(c).reshape(-1)[0]) for c in v)
    a = np.asarray(v)
    return float(a) if a.ndim == 0 else tuple(float(x) for x in a.reshape(-1))


#: "closure not computed yet" -- distinct from None, which is a computed
#: closure under no constraints
_UNSET = object()


def parent_closure(dataspec, rule: Rule):
    """`rule`'s own closure under `dataspec`'s constraints, for
    `materialize_child` to extend -- None when there are no constraints.
    Raises `ValueError` if `rule` is itself contradictory. Only needed
    for a search's seed: every child `materialize_child` builds comes
    with its own closure, which the search passes on instead."""
    if dataspec is None or not dataspec.constraints:
        return None
    return dataspec.propagate({lit.feature: True for lit in rule.conditions})


def materialize_child(
    data, dataspec, rule: Rule, closure, mask: FrozenSet[int], feature: int, handle,
) -> Optional[Tuple[Rule, FrozenSet[int], Any, Any]]:
    """The ``(child_rule, child_mask, child_handle, child_closure)``
    `Rule.specialize` would yield for adding `feature` to `rule`
    (`closure`: `rule`'s own, from `parent_closure` or an earlier
    `materialize_child`), or None where `specialize` would skip it as
    contradicting `dataspec`'s constraints. ``handle=None`` skips the
    cover refinement (``child_handle`` is then None too)."""
    if closure is not None:
        try:
            child_closure = dataspec.extend_closure(closure, {feature: True})
        except ValueError:
            return None
        determined = set(child_closure)
    else:
        child_closure, determined = None, {feature}
    child = Rule(
        rule.conditions + (Literal(feature),), target=rule.target, dataspec=dataspec,
        n_features=rule.n_features, default_fmt=rule.default_fmt,
    )
    child_handle = None if handle is None else data.refine_cover(handle, feature)
    return child, mask - determined, child_handle, child_closure


def _optimistic_stats(stats: RuleStats) -> RuleStats:
    """The best any refinement of a `(tp, fp)` rule could possibly cover:
    every true positive kept, every false positive shed -- `(tp, 0)`.
    `fn`/`tn` shift to hold `n_pos`/`n_neg` fixed; `length` stays at the
    parent's (a real refinement is at least one condition longer, so a
    length-penalized score here comes out no lower than anything actually
    reachable -- deliberately an upper bound). Shared by every search's
    `optimistic_pruning` (`BeamSearch`, `HillClimbing`, and
    `GainAscentHillClimbing`)."""
    return RuleStats(
        tp=stats.tp, fp=0, fn=stats.fn, tn=stats.fp + stats.tn, length=stats.length
    )


# ============================================================ objectives ====
#
# What a search maximizes. A search never calls a heuristic directly: it
# asks an `Objective` for a rule's statistics, for every open child's
# statistics at once, for their scores and for optimistic bounds. A
# `RuleHeuristic` is one objective (`HeuristicObjective`: class counts,
# `RuleStats`); a function of sums of arbitrary per-row values is another
# (`ValueSumObjective`: e.g. a boosting learner's gradients). Searches
# accept either -- `as_objective` wraps a heuristic.


class Objective(ABC):
    """What a rule search maximizes -- building block 1's counterpart of
    a heuristic, general enough for criteria that aren't functions of
    class counts (`ValueSumObjective`). A search uses only these
    operations, on *statistics* whose type the objective chooses
    (`RuleStats` for `HeuristicObjective`):

    - `rule_stats(data, target, rule, handle)` -- one rule's statistics;
    - `children(data, target, mask, handle, stats)` -- every open
      one-literal child's statistics at once, without building a child:
      ``(features, batch, same, dead)``, where `same` marks children that
      keep every contributing row (they must score exactly as the
      parent) and `dead` holds features whose child has nothing left to
      gain from (dropped, and masked out further down);
    - `batch_score(batch, parent)`, `score(stats)`, `score_one(stats)` --
      scores, higher is better (`score_one` computes one score the way
      `batch_score` does, for comparing with batch values);
    - `bound(stats)`, `bound_one(stats)`, `batch_bound(batch)` -- an
      upper bound on the score of any refinement (`supports_bound`);
    - `exact_bound(data, handle)` -- optionally a tighter bound from the
      covered rows themselves, for a search that has built the child
      anyway (`BranchAndBoundSearch`);
    - `take(batch, index)`, `at(batch, i)`, `concat(batches)` --
      selecting and combining batches;
    - `dead(stats)`, `consistent(stats)` -- nothing to gain any more /
      nothing left to remove (the `fp == 0` of a heuristic).

    `filtering`/`stopping` criteria receive the objective's statistics,
    so they need `RuleStats` (`supports_criteria`)."""

    #: whether `bound*` are admissible upper bounds
    supports_bound: bool = True
    #: whether the statistics are `RuleStats`, as `PrePruningCriterion`s need
    supports_criteria: bool = False

    def begin(self, data, target_class: Any, example_mask: Optional[np.ndarray]) -> None:
        """Called once at the start of a search, before anything else."""

    @abstractmethod
    def rule_stats(self, data, target_class: Any, rule: Rule, handle) -> Any:
        raise NotImplementedError

    @abstractmethod
    def children(self, data, target_class: Any, mask: FrozenSet[int], handle, stats) -> Tuple[np.ndarray, Any, np.ndarray, Set[int]]:
        raise NotImplementedError

    @abstractmethod
    def batch_score(self, batch, parent=None) -> Any:
        raise NotImplementedError

    def score(self, stats) -> Score:
        return self.score_one(stats)

    @abstractmethod
    def score_one(self, stats, parent=None) -> Score:
        raise NotImplementedError

    @abstractmethod
    def batch_bound(self, batch) -> Any:
        raise NotImplementedError

    def bound(self, stats) -> Score:
        return self.bound_one(stats)

    @abstractmethod
    def bound_one(self, stats) -> Score:
        raise NotImplementedError

    def exact_bound(self, data, handle) -> Optional[Score]:
        return None

    @abstractmethod
    def take(self, batch, index) -> Any:
        raise NotImplementedError

    @abstractmethod
    def at(self, batch, i: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def concat(self, batches: Sequence[Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def dead(self, stats) -> bool:
        raise NotImplementedError

    def consistent(self, stats) -> bool:
        return False


class HeuristicObjective(Objective):
    """A `RuleHeuristic` as an `Objective`: statistics are `RuleStats`
    (class counts for the search's target class), children are counted by
    `count_open_children` (one `batch_cover_counts` per node), the bound
    is the heuristic's own (`RuleHeuristic.bound`; by default the score
    at the ``(tp, 0)`` projection), and a child is dead once it covers no
    positive. Exactly what the searches did with a heuristic before
    objectives existed; every search wraps a heuristic in one
    (`as_objective`)."""

    supports_criteria = True

    def __init__(self, heuristic: RuleHeuristic):
        self.heuristic = heuristic

    @property
    def supports_bound(self) -> bool:
        return not isinstance(self.heuristic, GainHeuristic)

    def rule_stats(self, data, target_class, rule, handle) -> RuleStats:
        return stats_from_handle(data, target_class, rule, handle)

    def children(self, data, target_class, mask, handle, stats):
        features, tps, fps, fns, tns, same, dead = live_open_children(data, target_class, mask, handle, stats)
        return features, RuleStats(tp=tps, fp=fps, fn=fns, tn=tns, length=stats.length + 1), same, dead

    def batch_score(self, batch, parent=None):
        if parent is not None and isinstance(self.heuristic, GainHeuristic):
            return self.heuristic.batch_score(batch, parent)
        return self.heuristic.batch_score(batch)

    def score(self, stats) -> Score:
        return self.heuristic.score(stats)

    def score_one(self, stats, parent=None) -> Score:
        if parent is not None and isinstance(self.heuristic, GainHeuristic):
            return batch_score_one(self.heuristic, stats, parent)
        return batch_score_one(self.heuristic, stats)

    def batch_bound(self, batch):
        return self.heuristic.batch_bound(batch)

    def bound(self, stats) -> Score:
        return self.heuristic.bound(stats)

    def bound_one(self, stats) -> Score:
        v = np.asarray(self.heuristic.batch_bound(stats))
        return float(v) if v.ndim == 0 else tuple(float(x) for x in v.reshape(-1))

    def take(self, batch, index):
        length = batch.length if np.ndim(batch.length) == 0 else np.asarray(batch.length)[index]
        return RuleStats(tp=batch.tp[index], fp=batch.fp[index], fn=batch.fn[index], tn=batch.tn[index],
                         length=length)

    def at(self, batch, i: int) -> RuleStats:
        length = batch.length if np.ndim(batch.length) == 0 else int(np.asarray(batch.length)[i])
        return RuleStats(tp=batch.tp[i], fp=batch.fp[i], fn=batch.fn[i], tn=batch.tn[i], length=length)

    def concat(self, batches):
        lengths = [np.full(len(np.atleast_1d(b.tp)), b.length) if np.ndim(b.length) == 0 else np.asarray(b.length)
                   for b in batches]
        return RuleStats(tp=np.concatenate([b.tp for b in batches]), fp=np.concatenate([b.fp for b in batches]),
                         fn=np.concatenate([b.fn for b in batches]), tn=np.concatenate([b.tn for b in batches]),
                         length=np.concatenate(lengths))

    def dead(self, stats) -> bool:
        return stats.tp == 0

    def consistent(self, stats) -> bool:
        return stats.fp == 0

    def __repr__(self) -> str:
        return f"HeuristicObjective({self.heuristic!r})"


def as_objective(heuristic: Union[RuleHeuristic, Objective]) -> Objective:
    """`heuristic` itself if it's already an `Objective`, else wrapped in a
    `HeuristicObjective`."""
    return heuristic if isinstance(heuristic, Objective) else HeuristicObjective(heuristic)


def _check_criteria(objective: Objective, *criteria: Optional[PrePruningCriterion]) -> None:
    if not objective.supports_criteria and any(c is not None for c in criteria):
        raise ValueError(f"filtering/stopping criteria need RuleStats; {type(objective).__name__} "
                         "has other statistics")


@dataclass(frozen=True)
class ValueSums:
    """`ValueSumObjective`'s statistics: the per-row value columns summed
    over a rule's covered rows (`sums`, ``m`` -- or ``m x k`` for a batch
    of `k` rules), how many of them contribute (a row with a nonzero
    value; an exact integer, unlike the sums) and the rule length."""
    sums: np.ndarray
    count: Any
    length: Any


class ValueSumObjective(Objective):
    """An objective that is a function of sums of per-row values over the
    covered rows -- the general case behind `HeuristicObjective`'s class
    counts. A subclass gives the values (`columns`, ``n x m``, set before
    the search: they may change from one search to the next, like a
    boosting learner's gradients), the score of their sums
    (`score_sums`) and a bound from the sums (`bound_sums`, an upper
    bound on the score of every subset of the covered rows -- what a
    refinement can reach); optionally `exact_bound` from the covered
    rows. Children are summed with one `batch_cover_sums` per node, so
    every data representation works.

    A row contributes if any of its values is nonzero; the exact count
    of contributing covered rows decides what changes nothing (a child
    keeping all of them, `same`, gets exactly its parent's sums -- summed
    over other rows they could round differently) and what is dead (none
    left). The objective doesn't depend on a target class; criteria
    (`filtering`/`stopping`) aren't supported."""

    @abstractmethod
    def columns(self) -> np.ndarray:
        """``n x m`` per-row values (``n`` = all rows of the data)."""
        raise NotImplementedError

    @abstractmethod
    def score_sums(self, sums: np.ndarray, length: Any) -> np.ndarray:
        """Scores from ``m x k`` sums (and lengths), one per column."""
        raise NotImplementedError

    @abstractmethod
    def bound_sums(self, sums: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def begin(self, data, target_class, example_mask) -> None:
        values = np.asarray(self.columns(), dtype=float)
        contributing = (values != 0).any(axis=1)
        self._values = np.column_stack([values, contributing.astype(float)])

    def rule_stats(self, data, target_class, rule, handle) -> ValueSums:
        rows = data.cover_rows(handle)
        total = self._values[rows].sum(axis=0)
        return ValueSums(total[:-1], int(round(total[-1])), rule.length())

    def children(self, data, target_class, mask, handle, stats: ValueSums):
        features = np.asarray(sorted(mask), dtype=np.int64)
        if not len(features):
            empty = np.zeros((len(stats.sums), 0))
            return features, ValueSums(empty, np.zeros(0, dtype=np.int64), stats.length + 1), np.zeros(0, bool), set()
        totals = np.asarray(data.batch_cover_sums(handle, self._values, features.tolist()), dtype=float)
        sums, count = totals[:-1], np.rint(totals[-1]).astype(np.int64)
        same = count == stats.count
        if same.any():
            sums[:, same] = stats.sums[:, None]
        alive = count != 0
        dead = set(features[~alive].tolist())
        return (features[alive], ValueSums(sums[:, alive], count[alive], stats.length + 1),
                same[alive], dead)

    def batch_score(self, batch: ValueSums, parent=None):
        return self.score_sums(batch.sums, batch.length)

    def score_one(self, stats: ValueSums, parent=None) -> Score:
        return float(np.asarray(self.score_sums(stats.sums[:, None], stats.length)).reshape(-1)[0])

    def batch_bound(self, batch: ValueSums):
        return self.bound_sums(batch.sums)

    def bound_one(self, stats: ValueSums) -> Score:
        return float(np.asarray(self.bound_sums(stats.sums[:, None])).reshape(-1)[0])

    def take(self, batch: ValueSums, index) -> ValueSums:
        length = batch.length if np.ndim(batch.length) == 0 else np.asarray(batch.length)[index]
        return ValueSums(batch.sums[:, index], np.asarray(batch.count)[index], length)

    def at(self, batch: ValueSums, i: int) -> ValueSums:
        length = batch.length if np.ndim(batch.length) == 0 else int(np.asarray(batch.length)[i])
        return ValueSums(batch.sums[:, i], int(np.asarray(batch.count)[i]), length)

    def concat(self, batches):
        lengths = [np.full(b.sums.shape[1], b.length) if np.ndim(b.length) == 0 else np.asarray(b.length)
                   for b in batches]
        return ValueSums(np.concatenate([b.sums for b in batches], axis=1),
                         np.concatenate([np.asarray(b.count) for b in batches]), np.concatenate(lengths))

    def dead(self, stats: ValueSums) -> bool:
        return stats.count == 0


@dataclass(frozen=True)
class HeadStats(RuleStats):
    """`HeadObjective`'s statistics: the `RuleStats` of a rule's best
    head -- so that `filtering`/`stopping` criteria read them like any
    other -- plus its covered examples per class (`covered`, ``k``, or
    ``k x m`` for a batch), how many in-scope rows it covers (`count`,
    exact), the head (`head`, an index into `HeadObjective.classes`) and
    its score."""
    covered: Any = None
    count: Any = 0
    head: Any = 0
    score: Any = 0.0


class HeadObjective(Objective):
    """A `RuleHeuristic` as an objective for a rule of *any* class -- the
    search of a decision list, where the head isn't fixed beforehand
    (`target_class` None) but follows the covered examples. Each rule is
    counted per class, and the heuristic scores it for its best head
    (`RuleHeuristic.best_head`), so the head can change from a rule to
    its refinement. The searches run unchanged on it; the head of the
    rule they return is `head_of`.

    Counting is one `batch_cover_sums` per node over one indicator column
    per class (weighted by the scope), so every data representation works.
    The statistics are `HeadStats`, the best head's `RuleStats`, so
    `filtering`/`stopping` criteria apply to the head the rule would get.
    A gain heuristic gains over the parent's statistics for the same
    head, which makes heads comparable only at the first refinement,
    where all of them start from the scope: there the head is chosen, and
    then it stays. A rule is dead once it covers no example in scope, and
    consistent once all it covers are of its head's class."""

    supports_criteria = True

    def __init__(self, heuristic: RuleHeuristic):
        self.heuristic = heuristic
        self._gain = isinstance(heuristic, GainHeuristic)

    @property
    def supports_bound(self) -> bool:
        return not self._gain

    def begin(self, data, target_class, example_mask) -> None:
        if data.y is None:
            raise ValueError("a search for a rule of any class needs data.y")
        self.classes, index = np.unique(np.asarray(data.y), return_inverse=True)
        weights = data.scope_weights(example_mask)
        values = np.zeros((data.n_samples, len(self.classes) + 1))
        values[np.arange(data.n_samples), index.ravel()] = weights
        values[:, -1] = weights > 0
        self._values = values
        self.totals = values[:, :-1].sum(axis=0)

    def head_of(self, stats: HeadStats) -> Any:
        """The class `stats`' rule predicts."""
        return self.classes[int(stats.head)]

    def _stats(self, covered: np.ndarray, count: Any, length: Any, parent: Optional[HeadStats]) -> HeadStats:
        if self._gain and parent is None:   # a root has no parent to gain over: the covered majority
            head = covered.argmax(axis=0)
            score = np.zeros(covered.shape[1:])
        elif self._gain and np.ndim(parent.length) == 0 and parent.length > 0:
            # a gain is over the parent *for the same head*: comparing heads
            # is fair only from the root, where every head starts from its
            # prior -- further down, a rare head the parent covers at a low
            # precision would show a large gain from a handful of examples.
            # So the head chosen at the first refinement stays.
            head = np.full(covered.shape[1:], int(parent.head), dtype=np.int64)
            score = self.heuristic.head_scores(covered, self.totals, length, parent.covered)[int(parent.head)]
        else:
            score, head = self.heuristic.best_head(covered, self.totals, length,
                                                   parent.covered if self._gain else None)
        pick = lambda a: np.take_along_axis(a, np.asarray(head)[None, ...], axis=0)[0]   # noqa: E731
        tp = pick(covered)
        n_covered = covered.sum(axis=0)
        n_pos = self.totals[head]
        fp = n_covered - tp
        return HeadStats(tp=tp, fp=fp, fn=n_pos - tp, tn=(self.totals.sum() - n_pos) - fp, length=length,
                         covered=covered, count=count, head=head, score=score)

    def rule_stats(self, data, target_class, rule, handle) -> HeadStats:
        total = self._values[data.cover_rows(handle)].sum(axis=0)
        return self._stats(total[:-1], int(round(total[-1])), rule.length(), None)

    def children(self, data, target_class, mask, handle, stats: HeadStats):
        features = np.asarray(sorted(mask), dtype=np.int64)
        if not len(features):
            empty = np.zeros((len(self.classes), 0))
            return features, self._stats(empty, np.zeros(0, dtype=np.int64), stats.length + 1, stats), \
                np.zeros(0, bool), set()
        totals = np.asarray(data.batch_cover_sums(handle, self._values, features.tolist()), dtype=float)
        covered, count = totals[:-1], np.rint(totals[-1]).astype(np.int64)
        same = count == stats.count
        if same.any():
            covered[:, same] = stats.covered[:, None]
        alive = count != 0
        dead = set(features[~alive].tolist())
        return (features[alive], self._stats(covered[:, alive], count[alive], stats.length + 1, stats),
                same[alive], dead)

    def batch_score(self, batch: HeadStats, parent=None):
        return np.asarray(batch.score, dtype=float)

    def score_one(self, stats: HeadStats, parent=None) -> Score:
        return float(stats.score)

    def batch_bound(self, batch: HeadStats):
        return self.heuristic.head_bound(batch.covered, self.totals, batch.length)

    def bound_one(self, stats: HeadStats) -> Score:
        return float(self.heuristic.head_bound(stats.covered, self.totals, stats.length))

    def take(self, batch: HeadStats, index) -> HeadStats:
        length = batch.length if np.ndim(batch.length) == 0 else np.asarray(batch.length)[index]
        return HeadStats(tp=batch.tp[index], fp=batch.fp[index], fn=batch.fn[index], tn=batch.tn[index],
                         length=length, covered=batch.covered[:, index], count=np.asarray(batch.count)[index],
                         head=np.asarray(batch.head)[index], score=np.asarray(batch.score)[index])

    def at(self, batch: HeadStats, i: int) -> HeadStats:
        length = batch.length if np.ndim(batch.length) == 0 else int(np.asarray(batch.length)[i])
        return HeadStats(tp=float(batch.tp[i]), fp=float(batch.fp[i]), fn=float(batch.fn[i]),
                         tn=float(batch.tn[i]), length=length, covered=batch.covered[:, i],
                         count=int(np.asarray(batch.count)[i]), head=int(np.asarray(batch.head)[i]),
                         score=float(np.asarray(batch.score)[i]))

    def concat(self, batches):
        lengths = [np.full(b.covered.shape[1], b.length) if np.ndim(b.length) == 0 else np.asarray(b.length)
                   for b in batches]
        return HeadStats(
            tp=np.concatenate([b.tp for b in batches]), fp=np.concatenate([b.fp for b in batches]),
            fn=np.concatenate([b.fn for b in batches]), tn=np.concatenate([b.tn for b in batches]),
            length=np.concatenate(lengths), covered=np.concatenate([b.covered for b in batches], axis=1),
            count=np.concatenate([np.asarray(b.count) for b in batches]),
            head=np.concatenate([np.asarray(b.head) for b in batches]),
            score=np.concatenate([np.asarray(b.score) for b in batches]))

    def dead(self, stats: HeadStats) -> bool:
        return stats.count == 0

    def consistent(self, stats: HeadStats) -> bool:
        return stats.fp == 0

    def __repr__(self) -> str:
        return f"HeadObjective({self.heuristic!r})"


class BeamSearch(RuleSearch):
    """Standard top-down beam search: starting from `initial_candidates`,
    repeatedly specialize every rule currently in the beam by one
    literal (`Rule.specialize`), keep the `beam_width` highest-scoring
    results, and stop once no further specialization is possible, no
    beam member still looks capable of beating the best rule found so
    far (`optimistic_pruning`, on by default -- see below), or
    `max_conditions` refinement steps have happened (if given). Returns
    the single best-scoring rule seen at any point, not just whatever
    ends up in the final beam -- an earlier, shorter rule can outscore
    every one of its own descendants (ties broken toward the shorter
    rule, so a later-round rediscovery of an already-optimal candidate
    via extra, non-restricting conditions never displaces the shorter
    original in the beam). With `filtering` configured, only a
    filter-passing rule is ever eligible to be that best -- if none was
    reached, the search returns `None` (see `filtering`, below).

    `optimistic_pruning` (default `True`) bounds the search the way
    Fürnkranz's SeCo survey describes: a rule covering `tp` positives
    and `fp` negatives can, at very best, refine into one covering those
    same `tp` positives and *no* negatives, so if the heuristic's score
    at that hypothetical `(tp, 0)` point doesn't exceed the best score
    found so far, no refinement of this candidate ever will and it isn't
    specialized. This is checked against each beam member just before it
    would be expanded for the next round; once it holds for *every* rule
    in the beam at once, the whole search stops.

    "Best score found so far" means the best score of a rule that has
    actually passed `filtering` (`best_rule`'s score) -- that's the best
    result the search could return, so it's the only sound bound. With
    no `filtering` configured, that's just the plain running best. While
    nothing has passed `filtering` yet there's no bound at all and
    nothing is pruned -- so a strict `filtering` criterion makes
    optimistic pruning kick in later (a lower filtered-best is a weaker
    bound), never earlier, and can never cause the search to skip a
    branch it would otherwise have kept.

    This replaces an older unconditional "stop once some beam member has
    `fp == 0`" rule -- a pure rule's own optimistic bound is just its
    own score, which the running best already reflects, so the general
    check subsumes it -- and, on searches over interval-binarized
    numeric features, cuts out the long tail of ever-longer duplicate
    rules at the same coverage that used to pad the run out after the
    real answer was already in hand. It assumes the heuristic rewards
    covering more positives and fewer negatives (`Precision`, `Laplace`,
    `WRAcc`, `Correlation`, `Accuracy`, ... all do); turn it off for one
    that doesn't (e.g. `Entropy`, symmetric in the covered-class ratio).
    The `tp == 0` floor is separate and always on -- a rule covering no
    positives is degenerate for `target_class`, not merely unpromising,
    and every one of its descendants stays that way.

    `beam_width=1` is close to `HillClimbing` over an ordinary
    `RuleHeuristic` (`HillClimbing` adds a local-maximum stop and drops
    the running-best bookkeeping a single lineage doesn't need) -- but a
    `GainHeuristic` (e.g. `FoilGain`) can't be used here at all, at any
    `beam_width`: its score is only meaningful relative to its *own*
    parent, and as soon as more than one candidate is ever in flight
    (which `beam_width=1` doesn't prevent -- multiple children from the
    same round's single parent are still compared against each other,
    which is fine, but so is comparing scores computed against
    *different* parents the moment `initial_candidates` has more than
    one entry, or a future beam_width>1 round pools children from
    multiple beam members) there's no shared scale left to rank them
    on. Rather than carve out a narrow, guarded exception for the one
    case that happens to be safe, `GainHeuristic` support lives
    entirely in `GainAscentHillClimbing` instead, whose single lineage
    makes "only one parent, ever" a standing invariant rather than
    something that needs policing here.

    `filtering` never changes the beam itself -- every round's beam is
    selected purely by `heuristic`, regardless of `filtering`, and a
    candidate in the excluded region is still specialized like any
    other. What it changes is *which* candidate is eligible to be
    returned: each round, `best_rule` advances to the highest-scoring
    candidate that also has `filtering.accept(...)` (walking down that
    round's sorted candidates past any that don't, not just checking the
    very top one). A rule in the excluded region is never returned -- if
    the search never reaches a single filter-passing rule, it returns
    `None`, not a best-effort rejected one. `filtering` also feeds
    `optimistic_pruning`'s bound (see above): since only a
    filter-passing rule can be returned, only its score is a sound thing
    to prune against, so a strict `filtering` criterion only ever delays
    the optimistic cutoff, never brings it forward.

    `stopping` gates the result exactly as `filtering` does (a rule in
    its reject region is never returned), and *additionally* halts the
    search: `stopping.evaluate(...)` is checked once per round, from the
    first refinement onward (never against the seed), against that
    round's single best-scoring candidate (the same one that would seed
    next round's beam). The moment it fires, the search stops -- if
    `stopping.accept(...)` is also true for that candidate it's returned
    directly, otherwise the search returns `best_rule` as tracked so far
    (the best rule seen that was in every criterion's acceptable region;
    `None` if there wasn't one -- e.g. FOSSIL's `Correlation < 0.3`
    firing on the first refinement, before any rule reached 0.3, returns
    `None`, not that sub-threshold rule). "Best", not "last", since
    unlike `HillClimbing`, `BeamSearch` maintains that running best
    regardless. The only thing `stopping` does that `filtering` doesn't
    is quit early -- so it can miss a deeper rule that would have
    re-entered the acceptable region a few refinements later, which
    `filtering` (running the search out) would still find.
    """

    def __init__(
        self,
        beam_width: int = 5,
        max_conditions: Optional[int] = None,
        optimistic_pruning: bool = True,
    ):
        self.beam_width = beam_width
        self.max_conditions = max_conditions
        self.optimistic_pruning = optimistic_pruning

    def search(
        self,
        data: HorizontalBitmapRepresentation,
        target_class: Any,
        heuristic: RuleHeuristic,
        initial_candidates: Sequence[Tuple[Rule, FrozenSet[int]]],
        example_mask: Optional[np.ndarray] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
    ) -> Optional[Rule]:
        if not initial_candidates:
            raise ValueError("BeamSearch.search needs at least one initial candidate")
        dataspec = data.spec
        objective = as_objective(heuristic)
        _check_criteria(objective, filtering, stopping)
        objective.begin(data, target_class, example_mask)

        def promises_improvement(stats: RuleStats, threshold: Optional[Score]) -> bool:
            # optimistic pruning: if even the hypothetical perfect
            # refinement (tp, 0) can't out-score `threshold` -- the score
            # of the best filter-passing rule found so far, i.e. the best
            # result the search could actually return -- no actual
            # refinement of this candidate can either, so there's no
            # point specializing it. Subsumes the old unconditional
            # "stop once a beam member has fp == 0" floor (a pure rule's
            # own optimistic bound is just its own score, already
            # reflected in `threshold`). Assumes the heuristic rewards
            # covering more positives / fewer negatives -- see this
            # class's docstring.
            if not self.optimistic_pruning or threshold is None:
                return True
            # bound_one: `threshold` came from batch_score
            return objective.bound_one(stats) > threshold

        def is_eligible(rule: Rule, stats: RuleStats) -> bool:
            if objective.dead(stats):
                return False
            return filtering is None or filtering.accept(rule, stats, data, target_class, example_mask)

        # beam entries carry their own already-computed handle and stats:
        # the handle so the next round's specialize step can call
        # `refine_cover` on it instead of recomputing coverage for the
        # grown rule from scratch, the stats so scoring, the tp == 0
        # floor and the optimistic bound never need a second pass either --
        # and their constraint closure (`_UNSET` for the seeds, computed on
        # first use), which each child extends instead of recomputing, and
        # their score, which a child keeping all their rows inherits exactly
        beam: List[Tuple[Rule, FrozenSet[int], Any, RuleStats, Any, Score]] = []
        for rule, mask in initial_candidates:
            handle = handle_for(data, rule, example_mask)
            stats = objective.rule_stats(data, target_class, rule, handle)
            beam.append((rule, mask, handle, stats, _UNSET, objective.score_one(stats)))
        # best_rule tracks the best filtering-eligible rule seen *strictly
        # before* the round currently being evaluated (see the stopping
        # block below for why the timing matters) -- if none is ever seen
        # it stays None and the search returns None. best_score starts at
        # None (not a -inf sentinel) rather than assuming a plain float --
        # a LEF's score is a tuple, and tuple > float raises. Length-0
        # rules (the universal seed) are skipped: the empty conjunction is
        # never itself a meaningful "found rule" -- SeCo, ReducedError-
        # Pruning etc. all treat it specially -- so a search that stops
        # before refining anywhere should return None, not the seed.
        best_rule: Optional[Rule] = None
        best_score: Optional[Score] = None
        for rule, mask, handle, stats, _, s in beam:   # s: compared with batch_score values below
            if rule.length() == 0:
                continue
            if is_eligible(rule, stats) and (best_score is None or s > best_score):
                best_rule, best_score = rule, s

        depth = 0
        while beam and (self.max_conditions is None or depth < self.max_conditions):
            # only specialize beam members that could still lead
            # somewhere: not degenerate (tp > 0 -- every descendant of a
            # tp == 0 rule also has tp == 0), and, under optimistic
            # pruning, still holding an optimistic (tp, 0) bound that
            # beats `best_score` -- the score of the best rule that has
            # actually passed `filtering` so far (== the plain running
            # best when there's no filtering; None, so no pruning, while
            # nothing has passed it yet). That's the best result the
            # search could return, so it's the only sound bound. When
            # *nothing* in the beam qualifies, the whole search is done
            # -- this is what stops a beam from cycling through
            # ever-longer, same-coverage duplicate rules long after the
            # real answer was found (common with interval-binarized
            # numeric features), the job the old "some beam member has
            # fp == 0" break used to do.
            refinable = [
                entry for entry in beam
                if not objective.dead(entry[3]) and promises_improvement(entry[3], best_score)
            ]
            if not refinable:
                break
            # every parent's children are scored from counts alone
            # (`count_open_children`); only the ones that make the beam --
            # plus, under `filtering`, any walked past while advancing
            # `best_rule` below -- are ever built (`materialize_child`).
            #
            # A child covering no positives is dropped (some heuristics score
            # one well -- Laplace gives (0, 0) 0.5 -- and in the beam it would
            # end the search, since it can't be refined). Its condition is
            # also masked out of that parent's other children: coverage only
            # shrinks under refinement, so it covers no positives further
            # down either.
            #
            # different beam parents can reach the same rule (e.g. "a"
            # refining by b and "b" refining by a both reach {a, b}) -- Rule
            # equality is its condition set (target is fixed here), so the
            # key is the parent's feature bitmask plus the added feature, no
            # Rule needed. Their masks may differ only in dead conditions
            # masked out under one parent but not the other -- either mask
            # is sound -- so collapsing duplicates into the first arrival
            # loses nothing, and is the only thing standing between this and
            # an exponential blow-up: each surviving duplicate would
            # otherwise re-explore the same subtree of further refinements
            # independently, every round. A duplicate is contradictory
            # exactly when the first arrival is (contradiction is a property
            # of the condition set), so deduplicating before that check is
            # sound too.
            seen_keys: Set[int] = set()
            dead_by_parent: List[Set[int]] = []
            parts: List[Tuple[np.ndarray, ...]] = []
            batches: List[Any] = []
            for pi, (rule, mask, handle, stats, _, _) in enumerate(refinable):
                features, batch, same, dead = objective.children(data, target_class, mask, handle, stats)
                dead_by_parent.append(dead)
                bits = 0
                for lit in rule.conditions:
                    bits |= 1 << lit.feature
                keep: List[int] = []
                for j, f in enumerate(features.tolist()):
                    key = bits | (1 << f)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        keep.append(j)
                k = np.asarray(keep, dtype=np.intp)
                parts.append((np.full(len(k), pi), features[k], np.full(len(k), rule.length() + 1), same[k]))
                batches.append(objective.take(batch, k))
            c_parent, c_feature, c_length, c_same = (
                np.concatenate([p[j] for p in parts]) for j in range(4)
            )
            c_stats = objective.concat(batches)
            scores = objective.batch_score(c_stats)
            for i in np.flatnonzero(c_same).tolist():     # changes nothing: exactly its parent's score
                set_score(scores, i, refinable[int(c_parent[i])][5])
            # ties broken toward the shorter (more general) rule -- otherwise
            # an arbitrary, sort-order-dependent longer duplicate could win a
            # beam slot over an equally-good shorter one; equal keys stay in
            # arrival order (stable)
            order = rank_best_first(scores, c_length).tolist()

            def candidate(i: int) -> Tuple[Score, int, int, RuleStats]:
                return score_at(scores, i), int(c_parent[i]), int(c_feature[i]), objective.at(c_stats, i)

            contradictory = object()
            closures: Dict[int, Any] = {pi: entry[4] for pi, entry in enumerate(refinable)}

            def build(pi: int, f: int, with_handle: bool):
                prule, pmask, phandle, _, _, _ = refinable[pi]
                if closures[pi] is _UNSET:  # a seed's: computed on first use
                    try:
                        closures[pi] = parent_closure(dataspec, prule)
                    except ValueError:
                        closures[pi] = contradictory  # no consistent child at all
                if closures[pi] is contradictory:
                    return None
                return materialize_child(
                    data, dataspec, prule, closures[pi], pmask, f, phandle if with_handle else None,
                )

            new_beam: List[Tuple[Rule, FrozenSet[int], Any, RuleStats, Any, Score]] = []
            built: List[Tuple[Score, Rule, RuleStats]] = []
            walked = 0
            while walked < len(order) and len(new_beam) < self.beam_width:
                s, pi, f, st = candidate(order[walked])
                walked += 1
                child = build(pi, f, with_handle=True)
                if child is None:
                    continue  # contradicts its parent: never a candidate
                crule, cmask, chandle, cclosure = child
                new_beam.append((crule, cmask - dead_by_parent[pi], chandle, st, cclosure, s))
                built.append((s, crule, st))
            if not new_beam:
                break
            beam = new_beam

            top_rule, top_stats = beam[0][0], beam[0][3]

            # stopping halts the search: checked against this round's top
            # candidate only (the same rule that would seed next round).
            # Checked *before* best_rule is advanced from this round, so
            # `best_rule` still reflects only rounds strictly before the
            # one that triggered -- when stopping fires because the
            # frontier crossed the threshold, the search hands back the
            # best rule from *before* things went bad, or None if there
            # wasn't one yet (e.g. FOSSIL's `Correlation < 0.3` firing on
            # the first refinement returns None, not that sub-threshold
            # rule).
            if stopping is not None and stopping.evaluate(
                top_rule, top_stats, data, target_class, example_mask
            ):
                if stopping.accept(top_rule, top_stats, data, target_class, example_mask):
                    return top_rule
                return best_rule  # None if nothing acceptable came before this round

            # advance best_rule to this round's highest-scoring eligible
            # candidate (walking past any that fail `filtering` -- past the
            # beam, if need be), not necessarily the top one
            eligible = next(((s, rule) for s, rule, st in built if is_eligible(rule, st)), None)
            while eligible is None and walked < len(order):
                s, pi, f, st = candidate(order[walked])
                walked += 1
                child = build(pi, f, with_handle=False)
                if child is not None and is_eligible(child[0], st):
                    eligible = (s, child[0])
            if eligible is not None and (best_score is None or eligible[0] > best_score):
                best_rule, best_score = eligible[1], eligible[0]

            depth += 1

        return best_rule  # None if nothing eligible was ever found


class HillClimbing(RuleSearch):
    """Single-lineage greedy search: each round, specialize the current
    rule, score every child with a plain `RuleHeuristic`, move to
    whichever scores highest, stop at a **local maximum of the
    heuristic** -- the moment no child's score exceeds the current
    rule's own. Same idea as `BeamSearch(beam_width=1)`, but with that
    local-maximum stop it halts as soon as the greedy ascent peaks
    rather than running the lineage out; on a landscape that rises
    monotonically to a single peak (the usual case for `Precision`/
    `Laplace`/`Correlation` on a conjunctive concept) the two return the
    same rule, `HillClimbing` just reaching it with less work. Kept as
    its own class because a single lineage needs no beam dedup and no
    separately-tracked running best ("the last rule reached" is also the
    highest-scoring, since the walk only ever moves to a strictly better
    one). The same-scale parent/child comparison the local-maximum stop
    makes is valid precisely because one lineage means both are always
    scored the same way. A `GainHeuristic` needs `GainAscentHillClimbing`
    instead (this class rejects one).

    `optimistic_pruning` (default `True`) is exactly `BeamSearch`'s bound
    of the same name, restated for a single lineage: the best any
    refinement of a `(tp, fp)` rule could cover is `(tp, 0)`, so if the
    heuristic's score at that projection can't beat the best rule the
    search could still return -- the most recent filter-eligible rule, or
    (with no `filtering`) just the current one -- no real child can
    either, so the walk stops without enumerating this round's children.
    It subsumes the old `fp == 0` floor (a pure rule's `(tp, 0)`
    projection is itself), and, for a plain heuristic that rewards more
    `tp` / fewer `fp`, it never changes the result -- only skips work the
    local-maximum stop would have reached the same conclusion from. Turn
    it off for a heuristic that isn't monotone that way (`Entropy`,
    symmetric in the covered-class ratio). `tp == 0` stays a separate,
    always-on floor.

    `filtering` never affects which move gets taken -- the walk always
    follows the highest-scoring child, exactly as without it. It only
    changes what's *eligible to be returned*: this class tracks the most
    recently visited rule with `filtering.accept(...)` (a non-empty seed
    counts; the length-0 universal seed never does), and falls back to
    it whenever the walk's natural endpoint isn't itself eligible. If
    the walk never visits a single filter-passing rule, the search
    returns `None` -- a configured filter is a hard gate, not a
    best-effort preference.

    `stopping` is checked from the first refinement onward (never
    against the seed) -- each round, once the walk has moved to a new
    rule, `stopping.evaluate(...)` is checked on it *before* that rule
    is recorded as the fallback. The moment it fires, `accept(...)` on
    that same rule decides what's returned: itself, if accepted,
    otherwise the most recent rule from *strictly before* this round
    that passed both `filtering` and `stopping`'s accept (`None` if
    there wasn't one -- so `stopping` never returns a rule from its own
    reject region).

    `stop_at_local_optimum` (default `True`) is the local-maximum stop
    described above. Set it `False` and the walk instead keeps adding the
    best-scoring condition -- even one that lowers the score -- toward
    consistency (`fp == 0`). It then stops early only where growth is
    genuinely blocked or bounded: no `tp`-preserving condition left,
    `max_conditions`, or a `stopping` criterion firing (e.g.
    `ThresholdPrePruning(CoveredPositives(), 2, "<")` -- RIPPER's
    `minNo`, keeping every rule on at least two positives). This is
    RIPPER's / IREP's grow phase, and it is **only useful paired with
    post-pruning** (a `SingleRulePostProcessing` like
    `ReducedErrorPruning`): on its own it just grows maximally specific,
    overfit rules. `optimistic_pruning` has no effect once this is
    `False` -- it exists to skip work the local-maximum stop would reach
    anyway, and there is no such stop then. Leave it `True` outside the
    grow-then-prune recipe.
    """

    def __init__(
        self,
        max_conditions: Optional[int] = None,
        optimistic_pruning: bool = True,
        stop_at_local_optimum: bool = True,
    ):
        self.max_conditions = max_conditions
        self.optimistic_pruning = optimistic_pruning
        self.stop_at_local_optimum = stop_at_local_optimum

    # -- hooks the gain-ascent subclass overrides; everything else in
    #    `search` is shared single-lineage machinery. The base versions
    #    below are the plain-heuristic behaviour. --------------------------

    def _reject_heuristic(self, objective: Objective) -> None:
        if isinstance(getattr(objective, "heuristic", None), GainHeuristic):
            raise ValueError(
                f"{type(self).__name__} needs a plain RuleHeuristic, got "
                f"{type(objective.heuristic).__name__} -- use GainAscentHillClimbing for a GainHeuristic"
            )

    def _child_scores(self, objective: Objective, children: Any, parent_stats: Any) -> Any:
        return objective.batch_score(children)

    def _improvement_threshold(self, objective: Objective, parent_stats: Any) -> Score:
        # the best child must strictly beat this to be worth moving to:
        # the current rule's own score (a local maximum of the objective),
        # computed the way the children's are -- a child with the parent's
        # exact stats must tie, not win by a rounding difference
        return objective.score_one(parent_stats)

    def _improves(self, value: Score, threshold: Score) -> bool:
        """Whether a child scoring `value` is worth moving to."""
        return value > threshold

    def _optimistic_stop(self, objective: Objective, stats: Any, best_stats: Any) -> bool:
        # BeamSearch's optimality bound: if the objective's bound (for a
        # heuristic, its score at the (tp, 0) projection) can't beat the
        # best returnable rule so far (`best_stats` -- None before any
        # exists), no child down this lineage will either. Subsumes the
        # old fp == 0 floor.
        if best_stats is None:
            return False
        return objective.bound(stats) <= objective.score(best_stats)

    def search(
        self,
        data: HorizontalBitmapRepresentation,
        target_class: Any,
        heuristic: RuleHeuristic,
        initial_candidates: Sequence[Tuple[Rule, FrozenSet[int]]],
        example_mask: Optional[np.ndarray] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
    ) -> Optional[Rule]:
        if not initial_candidates:
            raise ValueError(f"{type(self).__name__}.search needs at least one initial candidate")
        if len(initial_candidates) > 1:
            raise ValueError(
                f"{type(self).__name__}.search needs exactly one initial candidate -- "
                "it only ever tracks a single lineage"
            )
        objective = as_objective(heuristic)
        self._reject_heuristic(objective)
        _check_criteria(objective, filtering, stopping)
        objective.begin(data, target_class, example_mask)
        dataspec = data.spec

        def is_eligible(rule: Rule, stats: RuleStats) -> bool:
            # returnable only if in the acceptable region of every
            # configured criterion -- so `stopping` (like `filtering`)
            # never lets the search hand back a rule from its own reject
            # region; it just *also* halts the walk (see below).
            if objective.dead(stats):
                return False
            for crit in (filtering, stopping):
                if crit is not None and not crit.accept(
                    rule, stats, data, target_class, example_mask
                ):
                    return False
            return True

        rule, mask = initial_candidates[0]
        handle = handle_for(data, rule, example_mask)
        stats = objective.rule_stats(data, target_class, rule, handle)
        # what `stopping` falls back to: the most recent eligible rule
        # from *before* the step that triggered it. The length-0 seed
        # never counts -- a walk that's stopped before refining anywhere
        # returns None, not the empty rule.
        last_eligible: Optional[Tuple[Rule, RuleStats]] = (
            (rule, stats) if rule.length() > 0 and is_eligible(rule, stats) else None
        )
        closure: Any = _UNSET

        depth = 0
        while mask and (self.max_conditions is None or depth < self.max_conditions):
            if objective.dead(stats):
                break  # degenerate for target_class -- every descendant stays tp == 0

            if not self.stop_at_local_optimum and objective.consistent(stats):
                break  # the rule is consistent -- nothing left to add

            if self.stop_at_local_optimum and self.optimistic_pruning and self._optimistic_stop(
                objective, stats, last_eligible[1] if last_eligible is not None else None
            ):
                break

            threshold = self._improvement_threshold(objective, stats)

            # score every child from counts alone, in one batch_score call;
            # build only the one moved to
            features, batch, same, dead = objective.children(data, target_class, mask, handle, stats)
            values = self._child_scores(objective, batch, stats)
            for i in np.flatnonzero(same).tolist():   # changes nothing: exactly the rule's own score
                set_score(values, i, threshold)
            if closure is _UNSET:  # only the seed's: every child brings its own
                try:
                    closure = parent_closure(dataspec, rule)
                except ValueError:
                    break  # the rule itself is contradictory: no consistent child
            best_child: Optional[Tuple[Rule, FrozenSet[int], Any, Any, RuleStats]] = None
            # stable: equal scores stay in feature order, the child a strict `>` scan keeps
            for i in rank_best_first(values).tolist():
                if self.stop_at_local_optimum and not self._improves(score_at(values, i), threshold):
                    break  # local optimum -- no child beats the current rule
                built = materialize_child(data, dataspec, rule, closure, mask, int(features[i]), handle)
                if built is not None:
                    best_child = (*built, objective.at(batch, i))
                    break
            if best_child is None:
                break  # nothing left to move to, or a local optimum

            rule, mask, handle, closure, stats = best_child
            mask = mask - dead
            depth += 1

            if stopping is not None and stopping.evaluate(rule, stats, data, target_class, example_mask):
                if stopping.accept(rule, stats, data, target_class, example_mask):
                    return rule
                return last_eligible[0] if last_eligible is not None else None

            if is_eligible(rule, stats):
                last_eligible = (rule, stats)

        if is_eligible(rule, stats):
            return rule
        return last_eligible[0] if last_eligible is not None else None

    def search_all(
        self,
        data: HorizontalBitmapRepresentation,
        target_class: Any,
        heuristic: RuleHeuristic,
        initial_candidates: Sequence[Tuple[Rule, FrozenSet[int]]],
        example_mask: Optional[np.ndarray] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
    ) -> List[Rule]:
        """With `branch_similarity` (`GainAscentHillClimbing`): every
        lineage's rule, see `_branching_walk`. Otherwise `search`'s."""
        if getattr(self, "branch_similarity", None) is None:
            return super().search_all(data, target_class, heuristic, initial_candidates,
                                      example_mask=example_mask, filtering=filtering, stopping=stopping)
        if len(initial_candidates) != 1:
            raise ValueError(f"{type(self).__name__}.search_all needs exactly one initial candidate")
        objective = as_objective(heuristic)
        self._reject_heuristic(objective)
        _check_criteria(objective, filtering, stopping)
        objective.begin(data, target_class, example_mask)
        return self._branching_walk(data, target_class, objective, initial_candidates[0],
                                    example_mask, filtering, stopping)

    def _branching_walk(self, data, target_class, objective, initial, example_mask, filtering, stopping) -> List[Rule]:
        """The single-lineage walk of `search`, except that at every step
        it also follows each child scoring at least ``best *
        branch_similarity`` (and worth moving to at all) -- a copy of the
        lineage, grown on the same way. CPAR's search (Yin & Han 2003).
        Copies are grown before the best child, depth first, and a rule
        reached a second time isn't grown again. Each lineage ends where
        `search`'s walk would, with the same `filtering`/`stopping`
        fallback, and yields that rule (not the length-0 seed); the result
        is every lineage's rule, in the order found, without repeats."""
        dataspec = data.spec
        results: Dict[Rule, None] = {}
        seen: Set[Rule] = set()

        def is_eligible(rule: Rule, stats: RuleStats) -> bool:
            if objective.dead(stats):
                return False
            return all(crit is None or crit.accept(rule, stats, data, target_class, example_mask)
                       for crit in (filtering, stopping))

        def end(rule: Optional[Rule]) -> None:
            if rule is not None and rule.length() > 0:
                results.setdefault(rule, None)

        def walk(rule, mask, handle, closure, stats, last_eligible, depth) -> None:
            if rule in seen:
                return
            seen.add(rule)
            fallback = last_eligible[0] if last_eligible is not None else None
            natural_end = rule if is_eligible(rule, stats) else fallback
            if not mask or (self.max_conditions is not None and depth >= self.max_conditions):
                return end(natural_end)
            if objective.dead(stats):
                return end(natural_end)
            if not self.stop_at_local_optimum and objective.consistent(stats):
                return end(natural_end)
            if self.stop_at_local_optimum and self.optimistic_pruning and self._optimistic_stop(
                objective, stats, last_eligible[1] if last_eligible is not None else None
            ):
                return end(natural_end)
            threshold = self._improvement_threshold(objective, stats)
            features, batch, same, dead = objective.children(data, target_class, mask, handle, stats)
            values = self._child_scores(objective, batch, stats)
            for i in np.flatnonzero(same).tolist():
                set_score(values, i, threshold)
            if closure is _UNSET:
                try:
                    closure = parent_closure(dataspec, rule)
                except ValueError:
                    return end(natural_end)
            chosen = []
            cut = None
            for i in rank_best_first(values).tolist():
                value = score_at(values, i)
                if self.stop_at_local_optimum and not self._improves(value, threshold):
                    break
                if cut is not None and value < cut:
                    break
                built = materialize_child(data, dataspec, rule, closure, mask, int(features[i]), handle)
                if built is None:
                    continue  # contradicts the rule: never a candidate
                if cut is None:  # the best child: copies must come within branch_similarity of it
                    cut = value * self.branch_similarity if value > 0 else value
                chosen.append((built, objective.at(batch, i)))
            if not chosen:
                return end(natural_end)
            for (child, cmask, chandle, cclosure), cstats in chosen[1:] + chosen[:1]:   # copies first
                if stopping is not None and stopping.evaluate(child, cstats, data, target_class, example_mask):
                    end(child if stopping.accept(child, cstats, data, target_class, example_mask) else fallback)
                    continue
                child_last = (child, cstats) if is_eligible(child, cstats) else last_eligible
                walk(child, cmask - dead, chandle, cclosure, cstats, child_last, depth + 1)

        rule, mask = initial
        handle = handle_for(data, rule, example_mask)
        stats = objective.rule_stats(data, target_class, rule, handle)
        last_eligible = (rule, stats) if rule.length() > 0 and is_eligible(rule, stats) else None
        walk(rule, mask, handle, _UNSET, stats, last_eligible, 0)
        return list(results)


class GainAscentHillClimbing(HillClimbing):
    """`HillClimbing` for a `GainHeuristic` (e.g. `FoilGain`). Where the
    base class ascends a *global* per-rule objective and stops at its
    local maximum, this follows the *local gradient*: each child is
    scored by `heuristic.score(child_stats, current_stats)` -- a gain
    relative to the current rule, not an absolute score -- and the walk
    stops once the best available gain is non-positive. 0 is the only
    shared reference a gain has (two gains against different parents
    aren't the same question), but it's always a meaningful one: a
    refinement that doesn't beat its own parent is never worth taking.

    This is the only search that can run a `GainHeuristic` at all --
    `BeamSearch` can hold several candidates in flight scored against
    different parents, with no shared scale to rank them (see its
    docstring); the base `HillClimbing` rejects a `GainHeuristic`
    outright. Because a gain has no cross-round scale, "the last rule the
    walk reached" is the only well-defined answer -- there's no
    "best ever seen" to track (see `GainHeuristic`'s own docstring).

    `optimistic_pruning` (default `True`) is the base-class bound
    rephrased for a gain: the best any refinement of a `(tp, fp)` rule
    could cover is `(tp, 0)`, so if the gain *there* (against the current
    rule) is already non-positive, no real child improves on it -- stop
    without enumerating children. Subsumes the old `fp == 0` floor (a
    pure rule's `(tp, 0)` projection is itself, gain 0). A gain's
    reference is always the immediate parent, so unlike the base class
    this needs no "best returnable so far" to bound against.

    `stop_at_local_optimum=False` (inherited) turns this into RIPPER's
    grow phase: keep adding the max-`FoilGain` condition -- even past the
    point gain goes non-positive -- toward consistency (`max_conditions`
    and a `stopping` criterion still bound it). Only useful feeding a
    `ReducedErrorPruning` post-processor; see `HillClimbing`'s docstring.

    `min_gain` (default none) raises the bar from "any positive gain" to
    "a gain of at least `min_gain`" (CPAR's 0.7). It must be positive: a
    condition that changes nothing gains exactly 0.

    `branch_similarity` (default none) makes the search yield several
    rules (`search_all`; `search` still returns the best-gain lineage's
    rule alone): at every step it also follows each child whose gain is
    at least ``branch_similarity`` times the best one (and worth moving to
    at all), as a copy of the rule that is grown on the same way -- CPAR's
    search (Yin & Han 2003), usable in any `SeCo` learner. See
    `HillClimbing._branching_walk`.
    """

    def __init__(
        self,
        max_conditions: Optional[int] = None,
        optimistic_pruning: bool = True,
        stop_at_local_optimum: bool = True,
        min_gain: Optional[float] = None,
        branch_similarity: Optional[float] = None,
    ):
        super().__init__(max_conditions, optimistic_pruning, stop_at_local_optimum)
        if min_gain is not None and not min_gain > 0:
            raise ValueError(f"min_gain must be positive (a condition changing nothing gains 0), got {min_gain}")
        if branch_similarity is not None and not 0.0 < branch_similarity <= 1.0:
            raise ValueError(f"branch_similarity must be in (0, 1], got {branch_similarity}")
        self.min_gain = min_gain
        self.branch_similarity = branch_similarity

    def _improves(self, value: Score, threshold: Score) -> bool:
        return value > threshold if self.min_gain is None else value >= self.min_gain

    def _reject_heuristic(self, objective: Objective) -> None:
        heuristic = getattr(objective, "heuristic", objective)
        if not isinstance(heuristic, GainHeuristic):
            raise ValueError(
                f"GainAscentHillClimbing needs a GainHeuristic (e.g. FoilGain), got "
                f"{type(heuristic).__name__} -- use HillClimbing for a plain RuleHeuristic"
            )

    def _child_scores(self, objective: Objective, children: Any, parent_stats: Any) -> Any:
        return objective.batch_score(children, parent_stats)

    def _improvement_threshold(self, objective: Objective, parent_stats: Any) -> Score:
        return 0.0  # a gain's own shared reference point

    def _optimistic_stop(self, objective: Objective, stats: Any, best_stats: Any) -> bool:
        return not self._improves(objective.heuristic.score(_optimistic_stats(stats), stats), 0.0)


class _Descending:
    """Heap key ordering scores from highest to lowest -- for floats and for
    `LEF`'s tuples alike, which can't simply be negated."""

    __slots__ = ("value",)

    def __init__(self, value: Score):
        self.value = value

    def __lt__(self, other: "_Descending") -> bool:
        return other.value < self.value


class BranchAndBoundSearch(RuleSearch):
    """Exhaustive search for the best rule -- with `k`, the `k` best -- by
    branch and bound, after OPUS (Webb, "OPUS: An efficient admissible
    algorithm for unordered search", JAIR 1995). Unlike `BeamSearch` and
    the hill climbers, the rule it returns is optimal: no conjunction of
    the open conditions (up to `max_conditions` of them) scores higher.

    The search space is the set-enumeration tree over the open
    conditions: a rule is only ever extended by conditions after its last
    one in feature order, so every condition set is generated at most
    once. Nodes are expanded best-first, by their *bound* -- the score of
    the best refinement conceivable, every covered positive kept and every
    covered negative dropped (the ``(tp, 0)`` projection `BeamSearch`'s
    `optimistic_pruning` uses). A node whose bound can't beat the `k`-th
    best rule found so far is never expanded, and the search ends as soon
    as no node left can. The bound is only admissible for a heuristic that
    rewards covering more positives and fewer negatives (`Laplace`,
    `WRAcc`, `Precision`, `Accuracy`, ...; not e.g. `Entropy`); the
    heuristic must also score each rule on its own -- a `GainHeuristic`
    is refused.

    As in OPUS, a condition is removed from the whole subtree below a
    node (not just from its own branch) once it can't help there: when
    the child adding it is bounded out (any rule containing that child
    is bounded by it, too), when it covers no positive (nor does any
    refinement), or when it drops no covered row (it changes nothing).
    Constraint propagation does the rest (`Rule.specialize`): conditions
    implied by the rule, or contradicting it, are never offered.

    With `dominance_pruning` (default), Webb's *cannotImprove* rule
    removes more: a condition that drops no covered negative (the same
    rule without it covers the same negatives and at least the same
    positives, so it scores at least as high), and a condition whose
    child is dominated by a sibling's -- the sibling covers every
    positive it covers and none of the negatives it drops -- since
    swapping the condition for the sibling's never lowers a score.
    Dominated rules aren't worse, only no better, so this is only sound
    for the single best rule: it is skipped with `k > 1` and with a
    `filtering`/`stopping` criterion (the dominating rule might be one
    they reject).

    The `k` best rules (`search_all`; `search` returns the best) have
    pairwise different coverage -- a rule covering exactly the same rows
    as one already kept is skipped; among equal scores, the one found
    first is kept. `filtering` restricts which rules may be kept;
    `stopping` also stops expanding a rule it fires on. One batched count
    per expanded node, so it runs on every data representation.

    Exhaustive means exponential in the worst case: `max_conditions`
    bounds the depth, and pruning keeps typical searches far from it.
    """

    def __init__(self, max_conditions: Optional[int] = None, k: int = 1, dominance_pruning: bool = True):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.max_conditions = max_conditions
        self.k = k
        self.dominance_pruning = dominance_pruning

    def search(self, data, target_class, heuristic, initial_candidates, example_mask=None,
               filtering=None, stopping=None) -> Optional[Rule]:
        rules = self._search(data, target_class, heuristic, initial_candidates, example_mask, filtering, stopping)
        return rules[0] if rules else None

    def search_all(self, data, target_class, heuristic, initial_candidates, example_mask=None,
                   filtering=None, stopping=None) -> List[Rule]:
        return self._search(data, target_class, heuristic, initial_candidates, example_mask, filtering, stopping)

    def _search(self, data, target_class, heuristic, initial_candidates, example_mask, filtering, stopping) -> List[Rule]:
        objective = as_objective(heuristic)
        if not objective.supports_bound:
            raise ValueError(f"BranchAndBoundSearch needs an objective scoring each rule on its own, got "
                             f"{getattr(objective, 'heuristic', objective)!r} (a gain is only meaningful "
                             "against its parent, so it has no bound)")
        if len(initial_candidates) != 1:
            raise ValueError("BranchAndBoundSearch needs exactly one initial candidate")
        _check_criteria(objective, filtering, stopping)
        objective.begin(data, target_class, example_mask)
        dataspec = data.spec
        dominance = (self.dominance_pruning and self.k == 1 and filtering is None and stopping is None
                     and isinstance(objective, HeuristicObjective))
        positive = np.asarray(data.y) == target_class if dominance else None
        best: List[Tuple[Score, int, Rule, bytes]] = []        # the k best, highest first
        found = itertools.count()

        def eligible(rule: Rule, stats: RuleStats) -> bool:
            if objective.dead(stats) or rule.length() == 0:
                return False
            return all(crit is None or crit.accept(rule, stats, data, target_class, example_mask)
                       for crit in (filtering, stopping))

        def threshold() -> Optional[Score]:
            return best[-1][0] if len(best) == self.k else None

        def offer(score: Score, rule: Rule, handle) -> None:
            th = threshold()
            if th is not None and not score > th:
                return
            key = np.packbits(data.cover_rows(handle)).tobytes() if self.k > 1 else b""
            if self.k > 1 and any(key == kept[3] for kept in best):
                return                                         # same rows as a rule already kept
            position = next((i for i, kept in enumerate(best) if score > kept[0]), len(best))
            best.insert(position, (score, next(found), rule, key))
            del best[self.k:]

        rule, mask = initial_candidates[0]
        handle = handle_for(data, rule, example_mask)
        stats = objective.rule_stats(data, target_class, rule, handle)
        if eligible(rule, stats):
            offer(objective.score_one(stats), rule, handle)
        last = max((lit.feature for lit in rule.conditions), default=-1)
        queue = [(None, 0, rule, mask, handle, _UNSET, stats, last)]   # the seed is always expanded
        pushed = itertools.count(1)

        while queue:
            bound, _, rule, mask, handle, closure, stats, last = heapq.heappop(queue)
            th = threshold()
            if bound is not None and th is not None and not bound.value > th:
                break                                          # best-first: nothing left can enter the k best
            if self.max_conditions is not None and rule.length() >= self.max_conditions:
                continue
            if stopping is not None and rule.length() > 0 and stopping.evaluate(
                    rule, stats, data, target_class, example_mask):
                continue
            open_later = frozenset(f for f in mask if f > last)
            if not open_later:
                continue
            features, batch, same, dead = objective.children(data, target_class, open_later, handle, stats)
            length = rule.length() + 1
            scores = objective.batch_score(batch)
            bounds = objective.batch_bound(batch)
            if closure is _UNSET:
                try:
                    closure = parent_closure(dataspec, rule)
                except ValueError:
                    continue                                   # contradictory: no consistent refinement
            removed: Set[int] = set(dead) | set(features[same].tolist())
            children = []
            for i in np.flatnonzero(~same).tolist():
                f = int(features[i])
                score, child_bound = score_at(scores, i), score_at(bounds, i)
                th = threshold()
                if th is not None and not score > th and not child_bound > th:
                    removed.add(f)                             # neither it nor any refinement can enter
                    continue
                built = materialize_child(data, dataspec, rule, closure, mask, f, handle)
                if built is None:
                    removed.add(f)                             # contradicts the rule
                    continue
                child, child_mask, child_handle, child_closure = built
                child_stats = objective.at(batch, i)
                exact = objective.exact_bound(data, child_handle)
                if exact is not None:
                    child_bound = min(child_bound, exact)      # the covered rows bound more tightly
                if eligible(child, child_stats):
                    offer(score, child, child_handle)
                children.append((child_bound, f, child, child_mask, child_handle, child_closure, child_stats))
            if dominance and children:
                removed |= self._dominated(data, positive, handle, stats, children)
            th = threshold()
            if th is not None:
                removed |= {c[1] for c in children if not c[0] > th}
            if self.max_conditions is not None and length >= self.max_conditions:
                continue                                       # the children can't be refined further
            for child_bound, f, child, child_mask, child_handle, child_closure, child_stats in children:
                if f in removed:
                    continue
                heapq.heappush(queue, (_Descending(child_bound), next(pushed), child, child_mask - removed,
                                       child_handle, child_closure, child_stats, f))
        return [kept[2] for kept in best]

    @staticmethod
    def _dominated(data, positive: np.ndarray, handle, stats: RuleStats, children: list) -> Set[int]:
        """The conditions OPUS's *cannotImprove* removes below a node:
        those dropping no covered negative, and those whose child a
        sibling dominates (covers a superset of its positives and a
        subset of its negatives). Of children covering exactly the same
        positives and negatives, the first stays. Counts preselect the
        candidate pairs; the covered rows decide."""
        rows = np.flatnonzero(data.cover_rows(handle))
        pos = positive[rows]
        covers = np.array([data.cover_rows(child[4])[rows] for child in children])
        P = np.packbits(covers & pos, axis=1)
        N = np.packbits(covers & ~pos, axis=1)
        features = np.array([child[1] for child in children])
        removed = set(features[~(np.packbits(~pos) & ~N).any(axis=1)].tolist())   # no negative dropped
        tp = np.array([float(child[6].tp) for child in children])
        fp = np.array([float(child[6].fp) for child in children])
        tol = 1e-9 * (1.0 + float(stats.tp) + float(stats.fp))
        I, J = np.nonzero((tp[:, None] >= tp[None, :] - tol) & (fp[:, None] <= fp[None, :] + tol))
        I, J = I[I != J], J[I != J]
        dominates = ~(P[J] & ~P[I]).any(axis=1) & ~(N[I] & ~N[J]).any(axis=1)
        pairs = set(zip(I[dominates].tolist(), J[dominates].tolist()))
        for i, j in pairs:
            if not ((j, i) in pairs and j < i):            # identical twins: only the later one goes
                removed.add(int(features[j]))
        return removed


class SearchSpaceInit(ABC):
    """Base for building block 5: where a single-rule search starts.
    Returns `(rule, open-feature-mask)` pairs, the same shape `RuleSearch.
    search`'s `initial_candidates` expects (and `Rule.specialize` itself
    returns) -- usually just one pair, but nothing here requires that.

    `example_mask` is the current covering scope (`SeCo`'s `remaining`,
    passed straight through by `SingleRuleLearner.learn_one_rule`, or
    `None` when there's no covering loop) -- most implementations ignore
    it; `SeedExample` needs it, to seed on an example that is still
    *uncovered*.
    """

    @abstractmethod
    def initial_candidates(
        self, data: HorizontalBitmapRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Sequence[Tuple[Rule, FrozenSet[int]]]:
        raise NotImplementedError


class EmptyRuleAllFeatures(SearchSpaceInit):
    """Default: start from the empty (universal) rule, every feature open."""

    def initial_candidates(
        self, data: HorizontalBitmapRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Sequence[Tuple[Rule, FrozenSet[int]]]:
        ds = data.spec
        empty = Rule([], target=target_class, dataspec=ds)
        return [(empty, frozenset(range(ds.n_features)))]


class FeatureSubset(SearchSpaceInit):
    """Start from the empty rule, but restrict the search to a named
    subset of features (by name or index) -- e.g. to bound search cost,
    or to exclude features known to be unusable for this `target_class`.
    """

    def __init__(self, allowed_features: Sequence[Union[str, int]]):
        self.allowed_features = tuple(allowed_features)

    def initial_candidates(
        self, data: HorizontalBitmapRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Sequence[Tuple[Rule, FrozenSet[int]]]:
        ds = data.spec
        empty = Rule([], target=target_class, dataspec=ds)
        mask = frozenset(ds.feature_index(f) for f in self.allowed_features)
        return [(empty, mask)]


class SeedExample(SearchSpaceInit):
    """AQ's search-space start: pick one *positive* example still
    uncovered by the rules learned so far (the "seed"), and search the
    space of its generalizations -- the empty rule, open only to the
    features the seed itself satisfies. Every rule the search can reach
    from there is therefore guaranteed to cover the seed (a proper
    subset of a True-valued feature set stays True on that example), so
    AQ never has to check "does this still cover the seed?" explicitly --
    the open mask enforces it.

    `strategy` picks *which* uncovered positive:
    - ``"first"`` (default) -- the lowest-indexed one; fully
      deterministic, and the order examples happen to sit in the data is
      as good as any other arbitrary choice for a noise-free covering.
    - ``"random"`` -- a uniformly random uncovered positive (`random_state`
      seeds it); over several runs this explores different rules first,
      which matters when `maxstar` trimming makes the search order-
      sensitive.
    - ``"index"`` -- seed on the exact row `index`, whatever its class
      (`target_class` is ignored -- callers pass the row's own label).
      `example_mask` is ignored too. This is how `pyrulearn.learners.pylord.PyLORD`
      seeds a search on *every* example in turn, rather than one per
      covering iteration.

    Explicit-negation features make "the features the seed satisfies"
    already include every "attribute X is absent" the seed warrants
    (its negation feature is the True one), so a seed rule can still
    grow conditions like ``not smoker`` -- nothing special is needed for
    negation here.

    Raises `ValueError` if there is no uncovered positive to seed on --
    `SeCo`'s own loop never calls this in that state (it checks first),
    but a direct caller would want to know rather than get a silent
    empty result.
    """

    def __init__(
        self, strategy: str = "first", random_state: Optional[int] = 0, index: Optional[int] = None,
    ):
        if strategy not in ("first", "random", "index"):
            raise ValueError(f"strategy must be 'first', 'random' or 'index', got {strategy!r}")
        if strategy == "index" and index is None:
            raise ValueError("SeedExample(strategy='index') needs index=")
        self.strategy = strategy
        self.random_state = random_state
        self.index = index

    def pick_seed(
        self, data: HorizontalBitmapRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> int:
        """The row this strategy seeds on: `index`, or the first / a
        random positive example within `example_mask`. Also used by
        `SeCo._covering_loop`, which picks seeds itself so that a seed
        with no acceptable rule can be set aside and the next one tried."""
        if self.strategy == "index":
            return int(self.index)
        positive = data.y == target_class
        support = data.scope(example_mask)[0]
        in_scope = positive if support is None else (positive & support)
        candidates = np.flatnonzero(in_scope)
        if candidates.size == 0:
            raise ValueError("SeedExample: no uncovered positive example to seed on")
        if self.strategy == "first":
            return int(candidates[0])
        rng = np.random.default_rng(self.random_state)
        return int(rng.choice(candidates))

    def initial_candidates(
        self, data: HorizontalBitmapRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Sequence[Tuple[Rule, FrozenSet[int]]]:
        ds = data.spec
        seed = self.pick_seed(data, target_class, example_mask)
        open_mask = frozenset(int(i) for i in data.features_of(seed))
        empty = Rule([], target=target_class, dataspec=ds)
        return [(empty, open_mask)]


# ================================================= seeds and heads ====
#
# A covering loop gets each rule from three independent choices: which
# example to explain next (the *seed*, `SeedSelection`, a learner's
# `seed=`), where the search starts (`SearchSpaceInit`, `space_init=`;
# with a seed, `SeedExample` -- the seed's generalizations) and the rule's
# head (`HeadSelection`, `target_class=`: a fixed class, or a class chosen
# per refinement, per rule or per seed example). A seed only sets
# the defaults of the other two -- the seed's generalizations and its
# class, AQ's seed covering -- and either can be overridden.


class SeedSelection(ABC):
    """Which example a covering loop explains next -- the seed. `pick`
    returns a row in scope (of class `target`, unless None), or None if
    there is none. A seed for which no acceptable rule is found is set
    aside and another one tried."""

    @abstractmethod
    def pick(self, data, scope, target: Any, rng: np.random.Generator) -> Optional[int]:
        raise NotImplementedError

    @staticmethod
    def _candidates(data, scope, target: Any) -> np.ndarray:
        support = data.scope(scope)[0]
        rows = np.ones(data.n_samples, dtype=bool) if support is None else support
        if target is not None:
            rows = rows & (np.asarray(data.y) == target)
        return np.flatnonzero(rows)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class FirstSeed(SeedSelection):
    """The lowest-indexed example in scope."""

    def pick(self, data, scope, target, rng):
        candidates = self._candidates(data, scope, target)
        return int(candidates[0]) if len(candidates) else None


class RandomSeed(SeedSelection):
    """A random example in scope, drawn from the covering loop's random
    generator (the learner's `random_state`)."""

    def pick(self, data, scope, target, rng):
        candidates = self._candidates(data, scope, target)
        return int(rng.choice(candidates)) if len(candidates) else None


class _SpaceInitSeed(SeedSelection):
    """A `SeedExample`'s own strategy (``"first"``/``"random"``) as the
    seed of a one-class covering loop -- how seeds were chosen before
    `seed=` existed, kept so that such a learner learns what it did."""

    def __init__(self, space_init: "SeedExample"):
        self.space_init = space_init

    def pick(self, data, scope, target, rng):
        return self.space_init.pick_seed(data, target, scope)


class HeadSelection(ABC):
    """How a covering loop gets its next rule, and that rule's head --
    what a learner's `target_class` holds besides a plain class label
    (which means `FixedClass`). `next_rules` returns the rule(s) of one
    search, each with its head; `seed` is the loop's seed (None without
    one) and `space_init` the search space for it (None: the learner's
    own)."""

    @abstractmethod
    def next_rules(self, learner: "SeCo", data, remaining, seed: Optional[int],
                   space_init: Optional[SearchSpaceInit]) -> List[Rule]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and vars(self) == vars(other)

    def __hash__(self) -> int:
        return hash(type(self))


class FixedClass(HeadSelection):
    """Every rule predicts `label` -- what a plain class label as
    `target_class` means."""

    def __init__(self, label: Any):
        self.label = label

    def next_rules(self, learner, data, remaining, seed, space_init):
        return learner._learn_rules(data, self.label, remaining, space_init)

    def __repr__(self) -> str:
        return f"FixedClass({self.label!r})"


class HeadPerRefinement(HeadSelection):
    """The class chosen per refinement: the heuristic chooses each rule's
    head at every step of its search (`HeadObjective`), so that it can
    change with each added condition (with a gain heuristic, only at the
    first) -- CN2's ordered mode."""

    def next_rules(self, learner, data, remaining, seed, space_init):
        return learner._learn_rules(data, None, remaining, space_init)


class HeadPerSeedExample(HeadSelection):
    """The class chosen per seed example, before the search: the seed's
    class -- AQ's seed covering. Needs a seed."""

    def next_rules(self, learner, data, remaining, seed, space_init):
        if seed is None:
            raise ValueError("HeadPerSeedExample() needs a seed (seed=FirstSeed() or RandomSeed())")
        return learner._learn_rules(data, np.asarray(data.y)[seed], remaining, space_init)


class HeadPerRule(HeadSelection):
    """The class chosen once per rule: one complete search per class
    still in scope, and the best of their rules, each scored by the
    learner's heuristic for its class on the remaining examples (a gain
    heuristic over the scope as a whole)."""

    def next_rules(self, learner, data, remaining, seed, space_init):
        heuristic = learner.single_rule_learner.heuristic
        if not isinstance(heuristic, RuleHeuristic):
            raise ValueError(f"HeadPerRule() compares rules by a RuleHeuristic, got {type(heuristic).__name__}")
        support = data.scope(remaining)[0]
        classes = np.unique(np.asarray(data.y) if support is None else np.asarray(data.y)[support])
        best, best_score = None, None
        for c in classes:
            found = learner._learn_rules(data, c, remaining, space_init)
            if not found:
                continue
            stats = RuleStats.from_rule(found[0], data, c, example_mask=remaining)
            if isinstance(heuristic, GainHeuristic):
                score = heuristic.score(stats, RuleStats.universal(stats.n_pos, stats.n_neg))
            else:
                score = heuristic.score(stats)
            if best_score is None or score > best_score:
                best, best_score = found[0], score
        return [] if best is None else [best]


class SingleRulePreparation(ABC):
    """Base for building block 3: what to do with the in-scope examples
    before searching for a single rule. `example_mask` is the incoming
    scope (e.g. a SeCo loop's "remaining" examples, or None for every
    row). Returns `(search_mask, context)`: `search_mask` is what the
    search itself is restricted to (a sub-mask of `example_mask`, or
    `example_mask` unchanged), `context` carries whatever the matching
    `SingleRulePostProcessing` needs (e.g. a held-out pruning mask).
    """

    @abstractmethod
    def prepare(
        self, data: HorizontalBitmapRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Any]:
        raise NotImplementedError


class NoSplit(SingleRulePreparation):
    """Default: search on every in-scope example, no held-out set."""

    def prepare(
        self, data: HorizontalBitmapRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Any]:
        return example_mask, None


class GrowPruneSplit(SingleRulePreparation):
    """RIPPER/IREP-style: split the in-scope examples into a growing set
    (what the search itself runs on) and a held-out pruning set (returned
    as `context`, meant for a matching `ReducedErrorPruning`), stratified
    by class via `sklearn.model_selection.train_test_split`.

    Confirmed as a real, not just hypothetical, failure mode: a `SeCo`
    covering loop's "remaining" scope shrinks every iteration, and can
    easily end up with too few examples of some class to stratify by the
    time a later rule is learned. Rather than let that raise and take
    down the whole covering loop, `prepare` falls back in two steps: a
    non-stratified split if stratification specifically isn't possible,
    then no split at all (search on every in-scope example, no pruning
    set -- same as `NoSplit`) if even that isn't possible (too few
    in-scope examples to split at all).

    A weighted scope (`WeightedCovering`) is split by rows (those with a
    positive weight), and both halves keep their rows' weights.
    """

    def __init__(self, prune_fraction: float = 0.33, random_state: Optional[int] = 0):
        self.prune_fraction = prune_fraction
        self.random_state = random_state

    def prepare(
        self, data: HorizontalBitmapRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Any]:
        from sklearn.model_selection import train_test_split

        n = data.n_samples
        support = data.scope(example_mask)[0]
        in_scope = np.arange(n) if support is None else np.flatnonzero(support)
        if len(in_scope) < 2:
            return example_mask, None  # too few examples to split at all

        labels = data.y[in_scope]
        try:
            grow_idx, prune_idx = train_test_split(
                in_scope, test_size=self.prune_fraction, random_state=self.random_state, stratify=labels,
            )
        except ValueError:
            try:
                grow_idx, prune_idx = train_test_split(
                    in_scope, test_size=self.prune_fraction, random_state=self.random_state,
                )
            except ValueError:
                return example_mask, None

        grow_mask = np.zeros(n, dtype=bool)
        grow_mask[grow_idx] = True
        prune_mask = np.zeros(n, dtype=bool)
        prune_mask[prune_idx] = True
        if example_mask is not None and np.asarray(example_mask).dtype != bool:   # keep the weights
            w = np.asarray(example_mask, dtype=float)
            return w * grow_mask, w * prune_mask
        return grow_mask, prune_mask


class SingleRulePostProcessing(ABC):
    """Base for building block 4: what to do with a single freshly-found
    rule before it's accepted. `context` is whatever the matching
    `SingleRulePreparation` produced (e.g. a pruning mask)."""

    @abstractmethod
    def postprocess(
        self, rule: Rule, data: HorizontalBitmapRepresentation, target_class: Any, context: Any,
    ) -> Rule:
        raise NotImplementedError


class NoPostProcessing(SingleRulePostProcessing):
    """Default: accept the rule as found, unchanged."""

    def postprocess(
        self, rule: Rule, data: HorizontalBitmapRepresentation, target_class: Any, context: Any,
    ) -> Rule:
        return rule


class ReducedErrorPruning(SingleRulePostProcessing):
    """RIPPER/IREP-style: `context` (from `GrowPruneSplit`) is the
    held-out pruning mask. Scores every *prefix* of `rule` down to length
    1 (the full rule, one shorter, ..., a single condition) against it
    and returns whichever prefix scores best -- RIPPER's own pruning
    metric picks the single best-scoring truncation directly, not a
    greedy walk that stops at the first non-improving removal, so that's
    what this does too. Ties favor the shorter (more general) prefix.
    `context=None` (no pruning set) is a no-op, returning `rule`
    unchanged.

    Deliberately never truncates all the way to length 0 (the empty,
    unconditional rule) even if it would score best on a small/skewed
    pruning sample -- confirmed directly to be a real failure mode, not
    a hypothetical one: an "always true" rule inserted into a `SeCo`
    rule set covers every remaining example at once (both classes),
    which both terminates the covering loop early and, at prediction
    time, fires on every example alongside whatever real rules exist,
    skewing predictions. `SeCo.fit` also rejects a length-0 rule coming
    from the search itself as a second layer of defense (see its own
    docstring) -- an unconditional rule is exactly what the *default*
    rule mechanism is already for, so there's no real capability lost
    by excluding it here specifically. If `rule` itself is already
    length 0 (the search's own conclusion, not this class truncating
    down to it), it's returned unchanged -- only *truncation* to empty
    is refused, not a rule that started that way.
    """

    def __init__(self, heuristic: RuleHeuristic):
        self.heuristic = heuristic

    def postprocess(
        self, rule: Rule, data: HorizontalBitmapRepresentation, target_class: Any, context: Any,
    ) -> Rule:
        if context is None:
            return rule

        def score(r: Rule) -> Score:
            stats = RuleStats.from_rule(r, data, target_class, example_mask=context)
            return self.heuristic.score(stats)

        best, best_score = rule, score(rule)
        for k in range(rule.length() - 1, 0, -1):  # stop at 1 -- never truncate to the empty rule
            truncated = Rule(
                rule.conditions[:k], target=rule.target, dataspec=rule.dataspec,
                n_features=rule.n_features, default_fmt=rule.default_fmt,
            )
            s = score(truncated)
            if s >= best_score:
                best, best_score = truncated, s
        return best


class SingleRuleLearner:
    """Ties the five building blocks together: `learn_one_rule` prepares
    the in-scope examples (block 3), gets a starting search space
    (block 5), runs the search (block 1, scored via block 2's
    `heuristic`), and post-processes the result (block 4). Defaults to
    the simplest configuration -- `BeamSearch`, no split, no
    post-processing, start from the empty rule with every feature open --
    RIPPER/IREP-style behavior needs `GrowPruneSplit`/`ReducedErrorPruning`
    passed explicitly.

    `heuristic` has no default -- picking one is a real modeling choice,
    not something to guess at silently. `filtering`/`stopping`
    (optional, independent, usable together, on top of `RuleSearch`'s
    own always-on `tp == 0` floor and optimistic-value pruning -- see
    `PrePruningCriterion`)
    are passed straight through to the search at call time, not baked
    into `search` itself -- keeps `RuleSearch` implementations reusable
    regardless of which policy is in effect, the same way `heuristic`
    already works.
    """

    def __init__(
        self,
        heuristic: RuleHeuristic,
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
        space_init: Optional[SearchSpaceInit] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
    ):
        self.heuristic = heuristic
        self.search = search if search is not None else BeamSearch()
        self.preparation = preparation if preparation is not None else NoSplit()
        self.postprocessing = postprocessing if postprocessing is not None else NoPostProcessing()
        self.space_init = space_init if space_init is not None else EmptyRuleAllFeatures()
        self._space_init_given = space_init is not None   # else a seed implies SeedExample()
        self.filtering = filtering
        self.stopping = stopping

    def learn_one_rule(
        self, data: HorizontalBitmapRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
        space_init: Optional[SearchSpaceInit] = None,
    ) -> Optional[Rule]:
        """Returns the post-processed single rule, or `None` if the search
        found nothing that passed its `filtering` criterion (propagated
        straight from `RuleSearch.search` -- post-processing is skipped).

        `space_init` overrides `self.space_init` for this call only --
        used by `SeCo`'s covering loops to search the generalizations of
        the seed they picked (`SeedExample(strategy="index")`)."""
        si = space_init if space_init is not None else self.space_init
        search_mask, context = self.preparation.prepare(data, target_class, example_mask)
        initial = si.initial_candidates(data, target_class, search_mask)
        objective = self._objective(target_class)
        rule = self.search.search(
            data, target_class, objective, initial,
            example_mask=search_mask, filtering=self.filtering, stopping=self.stopping,
        )
        if rule is None:
            return None
        rule = self._with_head(rule, data, objective, search_mask)
        return self.postprocessing.postprocess(rule, data, rule.target if target_class is None else target_class,
                                               context)

    def learn_rules(
        self, data: HorizontalBitmapRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
        space_init: Optional[SearchSpaceInit] = None,
    ) -> List[Rule]:
        """Every rule one search yields (`RuleSearch.search_all`), each
        post-processed -- one rule, or none, unless the search branches.
        What `SeCo`'s covering loop calls; same preparation and search
        space as `learn_one_rule`."""
        si = space_init if space_init is not None else self.space_init
        search_mask, context = self.preparation.prepare(data, target_class, example_mask)
        initial = si.initial_candidates(data, target_class, search_mask)
        objective = self._objective(target_class)
        rules = self.search.search_all(
            data, target_class, objective, initial,
            example_mask=search_mask, filtering=self.filtering, stopping=self.stopping,
        )
        rules = [self._with_head(rule, data, objective, search_mask) for rule in rules]
        return [self.postprocessing.postprocess(rule, data, rule.target if target_class is None else target_class,
                                                context) for rule in rules]

    def _objective(self, target_class: Any) -> Union[RuleHeuristic, Objective]:
        """What the search maximizes: `heuristic` for a fixed
        `target_class`; for none -- a rule of any class -- the heuristic
        choosing each rule's head (`HeadObjective`)."""
        if target_class is not None:
            return self.heuristic
        if isinstance(self.heuristic, Objective):
            raise ValueError(f"a rule of any class (target_class None) needs a RuleHeuristic to choose its "
                             f"head, got {type(self.heuristic).__name__}")
        return HeadObjective(self.heuristic)

    @staticmethod
    def _with_head(rule: Rule, data, objective, search_mask) -> Rule:
        """`rule` with the head the search chose for it (`HeadObjective`),
        or unchanged for a fixed target class -- before post-processing,
        which then keeps that head."""
        if not isinstance(objective, HeadObjective):
            return rule
        stats = objective.rule_stats(data, None, rule, handle_for(data, rule, search_mask))
        return Rule(rule.conditions, target=objective.head_of(stats), dataspec=rule.dataspec,
                    n_features=rule.n_features, default_fmt=rule.default_fmt)


class RuleSetOptimizer(ABC):
    """The rule-set optimizer: rework a whole class's rules *after* the
    covering loop has produced them, before they become a `RuleSet`.
    Where `SingleRulePostProcessing` prunes one rule in isolation, this
    reconsiders each rule against the rest of the set -- RIPPER's
    optimization phase (`ReplaceReviseOptimization`).

    Still binary-shaped, like every other `SeCo` building block: one
    `target_class`, negatives = everything else within `example_mask`.
    RIPPER runs it per class, inside a class-ordered decomposition
    (`pyrulearn.learners.multiclass.OrderedOneVsRest`), which is exactly how
    `Pypper` composes it.
    """

    @abstractmethod
    def optimize(
        self,
        rules: List[Rule],
        data: HorizontalBitmapRepresentation,
        target_class: Any,
        single_rule_learner: "SingleRuleLearner",
        example_mask: Optional[np.ndarray] = None,
    ) -> List[Rule]:
        """Return a (hopefully better) list of rules for `target_class`.
        `single_rule_learner` is `SeCo`'s own, handed in so alternatives
        are grown with the identical search/heuristic configuration."""
        raise NotImplementedError


# -- minimum description length (Cohen 1995, after Quinlan) ------------------

def _log2_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def _rule_theory_bits(n_conditions: int, n_possible: int) -> float:
    """Bits to transmit a rule with `n_conditions` conditions drawn from
    `n_possible` possible ones -- ``||k|| + k*log2(n/k) + (n-k)*log2(n/(n-k))``,
    halved to allow for redundancy among the features (Cohen 1995 sec. 2.2)."""
    k, n = min(n_conditions, n_possible), n_possible
    if n <= 0 or k <= 0:
        return 0.0
    p = k / n
    bits = math.log2(k) + k * math.log2(1.0 / p)
    if k < n:
        bits += (n - k) * math.log2(1.0 / (1.0 - p))
    return 0.5 * bits


def _exception_bits(n_covered: int, n_fp: int, n_uncovered: int, n_fn: int) -> float:
    """Bits to transmit which covered examples are false positives and
    which uncovered ones are false negatives -- ``log2(t+1) + log2(C(t, e))``
    per subset (the count, then which)."""
    def subset(t: int, e: int) -> float:
        return (math.log2(t + 1) + _log2_choose(t, e)) if t > 0 else 0.0
    return subset(n_covered, n_fp) + subset(n_uncovered, n_fn)


def _combined_cover(rules: Sequence[Rule], data: HorizontalBitmapRepresentation) -> np.ndarray:
    covered = np.zeros(data.n_samples, dtype=bool)
    for r in rules:
        covered |= r.covers_data(data)
    return covered


def rule_set_description_length(
    rules: Sequence[Rule],
    data: HorizontalBitmapRepresentation,
    target_class: Any,
    example_mask: Optional[np.ndarray] = None,
) -> float:
    """Total MDL of `rules` as a theory for the binary problem
    ``target_class`` vs. the rest (within `example_mask` if given): the
    sum of each rule's `_rule_theory_bits` plus the `_exception_bits` for
    the union of their coverage. With row weights (the data's, or a
    weighted `example_mask`) the example counts are sums of weights."""
    n_possible = data.spec.n_features
    theory = sum(_rule_theory_bits(r.length(), n_possible) for r in rules)
    w = data.scope_weights(example_mask)
    covered = _combined_cover(rules, data)
    pos = np.asarray(data.y) == target_class
    n_covered = float(w[covered].sum())
    n_fp = float(w[covered & ~pos].sum())
    n_uncovered = float(w[~covered].sum())
    n_fn = float(w[~covered & pos].sum())
    return theory + _exception_bits(n_covered, n_fp, n_uncovered, n_fn)


class ReplaceReviseOptimization(RuleSetOptimizer):
    """RIPPER's optimization phase (Cohen 1995 sec. 4). For `passes`
    rounds (RIPPER's default `k=2`), draw a fresh grow/prune split and
    walk the rules in order; for each rule `ri` build

    - a **replacement** -- a rule grown from scratch, and
    - a **revision** -- a rule grown by adding conditions to `ri`,

    each pruned (trailing conditions dropped) to minimize the *whole
    rule set's* error on the pruning split with `ri` swapped for it. Keep
    whichever of `{ri, replacement, revision}` gives the rule set the
    lowest `rule_set_description_length`. After each round, drop any rule
    whose removal lowers the total description length.

    Deliberately simplified: no separate "residual IREP*" step (Cohen
    re-covers positives left uncovered after DL-pruning) -- the input
    already comes from a full covering loop, so uncovered positives fall
    to the default like anywhere else.
    """

    def __init__(
        self,
        passes: int = 2,
        prune_fraction: float = 1.0 / 3.0,
        random_state: Optional[int] = 0,
    ):
        self.passes = passes
        self.prune_fraction = prune_fraction
        self.random_state = random_state

    def optimize(
        self,
        rules: List[Rule],
        data: HorizontalBitmapRepresentation,
        target_class: Any,
        single_rule_learner: "SingleRuleLearner",
        example_mask: Optional[np.ndarray] = None,
    ) -> List[Rule]:
        if not rules:
            return rules
        scope = example_mask

        for p in range(self.passes):
            rs = None if self.random_state is None else self.random_state + p
            grow_mask, prune_mask = GrowPruneSplit(self.prune_fraction, rs).prepare(data, target_class, scope)
            revised: List[Rule] = []
            for i, rule in enumerate(rules):
                context = revised + rules[i + 1:]  # the set with `rule` removed
                candidates = [rule]
                for start in (None, rule):
                    alt = self._grow_and_prune(
                        start, context, data, target_class, single_rule_learner,
                        grow_mask, prune_mask,
                    )
                    if alt is not None and alt.length() > 0:
                        candidates.append(alt)
                best = min(
                    candidates,
                    key=lambda c: rule_set_description_length(context + [c], data, target_class, scope),
                )
                revised.append(best)
            rules = self._dl_prune(revised, data, target_class, scope)
        return rules

    def _grow_and_prune(
        self, start: Optional[Rule], context: List[Rule], data: HorizontalBitmapRepresentation,
        target_class: Any, srl: "SingleRuleLearner",
        grow_mask: Optional[np.ndarray], prune_mask: Optional[np.ndarray],
    ) -> Optional[Rule]:
        ds = data.spec
        if start is None:
            initial = EmptyRuleAllFeatures().initial_candidates(data, target_class, grow_mask)
        else:
            open_feats = frozenset(range(ds.n_features)) - {l.feature for l in start.conditions}
            initial = [(Rule(list(start.conditions), target=target_class, dataspec=ds,
                             n_features=start.n_features), open_feats)]
        grown = srl.search.search(
            data, target_class, srl.heuristic, initial,
            example_mask=grow_mask, filtering=srl.filtering, stopping=srl.stopping,
        )
        if grown is None or grown.length() == 0:
            return None
        if prune_mask is None:
            return grown

        # prune trailing conditions to minimize the whole set's error on the pruning split
        w = data.scope_weights(prune_mask)
        pos = np.asarray(data.y) == target_class

        def err(rule: Rule) -> float:
            covered = _combined_cover(context + [rule], data)
            return float(w[covered & ~pos].sum()) + float(w[~covered & pos].sum())

        best, best_err = grown, err(grown)
        for k in range(grown.length() - 1, 0, -1):
            pref = Rule(grown.conditions[:k], target=target_class, dataspec=ds, n_features=grown.n_features)
            e = err(pref)
            if e <= best_err:
                best, best_err = pref, e
        return best

    def _dl_prune(
        self, rules: List[Rule], data: HorizontalBitmapRepresentation, target_class: Any,
        scope: Optional[np.ndarray],
    ) -> List[Rule]:
        kept = list(rules)
        for r in reversed(list(kept)):  # RIPPER: consider the last-added rules first
            without = [x for x in kept if x is not r]
            if rule_set_description_length(without, data, target_class, scope) < \
               rule_set_description_length(kept, data, target_class, scope):
                kept = without
        return kept


# -- covering strategies -----------------------------------------------------------

class CoveringState:
    """What one covering loop (one target class) keeps between rules:

    - `scope` -- what the next rule is learned on, passed to every
      building block as `example_mask`: a boolean mask (removal) or a
      weight vector (weighted covering);
    - `positive` -- the target-class rows;
    - `times_covered` -- per row, how many accepted rules cover it;
    - `errors` -- per row, how many accepted rules err on it (cover a
      negative, or leave a positive uncovered);
    - `round` -- the number of rules accepted so far;
    - `extra` -- anything else a `Reweighting` wants to keep.
    """

    def __init__(self, scope: np.ndarray, positive: np.ndarray):
        self.scope = scope
        self.positive = positive
        self.times_covered = np.zeros(len(positive), dtype=np.int64)
        self.errors = np.zeros(len(positive), dtype=np.int64)
        self.round = 0
        self.initial_positive_weight = float(np.sum(np.asarray(scope, dtype=float)[positive]))
        self.extra: Dict[str, Any] = {}

    def record(self, covered: np.ndarray) -> None:
        """Book an accepted rule covering the rows `covered`."""
        self.times_covered += covered
        self.errors += covered != self.positive
        self.round += 1


class CoveringStrategy(ABC):
    """How `SeCo`'s covering loop changes the scope -- which examples, with
    which weights, the next rule is learned on -- after each accepted
    rule, and when the loop is done. Works on a `CoveringState`."""

    @abstractmethod
    def start(self, data: Any, positive: np.ndarray) -> CoveringState:
        """The state before the first rule (`positive`: the target-class rows)."""
        raise NotImplementedError

    @abstractmethod
    def update(self, state: CoveringState, covered: np.ndarray) -> Optional[float]:
        """Book an accepted rule covering the rows `covered` and set the
        next scope. Returns the rule's weight if the strategy assigns one
        (boosting), else `None`."""
        raise NotImplementedError

    def exhausted(self, state: CoveringState) -> bool:
        """Whether the loop is done: no positive example left in scope."""
        return not np.any(state.positive & (state.scope > 0))

    def drop(self, state: CoveringState, row: int) -> None:
        """Take `row` out of the scope entirely -- a seed example no
        acceptable rule could be found for (see `SeCo._covering_loop`)."""
        state.scope = state.scope.copy()
        state.scope[row] = 0


class RemovalCovering(CoveringStrategy):
    """The classic separate-and-conquer policy (the default): every
    example an accepted rule covers -- both classes -- leaves the scope.
    The scope is a boolean mask."""

    def start(self, data: Any, positive: np.ndarray) -> CoveringState:
        return CoveringState(np.ones(len(positive), dtype=bool), positive)

    def update(self, state: CoveringState, covered: np.ndarray) -> Optional[float]:
        state.record(covered)
        state.scope = state.scope & ~covered
        return None


# -- reweighting schemes ---------------------------------------------------------

class Reweighting(ABC):
    """How weighted covering reweights the examples after each accepted
    rule -- the part in which weighted-covering algorithms differ, a
    pluggable component like a `RuleHeuristic`. `weights` returns the
    next scope's weights from a `CoveringState` that has already booked
    the new rule (`times_covered`, `errors`, `round` updated). A scheme
    that also fits a weight for each rule (boosting) sets
    `assigns_rule_weights` and returns it from `rule_weight`, which is
    called first, on the state before the new rule is booked.
    """

    #: whether `rule_weight` returns a fitted weight for each rule
    assigns_rule_weights: bool = False

    def rule_weight(self, state: CoveringState, covered: np.ndarray) -> Optional[float]:
        return None

    @abstractmethod
    def weights(self, state: CoveringState, covered: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


class MultiplicativeReweighting(Reweighting):
    """A positive covered by `k` rules weighs ``gamma ** k``; negatives
    keep weight 1 -- CN2-SD's multiplicative scheme (Lavrač et al. 2004),
    and CPAR's weight decay (Yin & Han 2003, ``gamma = 2/3``)."""

    def __init__(self, gamma: float = 0.5):
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        self.gamma = gamma

    def weights(self, state: CoveringState, covered: np.ndarray) -> np.ndarray:
        return np.where(state.positive, self.gamma ** state.times_covered, 1.0)


class AdditiveReweighting(Reweighting):
    """A positive covered by `k` rules weighs ``1 / (k + 1)``; negatives
    keep weight 1 -- CN2-SD's additive scheme (Lavrač et al. 2004)."""

    def weights(self, state: CoveringState, covered: np.ndarray) -> np.ndarray:
        return np.where(state.positive, 1.0 / (state.times_covered + 1.0), 1.0)


class AdaBoostReweighting(Reweighting):
    """Confidence-rated boosting (Schapire & Singer 1999), as in Slipper
    (Cohen & Singer 1999): a rule covering positives of weight ``W+`` and
    negatives of weight ``W-`` gets the weight (confidence)

        C = 1/2 * ln((W+ + eps) / (W- + eps))

    and every example it covers is multiplied by ``exp(-y * C)`` (``y =
    +1`` for positives, ``-1`` for negatives): positives covered by a
    confident rule lose weight, negatives it wrongly covers gain some. The
    weights are then rescaled to their previous total. `eps`, the
    smoothing, defaults to half a row's average weight (Slipper's
    ``1 / (2n)`` for weights summing to 1)."""

    assigns_rule_weights = True

    def __init__(self, eps: Optional[float] = None):
        self.eps = eps

    def confidence(self, w: np.ndarray, covered: np.ndarray, positive: np.ndarray) -> float:
        eps = self.eps if self.eps is not None else 0.5 * float(w.sum()) / len(w)
        w_pos = float(w[covered & positive].sum())
        w_neg = float(w[covered & ~positive].sum())
        return 0.5 * math.log((w_pos + eps) / (w_neg + eps))

    def rule_weight(self, state: CoveringState, covered: np.ndarray) -> Optional[float]:
        c = self.confidence(np.asarray(state.scope, dtype=float), covered, state.positive)
        state.extra["confidence"] = c
        return c

    def weights(self, state: CoveringState, covered: np.ndarray) -> np.ndarray:
        w = np.asarray(state.scope, dtype=float)
        y = np.where(state.positive, 1.0, -1.0)
        new = w * np.exp(-y * state.extra["confidence"] * covered)
        return new * (w.sum() / new.sum())


class LRIReweighting(Reweighting):
    """Lightweight Rule Induction (Weiss & Indurkhya 2000): every example
    weighs ``1 + e ** power`` (``power = 3``), where ``e`` is the number
    of accepted rules that err on it -- cover it though it's negative, or
    leave it uncovered though it's positive. Both classes are reweighted,
    and an example that keeps being misclassified quickly dominates (4
    errors: weight 65). As in the paper, once some ``e`` exceeds
    `max_errors` (32), all of them are halved (integer division), which
    keeps the weights bounded; ``None`` never halves. The halved counts
    are kept in ``state.extra["lri_errors"]``; `CoveringState.errors`
    keeps the plain counts."""

    def __init__(self, power: float = 3.0, max_errors: Optional[int] = 32):
        self.power = power
        self.max_errors = max_errors

    def weights(self, state: CoveringState, covered: np.ndarray) -> np.ndarray:
        e = state.extra.get("lri_errors")
        if e is None:
            e = np.zeros(len(state.positive), dtype=np.int64)
        e = e + (covered != state.positive)
        if self.max_errors is not None and e.max() > self.max_errors:
            e //= 2
        state.extra["lri_errors"] = e
        return 1.0 + e.astype(float) ** self.power


# -- stopping weighted covering ---------------------------------------------------

class CoveringStop(ABC):
    """When a weighted covering loop is done (weighted covering never runs
    out of examples by itself). Removal covering needs none."""

    @abstractmethod
    def done(self, state: CoveringState) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


class CoveredAtLeast(CoveringStop):
    """Done once every positive example is covered by at least `k`
    accepted rules (CN2-SD)."""

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k

    def done(self, state: CoveringState) -> bool:
        return bool(np.all(state.times_covered[state.positive] >= self.k))


class PositiveWeightBelow(CoveringStop):
    """Done once the positives' total weight has fallen below `fraction`
    of what it was at the start (CPAR, ``fraction = 0.05``)."""

    def __init__(self, fraction: float = 0.05):
        self.fraction = fraction

    def done(self, state: CoveringState) -> bool:
        w = float(np.sum(np.asarray(state.scope, dtype=float)[state.positive]))
        return w < self.fraction * state.initial_positive_weight


class Rounds(CoveringStop):
    """Done after `n` accepted rules (Slipper, LRI)."""

    def __init__(self, n: int):
        self.n = n

    def done(self, state: CoveringState) -> bool:
        return state.round >= self.n


class WeightedCovering(CoveringStrategy):
    """Weighted covering: covered examples stay in scope, reweighted by a
    `Reweighting` scheme (default `MultiplicativeReweighting(0.5)`,
    CN2-SD's), until `stop` (a `CoveringStop` or a sequence of them, any
    of which ends the loop; default `CoveredAtLeast(5)`). Later rules may
    then overlap earlier ones. `max_rounds` (default 100) is a safety net
    on top: weighting alone doesn't guarantee that every stop criterion is
    ever met. The weights multiply with the data's own row weights, if
    any.

    The loop keeps a rule that is found again (it still counts as a
    round and reweights); an unweighted rule set built from the loop
    drops such duplicates, which add nothing to it. Meant for unordered
    rule sets (`ConceptModel`/`ConceptSet`): a decision list's
    first-match semantics assumes removal.
    """

    def __init__(self, reweighting: Optional[Reweighting] = None,
                 stop: Union[CoveringStop, Sequence[CoveringStop], None] = None,
                 max_rounds: Optional[int] = 100):
        self.reweighting = reweighting if reweighting is not None else MultiplicativeReweighting(0.5)
        if stop is None:
            stop = CoveredAtLeast(5)
        self.stop = [stop] if isinstance(stop, CoveringStop) else list(stop)
        self.max_rounds = max_rounds

    def start(self, data: Any, positive: np.ndarray) -> CoveringState:
        return CoveringState(np.ones(len(positive), dtype=float), positive)

    def update(self, state: CoveringState, covered: np.ndarray) -> Optional[float]:
        weight = self.reweighting.rule_weight(state, covered)
        state.record(covered)
        dropped = state.scope == 0            # seeds dropped by `drop` stay out
        state.scope = np.where(dropped, 0.0, self.reweighting.weights(state, covered))
        return weight

    def exhausted(self, state: CoveringState) -> bool:
        if super().exhausted(state):
            return True
        if self.max_rounds is not None and state.round >= self.max_rounds:
            return True
        return any(s.done(state) for s in self.stop)

    def __repr__(self) -> str:
        return f"WeightedCovering(reweighting={self.reweighting!r}, stop={self.stop!r}, max_rounds={self.max_rounds!r})"


def _resolve_covering(covering: Optional[CoveringStrategy]) -> CoveringStrategy:
    return RemovalCovering() if covering is None else covering


class SeCo(DecomposingLearner, NativeRuleLearner):
    """Sequential covering: repeatedly call `single_rule_learner` for
    `target_class`, accept what it finds, and remove every example the
    accepted rule covers -- both classes, not just positives -- from
    further consideration (the standard separate-and-conquer covering
    policy; see Fürnkranz's survey), until `stop_covering` says to stop
    or no positive examples remain uncovered. Removing negatives too
    (not just the positives just explained) matches the textbook
    definition of "separate": each subsequent rule only ever needs to
    prove itself against what's left, not the whole original dataset.

    `stop_covering` is a `pyrulearn.pruning.PrePruningCriterion` -- the
    same type a search's `filtering=`/`stopping=`
    take -- but consulted only through its bare `evaluate()`, once per
    learned rule, between `SingleRuleLearner.learn_one_rule` calls: the
    moment it fires, the just-learned rule is discarded (never appended
    to `rules`) and the whole covering loop halts. `polarity`/`reject`/
    `accept` play no role here and can be left at their defaults --
    they exist to resolve *which* rule to hand back when a search stops
    partway through refining (`BeamSearch`'s best-so-far vs. the
    hill-climbers' last-visited), a question that doesn't arise at
    this level: there's exactly one covering loop, never a population
    of candidates to filter among, so a fired criterion always simply
    means "stop, and throw away what was just found" -- see
    `PrePruningCriterion`'s own docstring for the filtering/stopping
    distinction this collapses away here. `stats` (passed to `evaluate`
    already computed, same reasoning as everywhere else it's threaded
    through rather than recomputed) is the just-learned rule's stats
    *within the current covering scope* (`remaining` -- not the full
    dataset) -- e.g. `ThresholdPrePruning(UncoveredPositives(),
    -p, ">=")` reads `stats.fn` (how many positives would still need
    explaining once this rule is added) to stop once at most `p` remain.

    Unconditional floor, independent of any criterion configured here:
    `SeCo` stops once every positive example is covered (nothing left
    to explain), `single_rule_learner` returns `None` (its search found
    nothing passing its own `filtering` -- no acceptable rule left to
    add), the just-learned rule covers no new positives at all
    (`stats.tp == 0` within the remaining scope -- adding it would be
    pure waste, and letting the loop continue could add rule after rule
    that changes nothing), or the rule is length 0 (unconditional --
    confirmed directly as a real failure mode of `ReducedErrorPruning`
    on a noisy pruning sample, see its own docstring; this is a second
    layer of defense in case the search itself ever concludes an
    unconditional rule is "best", not just pruning degrading to one).
    These are all direct checks in `fit` itself, not routed through
    `stop_covering`, for the same reason `RuleSearch` keeps its own
    floor unwrapped -- see that class's docstring. `max_rules` is a
    separate, hardcoded resource budget for the same reason a search
    keeps `max_conditions` outside the criterion system too -- a hard
    cap, not a quality judgment.

    **Which model `fit` builds** (see `pyrulearn.learners.RuleLearner.fit`):

    - `fit(data, model=ConceptModel, label="a")` -- the definition of one
      concept: the covering loop for ``"a"`` vs. everything else. Setting
      `target_class="a"` at construction makes this the default (`label`
      then comes from `target_class`).
    - `fit(data)` with no `target_class` -- the family multi-class default
      (`_MULTICLASS_DEFAULT`): `ConceptSet` (one-vs-rest) for the SeCo
      family, `DecisionList` (one seed-covering loop) for `AQR`.
    - `fit(data, model=ConceptSet | ConceptCascade | PairwiseModel)` --
      one-vs-rest / ordered peeling / round robin, via `DecomposingLearner`
      (each fits per-class copies through the covering loop).
    - `fit(data, model=DecisionList)` -- one covering loop over all
      classes at once, rules in learn order, first match wins
      (`_decision_list_fit`): each rule from a seed (`seed=`), a search
      space and a head (`target_class=` a `HeadSelection`); without
      either, the best rule of any class (`HeadPerRefinement()`, CN2's ordered
      mode), or for a seeded learner a random seed, its generalizations
      and its class (AQ's seed covering). `fit(data)` builds it whenever
      `target_class` is a `HeadSelection` other than `FixedClass`, or a
      `seed` is given.
    - `fit(data, model=FlatRuleSet)` -- the same rules, as a rule set
      resolved by learn order (`ListCombiner()`), e.g. to compare other
      combiners on them.

    `SeCo` takes no default-prediction or combiner arguments -- it sets
    sensible ones on the result (`MajorityClass(data)` fallback;
    `HeuristicMaxCombiner()` for a `ConceptSet`) and leaves the rest to the caller. Reassign
    either on the returned model.

    Learned rules carry no declarative weight -- `HeuristicMaxCombiner()`
    (`Laplace` by default) scores each rule from
    its own *measured* `stats()` at predict time instead, the same
    default `sort_rules`/`covered_by` use for inspection; pass
    `combiner=HeuristicMaxCombiner(SomeHeuristic())` for a different one.

    `covering` (a `CoveringStrategy`, default `RemovalCovering`) is how
    the scope changes after each accepted rule: removing everything the
    rule covers (the description above), or `WeightedCovering` --
    reweighting the examples by a `Reweighting` scheme instead, so rules
    may overlap (CN2-SD, CPAR), until its `CoveringStop` criteria. Only
    the binary covering loop uses it; the all-classes seed covering
    (`model=DecisionList`) always removes. `SeCo` builds unweighted rule
    sets, so a reweighting that fits rule weights (boosting,
    `AdaBoostReweighting`) belongs to a boosting learner such as
    `pyrulearn.learners.boosting.Slipper`, and is refused here.

    `optimization` (a `RuleSetOptimizer`, default `None`) runs once the
    covering loop is done, reworking the whole class's rules before they
    become a `FlatRuleSet` -- RIPPER's optimization phase. Only in the
    binary path (`target_class` set, or a per-class stage of a decomposition);
    ignored by the all-classes seed covering (`model=DecisionList` /
    `FlatRuleSet`), which has no single target class.

    `seed` (a `SeedSelection`: `FirstSeed()`, `RandomSeed()`; default
    none) makes each search explain one uncovered example: by default it
    searches that example's generalizations (unless `space_init` is
    given) and, in a decision list, predicts its class (unless
    `target_class` is a `HeadSelection`). In a one-class covering loop
    the seed is an uncovered example of the class. A seed for which no
    acceptable rule is found is set aside and the next one tried.
    `target_class` takes a class (or `FixedClass`) for one concept, or
    `HeadPerRefinement()`, `HeadPerSeedExample()` or `HeadPerRule()` for a decision list
    (see "seeds and heads" above `HeadSelection`).
    """

    #: model type `fit(data)` builds with no `model=` and no `target_class`.
    #: `ConceptSet` (one-vs-rest) for the SeCo family; `AQR` overrides to
    #: `DecisionList` (one seed-covering loop).
    _MULTICLASS_DEFAULT: type = ConceptSet

    def __init__(
        self,
        single_rule_learner: SingleRuleLearner,
        target_class: Any = None,
        stop_covering: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
        random_state: Optional[int] = 0,
        optimization: Optional[RuleSetOptimizer] = None,
        covering: Optional[CoveringStrategy] = None,
        seed: Optional[SeedSelection] = None,
    ):
        self.single_rule_learner = single_rule_learner
        self.target_class = target_class
        self.stop_covering = stop_covering
        self.max_rules = max_rules
        self.optimization = optimization
        self.random_state = random_state
        self.covering = covering
        self.seed = seed

    def _provenance_params(self):
        params = super()._provenance_params()
        if params.get("seed") is None:   # absent unless given: keeps earlier cache keys valid
            params.pop("seed", None)
        return params

    def _default_model(self, data: HorizontalBitmapRepresentation) -> type:
        """`fit(data)` with no `model=`: one concept (`target_class` a
        class, or `FixedClass`), a decision list (`target_class` another
        `HeadSelection`, or a `seed`), or the family's multi-class default
        (`ConceptSet`; `DecisionList` for `AQR`)."""
        head = self.target_class
        if isinstance(head, HeadSelection):
            return ConceptModel if isinstance(head, FixedClass) else DecisionList
        if head is not None:
            return ConceptModel
        removal = isinstance(_resolve_covering(getattr(self, "covering", None)), RemovalCovering)
        # a decision list needs removal: with weighted covering (CPAR), a seed only seeds the class loops
        return DecisionList if self.seed is not None and removal else self._MULTICLASS_DEFAULT

    def _fixed_class(self, label: Any = None) -> Any:
        """The one class of a concept: `label`, or `target_class` (a class
        or `FixedClass`); None if there is none."""
        target = label if label is not None else self.target_class
        if isinstance(target, FixedClass):
            return target.label
        if isinstance(target, HeadSelection):
            raise ValueError(f"target_class={target!r} has no single class; pass label=")
        return target

    def _seed_selection(self, seed: Optional[SeedSelection], multiclass: bool) -> Optional[SeedSelection]:
        """The seed of a covering loop: `seed`, else the learner's own
        `seed=`, else -- with a `SeedExample` search space and no `seed=`,
        as before `seed=` existed -- its own strategy in a one-class loop
        and a random seed in a decision list (AQR's seed covering)."""
        if seed is None:
            seed = self.seed
        if seed is not None:
            return seed
        space_init = getattr(self.single_rule_learner, "space_init", None)
        if isinstance(space_init, SeedExample) and space_init.strategy != "index":
            return RandomSeed() if multiclass else _SpaceInitSeed(space_init)
        return None

    def _seed_space(self, seed: Optional[SeedSelection]) -> bool:
        """Whether each search generalizes the loop's seed: with a
        `SeedExample` search space, or with a seed and no `space_init`
        given (a seed implies its generalizations)."""
        space_init = getattr(self.single_rule_learner, "space_init", None)
        if isinstance(space_init, SeedExample):
            return space_init.strategy != "index"
        given = getattr(self.single_rule_learner, "_space_init_given", True)
        return (seed if seed is not None else self.seed) is not None and not given

    def _fit_binary(self, data: HorizontalBitmapRepresentation, positive: Any,
                    negative: Any = None) -> ConceptModel:
        """The decomposition primitive: a `ConceptModel` for `positive`,
        with `negative` (a pair sub-problem) or `MajorityClass` (one-vs-
        rest) as the fallback."""
        return self._fit_covering(data, label=positive, fallback=negative)

    def _covering_loop(self, data: HorizontalBitmapRepresentation, target: Any) -> List[Rule]:
        """The separate-and-conquer covering loop for one class `target`
        vs. the rest, plus `optimization` if set. Shared by the binary
        `_fit_default` path and `_fit_covering`.

        A learner that searches from a seed example (`SeedExample`, i.e.
        `AQR`) only ever sees the generalizations of that one example, so
        failing to find an acceptable rule for it -- or `stop_covering`
        rejecting the rule found -- says nothing about the other
        uncovered positives: that seed is just dropped from the uncovered
        examples (as if covered) and the next one tried, so the loop only
        stops once every positive has been covered or dropped. (One noisy
        seed would otherwise end the whole class.) Dropping a positive
        can't affect later rules' consistency, which only negatives
        decide. A learner searching all rules at once stops at its first
        failure, since no other search would find anything else.

        The covering scope (`state.scope`, a boolean mask or a weight
        vector) is maintained by `self.covering` (`CoveringStrategy`). A
        rule found again under weighted covering still counts as a round
        and reweights, but appears only once in the result."""
        positive = data.y == target
        covering = _resolve_covering(getattr(self, "covering", None))
        reweighting = getattr(covering, "reweighting", None)
        if reweighting is not None and reweighting.assigns_rule_weights:
            raise ValueError(f"{type(reweighting).__name__} fits rule weights, which SeCo's unweighted "
                             "rule sets can't hold -- use a boosting learner (e.g. Slipper)")
        state = covering.start(data, positive)
        rules: List[Rule] = []
        seeds = self._seed_selection(None, multiclass=False)
        seed_space = self._seed_space(None)
        rng = np.random.default_rng(self.random_state)

        while True:
            if self.max_rules is not None and len(rules) >= self.max_rules:
                break
            if covering.exhausted(state):
                break  # every positive covered (or dropped as a failed seed), or a stop criterion
            remaining = state.scope

            if seeds is not None:
                seed = seeds.pick(data, remaining, target, rng)
                if seed is None:
                    break
                found = self._learn_rules(data, target, remaining,
                                          SeedExample(strategy="index", index=seed) if seed_space else None)
            else:
                found = self._learn_rules(data, target, remaining)
            # one rule per search, unless the search branches (CPAR's): then
            # each, judged on the scope it was found in, then reweighting
            outcome = "next" if found else "unacceptable"
            for rule in found:
                if self.max_rules is not None and len(rules) >= self.max_rules:
                    outcome = "stop"
                    break
                stats = RuleStats.from_rule(rule, data, target, example_mask=remaining)
                if stats.tp > 0 and rule.length() == 0:
                    outcome = "stop"  # unconditional -- the default-rule mechanism covers this
                    break
                if stats.tp == 0 or (self.stop_covering is not None
                                     and self.stop_covering.evaluate(rule, stats, data, target, remaining)):
                    outcome = "unacceptable"
                    break
                rules.append(rule)
                covering.update(state, rule.covers_data_packed(data))
            if outcome == "stop":
                break
            if outcome == "unacceptable":
                # nothing (more) acceptable found (no rule, one covering
                # nothing new, or one `stop_covering` rejects)
                if seeds is not None:
                    covering.drop(state, seed)
                    continue
                break

        rules = list(dict.fromkeys(rules))   # a rule found again (weighted covering) counts once
        if self.optimization is not None:
            rules = self.optimization.optimize(
                rules, data, target, self.single_rule_learner, example_mask=None,
            )
        return rules

    def _learn_rules(self, data: HorizontalBitmapRepresentation, target: Any, remaining: np.ndarray,
                     space_init: Optional[SearchSpaceInit] = None) -> List[Rule]:
        """`single_rule_learner.learn_rules` -- or, for a single-rule
        learner of one's own with only `learn_one_rule`, its rule."""
        learner = self.single_rule_learner
        if hasattr(learner, "learn_rules"):
            return learner.learn_rules(data, target, remaining, space_init=space_init)
        rule = (learner.learn_one_rule(data, target, remaining) if space_init is None
                else learner.learn_one_rule(data, target, remaining, space_init=space_init))
        return [] if rule is None else [rule]

    @produces(ConceptModel)
    def _fit_covering(self, data: HorizontalBitmapRepresentation, *,
                      label: Any = None, fallback: Any = None) -> ConceptModel:
        """`fit(data, model=ConceptModel, label="a")`: the definition of
        one concept -- the binary covering loop for `label` (or
        `self.target_class`). `fallback` is the negative prediction for
        uncovered rows (`MajorityClass(data)` if not given)."""
        if data.y is None:
            raise ValueError("SeCo needs data.y")
        target = self._fixed_class(label)
        if target is None:
            raise ValueError("model=ConceptModel needs label= (or target_class set)")
        rules = annotate_rules(self._covering_loop(data, target), data)
        default = fallback if fallback is not None else MajorityClass(data)
        return annotate_default_rule(ConceptModel(rules, label=target, default_prediction=default), data)

    @produces(SingleRule)
    def _fit_one_rule(self, data: HorizontalBitmapRepresentation, *, label: Any = None) -> SingleRule:
        """`fit(data, model=SingleRule, label="a")`: just the single best
        rule the search finds for `label` -- one `learn_one_rule` call, no
        covering loop, no optimization."""
        if data.y is None:
            raise ValueError("SeCo needs data.y")
        target = self._fixed_class(label)
        if target is None:
            raise ValueError("model=SingleRule needs label= (or target_class set)")
        all_rows = np.ones(data.n_samples, dtype=bool)
        rule = self.single_rule_learner.learn_one_rule(data, target, all_rows)
        if rule is None:
            rule = Rule([], target=target, dataspec=data.spec)
        sr = SingleRule(rule, default_prediction=MajorityClass(data))
        sr.set_stats(data)
        return sr

    def _decision_list_fit(self, data: HorizontalBitmapRepresentation, target_class: Any = None,
                           seed: Optional[SeedSelection] = None) -> DecisionList:
        """`model=DecisionList`: one separate-and-conquer loop over *all*
        classes at once, rules in learn order, first match wins. Each
        round picks a seed if there is one (`seed`, else the learner's;
        see `_seed_selection`), gets the next rule from the head selection
        (`target_class`, else the learner's if it is a `HeadSelection`,
        else `HeadPerSeedExample()` with a seed and `HeadPerRefinement()` without), searched
        on the seed's generalizations if the seed implies them
        (`_seed_space`), and removes the examples it covers. A rule is
        refused if it covers no example of its head (`stats.tp == 0`), is
        unconditional, or `stop_covering` rejects it (judged for its
        head); then the seed is set aside and the next one tried, or,
        without a seed, the loop ends -- as it does once nothing is left.
        The default is the majority class of the examples no rule covers
        (of all of them if every example is covered).

        The combinations: AQ's seed covering (`RandomSeed()`, its
        generalizations, `HeadPerSeedExample()`; `AQR`'s default), CN2's ordered
        mode (no seed, all features, `HeadPerRefinement()`), one search per class
        (`HeadPerRule()`), or a seed with a free head (`HeadPerRefinement()`: the
        rule covers the seed but may predict another class)."""
        if data.y is None:
            raise ValueError("SeCo needs data.y")
        if not isinstance(_resolve_covering(getattr(self, "covering", None)), RemovalCovering):
            raise ValueError("a decision list (model=DecisionList) needs RemovalCovering: its first-match "
                             "semantics assumes covered examples are removed")
        seeds = self._seed_selection(seed, multiclass=True)
        seed_space = self._seed_space(seed)
        head = target_class if target_class is not None else self.target_class
        if not isinstance(head, HeadSelection):
            head = HeadPerSeedExample() if seeds is not None else HeadPerRefinement()
        if isinstance(head, FixedClass):
            raise ValueError("a decision list has rules of several classes; for one class use "
                             "model=ConceptModel")
        if isinstance(head, HeadPerSeedExample) and seeds is None:
            raise ValueError("HeadPerSeedExample() needs a seed (seed=FirstSeed() or RandomSeed())")
        rng = np.random.default_rng(self.random_state)
        remaining = np.ones(data.n_samples, dtype=bool)
        rules: List[Rule] = []
        while remaining.any():
            if self.max_rules is not None and len(rules) >= self.max_rules:
                break
            picked = seeds.pick(data, remaining, None, rng) if seeds is not None else None
            if seeds is not None and picked is None:
                break
            space = SeedExample(strategy="index", index=picked) if seed_space and picked is not None else None
            found = head.next_rules(self, data, remaining, picked, space)
            accepted = full = False
            for rule in found:
                if self.max_rules is not None and len(rules) >= self.max_rules:
                    full = True
                    break
                stats = RuleStats.from_rule(rule, data, rule.target, example_mask=remaining)
                if rule.length() == 0 or stats.tp == 0 or (
                        self.stop_covering is not None
                        and self.stop_covering.evaluate(rule, stats, data, rule.target, remaining)):
                    break
                rules.append(rule)
                remaining = remaining & ~rule.covers_data_packed(data)
                accepted = True
            if full:
                break
            if not accepted:
                if picked is None:
                    break
                remaining = remaining.copy()
                remaining[picked] = False   # set the seed aside, try another
        rules = annotate_rules(rules, data)
        default = MajorityClass(data, mask=remaining) if remaining.any() else MajorityClass(data)
        return annotate_default_rule(DecisionList(rules, default_prediction=default), data)

    @produces(DecisionList)
    def _fit_decision_list(self, data: HorizontalBitmapRepresentation, *, target_class: Any = None,
                           seed: Optional[SeedSelection] = None, **kw) -> DecisionList:
        """`fit(data, model=DecisionList)`: one covering loop over all
        classes at once, rules in learn order, first match wins -- see
        `_decision_list_fit`. `target_class=` (a `HeadSelection`) and
        `seed=` override the learner's for this fit."""
        return self._decision_list_fit(data, target_class, seed)

    @produces(FlatRuleSet)
    def _fit_decision_list_set(self, data: HorizontalBitmapRepresentation, *, target_class: Any = None,
                               seed: Optional[SeedSelection] = None, **kw) -> FlatRuleSet:
        """`fit(data, model=FlatRuleSet)`: the same rules as
        `model=DecisionList`, as a rule set resolved by learn order
        (`ListCombiner()`, so it predicts identically) -- e.g. to compare
        other combiners on the same rules via `predict(combiner=...)`."""
        dl = self._decision_list_fit(data, target_class, seed)
        model = FlatRuleSet(dl.rules, default_prediction=dl.default_prediction, combiner=ListCombiner())
        return annotate_default_rule(model, data)


class _ClassCountLaplace:
    """Mixin for a `SeCo` whose default heuristic is the Laplace estimate
    over the problem's classes, ``(tp+1)/(tp+fp+c)`` (Clark & Boswell's
    CN2, Webb's OPUS experiments): unless a heuristic was passed
    (`self._default_laplace` false), each `fit` sets
    `Laplace(n_classes=c)` with ``c`` the number of classes in the
    data."""

    def fit(self, data, model: Optional[type] = None, **model_kwargs):
        if getattr(self, "_default_laplace", False) and data.y is not None:
            n_classes = max(2, len(np.unique(np.asarray(data.y))))
            self.single_rule_learner.heuristic = Laplace(n_classes=n_classes)
        return super().fit(data, model, **model_kwargs)


class CN2(_ClassCountLaplace, SeCo):
    """CN2 (Clark & Niblett, 1989), in the Laplace-heuristic form from
    Clark & Boswell, 1991, as a `SeCo` instantiation -- a first example
    of the pattern this module is meant to support: a named algorithm is
    just a `SeCo` subclass whose constructor picks specific building
    blocks as defaults, while still accepting every one of them as an
    override.

    Two choices define CN2 here:
    - **search heuristic**: `Laplace` -- Clark & Boswell's replacement
      for original CN2's entropy-based one, with their number of classes
      in the denominator: each `fit` sets `Laplace(n_classes=c)` for the
      data's ``c`` classes unless `heuristic` is given (`Laplace()` for
      the two-class form whatever the data).
    - **significance test, as a *filtering* criterion**: Clark &
      Niblett's and Clark & Boswell's search replaces its best rule by a
      candidate only if the candidate is statistically significant
      (`LikelihoodRatio`, CN2's own G-test statistic) and better, and
      runs on until the beam is empty -- `BeamSearch`'s `filtering`
      (`ThresholdPrePruning(LikelihoodRatio(), threshold, operator="<")`,
      rejecting the insignificant ones). If a covering iteration finds
      no significant rule, the search returns `None` and `SeCo` ends
      the loop for that class: with Laplace, Clark & Boswell observe,
      the test acts mostly as this termination criterion. Two cheaper
      readings are available, both weaker under class noise
      (`demos/noise_robustness.py`): the test as `stopping` (halting a
      search the moment the round's best candidate is insignificant,
      ``CN2(stopping=ThresholdPrePruning(LikelihoodRatio(), 3.841,
      operator="<"))``), and the test of
      each search's result only, ending the covering loop
      (``CN2(significance_threshold=None, stop_covering=...)``). Pass
      `significance_threshold=None` to disable the criterion entirely
      (every rule the search lands on is accepted, same as calling
      `SeCo` directly with a `Laplace` heuristic and no `filtering`).

    `significance_threshold` defaults to 3.841, the chi-squared critical
    value at alpha=0.05 with 1 degree of freedom -- the significance
    level most commonly cited for CN2's own test; there's no single
    "the" CN2 default in the literature, so treat this as a reasonable
    starting point, not a fact about the original algorithm.

    `beam_width` defaults to 5 (the original paper's "star size"); pass
    `search` directly instead for anything beyond adjusting beam width
    (e.g. `max_conditions`, or a future non-beam `RuleSearch`) --
    `beam_width` is ignored if `search` is given explicitly.

    `preparation`/`postprocessing` default to `NoSplit`/
    `NoPostProcessing` -- unlike RIPPER/IREP, CN2 has no grow/prune split
    or post-hoc single-rule pruning; the significance test above is its
    only quality gate.

    `filtering`/`stopping`, if passed explicitly, override
    `significance_threshold`'s default construction entirely: the
    significance test is the default `filtering` criterion only while
    neither is given.

    `fit(data)`'s default `ConceptSet` uses `MicroVoteCombiner`, not the
    family's generic `HeuristicMaxCombiner()` -- see `_fit_one_vs_rest` below for
    why. Measured empirically to make little difference on real data
    (rows where covering rules actually disagree in target are rare, and
    even then the two combiners' accuracy differs by a fraction of a
    point either way); the point is matching what Clark & Boswell's own
    algorithm does, not a functional improvement.

    **Ordered CN2.** `fit(data, model=DecisionList)` is CN2's ordered
    mode: one covering loop over all classes, each rule the best of any
    class, predicting the class the heuristic chooses for it (with
    `Laplace`, the covered majority); the default is the majority of the
    examples no rule covers. `heuristic=Entropy()` gives the original's
    entropy over the whole class distribution. One difference remains:
    the significance test is the two-class one for the rule's head
    against the rest, not the original's statistic over all classes.
    `target_class=HeadPerRule()` -- one search per class, the best rule of
    them -- may be the better choice: in `demos/rule_models.py` it was
    clearly the best decision list for CN2, at about three times the
    training time.
    """

    def __init__(
        self,
        target_class: Any = None,
        heuristic: Optional[RuleHeuristic] = None,
        beam_width: int = 5,
        significance_threshold: Optional[float] = 3.841,
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
        space_init: Optional[SearchSpaceInit] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
        random_state: Optional[int] = 0,
        covering: Optional[CoveringStrategy] = None,
        seed: Optional[SeedSelection] = None,
        stop_covering: Optional[PrePruningCriterion] = None,
    ):
        self._default_laplace = heuristic is None
        heuristic = heuristic if heuristic is not None else Laplace()
        search = search if search is not None else BeamSearch(beam_width=beam_width)
        if stopping is None and filtering is None and significance_threshold is not None:
            filtering = ThresholdPrePruning(LikelihoodRatio(), significance_threshold, operator="<")
        single_rule_learner = SingleRuleLearner(
            heuristic=heuristic,
            search=search,
            preparation=preparation,
            postprocessing=postprocessing,
            space_init=space_init,
            filtering=filtering,
            stopping=stopping,
        )
        super().__init__(
            single_rule_learner=single_rule_learner,
            target_class=target_class,
            max_rules=max_rules,
            random_state=random_state,
            covering=covering,
            seed=seed,
            stop_covering=stop_covering,
        )

    @produces(ConceptSet)
    def _fit_one_vs_rest(self, data: HorizontalBitmapRepresentation, **kw) -> ConceptSet:
        """Clark & Boswell (1991)'s own unordered CN2 resolves a clash
        between rules of different classes covering the same row not by
        picking the single best-scoring rule (the family's generic
        `HeuristicMaxCombiner()`) but by summing each rule's own covered-training-
        example class distribution and predicting the largest total --
        `pyrulearn.combiners.MicroVoteCombiner`. Only this one producer is
        overridden: `model=ConceptModel` (one concept, nothing to
        reconcile) and the decision list (`model=DecisionList` /
        `FlatRuleSet` -- ordered CN2, resolved by learn order) are
        unaffected."""
        model = super()._fit_one_vs_rest(data, **kw)
        model.combiner = MicroVoteCombiner()
        return model


class AQR(SeCo):
    """AQR (Clark & Niblett, 1989) -- their reimplementation of Michalski's
    AQ, the algorithm CN2 was designed against, here as `CN2`'s sibling
    `SeCo` subclass. Three choices define it:

    - **search-space start: a single-example seed** (`SeedExample`). Each
      rule is grown as a generalization of one still-uncovered positive
      example, never from the fully-open empty rule -- the defining
      difference from `CN2`'s general-to-specific beam. See
      `SeedExample`'s docstring; ``space_init=SeedExample("random",
      random_state)`` picks a random uncovered positive instead of the
      first.

    - **evaluation: a `LEF`** (Michalski's Lexicographic Evaluation
      Functional, `pyrulearn.heuristics.LEF`). Clark & Niblett describe
      AQ as ranking a complex by *the number of positive examples it
      covers* -- `CoveredPositives`. That alone is a poor way to trim the
      partial star mid-search (it can't tell a nearly-consistent complex
      from an over-general one at the same positive coverage), so the
      default LEF adds two tie-breaks that are standard in Michalski's
      own AQ LEFs and inert at the point a consistent rule is finally
      chosen: `CoveredNegatives` (minimize covered negatives -- steers
      the beam toward completable complexes, the job negative-guided
      star growth does in classical AQ) then `MinimalLength` (minimize
      selectors). Pass any `RuleHeuristic` as `heuristic=` to change it
      -- a bare `CoveredPositives()` for the letter of the paper, a
      different `LEF`, or something else entirely (this is exactly
      `CN2`'s `heuristic=` override, which also accepts a `LEF`).

    - **stopping: consistency, as a *filtering* criterion**. AQR requires
      every rule to cover **no** negative examples -- it has no
      significance test and no noise tolerance (that lack is the whole
      point of Clark & Niblett's comparison: AQR overfits noisy data,
      CN2's significance test does not). That requirement is
      `ThresholdPrePruning(CoveredNegatives(), 0, "<")` as `filtering=`:
      a rule with `fp > 0` is never returned, and if a covering
      iteration can't reach a consistent rule at all, the search returns
      `None` and `SeCo` ends -- so un-fittable (e.g. noisy) data shows up
      as a short or empty ruleset. Pass `require_consistency=False` to
      drop the gate, or `filtering=`/`stopping=` to supply your own.

    The `SeCo` covering loop needs no adjustment for AQR: because every
    accepted rule is consistent, "remove every example the rule covers"
    removes only positives, and every negative stays in scope for every
    subsequent rule -- exactly classical AQ's covering.

    **What this is not (yet):** the search here is `BeamSearch` over the
    seed-restricted feature space, not AQ's literal *star* construction
    (grow the partial star by "add a selector that excludes a specific
    still-covered negative", trim to `maxstar` after each negative). With
    the consistency filter and a negatives-aware LEF the rules found are
    materially the same, and `beam_width == maxstar`, but the classical
    star is more efficient (its per-step candidate set is only the
    selectors that distinguish the seed from some covered negative) and
    *is* by definition "the set of maximally-general consistent
    complexes". A dedicated star `RuleSearch` can be passed as `search=`
    when one exists.

    `beam_width`/`maxstar` default to 5 (the paper's star size); ignored
    if `search` is passed explicitly. `preparation`/`postprocessing`
    default to `NoSplit`/`NoPostProcessing` -- AQR (in the 1989 paper)
    has neither a grow/prune split nor rule truncation.

    **Multi-class:** `AQR`'s `fit(data)` default is `model=DecisionList` --
    one covering loop over all classes (`SeCo._decision_list_fit`): pick a
    random uncovered example, take its label as the head, learn a
    consistent rule seeded on it, remove what it covers, repeat. That's
    AQ's own multi-class covering. A rule is consistent only within its
    own covering scope, so rules of different classes can still overlap
    on examples an earlier rule had already removed; the list resolves
    that by learn order, and prints in that order. Pass
    `model=ConceptSet` / `ConceptCascade` / `PairwiseModel` for a
    decomposition instead, or `model=ConceptModel, label=...` (or a
    `target_class`) for one binary problem. `random_state` seeds the
    example picks.
    """

    _MULTICLASS_DEFAULT = DecisionList

    def __init__(
        self,
        target_class: Any = None,
        heuristic: Optional[RuleHeuristic] = None,
        maxstar: int = 5,
        random_state: Optional[int] = 0,
        require_consistency: bool = True,
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
        space_init: Optional[SearchSpaceInit] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
        stop_covering: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
        covering: Optional[CoveringStrategy] = None,
        seed: Optional[SeedSelection] = None,
    ):
        heuristic = heuristic if heuristic is not None else LEF(
            CoveredPositives(), CoveredNegatives(), MinimalLength()
        )
        search = search if search is not None else BeamSearch(beam_width=maxstar)
        space_init = space_init if space_init is not None else SeedExample("first", random_state)
        if filtering is None and stopping is None and require_consistency:
            filtering = ThresholdPrePruning(CoveredNegatives(), 0, operator="<")
        single_rule_learner = SingleRuleLearner(
            heuristic=heuristic,
            search=search,
            preparation=preparation,
            postprocessing=postprocessing,
            space_init=space_init,
            filtering=filtering,
            stopping=stopping,
        )
        super().__init__(
            single_rule_learner=single_rule_learner,
            target_class=target_class,
            stop_covering=stop_covering,
            max_rules=max_rules,
            random_state=random_state,
            covering=covering,
            seed=seed,
        )


class PFoil(SeCo):
    """PFOIL (Mooney, 1995) -- FOIL's information-gain search heuristic
    (`FoilGain`) run propositionally via hill climbing, paired with
    Quinlan's own MDL-based encoding-length restriction
    (`pyrulearn.pruning.EncodingLengthRestriction`, see its own
    docstring and Quinlan, 1990, *Machine Learning* 5(3):239-266, p.
    251, verified directly against the primary source) as the quality
    gate, in `CN2`'s significance test's place.

    Two choices define PFoil here:
    - **search + heuristic**: `GainAscentHillClimbing` + `FoilGain` --
      `FoilGain` is a `GainHeuristic`, so `GainAscentHillClimbing` is the
      only valid search for it (see its docstring, and `BeamSearch`'s,
      for why gain scores can't be ranked across candidates in flight).
    - **stopping criterion**: `EncodingLengthRestriction`, as
      `stopping=` -- stop extending a clause once it costs more bits to
      write down than to list the positives it covers. Its default
      `polarity=True` already gives the same "leave" reading `CN2`'s
      significance test uses: `reject()==evaluate()`.

    Mooney's own PFOIL (the propositional restriction of FOIL described
    in Mooney, 1995) uses `FoilGain` as its search heuristic but does
    *not* itself apply an MDL stopping rule -- that's Quinlan's own
    contribution from the original (relational) FOIL paper, reused here
    since it needed no relational machinery to restate propositionally.
    Pass `mdl_stopping=False` (and no `filtering=`) to recover Mooney's
    own unrestricted-growth PFOIL exactly -- growth then only stops at
    `GainAscentHillClimbing`'s own unconditional floor (no move improves
    gain at all).

    `filtering`/`stopping`, if passed explicitly, override
    `mdl_stopping`'s default construction entirely -- e.g. pass
    `filtering=EncodingLengthRestriction()` to try it as a filter
    instead (only ever restricts which visited rule is eligible to be
    returned, never changes which move the walk takes -- see
    `GainAscentHillClimbing`'s own docstring).
    """

    def __init__(
        self,
        target_class: Any = None,
        heuristic: Optional[GainHeuristic] = None,
        mdl_stopping: bool = True,
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
        space_init: Optional[SearchSpaceInit] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
        random_state: Optional[int] = 0,
        covering: Optional[CoveringStrategy] = None,
        seed: Optional[SeedSelection] = None,
    ):
        heuristic = heuristic if heuristic is not None else FoilGain()
        search = search if search is not None else GainAscentHillClimbing()
        if stopping is None and filtering is None and mdl_stopping:
            stopping = EncodingLengthRestriction()
        single_rule_learner = SingleRuleLearner(
            heuristic=heuristic,
            search=search,
            preparation=preparation,
            postprocessing=postprocessing,
            space_init=space_init,
            filtering=filtering,
            stopping=stopping,
        )
        super().__init__(
            single_rule_learner=single_rule_learner,
            target_class=target_class,
            max_rules=max_rules,
            random_state=random_state,
            covering=covering,
            seed=seed,
        )


class PFossil(SeCo):
    """FOSSIL (Fürnkranz, 1994): `Correlation` (the four-field/phi
    correlation coefficient) as the rule-refinement heuristic, with
    `correlation_threshold` (default 0.3, FOSSIL's own published value)
    as the quality gate.

    **Hill climbing, not beam search.** FOSSIL's own search is a
    top-down hill-climbing loop -- add the one condition that most
    improves correlation, repeat, stop when nothing improves it -- and
    that's what `PFossil` now does: `HillClimbing` over `Correlation()`
    directly (`HillClimbing` ranks any plain `RuleHeuristic`; no
    `DeltaGain` wrapping needed). `HillClimbing`'s local-optimum stop --
    move on only to a child that beats the current rule's own
    correlation -- *is* "stop when correlation would no longer
    increase", so the search halts at the nearest correlation maximum,
    typically after a handful of conditions.

    This replaces an earlier `BeamSearch(beam_width=5)` + `Correlation`
    default that behaved badly on two fronts, both confirmed on the
    binary-UCI benchmark suite (`demos/seco_learners_comparison`):

    - *Runaway search.* `Correlation` evaluated at the beam's top
      candidate rarely drops below 0.3 as conditions are added (near-
      redundant interval literals barely move it), and its optimistic
      `(tp, 0)` bound sits near 1.0, so neither `stopping` nor
      `BeamSearch`'s optimistic pruning fired -- the beam ran to
      feature-mask exhaustion, exploring rules 40-90 conditions deep
      before returning a short one. Hill climbing stops at the
      correlation peak (~5 conditions), 5-25x faster.
    - *Over-general rules.* `Correlation` (like `WRAcc`; symmetric about
      the ROC diagonal) rewards a rule for covering the positive class
      *and* correctly excluding the negatives -- but in a first-match
      decision list only precision-on-what-fires matters. The beam
      would settle on high-coverage, ~55%-precision rules; sequential
      covering then blanketed the data with the target class and
      accuracy collapsed toward its base rate. Greedily following the
      correlation gradient from the empty rule adds negatives-removing
      conditions first and stops earlier, landing on tighter rules.

    `correlation_threshold` is wired as the `filtering` criterion
    ``ThresholdPrePruning(Correlation(), correlation_threshold, "<")``
    while neither `filtering` nor `stopping` is given: the local-optimum
    stop already decides *when to stop climbing*; the threshold only
    decides *whether the rule the climb lands on is worth keeping* -- if
    its correlation is below the threshold the search returns `None` and
    `SeCo` ends its covering loop. Passing the same criterion as
    `stopping` instead also halts the climb the moment correlation dips
    below it (returning the rule from just before, or `None` if there
    wasn't one yet) -- earlier than the natural peak, so it tends to
    underfit (often one rule). Either way a sub-threshold rule is never
    returned (before 2026, the stopping variant did append them, which
    is what wrecked accuracy). Pass `correlation_threshold=None` to drop
    the gate entirely (every rule the climb lands on is accepted).

    Beam search -- the pre-2026 default -- is still available explicitly:
    `PFossil(target_class=..., search=BeamSearch(beam_width=5),
    heuristic=Correlation())`.
    """

    def __init__(
        self,
        target_class: Any = None,
        heuristic: Optional[RuleHeuristic] = None,
        correlation_threshold: Optional[float] = 0.3,
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
        space_init: Optional[SearchSpaceInit] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
        random_state: Optional[int] = 0,
        covering: Optional[CoveringStrategy] = None,
        seed: Optional[SeedSelection] = None,
    ):
        search = search if search is not None else HillClimbing()
        if heuristic is None:
            heuristic = Correlation()
        if stopping is None and filtering is None and correlation_threshold is not None:
            filtering = ThresholdPrePruning(Correlation(), correlation_threshold, operator="<")
        single_rule_learner = SingleRuleLearner(
            heuristic=heuristic,
            search=search,
            preparation=preparation,
            postprocessing=postprocessing,
            space_init=space_init,
            filtering=filtering,
            stopping=stopping,
        )
        super().__init__(
            single_rule_learner=single_rule_learner,
            target_class=target_class,
            max_rules=max_rules,
            random_state=random_state,
            covering=covering,
            seed=seed,
        )


class Pypper(DecomposingLearner, NativeRuleLearner):
    """Pypper -- a re-implementation of RIPPER (Cohen, 1995, *Fast Effective
    Rule Induction*), not a port of Cohen's code: IREP\\* growth-and-pruning
    plus the `ReplaceReviseOptimization` phase, run per class. See the
    covering-loop stop below for how it differs from the original.

    `fit(data)` returns a `pyrulearn.models.ConceptCascade`: rules for the
    rarest class first, then the next, ..., the most frequent class as the
    catch-all default -- exactly RIPPER's class handling
    (least-frequent-first ordered peeling). Other model types via
    `fit(data, model=...)`:

    - `ConceptModel` -- binary Pypper for one class (`label=` / a
      `target_class`), IREP\\* + optimization, no peeling.
    - `SingleRule` -- just one grown-and-pruned rule for a class, no
      covering loop and no optimization phase.
    - `ConceptSet` -- one-vs-rest Pypper (each class its own full loop).
    - `PairwiseModel` -- round-robin Pypper.
    - `DecisionList` -- one covering loop over all classes, each rule
      grown for the class `FoilGain` prefers (see `_fit_decision_list`).
    - `FlatRuleSet` -- via the `ConceptSet -> FlatRuleSet` converter.

    (Pypper has no `FlatRuleSet` seed-covering producer -- it is not a
    seed-covering algorithm -- but still reaches `FlatRuleSet` by
    flattening its one-vs-rest `ConceptSet`.)

    Per class, a `SeCo` covering loop where each rule is:
    - **grown** by `FoilGain` on a 2/3 growing split (`GrowPruneSplit`),
      with `GainAscentHillClimbing(stop_at_local_optimum=False)` -- IREP's
      grow phase: add the max-gain condition until the rule is consistent
      (or it would drop below two covered positives, RIPPER's `minNo`,
      wired as a `stopping` criterion), *not* stopping at a gain peak,
    - **pruned** -- trailing conditions dropped by `ReducedErrorPruning`
      on the 1/3 held-out split (`Precision`, monotone with RIPPER's own
      ``(p-n)/(p+n)`` rule-value metric),
    then the whole class's rules go through `ReplaceReviseOptimization(k)`
    (`k=2` passes by default).

    The covering loop stops (`stop_covering=`) on whichever fires first
    of `pyrulearn.pruning.EncodingLengthRestriction` (FOIL's per-rule MDL
    stop) and IREP's rule (Fürnkranz & Widmer, 1994): halt once a pruned
    rule's precision on the covering scope drops below 0.5 -- a rule no
    better than a coin flip, and every rule after it, only degrades the
    set. This stands in for Cohen's exact "total ruleset DL is >64 bits
    over the minimum seen" rule (which would also need a retroactive
    trim-back the per-rule `PrePruningCriterion` interface can't express);
    the IREP stop is cruder but is itself a published stopping rule, and
    it fixes the same failure -- a covering loop that keeps tacking on
    low-precision rules until the MDL-guided optimization latches onto
    one of them. There is also no post-optimization residual IREP\\* --
    see `ReplaceReviseOptimization`.

    `k` -- optimization passes. `prune_fraction` -- held-out split size
    for grow/prune (both IREP\\* and the optimization). `grow_heuristic`/
    `search` -- override the growth heuristic / search (defaults `FoilGain`
    and `GainAscentHillClimbing(stop_at_local_optimum=False)`; pass your
    own `search` and you own its `stop_at_local_optimum` too).
    `random_state` seeds every split and the class-order tie-break.
    `seed`/`space_init` -- the example each rule explains and where its
    growth starts, as in `SeCo` (with a seed, its generalizations unless
    `space_init` is given).
    """

    def __init__(
        self,
        k: int = 2,
        prune_fraction: float = 1.0 / 3.0,
        random_state: Optional[int] = 0,
        max_rules: Optional[int] = None,
        grow_heuristic: Optional[RuleHeuristic] = None,
        search: Optional[RuleSearch] = None,
        target_class: Any = None,
        seed: Optional[SeedSelection] = None,
        space_init: Optional[SearchSpaceInit] = None,
    ):
        self.k = k
        self.prune_fraction = prune_fraction
        self.random_state = random_state
        self.max_rules = max_rules
        self.grow_heuristic = grow_heuristic
        self.search = search
        self.target_class = target_class
        self.seed = seed
        self.space_init = space_init

    def _provenance_params(self):
        params = super()._provenance_params()
        for name in ("seed", "space_init"):   # absent unless given: keeps earlier cache keys valid
            if params.get(name) is None:
                params.pop(name, None)
        return params

    def _stage_learner(self) -> SeCo:
        heuristic = self.grow_heuristic if self.grow_heuristic is not None else FoilGain()
        search = (
            self.search if self.search is not None
            else GainAscentHillClimbing(stop_at_local_optimum=False)
        )
        single_rule_learner = SingleRuleLearner(
            heuristic=heuristic,
            search=search,
            preparation=GrowPruneSplit(self.prune_fraction, self.random_state),
            postprocessing=ReducedErrorPruning(Precision()),
            space_init=self.space_init,
            stopping=ThresholdPrePruning(CoveredPositives(), 2, operator="<"),
        )
        return SeCo(
            single_rule_learner=single_rule_learner,
            stop_covering=AnyOf(
                EncodingLengthRestriction(),
                # IREP's stop (Fürnkranz & Widmer, 1994): once a pruned rule
                # is no better than a coin flip on the covering scope, adding
                # it -- and every rule after it -- only hurts.
                ThresholdPrePruning(Precision(), 0.5, operator="<"),
            ),
            max_rules=self.max_rules,
            optimization=ReplaceReviseOptimization(self.k, self.prune_fraction, self.random_state),
            target_class=self.target_class,
            random_state=self.random_state,
            seed=self.seed,
        )

    def _default_model(self, data: HorizontalBitmapRepresentation) -> type:
        head = self.target_class
        if isinstance(head, HeadSelection):
            return ConceptModel if isinstance(head, FixedClass) else DecisionList
        if head is not None:
            return ConceptModel
        return DecisionList if self.seed is not None else ConceptCascade

    @produces(ConceptModel)
    def _fit_concept(self, data: HorizontalBitmapRepresentation, *,
                     label: Any = None, fallback: Any = None) -> ConceptModel:
        """`fit(data, model=ConceptModel, label="a")` -- binary Pypper
        for one class: the configured IREP* + optimization covering loop,
        no peeling."""
        return self._stage_learner()._fit_covering(data, label=label, fallback=fallback)

    @produces(SingleRule)
    def _fit_one_rule(self, data: HorizontalBitmapRepresentation, *, label: Any = None) -> SingleRule:
        """`fit(data, model=SingleRule, label="a")` -- one grown-and-pruned
        rule for `label`, no covering loop and no optimization phase."""
        return self._stage_learner()._fit_one_rule(data, label=label)

    def _fit_binary(self, data: HorizontalBitmapRepresentation, positive: Any,
                    negative: Any = None) -> ConceptModel:
        return self._fit_concept(data, label=positive, fallback=negative)

    @produces(DecisionList)
    def _fit_decision_list(self, data: HorizontalBitmapRepresentation, *, target_class: Any = None,
                           seed: Optional[SeedSelection] = None, **kw) -> DecisionList:
        """`fit(data, model=DecisionList)` -- one IREP\\* covering loop over
        all classes at once (`SeCo._decision_list_fit`), no optimization
        phase: by default each rule grown for whichever class `FoilGain`
        prefers at its first refinement, then pruned for that class; with
        a seed (`seed=`), grown from the seed's features for its class."""
        return self._stage_learner()._decision_list_fit(data, target_class, seed)

