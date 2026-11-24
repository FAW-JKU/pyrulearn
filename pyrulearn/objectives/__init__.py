"""
pyrulearn.objectives
====================

What a rule search maximizes (Section 5 of the paper, "Objectives").
`Objective` is the interface a search uses: the statistics of a rule and
of all its one-condition refinements, computed on a data representation,
their scores and bounds. `HeuristicObjective` scores rules by a
`pyrulearn.heuristics.RuleHeuristic`, i.e., by their class counts;
`ValueSumObjective` sums per-example values over the covered examples,
e.g. the derivatives of a loss (`pyrulearn.objectives.losses`);
`HeadObjective` also chooses the head of a rule.

The heuristics themselves are in `pyrulearn.heuristics`, the losses of
the boosting learners and `LRIError` in `pyrulearn.objectives.losses`.
These classes used to live in `pyrulearn.learners.seco`, which still
re-exports them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..data.attributes import NumericGroup, ThresholdChain
from ..heuristics import LastRule, RuleHeuristic, RuleStats, Score
from ..pruning import PrePruningCriterion
from ..rule import Rule


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

    def criterion_bound(self, heuristic: RuleHeuristic, stats) -> Optional[float]:
        """An upper bound on `heuristic`'s score -- a criterion's, not the
        objective's -- over the refinements of a rule with `stats`, the
        statistics a criterion would see for them; None if there is none
        (`PrePruningCriterion.may_accept_refinements`)."""
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

    @property
    def relative_to_parent(self) -> bool:
        """Whether a refinement is scored relative to its parent (see
        `pyrulearn.heuristics.RuleHeuristic.relative_to_parent`): its
        score then takes the parent's statistics."""
        return bool(getattr(getattr(self, "heuristic", None), "relative_to_parent", False))

    def parents(self, stats: Sequence[Any], index: np.ndarray) -> Any:
        """The statistics of the parent of each refinement in a batch,
        `stats[index[i]]` for the i-th -- for `batch_score` relative to
        the parents."""
        def column(name):
            return np.asarray([float(getattr(st, name)) for st in stats])[index]
        return RuleStats(tp=column("tp"), fp=column("fp"), fn=column("fn"), tn=column("tn"))

    def restat(self, batch) -> Any:
        """`batch` -- statistics from another objective of the same kind,
        the refinement's -- as this objective's own: unchanged for one
        whose statistics don't depend on the heuristic; with the head this
        objective's heuristic chooses for a `HeadObjective`."""
        return batch


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

    supports_bound = True

    def rule_stats(self, data, target_class, rule, handle) -> RuleStats:
        return stats_from_handle(data, target_class, rule, handle)

    def children(self, data, target_class, mask, handle, stats):
        features, tps, fps, fns, tns, same, dead = live_open_children(data, target_class, mask, handle, stats)
        return features, RuleStats(tp=tps, fp=fps, fn=fns, tn=tns, length=stats.length + 1), same, dead

    def batch_score(self, batch, parent=None):
        if parent is not None and self.heuristic.relative_to_parent:
            return self.heuristic.batch_score(batch, parent)
        return self.heuristic.batch_score(batch)

    def score(self, stats) -> Score:
        return self.heuristic.score(stats)

    def score_one(self, stats, parent=None) -> Score:
        if parent is not None and self.heuristic.relative_to_parent:
            return batch_score_one(self.heuristic, stats, parent)
        return batch_score_one(self.heuristic, stats)

    def batch_bound(self, batch):
        return self.heuristic.batch_bound(batch)

    def bound(self, stats) -> Score:
        return self.heuristic.bound(stats)

    def bound_one(self, stats) -> Score:
        v = np.asarray(self.heuristic.batch_bound(stats))
        return float(v) if v.ndim == 0 else tuple(float(x) for x in v.reshape(-1))

    def criterion_bound(self, heuristic, stats) -> Optional[float]:
        return float(heuristic.bound(stats))

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


def selection_objective(objective: Objective, selection: Optional[Union[RuleHeuristic, Objective]],
                        data, target_class: Any, example_mask: Optional[np.ndarray]) -> Optional[Objective]:
    """The objective a search selects its result by (see `RuleSearch`):
    None without a `selection`, else `selection` as an objective of the
    same kind as the search's own `objective` -- for a fixed class or for
    a rule of any class -- started (`begin`) on the same data. A
    selection compares rules reached from different parents, so it can't
    be relative to the parent. Without a `selection`, an objective
    relative to the parent rule selects the last rule (`LastRule`): its
    scores along a refinement path have no common scale."""
    if selection is None:
        if not objective.relative_to_parent:
            return None
        selection = HeadObjective(LastRule()) if isinstance(objective, HeadObjective) else LastRule()
    sel = as_objective(selection)
    heuristic = getattr(sel, "heuristic", None)
    if sel.relative_to_parent:
        raise ValueError(f"a selection compares rules from different parents and can't score them relative to "
                         f"their parents, as {heuristic!r} does")
    if not any(isinstance(objective, kind) and isinstance(sel, kind) for kind in (HeuristicObjective, HeadObjective)):
        raise ValueError(f"a selection needs a heuristic of the same kind as the search's ({objective!r}); "
                         f"got {sel!r}")
    sel.begin(data, target_class, example_mask)
    return sel


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
    A heuristic relative to the parent rule (a gain such as `FoilGain`)
    scores each head against the parent's statistics for the same head,
    which makes heads comparable only at the first refinement, where all
    of them start from the scope: there the head is chosen, and then it
    stays. A rule is dead once it covers no example in scope, and
    consistent once all it covers are of its head's class."""

    supports_criteria = True

    def __init__(self, heuristic: RuleHeuristic):
        self.heuristic = heuristic
        self._relative = heuristic.relative_to_parent

    supports_bound = True

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

    def restat(self, batch: HeadStats) -> HeadStats:
        """`batch` with the head (and score) this objective's heuristic
        chooses -- e.g. a refinement's statistics for a selection."""
        return self._stats(batch.covered, batch.count, batch.length, None)

    def _stats(self, covered: np.ndarray, count: Any, length: Any, parent: Optional[HeadStats]) -> HeadStats:
        if self._relative and parent is not None and np.ndim(parent.length) == 0 and parent.length > 0:
            # relative to the parent *for the same head*: comparing heads is
            # fair only from the root, where every head starts from its
            # prior -- further down, a rare head the parent covers at a low
            # precision would show a large gain from a handful of examples.
            # So the head chosen at the first refinement stays.
            head = np.full(covered.shape[1:], int(parent.head), dtype=np.int64)
            score = self.heuristic.head_scores(covered, self.totals, length, parent.covered)[int(parent.head)]
        else:
            score, head = self.heuristic.best_head(
                covered, self.totals, length,
                parent.covered if self._relative and parent is not None else None)
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
        if parent is not None and self._relative:   # rescored relative to `parent`
            return float(self._stats(stats.covered, stats.count, stats.length, parent).score)
        return float(stats.score)

    def batch_bound(self, batch: HeadStats):
        return self.heuristic.head_bound(batch.covered, self.totals, batch.length)

    def bound_one(self, stats: HeadStats) -> Score:
        return float(self.heuristic.head_bound(stats.covered, self.totals, stats.length))

    def criterion_bound(self, heuristic, stats: HeadStats) -> Optional[float]:
        # over every head: a refinement may predict another class
        return float(heuristic.head_bound(stats.covered, self.totals, stats.length))

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
