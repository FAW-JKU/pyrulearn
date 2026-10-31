"""
pyrulearn.pruning
====================

`PrePruningCriterion`: a single per-candidate test, consumed three
different ways depending on who's asking --

- as a **filter** (a region no *returned* rule may occupy; checked
  continuously and independent of exploration -- a candidate sitting in
  the excluded region keeps getting refined exactly as it would
  otherwise), via `pyrulearn.learners.seco.BeamSearch`/`HillClimbing`'s
  `filtering=`,
- as a **stopping** trigger -- everything `filtering` does (a returned
  rule is never in the excluded region) *plus* halting the search the
  moment the frontier crosses into it (checked from the first
  refinement onward -- never against the seed candidate). Because it
  quits rather than keep looking, `stopping` can miss a rule that would
  have re-entered the acceptable region a few refinements deeper, which
  `filtering` alone would still find. Via those same classes'
  `stopping=`, or
- as the outer covering loop's own stop condition, via
  `pyrulearn.learners.seco.SeCo`'s `stop_covering=` -- consulted through bare
  `evaluate()` alone, with `polarity`/`accept`/`reject` playing no role
  (there's only ever one covering loop, never a population to filter
  among, so a fired criterion always just means "stop, and discard what
  was just found" -- see `SeCo`'s own docstring).

Same criterion, same `evaluate`/`accept`/`reject` methods regardless of
which of the three a given instance ends up passed to; see each
consuming class's own docstring for exactly how it applies them.

"Pre-pruning" (as opposed to `pyrulearn.learners.seco.ReducedErrorPruning`'s
*post*-pruning, which truncates an already-fully-grown rule afterward)
-- filtering and stopping both act mid-search, before a rule is ever
finished, unlike post-pruning's "grow it fully, then cut it back"; the
covering-loop use acts between complete rules, but still before the
*ruleset* as a whole is finished.

This module only defines the criteria themselves; every consumer
(`RuleSearch`/`BeamSearch`/`HillClimbing`/`GainAscentHillClimbing`/
`SeCo`) lives in `pyrulearn.learners.seco`.
"""

from __future__ import annotations

import itertools
import math
import operator as _operator
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from .heuristics import RuleHeuristic, RuleStats
from .data import DataRepresentation
from .rule import Rule

#: comparison operators `ThresholdPrePruning` accepts, by symbol
COMPARISON_OPERATORS = {
    ">=": _operator.ge, ">": _operator.gt,
    "<=": _operator.le, "<": _operator.lt,
    "==": _operator.eq, "!=": _operator.ne,
}


class PrePruningCriterion(ABC):
    """A single per-candidate test -- see the module docstring for how
    the same instance is consumed as a filter, a stopping trigger, or
    (via bare `evaluate()` alone, ignoring `polarity`/`accept`/`reject`
    entirely) `pyrulearn.learners.seco.SeCo`'s own `stop_covering`.

    `evaluate` is the one method a subclass implements: a raw,
    per-round predicate, phrased however is natural for the check at
    hand (e.g. "is this rule currently significant"). On its own it
    carries no notion of "good" or "bad" -- `polarity` (default `True`)
    says whether `evaluate() == True` means REJECT (the default) or
    ACCEPT. `reject`/`accept` are generic and never overridden:

        reject(...) == (evaluate(...) == polarity)
        accept(...) == not reject(...)

    Choosing `polarity` is a real decision, not boilerplate -- it's
    what lets the same shape of criterion express two different search
    behaviors when used for *stopping*. Phrase `evaluate` as "the
    condition that should make the search consider stopping" (it needs
    to become true at the exact moment that matters, whichever
    direction that is), then pick `polarity` to make `reject`/`accept`
    mean what you actually want for filtering:

    - CN2's significance test: `evaluate` = "is this rule no longer
      significant" (`LikelihoodRatio() < threshold`) -- becomes true
      exactly when significance is lost, the moment you want the
      search to stop and consider its answer. `polarity=True` (the
      default) gives `reject() == evaluate()`: reject exactly the
      insignificant rules -- the natural filtering meaning too.
    - "Stop once precision reaches 0.8": `evaluate` = "is precision now
      >= 0.8" -- becomes true exactly when you've reached "good
      enough". Here `accept()`, not `reject()`, should equal
      `evaluate()` directly, so `polarity=False`.

    A search consuming this as a stopping trigger checks `evaluate`
    each round; the moment it fires, `accept` on that same candidate
    decides what to return -- itself if `accept()` is `True` (in the
    CN2 example this branch is never taken: `reject()==evaluate()`
    means `accept()` is always `False` exactly when `evaluate()` just
    fired), otherwise the best rule the search had from *strictly
    before* the triggering round that was in every criterion's
    acceptable region (`pyrulearn.learners.seco.BeamSearch`: its running best;
    `pyrulearn.learners.seco.HillClimbing`: the most recent such rule -- they
    differ because one tracks a running best and the other doesn't; see
    each class's own docstring). That is `None` if there was no such
    rule -- so a criterion used as `stopping` never hands back a rule
    from its own reject region (e.g. FOSSIL's `Correlation < 0.3`
    firing on the first refinement returns `None`, not that
    sub-threshold rule).

    `stats` is passed in already computed (by whatever's driving the
    search) rather than recomputed here -- avoids a second `RuleStats.
    from_rule` pass over the same candidate purely for this check.

    Unconditional floor, independent of any criterion configured here:
    a `RuleSearch` stops specializing a candidate once its `stats.tp ==
    0` (provably pointless -- specialization only ever shrinks coverage,
    so every descendant would also have `tp == 0`), and, unless
    `optimistic_pruning=False`, once the heuristic's score at the
    candidate's optimistic `(tp, 0)` point (the best any refinement
    could ever reach) no longer beats the best rule found so far --
    `pyrulearn.learners.seco.BeamSearch`/`HillClimbing`/`GainAscentHillClimbing`
    own the details (and the flag), including how the whole search, not
    just one lineage, halts once *nothing* in flight still promises to
    improve. That optimistic bound subsumes the older `fp == 0` floor it
    replaced: a pure rule's `(tp, 0)` bound is just its own score, which
    the running best already reflects. These checks are direct scalar
    comparisons / single `heuristic.score` calls in the search loop
    itself, not routed through this class -- cheap and frequent enough
    that virtual dispatch would be a real fraction of their cost; a
    `PrePruningCriterion` is for genuinely configurable, typically
    heavier checks (e.g. CN2's significance test) layered on top of that
    floor.
    """

    polarity: bool = True

    @abstractmethod
    def evaluate(
        self,
        rule: Rule,
        stats: RuleStats,
        data: DataRepresentation,
        target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> bool:
        raise NotImplementedError

    def reject(
        self,
        rule: Rule,
        stats: RuleStats,
        data: DataRepresentation,
        target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> bool:
        return self.evaluate(rule, stats, data, target_class, example_mask) == self.polarity

    def accept(
        self,
        rule: Rule,
        stats: RuleStats,
        data: DataRepresentation,
        target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> bool:
        return not self.reject(rule, stats, data, target_class, example_mask)


class ThresholdPrePruning(PrePruningCriterion):
    """`evaluate()` is ``heuristic score <operator> threshold``. Phrase
    the comparison as whichever condition should make the search
    consider stopping (see `PrePruningCriterion` for why that's not
    always "the bad one"), and set `polarity` to get the filtering
    meaning you actually want. CN2's significance test is
    `ThresholdPrePruning(LikelihoodRatio(), critical_value, "<")` -- the
    default `polarity=True` already reads correctly: reject the
    insignificant ones -- the exact same instance also works unchanged
    as `pyrulearn.learners.seco.SeCo`'s `stop_covering=`, since that only ever
    reads `evaluate()`, which `polarity` doesn't affect.
    `ThresholdPrePruning(pyrulearn.heuristics.UncoveredPositives(),
    -p, ">=")` is another `stop_covering=` example: stop once at most
    `p` positives would remain uncovered. See `COMPARISON_OPERATORS`
    for every symbol `operator` accepts (`>=`, `>`, `<=`, `<`, `==`,
    `!=`).

    `heuristic` needs to return a plain float here (`threshold` is one,
    and comparing a tuple to a float raises `TypeError`) -- a
    `pyrulearn.heuristics.LEF` isn't usable directly as `heuristic`; pass
    one of its constituent heuristics instead if you need a threshold on
    a LEF-ranked search.
    """

    def __init__(
        self, heuristic: RuleHeuristic, threshold: float, operator: str = ">=", polarity: bool = True,
    ):
        if operator not in COMPARISON_OPERATORS:
            raise ValueError(f"operator must be one of {sorted(COMPARISON_OPERATORS)}, got {operator!r}")
        self.heuristic = heuristic
        self.threshold = threshold
        self.operator = operator
        self.polarity = polarity

    def evaluate(
        self,
        rule: Rule,
        stats: RuleStats,
        data: DataRepresentation,
        target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> bool:
        score = self.heuristic.score(stats)
        return COMPARISON_OPERATORS[self.operator](score, self.threshold)


def _log2_factorial(n: int) -> float:
    return math.lgamma(n + 1) / math.log(2)


def _log2_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


class EncodingLengthRestriction(PrePruningCriterion):
    """FOIL's own MDL-based stopping criterion (Quinlan, 1990, *Machine
    Learning* 5(3):239-266, p. 251, verified directly against the
    primary source): stop extending a clause the moment it becomes more
    expensive to write down than to simply list which examples it
    covers.

    Quinlan phrases this for first-order Horn clauses, where each
    literal is a relation applied to a tuple of (bound or free)
    variables; the bits needed to code one literal are 1 (negated or
    not) + log2(number of relations) (which relation) + log2(number of
    possible arguments) (which variables). This propositional collapse
    drops the last term entirely (a propositional literal binds no
    variables) and folds "which relation" into "which feature", giving
    `1 + log2(d)` bits per literal, with `d` = the number of features
    available in `data.spec` -- exactly the simplification
    the formula reduces to outside a relational setting.

    `evaluate()`: with `l = rule.length()` (literals so far) and `N`/`p`
    = `stats`'s total examples / covered positives (within whatever
    scope `stats` was computed against -- the search's own
    `example_mask`, same as everywhere else in this module),

        encoding_length = l * (1 + log2(d)) - log2(l!)
        benefit = log2(N) + log2(C(N, p))
        evaluate() = encoding_length > benefit

    fires the moment growing the clause further costs more bits than it
    saves -- exactly the round Quinlan's own criterion is meant to
    intervene at. The `-log2(l!)` term (Quinlan's own correction, kept
    as-is) accounts for every ordering of the same `l` literals meaning
    the same clause. `polarity=True` (the default, unchanged) gives the
    CN2-shaped reading: `reject()==evaluate()`, rejecting exactly the
    over-expensive rules -- the same "leave" direction CN2's
    significance test uses (see `PrePruningCriterion`'s own docstring),
    and the natural choice as a `stopping=` criterion for
    `pyrulearn.learners.seco.PFoil`.

    Never fires against the empty (length-0) rule -- `evaluate()`
    returns `False` unconditionally there, matching every other
    criterion in this module's convention that a seed candidate is
    never itself "too expensive" (an empty clause costs 0 bits to
    encode by construction).
    """

    def evaluate(
        self,
        rule: Rule,
        stats: RuleStats,
        data: DataRepresentation,
        target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> bool:
        length = rule.length()
        if length == 0:
            return False
        n_total = stats.tp + stats.fp + stats.fn + stats.tn
        if n_total <= 0:
            return False
        n_features = data.spec.n_features
        bits_per_literal = 1.0 + math.log2(n_features) if n_features > 0 else 1.0
        encoding_length = length * bits_per_literal - _log2_factorial(length)
        benefit = math.log2(n_total) + _log2_choose(n_total, stats.tp)
        return encoding_length > benefit


class AnyOf(PrePruningCriterion):
    """Rejects once *any* of `criteria` rejects (logical OR). Combines
    at the `reject()` level -- each constituent's own `polarity`
    already resolved -- rather than raw `evaluate()`, so this composes
    correctly no matter what polarity each constituent uses (combining
    raw `evaluate()` values directly would invert under De Morgan's law
    for any constituent using `polarity=False`). Short-circuits left to
    right.
    """

    def __init__(self, *criteria: PrePruningCriterion):
        self.criteria = criteria

    def evaluate(
        self,
        rule: Rule,
        stats: RuleStats,
        data: DataRepresentation,
        target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> bool:
        return any(c.reject(rule, stats, data, target_class, example_mask) for c in self.criteria)


class AllOf(PrePruningCriterion):
    """Rejects only once *every* one of `criteria` rejects (logical
    AND) -- see `AnyOf` for why this combines via each constituent's
    own `reject()`, not raw `evaluate()`. Short-circuits left to right.
    """

    def __init__(self, *criteria: PrePruningCriterion):
        self.criteria = criteria

    def evaluate(
        self,
        rule: Rule,
        stats: RuleStats,
        data: DataRepresentation,
        target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> bool:
        return all(c.reject(rule, stats, data, target_class, example_mask) for c in self.criteria)


class ProductiveRule(PrePruningCriterion):
    """Webb's filter for *productive* rules (Webb, "Discovering
    significant patterns", Machine Learning 2007; Magnum Opus): a rule is
    productive if it scores strictly higher than every generalization --
    every proper subset of its conditions, the empty rule (the class's
    prior) included. An unproductive rule adds conditions that don't
    improve on a simpler rule, like ``sex = male, parch != 6`` next to
    ``sex = male``. `evaluate()` is "unproductive", so with the default
    `polarity` a search's `filtering=` keeps only productive rules.

    `heuristic` is what "scores higher" means: `Precision` (the rule's
    confidence) by default, as in Webb's definition. Checking all
    ``2**length - 1`` generalizations makes it a filter for short rules
    (a top-k or branch-and-bound search with `max_conditions`); the
    generalizations' stats are cached for as long as `data`, the target
    and the example mask stay the same. As a `stopping=` criterion it
    would stop a lineage at its first unproductive rule -- use it for
    filtering.
    """

    def __init__(self, heuristic: Optional[RuleHeuristic] = None, polarity: bool = True):
        self.heuristic = heuristic
        self.polarity = polarity
        self._context: Optional[tuple] = None
        self._scores: dict = {}

    def _score_of(self, features: frozenset, data, target_class: Any, example_mask) -> Any:
        if (self._context is None or self._context[0] is not data or self._context[1] != target_class
                or self._context[2] is not example_mask):
            self._context = (data, target_class, example_mask)
            self._scores = {}
        score = self._scores.get(features)
        if score is None:
            rule = Rule(sorted(features), target=target_class, dataspec=data.spec)
            stats = RuleStats.from_rule(rule, data, target_class, example_mask=example_mask)
            score = self._scores[features] = self._heuristic().score(stats)
        return score

    def _heuristic(self) -> RuleHeuristic:
        if self.heuristic is not None:
            return self.heuristic
        from .heuristics import Precision
        return Precision()

    def evaluate(
        self,
        rule: Rule,
        stats: RuleStats,
        data: DataRepresentation,
        target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> bool:
        features = [lit.feature for lit in rule.conditions]
        if not features:
            return False
        own = self._heuristic().score(stats)
        for size in range(len(features)):
            for subset in itertools.combinations(features, size):
                if not own > self._score_of(frozenset(subset), data, target_class, example_mask):
                    return True
        return False
