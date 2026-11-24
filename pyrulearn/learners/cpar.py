"""
pyrulearn.learners.cpar
=======================

`CPAR` -- Classification based on Predictive Association Rules (Yin &
Han, SDM 2003): FOIL-style rule growing on weighted examples (their PRM,
Predictive Rule Mining) that follows every condition nearly as good as
the best one, so that one search can yield several rules; covered
positives are down-weighted instead of removed; and a class is predicted
by the mean expected accuracy of its best `k` rules covering the example.

Two implementations of the same algorithm, learning the same models:

- `CPAR` is a configuration of the separate-and-conquer framework
  `pyrulearn.learners.seco.SeCo`: `FoilGain` as the heuristic,
  `GainAscentHillClimbing(min_gain=..., branch_similarity=...)` as the
  search (the branching is what yields several rules per search),
  `WeightedCovering(MultiplicativeReweighting, PositiveWeightBelow)` as
  the covering, and `pyrulearn.combiners.TopKMeanCombiner` for
  prediction. Every part can be exchanged, and it runs on every data
  representation.
- `DenseCPAR` is the same algorithm specialized for speed: it scores
  every feature of a search node in one matrix product over a dense
  float copy of `data.X`, without constraint propagation, so it needs a
  `HorizontalBitmapRepresentation`. Same parameters, same models, 2-4x
  faster than `CPAR`.

The README (*Native learning algorithms*, "Components first, specialized
versions second") uses the pair as the example of how pyrulearn trades
the two off.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..combiners import RuleCombiner, TopKMeanCombiner
from ..data import HorizontalBitmapRepresentation
from ..heuristics import FoilGain, GeneralizedMEstimate, RuleHeuristic, RuleStats
from ..models import ConceptModel, ConceptSet, MajorityClass, annotate_default_rule, annotate_rules
from ..rule import Rule
from .base import DEFAULT_MAX_AUTO_CONVERT_CELLS, NativeRuleLearner, produces
from .seco import (
    CoveringStrategy, GainAscentHillClimbing, MultiplicativeReweighting, PositiveWeightBelow, SearchSpaceInit, SeCo,
    SeedSelection, SingleRuleLearner, WeightedCovering,
)


class _CPARParameters:
    """The parameters and the parts both implementations share."""

    def _set_cpar_params(
        self,
        decay: float,
        min_total_weight: float,
        min_gain: float,
        gain_similarity: float,
        k: int,
        max_length: Optional[int],
        max_rounds: int,
        heuristic: Optional[RuleHeuristic],
        covering: Optional[CoveringStrategy],
        combiner: Optional[RuleCombiner],
    ) -> None:
        if not 0.0 < gain_similarity <= 1.0:
            raise ValueError(f"gain_similarity must be in (0, 1], got {gain_similarity}")
        if not min_gain > 0:
            raise ValueError(f"min_gain must be positive (a condition changing nothing gains 0), got {min_gain}")
        if heuristic is not None and not heuristic.relative_to_parent:
            raise ValueError(
                f"CPAR needs a gain over the rule before the new condition, got {heuristic!r} -- e.g. "
                f"Adaptive(Gain({heuristic!r}))"
            )
        reweighting = getattr(covering, "reweighting", None)
        if reweighting is not None and reweighting.assigns_rule_weights:
            raise ValueError(f"{type(reweighting).__name__} fits rule weights, which CPAR's rule sets "
                             "can't hold -- use a boosting learner (e.g. Slipper)")
        self.decay = decay
        self.min_total_weight = min_total_weight
        self.min_gain = min_gain
        self.gain_similarity = gain_similarity
        self.k = k
        self.max_length = max_length
        self.max_rounds = max_rounds
        self.heuristic = heuristic
        self.covering = covering
        self.combiner = combiner

    def _heuristic(self) -> RuleHeuristic:
        return self.heuristic if self.heuristic is not None else FoilGain()

    def _covering(self) -> CoveringStrategy:
        """`covering`, or by default the paper's: positives covered by `k`
        rules weigh ``decay ** k``, until the positives' weight is below
        `min_total_weight` of the start, at most `max_rounds` rules."""
        if self.covering is not None:
            return self.covering
        return WeightedCovering(MultiplicativeReweighting(self.decay), PositiveWeightBelow(self.min_total_weight),
                                max_rounds=self.max_rounds)

    def _combiner(self, n_classes: int) -> RuleCombiner:
        if self.combiner is not None:
            return self.combiner
        return TopKMeanCombiner(GeneralizedMEstimate(m=n_classes, cost=1.0 / n_classes), k=self.k)

    def _admits(self, gain: float, threshold: Optional[float]) -> bool:
        """Whether a child with this `gain` is followed: at least
        `min_gain`, and once the best child is known, at least
        ``best * gain_similarity`` (`threshold`)."""
        return gain >= (self.min_gain if threshold is None else threshold)

    def _threshold(self, best: float) -> float:
        return max(best * self.gain_similarity, self.min_gain)

    def _check_classes(self, data: Any) -> int:
        if data.y is None:
            raise ValueError(f"{type(self).__name__} needs data.y")
        n_classes = len(np.unique(np.asarray(data.y)))
        if n_classes < 2:
            raise ValueError(f"{type(self).__name__} needs at least two classes")
        return n_classes


class CPAR(_CPARParameters, SeCo):
    """CPAR (Yin & Han 2003). For each class `c` against the rest:

    1. Grow rules on the current example weights, FOIL-style: starting
       from the empty rule, repeatedly add the condition with the highest
       weighted FOIL gain ``P* (log2 P*/(P*+N*) - log2 P/(P+N))`` (``P``,
       ``N``: the weights of the positives and negatives the rule covers,
       ``P*``, ``N*`` with the condition added), until no condition gains
       at least `min_gain`. Every other condition whose gain is within
       `gain_similarity` of the best (``gain >= best * gain_similarity``)
       starts a copy of the rule that is grown further the same way -- so
       one search can yield several rules.
    2. For each rule found, multiply the weights of the positives it
       covers by `decay` (`MultiplicativeReweighting`); negatives keep
       their weight.
    3. Repeat until the positives' total weight is below
       `min_total_weight` of the start (`PositiveWeightBelow`), or no rule
       is found.

    The model is a `ConceptSet` with one `ConceptModel` per class (rules
    found more than once appear once). A class is predicted by the mean
    expected accuracy ``(nc + 1) / (n + K)`` (``K`` classes; the rules'
    training stats) of its best `k` rules covering the example
    (`TopKMeanCombiner` with `GeneralizedMEstimate(m=K, cost=1/K)`); rows
    no rule covers get the majority class.

    Defaults are the paper's: ``decay = 2/3``, ``min_total_weight =
    0.05``, ``min_gain = 0.7``, ``gain_similarity = 0.99``, ``k = 5``.
    `gain_similarity = 1` (only exact ties copied) is close to PRM without
    copying. Numeric attributes come already binarized
    (`pyrulearn.data.io.build_dataspec`); the data's row weights multiply
    with the covering weights. `max_length` (default none) caps rule
    length, `max_rounds` the number of rules per class.

    **A configuration of `SeCo`.** CPAR is the separate-and-conquer
    framework with these building blocks -- nothing CPAR-specific is left
    in this class but the choice of them:

    - heuristic: `heuristic` (default `FoilGain()`), a gain over the rule
      before the condition (relative to the parent rule), since
      `min_gain` and `gain_similarity` are thresholds on that improvement;
      `Adaptive(Gain(h))` makes one from any heuristic `h`;
    - search: `GainAscentHillClimbing(max_conditions=max_length,
      min_gain=min_gain, branch_similarity=gain_similarity)` -- step 1;
      its `search_all` yields every lineage's rule, and `SeCo`'s covering
      loop accepts each;
    - covering: `covering` (default ``WeightedCovering(
      MultiplicativeReweighting(decay), PositiveWeightBelow(
      min_total_weight), max_rounds=max_rounds)``) -- steps 2 and 3; e.g.
      `AdditiveReweighting` (CN2-SD), or `RemovalCovering` for plain
      separate-and-conquer;
    - prediction: `combiner` (default the top-`k` mean above).

    So the same parts combine differently elsewhere: `PFoil` with a
    branching search, or `CPAR` with another covering. As a `SeCo` it also
    decomposes classes like the other SeCo learners (`model=ConceptCascade`,
    `model=PairwiseModel`), and runs on every data representation.

    `DenseCPAR` is the same algorithm specialized for speed on a
    `HorizontalBitmapRepresentation`: same parameters, same models, 2-4x
    faster. An implied condition never changes coverage, so its FOIL
    gain is 0, below `min_gain` -- `DenseCPAR` never picks one either,
    the search's constraint propagation only spares `CPAR` from scoring it.
    """

    def __init__(
        self,
        decay: float = 2.0 / 3.0,
        min_total_weight: float = 0.05,
        min_gain: float = 0.7,
        gain_similarity: float = 0.99,
        k: int = 5,
        max_length: Optional[int] = None,
        max_rounds: int = 1000,
        heuristic: Optional[RuleHeuristic] = None,
        covering: Optional[CoveringStrategy] = None,
        combiner: Optional[RuleCombiner] = None,
        target_class: Any = None,
        seed: Optional[SeedSelection] = None,
        space_init: Optional[SearchSpaceInit] = None,
    ):
        self._set_cpar_params(decay, min_total_weight, min_gain, gain_similarity, k, max_length, max_rounds,
                              heuristic, covering, combiner)
        search = GainAscentHillClimbing(max_conditions=max_length, optimistic_pruning=False,
                                        min_gain=min_gain, branch_similarity=gain_similarity)
        SeCo.__init__(self, SingleRuleLearner(self._heuristic(), search=search, space_init=space_init),
                      target_class=target_class, covering=self._covering(), seed=seed)

    @produces(ConceptSet)
    def _fit_one_vs_rest(self, data: Any, **kw) -> ConceptSet:
        n_classes = self._check_classes(data)
        model = super()._fit_one_vs_rest(data, **kw)
        model.combiner = self._combiner(n_classes)
        return model


class DenseCPAR(_CPARParameters, NativeRuleLearner):
    """`CPAR`, specialized for speed: same algorithm, same parameters,
    same models (see `CPAR` for both), 2-4x faster -- at the price of
    needing a `HorizontalBitmapRepresentation` and its dense matrix, and of
    deciding multi-class one way only (one-vs-rest, a `ConceptSet`).

    **Data representation.** `NATIVE_REPRESENTATIONS = (HorizontalBitmapRepresentation,)`
    -- `_fit_native` converts anything else via
    `NativeRuleLearner.ensure_representation`, same as `DenseENDER`. `_grow`
    scores every feature of a node at once, ``(wp * cov) @ Xf`` over a
    float copy of the whole matrix, and masks the ones already in the
    rule out of the *result*. It never removes an *implied* feature from
    consideration (no constraint propagation), so its open set stays
    close to full width the whole search -- exactly the regime where
    that wins: measured 15x faster than slicing to just the open
    features first, on `spambase` (`ROADMAP.md`'s "Design decisions" has
    the numbers and why this is the *opposite* regime from
    `BeamSearch`/`HillClimbing`).
    """

    #: see the "Data representation" paragraph above
    NATIVE_REPRESENTATIONS = (HorizontalBitmapRepresentation,)

    def __init__(
        self,
        decay: float = 2.0 / 3.0,
        min_total_weight: float = 0.05,
        min_gain: float = 0.7,
        gain_similarity: float = 0.99,
        k: int = 5,
        max_length: Optional[int] = None,
        max_rounds: int = 1000,
        heuristic: Optional[RuleHeuristic] = None,
        covering: Optional[CoveringStrategy] = None,
        combiner: Optional[RuleCombiner] = None,
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
    ):
        self._set_cpar_params(decay, min_total_weight, min_gain, gain_similarity, k, max_length, max_rounds,
                              heuristic, covering, combiner)
        self.max_auto_convert_cells = max_auto_convert_cells

    def _default_model(self, data: Any) -> type:
        return ConceptSet

    @produces(ConceptSet)
    def _fit_native(self, data: Any, **kw) -> ConceptSet:
        data = self.ensure_representation(data, self.max_auto_convert_cells, purpose="DenseCPAR's dense-matrix scoring")
        self._check_classes(data)
        y = np.asarray(data.y)
        classes = [c.item() if isinstance(c, np.generic) else c for c in np.unique(y)]
        X = np.asarray(data.X, dtype=bool)
        Xf = X.astype(float)
        base_w = np.ones(len(y)) if data.weights is None else data.weights.astype(float)
        concepts = []
        for c in classes:
            bodies = self._rules_for(X, Xf, y == c, base_w)
            rules = [Rule(list(b), target=c, dataspec=data.spec) for b in bodies]
            concepts.append(ConceptModel(annotate_rules(rules, data), label=c))
        model = ConceptSet(concepts, default_prediction=MajorityClass(data), combiner=self._combiner(len(classes)))
        return annotate_default_rule(model, data)

    # -- one class -----------------------------------------------------------

    def _rules_for(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray,
                   base_w: np.ndarray) -> List[Tuple[int, ...]]:
        """The covering loop for one class -- `SeCo`'s, for `CPAR`: until
        the covering strategy is done, one search on the data's row weights
        times the covering weights, then one covering update for every rule
        it yields. A rule found again still reweights but is kept once."""
        covering = self._covering()
        state = covering.start(None, positive)
        found: Dict[Tuple[int, ...], None] = {}
        while not covering.exhausted(state):
            bodies = self._grow(X, Xf, positive, base_w * state.scope)
            if not bodies:
                break
            for body in bodies:
                found.setdefault(body, None)
                covering.update(state, np.all(X[:, list(body)], axis=1))
        return list(found)

    def _grow(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray,
              w: np.ndarray) -> List[Tuple[int, ...]]:
        """The rules one search yields on the weights `w`: the best-gain
        lineage and every copy branched off at a nearly-as-good condition."""
        heuristic = self._heuristic()
        wp, wn = w * positive, w * ~positive
        n_pos, n_neg = float(wp.sum()), float(wn.sum())
        results: List[Tuple[int, ...]] = []
        seen: Set[frozenset] = set()

        def grow(body: List[int], cov: np.ndarray) -> None:
            key = frozenset(body)
            if key in seen:
                return
            seen.add(key)
            if self.max_length is not None and len(body) >= self.max_length:
                results.append(tuple(body))
                return
            p, n = float(wp[cov].sum()), float(wn[cov].sum())
            if p <= 0:
                return
            p_star = (wp * cov) @ Xf
            n_star = (wn * cov) @ Xf
            parent = RuleStats(tp=p, fp=n, fn=n_pos - p, tn=n_neg - n, length=len(body))
            children = RuleStats(tp=p_star, fp=n_star, fn=n_pos - p_star, tn=n_neg - n_star, length=len(body) + 1)
            gain = np.where(p_star > 0, heuristic.batch_score(children, parent), -np.inf)
            if body:
                gain[body] = -np.inf
            best = float(gain.max())
            if not self._admits(best, None):
                if body:
                    results.append(tuple(body))
                return
            threshold = self._threshold(best)
            order = np.argsort(-gain, kind="stable")
            chosen = [int(f) for f in order if self._admits(float(gain[f]), threshold)]
            for f in chosen[1:]:                         # the copies
                grow(body + [f], cov & X[:, f])
            grow(body + [chosen[0]], cov & X[:, chosen[0]])

        grow([], np.ones(len(w), dtype=bool))
        return [tuple(sorted(b)) for b in results]
