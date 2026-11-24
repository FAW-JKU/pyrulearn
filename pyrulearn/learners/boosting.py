"""
pyrulearn.learners.boosting
===========================

Rule learners that boost: each round reweights the training examples and
learns one rule with a fitted weight (confidence), and the model is the
sum of the rules' weights -- a `pyrulearn.models.LinearRuleModel`.

`Slipper` (Cohen & Singer, AAAI 1999) reweights with
`pyrulearn.learners.seco.AdaBoostReweighting`, the same component the
weighted covering framework uses (`pyrulearn.learners.seco.
WeightedCovering`), and grows its rules with the SeCo building blocks
(`HillClimbing`, `GrowPruneSplit`). `ENDER` (Dembczyński, Kotłowski &
Słowiński 2008/2010) is gradient boosting of rules for a pluggable
`BoostingLoss` -- `LogisticLoss` (the multinomial log-likelihood, the
default), `ExponentialLoss` (AdaBoost's) or `SigmoidLoss` -- and one of
the paper's minimization techniques (constant-step, gradient descent,
gradient boosting, simultaneous minimization) or MLRules' Newton
criterion; `Boomer` is its BOOMER configuration (Rapp et al. 2020,
single-label). `ORB` (optimal rule boosting, Boley et al., SDM 2021)
boosts rules that maximize the XGBoost-style gain (`XGBGain`, an
`Objective`), found by any rule search -- by default branch and bound,
the optimal rule; `DenseORB` is its dense-matrix specialization.

The losses (`BoostingLoss`, `MarginLoss` and their subclasses) and
`XGBGain` live in `pyrulearn.objectives.losses` and are re-exported
here.

`LRI` (Lightweight Rule Induction, Weiss & Indurkhya, ICML 2000) and its
dense-matrix specialization `DenseLRI` reweight the examples by the
cumulative errors of the rules so far, like boosting, but learn
unweighted DNF rules that vote (see `LRI`). `AdaBoost` (SAMME) boosts
whole models of any rule learner instead of single rules, and its
members vote with weights (see `AdaBoost`). Both used to have modules of
their own, `pyrulearn.learners.lri` and `pyrulearn.learners.adaboost`,
which still re-export them.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..data import DataRepresentation, HorizontalBitmapRepresentation
from ..heuristics import RuleHeuristic, RuleStats, SlipperZ
from ..models import (
    ConceptModel, EnsembleModel, LinearRuleModel, MajorityClass, RuleModel, annotate_default_rule, annotate_rules,
)
from ..objectives import count_open_children, stats_from_handle
from ..objectives.losses import (  # noqa: F401  (moved there; re-exported for compatibility)
    BoostingLoss, ExponentialLoss, LogisticLoss, LogisticMarginLoss, LRIError, MarginLoss, SigmoidLoss,
    SquaredMarginLoss, XGBGain,
)
from ..rule import Rule, WeightedRule
from .base import DEFAULT_MAX_AUTO_CONVERT_CELLS, NativeRuleLearner, produces
from .seco import (
    AdaBoostReweighting, BeamSearch, CoveringState, EmptyRuleAllFeatures, FeatureSubset, GrowPruneSplit,
    HillClimbing, LRIReweighting, Reweighting, RuleSearch, SearchSpaceInit, SingleRuleLearner,
    WeightedCovering, materialize_child, parent_closure,
)


class Slipper(NativeRuleLearner):
    """SLIPPER (Cohen & Singer 1999): confidence-rated boosting of rules.

    For one class `c` against the rest, every round

    1. splits the examples into a growing set and a pruning set
       (`prune_fraction`, stratified; each keeps its boosting weights);
    2. grows a rule on the growing set by greedily adding the condition
       that most increases ``sqrt(W+) - sqrt(W-)`` (`SlipperZ`, over the
       boosting weights of the covered positives and negatives), until no
       condition improves it (`HillClimbing`);
    3. prunes it: keeps the prefix minimizing the boosting loss on the
       pruning set, ``(1 - V+ - V-) + V+ exp(-C) + V- exp(C)``, with ``C``
       the prefix's confidence on the growing set and ``V+``/``V-`` the
       covered fractions of the pruning set's weight;
    4. compares it with the default rule (the empty rule, covering
       everything): whichever has the larger ``(sqrt(W+) - sqrt(W-))**2``
       on all examples is taken;
    5. gives the taken rule the confidence ``C = 1/2 ln((W+ + eps) /
       (W- + eps))`` on all examples, and reweights: every example it
       covers is multiplied by ``exp(-y C)`` (``y = +1`` for `c`, ``-1``
       otherwise), then the weights are rescaled
       (`AdaBoostReweighting`).

    After `n_rounds` rounds, the model predicts `c` where the covering
    rules' confidences sum to more than zero. A rule taken in several
    rounds appears once, with its confidences summed; the default rules
    sum into the intercept (the empty-body rule).

    The result is a `LinearRuleModel`. With two classes, one boosting run
    for `target_class` (default: the less frequent class, as RIPPER
    does); its rules all predict it, and the other class scores 0. With
    more classes, one run per class against the rest, their rules and
    intercepts together in one model: the class with the highest sum
    wins.

    Differences from Cohen & Singer: `n_rounds` is fixed (they chose it
    by internal cross-validation), and numeric attributes come already
    binarized (`pyrulearn.data.io.build_dataspec`) instead of being
    thresholded during growing. Row weights of the data are the initial
    boosting weights.
    """

    def __init__(
        self,
        n_rounds: int = 20,
        prune_fraction: float = 1.0 / 3.0,
        target_class: Any = None,
        max_conditions: Optional[int] = None,
        eps: Optional[float] = None,
        random_state: Optional[int] = 0,
    ):
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be at least 1, got {n_rounds}")
        self.n_rounds = n_rounds
        self.prune_fraction = prune_fraction
        self.target_class = target_class
        self.max_conditions = max_conditions
        self.eps = eps
        self.random_state = random_state

    def _default_model(self, data: Any) -> type:
        return LinearRuleModel

    @produces(LinearRuleModel)
    def _fit_native(self, data: Any, **kw) -> LinearRuleModel:
        if data.y is None:
            raise ValueError("Slipper needs data.y")
        y = np.asarray(data.y)
        labels, counts = np.unique(y, return_counts=True)
        classes = [c.item() if isinstance(c, np.generic) else c for c in labels]
        if len(classes) < 2:
            raise ValueError("Slipper needs at least two classes")
        if self.target_class is not None:
            targets = [self.target_class]
        elif len(classes) == 2:
            targets = [classes[int(np.argmin(counts))]]       # the less frequent class
        else:
            targets = classes
        rules: List[WeightedRule] = []
        runs = []
        for c in targets:
            learned = self._boost(data, c)
            rules += self._merge(learned, c, data.spec)
            runs.append((c, learned))
        model = LinearRuleModel(annotate_rules(rules, data), classes=classes)
        # round t of the model is round t of every class's run
        model.stages = [[]] + [[(c, learned[t][0], learned[t][1]) for c, learned in runs]
                               for t in range(self.n_rounds)]
        return model

    # -- one boosting run -----------------------------------------------------

    def _boost(self, data: Any, target: Any) -> List[Tuple[Tuple[int, ...], float]]:
        """One boosting run for `target`: per round, the body taken (``()``
        for the default rule) and its confidence."""
        base = data.with_weights(None)              # the boosting weights carry the data's weights
        positive = np.asarray(data.y) == target
        n = data.n_samples
        w = np.ones(n) if data.weights is None else data.weights.astype(float).copy()
        w *= n / w.sum()                            # mean 1: eps = 1/2 is Slipper's 1/(2n)
        reweighting = AdaBoostReweighting(eps=0.5 if self.eps is None else self.eps)
        state = CoveringState(w, positive)
        search = HillClimbing(max_conditions=self.max_conditions)
        heuristic = SlipperZ()
        everything = np.ones(n, dtype=bool)

        learned: List[Tuple[Tuple[int, ...], float]] = []
        for t in range(self.n_rounds):
            seed = None if self.random_state is None else self.random_state + t
            grow, prune = GrowPruneSplit(self.prune_fraction, seed).prepare(base, target, state.scope)
            initial = EmptyRuleAllFeatures().initial_candidates(base, target, grow)
            rule = search.search(base, target, heuristic, initial, example_mask=grow)
            if rule is not None and rule.length() > 0 and prune is not None:
                rule = self._prune(rule, base, positive, grow, prune, reweighting)

            covered = everything
            if rule is not None and rule.length() > 0:
                rule_cov = rule.covers_data_packed(base)
                if _objective(state.scope, rule_cov, positive) > _objective(state.scope, everything, positive):
                    covered = rule_cov
                else:
                    rule = None
            else:
                rule = None
            confidence = reweighting.rule_weight(state, covered)
            state.record(covered)
            state.scope = reweighting.weights(state, covered)
            body = () if rule is None else tuple(l.feature for l in rule.conditions)
            learned.append((body, confidence))

        return learned

    def _prune(self, rule: Rule, base: Any, positive: np.ndarray, grow: np.ndarray,
               prune: np.ndarray, reweighting: AdaBoostReweighting) -> Rule:
        """The prefix of `rule` (at least one condition) with the lowest
        boosting loss on the pruning weights `prune`; ties go to the shorter."""
        total = float(prune.sum())
        if total <= 0:
            return rule
        best, best_loss = rule, math.inf
        for k in range(rule.length(), 0, -1):
            prefix = rule if k == rule.length() else Rule(
                rule.conditions[:k], target=rule.target, dataspec=rule.dataspec, n_features=rule.n_features)
            cov = prefix.covers_data_packed(base)
            c = reweighting.confidence(grow, cov, positive)
            v_pos = float(prune[cov & positive].sum()) / total
            v_neg = float(prune[cov & ~positive].sum()) / total
            loss = (1.0 - v_pos - v_neg) + v_pos * math.exp(-c) + v_neg * math.exp(c)
            if loss <= best_loss:
                best, best_loss = prefix, loss
        return best

    @staticmethod
    def _merge(learned: List[Tuple[Tuple[int, ...], float]], target: Any, spec: Any) -> List[WeightedRule]:
        """One rule per distinct body, its confidences summed (in first-taken
        order); the default rules' sum is the intercept, listed first."""
        total: Dict[Tuple[int, ...], float] = {}
        for body, c in learned:
            total[body] = total.get(body, 0.0) + c
        intercept = total.pop((), None)
        out = [] if intercept is None else [WeightedRule([], target=target, dataspec=spec, weight=intercept)]
        out += [WeightedRule(list(body), target=target, dataspec=spec, weight=c)
                for body, c in total.items() if c != 0]
        return out


def _objective(w: np.ndarray, covered: np.ndarray, positive: np.ndarray) -> float:
    """``(sqrt(W+) - sqrt(W-))**2`` of the rows `covered` under weights `w`
    -- the larger, the lower the boosting loss the rule can reach."""
    return (math.sqrt(float(w[covered & positive].sum())) - math.sqrt(float(w[covered & ~positive].sum()))) ** 2


# ================================================================ ENDER ===


class ImpurityCriterion(ABC):
    """How `ENDER` scores a candidate rule while growing it -- its
    counterpart of a rule-evaluation heuristic, called the impurity
    ``L(rule)`` in the paper (lower is better; a rule is kept only if
    negative). Every criterion is a function of sums over the rule's
    covered rows, so it has two parts:

    - `terms` -- per-row, per-class quantities (``n x K`` arrays), zero
      outside the subsample `in_sample`, computed once per boosting round
      from the scores ``F``, the one-hot classes ``Y``, the row weights
      ``d`` and the loss's derivatives ``G``/``H``;
    - `impurity` -- the criterion from those terms summed over the
      covered rows (one ``K x n_conditions`` array per term), for every
      class and candidate condition at once.

    `check` refuses a loss the criterion doesn't fit."""

    def check(self, loss: BoostingLoss) -> None:
        pass

    @abstractmethod
    def terms(self, F, Y, d, G, H, loss: BoostingLoss, in_sample: np.ndarray) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError

    @abstractmethod
    def impurity(self, sums: Tuple[np.ndarray, ...], l2_regularization: float) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


class ConstantStep(ImpurityCriterion):
    """CS: the change of the loss if the covered rows' score for the
    class rose by `beta`. Works for any loss; `beta` trades off coverage
    against purity (larger: smaller, purer rules)."""

    def __init__(self, beta: float = 0.2):
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = beta

    def terms(self, F, Y, d, G, H, loss, in_sample):
        base = loss.values(F, Y, d)
        D = np.empty_like(F)
        for k in range(F.shape[1]):
            Fk = F.copy()
            Fk[:, k] += self.beta
            D[:, k] = loss.values(Fk, Y, d) - base
        return (D * in_sample[:, None],)

    def impurity(self, sums, l2_regularization):
        return sums[0]


class Gradient(ImpurityCriterion):
    """GD: the summed first derivative ``g`` -- the most general rules
    (`ConstantStep` as ``beta -> 0``)."""

    def terms(self, F, Y, d, G, H, loss, in_sample):
        return (G * in_sample[:, None],)

    def impurity(self, sums, l2_regularization):
        return sums[0]


class GradientBoosting(ImpurityCriterion):
    """GB: ``g / sqrt(covered weight)``."""

    def terms(self, F, Y, d, G, H, loss, in_sample):
        s = in_sample[:, None]
        return (G * s, (d[:, None] * s) * np.ones_like(G))

    def impurity(self, sums, l2_regularization):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sums[1] > 0, sums[0] / np.sqrt(sums[1]), 0.0)


class Simultaneous(ImpurityCriterion):
    """SM, for `ExponentialLoss` only: the loss with the rule's exact
    weight, ``-sqrt(W+) + sqrt(W-)`` (``W+``/``W-`` the weights of the
    covered examples of the voted/the other class)."""

    def check(self, loss):
        if not isinstance(loss, ExponentialLoss):
            raise ValueError("Simultaneous (method='simultaneous') needs the exponential loss")

    def terms(self, F, Y, d, G, H, loss, in_sample):
        w = H * in_sample[:, None]            # both columns carry the exponential weight
        return (w * Y, w * (1.0 - Y))

    def impurity(self, sums, l2_regularization):
        with np.errstate(invalid="ignore"):
            return -np.sqrt(sums[0]) + np.sqrt(sums[1])


class Newton(ImpurityCriterion):
    """MLRules' criterion ``g / sqrt(h + lambda)`` (``lambda``: the
    learner's `l2_regularization`), for convex losses."""

    def check(self, loss):
        if not loss.convex:
            raise ValueError(f"Newton (method='newton') needs a convex loss, not {loss!r}")

    def terms(self, F, Y, d, G, H, loss, in_sample):
        s = in_sample[:, None]
        return (G * s, H * s)

    def impurity(self, sums, l2_regularization):
        denom = sums[1] + l2_regularization
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, sums[0] / np.sqrt(denom), 0.0)


class _ENDERBase(NativeRuleLearner):
    """What `ENDER` and `DenseENDER` share: the boosting loop, the losses,
    the impurity criteria and the default rule. They differ only in how
    one rule is grown (`_grower`)."""

    def __init__(
        self,
        n_rules: int = 500,
        shrinkage: float = 0.1,
        subsample: float = 0.25,
        loss: Optional[BoostingLoss] = None,
        method: Optional[ImpurityCriterion] = None,
        l2_regularization: float = 0.0,
        early_stopping: bool = False,
        max_length: Optional[int] = None,
        random_state: Optional[int] = 0,
    ):
        if n_rules < 1:
            raise ValueError(f"n_rules must be at least 1, got {n_rules}")
        if not 0.0 < shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in (0, 1], got {shrinkage}")
        if not 0.0 < subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")
        loss = LogisticLoss() if loss is None else loss
        method = ConstantStep() if method is None else method
        if not isinstance(loss, BoostingLoss):
            raise ValueError(f"loss must be a BoostingLoss, got {loss!r}")
        if not isinstance(method, ImpurityCriterion):
            raise ValueError(f"method must be an ImpurityCriterion, got {method!r}")
        if l2_regularization < 0:
            raise ValueError(f"l2_regularization must be non-negative, got {l2_regularization}")
        self.n_rules = n_rules
        self.shrinkage = shrinkage
        self.subsample = subsample
        self.loss = loss
        self.method = method
        self.l2_regularization = l2_regularization
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.random_state = random_state
        self._criterion().check(loss)

    def _criterion(self) -> ImpurityCriterion:
        return self.method

    def _default_model(self, data: Any) -> type:
        return LinearRuleModel

    @produces(LinearRuleModel)
    def _fit_native(self, data: Any, **kw) -> LinearRuleModel:
        # a no-op for ENDER; DenseENDER converts to its dense matrix here
        data = self.ensure_representation(
            data, getattr(self, "max_auto_convert_cells", DEFAULT_MAX_AUTO_CONVERT_CELLS),
            purpose=f"{type(self).__name__}'s dense-matrix scoring",
        )
        if data.y is None:
            raise ValueError("ENDER needs data.y")
        y = np.asarray(data.y)
        labels, y_idx = np.unique(y, return_inverse=True)
        classes = [c.item() if isinstance(c, np.generic) else c for c in labels]
        K = len(classes)
        if K < 2:
            raise ValueError("ENDER needs at least two classes")
        if K > 2 and not self.loss.multiclass:
            raise ValueError(f"{self.loss!r} handles two classes only")
        n = len(y)
        Y = np.zeros((n, K))
        Y[np.arange(n), y_idx] = 1.0
        grow = self._grower(data)
        d = np.ones(n) if data.weights is None else data.weights.astype(float)
        rng = np.random.default_rng(self.random_state)

        F = np.zeros((n, K))
        k0, alpha0 = self._default_rule(F, Y, d)
        F[:, k0] += alpha0                                  # the default rule, not shrunk
        learned: List[Tuple[Tuple[int, ...], int, float]] = [((), k0, alpha0)]
        # per iteration, what it added (nothing if no rule was found or kept)
        stages: List[List[Tuple[Any, Tuple[int, ...], float]]] = [[(classes[k0], (), alpha0)]]

        size = max(1, int(round(self.subsample * n)))
        verdicts: List[bool] = []
        for _ in range(self.n_rules):
            G, H = self.loss.derivatives(F, Y, d)
            in_sample = np.zeros(n, dtype=bool)
            in_sample[rng.choice(n, size=size, replace=False)] = True
            found = grow(self._impurity_terms(F, Y, d, G, H, in_sample))
            stages.append([])
            if found is None:
                continue
            body, k, cov = found
            alpha = self._response(float(G[cov, k].sum()), float(H[cov, k].sum()))
            if not alpha > 0:
                continue
            F[cov, k] += self.shrinkage * alpha
            learned.append((body, k, self.shrinkage * alpha))
            stages[-1].append((classes[k], tuple(sorted(body)), self.shrinkage * alpha))
            if self.early_stopping:
                verdicts.append(self._acceptable(cov & ~in_sample, y_idx, k, d, K))
                if len(verdicts) >= 10 and sum(not v for v in verdicts[-10:]) >= 8:
                    break

        total: Dict[Tuple[Tuple[int, ...], int], float] = {}
        for body, k, a in learned:
            key = (tuple(sorted(body)), k)
            total[key] = total.get(key, 0.0) + a
        rules = [WeightedRule(list(body), target=classes[k], dataspec=data.spec, weight=a)
                 for (body, k), a in total.items() if a != 0]
        model = LinearRuleModel(annotate_rules(rules, data), classes=classes)
        model.stages = stages
        return model

    def _response(self, g: float, h: float) -> float:
        """The loss's rule weight; Newton steps get the L2 penalty."""
        if self.loss.iterative_response:
            h = h + self.l2_regularization
        return self.loss.response(g, h)

    # -- the default rule ------------------------------------------------------

    def _default_rule(self, F: np.ndarray, Y: np.ndarray, d: np.ndarray) -> Tuple[int, float]:
        """The class and weight of the rule covering everything that
        minimize the loss (the paper's Eq. 5): the weight by iterated
        Newton steps for a convex loss with Newton responses, else by the
        loss's own response (exact for the exponential loss, `beta` for
        the sigmoid)."""
        best: Tuple[float, int, float] = (math.inf, 0, 0.0)
        for k in range(F.shape[1]):
            alpha = 0.0
            for _ in range(25 if self.loss.iterative_response else 1):
                Fk = F.copy()
                Fk[:, k] += alpha
                G, H = self.loss.derivatives(Fk, Y, d)
                step = self._response(float(G[:, k].sum()), float(H[:, k].sum()))
                alpha = alpha + step if self.loss.iterative_response else step
                if abs(step) < 1e-10:
                    break
            if alpha <= 0:
                continue
            Fk = F.copy()
            Fk[:, k] += alpha
            value = self.loss.value(Fk, Y, d)
            if value < best[0]:
                best = (value, k, alpha)
        return best[1], best[2]

    # -- growing a rule ----------------------------------------------------------

    def _impurity_terms(self, F, Y, d, G, H, in_sample):
        """Per-row quantities whose sums over the covered rows give the
        criterion's impurity (see `_impurity`), restricted to the subsample."""
        return self._criterion().terms(F, Y, d, G, H, self.loss, in_sample)

    def _impurity(self, sums: Tuple[np.ndarray, ...]) -> np.ndarray:
        return self._criterion().impurity(sums, self.l2_regularization)

    def _grower(self, data: Any):
        """The step growing one rule: ``grow(terms) -> (body, class,
        covered rows)`` or None -- what `ENDER` and `DenseENDER` differ in."""
        raise NotImplementedError

    @staticmethod
    def _acceptable(holdout_cov: np.ndarray, y_idx: np.ndarray, k: int, d: np.ndarray, K: int) -> bool:
        w = d[holdout_cov]
        if w.sum() <= 0:
            return False
        error = float(w[y_idx[holdout_cov] != k].sum() / w.sum())
        return error < 1.0 - 1.0 / K


class ENDER(_ENDERBase):
    """ENDER: boosting of decision rules by forward stagewise minimization
    of a loss (Dembczyński, Kotłowski & Słowiński, DMKD 2010; its MLRules
    instance, ICML 2008), with a pluggable `BoostingLoss`.

    The model keeps a score per class; each rule votes for one class with
    a positive weight, and the class with the highest total wins. It
    starts from a default rule (covering everything) for one class, with
    the weight minimizing the loss. Then, for `n_rules` rounds:

    1. Draw a subsample (`subsample` of the rows, without replacement).
    2. Grow a rule on it: starting from the empty rule (impurity 0), add
       the condition, and choose the class, that minimize the impurity
       ``L(rule)`` of `method` -- an `ImpurityCriterion`, ENDER's
       counterpart of a rule-evaluation heuristic -- until no condition
       lowers it; the rule is kept only if
       its impurity is negative. With ``g``/``h`` the loss's first/second
       derivatives for a vote for the class, summed over the covered
       rows:

       - `ConstantStep(beta=0.2)` (CS, the default) -- the change of the
         loss if the covered rows' score for the class rose by `beta`:
         works for any loss, and `beta` trades off coverage against
         purity (larger: smaller, purer rules);
       - `Gradient()` (GD) -- ``g``: the most general rules (``beta -> 0``
         of constant-step);
       - `GradientBoosting()` (GB) -- ``g / sqrt(covered weight)``;
       - `Simultaneous()` (SM, `ExponentialLoss` only) -- the loss with
         the rule's exact weight: ``-sqrt(W+) + sqrt(W-)``, ``W+``/``W-``
         the weights of the covered examples of the voted/other class;
       - `Newton()` -- ``g / sqrt(h)``, MLRules' criterion (convex
         losses).

       The loss is a component too: `loss` takes a `BoostingLoss`
       (`LogisticLoss()`, the default, `ExponentialLoss()`,
       `SigmoidLoss(beta=0.2)`).
    3. Give it the loss's weight (`BoostingLoss.response`: a Newton step
       for `LogisticLoss`, the exact minimizer for `ExponentialLoss`, the
       constant `beta` for `SigmoidLoss`) computed on *all* rows -- which also
       regularizes it -- shrink it by `shrinkage` (``nu``), and add it to
       the scores.

    Defaults are the paper's constant-step logit setting (CS-Log: ``beta
    = 0.2``, ``nu = 0.1``, subsample 0.25, 500 rules), among its best and
    usable for any number of classes; its best-ranked, CS-Exp, is
    ``ENDER(loss=ExponentialLoss())`` with the same settings. MLRules is
    ``ENDER(method=Newton(), subsample=0.5)``. `LogisticLoss` handles any
    number of classes; `ExponentialLoss` and `SigmoidLoss` two (as in the
    paper, which also covers regression -- not here).

    `l2_regularization` (``lambda``, default 0) adds an L2 penalty on the
    rule weights to the Newton steps -- ``-sum g / (sum h + lambda)`` for
    the Newton-step weights (`LogisticLoss`, and the default rule) and
    ``sum g / sqrt(sum h + lambda)`` for `method=Newton()` -- as in
    BOOMER (Rapp et al. 2020), whose single-output case is then
    ``ENDER(method=Newton(), l2_regularization=...)``.

    `early_stopping` is MLRules': the rows left out of each subsample are
    a holdout set; a rule is acceptable if its error on the holdout rows
    it covers is below that of guessing among the classes (``1 - 1/K``),
    and growth stops once 8 of the last 10 rules weren't.

    The result is a `LinearRuleModel`; a rule found in several rounds
    appears once, its weights summed, and the default rule is the
    intercept of its class. Numeric attributes come already binarized
    (`pyrulearn.data.io.build_dataspec`) instead of being thresholded
    during the search; the data's row weights weight the loss.

    **Components and data representation.** The loss (`loss=`, a
    `BoostingLoss`) and the impurity criterion (`method=`, an
    `ImpurityCriterion`) are exchangeable parts. A rule is grown through
    the representation's own primitives, so `ENDER` runs on every data
    representation: the search walks a cover handle
    (`initial_cover`/`refine_cover`), sums the impurity terms of every
    open condition with one `batch_cover_sums` call per step, and builds
    only the condition it adds (`pyrulearn.learners.seco.materialize_child`),
    propagating constraints so that what a condition implies is never
    scored again. `DenseENDER` is the same algorithm specialized for speed
    on a dense matrix: same parameters, same models, 1-8x faster on
    `HorizontalBitmapRepresentation` (most with Newton steps on all rows; see
    `ROADMAP.md`). A condition that drops no row with a nonzero impurity
    term changes nothing; both count those rows exactly and never take
    such a condition for an improvement, which rounding could otherwise
    suggest.
    """

    def _grower(self, data: Any):
        dataspec = data.spec
        all_features = frozenset(range(dataspec.n_features))

        def grow(terms):
            n_terms, K = len(terms), terms[0].shape[1]
            contributing = (np.concatenate(terms, axis=1) != 0).any(axis=1)
            # n x (terms * K + 1): the last column counts covered contributing
            # rows -- an exact integer, unlike the other sums
            values = np.column_stack([np.concatenate(terms, axis=1), contributing.astype(float)])
            current_count = float(contributing.sum())
            rule = Rule([], dataspec=dataspec)
            handle = data.initial_cover(None)
            closure = parent_closure(dataspec, rule)
            mask = all_features
            body: List[int] = []
            best_k, current = -1, 0.0
            while mask and (self.max_length is None or len(body) < self.max_length):
                feats = sorted(mask)
                sums = data.batch_cover_sums(handle, values, feats)
                counts = sums[-1]
                crit = self._impurity(tuple(sums[t * K:(t + 1) * K] for t in range(n_terms)))
                if body:
                    # a condition dropping no contributing row changes nothing:
                    # exactly the current impurity, though summed over other rows
                    # it can round lower -- ENDER's dense sums can't, they always
                    # run over the same rows. (Not for the first condition, which
                    # ENDER compares with 0, not with the empty rule's impurity.)
                    crit[:, counts == current_count] = current
                while True:   # best (class, condition); same tie-break as ENDER: lowest class, then feature
                    k, j = np.unravel_index(int(np.argmin(crit)), crit.shape)
                    value = float(crit[k, j])
                    if not value < current:
                        built = None
                        break
                    built = materialize_child(data, dataspec, rule, closure, mask, feats[j], handle)
                    if built is not None:
                        break
                    crit[:, j] = np.inf                          # contradicts the rule
                if built is None:
                    break
                rule, mask, handle, closure = built
                current, best_k, current_count = value, int(k), counts[j]
                body.append(feats[j])
            if body and current < 0:
                return tuple(body), best_k, data.cover_rows(handle)
            return None
        return grow


class DenseENDER(_ENDERBase):
    """`ENDER`, specialized for speed: same algorithm, same parameters,
    same models (see `ENDER` for both) -- at the price of a dense float
    copy of the data.

    **Data representation.** `NATIVE_REPRESENTATIONS = (HorizontalBitmapRepresentation,)`
    -- `_fit_native` converts anything else via
    `NativeRuleLearner.ensure_representation` (`max_auto_convert_cells=`).
    Each step scores every class and condition at once, ``(t * c).T @
    Xf`` over a float copy of the whole matrix, with no constraint
    propagation; `ENDER` instead copies only the covered rows of the open
    conditions at every step, which costs it most where all rows count
    (Newton steps without subsampling: up to 8x on `spambase`).
    """

    #: see the "Data representation" paragraph above
    NATIVE_REPRESENTATIONS = (HorizontalBitmapRepresentation,)

    def __init__(
        self,
        n_rules: int = 500,
        shrinkage: float = 0.1,
        subsample: float = 0.25,
        loss: Optional[BoostingLoss] = None,
        method: Optional[ImpurityCriterion] = None,
        l2_regularization: float = 0.0,
        early_stopping: bool = False,
        max_length: Optional[int] = None,
        random_state: Optional[int] = 0,
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
    ):
        super().__init__(n_rules, shrinkage, subsample, loss, method, l2_regularization,
                         early_stopping, max_length, random_state)
        self.max_auto_convert_cells = max_auto_convert_cells

    def _grower(self, data: Any):
        """The step growing one rule: ``grow(terms) -> (body, class,
        covered rows)`` or None. Here, `_grow` on a dense float copy of
        the matrix."""
        X = np.asarray(data.X, dtype=bool)
        Xf = X.astype(float)

        def grow(terms):
            found = self._grow(X, Xf, terms)
            if found is None:
                return None
            body, k = found
            return body, k, np.all(X[:, list(body)], axis=1)
        return grow

    def _grow(self, X: np.ndarray, Xf: np.ndarray,
              terms: Tuple[np.ndarray, ...]) -> Optional[Tuple[Tuple[int, ...], int]]:
        """The rule (body, class) minimizing the impurity, grown greedily
        from the empty rule; `None` if no condition makes it negative."""
        cov = np.ones(X.shape[0], dtype=bool)
        body: List[int] = []
        best_k, current = -1, 0.0
        contributing = np.any([(t != 0).any(axis=1) for t in terms], axis=0).astype(float)
        current_count = float(contributing.sum())
        while self.max_length is None or len(body) < self.max_length:
            c = cov[:, None]
            crit = self._impurity(tuple((t * c).T @ Xf for t in terms))     # K x n_features
            counts = (contributing * cov) @ Xf                               # exact integers
            if body:
                # a condition dropping no contributing row changes nothing -- exactly
                # the current impurity, though the matrix product can round its
                # column differently (seen: a threshold implied by one already in
                # the rule winning by 4e-15). The first condition is compared with
                # 0, not with the empty rule's impurity, so it's left alone.
                crit[:, counts == current_count] = current
                crit[:, body] = np.inf
            k, f = np.unravel_index(int(np.argmin(crit)), crit.shape)
            value = float(crit[k, f])
            if not value < current:
                break
            current, best_k, current_count = value, int(k), counts[f]
            body.append(int(f))
            cov = cov & X[:, f]
        return (tuple(body), best_k) if body and current < 0 else None


class Boomer(ENDER):
    """BOOMER (Rapp, Loza Mencía, Fürnkranz, Nguyen & Hüllermeier, ECML PKDD
    2020) for single-label classification: gradient-boosted rules with the
    logistic loss and L2-regularized Newton steps -- `ENDER` with
    ``method=Newton()``, ``loss=LogisticLoss()`` and BOOMER's defaults: up to
    1000 rules, shrinkage 0.3, L2 weight 1.0, no subsampling. Each rule
    maximizes ``(sum g)**2 / (sum h + lambda)`` (as ``sum g / sqrt(sum h
    + lambda)``) and gets the weight ``-sum g / (sum h + lambda)``, shrunk.

    This is BOOMER's single-output case. BOOMER itself learns rules for
    several outputs (labels) at once -- multi-label classification, with
    rule heads predicting one or several labels and decomposable or
    non-decomposable losses -- which needs multi-label data and models
    that pyrulearn doesn't have yet (on the to-do list, see the README).
    With more than two classes, `Boomer` uses the multinomial logistic
    loss (each rule voting for one class), where BOOMER would treat the
    classes as labels. The original is interfaced as
    `pyrulearn.interfaces.boomer.MLRLBoomer` (binary classification),
    whose rule induction differs in details (e.g. its feature sampling),
    so the two don't produce identical models. `DenseBoomer` is the same
    on `DenseENDER`: same models, faster on a `HorizontalBitmapRepresentation`.
    """

    def __init__(
        self,
        n_rules: int = 1000,
        shrinkage: float = 0.3,
        l2_regularization: float = 1.0,
        subsample: float = 1.0,
        early_stopping: bool = False,
        max_length: Optional[int] = None,
        random_state: Optional[int] = 0,
    ):
        super().__init__(
            n_rules=n_rules, shrinkage=shrinkage, subsample=subsample, loss=LogisticLoss(), method=Newton(),
            l2_regularization=l2_regularization, early_stopping=early_stopping, max_length=max_length,
            random_state=random_state,
        )


class DenseBoomer(DenseENDER):
    """`Boomer` on `DenseENDER`: same parameters, same models, faster on a
    `HorizontalBitmapRepresentation` (Newton steps on all rows are where
    `ENDER`'s modular search costs most) -- see `Boomer` and `DenseENDER`."""

    def __init__(
        self,
        n_rules: int = 1000,
        shrinkage: float = 0.3,
        l2_regularization: float = 1.0,
        subsample: float = 1.0,
        early_stopping: bool = False,
        max_length: Optional[int] = None,
        random_state: Optional[int] = 0,
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
    ):
        super().__init__(
            n_rules=n_rules, shrinkage=shrinkage, subsample=subsample, loss=LogisticLoss(), method=Newton(),
            l2_regularization=l2_regularization, early_stopping=early_stopping, max_length=max_length,
            random_state=random_state, max_auto_convert_cells=max_auto_convert_cells,
        )


# ================================================== optimal rule boosting ===


class _ORBBase(NativeRuleLearner):
    """The boosting loop shared by `ORB` and `DenseORB`: everything but how
    a round's rule is found (`_prepare`, `_best_query`, `_covers`)."""

    def __init__(self, n_rules: int, loss: Optional[MarginLoss], reg: float, offset: bool):
        if n_rules < 1:
            raise ValueError(f"n_rules must be at least 1, got {n_rules}")
        loss = LogisticMarginLoss() if loss is None else loss
        if not isinstance(loss, MarginLoss):
            raise ValueError(f"loss must be a MarginLoss, got {loss!r}")
        if reg < 0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        self.n_rules = n_rules
        self.loss = loss
        self.reg = reg
        self.offset = offset

    def _default_model(self, data: Any) -> type:
        return LinearRuleModel

    def _derivatives(self, y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.loss.derivatives(y, s)

    @produces(LinearRuleModel)
    def _fit_native(self, data: Any, **kw) -> LinearRuleModel:
        if data.y is None:
            raise ValueError(f"{type(self).__name__} needs data.y")
        labels = np.unique(np.asarray(data.y))
        if len(labels) != 2:
            raise ValueError(f"{type(self).__name__} needs a binary target, got {len(labels)} classes")
        classes = [c.item() if isinstance(c, np.generic) else c for c in labels]
        y = np.where(np.asarray(data.y) == labels[1], 1.0, -1.0)
        ctx = self._prepare(data)
        d = np.ones(len(y)) if data.weights is None else data.weights.astype(float)
        s = np.zeros(len(y))
        learned: Dict[Tuple[int, ...], float] = {}
        for m in range(self.n_rules):
            g, h = self._derivatives(y, s)
            g, h = g * d, h * d
            if self.offset and m == 0:
                body: Tuple[int, ...] = ()
            else:
                body = self._best_query(ctx, g, h)
                if body is None:
                    break
            cov = self._covers(ctx, body) if body else np.ones(len(y), dtype=bool)
            w = -float(g[cov].sum()) / (self.reg + float(h[cov].sum()))
            s = s + w * cov
            learned[body] = learned.get(body, 0.0) + w
        rules = []
        for body, w in learned.items():
            if w == 0:
                continue
            target = classes[1] if w > 0 else classes[0]
            rules.append(WeightedRule(list(body), target=target, dataspec=data.spec, weight=abs(w)))
        return LinearRuleModel(annotate_rules(rules, data), classes=classes)


class ORB(_ORBBase):
    """Optimal rule boosting (Boley, Teshuva, Le Bodic & Webb, "Better
    short than greedy: Interpretable models through optimal rule
    boosting", SDM 2021; the `realkd` package), built from the rule
    search components: each round, `search` maximizes the `XGBGain`
    objective of the current scores.

    Binary classification with labels ``y = +1`` for the positive class
    (the second in sorted order) and ``-1`` for the other. The model is a
    sum of rules ``w * q(x)``; each round adds the rule whose query ``q``
    maximizes the XGBoost-style gain

        obj(q) = (sum_{i in q} g_i)**2 / (reg + sum_{i in q} h_i)

    (``g``/``h``: first/second derivatives of the loss at the current
    scores, times the row weights), with the weight
    ``w = -sum g / (reg + sum h)``. The objective is symmetric in the
    sign of ``sum g``; the weight's sign decides the class the rule votes
    for (a rule with a negative weight votes for the negative class with
    its absolute weight).

    `search` is the rule search, as in every `SeCo` learner (default
    `BeamSearch(beam_width=10)`): a `BranchAndBoundSearch` finds the
    optimal rule, as realkd's exhaustive search does, with `XGBGain`'s
    bounds, the paper's prefix/suffix bound included; a `HillClimbing`
    adds the best condition while the gain rises (realkd's greedy); a
    `BeamSearch` lies in between -- the default width 10 was the most
    accurate of greedy, beam widths 3/5/10 and branch and bound capped at
    3 conditions on 8 binary datasets (5-fold, 10 rules; see `ROADMAP.md`),
    while unbounded branch and bound often takes minutes per rule set.
    The search's `max_conditions` caps the
    rule length. The search runs on the data's own representation. The
    rule found is simplified by dropping conditions that don't change
    what it covers.

    `loss` is a `MarginLoss`: `LogisticMarginLoss()` (default) or
    `SquaredMarginLoss()`, realkd's two losses with its derivatives; `reg` is the L2 regularization ``lambda``; `offset=True`
    makes the first rule the empty one (an intercept). The result is a
    `LinearRuleModel`; a query found again adds to its weight. `DenseORB`
    is the same learner specialized for speed on a dense matrix, with its
    own two searches (exhaustive and greedy); it reproduces realkd
    exactly. With `BranchAndBoundSearch`/`HillClimbing` ORB finds rules of
    the same gain as DenseORB's exhaustive/greedy search; where several
    rules tie, they may pick different ones.
    """

    def __init__(
        self,
        n_rules: int = 10,
        loss: Optional[MarginLoss] = None,
        reg: float = 1.0,
        search: Optional[RuleSearch] = None,
        offset: bool = False,
    ):
        super().__init__(n_rules, loss, reg, offset)
        if search is not None and not isinstance(search, RuleSearch):
            raise ValueError(f"search must be a RuleSearch, got {search!r}")
        self.search = search if search is not None else BeamSearch(beam_width=10)

    def _prepare(self, data: Any):
        return data, self.search, EmptyRuleAllFeatures().initial_candidates(data, None)

    def _best_query(self, ctx, g: np.ndarray, h: np.ndarray) -> Optional[Tuple[int, ...]]:
        data, search, initial = ctx
        rule = search.search(data, None, XGBGain(g, h, reg=self.reg), initial)
        if rule is None or rule.length() == 0:
            return None
        return self._simplify(data, tuple(lit.feature for lit in rule.conditions))

    def _covers(self, ctx, body: Tuple[int, ...]) -> np.ndarray:
        data = ctx[0]
        return np.asarray(data.coverage(Rule(list(body), dataspec=data.spec)), dtype=bool)

    def _simplify(self, data: Any, body: Tuple[int, ...]) -> Tuple[int, ...]:
        """Drop conditions whose removal doesn't change the covered rows."""
        ctx = (data,)
        cov = self._covers(ctx, body)
        kept = list(body)
        for f in list(body):
            rest = [x for x in kept if x != f]
            if rest and np.array_equal(self._covers(ctx, tuple(rest)), cov):
                kept = rest
        return tuple(sorted(kept))


class DenseORB(_ORBBase):
    """`ORB`, specialized for speed on a dense matrix: the same boosting
    loop with two hand-written searches on ``data.X`` (any representation
    provides it). It reproduces the `realkd` package exactly -- the same
    rules and weights with either search, with or without an intercept
    (``tests/test_realkd_import.py``). See `ORB` for the algorithm and
    the parameters.

    `search="exhaustive"` (default) finds the optimal query by
    best-first branch-and-bound over conjunctions of the data's features
    (and their negation features), with the paper's bound: among the
    subsets of a query's covered examples, the objective is maximal at a
    prefix or a suffix of them sorted by ``g/h``. Refinements that don't
    change the covered set are skipped, and the query found is simplified
    by dropping conditions that don't change what it covers.
    `search="greedy"` adds the best condition until the objective stops
    improving -- one matrix multiply per step. `max_length` caps the
    query length.
    """

    def __init__(
        self,
        n_rules: int = 10,
        loss: Optional[MarginLoss] = None,
        reg: float = 1.0,
        search: str = "exhaustive",
        max_length: Optional[int] = None,
        offset: bool = False,
    ):
        super().__init__(n_rules, loss, reg, offset)
        if search not in ("exhaustive", "greedy"):
            raise ValueError(f"search must be 'exhaustive' or 'greedy', got {search!r}")
        self.search = search
        self.max_length = max_length

    def _prepare(self, data: Any) -> np.ndarray:
        return np.asarray(data.X, dtype=bool)

    def _covers(self, X: np.ndarray, body: Tuple[int, ...]) -> np.ndarray:
        return np.all(X[:, list(body)], axis=1)

    # -- the search -----------------------------------------------------------

    def _objective(self, g: np.ndarray, h: np.ndarray, cov: np.ndarray) -> float:
        return float(g[cov].sum()) ** 2 / (self.reg + float(h[cov].sum()))

    def _best_query(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> Optional[Tuple[int, ...]]:
        body = self._greedy(X, g, h) if self.search == "greedy" else self._branch_and_bound(X, g, h)
        if not body:
            return None
        return self._simplify(X, body)

    def _greedy(self, X, g, h) -> Tuple[int, ...]:
        cov = np.ones(X.shape[0], dtype=bool)
        body: List[int] = []
        current = -math.inf
        Xf = X.astype(float)
        while self.max_length is None or len(body) < self.max_length:
            gs = (g * cov) @ Xf
            hs = (h * cov) @ Xf
            obj = gs ** 2 / (self.reg + hs)
            obj[(cov[:, None] & X).sum(axis=0) == 0] = -math.inf     # empty extensions
            if body:
                obj[body] = -math.inf
            f = int(np.argmax(obj))
            if not obj[f] > current:
                break
            current = float(obj[f])
            body.append(f)
            cov = cov & X[:, f]
        return tuple(body)

    def _branch_and_bound(self, X, g, h) -> Tuple[int, ...]:
        import heapq

        n, n_features = X.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(h > 0, g / h, np.sign(g) * np.inf)
        order = np.argsort(-ratio, kind="stable")                  # examples by g/h, descending
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)

        def bound(cov: np.ndarray) -> float:
            idx = np.flatnonzero(cov)
            if idx.size == 0:
                return -math.inf
            idx = idx[np.argsort(rank[idx])]
            gq, hq = g[idx], h[idx]
            pre = np.cumsum(gq) ** 2 / (np.cumsum(hq) + self.reg)
            suf = np.cumsum(gq[::-1]) ** 2 / (np.cumsum(hq[::-1]) + self.reg)
            return float(max(pre.max(), suf.max()))

        best_body: Tuple[int, ...] = ()
        best_value = -math.inf
        root = np.ones(n, dtype=bool)
        heap = [(-bound(root), 0, (), root)]
        counter = 1
        while heap:
            neg_bound, _, body, cov = heapq.heappop(heap)
            if -neg_bound <= best_value:
                break                                            # best-bound-first: nothing left can win
            if self.max_length is not None and len(body) >= self.max_length:
                continue
            start = body[-1] + 1 if body else 0
            n_cov = int(cov.sum())
            for f in range(start, n_features):
                child = cov & X[:, f]
                n_child = int(child.sum())
                if n_child == 0 or n_child == n_cov:
                    continue                                     # empty, or covers the same rows
                value = self._objective(g, h, child)
                child_body = body + (f,)
                if value > best_value:
                    best_value, best_body = value, child_body
                b = bound(child)
                if b > best_value:
                    heapq.heappush(heap, (-b, counter, child_body, child))
                    counter += 1
        return best_body

    @staticmethod
    def _simplify(X: np.ndarray, body: Tuple[int, ...]) -> Tuple[int, ...]:
        """Drop conditions whose removal doesn't change the covered rows."""
        cov = np.all(X[:, list(body)], axis=1)
        kept = list(body)
        for f in list(body):
            rest = [x for x in kept if x != f]
            if rest and np.array_equal(np.all(X[:, rest], axis=1), cov):
                kept = rest
        return tuple(sorted(kept))


# ---------------------------------------------------------------- LRI ----


class DenseLRI(NativeRuleLearner):
    """Lightweight Rule Induction (Weiss & Indurkhya 2000), specialized for
    speed: its term search is hand-written on a dense float copy of the
    data matrix. `LRI` is the same algorithm built from the rule-search
    components, on every data representation.

    For each class `c` (every class, including both classes of a binary
    problem), `n_rules` DNF rules for `c` against the rest:

    1. Grow a conjunctive term by greedily adding the condition with the
       lowest weighted error ``err1 = FP + k * FN`` among the conditions
       that keep at least one true positive. ``k`` starts at 1 for each
       condition and is doubled while the cheapest condition overall
       would leave no true positive -- so a condition is always added
       while the term still covers negatives. The term stops at
       `max_length` conditions or when it covers no negative.
    2. Add the term to the rule. If the rule has fewer than `max_terms`
       terms and some positive is still uncovered, remove the cases the
       term covers and grow the next term on the rest.
    3. Count, for every training case, whether the finished rule errs on
       it (covers a negative, or leaves a positive uncovered), and
       reweight: ``1 + e**3`` with ``e`` the cumulative errors
       (`LRIReweighting`; halved once one exceeds 32).

    After `freeze_features_after` rules of a class, only the features its
    rules have used so far remain candidates (the paper's speed-up, 50 in
    its experiments; ``None`` never freezes). Rules found again are kept:
    they add votes, as in the paper.

    The model is an `EnsembleModel` of the DNF rules (`ConceptModel`s
    without a default, so each votes only where it fires): the class
    with the most satisfied rules wins; ties, and rows no rule fires on,
    go to the more frequent training class.

    Two readings of the paper to note. Its Table 2 grows a term "until
    FN = 0"; adding a condition can never lower FN, while a term covering
    no negative (FP = 0) can't improve any more, so the term stops at
    FP = 0 here. And "if no added condition adds a true positive, the
    cost of a false negative error is doubled" is read as above. Numeric
    attributes come already binarized (`pyrulearn.data.io.build_dataspec`)
    instead of being thresholded during the search; the data's row
    weights multiply with the error weights.
    """

    def __init__(
        self,
        n_rules: int = 50,
        max_terms: int = 4,
        max_length: int = 5,
        freeze_features_after: Optional[int] = 50,
        power: float = 3.0,
        max_errors: Optional[int] = 32,
    ):
        if n_rules < 1 or max_terms < 1 or max_length < 1:
            raise ValueError("n_rules, max_terms and max_length must be at least 1")
        self.n_rules = n_rules
        self.max_terms = max_terms
        self.max_length = max_length
        self.freeze_features_after = freeze_features_after
        self.power = power
        self.max_errors = max_errors

    def _default_model(self, data: Any) -> type:
        return EnsembleModel

    @produces(EnsembleModel)
    def _fit_native(self, data: Any, **kw) -> EnsembleModel:
        if data.y is None:
            raise ValueError("LRI needs data.y")
        y = np.asarray(data.y)
        classes = [c.item() if isinstance(c, np.generic) else c for c in np.unique(y)]
        if len(classes) < 2:
            raise ValueError("LRI needs at least two classes")
        X = np.asarray(data.X, dtype=bool)
        Xf = X.astype(float)
        base_w = np.ones(len(y)) if data.weights is None else data.weights.astype(float)
        members: List[ConceptModel] = []
        for c in classes:
            for terms in self._rules_for(X, Xf, y == c, base_w):
                rules = [Rule(list(t), target=c, dataspec=data.spec) for t in terms]
                members.append(ConceptModel(annotate_rules(rules, data), label=c))
        model = EnsembleModel(members, default_prediction=MajorityClass(data))
        return annotate_default_rule(model, data)

    # -- one class ----------------------------------------------------------

    def _rules_for(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray,
                   base_w: np.ndarray) -> List[List[Tuple[int, ...]]]:
        """`n_rules` DNF rules (each a list of terms) for `positive` vs. the rest."""
        n, n_features = X.shape
        state = CoveringState(base_w.copy(), positive)
        reweighting = LRIReweighting(power=self.power, max_errors=self.max_errors)
        allowed: Set[int] = set(range(n_features))
        used: Set[int] = set()
        rules: List[List[Tuple[int, ...]]] = []
        for r in range(self.n_rules):
            if self.freeze_features_after is not None and r == self.freeze_features_after and used:
                allowed = set(used)
            terms, covered = self._grow_rule(X, Xf, positive, state.scope, allowed)
            if not terms:
                break
            rules.append(terms)
            used.update(f for t in terms for f in t)
            state.record(covered)
            state.scope = base_w * reweighting.weights(state, covered)
        return rules

    def _grow_rule(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray, w: np.ndarray,
                   allowed: Set[int]) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
        """One DNF rule: its terms, and the rows it covers."""
        remaining = np.ones(len(w), dtype=bool)
        covered = np.zeros(len(w), dtype=bool)
        terms: List[Tuple[int, ...]] = []
        while len(terms) < self.max_terms:
            term, term_cov = self._grow_term(X, Xf, positive, w * remaining, allowed)
            if not term:
                break
            terms.append(term)
            covered |= term_cov
            if not np.any(positive & ~covered & (w > 0)):
                break                       # FN = 0: every positive covered
            remaining &= ~term_cov
        return terms, covered

    def _grow_term(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray, w: np.ndarray,
                   allowed: Set[int]) -> Tuple[Tuple[int, ...], np.ndarray]:
        """One conjunctive term grown on the weights `w` (zero for removed
        cases), and the rows (of all) it covers."""
        wp, wn = w * positive, w * ~positive
        n_pos = float(wp.sum())
        cov = np.ones(len(w), dtype=bool)
        body: List[int] = []
        candidates = np.array(sorted(allowed), dtype=np.int64)
        if n_pos <= 0 or candidates.size == 0:
            return (), cov & False
        fp = float(wn.sum())
        while len(body) < self.max_length and fp > 0:
            # weighted TP/FP of the term with each candidate condition added
            tp_c = (wp * cov) @ Xf[:, candidates]
            fp_c = (wn * cov) @ Xf[:, candidates]
            fn_c = n_pos - tp_c
            live = tp_c > 0
            if body:
                live &= ~np.isin(candidates, body)
            if not live.any():
                break
            dead = ~live
            if body:
                dead &= ~np.isin(candidates, body)
            k = 1.0
            while True:
                err = fp_c + k * fn_c
                best_live = float(err[live].min())
                if not dead.any() or float(err[dead].min()) >= best_live:
                    break
                k *= 2.0               # the cheapest condition would drop every true positive
            j = int(np.flatnonzero(live & (err == best_live))[0])
            f = int(candidates[j])
            body.append(f)
            cov = cov & X[:, f]
            fp = float(fp_c[j])
        return tuple(body), cov


# ======================================================= modular LRI ========


class LRI(NativeRuleLearner):
    """Lightweight Rule Induction (Weiss & Indurkhya 2000), built from the
    rule-search components: see `DenseLRI` for the algorithm in detail --
    the same learner, specialized for speed on a dense matrix, with its
    own term search.

    For each class `c` (every class, including both classes of a binary
    problem), `n_rules` DNF rules for `c` against the rest, one after
    another on the whole training data:

    1. A **term** is what a `SingleRuleLearner` finds: `search` (default
       ``HillClimbing(max_conditions=5, stop_at_local_optimum=False)`` --
       add the best condition each step, even a worsening one, until the
       term covers no negative or has 5 conditions) over the objective
       `heuristic` (default `LRIError`: the weighted ``FP + k * FN`` with
       the paper's adaptive `k`), on the cases with their current weights.
       Any search works; `max_conditions` of the search is the term
       length.
    2. A **rule** has up to `max_terms` terms: after each, the cases it
       covers are removed and the next term is grown on the rest, until
       every positive is covered.
    3. After each rule the cases are reweighted by `reweighting` (default
       `LRIReweighting(power=3, max_errors=32)`: ``1 + e**3``, `e` the
       number of rules so far that err on a case) through a
       `WeightedCovering` of `n_rules` rounds.

    After `freeze_features_after` rules of a class, only the features its
    rules have used so far stay open (a `FeatureSubset`; ``None`` never
    freezes). Rules found again are kept: they add votes.

    The model is an `EnsembleModel` of the DNF rules (`ConceptModel`s
    without a default, so each votes only where it fires): the class with
    the most satisfied rules wins; ties, and rows no rule fires on, go to
    the more frequent training class.

    Runs on every data representation, the search walking cover handles
    with constraint propagation -- a condition the term already implies is
    never added. `DenseLRI` doesn't propagate constraints: with negation or
    threshold features it can append a redundant condition where this
    learner takes the next best real one (and without such features both
    learn identical models). The data's row weights multiply with the
    error weights.
    """

    def __init__(
        self,
        n_rules: int = 50,
        max_terms: int = 4,
        search: Optional[RuleSearch] = None,
        heuristic: Optional[Union[RuleHeuristic, Objective]] = None,
        reweighting: Optional[Reweighting] = None,
        freeze_features_after: Optional[int] = 50,
    ):
        if n_rules < 1 or max_terms < 1:
            raise ValueError("n_rules and max_terms must be at least 1")
        if search is not None and not isinstance(search, RuleSearch):
            raise ValueError(f"search must be a RuleSearch, got {search!r}")
        self.n_rules = n_rules
        self.max_terms = max_terms
        self.search = search if search is not None else HillClimbing(max_conditions=5, stop_at_local_optimum=False)
        self.heuristic = heuristic if heuristic is not None else LRIError()
        self.reweighting = reweighting if reweighting is not None else LRIReweighting()
        self.freeze_features_after = freeze_features_after

    def _default_model(self, data: Any) -> type:
        return EnsembleModel

    @produces(EnsembleModel)
    def _fit_native(self, data: Any, **kw) -> EnsembleModel:
        if data.y is None:
            raise ValueError("LRI needs data.y")
        y = np.asarray(data.y)
        classes = [c.item() if isinstance(c, np.generic) else c for c in np.unique(y)]
        if len(classes) < 2:
            raise ValueError("LRI needs at least two classes")
        term_learner = SingleRuleLearner(self.heuristic, search=self.search)
        members: List[ConceptModel] = []
        for c in classes:
            for terms in self._rules_for(data, c, term_learner):
                members.append(ConceptModel(annotate_rules(terms, data), label=c))
        model = EnsembleModel(members, default_prediction=MajorityClass(data))
        return annotate_default_rule(model, data)

    def _rules_for(self, data: Any, c: Any, term_learner: SingleRuleLearner) -> List[List[Rule]]:
        """`n_rules` DNF rules (each a list of term rules) for `c` vs. the rest."""
        positive = np.asarray(data.y) == c
        covering = WeightedCovering(self.reweighting, stop=[], max_rounds=self.n_rules)
        state = covering.start(data, positive)
        space = None
        used: Set[int] = set()
        rules: List[List[Rule]] = []
        while not covering.exhausted(state):
            if (self.freeze_features_after is not None and state.round == self.freeze_features_after and used):
                space = FeatureSubset(sorted(used))
            terms, covered = self._grow_rule(data, c, positive, state.scope, term_learner, space)
            if not terms:
                break
            rules.append(terms)
            used.update(lit.feature for t in terms for lit in t.conditions)
            covering.update(state, covered)
        return rules

    def _grow_rule(self, data: Any, c: Any, positive: np.ndarray, scope: np.ndarray,
                   term_learner: SingleRuleLearner, space: Optional[SearchSpaceInit]) -> Tuple[List[Rule], np.ndarray]:
        """One DNF rule: its terms, and the rows it covers."""
        weights = data.scope_weights(scope)
        remaining = np.ones(len(positive), dtype=bool)
        covered = np.zeros(len(positive), dtype=bool)
        terms: List[Rule] = []
        while len(terms) < self.max_terms:
            term = term_learner.learn_one_rule(data, c, scope * remaining, space_init=space)
            if term is None or term.length() == 0:
                break
            terms.append(term)
            term_cov = np.asarray(data.coverage(term), dtype=bool)
            covered |= term_cov
            if not np.any(positive & ~covered & (weights > 0)):
                break                       # every positive covered
            remaining &= ~term_cov
        return terms, covered


# ----------------------------------------------------------- AdaBoost ----


class AdaBoost(NativeRuleLearner):
    """Up to `n_estimators` members, each `base_learner.fit` on the
    reweighted training data, as whatever model the base learner builds
    (e.g. `OneVsRest(Pypper())` for concept sets), combined into an
    `EnsembleModel` weighted by SAMME. A member's training error counts at
    least half an example, so that a member without errors doesn't get an
    unbounded weight.

    Two members end SAMME's boosting: one no better than random guessing
    (its weight would not be positive) and one without training errors
    (the weights would not change, and the next member would be the
    same). With `early_stopping` (the default), boosting stops there:
    the first is left out, unless it is the only member, and the second
    is kept. Without it, the weights are reset to those of a bootstrap
    sample (drawn from `random_state`) and boosting goes on -- the first
    member left out, the second kept -- until there are `n_estimators`
    members (or ``2 * n_estimators`` rounds have been tried).

    With `abstain` (the default), a member doesn't vote where its rules
    don't cover an example (`EnsembleModel`'s `abstain`), which suits
    members with rules for every class (one-vs-rest concept sets):
    boosting moves the weight onto few examples, so that a later member's
    default would decide most of the rest, with the full weight of the
    member. Only the vote changes: a member's training error still counts
    its default."""

    def __init__(self, base_learner: NativeRuleLearner, n_estimators: int = 10,
                 early_stopping: bool = True, random_state: Optional[int] = 0, abstain: bool = True):
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be at least 1, got {n_estimators}")
        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.abstain = abstain

    def _default_model(self, data: DataRepresentation) -> type:
        return EnsembleModel

    @produces(EnsembleModel)
    def _fit_native(self, data: DataRepresentation, **kw) -> EnsembleModel:
        if data.y is None:
            raise ValueError("AdaBoost needs data.y")
        y = np.asarray(data.y)
        n, k = data.n_samples, len(np.unique(y))
        base = np.ones(n) if data.weights is None else np.asarray(data.weights, dtype=float)
        w = base * (n / base.sum())
        rng = np.random.default_rng(self.random_state)
        members: List[RuleModel] = []
        weights: List[float] = []
        for _ in range(self.n_estimators if self.early_stopping else 2 * self.n_estimators):
            if len(members) == self.n_estimators:
                break
            member = self.base_learner.fit(data.with_weights(w))
            wrong = np.asarray(member.predict(data)) != y
            err = float(w[wrong].sum() / w.sum())
            guessing = err >= 1.0 - 1.0 / k - 1e-12   # no better than random guessing (up to rounding)
            if not guessing:
                members.append(member)
                weights.append(samme_weight(max(err, 0.5 / n), k))
            if guessing or err == 0.0:
                if self.early_stopping:
                    if not members:
                        members.append(member)
                        weights.append(1.0)
                    break
                draws = np.bincount(rng.integers(0, n, size=n), minlength=n)
                w = draws * base
                w = w * (n / w.sum())
                continue
            w = w * np.exp(weights[-1] * wrong)
            w = w * (n / w.sum())
        if not members:   # every round no better than guessing
            members, weights = [member], [1.0]
        model = EnsembleModel(members, default_prediction=MajorityClass(data), member_weights=weights,
                              abstain=self.abstain)
        return annotate_default_rule(model, data)


def samme_weight(err: float, n_classes: int, eps: float = 1e-10) -> float:
    """SAMME's member weight ``log((1 - err) / err) + log(K - 1)``, with
    `err` kept at least `eps` (a member without training errors)."""
    err = max(err, eps)
    return float(np.log((1.0 - err) / err) + np.log(n_classes - 1))
