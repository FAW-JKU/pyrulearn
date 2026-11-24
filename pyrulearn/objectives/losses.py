"""
pyrulearn.objectives.losses
===========================

Loss-based objectives: the losses of the boosting learners of
`pyrulearn.learners.boosting` and the weighted error of LRI.

- `BoostingLoss` (`LogisticLoss`, `ExponentialLoss`, `SigmoidLoss`) --
  multi-class losses for `ENDER`/`Boomer`, with one score per class and
  example;
- `MarginLoss` (`LogisticMarginLoss`, `SquaredMarginLoss`) -- binary
  losses for `ORB`, with labels ``+-1`` and one score per example, whose
  first and second derivatives `XGBGain` scores rules with;
- `LRIError` -- the weighted error ``FP + k * FN`` that LRI grows its
  terms by.

They used to live in `pyrulearn.learners.boosting` and
`pyrulearn.learners.lri`, which still re-export them.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..heuristics import RuleStats
from . import Objective, ValueSumObjective, count_open_children, stats_from_handle


class BoostingLoss(ABC):
    """A loss for `ENDER`. The model keeps one score per class and example,
    ``F`` (``n x K``); a rule votes for one class ``k`` by adding its
    weight ``alpha`` to that class's scores on the rows it covers.

    - `values` -- the per-example loss (times the example weights ``d``);
      `value` its total;
    - `derivatives` -- per example and class, the first and second
      derivative of the loss with respect to such a vote at ``alpha = 0``,
      times ``d``;
    - `response` -- a rule's weight from the sums of those derivatives
      over the rows it covers (default: the Newton step ``-g / h``).

    ``Y`` is the one-hot class matrix."""

    #: whether the loss handles more than two classes
    multiclass: bool = True
    #: whether its second derivative is usable for Newton steps (convex)
    convex: bool = True
    #: whether `response` is a Newton step, improvable by iterating
    #: (the default rule is fitted by iterating it)
    iterative_response: bool = True

    @abstractmethod
    def values(self, F: np.ndarray, Y: np.ndarray, d: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def value(self, F: np.ndarray, Y: np.ndarray, d: np.ndarray) -> float:
        return float(self.values(F, Y, d).sum())

    @abstractmethod
    def derivatives(self, F: np.ndarray, Y: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def response(self, g: float, h: float) -> float:
        return -g / h if h > 0 else 0.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _softmax(F: np.ndarray) -> np.ndarray:
    e = np.exp(F - F.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class LogisticLoss(BoostingLoss):
    """The multinomial negative log-likelihood, ``log sum_k exp(F_k) -
    F_y`` (MLRules; for two classes the logit loss). A vote for class
    ``k``: first derivative ``p_k - [y = k]``, second ``p_k (1 - p_k)``,
    with ``p`` the softmax of the scores. The rule weight is a Newton
    step (the paper's Eq. 15)."""

    def values(self, F, Y, d):
        m = F.max(axis=1)
        lse = m + np.log(np.exp(F - m[:, None]).sum(axis=1))
        return d * (lse - (F * Y).sum(axis=1))

    def derivatives(self, F, Y, d):
        P = _softmax(F)
        return d[:, None] * (P - Y), d[:, None] * P * (1.0 - P)


def _binary_margin(F: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``y`` (``+1`` for the second class) and ``f = F_1 - F_0``."""
    return np.where(Y[:, 1] > 0, 1.0, -1.0), F[:, 1] - F[:, 0]


class ExponentialLoss(BoostingLoss):
    """AdaBoost's exponential loss ``exp(-y f)`` for two classes, with
    ``f = F_1 - F_0`` and ``y = +1`` for the second class, ``-1`` for the
    first. A vote for a class moves ``f`` towards it; with ``w = exp(-y
    f)``, the first derivative is ``-w`` on that class's examples and
    ``+w`` on the others, the second ``w``. The rule weight is the exact
    minimizer ``1/2 ln((W+ + eps) / (W- + eps))`` (the paper's Eq. 14; ``W+``
    / ``W-`` the weights of the covered examples of the voted / the other
    class), smoothed as in Slipper (`eps`: half an example's average
    weight) so that a pure rule doesn't get an infinite weight."""

    multiclass = False
    iterative_response = False

    def __init__(self):
        self.eps = 0.5

    def values(self, F, Y, d):
        y, f = _binary_margin(F, Y)
        return d * np.exp(-y * f)

    def derivatives(self, F, Y, d):
        y, f = _binary_margin(F, Y)
        w = d * np.exp(-y * f)
        self.eps = 0.5 * float(w.mean())
        return np.stack([y * w, -y * w], axis=1), np.stack([w, w], axis=1)

    def response(self, g, h):
        w_pos, w_neg = (h - g) / 2.0, (h + g) / 2.0            # g = W- - W+, h = W+ + W-
        return 0.5 * math.log((max(w_pos, 0.0) + self.eps) / (max(w_neg, 0.0) + self.eps))


class SigmoidLoss(BoostingLoss):
    """The sigmoid loss ``1 / (1 + exp(y f))`` for two classes (``f`` as in
    `ExponentialLoss`): a smooth approximation of the 0-1 loss, bounded,
    and so less sensitive to outliers, but not convex -- so no Newton
    steps: a rule's weight is the constant step `beta` (default 0.2; the
    paper uses the same value as `ConstantStep`'s), and the Newton
    criterion is refused."""

    multiclass = False
    convex = False
    iterative_response = False

    def __init__(self, beta: float = 0.2):
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = beta

    def __repr__(self) -> str:
        return f"SigmoidLoss(beta={self.beta!r})"

    def values(self, F, Y, d):
        y, f = _binary_margin(F, Y)
        return d / (1.0 + np.exp(y * f))

    def derivatives(self, F, Y, d):
        y, f = _binary_margin(F, Y)
        L = 1.0 / (1.0 + np.exp(y * f))
        slope = d * L * (1.0 - L)                                # -dL/d(y f)
        curvature = d * L * (1.0 - L) * (1.0 - 2.0 * L)
        return np.stack([y * slope, -y * slope], axis=1), np.stack([curvature, curvature], axis=1)

    def response(self, g, h):
        return self.beta


class XGBGain(ValueSumObjective):
    """Optimal rule boosting's objective (Boley et al. 2021) as an
    `Objective`: the XGBoost-style gain of a rule that covers the rows
    ``Q``,

        (sum_{i in Q} g_i)**2 / (reg + sum_{i in Q} h_i)

    for one boosting round's per-row loss derivatives ``g``/``h`` (row
    weights already multiplied in). It is the loss reduction of the
    rule's best weight ``w = -sum g / (reg + sum h)`` under the
    second-order approximation of the loss.

    `sign` picks which rules count: ``None`` (default) both directions --
    the objective is symmetric, and the weight's sign decides the class
    afterwards (realkd's search); ``+1`` only rules that raise the scores
    (``sum g < 0``, for the class the scores favor), ``-1`` only rules that
    lower them. The empty rule scores ``-inf``: it isn't a rule this
    search looks for (a boosting learner's intercept is a separate step),
    so a greedy search always takes a first condition.

    Bounds, for any subset of a rule's covered rows (what a refinement can
    cover): from the sums, ``max(G+**2, G-**2) / reg`` -- ``G+``/``G-`` the
    sums of the positive/negative ``g`` -- infinite for ``reg = 0``; and,
    from the covered rows themselves (`exact_bound`), the paper's bound:
    the best prefix or suffix of the covered rows sorted by ``g/h``."""

    def __init__(self, g: np.ndarray, h: np.ndarray, reg: float = 1.0, sign: Optional[int] = None):
        if sign not in (None, 1, -1):
            raise ValueError(f"sign must be None, 1 or -1, got {sign!r}")
        self.g = np.asarray(g, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.reg = float(reg)
        self.sign = sign
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(self.h > 0, self.g / self.h, np.sign(self.g) * np.inf)
        order = np.argsort(-ratio, kind="stable")              # rows by g/h, descending
        self._rank = np.empty(len(self.g), dtype=np.int64)
        self._rank[order] = np.arange(len(self.g))

    def columns(self) -> np.ndarray:
        return np.column_stack([self.g, self.h, np.maximum(self.g, 0.0), np.minimum(self.g, 0.0)])

    def _gain(self, G, H):
        if self.sign == 1:
            G = np.minimum(G, 0.0)
        elif self.sign == -1:
            G = np.maximum(G, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return G ** 2 / (self.reg + H)

    def score_sums(self, sums: np.ndarray, length: Any) -> np.ndarray:
        gain = np.asarray(self._gain(sums[0], sums[1]), dtype=float)
        return np.where(np.asarray(length) == 0, -np.inf, gain)

    def bound_sums(self, sums: np.ndarray) -> np.ndarray:
        g_pos, g_neg = sums[2], sums[3]
        top = (np.maximum(g_pos ** 2, g_neg ** 2) if self.sign is None
               else g_neg ** 2 if self.sign == 1 else g_pos ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(top == 0, 0.0, top / self.reg)

    def exact_bound(self, data, handle) -> float:
        idx = np.flatnonzero(data.cover_rows(handle))
        if idx.size == 0:
            return 0.0
        idx = idx[np.argsort(self._rank[idx], kind="stable")]
        gq, hq = self.g[idx], self.h[idx]
        best = -math.inf
        if self.sign in (None, -1):                            # largest g/h first: sum g > 0
            best = max(best, float(np.max(self._gain(np.cumsum(gq), np.cumsum(hq)))))
        if self.sign in (None, 1):                             # smallest g/h first: sum g < 0
            best = max(best, float(np.max(self._gain(np.cumsum(gq[::-1]), np.cumsum(hq[::-1])))))
        return best


class MarginLoss(ABC):
    """A loss for `ORB`/`DenseORB`: binary labels ``y = +-1`` and one score
    ``s`` per example, as in `realkd`. `derivatives` gives the per-example
    first and second derivatives ``g``/``h`` with respect to ``s`` --
    what `XGBGain` scores rules with."""

    @abstractmethod
    def derivatives(self, y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class LogisticMarginLoss(MarginLoss):
    """``log(1 + exp(-y s))`` -- realkd's ``"logistic"`` loss."""

    def derivatives(self, y, s):
        sig = 1.0 / (1.0 + np.exp(y * s))                       # sigmoid(-y s)
        return -y * sig, sig * (1.0 - sig)


class SquaredMarginLoss(MarginLoss):
    """``(y - s)**2`` -- realkd's ``"squared"`` loss."""

    def derivatives(self, y, s):
        return 2.0 * (s - y), np.full_like(s, 2.0)


@dataclass(frozen=True)
class _TermStats(RuleStats):
    """`RuleStats` plus what `LRIError` needs along a term: `k`, the error
    weight of the step that produced them, and `body`, the features of the
    rule (of a batch of children: their parent's, and `feature` the one
    each adds)."""
    k: Any = 1.0
    body: Any = frozenset()
    feature: Any = None


def _per_child(value: Any, n: int) -> np.ndarray:
    """`value` as an array of `n` entries (a scalar repeated)."""
    return np.asarray(value) if np.ndim(value) else np.full(n, value)


def _bodies(value: Any, n: int) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = value
    return out


class LRIError(Objective):
    """The error Lightweight Rule Induction minimizes while it grows a term
    (Weiss & Indurkhya 2000, Table 2): the weighted ``FP + k * FN`` of the
    term (with the weights of the search's scope: LRI's, with the cases
    earlier terms of the rule cover removed). A search maximizes the
    score, so it is ``-(FP + k * FN)``.

    `k` defaults to the paper's adaptive choice, per step: 1, doubled while
    the cheapest condition overall -- among *all* conditions not yet in the
    term, those that would leave no true positive included -- would leave
    none. So a condition that keeps a true positive is always found, and
    ``k`` grows (towards keeping positives) only as far as that takes; a
    number fixes ``k`` for every step.

    Only conditions keeping a true positive are candidates (the others are
    dead, as for any heuristic: a child covering no positive); a term is
    consistent once it covers no negative, which ends its growth in
    `HillClimbing(stop_at_local_optimum=False)`. With the adaptive `k`,
    scores of different steps aren't comparable, so run it with a search
    that doesn't compare across steps (`HillClimbing` without the
    local-optimum stop, `BeamSearch(optimistic_pruning=False)`); with a
    fixed `k` the bound ``-k * FN`` (every false positive dropped) is
    admissible. The statistics are `RuleStats`, so filtering and stopping
    criteria work."""

    supports_criteria = True

    def __init__(self, k: Optional[float] = None):
        if k is not None and not k > 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self._universe: Optional[FrozenSet[int]] = None

    @property
    def supports_bound(self) -> bool:
        return self.k is not None

    def begin(self, data, target_class, example_mask) -> None:
        self._universe = None            # the first node of a search fixes its universe of conditions

    def rule_stats(self, data, target_class, rule, handle) -> _TermStats:
        base = stats_from_handle(data, target_class, rule, handle)
        return _TermStats(tp=base.tp, fp=base.fp, fn=base.fn, tn=base.tn, length=base.length,
                          k=1.0 if self.k is None else self.k,
                          body=frozenset(lit.feature for lit in rule.conditions))

    def _choose_k(self, tp, fp, fn) -> float:
        if self.k is not None:
            return float(self.k)
        live = tp > 0
        dead = ~live
        if not live.any() or not dead.any():
            return 1.0
        k = 1.0
        while k < 2.0 ** 60:
            err = fp + k * fn
            if float(err[dead].min()) >= float(err[live].min()):
                break
            k *= 2.0
        return k

    def children(self, data, target_class, mask, handle, stats: _TermStats):
        if self._universe is None:
            self._universe = frozenset(mask)
        open_features = self._universe - stats.body
        features, tp, fp, fn, tn, _ = count_open_children(data, target_class, open_features, handle, stats)
        features = np.asarray(features, dtype=np.int64)
        tp, fp, fn, tn = (np.asarray(a) for a in (tp, fp, fn, tn))
        k = self._choose_k(tp, fp, fn)
        in_mask = np.isin(features, np.fromiter(mask, dtype=np.int64, count=len(mask)))
        alive = (tp > 0) & in_mask
        dead = set(features[(tp <= 0) & in_mask].tolist())
        batch = _TermStats(tp=tp[alive], fp=fp[alive], fn=fn[alive], tn=tn[alive], length=stats.length + 1,
                           k=k, body=stats.body, feature=features[alive])
        # counts of children keeping every row are the parent's exactly: nothing to guard
        return features[alive], batch, np.zeros(int(alive.sum()), dtype=bool), dead

    def batch_score(self, batch: _TermStats, parent=None):
        return -(np.asarray(batch.fp, dtype=float) + batch.k * np.asarray(batch.fn, dtype=float))

    def score_one(self, stats: _TermStats, parent=None) -> float:
        return -(float(stats.fp) + float(stats.k) * float(stats.fn))

    def batch_bound(self, batch: _TermStats):
        return -(batch.k * np.asarray(batch.fn, dtype=float))

    def bound_one(self, stats: _TermStats) -> float:
        return -(float(stats.k) * float(stats.fn))

    def take(self, batch: _TermStats, index) -> _TermStats:
        n = len(np.atleast_1d(batch.tp))

        def cut(v):
            return np.asarray(v)[index] if np.ndim(v) else v
        body = batch.body if not isinstance(batch.body, np.ndarray) else batch.body[index]
        return _TermStats(tp=batch.tp[index], fp=batch.fp[index], fn=batch.fn[index], tn=batch.tn[index],
                          length=cut(batch.length), k=cut(batch.k), body=body, feature=cut(batch.feature))

    def at(self, batch: _TermStats, i: int) -> _TermStats:
        body = batch.body[i] if isinstance(batch.body, np.ndarray) else batch.body
        length = batch.length if np.ndim(batch.length) == 0 else int(np.asarray(batch.length)[i])
        k = batch.k if np.ndim(batch.k) == 0 else float(np.asarray(batch.k)[i])
        return _TermStats(tp=batch.tp[i], fp=batch.fp[i], fn=batch.fn[i], tn=batch.tn[i], length=length,
                          k=k, body=body | {int(batch.feature[i])})

    def concat(self, batches) -> _TermStats:
        sizes = [len(np.atleast_1d(b.tp)) for b in batches]
        return _TermStats(
            tp=np.concatenate([b.tp for b in batches]), fp=np.concatenate([b.fp for b in batches]),
            fn=np.concatenate([b.fn for b in batches]), tn=np.concatenate([b.tn for b in batches]),
            length=np.concatenate([_per_child(b.length, n) for b, n in zip(batches, sizes)]),
            k=np.concatenate([_per_child(b.k, n) for b, n in zip(batches, sizes)]),
            body=np.concatenate([_bodies(b.body, n) for b, n in zip(batches, sizes)]),
            feature=np.concatenate([np.asarray(b.feature) for b in batches]),
        )

    def dead(self, stats: _TermStats) -> bool:
        return stats.tp == 0

    def consistent(self, stats: _TermStats) -> bool:
        return stats.fp == 0

    def __repr__(self) -> str:
        return "LRIError()" if self.k is None else f"LRIError(k={self.k!r})"
