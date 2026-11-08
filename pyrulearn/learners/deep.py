"""
pyrulearn.learners.deep
=======================

Learners of `pyrulearn.models.DeepModel`s: fully connected networks of
alternating AND and OR layers with 0/1 edge weights over the data's
Boolean features -- formulas in negation normal form, read as stacked
rules. They differ only in how they find the weights:

- `DeepLocalSearch` -- stochastic hill climbing over edge flips (Beck &
  Fürnkranz 2021), with moves that keep the rules consistent with the
  `DataSpec`'s constraints (`contradictions`).
- `DeepGenetic` -- a memetic algorithm: crossover of aligned networks,
  mutation, a few hill-climbing steps per child.
- `DeepSAT` -- an exact fit by SAT solving, the smallest network by
  MaxSAT (Seip, Fürnkranz et al.).

`DeepRuleLearner` is their shared base: the architecture (`layers`), the
targets (one output node per class; for two classes one output for the
less frequent class, the other the default), the start weights (`init=`,
a `DeepInit`) and wrapping the weights into the model. A subclass
implements `_fit_weights`, and may override `_initialize`.

The starts (`DeepInit`):

- `RandomInit` -- random rules of a few conditions, Beck & Fürnkranz's
  start.
- `ModelInit` -- another `DeepModel`'s weights (chaining learners).
- `RuleSetInit` -- a rule set, or a learner fitted first: each output's
  rules in the first layer, passed up to the output.
"""

from __future__ import annotations

import math
from fractions import Fraction
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ..combiners import CountVoteCombiner
from ..data import DataRepresentation
from ..models import DeepModel, MajorityClass, RuleModel
from .base import NativeRuleLearner, RuleLearner, produces


def nnf_forward(weights: Sequence[np.ndarray], X: np.ndarray, first: str = "and") -> List[np.ndarray]:
    """The value of every node for every row of the Boolean matrix `X`
    (`(n_samples, n_features)`), one `(n_nodes, n_samples)` bool array per
    layer of `weights`, alternating from `first` (``"and"``: AND, OR, AND,
    ...): an AND node holds where none of its inputs is false, an OR node
    where one is true."""
    values: List[np.ndarray] = []
    below = X.T
    for i, w in enumerate(weights):
        values.append(_layer(w, below, _is_and(i, first)))
        below = values[-1]
    return values


def _is_and(layer: int, first: str) -> bool:
    return (layer % 2 == 0) == (first == "and")


def _layer(w: np.ndarray, below: np.ndarray, is_and: bool) -> np.ndarray:
    wi = w.astype(np.int32)
    if is_and:
        return wi @ (~below).astype(np.int32) == 0
    return wi @ below.astype(np.int32) > 0


class TracePoint(NamedTuple):
    """One new best network during a fit (`DeepRuleLearner(trace=True)`):
    the effort so far (for the search learners in candidate-rows: one
    candidate network evaluated on one row, a hill-climbing step on a
    batch counting every flip it evaluates on every row of the batch; for
    `DeepSAT` in solver calls), the seconds so far, the weighted training
    error, the weights, and for a learner that minimizes size (`DeepSAT`'s
    MaxSAT) the size it minimizes."""

    evaluations: int
    seconds: float
    train_error: float
    weights: List[np.ndarray]
    size: Optional[int] = None


def pruned_weights(weights: Sequence[np.ndarray]) -> List[np.ndarray]:
    """The backbone of `weights` (as `DeepModel.pruned`): every edge into a
    node the outputs don't depend on set to 0 -- the same function, no
    more edges."""
    used = [np.ones(weights[-1].shape[0], dtype=bool)]
    for w in reversed(weights[1:]):
        used.insert(0, w[used[0]].any(axis=0))
    return [w & u[:, None] for w, u in zip(weights, used)]


class _Tracker:
    """Counts a fit's effort and, with `keep`, records each new best."""

    def __init__(self, keep: bool):
        self.keep = keep
        self.evaluations = 0
        self.start = time.perf_counter()
        self.best = math.inf
        self.best_size = math.inf
        self.best_cost = math.inf
        self.points: List[TracePoint] = []

    def count(self, n: int) -> None:
        self.evaluations += int(n)

    def offer(self, weights: Sequence[np.ndarray], error: float, size: Optional[int] = None,
              cost: Optional[float] = None) -> None:
        """`weights` reach the training error `error` (and the `size` a
        learner minimizes): a new point if better -- a lower error, or the
        same error and a smaller size; with `cost` (a learner's objective
        trading the two), a lower cost."""
        if cost is not None:
            better = cost < self.best_cost
            if better:
                self.best_cost = cost
        else:
            better = error < self.best or (error == self.best and size is not None and size < self.best_size)
        if better:
            self.best = error
            self.best_size = size if size is not None else math.inf
            if self.keep:
                self.points.append(TracePoint(self.evaluations, time.perf_counter() - self.start, error,
                                              [w.copy() for w in weights], size))


class DeepInit(ABC):
    """The start weights of a `DeepRuleLearner`."""

    @abstractmethod
    def weights(self, data: DataRepresentation, labels: Sequence[Any], shapes: List[tuple],
                rng: np.random.Generator, first: str = "and") -> List[np.ndarray]:
        """Bool matrices of `shapes` (one per layer above the inputs, the
        output layer's rows in the order of `labels`, the layers
        alternating from `first`) for `data`."""


class RandomInit(DeepInit):
    """Random rules (Beck & Fürnkranz's start): a rule of the first layer
    takes each attribute with probability ``avg_rule_length /
    n_attributes``, with one random feature of it, drawn again while it
    has fewer than `min_rule_length` conditions (Beck: 0 -- but two empty
    rules are true on every row and keep an output OR true: no single flip
    changes that); a node of a layer in between gets one edge to a random
    node above it, and every edge with probability `init_prob`. An OR
    output: a single one is connected to every node below, several divide
    the nodes below among them (round-robin), so that the classes don't
    start out with the same rules. An AND output (all of its inputs must
    hold) gets one edge, round-robin. With `consistent`, a rule skips a
    feature that contradicts one it already has by the `DataSpec`'s
    constraints (`contradictions`); Beck's start is ``consistent=False,
    min_rule_length=0``. A first OR layer (`first` ``"or"``) gets its
    literals the same way, without the consistency check."""

    def __init__(self, avg_rule_length: float = 3, min_rule_length: int = 1, init_prob: float = 0.2,
                 consistent: bool = True):
        if not 0.0 <= init_prob <= 1.0:
            raise ValueError("init_prob is a probability")
        self.avg_rule_length = avg_rule_length
        self.min_rule_length = min_rule_length
        self.init_prob = init_prob
        self.consistent = consistent

    def weights(self, data, labels, shapes, rng, first="and"):
        groups = _attribute_groups(data.spec)
        consistent = self.consistent and first == "and"
        if consistent:
            apart = contradictions(data.spec)
        weights = [np.zeros(s, dtype=bool) for s in shapes]
        attributes = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        p = min(self.avg_rule_length, len(attributes)) / max(len(attributes), 1)
        least = min(self.min_rule_length, len(attributes))
        for k in range(shapes[0][0]):
            while True:
                weights[0][k] = False
                for feats in attributes:
                    if rng.random() < p:
                        j = feats[rng.integers(len(feats))]
                        if not (consistent and (apart[j] & weights[0][k]).any()):
                            weights[0][k, j] = True
                if weights[0][k].sum() >= least:
                    break
        for w in weights[1:-1]:
            for j in range(w.shape[1]):  # every node gets at least one edge upwards
                w[rng.integers(w.shape[0]), j] = True
                w[:, j] |= rng.random(w.shape[0]) < self.init_prob
        last = weights[-1]
        if len(weights) == 1:
            return weights  # the output is the first layer: its rules (or clauses) above
        last[:] = False
        if _is_and(len(weights) - 1, first):
            last[np.arange(last.shape[0]), np.arange(last.shape[0]) % last.shape[1]] = True
        elif last.shape[0] == 1:
            last[:] = True
        else:
            last[np.arange(last.shape[1]) % last.shape[0], np.arange(last.shape[1])] = True
        return weights


class ModelInit(DeepInit):
    """The weights of `model`, a `DeepModel` of the same architecture with
    the same outputs, its inputs matched to the data's by name."""

    def __init__(self, model: DeepModel):
        self.model = model

    def weights(self, data, labels, shapes, rng, first="and"):
        model = self.model.remap(data.spec)
        if model.first != first:
            raise ValueError(f"the initial model's first layer is {model.first!r}, not {first!r}")
        if list(model.outputs) != list(labels) or [w.shape for w in model.weights] != list(shapes):
            raise ValueError(f"the initial model has layers {model.layers} and outputs {model.outputs}, "
                             f"not {[s[0] for s in shapes[:-1]]} and {list(labels)}")
        return [w.copy() for w in model.weights]


class RuleSetInit(DeepInit):
    """A rule set as the start: `rules` is a rule model, or a `RuleLearner`
    fitted on the training data first. Each output's rules (those with its
    label as head, matched to the data's features by name) are nodes of
    the first AND layer, the output's OR over them; in a deeper network,
    each output's OR node is passed up through one node per layer (an AND
    or OR with that single input). The other nodes get `fill`'s start
    (default `RandomInit()`) but no edge upwards, so the network starts out
    as the rule set's DNF of every output. The first layer needs a node for
    every rule, every layer above it one per output; rules of other heads
    (the default class of a two-class problem) are left out. Needs a first
    AND layer; without hidden layers (an AND output), each output can take
    one rule only."""

    def __init__(self, rules: Union[RuleModel, RuleLearner], fill: Optional[DeepInit] = None):
        self.rules = rules
        self.fill = fill if fill is not None else RandomInit()

    def weights(self, data, labels, shapes, rng, first="and"):
        if first != "and":
            raise ValueError("RuleSetInit puts rules into a first AND layer (first='and')")
        model = self.rules.fit(data) if isinstance(self.rules, RuleLearner) else self.rules
        bodies = [[r for r in model.rules if r.target == label] for label in labels]
        n_rules = sum(len(b) for b in bodies)
        if len(shapes) == 1 and any(len(b) > 1 for b in bodies):
            raise ValueError("without hidden layers each output is one rule, but the rule set has more")
        if shapes[0][0] < n_rules:
            raise ValueError(f"{n_rules} rules for the outputs, but a first layer of {shapes[0][0]} nodes")
        if any(s[0] < len(labels) for s in shapes[1:-1]):
            raise ValueError(f"every layer above the first needs a node per output ({len(labels)})")
        weights = self.fill.weights(data, labels, shapes, rng, first)
        rule_nodes: List[List[int]] = []
        node = 0
        for rules in bodies:
            rule_nodes.append(list(range(node, node + len(rules))))
            for r in rules:
                weights[0][node] = False
                for lit in r.conditions:
                    weights[0][node, data.spec.feature_index(r.dataspec.feature_name(lit.feature))] = True
                node += 1
        if len(weights) == 1:
            return weights
        # the rules' nodes, and above them one node per output, feed only each other
        weights[1][:, :n_rules] = False
        for w in weights[2:]:
            w[:, :len(labels)] = False
        if len(weights) == 2:
            for k, nodes in enumerate(rule_nodes):
                weights[1][k] = False
                weights[1][k, nodes] = True
            return weights
        for k, nodes in enumerate(rule_nodes):
            weights[1][k] = False
            weights[1][k, nodes] = True
            for w in weights[2:]:
                w[k] = False
                w[k, k] = True
        return weights


class DeepRuleLearner(NativeRuleLearner):
    """Base of the `DeepModel` learners. `layers` are the hidden layer
    sizes, AND, OR, ..., AND (an odd number of them; by default one AND
    layer of ``ceil(sqrt(n))`` nodes for *n* features, a DNF; `first`, the
    type of the layer above the inputs, as in `DeepModel`); the output
    layer has one OR node per class, or for two classes one for the less
    frequent class, the other being the default -- or, with `target_class`,
    one for that class, the most frequent other class the default (with
    more classes, too: one class against the rest). With more classes, a row
    no output covers gets the most frequent class, a row several cover the
    class with the most covering rules (`CountVoteCombiner`). `init` (a
    `DeepInit`, default `RandomInit()`) gives the start weights, through
    `_initialize`, which a subclass may override.

    The learners are anytime algorithms: with `trace`, the model's `trace`
    lists every new best network on the training set during the fit
    (`TracePoint`s: effort, seconds, training error, weights), and
    `DeepModel.trace_model(i)` turns one into a model, e.g. to measure
    the test accuracy along the run."""

    def __init__(
        self,
        layers: Optional[Sequence[int]] = None,
        first: str = "and",
        init: Optional[DeepInit] = None,
        trace: bool = False,
        random_state: Optional[int] = 0,
        target_class: Optional[Any] = None,
    ):
        if first not in ("and", "or"):
            raise ValueError(f"first must be 'and' or 'or', got {first!r}")
        self.layers = None if layers is None else [int(n) for n in layers]
        self.first = first
        self.init = init if init is not None else RandomInit()
        self.trace = trace
        self.random_state = random_state
        self.target_class = target_class
        self._tracker = _Tracker(False)

    def _default_model(self, data: DataRepresentation) -> type:
        return DeepModel

    @produces(DeepModel)
    def _fit_native(self, data: DataRepresentation, **kw) -> DeepModel:
        if data.y is None:
            raise ValueError(f"{type(self).__name__} needs data.y")
        y = np.asarray(data.y)
        classes, counts = np.unique(y, return_counts=True)
        classes = [c.item() if isinstance(c, np.generic) else c for c in classes]
        if len(classes) < 2:
            raise ValueError(f"{type(self).__name__} needs at least two classes")
        if self.target_class is not None:
            if self.target_class not in classes:
                raise ValueError(f"target_class {self.target_class!r} is not a class of the data ({classes})")
            labels = [self.target_class]
        else:
            labels = [classes[int(np.argmin(counts))]] if len(classes) == 2 else classes
        targets = np.vstack([y == c for c in labels])
        X = np.asarray(data.X, dtype=bool)
        layers = (self.layers if self.layers is not None
                  else [math.ceil(math.sqrt(data.spec.n_features))])
        shapes = [w.shape for w in DeepModel(data.spec, labels, layers, first=self.first).weights]
        sample_weight = (np.ones(len(y)) if data.weights is None
                         else np.asarray(data.weights, dtype=float))
        rng = np.random.default_rng(self.random_state)
        self._tracker = _Tracker(self.trace)
        start = self._initialize(data, labels, shapes, rng)
        if [w.shape for w in start] != shapes:
            raise ValueError(f"the start weights have shapes {[w.shape for w in start]}, not {shapes}")
        weights = self._fit_weights(data, labels, X, targets, shapes,
                                    [np.asarray(w, dtype=bool) for w in start], sample_weight, rng)
        if len(labels) == 1:  # the other class, or the most frequent of the others
            others = [(n, c) for c, n in zip(classes, counts) if c != labels[0]]
            default = max(others, key=lambda nc: nc[0])[1]
        else:
            default = MajorityClass(data)
        model = DeepModel(data.spec, labels, weights=weights, default_prediction=default,
                          combiner=CountVoteCombiner(), first=self.first)
        model.trace = self._tracker.points if self.trace else None
        return model

    def _error(self, weights, X, targets, sample_weight) -> float:
        """The weighted training error of `weights` on (X, targets),
        counted as one candidate on every row."""
        self._tracker.count(X.shape[0])
        out = nnf_forward(weights, X, self.first)[-1]
        return float(((out != targets) * sample_weight).sum())

    def _initialize(self, data: DataRepresentation, labels: Sequence[Any], shapes: List[tuple],
                    rng: np.random.Generator) -> List[np.ndarray]:
        """The start weights: `init`'s."""
        return self.init.weights(data, labels, shapes, rng, self.first)

    def _fit_weights(self, data: DataRepresentation, labels: Sequence[Any], X: np.ndarray,
                     targets: np.ndarray, shapes: List[tuple], start: List[np.ndarray],
                     sample_weight: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
        """The 0/1 weight matrices (bool, of `shapes`) for `data` (`X` its
        Boolean matrix) and the output nodes' `targets` (`(n_outputs,
        n_samples)` bool, one row per entry of `labels`), starting from
        `start` (`_initialize`'s, which may be called for more starts)."""
        raise NotImplementedError


class DeepLocalSearch(DeepRuleLearner):
    """Stochastic hill climbing over the edges of a `DeepModel`, after Beck
    & Fürnkranz (2021, their code: github.com/f-beck/scikit-learn-rule-
    network). On a mini-batch of `batch_size` training rows, every single
    flip of an edge is evaluated and the one reaching the lowest error on
    the batch is made (the first such, in Beck's order), until no flip
    improves (or after `max_flips`). After each batch the weights are
    measured on the whole training set and the best ones kept; `n_epochs`
    passes over the batches (``None``: until an epoch brings no
    improvement), then one last hill climb on all rows from the best
    weights. With `optimize_last_layer`, each epoch also rebuilds the
    output layer greedily: no edges, then the edge that improves most,
    until none does.

    The start is `init`'s (default `RandomInit()`; Beck's is
    `RandomInit(min_rule_length=0, consistent=False)`).

    `moves` decides what flipping on an edge of the first layer does to
    the rule's other conditions:

    - ``"consistent"`` (default): it drops every condition that
      contradicts the new one by the `DataSpec`'s constraints
      (`contradictions`) -- the other values of the attribute, the
      condition's negation, a contradicting threshold. No flip makes a
      rule contradictory.
    - ``"attribute"``: it drops the other features of the same attribute
      (Beck's move; on nominal data without negation features the same
      as ``"consistent"``).
    - ``"free"``: nothing is dropped.

    Errors are counted per output node and weighted by the rows' weights;
    on unweighted two-class data that is the accuracy Beck climbs on.
    """

    MOVES = ("consistent", "attribute", "free")
    #: cells of a (candidates x nodes x rows) array evaluated at once
    _CHUNK_CELLS = 1 << 22

    def __init__(
        self,
        layers: Optional[Sequence[int]] = None,
        moves: str = "consistent",
        n_epochs: Optional[int] = 5,
        batch_size: Optional[int] = 50,
        max_flips: Optional[int] = None,
        optimize_last_layer: bool = False,
        init: Optional[DeepInit] = None,
        first: str = "and",
        target_class: Optional[Any] = None,
        trace: bool = False,
        random_state: Optional[int] = 0,
    ):
        super().__init__(layers=layers, first=first, init=init, trace=trace, random_state=random_state,
                         target_class=target_class)
        if moves not in self.MOVES:
            raise ValueError(f"moves must be one of {self.MOVES}, got {moves!r}")
        if (n_epochs is not None and n_epochs < 0) or (max_flips is not None and max_flips < 0):
            raise ValueError("n_epochs and max_flips must be at least 0")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.moves = moves
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_flips = max_flips
        self.optimize_last_layer = optimize_last_layer

    # -- the fit ----------------------------------------------------------

    def _fit_weights(self, data, labels, X, targets, shapes, start, sample_weight, rng):
        conflicts = self._conflicts(data.spec)
        weights = start
        n = X.shape[0]
        batch = n if self.batch_size is None else min(self.batch_size, n)
        n_batches = n // batch
        best = [w.copy() for w in weights]
        best_error = self._error(weights, X, targets, sample_weight)
        self._tracker.offer(weights, best_error)
        epochs = self.n_epochs if self.n_epochs is not None else np.iinfo(np.int32).max
        for _ in range(epochs):
            improved = False
            steps = [None] * n_batches + (["last"] if self.optimize_last_layer else [])
            for step in steps:
                if step is None:
                    rows = rng.permutation(n)[:batch]
                    self._climb(weights, X[rows], targets[:, rows], sample_weight[rows], conflicts)
                else:
                    self._rebuild_last_layer(weights, X, targets, sample_weight)
                error = self._error(weights, X, targets, sample_weight)
                self._tracker.offer(weights, error)
                if error < best_error:
                    best, best_error, improved = [w.copy() for w in weights], error, True
            if best_error == 0 or (self.n_epochs is None and not improved):
                break
        weights = [w.copy() for w in best]
        self._climb(weights, X, targets, sample_weight, conflicts, on_step=self._tracker.offer)
        return weights

    def _conflicts(self, spec: Any) -> Optional[np.ndarray]:
        """`[j, d]` True iff flipping on feature `j` in a rule drops `d`
        (`None`: nothing; always for a first OR layer)."""
        if self.moves == "free" or self.first != "and":
            return None
        if self.moves == "consistent":
            return contradictions(spec)
        groups = _attribute_groups(spec)
        out = groups[:, None] == groups[None, :]
        np.fill_diagonal(out, False)
        return out

    # -- hill climbing --------------------------------------------------------

    def _climb(self, weights, X, targets, sample_weight, conflicts, on_step=None) -> None:
        """Steepest ascent on (X, targets): make the best improving flip
        until none is left (or `max_flips`); `on_step(weights, error)` after
        each flip."""
        flips = 0
        inputs = X.T
        error = self._error(weights, X, targets, sample_weight)
        while self.max_flips is None or flips < self.max_flips:
            values = nnf_forward(weights, X, self.first)
            errors = [self._flip_errors(weights, values, inputs, l, targets, sample_weight,
                                        conflicts if l == 0 else None)
                      for l in range(len(weights))]
            self._tracker.count(sum(w.size for w in weights) * X.shape[0])
            # Beck's order: the first layer node by node, the others edge source by source
            order = [errors[0].ravel()] + [e.T.ravel() for e in errors[1:]]
            flat = np.concatenate(order)
            i = int(np.argmin(flat))
            if not flat[i] < error:
                return
            for l, e in enumerate(errors):
                if i < e.size:
                    k, j = divmod(i, e.shape[1]) if l == 0 else divmod(i, e.shape[0])[::-1]
                    self._flip(weights, l, k, j, conflicts if l == 0 else None)
                    break
                i -= e.size
            error = float(flat.min())
            flips += 1
            if on_step is not None:
                on_step(weights, error)

    @staticmethod
    def _flip(weights, l, k, j, conflicts) -> None:
        w = weights[l]
        if not w[k, j] and conflicts is not None:
            w[k, conflicts[j]] = False
        w[k, j] = not w[k, j]

    def _flip_errors(self, weights, values, inputs, l, targets, sample_weight, conflicts) -> np.ndarray:
        """The error after flipping each edge of layer `l`, shaped like its
        weights: per node, the node's values under every flip of one of its
        edges, then the layers above."""
        w = weights[l]
        below = inputs if l == 0 else values[l - 1]
        is_and = _is_and(l, self.first)
        out = np.empty(w.shape)
        b = below.shape[1]
        step = max(1, self._CHUNK_CELLS // max(1, b * max(x.shape[0] for x in weights[l:])))
        for k in range(w.shape[0]):
            for lo in range(0, w.shape[1], step):
                cand = self._candidate_values(w[k], below, is_and, conflicts, slice(lo, lo + step))
                out[k, lo:lo + step] = self._errors_above(weights, values, l, k, cand, targets, sample_weight,
                                                          self.first)
        return out

    @staticmethod
    def _candidate_values(row, below, is_and, conflicts, js) -> np.ndarray:
        """`(candidates, rows)`: the node's values after flipping each
        edge `j` in `js` (and, for an AND node with `conflicts`, dropping
        the conditions `j` conflicts with)."""
        present = row[js]
        if is_and:
            false = (~below).astype(np.int32)
            count = row.astype(np.int32) @ false
            delta = np.where(present[:, None], -false[js], false[js])
            if conflicts is not None:
                dropped = conflicts[js] & row[None, :] & ~present[:, None]
                delta = delta - dropped.astype(np.int32) @ false
            return count[None, :] + delta == 0
        true = below.astype(np.int32)
        count = row.astype(np.int32) @ true
        delta = np.where(present[:, None], -true[js], true[js])
        return count[None, :] + delta > 0

    @staticmethod
    def _errors_above(weights, values, l, k, cand, targets, sample_weight, first="and") -> np.ndarray:
        """The weighted error for each candidate row `cand` of node `k` in
        layer `l`, propagated through the layers above."""
        layer = np.broadcast_to(values[l], (cand.shape[0],) + values[l].shape).copy()
        layer[:, k] = cand
        for up in range(l + 1, len(weights)):
            wi = weights[up].astype(np.int32)
            if _is_and(up, first):
                layer = wi @ (~layer).astype(np.int32) == 0
            else:
                layer = wi @ layer.astype(np.int32) > 0
        return ((layer != targets[None]) * sample_weight[None, None, :]).sum(axis=(1, 2))

    def _rebuild_last_layer(self, weights, X, targets, sample_weight) -> None:
        last = weights[-1]
        last[:] = False
        error = self._error(weights, X, targets, sample_weight)
        while True:
            best, best_at = error, None
            for k, j in zip(*np.nonzero(~last)):
                last[k, j] = True
                e = self._error(weights, X, targets, sample_weight)
                last[k, j] = False
                if e < best:
                    best, best_at = e, (k, j)
            if best_at is None:
                return
            last[best_at] = True
            error = best


class DeepGenetic(DeepRuleLearner):
    """A memetic algorithm over the weights of a `DeepModel`: a genetic
    algorithm whose children are improved by a few hill-climbing steps.

    The population: `population` networks from `init` (default
    `RandomInit()`), the first ones replaced by the starts in `seeds`
    (further `DeepInit`s, e.g. `RuleSetInit(Pypper())`). Each generation
    keeps the `elite` best and fills up with children: two parents by
    tournaments of `tournament`, crossed over, mutated, then improved by
    up to `local_flips` steps of `DeepLocalSearch`'s hill climbing on a
    random mini-batch of `batch_size` rows. Fitness is the weighted
    training error, ties broken by fewer edges. After `generations`, the
    best network is climbed on all rows (`polish`).

    Crossover aligns the parents first: a network's hidden nodes can be
    permuted without changing what it computes, so node *i* of one parent
    and node *i* of the other need not have anything in common. Layer by
    layer from the inputs up, the second parent's hidden nodes are matched
    to the first's by their values on the training data (fewest differing
    rows, Hungarian method) and reordered. Then, by `crossover`:

    - ``"nodes"`` (default): the child takes each node -- its row of
      incoming edges -- from either parent with probability 1/2;
    - ``"subtree"``: a random node x of a random hidden layer of the
      second parent comes with its cone (every node below it connected to
      it through edges of weight 1, down to the first layer); everything
      else is the first parent's. x keeps the inputs it was built on; in
      a network with one hidden layer that swaps one rule;
    - ``"mixed"``: one of the two at random, per child;
    - ``"none"``: the child is a copy of the first parent (the control:
      mutation and hill climbing alone).

    Mutation flips on average `mutations` random edges, with the
    moves of `moves` (see `DeepLocalSearch`)."""

    CROSSOVERS = ("nodes", "subtree", "mixed", "none")
    #: rows the alignment compares nodes on (a random sample beyond this)
    _ALIGN_ROWS = 2000

    def __init__(
        self,
        layers: Optional[Sequence[int]] = None,
        population: int = 20,
        generations: int = 20,
        elite: int = 2,
        tournament: int = 3,
        mutations: float = 2.0,
        local_flips: int = 5,
        batch_size: int = 50,
        moves: str = "consistent",
        crossover: str = "nodes",
        polish: bool = True,
        seeds: Sequence[DeepInit] = (),
        init: Optional[DeepInit] = None,
        first: str = "and",
        target_class: Optional[Any] = None,
        trace: bool = False,
        random_state: Optional[int] = 0,
    ):
        super().__init__(layers=layers, first=first, init=init, trace=trace, random_state=random_state,
                         target_class=target_class)
        if population < 2 or not 0 <= elite < population or tournament < 1:
            raise ValueError("population must be at least 2, elite below it, tournament at least 1")
        if generations < 0 or local_flips < 0 or mutations < 0 or batch_size < 1:
            raise ValueError("generations, local_flips and mutations must be at least 0, batch_size 1")
        if len(seeds) > population:
            raise ValueError(f"{len(seeds)} seeds for a population of {population}")
        if moves not in DeepLocalSearch.MOVES:
            raise ValueError(f"moves must be one of {DeepLocalSearch.MOVES}, got {moves!r}")
        if crossover not in self.CROSSOVERS:
            raise ValueError(f"crossover must be one of {self.CROSSOVERS}, got {crossover!r}")
        self.population = population
        self.generations = generations
        self.elite = elite
        self.tournament = tournament
        self.mutations = mutations
        self.local_flips = local_flips
        self.batch_size = batch_size
        self.moves = moves
        self.crossover = crossover
        self.polish = polish
        self.seeds = list(seeds)

    def _fit_weights(self, data, labels, X, targets, shapes, start, sample_weight, rng):
        climber = DeepLocalSearch(moves=self.moves, max_flips=self.local_flips, first=self.first)
        climber._tracker = self._tracker
        conflicts = climber._conflicts(data.spec)
        align_rows = (np.arange(len(X)) if len(X) <= self._ALIGN_ROWS
                      else rng.choice(len(X), self._ALIGN_ROWS, replace=False))
        starts = [seed.weights(data, labels, shapes, rng) for seed in self.seeds]
        if len(starts) < self.population:
            starts.append(start)
        while len(starts) < self.population:
            starts.append(self._initialize(data, labels, shapes, rng))
        pop = [self._individual([np.asarray(w, dtype=bool) for w in ws], X, targets, sample_weight)
               for ws in starts]
        n, batch = len(X), min(self.batch_size, len(X))
        for _ in range(self.generations):
            pop.sort(key=lambda ind: ind[1])
            if pop[0][1][0] == 0:
                break
            children = pop[:self.elite]
            while len(children) < self.population:
                a, b = self._select(pop, rng), self._select(pop, rng)
                child = self._crossover(a[0], b[0], X[align_rows], rng)
                self._mutate(child, conflicts, rng)
                if self.local_flips:
                    rows = rng.choice(n, batch, replace=False)
                    climber._climb(child, X[rows], targets[:, rows], sample_weight[rows], conflicts)
                children.append(self._individual(child, X, targets, sample_weight))
            pop = children
        best = min(pop, key=lambda ind: ind[1])[0]
        if self.polish:
            polisher = DeepLocalSearch(moves=self.moves, first=self.first)
            polisher._tracker = self._tracker
            polisher._climb(best, X, targets, sample_weight, conflicts, on_step=self._tracker.offer)
        return best

    def _individual(self, weights, X, targets, sample_weight):
        error = self._error(weights, X, targets, sample_weight)
        self._tracker.offer(weights, error)
        return weights, (error, sum(int(w.sum()) for w in weights))

    def _select(self, pop, rng):
        picks = rng.choice(len(pop), size=min(self.tournament, len(pop)), replace=False)
        return min((pop[i] for i in picks), key=lambda ind: ind[1])

    @staticmethod
    def _aligned(a, b, X, first="and"):
        """`b`'s weights with its hidden nodes reordered to match `a`'s."""
        from scipy.optimize import linear_sum_assignment

        b = [w.copy() for w in b]
        va, vb = nnf_forward(a, X, first), nnf_forward(b, X, first)
        for l in range(len(b) - 1):
            ai, bi = va[l].astype(np.int32), vb[l].astype(np.int32)
            differ = ai @ (1 - bi).T + (1 - ai) @ bi.T  # rows on which node i of a and node j of b differ
            _, order = linear_sum_assignment(differ)
            b[l] = b[l][order]
            b[l + 1] = b[l + 1][:, order]
            vb[l] = vb[l][order]
        return b

    def _crossover(self, a, b, X, rng):
        if self.crossover == "none":
            return [w.copy() for w in a]
        self._tracker.count(2 * X.shape[0])
        b = self._aligned(a, b, X, self.first)
        mode = self.crossover if self.crossover != "mixed" else ("nodes", "subtree")[rng.integers(2)]
        if mode == "subtree":
            from_a = [~m for m in self._cone(b, int(rng.integers(len(b) - 1)), rng)]
        else:
            from_a = [rng.random(w.shape[0]) < 0.5 for w in a]
        return [np.where(m[:, None], wa, wb) for wa, wb, m in zip(a, b, from_a)]

    @staticmethod
    def _cone(weights, layer, rng) -> List[np.ndarray]:
        """Per layer, the nodes of a random node x of `layer` and of its
        cone: everything below x connected to it through edges of weight
        1 (nothing above x)."""
        masks = [np.zeros(w.shape[0], dtype=bool) for w in weights]
        masks[layer][rng.integers(weights[layer].shape[0])] = True
        for l in range(layer, 0, -1):
            masks[l - 1] = weights[l][masks[l]].any(axis=0)
        return masks

    def _mutate(self, weights, conflicts, rng):
        sizes = np.array([w.size for w in weights], dtype=float)
        for _ in range(rng.poisson(self.mutations)):
            l = int(rng.choice(len(weights), p=sizes / sizes.sum()))
            k, j = int(rng.integers(weights[l].shape[0])), int(rng.integers(weights[l].shape[1]))
            DeepLocalSearch._flip(weights, l, k, j, conflicts if l == 0 else None)


class DeepSAT(DeepRuleLearner):
    """The weights of a `DeepModel` by SAT solving, after Seip, Fürnkranz et
    al. ("Towards SAT-based learning of NNF networks", SYNASC 2025; the
    long version adds MaxSAT): an exact fit of the training data. Boolean
    variables are the weights and, per training example, the value of
    every node; clauses tie each node's value to its inputs (an AND node is
    true iff no input with an edge is false, an OR node iff an input with
    an edge is true; one auxiliary variable per edge for the direction
    that needs a witness) and each output to the example's label. The
    inputs and labels are constants, so a rule node of the first layer is
    true iff it has no edge to a feature false in the example -- no
    auxiliary variables there -- and a negative example's output needs
    none either. Solved through PySAT (``pip install python-sat``, the
    extra ``sat``).

    `size_penalty` (lambda) makes the examples soft: an example the
    network gets wrong costs 1 instead of being forbidden, a unit of size
    (by `minimize`) costs lambda, and the objective is a regularized loss,
    ``errors + lambda * size`` (the 0-1 loss with an L0 penalty; 0: the
    errors alone; > 0 needs `minimize`). ``None`` (default) keeps every
    example hard, an exact fit. Every assignment is then a network, so the
    linear search has one after its first solver call and is anytime from
    the start; rows with the same features and different labels need no
    `conflicts` rule (each label counts, the network gets one of them).
    Merged identical rows count as often as they occur; `sample_weight` is
    not used. The solvers get integer costs in the same ratio (lambda as a
    fraction p/q: a unit of size costs p, an error q) -- with fractional
    ones RC2 gets lost in rounding remainders that never reach 0. With soft
    examples, a given `init` is the linear search's first solution: the
    solver tries its weights first (as phases) and looks only for cheaper
    networks, so the result is never worse than the start; without
    `init` (default) there is no start.

    `minimize` turns it into MaxSAT, by default still fitting the data
    exactly:
    ``"inputs"`` minimizes the number of conditions (the edges of the first
    layer, summed over its nodes), ``"weights"`` the number of edges -- the
    smallest network fitting the data. `maxsat` decides how:

    - ``"linear"`` (default): linear search ("SAT-UNSAT"). Solve, then
      demand fewer edges than the network found (a cardinality constraint,
      an incremental totalizer, the bound passed as an assumption so the
      solver keeps what it learned), and repeat until no smaller network
      exists -- then the last one is optimal. With soft examples a second
      totalizer counts the errors, and ``errors + lambda * size`` must
      drop below the last network's cost (for every number r of errors, at
      least r errors allow at most so many edges). An anytime algorithm: every
      step gives a smaller network; with `max_seconds` it returns the best
      found so far, and with `trace` every one is a `TracePoint` (effort:
      solver calls; `size`: the objective).
    - ``"rc2"``: RC2, core-guided: works up from lower bounds and has a
      network only at the optimum -- often faster to prove it, but with
      nothing to return before; no time limit, and no start.

    Identical training rows are merged (the encoding needs one copy); rows
    with identical features but different labels make the problem
    unsatisfiable (with hard examples), so `conflicts` decides:
    ``"majority"`` (default) keeps the most frequent label of such a group (ties: the first in sorted
    order), ``"raise"`` refuses. An unsatisfiable problem (the
    architecture too small for the data) raises `ValueError`. `max_seconds`
    limits the solving: plain SAT raises a `TimeoutError` if it has no
    answer by then, the linear search returns its best network (or raises
    if it has none). `solver` is any of PySAT's; by default Glucose 4
    (``"g4"``) for SAT and the linear search -- it stops when the time is
    up, where CaDiCaL (``"cadical195"``, the papers') can't be interrupted
    by PySAT, so with it the time limit is only checked between solver
    calls -- and CaDiCaL for RC2, which has no time limit anyway and is
    much faster with it (on the Seip formula 12 s against 136 s).
    `max_conflicts` limits each solver call to that many conflicts
    instead (not for RC2, nor CaDiCaL): the same result on every machine,
    where a time limit depends on its speed; a call that runs out ends
    the linear search with its best network, and plain SAT with a
    `TimeoutError`. With hard examples `init` is not used.

    `constraints` adds the `DataSpec`'s constraints as clauses on the
    weights of a first AND layer, independent of the examples: no rule
    takes two conditions that contradict each other (`contradictions`),
    nor one that another of its conditions implies (`implications`), which
    would change nothing -- the rules stay logically consistent, and
    equivalent solutions drop out of the search."""

    MINIMIZE = (None, "inputs", "weights")
    MAXSAT = ("linear", "rc2")
    CONFLICTS = ("majority", "raise")
    #: solvers PySAT can't interrupt: the time limit holds only between calls
    UNINTERRUPTIBLE = ("cadical103", "cadical153", "cadical195", "lingeling")

    def __init__(
        self,
        layers: Optional[Sequence[int]] = None,
        minimize: Optional[str] = None,
        maxsat: str = "linear",
        conflicts: str = "majority",
        size_penalty: Optional[float] = None,
        max_seconds: Optional[float] = None,
        max_conflicts: Optional[int] = None,
        solver: Optional[str] = None,
        constraints: bool = True,
        init: Optional[DeepInit] = None,
        first: str = "and",
        target_class: Optional[Any] = None,
        trace: bool = False,
        random_state: Optional[int] = 0,
    ):
        super().__init__(layers=layers, first=first, init=init, trace=trace, random_state=random_state,
                         target_class=target_class)
        self.init = init  # None: no start (the base class would put in a random one)
        if minimize not in self.MINIMIZE:
            raise ValueError(f"minimize must be one of {self.MINIMIZE}, got {minimize!r}")
        if conflicts not in self.CONFLICTS:
            raise ValueError(f"conflicts must be one of {self.CONFLICTS}, got {conflicts!r}")
        if maxsat not in self.MAXSAT:
            raise ValueError(f"maxsat must be one of {self.MAXSAT}, got {maxsat!r}")
        self.minimize = minimize
        self.maxsat = maxsat
        if size_penalty is not None and not size_penalty >= 0:
            raise ValueError(f"size_penalty must be >= 0 (or None: an exact fit), got {size_penalty!r}")
        if size_penalty and minimize is None:
            raise ValueError("size_penalty > 0 needs minimize ('weights' or 'inputs'): the size it penalizes")
        self.conflicts = conflicts
        self.size_penalty = size_penalty
        self.max_seconds = max_seconds
        self.max_conflicts = max_conflicts
        self.solver = solver
        self.constraints = constraints

    def _initialize(self, data, labels, shapes, rng):
        """The start: `init`'s weights, or none (all False) without one."""
        if self.init is None:
            return [np.zeros(shape, dtype=bool) for shape in shapes]
        return self.init.weights(data, labels, shapes, rng, self.first)

    def _costs(self) -> Tuple[int, int]:
        """Integer costs (size unit, error) in the ratio `size_penalty`
        (RC2 gets lost in rounding remainders with fractional ones); (1, 0)
        for an exact fit."""
        if self.size_penalty is None:
            return 1, 0
        ratio = Fraction(self.size_penalty).limit_denominator(10 ** 6)
        return ratio.numerator, ratio.denominator

    def _fit_weights(self, data, labels, X, targets, shapes, start, sample_weight, rng):
        size_cost, error_cost = self._costs()
        soft = self.size_penalty is not None

        def report(weights) -> int:
            """Offer a network to the tracker -- its training error on all
            rows, its size and (when optimizing) its cost -- and return the
            cost: ``size_cost * size + error_cost * errors`` (the errors by
            the network itself; the solver's example variables may
            undercount)."""
            out = nnf_forward(weights, X_all, self.first)[-1]
            error = float(((out != targets_all) * sample_weight).sum())
            size = int(sum(w.sum() for w in (weights if self.minimize == "weights" else weights[:1])))
            size = size if self.minimize is not None else 0
            cost = size_cost * size
            if soft:
                rows_out = nnf_forward(weights, X, self.first)[-1]
                cost += error_cost * sum(c for i, g in enumerate(groups) for column, c in g
                                         if (rows_out[:, i] != column).any())
            self._tracker.offer(weights, error, size if self.minimize is not None else None,
                                cost if (edges or examples) else None)
            return cost

        X_all, targets_all = X, targets
        if soft:
            X, groups = self._row_groups(X, targets)
        else:
            X, targets = self._consistent_rows(X, targets)
            groups = None
        pool, hard, edges, examples = self._encode(X, targets, shapes, groups)
        if soft and size_cost == 0:
            edges = []  # only the errors count
        if self.constraints and self.first == "and":
            excluded = contradictions(data.spec) | implications(data.spec)
            pairs = [(i, j) for i, j in zip(*np.nonzero(np.triu(excluded | excluded.T, 1)))]
            hard += [[-pool.weight(0, k, int(i)), -pool.weight(0, k, int(j))]
                     for k in range(shapes[0][0]) for i, j in pairs]
        if (edges or examples) and self.maxsat == "linear":
            given = start if soft and self.init is not None else None
            return self._linear_search(pool, hard, edges, examples, shapes, report, given)
        model = self._solve(pool, hard, edges, examples)
        weights = self._weights(model, pool, shapes)
        self._tracker.count(1)
        report(weights)
        return weights

    def _call(self, solver, assumptions, deadline):
        """One solver call within the limits: `None` if one ran out."""
        import threading

        limited = deadline is not None or self.max_conflicts is not None
        if not limited:
            return solver.solve(assumptions=assumptions)
        if self.max_conflicts is not None:
            solver.conf_budget(self.max_conflicts)
        timer = None
        if deadline is not None:
            timer = threading.Timer(max(0.0, deadline - time.perf_counter()), solver.interrupt)
            timer.start()
        sat = solver.solve_limited(assumptions=assumptions, expect_interrupt=deadline is not None)
        if timer is not None:
            timer.cancel()
            solver.clear_interrupt()
        return sat

    def _solver_name(self, method: str = "sat") -> str:
        if self.solver is not None:
            return self.solver
        return "cadical195" if method == "rc2" else "g4"

    @staticmethod
    def _weights(model, pool, shapes) -> List[np.ndarray]:
        """The weight matrices of a solver model (a list of literals)."""
        true = {lit for lit in model if lit > 0}
        return [np.array([[pool.weight(l, k, j) in true for j in range(s[1])] for k in range(s[0])], dtype=bool)
                for l, s in enumerate(shapes)]

    def _linear_search(self, pool, hard, edges, examples, shapes, report, start=None) -> List[np.ndarray]:
        """Linear search: ever cheaper networks, until none is left or the
        time is up; the best found. Each network is pruned to its backbone
        before its cost bounds the next step. A `start` network (soft
        examples only), pruned too, is the first solution: the solver tries
        its weights first (as phases) and looks only for cheaper ones."""
        from pysat.card import ITotalizer
        from pysat.solvers import Solver

        size_cost, error_cost = self._costs()
        error_lits = [-e for e, c in examples for _ in range(c)]  # a copy per merged example
        deadline = None if self.max_seconds is None else time.perf_counter() + self.max_seconds
        interruptible = self._solver_name() not in self.UNINTERRUPTIBLE
        best, out_of_limits = None, False
        totalizers: Dict[str, Any] = {}
        if edges:  # up front, as the search of an exact fit has always had it
            totalizers["edges"] = ITotalizer(lits=edges, ubound=len(edges), top_id=pool.top)
        top = totalizers["edges"].top_id if edges else pool.top
        with Solver(name=self._solver_name(),
                    bootstrap_with=hard + (totalizers["edges"].cnf.clauses if edges else [])) as solver:
            assumptions: List[int] = []

            def tighten(cost) -> bool:
                """Demand a cheaper network than `cost`; False if none can be."""
                nonlocal assumptions
                if examples and "errors" not in totalizers:  # counting up to what this cost allows
                    ub = min(len(error_lits), cost // error_cost + 1)
                    totalizers["errors"] = ITotalizer(lits=error_lits, ubound=ub, top_id=top)
                    solver.append_formula(totalizers["errors"].cnf.clauses)
                clauses = self._cheaper(cost, totalizers, len(edges), len(error_lits), size_cost, error_cost)
                if clauses is None:
                    return False
                # a bound on the edges alone stays an assumption: the solver keeps what it learned
                assumptions = [c[0] for c in clauses if len(c) == 1 and not examples]
                solver.append_formula([c for c in clauses if not (len(c) == 1 and not examples)])
                return True

            go = True
            if start is not None:  # its backbone: the same function, the cost of what is used
                best = pruned_weights([np.asarray(w, dtype=bool) for w in start])
                solver.set_phases([pool.weight(l, k, j) if w[k, j] else -pool.weight(l, k, j)
                                   for l, w in enumerate(best) for k, j in np.ndindex(*w.shape)])
                go = tighten(report(best))
            while go:
                if deadline is not None and time.perf_counter() >= deadline:
                    out_of_limits = True
                    break
                sat = self._call(solver, assumptions, deadline if interruptible else None)
                self._tracker.count(1)
                if not sat:  # None: a limit ran out; False: nothing cheaper, the last one is optimal
                    out_of_limits = sat is None
                    break
                # its backbone: dead edges count in the solver's size but change nothing, so the
                # next bound starts from the edges in use
                best = pruned_weights(self._weights(solver.get_model(), pool, shapes))
                go = tighten(report(best))
        for totalizer in totalizers.values():
            totalizer.delete()
        if best is None:
            if out_of_limits:
                raise TimeoutError("no network found within the limits "
                                   f"(max_seconds={self.max_seconds}, max_conflicts={self.max_conflicts})")
            raise ValueError("no network of this architecture fits the training data (unsatisfiable)")
        return best

    @staticmethod
    def _cheaper(cost, totalizers, n_edges, n_errors, size_cost, error_cost) -> Optional[List[List[int]]]:
        """Clauses for ``size_cost * size + error_cost * errors < cost``
        (integers): for every r, at least r errors allow at most
        ``(cost - 1 - error_cost * r) // size_cost`` edges (and are
        impossible where that is negative). `None` if nothing is cheaper."""
        clauses = []
        for r in range(n_errors + 1):
            rest = cost - 1 - error_cost * r
            premise = [] if r == 0 else [-totalizers["errors"].rhs[r - 1]]  # "at least r errors"
            if rest < 0:
                if not premise:
                    return None
                clauses.append(premise)
                break
            if n_edges and rest // size_cost < n_edges:
                clauses.append(premise + [-totalizers["edges"].rhs[rest // size_cost]])
        return clauses

    def _consistent_rows(self, X: np.ndarray, targets: np.ndarray):
        """One row per distinct feature vector, with its majority label."""
        rows, inverse = np.unique(X, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        codes = np.unique(targets.T, axis=0, return_inverse=True)[1].ravel()
        out = np.zeros((targets.shape[0], len(rows)), dtype=bool)
        for r in range(len(rows)):
            members = np.flatnonzero(inverse == r)
            found, counts = np.unique(codes[members], return_counts=True)
            if len(found) > 1 and self.conflicts == "raise":
                raise ValueError("the training data has identical rows with different labels -- "
                                 "no network fits it (conflicts='majority' keeps the majority label)")
            best = found[np.argmax(counts)]
            out[:, r] = targets[:, members[np.flatnonzero(codes[members] == best)[0]]]
        return rows, out

    @staticmethod
    def _row_groups(X: np.ndarray, targets: np.ndarray):
        """One row per distinct feature vector, and per row its labels (as
        target columns) with how often each occurs."""
        rows, inverse = np.unique(X, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        groups = []
        for r in range(len(rows)):
            columns, counts = np.unique(targets[:, inverse == r].T, axis=0, return_counts=True)
            groups.append([(column.astype(bool), int(c)) for column, c in zip(columns, counts)])
        return rows, groups

    def _encode(self, X, targets, shapes, groups=None):
        """The clauses for (X, targets): hard ones and what to minimize, the
        edge variables and the examples. Without `groups` every label is a
        constant (an exact fit); with them (`_row_groups`) the outputs are
        variables and each label one example variable, true only if the
        network gets it right, with its count."""
        pool = _VariablePool(shapes)
        hard: List[List[int]] = []
        examples: List[Tuple[int, int]] = []
        n_layers = len(shapes)
        for i in range(X.shape[0]):
            # over the constant inputs, an AND node is true iff it has no edge to a false
            # feature, an OR node iff it has one to a true feature
            deciding = np.flatnonzero(~X[i]) if self.first == "and" else np.flatnonzero(X[i])
            sign = -1 if self.first == "and" else 1  # the polarity of "the node holds" in its clauses
            below: Optional[List[int]] = None  # activation variables of the layer below
            for l, (n_nodes, n_below) in enumerate(shapes):
                last = l == n_layers - 1
                constant = last and groups is None
                acts = [] if constant else [pool.new() for _ in range(n_nodes)]
                for k in range(n_nodes):
                    ws = [pool.weight(l, k, j) for j in range(n_below)]
                    a = bool(targets[k, i]) if constant else acts[k]
                    if l == 0:
                        if a is True or a is False:
                            holds = a if self.first == "and" else not a
                            if holds:  # AND true / OR false: no edge to a deciding feature
                                hard += [[-ws[j]] for j in deciding]
                            else:
                                hard.append([ws[j] for j in deciding])
                        else:  # a <-> no edge to a false feature (AND) / an edge to a true one (OR)
                            hard += [[sign * a, -ws[j]] for j in deciding]
                            hard.append([-sign * a] + [ws[j] for j in deciding])
                    elif _is_and(l, self.first):
                        self._and(pool, hard, a, ws, below)
                    else:
                        self._or(pool, hard, a, ws, below)
                below = acts
            if groups is not None:  # below: the outputs
                for column, count in groups[i]:
                    e = pool.new()
                    hard += [[-e, o if t else -o] for o, t in zip(below, column)]
                    examples.append((e, count))
        edges: List[int] = []
        if self.minimize is not None:
            for l, (n_nodes, n_below) in enumerate(shapes if self.minimize == "weights" else shapes[:1]):
                edges += [pool.weight(l, k, j) for k in range(n_nodes) for j in range(n_below)]
        return pool, hard, edges, examples

    @staticmethod
    def _and(pool, hard, a, ws, xs):
        """a <-> AND_j (w_j -> x_j); `a` a variable or a constant bool."""
        if a is True or a is False:
            if a:
                hard += [[-w, x] for w, x in zip(ws, xs)]
            else:
                t = [pool.new() for _ in ws]
                hard += [[-tj, w] for tj, w in zip(t, ws)] + [[-tj, -x] for tj, x in zip(t, xs)]
                hard.append(t)
            return
        hard += [[-a, -w, x] for w, x in zip(ws, xs)]
        t = [pool.new() for _ in ws]
        hard += [[-tj, w] for tj, w in zip(t, ws)] + [[-tj, -x] for tj, x in zip(t, xs)]
        hard.append([a] + t)

    @staticmethod
    def _or(pool, hard, a, ws, xs):
        """a <-> OR_j (w_j and x_j); `a` a variable or a constant bool."""
        if a is True or a is False:
            if a:
                t = [pool.new() for _ in ws]
                hard += [[-tj, w] for tj, w in zip(t, ws)] + [[-tj, x] for tj, x in zip(t, xs)]
                hard.append(t)
            else:
                hard += [[-w, -x] for w, x in zip(ws, xs)]
            return
        hard += [[a, -w, -x] for w, x in zip(ws, xs)]
        t = [pool.new() for _ in ws]
        hard += [[-tj, w] for tj, w in zip(t, ws)] + [[-tj, x] for tj, x in zip(t, xs)]
        hard.append([-a] + t)

    def _solve(self, pool, hard, edges, examples) -> List[int]:
        try:
            from pysat.solvers import Solver
        except ImportError as e:  # pragma: no cover
            raise ImportError("DeepSAT needs PySAT: pip install python-sat") from e
        if edges or examples:
            from pysat.examples.rc2 import RC2
            from pysat.formula import WCNF

            wcnf = WCNF()
            wcnf.extend(hard)
            size_cost, error_cost = self._costs()
            for w in edges:
                wcnf.append([-w], weight=size_cost)
            for e, count in examples:
                wcnf.append([e], weight=error_cost * count)
            with RC2(wcnf, solver=self._solver_name("rc2")) as rc2:  # no time limit
                model = rc2.compute()
            if model is None:
                raise ValueError("no network of this architecture fits the training data (unsatisfiable)")
            return self._full_model(model, pool)
        with Solver(name=self._solver_name(), bootstrap_with=hard) as solver:
            interruptible = self._solver_name() not in self.UNINTERRUPTIBLE
            deadline = (time.perf_counter() + self.max_seconds
                        if self.max_seconds is not None and interruptible else None)
            sat = self._call(solver, [], deadline)
            if sat is None:
                raise TimeoutError("the SAT solver found no answer within its limits "
                                   f"(max_seconds={self.max_seconds}, max_conflicts={self.max_conflicts})")
            if not sat:
                raise ValueError("no network of this architecture fits the training data (unsatisfiable)")
            return self._full_model(solver.get_model(), pool)

    @staticmethod
    def _full_model(model, pool) -> List[int]:
        """The model as a list indexed by variable - 1 (unmentioned: false)."""
        out = [-(v + 1) for v in range(pool.top)]
        for lit in model:
            if abs(lit) <= pool.top:
                out[abs(lit) - 1] = lit
        return out


class _VariablePool:
    """The SAT variables: the weights first (numbered layer by layer), then
    whatever `new` hands out (activations, auxiliary variables)."""

    def __init__(self, shapes):
        self.offsets = np.cumsum([0] + [a * b for a, b in shapes])
        self.shapes = shapes
        self.top = int(self.offsets[-1])

    def weight(self, l: int, k: int, j: int) -> int:
        return int(self.offsets[l]) + k * self.shapes[l][1] + j + 1

    def new(self) -> int:
        self.top += 1
        return self.top


def contradictions(spec) -> np.ndarray:
    """`(n_features, n_features)` bool: `[i, j]` True iff features `i`
    and `j` can't both be true by `spec`'s constraints (`DataSpec.
    propagate`): two values of a nominal attribute, a feature and its
    negation, contradicting thresholds. A `DataSpec` without typed
    attributes has no constraints, so nothing contradicts there."""
    n = spec.n_features
    out = np.zeros((n, n), dtype=bool)
    if spec.constraints:
        for i in range(n):
            for j, value in spec.propagate({i: True}).items():
                if value is False:
                    out[i, j] = True
    return out | out.T


def implications(spec) -> np.ndarray:
    """`(n_features, n_features)` bool: `[i, j]` True iff feature `i`
    implies feature `j` (`i != j`) by `spec`'s constraints (`DataSpec.
    propagate`), e.g. ``x=a`` implies ``x!=b``, ``x>=5`` implies
    ``x>=3``."""
    n = spec.n_features
    out = np.zeros((n, n), dtype=bool)
    if spec.constraints:
        for i in range(n):
            for j, value in spec.propagate({i: True}).items():
                if value is True and j != i:
                    out[i, j] = True
    return out


def _attribute_groups(spec) -> np.ndarray:
    """Per feature, a number for the attribute it derives from (a feature
    without an attribute is its own)."""
    names: dict = {}
    out = np.empty(spec.n_features, dtype=np.int64)
    for i, fs in enumerate(spec.feature_specs):
        key = fs.attribute if getattr(fs, "attribute", None) is not None else ("feature", i)
        out[i] = names.setdefault(key, len(names))
    return out
