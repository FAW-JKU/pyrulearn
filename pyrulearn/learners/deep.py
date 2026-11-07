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
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from ..combiners import CountVoteCombiner
from ..data import DataRepresentation
from ..models import DeepModel, MajorityClass, RuleModel
from .base import NativeRuleLearner, RuleLearner, produces


def nnf_forward(weights: Sequence[np.ndarray], X: np.ndarray) -> List[np.ndarray]:
    """The value of every node for every row of the Boolean matrix `X`
    (`(n_samples, n_features)`), one `(n_nodes, n_samples)` bool array per
    layer of `weights` (AND, OR, AND, ..., OR): an AND node holds where
    none of its inputs is false, an OR node where one is true."""
    values: List[np.ndarray] = []
    below = X.T
    for i, w in enumerate(weights):
        values.append(_layer(w, below, i % 2 == 0))
        below = values[-1]
    return values


def _layer(w: np.ndarray, below: np.ndarray, is_and: bool) -> np.ndarray:
    wi = w.astype(np.int32)
    if is_and:
        return wi @ (~below).astype(np.int32) == 0
    return wi @ below.astype(np.int32) > 0


class DeepInit(ABC):
    """The start weights of a `DeepRuleLearner`."""

    @abstractmethod
    def weights(self, data: DataRepresentation, labels: Sequence[Any], shapes: List[tuple],
                rng: np.random.Generator) -> List[np.ndarray]:
        """Bool matrices of `shapes` (one per layer above the inputs, the
        output layer's rows in the order of `labels`) for `data`."""


class RandomInit(DeepInit):
    """Random rules (Beck & Fürnkranz's start): a rule of the first layer
    takes each attribute with probability ``avg_rule_length /
    n_attributes``, with one random feature of it, drawn again while it
    has fewer than `min_rule_length` conditions (Beck: 0 -- but two empty
    rules are true on every row and keep an output OR true: no single flip
    changes that); a node of a layer in between gets one edge to a random
    node above it, and every edge with probability `init_prob`; a single
    output node is connected to every node below, several divide the nodes
    below among them (round-robin), so that the classes don't start out
    with the same rules. With `consistent`, a rule skips a feature that
    contradicts one it already has by the `DataSpec`'s constraints
    (`contradictions`); Beck's start is ``consistent=False,
    min_rule_length=0``."""

    def __init__(self, avg_rule_length: float = 3, min_rule_length: int = 1, init_prob: float = 0.2,
                 consistent: bool = True):
        if not 0.0 <= init_prob <= 1.0:
            raise ValueError("init_prob is a probability")
        self.avg_rule_length = avg_rule_length
        self.min_rule_length = min_rule_length
        self.init_prob = init_prob
        self.consistent = consistent

    def weights(self, data, labels, shapes, rng):
        groups = _attribute_groups(data.spec)
        if self.consistent:
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
                        if not (self.consistent and (apart[j] & weights[0][k]).any()):
                            weights[0][k, j] = True
                if weights[0][k].sum() >= least:
                    break
        for w in weights[1:-1]:
            for j in range(w.shape[1]):  # every node gets at least one edge upwards
                w[rng.integers(w.shape[0]), j] = True
                w[:, j] |= rng.random(w.shape[0]) < self.init_prob
        last = weights[-1]
        if last.shape[0] == 1:
            last[:] = True
        else:
            last[np.arange(last.shape[1]) % last.shape[0], np.arange(last.shape[1])] = True
        return weights


class ModelInit(DeepInit):
    """The weights of `model`, a `DeepModel` of the same architecture with
    the same outputs, its inputs matched to the data's by name."""

    def __init__(self, model: DeepModel):
        self.model = model

    def weights(self, data, labels, shapes, rng):
        model = self.model.remap(data.spec)
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
    (the default class of a two-class problem) are left out."""

    def __init__(self, rules: Union[RuleModel, RuleLearner], fill: Optional[DeepInit] = None):
        self.rules = rules
        self.fill = fill if fill is not None else RandomInit()

    def weights(self, data, labels, shapes, rng):
        model = self.rules.fit(data) if isinstance(self.rules, RuleLearner) else self.rules
        bodies = [[r for r in model.rules if r.target == label] for label in labels]
        n_rules = sum(len(b) for b in bodies)
        if shapes[0][0] < n_rules:
            raise ValueError(f"{n_rules} rules for the outputs, but a first layer of {shapes[0][0]} nodes")
        if any(s[0] < len(labels) for s in shapes[1:-1]):
            raise ValueError(f"every layer above the first needs a node per output ({len(labels)})")
        weights = self.fill.weights(data, labels, shapes, rng)
        rule_nodes: List[List[int]] = []
        node = 0
        for rules in bodies:
            rule_nodes.append(list(range(node, node + len(rules))))
            for r in rules:
                weights[0][node] = False
                for lit in r.conditions:
                    weights[0][node, data.spec.feature_index(r.dataspec.feature_name(lit.feature))] = True
                node += 1
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
    layer of ``ceil(sqrt(n))`` nodes for *n* features, a DNF); the output
    layer has one OR node per class, or for two classes one for the less
    frequent class, the other being the default. With more classes, a row
    no output covers gets the most frequent class, a row several cover the
    class with the most covering rules (`CountVoteCombiner`). `init` (a
    `DeepInit`, default `RandomInit()`) gives the start weights, through
    `_initialize`, which a subclass may override."""

    def __init__(
        self,
        layers: Optional[Sequence[int]] = None,
        init: Optional[DeepInit] = None,
        random_state: Optional[int] = 0,
    ):
        self.layers = None if layers is None else [int(n) for n in layers]
        self.init = init if init is not None else RandomInit()
        self.random_state = random_state

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
        labels = [classes[int(np.argmin(counts))]] if len(classes) == 2 else classes
        targets = np.vstack([y == c for c in labels])
        X = np.asarray(data.X, dtype=bool)
        layers = (self.layers if self.layers is not None
                  else [math.ceil(math.sqrt(data.spec.n_features))])
        shapes = [w.shape for w in DeepModel(data.spec, labels, layers).weights]  # validates them
        sample_weight = (np.ones(len(y)) if data.weights is None
                         else np.asarray(data.weights, dtype=float))
        rng = np.random.default_rng(self.random_state)
        start = self._initialize(data, labels, shapes, rng)
        if [w.shape for w in start] != shapes:
            raise ValueError(f"the start weights have shapes {[w.shape for w in start]}, not {shapes}")
        weights = self._fit_weights(X, targets, shapes, [np.asarray(w, dtype=bool) for w in start],
                                    sample_weight, rng, data.spec)
        default = (next(c for c in classes if c != labels[0]) if len(classes) == 2
                   else MajorityClass(data))
        return DeepModel(data.spec, labels, weights=weights, default_prediction=default,
                         combiner=CountVoteCombiner())

    def _initialize(self, data: DataRepresentation, labels: Sequence[Any], shapes: List[tuple],
                    rng: np.random.Generator) -> List[np.ndarray]:
        """The start weights: `init`'s."""
        return self.init.weights(data, labels, shapes, rng)

    def _fit_weights(self, X: np.ndarray, targets: np.ndarray, shapes: List[tuple],
                     start: List[np.ndarray], sample_weight: np.ndarray,
                     rng: np.random.Generator, spec: Any) -> List[np.ndarray]:
        """The 0/1 weight matrices (bool, of `shapes`) for `X` (over the
        features of `spec`) and the output nodes' `targets`
        (`(n_outputs, n_samples)` bool), starting from `start`."""
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
        random_state: Optional[int] = 0,
    ):
        super().__init__(layers=layers, init=init, random_state=random_state)
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

    def _fit_weights(self, X, targets, shapes, start, sample_weight, rng, spec):
        conflicts = self._conflicts(spec)
        weights = start
        n = X.shape[0]
        batch = n if self.batch_size is None else min(self.batch_size, n)
        n_batches = n // batch
        best = [w.copy() for w in weights]
        best_error = self._error(weights, X, targets, sample_weight)
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
                if error < best_error:
                    best, best_error, improved = [w.copy() for w in weights], error, True
            if best_error == 0 or (self.n_epochs is None and not improved):
                break
        weights = [w.copy() for w in best]
        self._climb(weights, X, targets, sample_weight, conflicts)
        return weights

    @staticmethod
    def _error(weights, X, targets, sample_weight) -> float:
        out = nnf_forward(weights, X)[-1]
        return float(((out != targets) * sample_weight).sum())

    def _conflicts(self, spec: Any) -> Optional[np.ndarray]:
        """`[j, d]` True iff flipping on feature `j` in a rule drops `d`."""
        if self.moves == "free":
            return None
        if self.moves == "consistent":
            return contradictions(spec)
        groups = _attribute_groups(spec)
        out = groups[:, None] == groups[None, :]
        np.fill_diagonal(out, False)
        return out

    # -- hill climbing --------------------------------------------------------

    def _climb(self, weights, X, targets, sample_weight, conflicts) -> None:
        """Steepest ascent on (X, targets): make the best improving flip
        until none is left (or `max_flips`)."""
        flips = 0
        inputs = X.T
        error = self._error(weights, X, targets, sample_weight)
        while self.max_flips is None or flips < self.max_flips:
            values = nnf_forward(weights, X)
            errors = [self._flip_errors(weights, values, inputs, l, targets, sample_weight,
                                        conflicts if l == 0 else None)
                      for l in range(len(weights))]
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
        is_and = l % 2 == 0
        out = np.empty(w.shape)
        b = below.shape[1]
        step = max(1, self._CHUNK_CELLS // max(1, b * max(x.shape[0] for x in weights[l:])))
        for k in range(w.shape[0]):
            for lo in range(0, w.shape[1], step):
                cand = self._candidate_values(w[k], below, is_and, conflicts, slice(lo, lo + step))
                out[k, lo:lo + step] = self._errors_above(weights, values, l, k, cand, targets, sample_weight)
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
    def _errors_above(weights, values, l, k, cand, targets, sample_weight) -> np.ndarray:
        """The weighted error for each candidate row `cand` of node `k` in
        layer `l`, propagated through the layers above."""
        layer = np.broadcast_to(values[l], (cand.shape[0],) + values[l].shape).copy()
        layer[:, k] = cand
        for up in range(l + 1, len(weights)):
            wi = weights[up].astype(np.int32)
            if up % 2 == 0:
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


def _attribute_groups(spec) -> np.ndarray:
    """Per feature, a number for the attribute it derives from (a feature
    without an attribute is its own)."""
    names: dict = {}
    out = np.empty(spec.n_features, dtype=np.int64)
    for i, fs in enumerate(spec.feature_specs):
        key = fs.attribute if getattr(fs, "attribute", None) is not None else ("feature", i)
        out[i] = names.setdefault(key, len(names))
    return out
