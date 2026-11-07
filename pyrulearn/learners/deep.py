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
import time
from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple, Optional, Sequence, Union

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


class TracePoint(NamedTuple):
    """One new best network during a fit (`DeepRuleLearner(trace=True)`):
    the effort so far in candidate-rows (one candidate network evaluated
    on one row; a hill-climbing step on a batch counts every flip it
    evaluates on every row of the batch), the seconds so far, the weighted
    training error and the weights."""

    evaluations: int
    seconds: float
    train_error: float
    weights: List[np.ndarray]


class _Tracker:
    """Counts a fit's effort and, with `keep`, records each new best."""

    def __init__(self, keep: bool):
        self.keep = keep
        self.evaluations = 0
        self.start = time.perf_counter()
        self.best = math.inf
        self.points: List[TracePoint] = []

    def count(self, n: int) -> None:
        self.evaluations += int(n)

    def offer(self, weights: Sequence[np.ndarray], error: float) -> None:
        """`weights` reach the training error `error`: a new point if best."""
        if error < self.best:
            self.best = error
            if self.keep:
                self.points.append(TracePoint(self.evaluations, time.perf_counter() - self.start, error,
                                              [w.copy() for w in weights]))


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
    `_initialize`, which a subclass may override.

    The learners are anytime algorithms: with `trace`, the model's `trace`
    lists every new best network on the training set during the fit
    (`TracePoint`s: effort, seconds, training error, weights), and
    `DeepModel.trace_model(i)` turns one into a model, e.g. to measure
    the test accuracy along the run."""

    def __init__(
        self,
        layers: Optional[Sequence[int]] = None,
        init: Optional[DeepInit] = None,
        trace: bool = False,
        random_state: Optional[int] = 0,
    ):
        self.layers = None if layers is None else [int(n) for n in layers]
        self.init = init if init is not None else RandomInit()
        self.trace = trace
        self.random_state = random_state
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
        labels = [classes[int(np.argmin(counts))]] if len(classes) == 2 else classes
        targets = np.vstack([y == c for c in labels])
        X = np.asarray(data.X, dtype=bool)
        layers = (self.layers if self.layers is not None
                  else [math.ceil(math.sqrt(data.spec.n_features))])
        shapes = [w.shape for w in DeepModel(data.spec, labels, layers).weights]  # validates them
        sample_weight = (np.ones(len(y)) if data.weights is None
                         else np.asarray(data.weights, dtype=float))
        rng = np.random.default_rng(self.random_state)
        self._tracker = _Tracker(self.trace)
        start = self._initialize(data, labels, shapes, rng)
        if [w.shape for w in start] != shapes:
            raise ValueError(f"the start weights have shapes {[w.shape for w in start]}, not {shapes}")
        weights = self._fit_weights(data, labels, X, targets, shapes,
                                    [np.asarray(w, dtype=bool) for w in start], sample_weight, rng)
        default = (next(c for c in classes if c != labels[0]) if len(classes) == 2
                   else MajorityClass(data))
        model = DeepModel(data.spec, labels, weights=weights, default_prediction=default,
                          combiner=CountVoteCombiner())
        model.trace = self._tracker.points if self.trace else None
        return model

    def _error(self, weights, X, targets, sample_weight) -> float:
        """The weighted training error of `weights` on (X, targets),
        counted as one candidate on every row."""
        self._tracker.count(X.shape[0])
        out = nnf_forward(weights, X)[-1]
        return float(((out != targets) * sample_weight).sum())

    def _initialize(self, data: DataRepresentation, labels: Sequence[Any], shapes: List[tuple],
                    rng: np.random.Generator) -> List[np.ndarray]:
        """The start weights: `init`'s."""
        return self.init.weights(data, labels, shapes, rng)

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
        trace: bool = False,
        random_state: Optional[int] = 0,
    ):
        super().__init__(layers=layers, init=init, trace=trace, random_state=random_state)
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

    def _climb(self, weights, X, targets, sample_weight, conflicts, on_step=None) -> None:
        """Steepest ascent on (X, targets): make the best improving flip
        until none is left (or `max_flips`); `on_step(weights, error)` after
        each flip."""
        flips = 0
        inputs = X.T
        error = self._error(weights, X, targets, sample_weight)
        while self.max_flips is None or flips < self.max_flips:
            values = nnf_forward(weights, X)
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
        trace: bool = False,
        random_state: Optional[int] = 0,
    ):
        super().__init__(layers=layers, init=init, trace=trace, random_state=random_state)
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
        climber = DeepLocalSearch(moves=self.moves, max_flips=self.local_flips)
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
            polisher = DeepLocalSearch(moves=self.moves)
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
    def _aligned(a, b, X):
        """`b`'s weights with its hidden nodes reordered to match `a`'s."""
        from scipy.optimize import linear_sum_assignment

        b = [w.copy() for w in b]
        va, vb = nnf_forward(a, X), nnf_forward(b, X)
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
        b = self._aligned(a, b, X)
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
