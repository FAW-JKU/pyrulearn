"""
pyrulearn.learners.deep
=======================

Learners of `pyrulearn.models.DeepModel`s: fully connected networks of
alternating AND and OR layers with 0/1 edge weights over the data's
Boolean features -- formulas in negation normal form, read as stacked
rules. They differ only in how they find the weights:

- `DeepLocalSearch` -- stochastic local search over edge flips.

`DeepRuleLearner` is their shared base: the architecture (`layers`), the
targets (one output node per class; for two classes one output for the
less frequent class, the other the default), the start weights
(`initial`, another `DeepModel`'s, for chaining learners) and wrapping
the weights into the model. A subclass implements `_fit_weights`.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence

import numpy as np

from ..combiners import CountVoteCombiner
from ..data import DataRepresentation
from ..models import DeepModel, MajorityClass
from .base import NativeRuleLearner, produces


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


def _node(row: np.ndarray, below: np.ndarray, is_and: bool) -> np.ndarray:
    """One node's values from its weight row and the layer below."""
    inputs = below[row]
    if is_and:
        return inputs.all(axis=0) if inputs.shape[0] else np.ones(below.shape[1], dtype=bool)
    return inputs.any(axis=0)


class DeepRuleLearner(NativeRuleLearner):
    """Base of the `DeepModel` learners. `layers` are the hidden layer
    sizes, AND, OR, ..., AND (an odd number of them; by default one AND
    layer of ``ceil(sqrt(n))`` nodes for *n* features, a DNF); the output
    layer has one OR node per class, or for two classes one for the less
    frequent class, the other being the default. A row no output covers
    gets the most frequent class, a row several cover the class with the
    most covering rules (`CountVoteCombiner`). `initial`, a `DeepModel`
    over the same features and of the same architecture, gives the start
    weights instead of the learner's own initialization."""

    def __init__(
        self,
        layers: Optional[Sequence[int]] = None,
        initial: Optional[DeepModel] = None,
        random_state: Optional[int] = 0,
    ):
        self.layers = None if layers is None else [int(n) for n in layers]
        self.initial = initial
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
        shape = DeepModel(data.spec, labels, layers)  # validates the architecture
        start = None
        if self.initial is not None:
            if [w.shape for w in self.initial.weights] != [w.shape for w in shape.weights]:
                raise ValueError(f"initial has layers {self.initial.layers} over "
                                 f"{self.initial.input_spec.n_features} features and "
                                 f"{len(self.initial.outputs)} outputs, not {layers} over "
                                 f"{data.spec.n_features} and {len(labels)}")
            start = [w.copy() for w in self.initial.weights]
        sample_weight = (np.ones(len(y)) if data.weights is None
                         else np.asarray(data.weights, dtype=float))
        rng = np.random.default_rng(self.random_state)
        weights = self._fit_weights(X, targets, [w.shape for w in shape.weights], start,
                                    sample_weight, rng)
        return DeepModel(data.spec, labels, weights=weights, default_prediction=MajorityClass(data),
                         combiner=CountVoteCombiner())

    def _fit_weights(self, X: np.ndarray, targets: np.ndarray, shapes: List[tuple],
                     start: Optional[List[np.ndarray]], sample_weight: np.ndarray,
                     rng: np.random.Generator) -> List[np.ndarray]:
        """The 0/1 weight matrices (bool, of `shapes`) for `X` and the
        output nodes' `targets` (`(n_outputs, n_samples)` bool), starting
        from `start` if given."""
        raise NotImplementedError


class DeepLocalSearch(DeepRuleLearner):
    """Stochastic local search over the edges of a `DeepModel` (after Beck
    & Fürnkranz). Starting from random weights (each edge present with
    probability `density`) or `initial`'s, each step flips one random
    edge, chosen uniformly among all edges of all layers. The flip is
    kept if it doesn't increase the training error -- the weighted number
    of output nodes that differ from their targets, summed over the rows
    -- and otherwise kept with probability `noise`, a random-walk step
    out of local optima. After `max_flips` steps, the best weights seen
    are kept; `restarts` further runs start again from random weights
    (not from `initial`) and the best run wins.

    Only the flipped node and the layers above it are evaluated again, so
    a step costs one node's values plus the matrix products above it."""

    def __init__(
        self,
        layers: Optional[Sequence[int]] = None,
        max_flips: int = 10000,
        noise: float = 0.01,
        density: float = 0.1,
        restarts: int = 0,
        initial: Optional[DeepModel] = None,
        random_state: Optional[int] = 0,
    ):
        super().__init__(layers=layers, initial=initial, random_state=random_state)
        if max_flips < 0 or restarts < 0:
            raise ValueError("max_flips and restarts must be at least 0")
        if not (0.0 <= noise <= 1.0 and 0.0 <= density <= 1.0):
            raise ValueError("noise and density are probabilities")
        self.max_flips = max_flips
        self.noise = noise
        self.density = density
        self.restarts = restarts

    def _fit_weights(self, X, targets, shapes, start, sample_weight, rng):
        best, best_error = None, math.inf
        for run in range(self.restarts + 1):
            weights = (start if run == 0 and start is not None
                       else [rng.random(s) < self.density for s in shapes])
            weights, error = self._search([w.copy() for w in weights], X, targets, sample_weight, rng)
            if error < best_error:
                best, best_error = weights, error
        return best

    def _search(self, weights, X, targets, sample_weight, rng):
        def error_of(out: np.ndarray) -> float:
            return float(((out != targets) * sample_weight).sum())

        values = nnf_forward(weights, X)
        error = error_of(values[-1])
        best, best_error = [w.copy() for w in weights], error
        sizes = np.array([w.size for w in weights], dtype=float)
        if sizes.sum() == 0:
            return best, best_error
        layer_p = sizes / sizes.sum()
        inputs = X.T
        for _ in range(self.max_flips):
            if best_error == 0:
                break
            layer = int(rng.choice(len(weights), p=layer_p))
            w = weights[layer]
            i, j = int(rng.integers(w.shape[0])), int(rng.integers(w.shape[1]))
            w[i, j] = not w[i, j]
            below = inputs if layer == 0 else values[layer - 1]
            new_values = values[:layer] + [values[layer].copy()]
            new_values[layer][i] = _node(w[i], below, layer % 2 == 0)
            for up in range(layer + 1, len(weights)):
                new_values.append(_layer(weights[up], new_values[-1], up % 2 == 0))
            new_error = error_of(new_values[-1])
            if new_error <= error or rng.random() < self.noise:
                values, error = new_values, new_error
                if error < best_error:
                    best, best_error = [x.copy() for x in weights], error
            else:
                w[i, j] = not w[i, j]
        return best, best_error
