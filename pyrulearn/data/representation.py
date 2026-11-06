"""
pyrulearn.data.representation
=================================

`DataRepresentation`: a concrete encoding of a dataset bound to a
`pyrulearn.data.spec.DataSpec`. `DataSpec` itself is pure schema (feature
names/order, typed attributes, constraints, missing-value policy) -- no
data; a `DataRepresentation` is where the actual values live, always
pointing back at the `DataSpec` it was built against.

`HorizontalBitmapRepresentation` is the canonical one: a `numpy.packbits`-
packed Boolean matrix (plus labels). `NListRepresentation` is the same
data as a PPC-tree / N-list vertical index (the one Huynh,
Fürnkranz & Beck's LORD builds); `SparseDataRepresentation` is a `scipy`
CSR/CSC encoding -- the N-list without the prefix tree. All three sit
behind the identical interface, so every `pyrulearn.learners.seco` learner runs
on any of them. All point at an ordinary `DataSpec`; further encodings
some future algorithm wants directly are further `DataRepresentation`
subclasses.

Every representation implements the primitives the rule-learning code
is written against: `coverage(rule)` (the boolean covered-rows vector)
and `features_of(row)` (a seed example's True features) for one-shot
use outside a search (prediction, pruning, `RuleStats.from_rule` --
`Rule.covers_data`/`Rule.covers_data_packed` just forward to
`data.coverage(self)`); and, for `pyrulearn.learners.seco.BeamSearch`/
`HillClimbing`'s incremental search fast path, `initial_cover`,
`refine_cover`, `cover_counts`, and `cover_rows`, built around an
opaque, representation-specific "cover handle" so each encoding can
make refining a rule by one more literal as cheap as its own storage
allows (see `NListRepresentation`'s own docstring for the interesting
case).

Because a `DataSpec` no longer carries any data of its own, the same
`DataSpec` instance can be shared verbatim across multiple
`DataRepresentation`s built from it (e.g. a train split and a test
split), guaranteeing identical feature indexing between them by
construction rather than by convention.
"""

from __future__ import annotations

import copy
import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

import numpy as np

from .spec import DataSpec

if TYPE_CHECKING:
    from ..rule import Rule


class DataRepresentation(ABC):
    """Base for a concrete encoding of a dataset bound to a `DataSpec`.

    Two abstract primitives make every representation a drop-in for the
    rule-learning machinery, regardless of how it stores the data
    internally:

    - `coverage(rule)` -- the boolean "which rows does this rule cover"
      vector every heuristic, search and covering loop is ultimately
      defined over (`RuleStats.from_rule` does the ``& example_mask &
      (y == class)`` counting on top of it). `Rule.covers_data` /
      `Rule.covers_data_packed` just forward here.
    - `features_of(row)` -- the feature indices that are True in one row,
      needed to seed a search from an example (`pyrulearn.learners.seco.
      SeedExample`).

    **Row weights.** `weights` (``None``: every row counts 1) are
    non-negative per-row weights that belong to the data -- e.g. how
    often a row was observed, or its importance -- given to the
    constructor or set with `with_weights`. Every count the learners
    make is then a sum of weights: the searches' `cover_counts`,
    `pyrulearn.evaluation.RuleStats.from_rule`, and a model's
    `evaluate`, and so the training stats rules store and print. The
    `example_mask` a search is restricted to may itself be a weight
    vector instead of a boolean mask (weighted covering, see
    `pyrulearn.learners.seco.WeightedCovering`); the two multiply
    (`scope`). Without weights the counts stay plain integers.
    """

    def __init__(self, spec: DataSpec):
        self.spec = spec
        self.y: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    # -- row weights -------------------------------------------------------

    def _set_weights(self, weights: Optional[Any]) -> None:
        if weights is None:
            self.weights = None
            return
        w = np.asarray(weights, dtype=float)
        if w.shape != (self.n_samples,):
            raise ValueError(f"weights must have shape ({self.n_samples},), got {w.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
        self.weights = w

    def with_weights(self, weights: Optional[Any]) -> "DataRepresentation":
        """The same data with row weights `weights` (``None`` removes
        them) -- a shallow copy sharing this representation's storage and
        index, so it costs no rebuild."""
        new = copy.copy(self)
        new._set_weights(weights)
        return new

    def scope(self, example_mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """``(support, weights)`` of a search restricted to `example_mask`
        -- a boolean mask, a non-negative weight vector, or ``None`` (all
        rows) -- on this data. `weights` is the effective per-row weight
        (this data's `weights` times `example_mask`), or ``None`` when
        every row in scope counts exactly 1 (no data weights, and no mask
        or a boolean one); `support` is the boolean mask of the rows in
        scope (weight > 0), or ``None`` for every row. Representations
        count with `weights` if given and plain row counts otherwise."""
        mask = None if example_mask is None else np.asarray(example_mask)
        if self.weights is None and (mask is None or mask.dtype == bool):
            return mask, None
        w = np.ones(self.n_samples) if self.weights is None else self.weights
        if mask is not None:
            w = w * mask.astype(float)
        return w > 0, w

    def scope_weights(self, example_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """The effective per-row weight of a search restricted to
        `example_mask`, always as a float array (1 per in-scope row
        without weights) -- for callers that just want weighted sums."""
        support, w = self.scope(example_mask)
        if w is not None:
            return w
        return np.ones(self.n_samples) if support is None else support.astype(float)

    @property
    @abstractmethod
    def n_samples(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def coverage(self, rule: "Rule") -> np.ndarray:
        """Boolean array of shape ``(n_samples,)`` -- ``[i]`` True iff
        every one of `rule`'s conditions holds for row ``i``."""
        raise NotImplementedError

    @abstractmethod
    def features_of(self, row: int) -> np.ndarray:
        """The integer feature indices that are True in `row` (ascending)
        -- the "open" features a rule seeded on that row can still add."""
        raise NotImplementedError

    # -- incremental coverage (search fast path) -----------------------

    @abstractmethod
    def initial_cover(self, example_mask: Optional[np.ndarray] = None) -> Any:
        """A representation-specific, opaque "cover handle" for the empty
        rule, restricted to `example_mask` if given. Masking a search's
        scope (a SeCo covering loop's shrinking "remaining" set, an
        IREP/RIPPER grow/prune split) is just the zeroth refinement --
        applying it once here, rather than re-applying it on every
        downstream `cover_counts` call the way `RuleStats.from_rule`'s
        `example_mask` does, is what lets a representation that keeps a
        compressed handle (`NListRepresentation`) narrow its own index
        once instead of masking a full row vector at every search node.
        """
        raise NotImplementedError

    @abstractmethod
    def refine_cover(self, handle: Any, feature: int) -> Any:
        """The handle for `handle`'s rule with one more literal
        (`feature`) added -- i.e. `handle` narrowed to rows/nodes where
        `feature` also holds. `pyrulearn.learners.seco.BeamSearch`/`HillClimbing`
        call this once per specialization step instead of recomputing
        `coverage` for the whole grown rule from scratch, so each
        representation can make refining as cheap as its own storage
        allows (see each subclass's implementation)."""
        raise NotImplementedError

    @abstractmethod
    def cover_counts(self, handle: Any, positive_class: Any) -> Tuple[int, int, int, int]:
        """``(tp, fp, fn, tn)`` for the rows covered by `handle`, against
        `self.y`, treating `positive_class` as positive. A plain tuple,
        not a `RuleStats` -- `length` is rule/search context, not
        something a coverage handle has, and `RuleStats` lives in
        `pyrulearn.heuristics`, which imports this module (a
        `RuleStats`-returning method here would be a circular import)."""
        raise NotImplementedError

    @abstractmethod
    def cover_rows(self, handle: Any) -> np.ndarray:
        """Boolean array of shape ``(n_samples,)``: which rows `handle`
        covers -- `coverage(rule)` restated for a handle instead of a
        `Rule`. Materializing the full array is usually only worth doing
        once a search has settled on a final handle, not at every node,
        which is why this is a separate call from `cover_counts` rather
        than something every refinement step pays for."""
        raise NotImplementedError

    @abstractmethod
    def batch_cover_counts(
        self, handle: Any, positive_class: Any, feature_indices: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(tp, fp, keeps_all)`` arrays, one entry per feature in
        `feature_indices`: the counts of `handle` refined by each feature,
        and whether that refinement drops no covered row with a nonzero
        weight. What every rule search counts a node's children with
        (`pyrulearn.learners.seco.count_open_children`); each
        representation batches it as its storage allows."""
        raise NotImplementedError

    def batch_cover_sums(self, handle: Any, values: np.ndarray, feature_indices: Sequence[int]) -> np.ndarray:
        """``values`` (``(n_samples, m)``, any per-row numbers -- e.g. a
        boosting loss's gradients) summed over the rows covered by
        `handle` refined by each feature in `feature_indices`: an ``(m,
        len(feature_indices))`` array. The generalization of class
        counting a boosting learner's search needs. This default refines
        and sums once per feature, so it works on every representation;
        a representation can override it with a batched version
        (`HorizontalBitmapRepresentation` does, as one matrix product)."""
        values = np.asarray(values, dtype=float)
        out = np.zeros((values.shape[1], len(feature_indices)))
        for j, f in enumerate(feature_indices):
            out[:, j] = values[self.cover_rows(self.refine_cover(handle, f))].sum(axis=0)
        return out

    # -- negation-feature toggling -----------------------------------

    def without_negations(self) -> "DataRepresentation":
        """A representation of the same concrete type over this data's
        feature space without negation features -- every ``not f`` /
        ``x != v`` feature dropped, via `DataSpec.without_negations`
        (a numeric attribute keeps both ``x >= t`` and ``x < t``). Returns `self` unchanged if there
        are no negation features. No data is recomputed: the surviving
        columns are exactly the columns that were already there."""
        new_spec, keep = self.spec.without_negations()
        if new_spec is self.spec:
            return self
        return self._reindexed(new_spec, keep, None)

    def with_negations(self) -> "DataRepresentation":
        """A representation of the same concrete type with a negation
        feature added for every test that lacks one, via
        `DataSpec.with_negations`. The added columns are synthesized as
        the logical complement of their positive partner -- see that
        method for the (missing-value) assumption that carries. Returns
        `self` unchanged if every feature already has its negation.

        For `SparseDataRepresentation` / `NListRepresentation` the added
        columns are near-50%-dense, so the result is much less compact
        than the input -- correct, but usually not what you want; prefer
        building the negated version from the raw data."""
        new_spec, source, is_complement = self.spec.with_negations()
        if new_spec.n_features == self.spec.n_features and not is_complement.any():
            return self
        return self._reindexed(new_spec, source, is_complement)

    def _reindexed(
        self, new_spec: DataSpec, source: np.ndarray, is_complement: Optional[np.ndarray]
    ) -> "DataRepresentation":
        """Build a representation of the same concrete type over
        `new_spec`, where new column ``j`` is old column ``source[j]``,
        logically negated where ``is_complement[j]`` (``is_complement is
        None`` -> plain column selection, no negation). Subclasses
        override with a storage-appropriate implementation; this fallback
        goes through the dense `X`."""
        cols = self.X[:, source]
        if is_complement is not None:
            cols = np.where(is_complement, ~cols, cols)
        return type(self)(new_spec, cols, self.y, weights=self.weights)

    # -- row subsetting ---------------------------------------------------

    def select_rows(self, mask: np.ndarray) -> "DataRepresentation":
        """A representation of the same concrete type over just the rows
        `mask` selects (a boolean mask or an integer index array),
        rebuilt from scratch -- `NListRepresentation` re-derives its
        tree, etc. The `DataSpec` (feature space, so rule feature indices)
        is unchanged, so rules learned on the subset still apply to the
        full data. Used to train one stage of a class-ordered
        decomposition on just the classes not yet peeled off."""
        y = None if self.y is None else self.y[mask]
        w = None if self.weights is None else self.weights[mask]
        return type(self)(self.spec, self.X[mask], y, weights=w)

    def relabel(self, new_y: Any) -> "DataRepresentation":
        """Same rows and feature space, a different `y` -- e.g.
        ``data.relabel(np.where(data.y == "a", "a", "rest"))`` for a
        one-vs-rest sub-problem given to a learner that can't fold the
        rest in itself (external binary learners)."""
        return type(self)(self.spec, self.X, np.asarray(new_y), weights=self.weights)


class HorizontalBitmapRepresentation(DataRepresentation):
    """The canonical Boolean bit-matrix encoding: a `numpy.packbits`-packed
    feature matrix (shape ``(n_samples, ceil(spec.n_features / 8))``) plus
    optional labels `y`. `coverage`/`Rule.covers_data_packed` and a
    search's primitives (`refine_cover`, `batch_cover_counts`,
    `batch_cover_sums`, `features_of`) all run against the packed form
    without ever touching `X`; the unpacked ``(n_samples, spec.n_features)``
    bool matrix `X` is reconstructed lazily on first access (see the
    property) -- for handing a plain array to external or dense learners.
    """

    #: the most cells `batch_cover_counts`/`batch_cover_sums` unpack at a
    #: time (covered rows x the bits of the open features' bytes), so
    #: their temporaries -- with weights, a float copy at eight bytes per
    #: cell -- stay small however large the data
    BATCH_BLOCK_CELLS = 1 << 18

    def __init__(self, spec: DataSpec, X, y: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None):
        super().__init__(spec)
        try:
            from scipy.sparse import issparse
        except ImportError:  # pragma: no cover
            issparse = lambda _o: False  # noqa: E731
        if issparse(X):
            X = X.toarray()
        X = np.asarray(X)
        if X.dtype != bool:
            X = X.astype(bool)
        if X.shape[1] != spec.n_features:
            raise ValueError(
                f"X has {X.shape[1]} columns but this DataSpec has "
                f"{spec.n_features} features"
            )
        self._n_samples: int = X.shape[0]
        self.y: Optional[np.ndarray] = None if y is None else np.asarray(y)
        self._packed: np.ndarray = np.packbits(X, axis=1)
        self._set_weights(weights)

    @functools.cached_property
    def X(self) -> np.ndarray:
        """The unpacked ``(n_samples, n_features)`` bool matrix, rebuilt
        from the packed cache on first access and held thereafter.
        `coverage`/`Rule.covers_data_packed` use `packed_rows()` and never
        trigger this; it's here for handing a plain array to external
        learners (sklearn / wittgenstein / imodels), for `Rule.covers_data`,
        and for `refine_cover`'s per-column access."""
        n_features = self.spec.n_features
        return np.unpackbits(self._packed, axis=1)[:, :n_features].astype(bool)

    def to_csr(self, chunk_cells: int = 1 << 16):
        """The matrix as a `scipy.sparse` CSR matrix (bool data, `int32`
        indices when they fit), unpacked a block of rows at a time -- so,
        unlike `X`, nothing dense of the full ``(n_samples, n_features)``
        size is formed, and nothing is cached here. What the sparse and
        N-list representations are built from (`from_boolean`)."""
        from scipy import sparse

        n, k = self._n_samples, self.spec.n_features
        index_dtype = np.int32 if n * k < 2 ** 31 else np.int64   # n * k bounds the nonzeros
        step = max(1, chunk_cells // max(k, 1))
        indptr = np.zeros(n + 1, dtype=index_dtype)
        cols = [np.zeros(0, dtype=index_dtype)]
        for start in range(0, n, step):
            block = np.unpackbits(self._packed[start:start + step], axis=1)[:, :k]
            indptr[start + 1:start + 1 + len(block)] = indptr[start] + np.cumsum(block.sum(axis=1, dtype=np.int64))
            cols.append(np.nonzero(block)[1].astype(index_dtype))   # row-major: ascending per row
            del block
        indices = np.concatenate(cols)
        del cols
        return sparse.csr_matrix((np.ones(len(indices), dtype=bool), indices, indptr), shape=(n, k))

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def packed_rows(self) -> np.ndarray:
        return self._packed

    def coverage(self, rule: "Rule") -> np.ndarray:
        """Vectorized bitmask subset check across all rows, against the
        `numpy.packbits` cache -- no per-feature Python loop over the
        data, and `X` is never materialized."""
        n_bytes = self._packed.shape[1]
        n_features = self.spec.n_features
        want_full = np.zeros(n_features, dtype=bool)
        for lit in rule.conditions:
            want_full[lit.feature] = True
        want = np.pad(np.packbits(want_full), (0, n_bytes - (n_features + 7) // 8))
        return np.all((self._packed & want) == want, axis=1)

    def features_of(self, row: int) -> np.ndarray:
        return np.flatnonzero(np.unpackbits(self._packed[row], count=self.spec.n_features))

    def _column(self, feature: int) -> np.ndarray:
        """Feature `feature`'s column as a bool array, read off its bit in
        the packed rows -- one byte per row is touched, nothing cached."""
        return (self._packed[:, feature >> 3] & np.uint8(0x80 >> (feature & 7))) != 0

    def _open_column_blocks(self, rows: np.ndarray, feature_indices: Sequence[int]):
        """The columns `feature_indices` of the rows `rows`, unpacked a
        block of rows at a time: yields ``(start, block)``, `block` the
        ``uint8`` 0/1 matrix of ``rows[start:start + len(block)]`` x the
        features. Only the bytes holding one of the features are unpacked,
        and a block holds at most `BATCH_BLOCK_CELLS` unpacked bits."""
        feats = np.asarray(feature_indices, dtype=np.intp)
        byte_idx, inverse = np.unique(feats >> 3, return_inverse=True)
        bit_pos = inverse.ravel() * 8 + (feats & 7)
        step = max(1, self.BATCH_BLOCK_CELLS // max(8 * len(byte_idx), 1))
        for start in range(0, len(rows), step):
            bits = np.unpackbits(self._packed[np.ix_(rows[start:start + step], byte_idx)], axis=1)
            yield start, bits[:, bit_pos]

    # -- incremental coverage (search fast path) -----------------------
    #
    # A handle is (cov, scope): `cov` is the current rule's coverage,
    # already ANDed with the original `example_mask` (so it's zero
    # outside it from the very first refinement on); `scope` is that
    # same mask (or `None`), carried through unchanged. `scope` has to be
    # tracked separately from `cov` -- `~cov` alone is True for *every*
    # out-of-scope row regardless of the rule, which would wrongly count
    # them as false negatives/negatives without it (see `cover_counts`).

    def initial_cover(self, example_mask: Optional[np.ndarray] = None):
        support, w = self.scope(example_mask)
        if support is None:
            return np.ones(self._n_samples, dtype=bool), None, None
        return support.copy(), support, w

    def refine_cover(self, handle, feature: int):
        cov, scope, w = handle
        # one bit per row, read from the packed bytes: as fast as a column
        # of the unpacked `X`, without unpacking (and caching) all of it
        return cov & self._column(feature), scope, w

    def cover_counts(self, handle, positive_class: Any) -> Tuple[int, int, int, int]:
        if self.y is None:
            raise ValueError("cover_counts needs labels (self.y)")
        cov, scope, w = handle
        pos_mask = self.y == positive_class
        if w is not None:
            return _weighted_counts(w, cov, pos_mask)
        in_scope = np.ones(self._n_samples, dtype=bool) if scope is None else scope
        tp = int(np.sum(cov & pos_mask))
        fp = int(np.sum(cov & ~pos_mask))
        fn = int(np.sum(~cov & pos_mask & in_scope))
        tn = int(np.sum(~cov & ~pos_mask & in_scope))
        return tp, fp, fn, tn

    def cover_rows(self, handle) -> np.ndarray:
        return handle[0]

    def batch_cover_sums(self, handle, values: np.ndarray, feature_indices: Sequence[int]) -> np.ndarray:
        """`DataRepresentation.batch_cover_sums` as a matrix product over
        the covered rows with a nonzero value only -- with a subsample,
        the rows outside it carry zeros and are skipped -- a block of rows
        at a time (`_open_column_blocks`), so that the float copy the
        product needs stays small."""
        values = np.asarray(values, dtype=float)
        rows = np.flatnonzero(handle[0] & (values != 0).any(axis=1))
        out = np.zeros((values.shape[1], len(feature_indices)))
        for start, block in self._open_column_blocks(rows, feature_indices):
            out += values[rows[start:start + len(block)]].T @ block
        return out

    def chain_cover_counts(
        self, handle, positive_class: Any, feature_indices: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(tp, fp)`` arrays, one entry per feature in `feature_indices`
        -- a numeric attribute's ``attr>=t`` thresholds, ascending, all
        still open under `handle`'s rule (`pyrulearn.learners.seco`'s
        `score_children` only ever calls this with such a list; see
        `ThresholdChain`/`NumericGroup` in `pyrulearn.data.attributes`).
        Computed in one pass over `handle`'s covered rows instead of one
        `refine_cover` + `cover_counts` per threshold: because the chain
        is monotonic, feature `k` (0-indexed) is True for a row exactly
        when that row satisfies at least ``k + 1`` of `feature_indices`
        -- so the per-row count of satisfied thresholds, histogrammed by
        class and summed from the top down, gives every threshold's
        tp/fp at once. Opt-in: this method doesn't exist on
        `DataRepresentation` or the other representations, so
        `score_children` falls back to the plain per-threshold path
        wherever it's absent. No search calls `score_children` any more
        (they count via `batch_cover_counts` -- `ROADMAP.md`'s "Build
        only the children a search follows").
        """
        if self.y is None:
            raise ValueError("chain_cover_counts needs labels (self.y)")
        cov, scope, w = handle
        pos_mask = self.y == positive_class
        m = len(feature_indices)
        satisfied = self.X[:, feature_indices].sum(axis=1)
        pos_idx = cov & pos_mask
        neg_idx = cov & ~pos_mask
        if w is not None:
            pos_hist = np.bincount(satisfied[pos_idx], weights=w[pos_idx], minlength=m + 1)
            neg_hist = np.bincount(satisfied[neg_idx], weights=w[neg_idx], minlength=m + 1)
        else:
            pos_hist = np.bincount(satisfied[pos_idx], minlength=m + 1)
            neg_hist = np.bincount(satisfied[neg_idx], minlength=m + 1)
        tp = np.cumsum(pos_hist[::-1])[::-1][1:]
        fp = np.cumsum(neg_hist[::-1])[::-1][1:]
        if w is None:
            tp = tp.astype(int)
            fp = fp.astype(int)
        return tp, fp

    def batch_cover_counts(
        self, handle, positive_class: Any, feature_indices: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(tp, fp, keeps_all)`` arrays, one entry per feature in `feature_indices`
        -- *any* open features, not necessarily from the same attribute
        or even numeric (unlike `chain_cover_counts`, no monotonic chain
        assumed) -- computed for every one of them in a single matrix
        multiply, the way `pyrulearn.learners.boosting.DenseENDER`'s `_grow`
        scores every candidate feature at once (``(t * c).T @ Xf``)
        instead of counting one at a time. What
        `pyrulearn.learners.seco.count_open_children` -- and so every
        rule search -- counts with on this representation.

        Works on the covered rows of the open columns only (uncovered
        rows would only add zero weights), unpacked from the packed bits
        a block of rows at a time (`_open_column_blocks`): without weights
        it counts the set bits per column, with weights it multiplies each
        block by them. The temporaries stay bounded by
        `BATCH_BLOCK_CELLS`, where an unpacked, float copy of all covered
        rows of the open columns at eight bytes per cell made a fit need
        about ten times the unpacked matrix (`ROADMAP.md`).

        `keeps_all` marks the features that drop no covered row with a
        nonzero weight: refining by one changes nothing, so its tp/fp
        equal the rule's own. Summing a different set of rows can round
        them differently, though -- weights are floats -- and a search
        could then take such a child for a (tiny) improvement; it uses
        `keeps_all` to give the child exactly its parent's score instead.
        """
        if self.y is None:
            raise ValueError("batch_cover_counts needs labels (self.y)")
        cov, scope, w = handle
        # only the covered rows that carry weight: the others would only
        # contribute zeros, and a deeper rule covers fewer and fewer of them
        rows = np.flatnonzero(cov if w is None else cov & (w != 0))
        pos = (self.y == positive_class)[rows]
        m = len(feature_indices)
        keeps_all = np.ones(m, dtype=bool)
        if w is None:
            tp = np.zeros(m, dtype=np.int64)
            fp = np.zeros(m, dtype=np.int64)
            for start, block in self._open_column_blocks(rows, feature_indices):
                p = pos[start:start + len(block)]
                tp += np.count_nonzero(block[p], axis=0)
                fp += np.count_nonzero(block[~p], axis=0)
                keeps_all &= block.all(axis=0)
            return tp, fp, keeps_all
        wr = w[rows]
        pos_w = np.where(pos, wr, 0.0)
        neg_w = np.where(pos, 0.0, wr)
        tp = np.zeros(m)
        fp = np.zeros(m)
        for start, block in self._open_column_blocks(rows, feature_indices):
            end = start + len(block)
            tp += pos_w[start:end] @ block
            fp += neg_w[start:end] @ block
            keeps_all &= block.all(axis=0)
        return tp, fp, keeps_all

    @classmethod
    def from_dataframe(
        cls, df, label_col: Optional[str] = None, spec: Optional[DataSpec] = None, name: Optional[str] = None
    ) -> "HorizontalBitmapRepresentation":
        """Build from a pandas DataFrame of 0/1 (or bool) feature columns.
        `spec` binds against an existing `DataSpec` (e.g. a training
        split's, when building a test representation); if omitted, one
        is built fresh from `df`'s (non-label) columns."""
        if label_col is not None:
            y = df[label_col].to_numpy()
            feat_df = df.drop(columns=[label_col])
        else:
            y = None
            feat_df = df
        if spec is None:
            spec = DataSpec(list(feat_df.columns), name=name)
        return cls(spec, feat_df.to_numpy(dtype=bool), y)

    @classmethod
    def from_scipy(cls, matrix, spec: DataSpec, y: Optional[np.ndarray] = None) -> "HorizontalBitmapRepresentation":
        """Build from a `scipy.sparse` matrix already matching `spec`'s
        feature order -- densified into the packed bit-matrix. (For a
        genuinely sparse dataset, `SparseDataRepresentation.from_scipy`
        keeps it sparse.)"""
        return cls(spec, matrix, y)

    @classmethod
    def from_xy(
        cls,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
        spec: Optional[DataSpec] = None,
        name: Optional[str] = None,
    ) -> "HorizontalBitmapRepresentation":
        """Build from an array-like ``X`` (plain numpy, not necessarily
        already Boolean) -- the ordinary ``X, y`` convention shared by
        scikit-learn and most other array-based ML tooling.

        If ``X`` is already 0/1-valued, it's used as-is. Otherwise pass
        ``threshold`` to binarize continuous features as ``X > threshold``
        (a single scalar, or an array broadcastable to a feature-wise
        threshold, e.g. per-feature medians). `spec` binds against an
        existing `DataSpec`; if omitted, one is built fresh, with
        `feature_names` defaulting to ``f0, f1, ...`` if not given.
        """
        X = np.asarray(X)
        Xb = (X > threshold) if threshold is not None else X.astype(bool)
        if spec is None:
            names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(Xb.shape[1])]
            spec = DataSpec(names, name=name)
        return cls(spec, Xb, y)

    def __repr__(self) -> str:
        return f"HorizontalBitmapRepresentation(spec={self.spec!r}, n_samples={self.n_samples})"


def _weighted_counts(w: np.ndarray, cov: np.ndarray, pos_mask: np.ndarray) -> Tuple[float, float, float, float]:
    """``(tp, fp, fn, tn)`` as sums of the weights `w` (zero outside the
    scope) -- `cov` the covered rows, `pos_mask` the positive ones."""
    pw, nw = w * pos_mask, w * ~pos_mask
    tp, fp = float(pw[cov].sum()), float(nw[cov].sum())
    return tp, fp, float(pw.sum()) - tp, float(nw.sum()) - fp


class _NListMaskContext:
    """Per-search state shared, unchanged, by every handle
    `NListRepresentation.initial_cover`/`refine_cover` produce for one
    search: the `example_mask` (or `None`), the effective row weights (or
    `None`), and -- built on first use, once per search -- the cumulative
    per-class sums of those over the rows in tree order. A node's rows are
    one contiguous range of that order, so its (masked, weighted) class
    counts are the difference of two rows of the cumulative sums: one pass
    over the rows per search, then two lookups per node."""

    __slots__ = ("mask", "weights", "_cum", "_scope_totals")

    def __init__(self, mask: Optional[np.ndarray], weights: Optional[np.ndarray] = None):
        self.mask = mask
        self.weights = weights          # effective row weights (zero outside the mask), or None
        self._cum: Optional[np.ndarray] = None
        self._scope_totals: Optional[np.ndarray] = None

    def scope_totals(self, rep: "NListRepresentation") -> np.ndarray:
        """Per-class row counts within this context's scope (all rows,
        or just the masked ones) -- computed once, reused for every
        `cover_counts` call in this search."""
        if self.mask is None:
            return rep._class_totals
        if self._scope_totals is None:
            if self.weights is not None:
                self._scope_totals = np.bincount(rep._y_idx, weights=self.weights, minlength=rep._n_classes)
            else:
                self._scope_totals = np.bincount(rep._y_idx[self.mask], minlength=rep._n_classes)
        return self._scope_totals

    def cum(self, rep: "NListRepresentation") -> np.ndarray:
        """`rep._class_cum` restricted to this context's scope: the
        representation's own (unmasked) sums if there's no mask, otherwise
        derived once from the mask or weights and cached."""
        if self.mask is None:
            return rep._class_cum
        if self._cum is None:
            self._cum = rep._cumulative_class_sums(self.weights if self.weights is not None else self.mask)
        return self._cum


def _class_value_columns(y: np.ndarray, positive_class: Any, w: Optional[np.ndarray]) -> np.ndarray:
    """Per row: (weight if positive, weight if negative, 1) -- summed over
    the rows that have a feature, the feature's tp, fp and how many of the
    rows it keeps (exact: a sum of ones). ``w=None``: every row weighs 1."""
    pos = (y == positive_class).astype(float)
    wr = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    return np.column_stack([wr * pos, wr * (1.0 - pos), np.ones(len(y))])


def _counts_from_sums(sums: np.ndarray, n_rows: int, feature_indices: Sequence[int],
                      weighted: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`batch_cover_counts`' ``(tp, fp, keeps_all)`` from the per-feature
    sums of `_class_value_columns` (``(n_features, 3)``) over `n_rows`
    contributing rows."""
    f = np.asarray(feature_indices, dtype=np.intp)
    tp, fp, kept = sums[f, 0], sums[f, 1], sums[f, 2]
    if not weighted:
        tp, fp = np.rint(tp).astype(int), np.rint(fp).astype(int)
    return tp, fp, kept == n_rows


class NListRepresentation(DataRepresentation):
    """A PPC-tree / N-list encoding of a Boolean dataset -- the vertical
    index Huynh, Fürnkranz & Beck's LORD builds before learning -- exposed
    behind the same `DataRepresentation` interface as
    `HorizontalBitmapRepresentation`, so every learner runs on it unchanged.
    The tree *is* the data: it is stored instead of the Boolean table, not
    next to it.

    How it works
    ------------
    Each row is the set of features that are True in it. Sort those
    features most-frequent-first and insert the resulting path into a
    trie; rows with a shared prefix share tree nodes, and each row ends at
    one node (the root for a row with no True feature). The nodes are
    numbered in pre-order, and the rows are kept in the order of the node
    they end at -- so the rows whose path passes through a node, those
    ending in its subtree, are one contiguous range of that order
    (`_row_lo`/`_row_hi`). Every node also keeps the bitmask of the
    features on its root-to-here path (`_words`). That is all that is
    stored per node; per row, only its place in the order (`_perm`) and the
    node it ends at (`_row_end`). A row's features are the bits of its end
    node's path, so `features_of` and `X` read the data back from the tree.

    A rule ``f1 ∧ ... ∧ fm`` covers exactly the rows whose path contains
    a node for every ``fi``. Take ``iy``, the deepest (least frequent) of
    the ``fi``; among the nodes registering ``iy`` (its N-list, contiguous
    in `_words` -- `_item_ptr`), keep those whose path bitmask has every
    ``fi``; the rule's coverage is the union of their row ranges (disjoint:
    no node of an item lies below another of the same item). Class counts
    of a node are differences of cumulative per-class sums over the rows
    in tree order (`_class_cum`) -- and under an `example_mask` or weights,
    of cumulative sums of those, built once per search
    (`_NListMaskContext`).

    `coverage`/`features_of` (used outside a search: prediction, pruning,
    `RuleStats.from_rule`) always recompute from scratch this way.
    `pyrulearn.learners.seco.BeamSearch`/`HillClimbing` instead go through
    `initial_cover`/`refine_cover`/`cover_counts` -- see the comment above
    `initial_cover` for the LORD-style incremental scheme: a handle tracks
    the deepest feature fixed so far (its "anchor") and the still-shrinking
    set of that item's nodes still satisfying the rule, so refining by a
    feature no deeper than the anchor is just a filter over the *current*
    (already narrow) set, and only refining by a strictly deeper feature
    needs a fresh look at that item's own N-list.

    Negation hurts this representation too: pairing every feature with
    its negation doubles each row's path length (every original feature
    *or* its negation is True, so both occupy a tree position), so fewer
    rows share a full prefix and the tree gets more nodes for the same
    underlying data. `without_negations()` before building (or on an
    already-negated representation) is the fix.

    Build it from an existing `HorizontalBitmapRepresentation` with
    `from_boolean`, or from raw arrays via the constructor / `from_xy` /
    `from_dataframe`, mirroring `HorizontalBitmapRepresentation`.
    """

    #: keep each node's pre-order number and subtree end after the build
    #: (`PrePostNListRepresentation` searches with them; here they are
    #: only needed to derive the row ranges)
    _KEEP_PRE_POST = False
    #: keep every node's path bitmask (`_words`); without them
    #: (`PrePostNListRepresentation`), the nodes keep their parent instead
    _PATH_BITMASKS = True

    def __init__(self, spec: DataSpec, X, y: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None):
        super().__init__(spec)
        from scipy import sparse

        csr = sparse.csr_matrix(X if sparse.issparse(X) else np.asarray(X, dtype=bool))
        if csr.dtype != bool:
            csr = csr.astype(bool)
        csr.eliminate_zeros()
        n, k = csr.shape
        if k != spec.n_features:
            raise ValueError(f"X has {k} columns but this DataSpec has {spec.n_features} features")
        self._n_samples: int = n
        self.y: Optional[np.ndarray] = None if y is None else np.asarray(y)
        self._build(csr)
        self._set_weights(weights)

    # -- construction ---------------------------------------------------

    def _build(self, csr) -> None:
        """Builds the tree from `csr` (``(n, k)``, bool) without a Python
        node graph: every row as the sequence of its features' ranks
        (most frequent first), padded with -1 into an ``(n, max_len)``
        matrix and sorted lexicographically -- in that order the tree's
        nodes appear in pre-order, a row's nodes from its longest common
        prefix with the previous row on are new, and the others are the
        previous rows' nodes at the same depth."""
        n, k = csr.shape
        self._n_features = k
        W = self._word_count = (k + 63) // 64 or 1
        freq = np.bincount(csr.indices, minlength=k)
        order = np.argsort(-freq, kind="stable")         # rank -> feature
        rank = np.empty(k, dtype=np.int64)
        rank[order] = np.arange(k)
        self._rank = rank

        lens = np.diff(csr.indptr).astype(np.int64)
        L = int(lens.max()) if n else 0
        # P[i, :lens[i]] = row i's feature ranks, ascending; -1 after
        P = np.full((n, max(L, 1)), -1, dtype=np.int32)
        if csr.nnz:
            row_of = np.repeat(np.arange(n), lens)
            ranks = rank[csr.indices]
            o = np.lexsort((ranks, row_of))
            P[row_of[o], np.arange(csr.nnz) - np.repeat(csr.indptr[:-1], lens)] = ranks[o]
        perm = np.lexsort(P.T[::-1]).astype(np.int64)    # rows in tree (pre-)order
        S, slens = P[perm], lens[perm]
        del P

        # longest common prefix with the previous row (in sorted order)
        lcp = np.zeros(n, dtype=np.int64)
        if n > 1:
            same = S[1:] == S[:-1]
            lcp[1:] = np.minimum(np.where(same.all(axis=1), L, same.argmin(axis=1)),
                                 np.minimum(slens[1:], slens[:-1]))
            del same
        depth = np.arange(S.shape[1])
        new = (depth >= lcp[:, None]) & (depth < slens[:, None])   # the tree's nodes, in pre-order
        ids = np.where(new, np.cumsum(new.ravel()).reshape(new.shape), 0).astype(np.int32)
        node_at = np.maximum.accumulate(ids, axis=0)    # every row's node at every depth of its path
        N = int(new.sum())                               # nodes 1..N; 0 is the root

        item = np.full(N + 1, -1, dtype=np.int64)
        node_depth = np.zeros(N + 1, dtype=np.int64)
        parent = np.zeros(N + 1, dtype=np.int64)
        ri, di = np.nonzero(new)                         # row-major: ascending node id
        item[1:] = order[S[ri, di]]
        node_depth[1:] = di
        parent[1:] = np.where(di > 0, node_at[ri, np.maximum(di - 1, 0)], 0)
        end_sorted = np.where(slens > 0, node_at[np.arange(n), np.maximum(slens - 1, 0)], 0) if n else \
            np.zeros(0, dtype=np.int64)
        del S, new, ids, node_at, ri, di

        # subtree sizes bottom-up, one depth level at a time; path bitmasks top-down
        size = np.ones(N + 1, dtype=np.int64)
        levels = [np.flatnonzero(node_depth[1:] == d) + 1 for d in range(L)]
        for nodes in reversed(levels):
            np.add.at(size, parent[nodes], size[nodes])
        if self._PATH_BITMASKS:
            words = np.zeros((N + 1, W), dtype=np.uint64)
            for nodes in levels:
                words[nodes] = words[parent[nodes]]
                f = item[nodes]
                words[nodes, f >> 6] |= np.left_shift(np.uint64(1), (f & 63).astype(np.uint64))
        subtree_end = np.arange(N + 1) + size - 1

        # per item, its nodes in pre-order: the N-lists, one flat array each
        flat = np.argsort(item[1:], kind="stable") + 1   # node ids grouped by item
        self._item_ptr = np.concatenate(([0], np.cumsum(np.bincount(item[1:], minlength=k)))).astype(np.int64)
        if self._PATH_BITMASKS:
            self._words = words[flat]
            del words
        self._row_lo = np.searchsorted(end_sorted, flat, side="left").astype(np.int32)
        self._row_hi = np.searchsorted(end_sorted, subtree_end[flat], side="right").astype(np.int32)
        if self._KEEP_PRE_POST:
            self._node_pre = flat.astype(np.int32)
            self._node_end = subtree_end[flat].astype(np.int32)
        position = np.full(N + 1, -1, dtype=np.int64)
        position[flat] = np.arange(N)
        if not self._PATH_BITMASKS:
            self._node_parent = position[parent[flat]].astype(np.int32)   # -1: a child of the root
        row_end = np.empty(n, dtype=np.int32)
        row_end[perm] = position[end_sorted]             # -1: the row ends at the root
        self._row_end = row_end
        self._perm = perm.astype(np.int32)
        self._init_labels()

    def _init_labels(self) -> None:
        """The per-class structures over `self.y`: the classes, every row's
        class index, the class totals and their cumulative sums in tree order."""
        if self.y is not None:
            classes, y_idx = np.unique(self.y, return_inverse=True)
            self._classes = classes
            self._class_index = {c: i for i, c in enumerate(classes)}
            self._y_idx = y_idx.astype(np.int64)
            self._n_classes = len(classes)
            self._class_totals = np.bincount(self._y_idx, minlength=self._n_classes)
            self._class_cum = self._cumulative_class_sums(None)
        else:
            self._classes = self._class_index = self._y_idx = None
            self._n_classes = self._class_totals = self._class_cum = None

    def _cumulative_class_sums(self, row_values: Optional[np.ndarray]) -> np.ndarray:
        """``(n_samples + 1, n_classes)``: row ``i`` the per-class sums of
        `row_values` (a bool mask or weights; 1 per row if ``None``) over
        the first ``i`` rows in tree order -- so the rows of a node, the
        range ``[lo, hi)``, sum to ``cum[hi] - cum[lo]``. Integer unless
        weighted."""
        n, c = self._n_samples, self._n_classes
        weighted = row_values is not None and row_values.dtype != bool
        per_row = np.zeros((n, c), dtype=float if weighted else np.int64)
        y = self._y_idx[self._perm]
        per_row[np.arange(n), y] = 1 if row_values is None else row_values[self._perm]
        out = np.zeros((n + 1, c), dtype=per_row.dtype)
        np.cumsum(per_row, axis=0, out=out[1:])
        return out

    # -- reading the data back from the tree ------------------------------

    def _words_of(self, item: int) -> np.ndarray:
        """`item`'s N-list: the path bitmasks of its nodes, in pre-order."""
        return self._words[self._item_ptr[item]:self._item_ptr[item + 1]]

    def _rows_of(self, positions: np.ndarray) -> np.ndarray:
        """The rows passing through the nodes at flat `positions` (no two
        of them on one path, so their ranges are disjoint), in tree order."""
        if len(positions) <= 8:      # the common case in a search: a few slices beat the vectorized gather
            perm, lo, hi = self._perm, self._row_lo, self._row_hi
            parts = [perm[lo[p]:hi[p]] for p in positions.tolist()]
            if not parts:
                return np.empty(0, dtype=np.int64)
            return parts[0] if len(parts) == 1 else np.concatenate(parts)
        # the positions in tree order as one cumulative sum: steps of 1 within
        # a range, and a jump from the end of one range to the start of the next
        lo, hi = self._row_lo[positions].astype(np.int64), self._row_hi[positions].astype(np.int64)
        ends = np.cumsum(hi - lo)
        steps = np.ones(ends[-1], dtype=np.int64)
        steps[0] = lo[0]
        steps[ends[:-1]] = lo[1:] - hi[:-1] + 1
        return self._perm[np.cumsum(steps)]

    def _path_features(self, ends: np.ndarray) -> np.ndarray:
        """``(len(ends), n_features)`` bool: the features on the paths of the
        nodes at flat positions `ends` (-1: the root, no feature) -- the
        bits of their path bitmasks."""
        words = self._words[np.maximum(ends, 0)]
        words[ends < 0] = 0
        bits = np.unpackbits(words.astype("<u8").view(np.uint8), axis=1, bitorder="little")
        return bits[:, :self._n_features].astype(bool)

    def _end_features(self, rows: np.ndarray) -> np.ndarray:
        """``(len(rows), n_features)`` bool: the features of `rows`, those on
        the path of the node each ends at."""
        return self._path_features(self._row_end[rows])

    def _decode(self, ends: np.ndarray, chunk_cells: int = 1 << 16):
        """`_path_features` of `ends` as a `scipy.sparse` CSR matrix, decoded
        a block at a time -- nothing dense of the full size is formed."""
        from scipy import sparse

        k = self._n_features
        step = max(1, chunk_cells // max(k, 1))
        blocks = [sparse.csr_matrix(self._path_features(ends[s:s + step])) for s in range(0, len(ends), step)]
        return sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, k), dtype=bool)

    def _csr(self):
        """The data as a `scipy.sparse` CSR matrix, decoded from the tree."""
        return self._decode(self._row_end)

    # -- the two DataRepresentation primitives -------------------------

    def _feature_words(self, features: Sequence[int]) -> np.ndarray:
        w = np.zeros(self._word_count, dtype=np.uint64)
        for f in features:
            w[f >> 6] |= np.uint64(1 << (int(f) & 63))
        return w

    def _coverage_of_features(self, feats: Sequence[int]) -> np.ndarray:
        """The from-scratch N-list computation behind `coverage`: look only
        at the *deepest* feature's own N-list, keep the nodes whose path
        contains every one of `feats`, and take their rows."""
        feats = list(feats)
        out = np.zeros(self._n_samples, dtype=bool)
        if not feats:
            out[:] = True
            return out
        iy = max(feats, key=lambda f: self._rank[f])   # deepest item in the tree
        words = self._words_of(iy)
        want = self._feature_words(feats)
        hit = np.flatnonzero(np.all((words & want) == want, axis=1))
        if hit.size:
            out[self._rows_of(self._item_ptr[iy] + hit)] = True
        return out

    def coverage(self, rule) -> np.ndarray:
        return self._coverage_of_features([int(l.feature) for l in rule.conditions])

    def features_of(self, row: int) -> np.ndarray:
        return np.flatnonzero(self._end_features(np.array([row]))[0]).astype(np.int64)

    # -- incremental coverage (search fast path) -----------------------
    #
    # A handle is (anchor_item, active, mask_words, ctx):
    #
    # - `anchor_item` is the *deepest* (least frequent) feature fixed so
    #   far, or `None` for the empty rule. `active` indexes into that
    #   item's own N-list (`_words_of(anchor_item)` etc.) -- the nodes
    #   whose path already satisfies every feature fixed so far.
    # - Refining by a feature no deeper than the anchor only needs to
    #   test the *new* bit against the already-narrow `active` set --
    #   every node still in it already satisfies every earlier bit, so
    #   there's nothing to recheck. `active` only ever shrinks, so this
    #   gets cheaper as the rule grows, the actual LORD-style win.
    # - Refining by a feature *deeper* than the anchor re-anchors: no
    #   node registered under the old anchor can tell us anything about
    #   an item that only appears *below* it, so this re-scans the new
    #   item's own N-list, filtered by the full accumulated mask --
    #   exactly `coverage`'s cost for that one item. (Pre/post codes
    #   give a second way, searching the new item's nodes inside the
    #   active nodes' subtrees: `PrePostNListRepresentation`, which
    #   does without the path bitmasks altogether.)
    # - `mask_words` is the accumulated feature bitmask (needed for the
    #   re-anchor case's full-mask test). `ctx` (`_NListMaskContext`)
    #   carries `example_mask` and the weights, shared unchanged by every
    #   handle from one `initial_cover` call, with the cumulative class
    #   sums that turn a node's row range into its counts.

    def initial_cover(self, example_mask: Optional[np.ndarray] = None):
        support, w = self.scope(example_mask)
        mask = None if support is None else np.array(support, dtype=bool, copy=True)
        return None, None, np.zeros(self._word_count, dtype=np.uint64), _NListMaskContext(mask, w)

    def refine_cover(self, handle, feature: int):
        anchor_item, active, mask_words, ctx = handle
        feature = int(feature)
        new_mask_words = mask_words.copy()
        new_mask_words[feature >> 6] |= np.uint64(1 << (feature & 63))

        if anchor_item is None or self._rank[feature] > self._rank[anchor_item]:
            words = self._words_of(feature)
            hit = np.all((words & new_mask_words) == new_mask_words, axis=1)
            return feature, np.flatnonzero(hit), new_mask_words, ctx

        wj, bit = feature >> 6, np.uint64(1 << (feature & 63))
        keep = (self._words_of(anchor_item)[active, wj] & bit) == bit
        return anchor_item, active[keep], new_mask_words, ctx

    def cover_counts(self, handle, positive_class: Any) -> Tuple[int, int, int, int]:
        if self.y is None:
            raise ValueError("cover_counts needs labels (self.y)")
        anchor_item, active, _, ctx = handle
        num = int if ctx.weights is None else float
        pos_idx = self._class_index[positive_class]
        scope_totals = ctx.scope_totals(self)
        n_pos = num(scope_totals[pos_idx])
        n_neg = num(scope_totals.sum()) - n_pos

        if anchor_item is None:
            tp, fp = n_pos, n_neg  # the empty rule covers everything in scope
        else:
            positions = self._item_ptr[anchor_item] + active
            cum = ctx.cum(self)
            active_counts = (cum[self._row_hi[positions]] - cum[self._row_lo[positions]]).sum(axis=0)
            tp = num(active_counts[pos_idx])
            fp = num(active_counts.sum()) - tp
        return tp, fp, n_pos - tp, n_neg - fp

    def cover_rows(self, handle) -> np.ndarray:
        anchor_item, active, _, ctx = handle
        if anchor_item is None:
            return np.ones(self._n_samples, dtype=bool) if ctx.mask is None else ctx.mask.copy()
        out = np.zeros(self._n_samples, dtype=bool)
        out[self._rows_of(self._item_ptr[anchor_item] + active)] = True
        if ctx.mask is not None:
            out &= ctx.mask
        return out

    # -- batched counting ----------------------------------------------
    #
    # Rows with the same feature set end at the same tree node, so the
    # covered rows' values are first summed per end node, and only the end
    # nodes that occur are multiplied with their path matrix (CSR: per end
    # node, the features on its path): cost grows with the distinct row
    # patterns covered, not the rows -- the compression the PPC-tree
    # stands for.

    #: the distinct row patterns are multiplied as a dense matrix when at least this
    #: share of its cells is true -- measured with ENDER on the synthetic data of
    #: `demos/representations.py`: dense pays off from about 2% (and is a third faster
    #: than the sparse product on real data with negation features, about 50%), while
    #: on wide sparse data (0.1%) it is several times slower
    DENSE_PATTERNS_MIN_DENSITY = 0.02
    #: ... and at most this many cells (one byte each)
    DENSE_PATTERNS_MAX_CELLS = 1 << 26
    #: the dense product converts the covered patterns to floats a block of at most this
    #: many cells at a time (8 bytes each), so its temporary copy stays small
    DENSE_PATTERNS_BLOCK_CELLS = 1 << 18

    @functools.cached_property
    def _row_patterns(self):
        """``(pattern of every row, the distinct patterns)`` -- a pattern is
        an end node; the patterns as a CSR matrix, or as a dense bool array
        when that is dense and small enough (see `DENSE_PATTERNS_MIN_DENSITY`)
        -- built once, on first use."""
        ends, pattern_of = np.unique(self._row_end, return_inverse=True)
        matrix = self._decode(ends)
        cells = matrix.shape[0] * matrix.shape[1]
        if cells and matrix.nnz >= self.DENSE_PATTERNS_MIN_DENSITY * cells \
                and cells <= self.DENSE_PATTERNS_MAX_CELLS:
            matrix = matrix.toarray()
        else:
            matrix = matrix.astype(float)
        return pattern_of.astype(np.int64).ravel(), matrix

    def _pattern_sums(self, rows: np.ndarray, values: np.ndarray) -> np.ndarray:
        """``values`` (one row per entry of `rows`) summed per feature over
        `rows`, through the distinct row patterns: ``(n_features, m)``."""
        pattern_of, matrix = self._row_patterns
        p = pattern_of[rows]
        n_patterns = matrix.shape[0]
        hit = np.flatnonzero(np.bincount(p, minlength=n_patterns))
        per_pattern = np.column_stack([np.bincount(p, weights=values[:, j], minlength=n_patterns)[hit]
                                       for j in range(values.shape[1])])
        if isinstance(matrix, np.ndarray):
            out = np.zeros((matrix.shape[1], per_pattern.shape[1]))
            step = max(1, self.DENSE_PATTERNS_BLOCK_CELLS // max(matrix.shape[1], 1))
            for s in range(0, len(hit), step):
                out += matrix[hit[s:s + step]].T.astype(float) @ per_pattern[s:s + step]
            return out
        return np.asarray(matrix[hit].T @ per_pattern)

    def batch_cover_counts(self, handle, positive_class: Any, feature_indices: Sequence[int]):
        """`HorizontalBitmapRepresentation.batch_cover_counts` for N-lists: tp,
        fp and ``keeps_all`` of every feature in `feature_indices`, from
        one pass over the covered rows' distinct patterns."""
        if self.y is None:
            raise ValueError("batch_cover_counts needs labels (self.y)")
        ctx = handle[3]
        w = ctx.weights
        cov = self.cover_rows(handle)
        rows = np.flatnonzero(cov if w is None else cov & (w != 0))
        values = _class_value_columns(self.y[rows], positive_class, None if w is None else w[rows])
        return _counts_from_sums(self._pattern_sums(rows, values), len(rows), feature_indices, w is not None)

    def batch_cover_sums(self, handle, values: np.ndarray, feature_indices: Sequence[int]) -> np.ndarray:
        """`DataRepresentation.batch_cover_sums` through the covered rows'
        distinct patterns (covered rows with a nonzero value only)."""
        values = np.asarray(values, dtype=float)
        rows = np.flatnonzero(self.cover_rows(handle) & (values != 0).any(axis=1))
        sums = self._pattern_sums(rows, values[rows])
        return sums[np.asarray(feature_indices, dtype=np.intp)].T

    # -- parity with HorizontalBitmapRepresentation ------------------------

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @functools.cached_property
    def X(self) -> np.ndarray:
        """The unpacked ``(n_samples, n_features)`` bool matrix, decoded
        from the tree on first access and held thereafter -- for handing a
        plain array to external tooling. The coverage path never triggers
        this, and neither do `select_rows`, `relabel` or
        `without_negations` (they decode to a sparse matrix instead)."""
        return self._end_features(np.arange(self._n_samples))

    def select_rows(self, mask: np.ndarray) -> "NListRepresentation":
        """The rows `mask` selects, as an N-list derived from this one's tree
        instead of built anew: the nodes no selected row passes through
        are dropped -- whole subtrees, since a node's rows are those ending
        in its subtree -- and the remaining ranges, end nodes and N-lists
        renumbered. The feature order stays this data's (most frequent
        first here, not in the subset); any fixed order gives a correct
        tree, so coverage and counts are exactly those of a rebuilt one.
        An integer index array that is not sorted and duplicate-free
        (rows reordered or repeated) falls back to building anew."""
        mask = np.asarray(mask)
        if mask.dtype != bool:
            idx = mask.astype(np.int64)
            if idx.size and (np.any(np.diff(idx) <= 0) or idx[0] < 0 or idx[-1] >= self._n_samples):
                y = None if self.y is None else self.y[idx]
                w = None if self.weights is None else self.weights[idx]
                return type(self)(self.spec, self._csr()[idx], y, weights=w)
            mask = np.zeros(self._n_samples, dtype=bool)
            mask[idx] = True
        if mask.shape != (self._n_samples,):
            raise ValueError(f"mask must have shape ({self._n_samples},), got {mask.shape}")
        return self._derived(mask)

    def relabel(self, new_y: Any) -> "NListRepresentation":
        """Same rows, a different `y`: shares the tree, only the per-class
        structures are new."""
        new_y = np.asarray(new_y)
        if new_y.shape != (self._n_samples,):
            raise ValueError(f"new_y must have shape ({self._n_samples},), got {new_y.shape}")
        new = self._copy_index()
        new.y = new_y
        new._init_labels()
        new._set_weights(self.weights)
        return new

    def _copy_index(self) -> "NListRepresentation":
        """A new representation sharing this one's tree arrays (they are
        never modified in place), without its lazily computed caches (`X`,
        the row patterns)."""
        new = object.__new__(type(self))
        DataRepresentation.__init__(new, self.spec)
        for name in ("_n_samples", "_n_features", "_word_count", "_rank", "_item_ptr",
                     "_row_lo", "_row_hi", "_row_end", "_perm"):
            setattr(new, name, getattr(self, name))
        if self._PATH_BITMASKS:
            new._words = self._words
        else:
            new._node_parent = self._node_parent
        if self._KEEP_PRE_POST:
            new._node_pre, new._node_end = self._node_pre, self._node_end
        return new

    def _derived(self, mask: np.ndarray) -> "NListRepresentation":
        """`select_rows` for a boolean `mask` over the rows, see there."""
        keep = mask[self._perm]                                  # tree order
        cs = np.concatenate(([0], np.cumsum(keep))).astype(np.int64)
        lo, hi = cs[self._row_lo], cs[self._row_hi]
        alive = hi > lo                                          # nodes with a selected row
        n_items = len(self._item_ptr) - 1
        item_of = np.repeat(np.arange(n_items), np.diff(self._item_ptr))

        new = self._copy_index()
        m = int(cs[-1])
        new._n_samples = m
        new._item_ptr = np.concatenate(([0], np.cumsum(np.bincount(item_of[alive], minlength=n_items)))
                                       ).astype(np.int64)
        new._row_lo = lo[alive].astype(np.int32)
        new._row_hi = hi[alive].astype(np.int32)
        if self._KEEP_PRE_POST:   # ancestor tests only compare these numbers: no renumbering needed
            new._node_pre, new._node_end = self._node_pre[alive], self._node_end[alive]
        new_id = np.cumsum(mask) - 1                              # old row -> new row
        new._perm = new_id[self._perm[keep]].astype(np.int32)
        new_position = np.cumsum(alive) - 1                       # old node -> new node
        if self._PATH_BITMASKS:
            new._words = self._words[alive]
        else:   # an alive node's parent has its rows too, so it is alive as well
            p = self._node_parent[alive]
            new._node_parent = np.where(p >= 0, new_position[np.maximum(p, 0)], -1).astype(np.int32)
        old_end = self._row_end[mask]
        new._row_end = np.where(old_end >= 0, new_position[np.maximum(old_end, 0)], -1).astype(np.int32)
        new.y = None if self.y is None else self.y[mask]
        new._init_labels()
        new._set_weights(None if self.weights is None else self.weights[mask])
        return new

    def _reindexed(self, new_spec, source, is_complement):
        if is_complement is not None and np.any(is_complement):
            return super()._reindexed(new_spec, source, is_complement)   # complements are dense anyway
        return type(self)(new_spec, self._csr()[:, np.asarray(source)], self.y, weights=self.weights)

    @classmethod
    def from_boolean(cls, data: "HorizontalBitmapRepresentation") -> "NListRepresentation":
        """Build the N-list index for an existing
        `HorizontalBitmapRepresentation`, reusing its `DataSpec`, labels and
        row weights."""
        return cls(data.spec, data.to_csr(), data.y, weights=data.weights)

    @classmethod
    def from_scipy(cls, matrix, spec: DataSpec, y: Optional[np.ndarray] = None) -> "NListRepresentation":
        """Build the PPC-tree / N-list index from a `scipy.sparse` matrix
        matching `spec`'s feature order -- no dense `(n_samples,
        n_features)` array is ever formed."""
        return cls(spec, matrix, y)

    @classmethod
    def from_dataframe(
        cls, df, label_col: Optional[str] = None, spec: Optional[DataSpec] = None, name: Optional[str] = None
    ) -> "NListRepresentation":
        return cls.from_boolean(
            HorizontalBitmapRepresentation.from_dataframe(df, label_col=label_col, spec=spec, name=name)
        )

    @classmethod
    def from_xy(
        cls,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
        spec: Optional[DataSpec] = None,
        name: Optional[str] = None,
    ) -> "NListRepresentation":
        return cls.from_boolean(
            HorizontalBitmapRepresentation.from_xy(
                X, y, feature_names=feature_names, threshold=threshold, spec=spec, name=name
            )
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(spec={self.spec!r}, n_samples={self.n_samples}, "
            f"n_tree_nodes={len(self._row_lo)})"
        )


class PrePostNListRepresentation(NListRepresentation):
    """`NListRepresentation` without the path bitmasks: the N-lists of the
    original PrePost algorithm (Deng et al.), which only know every node's
    pre-order number and subtree end. Node ``u`` is an ancestor of ``v``
    iff ``pre(u) < pre(v) <= end(u)``, and since no two nodes of one item
    lie on one path, an item's nodes, stored in pre-order, are disjoint
    intervals -- so every test a search needs is a binary search:

    - refining by a feature *deeper* than the anchor (see the comment above
      the base class's `initial_cover`): its nodes inside the active nodes'
      intervals (`_reanchor`);
    - refining by a feature *no deeper*: the active nodes with a node of the
      feature above them (`_with_ancestor`);
    - `coverage`: the deepest feature's nodes, filtered by the others' in
      the same way.

    A row's features are read back by walking from the node it ends at to
    the root, over every node's parent (`_node_parent`). Per node, this
    keeps five integers (pre-order number, subtree end, parent, row range)
    instead of the path bitmask -- one bit per feature -- and the row range.
    """

    _KEEP_PRE_POST = True
    _PATH_BITMASKS = False

    # -- reading the data back from the tree ------------------------------

    def _item_of(self, positions: np.ndarray) -> np.ndarray:
        """The item of the nodes at flat `positions` (the N-lists are
        consecutive in item order)."""
        return np.searchsorted(self._item_ptr, positions, side="right") - 1

    def _path_features(self, ends: np.ndarray) -> np.ndarray:
        """``(len(ends), n_features)`` bool: the features on the paths of the
        nodes at flat positions `ends` (-1: the root), collected by walking
        all of them up to the root together, one level per step."""
        out = np.zeros((len(ends), self._n_features), dtype=bool)
        cur = np.asarray(ends, dtype=np.int64)
        rows = np.flatnonzero(cur >= 0)
        cur = cur[rows]
        while cur.size:
            out[rows, self._item_of(cur)] = True
            cur = self._node_parent[cur].astype(np.int64)
            live = cur >= 0
            rows, cur = rows[live], cur[live]
        return out

    # -- the ancestor tests ------------------------------------------------

    def _with_ancestor(self, anchor_item: int, active: np.ndarray, feature: int) -> np.ndarray:
        """The `active` nodes of `anchor_item` (indices into its N-list)
        that have a node of `feature` on their path: for each, the last of
        `feature`'s nodes that starts no later must reach it."""
        s, e = self._item_ptr[feature], self._item_ptr[feature + 1]
        if active.size == 0 or e == s:
            return active[:0]
        pre_f, end_f = self._node_pre[s:e], self._node_end[s:e]
        p = self._node_pre[self._item_ptr[anchor_item] + active]
        i = np.searchsorted(pre_f, p, side="right") - 1
        return active[(i >= 0) & (end_f[np.maximum(i, 0)] >= p)]

    def _reanchor(self, anchor_item: int, active: np.ndarray, feature: int) -> np.ndarray:
        """`feature`'s nodes (indices into its N-list) below an `active` node
        of `anchor_item`: every node there has the active node's path, so
        they are the ones whose path has every feature so far. Searches
        from the smaller side: the feature's nodes inside each active
        interval, or the active interval around each of the feature's nodes."""
        s, e = self._item_ptr[feature], self._item_ptr[feature + 1]
        if active.size == 0 or e == s:
            return np.empty(0, dtype=np.int64)
        pre_f = self._node_pre[s:e]
        a = self._item_ptr[anchor_item] + active          # in pre-order, disjoint intervals
        a_pre, a_end = self._node_pre[a], self._node_end[a]
        if active.size >= e - s:
            i = np.searchsorted(a_pre, pre_f, side="right") - 1
            return np.flatnonzero((i >= 0) & (a_end[np.maximum(i, 0)] >= pre_f)).astype(np.int64)
        lo = np.searchsorted(pre_f, a_pre, side="left")
        hi = np.searchsorted(pre_f, a_end, side="right")
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        # vectorized ragged-range gather (no Python loop over `active`)
        return (np.repeat(lo - (np.cumsum(counts) - counts), counts) + np.arange(total)).astype(np.int64)

    # -- the two DataRepresentation primitives -------------------------

    def _coverage_of_features(self, feats: Sequence[int]) -> np.ndarray:
        feats = list(dict.fromkeys(int(f) for f in feats))
        out = np.zeros(self._n_samples, dtype=bool)
        if not feats:
            out[:] = True
            return out
        iy = max(feats, key=lambda f: self._rank[f])   # deepest item in the tree
        active = np.arange(self._item_ptr[iy + 1] - self._item_ptr[iy], dtype=np.int64)
        for f in feats:
            if f != iy:
                active = self._with_ancestor(iy, active, f)
        if active.size:
            out[self._rows_of(self._item_ptr[iy] + active)] = True
        return out

    # -- incremental coverage (search fast path) -----------------------
    #
    # The base class's handles, without the accumulated feature bitmask
    # (its slot is None): the ancestor tests need no more than the active
    # nodes of the anchor.

    def initial_cover(self, example_mask: Optional[np.ndarray] = None):
        support, w = self.scope(example_mask)
        mask = None if support is None else np.array(support, dtype=bool, copy=True)
        return None, None, None, _NListMaskContext(mask, w)

    def refine_cover(self, handle, feature: int):
        anchor_item, active, _, ctx = handle
        feature = int(feature)
        if anchor_item is None:
            m = self._item_ptr[feature + 1] - self._item_ptr[feature]
            return feature, np.arange(m, dtype=np.int64), None, ctx
        if self._rank[feature] > self._rank[anchor_item]:
            return feature, self._reanchor(anchor_item, active, feature), None, ctx
        return anchor_item, self._with_ancestor(anchor_item, active, feature), None, ctx


class SparseDataRepresentation(DataRepresentation):
    """A `scipy.sparse` encoding: the feature matrix as CSR (row slices
    for `features_of`) and CSC (column index-sets for `coverage`), plus
    optional labels `y`. Same `DataRepresentation` interface as
    `HorizontalBitmapRepresentation` / `NListRepresentation`, so every
    `pyrulearn.learners.seco` learner runs on it unchanged and produces identical
    rules.

    `coverage(rule)` intersects the rule's features' CSC column
    index-sets (each a sorted list of the rows where that one feature is
    True), rarest feature first -- Eclat's "vertical" tid-list
    intersection, ``O(sum of the involved features' supports)``, and no
    ``(n_rows, n_features)`` matrix is ever built. This is the N-list
    without the prefix tree: an `NListRepresentation` whose PPC-tree did
    no prefix merging (every row its own root-to-leaf path) would have
    exactly these per-feature lists as its N-lists.

    Best when the data is genuinely sparse -- one-hot expansions of
    high-cardinality nominals, bag-of-words. Negation actively works
    against that: whenever every feature has a paired negation (`x`, `not
    x`), each pair contributes exactly one True value to every row, so
    density is *exactly* 50% regardless of how skewed the underlying
    attributes are -- not "sparse" at all, and worse than the packed
    matrix has any reason to be. `without_negations()` (which drops the
    ``not f`` / ``x != v`` half) is what makes this representation worth
    using; build with it, or call it on an already-negated
    representation, before reaching for `Sparse`. A numeric attribute
    keeps both ``x >= t`` and ``x < t`` either way, so it stays dense.
    """

    def __init__(self, spec: DataSpec, X, y: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None):
        super().__init__(spec)
        from scipy import sparse

        if sparse.issparse(X):
            csr = X.tocsr()
        else:
            csr = sparse.csr_matrix(np.asarray(X, dtype=bool))
        if csr.dtype != bool:
            csr = csr.astype(bool)
        csr.eliminate_zeros()
        if csr.shape[1] != spec.n_features:
            raise ValueError(
                f"X has {csr.shape[1]} columns but this DataSpec has "
                f"{spec.n_features} features"
            )
        csr.sort_indices()
        self._n_samples: int = csr.shape[0]
        self.y: Optional[np.ndarray] = None if y is None else np.asarray(y)
        self._csr = csr
        self._csc = csr.tocsc()
        self._csc.sort_indices()
        self._set_weights(weights)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def coverage(self, rule: "Rule") -> np.ndarray:
        feats = [int(l.feature) for l in rule.conditions]
        out = np.zeros(self._n_samples, dtype=bool)
        if not feats:
            out[:] = True
            return out
        indptr, indices = self._csc.indptr, self._csc.indices
        feats.sort(key=lambda f: indptr[f + 1] - indptr[f])  # rarest first
        rows = indices[indptr[feats[0]]:indptr[feats[0] + 1]]
        for f in feats[1:]:
            if rows.size == 0:
                break
            rows = np.intersect1d(
                rows, indices[indptr[f]:indptr[f + 1]], assume_unique=True
            )
        out[rows] = True
        return out

    def features_of(self, row: int) -> np.ndarray:
        s, e = self._csr.indptr[row], self._csr.indptr[row + 1]
        return np.sort(self._csr.indices[s:e].astype(np.int64))

    @functools.cached_property
    def X(self) -> np.ndarray:
        return np.asarray(self._csr.todense(), dtype=bool)

    # -- incremental coverage (search fast path) -----------------------
    #
    # A handle is (rows, scope): `rows` is the current rule's row-id
    # array -- a tid-list, exactly what `coverage` builds by intersecting
    # one feature's column at a time, except now the intersection carries
    # across search steps instead of restarting from the full column set
    # for every candidate. `scope` is the original `example_mask` (or
    # `None`), carried through unchanged -- needed to get fn/tn's *total*
    # positive/negative counts right when a mask is in play (`rows`
    # alone, being already narrowed to the rule's own coverage, can't
    # answer "how many masked rows are there in total").

    def initial_cover(self, example_mask: Optional[np.ndarray] = None):
        support, w = self.scope(example_mask)
        if support is None:
            return np.arange(self._n_samples, dtype=np.int64), None, None
        return np.flatnonzero(support).astype(np.int64), support, w

    def refine_cover(self, handle, feature: int):
        rows, mask, w = handle
        indptr, indices = self._csc.indptr, self._csc.indices
        col = indices[indptr[feature]:indptr[feature + 1]]
        return np.intersect1d(rows, col, assume_unique=True), mask, w

    def cover_counts(self, handle, positive_class: Any) -> Tuple[int, int, int, int]:
        if self.y is None:
            raise ValueError("cover_counts needs labels (self.y)")
        rows, mask, w = handle
        if w is not None:
            pos = self.y == positive_class
            tp = float(w[rows][pos[rows]].sum())
            fp = float(w[rows].sum()) - tp
            n_pos = float(w[pos].sum())
            return tp, fp, n_pos - tp, float(w.sum()) - n_pos - fp
        tp = int(np.sum(self.y[rows] == positive_class))
        fp = int(rows.size) - tp
        y_scope = self.y if mask is None else self.y[mask]
        n_pos = int(np.sum(y_scope == positive_class))
        n_neg = y_scope.size - n_pos
        return tp, fp, n_pos - tp, n_neg - fp

    def cover_rows(self, handle) -> np.ndarray:
        rows = handle[0]
        out = np.zeros(self._n_samples, dtype=bool)
        out[rows] = True
        return out

    # -- batched counting: the covered rows of the CSR, in one product --

    def _row_sums(self, rows: np.ndarray, values: np.ndarray) -> np.ndarray:
        """``values`` (one row per entry of `rows`) summed per feature over
        `rows`: ``(n_features, m)``, one sparse product over their CSR rows
        -- cost grows with the covered rows' nonzeros."""
        return np.asarray(self._csr[rows].T @ values)

    def batch_cover_counts(self, handle, positive_class: Any, feature_indices: Sequence[int]):
        """`HorizontalBitmapRepresentation.batch_cover_counts` for sparse data:
        tp, fp and ``keeps_all`` of every feature in `feature_indices`, from
        one product over the covered rows (with a nonzero weight)."""
        if self.y is None:
            raise ValueError("batch_cover_counts needs labels (self.y)")
        rows, mask, w = handle
        if w is not None:
            rows = rows[w[rows] != 0]
        values = _class_value_columns(self.y[rows], positive_class, None if w is None else w[rows])
        return _counts_from_sums(self._row_sums(rows, values), len(rows), feature_indices, w is not None)

    def batch_cover_sums(self, handle, values: np.ndarray, feature_indices: Sequence[int]) -> np.ndarray:
        """`DataRepresentation.batch_cover_sums` as one product over the
        covered rows with a nonzero value."""
        values = np.asarray(values, dtype=float)
        rows = handle[0]
        rows = rows[(values[rows] != 0).any(axis=1)]
        return self._row_sums(rows, values[rows])[np.asarray(feature_indices, dtype=np.intp)].T

    def _reindexed(self, new_spec, source, is_complement):
        from scipy import sparse

        if is_complement is None:
            new = self._csc[:, source]
            return SparseDataRepresentation(new_spec, new, self.y, weights=self.weights)
        cols = []
        for j, s in enumerate(np.asarray(source)):
            col = self._csc[:, int(s)]
            if is_complement[j]:
                dense = np.ones((self._n_samples, 1), dtype=bool)
                dense[col.indices] = False
                cols.append(sparse.csc_matrix(dense))
            else:
                cols.append(col)
        return SparseDataRepresentation(new_spec, sparse.hstack(cols, format="csc"), self.y, weights=self.weights)

    @classmethod
    def from_scipy(cls, matrix, spec: DataSpec, y: Optional[np.ndarray] = None) -> "SparseDataRepresentation":
        """Wrap an existing `scipy.sparse` matrix (any format) that
        already matches `spec`'s feature order."""
        return cls(spec, matrix, y)

    @classmethod
    def from_boolean(cls, data: "HorizontalBitmapRepresentation") -> "SparseDataRepresentation":
        return cls(data.spec, data.to_csr(), data.y, weights=data.weights)

    @classmethod
    def from_dataframe(
        cls, df, label_col: Optional[str] = None, spec: Optional[DataSpec] = None, name: Optional[str] = None
    ) -> "SparseDataRepresentation":
        return cls.from_boolean(
            HorizontalBitmapRepresentation.from_dataframe(df, label_col=label_col, spec=spec, name=name)
        )

    @classmethod
    def from_xy(
        cls,
        X,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
        spec: Optional[DataSpec] = None,
        name: Optional[str] = None,
    ) -> "SparseDataRepresentation":
        return cls.from_boolean(
            HorizontalBitmapRepresentation.from_xy(
                X, y, feature_names=feature_names, threshold=threshold, spec=spec, name=name
            )
        )

    def __repr__(self) -> str:
        nnz = self._csr.nnz
        cells = self._n_samples * self.spec.n_features
        density = nnz / cells if cells else 0.0
        return (
            f"SparseDataRepresentation(spec={self.spec!r}, n_samples={self.n_samples}, "
            f"nnz={nnz}, density={density:.2f})"
        )


#: one word of a feature's bit-vector: 64 rows, row ``i`` at bit ``i % 64``
#: of word ``i // 64`` (little-endian, so the words' bytes are exactly
#: `numpy.packbits(..., bitorder="little")`'s)
_WORD = np.dtype("<u8")
_POPCOUNT_8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(words: np.ndarray) -> np.ndarray:
    """The number of set bits of `words` (`_WORD`s), summed over the last
    axis -- `numpy.bitwise_count` (a hardware popcount) where numpy has it
    (2.0 on), a byte lookup table otherwise."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(words).sum(axis=-1, dtype=np.int64)
    return _POPCOUNT_8[np.ascontiguousarray(words).view(np.uint8)].sum(axis=-1, dtype=np.int64)


class VerticalBitmapRepresentation(DataRepresentation):
    """The bit matrix of `HorizontalBitmapRepresentation`, packed the other way:
    one bit-vector per *feature*, over the examples (a ``(n_features,
    ceil(n_samples / 64))`` array of 64-bit words) -- the vertical bitmap
    layout of the MAFIA/SPAM frequent-pattern miners, where
    `HorizontalBitmapRepresentation` packs each example's features (horizontal)
    and `SparseDataRepresentation`/`NListRepresentation` keep per-feature
    lists of example indices (vertical, as lists).

    Covering a rule ANDs its features' bit-vectors -- ``n_samples / 64``
    words each, whatever the number of features; refining a rule by a
    feature is one more AND; counting is a popcount of the cover ANDed
    with a class's bit-vector. A search's batch counting
    (`batch_cover_counts`) ANDs the cover into the open features' words
    and popcounts them, over just the words the cover still touches. The
    memory is Boolean's -- one bit per cell, independent of the density --
    and nothing is unpacked for a search: `X` is built only for callers
    that want a plain matrix (external learners, row subsets).
    """

    def __init__(self, spec: DataSpec, X, y: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None):
        super().__init__(spec)
        try:
            from scipy.sparse import issparse
        except ImportError:  # pragma: no cover
            issparse = lambda _o: False  # noqa: E731
        if not issparse(X):
            X = np.asarray(X)
            if X.dtype != bool:
                X = X.astype(bool)
        if X.ndim != 2 or X.shape[1] != spec.n_features:
            raise ValueError(
                f"X has {X.shape[1] if X.ndim == 2 else '?'} columns but this DataSpec has "
                f"{spec.n_features} features"
            )
        n = X.shape[0]
        words = np.zeros((spec.n_features, (n + 63) // 64 * 8), dtype=np.uint8)
        if issparse(X):
            # set the bits of the nonzero cells directly -- no dense matrix
            coo = X.tocoo()
            nz = coo.data != 0
            rows, cols = coo.row[nz].astype(np.int64), coo.col[nz].astype(np.int64)
            np.bitwise_or.at(words, (cols, rows >> 3), (1 << (rows & 7)).astype(np.uint8))
        else:
            packed = np.packbits(X.T, axis=1, bitorder="little")
            words[:, :packed.shape[1]] = packed
        self._init_storage(words.view(_WORD), n, y, weights)

    def _init_storage(self, words: np.ndarray, n_samples: int, y, weights) -> None:
        self._words: np.ndarray = words
        self._n_samples: int = n_samples
        self.y: Optional[np.ndarray] = None if y is None else np.asarray(y)
        self._all: np.ndarray = self._pack(np.ones(n_samples, dtype=bool))
        self._class_words: dict = {}
        self._class_words_y: Optional[np.ndarray] = None
        self._set_weights(weights)

    @classmethod
    def _from_words(cls, spec: DataSpec, words: np.ndarray, n_samples: int, y, weights) -> "VerticalBitmapRepresentation":
        new = cls.__new__(cls)
        DataRepresentation.__init__(new, spec)
        new._init_storage(words, n_samples, y, weights)
        return new

    # -- packing ------------------------------------------------------------

    def _pack(self, rows: np.ndarray) -> np.ndarray:
        """A boolean row vector as a bit-vector of this data's words."""
        out = np.zeros((self._n_samples + 63) // 64 * 8, dtype=np.uint8)
        packed = np.packbits(rows, bitorder="little")
        out[:packed.size] = packed
        return out.view(_WORD)

    def _unpack(self, bits: np.ndarray) -> np.ndarray:
        """A bit-vector as a boolean row vector."""
        return np.unpackbits(bits.view(np.uint8), bitorder="little", count=self._n_samples).astype(bool)

    #: the most cells (features x rows) `_feature_sums` unpacks at once
    SUMS_BLOCK_CELLS = 1 << 22

    def _feature_sums(self, features: np.ndarray, rows: np.ndarray, values: np.ndarray) -> np.ndarray:
        """``values`` (one row per entry of `rows`, ascending) summed per
        feature in `features` over the rows where it holds: ``(len(features),
        m)``. When the rows are a large share of the data, a block of
        features is unpacked whole and multiplied; a few rows are gathered
        bit by bit instead."""
        out = np.zeros((len(features), values.shape[1]))
        if len(rows) == 0 or len(features) == 0:
            return out
        if len(rows) * 16 < self._n_samples:
            shift = (rows & 63).astype(np.uint64)
            bits = (self._words[np.ix_(features, rows >> 6)] >> shift) & np.uint64(1)
            return bits.astype(float) @ values
        every_row = len(rows) == self._n_samples
        step = max(1, self.SUMS_BLOCK_CELLS // self._n_samples)
        for start in range(0, len(features), step):
            bits = np.unpackbits(self._words[features[start:start + step]].view(np.uint8), axis=1,
                                 bitorder="little", count=self._n_samples)
            if not every_row:
                bits = bits[:, rows]
            out[start:start + step] = bits.astype(float) @ values
        return out

    def _class_bits(self, positive_class: Any) -> np.ndarray:
        """The bit-vector of the rows labelled `positive_class`, cached per
        class (and rebuilt if `y` is replaced)."""
        if self._class_words_y is not self.y:
            self._class_words = {}
            self._class_words_y = self.y
        bits = self._class_words.get(positive_class)
        if bits is None:
            bits = self._class_words[positive_class] = self._pack(self.y == positive_class)
        return bits

    # -- the primitives ---------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def coverage(self, rule: "Rule") -> np.ndarray:
        cov = self._all.copy()
        for lit in rule.conditions:
            cov &= self._words[lit.feature]
        return self._unpack(cov)

    def features_of(self, row: int) -> np.ndarray:
        bits = (self._words[:, row >> 6] >> np.uint64(row & 63)) & np.uint64(1)
        return np.flatnonzero(bits).astype(np.int64)

    @functools.cached_property
    def X(self) -> np.ndarray:
        """The unpacked ``(n_samples, n_features)`` bool matrix, built on
        first access; no search needs it."""
        bits = np.unpackbits(self._words.view(np.uint8), axis=1, bitorder="little", count=self._n_samples)
        return np.ascontiguousarray(bits.T, dtype=bool)

    # -- incremental coverage (search fast path) -----------------------
    #
    # A handle is (cov, scope, w): `cov` the bit-vector of the rows the
    # current rule covers within the scope, `scope` the bit-vector of the
    # rows in scope (or `None`: all), `w` the effective row weights (or
    # `None`), as for `HorizontalBitmapRepresentation`.

    def initial_cover(self, example_mask: Optional[np.ndarray] = None):
        support, w = self.scope(example_mask)
        if support is None:
            return self._all.copy(), None, None
        scope = self._pack(support)
        return scope.copy(), scope, w

    def refine_cover(self, handle, feature: int):
        cov, scope, w = handle
        return cov & self._words[feature], scope, w

    def cover_counts(self, handle, positive_class: Any) -> Tuple[int, int, int, int]:
        if self.y is None:
            raise ValueError("cover_counts needs labels (self.y)")
        cov, scope, w = handle
        if w is not None:
            return _weighted_counts(w, self._unpack(cov), self.y == positive_class)
        pos = self._class_bits(positive_class)
        in_scope = self._all if scope is None else scope
        tp = int(_popcount(cov & pos))
        fp = int(_popcount(cov)) - tp
        n_pos = int(_popcount(in_scope & pos))
        n_neg = int(_popcount(in_scope)) - n_pos
        return tp, fp, n_pos - tp, n_neg - fp

    def cover_rows(self, handle) -> np.ndarray:
        return self._unpack(handle[0])

    # -- batched counting ---------------------------------------------------

    #: the most words `batch_cover_counts` ANDs in one block (features x
    #: covered words), so its temporaries stay small on wide, long data
    BATCH_BLOCK_WORDS = 1 << 20

    def batch_cover_counts(self, handle, positive_class: Any, feature_indices: Sequence[int]):
        """`HorizontalBitmapRepresentation.batch_cover_counts` on bit-vectors:
        the cover's positive and negative rows ANDed into each open
        feature's words and popcounted, over just the words the cover
        still has a row in. With weights, a product over the covered rows'
        bits instead."""
        if self.y is None:
            raise ValueError("batch_cover_counts needs labels (self.y)")
        cov, scope, w = handle
        feats = np.asarray(feature_indices, dtype=np.intp)
        if w is not None:
            rows = np.flatnonzero(self._unpack(cov) & (w != 0))
            sums = self._feature_sums(feats, rows, _class_value_columns(self.y[rows], positive_class, w[rows]))
            return sums[:, 0], sums[:, 1], sums[:, 2] == len(rows)
        live = np.flatnonzero(cov)
        c = cov[live]
        p = c & self._class_bits(positive_class)[live]
        q = c ^ p
        tp = np.empty(len(feats), dtype=np.int64)
        fp = np.empty(len(feats), dtype=np.int64)
        step = max(1, self.BATCH_BLOCK_WORDS // max(len(live), 1))
        for start in range(0, len(feats), step):
            block = self._words[np.ix_(feats[start:start + step], live)]
            tp[start:start + step] = _popcount(block & p)
            fp[start:start + step] = _popcount(block & q)
        return tp, fp, tp + fp == int(_popcount(c))

    def batch_cover_sums(self, handle, values: np.ndarray, feature_indices: Sequence[int]) -> np.ndarray:
        """`DataRepresentation.batch_cover_sums` as one product over the
        covered rows with a nonzero value."""
        values = np.asarray(values, dtype=float)
        rows = np.flatnonzero(self._unpack(handle[0]) & (values != 0).any(axis=1))
        return self._feature_sums(np.asarray(feature_indices, dtype=np.intp), rows, values[rows]).T

    # -- derived representations -------------------------------------------

    def select_rows(self, mask: np.ndarray) -> "VerticalBitmapRepresentation":
        """The rows `mask` selects, re-packed a block of features at a time
        (never the whole unpacked matrix at once)."""
        rows = np.flatnonzero(mask) if np.asarray(mask).dtype == bool else np.asarray(mask, dtype=np.intp)
        n = len(rows)
        words = np.zeros((self.spec.n_features, (n + 63) // 64 * 8), dtype=np.uint8)
        step = max(1, (1 << 24) // max(self._n_samples, 1))
        for start in range(0, self.spec.n_features, step):
            bits = np.unpackbits(self._words[start:start + step].view(np.uint8), axis=1, bitorder="little",
                                 count=self._n_samples)[:, rows]
            packed = np.packbits(bits, axis=1, bitorder="little")
            words[start:start + step, :packed.shape[1]] = packed
        y = None if self.y is None else self.y[rows]
        w = None if self.weights is None else self.weights[rows]
        return self._from_words(self.spec, words.view(_WORD), n, y, w)

    def relabel(self, new_y: Any) -> "VerticalBitmapRepresentation":
        """The same bit-vectors (shared, not copied) with labels `new_y`."""
        new_y = np.asarray(new_y)
        if new_y.shape != (self._n_samples,):
            raise ValueError(f"new_y must have shape ({self._n_samples},), got {new_y.shape}")
        new = copy.copy(self)
        new.y = new_y
        new._class_words = {}
        new._class_words_y = new_y
        return new

    def _reindexed(self, new_spec, source, is_complement):
        words = self._words[np.asarray(source, dtype=np.intp)]
        if is_complement is not None and np.any(is_complement):
            words[is_complement] = ~words[is_complement] & self._all
        return self._from_words(new_spec, words, self._n_samples, self.y, self.weights)

    @classmethod
    def from_boolean(cls, data: "HorizontalBitmapRepresentation", chunk_cells: int = 1 << 22) -> "VerticalBitmapRepresentation":
        """Transpose `data`'s row-packed matrix a block of rows at a time,
        so nothing of the full unpacked size is formed."""
        n, k = data.n_samples, data.spec.n_features
        words = np.zeros((k, (n + 63) // 64 * 8), dtype=np.uint8)
        step = max(64, chunk_cells // max(k, 1) // 64 * 64)
        for start in range(0, n, step):
            block = np.unpackbits(data.packed_rows()[start:start + step], axis=1, count=k)
            packed = np.packbits(block.T, axis=1, bitorder="little")
            words[:, start // 8:start // 8 + packed.shape[1]] = packed
        return cls._from_words(data.spec, words.view(_WORD), n, data.y, data.weights)

    @classmethod
    def from_scipy(cls, matrix, spec: DataSpec, y: Optional[np.ndarray] = None) -> "VerticalBitmapRepresentation":
        return cls(spec, matrix, y)

    @classmethod
    def from_dataframe(
        cls, df, label_col: Optional[str] = None, spec: Optional[DataSpec] = None, name: Optional[str] = None
    ) -> "VerticalBitmapRepresentation":
        return cls.from_boolean(
            HorizontalBitmapRepresentation.from_dataframe(df, label_col=label_col, spec=spec, name=name)
        )

    @classmethod
    def from_xy(
        cls,
        X,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
        spec: Optional[DataSpec] = None,
        name: Optional[str] = None,
    ) -> "VerticalBitmapRepresentation":
        return cls.from_boolean(
            HorizontalBitmapRepresentation.from_xy(
                X, y, feature_names=feature_names, threshold=threshold, spec=spec, name=name
            )
        )

    def __repr__(self) -> str:
        return f"VerticalBitmapRepresentation(spec={self.spec!r}, n_samples={self.n_samples})"


#: Representation types, most to least generally preferred: what
#: `pyrulearn.data.io`'s loaders/`encode` and
#: `pyrulearn.experiments.runner.run_cv` build when no `representation=`
#: is given (`default_representation()`, the first entry), and how
#: `RuleLearner.ensure_representation` breaks ties when a learner's
#: `NATIVE_REPRESENTATIONS` names more than one type and the data matches
#: none of them. `VerticalBitmapRepresentation` first: on real data the
#: separate-and-conquer learners and PyLORD fit fastest on it, ENDER close
#: to the fastest, and it holds the least memory, one bit per cell
#: (`demos/representations_report.md`). `NListRepresentation` and
#: `PrePostNListRepresentation` next -- the same tree, the second with
#: pre/post codes instead of path bitmasks: less memory, a little slower
#: (on adult, 2.4 times less with negation, fits up to a fifth slower).
#: Read at call time, so reassigning this module attribute changes
#: the library-wide default.
REPRESENTATION_PREFERENCE_ORDER: Tuple[type, ...] = (
    VerticalBitmapRepresentation, NListRepresentation, PrePostNListRepresentation,
    HorizontalBitmapRepresentation, SparseDataRepresentation,
)


def default_representation() -> type:
    """The representation built when none is asked for: the first entry
    of `REPRESENTATION_PREFERENCE_ORDER`, looked up when called."""
    return REPRESENTATION_PREFERENCE_ORDER[0]
