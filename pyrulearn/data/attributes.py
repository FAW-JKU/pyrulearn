"""
pyrulearn.data.attributes
=========================

Typed attributes (boolean / nominal / numeric / ordinal / set /
hierarchical / relational) and the machinery to:

1. Generate the derived *Boolean* features a rule actually conditions on
   (e.g. nominal attribute ``color`` with domain ``{red, green, blue}``
   becomes three Boolean features ``color=red``, ``color=green``,
   ``color=blue``; numeric attribute ``age`` with cut points
   ``{20, 30, 40}`` becomes ``age>=20``, ``age>=30``, ``age>=40`` and
   their complements ``age<20``, ``age<30``, ``age<40``; ordinal
   attribute ``size`` with values ``small < medium < large`` becomes
   ``size=small``, ..., ``size<=small``, ``size>small``, ...; set
   attribute ``tags`` becomes one membership feature per possible tag;
   hierarchical attribute ``region`` becomes one equality feature per
   tree node). ``Rule``/``Literal`` never need to change for this -- they
   still just index into a flat Boolean feature space; what's new is that
   some features now carry *provenance* (which attribute(s), which test)
   via ``FeatureSpec``.

2. Derive the **constraints** that automatically hold among features
   coming from the same attribute:
   - a nominal (or Boolean) attribute's features -- the ``attr=v``
     equality tests (exactly one True: mutually exclusive + exhaustive)
     plus, if negation was declared, the parallel ``attr!=v`` tests
     (each the exact negation of its ``=v``) -- are governed by one
     ``NominalGroup`` per attribute. ``ExactlyOne`` is the degenerate
     no-negation case, kept for that and for merge compatibility.
   - a numeric attribute's ``attr>=t`` features form a monotonic chain
     (``attr>=t_high`` implies ``attr>=t_low`` for ``t_low<=t_high``),
     and the parallel ``attr<t`` features are their exact complements.
     One ``NumericGroup`` per attribute owns both (and one more for the
     ``le_thresholds`` family, ``attr<=t``/``attr>t``);
     ``ThresholdChain`` is the plain chain it is built on.
   - an ordinal attribute's ``attr=v``/``attr<=v``/``attr>v`` (and,
     with negation, ``attr!=v``) features all describe which position
     of the ordered domain the value is at; one ``OrdinalGroup`` per
     attribute propagates them through those positions.
   - hierarchy sibling nodes (children of the same parent) are mutually
     exclusive but *not* assumed exhaustive (``MutuallyExclusive``), and
     each node implies its parent (``Implies``, chained up the tree).
   - set-valued membership features and relational (multi-attribute
     derived) features carry no automatic constraints -- their values are
     independent as far as this module can infer.

   All are ``Constraint`` subclasses implementing a shared
   ``propagate(fixed)`` interface: given a partial True/False assignment
   over their features, return whatever *additional* assignments are
   forced, or raise ``ValueError`` if the partial assignment is already
   contradictory. ``DataSpec.propagate`` (see ``dataspec.py``) iterates
   all constraints to a fixed point, which is what lets a ``Rule`` ask
   "is this internally consistent?" and "what does this rule imply that
   it doesn't say explicitly?".
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

# A partial True/False assignment over feature indices -- what
# `Constraint.propagate`/`DataSpec.propagate` take and return. Not a
# single feature's representation (that's `FeatureSpec`); a mapping of several
# feature indices to the truth values fixed for them.
FeatureValues = Dict[int, bool]


class AttributeType(Enum):
    """How an attribute's raw values turn into Boolean features.

    BOOLEAN, BINARY and NOMINAL differ in how many values they test and
    whether their value list is known to be complete:

    - BOOLEAN -- a single tested value, a presence flag (``smoker``);
      its complement ``not smoker`` is a negation feature, generated only
      with negation on. The data-mining literature's *asymmetric binary*
      attribute, e.g. an item in a market basket.
    - BINARY -- a closed set of exactly two values (``sex=male`` /
      ``sex=female``), both always generated and each the other's
      negation, so no ``!=`` columns are ever needed. A value outside the
      two is treated like a missing one. The literature's *symmetric
      binary* attribute.
    - NOMINAL -- a possibly open list of values: ``x!=v`` columns (with
      negation on) are kept even for two values, since an unknown value
      must satisfy every ``x!=v``.

    NUMERIC and ORDINAL are the two ordered types:

    - NUMERIC -- a continuous quantity, tested at chosen thresholds:
      ``x>=t`` and its complement ``x<t`` (and ``x<=t``/``x>t`` for
      the separate ``le_thresholds`` family). Both directions always,
      whatever the negation setting -- one direction alone would let a
      rule bound the attribute from one side only.
    - ORDINAL -- a few ordered values, ``low < medium < high`` or a small
      set of numbers (a count, a rating): ``x=v`` for every value, and
      ``x<=v``/``x>v`` for every value but the largest; with negation
      also ``x!=v``. Values compare by their position in the declared
      order (numerically, if every value is a number).
    """
    BOOLEAN = "boolean"
    BINARY = "binary"
    NOMINAL = "nominal"
    NUMERIC = "numeric"
    ORDINAL = "ordinal"
    SET = "set"
    HIERARCHICAL = "hierarchical"
    RELATIONAL = "relational"


class MissingStrategy(Enum):
    """How `pyrulearn.data.io.binarize` handles a missing raw value when
    converting data into a `DataSpec`'s Boolean feature space. Resolved
    per call via an explicit `binarize(missing_strategy=...)` argument,
    falling back to the target `DataSpec`'s own `missing_strategy`, then
    to `DataSpec.DEFAULT_MISSING_STRATEGY` -- the same explicit > stored
    > class-wide resolution chain as `Rule.to_string`'s `fmt`.
    """

    NEVER_COVERS = "never_covers"
    """Every feature derived from the missing attribute is False for
    that example -- it never satisfies *any* literal conditioning on it,
    positive or negated (a closed-world "we don't know" reading, not
    "the negation holds"). The default; needs nothing declared on the
    attribute."""

    MAJORITY = "majority"
    """Impute the missing raw value before evaluating: the column
    median for a NUMERIC attribute, the median value for an ORDINAL one
    (the lower of the two middle values for an even count, so always one
    of its values), the most frequent value (mode) otherwise."""

    RANDOM = "random"
    """Impute the missing raw value with another, uniformly randomly
    chosen non-missing example's value from the same column."""

    SEPARATE = "separate"
    """Route the missing value into a dedicated feature, treating
    "missing" as its own value alongside the attribute's declared
    domain. Requires the attribute to have been built with
    `missing_name=` (`DataSpecBuilder.add_nominal`/`add_numeric`/
    `add_ordinal`) -- raises if no such feature was declared."""


@dataclass(frozen=True)
class Hierarchy:
    """A tree over a HIERARCHICAL attribute's values (e.g. continent >
    country > state > city). `edges` is a tuple of (node, parent) pairs,
    with `parent` `None` for roots -- stored as a flat tuple (rather than
    a dict) so `Hierarchy` stays hashable/immutable like the rest of this
    module.
    """
    edges: Tuple[Tuple[Any, Optional[Any]], ...]

    @property
    def nodes(self) -> Tuple[Any, ...]:
        return tuple(n for n, _ in self.edges)

    def parent_of(self, node: Any) -> Optional[Any]:
        for n, p in self.edges:
            if n == node:
                return p
        raise KeyError(node)

    def ancestors(self, node: Any) -> Tuple[Any, ...]:
        """Ancestors of `node`, nearest first, root last."""
        chain = []
        cur = self.parent_of(node)
        while cur is not None:
            chain.append(cur)
            cur = self.parent_of(cur)
        return tuple(chain)

    @classmethod
    def from_nested(cls, tree: Dict[Any, Any]) -> "Hierarchy":
        """Build from a nested dict of the shape
        ``{"Europe": {"France": {"Paris": {}, "Lyon": {}}, "Germany": {}}, "Asia": {}}``
        -- each key a node, each value the dict of its children (``{}``
        for a leaf).
        """
        edges: list = []

        def walk(node, subtree, par):
            edges.append((node, par))
            for child, grandchildren in subtree.items():
                walk(child, grandchildren, node)

        for root, subtree in tree.items():
            walk(root, subtree, None)
        return cls(tuple(edges))


@dataclass(frozen=True)
class Attribute:
    """A source attribute a rule's conditions may ultimately be about.

    `domain` is the set of nominal category values for NOMINAL
    attributes, the two values of a BINARY attribute, the ascending tuple of ``>=`` threshold cut points
    actually used to binarize a NUMERIC attribute (not the attribute's
    full continuous range -- just the cuts you chose to test against),
    the values of an ORDINAL attribute in ascending order, or the set of
    possible values for a SET attribute. `le_domain` is a
    NUMERIC attribute's *separate* ascending tuple of ``<=`` cut points,
    if any were given via `add_numeric`'s `le_thresholds=` -- `None`
    (not just empty) when none were, same as every other unused
    optional field here; `None` for every other attribute type.
    `hierarchy` is set for HIERARCHICAL attributes (and `domain` holds
    all of its node values, flattened). `sources` is set for RELATIONAL
    attributes: the names of the other attributes this derived
    attribute reads.

    `missing_name` is the value used for this attribute's dedicated
    "value is missing" feature, if one was declared (via `add_nominal`'s
    or `add_numeric`'s `missing_name=`) -- `None` if none was. For
    NOMINAL/HIERARCHICAL attributes it rides in the same exhaustive
    `ExactlyOne`/sibling group as the declared domain (missing is just
    one more value); for NUMERIC it's a standalone feature outside the
    `ThresholdChain` -- a numeric value can be simultaneously "missing"
    and fail every `>=` test, so deliberately no constraint links the
    two; for ORDINAL it's one more position of the `OrdinalGroup`,
    outside the order (no ``<=``/``>`` test holds there). `missing_values` are raw values (besides `None`/NaN) that
    should be treated as missing when reading data for this attribute,
    e.g. ``("?",)`` for the common ARFF/UCI convention.
    """
    name: str
    type: AttributeType
    domain: Optional[Tuple[Any, ...]] = None
    le_domain: Optional[Tuple[Any, ...]] = None
    hierarchy: Optional[Hierarchy] = None
    sources: Optional[Tuple[str, ...]] = None
    missing_name: Optional[Any] = None
    missing_values: Tuple[Any, ...] = ()
    #: whether negation features (``not name`` / ``name!=v``) were
    #: generated for this attribute -- BOOLEAN/NOMINAL/ORDINAL only (a
    #: BINARY attribute's two values are each other's negation either
    #: way, and a NUMERIC attribute always has both ``>=`` and ``<``).
    negation: bool = False



@dataclass(frozen=True)
class FeatureSpec:
    """Provenance for one Boolean feature (one bit position in a Rule's
    bitmask): which attribute(s) it derives from and what test it encodes.

    `op` is ``None`` for a plain, attribute-less Boolean feature (the
    original use case -- the feature *is* the literal, no attribute
    behind it), ``"not"`` for the negation of a Boolean attribute
    (renders as ``not name``), ``"=="``/``"!="`` for a nominal or
    hierarchical-node / ordinal value equality / inequality test,
    ``">="``/``"<"`` for a numeric threshold test and its exact
    complement, ``"<="``/``">"`` for the separate ``le_thresholds``
    family and for an ordinal attribute's order tests (see
    `DataSpecBuilder.add_numeric` -- ``<=`` is *not* the complement of
    ``>=``, they disagree at ``==t``), ``"has"`` for a set-valued
    membership test, or ``"expr"`` for a relational feature derived from
    multiple attributes. Only ``"not"``/``"!="`` features are negation
    features, generated only when the attribute was declared with
    negation on (`DataSpecBuilder`'s `negation=`); ``"<"``/``">"`` are
    always there. `attribute`/`value` are ``None`` in the plain-Boolean
    case; `attributes`/`expression` are only set for ``"expr"``.
    """
    index: int
    name: str
    attribute: Optional[str] = None
    attributes: Optional[Tuple[str, ...]] = None
    op: Optional[str] = None
    value: Any = None
    expression: Optional[str] = None


def evaluate_feature(spec: FeatureSpec, value: Any) -> Optional[bool]:
    """Evaluate one derived feature's test against a raw attribute value
    (e.g. one cell from a CSV/ARFF row) -- the inverse of what
    `FeatureSpec` records for display: given the raw value, what should
    this Boolean feature be?

    Returns ``None`` for a missing value (``None`` or NaN) so callers can
    decide how to handle it -- this module has no missing-value
    data of its own. For a plain Boolean feature (``op is
    None``), the raw value is just coerced with ``bool()``; this works
    for actual bools/0/1/numeric values but is ambiguous for strings like
    ``"False"`` (truthy in Python) -- convert such columns to real
    booleans before calling this.

    ``op == "expr"`` (relational features) has no evaluation rule of its
    own, only a display `expression` (see `DataSpecBuilder.add_relational`):
    its column holds the precomputed truth value, which is coerced with
    ``bool()`` like a plain Boolean feature's.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if spec.op is None:
        return bool(value)
    if spec.op == "not":
        return not bool(value)
    if spec.op == "==":
        return value == spec.value
    if spec.op == "!=":
        return value != spec.value
    if spec.op == ">=":
        return value >= spec.value
    if spec.op == ">":
        return value > spec.value
    if spec.op == "<":
        return value < spec.value
    if spec.op == "<=":
        return value <= spec.value
    if spec.op == "has":
        return spec.value in value
    if spec.op == "expr":
        return bool(value)  # the column is the precomputed value
    raise ValueError(f"Unknown FeatureSpec op {spec.op!r}")


class Constraint(ABC):
    """A relationship among a fixed set of feature indices, all derived
    from the same attribute. Subclasses implement `propagate`.
    """

    def __init__(self, feature_indices: Sequence[int]):
        self.feature_indices: Tuple[int, ...] = tuple(feature_indices)

    @abstractmethod
    def propagate(self, fixed: FeatureValues) -> FeatureValues:
        """Given a dict of feature_index -> True/False for *some subset*
        of `self.feature_indices` (and possibly others, which this
        constraint ignores), return a dict of *additional* forced
        assignments (features not already keyed in `fixed`). Must not
        mutate `fixed`. Raises ValueError if `fixed` already contains a
        combination this constraint rules out.
        """
        raise NotImplementedError


class ExactlyOne(Constraint):
    """Mutual exclusion + exhaustiveness for a nominal attribute's
    equality tests: exactly one of `feature_indices` is True.

    Models: "if Attribute=Value is true, every feature testing a
    *different* value of that attribute must be false" -- and the
    converse (if all-but-one are ruled out, the remaining one is forced
    true).
    """

    def propagate(self, fixed: FeatureValues) -> FeatureValues:
        true_ones = [i for i in self.feature_indices if fixed.get(i) is True]
        if len(true_ones) > 1:
            raise ValueError(
                f"Multiple values fixed True in mutually-exclusive group "
                f"{self.feature_indices}: {true_ones}"
            )
        forced: FeatureValues = {}
        if true_ones:
            t = true_ones[0]
            for i in self.feature_indices:
                if i == t:
                    continue
                if i not in fixed:
                    forced[i] = False
        else:
            false_ones = [i for i in self.feature_indices if fixed.get(i) is False]
            remaining = [i for i in self.feature_indices if i not in false_ones]
            if not remaining:
                raise ValueError(
                    f"All values fixed False in exhaustive group "
                    f"{self.feature_indices}, but exactly one must hold"
                )
            if len(remaining) == 1:
                forced[remaining[0]] = True
        return forced


class ThresholdChain(Constraint):
    """Monotonic chain for a numeric attribute's threshold tests
    (``attr >= t``). `feature_indices` must be given in **ascending
    threshold order**.

    Models: "if Attribute>=Value is true, every feature testing a
    *smaller* threshold of that attribute must also be true" -- and the
    converse (a False at a low threshold forces False at every higher
    threshold).
    """

    def propagate(self, fixed: FeatureValues) -> FeatureValues:
        idxs = self.feature_indices
        true_pos = [p for p, i in enumerate(idxs) if fixed.get(i) is True]
        false_pos = [p for p, i in enumerate(idxs) if fixed.get(i) is False]
        forced: FeatureValues = {}

        if true_pos:
            max_true = max(true_pos)
            for p in range(0, max_true):
                i = idxs[p]
                if fixed.get(i) is False:
                    raise ValueError(
                        f"Contradiction in threshold chain {idxs}: position {p} "
                        f"must be True (implied by a higher threshold) but is fixed False"
                    )
                if i not in fixed:
                    forced[i] = True

        if false_pos:
            min_false = min(false_pos)
            for p in range(min_false + 1, len(idxs)):
                i = idxs[p]
                if fixed.get(i) is True:
                    raise ValueError(
                        f"Contradiction in threshold chain {idxs}: position {p} "
                        f"must be False (implied by a lower threshold) but is fixed True"
                    )
                if i not in fixed:
                    forced[i] = False

        return forced


class MutuallyExclusive(Constraint):
    """At most one of `feature_indices` is True -- unlike `ExactlyOne`,
    does *not* force the last remaining feature True when all others are
    ruled out (the group isn't assumed exhaustive). Used for sibling
    nodes in a `Hierarchy`: two children of the same parent can't both
    hold, but "none of the known children" is a valid state (the example
    just isn't known/specified at that granularity).
    """

    def propagate(self, fixed: FeatureValues) -> FeatureValues:
        true_ones = [i for i in self.feature_indices if fixed.get(i) is True]
        if len(true_ones) > 1:
            raise ValueError(
                f"Multiple values fixed True in mutually-exclusive group "
                f"{self.feature_indices}: {true_ones}"
            )
        forced: FeatureValues = {}
        if true_ones:
            t = true_ones[0]
            for i in self.feature_indices:
                if i != t and i not in fixed:
                    forced[i] = False
        return forced


class Implies(Constraint):
    """One-directional implication between two features: `antecedent`
    True forces `consequent` True, and (contrapositive) `consequent`
    False forces `antecedent` False. Used for `Hierarchy` parent/child
    edges (e.g. ``city=Paris`` implies ``country=France``); chains of
    these compose via repeated `propagate` to and up a whole hierarchy
    path.
    """

    def __init__(self, antecedent: int, consequent: int):
        super().__init__((antecedent, consequent))
        self.antecedent = antecedent
        self.consequent = consequent

    def propagate(self, fixed: FeatureValues) -> FeatureValues:
        a, c = fixed.get(self.antecedent), fixed.get(self.consequent)
        if a is True and c is False:
            raise ValueError(
                f"Contradiction: feature {self.antecedent} implies feature "
                f"{self.consequent}, but {self.antecedent} is fixed True and "
                f"{self.consequent} is fixed False"
            )
        forced: FeatureValues = {}
        if a is True and self.consequent not in fixed:
            forced[self.consequent] = True
        if c is False and self.antecedent not in fixed:
            forced[self.antecedent] = False
        return forced


class NominalGroup(Constraint):
    """A nominal (or Boolean) attribute's full derived-feature logic in
    one constraint: `eq` are the value-equality features (``attr=v``),
    exactly one of which is True (mutually exclusive *and* exhaustive,
    like `ExactlyOne`); `ne`, if non-empty, are the value-inequality
    features (``attr!=v``) parallel to `eq` -- ``ne[k]`` is the exact
    negation of ``eq[k]``, True iff ``eq[k]`` is False. `ne` is empty
    when the attribute was declared without negation, in which case this
    behaves exactly as `ExactlyOne`.

    `propagate` handles both directions: a fixed ``eq[j]``/``ne[j]``
    forces every sibling, and once all-but-one value is ruled out the
    last one is forced True.
    """

    def __init__(self, eq: Sequence[int], ne: Sequence[int] = ()):
        eq = tuple(eq)
        ne = tuple(ne)
        if ne and len(ne) != len(eq):
            raise ValueError("NominalGroup: `ne` must be empty or parallel to `eq`")
        super().__init__(eq + ne)
        self.eq = eq
        self.ne = ne

    def propagate(self, fixed: FeatureValues) -> FeatureValues:
        n = len(self.eq)
        eq_true = [False] * n
        eq_false = [False] * n
        for k in range(n):
            ev = fixed.get(self.eq[k])
            nv = fixed.get(self.ne[k]) if self.ne else None
            if ev is True or nv is False:
                eq_true[k] = True
            if ev is False or nv is True:
                eq_false[k] = True
            if eq_true[k] and eq_false[k]:
                raise ValueError(
                    f"Contradiction in nominal group {self.feature_indices}: "
                    f"value at position {k} fixed both True and False"
                )

        trues = [k for k in range(n) if eq_true[k]]
        if len(trues) > 1:
            raise ValueError(
                f"Multiple values fixed True in nominal group {self.feature_indices}: {trues}"
            )

        winner: Optional[int] = trues[0] if trues else None
        if winner is None:
            not_false = [k for k in range(n) if not eq_false[k]]
            if not not_false:
                raise ValueError(
                    f"All values fixed False in exhaustive nominal group {self.feature_indices}"
                )
            if len(not_false) == 1:
                winner = not_false[0]

        forced: FeatureValues = {}
        for k in range(n):
            if winner is not None:
                truth: Optional[bool] = (k == winner)
            elif eq_false[k]:
                truth = False
            else:
                truth = None
            if truth is None:
                continue
            if self.eq[k] not in fixed:
                forced[self.eq[k]] = truth
            if self.ne and self.ne[k] not in fixed:
                forced[self.ne[k]] = not truth
        return forced


class NumericGroup(Constraint):
    """A numeric attribute's full derived-feature logic: `ge` are the
    ``attr>=t`` features **in ascending threshold order** (a monotonic
    chain, as `ThresholdChain`); `lt`, if non-empty, are the ``attr<t``
    features parallel to `ge` -- ``lt[k]`` is the exact complement of
    ``ge[k]`` (they partition the non-missing values, disagreeing
    nowhere, unlike ``>=`` vs ``<=``). `lt` is empty when the attribute
    was declared without negation, in which case this behaves exactly as
    `ThresholdChain`.
    """

    def __init__(self, ge: Sequence[int], lt: Sequence[int] = ()):
        ge = tuple(ge)
        lt = tuple(lt)
        if lt and len(lt) != len(ge):
            raise ValueError("NumericGroup: `lt` must be empty or parallel to `ge`")
        super().__init__(ge + lt)
        self.ge = ge
        self.lt = lt
        self._chain = ThresholdChain(ge)

    def propagate(self, fixed: FeatureValues) -> FeatureValues:
        # fold any known `lt[k]` into an equivalent `ge[k]` assignment,
        # then run the plain `>=` threshold chain, then mirror back.
        ge_view = dict(fixed)
        for k, li in enumerate(self.lt):
            v = fixed.get(li)
            if v is None:
                continue
            gi = self.ge[k]
            implied = not v
            if ge_view.get(gi) is not None and ge_view[gi] != implied:
                raise ValueError(
                    f"Contradiction in numeric group {self.feature_indices}: "
                    f"feature {gi} and its complement {li} both fixed"
                )
            ge_view[gi] = implied

        chain_forced = self._chain.propagate(ge_view)
        forced: FeatureValues = {}
        for k, gi in enumerate(self.ge):
            val = chain_forced.get(gi, ge_view.get(gi))
            if val is None:
                continue
            if gi not in fixed:
                forced[gi] = val
            if self.lt and self.lt[k] not in fixed:
                forced[self.lt[k]] = not val
        return forced


def is_numeric_domain(domain: Optional[Sequence[Any]]) -> bool:
    """Whether every value of an ORDINAL attribute's `domain` is a number
    (not a bool) -- such an attribute compares raw values numerically,
    so a value outside the domain still satisfies the right ``<=``/``>``
    tests; any other domain compares by position in its declared order."""
    import numbers
    return bool(domain) and all(
        isinstance(v, numbers.Number) and not isinstance(v, bool) for v in domain
    )


class OrdinalGroup(Constraint):
    """An ordinal attribute's full derived-feature logic in one
    constraint. The attribute's value is at exactly one of `n_positions`
    positions -- its ``k`` values in order, plus, if declared, a
    ``missing_name`` position outside the order -- and every feature is
    true on a fixed set of positions: ``x=v_i`` on ``{i}``, ``x!=v_i``
    on every value but ``i``, ``x<=v_i`` on ``{0..i}``, ``x>v_i`` on
    ``{i+1..k-1}``. `masks` maps each feature index to that set, as a
    bitmask.

    `propagate` intersects the positions the fixed features leave
    possible (a True feature its own set, a False one the complement)
    and forces every other feature that is true on all of them, or on
    none -- so ``x=v_2`` forces ``x<=v_3`` and ``x>v_1``, and ``x<=v_3``
    with ``x>v_2`` forces ``x=v_3``. An empty intersection is a
    contradiction. Like `NominalGroup`, it assumes the value is at one
    of the positions (exhaustive), which holds on the data the domain
    was taken from (the training data, when `build_dataspec` infers
    it). A missing value without a ``missing_name`` position falsifies
    every feature; so does a value outside a non-numeric order. An
    unseen number satisfies the ``<=``/``>`` tests it passes and no
    ``=``, so what the group derives (``x>1`` and ``x<=3`` of 0, 1, 3
    imply ``x=3``) need not hold for it -- it only matters for implied
    conditions and their display, never for which examples a rule
    covers, which the data decides.
    """

    def __init__(self, masks: Dict[int, int], n_positions: int):
        super().__init__(sorted(masks))
        self.masks: Dict[int, int] = dict(masks)
        self.n_positions = n_positions
        self._full = (1 << n_positions) - 1

    def propagate(self, fixed: FeatureValues) -> FeatureValues:
        possible = self._full
        for i, m in self.masks.items():
            v = fixed.get(i)
            if v is True:
                possible &= m
            elif v is False:
                possible &= ~m & self._full
        if not possible:
            raise ValueError(
                f"Contradiction in ordinal group {self.feature_indices}: "
                "no value satisfies the fixed features"
            )
        forced: FeatureValues = {}
        for i, m in self.masks.items():
            if i in fixed:
                continue
            if not possible & ~m:
                forced[i] = True
            elif not possible & m:
                forced[i] = False
        return forced
