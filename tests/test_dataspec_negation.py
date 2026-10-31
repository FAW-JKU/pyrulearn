"""`DataSpec.with_negations` / `DataSpec.without_negations` -- the schema
side of the data-level negation toggle."""

import numpy as np
import pytest

from pyrulearn.data.attributes import ExactlyOne, NominalGroup, NumericGroup
from pyrulearn.data import DataSpec, DataSpecBuilder


def _spec(negation):
    b = DataSpecBuilder(negation=negation)
    b.add_boolean("smoker")
    b.add_nominal("color", ["red", "green", "blue"])
    b.add_numeric("age", [30, 50])
    return b.build(name="t")


def test_without_negations_drops_exactly_the_negation_features():
    neg = _spec(True)
    pos, keep = neg.without_negations()
    # a numeric attribute's `<` features are no negation features: kept
    assert pos.feature_names == ["smoker", "color=red", "color=green", "color=blue",
                                 "age>=30", "age>=50", "age<30", "age<50"]
    assert not pos.has_negation_features
    assert [neg.feature_names[i] for i in keep] == pos.feature_names


def test_without_negations_regenerates_constraints():
    pos, _ = _spec(True).without_negations()
    kinds = {type(c) for c in pos.constraints}
    assert NominalGroup not in kinds          # -> ExactlyOne
    assert ExactlyOne in kinds
    assert NumericGroup in kinds              # numeric attributes unchanged


def test_without_negations_is_identity_when_there_is_nothing_to_drop():
    pos = _spec(False)
    same, keep = pos.without_negations()
    assert same is pos
    assert keep.tolist() == list(range(pos.n_features))
    plain = DataSpec(["a", "b", "c"])
    assert plain.without_negations()[0] is plain


def test_with_negations_adds_the_missing_partners():
    pos = _spec(False)
    neg, source, is_complement = pos.with_negations()
    assert neg.feature_names == _spec(True).feature_names
    # every added feature is flagged as the complement of an existing one
    for j, nm in enumerate(neg.feature_names):
        if nm in set(pos.feature_names):
            assert not is_complement[j] and pos.feature_name(source[j]) == nm
        else:
            assert is_complement[j]
            assert neg.feature_name(neg.negation_of(j)) == pos.feature_name(source[j])


def test_with_negations_reconstructs_the_data_matrix():
    rng = np.random.default_rng(0)
    base = rng.random((200, 3)) < 0.5              # smoker + 2 more bools
    b = DataSpecBuilder(negation=False)
    for n in ("smoker", "p", "q"):
        b.add_boolean(n)
    pos = b.build()

    neg, source, is_complement = pos.with_negations()
    reconstructed = np.where(is_complement, ~base[:, source], base[:, source])

    b2 = DataSpecBuilder(negation=True)
    for n in ("smoker", "p", "q"):
        b2.add_boolean(n)
    native = b2.build()
    native_X = np.empty((200, 6), dtype=bool)
    native_X[:, 0::2] = base
    native_X[:, 1::2] = ~base

    assert neg.feature_names == native.feature_names
    assert np.array_equal(reconstructed, native_X)


def test_with_negations_is_identity_when_every_feature_already_has_one():
    neg = _spec(True)
    same, source, is_complement = neg.with_negations()
    assert same.feature_names == neg.feature_names
    assert not is_complement.any()
    assert source.tolist() == list(range(neg.n_features))
