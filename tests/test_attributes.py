import numpy as np
import pytest
from pyrulearn.data import DataSpec, DataSpecBuilder
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.data.attributes import (
    Attribute, AttributeType, ExactlyOne, Implies, MutuallyExclusive, NominalGroup,
    NumericGroup, ThresholdChain, evaluate_feature, FeatureSpec,
)
from pyrulearn.rule import Rule, Literal


def test_nominal_group_propagation_both_directions():
    # eq = color=red/green/blue (idx 0-2), ne = color!=red/green/blue (idx 3-5)
    g = NominalGroup([0, 1, 2], [3, 4, 5])
    assert g.propagate({0: True}) == {1: False, 2: False, 3: False, 4: True, 5: True}
    assert g.propagate({3: True}) == {0: False}                       # can't pick a winner yet
    assert g.propagate({3: True, 4: True}) == {0: False, 1: False, 2: True, 5: False}
    with pytest.raises(ValueError):
        g.propagate({3: True, 4: True, 5: True})                      # exhaustive violated
    with pytest.raises(ValueError):
        g.propagate({0: True, 1: True})                               # mutual exclusion violated
    # ne empty -> behaves exactly like ExactlyOne
    assert NominalGroup([0, 1, 2]).propagate({0: True}) == {1: False, 2: False}


def test_numeric_group_complement_and_chain():
    # ge = age>=30, age>=40 (idx 0-1); lt = age<30, age<40 (idx 2-3)
    g = NumericGroup([0, 1], [2, 3])
    assert g.propagate({1: True}) == {0: True, 2: False, 3: False}     # >=40 -> >=30, both <t false
    assert g.propagate({2: True}) == {0: False, 1: False, 3: True}     # <30 -> not>=30 -> not>=40 -> <40
    with pytest.raises(ValueError):
        g.propagate({0: True, 2: True})                               # age>=30 and age<30
    # lt empty -> behaves like ThresholdChain
    assert NumericGroup([0, 1]).propagate({1: True}) == {0: True}


def test_evaluate_feature_negation_ops_are_conservative_on_missing():
    ne = FeatureSpec(0, "color!=red", attribute="color", op="!=", value="red")
    lt = FeatureSpec(1, "age<30", attribute="age", op="<", value=30)
    nb = FeatureSpec(2, "not smoker", attribute="smoker", op="not")
    assert evaluate_feature(ne, "green") is True and evaluate_feature(ne, "red") is False
    assert evaluate_feature(lt, 25) is True and evaluate_feature(lt, 30) is False
    assert evaluate_feature(nb, False) is True and evaluate_feature(nb, True) is False
    assert evaluate_feature(ne, None) is None      # missing -> None -> binarize maps to 0
    assert evaluate_feature(lt, float("nan")) is None
    assert evaluate_feature(nb, None) is None


def test_builder_negation_generates_paired_features_and_group_constraints():
    b = DataSpecBuilder(negation=True)
    s = b.add_boolean("smoker")
    c = b.add_nominal("color", ["red", "green", "blue"])
    a = b.add_numeric("age", [30, 40])
    ds = b.build()

    assert ds.feature_names == [
        "smoker", "not smoker",
        "color=red", "color=green", "color=blue", "color!=red", "color!=green", "color!=blue",
        "age>=30", "age>=40", "age<30", "age<40",
    ]
    assert int(s) == 0 and s.negative == 1 and s.all == [0, 1]
    assert dict(c) == {"red": 2, "green": 3, "blue": 4}
    assert c.negative == {"red": 5, "green": 6, "blue": 7}
    assert dict(a) == {30: 8, 40: 9} and a.negative == {30: 10, 40: 11}
    assert [type(x).__name__ for x in ds.constraints] == \
        ["MutuallyExclusive", "NominalGroup", "NumericGroup"]

    assert ds.negation_of("color=red") == ds.feature_index("color!=red")
    assert ds.negation_of("color!=red") == ds.feature_index("color=red")
    assert ds.negation_of("age>=30") == ds.feature_index("age<30")
    assert ds.negation_of("smoker") == ds.feature_index("not smoker")
    # color=red True propagates through the whole group
    closure = ds.propagate({c["red"]: True})
    assert closure[c["green"]] is False and closure[c.negative["red"]] is False
    assert closure[c.negative["green"]] is True


def test_builder_negation_per_attribute_override():
    b = DataSpecBuilder(negation=True)
    b.add_nominal("keep", ["a", "b"])
    b.add_nominal("drop", ["x", "y"], negation=False)
    ds = b.build()
    assert "keep!=a" in ds.feature_names and "drop!=x" not in ds.feature_names
    assert ds.negation_of("keep=a") is not None
    assert ds.negation_of("drop=x") is None


def test_builder_generates_features_and_constraints():
    b = DataSpecBuilder(negation=False)
    smoker_idx = b.add_boolean("smoker")
    color_idx = b.add_nominal("color", ["red", "green", "blue"])
    age_idx = b.add_numeric("age", [20, 30, 40])
    ds = b.build()

    # without negation, still both directions for a numeric attribute
    assert ds.feature_names == [
        "smoker", "color=red", "color=green", "color=blue",
        "age>=20", "age>=30", "age>=40", "age<20", "age<30", "age<40",
    ]
    assert smoker_idx == 0
    assert color_idx == {"red": 1, "green": 2, "blue": 3}
    assert age_idx == {20: 4, 30: 5, 40: 6}
    assert age_idx.negative == {20: 7, 30: 8, 40: 9}
    assert len(ds.constraints) == 2  # one ExactlyOne, one NumericGroup
    assert isinstance(ds.constraints[0], ExactlyOne)
    assert isinstance(ds.constraints[1], NumericGroup)


def test_nominal_mutual_exclusion_propagation():
    b = DataSpecBuilder()
    idx = b.add_nominal("color", ["red", "green", "blue"])
    ds = b.build()

    # color=red True forces green/blue False
    closure = ds.propagate({idx["red"]: True})
    assert closure[idx["green"]] is False
    assert closure[idx["blue"]] is False

    # ruling out two of three forces the third True
    closure2 = ds.propagate({idx["red"]: False, idx["green"]: False})
    assert closure2[idx["blue"]] is True

    # contradiction: two different values both True
    try:
        ds.propagate({idx["red"]: True, idx["blue"]: True})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_numeric_threshold_chain_propagation():
    b = DataSpecBuilder()
    idx = b.add_numeric("age", [20, 30, 40])
    ds = b.build()

    # age>=40 True forces age>=30 and age>=20 True
    closure = ds.propagate({idx[40]: True})
    assert closure[idx[30]] is True
    assert closure[idx[20]] is True

    # age>=20 False forces age>=30 and age>=40 False
    closure2 = ds.propagate({idx[20]: False})
    assert closure2[idx[30]] is False
    assert closure2[idx[40]] is False

    # contradiction: low threshold False but high threshold True
    try:
        ds.propagate({idx[20]: False, idx[40]: True})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_add_numeric_without_le_thresholds_is_unchanged():
    # le_thresholds defaults to () -- a complete no-op: the same plain
    # ``{threshold: '>=' index}`` dict, with the ``<`` complements
    b = DataSpecBuilder(negation=False)
    idx = b.add_numeric("age", [20, 30, 40])
    ds = b.build()

    assert isinstance(idx, dict)
    assert idx == {20: 0, 30: 1, 40: 2}
    assert idx.le is None
    assert ds.feature_names == ["age>=20", "age>=30", "age>=40", "age<20", "age<30", "age<40"]
    assert len(ds.constraints) == 1


def test_add_numeric_le_thresholds_creates_separate_family():
    b = DataSpecBuilder(negation=False)
    result = b.add_numeric("age", [30, 40], le_thresholds=[10, 20])
    ds = b.build()

    # NumericFeatureIndices, but still positionally unpackable
    assert result.ge == {30: 0, 40: 1}
    ge_idx, le_idx = result
    assert ge_idx == {30: 0, 40: 1}
    assert set(le_idx) == {10, 20}  # indices assigned in descending order internally, values don't matter here
    assert set(ds.feature_names) == {"age>=30", "age>=40", "age<30", "age<40",
                                     "age<=10", "age<=20", "age>10", "age>20"}
    assert set(le_idx.negative) == {10, 20}  # the > complements
    # one NumericGroup per family
    assert len(ds.constraints) == 2


def test_le_threshold_chain_propagation_mirrors_ge():
    # <=10 True (x<=10) must force <=20 True (x<=20, a looser bound);
    # <=20 False (x>20) must force <=10 False (x>10, an even tighter miss)
    # -- the reused, unmodified ThresholdChain gets this right because the
    # <= family's feature indices are fed to it in descending threshold
    # order, mirroring <='s reversed monotonicity
    b = DataSpecBuilder()
    ge_idx, le_idx = b.add_numeric("age", [], le_thresholds=[10, 20, 30])
    ds = b.build()

    closure = ds.propagate({le_idx[10]: True})
    assert closure[le_idx[20]] is True
    assert closure[le_idx[30]] is True

    closure2 = ds.propagate({le_idx[20]: False})
    assert closure2[le_idx[10]] is False

    # age<=10 True (age<=10) together with age<=30 False (age>30) is the
    # actual contradiction -- age<=10 and age>30 can't both hold
    try:
        ds.propagate({le_idx[10]: True, le_idx[30]: False})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ge_and_le_families_have_no_cross_reasoning():
    # deliberate: age>=30 and age<=25 together is never flagged
    # inconsistent (no ThresholdBound-style cross-family constraint) --
    # a rule stating both simply never covers any real example instead,
    # which is fine for rules read in from an external source
    b = DataSpecBuilder(negation=False)
    ge_idx, le_idx = b.add_numeric("age", [30], le_thresholds=[25])
    ds = b.build()

    closure = ds.propagate({ge_idx[30]: True, le_idx[25]: True})
    # only each family's own complements are forced, nothing across, no raise
    assert closure == {ge_idx[30]: True, le_idx[25]: True,
                       ge_idx.negative[30]: False, le_idx.negative[25]: False}


def test_ge_and_le_at_same_value_behaves_as_equality():
    # giving the *same* value to both ge_thresholds and le_thresholds is
    # how to represent attribute==value as a conjunction -- is_consistent
    # doesn't flag it (unlike a genuine ge>le contradiction, it's not
    # inconsistent), and at the data level only rows exactly equal to the
    # value satisfy both literals
    import pandas as pd
    from pyrulearn.data.io import binarize
    from pyrulearn.data import HorizontalBitmapRepresentation
    from pyrulearn.rule import Rule, Literal

    b = DataSpecBuilder()
    ge_idx, le_idx = b.add_numeric("age", [30], le_thresholds=[30])
    ds = b.build()

    r = Rule([ge_idx[30], le_idx[30]], target="pos", dataspec=ds)
    assert r.is_consistent()

    df = pd.DataFrame({"age": [28, 29, 30, 31, 32]})
    X = binarize(ds, df)
    rep = HorizontalBitmapRepresentation(ds, X, np.array(["?"] * 5))
    covered = r.covers_data(rep)
    assert list(covered) == [False, False, True, False, False]


def test_le_feature_binarizes_directly_from_raw_numeric_column():
    # the whole point: unlike an atomic Boolean workaround, a proper <=
    # family derives correctly from a plain raw numeric column
    import pandas as pd
    from pyrulearn.data.io import binarize

    b = DataSpecBuilder()
    ge_idx, le_idx = b.add_numeric("income", [], le_thresholds=[19992])
    ds = b.build()

    df = pd.DataFrame({"income": [10000, 19992, 20000, 30000]})
    X = binarize(ds, df)
    assert list(X[:, le_idx[19992]]) == [True, True, False, False]


def test_le_literal_rendering():
    b = DataSpecBuilder(negation=True)
    ge_idx, le_idx = b.add_numeric("income", [], le_thresholds=[19992])
    ds = b.build()

    r_pos = Rule([le_idx[19992]], target="pos", dataspec=ds)
    r_neg = Rule([ds.negation_of(le_idx[19992])], target="neg", dataspec=ds)
    assert r_pos.to_string("logic") == "income <= 19992 → pos"
    assert r_neg.to_string("logic") == "income > 19992 → neg"
    assert r_pos.to_string("prolog") == "pos(X) :- income(X, V1), V1 <= 19992."
    assert r_neg.to_string("prolog") == "neg(X) :- income(X, V1), V1 > 19992."


def test_rule_consistency_and_implied_conditions():
    b = DataSpecBuilder(negation=True)
    color_idx = b.add_nominal("color", ["red", "green", "blue"])
    age_idx = b.add_numeric("age", [20, 30, 40])
    ds = b.build()

    # consistent rule: color=red and age>=30
    r_ok = Rule([color_idx["red"], age_idx[30]], target="pos", dataspec=ds)
    assert r_ok.is_consistent()
    # implied *positive* conditions: color!=green, color!=blue (from
    # ExactlyOne), age>=20 (lower threshold implied by age>=30)
    implied = {l.feature for l in r_ok.implied_conditions()}
    assert implied == {color_idx.negative["green"], color_idx.negative["blue"], age_idx[20]}

    # inconsistent rule: color=red and color=blue
    r_bad = Rule([color_idx["red"], color_idx["blue"]], dataspec=ds)
    assert not r_bad.is_consistent()


def test_display_formats_with_typed_attributes():
    b = DataSpecBuilder(negation=True)
    color_idx = b.add_nominal("color", ["red", "green"])
    age_idx = b.add_numeric("age", [30])
    ds = b.build()

    r = Rule([color_idx["red"], age_idx.negative[30]], target="risk", dataspec=ds)
    assert r.to_string("logic") == "color = red ∧ age < 30 → risk"
    assert r.to_string("conditions") == "color = red, age < 30"
    assert r.to_string("prolog") == "risk(X) :- color(X, red), age(X, V1), V1 < 30."

    # a negated nominal condition is a positive literal on the != feature
    r2 = Rule([color_idx.negative["red"]], dataspec=ds)
    assert r2.to_string("logic") == "color ≠ red"
    assert r2.to_string("logic", ascii=True) == "color != red"


def test_set_valued_features_are_independent():
    b = DataSpecBuilder()
    idx = b.add_set("tags", ["urgent", "billing", "bug"])
    ds = b.build()

    assert ds.feature_names == ["tags has urgent", "tags has billing", "tags has bug"]
    assert ds.constraints == ()  # no mutual exclusion / exhaustiveness

    # any combination can hold simultaneously, including all or none
    closure = ds.propagate({idx["urgent"]: True, idx["billing"]: True})
    assert idx["bug"] not in closure

    r = Rule([idx["urgent"], idx["bug"]], target="p1", dataspec=ds)
    assert r.to_string("logic") == "urgent ∈ tags ∧ bug ∈ tags → p1"
    assert r.to_string("logic", ascii=True) == "urgent in tags AND bug in tags -> p1"
    assert r.to_string("prolog") == "p1(X) :- tags(X, urgent), tags(X, bug)."


def test_hierarchical_features_propagation_and_display():
    b = DataSpecBuilder()
    idx = b.add_hierarchical("region", {
        "Europe": {"France": {"Paris": {}, "Lyon": {}}, "Germany": {}},
        "Asia": {},
    })
    ds = b.build()

    assert isinstance(ds.constraints[0], MutuallyExclusive) or isinstance(ds.constraints[0], Implies)

    # region=Paris forces region=France and region=Europe True
    closure = ds.propagate({idx["Paris"]: True})
    assert closure[idx["France"]] is True
    assert closure[idx["Europe"]] is True
    # siblings of an implied ancestor are forced False (mutual exclusion)
    assert closure[idx["Germany"]] is False
    assert closure[idx["Asia"]] is False
    # Lyon (sibling of Paris) is also ruled out
    assert closure[idx["Lyon"]] is False

    # region=Europe alone does NOT force a specific country (not exhaustive)
    closure2 = ds.propagate({idx["Europe"]: True})
    assert idx["France"] not in closure2
    assert idx["Germany"] not in closure2

    # contrapositive: region != Europe forces region != France, != Paris, != Lyon
    closure3 = ds.propagate({idx["Europe"]: False})
    assert closure3[idx["France"]] is False
    assert closure3[idx["Paris"]] is False
    assert closure3[idx["Lyon"]] is False

    # contradiction: Paris True but France explicitly False
    try:
        ds.propagate({idx["Paris"]: True, idx["France"]: False})
        assert False, "expected ValueError"
    except ValueError:
        pass

    r = Rule([idx["Paris"]], target="eu_customer", dataspec=ds)
    assert r.to_string("logic") == "region = Paris → eu_customer"
    # only positive implications (hierarchy nodes have no != feature):
    # Paris -> France, Europe
    assert {l.feature for l in r.implied_conditions()} == {idx["France"], idx["Europe"]}


def test_relational_feature_provenance_and_display():
    b = DataSpecBuilder()
    b.add_numeric("age", [30])
    b.add_numeric("income", [50000])
    rel_idx = b.add_relational(
        "income_gt_age_scaled", attributes=["income", "age"], expression="income > age * 1000"
    )
    ds = b.build()

    assert ds.attributes["income_gt_age_scaled"].type == AttributeType.RELATIONAL
    assert ds.attributes["income_gt_age_scaled"].sources == ("income", "age")
    spec = ds.feature_spec(rel_idx)
    assert spec.attributes == ("income", "age")
    assert spec.expression == "income > age * 1000"

    r = Rule([rel_idx], target="qualifies", dataspec=ds)
    assert r.to_string("logic") == "income > age * 1000 → qualifies"
    assert r.to_string("prolog") == "qualifies(X) :- income_gt_age_scaled(X)."


def test_binary_attribute_has_two_value_features_paired_as_negations():
    for negation in (True, False):
        b = DataSpecBuilder(negation=negation)
        idx = b.add_binary("sex", ["male", "female"])
        ds = b.build()
        # never any != columns, whatever the negation setting
        assert ds.feature_names == ["sex=male", "sex=female"]
        assert ds.negation_of("sex=male") == ds.feature_index("sex=female")
        assert ds.negation_of("sex=female") == ds.feature_index("sex=male")
        assert idx.negative == {"male": idx["female"], "female": idx["male"]}
        assert ds.attributes["sex"].type == AttributeType.BINARY


def test_binary_unknown_and_missing_values_cover_neither_value():
    import pandas as pd
    from pyrulearn.data.io import binarize
    b = DataSpecBuilder()
    b.add_binary("sex", ["male", "female"])
    X = binarize(b.build(), pd.DataFrame({"sex": ["male", "female", "other", None]}))
    assert X.astype(int).tolist() == [[1, 0], [0, 1], [0, 0], [0, 0]]


def test_binary_needs_two_distinct_values():
    for values in (["a"], ["a", "b", "c"], ["a", "a"]):
        with pytest.raises(ValueError):
            DataSpecBuilder().add_binary("x", values)


def test_two_valued_nominal_keeps_its_inequality_columns():
    # a nominal's value list may be incomplete: an unknown value must
    # satisfy every x!=v, so x!=a is not x=b
    import pandas as pd
    from pyrulearn.data.io import binarize
    b = DataSpecBuilder(negation=True)
    b.add_nominal("x", ["a", "b"])
    ds = b.build()
    assert ds.feature_names == ["x=a", "x=b", "x!=a", "x!=b"]
    assert ds.negation_of("x=a") == ds.feature_index("x!=a")
    X = binarize(ds, pd.DataFrame({"x": ["c", None]}))
    assert X.astype(int).tolist() == [[0, 0, 1, 1], [0, 0, 0, 0]]


def test_negation_toggles_keep_a_binary_attributes_pairing():
    for negation in (True, False):
        b = DataSpecBuilder(negation=negation)
        b.add_binary("sex", ["male", "female"])
        b.add_nominal("c", ["a", "b", "c"])
        ds = b.build()
        for toggled in (ds.with_negations()[0], ds.without_negations()[0]):
            assert toggled.feature_names[:2] == ["sex=male", "sex=female"]
            assert toggled.negation_of("sex=male") == toggled.feature_index("sex=female")


def test_extend_closure_matches_propagate_from_scratch():
    import pandas as pd
    from pyrulearn.data.io import build_dataspec

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=60), "z": rng.integers(0, 9, size=60).astype(float),
        "c": rng.choice(["r", "g", "b", "k"], size=60), "b": rng.choice(["y", "n"], size=60),
        "t": rng.choice(["p", "q"], size=60),
    })
    spec = build_dataspec(df, target="t", discretizer=TreeDiscretizer(max_intervals=5)).build()
    n = spec.n_features
    checked = 0
    for _ in range(400):
        base = {int(i): True for i in rng.choice(n, size=rng.integers(0, 3), replace=False)}
        try:
            closure = spec.propagate(base)
        except ValueError:
            continue
        new = {int(i): bool(rng.integers(0, 2)) for i in rng.choice(n, size=rng.integers(1, 3), replace=False)}
        try:
            expected = spec.propagate({**base, **new}) if all(closure.get(i, v) == v for i, v in new.items()) else None
        except ValueError:
            expected = None
        if expected is None:
            with pytest.raises(ValueError):
                spec.extend_closure(closure, new)
        else:
            assert spec.extend_closure(closure, new) == expected
        checked += 1
    assert checked > 200
