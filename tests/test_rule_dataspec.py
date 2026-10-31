import numpy as np
from pyrulearn.data import DataSpec, DataSpecBuilder
from pyrulearn.data import HorizontalBitmapRepresentation
from pyrulearn.rule import Rule, Literal


def _bool_spec(names, negation=True):
    b = DataSpecBuilder(negation=negation)
    for n in names:
        b.add_boolean(n)
    return b.build()


def test_coverage_matches_brute_force():
    rng = np.random.default_rng(42)
    n, d = 500, 10
    X = rng.integers(0, 2, size=(n, d)).astype(bool)
    ds = DataSpec([f"f{i}" for i in range(d)])
    rep = HorizontalBitmapRepresentation(ds, X)
    for _ in range(20):
        k = int(rng.integers(0, 4))
        feats = [int(i) for i in rng.choice(d, size=k, replace=False)]
        r = Rule(feats, dataspec=ds)
        expected = np.array([all(X[i, f] for f in feats) for i in range(n)])
        assert np.array_equal(r.covers_data(rep), expected)
        assert np.array_equal(r.covers_data_packed(rep), expected)
        for i in range(0, n, 37):
            assert r.covers(X[i].tolist()) == bool(expected[i])


def test_ordering():
    ds = _bool_spec(["age_gt_30", "smoker", "high_bp", "diabetic"])
    not_diabetic = ds.negation_of("diabetic")
    # construct with a specific, meaningful order (the order tests were added)
    r = Rule([ds.feature_index("smoker"), not_diabetic, ds.feature_index("age_gt_30")],
             target="risk", dataspec=ds, ordered=True)
    assert r.to_string("logic") == "smoker ∧ ¬diabetic ∧ age_gt_30 → risk"

    r_unordered = Rule([ds.feature_index("smoker"), not_diabetic, ds.feature_index("age_gt_30")],
                       target="risk", dataspec=ds, ordered=False)
    assert r_unordered.to_string("logic") == "age_gt_30 ∧ smoker ∧ ¬diabetic → risk"

    assert r == r_unordered
    assert r.covers_bits(0b1) == r_unordered.covers_bits(0b1)

    r_reordered = r_unordered.reorder([not_diabetic, ds.feature_index("age_gt_30"), ds.feature_index("smoker")])
    assert r_reordered.ordered is True
    assert r_reordered.to_string("logic") == "¬diabetic ∧ age_gt_30 ∧ smoker → risk"


def test_output_formats():
    ds = _bool_spec(["age_gt_30", "smoker", "high_bp"])
    r = Rule.from_pos_neg(pos=[ds.feature_index("age_gt_30"), ds.feature_index("smoker")],
                          neg=[ds.feature_index("high_bp")], target="high_risk", dataspec=ds)

    assert r.to_string("logic") == "age_gt_30 ∧ smoker ∧ ¬high_bp → high_risk"
    assert r.to_string("logic", ascii=True) == "age_gt_30 AND smoker AND NOT high_bp -> high_risk"
    assert r.to_string("prolog") == "high_risk(X) :- age_gt_30(X), smoker(X), \\+high_bp(X)."
    assert r.to_string("conditions") == "age_gt_30, smoker, ¬high_bp"

    # attribute-less rule falls back to f{i} names; every literal is positive now
    r_nods = Rule([0, 2], n_features=3)
    assert r_nods.to_string("logic") == "f0 ∧ f2"
    assert r_nods.to_string("pattern") == "1 0 1"
    assert repr(r_nods) == "Rule(rule(X) :- f0(X), f2(X).)"
    assert r_nods.to_string() == "rule(X) :- f0(X), f2(X)."


def test_prolog_format_quotes_non_atom_targets():
    # a target that isn't a valid bare Prolog atom (must start lowercase --
    # a leading digit or uppercase letter is a variable, not a predicate
    # name) gets single-quoted in the head; "rule" (no target) is untouched
    r_digit = Rule([0], target="1", n_features=1)
    assert r_digit.to_string("prolog") == "'1'(X) :- f0(X)."
    r_upper = Rule([0], target="Yes", n_features=1)
    assert r_upper.to_string("prolog") == "'Yes'(X) :- f0(X)."
    r_lower = Rule([0], target="survived", n_features=1)
    assert r_lower.to_string("prolog") == "survived(X) :- f0(X)."
    r_none = Rule([0], n_features=1)
    assert r_none.to_string("prolog") == "rule(X) :- f0(X)."


def test_pretty_prolog_puts_each_condition_on_its_own_indented_line():
    from pyrulearn.rule import WeightedRule
    ds = DataSpec(["f0", "f1"])
    rule = Rule([0, 1], target="pos", dataspec=ds)
    assert rule.to_string("prolog", pretty=True) == "pos(X) :-\n    f0(X),\n    f1(X)."
    assert rule.to_string("prolog") == "pos(X) :- f0(X), f1(X)."          # default unchanged
    assert Rule([], target="pos", dataspec=ds).to_string("prolog", pretty=True) == "pos(X) :- true."
    weighted = WeightedRule([0], target="pos", dataspec=ds, weight=0.8)
    assert weighted.to_string("prolog", pretty=True) == "0.8::pos(X) :-\n    f0(X)."
    assert rule.to_string("logic", pretty=True) == rule.to_string("logic")  # other formats: unaffected


def test_pattern_format_truncates_long_vectors_unless_pretty():
    n = Rule.PATTERN_TRUNCATE_AT
    short = Rule([0, 2], n_features=n)             # exactly at the threshold: no truncation
    assert "..." not in short.to_string("pattern")
    assert short.to_string("pattern", pretty=True) == short.to_string("pattern")

    long_ = Rule([0, n], n_features=n + 5)         # one condition past the truncation point
    truncated = long_.to_string("pattern")
    assert truncated.endswith("...")
    assert truncated.split()[:-1] == ["1"] + ["0"] * (n - 1)      # first n tokens, "..." stripped
    full = long_.to_string("pattern", pretty=True)
    assert "..." not in full
    tokens = full.split()
    assert len(tokens) == n + 5
    assert [i for i, t in enumerate(tokens) if t == "1"] == [0, n]


def test_prolog_format_gives_each_condition_its_own_variable():
    # two conditions that each introduce a fresh Prolog variable (>=/</
    # !=) must NOT reuse the same variable name -- "age(X, V), V < 30,
    # income(X, V), V >= 50000" would force age's and income's values to
    # unify, which is wrong: they're unrelated quantities
    b = DataSpecBuilder(negation=True)
    age_idx = b.add_numeric("age", [30])
    income_idx = b.add_numeric("income", [50000])
    color_idx = b.add_nominal("color", ["red", "blue"])
    ds = b.build()

    r = Rule([age_idx.negative[30], income_idx[50000], color_idx.negative["red"]],
            target="pos", dataspec=ds, ordered=True)
    assert r.to_string("prolog") == (
        "pos(X) :- age(X, V1), V1 < 30, income(X, V2), V2 >= 50000, "
        "color(X, V3), V3 \\= red."
    )
    # a single such condition still gets numbered, for consistency
    r_one = Rule([age_idx.negative[30]], target="pos", dataspec=ds)
    assert r_one.to_string("prolog") == "pos(X) :- age(X, V1), V1 < 30."


def test_default_fmt_configuration():
    r = Rule([0, 1], target="pos", n_features=3)
    assert Rule.DEFAULT_FORMAT == "prolog"
    assert r.to_string() == "pos(X) :- f0(X), f1(X)."
    assert repr(r) == "Rule(pos(X) :- f0(X), f1(X).)"

    r_logic = Rule([0, 1], target="pos", n_features=3, default_fmt="logic")
    assert r_logic.to_string() == "f0 ∧ f1 → pos"
    assert repr(r_logic) == "Rule(f0 ∧ f1 → pos)"

    assert r_logic.to_string(fmt="conditions") == "f0, f1"

    reordered = r_logic.reorder([1, 0])
    assert reordered.default_fmt == "logic"

    other = Rule([0], target="pos", n_features=3)  # no default_fmt
    assert r_logic.generalize(other).default_fmt == "logic"
    assert other.generalize(r_logic).default_fmt == "logic"

    try:
        Rule.DEFAULT_FORMAT = "logic"
        assert r.to_string() == "f0 ∧ f1 → pos"
        assert r_logic.to_string() == "f0 ∧ f1 → pos"
    finally:
        Rule.DEFAULT_FORMAT = "prolog"


def test_boolean_representation_and_dedup():
    ds = DataSpec(["a", "b", "c"])
    X = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=bool)
    rep = HorizontalBitmapRepresentation(ds, X, y=np.array(["x", "y", "x"]))
    assert rep.spec is ds and rep.n_samples == 3

    # duplicate condition on the same feature collapses silently, first position kept
    r = Rule([Literal(0), Literal(1), Literal(0)], dataspec=ds)
    assert r.conditions == (Literal(0), Literal(1))


def test_rule_remap_by_feature_name():
    old_ds = _bool_spec(["smoker", "age_gt_30", "high_bp"])
    r = Rule.from_pos_neg(pos=[old_ds.feature_index("age_gt_30"), old_ds.feature_index("smoker")],
                          neg=[old_ds.feature_index("high_bp")], target="risk", dataspec=old_ds)

    new_ds = _bool_spec(["high_bp", "extra_unused", "age_gt_30", "smoker"])
    remapped = r.remap(new_ds)

    assert remapped.dataspec is new_ds
    assert remapped.target == "risk"
    assert remapped.pos == tuple(sorted((
        new_ds.feature_index("age_gt_30"), new_ds.feature_index("smoker"),
        new_ds.feature_index("not high_bp"),
    )))
    assert {new_ds.feature_name(i) for i in remapped.pos} == \
           {old_ds.feature_name(i) for i in r.pos}


def test_rule_remap_raises_on_unmatched_feature():
    old_ds = DataSpec(["smoker", "age_gt_30"])
    r = Rule([0], dataspec=old_ds)
    try:
        r.remap(DataSpec(["age_gt_30"]))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "smoker" in str(e)


def test_rule_remap_raises_without_dataspec():
    r = Rule([0], n_features=2)
    try:
        r.remap(DataSpec(["f0", "f1"]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_generalize_and_is_more_specific():
    ds = _bool_spec(["a", "b", "c", "d", "e"])
    a, b, c, d = (ds.feature_index(x) for x in "abcd")
    r1 = Rule.from_pos_neg([a, b], [c], dataspec=ds)
    r2 = Rule.from_pos_neg([a], [c, d], dataspec=ds)
    g = r1.generalize(r2)
    assert g.pos == tuple(sorted((a, ds.negation_of(c))))
    assert r1.is_more_specific(g)
    assert r2.is_more_specific(g)
    assert not g.is_more_specific(r1)
    assert g.is_more_general(r1)
    assert g.is_more_general(r2)
    assert not r1.is_more_general(g)


def test_is_more_specific_ignores_target_by_default():
    ds = DataSpec(["a", "b", "c"])
    r1 = Rule([0, 1], dataspec=ds, target="yes")
    r2 = Rule([0], dataspec=ds, target="no")
    assert r1.is_more_specific(r2)
    assert r2.is_more_general(r1)
    assert not r1.is_more_specific(r2, include_target=True)
    assert not r2.is_more_general(r1, include_target=True)


def test_incomparable_rules_are_neither_more_specific_nor_more_general():
    ds = DataSpec(["a", "b", "c"])
    r1 = Rule([0], dataspec=ds)
    r2 = Rule([1], dataspec=ds)
    assert not r1.is_more_specific(r2)
    assert not r1.is_more_general(r2)
    assert not r2.is_more_specific(r1)
    assert not r2.is_more_general(r1)


def test_rule_comparison_operators():
    ds = DataSpec(["a", "b", "c", "d"])
    specific = Rule([0, 1], dataspec=ds, target="yes")
    general = Rule([0], dataspec=ds, target="yes")
    same = Rule([0], dataspec=ds, target="yes")
    other_target = Rule([0], dataspec=ds, target="no")
    incomparable = Rule([2], dataspec=ds, target="yes")

    assert specific <= general and specific < general
    assert not (general <= specific)
    assert general >= specific and general > specific
    assert not (specific >= general)
    assert general <= same and general >= same and not (general < same) and not (general > same)
    assert general == same

    assert not (general <= other_target)
    assert not (general >= other_target)
    assert general != other_target

    assert not (general <= incomparable)
    assert not (general >= incomparable)

    assert specific.__le__(5) is NotImplemented


def test_specialize_no_dataspec_tries_every_feature():
    r = Rule([], n_features=3)  # no dataspec -- no pruning
    children = r.specialize()
    assert len(children) == 3  # one per feature (every literal positive now)
    for child, mask in children:
        assert child.length() == 1
        feat = child.conditions[0].feature
        assert mask == frozenset({0, 1, 2}) - {feat}


def test_specialize_default_mask_excludes_already_used_features():
    ds = DataSpec(["a", "b", "c"])
    r = Rule([0], dataspec=ds)
    children = r.specialize(ds)
    used = {child.conditions[-1].feature for child, _ in children}
    assert used == {1, 2}
    for _, mask in children:
        assert 0 not in mask


def test_specialize_explicit_mask_restricts_candidates():
    ds = DataSpec(["a", "b", "c"])
    r = Rule([], dataspec=ds)
    children = r.specialize(ds, mask=frozenset({1}))
    assert {child.conditions[0].feature for child, _ in children} == {1}
    assert len(children) == 1


def test_specialize_prunes_redundant_threshold_chain_conditions():
    b = DataSpecBuilder()
    idx = b.add_numeric("age", [20, 30, 40])
    ds = b.build()
    children = Rule([], dataspec=ds).specialize(ds)

    match = [(c, m) for c, m in children if c.conditions == (Literal(idx[40]),)]
    assert len(match) == 1
    _, mask = match[0]
    # age>=40 implies age>=30 and age>=20 -- re-offering either is redundant
    assert idx[40] not in mask and idx[30] not in mask and idx[20] not in mask


def test_specialize_prunes_contradictory_nominal_conditions():
    b = DataSpecBuilder()
    idx = b.add_nominal("color", ["red", "green", "blue"])
    ds = b.build()
    children = Rule([], dataspec=ds).specialize(ds)

    match = [(c, m) for c, m in children if c.conditions == (Literal(idx["red"]),)]
    assert len(match) == 1
    child, mask = match[0]
    # color=red forces green/blue False -- both masked out
    assert idx["red"] not in mask and idx["green"] not in mask and idx["blue"] not in mask

    # forcing green as a candidate: green contradicts red, so no child is produced
    forced = child.specialize(ds, mask=frozenset({idx["green"]}))
    assert forced == []


def _color_spec():
    b = DataSpecBuilder(negation=True)
    color = b.add_nominal("color", ["red", "green", "blue", "black", "white"])
    smoker = b.add_boolean("smoker")
    return b.build(), color.negative, smoker


def test_pretty_shows_several_negations_on_one_attribute_as_a_value_set():
    spec, neq, smoker = _color_spec()
    two = Rule([smoker, neq["red"], neq["blue"]], target="t", dataspec=spec)
    assert two.to_string(fmt="conditions") == "color ≠ red, color ≠ blue, smoker"   # plain: every condition
    assert two.to_string(fmt="conditions", pretty=True) == "color ∉ {red, blue}, smoker"
    assert two.to_string(fmt="conditions", ascii=True, pretty=True) == "color not in {red, blue}, smoker"
    assert two.to_string(fmt="prolog", pretty=True) == (
        "t(X) :-\n    color(X, V1), \\+ member(V1, [red, blue]),\n    smoker(X).")
    assert two.length() == 3                                   # only the display changes
    three = Rule([neq["red"], neq["blue"], neq["black"]], target="t", dataspec=spec)
    assert three.to_string(fmt="logic", pretty=True) == "color ∈ {green, white} → t"   # the shorter form
    assert three.to_string(fmt="prolog", pretty=True) == "t(X) :-\n    color(X, V1), member(V1, [green, white])."
    four = Rule([neq["red"], neq["blue"], neq["black"], neq["white"]], target="t", dataspec=spec)
    assert four.to_string(fmt="conditions", pretty=True) == "color = green"
    assert four.to_string(fmt="prolog", pretty=True) == "t(X) :-\n    color(X, green)."


def test_a_single_negation_stays_a_negation():
    spec, neq, smoker = _color_spec()
    one = Rule([smoker, neq["red"]], target="t", dataspec=spec)
    assert one.to_string(fmt="conditions", pretty=True) == "color ≠ red, smoker"
    assert one.to_string(fmt="prolog") == r"t(X) :- color(X, V1), V1 \= red, smoker(X)."


def test_pretty_shows_thresholds_on_one_numeric_attribute_as_an_interval():
    b = DataSpecBuilder(negation=True)
    age = b.add_numeric("age", ge_thresholds=[20, 30, 40])
    smoker = b.add_boolean("smoker")
    spec = b.build()
    between = Rule([age[20], age.negative[40], smoker], target="t", dataspec=spec)
    assert between.to_string(fmt="conditions") == "age >= 20, age < 40, smoker"
    assert between.to_string(fmt="conditions", pretty=True) == "20 <= age < 40, smoker"
    assert between.to_string(fmt="prolog", pretty=True) == "t(X) :-\n    age(X, V1), V1 >= 20, V1 < 40,\n    smoker(X)."
    tightest = Rule([age[20], age[30]], target="t", dataspec=spec)
    assert tightest.to_string(fmt="conditions", pretty=True) == "age >= 30"
    single = Rule([age.negative[30]], target="t", dataspec=spec)
    assert single.to_string(fmt="conditions", pretty=True) == "age < 30"


def test_pretty_model_printing_compresses_too():
    from pyrulearn.models import ConceptModel
    spec, neq, smoker = _color_spec()
    model = ConceptModel([Rule([neq["red"], neq["blue"], smoker], target="t", dataspec=spec)], label="t",
                         default_prediction="f")
    assert "color ∉ {red, blue}" in model.to_string(fmt="logic", pretty=True, show_stats=False)
    assert "color ≠ red ∧ color ≠ blue" in model.to_string(fmt="logic", show_stats=False)
