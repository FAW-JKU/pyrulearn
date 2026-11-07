import itertools

import numpy as np
import pytest

from pyrulearn.combiners import CountVoteCombiner, HeuristicMaxCombiner, ListCombiner, MicroVoteCombiner
from pyrulearn.data import HorizontalBitmapRepresentation
from pyrulearn.data import DataSpec, DataSpecBuilder, merge_dataspecs
from pyrulearn.rule import Rule, WeightedRule
from pyrulearn.models import (
    AccuracyWeightedVote, CompositeModel, ConceptCascade, ConceptModel, ConceptSet, DecisionList,
    DeepModel, DefaultPrediction, DisjointRuleSet, EnsembleModel, FlatRuleSet, MajorityClass,
    PairwiseModel, Provenance, RuleList, RuleModel, RuleSet, SingleRule, WeightedVote,
    can_convert, conceptcascade_to_decision_list, conceptset_to_flatruleset, convert,
    ensemblemodel_to_flatruleset,
    flatruleset_to_conceptset, flatruleset_to_decision_list,
    annotate_default_rule, annotate_rules,
)
from pyrulearn.evaluation import RuleStats

DS = DataSpec(["p", "q", "r"])
#            p  q  r
X = np.array([[1, 0, 0],   # 0
              [0, 1, 0],   # 1
              [1, 1, 0],   # 2
              [0, 0, 1],   # 3
              [0, 0, 0]],  # 4
             dtype=bool)
Y = np.array(["a", "b", "a", "b", "a"], dtype=object)
DATA = HorizontalBitmapRepresentation(DS, X, Y)


# -- WeightedRule ------------------------------------------------------

def test_weighted_rule_is_rule_plus_one_declarative_field():
    wr = WeightedRule([0], target="a", dataspec=DS, weight=0.8)
    assert wr.weight == 0.8
    assert wr == Rule([0], target="a", dataspec=DS)             # weight not in identity
    assert hash(wr) == hash(Rule([0], target="a"))
    assert {wr, Rule([0], target="a")} == {wr}


def test_weighted_rule_printing():
    wr = WeightedRule([0, 2], target="a", dataspec=DS, weight=0.75)
    assert wr.to_string("prolog") == "0.75::a(X) :- p(X), r(X)."
    assert wr.to_string("logic", ascii=True) == "p AND r -> a  [0.75]"
    assert repr(wr).startswith("WeightedRule(")


def test_weighted_rule_structural_copies_keep_weight():
    wr = WeightedRule([0, 1], target="a", dataspec=DS, weight=0.9, ordered=False)
    assert wr.reorder([1, 0]).weight == 0.9
    assert wr.remap(DataSpec(["x", "p", "q", "r"])).weight == 0.9
    child, _ = wr.specialize()[0]
    assert type(child) is Rule


# -- hierarchy: the two axes -------------------------------------------

def test_family_membership():
    ds = DS
    r = [Rule([0], target="a", dataspec=ds)]
    assert isinstance(FlatRuleSet(r), RuleSet) and not isinstance(FlatRuleSet(r), RuleList)
    assert isinstance(ConceptModel(r), RuleSet)
    assert isinstance(ConceptSet.from_rules(r), RuleSet)
    assert isinstance(DisjointRuleSet(r), RuleSet)
    assert isinstance(DecisionList(r), RuleList) and not isinstance(DecisionList(r), RuleSet)
    assert isinstance(ConceptCascade([ConceptModel(r, label="a")]), RuleList)
    assert isinstance(EnsembleModel([SingleRule(r[0])]), CompositeModel)
    # RuleSet / RuleList are abstract family bases -- not instantiable
    with pytest.raises(TypeError):
        RuleSet()
    with pytest.raises(TypeError):
        RuleList()
    # and every model, base or composite, is a RuleModel
    for m in (SingleRule(r[0]), ConceptModel(r), FlatRuleSet(r), DisjointRuleSet(r),
              DecisionList(r), ConceptSet.from_rules(r),
              ConceptCascade([ConceptModel(r, label="a")]),
              EnsembleModel([SingleRule(r[0])]),
              PairwiseModel([("a", "b", FlatRuleSet(r))])):
        assert isinstance(m, RuleModel)


# -- SingleRule --------------------------------------------------------

def test_single_rule_predicts_head_or_default():
    m = SingleRule(Rule([0], target="a", dataspec=DS), default_prediction="b")
    assert list(m.predict(DATA)) == ["a", "b", "a", "b", "b"]
    assert m.rule.target == "a" and len(m) == 1
    assert type(m.filter("a")) is SingleRule
    assert type(m.filter("z")) is FlatRuleSet and len(m.filter("z")) == 0


def test_flat_containers_store_single_rule_not_bare_rule():
    r1 = Rule([0], target="a", dataspec=DS)
    r2 = WeightedRule([1], target="b", dataspec=DS, weight=0.7)
    fs = FlatRuleSet([r1, r2])
    assert all(type(r) is SingleRule for r in fs.rules)
    assert [r.rule for r in fs.rules] == [r1, r2]


def test_single_rule_forwards_reads_to_the_wrapped_rule():
    r = WeightedRule([0, 1], target="a", dataspec=DS, weight=0.42)
    sr = SingleRule(r)
    assert sr.target == "a"
    assert sr.conditions == r.conditions
    assert sr.dataspec is DS
    assert sr.pos == r.pos
    assert sr.weight == 0.42
    assert sr.to_string("logic") == r.to_string("logic")  # fmt positional-first, like Rule.to_string
    assert sr.to_string(fmt="prolog") == r.to_string(fmt="prolog")
    assert sr.rules == [sr]           # recursion base case


def test_single_rule_weight_is_settable_through_the_wrapper():
    r = WeightedRule([0], target="a", dataspec=DS, weight=0.5)
    fs = FlatRuleSet([r])
    fs.rules[0].weight = 0.05
    assert r.weight == 0.05           # mutation reaches the real underlying rule
    assert fs.rules[0].weight == 0.05  # and is visible again through the same wrapper


def test_single_rule_missing_attribute_still_raises_attribute_error():
    sr = SingleRule(Rule([0], target="a", dataspec=DS))
    with pytest.raises(AttributeError):
        sr.this_attribute_does_not_exist_anywhere


def test_filter_and_remap_keep_producing_single_rule_wrapped_containers():
    r1 = Rule([0], target="a", dataspec=DS)
    r2 = Rule([1], target="b", dataspec=DS)
    fs = FlatRuleSet([r1, r2])

    filtered = fs.filter("a")
    assert all(type(r) is SingleRule for r in filtered.rules)

    ds2 = DataSpec(["p", "q", "r"])
    remapped = fs.remap(ds2)
    assert all(type(r) is SingleRule for r in remapped.rules)
    assert remapped.rules[0].dataspec is ds2


# -- ConceptModel ------------------------------------------------------

def test_concept_model_predicts_label_or_none():
    cm = ConceptModel([Rule([0], target="a", dataspec=DS), Rule([1], target="a", dataspec=DS)])
    assert cm.label == "a"
    assert list(cm.predict(DATA)) == ["a", "a", "a", None, None]
    assert cm.default_prediction is None
    assert len(cm.filter("a")) == 2 and len(cm.filter("b")) == 0


def test_concept_model_rejects_mixed_heads():
    with pytest.raises(ValueError):
        ConceptModel([Rule([0], target="a"), Rule([1], target="b")])


# -- FlatRuleSet -------------------------------------------------------

def test_flat_rule_set_resolves_by_combiner_and_swaps():
    rs = FlatRuleSet([Rule([0], target="a", dataspec=DS), Rule([1], target="b", dataspec=DS)],
                     default_prediction="a", combiner=ListCombiner())
    assert list(rs.predict(DATA)) == ["a", "b", "a", "a", "a"]  # row2 both -> list: first
    assert isinstance(rs.filter("a").combiner, ListCombiner)
    rs.combiner = CountVoteCombiner()
    assert set(np.unique(rs.predict(DATA))) <= {"a", "b"}
    assert type(rs.remap(DataSpec(["p", "q", "r", "s"]))) is FlatRuleSet


# -- DecisionList ------------------------------------------------------

def test_decision_list_is_first_match():
    dl = DecisionList([Rule([0], target="a", dataspec=DS), Rule([], target="b", dataspec=DS)],
                      default_prediction=None)
    assert list(dl.predict(DATA)) == ["a", "b", "a", "b", "b"]


def test_decision_list_predict_and_firing_coverage():
    X = np.array([
        [1, 0],   # only a
        [0, 1],   # only b
        [0, 0],   # neither
        [1, 1],   # both -- r1 should claim this one, not r2
    ], dtype=bool)
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X)
    r1 = Rule.from_pos_neg(pos=[0], target="A", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="B", dataspec=ds)
    r3 = Rule.from_pos_neg(target="default", dataspec=ds)  # no conditions -> catch-all
    dl = DecisionList([r1, r2, r3])

    assert list(dl.predict(rep)) == ["A", "B", "default", "A"]

    cov = dl.coverage_matrix(rep)          # raw coverage ignores order/position entirely
    unique = dl._unique_mask(cov)          # "fires": matches, and not already claimed
    assert list(cov.sum(axis=1)) == [2, 2, 4]
    assert list(unique.sum(axis=1)) == [2, 1, 1]


# -- DisjointRuleSet ---------------------------------------------------

def test_disjoint_rule_set_predicts_the_one_covering_rule():
    drs = DisjointRuleSet([Rule([0], target="a", dataspec=DS), Rule([2], target="b", dataspec=DS)],
                          default_prediction="a")
    assert list(drs.predict(DATA)) == ["a", "a", "a", "b", "a"]


# -- ConceptSet --------------------------------------------------------

def test_concept_set_blocks_and_matches_flat_when_no_conflict():
    rules = [Rule([0], target="a", dataspec=DS), Rule([2], target="b", dataspec=DS)]
    cs = ConceptSet.from_rules(rules, default_prediction="a", combiner=HeuristicMaxCombiner())
    assert sorted(cs.concept_labels) == ["a", "b"]
    assert len(cs.concept_for("a")) == 1
    assert cs.labels == ["a", "b"]
    flat = FlatRuleSet(rules, default_prediction="a", combiner=HeuristicMaxCombiner())
    assert list(cs.predict(DATA)) == list(flat.predict(DATA)) == ["a", "a", "a", "b", "a"]
    assert len(cs.filter("b").concepts) == 1
    cs.combiner = CountVoteCombiner()
    assert isinstance(cs.filter("a").combiner, CountVoteCombiner)


# -- ConceptCascade ----------------------------------------------------

def test_concept_cascade_first_firing_concept_wins():
    a = ConceptModel([Rule([0], target="a", dataspec=DS)], label="a")
    b = ConceptModel([Rule([1], target="b", dataspec=DS)], label="b")
    assert list(ConceptCascade([a, b], default_prediction="b").predict(DATA)) == ["a", "b", "a", "b", "b"]
    assert list(ConceptCascade([b, a], default_prediction="a").predict(DATA))[2] == "b"


def test_concept_cascade_unique_mask_credits_the_whole_first_concept():
    # two rules in concept "a", both cover row2 [p,q] (p and p&q): the
    # credit is per rule, not just per concept
    a = ConceptModel([Rule([0], target="a", dataspec=DS), Rule([0, 1], target="a", dataspec=DS)], label="a")
    b = ConceptModel([Rule([1], target="b", dataspec=DS)], label="b")
    casc = ConceptCascade([a, b], default_prediction="b")
    cov = casc.coverage_matrix(DATA)
    um = casc._unique_mask(cov)
    # row2 -> concept "a" fires first; BOTH of a's covering rules get unique credit
    assert um[0, 2] and um[1, 2]


# -- composite ---------------------------------------------------------

def test_ensemble_model_votes_over_members():
    m1 = SingleRule(Rule([0], target="a", dataspec=DS), default_prediction=None)   # p -> a
    m2 = SingleRule(Rule([1], target="b", dataspec=DS), default_prediction=None)   # q -> b
    m3 = SingleRule(Rule([], target="a", dataspec=DS), default_prediction=None)    # TRUE -> a
    ens = EnsembleModel([m1, m2, m3], default_prediction="b")
    # row1: a (m3) vs b (m2) tie -- never decided by member order; no stats
    # to read training frequencies from, so the label that sorts first: a
    assert list(ens.predict(DATA)) == ["a", "a", "a", "a", "a"]
    assert isinstance(ens, CompositeModel) and len(ens.rules) == 3
    with pytest.raises(TypeError):
        ens.add(Rule([0], target="a"))
    assert "3 members" in repr(ens)
    ensw = EnsembleModel([m1, m2, m3], member_weights=[1, 5, 1], default_prediction="b")
    assert list(ensw.predict(DATA))[2] == "b"   # m2->b weighted 5 beats m1+m3 -> a


def test_ensemble_vote_ties_never_depend_on_member_order():
    ds = DataSpec(["a"])

    def members(y):
        train = HorizontalBitmapRepresentation(ds, np.array([[1]] * len(y), dtype=bool), np.array(y))
        good = ConceptModel(annotate_rules([Rule([0], target="good", dataspec=ds)], train), label="good")
        bad = ConceptModel(annotate_rules([Rule([0], target="bad", dataspec=ds)], train), label="bad")
        return good, bad

    row = HorizontalBitmapRepresentation(ds, np.array([[1]], dtype=bool))
    for y, expected in ((["good"] * 3 + ["bad"], "good"),     # good more frequent
                        (["good"] + ["bad"] * 3, "bad"),       # bad more frequent
                        (["good", "bad"], "bad")):             # level -> the label sorting first
        good, bad = members(y)
        for order in ([good, bad], [bad, good]):
            assert EnsembleModel(order).predict(row)[0] == expected, (y, order)
    # an unequal weighted vote isn't a tie
    good, bad = members(["good", "bad"])
    assert EnsembleModel([good, bad], member_weights=[0.6, 0.4]).predict(row)[0] == "good"


def test_pairwise_model_round_robin_vote():
    ab = FlatRuleSet([Rule([0], target="a", dataspec=DS)], default_prediction="b")
    ac = FlatRuleSet([Rule([0], target="a", dataspec=DS)], default_prediction="c")
    bc = FlatRuleSet([Rule([1], target="b", dataspec=DS)], default_prediction="c")
    pm = PairwiseModel([("a", "b", ab), ("a", "c", ac), ("b", "c", bc)], default_prediction="a")
    assert pm.pairs == [("a", "b"), ("a", "c"), ("b", "c")]
    assert list(pm.predict(DATA))[:2] == ["a", "b"]
    assert type(pm.filter("a")) is PairwiseModel and pm.filter("a").pairs == pm.pairs


def test_pairwise_ties_use_training_frequencies_from_stats_when_no_priors_are_given():
    ds = DataSpec(["a"])
    # a 2-class model whose only pair abstains-free vote is a tie: one
    # member per direction, each voting for its own positive class
    y = ["yes"] * 3 + ["no"]
    train = HorizontalBitmapRepresentation(ds, np.array([[1]] * 4, dtype=bool), np.array(y))
    yes = ConceptModel(annotate_rules([Rule([0], target="yes", dataspec=ds)], train), label="yes")
    no = ConceptModel(annotate_rules([Rule([0], target="no", dataspec=ds)], train), label="no")
    row = HorizontalBitmapRepresentation(ds, np.array([[1]], dtype=bool))
    # no label_priors recorded: read from the rules' stats -> "yes" (3 of 4)
    for members in ([("yes", "no", yes), ("no", "yes", no)], [("no", "yes", no), ("yes", "no", yes)]):
        assert PairwiseModel(members).predict(row)[0] == "yes"


def _nnf_reference(layers, X):
    """A layered NNF evaluated directly: AND = all inputs with weight 1,
    OR = any input with weight 1."""
    v = X.astype(bool)
    for i, w in enumerate(layers):
        w = np.asarray(w, dtype=bool)
        if i % 2 == 0:
            v = np.all(~w[None, :, :] | v[:, None, :], axis=2)
        else:
            v = np.any(w[None, :, :] & v[:, None, :], axis=2)
    return v


def _random_layers(rng, n_in, widths, density):
    layers, below = [], n_in
    for width in widths:
        layers.append((rng.random((width, below)) < density).astype(int))
        below = width
    return layers


def test_deep_model_is_its_nnf():
    rng = np.random.default_rng(0)
    spec = DataSpec([f"f{i}" for i in range(6)])
    X = rng.random((200, 6)) < 0.5
    data = HorizontalBitmapRepresentation(spec, X, np.array(["x"] * 200))
    for _ in range(30):
        layers = _random_layers(rng, 6, [5, 4, 3, 1], 0.4)
        dm = DeepModel(spec, ["yes"], weights=layers, default_prediction="no")
        expected = _nnf_reference(layers, X)[:, 0]
        assert list(dm.predict(data)) == ["yes" if e else "no" for e in expected]
        assert dm.concept_values(data)[-1].tolist() == expected.tolist()
        # the rule views predict the same, simplified or not
        for simplify in (False, True):
            cv = {}
            for m in dm.to_concepts(simplify):
                v = np.zeros(200, dtype=bool)
                for r in m.rules:
                    c = np.ones(200, dtype=bool)
                    for lit in r.conditions:
                        c &= X[:, lit.feature] if lit.feature < 6 else cv[r.dataspec.feature_names[lit.feature]]
                    v |= c
                cv[m.label] = v
            assert cv["yes"].tolist() == expected.tolist()


def test_deep_model_rule_view():
    spec = DataSpec(["a", "b", "c"])
    W1 = [[1, 1, 0], [0, 0, 1], [0, 0, 0]]  # a & b, c, (empty: true)
    W2 = [[1, 1, 0], [0, 0, 0]]             # h1_1 = (a & b) | c, h1_2 = (no rules: false)
    W3 = [[1, 0], [1, 1]]                    # h1_1, h1_1 & h1_2
    W4 = [[1, 1]]
    dm = DeepModel(spec, ["yes"], weights=[W1, W2, W3, W4], default_prediction="no")
    assert repr(dm) == "DeepModel(3 inputs, layers [3, 2, 2], 1 outputs)" and dm.layers == [3, 2, 2]
    assert [m.label for m in dm.members] == ["h1_1", "h1_2", "yes"]
    assert dm.hidden_concepts == ["h1_1", "h1_2"] and dm.outputs == ["yes"] and dm.labels == ["no", "yes"]
    assert dm.dataspec.feature_names == ["a", "b", "c", "h1_1", "h1_2", "yes"]
    assert [len(m.rules) for m in dm.members] == [2, 0, 2]
    X = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
    data = HorizontalBitmapRepresentation(spec, X)
    assert dm.concept_values(data).tolist() == [[True, True, False], [False] * 3, [True, True, False]]
    assert dm.coverage_matrix(data).tolist() == [
        [True, False, False], [False, True, False], [True, True, False], [False] * 3]
    assert list(dm.predict(data)) == ["yes", "yes", "no"]
    # the rule using the false h1_2 goes, the single-rule `yes :- h1_1` takes h1_1's rules
    simple = dm.to_concepts()
    assert [m.label for m in simple] == ["yes"]
    assert sorted(r.rule.to_string() for r in simple[0].rules) == ["yes(X) :- a(X), b(X).", "yes(X) :- c(X)."]
    assert "h1_" not in str(dm) and "% concept: h1_1" in dm.to_string(simplify=False)


def test_deep_model_inlines_a_single_rule_concept():
    spec = DataSpec(["a", "b", "c"])
    # h1_1 = a & b (one rule); yes = (h1_1 & c) | c
    dm = DeepModel(spec, ["yes"], weights=[[[1, 1, 0]], [[1]], [[1], [0]], [[1, 1]]])
    dm_c = DeepModel(spec, ["yes"],
                     weights=[[[1, 1, 0], [0, 0, 1]], [[1, 0], [0, 1]], [[1, 1], [0, 1]], [[1, 1]]])
    (yes,) = dm_c.to_concepts()
    assert sorted(sorted(spec.feature_names[l.feature] for l in r.conditions) for r in yes.rules) == [
        ["a", "b", "c"], ["c"]]
    (yes,) = dm.to_concepts()
    assert [r.rule.to_string() for r in yes.rules] == ["yes(X) :- a(X), b(X).", "yes(X) :- true."]


def test_deep_model_checks_its_layers():
    spec = DataSpec(["a", "b"])
    with pytest.raises(ValueError, match="odd number"):
        DeepModel(spec, ["yes"], weights=[[[1, 0]]])
    with pytest.raises(ValueError, match="odd number"):
        DeepModel(spec, ["yes"], [4, 2])
    with pytest.raises(ValueError, match="not \\[2\\]"):
        DeepModel(spec, ["yes"], [2], weights=[[[1, 0]], [[1]]])
    with pytest.raises(ValueError, match="0 or 1"):
        DeepModel(spec, ["yes"], weights=[[[2, 0]], [[1]]])
    with pytest.raises(ValueError, match="columns"):
        DeepModel(spec, ["yes"], weights=[[[1, 0, 1]], [[1]]])
    with pytest.raises(ValueError, match="labels for"):
        DeepModel(spec, ["yes", "no"], weights=[[[1, 0]], [[1]]])
    with pytest.raises(ValueError, match="clash"):
        DeepModel(spec, ["a"], weights=[[[1, 0]], [[1]]])


def test_deep_model_filter_remap_and_conflicts():
    spec = DataSpec(["a", "b"])
    dm = DeepModel(spec, ["p", "q"], weights=[[[1, 0], [0, 1]], [[1, 0], [0, 1]]], combiner=CountVoteCombiner())
    assert isinstance(dm, CompositeModel) and dm.outputs == ["p", "q"]
    assert dm.filter("q").outputs == ["q"] and dm.filter("q").weights[-1].tolist() == [[False, True]]
    wider = DataSpec(["z", "b", "a"])
    data = HorizontalBitmapRepresentation(wider, np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=bool))
    moved = dm.remap(wider)
    assert list(moved.predict(data)[:2]) == ["p", "q"]
    assert moved.predict(data)[2] in ("p", "q")  # both fire: the combiner decides
    with pytest.raises(TypeError):
        dm._with_members([])


def test_deep_model_seip_formula():
    # (a | b) & (c | d) & (e | (f & g)), the example of Seip et al.: three hidden concepts,
    # one rule for the class over them
    spec = DataSpec(list("abcdefg"))
    W1 = np.zeros((6, 7), dtype=int)
    for node, inputs in enumerate([[0], [1], [2], [3], [4], [5, 6]]):
        W1[node, inputs] = 1
    W2 = [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]
    dm = DeepModel(spec, ["y"], weights=[W1, W2, [[1, 1, 1]], [[1]]])
    X = np.array(list(itertools.product([0, 1], repeat=7)), dtype=bool)
    a, b, c, d, e, f, g = X.T
    expected = (a | b) & (c | d) & (e | (f & g))
    assert ((dm.predict(HorizontalBitmapRepresentation(spec, X)) == "y") == expected).all()
    assert dm.to_string() == dm.to_string(simplify=False) == """\
% concept: h1_1
h1_1(X) :- a(X).
h1_1(X) :- b(X).

% concept: h1_2
h1_2(X) :- c(X).
h1_2(X) :- d(X).

% concept: h1_3
h1_3(X) :- e(X).
h1_3(X) :- f(X), g(X).

% class: y
y(X) :- h1_1(X), h1_2(X), h1_3(X)."""


def test_deep_model_from_its_architecture():
    spec = DataSpec([f"f{i}" for i in range(8)])
    dm = DeepModel(spec, ["a", "b"], [32, 16, 10, 5, 2], default_prediction="none")
    assert dm.layers == [32, 16, 10, 5, 2]
    assert [w.shape for w in dm.weights] == [(32, 8), (16, 32), (10, 16), (5, 10), (2, 5), (2, 2)]
    assert not any(w.any() for w in dm.weights)
    assert dm.hidden_concepts == [f"h1_{j}" for j in range(1, 17)] + [f"h2_{j}" for j in range(1, 6)]
    data = HorizontalBitmapRepresentation(spec, np.ones((3, 8), dtype=bool))
    assert list(dm.predict(data)) == ["none"] * 3  # no edges: every OR node is false
    assert DeepModel(spec, ["a"], [4, 2, 3]).layers == [4, 2, 3]
    assert DeepModel(spec, ["a"]).layers == [3]  # ceil(sqrt(8)): a DNF
    assert DeepModel(DataSpec([f"f{i}" for i in range(100)]), ["a"]).layers == [10]


# -- converters --------------------------------------------------------

def test_converters_and_can_convert():
    rs = FlatRuleSet([Rule([0], target="a", dataspec=DS), Rule([1], target="b", dataspec=DS)],
                     default_prediction="a")
    assert can_convert(FlatRuleSet, DecisionList) and not can_convert(FlatRuleSet, ConceptCascade)
    assert type(flatruleset_to_decision_list(rs)) is DecisionList
    assert sorted(flatruleset_to_conceptset(rs).concept_labels) == ["a", "b"]
    assert type(convert(rs, DecisionList)) is DecisionList
    with pytest.raises(TypeError):
        convert(rs, ConceptCascade)
    casc = ConceptCascade([ConceptModel([Rule([0], target="a", dataspec=DS)], label="a")],
                          default_prediction="b")
    assert type(conceptcascade_to_decision_list(casc)) is DecisionList

    ens = EnsembleModel(
        [DisjointRuleSet([Rule([0], target="a", dataspec=DS), Rule([1], target="b", dataspec=DS)]),
         DisjointRuleSet([Rule([2], target="b", dataspec=DS)])],
        default_prediction="a",
    )
    assert can_convert(EnsembleModel, FlatRuleSet)
    flat = ensemblemodel_to_flatruleset(ens)
    assert type(flat) is FlatRuleSet and len(flat.rules) == 3
    assert flat.default_prediction == "a" and isinstance(flat.combiner, CountVoteCombiner)
    assert type(convert(ens, FlatRuleSet)) is FlatRuleSet


# -- evaluate ----------------------------------------------------------

def test_evaluate_returns_a_model_stats_and_stores_nothing():
    rs = FlatRuleSet([Rule([0], target="a", dataspec=DS), Rule([1], target="a", dataspec=DS)],
                     default_prediction="a")
    st = rs.evaluate(DATA)
    assert type(st).__name__ == "ModelStats"
    assert st.n_rows == 5
    assert st.n_rules == 2
    assert st.n_conditions == 2   # one condition per rule
    assert st.confusion is not None
    assert st.confusion.labels == ["a", "b"]
    assert 0.0 <= st.confusion.accuracy <= 1.0
    # a pure measurement: nothing is stored on the model or its rules
    assert not hasattr(rs, "stats")
    assert all(r.stats() is None for r in rs.rules)
    assert rs.evaluate(DATA) is not st


def test_stats_confusion_is_none_without_labels():
    rep_no_y = HorizontalBitmapRepresentation(DS, X)  # no labels available to score a confusion matrix against
    rs = FlatRuleSet([Rule([0], target="a", dataspec=DS)], default_prediction="a")
    st = rs.evaluate(rep_no_y)
    assert st.confusion is None
    assert st.n_rules == 1


# -- provenance --------------------------------------------------------

def test_provenance_defaults_to_none_and_is_a_plain_settable_attribute():
    m = FlatRuleSet([Rule([0], target="a", dataspec=DS)])
    assert m.provenance is None
    m.provenance = Provenance(learner="CN2", params={"beam_width": 5}, source=None)
    assert m.provenance.learner == "CN2"
    assert m.provenance.params == {"beam_width": 5}
    assert m.provenance.source is None


def test_provenance_equality_and_repr():
    p1 = Provenance(learner="JRip", params={"F": 3}, source="weka.classifiers.rules.JRip")
    p2 = Provenance(learner="JRip", params={"F": 3}, source="weka.classifiers.rules.JRip")
    assert p1 == p2
    assert p1 != Provenance(learner="PART", params={"F": 3}, source="weka.classifiers.rules.JRip")
    assert "JRip" in repr(p1) and "weka.classifiers.rules.JRip" in repr(p1)
    assert "source" not in repr(Provenance(learner="CN2", params={}))  # no source -> omitted


def _rules_with_provenance():
    """Two SingleRules, each its own fresh Provenance instance (matching
    the importers' per-rule stamping convention), targeting "a" and "b"."""
    prov_a = Provenance(learner="CN2", params={"target": "a"})
    prov_b = Provenance(learner="CN2", params={"target": "b"})
    ra = SingleRule(Rule([0], target="a", dataspec=DS))
    ra.provenance = prov_a
    rb = SingleRule(Rule([1], target="b", dataspec=DS))
    rb.provenance = prov_b
    return ra, rb, prov_a, prov_b


def test_flat_rules_filter_and_remap_carry_container_and_rule_provenance():
    ra, rb, prov_a, prov_b = _rules_with_provenance()
    fs = FlatRuleSet([ra, rb])
    fs.provenance = Provenance(learner="RandomForest", params={"n_estimators": 5})

    filtered = fs.filter("a")
    assert filtered.provenance is fs.provenance          # container-level: carried, same object
    assert filtered.rules[0].provenance is prov_a         # per-rule: same object (no rebuild)

    ds2 = DataSpec(["p", "q", "r"])                       # same names -> valid remap target
    remapped = fs.remap(ds2)
    assert remapped.provenance is fs.provenance           # container-level: carried
    assert remapped.rules[0].provenance is prov_a         # per-rule: carried across the rebuild
    assert remapped.rules[1].provenance is prov_b
    assert remapped.rules[0].dataspec is ds2              # actually remapped, not just copied


def test_concept_model_filter_and_remap_carry_provenance():
    ra, _, prov_a, _ = _rules_with_provenance()
    cm = ConceptModel([ra], label="a")
    cm.provenance = Provenance(learner="SeCo", params={"target_class": "a"})

    ds2 = DataSpec(["p", "q", "r"])
    assert cm.filter("a").provenance is cm.provenance
    remapped = cm.remap(ds2)
    assert remapped.provenance is cm.provenance
    assert remapped.rules[0].provenance is prov_a


def test_single_rule_filter_and_remap_carry_provenance():
    ra, _, prov_a, _ = _rules_with_provenance()
    assert ra.filter("a") is ra                           # matches -> same object, trivially preserved
    assert ra.filter("a").provenance is prov_a

    empty = ra.filter("zzz")                              # no match -> a fresh empty FlatRuleSet
    assert empty.provenance is prov_a                     # still carried, for consistency

    ds2 = DataSpec(["p", "q", "r"])
    remapped = ra.remap(ds2)
    assert remapped is not ra                             # remap always rebuilds
    assert remapped.provenance is prov_a


def test_concept_indexed_filter_and_remap_carry_provenance():
    ra, rb, _, _ = _rules_with_provenance()
    cs = ConceptSet.from_rules([ra, rb])
    cs.provenance = Provenance(learner="OneVsRest", params={})

    ds2 = DataSpec(["p", "q", "r"])
    assert cs.filter("a").provenance is cs.provenance
    assert cs.remap(ds2).provenance is cs.provenance

    cascade = ConceptCascade([ConceptModel([ra], label="a"), ConceptModel([rb], label="b")])
    cascade.provenance = Provenance(learner="OrderedOneVsRest", params={})
    assert cascade.filter("a").provenance is cascade.provenance
    assert cascade.remap(ds2).provenance is cascade.provenance


def test_composite_filter_and_remap_carry_provenance():
    ra, rb, _, _ = _rules_with_provenance()
    ens = EnsembleModel([SingleRule(ra.rule), SingleRule(rb.rule)])
    ens.provenance = Provenance(learner="RandomForest", params={})

    ds2 = DataSpec(["p", "q", "r"])
    assert ens.filter("a").provenance is ens.provenance
    assert ens.remap(ds2).provenance is ens.provenance


def test_converters_carry_provenance():
    ra, rb, _, _ = _rules_with_provenance()
    fs = FlatRuleSet([ra, rb])
    fs.provenance = Provenance(learner="RandomForest", params={})
    assert flatruleset_to_decision_list(fs).provenance is fs.provenance
    assert flatruleset_to_conceptset(fs).provenance is fs.provenance

    cs = ConceptSet.from_rules([ra, rb])
    cs.provenance = Provenance(learner="OneVsRest", params={})
    assert conceptset_to_flatruleset(cs).provenance is cs.provenance

    ens = EnsembleModel([SingleRule(ra.rule), SingleRule(rb.rule)])
    ens.provenance = Provenance(learner="RandomForest", params={})
    assert ensemblemodel_to_flatruleset(ens).provenance is ens.provenance

    cascade = ConceptCascade([ConceptModel([ra], label="a"), ConceptModel([rb], label="b")])
    cascade.provenance = Provenance(learner="OrderedOneVsRest", params={})
    assert conceptcascade_to_decision_list(cascade).provenance is cascade.provenance


# -- annotate_rules ----------------------------------------------------

def test_annotate_rules_wraps_and_populates_stats_against_given_data():
    r1 = Rule([0], target="a", dataspec=DS)          # p -> a: covers rows 0, 2
    r2 = Rule([2], target="b", dataspec=DS)          # r -> b: covers row 3
    out = annotate_rules([r1, r2], DATA)
    assert all(type(sr) is SingleRule for sr in out)
    for sr, r in zip(out, [r1, r2]):
        expected = RuleStats.from_rule(r, DATA, r.target)
        got = sr.stats().confusion.rule_stats(r.target)
        assert (got.tp, got.fp, got.fn, got.tn) == (expected.tp, expected.fp, expected.fn, expected.tn)


def test_annotate_rules_is_idempotent_on_an_already_single_rule():
    sr = SingleRule(Rule([0], target="a", dataspec=DS))
    out = annotate_rules([sr], DATA)
    assert out[0] is sr                    # not re-wrapped
    assert out[0].stats() is not None


def test_annotate_rules_with_data_none_skips_stats():
    r = Rule([0], target="a", dataspec=DS)
    out = annotate_rules([r], None)
    assert type(out[0]) is SingleRule       # still wrapped
    assert out[0].stats() is None           # but nothing measured


def test_annotate_default_rule_populates_the_materialized_default_rules_stats():
    fs = FlatRuleSet([Rule([0], target="a", dataspec=DS)], default_prediction="b")
    result = annotate_default_rule(fs, DATA)
    assert result is fs                     # returned unchanged, for chaining
    dr_stats = fs.default_rule.stats()
    assert dr_stats is not None and dr_stats.confusion is not None


def test_annotate_default_rule_is_a_noop_when_there_is_no_default_rule():
    fs = FlatRuleSet([Rule([0], target="a", dataspec=DS)])  # default_prediction=None -> abstain
    assert fs.default_rule is None
    result = annotate_default_rule(fs, DATA)                # must not raise
    assert result is fs


# -- default_prediction ------------------------------------------------

def test_predict_default_prediction_bare_label_and_none():
    ds = DataSpec(["a", "b"])
    X3 = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    rep = HorizontalBitmapRepresentation(ds, X3)
    r = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)  # covers row 0 only

    fs = FlatRuleSet([r])
    assert list(fs.predict(rep)) == ["pos", None, None]  # no default_prediction -> None
    fs.default_prediction = "neg"
    assert list(fs.predict(rep)) == ["pos", "neg", "neg"]

    dl = DecisionList([r])
    assert list(dl.predict(rep)) == ["pos", None, None]
    dl.default_prediction = "neg"
    assert list(dl.predict(rep)) == ["pos", "neg", "neg"]


def test_default_rule_is_a_single_rule_with_its_own_stats_and_provenance():
    fs = FlatRuleSet([Rule([0], target="a", dataspec=DS)], default_prediction="b")
    dr = fs.default_rule
    assert type(dr) is SingleRule
    assert dr.target == "b" and dr.conditions == ()
    assert dr.stats() is None                   # nothing measured yet
    dr.set_stats(DATA)                           # fall-through stats land here, like any other leaf
    assert fs.default_rule is dr and fs.default_rule.stats() is not None
    assert fs.default_rule.provenance is None   # settable, just like any other RuleModel


def test_default_rule_reassignment_discards_materialized_rule_and_stats():
    r = Rule.from_pos_neg(pos=[0], target="a", dataspec=DS)
    fs = FlatRuleSet([r], default_prediction="fallback")
    dr = fs.default_rule
    assert dr is not None and dr.target == "fallback" and len(dr.conditions) == 0
    assert fs.default_rule is dr  # cached: same object each access

    fs.default_rule.set_stats(DATA)
    assert fs.default_rule.stats() is not None

    # reassigning the policy discards the materialized rule (and its stats)
    fs.default_prediction = "other"
    assert fs.default_rule is not dr
    assert fs.default_rule.target == "other"
    assert fs.default_rule.stats() is None

    fs.default_prediction = None
    assert fs.default_rule is None


class _ParityDefault(DefaultPrediction):
    def predict(self, rules, data, example_idx):
        return "even" if example_idx % 2 == 0 else "odd"


def test_object_policy_gives_per_example_fallback():
    # a fallback that can't be expressed as one static Rule: predict
    # "even" or "odd" depending on the uncovered example's own index
    ds = DataSpec(["a"])
    r = Rule.from_pos_neg(pos=[0], target="matched", dataspec=ds)  # covers only a=True
    X4 = np.array([[1], [0], [0], [0]], dtype=bool)
    data_rep = HorizontalBitmapRepresentation(ds, X4)

    fs = FlatRuleSet([r], default_prediction=_ParityDefault())
    assert list(fs.predict(data_rep)) == ["matched", "odd", "even", "odd"]
    assert fs.default_rule is None  # non-constant policy -> no materialized view

    dl = DecisionList([r], default_prediction=_ParityDefault())
    assert list(dl.predict(data_rep)) == ["matched", "odd", "even", "odd"]


def test_majority_class_default_and_distribution():
    m = SingleRule(Rule([0], target="a", dataspec=DS), default_prediction=MajorityClass(DATA))
    assert list(m.predict(DATA)) == ["a", "a", "a", "a", "a"]
    dist = m.predict_distribution(DATA)
    assert dist.shape == (5, 1) and np.allclose(dist, 1.0)


def test_majority_class_materializes_as_the_default_rules_target():
    ds = DataSpec(["a"])
    X5 = np.array([[1], [0], [0], [0]], dtype=bool)
    y5 = np.array(["a", "b", "b", "c"])
    rep = HorizontalBitmapRepresentation(ds, X5, y5)
    r = Rule.from_pos_neg(pos=[0], target="a", dataspec=ds)  # covers row 0

    fs = FlatRuleSet([r], default_prediction=MajorityClass(rep))
    assert fs.default_rule.target == "b"  # majority over all rows
    assert list(fs.predict(rep)) == ["a", "b", "b", "b"]


def test_filter_remap_to_rulelist_preserve_default_prediction():
    r_a = Rule.from_pos_neg(pos=[0], target="a", n_features=2)
    r_b = Rule.from_pos_neg(pos=[1], target="b", n_features=2)
    fs = FlatRuleSet([r_a, r_b], default_prediction="fallback")

    # key= a stats-free callable -- ordering itself is exercised in
    # test_evaluation.py::test_ruleset_to_rulelist instead
    rl = fs.to_rulelist(key=lambda r: r.target)
    assert rl.default_prediction == "fallback"

    filtered = fs.filter("a")
    assert filtered.default_prediction == "fallback"
    assert isinstance(filtered, FlatRuleSet)


def test_filter_remap_to_rulelist_preserve_object_policy():
    strategy = _ParityDefault()
    r_a = Rule.from_pos_neg(pos=[0], target="a", n_features=2)
    fs = FlatRuleSet([r_a], default_prediction=strategy)

    # a non-constant strategy means default_rule reads as None, but the
    # policy object itself must survive filter()/to_rulelist()/remap() unchanged
    assert fs.default_rule is None
    assert fs.filter("a").default_prediction is strategy
    assert fs.to_rulelist(key=lambda r: r.target).default_prediction is strategy

    ds1 = DataSpec(["a", "b"])
    r = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds1)
    fs2 = FlatRuleSet([r], default_prediction=strategy)
    ds2 = DataSpec(["b", "a"])  # same names, different order
    assert fs2.remap(ds2).default_prediction is strategy


def test_default_prediction_is_abstract():
    with pytest.raises(TypeError):
        DefaultPrediction()


def test_remap_onto_merged_dataspec():
    # two "independently imported models" over the same conceptual
    # attributes, each with its own per-model DataSpec/feature order
    b1 = DataSpecBuilder()
    b1.add_nominal("color", ["red", "blue"])
    b1.add_numeric("age", [30])
    ds1 = b1.build()
    r1 = Rule.from_pos_neg(pos=[ds1.feature_index("color=red")], target="pos", dataspec=ds1)
    model1 = FlatRuleSet([r1], default_prediction="neg")

    b2 = DataSpecBuilder()
    b2.add_numeric("age", [40])  # different threshold than model1's
    b2.add_nominal("color", ["red", "blue", "green"])  # extra category than model1's
    ds2 = b2.build()
    r2 = Rule.from_pos_neg(pos=[ds2.feature_index("age>=40")], target="pos", dataspec=ds2)
    model2 = DisjointRuleSet([r2])

    # merge_dataspecs is the union of both models' features -- remap
    # never fails to find a match, by construction
    shared = merge_dataspecs(ds1, ds2).build()
    remapped1 = model1.remap(shared)
    remapped2 = model2.remap(shared)

    assert isinstance(remapped1, FlatRuleSet) and not isinstance(remapped1, DisjointRuleSet)
    assert isinstance(remapped2, DisjointRuleSet)
    assert remapped1.default_rule.target == "neg"
    assert remapped1.default_rule.dataspec is shared

    # only ONE shared dataset needs attaching for both remapped models
    # to be scored together
    X = np.zeros((3, shared.n_features), dtype=bool)
    X[0, shared.feature_index("color=red")] = True
    X[0, shared.feature_index("age>=30")] = True
    X[0, shared.feature_index("age>=40")] = True
    X[1, shared.feature_index("color=blue")] = True
    X[2, shared.feature_index("color=green")] = True
    shared_rep = HorizontalBitmapRepresentation(shared, X)

    assert list(remapped1.predict(shared_rep)) == ["pos", "neg", "neg"]
    assert list(remapped2.predict(shared_rep)) == ["pos", None, None]


# -- is_disjoint / coverage --------------------------------------------

def test_disjoint_rule_set_is_disjoint_and_exhaustive():
    from _negation_helpers import make_rule, neg_spec, neg_X

    X3 = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    ds = neg_spec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, neg_X(X3))
    r_a = make_rule(ds, pos=["a"], target="a")  # row 0 only
    r_b = make_rule(ds, pos=["b"], target="b")  # row 1 only

    not_exhaustive = DisjointRuleSet([r_a, r_b])  # row 2 uncovered
    assert not_exhaustive.is_disjoint(rep) is True
    assert not_exhaustive.is_exhaustive(rep) is False

    r_c = make_rule(ds, neg=["a", "b"], target="c")  # row 2 only
    full_partition = DisjointRuleSet([r_a, r_b, r_c])
    assert full_partition.is_disjoint(rep) is True
    assert full_partition.is_exhaustive(rep) is True

    r_overlap = make_rule(ds, neg=["b"], target="x")  # rows 0,2 -- overlaps r_a on row 0
    overlapping = DisjointRuleSet([r_a, r_overlap])
    assert overlapping.is_disjoint(rep) is False


def test_is_disjoint_usable_as_precondition_on_a_plain_ruleset():
    ds = DataSpec(["a", "b"])
    X3 = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    rep = HorizontalBitmapRepresentation(ds, X3)
    r_a = Rule.from_pos_neg(pos=[0], target="a", dataspec=ds)
    r_b = Rule.from_pos_neg(pos=[1], target="b", dataspec=ds)

    fs = FlatRuleSet([r_a, r_b])
    assert fs.is_disjoint(rep) is True
    assert fs.is_exhaustive(rep) is False
    drs = DisjointRuleSet(fs.rules, default_prediction=fs.default_prediction)
    assert drs.is_disjoint(rep) is True

    dl = DecisionList([r_a, r_b])
    assert dl.is_disjoint(rep) is True  # available on RuleList too, same base method


def test_coverage_space_and_coverage_path():
    ds = DataSpec(["a", "b", "c"])
    X4 = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=bool)
    y4 = np.array(["pos", "pos", "neg"])
    rep = HorizontalBitmapRepresentation(ds, X4, y4)
    r1 = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)   # rows 0,1
    r2 = Rule.from_pos_neg(pos=[2], target="neg", dataspec=ds)   # rows 0,2

    fs = FlatRuleSet([r1, r2])
    pts = fs.coverage_space(rep, "pos")
    assert pts.tolist() == [[0, 2], [1, 1]]  # r1: 0 neg/2 pos; r2: 1 neg/1 pos

    dl = DecisionList([r1, r2])
    path = dl.coverage_path(rep, "pos")
    assert path.tolist() == [[0, 0], [0, 2], [1, 2]]


# -- to_string ---------------------------------------------------------

def test_flat_rule_set_to_string_grouped_by_class():
    ds = DataSpec(["age_gt_30", "smoker", "high_bp"])
    r1 = Rule.from_pos_neg(pos=[0, 1], target="high_risk", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[2], target="high_risk", dataspec=ds)
    fs = FlatRuleSet([r1, r2], default_prediction="low_risk")

    prolog = fs.to_string(fmt="prolog")
    assert "% class: high_risk" in prolog
    assert "% default" in prolog
    assert prolog.index("% class: high_risk") < prolog.index("% default")
    assert "high_risk(X) :- age_gt_30(X), smoker(X)." in prolog
    assert "high_risk(X) :- high_bp(X)." in prolog
    assert "low_risk(X) :- true." in prolog  # the default rule, no conditions

    logic = fs.to_string(fmt="logic")
    # both rules share a target -> collapse into one parenthesized DNF block
    assert "(age_gt_30 ∧ smoker)" in logic
    assert "∨ (high_bp)" in logic
    assert "→ high_risk" in logic

    # every rule renders in the *same* requested format regardless of
    # each rule's own default_fmt
    r1.default_fmt = "conditions"
    assert "age_gt_30(X), smoker(X)" in fs.to_string(fmt="prolog")


def test_flat_rule_set_to_string_no_default_rule_no_section():
    r = Rule.from_pos_neg(pos=[0], target="a", n_features=2)
    fs = FlatRuleSet([r])
    assert "default" not in fs.to_string(fmt="prolog")


def test_decision_list_to_string_sequential_and_if_elif_else():
    ds = DataSpec(["a", "b"])
    r1 = Rule.from_pos_neg(pos=[0], target="A", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="B", dataspec=ds)
    dl = DecisionList([r1, r2], default_prediction="C")

    logic = dl.to_string(fmt="logic")
    lines = logic.splitlines()
    assert lines[0] == "% conflict resolution: first matching rule" and lines[1] == ""
    assert lines[2].startswith("if  ") and lines[2].endswith("→ A")
    assert lines[3].startswith("elif") and lines[3].endswith("→ B")
    assert lines[4] == "else → C"

    prolog = dl.to_string(fmt="prolog", show_resolution=False)
    plines = prolog.splitlines()
    # uppercase-leading targets aren't valid bare Prolog atoms (that's a
    # variable, not a predicate name) -- quoted, same as a Rule printed alone
    assert plines[0] == "'A'(X) :- a(X)."
    assert plines[1] == "'B'(X) :- b(X)."
    assert plines[2] == "% default"
    assert plines[3] == "'C'(X) :- true."


def test_list_resolved_rule_set_prints_in_its_deciding_order():
    # combiner=ListCombiner(): the first covering rule in list order decides, so
    # printing grouped by label would hide what decides -- it prints like
    # the equivalent DecisionList instead (and predicts like it)
    ds = DataSpec(["a", "b"])
    r1 = Rule.from_pos_neg(pos=[0], target="Z", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="A", dataspec=ds)
    fs = FlatRuleSet([r1, r2], default_prediction="C", combiner=ListCombiner())
    dl = DecisionList([r1, r2], default_prediction="C")
    for fmt in ("logic", "prolog"):
        assert fs.to_string(fmt=fmt) == dl.to_string(fmt=fmt)
    assert "% class:" not in fs.to_string(fmt="prolog")
    rep = HorizontalBitmapRepresentation(ds, np.array([[1, 1], [0, 1], [0, 0]], dtype=bool))
    assert list(fs.predict(rep)) == list(dl.predict(rep)) == ["Z", "A", "C"]
    # other combiners, and a single-head set, still print grouped by label
    assert "% class:" in FlatRuleSet([r1, r2], default_prediction="C").to_string(fmt="prolog")
    one_head = FlatRuleSet([r1, Rule.from_pos_neg(pos=[1], target="Z", dataspec=ds)], combiner=ListCombiner())
    assert "% class: Z" in one_head.to_string(fmt="prolog")


def test_conflict_resolution_line_only_where_rules_can_conflict():
    ds = DataSpec(["a", "b"])
    ra = Rule.from_pos_neg(pos=[0], target="A", dataspec=ds)
    rb = Rule.from_pos_neg(pos=[1], target="B", dataspec=ds)
    ra2 = Rule.from_pos_neg(pos=[1], target="A", dataspec=ds)
    head = "% conflict resolution: "
    assert FlatRuleSet([ra, rb]).to_string(fmt="prolog").startswith(head + "max Laplace\n")
    assert FlatRuleSet([ra, rb], combiner=CountVoteCombiner()).to_string(fmt="prolog").startswith(
        head + "voting (one vote per covering rule)\n")
    assert DecisionList([ra, rb]).to_string(fmt="prolog").startswith(head + "first matching rule\n")
    # nothing to resolve: one head only, disjoint rules, a lone rule
    for model in (FlatRuleSet([ra, ra2]), ConceptModel([ra, ra2], label="A"),
                  DisjointRuleSet([ra, rb]), SingleRule(ra)):
        assert head not in model.to_string(fmt="prolog"), type(model).__name__
    assert head not in FlatRuleSet([ra, rb]).to_string(fmt="prolog", show_resolution=False)


def test_repr_identifies_and_print_renders_every_model():
    ds = DataSpec(["f0", "f1"])
    data = HorizontalBitmapRepresentation(ds, np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
                                     np.array(["pos", "neg", "neg"]))
    sr = annotate_rules([Rule([0, 1], target="pos", dataspec=ds)], data)[0]
    assert repr(sr) == "SingleRule(2 conditions)"
    assert repr(SingleRule(Rule([0], target="pos", dataspec=ds))) == "SingleRule(1 condition)"
    assert str(sr) == sr.to_string() == "pos(X) :- f0(X), f1(X).  % (0/1)"  # covers row 1: neg
    fs = FlatRuleSet([sr, Rule([1], target="neg", dataspec=ds)])
    assert repr(fs) == "FlatRuleSet(2 rules)"
    assert str(fs) == fs.to_string()


def test_pretty_prolog_puts_a_rules_stats_after_its_head():
    ds = DataSpec(["f0", "f1"])
    data = HorizontalBitmapRepresentation(ds, np.array([[1, 1], [1, 0], [0, 1]], dtype=bool),
                                     np.array(["pos", "neg", "neg"]))
    rules = annotate_rules([Rule([0, 1], target="pos", dataspec=ds), Rule([1], target="neg", dataspec=ds)], data)
    dl = annotate_default_rule(DecisionList(rules, default_prediction="neg"), data)
    assert dl.to_string(pretty=True, show_resolution=False).splitlines() == [
        "pos(X) :-  % (1/0)", "    f0(X),", "    f1(X).",
        "neg(X) :-  % (1/1)", "    f1(X).",
        "% default", "neg(X) :- true.  % (2/1)",
    ]
    assert str(rules[0]) == "pos(X) :- f0(X), f1(X).  % (1/0)"          # not pretty by default
    assert rules[0].to_string(pretty=True) == "pos(X) :-  % (1/0)\n    f0(X),\n    f1(X)."
    # a one-line rule (the default's "true.", no conditions) has nowhere
    # to put the comment but trailing -- see "neg(X) :- true.  % (2/1)" above
    # logic format: pretty changes nothing yet
    assert dl.to_string(fmt="logic", pretty=True) == dl.to_string(fmt="logic")


def test_headless_formats_name_the_class_inside_models():
    # "conditions"/"pattern" render only a rule's body; inside a model the
    # class must still be visible
    ds = DataSpec(["f0", "f1"])
    data = HorizontalBitmapRepresentation(ds, np.array([[1, 1], [1, 0], [0, 1]], dtype=bool),
                                     np.array(["pos", "neg", "neg"]))
    rules = annotate_rules([Rule([0, 1], target="pos", dataspec=ds), Rule([1], target="neg", dataspec=ds)], data)
    dl = annotate_default_rule(DecisionList(rules, default_prediction="neg"), data)
    # the default rule is labeled like every other rule here, "x:", with
    # "- default" inside its own comment instead of a separate "% default"
    # prefix (there's no body to show, so no line-up padding needed for it)
    assert dl.to_string(fmt="conditions", show_resolution=False).splitlines() == [
        "pos: f0, f1  % (1/0)", "neg: f1  % (1/1)", "neg:  % (2/1 - default)"]
    assert dl.to_string(fmt="pattern", show_resolution=False).splitlines() == [
        "pos: 1 1  % (1/0)", "neg: 0 1  % (1/1)", "neg:  % (2/1 - default)"]
    # a rule set already has class headers; only its default names the class
    fs = annotate_default_rule(FlatRuleSet(rules, default_prediction="neg"), data)
    text = fs.to_string(fmt="conditions")
    assert "% class: pos\nf0, f1  % (1/0)" in text and text.endswith("neg:  % (2/1 - default)")
    # labels are padded to the longest class name, so the bodies line up
    mixed = DecisionList([Rule([0], target="yes", dataspec=ds), Rule([1], target="n", dataspec=ds)])
    assert mixed.to_string(fmt="pattern", show_resolution=False).splitlines() == ["yes: 1 0", "n:   0 1"]
    # headed formats are unchanged, and a lone rule stays a bare body
    assert "% default\nneg(X) :- true.  % (2/1)" in dl.to_string()
    assert rules[0].rule.to_string("conditions") == "f0, f1"


# -- to_string: training stats -----------------------------------------

def test_to_string_prints_only_the_rules_own_frozen_stats():
    ds = DataSpec(["age_gt_30", "smoker", "high_bp"])
    r1 = Rule.from_pos_neg(pos=[0, 1], target="high_risk", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[2], target="high_risk", dataspec=ds)

    # rules without stats print bare; there is no data= to measure against
    fs = FlatRuleSet([r1, r2])
    assert "% (" not in fs.to_string(fmt="prolog")
    with pytest.raises(TypeError):
        fs.to_string(fmt="prolog", data=None)

    train = HorizontalBitmapRepresentation(ds, np.array([[1, 1, 0], [0, 0, 1]], dtype=bool),
                                      np.array(["high_risk", "low_risk"]))
    fs = FlatRuleSet(annotate_rules([r1, r2], train))
    before = fs.to_string(fmt="prolog")
    assert "high_risk(X) :- age_gt_30(X), smoker(X).  % (1/0)" in before
    assert "high_risk(X) :- high_bp(X).  % (0/1)" in before

    # evaluating on other data stores nothing, so it never changes what's
    # printed (nor what's predicted)
    test = HorizontalBitmapRepresentation(ds, np.array([[1, 1, 1]] * 3, dtype=bool), np.array(["high_risk"] * 3))
    fs.evaluate(test)
    for r in fs.rules:
        r.evaluate(test)
    assert fs.to_string(fmt="prolog") == before
    # show_stats=False: the bare rules
    assert "% (" not in fs.to_string(fmt="prolog", show_stats=False)


def test_training_stats_are_frozen_and_only_replaced_deliberately():
    ds = DataSpec(["a"])
    train = HorizontalBitmapRepresentation(ds, np.array([[1], [1], [0]], dtype=bool), np.array(["x", "y", "y"]))
    other = HorizontalBitmapRepresentation(ds, np.array([[1], [0]], dtype=bool), np.array(["x", "x"]))
    sr = annotate_rules([Rule([0], target="x", dataspec=ds)], train)[0]
    assert sr.to_string("prolog").endswith("% (1/1)")

    for overwrite in (lambda: sr.set_stats(other), lambda: annotate_rules([sr], other),
                      lambda: sr.set_stats_from_counts({"x": 1}, {"x": 2})):
        with pytest.raises(ValueError, match="frozen"):
            overwrite()
    assert sr.to_string("prolog").endswith("% (1/1)")  # untouched

    sr.reset_stats(other)                               # deliberate replacement
    assert sr.to_string("prolog").endswith("% (1/0)")
    annotate_rules([sr], train, reset=True)
    assert sr.to_string("prolog").endswith("% (1/1)")
    # copy=True annotates fresh SingleRules and leaves the given ones alone
    (fresh,) = annotate_rules([sr], other, copy=True)
    assert fresh is not sr and fresh.rule is sr.rule
    assert fresh.to_string("prolog").endswith("% (1/0)") and sr.to_string("prolog").endswith("% (1/1)")


def test_remap_and_filter_keep_the_frozen_stats():
    ds = DataSpec(["a", "b"])
    train = HorizontalBitmapRepresentation(ds, np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
                                      np.array(["x", "y", "y"]))
    rules = annotate_rules([Rule([0], target="x", dataspec=ds), Rule([1], target="y", dataspec=ds)], train)
    fs = annotate_default_rule(FlatRuleSet(rules, default_prediction="y"), train)
    before = fs.to_string(fmt="prolog")
    wider = DataSpec(["b", "a", "c"])                    # different order, extra feature
    remapped = fs.remap(wider)
    assert remapped.to_string(fmt="prolog") == before
    assert remapped.default_rule.stats() is not None
    assert "x(X) :- a(X).  % (1/1)" in fs.filter("x").to_string(fmt="prolog")


def test_to_string_coverage_correctness_counts():
    # row0: covered by r1 only, label matches target (correct)
    # row1: covered by r2 only, label does NOT match target (wrong)
    # row2: covered by both r1 and r2, label matches target
    ds = DataSpec(["age_gt_30", "smoker", "high_bp"])
    r1 = Rule.from_pos_neg(pos=[0, 1], target="high_risk", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[2], target="high_risk", dataspec=ds)
    X = np.array([
        [1, 1, 0],  # row0: age_gt_30 & smoker -> covered by r1 only
        [0, 0, 1],  # row1: high_bp -> covered by r2 only
        [1, 1, 1],  # row2: all three -> covered by both
    ], dtype=bool)
    y = np.array(["high_risk", "low_risk", "high_risk"])
    fs = FlatRuleSet(annotate_rules([r1, r2], HorizontalBitmapRepresentation(ds, X, y)))

    prolog = fs.to_string(fmt="prolog")
    # r1: covers rows 0,2, both high_risk (its own target) -> tp=2, fp=0
    assert "high_risk(X) :- age_gt_30(X), smoker(X).  % (2/0)" in prolog
    # r2: covers rows 1,2 -- row1 is low_risk (wrong), row2 is high_risk -> tp=1, fp=1
    assert "high_risk(X) :- high_bp(X).  % (1/1)" in prolog


# -- to_string: class distributions ------------------------------------

def _three_class_dog_rule():
    # one rule, trained on 19 rows; it covers 16: 1 bird, 1 cat, 14 dog
    ds = DataSpec(["barks"])
    X = np.array([[1]] * 16 + [[0]] * 3, dtype=bool)
    y = np.array(["bird"] + ["cat"] + ["dog"] * 14 + ["bird", "cat", "dog"])
    data = HorizontalBitmapRepresentation(ds, X, y)
    rule = Rule.from_pos_neg(pos=[0], target="dog", dataspec=ds)
    return FlatRuleSet(annotate_rules([rule], data)), data


def test_to_string_prints_the_full_distribution_only_for_a_distributioncombiner():
    fs, _ = _three_class_dog_rule()

    # a DistributionCombiner and >2 classes -> the full per-class breakdown,
    # with a printed-once legend giving its order (sorted: bird, cat, dog)
    fs.combiner = MicroVoteCombiner()
    text = fs.to_string(fmt="prolog")
    assert text.splitlines()[0] == "% classes: [bird, cat, dog]"
    assert "dog(X) :- barks(X).  % [1, 1, 14]" in text

    # "max" (the default) -- no distribution, no legend, plain (tp/fp)
    fs.combiner = HeuristicMaxCombiner()
    text = fs.to_string(fmt="prolog")
    assert "% classes:" not in text
    assert "dog(X) :- barks(X).  % (14/2)" in text


def test_to_string_skips_the_distribution_for_a_binary_problem_even_with_a_distributioncombiner():
    # only 2 classes -- the distribution vector would just be (tp/fp) reordered,
    # so it's redundant and stays suppressed even though the combiner would want it
    ds = DataSpec(["barks"])
    X = np.array([[1]] * 5 + [[0]] * 5, dtype=bool)
    y = np.array(["dog"] * 5 + ["cat"] * 5)
    rule = Rule.from_pos_neg(pos=[0], target="dog", dataspec=ds)
    fs = FlatRuleSet(annotate_rules([rule], HorizontalBitmapRepresentation(ds, X, y)), combiner=MicroVoteCombiner())

    text = fs.to_string(fmt="prolog")
    assert "% classes:" not in text
    assert "dog(X) :- barks(X).  % (5/0)" in text


def test_to_string_singlerule_default_never_prints_the_distribution():
    # SingleRule.resolution is Exclusive, never a Combine -- by default there's
    # only one rule, so no distribution-scored disagreement to make visible
    _, data = _three_class_dog_rule()
    sr = annotate_rules([Rule.from_pos_neg(pos=[0], target="dog", dataspec=data.spec)], data)[0]
    text = sr.to_string(fmt="prolog")
    assert "% classes:" not in text
    assert text == "dog(X) :- barks(X).  % (14/2)"


def test_to_string_show_distribution_forces_the_choice_either_way():
    fs, data = _three_class_dog_rule()  # combiner=HeuristicMaxCombiner() (the default), 3 classes

    # show_distribution=True forces the vector even though "max" never needs it
    text = fs.to_string(fmt="prolog", show_distribution=True)
    assert text.splitlines()[0] == "% classes: [bird, cat, dog]"
    assert "dog(X) :- barks(X).  % [1, 1, 14]" in text

    # show_distribution=False suppresses it even under a genuine DistributionCombiner
    fs.combiner = MicroVoteCombiner()
    text = fs.to_string(fmt="prolog", show_distribution=False)
    assert "% classes:" not in text
    assert "dog(X) :- barks(X).  % (14/2)" in text

    # forcing it on works even where the model structurally never has a
    # DistributionCombiner at all: a lone SingleRule, and a binary problem
    sr = annotate_rules([Rule.from_pos_neg(pos=[0], target="dog", dataspec=data.spec)], data)[0]
    assert (sr.to_string(fmt="prolog", show_distribution=True)
            == "% classes: [bird, cat, dog]\n\ndog(X) :- barks(X).  % [1, 1, 14]")

    ds2 = DataSpec(["barks"])
    binary_data = HorizontalBitmapRepresentation(
        ds2, np.array([[1]] * 5 + [[0]] * 5, dtype=bool), np.array(["dog"] * 5 + ["cat"] * 5))
    binary_fs = FlatRuleSet(annotate_rules([Rule.from_pos_neg(pos=[0], target="dog", dataspec=ds2)],
                                           binary_data))  # default "max"
    text = binary_fs.to_string(fmt="prolog", show_distribution=True)
    assert "% classes: [cat, dog]" in text
    assert "dog(X) :- barks(X).  % [0, 5]" in text  # cat=0, dog=5, class order [cat, dog]


def test_to_string_show_classes_is_independent_of_show_distribution():
    fs, _ = _three_class_dog_rule()  # combiner=HeuristicMaxCombiner(), no rule would show a vector by default

    # show_classes=True prints the legend even though no rule shows a distribution
    text = fs.to_string(fmt="prolog", show_classes=True)
    assert text.splitlines()[0] == "% classes: [bird, cat, dog]"
    assert "dog(X) :- barks(X).  % (14/2)" in text  # still plain (tp/fp) -- show_distribution untouched

    # show_classes=False suppresses the legend even while a distribution IS shown
    fs.combiner = MicroVoteCombiner()
    text = fs.to_string(fmt="prolog", show_classes=False)
    assert "% classes:" not in text
    assert "dog(X) :- barks(X).  % [1, 1, 14]" in text  # the vector itself is untouched


# -- to_string: composite models ---------------------------------------

def _three_class_fixture_data():
    ds = DataSpec(["a", "b"])
    X = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [0, 1]], dtype=bool)
    y = np.array(["cat", "cat", "dog", "dog", "bird", "bird"])
    return HorizontalBitmapRepresentation(ds, X, y)


def _three_class_pairwise_fixture():
    # cat/dog/bird, 3 pairs, each sub-model's rule only ever explicitly
    # predicts ONE of its own two classes -- the other only ever surfaces as
    # that sub-model's own default_prediction. Each sub-model's rules are
    # annotated on its own pair's rows, as a pairwise fit does.
    data = _three_class_fixture_data()
    ds = data.spec

    def concept(pair, feature, label, default):
        rows = data.select_rows(np.isin(data.y, list(pair)))
        return ConceptModel(annotate_rules([Rule.from_pos_neg(pos=[feature], target=label, dataspec=ds)], rows),
                            label=label, default_prediction=default)

    return (concept(("cat", "dog"), 0, "cat", "dog"), concept(("cat", "bird"), 0, "cat", "bird"),
            concept(("dog", "bird"), 1, "dog", "bird"), data)


def test_pairwisemodel_to_string_forces_each_pairs_own_two_classes_by_default():
    cat_dog, cat_bird, dog_bird, _ = _three_class_pairwise_fixture()
    pw = PairwiseModel(
        [("cat", "dog", cat_dog), ("cat", "bird", cat_bird), ("dog", "bird", dog_bird)],
        combiner=AccuracyWeightedVote(), member_weights=[0.9, 0.75, 0.6], default_prediction="cat",
    )
    text = pw.to_string(fmt="prolog")

    # conflict resolution, then the top-level legend: every class this model spans
    assert text.splitlines()[0] == "% conflict resolution: pairwise vote weighted by each pair's training accuracy"
    assert text.splitlines()[1] == "% classes: [bird, cat, dog]"
    # each pair's own header names it and, for accuracy_vote, its member weight
    assert "% pair: cat vs dog  (member weight: 0.9)" in text
    assert "% pair: cat vs bird  (member weight: 0.75)" in text
    assert "% pair: dog vs bird  (member weight: 0.6)" in text
    # each pair's own sub-model is forced to show its own two-class legend,
    # even though its one rule only ever explicitly predicts one of them
    assert "% classes: [cat, dog]" in text
    assert "% classes: [bird, cat]" in text
    assert "% classes: [bird, dog]" in text
    # each sub-model prints the stats it was trained with, on its own pair's
    # rows: cat_dog's rule (a=1 -> cat) over the 2 cat + 2 dog rows -> 2/0
    assert "cat(X) :- a(X).  % (2/0)" in text
    # cat_bird's *same* rule, over cat+bird rows -- the one bird row with
    # a=1 is a false positive there -- 2/1
    assert "cat(X) :- a(X).  % (2/1)" in text


def test_pairwise_weighted_vote_heuristic_is_configurable_and_printed():
    from pyrulearn.heuristics import Precision
    cat_dog, cat_bird, dog_bird, data = _three_class_pairwise_fixture()
    members = [("cat", "dog", cat_dog), ("cat", "bird", cat_bird), ("dog", "bird", dog_bird)]
    head = "% conflict resolution: "
    assert PairwiseModel(members).to_string(fmt="prolog").startswith(head + "pairwise vote\n")
    lap = PairwiseModel(members, combiner=WeightedVote())
    assert lap.to_string(fmt="prolog").startswith(
        head + "pairwise vote weighted by Laplace of each pair's deciding rule\n")
    prec = PairwiseModel(members, combiner=WeightedVote(heuristic=Precision()))
    assert prec.to_string(fmt="prolog").startswith(
        head + "pairwise vote weighted by Precision of each pair's deciding rule\n")
    # the weight is that heuristic on the deciding rule's frozen stats
    rule = cat_bird.rules[0]                        # (2/1) on its pair's rows
    assert prec.combiner.rule_weight(rule) == pytest.approx(2 / 3)
    assert lap.combiner.rule_weight(rule) == pytest.approx(3 / 5)
    assert WeightedVote().rule_weight(SingleRule(Rule([0], target="cat", dataspec=data.spec))) == 0.5
    assert head not in lap.to_string(fmt="prolog", show_resolution=False)


def test_pairwisemodel_to_string_show_classes_false_suppresses_everything():
    cat_dog, cat_bird, dog_bird, _ = _three_class_pairwise_fixture()
    pw = PairwiseModel([("cat", "dog", cat_dog), ("cat", "bird", cat_bird), ("dog", "bird", dog_bird)])
    text = pw.to_string(fmt="prolog", show_classes=False)
    assert "% classes:" not in text


def test_ensemblemodel_to_string_shows_member_weights_and_top_level_legend():
    # members trained on the full data -- unlike pairwise sub-models
    data = _three_class_fixture_data()
    ds = data.spec
    cat = ConceptModel(annotate_rules([Rule.from_pos_neg(pos=[0], target="cat", dataspec=ds)], data),
                       label="cat", default_prediction="dog")
    dog = ConceptModel(annotate_rules([Rule.from_pos_neg(pos=[1], target="dog", dataspec=ds)], data),
                       label="dog", default_prediction="bird")
    ens = EnsembleModel([cat, dog], member_weights=[0.7, 0.3])
    text = ens.to_string(fmt="prolog")
    # .labels is rule heads + the ensemble's OWN default (unset here) --
    # "bird" never appears as either, only as a *sub-model's own* default,
    # so it's genuinely outside this model's declared label set
    assert "% classes: [cat, dog]" in text
    assert "% member 0  (weight: 0.7)" in text
    assert "% member 1  (weight: 0.3)" in text
    # cat's rule (a=1 -> cat) over all 6 training rows also covers the a=1
    # bird row as a false positive
    assert "cat(X) :- a(X).  % (2/1)" in text


def test_ensemble_prints_its_conflict_resolution():
    data = _three_class_fixture_data()
    ds = data.spec
    cat = ConceptModel(annotate_rules([Rule.from_pos_neg(pos=[0], target="cat", dataspec=ds)], data), label="cat")
    dog = ConceptModel(annotate_rules([Rule.from_pos_neg(pos=[1], target="dog", dataspec=ds)], data), label="dog")
    head = "% conflict resolution: "
    assert EnsembleModel([cat, dog]).to_string(fmt="prolog").startswith(head + "vote of members\n")
    assert EnsembleModel([cat, dog], member_weights=[0.7, 0.3]).to_string(fmt="prolog").startswith(
        head + "weighted vote of members\n")
    assert head not in EnsembleModel([cat, dog]).to_string(fmt="prolog", show_resolution=False)


# -- covered_by --------------------------------------------------------

def _covered_by_fixture():
    ds = DataSpec(["p", "q", "r"])
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
    y = np.array(["a", "a", "b", "b", "b", "a"])
    data = HorizontalBitmapRepresentation(ds, X, y)
    rp = Rule([0], target="a", dataspec=ds)       # p -> a: covers rows 0,1,2 (2/3 correct)
    rpq = Rule([0, 1], target="a", dataspec=ds)   # p & q -> a: covers rows 1,2 (1/2 correct)
    rr = Rule([2], target="b", dataspec=ds)       # r -> b: covers rows 2,3,4 (all correct)
    rq = Rule([1], target="b", dataspec=ds)       # q -> b: covers rows 1,2,3 (2/3 correct)
    # covered_by() calls predict() internally (to put the predicted
    # class's block first) -- annotate real stats so both the default
    # "max" combiner and covered_by's own default (Laplace-on-stats)
    # within-block sort have something to score. Laplace scores here:
    # rp=(2+1)/5=0.6, rpq=(1+1)/4=0.5, rr=(3+1)/5=0.8, rq=(2+1)/5=0.6
    # -- rq/rp tie exactly, broken by insertion order (rp built first).
    rules = annotate_rules([rp, rpq, rr, rq], data)
    fs = FlatRuleSet(rules, default_prediction="a")
    return ds, data, y, fs


def test_covered_by_blocks_predicted_class_first_sorted_by_score():
    ds, data, y, fs = _covered_by_fixture()
    preds = list(fs.predict(data))
    cb = fs.covered_by(data)  # default by=None -> Laplace-on-measured-stats

    assert len(cb) == data.n_samples
    for j, rules in enumerate(cb):
        if not rules:
            continue
        first_target = rules[0].target
        if preds[j] in {r.target for r in rules}:
            assert first_target == preds[j]
        targets = [r.target for r in rules]
        assert targets == sorted(targets, key=lambda t: targets.index(t))  # no interleaving

    # ex2 = [p,q,r]: covered by all 4, predicted "b" -> b-block (r:0.8, q:0.6) then a-block (p:0.6, pq:0.5)
    assert [r.pos for r in cb[2]] == [(2,), (1,), (0,), (0, 1)]
    # ex5 = [] : nothing covers -> the default rule
    assert len(cb[5]) == 1 and cb[5][0] is fs.default_rule


def test_covered_by_other_blocks_ordered_by_strongest_rule():
    ds = DataSpec(["p", "q", "s"])
    X = np.array([[1, 1, 1]], dtype=bool)
    data = HorizontalBitmapRepresentation(ds, X, np.array(["a"]))
    ra = Rule([0], target="a", dataspec=ds)
    rb = Rule([1], target="b", dataspec=ds)
    rc = Rule([2], target="c", dataspec=ds)
    # combiner=ListCombiner() so predict() (called internally by covered_by()) doesn't
    # need stats -- ra is first in list order, matching this fixture's
    # intended "a" prediction. b/c are structurally tied under any real
    # measurement on this single row, so an explicit `by=` (a plain
    # callable, no stats needed) exercises the "other blocks by their
    # strongest rule" mechanism directly rather than the stats-based default.
    fs = FlatRuleSet([ra, rb, rc], default_prediction="a", combiner=ListCombiner())
    scores = {"a": 0.95, "b": 0.40, "c": 0.70}
    (block,) = fs.covered_by(data, by=lambda r: scores[r.target])
    # predicted "a" first, then the *other* blocks by their strongest rule: c (0.70) before b (0.40)
    assert [r.target for r in block] == ["a", "c", "b"]


def test_covered_by_on_a_decision_list_and_with_no_default():
    ds = DataSpec(["a", "b"])
    X = np.array([[1, 1], [0, 0]], dtype=bool)
    data = HorizontalBitmapRepresentation(ds, X, np.array(["B", "A"]))
    dl = DecisionList([Rule([1], target="B", dataspec=ds), Rule([0], target="A", dataspec=ds)],
                     default_prediction=None)  # no constant default
    cb = dl.covered_by(data)  # singleton blocks -- no scoring needed, no stats required
    assert [r.target for r in cb[0]] == ["B", "A"]   # first-match B predicted -> its block first
    assert cb[1] == []                                # nothing covers, no default rule
