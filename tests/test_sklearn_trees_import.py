import numpy as np
from pyrulearn.combiners import CountVoteCombiner
from pyrulearn import HorizontalBitmapRepresentation, DataSpec, DataSpecBuilder, Rule

from _negation_helpers import neg_spec, neg_X


def test_stamp_rule_provenance_gives_each_rule_its_own_instance():
    from pyrulearn.interfaces.base import PatternStringImporter
    from pyrulearn.models import SingleRule

    ds = DataSpec(["a", "b"])
    r1 = Rule.from_pattern_string("1 0", dataspec=ds, target="pos")
    r2 = Rule.from_pattern_string("0 1", dataspec=ds, target="pos")
    importer = PatternStringImporter()
    wrapped = importer._stamp_rule_provenance([r1, r2], n_rules=2)

    assert all(type(r) is SingleRule for r in wrapped)
    assert [r.rule for r in wrapped] == [r1, r2]           # same underlying Rule, just wrapped
    assert all(r.provenance.learner == "PatternStringImporter" for r in wrapped)
    assert all(r.provenance.params == {"n_rules": 2} for r in wrapped)
    assert wrapped[0].provenance is not wrapped[1].provenance     # distinct instances (Option B)

    # an already-SingleRule item passes through, gets re-stamped in place
    already = SingleRule(r1)
    same_object = importer._stamp_rule_provenance([already], tag="x")[0]
    assert same_object is already
    assert same_object.provenance.params == {"tag": "x"}


def test_from_sklearn_tree_returns_disjoint_ruleset():
    from sklearn.tree import DecisionTreeClassifier
    from pyrulearn.interfaces.sklearn import from_sklearn_tree
    from pyrulearn.models import DisjointRuleSet

    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(100, 4)).astype(bool)
    y = np.where(X[:, 0] & ~X[:, 1], "pos", "neg")
    ds = neg_spec([f"f{i}" for i in range(4)])
    rep = HorizontalBitmapRepresentation(ds, neg_X(X), y)

    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    rules = from_sklearn_tree(clf, dataspec=ds)
    assert isinstance(rules, DisjointRuleSet)
    assert rules.is_disjoint(rep)
    assert rules.is_exhaustive(rep)  # every example reaches exactly one leaf
    # importer used directly (no learner) still stamps model-level provenance
    assert rules.provenance.learner == "SklearnTreeImporter"
    assert rules.provenance.source == "sklearn.tree.DecisionTreeClassifier"
    assert rules.provenance.params["max_depth"] == 3
    # ... and every individual rule now carries its own provenance too
    assert all(r.provenance is not None for r in rules.rules)
    assert rules.rules[0].provenance.learner == "SklearnTreeImporter"
    assert rules.rules[0].provenance.source == "sklearn.tree.DecisionTreeClassifier"
    assert len({id(r.provenance) for r in rules.rules}) == len(rules.rules)  # one instance each, not shared


def test_sklearn_tree_importer_binds_to_given_dataspec_and_tags_provenance():
    from sklearn.tree import DecisionTreeClassifier
    from pyrulearn.interfaces.sklearn import SklearnTreeImporter, from_sklearn_tree
    from pyrulearn.models import DisjointRuleSet

    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(100, 4)).astype(bool)
    y = np.where(X[:, 0] & ~X[:, 1], "pos", "neg")
    b = DataSpecBuilder(negation=True)
    for i in range(4):
        b.add_boolean(f"f{i}")
    ds = b.build()

    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    # dataspec= binds rules directly to the caller's own DataSpec object
    # (not a throwaway shadow copy built from just its feature names)
    rules_a = from_sklearn_tree(clf, dataspec=ds)
    assert all(r.dataspec is ds for r in rules_a)

    rules_b = SklearnTreeImporter().import_model(clf, ds)
    assert isinstance(rules_b, DisjointRuleSet)
    for r in rules_b:
        assert r.provenance.source == "sklearn.tree.DecisionTreeClassifier"
        assert r.provenance.learner == "SklearnTreeImporter"
        assert r.provenance.params["max_depth"] == 3


def test_sklearn_tree_importer_raises_on_feature_count_mismatch():
    from sklearn.tree import DecisionTreeClassifier
    from pyrulearn.interfaces.sklearn import SklearnTreeImporter

    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(50, 4)).astype(bool)
    y = np.where(X[:, 0], "pos", "neg")
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)

    wrong_ds = DataSpec(["only", "three", "features"])
    try:
        SklearnTreeImporter().import_model(clf, wrong_ds)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "features" in str(e)


def test_sklearn_tree_infer_dataspec_matches_tree_exactly_on_raw_data():
    # the whole point of infer_dataspec: fit directly on RAW continuous
    # data (no pre-binarization, no artificial threshold cap) and still
    # get a faithful rule-set re-expression of the tree
    from sklearn.tree import DecisionTreeClassifier
    from pyrulearn.interfaces.sklearn import SklearnTreeImporter
    from pyrulearn.data.io import binarize
    import pandas as pd

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = np.where((X[:, 0] > 0.5) & (X[:, 1] < -0.2), "pos", "neg")
    names = ["age", "income", "score"]

    clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    importer = SklearnTreeImporter()
    ds = importer.infer_dataspec(clf, feature_names=names)

    # every discovered threshold is one the tree actually used -- not a
    # fixed, pre-chosen cap
    real_thresholds = {float(t) for f, t in zip(clf.tree_.feature, clf.tree_.threshold) if f != -2}
    discovered_thresholds = {spec.value for spec in ds.feature_specs}
    assert discovered_thresholds == real_thresholds

    rules = importer.import_model(clf, ds, feature_names=names)
    Xb = binarize(ds, pd.DataFrame(X, columns=names))
    rep = HorizontalBitmapRepresentation(ds, Xb, y)

    tree_preds = clf.predict(X)
    rule_preds = rules.predict(rep)
    assert np.array_equal(tree_preds, rule_preds)


def test_random_forest_infer_dataspec_unions_thresholds_across_trees():
    from sklearn.ensemble import RandomForestClassifier
    from pyrulearn.interfaces.sklearn import RandomForestImporter, SklearnTreeImporter

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = np.where((X[:, 0] > 0.5) & (X[:, 1] < -0.2), "pos", "neg")
    names = ["age", "income", "score"]

    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0).fit(X, y)
    importer = RandomForestImporter()
    ds = importer.infer_dataspec(rf, feature_names=names)
    rules = importer.import_model(rf, ds, feature_names=names)

    # rule count matches importing each tree individually against the
    # SAME shared, forest-discovered dataspec and summing -- same
    # faithfulness check as the plain-Boolean forest test above
    expected_total = sum(
        len(SklearnTreeImporter().import_model(t, ds, feature_names=names)) for t in rf.estimators_
    )
    assert len(rules) == expected_total


def test_random_forest_importer_returns_single_ruleset():
    from sklearn.ensemble import RandomForestClassifier
    from pyrulearn.interfaces.sklearn import RandomForestImporter, SklearnTreeImporter, from_random_forest
    from pyrulearn.models import DisjointRuleSet, FlatRuleSet

    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 4)).astype(bool)
    y = np.where(X[:, 0] & ~X[:, 1], "pos", "neg")
    ds = neg_spec([f"f{i}" for i in range(4)])
    rep = HorizontalBitmapRepresentation(ds, neg_X(X), y)

    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0).fit(X, y)
    rules = RandomForestImporter().import_model(rf, ds)

    assert isinstance(rules, FlatRuleSet)
    assert not isinstance(rules, DisjointRuleSet)  # overlapping across trees, not disjoint

    # rule count matches importing each tree individually and summing
    expected_total = sum(len(SklearnTreeImporter().import_model(t, ds)) for t in rf.estimators_)
    assert len(rules) == expected_total

    # provenance: every rule knows which tree it came from
    tree_indices = {r.provenance.params["tree_index"] for r in rules}
    assert tree_indices == set(range(5))
    for r in rules:
        assert r.provenance.source == "sklearn.ensemble.RandomForestClassifier"
        assert r.provenance.params["n_estimators"] == 5

    # every example gets exactly n_estimators covering rules -- each
    # tree is individually exhaustive, so the combined set always is too
    cov = rules.coverage_matrix(rep)
    assert list(cov.sum(axis=0)) == [5] * rep.n_samples

    # unweighted majority vote reasonably tracks the true labels (sanity
    # check, not exactness -- sklearn's own predict() does soft/
    # probability-averaged voting, not hard majority voting)
    preds = rules.predict(rep, combiner=CountVoteCombiner())
    assert np.mean(preds == y) > 0.9

    rules2 = from_random_forest(rf, dataspec=ds)
    assert isinstance(rules2, FlatRuleSet)


def test_random_forest_import_model_with_data_gives_every_leaf_measured_stats():
    # DistributionCombiner (micro_vote/macro_vote/micro_max/macro_max)
    # requires every covering rule to have measured stats -- for a forest,
    # that means passing data= to import_model (see its docstring); the
    # actual micro-vs-macro divergence mechanics are unit-tested directly,
    # with hand-crafted differently-sized leaves, in
    # test_combiners.py::test_distribution_combiners_micro_vs_macro_genuinely_diverge.
    from sklearn.ensemble import RandomForestClassifier
    from pyrulearn.interfaces.sklearn import RandomForestImporter
    from pyrulearn.combiners import MicroVoteCombiner, MacroVoteCombiner

    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(300, 6)).astype(bool)
    noise = rng.random(300) < 0.3
    y_clean = np.where(X[:, 0] & ~X[:, 1], "pos", "neg")
    y = np.where(noise, np.where(y_clean == "pos", "neg", "pos"), y_clean)
    ds = neg_spec([f"f{i}" for i in range(6)])
    rep = HorizontalBitmapRepresentation(ds, neg_X(X), y)

    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1).fit(X, y)
    bare_rules = RandomForestImporter().import_model(rf, ds)  # no data= -- no stats
    assert all(r.stats() is None for r in bare_rules)
    try:
        bare_rules.predict(rep, combiner=MicroVoteCombiner())
        assert False, "expected ValueError without measured stats"
    except ValueError:
        pass

    rules = RandomForestImporter().import_model(rf, ds, data=rep)  # data= -> real measured stats
    assert all(r.stats() is not None for r in rules)
    preds_micro = rules.predict(rep, combiner=MicroVoteCombiner())
    preds_macro = rules.predict(rep, combiner=MacroVoteCombiner())
    assert set(preds_micro) <= {"pos", "neg"} and set(preds_macro) <= {"pos", "neg"}


def test_tree_learners_on_the_fit_switcher():
    # SKLDecisionTree / SKLRandomForest route through fit(data, model=...) like
    # any other learner (the external-wrapper switcher).
    from pyrulearn.interfaces.sklearn import SKLDecisionTree, SKLRandomForest
    from pyrulearn.models import ConceptSet, DisjointRuleSet, EnsembleModel, FlatRuleSet

    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 5)).astype(bool)
    y = np.where(X[:, 0] & ~X[:, 1], "a", np.where(X[:, 2], "b", "c"))
    ds = neg_spec([f"f{i}" for i in range(5)])
    rep = HorizontalBitmapRepresentation(ds, neg_X(X), y)

    dt = SKLDecisionTree(max_depth=4, random_state=0)
    assert dt.produces() >= {DisjointRuleSet, ConceptSet}
    m = dt.fit(rep)
    assert type(m) is DisjointRuleSet
    assert type(dt.fit(rep, model=ConceptSet)) is ConceptSet
    assert set(m.predict(rep)) <= {"a", "b", "c"}

    rf = SKLRandomForest(n_estimators=6, max_depth=3, random_state=0)
    assert rf.produces() == {EnsembleModel, FlatRuleSet}
    ens = rf.fit(rep)
    assert type(ens) is EnsembleModel and len(ens.members) == 6
    assert all(type(mem) is DisjointRuleSet for mem in ens.members)
    assert type(rf.fit(rep, model=FlatRuleSet)) is FlatRuleSet
    assert set(ens.predict(rep)) <= {"a", "b", "c"}

    # each member's rules share one tree_index (via their own .provenance,
    # not shared meta) and no two members' rules share a tree_index
    seen_indices = set()
    for mem in ens.members:
        indices = {r.provenance.params["tree_index"] for r in mem.rules}
        assert len(indices) == 1          # one tree per member
        assert indices.isdisjoint(seen_indices)
        seen_indices |= indices


def test_sklearn_tree_importer_stamps_stats_only_when_data_is_given():
    from sklearn.tree import DecisionTreeClassifier
    from pyrulearn.evaluation import RuleStats
    from pyrulearn.interfaces.sklearn import SKLDecisionTree, SklearnTreeImporter

    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(150, 4)).astype(bool)
    y = np.where(X[:, 0] & ~X[:, 1], "pos", "neg")
    ds = neg_spec([f"f{i}" for i in range(4)])
    rep = HorizontalBitmapRepresentation(ds, neg_X(X), y)

    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(rep.X, y)
    bare = SklearnTreeImporter().import_model(clf, ds)  # no data= -- standalone call
    assert all(r.stats() is None for r in bare.rules)

    fitted = SKLDecisionTree(max_depth=3, random_state=0).fit(rep)  # fit() round trip -> data=rep passed through
    for r in fitted.rules:
        expected = RuleStats.from_rule(r.rule, rep, r.target)
        got = r.stats().confusion.rule_stats(r.target)
        assert (got.tp, got.fp, got.fn, got.tn) == (expected.tp, expected.fp, expected.fn, expected.tn)


def test_prettify_threshold_finds_the_coarsest_value_in_the_real_data_gap():
    from pyrulearn.data.discretization import prettify_threshold

    sorted_unique = np.array([1.0, 4.0, 15.1458, 15.2, 30.0])
    ugly = (15.1458 + 15.2) / 2  # 15.172899999999998 -- sklearn's own style of split point
    pretty = prettify_threshold(ugly, sorted_unique)
    assert pretty == 15.17                       # coarsely rounded, not sklearn's raw many-digit value
    assert 15.1458 < pretty < 15.2               # still the same open gap -- identical partition

    # a value sitting in a genuinely tight real gap can't be shortened
    # below the precision that gap actually needs
    tight_unique = np.array([15.1458, 15.14581])
    tight = prettify_threshold(15.145805, tight_unique)
    assert 15.1458 < tight < 15.14581


def test_tree_thresholds_prettifies_without_changing_the_partition():
    from pyrulearn.interfaces.sklearn import tree_thresholds

    # two distinct values only, so there is exactly one possible split,
    # at their exact midpoint -- sklearn would return that midpoint
    # verbatim (an ugly float; see the assertion below), unprettified
    values = np.array([15.1458] * 5 + [15.2] * 5)
    y = np.array([0] * 5 + [1] * 5)
    raw_midpoint = (15.1458 + 15.2) / 2
    assert repr(raw_midpoint) == "15.172899999999998"  # documents the artifact this fix removes

    thresholds = tree_thresholds(values, y, max_intervals=2)
    assert thresholds == [15.17]
    # the prettified threshold partitions this column exactly like the
    # raw midpoint would -- rounding changed nothing but the display
    assert np.array_equal(values >= thresholds[0], values >= raw_midpoint)
