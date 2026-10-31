"""`NListRepresentation`, `PrePostNListRepresentation` and
`SparseDataRepresentation` must all be drop-ins for
`HorizontalBitmapRepresentation`: identical `coverage`, identical seed
features, identical rules from every `pyrulearn.learners.seco` learner -- plus
the `with_negations` / `without_negations` toggle on all four."""

import numpy as np
import pandas as pd
import pytest

from pyrulearn import (
    HorizontalBitmapRepresentation,
    DataSpec,
    NListRepresentation,
    PrePostNListRepresentation,
    Rule,
    SparseDataRepresentation,
    VerticalBitmapRepresentation,
)
from pyrulearn.models import DecisionList, FlatRuleSet
from pyrulearn.data import DataSpecBuilder
from pyrulearn.heuristics import RuleStats
from pyrulearn.learners.boosting import DenseENDER, ENDER
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.learners.seco import AQR, CN2, PFoil, PFossil

from _negation_helpers import neg_spec, neg_X

ALT_REPS = [NListRepresentation, PrePostNListRepresentation, SparseDataRepresentation,
            VerticalBitmapRepresentation]
ALL_REPS = [HorizontalBitmapRepresentation, NListRepresentation, PrePostNListRepresentation, SparseDataRepresentation,
            VerticalBitmapRepresentation]


def _pos_spec(names):
    b = DataSpecBuilder(negation=False)
    for n in names:
        b.add_boolean(n)
    return b.build()


def _dataset(n=400, k=14, seed=0, negation=False):
    rng = np.random.default_rng(seed)
    base = rng.random((n, k)) < rng.uniform(0.2, 0.6, size=k)
    concept = (base[:, 0] & ~base[:, 1]) | (base[:, 2] & base[:, 3])
    y = np.where(concept ^ (rng.random(n) < 0.05), "pos", "neg")
    if negation:
        ds = neg_spec([f"a{i}" for i in range(k)])
        return HorizontalBitmapRepresentation(ds, neg_X(base), y), ds
    ds = DataSpec([f"f{i}" for i in range(k)])
    return HorizontalBitmapRepresentation(ds, base, y), ds


def _random_rules(ds, rng, n_rules=300, max_len=5):
    k = ds.n_features
    out = []
    for _ in range(n_rules):
        length = int(rng.integers(0, max_len))
        feats = sorted(int(f) for f in rng.choice(k, size=length, replace=False))
        out.append(Rule(feats, target="pos", dataspec=ds))
    return out


@pytest.mark.parametrize("rep_cls", ALT_REPS)
@pytest.mark.parametrize("negation", [False, True])
def test_coverage_matches_boolean(rep_cls, negation):
    brep, ds = _dataset(seed=1, negation=negation)
    rep = rep_cls.from_boolean(brep)
    rng = np.random.default_rng(0)
    for rule in _random_rules(ds, rng, n_rules=500):
        assert np.array_equal(brep.coverage(rule), rep.coverage(rule)), rule
    assert rep.coverage(Rule([], target="pos", dataspec=ds)).all()  # empty rule covers all


@pytest.mark.parametrize("rep_cls", ALT_REPS)
def test_features_of_and_X_match(rep_cls):
    brep, ds = _dataset(seed=2)
    rep = rep_cls.from_boolean(brep)
    assert np.array_equal(rep.X, brep.X)
    for i in range(brep.n_samples):
        assert np.array_equal(rep.features_of(i), brep.features_of(i))


@pytest.mark.parametrize("rep_cls", ALT_REPS)
def test_example_mask_counts_match(rep_cls):
    brep, ds = _dataset(n=500, seed=3)
    rep = rep_cls.from_boolean(brep)
    rng = np.random.default_rng(1)
    mask = rng.random(500) < 0.55
    for rule in _random_rules(ds, rng, n_rules=200, max_len=4):
        sb = RuleStats.from_rule(rule, brep, "pos", example_mask=mask)
        sr = RuleStats.from_rule(rule, rep, "pos", example_mask=mask)
        assert (sb.tp, sb.fp, sb.fn, sb.tn) == (sr.tp, sr.fp, sr.fn, sr.tn), rule


@pytest.mark.parametrize("negation", [False, True])
def test_seco_learners_identical_across_representations(negation):
    brep, ds = _dataset(n=500, k=16, seed=7, negation=negation)
    nrep = NListRepresentation.from_boolean(brep)
    pprep = PrePostNListRepresentation.from_boolean(brep)
    srep = SparseDataRepresentation.from_boolean(brep)
    for make in (
        lambda: CN2(target_class="pos"),
        lambda: PFoil(target_class="pos"),
        lambda: PFossil(target_class="pos"),
        lambda: AQR(target_class="pos", maxstar=4),
        lambda: PyLORD(m=0.1, random_state=0),
    ):
        rb, rn, rp, rs = make().fit(brep), make().fit(nrep), make().fit(pprep), make().fit(srep)
        kb = {(r.pos, r.target) for r in rb.rules}
        assert kb == {(r.pos, r.target) for r in rn.rules}
        assert kb == {(r.pos, r.target) for r in rp.rules}
        assert kb == {(r.pos, r.target) for r in rs.rules}
        pb = np.asarray(rb.predict(brep))
        assert np.array_equal(pb, np.asarray(rn.predict(nrep)))
        assert np.array_equal(pb, np.asarray(rp.predict(pprep)))
        assert np.array_equal(pb, np.asarray(rs.predict(srep)))


@pytest.mark.parametrize("cls", [ENDER, DenseENDER])
def test_ender_identical_across_representations(cls):
    """`ENDER` grows its rules through the representation's own primitives
    (`initial_cover`/`refine_cover`/`batch_cover_sums`); `DenseENDER`
    converts every representation to a dense matrix first. Both must
    learn the same model on every representation -- and the same as each
    other (`tests/test_boosting.py`)."""
    brep, ds = _dataset(n=300, k=12, seed=11)
    nrep = NListRepresentation.from_boolean(brep)
    pprep = PrePostNListRepresentation.from_boolean(brep)
    srep = SparseDataRepresentation.from_boolean(brep)

    def rules_of(model):
        return {(r.target, frozenset(l.feature for l in r.conditions), round(float(r.weight), 9))
                for r in model.rules}

    make = lambda: cls(n_rules=30, random_state=0)
    mb, mn, mp, ms = make().fit(brep), make().fit(nrep), make().fit(pprep), make().fit(srep)
    kb = rules_of(mb)
    assert kb == rules_of(mn)
    assert kb == rules_of(mp)
    assert kb == rules_of(ms)
    pb = np.asarray(mb.predict(brep))
    assert np.array_equal(pb, np.asarray(mn.predict(nrep)))
    assert np.array_equal(pb, np.asarray(mp.predict(pprep)))
    assert np.array_equal(pb, np.asarray(ms.predict(srep)))


def test_seco_learners_identical_on_wide_data():
    """Same check as test_seco_learners_identical_across_representations,
    but wide enough (k=150, a broad range of feature frequencies) that a
    real search actually drives `PrePostNListRepresentation.refine_cover`
    into its "deeper feature" re-anchor case repeatedly, including the
    interval-join branch -- not just the narrow k=16 case."""
    rng = np.random.default_rng(3)
    n, k = 1200, 150
    X = rng.random((n, k)) < rng.uniform(0.02, 0.5, size=k)
    concept = (X[:, 0] & ~X[:, 1]) | (X[:, 2] & X[:, 3])
    y = np.where(concept ^ (rng.random(n) < 0.05), "pos", "neg")
    ds = DataSpec([f"f{i}" for i in range(k)])
    brep = HorizontalBitmapRepresentation(ds, X, y)
    nrep = NListRepresentation.from_boolean(brep)
    pprep = PrePostNListRepresentation.from_boolean(brep)
    for make in (
        lambda: CN2(target_class="pos"),
        lambda: PFoil(target_class="pos"),
        lambda: PFossil(target_class="pos"),
        lambda: AQR(target_class="pos", maxstar=4),
        lambda: PyLORD(m=0.1, random_state=0),
    ):
        rb, rn, rp = make().fit(brep), make().fit(nrep), make().fit(pprep)
        kb = {(r.pos, r.target) for r in rb.rules}
        assert kb == {(r.pos, r.target) for r in rn.rules}
        assert kb == {(r.pos, r.target) for r in rp.rules}
        pb = np.asarray(rb.predict(brep))
        assert np.array_equal(pb, np.asarray(rn.predict(nrep)))
        assert np.array_equal(pb, np.asarray(rp.predict(pprep)))


@pytest.mark.parametrize("rep_cls", ALT_REPS)
def test_ruleclassifier_coverage_methods_match_boolean(rep_cls):
    """`RuleModel`'s coverage/prediction methods only touch the
    `DataRepresentation` ABC surface (`coverage`, `n_samples`, `y`), so
    they must give identical results on any data."""
    brep, ds = _dataset(n=500, k=16, seed=11)
    rep = rep_cls.from_boolean(brep)
    learned = CN2(target_class="pos").fit(brep)          # a FlatRuleSet with a default
    rules = list(learned.rules)

    for Cls in (FlatRuleSet, DecisionList):
        cb, cx = Cls(rules, default_prediction="neg"), Cls(rules, default_prediction="neg")

        assert np.array_equal(cb.coverage_matrix(brep), cx.coverage_matrix(rep))
        assert np.array_equal(np.asarray(cb.predict(brep)), np.asarray(cx.predict(rep)))
        assert cb.is_disjoint(brep) == cx.is_disjoint(rep)
        assert cb.is_exhaustive(brep) == cx.is_exhaustive(rep)
        assert np.array_equal(cb.coverage_space(brep, "pos"), cx.coverage_space(rep, "pos"))

        sb, sx = cb.evaluate(brep), cx.evaluate(rep)
        assert sb.confusion.labels == sx.confusion.labels and \
            np.array_equal(sb.confusion.counts, sx.confusion.counts)
        db, dx = cb.default_rule.evaluate(brep), cx.default_rule.evaluate(rep)
        assert db.confusion.labels == dx.confusion.labels and \
            np.array_equal(db.confusion.counts, dx.confusion.counts)


@pytest.mark.parametrize("rep_cls", ALL_REPS)
def test_select_rows_subsets_and_stays_the_same_type(rep_cls):
    brep, ds = _dataset(n=300, k=12, seed=13)
    rep = rep_cls.from_boolean(brep) if rep_cls is not HorizontalBitmapRepresentation else brep
    mask = brep.y == "pos"

    sub = rep.select_rows(mask)
    assert type(sub) is rep_cls
    assert sub.n_samples == int(mask.sum())
    assert np.array_equal(sub.y, brep.y[mask])
    assert np.array_equal(np.asarray(sub.X), np.asarray(brep.X)[mask])
    assert sub.spec is rep.spec  # feature space unchanged

    # a rule's coverage on the subset == its coverage on the full data, masked
    rng = np.random.default_rng(0)
    for rule in _random_rules(ds, rng, n_rules=50, max_len=4):
        assert np.array_equal(sub.coverage(rule), brep.coverage(rule)[mask])


def _masks(n, y, rng):
    yield "random", rng.random(n) < 0.4
    yield "one class", y == "pos"
    yield "all rows", np.ones(n, dtype=bool)
    yield "a single row", np.arange(n) == 7
    yield "no rows", np.zeros(n, dtype=bool)
    yield "sorted indices", np.flatnonzero(rng.random(n) < 0.3)


@pytest.mark.parametrize("rep_cls", [NListRepresentation, PrePostNListRepresentation])
@pytest.mark.parametrize("negation", [False, True])
def test_nlist_subset_derived_from_the_tree_matches_a_rebuilt_one(rep_cls, negation):
    """select_rows derives the subset's tree from the existing one (keeping its
    feature order); everything a learner can observe must equal a fresh build."""
    brep, ds = _dataset(n=300, k=12, seed=21, negation=negation)
    X, y = np.asarray(brep.X), brep.y
    weights = np.random.default_rng(1).uniform(0.5, 2.0, size=len(y))
    rep = rep_cls(ds, X, y, weights=weights)
    rng = np.random.default_rng(5)
    rules = _random_rules(ds, rng, n_rules=60, max_len=4)
    for name, mask in _masks(len(y), y, rng):
        sub = rep.select_rows(mask)
        fresh = rep_cls(ds, X[mask], y[mask], weights=weights[mask])
        assert type(sub) is rep_cls and sub.n_samples == fresh.n_samples, name
        assert np.array_equal(sub.y, fresh.y) and np.array_equal(sub.weights, fresh.weights), name
        assert np.array_equal(np.asarray(sub.X), np.asarray(fresh.X)), name
        for row in range(sub.n_samples):
            assert np.array_equal(sub.features_of(row), fresh.features_of(row)), name
        for rule in rules:
            assert np.array_equal(sub.coverage(rule), fresh.coverage(rule)), name
        if sub.n_samples and len(set(sub.y)) > 1:
            # the search path: incremental cover handles and their class counts
            for rule in rules[:20]:
                hs, hf = sub.initial_cover(), fresh.initial_cover()
                for f in rule.pos:
                    hs, hf = sub.refine_cover(hs, f), fresh.refine_cover(hf, f)
                # weighted sums: same rows, added up in another order (the tree keeps the
                # full data's feature order), so equal up to floating-point rounding
                assert np.allclose(sub.cover_counts(hs, "pos"), fresh.cover_counts(hf, "pos"),
                                   rtol=1e-12, atol=0), name
            for learner in (CN2(random_state=0), PFossil(random_state=0)):
                assert str(learner.fit(sub)) == str(learner.fit(fresh)), (name, type(learner).__name__)
    nested = rep.select_rows(y == "pos").select_rows(np.arange(int((y == "pos").sum())) % 2 == 0)
    once = rep_cls(ds, X[y == "pos"][::2], y[y == "pos"][::2], weights=weights[y == "pos"][::2])
    assert np.array_equal(np.asarray(nested.X), np.asarray(once.X))
    assert all(np.array_equal(nested.coverage(r), once.coverage(r)) for r in rules)


@pytest.mark.parametrize("rep_cls", [NListRepresentation, PrePostNListRepresentation])
def test_nlist_select_rows_with_unsorted_or_repeated_indices_rebuilds(rep_cls):
    brep, ds = _dataset(n=120, k=10, seed=3)
    X, y = np.asarray(brep.X), brep.y
    rep = rep_cls(ds, X, y)
    for idx in (np.array([5, 3, 9, 3, 0]), np.array([10, 2, 77])):
        sub = rep.select_rows(idx)
        assert np.array_equal(np.asarray(sub.X), X[idx]) and np.array_equal(sub.y, y[idx])


@pytest.mark.parametrize("rep_cls", [NListRepresentation, PrePostNListRepresentation])
def test_nlist_relabel_shares_the_tree(rep_cls):
    brep, ds = _dataset(n=200, k=12, seed=8)
    X, y = np.asarray(brep.X), brep.y
    rep = rep_cls(ds, X, y)
    new_y = np.where(np.arange(len(y)) % 3 == 0, "a", np.where(y == "pos", "b", "c"))
    rel, fresh = rep.relabel(new_y), rep_cls(ds, X, new_y)
    assert rel._words is rep._words  # shared, not rebuilt
    assert np.array_equal(rel.y, new_y) and np.array_equal(rep.y, y)  # the original keeps its labels
    for rule in _random_rules(ds, np.random.default_rng(2), n_rules=40, max_len=3):
        assert np.array_equal(rel.coverage(rule), fresh.coverage(rule))
    assert str(CN2(random_state=0).fit(rel)) == str(CN2(random_state=0).fit(fresh))


@pytest.mark.parametrize("rep_cls", ALT_REPS)
def test_constructors_mirror_boolean(rep_cls):
    rng = np.random.default_rng(4)
    X = rng.random((120, 8)) < 0.4
    y = np.where(X[:, 0], "a", "b")

    from_xy = rep_cls.from_xy(X, y, feature_names=[f"g{i}" for i in range(8)])
    assert from_xy.spec.feature_names == [f"g{i}" for i in range(8)]
    assert np.array_equal(from_xy.X, X)
    assert np.array_equal(from_xy.y, y)

    df = pd.DataFrame(X.astype(int), columns=[f"g{i}" for i in range(8)])
    df["label"] = y
    from_df = rep_cls.from_dataframe(df, label_col="label")
    assert np.array_equal(from_df.X, X)
    assert list(from_df.y) == list(y)


@pytest.mark.parametrize("rep_cls", ALL_REPS)
def test_from_scipy_accepts_a_sparse_matrix(rep_cls):
    sparse = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(8)
    X = (rng.random((80, 6)) < 0.3)
    ds = DataSpec([f"f{i}" for i in range(6)])
    rep = rep_cls.from_scipy(sparse.csr_matrix(X), ds, np.where(X[:, 0], "a", "b"))
    assert np.array_equal(rep.X, X)
    rule = Rule([0, 2], target="a", dataspec=ds)
    assert np.array_equal(rep.coverage(rule), X[:, 0] & X[:, 2])


@pytest.mark.parametrize("rep_cls", ALT_REPS)
def test_all_zero_row_and_shape_guard(rep_cls):
    ds = DataSpec(["a", "b", "c"])
    X = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    rep = rep_cls(ds, X, np.array(["p", "p", "n"]))
    assert rep.features_of(0).size == 0
    assert rep.coverage(Rule([0], target="p", dataspec=ds)).tolist() == [False, True, False]
    with pytest.raises(ValueError):
        rep_cls(ds, np.zeros((3, 5), dtype=bool))


@pytest.mark.parametrize("rep_cls", ALL_REPS)
@pytest.mark.parametrize("use_mask", [False, True])
def test_incremental_cover_matches_rulestats(rep_cls, use_mask):
    """initial_cover/refine_cover/cover_counts/cover_rows -- the
    BeamSearch/HillClimbing fast path -- must agree with the ordinary
    RuleStats.from_rule/coverage path exactly, feature-by-feature."""
    brep, ds = _dataset(n=400, k=12, seed=13)
    rep = brep if rep_cls is HorizontalBitmapRepresentation else rep_cls.from_boolean(brep)
    rng = np.random.default_rng(5)
    mask = (rng.random(400) < 0.6) if use_mask else None
    for rule in _random_rules(ds, rng, n_rules=200, max_len=5):
        handle = rep.initial_cover(mask)
        for lit in rule.conditions:
            handle = rep.refine_cover(handle, lit.feature)
        got = rep.cover_counts(handle, "pos")
        want = RuleStats.from_rule(rule, brep, "pos", example_mask=mask)
        assert got == (want.tp, want.fp, want.fn, want.tn), (rep_cls.__name__, rule, use_mask)

        full_cov = brep.coverage(rule)
        want_rows = full_cov if mask is None else (full_cov & mask)
        assert np.array_equal(rep.cover_rows(handle), want_rows)


@pytest.mark.parametrize("rep_cls", [NListRepresentation, PrePostNListRepresentation])
def test_nlist_incremental_cover_on_wide_varied_frequency_data(rep_cls):
    """Same equivalence check as test_incremental_cover_matches_rulestats,
    but wide and with a broad spread of feature frequencies, so
    `refine_cover`'s "deeper feature" re-anchor case (not just "no deeper
    than the anchor") gets exercised repeatedly, both masked and
    unmasked -- including, for `PrePostNListRepresentation`, both of
    `_reanchor`'s two strategies."""
    rng = np.random.default_rng(7)
    n, k = 2000, 300
    X = rng.random((n, k)) < rng.uniform(0.02, 0.5, size=k)  # wide range of frequencies
    y = np.where(X[:, 0] & ~X[:, 1], "pos", "neg")
    ds = DataSpec([f"f{i}" for i in range(k)])
    brep = HorizontalBitmapRepresentation(ds, X, y)
    rep = rep_cls.from_boolean(brep)

    branch_counts = {"bitmask": 0, "interval": 0}
    orig_reanchor = getattr(rep_cls, "_reanchor", None)
    if orig_reanchor is not None:
        def counted(self, anchor_item, active, feature, mask_words):
            words = self._words_of(feature)
            branch_counts["bitmask" if (active.size == 0 or active.size >= words.shape[0]) else "interval"] += 1
            return orig_reanchor(self, anchor_item, active, feature, mask_words)
        rep_cls._reanchor = counted

    try:
        for trial in range(200):
            length = int(rng.integers(1, 10))
            order = [int(f) for f in rng.choice(k, size=length, replace=False)]
            mask = None if trial % 3 else (rng.random(n) < 0.7)
            h = rep.initial_cover(mask)
            for step, f in enumerate(order, start=1):
                h = rep.refine_cover(h, f)
                rule = Rule(sorted(order[:step]), target="pos", dataspec=ds)
                want = RuleStats.from_rule(rule, brep, "pos", example_mask=mask)
                assert rep.cover_counts(h, "pos") == (want.tp, want.fp, want.fn, want.tn)
                full_cov = brep.coverage(rule)
                want_rows = full_cov if mask is None else (full_cov & mask)
                assert np.array_equal(rep.cover_rows(h), want_rows)
    finally:
        if orig_reanchor is not None:
            rep_cls._reanchor = orig_reanchor

    if orig_reanchor is not None:
        assert branch_counts["interval"] > 0 and branch_counts["bitmask"] > 0


def test_nlist_tree_is_more_compact_than_the_dense_matrix():
    # rows sharing prefixes collapse: fewer tree nodes than n*k matrix cells
    brep, _ = _dataset(n=600, k=20, seed=11)
    nrep = NListRepresentation.from_boolean(brep)
    n_nodes = len(nrep._words)
    assert n_nodes < brep.n_samples * brep.spec.n_features


def test_sparse_is_the_nlist_without_the_tree():
    # SparseDataRepresentation's per-feature row lists ARE the N-lists of a
    # PPC-tree that did no prefix merging -- coverage must agree exactly.
    brep, ds = _dataset(n=300, k=12, seed=5)
    srep = SparseDataRepresentation.from_boolean(brep)
    nrep = NListRepresentation.from_boolean(brep)
    rng = np.random.default_rng(2)
    for rule in _random_rules(ds, rng, n_rules=300, max_len=4):
        cov = brep.coverage(rule)
        assert np.array_equal(cov, srep.coverage(rule))
        assert np.array_equal(cov, nrep.coverage(rule))


# -- negation toggles ------------------------------------------------

@pytest.mark.parametrize("rep_cls", ALL_REPS)
def test_without_then_with_negations_round_trips(rep_cls):
    names = list("abcde")
    rng = np.random.default_rng(3)
    base = rng.random((300, 5)) < 0.4
    y = np.where(base[:, 0] & ~base[:, 1], "pos", "neg")
    bneg = HorizontalBitmapRepresentation(neg_spec(names), neg_X(base), y)
    rep = bneg if rep_cls is HorizontalBitmapRepresentation else rep_cls.from_boolean(bneg)

    pos = rep.without_negations()
    assert type(pos) is rep_cls
    assert pos.spec.n_features == 5 and not pos.spec.has_negation_features
    assert np.array_equal(pos.X, neg_X(base)[:, 0::2])   # positive columns, untouched

    back = pos.with_negations()
    assert type(back) is rep_cls
    assert back.spec.n_features == 10
    assert np.array_equal(back.X, bneg.X)

    r_pos = Rule([pos.spec.feature_index("a")], target="pos", dataspec=pos.spec)
    r_neg = Rule([bneg.spec.feature_index("a")], target="pos", dataspec=bneg.spec)
    assert np.array_equal(pos.coverage(r_pos), bneg.coverage(r_neg))


@pytest.mark.parametrize("rep_cls", ALL_REPS)
def test_without_negations_is_identity_without_negation_features(rep_cls):
    rng = np.random.default_rng(6)
    X = rng.random((100, 4)) < 0.5
    rep = rep_cls(_pos_spec(list("abcd")), X, None)
    assert rep.without_negations() is rep


def test_negation_toggle_gives_every_learner_the_same_rules_as_a_native_build():
    # toggling the data must match building the spec that way from scratch
    names = list("abcdef")
    rng = np.random.default_rng(9)
    base = rng.random((360, 6)) < 0.4
    y = np.where(base[:, 0] & ~base[:, 1] | base[:, 2], "pos", "neg")

    native_pos = HorizontalBitmapRepresentation(_pos_spec(names), base, y)
    toggled_pos = HorizontalBitmapRepresentation(neg_spec(names), neg_X(base), y).without_negations()

    for make in (lambda: CN2(target_class="pos"), lambda: PFoil(target_class="pos")):
        a = {(r.pos, r.target) for r in make().fit(native_pos).rules}
        b = {(r.pos, r.target) for r in make().fit(toggled_pos).rules}
        # feature indices line up because without_negations keeps names/order
        assert a == b


def test_preference_order_and_default_representation(monkeypatch):
    import pandas as pd
    from pyrulearn.data import representation
    from pyrulearn.data.io import encode
    from pyrulearn.learners import base

    assert representation.REPRESENTATION_PREFERENCE_ORDER == (
        VerticalBitmapRepresentation, NListRepresentation, PrePostNListRepresentation,
        HorizontalBitmapRepresentation, SparseDataRepresentation)
    assert representation.default_representation() is VerticalBitmapRepresentation
    assert base.REPRESENTATION_PREFERENCE_ORDER is representation.REPRESENTATION_PREFERENCE_ORDER
    ds = DataSpec(["f0", "f1"])
    df = pd.DataFrame({"f0": [True, False], "f1": [False, True]})
    assert isinstance(encode(ds, df), VerticalBitmapRepresentation)
    # read when called: reordering the module attribute changes the library-wide default
    monkeypatch.setattr(representation, "REPRESENTATION_PREFERENCE_ORDER",
                        (HorizontalBitmapRepresentation, NListRepresentation))
    assert representation.default_representation() is HorizontalBitmapRepresentation
    assert type(encode(ds, df)) is HorizontalBitmapRepresentation


@pytest.mark.parametrize("rep_cls", [NListRepresentation, PrePostNListRepresentation, SparseDataRepresentation,
                                     VerticalBitmapRepresentation])
def test_external_learner_converts_any_representation(rep_cls):
    from pyrulearn.interfaces.sklearn import SKLDecisionTree
    brep, _ = _dataset(n=300, k=10, seed=3, negation=True)
    other = rep_cls.from_boolean(brep)
    learner = SKLDecisionTree(max_depth=3, random_state=0)
    assert learner.ensure_representation(other, None, announce=False).__class__ is HorizontalBitmapRepresentation
    mb, mo = SKLDecisionTree(max_depth=3, random_state=0).fit(brep), learner.fit(other)
    assert str(mb) == str(mo)
    assert np.array_equal(np.asarray(mb.predict(brep)), np.asarray(mo.predict(other)))


@pytest.mark.parametrize("chunk_cells", [1, 37, 1 << 16])
@pytest.mark.parametrize("n", [0, 1, 129])
def test_to_csr_is_the_matrix_without_caching_it(n, chunk_cells):
    # built a block of rows at a time (chunk_cells=1: one row per block), int32 indices,
    # and the Boolean source keeps no unpacked X -- what from_boolean of the sparse and
    # N-list representations relies on
    rng = np.random.default_rng(n)
    X = rng.random((n, 19)) < 0.3
    brep = HorizontalBitmapRepresentation(DataSpec([f"f{i}" for i in range(19)]), X, np.array(["a"] * n))
    csr = brep.to_csr(chunk_cells=chunk_cells)
    assert csr.shape == (n, 19) and csr.dtype == bool
    assert csr.indices.dtype == np.int32 and csr.indptr.dtype == np.int32
    assert np.array_equal(csr.toarray(), X) and csr.has_sorted_indices
    assert "X" not in vars(brep)
    for rep_cls in ALT_REPS:
        rep_cls.from_boolean(brep)
        assert "X" not in vars(brep)


def test_boolean_data_representation_is_removed():
    import importlib

    for module in ("pyrulearn", "pyrulearn.data", "pyrulearn.data.representation"):
        assert not hasattr(importlib.import_module(module), "BooleanDataRepresentation")
