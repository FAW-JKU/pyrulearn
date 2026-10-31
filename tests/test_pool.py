from pyrulearn.learners.ids import GreedySelection
import numpy as np
import pytest

from pyrulearn.data import HorizontalBitmapRepresentation, NListRepresentation
from pyrulearn.learners.associative import CARMiner, sort_by_measured_precedence
from pyrulearn.learners.associative import CBA, CMAR
from pyrulearn.learners.ids import IDS
from pyrulearn.models import DecisionList, FlatRuleSet, PooledRuleSet, RuleView, convert

from _negation_helpers import neg_spec, neg_X


def _data(n=220, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(n, 5)).astype(bool)
    y = np.where(raw[:, 0] & ~raw[:, 1], "x", np.where(raw[:, 2] & raw[:, 3], "y", "z"))
    return NListRepresentation.from_boolean(HorizontalBitmapRepresentation(neg_spec(list("abcde")), neg_X(raw), y))


def _pool(data):
    return CARMiner(min_support=0.02, min_confidence=0.5, max_len=3).fit(data)


def _sig(rules):
    return [(tuple(l.feature for l in r.conditions), r.target) for r in rules]


def test_mined_pool_is_a_lazy_flat_rule_set_that_builds_nothing_until_looked_at():
    data = _data()
    pool = _pool(data)
    assert isinstance(pool, FlatRuleSet) and isinstance(pool, PooledRuleSet)
    assert isinstance(pool.rules, RuleView)
    assert len(pool.rules) > 50
    assert len(pool._cache) == 0          # nothing built yet, however many rules there are


def test_indexing_builds_once_and_keeps_one_persistent_object():
    pool = _pool(_data())
    a, b = pool.rules[3], pool.rules[3]
    assert a is b and len(pool._cache) == 1
    assert pool.rules[-1] is pool.rules[len(pool.rules) - 1]
    assert isinstance(pool.rules[2:9], RuleView) and len(pool.rules[2:9]) == 7
    a.weight = 0.5                                   # mutations persist because the object does
    assert pool.rules[3].weight == 0.5
    with pytest.raises(IndexError):
        pool.rules[len(pool.rules)]


def test_iter_transient_builds_equal_rules_without_caching():
    pool = _pool(_data())
    first = list(pool.rules.iter_transient())
    assert len(pool._cache) == 0
    assert _sig(first) == _sig(pool.rules)           # (this second pass caches, the first did not)
    assert first[0] is not pool.rules[0] and _sig([first[0]]) == _sig([pool.rules[0]])


def test_vectorized_precedence_order_equals_sorting_the_built_rules():
    data = _data()
    pool = _pool(data)
    fast = sort_by_measured_precedence(pool.rules)
    assert isinstance(fast, RuleView) and len(pool._cache) == 0     # sorted without building anything
    slow = sort_by_measured_precedence(list(pool.rules))            # the generic path, on real rules
    assert _sig(fast) == _sig(slow)


def test_for_target_targets_and_head_per_target():
    pool = _pool(_data())
    view = pool.rules
    assert set(view.targets()) == {"x", "y", "z"}
    assert all(r.target == "x" for r in view.for_target("x"))
    assert len(view.for_target("nope")) == 0

    ordered = view.precedence_sorted()
    capped = ordered.head_per_target(4)
    expected, seen = [], {}
    for r in ordered:                                                # the loop-based cap (IDS's list path)
        seen[r.target] = seen.get(r.target, 0) + 1
        if seen[r.target] <= 4:
            expected.append(r)
    assert _sig(capped) == _sig(expected)


def test_filter_returns_a_pool_and_add_is_unsupported():
    pool = _pool(_data())
    only_x = pool.filter("x")
    assert isinstance(only_x, PooledRuleSet) and all(r.target == "x" for r in only_x.rules)
    assert len(only_x.rules) == len(pool.rules.for_target("x"))
    with pytest.raises(TypeError, match="columnar"):
        pool.add(pool.rules[0])


def test_a_pooled_set_predicts_like_the_equivalent_eager_flat_rule_set_and_converts():
    data = _data()
    pool = _pool(data)
    eager = FlatRuleSet(list(pool.rules), default_prediction=pool.default_prediction)
    assert list(pool.predict(data)) == list(eager.predict(data))
    assert isinstance(convert(pool, DecisionList), DecisionList)       # via the FlatRuleSet converter (MRO)


def test_distillers_give_identical_models_from_a_pooled_or_an_eager_pool():
    data = _data()
    pool = _pool(data)
    eager = FlatRuleSet(list(pool.rules), default_prediction=pool.default_prediction)

    cba_p, cba_e = CBA(rules=pool).fit(data), CBA(rules=eager).fit(data)
    assert _sig(cba_p.rules) == _sig(cba_e.rules) and cba_p.default_prediction == cba_e.default_prediction

    cmar_p, cmar_e = CMAR(rules=pool).fit(data), CMAR(rules=eager).fit(data)
    assert sorted(_sig(cmar_p.rules)) == sorted(_sig(cmar_e.rules))

    kw = dict(rule_cutoff=8, optimizer=GreedySelection(), tune_lambdas=False, lambda_weights=(0.2, 0.2, 0.2, 0.2, 1, 1, 2))
    ids_p, ids_e = IDS(rules=pool, **kw).fit(data), IDS(rules=eager, **kw).fit(data)
    assert _sig(ids_p.rules) == _sig(ids_e.rules)


def test_cba_over_a_lazy_pool_builds_only_what_it_scans():
    data = _data()
    pool = _pool(data)
    CBA(rules=pool).fit(data)
    # coverage_select streams (iter_transient), so the pool's own cache stays empty even though
    # the pool holds hundreds of rules
    assert len(pool._cache) == 0
