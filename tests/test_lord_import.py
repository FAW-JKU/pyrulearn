import os

import numpy as np
import pytest

from pyrulearn import HorizontalBitmapRepresentation, Rule
from pyrulearn.data import DataSpecBuilder
from pyrulearn.heuristics import RuleStats
from pyrulearn.interfaces.lord import JavaLORD, LORDImporter, _parse_rules, run_lord
from pyrulearn.models import ConceptSet, FlatRuleSet

from _negation_helpers import neg_spec, neg_X

# The "Rule set:" block, verbatim in shape from a real `run.LordRun` on a
# 0/1 CSV (LORD's committed bin/ classes + weka jar) -- p/n are floats,
# heuristic_value is full-precision, the block is fenced by 84-dash rules
# and the whole file is prefixed by thread/timing chatter.
EG_OUTPUT = """\
Execute algorithm Lord on dataset:
Metric type: MESTIMATE, Argument: 0.100000
preprocess time: 5 ms
\tSearchRuleThread 0 founds 6 rules, finished in 7 ms
\tFilterRuleThread 0 finished in 0 ms
Learning time: 12 ms
------------------------------------------------------------------------------------
Rule set:
IF (f0=1) & (f3=0) THEN (Class=pos)\t(p=40.0, n=0.0, heuristic_value=0.98)
IF (f2=1) THEN (Class=pos)\t(p=55.0, n=2.0, heuristic_value=0.87)
IF (f0=0) THEN (Class=neg)\t(p=90.0, n=5.0, heuristic_value=0.79)

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Rule count: 3
Accuracy: 0.910000
Finished.
"""

NOMINAL_EG_OUTPUT = """\
Rule set:
IF (color=red) & (size=big) THEN (Class=yes)\t(p=12, n=1, heuristic_value=0.71)
IF (color=blue) THEN (Class=no)\t(p=20, n=0, heuristic_value=0.95)
"""


def _bool_spec(k):
    b = DataSpecBuilder(negation=True)
    for i in range(k):
        b.add_boolean(f"f{i}")
    return b.build()


def test_parse_rules_reads_the_rule_set_block():
    parsed = _parse_rules(EG_OUTPUT)
    assert len(parsed) == 3
    conds, target = parsed[0]
    assert conds == [("f0", "1"), ("f3", "0")]
    assert target == "pos"
    assert parsed[2][1] == "neg"


def test_parse_ignores_non_rule_lines():
    noisy = "garbage\n" + EG_OUTPUT + "\nNumber of rules: 3\nsome trailing text\n"
    assert len(_parse_rules(noisy)) == 3


def test_import_binds_to_given_spec_and_maps_zero_to_the_negation_feature():
    ds = _bool_spec(4)
    rules = LORDImporter(dataspec=ds).parse(EG_OUTPUT)
    assert isinstance(rules, FlatRuleSet)
    assert [r.target for r in rules.rules] == ["pos", "pos", "neg"]

    r0 = rules.rules[0]  # (f0=1) & (f3=0)
    assert set(r0.pos) == {ds.feature_index("f0"), ds.feature_index("not f3")}
    r2 = rules.rules[2]  # (f0=0)
    assert set(r2.pos) == {ds.feature_index("not f0")}


def test_provenance_tagged():
    rules = LORDImporter(dataspec=_bool_spec(4)).parse(EG_OUTPUT)
    for r in rules.rules:
        assert r.provenance.source == "LORD (Huynh, Fürnkranz & Beck, 2023)"
        assert r.provenance.learner == "LORDImporter"


def test_infer_dataspec_treats_binary_attrs_as_boolean():
    ds = LORDImporter().infer_dataspec(EG_OUTPUT)
    # f0/f2/f3 seen with values in {0,1} -> boolean, each paired with a negation
    assert set(ds.feature_names) == {"f0", "not f0", "f2", "not f2", "f3", "not f3"}


def test_infer_dataspec_treats_multivalued_attrs_as_nominal():
    rules = LORDImporter().parse(NOMINAL_EG_OUTPUT)
    ds = rules.rules[0].dataspec
    assert {"color=red", "color=blue", "size=big"} <= set(ds.feature_names)
    assert set(rules.rules[0].pos) == {ds.feature_index("color=red"), ds.feature_index("size=big")}


def test_disjunctive_selector_raises_clearly():
    with pytest.raises(ValueError, match="disjunctive"):
        _parse_rules("IF (f0=1||f0=2) THEN (Class=pos)\t(p=1, n=0, heuristic_value=0.5)")


def test_end_to_end_predict_is_best_rule_wins():
    # f0=1 & ~f3 -> pos (w .98); f2 -> pos (w .87); ~f0 -> neg (w .79)
    ds = _bool_spec(4)
    raw = np.array([
        [1, 0, 0, 0],  # r0 fires (pos, .98)
        [1, 0, 1, 0],  # r0 (.98) and r1 (.87) both fire -> r0 wins -> pos
        [0, 0, 1, 0],  # r1 (.87, pos) and r2 (.79, neg) fire -> r1 wins -> pos
        [0, 0, 0, 0],  # only r2 -> neg
        [1, 0, 0, 1],  # f3 set -> r0 blocked; nothing fires -> default
    ], dtype=bool)
    X = np.empty((5, 8), dtype=bool)
    X[:, 0::2] = raw
    X[:, 1::2] = ~raw
    y = np.array(["pos", "pos", "pos", "neg", "pos"])
    rep = HorizontalBitmapRepresentation(ds, X, y)

    rules = LORDImporter(dataspec=ds).parse(EG_OUTPUT, data=rep)  # data= -> real measured stats
    rules.default_prediction = "pos"  # majority, set by caller

    preds = np.asarray(rules.predict(rep))  # RuleSet.predict defaults to HeuristicMaxCombiner (measured stats)
    assert list(preds) == ["pos", "pos", "pos", "neg", "pos"]


def test_placeholder_features_bind_by_position_not_name():
    # a fit() round-trip writes ds's columns as f0..fN; parse must map
    # f{i} to column i positionally, even when ds's real feature names
    # are nothing like "f0"
    ds = neg_spec(["aa", "bb", "cc", "dd"])  # aa,not aa,bb,not bb,... -> 0..7
    rules = LORDImporter(dataspec=ds, placeholder_features=True).parse(EG_OUTPUT)
    # (f0=1) & (f3=0): f0=1 -> +col 0 ; f3=0 -> -col 3 -> paired negation (col 2)
    assert sorted(l.feature for l in rules.rules[0].conditions) == [0, 2]
    assert [r.target for r in rules.rules] == ["pos", "pos", "neg"]


def test_placeholder_features_requires_a_dataspec():
    with pytest.raises(ValueError, match="requires an explicit dataspec"):
        LORDImporter(placeholder_features=True).parse(EG_OUTPUT)


def test_javalord_is_a_fit_learner():
    caps = JavaLORD().produces()
    assert FlatRuleSet in caps and ConceptSet in caps
    with pytest.raises(ValueError, match="variant"):
        JavaLORD(variant="nope")


@pytest.mark.skipif(
    not (os.environ.get("LORD_CLASSPATH") or os.environ.get("LORD_JAR")),
    reason="LORD_CLASSPATH not set",
)
def test_javalord_fits_end_to_end():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 2, size=(200, 4)).astype(bool)
    y = np.where(raw[:, 0] & ~raw[:, 1], "pos", "neg")
    ds = neg_spec([f"x{i}" for i in range(4)])
    rep = HorizontalBitmapRepresentation(ds, neg_X(raw), y)

    model = JavaLORD(timeout=120).fit(rep)
    assert type(model) is FlatRuleSet and model.default_prediction is not None
    acc = float(np.mean(np.asarray(model.predict(rep)) == y))
    assert acc > 0.8, f"JavaLORD train accuracy only {acc:.3f}"
    assert type(JavaLORD(timeout=120).fit(rep, model=ConceptSet)) is ConceptSet


@pytest.mark.skipif(
    not (os.environ.get("LORD_CLASSPATH") or os.environ.get("LORD_JAR")),
    reason="LORD_CLASSPATH not set",
)
def test_run_lord_end_to_end():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 2, size=(200, 4)).astype(int)
    y = np.where((raw[:, 0] == 1) & (raw[:, 1] == 0), "pos", "neg")
    header = [f"f{i}" for i in range(4)] + ["Class"]
    rows = [list(raw[i]) + [y[i]] for i in range(200)]

    text = run_lord(rows, header, metric="mestimate", metric_arg=0.1, timeout=120)
    rep = HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw.astype(bool)), y)
    rules = LORDImporter(dataspec=neg_spec([f"f{i}" for i in range(4)])).parse(text, data=rep)
    assert len(rules.rules) > 0
    rules.default_prediction = "neg"
    acc = float(np.mean(np.asarray(rules.predict(rep)) == y))
    assert acc > 0.8
