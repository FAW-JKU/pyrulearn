from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyrulearn import HorizontalBitmapRepresentation, DataSpec
from pyrulearn.models import DecisionList, FlatRuleSet
from pyrulearn.data.io import binarize
from pyrulearn.interfaces.weka import JRipImporter

_ARFF_PATH = Path(__file__).resolve().parent / "fixtures" / "jrip_test.arff"

# Captured verbatim from a real `weka.jar` 3.8.7 run:
#   java -cp weka.jar weka.classifiers.rules.JRip -t jrip_test.arff -x 2
# against a 400-row synthetic dataset with label = ((age>=40 & color=="red")
# | income<20000) XOR ~4% noise -- not hand-crafted to match this parser.
REAL_JRIP_STDOUT = """
=== Classifier model (full training set) ===

JRIP rules:
===========

(color = red) and (age >= 41) => label=pos (81.0/2.0)
(income <= 19992) => label=pos (32.0/1.0)
 => label=neg (287.0/15.0)

Number of Rules : 3


Time taken to build model: 0.1 seconds

=== Error on training data ===

Correctly Classified Instances         382               95.5    %
Incorrectly Classified Instances        18                4.5    %
Kappa statistic                          0.8925

=== Confusion Matrix ===

   a   b   <-- classified as
 110  15 |   a = pos
   3 272 |   b = neg
"""

REAL_JRIP_RULES_ONLY = """
(color = red) and (age >= 41) => label=pos (81.0/2.0)
(income <= 19992) => label=pos (32.0/1.0)
 => label=neg (287.0/15.0)
"""


def _real_arff_dataframe():
    # the exact file fed to the real Weka run, not a re-derived
    # reconstruction -- regenerating it from scratch via the same RNG
    # seed is fragile (any unrelated extra draw earlier in that script,
    # even of a column never written to the file, shifts every later
    # draw), so read the file itself instead
    from scipy.io import arff as scipy_arff

    data, _meta = scipy_arff.loadarff(str(_ARFF_PATH))
    df = pd.DataFrame(data)
    for c in df.select_dtypes([object]).columns:
        df[c] = df[c].str.decode("utf-8")
    return df


def test_parses_real_jrip_output_as_rulelist():
    # JRip's own semantics are ordered/first-match-wins (classic RIPPER
    # sequential covering, genuinely multi-class), unlike wittgenstein's
    # binary-only IREP/RIPPER importers which produce a RuleSet -- see
    # the module docstring
    importer = JRipImporter()
    rules = importer.parse(REAL_JRIP_RULES_ONLY)

    assert isinstance(rules, DecisionList)
    assert not isinstance(rules, FlatRuleSet)
    assert len(rules.rules) == 2
    assert rules.default_rule is not None
    assert len(rules.default_rule.conditions) == 0
    assert rules.default_rule.target == "neg"
    assert [r.target for r in rules.rules] == ["pos", "pos"]
    assert set(importer.dataspec.feature_names) == {
        "age>=41.0", "age<41.0", "color=red", "color!=red",
        "income<=19992.0", "income>19992.0",
    }


def test_parsing_ignores_everything_but_rule_lines():
    # the full raw Weka console dump (evaluation stats, confusion matrix,
    # header/footer) should parse identically to the isolated rule block --
    # the module docstring's claim that only lines containing "=>" matter
    a = JRipImporter().parse(REAL_JRIP_STDOUT)
    b = JRipImporter().parse(REAL_JRIP_RULES_ONLY)
    assert len(a.rules) == len(b.rules) == 2
    assert [r.target for r in a.rules] == [r.target for r in b.rules]
    assert a.default_rule.target == b.default_rule.target == "neg"


def test_provenance_tagged_on_rules_and_default():
    rules = JRipImporter().parse(REAL_JRIP_RULES_ONLY)
    for r in list(rules.rules) + [rules.default_rule]:
        assert r.provenance.source == "weka.classifiers.rules.JRip"
        assert r.provenance.learner == "JRipImporter"


def test_end_to_end_predict_matches_weka_exactly():
    # the strongest check: predictions on the real underlying data must
    # match both JRip's own rule logic exactly (0 mismatches) and Weka's
    # own reported training accuracy (95.5%, from the real run's stdout)
    importer = JRipImporter()
    rules = importer.parse(REAL_JRIP_RULES_ONLY)
    ds = importer.dataspec

    df = _real_arff_dataframe()
    # no precomputed helper column needed: "income<=19992" is a proper
    # add_numeric(le_thresholds=...) feature, binarized straight from the
    # raw "income" column like any other numeric threshold
    X = binarize(ds, df)
    rep = HorizontalBitmapRepresentation(ds, X, df["label"].to_numpy())
    preds = np.asarray(rules.predict(rep))

    expected = np.where(
        (df["color"] == "red") & (df["age"] >= 41), "pos",
        np.where(df["income"] <= 19992, "pos", "neg"),
    )
    assert np.array_equal(preds, expected)
    accuracy = (preds == df["label"].to_numpy()).mean()
    assert accuracy == pytest.approx(0.955, abs=1e-3)


def test_le_feature_binarizes_directly_from_raw_column():
    # <=/> conditions are a genuine add_numeric(le_thresholds=...) family
    # now, not an atomic Boolean feature -- binarize derives them from the
    # raw numeric column on its own, no precomputed helper column needed
    importer = JRipImporter()
    importer.parse(REAL_JRIP_RULES_ONLY)
    ds = importer.dataspec
    df = _real_arff_dataframe()
    X = binarize(ds, df)  # would raise if this were still the atomic-feature workaround
    le_idx = ds.feature_index("income<=19992.0")
    assert np.array_equal(X[:, le_idx], (df["income"] <= 19992).to_numpy())


def test_numeric_operators_all_four_forms_and_multiclass():
    # real captured output only exercised >=/<=/= ; this synthetic (but
    # correctly-formatted, stats included) text exercises the other two
    # forms and a genuinely 3-class rule set
    text = """
    (age < 30) and (score > 5.5) => label=young (10.0/0.0)
    (age >= 30) and (score <= 5.5) => label=old (10.0/1.0)
     => label=mid (5.0/0.0)
    """
    importer = JRipImporter()
    rules = importer.parse(text)
    ds = importer.dataspec

    # since the explicit-negation redesign, "score>5.5" and "age<30" are
    # their own features -- each the exact paired negation of the <=/>=
    # family feature it complements (linked by a MutuallyExclusive /
    # NumericGroup constraint), rather than a negative literal.
    assert set(ds.feature_names) == {
        "age>=30.0", "age<30.0", "score<=5.5", "score>5.5",
    }
    assert [r.target for r in rules.rules] == ["young", "old"]
    assert rules.default_rule.target == "mid"

    # "age < 30" must be the negation feature paired with age>=30.0;
    # "score > 5.5" the one paired with score<=5.5
    assert ds.negation_of(ds.feature_index("age>=30.0")) == ds.feature_index("age<30.0")
    assert ds.negation_of(ds.feature_index("score<=5.5")) == ds.feature_index("score>5.5")
    feats0 = set(rules.rules[0].pos)
    assert ds.feature_index("age<30.0") in feats0
    assert ds.feature_index("score>5.5") in feats0

    df = pd.DataFrame({"age": [20, 40, 20, 40], "score": [8.0, 2.0, 2.0, 8.0]})
    X = binarize(ds, df)  # both families derive directly from raw age/score columns
    rep = HorizontalBitmapRepresentation(ds, X, np.array(["?"] * 4))
    preds = list(rules.predict(rep))
    assert preds == ["young", "old", "mid", "mid"]


def test_class_values_with_spaces_parse():
    # Weka prints nominal class values verbatim and unquoted, so a value
    # like glass's "vehic wind float" lands mid-line between "=>" and the
    # trailing "(covered/errors)" -- the consequent capture must not stop
    # at the first space
    text = """
    (f43 <= 0) and (f42 >= 1) => class=vehic wind float (5.0/1.0)
    (f7 >= 1) => class=build wind non-float (40.0/3.0)
     => class=headlamps (12.0/0.0)
    """
    rules = JRipImporter().parse(text)
    assert [r.target for r in rules.rules] == [
        "vehic wind float", "build wind non-float",
    ]
    assert rules.default_rule.target == "headlamps"


def test_reuses_dataspec_across_multiple_parse_calls():
    # workflow-1-style reuse: pass an already-built dataspec (e.g. from an
    # earlier infer_dataspec call) so a second parse binds to the same
    # object rather than discovering its own
    ds = JRipImporter().infer_dataspec(REAL_JRIP_RULES_ONLY)
    importer = JRipImporter(dataspec=ds)
    rules = importer.parse(REAL_JRIP_RULES_ONLY)
    assert importer.dataspec is ds
    assert rules.rules[0].dataspec is ds


def test_raises_on_multiple_default_rules():
    text = """
    (a = x) => label=pos (5.0/0.0)
     => label=neg (3.0/0.0)
     => label=other (1.0/0.0)
    """
    with pytest.raises(ValueError, match="more than one"):
        JRipImporter().parse(text)


def test_raises_on_unparseable_condition():
    text = "(a ~ x) => label=pos (5.0/0.0)"
    with pytest.raises(ValueError, match="couldn't parse condition"):
        JRipImporter().parse(text)


JRIP_INEQUALITY_RULES = """(color = blue) => label=neg (10.0/0.0)
(color != green) => label=pos (5.0/0.0)
 => label=neg (3.0/0.0)"""


def test_inequality_on_a_value_list_seen_only_in_the_rules_still_covers_unseen_values():
    # the rules mention only blue/green, but the data also has red: the
    # inferred attribute must stay NOMINAL, so `color != green` covers red
    importer = JRipImporter()
    model = importer.parse(JRIP_INEQUALITY_RULES)
    ds = importer.dataspec
    rep = HorizontalBitmapRepresentation(ds, binarize(ds, pd.DataFrame({"color": ["red", "green"]})),
                                    np.array(["pos", "neg"]))
    assert list(model.predict(rep)) == ["pos", "neg"]
