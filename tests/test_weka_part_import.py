from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyrulearn import HorizontalBitmapRepresentation, DataSpec
from pyrulearn.models import DecisionList, FlatRuleSet
from pyrulearn.data.io import binarize
from pyrulearn.interfaces.weka import PARTImporter

_ARFF_PATH = Path(__file__).resolve().parent / "fixtures" / "jrip_test.arff"

# Captured verbatim from a real `weka.jar` 3.8.7 run:
#   java -cp weka.jar weka.classifiers.rules.PART -t jrip_test.arff -x 2
# against the same 400-row synthetic dataset test_weka_jrip_import.py uses.
REAL_PART_STDOUT = """
=== Classifier model (full training set) ===

PART decision list
------------------

income > 19992 AND
color = green: neg (125.0/8.0)

income > 20292 AND
color = blue: neg (117.0/4.0)

age > 39: pos (100.0/3.0)

income > 20373: neg (45.0/3.0)

: pos (13.0)

Number of Rules  : \t5


Time taken to build model: 0.09 seconds

=== Error on training data ===

Correctly Classified Instances         382               95.5    %
Incorrectly Classified Instances        18                4.5    %

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,880    0,011    0,973      0,880    0,924      0,895    0,944     0,914     pos
                 0,989    0,120    0,948      0,989    0,968      0,895    0,944     0,954     neg
Weighted Avg.    0,955    0,086    0,956      0,955    0,954      0,895    0,944     0,941

=== Confusion Matrix ===

   a   b   <-- classified as
 110  15 |   a = pos
   3 272 |   b = neg
"""

REAL_PART_RULES_ONLY = """
income > 19992 AND
color = green: neg (125.0/8.0)

income > 20292 AND
color = blue: neg (117.0/4.0)

age > 39: pos (100.0/3.0)

income > 20373: neg (45.0/3.0)

: pos (13.0)
"""


def _real_arff_dataframe():
    from scipy.io import arff as scipy_arff

    data, _meta = scipy_arff.loadarff(str(_ARFF_PATH))
    df = pd.DataFrame(data)
    for c in df.select_dtypes([object]).columns:
        df[c] = df[c].str.decode("utf-8")
    return df


def test_parses_real_part_output_as_rulelist():
    importer = PARTImporter()
    rules = importer.parse(REAL_PART_RULES_ONLY)

    assert isinstance(rules, DecisionList)
    assert not isinstance(rules, FlatRuleSet)
    assert len(rules.rules) == 4
    assert rules.default_rule is not None
    assert len(rules.default_rule.conditions) == 0
    assert rules.default_rule.target == "pos"
    assert [r.target for r in rules.rules] == ["neg", "neg", "pos", "neg"]
    assert set(importer.dataspec.feature_names) == {
        "age<=39.0", "income<=20373.0", "income<=20292.0", "income<=19992.0",
        "color=blue", "color=green",
        "age>39.0", "income>20373.0", "income>20292.0", "income>19992.0",
        "color!=blue", "color!=green",
    }


def test_parsing_ignores_everything_but_rule_blocks():
    # the full raw Weka console dump (evaluation stats, confusion matrix,
    # header/footer, "Number of Rules" line) parses identically to the
    # isolated rule blocks -- confirms the blank-line-block filter is
    # safe against a whole console dump, same claim JRip's "=>" filter makes
    a = PARTImporter().parse(REAL_PART_STDOUT)
    b = PARTImporter().parse(REAL_PART_RULES_ONLY)
    assert len(a.rules) == len(b.rules) == 4
    assert [r.target for r in a.rules] == [r.target for r in b.rules]
    assert a.default_rule.target == b.default_rule.target == "pos"


def test_multi_condition_and_continuation_parses_correctly():
    rules = PARTImporter().parse(REAL_PART_RULES_ONLY)
    ds = PARTImporter().infer_dataspec(REAL_PART_RULES_ONLY)

    r0 = rules.rules[0]  # "income > 19992 AND\ncolor = green: neg (125.0/8.0)"
    feats = set(r0.pos)
    assert ds.feature_index("income>19992.0") in feats  # income > 19992 == NOT(income<=19992)
    assert ds.feature_index("color=green") in feats
    assert len(r0.conditions) == 2


def test_provenance_tagged_on_rules_and_default():
    rules = PARTImporter().parse(REAL_PART_RULES_ONLY)
    for r in list(rules.rules) + [rules.default_rule]:
        assert r.provenance.source == "weka.classifiers.rules.PART"
        assert r.provenance.learner == "PARTImporter"


def test_end_to_end_predict_matches_weka_exactly():
    importer = PARTImporter()
    rules = importer.parse(REAL_PART_RULES_ONLY)
    ds = importer.dataspec

    df = _real_arff_dataframe()
    # all four numeric conditions are ">" here, i.e. negated <=-family
    # literals -- binarize derives them straight from the raw columns,
    # no precomputed helper column needed (same as JRip's <= family)
    X = binarize(ds, df)
    rep = HorizontalBitmapRepresentation(ds, X, df["label"].to_numpy())
    preds = np.asarray(rules.predict(rep))

    expected = np.where(
        (df["income"] > 19992) & (df["color"] == "green"), "neg",
        np.where(
            (df["income"] > 20292) & (df["color"] == "blue"), "neg",
            np.where(df["age"] > 39, "pos", np.where(df["income"] > 20373, "neg", "pos")),
        ),
    )
    assert np.array_equal(preds, expected)
    accuracy = (preds == df["label"].to_numpy()).mean()
    assert accuracy == pytest.approx(0.955, abs=1e-3)


def test_le_operator_form_also_parses_correctly():
    # the real captured sample only exercised ">" for numeric conditions;
    # a synthetic (but correctly PART-formatted) block confirms "<=" --
    # C4.5's other split direction -- works too
    text = """
    age <= 30 AND
    color = red: young (10.0/1.0)

    score <= 5.5: mid (8.0)

    : old (20.0/2.0)
    """
    importer = PARTImporter()
    rules = importer.parse(text)
    ds = importer.dataspec

    assert set(ds.feature_names) == {
        "age<=30.0", "age>30.0", "color=red", "color!=red", "score<=5.5", "score>5.5",
    }
    assert [r.target for r in rules.rules] == ["young", "mid"]
    assert rules.default_rule.target == "old"

    feats0 = set(rules.rules[0].pos)
    assert ds.feature_index("age<=30.0") in feats0  # plain positive literal, not negated
    assert ds.feature_index("color=red") in feats0


def test_reuses_dataspec_across_multiple_parse_calls():
    ds = PARTImporter().infer_dataspec(REAL_PART_RULES_ONLY)
    importer = PARTImporter(dataspec=ds)
    rules = importer.parse(REAL_PART_RULES_ONLY)
    assert importer.dataspec is ds
    assert rules.rules[0].dataspec is ds


def test_raises_on_multiple_default_rules():
    text = """
    age <= 30: pos (5.0/0.0)

    : neg (3.0)

    : other (1.0)
    """
    with pytest.raises(ValueError, match="more than one"):
        PARTImporter().parse(text)


def test_raises_on_missing_and_continuation():
    # a non-final condition line that doesn't end in " AND" is malformed
    text = """
    age <= 30
    color = red: pos (5.0/0.0)
    """
    with pytest.raises(ValueError, match="AND"):
        PARTImporter().parse(text)


def test_raises_when_no_rule_blocks_found():
    with pytest.raises(ValueError, match="no PART rule blocks"):
        PARTImporter().parse("nothing resembling a PART rule here\n")
