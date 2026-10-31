from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyrulearn import HorizontalBitmapRepresentation, DataSpec
from pyrulearn.models import DecisionList, DisjointRuleSet
from pyrulearn.data.io import binarize
from pyrulearn.interfaces.weka import J48Importer

_ARFF_PATH = Path(__file__).resolve().parent / "fixtures" / "jrip_test.arff"
_DEEP_ARFF_PATH = Path(__file__).resolve().parent / "fixtures" / "j48_deep_test.arff"

# Captured verbatim from a real `weka.jar` 3.8.7 run:
#   java -cp weka.jar weka.classifiers.trees.J48 -t jrip_test.arff -x 2
# against the same 400-row synthetic dataset test_weka_jrip_import.py uses.
REAL_J48_STDOUT = """
=== Classifier model (full training set) ===

J48 pruned tree
------------------

income <= 19992: pos (40.0/1.0)
income > 19992
|   color = red
|   |   age <= 40: neg (45.0/3.0)
|   |   age > 40: pos (73.0/2.0)
|   color = green: neg (125.0/8.0)
|   color = blue: neg (117.0/4.0)

Number of Leaves  : \t5

Size of the tree : \t8


Time taken to build model: 0.08 seconds

=== Error on training data ===

Correctly Classified Instances         382               95.5    %
Incorrectly Classified Instances        18                4.5    %

=== Confusion Matrix ===

   a   b   <-- classified as
 110  15 |   a = pos
   3 272 |   b = neg
"""

REAL_J48_TREE_ONLY = """J48 pruned tree
------------------

income <= 19992: pos (40.0/1.0)
income > 19992
|   color = red
|   |   age <= 40: neg (45.0/3.0)
|   |   age > 40: pos (73.0/2.0)
|   color = green: neg (125.0/8.0)
|   color = blue: neg (117.0/4.0)
"""

# a deliberately deep tree (33 leaves, depth 5, mixed numeric/nominal,
# multiple multi-way nominal splits) to stress-test the depth-stack
# backtracking beyond the shallow example above -- also captured verbatim
DEEP_J48_TREE = """J48 pruned tree
------------------

d = p
|   e = u
|   |   a <= 2: neg (9.0/1.0)
|   |   a > 2: pos (120.0/33.0)
|   e = v: pos (140.0/18.0)
|   e = w: pos (125.0/40.0)
d = q
|   e = u: neg (126.0/34.0)
|   e = v
|   |   a <= 2: neg (12.0/1.0)
|   |   a > 2
|   |   |   b <= 39: pos (106.0/31.0)
|   |   |   b > 39
|   |   |   |   b <= 48
|   |   |   |   |   c <= 7: pos (2.0)
|   |   |   |   |   c > 7: neg (21.0/5.0)
|   |   |   |   b > 48: pos (2.0)
|   e = w: neg (136.0/33.0)
d = r
|   e = u
|   |   b <= 47: neg (129.0/26.0)
|   |   b > 47: pos (6.0/1.0)
|   e = v: pos (137.0/67.0)
|   e = w: neg (152.0/35.0)
d = s
|   e = u
|   |   a <= 45: neg (119.0/36.0)
|   |   a > 45: pos (10.0/2.0)
|   e = v
|   |   a <= 4: neg (18.0/3.0)
|   |   a > 4: pos (121.0/46.0)
|   e = w
|   |   b <= 24
|   |   |   b <= 13: neg (24.0/5.0)
|   |   |   b > 13: pos (26.0/9.0)
|   |   b > 24: neg (74.0/14.0)
d = t
|   e = u: neg (121.0/10.0)
|   e = v
|   |   a <= 3: neg (12.0)
|   |   a > 3
|   |   |   a <= 6
|   |   |   |   a <= 5
|   |   |   |   |   a <= 4: pos (3.0)
|   |   |   |   |   a > 4: neg (3.0)
|   |   |   |   a > 5: pos (5.0)
|   |   |   a > 6: neg (113.0/29.0)
|   e = w
|   |   c <= 12: neg (37.0)
|   |   c > 12
|   |   |   a <= 32: neg (59.0/3.0)
|   |   |   a > 32
|   |   |   |   c <= 15: pos (5.0/1.0)
|   |   |   |   c > 15
|   |   |   |   |   a <= 34: pos (3.0/1.0)
|   |   |   |   |   a > 34: neg (24.0/2.0)
"""


def _real_arff_dataframe():
    from scipy.io import arff as scipy_arff

    data, _meta = scipy_arff.loadarff(str(_ARFF_PATH))
    df = pd.DataFrame(data)
    for c in df.select_dtypes([object]).columns:
        df[c] = df[c].str.decode("utf-8")
    return df


def test_parses_real_j48_output_as_disjoint_ruleset():
    importer = J48Importer()
    rules = importer.parse(REAL_J48_TREE_ONLY)

    assert isinstance(rules, DisjointRuleSet)
    assert not isinstance(rules, DecisionList)
    assert len(rules.rules) == 5
    assert set(importer.dataspec.feature_names) == {
        "age<=40.0", "age>40.0", "income<=19992.0", "income>19992.0",
        "color=blue", "color=green", "color=red",
        "color!=blue", "color!=green", "color!=red",
    }


def test_confusion_matrix_trap_line_is_not_misparsed():
    # ' 110  15 |   a = pos' in the confusion matrix reads as both a
    # "|   "-indented node AND a valid "attr = value" condition -- a real
    # false-positive risk generic line-scanning would hit (unlike JRip's
    # "=>" filter or PART's block terminator); confirms header-anchoring
    # avoids it, on the *whole* raw console dump
    rules_isolated = J48Importer().parse(REAL_J48_TREE_ONLY)
    rules_full_dump = J48Importer().parse(REAL_J48_STDOUT)
    assert len(rules_full_dump.rules) == len(rules_isolated.rules) == 5
    assert sorted(r.to_string("logic") for r in rules_full_dump.rules) == \
        sorted(r.to_string("logic") for r in rules_isolated.rules)


def test_multiway_nominal_split_parses_correctly():
    # C4.5/J48, unlike CART, splits nominal attributes into more than two
    # branches -- "color" fans out into red/green/blue under one parent
    rules = J48Importer().parse(REAL_J48_TREE_ONLY)
    logic_strs = sorted(r.to_string("logic") for r in rules.rules)
    assert any("color = green" in s for s in logic_strs)
    assert any("color = blue" in s for s in logic_strs)
    assert any("color = red" in s for s in logic_strs)


def test_provenance_tagged_on_rules():
    rules = J48Importer().parse(REAL_J48_TREE_ONLY)
    for r in rules.rules:
        assert r.provenance.source == "weka.classifiers.trees.J48"
        assert r.provenance.learner == "J48Importer"


def test_end_to_end_predict_matches_weka_exactly_and_is_disjoint_exhaustive():
    importer = J48Importer()
    rules = importer.parse(REAL_J48_TREE_ONLY)
    ds = importer.dataspec

    df = _real_arff_dataframe()
    X = binarize(ds, df)
    rep = HorizontalBitmapRepresentation(ds, X, df["label"].to_numpy())
    preds = np.asarray(rules.predict(rep))

    expected = np.where(
        df["income"] <= 19992, "pos",
        np.where(
            df["color"] == "red", np.where(df["age"] <= 40, "neg", "pos"),
            "neg",  # both color=green and color=blue branches predict neg
        ),
    )
    assert np.array_equal(preds, expected)
    accuracy = (preds == df["label"].to_numpy()).mean()
    assert accuracy == pytest.approx(0.955, abs=1e-3)

    # a genuine decision tree's leaves are disjoint and exhaustive by
    # construction -- confirmed, not just assumed from the DisjointRuleSet type
    assert rules.is_disjoint(rep)
    assert rules.is_exhaustive(rep)


def test_deep_tree_with_multilevel_backtracking():
    # stress test: 33 leaves, depth up to 5, several multi-way splits and
    # several depth changes of more than one level between siblings --
    # exercises del stack[depth:]'s backtracking well beyond the shallow
    # example above
    importer = J48Importer()
    rules = importer.parse(DEEP_J48_TREE)
    assert len(rules.rules) == 33

    from scipy.io import arff as scipy_arff

    data, _meta = scipy_arff.loadarff(str(_DEEP_ARFF_PATH))
    df = pd.DataFrame(data)
    for c in df.select_dtypes([object]).columns:
        df[c] = df[c].str.decode("utf-8")

    ds = importer.dataspec
    X = binarize(ds, df)
    rep = HorizontalBitmapRepresentation(ds, X, df["label"].to_numpy())
    preds = np.asarray(rules.predict(rep))
    accuracy = (preds == df["label"].to_numpy()).mean()
    # Weka's own real run on this exact tree reported 75.7% training accuracy
    assert accuracy == pytest.approx(0.757, abs=1e-3)
    assert rules.is_disjoint(rep)
    assert rules.is_exhaustive(rep)


def test_reuses_dataspec_across_multiple_parse_calls():
    ds = J48Importer().infer_dataspec(REAL_J48_TREE_ONLY)
    importer = J48Importer(dataspec=ds)
    rules = importer.parse(REAL_J48_TREE_ONLY)
    assert importer.dataspec is ds
    assert rules.rules[0].dataspec is ds


def test_raises_without_header():
    text = "income <= 19992: pos (40.0/1.0)\n"
    with pytest.raises(ValueError, match="J48"):
        J48Importer().parse(text)


def test_raises_when_no_leaves_found():
    text = "J48 pruned tree\n------------------\n\nincome <= 19992\n"
    with pytest.raises(ValueError, match="no leaves"):
        J48Importer().parse(text)


# Captured verbatim from a real weka.jar 3.8.7 run (a fold where no
# attribute was predictive enough to split on, so J48 prunes all the way
# back to the root and predicts the majority class unconditionally) --
# note there's no blank line between the divider and the tree body here,
# unlike every other captured example above: Weka only prints that blank
# separator when the tree actually has a body to separate it from.
REAL_J48_SINGLE_LEAF_STDOUT = """
=== Classifier model (full training set) ===

J48 pruned tree
------------------
: 1 (499.0/119.0)

Number of Leaves  : \t1

Size of the tree : \t1


Time taken to build model: 0.17 seconds
"""


def test_single_leaf_tree_parses_as_one_unconditional_rule():
    # regression test: _extract_j48_tree_lines used to assume a blank
    # line always separates the divider from the tree body (true for
    # every multi-leaf tree, since each split line starts with the
    # recursive dump's own leading newline) -- but a single-leaf tree
    # has no such line, so that assumption skipped straight past the
    # tree's one real line into blank space, and "no leaves found" fired
    # for a perfectly valid model.
    importer = J48Importer()
    rules = importer.parse(REAL_J48_SINGLE_LEAF_STDOUT)
    assert isinstance(rules, DisjointRuleSet)
    assert len(rules.rules) == 1
    rule = rules.rules[0]
    assert rule.rule.conditions == ()
    assert rule.target == "1"
    assert importer.dataspec.feature_names == []


MULTIWORD_LABEL_TREE = """J48 pruned tree
------------------

Ba <= 0.27
|   RI <= 1.5241: build wind float (70.0/20.0)
|   RI > 1.5241: containers (13.0/2.0)
Ba > 0.27: vehic wind non-float (17.0)

Number of Leaves  : 	3

Size of the tree : 	5
"""


def test_multiword_class_labels_are_kept():
    # regression test: the leaf regex took the target as a single \S+
    # token, so every leaf whose label contains a space (e.g. glass's
    # "build wind float") was silently dropped.
    rules = J48Importer().parse(MULTIWORD_LABEL_TREE)
    assert sorted(r.target for r in rules.rules) == [
        "build wind float", "containers", "vehic wind non-float"]
