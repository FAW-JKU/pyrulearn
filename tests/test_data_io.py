import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyrulearn.data.attributes import FeatureSpec, MissingStrategy, evaluate_feature
from pyrulearn.data import (
    HorizontalBitmapRepresentation, DataSpec, DataSpecBuilder, NListRepresentation, SparseDataRepresentation,
    VerticalBitmapRepresentation,
)
from pyrulearn.data.io import (
    binarize, build_dataspec, encode, read_arff, read_csv, validate_dataspec, write_arff, write_csv,
)
from pyrulearn.rule import Rule

_ENCODE_REPS = [HorizontalBitmapRepresentation, NListRepresentation, SparseDataRepresentation,
                VerticalBitmapRepresentation]


CSV_TEXT = """color,age,label
red,20,neg
red,22,neg
red,45,pos
blue,21,neg
blue,50,pos
blue,55,pos
green,19,neg
green,60,pos
"""

ARFF_TEXT = """@relation weather

@attribute outlook {sunny, overcast, rainy}
@attribute temperature numeric
@attribute play {yes, no}

@data
sunny,85,no
sunny,80,no
overcast,83,yes
rainy,70,yes
rainy,68,yes
rainy,65,no
overcast,64,yes
sunny,72,no
sunny,69,yes
rainy,75,yes
sunny,75,yes
overcast,72,yes
overcast,81,yes
rainy,71,no
"""


def test_evaluate_feature():
    plain = FeatureSpec(0, "smoker")
    assert evaluate_feature(plain, True) is True
    assert evaluate_feature(plain, 0) is False
    assert evaluate_feature(plain, 1) is True

    eq = FeatureSpec(1, "color=red", attribute="color", op="==", value="red")
    assert evaluate_feature(eq, "red") is True
    assert evaluate_feature(eq, "blue") is False

    ge = FeatureSpec(2, "age>=30", attribute="age", op=">=", value=30)
    assert evaluate_feature(ge, 35) is True
    assert evaluate_feature(ge, 29) is False

    has = FeatureSpec(3, "tags has urgent", attribute="tags", op="has", value="urgent")
    assert evaluate_feature(has, {"urgent", "bug"}) is True
    assert evaluate_feature(has, {"bug"}) is False

    assert evaluate_feature(plain, None) is None
    assert evaluate_feature(plain, float("nan")) is None

    expr = FeatureSpec(4, "income_gt_age", attributes=("income", "age"), op="expr", expression="income > age*1000")
    try:
        evaluate_feature(expr, 1)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_read_csv_infer():
    rep = read_csv(io.StringIO(CSV_TEXT), target="label")
    assert rep.n_samples == 8
    assert {"color=red", "color=blue", "color=green"} <= set(rep.spec.feature_names)
    age_feats = [n for n in rep.spec.feature_names if n.startswith("age>=")]
    assert len(age_feats) >= 1
    assert list(rep.y) == ["neg", "neg", "pos", "neg", "pos", "pos", "neg", "pos"]

    # the age threshold(s) found should perfectly separate pos/neg on this toy data
    age_idx = rep.spec.feature_index(age_feats[0])
    r = Rule.from_pos_neg(pos=[age_idx], target="pos", dataspec=rep.spec)
    cov = r.covers_data(rep)
    assert list(cov) == [False, False, True, False, True, True, False, True]


def test_read_csv_given_dataspec():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue", "green"])
    b.add_numeric("age", [30])
    given = b.build()
    rep = read_csv(io.StringIO(CSV_TEXT), dataspec=given, target="label")
    assert rep.spec is given

    age_idx = rep.spec.feature_index("age>=30")
    color_red_idx = rep.spec.feature_index("color=red")
    assert list(rep.X[:, age_idx]) == [False, False, True, False, True, True, False, True]
    assert list(rep.X[:, color_red_idx]) == [True, True, True, False, False, False, False, False]


def test_read_csv_strict_true_raises_on_mismatch():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])  # "green" missing from the domain on purpose
    b.add_numeric("age", [30])
    given = b.build()
    try:
        read_csv(io.StringIO(CSV_TEXT), dataspec=given, target="label", strict=True)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_read_csv_strict_false_allows_mismatch():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])  # "green" missing from the domain on purpose
    b.add_numeric("age", [30])
    given = b.build()
    rep = read_csv(io.StringIO(CSV_TEXT), dataspec=given, target="label", strict=False)

    red_idx = rep.spec.feature_index("color=red")
    blue_idx = rep.spec.feature_index("color=blue")
    colors = ["red", "red", "red", "blue", "blue", "blue", "green", "green"]
    green_rows = [i for i, v in enumerate(colors) if v == "green"]
    # rows with the unmodeled "green" category come back False on every color feature
    for i in green_rows:
        assert rep.X[i, red_idx] == False and rep.X[i, blue_idx] == False


def test_validate_dataspec_mismatches():
    import pandas as pd

    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])  # missing "green" category on purpose
    b.add_nominal("shape", ["circle", "square"])  # doesn't exist in the data at all
    ds = b.build()
    df = pd.read_csv(io.StringIO(CSV_TEXT))
    problems = validate_dataspec(ds, df)
    assert any("shape" in p and "no matching column" in p for p in problems)
    assert any("color" in p and "green" in p for p in problems)


def test_binarize_never_covers_missing_value_by_default():
    # row 2's color is missing -> every color=* feature is False for it
    # (NEVER_COVERS, the default), no exception
    text = "color,age,label\nred,20,neg\nblue,30,neg\n,25,pos\n"
    rep = read_csv(io.StringIO(text), target="label")
    red_idx = rep.spec.feature_index("color=red")
    blue_idx = rep.spec.feature_index("color=blue")
    assert rep.X[2, red_idx] == False
    assert rep.X[2, blue_idx] == False


def test_missing_strategy_majority_imputation():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])
    b.add_numeric("age", [30])
    ds = b.build()
    df = pd.DataFrame({
        "color": ["red", "red", "blue", None],   # mode among known values: "red"
        "age": [20.0, 40.0, 50.0, None],          # median among known values: 40 (>=30)
    })
    X = binarize(ds, df, missing_strategy=MissingStrategy.MAJORITY)
    color_red_idx = ds.feature_index("color=red")
    age_idx = ds.feature_index("age>=30")
    assert X[3, color_red_idx] == True
    assert X[3, age_idx] == True


def test_missing_strategy_random_imputation_reproducible():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])
    ds = b.build()
    df = pd.DataFrame({"color": ["red", "blue", "red", "blue", None]})

    X1 = binarize(ds, df, missing_strategy=MissingStrategy.RANDOM, random_state=0)
    X2 = binarize(ds, df, missing_strategy=MissingStrategy.RANDOM, random_state=0)
    assert (X1 == X2).all()  # same seed -> reproducible

    red_idx = ds.feature_index("color=red")
    blue_idx = ds.feature_index("color=blue")
    # the imputed value is always one of the column's known categories
    assert X1[4, red_idx] != X1[4, blue_idx]


def test_missing_strategy_separate_nominal_and_numeric():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"], missing_name="<missing>")
    b.add_numeric("age", [30], missing_name="<missing>")
    ds = b.build()
    df = pd.DataFrame({
        "color": ["red", None],
        "age": [40.0, None],
    })
    X = binarize(ds, df, missing_strategy=MissingStrategy.SEPARATE)

    color_red_idx = ds.feature_index("color=red")
    color_blue_idx = ds.feature_index("color=blue")
    color_missing_idx = ds.feature_index("color=<missing>")
    assert list(X[:, color_red_idx]) == [True, False]
    assert list(X[:, color_blue_idx]) == [False, False]
    assert list(X[:, color_missing_idx]) == [False, True]

    age_ge30_idx = ds.feature_index("age>=30")
    age_missing_idx = ds.feature_index("age=<missing>")
    assert list(X[:, age_ge30_idx]) == [True, False]
    assert list(X[:, age_missing_idx]) == [False, True]


def test_missing_strategy_separate_without_declared_feature_raises():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])  # no missing_name declared
    ds = b.build()
    df = pd.DataFrame({"color": ["red", None]})
    try:
        binarize(ds, df, missing_strategy=MissingStrategy.SEPARATE)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "missing_name" in str(e) or "missing-value feature" in str(e)


def test_missing_values_sentinel_recognition():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"], missing_values=("?",))
    ds = b.build()
    df = pd.DataFrame({"color": ["red", "?"]})
    X = binarize(ds, df)  # NEVER_COVERS default
    red_idx = ds.feature_index("color=red")
    blue_idx = ds.feature_index("color=blue")
    assert list(X[:, red_idx]) == [True, False]
    assert list(X[:, blue_idx]) == [False, False]


# -- encode: binarize's sparse-intermediate cousin --------------------------
# Same scenarios as the binarize missing-strategy tests above, but checked
# against encode() for every representation -- encode must agree with
# binarize(...) + rep_cls(...) exactly, not just "look plausible", since
# the whole point is dropping in as a replacement.

@pytest.mark.parametrize("rep_cls", _ENCODE_REPS)
def test_encode_matches_binarize(rep_cls):
    text = "color,age,label\nred,20,neg\nblue,30,neg\n,25,pos\n"
    boolean_rep = read_csv(io.StringIO(text), target="label")
    spec = boolean_rep.spec
    df = pd.DataFrame({"color": ["red", "blue", None], "age": [20, 30, 25]})
    y = np.array(["neg", "neg", "pos"])
    expected = rep_cls(spec, binarize(spec, df), y)
    got = encode(spec, df, y=y, representation=rep_cls)
    assert isinstance(got, rep_cls)
    assert np.array_equal(got.X, expected.X)
    assert np.array_equal(got.y, expected.y)


def test_encode_with_no_representation_arg_defaults_to_vertical_bitmap():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])
    ds = b.build()
    df = pd.DataFrame({"color": ["red", "blue", "red"]})
    got = encode(ds, df)
    assert isinstance(got, VerticalBitmapRepresentation)


@pytest.mark.parametrize("rep_cls", _ENCODE_REPS)
def test_encode_matches_binarize_majority_imputation(rep_cls):
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])
    b.add_numeric("age", [30])
    ds = b.build()
    df = pd.DataFrame({
        "color": ["red", "red", "blue", None],
        "age": [20.0, 40.0, 50.0, None],
    })
    expected = rep_cls(ds, binarize(ds, df, missing_strategy=MissingStrategy.MAJORITY))
    got = encode(ds, df, representation=rep_cls, missing_strategy=MissingStrategy.MAJORITY)
    assert np.array_equal(got.X, expected.X)


@pytest.mark.parametrize("rep_cls", _ENCODE_REPS)
def test_encode_matches_binarize_random_imputation_reproducible(rep_cls):
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])
    ds = b.build()
    df = pd.DataFrame({"color": ["red", "blue", "red", "blue", None]})
    expected = rep_cls(ds, binarize(ds, df, missing_strategy=MissingStrategy.RANDOM, random_state=0))
    got1 = encode(ds, df, representation=rep_cls, missing_strategy=MissingStrategy.RANDOM, random_state=0)
    got2 = encode(ds, df, representation=rep_cls, missing_strategy=MissingStrategy.RANDOM, random_state=0)
    assert np.array_equal(got1.X, expected.X)
    assert np.array_equal(got1.X, got2.X)  # same seed -> reproducible


@pytest.mark.parametrize("rep_cls", _ENCODE_REPS)
def test_encode_matches_binarize_separate_nominal_and_numeric(rep_cls):
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"], missing_name="<missing>")
    b.add_numeric("age", [30], missing_name="<missing>")
    ds = b.build()
    df = pd.DataFrame({"color": ["red", None], "age": [40.0, None]})
    expected = rep_cls(ds, binarize(ds, df, missing_strategy=MissingStrategy.SEPARATE))
    got = encode(ds, df, representation=rep_cls, missing_strategy=MissingStrategy.SEPARATE)
    assert np.array_equal(got.X, expected.X)


def test_encode_separate_without_declared_feature_raises():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])  # no missing_name declared
    ds = b.build()
    df = pd.DataFrame({"color": ["red", None]})
    with pytest.raises(ValueError, match="missing_name|missing-value feature"):
        encode(ds, df, missing_strategy=MissingStrategy.SEPARATE)


@pytest.mark.parametrize("rep_cls", _ENCODE_REPS)
def test_encode_passes_weights_through(rep_cls):
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"])
    ds = b.build()
    df = pd.DataFrame({"color": ["red", "blue", "red"]})
    w = np.array([1.0, 2.0, 3.0])
    got = encode(ds, df, representation=rep_cls, weights=w)
    assert np.array_equal(got.weights, w)


def test_dataspec_missing_strategy_default_resolution():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue"], missing_name="<missing>")
    ds = b.build(missing_strategy=MissingStrategy.SEPARATE)
    df = pd.DataFrame({"color": ["red", None]})
    missing_idx = ds.feature_index("color=<missing>")

    # no explicit missing_strategy= to binarize() -> resolves to ds.missing_strategy
    X = binarize(ds, df)
    assert list(X[:, missing_idx]) == [False, True]

    # an explicit argument still overrides the DataSpec's own default
    X2 = binarize(ds, df, missing_strategy=MissingStrategy.NEVER_COVERS)
    assert list(X2[:, missing_idx]) == [False, False]


def test_default_missing_strategy_is_never_covers():
    assert DataSpec.DEFAULT_MISSING_STRATEGY == MissingStrategy.NEVER_COVERS


def test_read_arff_infer():
    rep = read_arff(io.StringIO(ARFF_TEXT), target="play")
    assert rep.n_samples == 14
    assert "outlook=sunny" in rep.spec.feature_names
    assert any(n.startswith("temperature>=") for n in rep.spec.feature_names)
    assert list(rep.y) == [
        "no", "no", "yes", "yes", "yes", "no", "yes",
        "no", "yes", "yes", "yes", "yes", "yes", "no",
    ]


def test_read_arff_given_dataspec():
    b = DataSpecBuilder()
    b.add_nominal("outlook", ["sunny", "overcast", "rainy"])
    b.add_numeric("temperature", [75])
    given = b.build()
    rep = read_arff(io.StringIO(ARFF_TEXT), dataspec=given, target="play")
    assert rep.spec is given

    idx = rep.spec.feature_index("temperature>=75")
    temps = [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71]
    expected = [t >= 75 for t in temps]
    assert list(rep.X[:, idx]) == expected


def test_write_arff_round_trips_through_read_arff(tmp_path):
    df = pd.DataFrame({
        "outlook": ["sunny", "sunny", "overcast", "rainy", "rainy"],
        "temperature": [85, 80, 83, 70, 68],
        "play": ["no", "no", "yes", "yes", "yes"],
    })
    path = tmp_path / "weather.arff"
    write_arff(df, target="play", path=str(path), arff_types={"outlook": "nominal", "temperature": "numeric"})

    rep = read_arff(str(path), target="play")
    assert rep.n_samples == 5
    assert "outlook=sunny" in rep.spec.feature_names
    assert any(n.startswith("temperature>=") for n in rep.spec.feature_names)
    assert list(rep.y) == ["no", "no", "yes", "yes", "yes"]


def test_write_arff_quotes_nominal_values_needing_it():
    # regression test for a real bug: an *unquoted* nominal category
    # containing a space (e.g. "no checking") makes Weka's own ArffLoader
    # fail -- with a wholly unrelated-looking error pointing at a
    # *different* attribute -- confirmed directly against a real
    # weka.jar run, not just reasoned about here. This does NOT round-trip
    # through read_arff when quoting is needed (see write_arff's own
    # docstring for exactly why: a scipy limitation, not a pyrulearn bug),
    # so this test checks the written text directly instead of read_arff.
    df = pd.DataFrame({
        "checking_status": ["<0", "no checking", "<0", "0<=X<200"],
        "class": ["bad", "good", "bad", "good"],
    })
    buf = io.StringIO()
    write_arff(df, target="class", path=buf, arff_types={"checking_status": "nominal"})
    text = buf.getvalue()

    assert "'no checking'" in text  # quoted: contains a space
    assert "<0" in text and "'<0'" not in text  # not quoted: no special chars
    assert "'0<=X<200'" not in text  # not quoted either -- no space/comma/braces
    # data rows quote it too, matching the declaration -- Weka needs this
    # in both places (confirmed directly), even though it means this
    # specific file can't round-trip through read_arff (see write_arff's
    # docstring)
    assert "'no checking',good" in text


def test_write_arff_quotes_attribute_names_needing_it():
    # regression test for a real bug found running the workflow-comparison
    # demo against OpenML's mushroom dataset: sklearn's fetch_openml
    # literally names one column "bruises%3F" (OpenML's own percent-encoding
    # of "bruises?"). Unquoted, "@attribute bruises%3F {f,t}" is corrupted by
    # ARFF's own rule that '%' starts a comment *anywhere* on a line, not
    # just at line start -- truncating the whole domain declaration and
    # making Weka fail with an unrelated-looking "Can't open file" error.
    # Confirmed directly against a real weka.jar run (both the break and
    # the fix), not just reasoned about here.
    df = pd.DataFrame({
        "bruises%3F": ["f", "t", "f"],
        "class": ["e", "p", "e"],
    })
    buf = io.StringIO()
    write_arff(df, target="class", path=buf, arff_types={"bruises%3F": "nominal"})
    text = buf.getvalue()

    assert "@attribute 'bruises%3F'" in text
    assert "@attribute bruises%3F " not in text  # unquoted form would corrupt parsing


def test_write_arff_feature_names_override():
    # the motivating case: write already-binarized 0/1 values under safe
    # placeholder names instead of a DataSpec's own '>='/'='-laden ones
    b = DataSpecBuilder()
    b.add_numeric("age", [30])
    b.add_nominal("color", ["red", "blue"])
    ds = b.build()

    bool_df = pd.DataFrame({
        ds.feature_names[0]: [1, 0, 1],
        ds.feature_names[1]: [1, 0, 0],
        ds.feature_names[2]: [0, 1, 1],
        "label": ["pos", "neg", "pos"],
    })
    buf = io.StringIO()
    write_arff(bool_df, target="label", path=buf, feature_names=["f0", "f1", "f2"])
    text = buf.getvalue()

    assert "@attribute f0 numeric" in text
    assert ds.feature_names[0] not in text  # the real ">="-laden name never appears
    assert "1,1,0,pos" in text

    buf.seek(0)
    rep = read_arff(buf, target="label")
    assert any(n.startswith(("f0", "f1", "f2")) for n in rep.spec.feature_names)


def test_write_csv_round_trips_through_read_csv(tmp_path):
    df = pd.DataFrame({
        "color": ["red", "red", "blue", "green"],
        "age": [20, 45, 21, 60],
        "label": ["neg", "pos", "neg", "pos"],
    })
    path = tmp_path / "toy.csv"
    write_csv(df, target="label", path=str(path))

    rep = read_csv(str(path), target="label")
    assert rep.n_samples == 4
    assert {"color=red", "color=blue", "color=green"} <= set(rep.spec.feature_names)
    assert list(rep.y) == ["neg", "pos", "neg", "pos"]


def test_write_csv_feature_names_override():
    df = pd.DataFrame({"color": ["red", "blue"], "age": [20, 45], "label": ["neg", "pos"]})
    buf = io.StringIO()
    write_csv(df, target="label", path=buf, feature_names=["f0", "f1"])
    text = buf.getvalue()

    assert text.splitlines()[0] == "f0,f1,label"
    assert "color" not in text.split("\n")[0]

    buf.seek(0)
    rep = read_csv(buf, target="label")
    assert any(n.startswith("f0") for n in rep.spec.feature_names)


def test_write_csv_wrong_feature_names_length_raises():
    df = pd.DataFrame({"color": ["red", "blue"], "age": [20, 45], "label": ["neg", "pos"]})
    try:
        write_csv(df, target="label", path=io.StringIO(), feature_names=["f0"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "feature_names" in str(e)


def test_build_dataspec_makes_two_valued_columns_binary():
    from pyrulearn.data.attributes import AttributeType
    df = pd.DataFrame({
        "sex": ["m", "f", "m", "f"],
        "c": ["a", "b", "c", "a"],
        "y": ["p", "n", "p", "n"],
    })
    ds = build_dataspec(df, target="y").build()
    assert ds.attributes["sex"].type == AttributeType.BINARY
    assert ds.attributes["c"].type == AttributeType.NOMINAL
    assert "sex!=f" not in ds.feature_names and "c!=a" in ds.feature_names
    # a declared domain wins over the observed values
    ds = build_dataspec(df, target="y", domains={"sex": ["f", "m", "x"]}).build()
    assert ds.attributes["sex"].type == AttributeType.NOMINAL


ARFF_WITH_MISSING = """@relation t
@attribute sex {male,female}
@attribute c {a,b,c}
@attribute y {p,n}
@data
male,a,p
?,b,n
female,?,p
"""


def test_read_arff_treats_question_mark_as_missing_and_uses_the_header_domain():
    from pyrulearn.data.attributes import AttributeType
    rep = read_arff(io.StringIO(ARFF_WITH_MISSING), target="y")
    ds = rep.spec
    assert ds.attributes["sex"].type == AttributeType.BINARY
    assert ds.attributes["c"].type == AttributeType.NOMINAL
    assert list(ds.attributes["c"].domain) == ["a", "b", "c"]  # 'c' never occurs, still declared
    assert not any("?" in n for n in ds.feature_names)
    row = dict(zip(ds.feature_names, rep.X[1].astype(int)))  # sex missing
    assert row["sex=male"] == 0 and row["sex=female"] == 0
    row = dict(zip(ds.feature_names, rep.X[2].astype(int)))  # c missing: no c feature holds
    assert all(v == 0 for n, v in row.items() if n.startswith("c"))


def test_bool_columns_are_boolean_attributes():
    df = pd.DataFrame({"a": [True, False, True], "c": ["x", "y", "x"], "y": ["p", "n", "p"]})
    spec = build_dataspec(df, target="y").build()
    assert spec.feature_names == ["a", "not a", "c=x", "c=y"]
    X = np.asarray(encode(spec, df, df["y"].to_numpy()).X)
    assert X[:, :2].astype(int).tolist() == [[1, 0], [0, 1], [1, 0]]
    assert build_dataspec(df, target="y", include_negations=False).build().feature_names[:1] == ["a"]
