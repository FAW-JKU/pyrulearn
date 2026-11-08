import numpy as np
import pytest

from pyrulearn.datasets import KRKIllegal, krk_illegal, krk_is_illegal


def _illegal(position):
    """`position` like "a1 c3 e8": white king, white rook, black king."""
    coords = []
    for square in position.split():
        coords += ["abcdefgh".index(square[0]), int(square[1]) - 1]
    return bool(krk_is_illegal(*coords))


@pytest.mark.parametrize("position, illegal", [
    ("a1 c3 e8", False),
    ("a1 a1 e8", True),   # two pieces on one square
    ("d4 a1 d4", True),
    ("d4 e5 e6", True),   # the kings next to each other ...
    ("d4 a1 f6", False),  # ... or not
    ("a1 c3 c8", True),   # the rook gives check along the file ...
    ("a1 c3 h3", True),   # ... or the rank
    ("c5 c3 c8", False),  # the white king between them blocks it
    ("c1 c3 c8", True),   # behind the rook it doesn't
    ("e3 c3 h3", False),
    ("h8 c3 d4", False),  # diagonal: a rook doesn't attack it
])
def test_krk_illegality_rules(position, illegal):
    assert _illegal(position) == illegal


def test_krk_illegal_samples():
    full = krk_illegal(None)
    assert len(full) == 8 ** 6 and (full["class"] == "illegal").sum() == 86976  # about a third
    d = krk_illegal(1000, random_state=1)
    assert len(d) == 1000 and not d.drop(columns="class").duplicated().any()
    assert d.equals(krk_illegal(1000, random_state=1)) and not d.equals(krk_illegal(1000, random_state=2))
    assert list(d.columns) == ["wkf", "wkr", "wrf", "wrr", "bkf", "bkr", "class"]
    assert set(d["wkf"]) <= set("abcdefgh") and set(d["bkr"]) <= set("12345678")
    with pytest.raises(ValueError):
        krk_illegal(0)


def test_krk_illegal_as_a_dataset():
    df, target = KRKIllegal(500).load()
    assert target == "class" and len(df) == 500
    assert KRKIllegal(500).name == "krk-illegal-500" and KRKIllegal(500, 3).name == "krk-illegal-500-seed3"
    assert KRKIllegal(500).openml_id is None
    np.testing.assert_array_equal(df["class"], krk_illegal(500)["class"])


def test_krk_relations_are_relational_features():
    from pyrulearn.data.io import build_dataspec, encode

    d = krk_illegal(300, relations=True)
    assert list(d.columns[:6]) == ["wkf", "wkr", "wrf", "wrr", "bkf", "bkr"] and d.columns[-1] == "class"
    assert len(d.columns) == 6 + 24 + 1
    files = {c: d[c].astype(str).map("abcdefgh".index).astype(int) for c in ("wrf", "bkf")}
    np.testing.assert_array_equal(d["eq(wrf,bkf)"], files["wrf"] == files["bkf"])
    np.testing.assert_array_equal(d["adj(wrf,bkf)"], (files["wrf"] - files["bkf"]).abs() == 1)
    np.testing.assert_array_equal(d["lt(bkf,wrf)"], files["bkf"] < files["wrf"])
    # a fold keeps them relational: sources, expression, a negation, values from the columns
    fold = d.iloc[:200].reset_index(drop=True)
    spec = build_dataspec(fold, target="class").build()
    attr = spec.attributes["eq(wrf,bkf)"]
    assert attr.type.name == "RELATIONAL" and attr.sources == ("wrf", "bkf")
    assert spec.feature_spec(spec.feature_index("eq(wrf,bkf)")).expression == "wrf = bkf"
    X = np.asarray(encode(spec, fold, fold["class"].to_numpy()).X, dtype=bool)
    np.testing.assert_array_equal(X[:, spec.feature_index("eq(wrf,bkf)")], fold["eq(wrf,bkf)"])
    np.testing.assert_array_equal(X[:, spec.feature_index("not eq(wrf,bkf)")], ~fold["eq(wrf,bkf)"])
    assert KRKIllegal(300, relations=True).name == "krk-illegal-300-relational"


def test_krk_relations_give_an_exact_theory():
    # illegal iff: two pieces on one square, kings adjacent, or a check the white king doesn't block
    d = krk_illegal(None, relations=True)
    same = lambda p, q: d[f"eq({p}f,{q}f)"] & d[f"eq({p}r,{q}r)"]  # noqa: E731
    near = lambda x, y: d[f"eq({x},{y})"] | d[f"adj({x},{y})"]  # noqa: E731
    between = lambda d_: (d[f"lt(wr{d_},wk{d_})"] & d[f"lt(wk{d_},bk{d_})"]) | (  # noqa: E731
        d[f"lt(bk{d_},wk{d_})"] & d[f"lt(wk{d_},wr{d_})"])
    illegal = (same("wk", "wr") | same("wk", "bk") | same("wr", "bk")
               | (near("wkf", "bkf") & near("wkr", "bkr"))
               | (d["eq(wrf,bkf)"] & ~(d["eq(wkf,wrf)"] & between("r")))
               | (d["eq(wrr,bkr)"] & ~(d["eq(wkr,wrr)"] & between("f"))))
    np.testing.assert_array_equal(illegal, d["class"] == "illegal")


def test_krk_relations_only():
    d = krk_illegal(200, relations=True, coordinates=False)
    assert len(d.columns) == 24 + 1 and "wkf" not in d.columns
    assert KRKIllegal(200, relations=True, coordinates=False).name == "krk-illegal-200-relational-only"
    with pytest.raises(ValueError, match="needs relations"):
        krk_illegal(200, coordinates=False)
