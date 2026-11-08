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


def test_parity():
    from pyrulearn.datasets import Parity, parity

    d = parity(3, 2)
    assert len(d) == 32 and list(d.columns) == ["x1", "x2", "x3", "x4", "x5", "class"]
    assert d["x1"].dtype == bool
    odd = (d[["x1", "x2", "x3"]].sum(axis=1) % 2 == 1)
    np.testing.assert_array_equal(d["class"] == "odd", odd)  # x4, x5 don't matter
    s = parity(4, 6, n=300, random_state=1)
    assert len(s) == 300 and not s.drop(columns="class").duplicated().any()
    noisy = parity(4, 6, noise=0.1, random_state=2)
    clean = parity(4, 6)
    assert 0.05 < (noisy["class"] != clean["class"]).mean() < 0.15
    assert Parity().name == "parity-4-of-10" and Parity(3, 5, n=100, noise=0.05).name == "parity-3-of-8-n100-noise0.05"
    df, target = Parity(3, 2).load()
    assert target == "class" and df.equals(d)
    with pytest.raises(ValueError):
        parity(0, 3)


def test_parity_groups():
    from pyrulearn.datasets import ParityGroups, parity_groups

    d = parity_groups(4, 3, 2, n_irrelevant=1)
    assert len(d) == 2 ** 13 and list(d.columns) == [f"x{i}" for i in range(1, 14)] + ["class"]
    x = d.drop(columns="class").to_numpy()
    odd = x[:, :12].reshape(-1, 4, 3).sum(axis=2) % 2 == 1
    np.testing.assert_array_equal(d["class"] == "yes", odd.sum(axis=1) >= 2)  # x13 doesn't matter
    assert abs((d["class"] == "yes").mean() - 11 / 16) < 1e-12
    s = parity_groups(n=500, random_state=3)
    assert len(s) == 500 and not s.drop(columns="class").duplicated().any()
    assert ParityGroups().name == "parity-groups-2-of-4x3"
    assert ParityGroups(4, 2, 2, n_irrelevant=4, n=1000).name == "parity-groups-2-of-4x2-plus4-n1000"
    df, target = ParityGroups(3, 2, 1).load()
    assert target == "class" and df.equals(parity_groups(3, 2, 1))
    with pytest.raises(ValueError, match="threshold"):
        parity_groups(3, 2, 4)


def test_parity_groups_has_the_small_deep_network():
    # the network of the docstring -- each group's parity in layers 1-2, the pairs of groups in
    # layer 3 (AND), their OR at the output -- computes the concept exactly, with 82 edges
    from itertools import combinations, product

    from pyrulearn.datasets import parity_groups
    from pyrulearn.learners.deep import nnf_forward

    d = parity_groups(4, 3, 2)
    x = d.drop(columns="class").to_numpy()
    X = np.hstack([x, ~x])  # x1..x12, then not x1..not x12
    odd_terms = [t for t in product((0, 1), repeat=3) if sum(t) % 2 == 1]
    w1 = np.zeros((16, 24), dtype=bool)
    for g in range(4):
        for k, t in enumerate(odd_terms):
            for i, bit in enumerate(t):
                w1[4 * g + k, 3 * g + i + (0 if bit else 12)] = True
    w2 = np.zeros((4, 16), dtype=bool)
    for g in range(4):
        w2[g, 4 * g:4 * g + 4] = True
    pairs = list(combinations(range(4), 2))
    w3 = np.zeros((6, 4), dtype=bool)
    for k, (a, b) in enumerate(pairs):
        w3[k, [a, b]] = True
    w4 = np.ones((1, 6), dtype=bool)
    weights = [w1, w2, w3, w4]
    assert sum(int(w.sum()) for w in weights) == 48 + 16 + 12 + 6
    out = nnf_forward(weights, X)[-1][0]
    np.testing.assert_array_equal(out, d["class"] == "yes")
