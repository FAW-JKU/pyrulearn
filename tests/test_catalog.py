"""The dataset catalog: the shipped file's consistency and the selection
logic. Offline -- nothing here downloads data."""

import pandas as pd
import pytest

from pyrulearn.experiments.catalog import (
    ATTRIBUTE_KINDS, MEDIUM_MAX, SIZES, SMALL_MAX, TASKS, Catalog, CatalogEntry, prepare_frame,
)

KNOWN_TAGS = {"cc18", "lord", "xai", "classic"}


@pytest.fixture(scope="module")
def cat():
    return Catalog.default()


def test_shipped_catalog_is_consistent(cat):
    assert len(cat) > 50
    for e in cat:
        assert e.tags and set(e.tags) <= KNOWN_TAGS, e.name
        assert e.openml_id is not None or e.uci or e.kaggle, f"{e.name} has no source"
        if e.loadable:
            # the statistics tools/build_catalog.py fills in are all there
            for f in ("n_instances", "n_features", "n_numeric", "n_nominal", "n_classes"):
                assert getattr(e, f) is not None, (e.name, f)
            assert e.n_numeric + e.n_nominal == e.n_features, e.name
            assert e.task in TASKS and e.size in SIZES and e.attributes in ATTRIBUTE_KINDS, e.name
    ids = [e.openml_id for e in cat if e.openml_id is not None]
    assert len(ids) == len(set(ids)), "an OpenML dataset appears twice"


def test_shipped_catalog_round_trips(cat):
    again = Catalog.from_json(cat.to_json())
    assert [vars(e) for e in again] == [vars(e) for e in cat]


def test_shipped_catalog_covers_every_category(cat):
    for task in TASKS:
        for size in SIZES:
            assert cat.select(task=task, size=size), (task, size)
    for kind in ATTRIBUTE_KINDS:
        assert cat.select(attributes=kind), kind
    for tag in KNOWN_TAGS:
        assert cat.select(tags=tag, loadable=None), tag


def test_categories_follow_the_counts():
    e = CatalogEntry("x", n_classes=2, n_instances=SMALL_MAX - 1, n_numeric=0, n_nominal=3)
    assert (e.task, e.size, e.attributes) == ("binary", "small", "categorical")
    e = CatalogEntry("x", n_classes=3, n_instances=MEDIUM_MAX, n_numeric=2, n_nominal=0)
    assert (e.task, e.size, e.attributes) == ("multiclass", "medium", "numeric")
    e = CatalogEntry("x", n_classes=5, n_instances=MEDIUM_MAX + 1, n_numeric=2, n_nominal=1)
    assert (e.size, e.attributes) == ("large", "mixed")
    assert CatalogEntry("x").task is None and not CatalogEntry("x").loadable


def _toy():
    return Catalog([
        CatalogEntry("a", openml_id=1, tags=["lord"], n_classes=2, n_instances=100, n_numeric=0, n_nominal=4,
                     missing_rate=0.0),
        CatalogEntry("b", openml_id=2, tags=["cc18"], n_classes=2, n_instances=5000, n_numeric=3, n_nominal=0,
                     missing_rate=0.1),
        CatalogEntry("c", openml_id=3, tags=["cc18", "xai"], n_classes=4, n_instances=50000, n_numeric=1,
                     n_nominal=1, missing_rate=0.0),
        CatalogEntry("d", uci="https://archive.ics.uci.edu/dataset/1/x", tags=["lord"], n_instances=10**6),
    ])


def test_select_combines_axes_with_and_and_values_with_or():
    cat = _toy()
    names = lambda es: [e.name for e in es]  # noqa: E731
    assert names(cat.select(task="binary")) == ["a", "b"]
    assert names(cat.select(task="binary", size=["small", "medium"])) == ["a", "b"]
    assert names(cat.select(task="binary", size="medium")) == ["b"]
    assert names(cat.select(attributes=["categorical", "mixed"])) == ["a", "c"]
    assert names(cat.select(tags="xai")) == ["c"]
    assert names(cat.select(missing=True)) == ["b"]
    # pointer-only entries are skipped unless asked for
    assert names(cat.select(tags="lord")) == ["a"]
    assert names(cat.select(tags="lord", loadable=None)) == ["a", "d"]
    with pytest.raises(ValueError):
        cat.select(size="huge")
    with pytest.raises(KeyError):
        cat.select(names="nope")


def test_parse_sorts_words_onto_their_axis():
    cat = _toy()
    names = lambda es: [e.name for e in es]  # noqa: E731
    assert names(cat.parse("binary,small,medium")) == ["a", "b"]
    assert names(cat.parse("cc18")) == ["b", "c"]
    assert names(cat.parse("multiclass,c")) == ["c"]
    assert names(cat.parse("a,b")) == ["a", "b"]
    assert names(cat.parse("binary,small,c")) == ["a", "c"]  # names add to the filtered selection
    assert names(cat.parse("all")) == ["a", "b", "c"]
    with pytest.raises(ValueError):
        cat.parse("binary,nonsense")


def test_select_n_picks_a_reproducible_random_subset(cat):
    pool = cat.select(task="binary", size="medium")
    three = cat.select(task="binary", size="medium", n=3, random_state=1)
    assert len(three) == 3 and all(e in pool for e in three)
    assert [e.name for e in three] == [e.name for e in pool if e in three]  # catalog order kept
    assert three == cat.select(task="binary", size="medium", n=3, random_state=1)
    # different seeds eventually give different picks
    assert len({tuple(e.name for e in cat.select(task="binary", size="medium", n=3, random_state=s))
                for s in range(10)}) > 1
    assert cat.select(task="binary", size="medium", n=0) == []
    with pytest.raises(ValueError):
        cat.select(task="binary", size="medium", n=len(pool) + 1)


def test_parse_takes_a_number_for_a_random_pick():
    cat = _toy()
    assert len(cat.parse("binary,1", random_state=0)) == 1
    assert len(cat.parse("2", random_state=0)) == 2           # from every loadable entry
    picked = cat.parse("binary,1,c", random_state=0)          # names still added on top
    assert len(picked) == 2 and picked[-1].name == "c"
    with pytest.raises(ValueError):
        cat.parse("binary,1,2")


def test_select_by_names_keeps_the_given_order():
    cat = _toy()
    assert [e.name for e in cat.select(names=["c", "a", "b"])] == ["c", "a", "b"]
    assert [e.name for e in cat.select(names=["c", "a"], task="binary")] == ["a"]  # other criteria still apply
    assert [e.name for e in cat.parse("c,a")] == ["c", "a"]


def test_summary_takes_the_same_criteria_as_select():
    cat = _toy()
    lines = cat.summary().splitlines()
    assert len(lines) == 1 + len(cat)                          # header + every entry, pointers included
    lines = cat.summary(task="binary").splitlines()
    assert [ln.split()[0] for ln in lines[1:]] == ["a", "b"]
    assert "d" in [ln.split()[0] for ln in cat.summary(tags="lord").splitlines()[1:]]
    assert "d" not in [ln.split()[0] for ln in cat.summary(tags="lord", loadable=True).splitlines()[1:]]


def test_prepare_applies_the_curated_fixes():
    e = CatalogEntry("x", rename={"V1": "age", "V2": "sex", "V3": "leak", "V4": "code"},
                     drop=["leak"], nominal=["code"], missing_markers={"age": [0]})
    df = pd.DataFrame({"V1": [30.0, 0.0, 50.0, 40.0], "V2": ["m", "f", "m", "m"],
                       "V3": [1, 2, 3, 4], "V4": [1.0, 2.0, 3.0, None],
                       "const": [7, 7, 7, 7], "y": ["p", "n", None, "p"]})
    out = e.prepare(df, "y")
    assert list(out.columns) == ["age", "sex", "code", "y"]   # leak dropped, constant dropped
    assert len(out) == 3                                      # row without a class dropped
    assert pd.isna(out["age"].iloc[1])                        # 0 marks "not measured"
    assert str(out["code"].dtype) == "category"
    assert list(out["code"].iloc[:2]) == ["1", "2"] and pd.isna(out["code"].iloc[2])


def test_prepare_frame_keeps_degenerate_columns_when_asked():
    df = pd.DataFrame({"const": [1, 1], "y": ["a", "b"]})
    assert list(prepare_frame(df, "y", drop_degenerate=False).columns) == ["const", "y"]
