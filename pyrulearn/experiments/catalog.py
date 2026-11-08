"""
pyrulearn.experiments.catalog
======================

A catalog of benchmark datasets: metadata and download pointers only --
no data is bundled. Each entry names where the data comes from (an
OpenML id, plus UCI/Kaggle/original-source links where known) and
carries enough statistics to *select* datasets without downloading
them::

    from pyrulearn.experiments.catalog import Catalog

    cat = Catalog.default()
    for entry in cat.select(task="binary", size=["small", "medium"]):
        df, target = entry.load()          # downloaded once, then cached

Three computed categories organize the pool:

- `task` -- ``"binary"`` (2 classes) or ``"multiclass"`` (3 or more).
- `size` -- ``"small"`` (fewer than `SMALL_MAX` rows), ``"medium"`` (up
  to `MEDIUM_MAX`), ``"large"`` (more).
- `attributes` -- ``"categorical"`` (every attribute nominal),
  ``"numeric"`` (every attribute numeric) or ``"mixed"``.

`tags` record which collection an entry belongs to: ``cc18`` (the
OpenML-CC18 benchmark suite, OpenML study 99, minus its image/signal/text
datasets with more than 200 features), ``lord`` (the datasets of the
LORD evaluation, Huynh, Fürnkranz & Beck 2023), ``xai`` (classic
explainable-AI datasets such as adult or COMPAS), ``classic`` (the
traditional rule-learning benchmarks: vote, mushroom, soybean, ...),
``deep`` (the datasets of the deep rule learning evaluations, Beck &
Fürnkranz 2021 and Seip et al.: their 20 artificial datasets, monk-1/2/3,
tic-tac-toe, vote, kr-vs-kp, mushroom and binary versions of car and
connect-4) and ``artificial`` (generated data, e.g. truth tables
labeled by a random formula).

`load` fetches through `sklearn.datasets.fetch_openml`, whose own cache
(``~/scikit_learn_data`` unless `data_home` or ``$PYRULEARN_DATA_HOME``
says otherwise) makes every later load offline. It returns the raw data
with real missing values -- no imputation -- after applying the entry's
curated fixes: `rename` restores documented attribute names where the
OpenML upload anonymized them (``V1``, ``x1``, ...), `drop` removes
identifier and leakage columns, `missing_markers` (values that encode
"not measured", e.g. Pima diabetes's zeros) become missing, and
`nominal` columns (categories stored as integer codes) become
categorical. An entry not on OpenML can name a `csv_url` instead: a
CSV file, downloaded once into the same data home (``pyrulearn_csv/``)
and read with pandas (empty cells missing). Entries with neither are
pointers only (`loadable` is False).

The catalog file (``catalog.json``, next to this module) is maintained
with ``tools/build_catalog.py``: the curated fields (`openml_id`,
`target`, `tags`, `nominal`, `missing_markers`, links, `notes`) are
edited by hand, the statistics are filled in from OpenML.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

#: `size` boundaries, in rows: small < SMALL_MAX <= medium <= MEDIUM_MAX < large.
SMALL_MAX = 1_000
MEDIUM_MAX = 10_000

TASKS = ("binary", "multiclass")
SIZES = ("small", "medium", "large")
ATTRIBUTE_KINDS = ("categorical", "numeric", "mixed")

#: Fields filled in by tools/build_catalog.py rather than by hand.
COMPUTED_FIELDS = (
    "openml_name", "openml_version", "original_url", "license",
    "n_instances", "n_features", "n_numeric", "n_nominal", "n_classes",
    "majority_rate", "missing_rate",
)


@dataclass
class CatalogEntry:
    """One dataset: where to get it, how to read it, and its statistics.

    Curated: `name`, `openml_id` or `csv_url` (where the data comes
    from), `target` (only when OpenML's default target is missing or
    wrong; always for a `csv_url`), `tags`, `uci`/`kaggle` (page URLs),
    `rename` (``{openml_column: name}``, applied first -- every other
    curated field uses the new names), `drop` (identifier or leakage
    columns), `nominal` (columns stored as numbers that are really
    categories), `missing_markers` (``{column: [values meaning
    missing]}``), `notes`.
    Computed: see `COMPUTED_FIELDS`; `missing_rate` is the fraction of
    rows with at least one missing value (after `missing_markers`),
    `majority_rate` the fraction of the most frequent class.
    """

    name: str
    openml_id: Optional[int] = None
    csv_url: Optional[str] = None
    target: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    uci: Optional[str] = None
    kaggle: Optional[str] = None
    rename: Dict[str, str] = field(default_factory=dict)
    drop: List[str] = field(default_factory=list)
    nominal: List[str] = field(default_factory=list)
    missing_markers: Dict[str, List[Any]] = field(default_factory=dict)
    notes: Optional[str] = None
    # computed
    openml_name: Optional[str] = None
    openml_version: Optional[int] = None
    original_url: Optional[str] = None
    license: Optional[str] = None
    n_instances: Optional[int] = None
    n_features: Optional[int] = None
    n_numeric: Optional[int] = None
    n_nominal: Optional[int] = None
    n_classes: Optional[int] = None
    majority_rate: Optional[float] = None
    missing_rate: Optional[float] = None

    # -- categories ---------------------------------------------------------

    @property
    def task(self) -> Optional[str]:
        if self.n_classes is None:
            return None
        return "binary" if self.n_classes == 2 else "multiclass"

    @property
    def size(self) -> Optional[str]:
        if self.n_instances is None:
            return None
        if self.n_instances < SMALL_MAX:
            return "small"
        return "medium" if self.n_instances <= MEDIUM_MAX else "large"

    @property
    def attributes(self) -> Optional[str]:
        if self.n_numeric is None or self.n_nominal is None:
            return None
        if self.n_numeric == 0:
            return "categorical"
        return "numeric" if self.n_nominal == 0 else "mixed"

    @property
    def has_missing(self) -> Optional[bool]:
        return None if self.missing_rate is None else self.missing_rate > 0

    @property
    def loadable(self) -> bool:
        return self.openml_id is not None or self.csv_url is not None

    @property
    def openml_url(self) -> Optional[str]:
        return None if self.openml_id is None else f"https://www.openml.org/d/{self.openml_id}"

    # -- loading ------------------------------------------------------------

    def load(self, data_home: Optional[str] = None, drop_degenerate: bool = True):
        """Download (or read from the cache) and return ``(df, target)``:
        a pandas DataFrame with real missing values (never imputed) and
        the name of its class column. `rename`, `drop`, `missing_markers`
        and `nominal` are applied; rows with a missing class are dropped;
        `drop_degenerate` (default) also drops attribute columns that are
        entirely missing or constant, which carry no information and
        which `pyrulearn.data.io.build_dataspec` can't discretize.

        Needs `pandas`; raises `ValueError` for a pointer-only entry.
        """
        if self.openml_id is None and self.csv_url is None:
            links = ", ".join(u for u in (self.uci, self.kaggle, self.original_url) if u) or "no link recorded"
            raise ValueError(f"{self.name!r} has no OpenML id and can't be loaded automatically ({links})")
        if self.openml_id is None:
            if not self.target:
                raise ValueError(f"{self.name!r} reads a CSV file -- set the entry's `target`")
            return self.prepare(fetch_csv_frame(self.csv_url, data_home), self.target, drop_degenerate), self.target
        frame, default_target = fetch_openml_frame(self.openml_id, data_home)
        target = self.target or self.rename.get(default_target, default_target)
        if not target:
            raise ValueError(f"OpenML sets no default target for {self.name!r} -- set the entry's `target`")
        return self.prepare(frame, target, drop_degenerate), target

    def prepare(self, df, target: str, drop_degenerate: bool = True):
        """Apply this entry's curated fixes to a freshly fetched frame
        (see `load`)."""
        return prepare_frame(df.rename(columns=self.rename), target, self.nominal, self.missing_markers,
                             drop_degenerate, drop=self.drop)

    def __repr__(self) -> str:
        cats = "/".join(str(c) for c in (self.task, self.size, self.attributes))
        return f"CatalogEntry({self.name!r}, openml_id={self.openml_id}, {cats})"


def fetch_openml_frame(openml_id: int, data_home: Optional[str] = None):
    """``(frame, default_target)``: every column of OpenML dataset
    `openml_id` (cached by scikit-learn), and OpenML's default target
    attribute (`None` if it sets none)."""
    import warnings
    from sklearn.datasets import fetch_openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bunch = fetch_openml(data_id=openml_id, as_frame=True, parser="auto", target_column=None,
                             data_home=data_home or os.environ.get("PYRULEARN_DATA_HOME"))
    default = bunch.details.get("default_target_attribute") or None
    if default and "," in default:
        raise ValueError(f"OpenML dataset {openml_id} has several default targets ({default})")
    return bunch.frame.copy(), default


def fetch_csv_frame(url: str, data_home: Optional[str] = None):
    """The CSV file at `url` as a DataFrame (empty cells missing), read
    from ``pyrulearn_csv/`` in the data home (`data_home`,
    ``$PYRULEARN_DATA_HOME`` or ``~/scikit_learn_data``), where the first
    call downloads it."""
    import hashlib
    import urllib.request

    import pandas as pd

    home = data_home or os.environ.get("PYRULEARN_DATA_HOME") or os.path.join("~", "scikit_learn_data")
    folder = os.path.join(os.path.expanduser(home), "pyrulearn_csv")
    path = os.path.join(folder, hashlib.sha1(url.encode("utf-8")).hexdigest()[:12] + "_"
                        + os.path.basename(url.split("?")[0]))
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    return pd.read_csv(path)


def prepare_frame(df, target: str, nominal: Sequence[str] = (), missing_markers: Optional[Dict[str, List[Any]]] = None,
                  drop_degenerate: bool = True, drop: Sequence[str] = ()):
    """The loading fixes `CatalogEntry.load` applies (after `rename`),
    on an already fetched DataFrame."""
    import pandas as pd

    df = df.drop(columns=list(drop))
    for col, values in (missing_markers or {}).items():
        df[col] = df[col].mask(df[col].isin(values))
    for col in nominal:
        # keep the codes readable ("1", not 1.0) and missing values missing
        df[col] = df[col].map(lambda v: v if pd.isna(v) else _code_label(v)).astype("category")
    df = df[df[target].notna()].reset_index(drop=True)
    if drop_degenerate:
        keep = [c for c in df.columns
                if c == target or df[c].nunique(dropna=True) > 1]
        df = df[keep]
    return df


def _code_label(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v)


# -- the catalog --------------------------------------------------------------

Selector = Union[None, str, Sequence[str]]


def _as_set(value: Selector) -> Optional[set]:
    if value is None:
        return None
    return {value} if isinstance(value, str) else set(value)


class Catalog:
    """An ordered collection of `CatalogEntry`s, indexed by name.
    `Catalog.default()` reads the packaged ``catalog.json``."""

    def __init__(self, entries: Iterable[CatalogEntry]):
        self.entries: List[CatalogEntry] = list(entries)
        self._by_name = {e.name: e for e in self.entries}
        if len(self._by_name) != len(self.entries):
            raise ValueError("catalog entry names must be unique")

    @classmethod
    def default(cls) -> "Catalog":
        with resources.files(__package__).joinpath("catalog.json").open(encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    @classmethod
    def from_file(cls, path: str) -> "Catalog":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Catalog":
        known = set(CatalogEntry.__dataclass_fields__)
        entries = []
        for raw in data["datasets"]:
            unknown = set(raw) - known
            if unknown:
                raise ValueError(f"catalog entry {raw.get('name')!r} has unknown fields {sorted(unknown)}")
            entries.append(CatalogEntry(**raw))
        return cls(entries)

    def to_json(self) -> Dict[str, Any]:
        """The file format: every field in declaration order, `None`
        and empty curated fields left out."""
        out = []
        for e in self.entries:
            row = {}
            for name in CatalogEntry.__dataclass_fields__:
                v = getattr(e, name)
                if v is None or v == [] or v == {}:
                    continue
                row[name] = v
            out.append(row)
        return {"datasets": out}

    def __iter__(self) -> Iterator[CatalogEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> CatalogEntry:
        return self._by_name[name]

    def select(
        self,
        task: Selector = None,
        size: Selector = None,
        attributes: Selector = None,
        tags: Selector = None,
        names: Selector = None,
        missing: Optional[bool] = None,
        loadable: Optional[bool] = True,
        n: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> List[CatalogEntry]:
        """The entries matching every given criterion -- in the order
        `names` lists them if given, else in catalog order. Each
        criterion takes one value or a list of alternatives (a list
        means *any of*): `task` from `TASKS`, `size` from `SIZES`,
        `attributes` from `ATTRIBUTE_KINDS`, `tags` (an entry matches if
        it has any of them), `names`. `missing` True/False keeps entries
        with/without missing values. `loadable` (default True) skips
        pointer-only entries; `None` keeps both.

        `n` picks that many of the matching entries at random (keeping
        the order above) -- e.g. ``select(task="binary", size="medium",
        n=3)``; `random_state` seeds the pick for a reproducible choice.
        Raises `ValueError` if fewer than `n` entries match."""
        for value, allowed, axis in ((task, TASKS, "task"), (size, SIZES, "size"),
                                     (attributes, ATTRIBUTE_KINDS, "attributes")):
            bad = (_as_set(value) or set()) - set(allowed)
            if bad:
                raise ValueError(f"unknown {axis} value(s) {sorted(bad)}; expected one of {allowed}")
        name_order = [names] if isinstance(names, str) else list(names or ())
        task, size, attributes = _as_set(task), _as_set(size), _as_set(attributes)
        tags, names = _as_set(tags), _as_set(names)
        if names:
            unknown = names - set(self._by_name)
            if unknown:
                raise KeyError(f"no catalog entries named {sorted(unknown)}")

        def keep(e: CatalogEntry) -> bool:
            return ((task is None or e.task in task)
                    and (size is None or e.size in size)
                    and (attributes is None or e.attributes in attributes)
                    and (tags is None or bool(tags & set(e.tags)))
                    and (names is None or e.name in names)
                    and (missing is None or e.has_missing is missing)
                    and (loadable is None or e.loadable is loadable))

        chosen = [e for e in self.entries if keep(e)]
        if name_order:
            rank = {name: i for i, name in enumerate(dict.fromkeys(name_order))}
            chosen.sort(key=lambda e: rank[e.name])
        if n is None:
            return chosen
        if n < 0 or n > len(chosen):
            raise ValueError(f"asked for {n} datasets, but {len(chosen)} match")
        picked = set(random.Random(random_state).sample(range(len(chosen)), n))
        return [e for i, e in enumerate(chosen) if i in picked]

    def parse(self, spec: str, random_state: Optional[int] = None) -> List[CatalogEntry]:
        """Select from a compact comma-separated string, e.g. for a
        demo's ``--datasets`` option: ``"binary,small,medium"``,
        ``"lord"``, ``"categorical,multiclass"``, ``"vote,mushroom"``,
        ``"binary,medium,3"``, ``"all"``. Words are sorted onto their
        axis (task, size, attributes, tag); several words on one axis
        mean *any of*, different axes must all hold. A number picks that
        many of the matches at random (`select`'s `n`, seeded by
        `random_state`). Dataset names are added to whatever the other
        words select (or form the whole selection if there are no other
        words)."""
        words = [w.strip() for w in spec.split(",") if w.strip()]
        if words == ["all"]:
            return self.select()
        axes: Dict[str, List[str]] = {"task": [], "size": [], "attributes": [], "tags": [], "names": []}
        all_tags = {t for e in self.entries for t in e.tags}
        n: Optional[int] = None
        for w in words:
            if w.isdigit():
                if n is not None:
                    raise ValueError(f"more than one number in {spec!r}")
                n = int(w)
            elif w in TASKS:
                axes["task"].append(w)
            elif w in SIZES:
                axes["size"].append(w)
            elif w in ATTRIBUTE_KINDS:
                axes["attributes"].append(w)
            elif w in all_tags:
                axes["tags"].append(w)
            elif w in self._by_name:
                axes["names"].append(w)
            else:
                raise ValueError(f"{w!r} is neither a category, a tag nor a dataset name in the catalog")
        filters = {k: v for k, v in axes.items() if v and k != "names"}
        chosen = (self.select(**filters, n=n, random_state=random_state)
                  if filters or n is not None else [])
        extra = self.select(names=axes["names"], loadable=None) if axes["names"] else []
        seen = {e.name for e in chosen}
        return chosen + [e for e in extra if e.name not in seen]

    def summary(self, **criteria) -> str:
        """A one-line-per-dataset text table: the whole catalog, or the
        entries `select(**criteria)` picks -- e.g.
        ``summary(task="binary", size="small")``. Unlike `select`,
        pointer-only entries are included unless ``loadable=True`` is
        passed."""
        criteria.setdefault("loadable", None)
        entries = self.select(**criteria)
        lines = [f"{'name':40s} {'task':10s} {'size':6s} {'attrs':11s} {'rows':>9s} "
                 f"{'feats':>5s} {'cls':>3s}  tags"]
        for e in entries:
            lines.append(f"{e.name:40s} {e.task or '-':10s} {e.size or '-':6s} {e.attributes or '-':11s} "
                         f"{e.n_instances if e.n_instances is not None else '-':>9} "
                         f"{e.n_features if e.n_features is not None else '-':>5} "
                         f"{e.n_classes if e.n_classes is not None else '-':>3}  {','.join(e.tags)}")
        return "\n".join(lines)
