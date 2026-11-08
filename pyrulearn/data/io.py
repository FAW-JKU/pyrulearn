"""
pyrulearn.data.io
=====================

Read external tabular data (ARFF, CSV) into a
`pyrulearn.data.representation.DataRepresentation` via `read_arff`/
`read_csv`, in three ways:

1. **Validate** an already-built `DataSpec` against a file's header
   (`validate_dataspec`) -- report missing attributes, declared/inferred
   type mismatches, and nominal categories the file has that the
   `DataSpec` doesn't know about.
2. **Read with a given `DataSpec`** (pass `dataspec=` to `read_arff`/
   `read_csv`) -- encode the file's raw values against the `DataSpec`'s
   existing `FeatureSpec`s (via `pyrulearn.data.attributes.evaluate_feature`).
3. **Read and infer the `DataSpec`** (`dataspec=None`, the default) --
   build one attribute per column (nominal/numeric) from the file's own
   declared types (ARFF) or a cardinality-based heuristic (CSV), then
   encode the same way.

Both readers build the chosen representation via `encode` -- a
`DataSpec` + a dataframe of raw values, accumulated into a
`scipy.sparse` intermediate, straight into whichever `representation`
is asked for (`read_arff`/`read_csv`'s own `representation=` parameter,
default `pyrulearn.data.representation.default_representation()`, the
first entry of `REPRESENTATION_PREFERENCE_ORDER`,
`VerticalBitmapRepresentation` -- see `demos/representations_report.md`
for how the representations compare). `binarize` is `encode`'s
older, simpler sibling: a `DataSpec` + a dataframe of raw values -> a
dense Boolean matrix, kept as the low-level helper for callers that
want that dense matrix directly or explicitly want
`HorizontalBitmapRepresentation` without going through `encode` -- pass
`representation=HorizontalBitmapRepresentation` to `read_arff`/`read_csv`/
`encode` for that, or call `binarize` yourself and wrap the result.

Numeric columns without pre-given thresholds are discretized by a
`pyrulearn.data.discretization.Discretizer`, passed as ``discretizer=``:
by default `TreeDiscretizer()` (a single-feature decision tree,
`pyrulearn.interfaces.sklearn.tree_thresholds`), or e.g.
`EqualFrequencyDiscretizer`, optionally wrapped in `BoundaryCorrection`.
A supervised discretizer (the default included) needs a `target`
column to discretize a numeric column.

This module requires ``pandas`` (only this module does); ARFF support
additionally needs ``scipy`` (imported lazily, only inside `read_arff`).
`tree_thresholds` (from `pyrulearn.interfaces.sklearn`, needing
scikit-learn) is imported lazily by `TreeDiscretizer`, only when
actually invoked -- reading with an already-complete `DataSpec` (every
numeric attribute already has thresholds) skips it entirely, even though scikit-learn
itself is a hard dependency of `pyrulearn` overall (via
`pyrulearn.interfaces.sklearn.RuleSetClassifier`).

Missing raw values are handled per `pyrulearn.data.attributes.MissingStrategy`
(see `binarize`) -- resolved from an explicit `missing_strategy=`
argument, else the target `DataSpec`'s own `missing_strategy`, else
`DataSpec.DEFAULT_MISSING_STRATEGY` (``NEVER_COVERS``). Set-valued,
hierarchical, and relational attributes are out of scope for *inference*
(task 3): ARFF/CSV headers have no way to declare them, so `build_dataspec`
never produces them, and relational features can't be evaluated at all
(see `evaluate_feature`) -- only a `dataspec=` built by hand with
nominal/numeric/boolean attributes is supported end to end for those.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
import pandas as pd

from .attributes import AttributeType, FeatureSpec, MissingStrategy, evaluate_feature, is_numeric_domain
from .spec import DataSpec, DataSpecBuilder
from .discretization import DEFAULT_MAX_INTERVALS, Discretizer, TreeDiscretizer  # noqa: F401 (re-exported)
from .representation import HorizontalBitmapRepresentation, DataRepresentation, default_representation

# Default cap on the distinct values of a numeric column that is still
# inferred ORDINAL rather than NUMERIC (`max_ordinal_values` below) --
# independent of the discretizer, so that varying the discretization
# doesn't change which columns are ordinal.
DEFAULT_MAX_ORDINAL_VALUES = 10


# -- column -> Attribute inference -------------------------------------------

def _infer_csv_column_type(
    series: pd.Series,
    max_ordinal_values: int = DEFAULT_MAX_ORDINAL_VALUES,
    categorical_min_ratio: float = 0.5,
) -> str:
    """Heuristic type guess for one CSV column, since CSV carries no
    declared types: 'numeric' if the dtype is numeric (and not boolean)
    AND cardinality looks continuous -- more than `max_ordinal_values`
    distinct values (an absolute floor, for large datasets where even a
    small fraction of rows being distinct is still plenty of values), OR
    more than `categorical_min_ratio` of rows are distinct (a relative
    floor, so this still works on small datasets where the absolute
    count is never going to be large). A numeric dtype with only a
    handful of distinct values (a count, a rating, an encoded category)
    is 'ordinal' from three values on, keeping their order and testing
    each value on its own (`DataSpecBuilder.add_ordinal`); with two
    values or fewer it is 'nominal' (two values make a BINARY
    attribute). A non-numeric (or boolean) dtype is 'nominal'.
    """
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "nominal"
    n = len(series)
    n_unique = series.nunique(dropna=True)
    if n_unique > max_ordinal_values or n_unique > categorical_min_ratio * n:
        return "numeric"
    return "ordinal" if n_unique >= 3 else "nominal"


def build_dataspec(
    df: pd.DataFrame,
    target: Optional[str] = None,
    arff_types: Optional[Dict[str, str]] = None,
    discretizer: Optional[Discretizer] = None,
    include_negations: bool = True,
    negation_overrides: Optional[Dict[str, bool]] = None,
    domains: Optional[Dict[str, Sequence[Any]]] = None,
    skip_unusable: bool = False,
    max_ordinal_values: int = DEFAULT_MAX_ORDINAL_VALUES,
) -> DataSpecBuilder:
    """Infer one attribute per (non-target) column of `df` and return the
    `DataSpecBuilder` that generated them -- the shared core of "read and
    infer the DataSpec" (task 3) for both `read_arff` and `read_csv`.

    `arff_types` (from `read_arff`'s ARFF metadata) gives declared types
    per column ('numeric'/'nominal'/'ordinal'/...); columns missing from
    it (the CSV case) fall back to `_infer_csv_column_type`'s heuristic,
    which makes a numeric column with three to `max_ordinal_values`
    distinct values (default 10) ORDINAL (its sorted observed values,
    `DataSpecBuilder.add_ordinal`). Numeric
    columns are discretized by `discretizer` (a
    `pyrulearn.data.discretization.Discretizer`; default
    `TreeDiscretizer()`, a decision tree against `df[target]` with at most
    `DEFAULT_MAX_INTERVALS` intervals) -- a supervised one requires
    `target`.

    A nominal column's values are its declared ones in `domains` (e.g.
    an ARFF header's value list), else the non-missing values observed
    in `df`. Exactly two values make it a BINARY attribute
    (`DataSpecBuilder.add_binary`: ``x=a``/``x=b``, each the other's
    negation, no ``!=`` columns); otherwise it's NOMINAL. A column of
    dtype ``bool`` is a BOOLEAN attribute (`DataSpecBuilder.add_boolean`:
    ``x``, and ``not x`` with negations). A column named in
    ``df.attrs["relational"]`` (``{column: {"sources": [...],
    "expression": "..."}}``, e.g. from `pyrulearn.datasets.krk_illegal`)
    is a RELATIONAL attribute instead (`DataSpecBuilder.add_relational`,
    negated like the others): a precomputed Boolean feature derived from
    the `sources` columns. pandas keeps `attrs` through row selections,
    so a cross-validation fold carries it.

    `include_negations` (default True) is the builder-wide default for
    whether each attribute also gets explicit negation features
    (``x!=v``, ``not f``); `negation_overrides` (``{column: bool}``)
    overrides it per column. Neither affects a BINARY or NUMERIC
    attribute: a numeric attribute always gets ``x>=t`` and ``x<t``.

    A numeric column the discretization finds no split for, or a nominal
    column with no observed value, raises `ValueError` -- unless
    `skip_unusable` is set, which leaves such columns out instead (for
    automated runs such as cross-validation, where a column can be
    constant or entirely missing within one training fold).
    """
    overrides = negation_overrides or {}
    discretizer = discretizer if discretizer is not None else TreeDiscretizer()
    builder = DataSpecBuilder(negation=include_negations)
    y = df[target].to_numpy() if target is not None else None

    relational = df.attrs.get("relational", {})
    for col in df.columns:
        if col == target:
            continue
        declared = (arff_types or {}).get(col)
        negation = overrides.get(col)
        if col in relational:
            rel = relational[col]
            builder.add_relational(col, list(rel.get("sources", ())), expression=rel.get("expression"),
                                   negation=include_negations if negation is None else negation)
            continue
        kind = declared if declared is not None else _infer_csv_column_type(df[col], max_ordinal_values)

        if declared is None and pd.api.types.is_bool_dtype(df[col]):
            builder.add_boolean(col, negation=negation)
        elif kind in ("nominal", "string"):
            declared_domain = (domains or {}).get(col)
            domain = (list(declared_domain) if declared_domain is not None
                      else sorted(df[col].dropna().unique().tolist()))
            if not domain and skip_unusable:
                continue
            if len(domain) == 2:
                builder.add_binary(col, domain)
            else:
                builder.add_nominal(col, domain, negation=negation)
        elif kind == "ordinal":
            values = sorted(df[col].dropna().unique().tolist())
            if not values and skip_unusable:
                continue
            builder.add_ordinal(col, values, negation=negation)
        elif kind == "numeric":
            if y is None and discretizer.supervised:
                raise ValueError(
                    f"Cannot auto-discretize numeric column {col!r} with the supervised "
                    f"{discretizer!r} without a target column -- pass target=, an "
                    "unsupervised discretizer, or thresholds via an explicit DataSpec."
                )
            thresholds = discretizer.thresholds(df[col].to_numpy(dtype=float), y)
            if not thresholds:
                if skip_unusable:
                    continue
                raise ValueError(
                    f"{discretizer!r} found no threshold for numeric column {col!r}; "
                    "provide thresholds explicitly via a DataSpec if this attribute "
                    "should still be included."
                )
            builder.add_numeric(col, thresholds)
        else:
            raise ValueError(f"Unsupported column type {kind!r} for column {col!r}")

    return builder


# -- validation (task 1) ------------------------------------------------------

def validate_dataspec(
    dataspec: DataSpec,
    df: pd.DataFrame,
    arff_types: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Compare `dataspec`'s typed attributes against `df`'s columns.
    Returns a list of human-readable problem descriptions (empty if
    compatible): missing attributes, declared/inferred type mismatches,
    and (for nominal attributes) categories present in `df` but outside
    the attribute's known domain.
    """
    problems: List[str] = []
    for name, attr in dataspec.attributes.items():
        if name not in df.columns:
            problems.append(f"attribute {name!r} has no matching column in the data")
            continue
        declared = (arff_types or {}).get(name)
        kind = declared if declared is not None else _infer_csv_column_type(df[name])
        value_typed = attr.type in (AttributeType.NOMINAL, AttributeType.BINARY)
        if value_typed and kind not in ("nominal", "string"):
            problems.append(f"attribute {name!r} is {attr.type.name} but column {name!r} looks {kind!r}")
        elif attr.type == AttributeType.NUMERIC and kind not in ("numeric", "ordinal"):
            problems.append(f"attribute {name!r} is NUMERIC but column {name!r} looks {kind!r}")
        elif attr.type == AttributeType.ORDINAL:
            numeric_values = is_numeric_domain(attr.domain)
            if numeric_values and kind not in ("numeric", "ordinal"):
                problems.append(f"attribute {name!r} is ORDINAL over numbers but column {name!r} looks {kind!r}")
            elif not numeric_values and kind not in ("nominal", "string"):
                problems.append(f"attribute {name!r} is ORDINAL but column {name!r} looks {kind!r}")
            value_typed = not numeric_values  # numbers outside the domain still compare
        if value_typed and attr.domain is not None:
            unknown = sorted(set(df[name].dropna().unique().tolist()) - set(attr.domain))
            if unknown:
                problems.append(
                    f"attribute {name!r} has values in the data outside its known domain: {unknown}"
                )
    return problems


# -- binarization (tasks 2 & 3 share this) ------------------------------------

def _prepared_column(raw: pd.Series, attr, specs: List[FeatureSpec], resolved: MissingStrategy,
                     rng, source_col: str):
    """One source column made ready for `evaluate_feature`, the shared
    first step of `binarize` and `encode`: declared `missing_values` set
    to missing, NUMERIC values (and ORDINAL ones over numbers) coerced to
    numbers, ORDINAL ones over other values replaced by their position in
    the declared order (a value outside it counts as missing) -- with
    `specs` rewritten to compare positions too -- and, for `MAJORITY`/
    `RANDOM`, missing values imputed. Returns ``(values, missing_mask,
    specs)``, the mask taken after imputation."""
    if attr is not None and attr.missing_values:
        missing_set = set(attr.missing_values)
        raw = raw.apply(lambda v: None if v in missing_set else v)
    ordinal = attr is not None and attr.type == AttributeType.ORDINAL
    numeric = attr is not None and (attr.type == AttributeType.NUMERIC
                                    or (ordinal and is_numeric_domain(attr.domain)))
    if numeric:
        raw = pd.to_numeric(raw, errors="coerce")
    elif ordinal:
        position = {v: p for p, v in enumerate(attr.domain)}
        raw = raw.map(lambda v: position.get(v, np.nan)).astype(float)
        specs = [s if s.value not in position else
                 FeatureSpec(s.index, s.name, attribute=s.attribute, op=s.op, value=position[s.value])
                 for s in specs]
    missing_mask = raw.isna()

    if missing_mask.any() and resolved in (MissingStrategy.MAJORITY, MissingStrategy.RANDOM):
        known = raw.dropna()
        if known.empty:
            raise ValueError(f"Column {source_col!r} has no non-missing values to impute from")
        if resolved == MissingStrategy.MAJORITY:
            if ordinal:  # the lower median, so always one of the values
                ordered = np.sort(known.to_numpy(dtype=float))
                fill = ordered[(len(ordered) - 1) // 2]
            elif numeric:
                fill = float(known.astype(float).median())
            else:
                fill = known.mode().iloc[0]
            raw = raw.fillna(fill)
        else:  # RANDOM
            raw = raw.copy()
            idx = raw.index[missing_mask]
            raw.loc[idx] = rng.choice(known.to_numpy(), size=len(idx))
        missing_mask = raw.isna()  # now all False
    return raw, missing_mask, specs


def binarize(
    dataspec: DataSpec,
    df: pd.DataFrame,
    missing_strategy: Optional[MissingStrategy] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Evaluate every entry of `dataspec.feature_specs` against `df`'s
    raw values (via `pyrulearn.data.attributes.evaluate_feature`), in
    `feature_names` order, producing the Boolean matrix ready for
    `pyrulearn.data.HorizontalBitmapRepresentation`.

    `missing_strategy` resolves (explicit argument > `dataspec`'s own
    `missing_strategy` > `DataSpec.DEFAULT_MISSING_STRATEGY`) to one of
    `pyrulearn.data.attributes.MissingStrategy`, applied per source column
    (an attribute's `missing_values`, if declared, are also recognized
    as missing alongside real `None`/NaN):

    - ``NEVER_COVERS`` (default): every feature derived from a missing
      value is left False.
    - ``MAJORITY``/``RANDOM``: the raw value is imputed (column median
      for a NUMERIC attribute, the lower median value for an ORDINAL
      one, else mode; or a uniformly random other non-missing value --
      `random_state` seeds this one) before evaluating normally.
    - ``SEPARATE``: routes into the attribute's dedicated missing
      feature (see `DataSpecBuilder.add_nominal`/`add_numeric`/
      `add_ordinal`'s `missing_name=`) -- raises if the attribute has
      none declared.

    An ORDINAL attribute's values compare by their position in its
    declared order, or numerically if they are all numbers; a value
    outside a non-numeric order counts as missing.
    """
    resolved = missing_strategy or dataspec.missing_strategy or DataSpec.DEFAULT_MISSING_STRATEGY
    n = len(df)
    X = np.zeros((n, dataspec.n_features), dtype=bool)
    rng = np.random.default_rng(random_state) if resolved == MissingStrategy.RANDOM else None

    specs_by_col: Dict[str, List] = {}
    for spec in dataspec.feature_specs:
        source_col = spec.attribute if spec.attribute is not None else spec.name
        specs_by_col.setdefault(source_col, []).append(spec)

    for source_col, specs in specs_by_col.items():
        if source_col not in df.columns:
            names = [s.name for s in specs]
            raise ValueError(f"Column {source_col!r} (for feature(s) {names}) not found in data")
        attr = dataspec.attributes.get(source_col)
        raw, missing_mask, specs = _prepared_column(df[source_col], attr, specs, resolved, rng, source_col)

        for spec in specs:
            results = raw.apply(lambda v: evaluate_feature(spec, v))
            # unresolved (still-missing) results never cover, regardless
            # of which feature/polarity is being evaluated -- exactly
            # NEVER_COVERS's semantics, and also SEPARATE's default for
            # every feature *except* the dedicated missing feature,
            # which the block below corrects to True.
            X[:, spec.index] = results.to_numpy(dtype=object) == True

        if resolved == MissingStrategy.SEPARATE and missing_mask.any():
            missing_spec = next(
                (s for s in specs if attr is not None and attr.missing_name is not None
                 and s.op == "==" and s.value == attr.missing_name),
                None,
            )
            if missing_spec is None:
                raise ValueError(
                    f"missing_strategy=SEPARATE selected, but attribute {source_col!r} has no "
                    "declared missing-value feature -- pass missing_name= to add_nominal/add_numeric "
                    "when building this attribute, or choose a different missing_strategy."
                )
            X[missing_mask.to_numpy(), missing_spec.index] = True

    return X


def encode(
    dataspec: DataSpec,
    df: pd.DataFrame,
    y: Optional[Any] = None,
    representation: Optional[Type[DataRepresentation]] = None,
    missing_strategy: Optional[MissingStrategy] = None,
    random_state: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> DataRepresentation:
    """Like `binarize`, but builds `representation` (default
    `pyrulearn.data.representation.default_representation()`, the first
    entry of `REPRESENTATION_PREFERENCE_ORDER` -- `VerticalBitmapRepresentation`)
    directly from a `scipy.sparse` intermediate instead of
    writing into a dense `(n_samples, n_features)` array first.

    `binarize` always allocates that array, even when the representation
    actually wanted is `NListRepresentation` or `SparseDataRepresentation`
    -- both of which build their own, more compact storage *from* it,
    so the dense array was only ever a transient step, not something
    either representation keeps. For data where `n_samples *
    n_features` is the whole reason to prefer one of those
    representations, paying for it anyway, even transiently, defeats
    the point (`ROADMAP.md`'s 0.3.0 plan). This function evaluates every
    `dataspec.feature_specs` entry exactly as `binarize` does (same
    `missing_strategy` resolution, same per-column grouping, same
    `evaluate_feature` calls) but accumulates each derived feature's
    *true row indices* instead of writing into a dense column, then
    builds one `scipy.sparse.csc_matrix` from all of them at once --
    memory proportional to the number of true entries, never to
    `n_samples * n_features`. `HorizontalBitmapRepresentation.__init__` and
    `SparseDataRepresentation.__init__` already accept a `scipy.sparse`
    matrix directly; `NListRepresentation.__init__` now does too,
    reading rows from its CSR structure instead of `np.flatnonzero` on a
    dense row -- so whichever `representation` is requested, this
    sparse intermediate is the only matrix ever built. Choosing
    `HorizontalBitmapRepresentation` still ends up with a dense array
    eventually (that's its own storage), just built by its constructor
    from this sparse intermediate's `.toarray()`, not filled in here.

    `y`/`weights`, if given, are passed straight to `representation`'s
    constructor alongside the encoded matrix -- the same one-line
    `SomeRepresentation(spec, binarize(spec, df), df[target].to_numpy())`
    every demo currently writes by hand, just with the matrix building
    and the representation choice combined into one call.
    """
    from scipy import sparse

    if representation is None:
        representation = default_representation()
    resolved = missing_strategy or dataspec.missing_strategy or DataSpec.DEFAULT_MISSING_STRATEGY
    n = len(df)
    rng = np.random.default_rng(random_state) if resolved == MissingStrategy.RANDOM else None

    row_chunks: List[np.ndarray] = []
    col_chunks: List[np.ndarray] = []

    def _add(rows: np.ndarray, col: int) -> None:
        if rows.size:
            row_chunks.append(rows)
            col_chunks.append(np.full(rows.size, col, dtype=np.int64))

    specs_by_col: Dict[str, List] = {}
    for spec in dataspec.feature_specs:
        source_col = spec.attribute if spec.attribute is not None else spec.name
        specs_by_col.setdefault(source_col, []).append(spec)

    for source_col, specs in specs_by_col.items():
        if source_col not in df.columns:
            names = [s.name for s in specs]
            raise ValueError(f"Column {source_col!r} (for feature(s) {names}) not found in data")
        attr = dataspec.attributes.get(source_col)
        raw, missing_mask, specs = _prepared_column(df[source_col], attr, specs, resolved, rng, source_col)

        for spec in specs:
            results = raw.apply(lambda v: evaluate_feature(spec, v))
            _add(np.flatnonzero(results.to_numpy(dtype=object) == True), spec.index)

        if resolved == MissingStrategy.SEPARATE and missing_mask.any():
            missing_spec = next(
                (s for s in specs if attr is not None and attr.missing_name is not None
                 and s.op == "==" and s.value == attr.missing_name),
                None,
            )
            if missing_spec is None:
                raise ValueError(
                    f"missing_strategy=SEPARATE selected, but attribute {source_col!r} has no "
                    "declared missing-value feature -- pass missing_name= to add_nominal/add_numeric "
                    "when building this attribute, or choose a different missing_strategy."
                )
            _add(np.flatnonzero(missing_mask.to_numpy()), missing_spec.index)

    rows = np.concatenate(row_chunks) if row_chunks else np.empty(0, dtype=np.int64)
    cols = np.concatenate(col_chunks) if col_chunks else np.empty(0, dtype=np.int64)
    data = np.ones(rows.shape[0], dtype=bool)
    X = sparse.coo_matrix((data, (rows, cols)), shape=(n, dataspec.n_features), dtype=bool).tocsc()
    return representation(dataspec, X, y, weights=weights)


# -- readers -------------------------------------------------------------------

def read_arff(
    source,
    dataspec: Optional[DataSpec] = None,
    target: Optional[str] = None,
    discretizer: Optional[Discretizer] = None,
    strict: bool = True,
    missing_strategy: Optional[MissingStrategy] = None,
    random_state: Optional[int] = None,
    include_negations: bool = True,
    max_ordinal_values: int = DEFAULT_MAX_ORDINAL_VALUES,
    representation: Optional[Type[DataRepresentation]] = None,
) -> DataRepresentation:
    """Read an ARFF file (path or file-like object) into a
    `DataRepresentation` -- `representation` (default
    `default_representation()`, `VerticalBitmapRepresentation`, via `encode`; pass
    `HorizontalBitmapRepresentation` for the dense matrix instead).

    If `dataspec` is given, the file's header is validated against it
    (see `validate_dataspec`) before encoding -- `strict=True`
    (default) raises on any mismatch found. If `dataspec` is omitted,
    one is inferred from ARFF's own declared attribute types
    (`build_dataspec`, numeric columns discretized by `discretizer`). `missing_strategy`/`random_state` are passed
    through to `encode`.

    Nominal *data* values wrapped in quotes (``'like this'``/``"like
    this"``, needed for a category containing whitespace or another
    ARFF-special character -- see `write_arff`'s `_arff_quote`) are
    unquoted here explicitly: scipy's own ARFF reader (this function's
    dependency) correctly strips quotes from a *declared* attribute's
    ``{...}`` value list, but leaves them on *data-row* values verbatim,
    which then fails to match the (unquoted) declared value -- confirmed
    directly, not assumed, while tracking down why a file `write_arff`
    wrote for Weka didn't round-trip through this function.
    """
    from scipy.io import arff

    data, meta = arff.loadarff(source)
    df = pd.DataFrame(data)
    arff_types = {name: meta[name][0] for name in meta.names()}
    # the header's declared value lists: complete by ARFF's own rules, so
    # an attribute declared with two values is BINARY (see build_dataspec)
    domains = {name: list(meta[name][1]) for name in meta.names() if meta[name][0] == "nominal"}
    for col in df.columns:
        if arff_types.get(col) == "nominal" and df[col].dtype == object:
            # scipy returns ARFF's missing marker, an unquoted ?, verbatim --
            # map it to None so it's handled as missing, not as a category
            df[col] = df[col].apply(
                lambda v: (None if v == b"?" else _arff_unquote(v.decode("utf-8")))
                if isinstance(v, bytes) else v
            )

    if dataspec is not None:
        problems = validate_dataspec(dataspec, df, arff_types=arff_types)
        if problems and strict:
            raise ValueError("DataSpec does not match the ARFF header:\n" + "\n".join(problems))
    else:
        dataspec = build_dataspec(
            df, target=target, arff_types=arff_types, discretizer=discretizer,
            include_negations=include_negations, max_ordinal_values=max_ordinal_values, domains=domains,
        ).build()

    y = df[target].to_numpy() if target is not None else None
    return encode(dataspec, df, y, representation=representation,
                 missing_strategy=missing_strategy, random_state=random_state)


def read_csv(
    source,
    dataspec: Optional[DataSpec] = None,
    target: Optional[str] = None,
    discretizer: Optional[Discretizer] = None,
    strict: bool = True,
    missing_strategy: Optional[MissingStrategy] = None,
    random_state: Optional[int] = None,
    include_negations: bool = True,
    max_ordinal_values: int = DEFAULT_MAX_ORDINAL_VALUES,
    representation: Optional[Type[DataRepresentation]] = None,
    **read_csv_kwargs,
) -> DataRepresentation:
    """Read a CSV file (path or file-like object) into a
    `DataRepresentation` -- `representation` (default
    `default_representation()`, `VerticalBitmapRepresentation`, via `encode`; pass
    `HorizontalBitmapRepresentation` for the dense matrix instead). CSV carries no declared column types,
    so when `dataspec` is omitted, types are guessed heuristically
    (`_infer_csv_column_type`); pass an explicit `dataspec` whenever that
    guess isn't good enough; numeric columns are discretized by
    `discretizer` (see `build_dataspec`). `missing_strategy`/`random_state` are
    passed through to `encode`. `**read_csv_kwargs` are passed through
    to `pandas.read_csv`.
    """
    df = pd.read_csv(source, **read_csv_kwargs)

    if dataspec is not None:
        problems = validate_dataspec(dataspec, df)
        if problems and strict:
            raise ValueError("DataSpec does not match the CSV header:\n" + "\n".join(problems))
    else:
        dataspec = build_dataspec(
            df, target=target, discretizer=discretizer,
            include_negations=include_negations, max_ordinal_values=max_ordinal_values,
        ).build()

    y = df[target].to_numpy() if target is not None else None
    return encode(dataspec, df, y, representation=representation,
                 missing_strategy=missing_strategy, random_state=random_state)


_ARFF_NEEDS_QUOTE = re.compile(r"[\s,{}%'\"]")


def _arff_quote(value: str) -> str:
    """Quote `value` for ARFF if it contains whitespace or any of ARFF's
    own special characters (``,``, ``{``, ``}``, ``%``, quotes) --
    confirmed directly against Weka's own `ArffLoader`, not assumed: an
    *unquoted* nominal category containing a space (e.g. ``"no
    checking"``) made it fail with a wholly unrelated-looking error
    (``"nominal attribute (credit_history) cannot have duplicate labels
    (existing)"`` -- its tokenizer doesn't stop an unquoted value at a
    comma the way a plain CSV reader would). scipy's own ARFF reader
    (`read_arff`'s own dependency) is more lenient and accepts the
    unquoted form fine, which is exactly why round-tripping through
    `read_arff` alone didn't catch this.
    """
    if not _ARFF_NEEDS_QUOTE.search(value):
        return value
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


_ARFF_QUOTED_RE = re.compile(r"^(['\"])(.*)\1$", re.DOTALL)


def _arff_unquote(value: str) -> str:
    """Inverse of `_arff_quote`: strip a single matching layer of
    ``'...'``/``"..."`` quoting, if present, and reverse the
    backslash-escaping `_arff_quote` applies. A value that was never
    quoted (the common case) passes through unchanged.
    """
    m = _ARFF_QUOTED_RE.match(value)
    if not m:
        return value
    quote, inner = m.group(1), m.group(2)
    return inner.replace(f"\\{quote}", quote).replace("\\\\", "\\")


def write_arff(
    df: pd.DataFrame,
    target: str,
    path,
    arff_types: Optional[Dict[str, str]] = None,
    feature_names: Optional[Sequence[str]] = None,
    relation: str = "pyrulearn",
) -> None:
    """Write `df` (feature columns plus `target`) to an ARFF file at
    `path` (a path or a file-like object) -- the write-side counterpart
    to `read_arff`.

    `arff_types` maps column name -> ``"nominal"``/``"numeric"``, the
    same convention `build_dataspec` takes; a column not listed there
    defaults to nominal if its dtype isn't numeric, numeric otherwise.

    `feature_names`, if given, overrides the ``@attribute`` names
    actually written for the feature columns -- by *position*, not by
    matching `df`'s own column names. The motivating case: writing out
    an already-binarized `DataSpec`'s own feature values (e.g.
    `binarize(ds, df)`) under that `DataSpec`'s own derived names like
    ``"age>=30"``/``"color=red"`` would corrupt any downstream
    *text*-based rule parser that uses those same characters (``>=``,
    ``<=``, ``=``) to recognize where a condition's operator is --
    `pyrulearn.interfaces.weka`'s importers all work this way, so
    feeding Weka an attribute literally named ``"age>=30"`` and then
    trying to re-parse its own printed condition text back would
    misread the embedded operator as part of the split. Passing safe
    placeholder names (``["f0", "f1", ...]``) sidesteps that; `df`'s own
    real values are written unchanged either way, only the declared
    attribute names differ.

    Nominal values containing whitespace or another ARFF-special
    character are quoted (`_arff_quote`) -- required by Weka's own
    `ArffLoader` for both the declared ``{...}`` value list *and* data
    rows (confirmed directly, not assumed). The same quoting applies to
    `relation`, `target`, and every `@attribute` name (`name`/`feature_names`
    or `df`'s own column names) -- not just values: OpenML's `mushroom`
    dataset, for instance, really does name a column ``"bruises%3F"``
    (its own percent-encoding of ``"bruises?"``), and ARFF treats ``%``
    as a comment marker *anywhere* on a line, not just at line start, so
    an unquoted ``@attribute bruises%3F {f,t}`` silently truncates to
    ``@attribute bruises`` with the rest read as a comment -- Weka then
    fails with the same unhelpfully generic "Can't open file" error
    described above, confirmed directly against a real `.arff` this
    produced before this quoting was added. One real limitation this
    creates, worth knowing rather than discovering by surprise: scipy's
    own ARFF reader (`read_arff`'s dependency) validates a data row's
    nominal values against the declared domain *during* parsing, before
    any unquoting could apply, and raises immediately on a quoted value
    -- so a file with quote-needing categories written here is readable
    by Weka but will *not* round-trip through `read_arff`. This isn't a
    pyrulearn bug to route around silently; if you need one file
    readable both ways, avoid nominal categories that need quoting (no
    whitespace/commas/braces), or keep evaluation in Python against the
    original `DataFrame` rather than reading the written file back.
    """
    feature_cols = [c for c in df.columns if c != target]
    names = list(feature_names) if feature_names is not None else feature_cols
    if len(names) != len(feature_cols):
        raise ValueError(
            f"feature_names has {len(names)} entries but df has {len(feature_cols)} feature columns"
        )

    types = arff_types or {}

    def _is_nominal(col: str) -> bool:
        t = types.get(col)
        return t == "nominal" if t is not None else not pd.api.types.is_numeric_dtype(df[col])

    classes = sorted(df[target].dropna().astype(str).unique().tolist())

    lines = [f"@relation {_arff_quote(relation)}", ""]
    for col, name in zip(feature_cols, names):
        if _is_nominal(col):
            cats = sorted(df[col].dropna().astype(str).unique().tolist())
            lines.append(f"@attribute {_arff_quote(name)} {{{','.join(_arff_quote(c) for c in cats)}}}")
        else:
            lines.append(f"@attribute {_arff_quote(name)} numeric")
    lines.append(f"@attribute {_arff_quote(target)} {{{','.join(_arff_quote(c) for c in classes)}}}")
    lines.append("")
    lines.append("@data")

    cols = feature_cols + [target]
    data = df[cols].astype(str)
    data = data.where(df[cols].notna(), "?")
    data = data.map(_arff_quote)
    rows = data.agg(",".join, axis=1).tolist()

    body = "\n".join(lines) + "\n" + "\n".join(rows) + "\n"
    if hasattr(path, "write"):
        path.write(body)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)


def write_csv(
    df: pd.DataFrame,
    target: str,
    path,
    feature_names: Optional[Sequence[str]] = None,
    **to_csv_kwargs,
) -> None:
    """Write `df` (feature columns plus `target`) to a CSV file at
    `path` (a path or a file-like object) -- the write-side counterpart
    to `read_csv`.

    `feature_names`, if given, overrides the header names actually
    written for the feature columns -- by *position*, not by matching
    `df`'s own column names -- the same convention `write_arff`'s
    `feature_names=` uses (see its docstring for the motivating case:
    avoiding column names that would corrupt a downstream text-based
    rule parser). `target`'s own column header is always kept as-is.
    CSV carries no declared attribute types the way ARFF does, so
    there's no `arff_types=` equivalent here.

    `**to_csv_kwargs` are passed through to `DataFrame.to_csv`;
    `index=False` is the default here (matching `read_csv`'s own
    expectation that the file has no index column), overridable like
    any other kwarg.
    """
    feature_cols = [c for c in df.columns if c != target]
    names = list(feature_names) if feature_names is not None else feature_cols
    if len(names) != len(feature_cols):
        raise ValueError(
            f"feature_names has {len(names)} entries but df has {len(feature_cols)} feature columns"
        )

    out = df[feature_cols + [target]].copy()
    out.columns = names + [target]
    to_csv_kwargs.setdefault("index", False)
    out.to_csv(path, **to_csv_kwargs)
