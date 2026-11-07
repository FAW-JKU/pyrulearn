"""
pyrulearn.data
=================

Everything about representing data, split by concern:

- `pyrulearn.data.attributes` -- the building blocks of a feature space:
  typed attributes (`AttributeType`, `Attribute`), the Boolean features
  they derive (`FeatureSpec`, `evaluate_feature`), the constraints among
  those features, and `MissingStrategy`.
- `pyrulearn.data.spec` -- `DataSpec`, a pure feature-space specification
  (names, typed attributes, constraints, missing-value policy), no data.
- `pyrulearn.data.representation` -- `DataRepresentation` (ABC) and its
  three base representations, the actual data bound to a `DataSpec`:
  `HorizontalBitmapRepresentation` (bit-packed matrix), `SparseDataRepresentation`
  (scipy CSR/CSC) and `NListRepresentation` (PPC-tree / N-list index;
  `PrePostNListRepresentation` is the same tree without the path bitmasks,
  for less memory). All implement the
  same `coverage(rule)` interface, so every rule learner runs on any of them.
- `pyrulearn.data.discretization` -- discretizers, turning a numeric
  column into thresholds: `TreeDiscretizer` (the default),
  `FusinterDiscretizer`, `EqualFrequencyDiscretizer`, `EqualWidthDiscretizer`,
  and the `BoundaryCorrection` and `PerClass` wrappers.
- `pyrulearn.data.io` -- `read_arff`/`read_csv`/`write_arff`/`write_csv`/
  `binarize`/`build_dataspec`, reading/writing a `HorizontalBitmapRepresentation`
  from/to ARFF or CSV. Requires `pandas` -- **not** re-exported here, so a
  bare `import pyrulearn` (or `import pyrulearn.data`) still doesn't need
  it; `import pyrulearn.data.io` (or the functions it's actually used
  through) does.

Both `spec` and `representation` are re-exported here, so
``from pyrulearn.data import DataSpec, HorizontalBitmapRepresentation`` keeps
working as a single import.
"""

from __future__ import annotations

from .spec import DataSpec, DataSpecBuilder, merge_dataspecs
from .discretization import (BoundaryCorrection, Discretizer, EqualFrequencyDiscretizer, EqualWidthDiscretizer,
                             FusinterDiscretizer, PerClass, TreeDiscretizer)
from .representation import (
    DataRepresentation,
    HorizontalBitmapRepresentation,
    NListRepresentation,
    PrePostNListRepresentation,
    REPRESENTATION_PREFERENCE_ORDER,
    SparseDataRepresentation,
    VerticalBitmapRepresentation,
    default_representation,
)

__all__ = [
    "DataSpec", "DataSpecBuilder", "merge_dataspecs",
    "DataRepresentation", "HorizontalBitmapRepresentation",
    "NListRepresentation", "PrePostNListRepresentation", "SparseDataRepresentation", "VerticalBitmapRepresentation",
    "REPRESENTATION_PREFERENCE_ORDER", "default_representation",
    "Discretizer", "TreeDiscretizer", "FusinterDiscretizer", "EqualFrequencyDiscretizer", "EqualWidthDiscretizer",
    "BoundaryCorrection", "PerClass",
]

