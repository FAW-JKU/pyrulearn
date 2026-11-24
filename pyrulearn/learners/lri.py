"""
pyrulearn.learners.lri
======================

`LRI` and `DenseLRI` now live in `pyrulearn.learners.boosting`, and
`LRIError` in `pyrulearn.objectives.losses`; this module re-exports them.
"""

from ..objectives.losses import LRIError  # noqa: F401
from .boosting import LRI, DenseLRI  # noqa: F401
