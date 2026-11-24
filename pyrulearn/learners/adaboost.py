"""
pyrulearn.learners.adaboost
===========================

`AdaBoost` (and `samme_weight`) now live in `pyrulearn.learners.boosting`;
this module re-exports them.
"""

from .boosting import AdaBoost, samme_weight  # noqa: F401
