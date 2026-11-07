"""
pyrulearn.learners
=====================

`RuleLearner` and its shared machinery -- everything about turning a
`DataRepresentation` into rules via a uniform `fit(data, model=None) ->
RuleModel`, re-exported here from `pyrulearn.learners.base`:
`RuleLearner`, `ExternalRuleLearner`, `NativeRuleLearner`,
`DecomposingLearner` (the `fit(data, model=ConceptSet |
ConceptCascade | PairwiseModel)` multiclass-by-binary-decomposition
switcher), `RelabelingExternalLearner` (external algorithms that can't
fold "the rest" into one label themselves), and the `@produces(...)`
decorator.

Concrete learners live in their own submodules, same
group-by-shared-implementation convention as `pyrulearn.interfaces`:
`pyrulearn.learners.seco` (`SeCo` framework -- `CN2`, `AQR`, `PFoil`,
`PFossil`, `Pypper`), `pyrulearn.learners.pylord` (`PyLORD`),
`pyrulearn.learners.associative` (`CARMiner`, the
`RuleDistiller` mixin shared by the classifiers that consume a rule
pool, and the `CBA` and `CMAR` classifiers built on it),
`pyrulearn.learners.ids` (`IDS`), `pyrulearn.learners.rulefit`
(`RuleFit`, a sparse linear model over a rule pool),
`pyrulearn.learners.boosting` (`Slipper`, `ENDER`, `DenseENDER`, `Boomer`, `DenseBoomer`, `ORB`, `DenseORB`), `pyrulearn.learners.lri`
(`LRI`, `DenseLRI`), `pyrulearn.learners.bagging` (`Bagging`, over any
base learner), `pyrulearn.learners.adaboost` (`AdaBoost`, SAMME over any
base learner), `pyrulearn.learners.cpar` (`CPAR`, `DenseCPAR`), `pyrulearn.learners.opus`
(`Opus`, `OpusTopK`), `pyrulearn.learners.deep` (`DeepLocalSearch`,
`DeepGenetic`, learning `DeepModel`s), and
`pyrulearn.learners.multiclass` (`OneVsRest`/`OrderedOneVsRest`/
`Pairwise` -- thin sugar over `DecomposingLearner`). External-tool
learners (sklearn/wittgenstein/imodels/Weka/LORD) live next to their
importer in `pyrulearn.interfaces` instead, since each pairs a learner
with the `ObjectRuleImporter`/`StringRuleImporter` that reads its output
back -- not duplicated here.
"""

from __future__ import annotations

from .base import (
    DecomposingLearner,
    ExternalRuleLearner,
    NativeRuleLearner,
    RelabelingExternalLearner,
    RuleLearner,
    produces,
)

__all__ = [
    "RuleLearner", "ExternalRuleLearner", "NativeRuleLearner",
    "DecomposingLearner", "RelabelingExternalLearner", "produces",
]
