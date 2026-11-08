"""
pyrulearn: representing and analyzing learned propositional logic rules.

Quick tour
----------
>>> import numpy as np
>>> from pyrulearn import DataSpec, Rule
>>> from pyrulearn.data import HorizontalBitmapRepresentation
>>> X = np.array([[1,0,1],[1,1,0],[0,0,1]], dtype=bool)
>>> y = np.array(["pos","pos","neg"])
>>> ds = DataSpec(["a","b","c"])
>>> rep = HorizontalBitmapRepresentation(ds, X, y)
>>> r = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
>>> r.covers_data(rep)
array([ True,  True, False])

See also: ``pyrulearn.data.attributes`` (typed nominal/numeric/set/hierarchical/
relational attributes and the constraints they imply),
``pyrulearn.data`` (``DataRepresentation``/
``HorizontalBitmapRepresentation`` -- the actual data bound to a ``DataSpec``,
which is pure schema),
``pyrulearn.models`` (the ``RuleModel`` hierarchy -- ``FlatRuleSet``/
``ConceptModel``/``DisjointRuleSet``/``ConceptSet`` (unordered),
``DecisionList``/``ConceptCascade`` (ordered), and composites
(``EnsembleModel``/``PairwiseModel``/``DeepModel``) -- collections of
rules that make predictions together),
``pyrulearn.interfaces.sklearn`` (``RuleSetClassifier``, the scikit-learn-
compatible wrapper around a ``FlatRuleSet`` -- scikit-learn is a hard
dependency of pyrulearn because of this class),
``pyrulearn.combiners`` (``RuleCombiner`` -- how ``RuleSet.predict``
combines several simultaneously-covering rules: ``ListCombiner``
(position), ``CountVoteCombiner`` (plain majority vote),
``HeuristicCombiner``'s ``HeuristicMaxCombiner``/``HeuristicVoteCombiner``
(a ``RuleHeuristic`` scored against each rule's own measured stats), or
``DistributionCombiner``'s ``MicroVoteCombiner``/``MacroVoteCombiner``/
``MicroMaxCombiner``/``MacroMaxCombiner`` (each rule's full per-class
distribution -- ``MacroVoteCombiner`` is the one that matches sklearn's
own soft-voting ensembles)),
``pyrulearn.data.io`` (reading ARFF/CSV into a ``HorizontalBitmapRepresentation``),
``pyrulearn.interfaces`` (``RuleImporter`` and its two shapes --
``ObjectRuleImporter`` for an already-fitted live model object, e.g.
``pyrulearn.interfaces.sklearn.SklearnTreeImporter``;
``StringRuleImporter`` for a string/serialized rule format, e.g.
``PatternStringImporter`` -- sharing one provenance-tagging mechanism
and lookup registry),
``pyrulearn.learners`` (``RuleLearner``'s ``fit(data) ->
RuleModel`` -- ``ExternalRuleLearner`` runs an external algorithm
and converts its output via an existing ``ObjectRuleImporter``, e.g.
``pyrulearn.interfaces.sklearn.SKLRandomForest``; ``NativeRuleLearner``
induces directly, no importer involved),
``pyrulearn.heuristics`` (``RuleHeuristic`` -- pluggable rule-evaluation
heuristics scored from ``RuleStats``, e.g. ``Precision``/``Laplace``/
``MEstimate``/``WRAcc``/``FoilGain``),
``pyrulearn.evaluation`` (measured statistics -- ``RuleStats``,
``ConfusionMatrix``, ``ModelStats``, ``sort_rules`` -- and coverage-space /
refinement-graph plotting).
"""

from .combiners import (
    CountVoteCombiner,
    DistributionCombiner,
    HeuristicCombiner,
    HeuristicMaxCombiner,
    HeuristicVoteCombiner,
    ListCombiner,
    MacroMaxCombiner,
    MacroVoteCombiner,
    MicroMaxCombiner,
    MicroVoteCombiner,
    RuleCombiner,
)
from .evaluation import sort_rules
from .heuristics import (
    Accuracy,
    ChiSquare,
    Correlation,
    Coverage,
    CoveredNegatives,
    CoveredPositives,
    CoverageDifference,
    Entropy,
    FBeta,
    DeltaGain,
    FoilGain,
    GainHeuristic,
    GeneralizedMEstimate,
    GHeuristic,
    Laplace,
    LEF,
    LengthPenalized,
    MinimalLength,
    LikelihoodRatio,
    LinearCost,
    LinearCostRates,
    MEstimate,
    Precision,
    Recall,
    RuleHeuristic,
    RuleStats,
    Support,
    UncoveredNegatives,
    UncoveredPositives,
    WRAcc,
    YoudenJ,
)
from .interfaces import (
    ObjectRuleImporter,
    PatternStringImporter,
    RuleImporter,
    StringRuleImporter,
    get_importer,
    register_importer,
)
from .learners import ExternalRuleLearner, NativeRuleLearner, RuleLearner
from .models import DefaultPrediction, DisjointRuleSet, MajorityClass
from .data import (
    BoundaryCorrection,
    DataRepresentation,
    DataSpec,
    DataSpecBuilder,
    HorizontalBitmapRepresentation,
    NListRepresentation,
    PrePostNListRepresentation,
    SparseDataRepresentation,
    VerticalBitmapRepresentation,
    Discretizer,
    EqualFrequencyDiscretizer,
    EqualWidthDiscretizer,
    FusinterDiscretizer,
    PerClass,
    TreeDiscretizer,
    merge_dataspecs,
)
from .rule import Literal, Rule
from .interfaces.sklearn import RuleSetClassifier

__all__ = [
    "DataSpec", "DataSpecBuilder", "Literal", "Rule",
    "DataRepresentation", "HorizontalBitmapRepresentation", "NListRepresentation",
    "PrePostNListRepresentation", "SparseDataRepresentation", "VerticalBitmapRepresentation",
    "DisjointRuleSet", "merge_dataspecs",
    "Discretizer", "TreeDiscretizer", "FusinterDiscretizer", "EqualFrequencyDiscretizer",
    "EqualWidthDiscretizer", "BoundaryCorrection", "PerClass",
    "RuleSetClassifier", "sort_rules",
    "RuleImporter", "ObjectRuleImporter", "StringRuleImporter", "PatternStringImporter",
    "register_importer", "get_importer",
    "RuleLearner", "ExternalRuleLearner", "NativeRuleLearner",
    "RuleCombiner", "ListCombiner", "CountVoteCombiner",
    "HeuristicCombiner", "HeuristicMaxCombiner", "HeuristicVoteCombiner",
    "DistributionCombiner", "MicroVoteCombiner", "MacroVoteCombiner", "MicroMaxCombiner", "MacroMaxCombiner",
    "DefaultPrediction", "MajorityClass",
    "RuleHeuristic", "GainHeuristic", "RuleStats",
    "CoveredPositives", "CoveredNegatives", "UncoveredPositives", "UncoveredNegatives",
    "Precision", "Recall", "FBeta", "Laplace", "MEstimate",
    "GeneralizedMEstimate", "GHeuristic",
    "WRAcc", "YoudenJ",
    "Accuracy", "CoverageDifference", "Support", "Coverage",
    "LinearCost", "LinearCostRates", "LengthPenalized", "MinimalLength", "FoilGain", "DeltaGain", "Correlation", "Entropy", "LikelihoodRatio", "ChiSquare",
    "LEF",
]
__version__ = "0.6.0"

