import numpy as np
import pytest

from pyrulearn.data import HorizontalBitmapRepresentation
from pyrulearn.data import DataSpec
from pyrulearn.learners.seco import CN2
from pyrulearn.models import (
    ConceptCascade, ConceptModel, ConceptSet, DecisionList, FlatRuleSet,
    PairwiseModel, RuleModel, SingleRule,
)
from pyrulearn.interfaces.sklearn import SKLDecisionTree

# 3 cleanly separable classes: class k <-> feature k
_ds = DataSpec(["f0", "f1", "f2"])
_rng = np.random.default_rng(0)
_lbl = _rng.integers(0, 3, size=180)
_X = np.zeros((180, 3), dtype=bool)
_X[np.arange(180), _lbl] = True
_y = np.array(["a", "b", "c"], dtype=object)[_lbl]
DATA = HorizontalBitmapRepresentation(_ds, _X, _y)


def test_fit_model_switch_produces_the_requested_type():
    cn2 = CN2()
    assert type(cn2.fit(DATA, model=ConceptSet)) is ConceptSet
    assert type(cn2.fit(DATA, model=ConceptCascade)) is ConceptCascade
    assert type(cn2.fit(DATA, model=PairwiseModel)) is PairwiseModel
    assert type(cn2.fit(DATA, model=FlatRuleSet)) is FlatRuleSet     # seed covering
    cm = cn2.fit(DATA, model=ConceptModel, label="a")
    assert type(cm) is ConceptModel and cm.label == "a"
    assert set(np.unique(cm.predict(DATA))) <= {"a", "b", "c"}       # covered -> a, else default


def test_fit_reaches_a_type_only_via_a_converter():
    cn2 = CN2()
    # SeCo has no DecisionList producer, but FlatRuleSet -> DecisionList converts
    dl = cn2.fit(DATA, model=DecisionList)
    assert type(dl) is DecisionList


def test_produces_reports_the_capability_set():
    caps = CN2().produces()
    assert {SingleRule, ConceptModel, FlatRuleSet, ConceptSet, ConceptCascade,
            PairwiseModel} <= caps
    assert DecisionList in caps          # via FlatRuleSet / ConceptCascade converters
    from pyrulearn.models import DisjointRuleSet, EnsembleModel
    assert DisjointRuleSet not in caps and EnsembleModel not in caps


def test_fit_rejects_an_unreachable_type_and_stray_kwargs():
    from pyrulearn.models import DisjointRuleSet
    cn2 = CN2()
    with pytest.raises(TypeError, match="cannot produce"):
        cn2.fit(DATA, model=DisjointRuleSet)     # SeCo rules overlap -> no producer, no converter
    with pytest.raises(TypeError, match="model_kwargs"):
        cn2.fit(DATA, label="a")                 # kwargs without model=


def test_fit_no_model_is_the_learner_default():
    m = CN2().fit(DATA)
    assert type(m) is ConceptSet
    assert set(np.unique(np.asarray(m.predict(DATA)))) <= {"a", "b", "c"}


def test_a_learner_can_disable_one_producer():
    class NoPairwiseCN2(CN2):
        _fit_pairwise = None

    lc = NoPairwiseCN2()
    assert PairwiseModel not in lc.produces()
    assert type(lc.fit(DATA, model=ConceptSet)) is ConceptSet     # others still work
    with pytest.raises(TypeError, match="cannot produce"):
        lc.fit(DATA, model=PairwiseModel)


# ------------------------------------------------------------------- provenance ---

def test_fit_stamps_provenance_with_the_learner_and_its_params():
    cn2 = CN2(random_state=7)
    m = cn2.fit(DATA, model=ConceptSet)
    assert m.provenance is not None
    assert m.provenance.learner == "CN2"
    assert m.provenance.params["random_state"] == 7
    assert m.provenance.source is None       # native learner, no external algorithm


def test_fit_stamps_model_kwargs_into_provenance_params():
    m = CN2().fit(DATA, model=ConceptModel, label="a")
    assert m.provenance.params["label"] == "a"


def test_fit_stamps_provenance_through_a_converter():
    # SeCo has no DecisionList producer -- reached via FlatRuleSet -> DecisionList
    m = CN2().fit(DATA, model=DecisionList)
    assert m.provenance is not None and m.provenance.learner == "CN2"


def test_external_learner_provenance_carries_the_source_library():
    from _negation_helpers import neg_spec, neg_X
    ds = neg_spec(["f0", "f1", "f2"])
    rep = HorizontalBitmapRepresentation(ds, neg_X(_X), _y)

    dt = SKLDecisionTree(max_depth=2, random_state=0)
    m = dt.fit(rep)
    assert m.provenance.learner == "SKLDecisionTree"
    assert m.provenance.source == "sklearn.tree.DecisionTreeClassifier"
    assert m.provenance.params["max_depth"] == 2


def test_external_learners_are_prefixed_and_old_names_are_gone():
    from pyrulearn.interfaces import imodels, lord, sklearn, weka, wittgenstein
    from pyrulearn.learners.boosting import Slipper
    from pyrulearn.learners.seco import CN2

    assert sklearn.SKLDecisionTree().display_name == "SKL:DecisionTree"
    assert wittgenstein.WittRIPPER(pos_class="a").display_name == "Witt:RIPPER"
    assert imodels.IModSlipper().display_name == "IMod:Slipper"
    assert weka.WekaJ48().display_name == "Weka:J48"
    assert lord.JavaLORD().display_name == "JavaLORD"
    assert CN2().display_name == "CN2" and Slipper().display_name == "Slipper"
    for module, old_name in [(weka, "JRip"), (wittgenstein, "RIPPERk"), (sklearn, "DecisionTree"),
                             (lord, "LordJar"), (lord, "JavaLord"), (imodels, "ImodelsRuleFit")]:
        assert not hasattr(module, old_name)


def test_mlrl_boomer_is_prefixed():
    from pyrulearn.interfaces import boomer

    assert boomer.MLRLBoomer().display_name == "MLRL:Boomer"
    assert not hasattr(boomer, "MlrlBoomer")
