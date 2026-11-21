"""External learners: the run time of the external algorithm, the target
class of the wittgenstein learners, and `native_negation`.

- `ExternalRuleLearner.fit` records the external algorithm's own run time
  as `Provenance.run_time` (a native learner's stays `None`);
- `WittRIPPER`/`WittIREP` learn their rules for the minority class unless
  told otherwise, as Ripper does;
- with `native_negation=True`, wittgenstein (and Weka) see only the features
  that are not negations, and their ``= False`` conditions land on the
  negation features again: the model is the same as with the negation
  features (wittgenstein) -- while Pypper, which only tests features for
  being true, needs them.

The Weka tests need a real ``weka.jar`` (``$WEKA_JAR``, optionally
``$WEKA_JAVA``) and skip otherwise.
"""

import os

import numpy as np
import pytest

from pyrulearn.data import HorizontalBitmapRepresentation
from pyrulearn.interfaces.weka import WekaJRip, weka_build_time
from pyrulearn.interfaces.wittgenstein import WittIREP, WittRIPPER
from pyrulearn.learners.seco import Pypper
from pyrulearn.models import ConceptModel

from _negation_helpers import neg_spec, neg_X

_HAS_WEKA = bool(os.environ.get("WEKA_JAR"))


def _data(n=300, d=6, positive_rate=None, seed=0, concept=None):
    """Random Boolean data over `d` features with their negations; the
    class is `concept(X)` (default: f0 and not f1) with 5% noise."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d)) < 0.5
    y = (X[:, 0] & ~X[:, 1]) if concept is None else concept(X)
    y = np.where(rng.random(n) < 0.05, ~y, y)
    y = np.where(y, "pos", "neg")
    return HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(d)]), neg_X(X), y)


# -- run time ----------------------------------------------------------------

def test_external_run_time_recorded():
    rep = _data()
    model = WittRIPPER(pos_class="pos", random_state=0).fit(rep)
    assert model.provenance.run_time is not None and model.provenance.run_time > 0


def test_external_run_time_summed_over_a_decomposition():
    rep = _data()
    from pyrulearn.models import ConceptSet
    model = WittRIPPER(random_state=0).fit(rep, model=ConceptSet)
    assert model.provenance.run_time is not None and model.provenance.run_time > 0


def test_native_run_time_is_none():
    model = Pypper(random_state=0).fit(_data())
    assert model.provenance.run_time is None


def test_weka_build_time_parsed():
    assert weka_build_time("Time taken to build model: 0.05 seconds\n") == 0.05
    assert weka_build_time("no timing here") is None


# -- the target class --------------------------------------------------------

def test_witt_learns_the_minority_class_by_default():
    # "neg" sorts first, but "pos" is the minority: f0 and not f1 holds for about a quarter
    rep = _data()
    assert (np.asarray(rep.y) == "pos").mean() < 0.5
    for cls in (WittRIPPER, WittIREP):
        model = cls(random_state=0).fit(rep, model=ConceptModel)
        assert model.label == "pos"
    # and the majority class if that's the one that sorts last
    flipped = HorizontalBitmapRepresentation(rep.spec, rep.X, np.where(np.asarray(rep.y) == "pos", "zz", "aa"))
    assert WittRIPPER(random_state=0).fit(flipped, model=ConceptModel).label == "zz"


def test_witt_neg_class_still_decides():
    rep = _data()
    assert WittRIPPER(neg_class="pos", random_state=0).fit(rep, model=ConceptModel).label == "neg"


# -- native negation ---------------------------------------------------------

@pytest.mark.parametrize("cls", [WittRIPPER, WittIREP])
def test_witt_native_negation_gives_the_same_model(cls):
    rep = _data()
    with_neg = cls(pos_class="pos", random_state=0).fit(rep)
    native = cls(pos_class="pos", random_state=0, native_negation=True).fit(rep)
    # the rules are expressed over the full feature space either way ...
    assert native.rules and all(r.dataspec is rep.spec for r in native.rules)
    # ... and the negation the concept needs (not f1) is there
    assert any("¬f1" in r.to_string("logic", show_stats=False) for r in native.rules)
    assert np.array_equal(np.asarray(with_neg.predict(rep)), np.asarray(native.predict(rep)))


def test_pypper_needs_the_negation_features():
    rep = _data()
    with_neg = Pypper(random_state=0).fit(rep)
    without = Pypper(random_state=0).fit(rep.without_negations())
    acc = lambda m, r: np.mean(np.asarray(m.predict(r)) == np.asarray(r.y))  # noqa: E731
    # f0 and not f1 can't be expressed without the negation of f1
    assert acc(with_neg, rep) > acc(without, rep.without_negations()) + 0.05


@pytest.mark.skipif(not _HAS_WEKA, reason="needs $WEKA_JAR")
def test_jrip_native_negation_and_run_time():
    rep = _data()
    model = WekaJRip(native_negation=True).fit(rep)
    assert all(r.dataspec is rep.spec for r in model.rules)
    assert any("¬f1" in r.to_string("logic", show_stats=False) for r in model.rules)
    assert np.mean(np.asarray(model.predict(rep)) == np.asarray(rep.y)) > 0.9
    # Weka's own build time, not the subprocess's wall time
    assert model.provenance.run_time is not None and model.provenance.run_time < 1.0
