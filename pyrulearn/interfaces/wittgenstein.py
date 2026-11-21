"""
pyrulearn.interfaces.wittgenstein
==========================================

`ObjectRuleImporter`s for the `wittgenstein` package's rule-induction
algorithms -- `IREPImporter` for `wittgenstein.IREP`, `RIPPERImporter`
for `wittgenstein.RIPPER` -- grouped together since both share the same
underlying `Ruleset`/`Rule`/`Cond` structure (both subclass
wittgenstein's own `AbstractRulesetClassifier`), hence the same
conversion logic, `_rules_from_ruleset` below.

Both algorithms are fundamentally *binary*, confirmed against the
actual library (version 0.3.5) rather than assumed: `fit()` raises
without an explicit `pos_class` for anything beyond two/boolean-like
classes, but -- also confirmed directly, not assumed -- given an
*explicit* `pos_class` it silently proceeds on a `y` with any number of
classes, treating everything that isn't `pos_class` as one undifferentiated
"negative" with no warning at all; there is no internal one-vs-rest
cascade for more classes. `WittIREP`/`WittRIPPER.fit_external` (below) refuse
this outright (`_check_binary_target`) rather than let it happen
unnoticed. The fitted model's own `.predict()` returns plain booleans
("matched `pos_class`" or not) -- it does *not* retain what the actual
fallback *label* should be for what it didn't match. Genuine
multi-class support here, if wanted later, would mean pyrulearn doing
its own one-vs-rest loop on top of repeated binary fits, not something
wittgenstein hands us for free.

Since a fitted ruleset's rules all predict that same single `pos_class`,
multiple rules covering one example never actually conflict the way a
decision tree's differently-labeled leaves can -- any `RuleCombiner`
gives the same answer, so no special conflict-resolution logic is
needed for the `RuleSet` these become. What *is* needed is a fallback
for whatever isn't covered: `IREPImporter`/`RIPPERImporter`, used
directly on an already-fitted model, leave `default_prediction` as
`None` (there being no reliable place to recover the fallback label from
the model alone); `WittIREP`/`WittRIPPER` (the `ExternalRuleLearner` wrappers, below)
compute it from `data.y` at fit time and wire it up
automatically -- see their docstrings for exactly when that's possible.

**Numeric/nominal attributes**: two ways to handle a model fit on raw
(not already-Boolean) columns, same two workflows as
`pyrulearn.interfaces.sklearn` (see `pyrulearn.learners`'s module
docstring):

- **Workflow 1** -- binarize first via `pyrulearn.data.io.
  build_dataspec`/`binarize` so wittgenstein only ever sees
  already-Boolean columns named after real `DataSpec` features
  (``age>=20``, ``color=red``, ...); every `Cond` it then produces is
  already a plain Boolean literal on a feature already in the target
  `DataSpec`, complete with its `ThresholdChain`/`ExactlyOne`
  provenance:

  ```python
  from pyrulearn.data.io import build_dataspec, binarize
  from pyrulearn.data import HorizontalBitmapRepresentation

  ds = build_dataspec(df, target="label", arff_types={...}).build()
  X = binarize(ds, df)
  rep = HorizontalBitmapRepresentation(ds, X, df["label"].to_numpy())
  rules = WittRIPPER(pos_class="pos").fit(rep)
  ```

- **Workflow 2** -- fit directly on raw columns and let
  `infer_dataspec` parse wittgenstein's *own* discretization/nominal
  output back into typed attributes: a bin-range `Cond` like
  ``num_feat=0.53 - 0.84`` becomes two `<=`-family threshold literals
  (`num_feat<=0.53` False, `num_feat<=0.84` True -- see
  `_parse_numeric_cond_value` for why *both* bounds are `<=`-family, not
  `>=`: wittgenstein's own bins are right-closed/left-open,
  `(lo, hi]`, confirmed directly against `wittgenstein.discretize.
  BinTransformer` rather than assumed), an unbounded one like
  ``num_feat=>1.32`` becomes one, and a bare category like ``color=red``
  becomes a nominal equality literal. Which of these a given attribute gets is decided
  per attribute (not globally) from the actual `Cond.val`s seen for it
  -- all-Boolean stays plain Boolean, all-parseable-as-a-bin becomes
  NUMERIC, anything else becomes NOMINAL (one value that doesn't parse
  is enough to make the whole attribute NOMINAL -- e.g. `credit-g`'s
  `checking_status`, whose actual categories include `"<0"`,
  `"0<=X<200"`, and `"no checking"`: `"<0"` alone would parse as a
  numeric bin, but `"no checking"` never could, so the attribute is
  correctly NOMINAL overall, and `_rules_from_ruleset` -- which looks up
  this same per-attribute decision rather than re-parsing each value on
  its own -- treats `"<0"` as one literal category, not a threshold).
  Caveat: this is pattern matching on wittgenstein's own string format,
  not real type information -- a NOMINAL column whose category labels
  *all* happen to look like a bin/range string (unlikely, but possible)
  would still be misdetected as NUMERIC.

This module requires the `wittgenstein` package (an optional dependency
-- only importing this specific module pulls it in, per
`pyrulearn.interfaces`'s own dependency-isolation convention).
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import ObjectRuleImporter, register_importer
from ..data.attributes import AttributeType
from ..data import DataRepresentation
from ..data import DataSpec, DataSpecBuilder
from ..learners import DecomposingLearner, ExternalRuleLearner
from ..models import ConceptModel
from ..rule import Rule

# wittgenstein's own numeric-bin Cond.val formats (see bin_transformer_):
# "<t", ">t", or "lo - hi" -- always space-dash-space between bounds, so a
# negative bound's own leading "-" (no space) is never ambiguous with it.
_NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_RANGE_RE = re.compile(rf"^({_NUM}) - ({_NUM})$")
_UNBOUNDED_RE = re.compile(rf"^([<>])({_NUM})$")


def _parse_numeric_cond_value(val: str) -> Optional[List[Tuple[float, bool]]]:
    """Parse a wittgenstein numeric-bin `Cond.val` string into `<=`-family
    threshold literals -- `(threshold, is_true)` pairs, ANDed together,
    meaning the attribute's `<=threshold` derived feature must be
    True/False.

    wittgenstein's own bins are right-closed, left-open -- confirmed
    directly against `wittgenstein.discretize.BinTransformer`, not
    assumed: its `fit`'s own docstring states "min is exclusive; max is
    inclusive," `_thresholds_to_intervals` builds each bin as
    `pd.Interval(..., closed='right')`, and empirically, transforming a
    value exactly equal to a bin's lower edge lands it in the
    *previous* bin, while a value equal to the upper edge stays in the
    bin. So ``"lo - hi"`` means `(lo, hi]`, i.e. `x>lo AND x<=hi`, not
    `x>=lo AND x<hi`: both bounds are `<=`-family tests -- the lower
    bound is that family's *negated* test (`x>lo` is the exact
    complement of `x<=lo`), the upper bound its plain positive one --
    giving `[(lo, False), (hi, True)]`. Likewise ``">t"`` means `(t,
    inf)`, i.e. `x>t` (exclusive) -> `[(t, False)]`; ``"<t"`` means
    `(-inf, t]`, i.e. `x<=t` (inclusive) -> `[(t, True)]`. (An earlier
    version of this function had all four of these backwards --
    treating the range as `[lo, hi)` via the `>=` family -- which only
    misclassifies examples landing exactly on a bin edge, hence going
    unnoticed until checked directly against `BinTransformer`.)

    Returns `None` if `val` doesn't match either pattern -- i.e. it's a
    genuine nominal category, not a numeric bin.
    """
    m = _RANGE_RE.match(val)
    if m:
        return [(float(m.group(1)), False), (float(m.group(2)), True)]
    m = _UNBOUNDED_RE.match(val)
    if m:
        op, t = m.group(1), float(m.group(2))
        return [(t, op == "<")]
    return None


def _rules_from_ruleset(
    ruleset, dataspec: DataSpec, target: Any, feature_names: Optional[Sequence[str]] = None
) -> List[Rule]:
    """One `Rule` per `wittgenstein.base.Rule` in a fitted `IREP`/
    `RIPPER` model's `.ruleset_`; every produced `Rule` predicts the
    same `target` (`model.pos_class`), since that's the only class such
    a ruleset ever predicts.

    Without `feature_names` (default): every `Cond.val` must already be
    Boolean (`dataspec` built by hand or via `build_dataspec`/`binarize`
    before fitting -- workflow 1, see the module docstring) -- becomes a
    literal on `dataspec.feature_index(cond.feature)` directly; a
    non-Boolean value raises.

    With `feature_names`: `dataspec` must be one `infer_dataspec` built
    from this same `model` (workflow 2). Which of NUMERIC or NOMINAL a
    `Cond.val` becomes is decided by looking up `cond.feature`'s
    *already-discovered* type in `dataspec` -- not by re-parsing the
    value independently -- since `infer_dataspec` only calls an
    attribute NUMERIC when *every* value it saw for it parsed as a bin;
    a single ambiguous value (e.g. a NOMINAL category that happens to
    read like a bin, such as `"<0"` as a literal `checking_status`
    category, not a wittgenstein threshold) must be resolved the same
    way discovery already resolved it, or the two would disagree and
    this would raise a spurious `KeyError` from `dataspec.feature_index`.
    A NUMERIC attribute's value is parsed via `_parse_numeric_cond_value`
    into one or two threshold literals (`f"{feature}>={threshold}"`);
    anything else becomes one nominal-equality literal
    (`f"{feature}={val}"`).
    """
    rules: List[Rule] = []
    for wrule in ruleset.rules:
        pos: List[int] = []
        neg: List[int] = []
        for cond in wrule.conds:
            if isinstance(cond.val, (bool, np.bool_)):
                idx = dataspec.feature_index(cond.feature)
                (pos if cond.val else neg).append(idx)
                continue
            if feature_names is None:
                raise ValueError(
                    f"Cond on feature {cond.feature!r} has non-Boolean value {cond.val!r} -- "
                    "pass feature_names= (with a dataspec built via infer_dataspec) to convert "
                    "numeric/nominal Conds, or fit on already-Boolean-encoded columns instead"
                )
            attr = dataspec.attributes.get(cond.feature)
            if attr is not None and attr.type == AttributeType.NUMERIC:
                parsed = _parse_numeric_cond_value(cond.val) if isinstance(cond.val, str) else None
                if parsed is None:
                    raise ValueError(
                        f"Cond value {cond.val!r} on feature {cond.feature!r} doesn't match a "
                        "recognized numeric-bin format, but infer_dataspec discovered this "
                        "attribute as NUMERIC -- dataspec wasn't built from this same model"
                    )
                for threshold, is_true in parsed:
                    idx = dataspec.feature_index(f"{cond.feature}<={threshold}")
                    (pos if is_true else neg).append(idx)
            else:
                pos.append(dataspec.feature_index(f"{cond.feature}={cond.val}"))
        rules.append(Rule.from_pos_neg(pos=pos, neg=neg, target=target, dataspec=dataspec))
    return rules


def _dataspec_from_wittgenstein(ruleset, feature_names: Sequence[str]) -> DataSpec:
    """Shared `infer_dataspec` body for `IREPImporter`/`RIPPERImporter`
    (workflow 2): every distinct `Cond.val` actually used, grouped by
    raw feature name, decides that attribute's type -- see the module
    docstring for the exact rule. A name never referenced by any rule
    contributes no attribute at all (same reasoning as
    `pyrulearn.interfaces.sklearn._dataspec_from_thresholds`).
    """
    by_col: Dict[str, set] = {}
    for wrule in ruleset.rules:
        for cond in wrule.conds:
            by_col.setdefault(cond.feature, set()).add(cond.val)

    builder = DataSpecBuilder()
    for name in feature_names:
        vals = by_col.get(name)
        if not vals:
            continue
        if all(isinstance(v, (bool, np.bool_)) for v in vals):
            builder.add_boolean(name)
            continue
        thresholds: set = set()
        is_numeric = True
        for v in vals:
            parsed = _parse_numeric_cond_value(v) if isinstance(v, str) else None
            if parsed is None:
                is_numeric = False
                break
            thresholds.update(t for t, _ in parsed)
        if is_numeric:
            builder.add_numeric(name, le_thresholds=sorted(thresholds))
        else:
            builder.add_nominal(name, sorted(vals))
    return builder.build()


def _default_target_for(model) -> Optional[Any]:
    """`RuleSet`'s `default_prediction` for a fitted `IREP`/`RIPPER`
    model -- `None` unless `fit_external` (below) already computed and
    stashed the fallback label as `model._pyrulearn_default_target` (see
    the module docstring for why the model alone never has this)."""
    return getattr(model, "_pyrulearn_default_target", None)


class IREPImporter(ObjectRuleImporter):
    """Extracts a `RuleSet` from a fitted `wittgenstein.IREP` model --
    one `Rule` per rule in `model.ruleset_`, all predicting
    `model.pos_class`. See the module docstring for the two ways to
    handle numeric/nominal attributes, and why `default_prediction` is
    left `None` here either way (use `WittIREP`, below, for that).
    """

    SOURCE = "wittgenstein.IREP"

    def import_model(self, model, dataspec: DataSpec, feature_names: Optional[Sequence[str]] = None,
                     data: Optional[DataRepresentation] = None) -> ConceptModel:
        """Without `feature_names`: `dataspec`'s feature order must
        match the Boolean columns `model` was trained on (workflow 1).
        With it: `dataspec` must be one `infer_dataspec` built from this
        same `model` and `feature_names` (workflow 2). Extracted rules
        bind to `dataspec` directly either way.
        """
        rules = _rules_from_ruleset(model.ruleset_, dataspec, model.pos_class, feature_names=feature_names)
        rules = self._stamp_rule_provenance(rules, pos_class=model.pos_class)
        rules = self._stamp_rule_stats(rules, data)
        return self._stamp_provenance(
            ConceptModel(rules, label=model.pos_class, default_prediction=_default_target_for(model)),
            pos_class=model.pos_class,
        )

    def infer_dataspec(self, model, feature_names: Sequence[str]) -> DataSpec:
        """Discover a typed attribute per raw feature `model`'s fitted
        ruleset actually used -- see the module docstring for how each
        attribute's type is decided. Pass the result (and the same
        `feature_names`) to `import_model`.
        """
        return _dataspec_from_wittgenstein(model.ruleset_, feature_names)


register_importer("irep", IREPImporter)


class RIPPERImporter(ObjectRuleImporter):
    """Extracts a `RuleSet` from a fitted `wittgenstein.RIPPER` model --
    identical conversion to `IREPImporter` (both wittgenstein algorithms
    share the same `Ruleset`/`Rule`/`Cond` structure), only the `SOURCE`
    tag differs. See `IREPImporter`'s docstring and the module docstring
    for the shared details.
    """

    SOURCE = "wittgenstein.RIPPER"

    def import_model(self, model, dataspec: DataSpec, feature_names: Optional[Sequence[str]] = None,
                     data: Optional[DataRepresentation] = None) -> ConceptModel:
        rules = _rules_from_ruleset(model.ruleset_, dataspec, model.pos_class, feature_names=feature_names)
        rules = self._stamp_rule_provenance(rules, pos_class=model.pos_class)
        rules = self._stamp_rule_stats(rules, data)
        return self._stamp_provenance(
            ConceptModel(rules, label=model.pos_class, default_prediction=_default_target_for(model)),
            pos_class=model.pos_class,
        )

    def infer_dataspec(self, model, feature_names: Sequence[str]) -> DataSpec:
        """See `IREPImporter.infer_dataspec` -- identical logic."""
        return _dataspec_from_wittgenstein(model.ruleset_, feature_names)


register_importer("ripper", RIPPERImporter)


def _check_binary_target(y: np.ndarray, learner_name: str) -> None:
    """`WittIREP`/`WittRIPPER` fit silently on more-than-two-class `y` --
    confirmed directly against the library: given an explicit
    `pos_class`, wittgenstein just treats everything else as negative,
    discarding the distinction between the other classes with no
    warning at all. Raise here instead of letting that happen unnoticed
    -- real multi-class support, if wanted later, needs pyrulearn doing
    its own one-vs-rest handling on top (see the module docstring), not
    something to fall into by accident.
    """
    classes = np.unique(y)
    if len(classes) > 2:
        raise ValueError(
            f"{learner_name} is fundamentally binary (see the module docstring) but y has "
            f"{len(classes)} distinct classes ({list(classes)}) -- wittgenstein would silently "
            "collapse every non-pos_class label into \"negative\" rather than erroring, discarding "
            "the distinction between them. Binarize the target yourself first if that's really "
            "what you want, e.g. np.where(y == pos_class, pos_class, other_label)."
        )


def _resolve_default_target(y: np.ndarray, pos_class: Any, neg_class: Optional[Any]) -> Optional[Any]:
    """The label `WittIREP`/`WittRIPPER`'s `default_prediction` should be for
    examples no learned rule covers. `neg_class`, if given, wins
    outright. Otherwise: exactly one other label besides `pos_class`
    present in `y` (the ordinary binary case) is used automatically;
    anything else (more than one other label, i.e. genuinely
    multi-class data wittgenstein itself doesn't support) leaves it
    unresolved (`None`, i.e. abstain) rather than guessing which of
    several other labels should stand in for "not positive" -- pass
    `neg_class=` explicitly if you want a fallback in that case anyway.
    """
    if neg_class is not None:
        return neg_class
    others = [c for c in np.unique(y) if c != pos_class]
    return others[0] if len(others) == 1 else None


class _WittgensteinLearner(DecomposingLearner, ExternalRuleLearner):
    """Shared plumbing for `WittIREP` / `WittRIPPER`.

    `fit(data)` -> a `pyrulearn.models.ConceptModel` for `pos_class`
    (`neg_class` -- or the sole other label -- as the fallback);
    wittgenstein is binary, so `data.y` must have ≤ 2 labels. Without a
    `pos_class`, the rules are learned for the label other than
    `neg_class`, if given, else for the minority class -- as Ripper does
    itself, leaving the majority class to the default rule.

    `native_negation=True` hands wittgenstein only the features that are
    not negations of others (`DataRepresentation.without_negations`): it
    tests a Boolean column for both values anyway, so its ``= False``
    conditions provide exactly the negations -- the same conditions as
    with them, without evaluating each one twice. Default `False`.
    `fit(data, model=ConceptModel, label="a")` sets the target
    explicitly. `fit(data, model=ConceptSet | ConceptCascade |
    PairwiseModel)` decomposes: each binary sub-fit relabels the rest to
    ``"rest"`` first (`DecomposingLearner._fit_binary`).
    """

    NATIVE_MODEL = ConceptModel
    _LIB_CLS: str  # "IREP" | "RIPPER"

    def __init__(self, pos_class: Any = None, neg_class: Optional[Any] = None,
                 native_negation: bool = False, **params):
        self.pos_class = pos_class
        self.neg_class = neg_class
        self.native_negation = native_negation
        self.params = params

    def _import(self, data):
        """`ExternalRuleLearner._import`, but with `native_negation=True`
        wittgenstein sees only the features that are not negations of
        others: it tests a Boolean column for both values anyway, and its
        ``= False`` conditions land on the paired negation features of
        `data.spec` again on import (by name -- the surviving features
        keep theirs)."""
        sub = data.without_negations() if self.native_negation else data
        fitted = self.fit_external(self.prepare(sub), sub.y, feature_names=sub.spec.feature_names)
        return self.IMPORTER().import_model(fitted, data.spec, data=data)

    def fit_external(self, X, y, feature_names=None):
        import wittgenstein as lw

        if y is None:
            raise ValueError(f"{type(self).__name__} needs labels (y) to fit")
        pos = self.pos_class
        if pos is None:
            labels, counts = np.unique(y, return_counts=True)
            if len(labels) != 2:
                raise ValueError(
                    f"{type(self).__name__} needs pos_class (or fit(model=ConceptModel, label=...))"
                )
            if self.neg_class is not None:
                pos = next(c for c in labels if c != self.neg_class)
            else:  # as Ripper itself: rules for the minority class (ties: the first label)
                pos = labels[int(np.argmin(counts))]
        _check_binary_target(y, type(self).__name__)
        model = getattr(lw, self._LIB_CLS)(**self.params)
        model.fit(X, y=y, pos_class=pos, feature_names=feature_names)
        model._pyrulearn_default_target = _resolve_default_target(y, pos, self.neg_class)
        return model

    def _fit_binary(self, data, positive, negative=None):
        lc = copy.copy(self)
        lc.pos_class = positive
        if negative is not None:                       # pairwise: data is already the pair
            lc.neg_class = negative
            return lc._fit_native(data)
        y = np.asarray(data.y)                          # one-vs-rest: fold the rest in ourselves
        lc.neg_class = "rest"
        return lc._fit_native(data.relabel(np.where(y == positive, positive, "rest")))


class WittIREP(_WittgensteinLearner):
    """`wittgenstein.IREP`. `**params` are its constructor args
    (`prune_size=`, `random_state=`, ...). See `_WittgensteinLearner`."""

    IMPORTER = IREPImporter
    _LIB_CLS = "IREP"


class WittRIPPER(_WittgensteinLearner):
    """`wittgenstein.RIPPER` (IREP\\* + `k` optimization passes -- hence
    the "k"). `**params` are its constructor args (`k=`, `prune_size=`,
    ...). See `_WittgensteinLearner`."""

    IMPORTER = RIPPERImporter
    _LIB_CLS = "RIPPER"

# -- naming ------------------------------------------------------------------

_WittgensteinLearner.TOOL = "Witt"
