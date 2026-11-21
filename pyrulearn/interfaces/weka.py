"""
pyrulearn.interfaces.weka
===============================

`StringRuleImporter`s for Weka's text-dump rule formats --
`JRipImporter` for `weka.classifiers.rules.JRip`, `PARTImporter` for
`weka.classifiers.rules.PART`, and `J48Importer` for
`weka.classifiers.trees.J48` so far; other Weka rule/tree learners with
their own text dump belong in this module too when added, per this
package's group-by-shared-implementation convention. All three share
everything below the line/block-level: once a source-specific parser
has produced `(conditions, target)` tuples,
`_infer_dataspec_from_conditions`/`_rule_from_conditions` build the
`DataSpec`/`Rule`s identically regardless of which classifier's text
they came from -- only the surrounding line/block/tree structure
differs.

Weka is not a Python library, so unlike every other concrete importer in
`pyrulearn.interfaces` so far, this one is genuinely a `StringRuleImporter`,
not an `ObjectRuleImporter`: there is no live fitted Python object to
inspect, only JRip's own printed rule-set text -- e.g. from
``java -cp weka.jar weka.classifiers.rules.JRip -t data.arff`` -- which
looks like this (captured from a real `weka.jar` 3.8.7 run, not assumed)::

    JRIP rules:
    ===========

    (color = red) and (age >= 41) => label=pos (81.0/2.0)
    (income <= 19992) => label=pos (32.0/1.0)
     => label=neg (287.0/15.0)

    Number of Rules : 3

`parse` only looks for lines containing ``"=>"``, verified against real
Weka output that this is a safe filter for the *whole* raw console dump,
not just the isolated rule block -- none of the surrounding
header/footer or evaluation-stats tables (accuracy, confusion matrix,
...) contain that token. So the complete captured stdout can be handed
to `parse` directly, no need to first isolate the rule block by hand.

**Produces a `RuleList`, not a `RuleSet`** -- like
`pyrulearn.interfaces.imodels.BayesianRuleListImporter`, and unlike
`pyrulearn.interfaces.wittgenstein`'s IREP/RIPPER importers. Same
distinction as made there: wittgenstein's extracted rules never disagree
in target (binary, single `pos_class`), so order doesn't matter for
prediction; JRip is genuinely **multi-class** (classic RIPPER
sequential-covering induction -- one class's rules at a time, in printed
order, falling back to a default/majority class), so two rules with
different targets can both cover the same example on new data, and only
trying them in the printed order (first match wins, exactly matching
JRip's own prediction semantics) reproduces its predictions correctly.
The trailing, condition-less ``=> label=<value> (...)`` line becomes the
`RuleList`'s `default_prediction`.

**Numeric conditions.** Real JRip output uses all four of ``>=``,
``<``, ``<=``, ``>`` (see the example above). Since the explicit-negation
redesign, every one is its own derived feature -- there are no negative
literals -- and `DataSpecBuilder.add_numeric` (with negation on, the
default) generates them in exactly-complementary pairs linked by a
`NumericGroup` constraint:

- ``attr >= T`` and ``attr < T`` -> features ``f"{attr}>={T}"`` /
  ``f"{attr}<{T}"``, exact negations of each other.
- ``attr <= T`` and ``attr > T`` -> features ``f"{attr}<={T}"`` /
  ``f"{attr}>{T}"``, exact negations of each other.

All of them binarize straight from the same raw numeric column (e.g.
`pyrulearn.data.io.binarize` derives ``income<=19992`` from a plain
``"income"`` column on its own -- no precomputed helper column needed),
and each ``>=`` / ``<=`` family keeps its own `NumericGroup`'s
monotonic-implication reasoning. The ``>=`` and ``<=`` families aren't
cross-checked against each other, though: nothing stops a rule from
stating both ``age>=30`` and ``age<=25`` (impossible to satisfy) without
`Rule.is_consistent` flagging it -- deliberately not built, since rules
parsed from JRip's own text are read in as-is rather than hand-composed,
and such a rule would simply never cover any real example, which is a
sufficient outcome here.

**Nominal conditions** (``attr = value`` / ``attr != value``) go through
`DataSpecBuilder.add_nominal`, one call per attribute with every
category *actually seen* for it across the whole rule set -- exactly
`pyrulearn.interfaces.wittgenstein`'s workflow-2 `infer_dataspec`
precedent, including its same accepted caveat: the resulting
`NominalGroup` constraint technically claims those are the attribute's
*only* categories, when really they're only the ones the rule set
happened to reference. ``attr = value`` is a positive literal on the
``f"{attr}={value}"`` feature; ``attr != value`` (not seen in practice
-- Weka's own `NominalAntd` always tests equality against one specific
value -- but handled regardless) is a positive literal on the paired
``f"{attr}!={value}"`` negation feature: exact either way, no boundary
concerns the way numeric conditions have.

**The trailing ``(covered/errors)`` on each rule line is parsed only to
validate/delimit the line**, not retained on the rule: Weka's own
per-rule support is scoped to its *covering loop* (rows not already
claimed by an earlier rule in the same sequence, for JRip/PART's ordered
decision lists) -- a genuinely different quantity from a rule's own
*measured* `stats()` (`_stamp_rule_stats`/`pyrulearn.models.
annotate_rules`, wired into the `fit()` round trip), which measures the
rule as a standalone predictor (its raw coverage of the whole dataset,
the same definition every other learner's rules get). The two agree
only where nothing upstream can have already claimed a row (e.g. J48's
pairwise-disjoint leaves) -- since there's no principled single place to
store a covering-loop-scoped number that would silently disagree with
`stats()` for JRip/PART, it's simply not kept; annotate via `data=`/
`fit()` for real, comparable numbers instead.

JRip is natively multi-class (unlike wittgenstein's IREP/RIPPER or
`imodels`' Bayesian Rule List/Rule Set, all binary or binary-with-an-
explicit-positive-class) -- each rule's target is read straight from its
consequent, no `pos_class`/binary restriction needed on this side.

**`J48Importer`** (`weka.classifiers.trees.J48`, Weka's C4.5
reimplementation) produces a `pyrulearn.models.DisjointRuleSet`, not
a `DecisionList`: unlike JRip/PART's sequential covering, a decision tree's
leaves already partition the feature space disjointly and exhaustively
by construction -- the same reasoning
`pyrulearn.interfaces.sklearn.SklearnTreeImporter` uses for its own
`DisjointRuleSet` result -- so there's no condition-less default rule to
look for, and no rule-order semantics to preserve. J48's printed text is
an indented tree, not a flat rule list -- captured from a real run, not
assumed::

    J48 pruned tree
    ------------------

    income <= 19992: pos (40.0/1.0)
    income > 19992
    |   color = red
    |   |   age <= 40: neg (45.0/3.0)
    |   |   age > 40: pos (73.0/2.0)
    |   color = green: neg (125.0/8.0)
    |   color = blue: neg (117.0/4.0)

    Number of Leaves  : 	5

Each ``|   `` group is one level of indentation depth; a line ending in
``: target (stats)`` is a leaf (its rule is the conjunction of every
ancestor condition still open at that depth, plus its own, if any); a
line with no such suffix is an internal split node whose condition
applies to everything nested under it. `_collect_j48_conditions` walks
the lines with a depth-indexed stack -- push a condition when entering a
deeper split, truncate the stack to the current depth before each line
(that's the backtrack when a sibling appears at a shallower or equal
depth) -- which handles a nominal split fanning out into more than two
children (C4.5, unlike CART, doesn't restrict splits to binary) exactly
as easily as a binary one: each sibling is just another line at the same
depth, and popping back to that depth via the stack doesn't care how
many siblings came before it. Verified against a deliberately deep,
33-leaf/depth-5 tree, not just the shallow example above.

Unlike JRip/PART, the tree can't be picked out of a whole raw console
dump by scanning every line for a content pattern: the confusion
matrix's own column layout can produce a line that reads as *both* a
``|   ``-indented node *and* a valid ``attr = value`` condition (e.g.
`` 110  15 |   a = pos``), which would otherwise be misparsed as a
depth-1 split. `parse` instead anchors explicitly on the ``J48 ...
tree`` header line, its ``----`` divider, and the blank line that
follows it (present even in the shallow example above -- easy to miss,
confirmed directly against real output), then reads until the next
blank line.

**`PARTImporter`** (`weka.classifiers.rules.PART`, Frank & Witten's
"partial decision trees" rule learner -- separate-and-conquer over a
pruned partial C4.5 tree each round, taking its "best" leaf as one rule)
produces the same kind of `RuleList`, for the same reason as JRip
(genuinely multi-class, ordered, first-match-wins), but its printed text
is laid out completely differently -- captured from a real run, not
assumed::

    PART decision list
    ------------------

    income > 19992 AND
    color = green: neg (125.0/8.0)

    income > 20292 AND
    color = blue: neg (117.0/4.0)

    age > 39: pos (100.0/3.0)

    income > 20373: neg (45.0/3.0)

    : pos (13.0)

    Number of Rules  : 	5

No ``=>``, no parentheses around conditions, one condition per line with
a trailing `` AND`` on every line but the last (which instead ends in
``: target (stats)``), and rules are blank-line-delimited blocks rather
than one line each -- so `PARTImporter` has its own block-level parser
(`_collect_part_conditions`), unrelated to JRip's line regex. Two
smaller format differences worth knowing: the trailing stats can omit
the error count entirely when it's zero (``(13.0)``, not JRip's always-
both-numbers ``(13.0/0.0)``), and numeric splits are classic C4.5-style
``<=``/``>`` (never ``>=``/``<``) -- both handled by the same shared
`_split_condition`/`_infer_dataspec_from_conditions`/
`_rule_from_conditions` JRip already uses, since those only care about
the six recognized operator strings, not which classifier produced
them. The blank-line-block filter is checked against a whole raw
console dump the same way JRip's ``"=>"`` filter is: every other
section (accuracy tables, confusion matrix, the header banner) never
ends a block in ``: word (numbers)``, so `parse` can take the full
captured stdout directly.

**Running Weka, not just parsing it.** `run_weka` writes a
`DataRepresentation`'s already-Boolean matrix to a temporary ARFF (safe
`f0..fN` placeholder attribute names), invokes ``java -cp <weka.jar>``
as a subprocess, and returns the printed model text for `parse` --
so `WekaJRip` / `WekaPART` / `WekaJ48` are full `pyrulearn.learners` learners with a
uniform `fit(data, model=...)`, not import-only adapters. The
text-parsing path stays standard-library only; `run_weka` (and the
learner classes) additionally use `subprocess` and, lazily,
`pandas` / `pyrulearn.data.io.write_arff`.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .base import StringRuleImporter, register_importer
from .. import models
from ..models import DecisionList, DisjointRuleSet
from ..data import DataSpec, DataSpecBuilder
from ..learners import RelabelingExternalLearner, produces
from ..rule import Rule

# longest operators first, so ">="/"<="/"!=" are never mistaken for
# ">"/"<"/"=" with a stray extra character
_OP_RE = re.compile(r"^(.+?)\s*(>=|<=|!=|>|<|=)\s*(.+)$")
_RULE_LINE_RE = re.compile(
    r"^(?P<antecedent>.*?)\s*=>\s*(?P<consequent>.+?)\s*"
    r"\(\s*(?P<covered>[\d.]+)\s*(?:/\s*(?P<errors>[\d.]+)\s*)?\)\s*$"
)

ParsedRule = Tuple[List[Tuple[str, str, str]], str]


def _split_condition(text: str) -> Tuple[str, str, str]:
    """`"color = red"` -> `("color", "=", "red")`."""
    m = _OP_RE.match(text.strip())
    if not m:
        raise ValueError(f"couldn't parse condition {text!r} as 'attr OP value'")
    attr, op, value = m.groups()
    return attr.strip(), op, value.strip()


def _split_antecedent(antecedent: str) -> List[str]:
    if not antecedent:
        return []
    parts = []
    for part in antecedent.split(" and "):
        part = part.strip()
        if part.startswith("(") and part.endswith(")"):
            part = part[1:-1].strip()
        parts.append(part)
    return parts


def _collect_conditions(source: str) -> List[ParsedRule]:
    """Every rule line in `source` (any line containing `"=>"` -- see
    the module docstring for why that's a safe filter for a whole raw
    console dump), parsed into `(conditions, target)`. The trailing
    ``(covered[/errors])`` stats suffix must still match (it's what
    confirms this is really a well-formed rule line and delimits where
    the consequent ends), but the numbers themselves aren't kept -- see
    the module docstring.
    """
    parsed: List[ParsedRule] = []
    for line in source.splitlines():
        if "=>" not in line:
            continue
        m = _RULE_LINE_RE.match(line.strip())
        if not m:
            raise ValueError(f"line looks like a JRip rule but doesn't parse: {line!r}")
        antecedent = m.group("antecedent").strip()
        consequent = m.group("consequent")
        if "=" not in consequent:
            raise ValueError(f"couldn't parse consequent {consequent!r} in line {line!r}")
        _, target = consequent.split("=", 1)
        conditions = [_split_condition(c) for c in _split_antecedent(antecedent)]
        parsed.append((conditions, target))
    return parsed


def _infer_dataspec_from_conditions(parsed: Sequence[ParsedRule]) -> DataSpec:
    ge_thresholds: Dict[str, Set[float]] = {}
    le_thresholds: Dict[str, Set[float]] = {}
    nominal_categories: Dict[str, Set[str]] = {}

    for conditions, _target in parsed:
        for attr, op, value in conditions:
            if op in (">=", "<"):
                ge_thresholds.setdefault(attr, set()).add(float(value))
            elif op in ("<=", ">"):
                le_thresholds.setdefault(attr, set()).add(float(value))
            elif op in ("=", "!="):
                nominal_categories.setdefault(attr, set()).add(value)
            else:
                raise ValueError(f"unrecognized operator {op!r} for attribute {attr!r}")

    # negation on: `<` / `>` / `!=` conditions become positive literals on
    # the paired negation feature (age<41, income>19992, color!=red).
    builder = DataSpecBuilder(negation=True)
    for attr in sorted(set(ge_thresholds) | set(le_thresholds)):
        builder.add_numeric(
            attr,
            sorted(ge_thresholds.get(attr, ())),
            le_thresholds=sorted(le_thresholds.get(attr, ())),
        )
    for attr, categories in nominal_categories.items():
        builder.add_nominal(attr, sorted(categories))
    return builder.build()


def _rule_from_conditions(conditions: Sequence[Tuple[str, str, str]], target: str,
                          dataspec: DataSpec) -> Rule:
    pos: List[int] = []
    neg: List[int] = []
    for attr, op, value in conditions:
        if op == ">=":
            pos.append(dataspec.feature_index(f"{attr}>={float(value)}"))
        elif op == "<":
            neg.append(dataspec.feature_index(f"{attr}>={float(value)}"))
        elif op == "<=":
            pos.append(dataspec.feature_index(f"{attr}<={float(value)}"))
        elif op == ">":
            neg.append(dataspec.feature_index(f"{attr}<={float(value)}"))
        elif op == "=":
            pos.append(dataspec.feature_index(f"{attr}={value}"))
        elif op == "!=":
            neg.append(dataspec.feature_index(f"{attr}={value}"))
        else:
            raise ValueError(f"unrecognized operator {op!r} for attribute {attr!r}")
    return Rule.from_pos_neg(pos=pos, neg=neg, target=target, dataspec=dataspec)


#: values a placeholder `f{i}` column tests True against (Weka writes bool
#: columns as `{False,True}` nominal; `1`/`t`/`yes` accepted defensively).
_PLACEHOLDER_TRUE = {"true", "t", "1", "yes", "y"}
_PLACEHOLDER_FALSE = {"false", "f", "0", "no", "n"}


def _placeholder_rule_from_conditions(conditions: Sequence[Tuple[str, str, str]], target: str,
                                      dataspec: DataSpec, column_map: Optional[Sequence[int]] = None) -> Rule:
    """`_rule_from_conditions`'s counterpart for a `fit()` round-trip,
    where the ARFF was written with safe placeholder attribute names
    (`f0`, `f1`, ...) for `dataspec`'s already-Boolean columns -- so each
    condition is ``f{i} = <bool>`` and binds to `dataspec` feature ``i``
    **by position** (`Rule.remap` is by-name, so it can't do this). A
    ``= True`` test is a positive literal on that feature, ``= False`` a
    negative one (which `Rule.from_pos_neg` lands on the paired negation
    feature); ``!=`` flips. With `column_map`, placeholder ``f{i}`` is
    `dataspec` feature ``column_map[i]`` instead (the columns Weka saw
    were a subset of `dataspec`'s -- `native_negation=True`).
    """
    pos: List[int] = []
    neg: List[int] = []
    for attr, op, value in conditions:
        if not (attr.startswith("f") and attr[1:].isdigit()):
            raise ValueError(
                f"placeholder_features=True expects f0/f1/... attribute names, got {attr!r}"
            )
        idx = int(attr[1:])
        if column_map is not None:
            idx = int(column_map[idx])
        v = str(value).strip().lower()
        if v in _PLACEHOLDER_TRUE:
            truthy = True
        elif v in _PLACEHOLDER_FALSE:
            truthy = False
        else:
            raise ValueError(f"placeholder feature {attr!r} tested against non-Boolean value {value!r}")
        if op == "!=":
            truthy = not truthy
        elif op != "=":
            raise ValueError(f"placeholder feature {attr!r} used with operator {op!r} (expected '='/'!=')")
        (pos if truthy else neg).append(idx)
    return Rule.from_pos_neg(pos=pos, neg=neg, target=target, dataspec=dataspec)


def _rules_and_default(
    parsed: Sequence[ParsedRule], dataspec: DataSpec, rule_builder=_rule_from_conditions,
) -> Tuple[List[Rule], Optional[Any]]:
    """Shared by `JRipImporter`/`PARTImporter`'s `parse`: turn already
    line/block-parsed `(conditions, target)` tuples into `Rule`s plus the
    condition-less default line's `target`, once the source-format-
    specific parsing is done. `target` is `None` when there is no
    default line. Raises if more than one condition-less rule is present.
    """
    rules: List[Rule] = []
    default_target: Optional[Any] = None
    seen_default = False
    for conditions, target in parsed:
        if not conditions:
            if seen_default:
                raise ValueError("source has more than one condition-less (default) rule")
            seen_default = True
            default_target = target
        else:
            rules.append(rule_builder(conditions, target, dataspec))
    return rules, default_target


def _rule_builder_for(importer: Any):
    """`_placeholder_rule_from_conditions` when the importer was built with
    `placeholder_features=True` (a `fit()` round-trip -- attributes are
    `f0/f1/...`, bound to `importer.dataspec` by position), else the
    default by-name `_rule_from_conditions`."""
    if getattr(importer, "placeholder_features", False):
        if importer.dataspec is None:
            raise ValueError("placeholder_features=True requires an explicit dataspec=")
        column_map = getattr(importer, "column_map", None)
        if column_map is None:
            return _placeholder_rule_from_conditions
        return lambda conditions, target, dataspec: _placeholder_rule_from_conditions(
            conditions, target, dataspec, column_map=column_map)
    return _rule_from_conditions


def _finalize_rulelist(
    importer: Any, rules: List[Rule], default_target: Optional[Any],
    data: Optional[Any] = None,
) -> "DecisionList":
    """Build the `DecisionList` and provenance-tag every rule (each
    rule's own `.provenance`, and the model-level `.provenance`). `data`,
    when given by a `fit()` round trip, also gets each rule its own
    measured `stats` -- see `RuleImporter._stamp_rule_stats`.
    """
    rules = importer._stamp_rule_provenance(rules, n_rules=len(rules))
    rules = importer._stamp_rule_stats(rules, data)

    rule_list = DecisionList(rules, default_prediction=default_target)
    if default_target is not None:
        importer._stamp_rule_provenance([rule_list.default_rule], n_rules=len(rules))
    return importer._stamp_provenance(rule_list, n_rules=len(rules))


class JRipImporter(StringRuleImporter):
    """Extracts a `pyrulearn.models.DecisionList` from
    `weka.classifiers.rules.JRip`'s printed rule-set text (or a full raw
    Weka console dump containing it). See the module docstring for the
    condition grammar, the `DecisionList`-not-`FlatRuleSet` reasoning, and how
    numeric conditions map onto `DataSpecBuilder.add_numeric`'s two
    independent `>=`/`<=` threshold families.

    Unlike `ObjectRuleImporter`'s `import_model(model, dataspec)`, which
    needs an explicit `dataspec` because a live model object carries no
    separate discovery step of its own, `parse` here builds and caches
    one automatically from `source` itself when none was given at
    construction -- there's no live model to inspect independently of
    the text, so a single auto-inferred pass is the natural default;
    pass `dataspec=` explicitly (e.g. one built by an earlier
    `infer_dataspec` call, to reuse across a train/test pair of runs, or
    a hand-built one to share with other imported rule sets) to bind
    against an existing `DataSpec` instead.
    """

    SOURCE = "weka.classifiers.rules.JRip"

    def __init__(self, dataspec: Optional[DataSpec] = None, placeholder_features: bool = False,
                 column_map: Optional[Sequence[int]] = None):
        self.dataspec = dataspec
        #: with `placeholder_features`: the `dataspec` feature each
        #: placeholder column ``f{i}`` is (`None`: feature ``i`` itself)
        self.column_map = column_map
        #: when True, `source`'s attributes are `f0/f1/...` placeholders
        #: (written by a `fit()` round-trip) that bind to `dataspec`'s
        #: columns by position -- see `_placeholder_rule_from_conditions`.
        self.placeholder_features = placeholder_features

    def infer_dataspec(self, source: str) -> DataSpec:
        """Discover a typed `DataSpec` directly from JRip's rule text --
        see the module docstring for how each condition's type is
        decided. Pass the result to a later `parse` call (as
        `dataspec=` at construction) to reuse it, e.g. against a
        companion test-set run's text.
        """
        return _infer_dataspec_from_conditions(_collect_conditions(source))

    def parse(self, source: str, data: Optional[Any] = None) -> DecisionList:
        """Parse `source` into a `DecisionList`, in JRip's own printed
        (first-match-wins) order, with the trailing condition-less rule
        becoming `default_prediction`. Builds and caches `self.dataspec` from
        `source` itself on first use if none was supplied to `__init__`
        (unless `placeholder_features=True`, which requires an explicit one).

        `data`, when given (only by a `fit()` round trip -- see the
        module's `_WekaRuleLearner._import`), is the exact
        `DataRepresentation` written out for JRip; each returned rule
        (and the default rule) gets its own measured `stats` against it.
        `None` (a bare `parse` call on captured text) leaves rules
        un-annotated, unchanged from before.
        """
        if self.dataspec is None and not self.placeholder_features:
            self.dataspec = self.infer_dataspec(source)
        builder = _rule_builder_for(self)
        dataspec = self.dataspec

        parsed = _collect_conditions(source)
        if not parsed:
            raise ValueError("no JRip rule lines found in source (expected lines containing '=>')")

        rules, default_target = _rules_and_default(parsed, dataspec, builder)
        return _finalize_rulelist(self, rules, default_target, data)


register_importer("weka_jrip", JRipImporter)


# PART's own rule terminator: "<conditions>: target (covered[.0][/errors[.0]])"
# -- the tail (possibly empty) of a rule's last line, before its stats.
_PART_TERMINATOR_RE = re.compile(
    r"^(?P<tail>.*?):\s*(?P<target>\S(?:.*?\S)?)\s*"  # target may contain spaces
    r"\(\s*(?P<covered>[\d.]+)\s*(?:/\s*(?P<errors>[\d.]+)\s*)?\)\s*$"
)
# a non-final condition line always ends in a trailing " AND"
_PART_AND_SUFFIX_RE = re.compile(r"^(.*?)\s+AND$")


def _collect_part_conditions(source: str) -> List[ParsedRule]:
    """Every PART rule block in `source` -- blank-line-delimited; a
    block is a rule if its last (non-blank) line matches
    ``"<conditions>: target (stats)"`` -- parsed into `(conditions,
    target)`, the same shape `_collect_conditions` produces for JRip
    (the trailing stats suffix must still match, to confirm this really
    is a rule block, but its numbers aren't kept -- see the module
    docstring). Safe against a full raw console dump the same way: no
    other section (accuracy tables, confusion matrix, the header banner)
    ever ends a blank-line-delimited block that way.
    """
    parsed: List[ParsedRule] = []
    for block in re.split(r"\n\s*\n", source):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        m = _PART_TERMINATOR_RE.match(lines[-1])
        if not m:
            continue  # not a rule block
        condition_lines = lines[:-1]
        tail = m.group("tail").strip()
        if tail:
            condition_lines = condition_lines + [tail]

        conditions = []
        for i, line in enumerate(condition_lines):
            and_m = _PART_AND_SUFFIX_RE.match(line)
            if and_m is None and i < len(condition_lines) - 1:
                raise ValueError(
                    f"PART condition line doesn't end in ' AND' as expected: {line!r}"
                )
            cond_text = and_m.group(1).strip() if and_m else line
            conditions.append(_split_condition(cond_text))

        target = m.group("target")
        parsed.append((conditions, target))
    return parsed


class PARTImporter(StringRuleImporter):
    """Extracts a `pyrulearn.models.DecisionList` from
    `weka.classifiers.rules.PART`'s printed decision-list text (or a
    full raw Weka console dump containing it). See the module docstring
    for PART's block-structured condition grammar (unrelated to JRip's
    line format) and why it's also a `DecisionList`.

    Same `dataspec=`/auto-discovery convention as `JRipImporter`: pass
    one at construction to bind against an existing `DataSpec`, or leave
    it unset to discover and cache one from `source` on first `parse`.
    """

    SOURCE = "weka.classifiers.rules.PART"

    def __init__(self, dataspec: Optional[DataSpec] = None, placeholder_features: bool = False,
                 column_map: Optional[Sequence[int]] = None):
        self.dataspec = dataspec
        #: with `placeholder_features`: the `dataspec` feature each
        #: placeholder column ``f{i}`` is (`None`: feature ``i`` itself)
        self.column_map = column_map
        self.placeholder_features = placeholder_features

    def infer_dataspec(self, source: str) -> DataSpec:
        """Discover a typed `DataSpec` directly from PART's rule text --
        same condition-type rules as `JRipImporter.infer_dataspec`, just
        fed PART's own block-parsed conditions instead of JRip's.
        """
        return _infer_dataspec_from_conditions(_collect_part_conditions(source))

    def parse(self, source: str, data: Optional[Any] = None) -> DecisionList:
        """Parse `source` into a `DecisionList`, in PART's own printed
        (first-match-wins) order, with the trailing condition-less rule
        becoming `default_prediction`. Builds and caches `self.dataspec` from
        `source` itself on first use if none was supplied to `__init__`
        (unless `placeholder_features=True`, which requires an explicit one).

        `data`, when given (only by a `fit()` round trip), gets each
        returned rule its own measured `stats` against it -- see
        `JRipImporter.parse`.
        """
        if self.dataspec is None and not self.placeholder_features:
            self.dataspec = self.infer_dataspec(source)
        builder = _rule_builder_for(self)
        dataspec = self.dataspec

        parsed = _collect_part_conditions(source)
        if not parsed:
            raise ValueError(
                "no PART rule blocks found in source (expected blank-line-delimited "
                "blocks ending in '<conditions>: target (stats)')"
            )

        rules, default_target = _rules_and_default(parsed, dataspec, builder)
        return _finalize_rulelist(self, rules, default_target, data)


register_importer("weka_part", PARTImporter)


_J48_HEADER_RE = re.compile(r"^J48 \S+ tree$")
_J48_DIVIDER_RE = re.compile(r"^-+$")
_J48_INDENT_RE = re.compile(r"^(?:\|   )*")
# a leaf line: "<conditions>: target (stats)" -- same terminator shape as
# PART's, reused via the same regex family rather than redefined
_J48_LEAF_RE = _PART_TERMINATOR_RE


def _extract_j48_tree_lines(source: str) -> List[str]:
    """Locate and return just the tree's own lines from `source` (the
    isolated tree text, or a full raw Weka console dump). Unlike JRip's
    ``"=>"`` filter or PART's block-terminator regex, J48's tree can't
    be picked out of a whole dump by content alone: the confusion
    matrix's own column layout can produce a line that reads as *both*
    a ``|   ``-indented node *and* a valid ``attr = value`` condition
    (e.g. `` 110  15 |   a = pos``), which would otherwise be misparsed
    as a depth-1 split -- so this anchors explicitly on the header
    instead.
    """
    lines = source.splitlines()
    start = None
    for i, line in enumerate(lines):
        if _J48_HEADER_RE.match(line.strip()) and i + 1 < len(lines) and _J48_DIVIDER_RE.match(lines[i + 1].strip()):
            start = i + 2  # right after header, divider
            break
    if start is None:
        raise ValueError(
            "no 'J48 ... tree' header found in source -- expected a line matching "
            "'J48 <word> tree' immediately followed by a '----' divider line"
        )
    # normally one blank line separates the divider from the tree body --
    # except for a *single-leaf* tree (the root itself is a leaf, no split
    # at all: "no attribute is predictive enough to split on", or the
    # training fold's own class column ends up constant), where Weka
    # prints that one leaf line immediately after the divider with no
    # blank line first. Skip at most one blank line here rather than
    # assuming it's always there, or a single-leaf tree's one real line
    # gets skipped too, leaving nothing (confirmed directly: `"J48 pruned
    # tree\n------------------\n: 1 (499.0/119.0)\n\nNumber of Leaves..."`).
    if start < len(lines) and not lines[start].strip():
        start += 1
    end = start
    while end < len(lines) and lines[end].strip():
        end += 1
    return lines[start:end]


def _collect_j48_conditions(source: str) -> List[ParsedRule]:
    """Every leaf in `source`'s J48 tree, parsed into `(conditions,
    target)` -- the same shape `_collect_conditions`/
    `_collect_part_conditions` produce, so it feeds the same shared
    `_infer_dataspec_from_conditions`/`_rule_from_conditions`. Walks the
    tree's lines with a depth-indexed stack of still-open ancestor
    conditions: `del stack[depth:]` truncates back to the current line's
    depth before processing it (a no-op when descending one level
    deeper, the actual backtrack when a sibling appears at the same or
    a shallower depth than the previous line) -- correct regardless of
    how many siblings a split has, since each sibling is just another
    line at that same depth.
    """
    stack: List[Tuple[str, str, str]] = []
    parsed: List[ParsedRule] = []
    for raw_line in _extract_j48_tree_lines(source):
        indent = _J48_INDENT_RE.match(raw_line).group(0)
        depth = len(indent) // 4
        content = raw_line[len(indent):].strip()
        del stack[depth:]

        leaf_m = _J48_LEAF_RE.match(content)
        if leaf_m:
            cond_text = leaf_m.group("tail").strip()
            conditions = list(stack)
            if cond_text:
                conditions.append(_split_condition(cond_text))
            target = leaf_m.group("target")
            parsed.append((conditions, target))
        else:
            stack.append(_split_condition(content))
    return parsed


class J48Importer(StringRuleImporter):
    """Extracts a `pyrulearn.models.DisjointRuleSet` from
    `weka.classifiers.trees.J48`'s printed pruned-tree text (or a full
    raw Weka console dump containing it) -- one `Rule` per leaf, its
    conditions the conjunction of every split condition on its
    root-to-leaf path. See the module docstring for the tree grammar and
    why this produces a `DisjointRuleSet`, not a `DecisionList`.

    Same `dataspec=`/auto-discovery convention as `JRipImporter`/
    `PARTImporter`: pass one at construction to bind against an existing
    `DataSpec`, or leave it unset to discover and cache one from
    `source` on first `parse`.
    """

    SOURCE = "weka.classifiers.trees.J48"

    def __init__(self, dataspec: Optional[DataSpec] = None, placeholder_features: bool = False,
                 column_map: Optional[Sequence[int]] = None):
        self.dataspec = dataspec
        #: with `placeholder_features`: the `dataspec` feature each
        #: placeholder column ``f{i}`` is (`None`: feature ``i`` itself)
        self.column_map = column_map
        self.placeholder_features = placeholder_features

    def infer_dataspec(self, source: str) -> DataSpec:
        """Discover a typed `DataSpec` directly from J48's tree text --
        same condition-type rules as `JRipImporter.infer_dataspec`.
        """
        return _infer_dataspec_from_conditions(_collect_j48_conditions(source))

    def parse(self, source: str, data: Optional[Any] = None) -> DisjointRuleSet:
        """Parse `source` into a `DisjointRuleSet`, one `Rule` per leaf
        -- no `default_rule`, since a tree's leaves are already
        exhaustive by construction. Builds and caches `self.dataspec`
        from `source` itself on first use if none was supplied to
        `__init__` (unless `placeholder_features=True`, which requires an
        explicit one).

        `data`, when given (only by a `fit()` round trip), gets each
        returned rule its own measured `stats` against it -- see
        `JRipImporter.parse`.
        """
        if self.dataspec is None and not self.placeholder_features:
            self.dataspec = self.infer_dataspec(source)
        builder = _rule_builder_for(self)
        dataspec = self.dataspec

        parsed = _collect_j48_conditions(source)
        if not parsed:
            raise ValueError("no leaves found in J48 tree text")

        rules = [builder(conditions, target, dataspec) for conditions, target in parsed]
        rules = self._stamp_rule_provenance(rules, n_rules=len(rules))
        rules = self._stamp_rule_stats(rules, data)
        return self._stamp_provenance(DisjointRuleSet(rules), n_rules=len(rules))


register_importer("weka_j48", J48Importer)


# ============================================================ running Weka ====

def _resolve_weka(jar: Optional[str], java: Optional[str]) -> Tuple[str, str]:
    """`(jar, java)` for a Weka subprocess: `jar` from the argument or
    ``$WEKA_JAR``; `java` from the argument, else ``$WEKA_JAVA``, else
    ``java`` on `PATH`. Raises if no jar is available. (Same shape as
    `lord._lord_classpath`.)"""
    jar = jar or os.environ.get("WEKA_JAR")
    if not jar:
        raise RuntimeError(
            "no weka.jar available -- pass jar= to the learner or set $WEKA_JAR "
            "(e.g. 'C:/Program Files/Weka-3-8-7/weka.jar'); $WEKA_JAVA overrides the "
            "java executable"
        )
    return jar, (java or os.environ.get("WEKA_JAVA") or "java")


def _weka_cli_options(options: Optional[Dict[str, Any]]) -> List[str]:
    """`{"F": 3, "O": 2, "P": True}` -> ``["-F", "3", "-O", "2", "-P"]`` --
    a bare flag for a ``True`` value, ``-k v`` otherwise; ``False``/``None``
    drops the flag."""
    out: List[str] = []
    for key, value in (options or {}).items():
        flag = key if key.startswith("-") else f"-{key}"
        if value is True:
            out.append(flag)
        elif value not in (False, None):
            out += [flag, str(value)]
    return out


def _write_boolean_arff(X: Any, y: Any, path: str, block_cells: int = 1 << 22) -> None:
    """Write the Boolean matrix `X` (``n x k``) and labels `y` as an ARFF
    file for Weka -- the same text `pyrulearn.data.io.write_arff` writes
    for the same data (its line endings are the platform's, these always
    ``\n``) (features ``f0..f{k-1}`` as nominal
    attributes with the values ``False``/``True`` that occur in the
    column, then ``class`` with `y`'s distinct values, sorted), but built
    as byte arrays a block of rows at a time: the general writer converts
    and quotes every cell on its own, which on 32,000 x 278 cells took
    17 s, longer than many of the fits it was writing for.

    The value names stay ``False``/``True`` on purpose: JRip seeds its
    random grow/prune splits partly from the text of an instance
    (`weka.core.Instances.getRandomNumberGenerator`), so ``0``/``1``
    would make it learn different rules from the same data."""
    import numpy as np

    from ..data.io import _arff_quote

    X = np.asarray(X, dtype=bool)
    n, k = X.shape
    labels, codes = np.unique(np.asarray(y).astype(str), return_inverse=True)
    quoted = [_arff_quote(str(c)).encode("utf-8") for c in labels]
    any_true, all_true = X.any(axis=0), X.all(axis=0)
    domains = ["{" + ",".join((["False"] if not t_all else []) + (["True"] if t_any else [])) + "}"
               for t_any, t_all in zip(any_true, all_true)]
    header = (["@relation pyrulearn", ""]
              + [f"@attribute f{i} {d}" for i, d in enumerate(domains)]
              + [f"@attribute class {{{','.join(q.decode('utf-8') for q in quoted)}}}", "", "@data", ""])
    tokens = ((False, b"False,"), (True, b"True,"))
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("utf-8"))
        step = max(1, block_cells // max(k, 1))
        for start in range(0, n, step):
            block = X[start:start + step]
            widths = np.where(block, 5, 6).ravel()          # len("True,"), len("False,")
            ends = np.cumsum(widths)
            starts = ends - widths
            flat = np.empty(int(ends[-1]) if len(ends) else 0, dtype=np.uint8)
            cells = block.ravel()
            for value, token in tokens:
                at = starts[cells == value]
                for j, ch in enumerate(token):
                    flat[at + j] = ch
            text = flat.tobytes()
            row_ends = ends[k - 1::k] if k else np.zeros(len(block), dtype=np.int64)
            row_starts = np.concatenate(([0], row_ends[:-1]))
            f.write(b"".join(text[a:b] + quoted[c] + b"\n"
                             for a, b, c in zip(row_starts, row_ends, codes[start:start + step])))


def run_weka(
    weka_class: str,
    X: Any,
    y: Any,
    *,
    jar: Optional[str] = None,
    java: Optional[str] = None,
    timeout: Optional[float] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Fit `weka_class` (e.g. ``"weka.classifiers.rules.JRip"``) on the
    already-Boolean matrix `X` / labels `y` via ``java -cp <weka.jar>``,
    and return its printed stdout -- exactly the text this module's
    importers parse.

    `X`'s columns are written as ``{False,True}`` nominal ARFF attributes
    with safe placeholder names ``f0..fN`` (a `DataSpec`'s own ``age>=30``-style
    names would collide with the parsers' operator regexes -- see
    `write_arff`'s docstring) by `_write_boolean_arff`, so the caller pairs
    this with ``Importer(dataspec=<real spec>, placeholder_features=True)``
    to bind the rules back by position. ``-no-cv`` skips Weka's own
    cross-validation; `options` passes classifier flags (see
    `_weka_cli_options`). Raises `RuntimeError` (stderr included) if Weka
    exits non-zero or prints no model.
    """
    jar, java = _resolve_weka(jar, java)
    with tempfile.TemporaryDirectory(prefix="weka_") as d:
        arff = os.path.join(d, "train.arff")
        _write_boolean_arff(X, y, arff)
        cmd = [java, "-Duser.language=en", "-Duser.country=US", "-cp", jar, weka_class,
               "-t", arff, "-no-cv", *_weka_cli_options(options)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    # Weka's `-t` always prints an "=== ... ===" evaluation section on success;
    # its CLI wrapper is otherwise unhelpfully generic (a bad ARFF surfaces only
    # as "Can't open file"), so echo stderr on failure.
    if result.returncode != 0 or "=== " not in result.stdout:
        raise RuntimeError(
            f"{weka_class} produced no usable model (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def weka_build_time(text: str) -> Optional[float]:
    """The seconds Weka reports for building its model ("Time taken to
    build model: 0.05 seconds") in its printed output, or `None` if it
    doesn't report one."""
    m = re.search(r"Time taken to build model:\s*([\d.]+)\s*seconds", text)
    return float(m.group(1)) if m else None


class _WekaRuleLearner(RelabelingExternalLearner):
    """Shared base: write the training data to ARFF, run a Weka rule/tree
    classifier as a subprocess, read its printed rules back through the
    matching `StringRuleImporter` (`placeholder_features=True`, so rules
    bind to the training `DataSpec` by column position).

    `fit(data)` -> `NATIVE_MODEL`; `fit(data, model=ConceptSet |
    ConceptCascade | PairwiseModel)` for the binary-decomposition
    variants (via `RelabelingExternalLearner`). `jar=` / `java=` locate
    the JVM (else ``$WEKA_JAR`` / ``$WEKA_JAVA``); ``**weka_options`` are
    passed through as classifier CLI flags (JRip's ``-F``/``-N``/``-O``,
    J48/PART's ``-C``/``-M``, ...).

    `native_negation=True` hands Weka only the features that are not
    negations of others (`DataRepresentation.without_negations`): Weka
    tests a Boolean column both for ``True`` and for ``False`` anyway, so
    its own ``= False`` tests provide exactly the negations (and land on
    the paired negation features again on import) -- the same conditions
    as with them, but without evaluating each one twice. Default `False`:
    every column as it is.
    """

    WEKA_CLASS: str
    IMPORTER: type

    def __init__(self, *, jar: Optional[str] = None, java: Optional[str] = None,
                 timeout: Optional[float] = None, native_negation: bool = False, **weka_options: Any):
        self.jar = jar
        self.java = java
        self.timeout = timeout
        self.native_negation = native_negation
        self.weka_options = weka_options

    def fit_external(self, X, y, feature_names=None):
        text = run_weka(self.WEKA_CLASS, X, y, jar=self.jar, java=self.java,
                        timeout=self.timeout, options=self.weka_options)
        # the run time is Weka's own "Time taken to build model" (10 ms
        # resolution), not the subprocess's wall time, which also includes
        # starting the JVM, reading the ARFF file and printing the model
        self._reported_run_time = weka_build_time(text)
        return text

    def _import(self, data):
        if self.native_negation:
            _, keep = data.spec.without_negations()
            sub = data.without_negations()
        else:
            keep, sub = None, data
        text = self.fit_external(sub.X, sub.y, feature_names=sub.spec.feature_names)
        return self.IMPORTER(dataspec=data.spec, placeholder_features=True,
                             column_map=keep).parse(text, data=data)


class WekaJRip(_WekaRuleLearner):
    """`weka.classifiers.rules.JRip` (Weka's RIPPER). `fit(data)` ->
    `pyrulearn.models.DecisionList` (JRip is an ordered, multi-class
    decision list with a trailing default). `fit(data, model=SingleRule,
    label="a")` -- see `_fit_one_rule`."""

    WEKA_CLASS = "weka.classifiers.rules.JRip"
    IMPORTER = JRipImporter
    NATIVE_MODEL = models.DecisionList

    @produces(models.SingleRule)
    def _fit_one_rule(self, data, *, label=None) -> "models.SingleRule":
        """`fit(data, model=SingleRule, label="a")` -- the first rule
        `label`'s own segment of JRip's covering loop produces: run the
        full multiclass fit, then take the first rule in JRip's own
        printed order whose target is `label`. JRip processes one class
        at a time, each fully covered before the next, so that rule is
        exactly what `label`'s covering-loop segment would find first --
        the same "first rule of a covering run" reading `SeCo`/`Pypper`
        use for their own native `SingleRule`, not a best-of-many pick
        (JRip's induction has no per-class covering primitive we could
        call directly, unlike our own SeCo family, so this is the
        closest equivalent that doesn't invent a selection criterion the
        algorithm itself doesn't make).

        `label` may legitimately get no explicit rule at all -- JRip
        always leaves exactly one class (typically the most frequent) as
        the trailing catch-all default, with zero rules of its own; that
        falls back to an empty (always-true) rule for `label`, the same
        "search found nothing" fallback `SeCo`/`Pypper`'s own
        `_fit_one_rule` uses.
        """
        if label is None:
            raise ValueError("model=SingleRule needs label=")
        decision_list = self._fit_native(data)
        for r in decision_list.rules:
            if r.target == label:
                sr = models.SingleRule(r.rule, default_prediction=models.MajorityClass(data))
                sr.set_stats(data)
                return sr
        empty = Rule([], target=label, dataspec=data.spec)
        sr = models.SingleRule(empty, default_prediction=models.MajorityClass(data))
        sr.set_stats(data)
        return sr


class WekaPART(_WekaRuleLearner):
    """`weka.classifiers.rules.PART` (Frank & Witten's partial-C4.5 rule
    learner). `fit(data)` -> `pyrulearn.models.DecisionList`."""

    WEKA_CLASS = "weka.classifiers.rules.PART"
    IMPORTER = PARTImporter
    NATIVE_MODEL = models.DecisionList


class WekaJ48(_WekaRuleLearner):
    """`weka.classifiers.trees.J48` (Weka's C4.5). `fit(data)` ->
    `pyrulearn.models.DisjointRuleSet` (one rule per leaf; a tree's leaves
    partition the space)."""

    WEKA_CLASS = "weka.classifiers.trees.J48"
    IMPORTER = J48Importer
    NATIVE_MODEL = models.DisjointRuleSet

# -- naming ------------------------------------------------------------------

_WekaRuleLearner.TOOL = "Weka"
