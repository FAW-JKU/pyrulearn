"""
demos/refinement_heuristics.py
==============================

How a separate-and-conquer learner should judge the refinements of a
rule: from the origin of the coverage space or from the rule being
refined, by what a refinement covers or by what it excludes, and how
the rule on the refinement path is chosen. The covering learner and the
datasets are those of `heuristic_comparison` (after Janssen &
Fürnkranz, 2010): greedy top-down refinement until no refinement is
left (`BeamSearch(beam_width=1)`), the best rule on the path by a
selection heuristic, no stopping criterion, no pruning, ordered class
binarization. It shares that demo's result cache, so the learners both
run are computed once.

Two operations change a heuristic ``h`` for refinement:

- **relativization** (`Relative(h)`): ``h`` scored among the examples of
  the rule being refined (its parent), whose class distribution is the
  prior -- the parent is the universal rule of the frame;
- **inversion** (`Inverted(h)`): ``h(N - n, P - p)``, a rule judged by
  the examples it excludes, with the classes exchanged.

Stecher, Janssen & Fürnkranz's inverted heuristics combine both,
`Relative(Inverted(h))`. Sections:

1. **m-estimates for refinement and for selection** -- `MEstimate(m_r)`
   refines, `MEstimate(m_s)` selects, for every pair of m in `MATRIX_M`:
   absolute and relative, regular and inverted (four m x m matrices; the
   first two as in the paper's section on inverted heuristics).
2. **Absolute vs. relative, regular vs. inverted** -- the same 2 x 2 for
   the other heuristics whose scores depend on the frame (WRA, relative
   cost, F-measure, correlation, Foil gain), and Laplace, which doesn't.
3. **Relative heuristics and m** -- the best selection per m for
   refinement, up to m -> infinity (WRA).
4. **Beam search** -- scores relative to different parents have no
   common scale: beam width 1 vs. 5, levels (`Relative(h)`) vs.
   improvements (`Relative(Gain(h))`).
5. **Weighted gains** -- ``weight(r) * (base(r) - base(r0))``, the gain
   over the universal rule of the frame (`WeightedGain`), with Klösgen's
   weights ``c^a`` (``c`` the covered examples or positives, ``a`` in 0,
   0.5, 1, 2) and a linear (precision) or logarithmic (information
   content) base, absolute and relative: WRA and Foil gain are two of
   them.

Run: `python demos/refinement_heuristics.py` with no arguments is the
**quick** default (`heuristic_comparison`'s quick datasets, 3-fold,
fewer m); its report/plots go to ``refinement_heuristics_quick_*`` and
are not checked in. `--full` runs `heuristic_comparison`'s datasets
with its folds and writes the canonical ``refinement_heuristics_report.md``
(plots in ``refinement_heuristics_plots/``). Needs the `experiments`
extra.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import heuristic_comparison as hc
from heuristic_comparison import HeuristicSeCo
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.evaluation import RuleStats
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import format_accuracy, format_mean_count, render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import mean_rank, win_counts
from pyrulearn.heuristics import (
    Correlation, Coverage, CoveredPositives, FBeta, FoilGain, Gain, InformationContent, Inverted, Laplace,
    LinearCostRates, MEstimate, Precision, Adaptive, RuleHeuristic, Support, WeightedGain, WRAcc,
)
from pyrulearn.learners.seco import BeamSearch, GainAscentHillClimbing, SeCo, SingleRuleLearner
from pyrulearn.models import ConceptCascade, ConceptModel
from pyrulearn.pruning import ThresholdPrePruning
from pyrulearn.heuristics import CoverageDifference

HERE = os.path.dirname(__file__)
REPORT_PATH = os.path.join(HERE, "refinement_heuristics_report.md")
QUICK_REPORT_PATH = os.path.join(HERE, "refinement_heuristics_quick_report.md")
PLOTS_DIR = os.path.join(HERE, "refinement_heuristics_plots")
CACHE_DIR = hc.CACHE_DIR          # shared: the learners both demos run are computed once

MATRIX_M = hc.MATRIX_M            # m for refinement x m for selection
QUICK_MATRIX_M = (1, 4, 16)
M_REFINE = 16                     # m of the m-estimates in sections 2 and 4
SECTION2_SELECT = MATRIX_M        # the selections of section 2 (every m of the matrix)
BEAM_SELECT = (4, 8, 16)          # the selections of sections 4 and 5
#: the selections a "best selection" picks from in sections 2 and 3 and the ranks
REPORT_SELECT = SECTION2_SELECT
#: the one selection of the "fixed selection" tables of sections 2 and 5, next to the best
#: selection: with it, only the refinement heuristics differ (in BEAM_SELECT, so that the
#: gains have it, too)
FIXED_SELECT = 8
QUICK_SELECT = (4, 16)
QUICK_FIXED_SELECT = 4
BEAM_WIDTH = 5

N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


def _m(m) -> str:
    return hc._m_text(m)


class Power(RuleHeuristic):
    """``base ** exponent`` -- e.g. the square root of `Support`, the
    weight of Klösgen's quality function sqrt(coverage) * (precision -
    prior)."""

    def __init__(self, base: RuleHeuristic, exponent: float):
        self.base = base
        self.exponent = exponent

    def score(self, stats: RuleStats) -> float:
        return float(self.base.score(stats)) ** self.exponent

    def batch_score(self, stats: RuleStats) -> np.ndarray:
        return np.asarray(self.base.batch_score(stats), dtype=float) ** self.exponent

    def __repr__(self) -> str:
        return f"{self.base!r}^{self.exponent:g}"


class LabeledSeCo(SeCo):
    """`heuristic_comparison`'s covering learner with any search and
    heuristic, named by `label` (the learner's cache key, as there)."""

    def __init__(self, label: str, heuristic, selection=None, search=None, random_state: Optional[int] = None):
        extra = {} if selection is None else {"selection": selection}
        super().__init__(
            single_rule_learner=SingleRuleLearner(heuristic=heuristic,
                                                  search=search if search is not None else BeamSearch(beam_width=1),
                                                  **extra),
            stop_covering=ThresholdPrePruning(CoverageDifference(), 0, operator="<="),
            random_state=random_state,
        )
        self.label = label

    @property
    def display_name(self) -> str:
        return self.label

    def _provenance_params(self):
        return {"label": self.label}

    def _default_model(self, data) -> type:
        return ConceptModel if self.target_class is not None else ConceptCascade


# ------------------------------------------------------------ variants ----

#: the four ways to refine by a heuristic, and their label prefixes; the
#: relative inverted one is `heuristic_comparison`'s "Inverted(...)"
FRAMES = (("absolute", ""), ("relative", "Relative"), ("absolute inverted", "absolute Inverted"),
          ("relative inverted", "Inverted"))


def _refiner(kind: str, h: RuleHeuristic, relative_h: Optional[RuleHeuristic] = None) -> RuleHeuristic:
    """`h` refining as `kind`; `relative_h` is the relative heuristic if
    it isn't `Relative(h)` (Foil gain)."""
    relative = relative_h if relative_h is not None else Adaptive(h)
    if kind == "absolute":
        return h
    if kind == "relative":
        return relative
    return Inverted(h) if kind == "absolute inverted" else Inverted(relative)


def _name(kind: str, name: str) -> str:
    prefix = dict(FRAMES)[kind]
    return f"{prefix}({name})" if prefix else name


def matrix_label(kind: str, m_refine, m_select) -> str:
    """Section 1's learner labels: `heuristic_comparison`'s for the absolute
    and the relative inverted m-estimate (the same learners)."""
    if kind == "absolute":
        return hc._matrix_label(False, m_refine, m_select)
    if kind == "relative inverted":
        return hc._matrix_label(True, m_refine, m_select)
    return f"refine {_name(kind, f'm={_m(m_refine)}')}, select m={_m(m_select)}"


def build_matrix_learners(ms) -> List[SeCo]:
    learners = []
    for kind, _ in FRAMES:
        for m_r in ms:
            for m_s in ms:
                label = matrix_label(kind, m_r, m_s)
                if kind == "absolute" and not label.startswith("refine "):
                    learners.append(HeuristicSeCo(label, MEstimate(m=m_r), random_state=hc.RANDOM_STATE))
                    continue
                selection = None if kind == "absolute" and m_r == m_s else MEstimate(m=m_s)
                learners.append(HeuristicSeCo(label, _refiner(kind, MEstimate(m=m_r)), random_state=hc.RANDOM_STATE,
                                              selection=selection))
    return learners


#: section 2: name -> (absolute heuristic, its relative version or None,
#: the frames that differ from the others -- inverting a symmetric
#: heuristic changes nothing, relativizing one that doesn't depend on the
#: frame neither)
def _section2_heuristics():
    all_frames = tuple(kind for kind, _ in FRAMES)
    return {
        "WRA": (WRAcc(), None, ("absolute", "relative")),
        "relative cost (c_r=0.342)": (LinearCostRates(cost_ratio=hc._cost_ratio(0.342)), None, all_frames),
        "F-measure (beta=0.5)": (FBeta(beta=0.5), None, all_frames),
        "Correlation": (Correlation(), None, ("absolute", "relative")),
        "Foil gain": (WeightedGain(CoveredPositives(), InformationContent()), FoilGain(), all_frames),
        "Laplace": (Laplace(), None, ("absolute", "absolute inverted", "relative inverted")),
    }


def section2_label(kind: str, name: str, m_select) -> str:
    return f"refine {_name(kind, name)}, select m={_m(m_select)}"


def build_section2_learners(selects) -> List[SeCo]:
    learners = []
    for name, (h, relative_h, kinds) in _section2_heuristics().items():
        for kind in kinds:
            for m_s in selects:
                learners.append(LabeledSeCo(section2_label(kind, name, m_s), _refiner(kind, h, relative_h),
                                            selection=MEstimate(m=m_s), random_state=hc.RANDOM_STATE))
    # Foil's own way: climb while the gain is positive, return the rule it ends at
    learners.append(LabeledSeCo("Foil gain", FoilGain(), search=GainAscentHillClimbing(),
                                random_state=hc.RANDOM_STATE))
    learners.append(LabeledSeCo("Foil gain over the root", WeightedGain(CoveredPositives(), InformationContent()),
                                search=GainAscentHillClimbing(), random_state=hc.RANDOM_STATE))
    return learners


#: section 4: the beam's refinement heuristics, name -> heuristic
def _beam_heuristics() -> Dict[str, RuleHeuristic]:
    m = MEstimate(m=M_REFINE)
    return {
        f"m={_m(M_REFINE)}": m,
        f"Relative(m={_m(M_REFINE)})": Adaptive(m),
        f"Inverted(m={_m(M_REFINE)})": Adaptive(Inverted(m)),
        f"Relative(Gain(m={_m(M_REFINE)}))": Adaptive(Gain(m)),
        f"Relative(Gain(absolute Inverted(m={_m(M_REFINE)})))": Adaptive(Gain(Inverted(m))),
        "WRA": WRAcc(),
        "Relative(WRA)": Adaptive(WRAcc()),
    }


def beam_label(width: int, name: str, m_select) -> str:
    if width == 1:   # sections 1 and 2's learners, where they have them
        if name == f"m={_m(M_REFINE)}":
            return hc._matrix_label(False, M_REFINE, m_select)
        if name == f"Relative(m={_m(M_REFINE)})":
            return matrix_label("relative", M_REFINE, m_select)
        if name == f"Inverted(m={_m(M_REFINE)})":
            return hc._matrix_label(True, M_REFINE, m_select)
        if name in ("WRA", "Relative(WRA)"):
            return section2_label("absolute" if name == "WRA" else "relative", "WRA", m_select)
        return f"refine {name}, select m={_m(m_select)}"
    return f"beam {width}: refine {name}, select m={_m(m_select)}"


def build_beam_learners(selects) -> List[SeCo]:
    learners = []
    for name, h in _beam_heuristics().items():
        for m_s in selects:
            for width in (1, BEAM_WIDTH):
                label = beam_label(width, name, m_s)
                if width == 1 and "Gain(" not in name:
                    continue   # built in section 1 or 2
                learners.append(LabeledSeCo(label, h, selection=MEstimate(m=m_s),
                                            search=BeamSearch(beam_width=width), random_state=hc.RANDOM_STATE))
    return learners


#: section 5: weights and bases of `WeightedGain`
def _weights() -> Dict[str, Optional[RuleHeuristic]]:
    """Klösgen's family of weights, ``c^a`` for a coverage ``c`` -- of
    the covered examples (``p + n``) or of the covered positives (``p``)
    -- and ``a`` in 0.5, 1, 2; ``a = 0`` is no weight. Within one frame, a
    rescaled coverage (`Support` for `Coverage`) ranks alike."""
    return {
        "1": None,
        "sqrt(p)": Power(CoveredPositives(), 0.5), "p": CoveredPositives(), "p^2": Power(CoveredPositives(), 2.0),
        "sqrt(p + n)": Power(Support(), 0.5), "p + n": Coverage(), "(p + n)^2": Power(Coverage(), 2.0),
    }


def _bases() -> Dict[str, RuleHeuristic]:
    return {"precision": Precision(), "information content": InformationContent()}


def gain_label(frame: str, weight: str, base: str, m_select) -> str:
    gain = f"gain of {base} weighted by {weight}"
    return f"refine {'Relative(' + gain + ')' if frame == 'relative' else gain}, select m={_m(m_select)}"


#: section 5's paired comparisons: (frame, weight, base) of two gains, named
GAIN_PAIRS = (
    (("absolute", "p + n", "precision"), ("absolute", "sqrt(p + n)", "precision"), "WRA vs. a weaker weight"),
    (("absolute", "p + n", "precision"), ("absolute", "p", "precision"), "WRA vs. weight p"),
    (("absolute", "p + n", "precision"), ("relative", "p + n", "precision"), "WRA vs. relative WRA"),
    (("relative", "p + n", "precision"), ("relative", "p^2", "precision"), "relative WRA vs. weight p^2"),
    (("relative", "p", "information content"), ("relative", "p + n", "information content"),
     "Foil gain vs. weight p + n"),
    (("relative", "p", "information content"), ("relative", "p + n", "precision"), "Foil gain vs. relative WRA"),
    (("absolute", "p", "information content"), ("absolute", "sqrt(p)", "information content"),
     "absolute Foil gain vs. weight sqrt(p)"),
    (("absolute", "p", "information content"), ("relative", "p", "information content"),
     "absolute vs. relative Foil gain"),
) + tuple(((frame, weight, "precision"), (frame, weight, "information content"),
           f"{frame} {weight}: precision vs. information content")
          for frame in ("absolute", "relative") for weight in ("sqrt(p)", "p", "p^2", "sqrt(p + n)", "p + n",
                                                               "(p + n)^2"))


def _paired(results, a: str, b: str):
    """Wins and losses of `b` over `a` across datasets (mean accuracy per
    dataset), and the two-sided Wilcoxon signed-rank test's and sign test's
    p-values (ties left out of the sign test)."""
    from scipy.stats import binomtest, wilcoxon
    means = results[results["learner"].isin([a, b])].groupby(["dataset", "learner"])["accuracy"].mean().unstack()
    means = means.dropna()
    diff = means[b] - means[a]
    wins, losses = int((diff > 1e-12).sum()), int((diff < -1e-12).sum())
    p_value = wilcoxon(means[b], means[a]).pvalue if wins + losses > 0 else 1.0
    sign_p = binomtest(wins, wins + losses).pvalue if wins + losses > 0 else 1.0
    return len(means), wins, losses, float(diff.mean()), p_value, sign_p


def build_gain_learners(selects) -> List[SeCo]:
    learners = []
    for weight_name, weight in _weights().items():
        for base_name, base in _bases().items():
            gain = WeightedGain(weight, base)
            for frame in ("absolute", "relative"):
                for m_s in selects:
                    learners.append(LabeledSeCo(gain_label(frame, weight_name, base_name, m_s),
                                                gain if frame == "absolute" else Adaptive(gain),
                                                selection=MEstimate(m=m_s), random_state=hc.RANDOM_STATE))
    return learners


def build_learners(quick: bool) -> List[SeCo]:
    ms = QUICK_MATRIX_M if quick else MATRIX_M
    selects = QUICK_SELECT if quick else SECTION2_SELECT
    beam_selects = QUICK_SELECT if quick else BEAM_SELECT
    learners = (build_matrix_learners(ms) + build_section2_learners(selects) + build_beam_learners(beam_selects)
                + build_gain_learners(beam_selects))
    unique: Dict[str, SeCo] = {}
    for learner in learners:
        unique.setdefault(learner.display_name, learner)
    return list(unique.values())


# ------------------------------------------------------------- report ----

MATRIX_MEASURES = (("accuracy", "Accuracy", format_accuracy),
                   ("n_rules", "Number of rules", format_mean_count),
                   ("conds_per_rule", "Conditions per rule", format_mean_count),
                   ("n_conditions", "Conditions in the rule set", format_mean_count))
#: section 1's matrices, in the order the paper's figure script reads them
MATRIX_TITLES = (("absolute", "Regular m-estimate"), ("relative inverted", "Inverted m-estimate"),
                 ("relative", "Relative m-estimate"), ("absolute inverted", "Absolute inverted m-estimate"))


def _means(results) -> pd.DataFrame:
    """Means across every dataset and fold; conditions per rule as in
    `heuristic_comparison`, the mean of each fold's ratio."""
    results = results.assign(conds_per_rule=results["n_conditions"] / results["n_rules"])
    return results.groupby("learner")[["accuracy", "n_rules", "n_conditions", "conds_per_rule"]].mean(
        numeric_only=True)


def _matrix(means, kind: str, measure: str, ms) -> pd.DataFrame:
    return pd.DataFrame([[means[measure].get(matrix_label(kind, r, c), np.nan) for c in ms] for r in ms],
                        index=list(ms), columns=list(ms))


def _matrix_table(matrix, fmt) -> str:
    rows = ["| refine \\ select | " + " | ".join(f"m={_m(c)}" for c in matrix.columns) + " |\n",
            "|---|" + "--:|" * len(matrix.columns) + "\n"]
    for r in matrix.index:
        cells = ["n/a" if np.isnan(matrix.loc[r, c]) else fmt(matrix.loc[r, c]) for c in matrix.columns]
        rows.append(f"| m={_m(r)} | " + " | ".join(cells) + " |\n")
    return "".join(rows) + "\n"


def _m_s(label: str) -> str:
    """The m for selection of a learner's label (the plain m-estimate's own m)."""
    if "select m=" in label:
        return label.rsplit("select m=", 1)[1]
    return label.rsplit("m=", 1)[1].rstrip(")")


def _best_over_selection(means, labels: Sequence[str]) -> Tuple[float, float, Optional[str]]:
    """The accuracy and size of the most accurate of `labels`, and its label."""
    have = [l for l in labels if l in means.index and np.isfinite(means.loc[l, "accuracy"])]
    if not have:
        return np.nan, np.nan, None
    best = max(have, key=lambda l: means.loc[l, "accuracy"])
    return means.loc[best, "accuracy"], means.loc[best, "n_conditions"], best


def _description(datasets, n_folds: int, quick: bool) -> str:
    mode = (f"**Quick run** ({len(datasets)} small datasets, {n_folds}-fold, m in "
            f"{{{', '.join(_m(m) for m in QUICK_MATRIX_M)}}}) -- a fast sanity check. Full run: "
            "`python demos/refinement_heuristics.py --full`.\n\n" if quick else
            f"**Full run** ({len(datasets)} datasets, {n_folds}-fold).\n\n")
    return f"""\
{mode}The covering learner and the datasets of the
[heuristic comparison](heuristic_comparison_report.md) (after Janssen &
Fürnkranz, 2010): greedy top-down refinement until no refinement is left
(`BeamSearch(beam_width=1)`, in section 4 also width {BEAM_WIDTH}), the
best rule on the refinement path by a separate selection heuristic, the
m-estimate (`SingleRuleLearner(h, selection=MEstimate(m_s))`), no
stopping criterion and no pruning; covering stops once a new rule covers
no more positives than negatives; ordered class binarization. Numeric
attributes are discretized per training fold
(`TreeDiscretizer(max_intervals={hc.MAX_INTERVALS})`); each fit is capped at
{hc.FIT_TIMEOUT:.0f}s, and a fold that times out fails the learner's whole
cross-validation on that dataset.

A heuristic `h` refines in one of four frames: **absolute** (`h`, judged
from the origin of the coverage space of the whole data), **relative**
(`Relative(h)`: among the examples of the rule being refined, its class
distribution as the prior), **absolute inverted** (`Inverted(h)`:
`h(N - n, P - p)`, judged by the examples a rule excludes, from the upper
right corner of the whole coverage space), and **relative inverted**
(`Relative(Inverted(h))`: `h(N' - n, P' - p)` among the parent's `P'`
positives and `N'` negatives -- the inverted heuristics of Stecher,
Janssen & Fürnkranz, ECML PKDD 2014). Scores relative to the parent have
no common scale across parents: the selection, always absolute, picks the
rule that is returned.

Measures: test accuracy, number of rules and conditions; means across
every dataset and fold.
"""


def write_report(results, datasets, n_folds: int, quick: bool, report_path: str) -> None:
    ms = QUICK_MATRIX_M if quick else MATRIX_M
    selects = QUICK_SELECT if quick else REPORT_SELECT
    beam_selects = QUICK_SELECT if quick else BEAM_SELECT
    fixed = QUICK_FIXED_SELECT if quick else FIXED_SELECT
    means = _means(results)

    def _fixed_cell(label: str) -> str:
        if label not in means.index:
            return "n/a"
        return f"{format_accuracy(means.loc[label, 'accuracy'])} ({format_mean_count(means.loc[label, 'n_conditions'])})"
    plots = "refinement_heuristics_plots"
    tag = "_quick" if quick else ""
    lines = ["# Refinement heuristics: absolute and relative, regular and inverted\n\n",
             render_setup_section(_description(datasets, n_folds, quick))]

    # 1. matrices
    lines += [
        "## m-estimates for refinement and for selection\n\n",
        "`MEstimate(m_r)` refines and `MEstimate(m_s)` selects the rule on the refinement path, "
        f"for every pair of m in {{{', '.join(_m(m) for m in ms)}}}, in each of the four frames; the "
        "diagonal of the regular (absolute) matrix is the plain m-estimate. Rows: m for refinement, "
        "columns: m for selection.\n\n",
        f"![the four matrices, accuracy]({plots}/matrices_accuracy{tag}.png)\n\n",
        f"![the four matrices, conditions per rule]({plots}/matrices_conds_per_rule{tag}.png)\n\n",
    ]
    for measure, title, fmt in MATRIX_MEASURES:
        lines.append(f"### {title}\n\n")
        for kind, name in MATRIX_TITLES:
            lines.append(f"{name} for refinement:\n\n" + _matrix_table(_matrix(means, kind, measure, ms), fmt))

    # 2. the 2 x 2 per heuristic
    lines += [
        "## Absolute vs. relative, regular vs. inverted\n\n",
        f"Each heuristic refines in every frame where it differs, with the m-estimate selecting "
        f"(m_s in {{{', '.join(_m(m) for m in selects)}}}); shown is the best selection, "
        "with its m_s. Inverting a symmetric heuristic (WRA, correlation) changes nothing, and "
        "relativizing one that doesn't depend on the frame (Laplace) neither -- unless it is inverted, "
        "as the inverted Laplace estimate depends on it. The absolute Foil "
        "gain is the gain over the universal rule (`WeightedGain(CoveredPositives(), "
        "InformationContent())`), the relative one Foil's own (`FoilGain()`). The m-estimate "
        f"(m_r = {_m(M_REFINE)}) is from section 1.\n\n",
        f"![absolute to relative]({plots}/frames{tag}.png)\n\n",
        "| heuristic | " + " | ".join(kind for kind, _ in FRAMES) + " |\n|---|" + "--:|" * len(FRAMES) + "\n",
    ]
    rows = dict(_section2_heuristics())
    for name in [f"m-estimate (m={_m(M_REFINE)})"] + list(rows):
        cells = []
        for kind, _ in FRAMES:
            if name.startswith("m-estimate"):
                labels = [matrix_label(kind, M_REFINE, s) for s in selects]
            elif kind in rows[name][2]:
                labels = [section2_label(kind, name, s) for s in selects]
            else:
                labels = []
            acc, conds, best = _best_over_selection(means, labels)
            cells.append("--" if best is None else
                         f"{format_accuracy(acc)} ({format_mean_count(conds)}; m_s={_m_s(best)})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |\n")
    lines.append("\nCells: accuracy (conditions in the rule set; the best m for selection). "
                 "--: the same as the frame to its left.\n\n")
    lines += [
        f"### Fixed selection (m_s = {_m(fixed)})\n\n",
        f"As above, with the m-estimate selecting with m_s = {_m(fixed)} for every heuristic and "
        "frame, so that only the refinement heuristics differ.\n\n",
        "| heuristic | " + " | ".join(kind for kind, _ in FRAMES) + " |\n|---|" + "--:|" * len(FRAMES) + "\n",
    ]
    for name in [f"m-estimate (m={_m(M_REFINE)})"] + list(rows):
        cells = []
        for kind, _ in FRAMES:
            if name.startswith("m-estimate"):
                cells.append(_fixed_cell(matrix_label(kind, M_REFINE, fixed)))
            elif kind in rows[name][2]:
                cells.append(_fixed_cell(section2_label(kind, name, fixed)))
            else:
                cells.append("--")
        lines.append(f"| {name} | " + " | ".join(cells) + " |\n")
    lines.append("\nCells: accuracy (conditions in the rule set). --: the same as the frame to its left.\n\n")
    foil = [l for l in ("Foil gain", "Foil gain over the root") if l in means.index]
    if foil:
        lines.append("Foil's own search, climbing while the gain is positive and returning the rule it "
                     "ends at (`GainAscentHillClimbing`, no selection): " + "; ".join(
                         f"{l}: {format_accuracy(means.loc[l, 'accuracy'])} with "
                         f"{format_mean_count(means.loc[l, 'n_conditions'])} conditions" for l in foil) + ".\n\n")

    # 3. relative heuristics and m
    lines += [
        "## Relative heuristics and m\n\n",
        "The best selection for each m for refinement, in each frame; m -> infinity is WRA "
        "(absolute: WRA, relative: relative WRA; inverting it changes nothing).\n\n",
        f"![best accuracy per m for refinement]({plots}/m_refine{tag}.png)\n\n",
        "| m for refinement | " + " | ".join(kind for kind, _ in FRAMES) + " |\n|---|" + "--:|" * len(FRAMES) + "\n",
    ]
    for m_r in list(ms) + ["inf"]:
        cells = []
        for kind, _ in FRAMES:
            if m_r == "inf":
                frame = "relative" if kind.startswith("relative") else "absolute"
                labels = [section2_label(frame, "WRA", s) for s in selects]
            else:
                labels = [matrix_label(kind, m_r, s) for s in selects]
            acc, conds, best = _best_over_selection(means, labels)
            cells.append("n/a" if best is None else f"{format_accuracy(acc)} ({format_mean_count(conds)})")
        lines.append(f"| {m_r if m_r == 'inf' else _m(m_r)} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    # 4. beam
    lines += [
        "## Beam search\n\n",
        f"Beam width 1 vs. {BEAM_WIDTH}, the m-estimate selecting (m_s in "
        f"{{{', '.join(_m(m) for m in beam_selects)}}}). A beam ranks the refinements of all its "
        "rules together: scores relative to different parents are compared, which have no common "
        "scale. `Gain(...)` ranks by the improvement over the parent instead of the level; with "
        "width 1 the two rank the refinements of a rule alike.\n\n",
        "| refinement | m_s | " + " | ".join(f"width {w}" for w in (1, BEAM_WIDTH)) + " |\n|---|--:|--:|--:|\n",
    ]
    for name in _beam_heuristics():
        for m_s in beam_selects:
            cells = []
            for width in (1, BEAM_WIDTH):
                label = beam_label(width, name, m_s)
                cells.append(f"{format_accuracy(means.loc[label, 'accuracy'])} "
                             f"({format_mean_count(means.loc[label, 'n_conditions'])})"
                             if label in means.index else "n/a")
            lines.append(f"| {name} | {_m(m_s)} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    # 5. weighted gains
    lines += [
        "## Weighted gains\n\n",
        "`WeightedGain(weight, base)`: ``weight(r) * (base(r) - base(r0))``, the gain over the "
        "universal rule ``r0`` of the frame -- over the root (absolute) or over the parent "
        "(relative). The weights are Klösgen's family ``c^a``, with ``c`` the covered examples "
        "(`p + n`) or positives (`p`); the base is linear (precision) or logarithmic "
        "(information content). `p + n` and precision is WRA, `p` and information content "
        "Foil gain, `sqrt(p + n)` and precision Klösgen's quality function, weight 1 ranks as "
        "the base itself. Best selection (m_s in "
        f"{{{', '.join(_m(m) for m in beam_selects)}}}).\n\n",
        "| weight | base | absolute | relative |\n|---|---|--:|--:|\n",
    ]
    for weight in _weights():
        for base in _bases():
            cells = []
            for frame in ("absolute", "relative"):
                acc, conds, best = _best_over_selection(
                    means, [gain_label(frame, weight, base, s) for s in beam_selects])
                cells.append("n/a" if best is None else
                             f"{format_accuracy(acc)} ({format_mean_count(conds)}; m_s={_m_s(best)})")
            lines.append(f"| {weight} | {base} | " + " | ".join(cells) + " |\n")
    lines.append("\nCells: accuracy (conditions in the rule set; the best m for selection).\n\n")

    lines += [
        f"### Fixed selection (m_s = {_m(fixed)})\n\n",
        f"As above, with the m-estimate selecting with m_s = {_m(fixed)} for every gain.\n\n",
        "| weight | base | absolute | relative |\n|---|---|--:|--:|\n",
    ]
    for weight in _weights():
        for base in _bases():
            cells = [_fixed_cell(gain_label(frame, weight, base, fixed)) for frame in ("absolute", "relative")]
            lines.append(f"| {weight} | {base} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    # significance of the weighted gains' differences, with the best and with the fixed selection
    def _best_gain(frame, weight, base):
        return _best_over_selection(means, [gain_label(frame, weight, base, s) for s in beam_selects])[2]

    def _fixed_gain(frame, weight, base):
        label = gain_label(frame, weight, base, fixed)
        return label if label in means.index else None

    for title, what, pick in (("Paired comparisons", "its best selection (as above)", _best_gain),
                              (f"Paired comparisons, fixed selection (m_s = {_m(fixed)})",
                               f"the selection m_s = {_m(fixed)}", _fixed_gain)):
        lines += [
            f"### {title}\n\n",
            f"Each gain with {what}; per dataset the mean accuracy across "
            "folds; wins/losses of the second over the first, the mean difference, and the "
            "two-sided Wilcoxon signed-rank test (p) and sign test (sign p) across datasets "
            f"(not corrected for the {len(GAIN_PAIRS)} tests).\n\n",
            "| comparison | first | second | datasets | wins/losses | difference | p | sign p |\n"
            "|---|---|---|--:|--:|--:|--:|--:|\n",
        ]
        for first, second, name in GAIN_PAIRS:
            labels = [pick(*first), pick(*second)]
            if None in labels:
                continue
            n, wins, losses, diff, p_value, sign_p = _paired(results, labels[0], labels[1])
            lines.append(f"| {name} | {labels[0]} | {labels[1]} | {n} | {wins}/{losses} | "
                         f"{100 * diff:+.2f} | {p_value:.3f} | {sign_p:.3f} |\n")
        lines.append("\n")

    # ranks over the best variants
    lines += ["## Ranks\n\n",
              "Win counts and mean accuracy ranks across datasets among the best selection of each "
              "frame of section 2 and of the m-estimate.\n\n"]
    chosen = []
    for name in [f"m-estimate (m={_m(M_REFINE)})"] + list(rows):
        for kind, _ in FRAMES:
            if name.startswith("m-estimate"):
                labels = [matrix_label(kind, M_REFINE, s) for s in selects]
            elif kind in rows[name][2]:
                labels = [section2_label(kind, name, s) for s in selects]
            else:
                continue
            best = _best_over_selection(means, labels)[2]
            if best is not None:
                chosen.append(best)
    if chosen:
        subset = results[results["learner"].isin(chosen)]
        ranks, wins = mean_rank(subset, "accuracy"), win_counts(subset, "accuracy")
        lines.append("| learner | accuracy | wins | mean rank |\n|---|--:|--:|--:|\n")
        for learner in sorted(chosen, key=lambda l: ranks[l]):
            lines.append(f"| {learner} | {format_accuracy(means.loc[learner, 'accuracy'])} | "
                         f"{wins.get(learner, 0.0):.1f} | {ranks[learner]:.2f} |\n")
        lines.append("\n")
    n_failed = int(results["error"].notna().sum())
    lines.append(f"Failed folds: {n_failed}.\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {report_path}")


# -------------------------------------------------------------- plots ----

def write_plots(results, quick: bool) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    tag = "_quick" if quick else ""
    ms = QUICK_MATRIX_M if quick else MATRIX_M
    selects = QUICK_SELECT if quick else REPORT_SELECT
    means = _means(results)

    for measure, cmap, label, text in (("accuracy", "RdYlGn", "mean test accuracy", lambda v: f"{100 * v:.2f}"),
                                       ("conds_per_rule", "YlOrBr", "mean conditions per rule",
                                        lambda v: f"{v:.1f}")):
        matrices = [(_matrix(means, kind, measure, ms), title) for kind, title in MATRIX_TITLES]
        values = np.concatenate([m.to_numpy().ravel() for m, _ in matrices])
        if not np.isfinite(values).any():
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
        for ax, (matrix, title) in zip(axes.ravel(), matrices):
            data = matrix.to_numpy()
            image = ax.imshow(data, cmap=cmap, vmin=np.nanmin(values), vmax=np.nanmax(values))
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if np.isfinite(data[i, j]):
                        ax.text(j, i, text(data[i, j]), ha="center", va="center", fontsize=7)
            ticks = [_m(m) for m in ms]
            ax.set_xticks(range(len(ticks)), ticks)
            ax.set_yticks(range(len(ticks)), ticks)
            ax.set_xlabel("m for selection")
            ax.set_ylabel("m for refinement")
            ax.set_title(f"{title} refines")
        bar = fig.colorbar(image, ax=list(axes.ravel()), shrink=0.8)
        bar.set_label(label)
        if measure == "accuracy":
            bar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        fig.savefig(os.path.join(PLOTS_DIR, f"matrices_{measure}{tag}.png"), dpi=100)
        plt.close(fig)

    # absolute -> relative (-> inverted) per heuristic
    colors = dict(zip((kind for kind, _ in FRAMES), ("tab:blue", "tab:orange", "tab:purple", "tab:green")))
    fig, ax = plt.subplots(figsize=(9, 6))
    heuristics = dict(_section2_heuristics())
    for name in [f"m-estimate (m={_m(M_REFINE)})"] + list(heuristics):
        points = {}
        for kind, _ in FRAMES:
            if name.startswith("m-estimate"):
                labels = [matrix_label(kind, M_REFINE, s) for s in selects]
            elif kind in heuristics[name][2]:
                labels = [section2_label(kind, name, s) for s in selects]
            else:
                continue
            acc, conds, best = _best_over_selection(means, labels)
            if best is not None:
                points[kind] = (conds, acc)
        start = points.get("absolute")
        for kind, (conds, acc) in points.items():
            ax.scatter(conds, acc, color=colors[kind], s=30, zorder=3)
            if start is not None and kind != "absolute":
                ax.annotate("", (conds, acc), start, arrowprops=dict(arrowstyle="->", color=colors[kind], lw=0.8))
        if start is not None:
            ax.annotate(name, start, xytext=(4, 3), textcoords="offset points", fontsize=7)
    for kind, color in colors.items():
        ax.scatter([], [], color=color, label=kind)
    ax.set_xscale("log")
    ax.set_xlabel("mean number of conditions (log)")
    ax.set_ylabel("mean test accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("from the absolute heuristic to the other frames (best selection)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"frames{tag}.png"), dpi=110)
    plt.close(fig)

    # best accuracy per m for refinement
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(len(ms))) + [len(ms) + 0.5]
    for kind, _ in FRAMES:
        ys = [_best_over_selection(means, [matrix_label(kind, m_r, s) for s in selects])[0] for m_r in ms]
        frame = "relative" if kind.startswith("relative") else "absolute"
        ys.append(_best_over_selection(means, [section2_label(frame, "WRA", s) for s in selects])[0])
        ax.plot(xs[:-1], ys[:-1], "o-", color=colors[kind], label=kind)
        ax.plot(xs[-2:], ys[-2:], ":", color=colors[kind])
        ax.scatter(xs[-1:], ys[-1:], color=colors[kind], marker="s")
    ax.set_xticks(xs, [_m(m) for m in ms] + ["∞ (WRA)"])
    ax.set_xlabel("m for refinement")
    ax.set_ylabel("mean test accuracy (best selection)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"m_refine{tag}.png"), dpi=110)
    plt.close(fig)
    print(f"Plots  -> {PLOTS_DIR}")


def main(full: bool = False) -> None:
    quick = not full
    spec = hc.QUICK_DATASETS_SPEC if quick else hc.FULL_DATASETS_SPEC
    n_folds = hc.QUICK_FOLDS if quick else hc.N_FOLDS
    datasets = Catalog.default().parse(spec, random_state=hc.RANDOM_STATE)
    learners = build_learners(quick)
    print(f"{len(learners)} learners")
    results = run_cv(learners, datasets, n_folds=n_folds, fit_timeout=hc.FIT_TIMEOUT,
                     discretizer=TreeDiscretizer(max_intervals=hc.MAX_INTERVALS), random_state=hc.RANDOM_STATE,
                     cache_dir=CACHE_DIR, n_jobs=_n_jobs())
    write_report(results, datasets, n_folds, quick, QUICK_REPORT_PATH if quick else REPORT_PATH)
    write_plots(results, quick)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="the full run (heuristic_comparison's datasets and folds)")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores unless "
                             "given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(full=args.full)
