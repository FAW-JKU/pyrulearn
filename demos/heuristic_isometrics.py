"""
Demo: `RuleHeuristic` isometrics via `CoverageSpace`.

Two things, each rendered in both a raw coverage space and a normalized
ROC space:

1. Every defined heuristic's isometrics, plotted standalone (default
   `CoverageSpace` dimensions -- no dataset involved at all).
2. One example rule's refinement path (see `pyrulearn.evaluation.
   rule_refinement_path`), layered on top of `Precision`'s and
   `Accuracy`'s isometrics on the same space -- a concrete illustration
   of how a greedy, precision-ordered refinement moves through
   Precision's isometrics (always "uphill", since that's what it's
   greedily optimizing) versus Accuracy's (not necessarily uphill, since
   the rule isn't optimizing that heuristic at all).
3. `Precision`/`Laplace`/`MEstimate` side by side, illustrating why the
   pencil family's *pivot* matters: `Laplace`'s pivot is at (-1, -1)
   instead of `Precision`'s origin, and `MEstimate(m)`'s slides along
   that same line as `m` grows -- but a shift of a few units is
   invisible at the default 100/160-sized space (a handful of examples
   out of 100+ barely moves anything), so this uses a much smaller
   space instead, where the same absolute shift is a real fraction of
   the total.
4. Inverted heuristics (`Inverted(h)`, h(N-n, P-p)): `h` evaluated
   from the upper right corner of coverage space, on the examples a rule
   excludes, with the classes exchanged. Relative to the parent rule
   (`Relative(Inverted(h))`, Stecher et al.'s inverted heuristics) the
   corner is the rule being refined; the plots show the first
   refinement, from the universal rule. A summary table at the top gives
   every heuristic's formula, regular and inverted; the section at the
   end shows the regular and the inverted isometrics side by side for
   every heuristic where inversion changes the ranking.

`LengthPenalized` (needs `stats.length`, which defaults to 0 here --
indistinguishable from its wrapped heuristic on this kind of plot) is
left out of the heuristic grid for that reason. `FoilGain` is relative
to the parent rule; without one (as for a bare grid point) it is the
gain over the universal rule, the plotted quantity throughout. The main grid's `MEstimate`/
`GHeuristic` instances use `m=20`/`g=30` (not smaller, more typical
values) since, unlike `Laplace`'s fixed pivot shift, their pivot- and
slope-changing effects are already visible even at the default space
size with these larger parameter values.
"""

import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyrulearn import HorizontalBitmapRepresentation, DataSpec, DataSpecBuilder, Rule
from pyrulearn.evaluation import CoverageSpace
from pyrulearn.heuristics import (
    Accuracy,
    Correlation,
    Coverage,
    CoverageDifference,
    Entropy,
    FBeta,
    FoilGain,
    GeneralizedMEstimate,
    GHeuristic,
    Inverted,
    Laplace,
    LikelihoodRatio,
    LinearCost,
    LinearCostRates,
    MEstimate,
    CoveredNegatives,
    Precision,
    Recall,
    Support,
    WRAcc,
    YoudenJ,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(HERE, "heuristic_isometrics_plots")
REPORT_PATH = os.path.join(HERE, "heuristic_isometrics_report.md")
COMPARISON_REPORT = os.path.join(HERE, "heuristic_comparison_report.md")

# (name, heuristic, one-line description for the report); p/n = covered
# positives/negatives, P/N = all positives/negatives
HEURISTICS = [
    ("CoveredNegatives", CoveredNegatives(),
     "h = -n. Only consistency counts: vertical isometrics."),
    ("Recall", Recall(),
     "h = p/P. Only positive coverage counts: horizontal isometrics."),
    ("Precision", Precision(),
     "h = p/(p+n). Lines through the origin: purity, regardless of coverage."),
    ("FBeta(1)", FBeta(),
     "Harmonic mean of precision and recall. Precision-like lines, but pivoting "
     "at (-beta^2 * P, 0) left of the origin, so coverage gains weight."),
    ("Laplace", Laplace(),
     "h = (p+1)/(p+n+2). Precision with its pivot moved to (-1, -1)."),
    ("MEstimate(20)", MEstimate(20),
     "h = (p + m * P/(P+N))/(p+n+m). The pivot moves out along the negative "
     "diagonal as m grows; from precision (m = 0) towards WRAcc (m -> infinity)."),
    ("GeneralizedMEstimate(20, 0.9)", GeneralizedMEstimate(20, 0.9),
     "h = (p + m * c)/(p+n+m): the m-estimate with a free prior c instead of P/(P+N)."),
    ("GHeuristic(30)", GHeuristic(30),
     "h = p/(n+g) (Gamberger & Lavrac). Precision-like lines pivoting at (-g, 0)."),
    ("WRAcc", WRAcc(),
     "Weighted relative accuracy, coverage times precision gain over P/(P+N). "
     "Lines parallel to the diagonal."),
    ("YoudenJ", YoudenJ(),
     "h = p/P - n/N. Lines parallel to the diagonal; ranks rules exactly like WRAcc."),
    ("Accuracy", Accuracy(),
     "h = (p + N - n)/(P+N). Parallel lines of slope 1."),
    ("CoverageDifference", CoverageDifference(),
     "h = p - n. Ranks rules exactly like Accuracy."),
    ("Support", Support(),
     "h = (p+n)/(P+N). Anti-diagonal lines: coverage regardless of class."),
    ("Coverage", Coverage(),
     "h = p + n. Ranks rules exactly like Support."),
    ("LinearCost(2.0)", LinearCost(2.0),
     "h = p - c * n (the paper's cost measure, up to scaling). Parallel lines of slope c."),
    ("LinearCostRates(2.0)", LinearCostRates(2.0),
     "h = p/P - c * n/N (the relative cost measure, up to scaling). Parallel lines "
     "of slope c in ROC space."),
    ("Correlation", Correlation(),
     "The phi coefficient between 'rule covers' and 'is positive' (FOSSIL). "
     "Curved isometrics, symmetric around the diagonal."),
    ("Entropy", Entropy(),
     "Negated class entropy among covered examples (CN2's original heuristic). "
     "Symmetric: pure negative rules score as well as pure positive ones."),
    ("LikelihoodRatio", LikelihoodRatio(),
     "CN2's significance statistic: how far the covered class distribution is from "
     "the prior. Grows with coverage, in both directions away from the diagonal."),
    ("FoilGain (vs. universal rule)", FoilGain(),
     "FOIL's information gain, here relative to the universal rule (in a search, it is "
     "relative to the parent rule): p * (log2 precision - log2 P/(P+N))."),
]

# formulas for the summary table: name -> (h(p, n), h(N - n, P - p)).
# "unchanged": the inverted heuristic ranks every rule the same way
# (`Inverted` warns); "n.a.": it can't be inverted.
# Both are checked against the library in `_formula_table`.
UNCHANGED, NOT_APPLICABLE = "unchanged", "n.a."
FORMULAS = {
    "CoveredNegatives": ("-n", "-(P - p), ranks like Recall"),
    "Recall": ("p/P", "(N - n)/N, ranks like CoveredNegatives"),
    "Precision": ("p/(p + n)", "(N - n)/(N - n + P - p)"),
    "FBeta(1)": ("(1 + β²) p/(β² P + p + n)", "(1 + β²)(N - n)/(β² N + N - n + P - p)"),
    "Laplace": ("(p + 1)/(p + n + 2)", "(N - n + 1)/(N - n + P - p + 2)"),
    "MEstimate(20)": ("(p + m P/(P + N))/(p + n + m)", "(N - n + m N/(P + N))/(N - n + P - p + m)"),
    "GeneralizedMEstimate(20, 0.9)": ("(p + m c)/(p + n + m)", "(N - n + m c)/(N - n + P - p + m)"),
    "GHeuristic(30)": ("p/(n + g)", "(N - n)/(P - p + g)"),
    "WRAcc": ("(p + n)/(P + N) · (p/(p + n) - P/(P + N))", UNCHANGED),
    "YoudenJ": ("p/P - n/N", UNCHANGED),
    "Accuracy": ("(p + N - n)/(P + N)", UNCHANGED),
    "CoverageDifference": ("p - n", UNCHANGED),
    "Support": ("(p + n)/(P + N)", "(P + N - p - n)/(P + N), the reverse ranking"),
    "Coverage": ("p + n", "P + N - p - n, the reverse ranking"),
    "LinearCost(2.0)": ("p - c n", "N - n - c (P - p), ranks like LinearCost(1/c)"),
    "LinearCostRates(2.0)": ("p/P - c n/N", "(N - n)/N - c (P - p)/P, ranks like LinearCostRates(1/c)"),
    "Correlation": ("(p N - n P)/√(P N (p + n)(P + N - p - n))", UNCHANGED),
    "Entropy": ("π log₂ π + (1 - π) log₂ (1 - π), π = p/(p + n)",
                "the same with π = (N - n)/(N - n + P - p): the purity of the uncovered examples"),
    "LikelihoodRatio": ("2 (p ln(p/e_p) + n ln(n/e_n)), e_p = (p + n) P/(P + N), e_n = (p + n) N/(P + N)",
                        "the same for the uncovered examples: N - n and P - p against their expected "
                        "shares of P + N - p - n"),
    "FoilGain (vs. universal rule)": ("p (log₂ p/(p + n) - log₂ P/(P + N))",
                                      "(N - n)(log₂ (N - n)/(N - n + P - p) - log₂ N/(P + N))"),
}


def _inversion_kind(h) -> str:
    """"inverted", "unchanged" (same ranking either way) or "n.a.", from the library."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            Inverted(h)
        except TypeError:
            return NOT_APPLICABLE
    return UNCHANGED if any("the same way inverted or not" in str(w.message) for w in caught) else "inverted"


def _inverted(h):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Inverted(h)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_heuristic_isometrics(normalized: bool) -> None:
    kind = "roc" if normalized else "raw"
    for name, h, _ in HEURISTICS:
        fig, ax = plt.subplots(figsize=(5, 5))
        space = CoverageSpace(ax=ax, normalized=normalized, title=name)
        # a relative heuristic like FoilGain: relative to the universal rule
        h.plot_isometrics(space=space, levels=8)
        fig.tight_layout()
        out_path = os.path.join(PLOTS_DIR, _isometrics_file(name, kind))
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(HEURISTICS)} isometrics plots ({kind}) to {PLOTS_DIR}")


def _isometrics_file(name: str, kind: str) -> str:
    return f"isometrics_{_slug(name)}_{kind}.png"


def _inverted_file(name: str, kind: str) -> str:
    return f"inverted_{_slug(name)}_{kind}.png"


def _invertible():
    return [(name, h) for name, h, _ in HEURISTICS if _inversion_kind(h) == "inverted"]


def plot_inverted_isometrics(normalized: bool) -> None:
    kind = "roc" if normalized else "raw"
    invertible = _invertible()
    for name, h in invertible:
        fig, ax = plt.subplots(figsize=(5, 5))
        space = CoverageSpace(ax=ax, normalized=normalized, title=f"Inverted({name})")
        _inverted(h).plot_isometrics(space=space, levels=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, _inverted_file(name, kind)), dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(invertible)} inverted isometrics plots ({kind}) to {PLOTS_DIR}")


def build_example_rule():
    rng = np.random.default_rng(0)
    n, d = 300, 6
    X = rng.integers(0, 2, size=(n, d)).astype(bool)
    # ground truth: y = f0 AND NOT f1 AND (f2 OR f3), with a bit of noise
    y_clean = X[:, 0] & ~X[:, 1] & (X[:, 2] | X[:, 3])
    noise = rng.random(n) < 0.05
    y = np.where(noise, ~y_clean, y_clean)
    y = np.where(y, "pos", "neg")

    # explicit-negation feature space (the default): f_i paired with "not f_i"
    _b = DataSpecBuilder()
    for i in range(d):
        _b.add_boolean(f"f{i}")
    ds = _b.build()
    X_full = np.empty((n, 2 * d), dtype=bool)
    X_full[:, 0::2] = X
    X_full[:, 1::2] = ~X
    rep = HorizontalBitmapRepresentation(ds, X_full, y)
    rule = Rule.from_pos_neg(
        pos=[ds.feature_index("f0"), ds.feature_index("f2")],
        neg=[ds.feature_index("f1")], target="pos", dataspec=ds,
    )
    ordered = rule.order_by_precision(rep)  # a meaningful step-by-step order to plot
    return ordered, rep


def plot_refinement_with_isometrics(normalized: bool, out_name: str) -> None:
    rule, rep = build_example_rule()
    space = CoverageSpace.from_data(
        rep, "pos", normalized=normalized,
        title="Rule refinement over Precision/Accuracy isometrics",
    )
    # thin dashed lines, in colors distinct from the refinement path's own
    # (plot_rule_refinement draws it in "C1", matplotlib's default orange)
    # -- solid same-color isometrics would be hard to tell apart from the
    # path itself
    iso_style = dict(levels=8, linestyles="dashed", linewidths=0.8)
    Precision().plot_isometrics(space=space, colors="tab:blue", **iso_style)
    Accuracy().plot_isometrics(space=space, colors="tab:green", **iso_style)
    # unlabeled contour sets don't show up in a legend on their own -- add
    # invisible proxy lines so the final legend() call (inside
    # plot_rule_refinement) can tell the two heuristics' isometrics apart
    space.ax.plot([], [], color="tab:blue", linestyle="dashed", label="Precision isometrics")
    space.ax.plot([], [], color="tab:green", linestyle="dashed", label="Accuracy isometrics")
    space.plot_rule_refinement(rule, rep, annotate=True)

    # highlight the single Precision isometric that passes exactly
    # through the final rule's own coverage -- every other point on
    # this one line is, by definition, exactly as precise as the rule
    # itself. mark_point=False since plot_rule_refinement already
    # scattered the final point.
    Precision().plot_isometric_through_rule(
        rule, rep, space=space, mark_point=False, colors="black", linewidths=1.2,
    )
    space.ax.plot([], [], color="black", label="Precision isometric through final rule")
    space.ax.legend(loc="lower right", fontsize=8)  # refresh legend with the label just added

    out_path = os.path.join(PLOTS_DIR, out_name)
    space.ax.figure.savefig(out_path, dpi=110)
    plt.close(space.ax.figure)
    print(f"Saved {out_path}")


# a small space, not the default 100/160: Laplace's pivot shift to
# (-1, -1) is a real fraction of a space this size, invisible at the
# default scale (see module docstring)
PRECISION_FAMILY_N_POS = 10
PRECISION_FAMILY_N_NEG = 16
PRECISION_FAMILY = [
    ("Precision", Precision(), "tab:blue"),
    ("Laplace", Laplace(), "tab:orange"),
    ("MEstimate(2)", MEstimate(2), "tab:green"),
    ("MEstimate(20)", MEstimate(20), "tab:red"),
]


def plot_precision_family_comparison(normalized: bool, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    space = CoverageSpace(
        n_pos=PRECISION_FAMILY_N_POS, n_neg=PRECISION_FAMILY_N_NEG, normalized=normalized, ax=ax,
        title=f"Precision/Laplace/MEstimate pivots ({PRECISION_FAMILY_N_POS}/{PRECISION_FAMILY_N_NEG})",
    )
    for name, h, color in PRECISION_FAMILY:
        h.plot_isometrics(space=space, levels=6, colors=color, linestyles="dashed", linewidths=0.8)
        ax.plot([], [], color=color, linestyle="dashed", label=name)
    ax.legend(fontsize=8, loc="lower right")

    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")


def _anchor(title: str) -> str:
    """GitHub's anchor for a heading."""
    return "#" + re.sub(r"[^a-z0-9 _-]", "", title.lower()).replace(" ", "-")


def _formula_table() -> str:
    rows = ["| heuristic | h(p, n) | inverted: h(N - n, P - p) |\n|---|---|---|\n"]
    for name, h, _ in HEURISTICS:
        regular, inverted = FORMULAS[name]
        kind = _inversion_kind(h)
        expected = inverted if inverted in (UNCHANGED, NOT_APPLICABLE) else "inverted"
        if kind != expected:
            raise AssertionError(f"{name}: the table says {expected!r}, the library {kind!r}")
        cell = inverted if kind != "inverted" else f"[{inverted}]({_anchor(f'Inverted {name}')})"
        rows.append(f"| {name} | {regular} | {cell} |\n")
    return "".join(rows) + "\n"


def _side_by_side(left: str, right: str, alt: str) -> str:
    plots = os.path.basename(PLOTS_DIR)
    return (f'<img src="{plots}/{left}" alt="{alt}" width="49%"> '
            f'<img src="{plots}/{right}" alt="Inverted({alt})" width="49%">\n\n')


def _pair(left: str, right: str, alt: str) -> str:
    plots = os.path.basename(PLOTS_DIR)
    return (f'<img src="{plots}/{left}" alt="{alt}, coverage space" width="49%"> '
            f'<img src="{plots}/{right}" alt="{alt}, ROC space" width="49%">\n\n')


def write_report() -> None:
    comparison = os.path.relpath(COMPARISON_REPORT, HERE).replace(os.sep, "/")
    script = os.path.relpath(os.path.abspath(__file__), os.path.join(HERE, "..")).replace(os.sep, "/")
    lines = [
        "# Rule learning heuristics: isometrics in coverage space\n\n",
        "A rule learning heuristic scores a rule by the positives `p` and negatives `n` it "
        "covers, given the totals `P` and `N`. Its *isometrics* -- the lines of equal "
        "score in coverage space (x = negatives covered, y = positives covered) -- show "
        "its preference structure at a glance: how it trades off consistency against "
        "coverage (Fürnkranz & Flach, 2005). This demo plots them for every heuristic in "
        "`pyrulearn.heuristics`, each in raw coverage space (counts) and normalized ROC "
        "space (rates), for an abstract dataset with P = "
        f"{CoverageSpace.DEFAULT_N_POS} and N = {CoverageSpace.DEFAULT_N_NEG}. The dashed "
        "gray diagonal is the random-guess line; brighter isometrics are better.\n\n",
        "All plots come straight from the library, no plotting code of their own: "
        "`pyrulearn.evaluation.CoverageSpace` draws the space, and every heuristic can "
        "draw its own isometrics into it.\n\n",
        "```python\n"
        "from pyrulearn.evaluation import CoverageSpace\n"
        "from pyrulearn.heuristics import MEstimate\n\n"
        "space = CoverageSpace(normalized=False)          # or CoverageSpace.from_data(rep, target)\n"
        "MEstimate(20).plot_isometrics(space=space, levels=8)\n"
        "space.plot_rule_refinement(rule, rep)            # a rule's refinement path on top\n"
        "```\n\n",
        "How the heuristics actually perform as search heuristics -- accuracy and theory "
        "size inside one separate-and-conquer learner, after Janssen & Fürnkranz (Machine "
        f"Learning, 2010) -- is tested in the heuristic comparison demo: see the "
        f"[full results]({comparison}).\n\n",
        f"Run: `python {script}` (no data or extras needed beyond matplotlib).\n\n",
        "## The heuristics at a glance\n\n",
        "`p`/`n`: covered positives/negatives, `P`/`N`: all positives/negatives; "
        "`m`, `c`, `g`, `β`: the parameters in the heuristic's name. The inverted "
        "heuristic `Inverted(h)` (Stecher, Janssen & Fürnkranz, 2014) evaluates `h` on "
        "what a rule leaves out, with the classes exchanged: h(N - n, P - p), the "
        "excluded negatives counted as positives and the excluded positives as negatives. "
        "*unchanged*: the inverted heuristic ranks every rule the same way (relative to the "
        "parent rule, inverting it only restricts it to the examples of the rule being "
        "refined, as `Relative` does). The links lead to the inverted isometrics.\n\n",
        _formula_table(),
        "## Isometrics per heuristic\n\n",
        "Left: coverage space (counts), right: ROC space (rates). `p`/`n`: covered "
        "positives/negatives, `P`/`N`: all positives/negatives. `LengthPenalized` is "
        "left out: it needs a rule length, which a point in coverage space doesn't have.\n\n",
    ]
    for name, _, description in HEURISTICS:
        lines.append(f"### {name}\n\n{description}\n\n")
        lines.append(_pair(_isometrics_file(name, "raw"), _isometrics_file(name, "roc"), name))
    lines += [
        "## A rule's refinement path over the isometrics\n\n",
        "One rule on a small synthetic dataset, its conditions added in order of "
        "precision (`Rule.order_by_precision`), drawn as a path through coverage space "
        "(`CoverageSpace.plot_rule_refinement`) over Precision's (blue) and Accuracy's "
        "(green) isometrics. The path always moves to better Precision isometrics -- that "
        "is what the order optimizes -- but not necessarily to better Accuracy ones. The "
        "black line is the Precision isometric through the final rule "
        "(`plot_isometric_through_rule`): every point on it is exactly as precise.\n\n",
        _pair("refinement_raw.png", "refinement_roc.png", "Rule refinement path"),
        "## Where the precision family pivots\n\n",
        "Precision, Laplace and the m-estimate all have isometrics that are lines through "
        "one pivot point: the origin for Precision, (-1, -1) for Laplace, and further out "
        "along the negative diagonal for the m-estimate as `m` grows. In a space of "
        "realistic size a shift of one or two examples is invisible, so this plot uses a "
        f"small space (P = {PRECISION_FAMILY_N_POS}, N = {PRECISION_FAMILY_N_NEG}).\n\n",
        _pair("precision_family_raw.png", "precision_family_roc.png", "Precision family"),
        "## Inverted heuristics\n\n",
        "A top-down search starts with the universal rule in the upper right corner of "
        "coverage space, (N, P), and moves towards the origin. A regular heuristic judges "
        "a rule from the origin: Precision's isometrics all go through it. The inverted "
        "heuristic judges it from the corner the search starts from: inverted Precision's "
        "isometrics all go through (N, P), and it prefers the rule whose excluded examples "
        "are the purest in negatives -- the refinement that removes negatives and keeps "
        "positives. Relative to the parent rule (`Relative(Inverted(h))`), the corner is "
        "the rule being refined: a refinement of a rule covering P̂ positives and N̂ "
        "negatives scores h(N̂ - n, P̂ - p), and the plots show the first refinement, from "
        "the universal rule. Stecher, Janssen & Fürnkranz (2014) refine a rule by such a "
        "relative inverted heuristic and choose the rule on the refinement path by a "
        "regular one (`SingleRuleLearner(Relative(Inverted(Precision())), "
        "selection=Laplace())`); the "
        f"[heuristic comparison]({comparison}#m-estimates-for-refinement-and-for-selection) tests "
        "this.\n\n",
        "Left: the regular heuristic, right: the inverted one, both in coverage space "
        "(counts); ROC space only rescales the axes. Only the heuristics where inversion "
        f"changes the ranking (see [the table]({_anchor('The heuristics at a glance')})).\n\n",
    ]
    for name, _ in _invertible():
        regular, inverted = FORMULAS[name]
        lines.append(f"### Inverted {name}\n\nh = {regular} becomes {inverted}.\n\n")
        lines.append(_side_by_side(_isometrics_file(name, "raw"), _inverted_file(name, "raw"), name))
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {REPORT_PATH}")


def main():
    print(f"Plotting isometrics for {len(HEURISTICS)} heuristics "
          "(LengthPenalized skipped -- see module docstring).")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_heuristic_isometrics(normalized=False)
    plot_heuristic_isometrics(normalized=True)
    plot_refinement_with_isometrics(normalized=False, out_name="refinement_raw.png")
    plot_refinement_with_isometrics(normalized=True, out_name="refinement_roc.png")
    plot_precision_family_comparison(normalized=False, out_name="precision_family_raw.png")
    plot_precision_family_comparison(normalized=True, out_name="precision_family_roc.png")
    plot_inverted_isometrics(normalized=False)
    write_report()


if __name__ == "__main__":
    main()
