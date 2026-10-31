import matplotlib
matplotlib.use("Agg")

import numpy as np
from pyrulearn import HorizontalBitmapRepresentation, DataSpec, Rule
from pyrulearn.models import DecisionList, FlatRuleSet, annotate_rules
from pyrulearn.evaluation import (
    ABSTAIN, ConfusionMatrix, ModelStats,
    CoverageSpace, _apply_aspect, coverage_space_auc, coverage_space_plot, rule_refinement_path,
    rule_refinement_plot, sort_rules,
)
from pyrulearn.heuristics import FoilGain, MinimalLength, Precision, RuleStats, WRAcc

from _negation_helpers import make_rule, neg_spec, neg_X


def test_aspect_ratio():
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    _apply_aspect(ax, x_extent=9, y_extent=3, aspect="square", max_aspect_ratio=3.0)
    assert ax.get_box_aspect() is None

    _, ax = plt.subplots()
    _apply_aspect(ax, x_extent=30, y_extent=3, aspect="rectangular", max_aspect_ratio=3.0)
    assert abs(ax.get_box_aspect() - (3 / 30)) < 1e-9

    _, ax = plt.subplots()
    _apply_aspect(ax, x_extent=30, y_extent=3, aspect="auto", max_aspect_ratio=3.0)
    assert abs(ax.get_box_aspect() - (1 / 3)) < 1e-9  # clamped down from 0.1

    _, ax = plt.subplots()
    _apply_aspect(ax, x_extent=6, y_extent=3, aspect="auto", max_aspect_ratio=3.0)
    assert abs(ax.get_box_aspect() - 0.5) < 1e-9  # within range, unclamped

    plt.close("all")


def test_normalized_coverage_space_is_always_square():
    # a heavily skewed raw ratio must NOT distort the ROC-space box --
    # normalized display is always [0,1] x [0,1], a perfect square,
    # regardless of the underlying n_pos/n_neg or the aspect= setting
    for aspect in ("auto", "rectangular"):
        space = CoverageSpace(n_pos=5, n_neg=500, normalized=True, aspect=aspect)
        assert abs(space.ax.get_box_aspect() - 1.0) < 1e-9
    # non-normalized still reflects the true (skewed) ratio under "rectangular"
    space = CoverageSpace(n_pos=5, n_neg=500, normalized=False, aspect="rectangular")
    assert abs(space.ax.get_box_aspect() - (5 / 500)) < 1e-9


def test_coverage_space_plot_ruleset_uses_raw_counts():
    X = np.array([
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],
    ], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    r1 = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="pos", dataspec=ds)
    rs = FlatRuleSet([r1, r2])

    ax = coverage_space_plot(rs, rep, positive_class="pos", show_refinements=False)
    expected = rs.coverage_space(rep, "pos")
    offsets = ax.collections[0].get_offsets()
    assert list(offsets[:, 0]) == list(expected[:, 0])
    assert list(offsets[:, 1]) == list(expected[:, 1])


def test_coverage_space_plot_rulelist_cumulative_path():
    X = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1],
    ], dtype=bool)
    y = np.array(["pos", "pos", "neg", "pos"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    r1 = Rule.from_pos_neg(pos=[0], target="A", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="B", dataspec=ds)
    r3 = Rule.from_pos_neg(target="default", dataspec=ds)
    rl = DecisionList([r1, r2, r3])

    ax = coverage_space_plot(rl, rep, positive_class="pos")
    expected = rl.coverage_path(rep, "pos")
    # _plot_arrow_path scatters the points and draws len(pts)-1 arrow annotations
    offsets = ax.collections[0].get_offsets()
    assert list(offsets[:, 0]) == list(expected[:, 0])
    assert list(offsets[:, 1]) == list(expected[:, 1])
    assert offsets[0, 0] == 0 and offsets[0, 1] == 0
    assert len(ax.texts) == len(expected) - 1  # one arrow annotation per step


def test_upper_hull():
    from pyrulearn.evaluation import _upper_hull

    # (2,2) interior, (4,0) on the *lower* boundary -- neither belongs on
    # the ascending-x, max-y "upper" boundary used for AUC
    pts = np.array([[0, 0], [0, 4], [4, 4], [4, 0], [2, 2]])
    upper = _upper_hull(pts)
    assert [tuple(p) for p in upper] == [(0, 0), (0, 4), (4, 4)]


def test_coverage_space_plot_show_convex_hull():
    X = np.array([
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],
    ], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    r1 = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="pos", dataspec=ds)
    rs = FlatRuleSet([r1, r2])

    ax = coverage_space_plot(rs, rep, positive_class="pos", show_refinements=False, show_convex_hull=True)
    hull_lines = [l for l in ax.get_lines() if l.get_label().startswith("convex hull")]
    assert len(hull_lines) == 1
    xs, ys = hull_lines[0].get_xdata(), hull_lines[0].get_ydata()
    points = list(zip(xs, ys))
    assert points[0] == (0, 0) and points[-1] == (0, 0)  # closed back to the origin
    assert (2, 0) in points  # goes via (N, 0) -- not a bare diagonal back to (0,0)
    assert "AUC=" in hull_lines[0].get_label()


def test_convex_hull_restricted_to_same_target():
    # a mix of pos- and neg-target rules over the same DataSpec -- the
    # hull/AUC for positive_class="pos" must only see the pos-target rule
    X = np.array([
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],
    ], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    r_pos = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)  # covers rows 0,1: 2 pos, 0 neg
    r_neg = Rule.from_pos_neg(pos=[1], target="neg", dataspec=ds)  # covers rows 1,2: 1 pos, 1 neg
    rs = FlatRuleSet([r_pos, r_neg])

    auc_pos = coverage_space_auc(rs, rep, "pos")
    # only r_pos (0 neg, 2 pos) contributes -> hull is (0,0)-(0,2)-(2,2), a perfect classifier
    assert abs(auc_pos - 1.0) < 1e-9

    auc_neg = coverage_space_auc(rs, rep, "neg")
    # only r_neg contributes: for target="neg", r_neg covers 1 non-neg row and 1 neg row
    # out of 2 of each -- point (1,1) sits exactly on the diagonal, so AUC is exactly 0.5
    assert abs(auc_neg - 0.5) < 1e-9

    ax = coverage_space_plot(rs, rep, positive_class="pos", show_refinements=False, show_convex_hull=True)
    hull_line = next(l for l in ax.get_lines() if l.get_label().startswith("convex hull"))
    assert f"{auc_pos:.3f}" in hull_line.get_label()


def test_coverage_space_auc_no_matching_rules_is_random_baseline():
    X = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    r = Rule.from_pos_neg(pos=[0], target="other", dataspec=ds)
    rs = FlatRuleSet([r])

    auc = coverage_space_auc(rs, rep, "pos")
    assert abs(auc - 0.5) < 1e-9  # no pos-target rules -> trivial (0,0)-(N,P) hull


def test_ruleset_to_rulelist():
    from pyrulearn.models import annotate_rules
    ds = DataSpec(["f0", "f1", "f2"])
    # r1 (a): 1/1 correct -> Laplace (1+1)/(1+0+2)=0.667
    # r2 (b): 3/3 correct -> Laplace (3+1)/(3+0+2)=0.8 (highest)
    # r3 (c): 2/2 correct -> Laplace (2+1)/(2+0+2)=0.75
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
    y = np.array(["a", "b", "b", "b", "c", "c"])
    data = HorizontalBitmapRepresentation(ds, X, y)
    r1 = Rule.from_pos_neg(pos=[0], target="a", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="b", dataspec=ds)
    r3 = Rule.from_pos_neg(pos=[2], target="c", dataspec=ds)
    rs = FlatRuleSet(annotate_rules([r1, r2, r3], data))

    rl = rs.to_rulelist()
    assert isinstance(rl, DecisionList)
    assert [r.target for r in rl] == ["b", "c", "a"]  # descending Laplace: 0.8, 0.75, 0.667


def _covering_fixture():
    ds = DataSpec(["p", "q", "r"])
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
    y = np.array(["a", "a", "b", "b", "b", "a"])
    data = HorizontalBitmapRepresentation(ds, X, y)
    rp = Rule([0], target="a", dataspec=ds)       # p -> a: tp=2,fp=1 -> Laplace 0.6
    rpq = Rule([0, 1], target="a", dataspec=ds)   # p & q -> a: tp=1,fp=1 -> Laplace 0.5
    rr = Rule([2], target="b", dataspec=ds)       # r -> b: tp=3,fp=0 -> Laplace 0.8
    rq = Rule([1], target="b", dataspec=ds)       # q -> b: tp=2,fp=1 -> Laplace 0.6 (ties rp)
    rules = annotate_rules([rp, rpq, rr, rq], data)
    return ds, data, y, FlatRuleSet(rules, default_prediction="a")


def test_sort_rules_by_laplace_length_heuristic_and_callable():
    ds, data, _, rs = _covering_fixture()
    by_w = sort_rules(rs.rules, by=None)                                   # None -> Laplace-on-stats, the default
    # descending: rr(0.8), then rp/rq tied at 0.6 (stable -> rp first, it's
    # earlier in rs.rules), then rpq(0.5) -- identify rules by their .pos
    assert [r.pos for r in by_w] == [(2,), (0,), (1,), (0, 1)]
    assert sort_rules(rs.rules) == by_w                                    # None is the default
    # MinimalLength is a heuristic, and needs no `data` (needs_data is False)
    assert [r.length() for r in sort_rules(rs.rules, by=MinimalLength())][:1] == [1]  # shorter first
    assert sort_rules(rs.rules, by=None, descending=False)[0].pos == (0, 1)  # lowest score (0.5) first
    assert sort_rules(rs.rules, by=lambda r: r.length())[0].length() == 2
    # a coverage heuristic scores each rule against `data`
    assert sort_rules(rs.rules, by=Precision(), data=data)[0].pos == (2,)   # r -> b, precision 1.0
    try:
        sort_rules(rs.rules, by=Precision())                               # needs data
        assert False
    except ValueError:
        pass
    try:
        sort_rules(rs.rules, by="weight")                                  # strings no longer accepted
        assert False
    except ValueError:
        pass


def test_rule_refinement_path_monotonic_and_endpoints():
    X = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=bool)
    y = np.array(["pos", "pos", "pos", "neg", "neg", "neg"])
    ds = DataSpec(["a", "b", "c"])
    rep = HorizontalBitmapRepresentation(ds, X, y)

    r = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)  # unordered
    path = rule_refinement_path(r, rep, positive_class="pos")

    n_neg = int(np.sum(y != "pos"))
    n_pos = int(np.sum(y == "pos"))
    assert tuple(path[0]) == (n_neg, n_pos)  # empty rule covers everyone

    cov = r.covers_data_packed(rep)
    assert tuple(path[-1]) == (
        int(np.sum(cov & (y != "pos"))), int(np.sum(cov & (y == "pos"))),
    )
    assert all(path[i][0] >= path[i + 1][0] and path[i][1] >= path[i + 1][1] for i in range(len(path) - 1))


def test_rule_refinement_defaults_positive_class_to_rule_target():
    X = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=bool)
    y = np.array(["pos", "pos", "pos", "neg", "neg", "neg"])
    ds = neg_spec(["a", "b", "c"])
    rep = HorizontalBitmapRepresentation(ds, neg_X(X), y)

    # a rule whose head is "neg" -- omitting positive_class must use "neg",
    # not silently default to whichever class the caller happens to be
    # thinking of, since it's a real footgun to get backwards
    r = make_rule(ds, neg=["a"], target="neg")  # ¬a -> neg
    path_default = rule_refinement_path(r, rep)
    path_explicit = rule_refinement_path(r, rep, positive_class="neg")
    assert np.array_equal(path_default, path_explicit)

    # and it must NOT match what "pos" would have given (the actual bug:
    # both were silently swapped before this fix)
    path_wrong = rule_refinement_path(r, rep, positive_class="pos")
    assert not np.array_equal(path_default, path_wrong)

    ax = rule_refinement_plot(r, rep)
    assert "'neg'" in ax.get_ylabel()


def test_coverage_space_default_dims():
    space = CoverageSpace()
    assert space.n_pos < space.n_neg  # smaller value on the Pos axis
    assert space.n_pos == CoverageSpace.DEFAULT_N_POS == 100
    assert space.n_neg == CoverageSpace.DEFAULT_N_NEG == 160


def test_coverage_space_requires_both_dims_or_neither():
    CoverageSpace(n_pos=10, n_neg=20)  # both given: fine
    CoverageSpace()  # neither given: fine (default)
    try:
        CoverageSpace(n_pos=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        CoverageSpace(n_neg=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_coverage_space_show_labels_defaults_to_whether_dims_were_given():
    # default (arbitrary) dims -> generic axis text, symbolic endpoint
    # tick labels ("N"/"P") -- no concrete (misleading) numbers, but
    # still real axis text and endpoint ticks
    abstract = CoverageSpace()
    assert abstract.show_labels is False
    assert abstract.ax.get_xlabel() == "negatives covered"
    assert abstract.ax.get_ylabel() == "positives covered"
    assert [t.get_text() for t in abstract.ax.get_xticklabels()] == ["0", "N"]
    assert [t.get_text() for t in abstract.ax.get_yticklabels()] == ["0", "P"]

    # normalized + abstract -> both endpoints are literally (0, 1), always
    # true regardless of the placeholder counts behind them
    abstract_roc = CoverageSpace(normalized=True)
    assert [t.get_text() for t in abstract_roc.ax.get_xticklabels()] == ["0", "1"]
    assert [t.get_text() for t in abstract_roc.ax.get_yticklabels()] == ["0", "1"]

    # real dims given -> the real counts are shown, not symbolic placeholders
    real = CoverageSpace(n_pos=10, n_neg=20)
    assert real.show_labels is True
    assert real.ax.get_xlabel() == "negatives covered (of 20)"
    assert real.ax.get_ylabel() == "positives covered (of 10)"

    # explicit override always wins either way
    forced_off = CoverageSpace(n_pos=10, n_neg=20, show_labels=False)
    assert [t.get_text() for t in forced_off.ax.get_xticklabels()] == ["0", "N"]
    forced_on = CoverageSpace(show_labels=True)
    assert forced_on.ax.get_xlabel() == f"negatives covered (of {CoverageSpace.DEFAULT_N_NEG})"


def test_coverage_space_has_light_minor_grid():
    from matplotlib.ticker import AutoMinorLocator

    for space in (CoverageSpace(), CoverageSpace(n_pos=10, n_neg=20), CoverageSpace(normalized=True)):
        assert isinstance(space.ax.xaxis.get_minor_locator(), AutoMinorLocator)
        assert isinstance(space.ax.yaxis.get_minor_locator(), AutoMinorLocator)


def test_coverage_space_grid_cells_are_square_not_flat_5x5():
    # abstract (no dims given) rectangular space: 100 (Pos) x 160 (Neg) --
    # grid divisions must scale with the true extent ratio (5 on the
    # shorter axis, proportionally more on the longer one, i.e. 5x8),
    # not just "5 either way" regardless of the box's actual shape
    space = CoverageSpace()  # DEFAULT_N_POS=100, DEFAULT_N_NEG=160
    assert space.ax.yaxis.get_minor_locator().ndivs == 5
    assert space.ax.xaxis.get_minor_locator().ndivs == round(5 * 160 / 100) == 8

    # normalized (always square) -> still 5x5
    roc_space = CoverageSpace(normalized=True)
    assert roc_space.ax.xaxis.get_minor_locator().ndivs == 5
    assert roc_space.ax.yaxis.get_minor_locator().ndivs == 5


def test_plot_isometrics_skips_clabel_when_show_labels_false():
    space = CoverageSpace()  # default dims -> show_labels False
    assert space.show_labels is False
    Precision().plot_isometrics(space=space, levels=5)
    assert len(space.ax.texts) == 0  # no clabel value labels drawn

    space2 = CoverageSpace(n_pos=10, n_neg=16)  # real dims -> show_labels True
    Precision().plot_isometrics(space=space2, levels=5)
    assert len(space2.ax.texts) > 0


def test_coverage_space_from_data_matches_data():
    X = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg"])
    rep = HorizontalBitmapRepresentation(DataSpec(["a", "b"]), X, y)
    space = CoverageSpace.from_data(rep, "pos")
    assert space.n_pos == 2 and space.n_neg == 2


def test_coverage_space_normalized_scatter_gives_rates():
    space = CoverageSpace(n_pos=4, n_neg=8, normalized=True)
    points = np.array([[8, 4], [4, 2], [0, 0]])  # (neg, pos)
    coll = space.scatter(points)
    offsets = coll.get_offsets()
    # normalized: fpr = neg/8, tpr = pos/4
    assert list(offsets[:, 0]) == [1.0, 0.5, 0.0]
    assert list(offsets[:, 1]) == [1.0, 0.5, 0.0]
    assert space.ax.get_xlabel() == "FPR" and space.ax.get_ylabel() == "TPR"


def test_coverage_space_plot_ruleset_delegates_and_matches_standalone():
    # CoverageSpace.plot_ruleset (used internally by coverage_space_plot)
    # is directly usable/composable on its own too
    X = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    r1 = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="pos", dataspec=ds)
    rs = FlatRuleSet([r1, r2])

    space = CoverageSpace.from_data(rep, "pos")
    space.plot_ruleset(rs, rep, "pos", show_refinements=False)
    offsets = space.ax.collections[0].get_offsets()
    expected = rs.coverage_space(rep, "pos")
    assert list(offsets[:, 0]) == list(expected[:, 0])
    assert list(offsets[:, 1]) == list(expected[:, 1])


def test_plot_isometrics_returns_coverage_space_with_contour():
    space = Precision().plot_isometrics(levels=5)
    assert isinstance(space, CoverageSpace)
    # default space when none is given
    assert space.n_pos == CoverageSpace.DEFAULT_N_POS and space.n_neg == CoverageSpace.DEFAULT_N_NEG
    # contour lines were actually drawn (added as collections on the axes)
    assert len(space.ax.collections) > 0
    # default (arbitrary) dims -> show_labels defaults False -> no clabel text
    assert len(space.ax.texts) == 0


def test_plot_isometrics_sized_to_representation_and_layerable():
    X = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    r1 = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="pos", dataspec=ds)
    rs = FlatRuleSet([r1, r2])

    space = CoverageSpace.from_data(rep, "pos")
    WRAcc().plot_isometrics(space=space, levels=5)
    assert space.n_pos == 2 and space.n_neg == 2  # sized to rep, not the golden-ratio default

    # further layering onto the same space works (the actual point of returning `space`)
    n_collections_before = len(space.ax.collections)  # contour already added some
    space.plot_ruleset(rs, rep, "pos", show_refinements=False)
    # the ruleset's scatter is the last collection added, after the contour's
    offsets = space.ax.collections[-1].get_offsets()
    assert len(space.ax.collections) > n_collections_before
    expected = rs.coverage_space(rep, "pos")
    assert list(offsets[:, 0]) == list(expected[:, 0])


def test_plot_isometrics_filled_skips_clabel():
    space = CoverageSpace(n_pos=10, n_neg=10)
    n_texts_before = len(space.ax.texts)
    Precision().plot_isometrics(space=space, levels=5, filled=True)
    # filled contours (contourf) aren't clabel'd -- no new text artists
    assert len(space.ax.texts) == n_texts_before


def test_plot_isometrics_raises_for_foil_gain_without_parent():
    # FoilGain (a GainHeuristic) needs a parent RuleStats to score
    # anything -- without a fixed parent= supplied, plot_isometrics has
    # nothing to pass as the second argument to score(), so this must fail
    try:
        FoilGain().plot_isometrics(space=CoverageSpace(n_pos=10, n_neg=10), levels=3)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_plot_isometrics_with_parent_makes_foil_gain_plottable():
    # a fixed parent= (e.g. the universal rule) is what makes a
    # gain-style heuristic like FoilGain plottable at all
    space = CoverageSpace(n_pos=10, n_neg=10)
    universal = RuleStats.universal(space.n_pos, space.n_neg)
    n_collections_before = len(space.ax.collections)
    result = FoilGain().plot_isometrics(space=space, levels=5, parent=universal)
    assert result is space
    assert len(space.ax.collections) > n_collections_before


def test_plot_isometric_through_point_marks_the_point():
    space = CoverageSpace(n_pos=10, n_neg=16)
    n_collections_before = len(space.ax.collections)
    result = Precision().plot_isometric_through_point(4, 6, space=space)
    assert result is space
    # one contour path (a single level) + one scatter marker
    assert len(space.ax.collections) == n_collections_before + 2
    marker_offsets = space.ax.collections[-1].get_offsets()
    assert list(marker_offsets[0]) == [4, 6]


def test_plot_isometric_through_point_mark_point_false_skips_marker():
    space = CoverageSpace(n_pos=10, n_neg=16)
    n_collections_before = len(space.ax.collections)
    Precision().plot_isometric_through_point(4, 6, space=space, mark_point=False)
    assert len(space.ax.collections) == n_collections_before + 1  # contour only


def test_plot_isometric_through_rule_uses_rule_coverage_and_real_dims():
    X = np.array([
        [1, 1], [1, 1], [1, 0], [0, 1], [0, 1], [0, 0],  # positives
        [0, 1], [0, 1], [0, 1], [0, 0],                   # negatives
    ], dtype=bool)
    y = np.array(["pos"] * 6 + ["neg"] * 4)
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    rule = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)  # a=True: rows 0,1,2 -> tp=3, fp=0

    space = Precision().plot_isometric_through_rule(rule, rep)
    # sized to the real data (n_pos=6, n_neg=4), not the abstract default
    assert space.n_pos == 6 and space.n_neg == 4
    marker_offsets = space.ax.collections[-1].get_offsets()
    assert list(marker_offsets[0]) == [0, 3]  # fp=0, tp=3


def test_plot_isometric_through_rule_raises_without_positive_class():
    r_no_target = Rule.from_pos_neg(pos=[0], n_features=2)  # no target
    X = np.array([[1, 0], [0, 1]], dtype=bool)
    y = np.array(["pos", "neg"])
    rep = HorizontalBitmapRepresentation(DataSpec(["a", "b"]), X, y)
    try:
        Precision().plot_isometric_through_rule(r_no_target, rep)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_order_by_precision():
    X = np.array([
        [1, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 1],
        [0, 0],
    ], dtype=bool)
    y = np.array(["neg", "neg", "pos", "pos", "pos", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)

    r = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)  # unordered, canonical: a then b
    ordered = r.order_by_precision(rep)

    assert ordered.ordered is True
    # b=True alone is perfectly precise (3/3 pos); a=True alone is only 1/3 -> b picked first
    assert [lit.feature for lit in ordered.conditions] == [1, 0]
    # same rule, just reordered -- coverage is unaffected
    assert ordered.pos == r.pos


# ------------------------------------------------------------- ConfusionMatrix ---

def test_confusion_matrix_from_predictions_basic():
    y_true = ["a", "a", "b", "b", "c"]
    y_pred = ["a", "b", "b", "b", "c"]
    cm = ConfusionMatrix.from_predictions(y_true, y_pred, labels=["a", "b", "c"])
    assert cm.labels == ["a", "b", "c"]
    assert cm.n_total == 5
    assert cm.accuracy == 4 / 5   # only row 1 (true a, predicted b) is wrong


def test_confusion_matrix_rule_stats_differs_by_label_not_just_relabeled():
    # a genuine 2-label case: binary(a) and binary(b) are mirror images
    # (tp/tn and fp/fn swap), not the same numbers under a different name
    y_true = ["a", "a", "a", "b", "b"]
    y_pred = ["a", "a", "b", "b", "b"]
    cm = ConfusionMatrix.from_predictions(y_true, y_pred, labels=["a", "b"])
    rs_a = cm.rule_stats("a")
    rs_b = cm.rule_stats("b")
    assert (rs_a.tp, rs_a.fp, rs_a.fn, rs_a.tn) == (2, 0, 1, 2)
    assert (rs_b.tp, rs_b.fp, rs_b.fn, rs_b.tn) == (2, 1, 0, 2)
    assert rs_a.tp == rs_b.tn and rs_a.tn == rs_b.tp  # mirror, not identical


def test_confusion_matrix_abstained_predictions_get_their_own_bucket():
    y_true = ["a", "a", "b"]
    y_pred = ["a", None, "b"]
    cm = ConfusionMatrix.from_predictions(y_true, y_pred, labels=["a", "b"])
    assert ABSTAIN in cm.labels
    # the abstained row counts against accuracy (never excluded from the denominator)
    assert cm.accuracy == 2 / 3
    rs_a = cm.rule_stats("a")
    assert rs_a.tp == 1 and rs_a.fn == 1   # the abstained "a" row is a false negative for "a"


def test_confusion_matrix_labels_outside_the_given_set_are_folded_in():
    cm = ConfusionMatrix.from_predictions(["a", "b", "z"], ["a", "b", "z"], labels=["a", "b"])
    assert "z" in cm.labels
    assert cm.n_total == 3


# ----------------------------------------------------------------- ModelStats ---

def test_model_stats_lives_alongside_rulestats_and_confusionmatrix():
    # ModelStats itself is exercised end-to-end via RuleModel.stats() in
    # test_models.py -- this just confirms it's constructible directly here,
    # in the same module as the RuleStats/ConfusionMatrix it wraps.
    cm = ConfusionMatrix.from_predictions(["a", "a", "b"], ["a", "b", "b"])
    st = ModelStats(n_rows=3, confusion=cm, n_rules=2, n_conditions=4)
    assert st.n_rows == 3 and st.n_rules == 2 and st.n_conditions == 4
    assert st.confusion is cm
    assert "accuracy" in repr(st)
    st_no_confusion = ModelStats(n_rows=3, confusion=None, n_rules=2, n_conditions=4)
    assert "accuracy" not in repr(st_no_confusion)
