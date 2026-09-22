import itertools
from functools import partial
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import same_color, to_rgb, to_rgba

import pytest
from pytest import approx
from numpy.testing import (
    assert_array_equal,
    assert_array_less,
    assert_array_almost_equal,
)

from seaborn import categorical as cat

from seaborn._base import categorical_order
from seaborn._compat import get_colormap, get_legend_handles
from seaborn._testing import assert_plots_equal
from seaborn.categorical import (
    _CategoricalPlotter,
    Beeswarm,
    BoxPlotContainer,
    catplot,
    barplot,
    boxplot,
    boxenplot,
    pointplot,
    stripplot,
    swarmplot,
    violinplot,
)
from seaborn.palettes import color_palette
from seaborn.utils import _draw_figure, _version_predates, desaturate


PLOT_FUNCS = [
    catplot,
    barplot,
    boxplot,
    boxenplot,
    pointplot,
    stripplot,
    swarmplot,
    violinplot,
]


class TestCategoricalPlotterNew:

    @pytest.mark.parametrize(
        "func,kwargs",
        itertools.product(
            PLOT_FUNCS,
            [
                {"x": "x", "y": "a"},
                {"x": "a", "y": "y"},
                {"x": "y"},
                {"y": "x"},
            ],
        ),
    )
    def test_axis_labels(self, long_df, func, kwargs):

        func(data=long_df, **kwargs)

        ax = plt.gca()
        for axis in "xy":
            val = kwargs.get(axis, "")
            label_func = getattr(ax, f"get_{axis}label")
            assert label_func() == val

    @pytest.mark.parametrize("func", PLOT_FUNCS)
    def test_empty(self, func):

        func()
        ax = plt.gca()
        assert not ax.collections
        assert not ax.patches
        assert not ax.lines

        func(x=[], y=[])
        ax = plt.gca()
        assert not ax.collections
        assert not ax.patches
        assert not ax.lines

    def test_redundant_hue_backcompat(self, long_df):

        p = _CategoricalPlotter(
            data=long_df,
            variables={"x": "s", "y": "y"},
        )

        color = None
        palette = dict(zip(long_df["s"].unique(), color_palette()))
        hue_order = None

        palette, _ = p._hue_backcompat(color, palette, hue_order, force_hue=True)

        assert p.variables["hue"] == "s"
        assert_array_equal(p.plot_data["hue"], p.plot_data["x"])
        assert all(isinstance(k, str) for k in palette)


class SharedAxesLevelTests:

    def orient_indices(self, orient):
        pos_idx = ["x", "y"].index(orient)
        val_idx = ["y", "x"].index(orient)
        return pos_idx, val_idx

    @pytest.fixture
    def common_kws(self):
        return {}

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_labels_long(self, long_df, orient):

        depend = {"x": "y", "y": "x"}[orient]
        kws = {orient: "a", depend: "y", "hue": "b"}

        ax = self.func(long_df, **kws)

        # To populate texts; only needed on older matplotlibs
        _draw_figure(ax.figure)

        assert getattr(ax, f"get_{orient}label")() == kws[orient]
        assert getattr(ax, f"get_{depend}label")() == kws[depend]

        get_ori_labels = getattr(ax, f"get_{orient}ticklabels")
        ori_labels = [t.get_text() for t in get_ori_labels()]
        ori_levels = categorical_order(long_df[kws[orient]])
        assert ori_labels == ori_levels

        legend = ax.get_legend()
        assert legend.get_title().get_text() == kws["hue"]

        hue_labels = [t.get_text() for t in legend.texts]
        hue_levels = categorical_order(long_df[kws["hue"]])
        assert hue_labels == hue_levels

    def test_labels_wide(self, wide_df):

        wide_df = wide_df.rename_axis("cols", axis=1)
        ax = self.func(wide_df)

        # To populate texts; only needed on older matplotlibs
        _draw_figure(ax.figure)

        assert ax.get_xlabel() == wide_df.columns.name
    
# ... [truncated] ...
data is long_df

        g2 = catplot(x=long_df["a"], y=long_df["y"], col=long_df["c"])
        assert g2.data.equals(long_df[["a", "y", "c"]])

    @pytest.mark.parametrize("var", ["col", "row"])
    def test_array_faceter(self, long_df, var):

        g1 = catplot(data=long_df, x="y", **{var: "a"})
        g2 = catplot(data=long_df, x="y", **{var: long_df["a"].to_numpy()})

        for ax1, ax2 in zip(g1.axes.flat, g2.axes.flat):
            assert_plots_equal(ax1, ax2)

    def test_invalid_kind(self, long_df):

        with pytest.raises(ValueError, match="Invalid `kind`: 'wrong'"):
            catplot(long_df, kind="wrong")

    def test_legend_with_auto(self):

        g1 = catplot(self.df, x="g", y="y", hue="g", legend='auto')
        assert g1._legend is None

        g2 = catplot(self.df, x="g", y="y", hue="g", legend=True)
        assert g2._legend is not None

    def test_weights_warning(self, long_df):

        with pytest.warns(UserWarning, match="The `weights` parameter"):
            g = catplot(long_df, x="a", y="y", weights="z")
        assert g.ax is not None


class TestBeeswarm:

    def test_could_overlap(self):

        p = Beeswarm()
        neighbors = p.could_overlap(
            (1, 1, .5),
            [(0, 0, .5),
             (1, .1, .2),
             (.5, .5, .5)]
        )
        assert_array_equal(neighbors, [(.5, .5, .5)])

    def test_position_candidates(self):

        p = Beeswarm()
        xy_i = (0, 1, .5)
        neighbors = [(0, 1, .5), (0, 1.5, .5)]
        candidates = p.position_candidates(xy_i, neighbors)
        dx1 = 1.05
        dx2 = np.sqrt(1 - .5 ** 2) * 1.05
        assert_array_equal(
            candidates,
            [(0, 1, .5), (-dx1, 1, .5), (dx1, 1, .5), (dx2, 1, .5), (-dx2, 1, .5)]
        )

    def test_find_first_non_overlapping_candidate(self):

        p = Beeswarm()
        candidates = [(.5, 1, .5), (1, 1, .5), (1.5, 1, .5)]
        neighbors = np.array([(0, 1, .5)])

        first = p.first_non_overlapping_candidate(candidates, neighbors)
        assert_array_equal(first, (1, 1, .5))

    def test_beeswarm(self, long_df):

        p = Beeswarm()
        data = long_df["y"]
        d = data.diff().mean() * 1.5
        x = np.zeros(data.size)
        y = np.sort(data)
        r = np.full_like(y, d)
        orig_xyr = np.c_[x, y, r]
        swarm = p.beeswarm(orig_xyr)[:, :2]
        dmat = np.sqrt(np.sum(np.square(swarm[:, np.newaxis] - swarm), axis=-1))
        triu = dmat[np.triu_indices_from(dmat, 1)]
        assert_array_less(d, triu)
        assert_array_equal(y, swarm[:, 1])

    def test_add_gutters(self):

        p = Beeswarm(width=1)

        points = np.zeros(10)
        t_fwd = t_inv = lambda x: x
        assert_array_equal(points, p.add_gutters(points, 0, t_fwd, t_inv))

        points = np.array([0, -1, .4, .8])
        msg = r"50.0% of the points cannot be placed.+$"
        with pytest.warns(UserWarning, match=msg):
            new_points = p.add_gutters(points, 0, t_fwd, t_inv)
        assert_array_equal(new_points, np.array([0, -.5, .4, .5]))


class TestBoxPlotContainer:

    @pytest.fixture
    def container(self, wide_array):

        ax = mpl.figure.Figure().subplots()
        artist_dict = ax.boxplot(wide_array)
        return BoxPlotContainer(artist_dict)

    def test_repr(self, container, wide_array):

        n = wide_array.shape[1]
        assert str(container) == f"<BoxPlotContainer object with {n} boxes>"

    def test_iteration(self, container):
        for artist_tuple in container:
            for attr in ["box", "median", "whiskers", "caps", "fliers", "mean"]:
                assert hasattr(artist_tuple, attr)

    def test_label(self, container):

        label = "a box plot"
        container.set_label(label)
        assert container.get_label() == label

    def test_children(self, container):

        children = container.get_children()
        for child in children:
            assert isinstance(child, mpl.artist.Artist)

