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
    countplot,
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

    @pytest.mark.parametrize("func", PLOT_FUNCS)
    @pytest.mark.parametrize("orient", ["x", "y"])
    @pytest.mark.parametrize("order", [None, ["a", "b", "c"]])
    @pytest.mark.parametrize("sort", [None, True, False])
    @pytest.mark.parametrize("hue_order", [None, ["a", "b", "c"]])
    @pytest.mark.parametrize("hue", [None, "b"])
    @pytest.mark.parametrize("palette", [None, "dark", "bright", "pastel"])
    @pytest.mark.parametrize("dodge", [None, False, True])
    @pytest.mark.parametrize("width", [None, 1, .5, .25])
    @pytest.mark.parametrize("errwidth", [None, 1, 2, 3])
    @pytest.mark.parametrize("n_boot", [None, 1000])
    @pytest.mark.parametrize("units", [None, "c"])
    @pytest.mark.parametrize("estimator", [None, "mean", "median"])
    @pytest.mark.parametrize("ci", [None, 68, 95, "sd"])
    @pytest.mark.parametrize("capsize", [None, 0, .1, .25, .5])
    @pytest.mark.parametrize("join", [None, False, True])
    @pytest.mark.parametrize("share", [None, "row", "col", "all"])
    @pytest.mark.parametrize("legend", [None, False, True, "auto"])
    @pytest.mark.parametrize("errcolor", [None, "red"])
    @pytest.mark.parametrize("linewidth", [None, 1, 2, 3])
    @pytest.mark.parametrize("color", [None, "red"])
    @pytest.mark.parametrize("dashes", [None, (1, 1)])
    @pytest.mark.parametrize("scale", [None, "area", "count", "width"])
    @pytest.mark.parametrize("scale_hue", [None, "area", "count", "width"])
    @pytest.mark.parametrize("linewidth_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit", [None, "area", "count", "width"])
    @pytest.mark.parametrize("scale_hue_unit", [None, "area", "count", "width"])
    @pytest.mark.parametrize("scale_unit_hue", [None, "area", "count", "width"])
    @pytest.mark.parametrize("scale_unit_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale", [None, 1, 2, 3])
    @pytest.mark.parametrize("scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale_hue_scale_unit_scale