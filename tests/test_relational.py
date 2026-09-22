from itertools import product
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import same_color, to_rgba

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from seaborn.palettes import color_palette
from seaborn._base import categorical_order, unique_markers

from seaborn.relational import (
    _RelationalPlotter,
    _LinePlotter,
    _ScatterPlotter,
    relplot,
    lineplot,
    scatterplot
)

from seaborn.utils import _draw_figure, _version_predates
from seaborn._compat import get_colormap, get_legend_handles
from seaborn._testing import assert_plots_equal


@pytest.fixture(params=[
    dict(x="x", y="y"),
    dict(x="t", y="y"),
    dict(x="a", y="y"),
    dict(x="x", y="y", hue="y"),
    dict(x="x", y="y", hue="a"),
    dict(x="x", y="y", size="a"),
    dict(x="x", y="y", style="a"),
    dict(x="x", y="y", hue="s"),
    dict(x="x", y="y", size="s"),
    dict(x="x", y="y", style="s"),
    dict(x="x", y="y", hue="a", style="a"),
    dict(x="x", y="y", hue="a", size="b", style="b"),
])
def long_semantics(request):
    return request.param


class Helpers:

    @pytest.fixture
    def levels(self, long_df):
        return {var: categorical_order(long_df[var]) for var in ["a", "b"]}

    def scatter_rgbs(self, collections):
        rgbs = []
        for col in collections:
            rgb = tuple(col.get_facecolor().squeeze()[:3])
            rgbs.append(rgb)
        return rgbs

    def paths_equal(self, *args):

        equal = all([len(a) == len(args[0]) for a in args])

        for p1, p2 in zip(*args):
            equal &= np.array_equal(p1.vertices, p2.vertices)
            equal &= np.array_equal(p1.codes, p2.codes)
        return equal


class SharedAxesLevelTests:

    def test_color(self, long_df):

        ax = plt.figure().subplots()
        self.func(data=long_df, x="x", y="y", ax=ax)
        assert self.get_last_color(ax) == to_rgba("C0")

        ax = plt.figure().subplots()
        self.func(data=long_df, x="x", y="y", ax=ax)
        self.func(data=long_df, x="x", y="y", ax=ax)
        assert self.get_last_color(ax) == to_rgba("C1")

        ax = plt.figure().subplots()
        self.func(data=long_df, x="x", y="y", color="C2", ax=ax)
        assert self.get_last_color(ax) == to_rgba("C2")

        ax = plt.figure().subplots()
        self.func(data=long_df, x="x", y="y", c="C2", ax=ax)
        assert self.get_last_color(ax) == to_rgba("C2")


class TestRelationalPlotter(Helpers):

    def test_wide_df_variables(self, wide_df):

        p = _RelationalPlotter()
        p.assign_variables(data=wide_df)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == np.prod(wide_df.shape)

        x = p.plot_data["x"]
        expected_x = np.tile(wide_df.index, wide_df.shape[1])
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = wide_df.to_numpy().ravel(order="f")
        assert_array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(wide_df.columns.to_numpy(), wide_df.shape[0])
        assert_array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == wide_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == wide_df.columns.name
        assert p.variables["style"] == wide_df.columns.name

    def test_wide_df_with_nonnumeric_variables(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        numeric_df = long_df.select_dtypes("number")

        assert len(p.plot_data) == np.prod(numeric_df.shape)

        x = p.plot_data["x"]
        expected_x =
# ... [truncated] ...
        for key in set(legend_data[0]) & set(legend_data[1]):
            if key == "y":
                # At some point (circa 3.0) matplotlib auto-added pandas series
                # with a valid name into the legend, which messes up this test.
                # I can't track down when that was added (or removed), so let's
                # just anticipate and ignore it here.
                continue
            assert legend_data[0][key] == legend_data[1][key]

    def test_datetime_scale(self, long_df):

        ax = scatterplot(data=long_df, x="t", y="y")
        # Check that we avoid weird matplotlib default auto scaling
        # https://github.com/matplotlib/matplotlib/issues/17586
        ax.get_xlim()[0] > ax.xaxis.convert_units(np.datetime64("2002-01-01"))

    def test_unfilled_marker_edgecolor_warning(self, long_df):  # GH2636

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scatterplot(data=long_df, x="x", y="y", marker="+")

    def test_short_form_kwargs(self, long_df):

        ax = scatterplot(data=long_df, x="x", y="y", ec="g")
        pts = ax.collections[0]
        assert same_color(pts.get_edgecolors().squeeze(), "g")

    def test_scatterplot_vs_relplot(self, long_df, long_semantics):

        ax = scatterplot(data=long_df, **long_semantics)
        g = relplot(data=long_df, kind="scatter", **long_semantics)

        for s_pts, r_pts in zip(ax.collections, g.ax.collections):

            assert_array_equal(s_pts.get_offsets(), r_pts.get_offsets())
            assert_array_equal(s_pts.get_sizes(), r_pts.get_sizes())
            assert_array_equal(s_pts.get_facecolors(), r_pts.get_facecolors())
            assert self.paths_equal(s_pts.get_paths(), r_pts.get_paths())

    def test_scatterplot_smoke(
        self,
        wide_df, wide_array,
        flat_series, flat_array, flat_list,
        wide_list_of_series, wide_list_of_arrays, wide_list_of_lists,
        long_df, null_df, object_df
    ):

        f, ax = plt.subplots()

        scatterplot(x=[], y=[])
        ax.clear()

        scatterplot(data=wide_df)
        ax.clear()

        scatterplot(data=wide_array)
        ax.clear()

        scatterplot(data=wide_list_of_series)
        ax.clear()

        scatterplot(data=wide_list_of_arrays)
        ax.clear()

        scatterplot(data=wide_list_of_lists)
        ax.clear()

        scatterplot(data=flat_series)
        ax.clear()

        scatterplot(data=flat_array)
        ax.clear()

        scatterplot(data=flat_list)
        ax.clear()

        scatterplot(x="x", y="y", data=long_df)
        ax.clear()

        scatterplot(x=long_df.x, y=long_df.y)
        ax.clear()

        scatterplot(x=long_df.x, y="y", data=long_df)
        ax.clear()

        scatterplot(x="x", y=long_df.y.to_numpy(), data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", style="a", data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", style="b", data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", style="a", data=null_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", style="b", data=null_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", size="a", data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", size="s", data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", size="a", data=null_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", size="s", data=null_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="f", data=object_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="c", size="f", data=object_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="f", size="s", data=object_df)
        ax.clear()

