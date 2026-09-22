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
        expected_x = numeric_df.index

        y = p.plot_data["y"]
        expected_y = numeric_df.to_numpy().ravel(order="f")

        hue = p.plot_data["hue"]
        expected_hue = numeric_df.columns

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == numeric_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == numeric_df.columns.name
        assert p.variables["style"] == numeric_df.columns.name

    def test_wide_df_with_nonnumeric_variables_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        numeric_df = long_df.select_dtypes("number")

        assert len(p.plot_data) == np.prod(numeric_df.shape)

        x = p.plot_data["x"]
        expected_x = numeric_df.index

        y = p.plot_data["y"]
        expected_y = numeric_df.to_numpy().ravel(order="f")

        hue = p.plot_data["hue"]
        expected_hue = numeric_df.columns

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == numeric_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == numeric_df.columns.name
        assert p.variables["style"] == numeric_df.columns.name

    def test_long_df_variables(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_wide_df_variables_with_missing_values(self, wide_df):

        p = _RelationalPlotter()
        p.assign_variables(data=wide_df)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(wide_df)

        x = p.plot_data["x"]
        expected_x = wide_df.index

        y = p.plot_data["y"]
        expected_y = wide_df.to_numpy().ravel(order="f")

        hue = p.plot_data["hue"]
        expected_hue = wide_df.columns

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == wide_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == wide_df.columns.name
        assert p.variables["style"] == wide_df.columns.name

    def test_wide_df_variables_with_missing_values_and_non_numeric_columns(self, wide_df):

        p = _RelationalPlotter()
        p.assign_variables(data=wide_df)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(wide_df)

        x = p.plot_data["x"]
        expected_x = wide_df.index

        y = p.plot_data["y"]
        expected_y = wide_df.to_numpy().ravel(order="f")

        hue = p.plot_data["hue"]
        expected_hue = wide_df.columns

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == wide_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == wide_df.columns.name
        assert p.variables["style"] == wide_df.columns.name

    def test_long_df_variables_with_missing_values_and_non_numeric_columns(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_wide_df_variables_with_missing_values_and_non_numeric_columns(self, wide_df):

        p = _RelationalPlotter()
        p.assign_variables(data=wide_df)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(wide_df)

        x = p.plot_data["x"]
        expected_x = wide_df.index

        y = p.plot_data["y"]
        expected_y = wide_df.to_numpy().ravel(order="f")

        hue = p.plot_data["hue"]
        expected_hue = wide_df.columns

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == wide_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == wide_df.columns.name
        assert p.variables["style"] == wide_df.columns.name

    def test_wide_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, wide_df):

        p = _RelationalPlotter()
        p.assign_variables(data=wide_df)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(wide_df)

        x = p.plot_data["x"]
        expected_x = wide_df.index

        y = p.plot_data["y"]
        expected_y = wide_df.to_numpy().ravel(order="f")

        hue = p.plot_data["hue"]
        expected_hue = wide_df.columns

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == wide_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == wide_df.columns.name
        assert p.variables["style"] == wide_df.columns.name

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df)

        x = p.plot_data["x"]
        expected_x = long_df.x

        y = p.plot_data["y"]
        expected_y = long_df.y

        hue = p.plot_data["hue"]
        expected_hue = long_df.a

        style = p.plot_data["style"]
        expected_style = hue

        assert_array_equal(x, expected_x)
        assert_array_equal(y, expected_y)
        assert_array_equal(hue, expected_hue)
        assert_array_equal(style, expected_style)

        assert p.variables["x"] == "x"
        assert p.variables["y"] == "y"
        assert p.variables["hue"] == "a"
        assert p.variables["style"] == "a"

    def test_long_df_variables_with_missing_values_and_non_numeric_columns_and_missing_values(self, long_df):

        p = _RelationalPlotter()
        p.assign_variables(data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == len(long_df