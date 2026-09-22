"""The classes for specifying and compiling a declarative visualization."""
from __future__ import annotations

import io
import os
import re
import inspect
import itertools
import textwrap
from contextlib import contextmanager
from collections import abc
from collections.abc import Callable, Generator
from typing import Any, List, Literal, Optional, cast
from xml.etree import ElementTree

from cycler import cycler
import pandas as pd
from pandas import DataFrame, Series, Index
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move
from seaborn._core.scales import Scale
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.properties import PROPERTIES, Property
from seaborn._core.typing import (
    DataSource,
    VariableSpec,
    VariableSpecList,
    OrderSpec,
    Default,
)
from seaborn._core.exceptions import PlotSpecError
from seaborn._core.rules import categorical_order
from seaborn._compat import set_layout_engine
from seaborn.rcmod import axes_style, plotting_context
from seaborn.palettes import color_palette

from typing import TYPE_CHECKING, TypedDict
if TYPE_CHECKING:
    from matplotlib.figure import SubFigure


default = Default()


# ---- Definitions for internal specs ---------------------------------------------- #


class Layer(TypedDict, total=False):

    mark: Mark  # TODO allow list?
    stat: Stat | None  # TODO allow list?
    move: Move | list[Move] | None
    data: PlotData
    source: DataSource
    vars: dict[str, VariableSpec]
    orient: str
    legend: bool
    label: str | None


class FacetSpec(TypedDict, total=False):

    variables: dict[str, VariableSpec]
    structure: dict[str, list[str]]
    wrap: int | None


class PairSpec(TypedDict, total=False):

    variables: dict[str, VariableSpec]
    structure: dict[str, list[str]]
    cross: bool
    wrap: int | None


# --- Local helpers ---------------------------------------------------------------- #


@contextmanager
def theme_context(params: dict[str, Any]) -> Generator:
    """Temporarily modify specifc matplotlib rcParams."""
    orig_params = {k: mpl.rcParams[k] for k in params}
    color_codes = "bgrmyck"
    nice_colors = [*color_palette("deep6"), (.15, .15, .15)]
    orig_colors = [mpl.colors.colorConverter.colors[x] for x in color_codes]
    # TODO how to allow this to reflect the color cycle when relevant?
    try:
        mpl.rcParams.update(params)
        for (code, color) in zip(color_codes, nice_colors):
            mpl.colors.colorConverter.colors[code] = color
        yield
    finally:
        mpl.rcParams.update(orig_params)
        for (code, color) in zip(color_codes, orig_colors):
            mpl.colors.colorConverter.colors[code] = color


def build_plot_signature(cls):
    """
    Decorator function for giving Plot a useful signature.

    Currently this mostly saves us some duplicated typing, but we would
    like eventually to have a way of registering new semantic properties,
    at which point dynamic signature generation would become more important.

    """
    sig = inspect.signature(cls)
    params = [
        inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
        inspect.Parameter("data", inspect.Parameter.KEYWORD_ONLY, default=None)
    ]
    params.extend([
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
        for name in PROPERTIES
    ])
    new_sig = sig.replace(parameters=params)
    cls.__signature__ = new_sig

    known_properties = textwrap.fill(
        ", ".join([f"|{p}|" for p in PROPERTIES]),
        width=78, subsequent_indent=" " * 8,
    )

    if cls.__doc__ is not None:  # support python -OO mode
        cls.__doc__ = cls.__doc__
# ... [truncated] ...
        existing_artists[i] += tuple([new_artist])

        # When using pyplot, an "external" legend won't be shown, so this
        # keeps it inside the axes (though still attached to the figure)
        # This is necessary because matplotlib layout engines currently don't
        # support figure legends — ideally this will change.
        loc = "center right" if self._pyplot else "center left"

        base_legend = None
        for (name, _), (handles, labels) in merged_contents.items():

            legend = mpl.legend.Legend(
                self._figure,
                handles,  # type: ignore  # matplotlib/issues/26639
                labels,
                title=name,
                loc=loc,
                bbox_to_anchor=(.98, .55),
            )

            if base_legend:
                # Matplotlib has no public API for this so it is a bit of a hack.
                # Ideally we'd define our own legend class with more flexibility,
                # but that is a lot of work!
                base_legend_box = base_legend.get_children()[0]
                this_legend_box = legend.get_children()[0]
                base_legend_box.get_children().extend(this_legend_box.get_children())
            else:
                base_legend = legend
                self._figure.legends.append(legend)

    def _finalize_figure(self, p: Plot) -> None:

        for sub in self._subplots:
            ax = sub["ax"]
            for axis in "xy":
                axis_key = sub[axis]
                axis_obj = getattr(ax, f"{axis}axis")

                # Axis limits
                if axis_key in p._limits or axis in p._limits:
                    convert_units = getattr(ax, f"{axis}axis").convert_units
                    a, b = p._limits.get(axis_key) or p._limits[axis]
                    lo = a if a is None else convert_units(a)
                    hi = b if b is None else convert_units(b)
                    if isinstance(a, str):
                        lo = cast(float, lo) - 0.5
                    if isinstance(b, str):
                        hi = cast(float, hi) + 0.5
                    ax.set(**{f"{axis}lim": (lo, hi)})

                if axis_key in self._scales:  # TODO when would it not be?
                    self._scales[axis_key]._finalize(p, axis_obj)

        if (engine_name := p._layout_spec.get("engine", default)) is not default:
            # None is a valid arg for Figure.set_layout_engine, hence `default`
            set_layout_engine(self._figure, engine_name)
        elif p._target is None:
            # Don't modify the layout engine if the user supplied their own
            # matplotlib figure and didn't specify an engine through Plot
            # TODO switch default to "constrained"?
            # TODO either way, make configurable
            set_layout_engine(self._figure, "tight")

        if (extent := p._layout_spec.get("extent")) is not None:
            engine = get_layout_engine(self._figure)
            if engine is None:
                self._figure.subplots_adjust(*extent)
            else:
                # Note the different parameterization for the layout engine rect...
                left, bottom, right, top = extent
                width, height = right - left, top - bottom
                try:
                    # The base LayoutEngine.set method doesn't have rect= so we need
                    # to avoid typechecking this statement. We also catch a TypeError
                    # as a plugin LayoutEngine may not support it either.
                    # Alternatively we could guard this with a check on the engine type,
                    # but that would make later-developed engines would un-useable.
                    engine.set(rect=[left, bottom, width, height])  # type: ignore
                except TypeError:
                    # Should we warn / raise? Note that we don't expect to get here
                    # under any normal circumstances.
                    pass

