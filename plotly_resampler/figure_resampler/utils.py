"""Utility functions for the figure_resampler submodule."""

import math

import pandas as pd
from plotly.basedatatypes import BaseFigure

try:  # Fails when IPywidgets is not installed
    from plotly.basewidget import BaseFigureWidget
except (ImportError, ModuleNotFoundError):
    BaseFigureWidget = type(None)

from typing import Any

### Checks for the figure type


def is_figure(figure: Any) -> bool:
    """Check if the figure is a plotly go.Figure or a FigureResampler.

    !!! note

        This method does not use isinstance(figure, go.Figure) as this will not work
        when go.Figure is decorated (after executing the
        ``register_plotly_resampler`` function).

    Parameters
    ----------
    figure : Any
        The figure to check.

    Returns
    -------
    bool
        True if the figure is a plotly go.Figure or a FigureResampler.
    """
    return isinstance(figure, BaseFigure) and (not isinstance(figure, BaseFigureWidget))


def is_figurewidget(figure: Any):
    """Check if the figure is a plotly go.FigureWidget or a FigureWidgetResampler.

    !!! note

        This method does not use isinstance(figure, go.FigureWidget) as this will not
        work when go.FigureWidget is decorated (after executing the
        ``register_plotly_resampler`` function).

    Parameters
    ----------
    figure : Any
        The figure to check.

    Returns
    -------
    bool
        True if the figure is a plotly go.FigureWidget or a FigureWidgetResampler.
    """
    return isinstance(figure, BaseFigureWidget)


def is_fr(figure: Any) -> bool:
    """Check if the figure is a FigureResampler.

    !!! note

        This method will not return True if the figure is a plotly go.Figure.

    Parameters
    ----------
    figure : Any
        The figure to check.

    Returns
    -------
    bool
        True if the figure is a FigureResampler.
    """
    from plotly_resampler import FigureResampler

    return isinstance(figure, FigureResampler)


def is_fwr(figure: Any) -> bool:
    """Check if the figure is a FigureWidgetResampler.

    !!! note

        This method will not return True if the figure is a plotly go.FigureWidget.

    Parameters
    ----------
    figure : Any
        The figure to check.

    Returns
    -------
    bool
        True if the figure is a FigureWidgetResampler.
    """
    from plotly_resampler import FigureWidgetResampler

    return isinstance(figure, FigureWidgetResampler)


### Rounding functions for bin size


def timedelta_to_str(td: pd.Timedelta) -> str:
    """Construct a tight string representation for the given timedelta arg.

    Parameters
    ----------
    td: pd.Timedelta
        The timedelta for which the string representation is constructed

    Returns
    -------
    str:
        The tight string bounds of format '$d-$h$m$s.$ms'.
        If the timedelta is negative, the string starts with 'NEG'.

    """
    out_str = ""

    # Edge case if we deal with negative
    if td < pd.Timedelta(seconds=0):
        td *= -1
        out_str += "NEG"

    # Note: this must happen after the *= -1
    c = td.components
    if c.days > 0:
        out_str += f"{c.days}D"
    if c.hours > 0 or c.minutes > 0 or c.seconds > 0 or c.milliseconds > 0:
        out_str += "_" if out_str else ""  # add seperator if non-empty

    if c.hours > 0:
        out_str += f"{c.hours}h"
    if c.minutes > 0:
        out_str += f"{c.minutes}m"
    if c.seconds > 0:
        if c.milliseconds:
            out_str += (
                f"{c.seconds}.{str(c.milliseconds / 1000).split('.')[-1].rstrip('0')}s"
            )
        else:
            out_str += f"{c.seconds}s"
    elif c.milliseconds > 0:
        out_str += f"{c.milliseconds}ms"
    if c.microseconds > 0:
        out_str += f"{c.microseconds}us"
    if c.nanoseconds > 0:
        out_str += f"{c.nanoseconds}ns"
    return out_str


def round_td_str(td: pd.Timedelta) -> str:
    """Round a timedelta to the nearest unit and convert to a string.

    Parameters
    ----------
    td : pd.Timedelta
        The timedelta to round.

    Returns
    -------
    str
        The rounded timedelta as a string.
        If the timedelta is == 0, None is returned.

    !!! info "See Also"
        [`timedelta_to_str`][figure_resampler.utils.timedelta_to_str]

    """
    for t_s in ("D", "h", "min", "s", "ms", "us", "ns"):
        if td > 0.95 * pd.Timedelta(f"1{t_s}"):
            return timedelta_to_str(td.round(t_s))


def round_number_str(number: float) -> str:
    """Round a number to the nearest unit and convert to a string.

    Parameters
    ----------
    number : float
        The number to round.

    Returns
    -------
    str
        The rounded number as a string.
        If the number is == 0, None is returned.

    """
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number > 0.95:
        for unit, scaling in [
            ("T", int(1e12)),  # Trillion
            ("B", int(1e9)),  # Billion
            ("M", int(1e6)),  # Million
            ("k", int(1e3)),  # Thousand
        ]:
            if number / scaling > 0.95:
                return f"{round(number / scaling)}{unit}"
        return sign + str(round(number))
    if number > 0:  # avoid log10(0)
        # we have a number between 0-0.95 -> round till nearest non-zero digit
        return sign + str(round(number, 1 + abs(int(math.log10(number)))))
