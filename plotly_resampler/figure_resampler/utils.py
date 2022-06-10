from plotly.basedatatypes import BaseFigure
from plotly.basewidget import BaseFigureWidget

from typing import Any


def _is_figure(figure: Any) -> bool:
    """Check if the figure is a plotly go.Figure.
    Note: this method does not use isinstance(figure, go.Figure) as this will not work
    when go.Figure is decorated (after executing the the `register_plotly_resampler`
    function).

    Parameters
    ----------
    figure : Any
        The figure to check.

    Returns
    -------
    bool
        True if the figure is a plotly go.Figure.
    """

    return isinstance(figure, BaseFigure) and not isinstance(figure, BaseFigureWidget)


def _is_figurewidget(figure: Any):
    """Check if the figure is a plotly go.FigureWidget.

    Note: this method does not use isinstance(figure, go.FigureWidget) as this will not
    work when go.FigureWidget is decorated (after executing the the
    `register_plotly_resampler` function).

    Parameters
    ----------
    figure : Any
        The figure to check.

    Returns
    -------
    bool
        True if the figure is a plotly go.FigureWidget.
    """
    return isinstance(figure, BaseFigureWidget)
