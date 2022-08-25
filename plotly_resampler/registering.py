"""Register plotly-resampler to (un)wrap plotly-graph-objects."""

__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt, Emiel Deprost"

from plotly_resampler import FigureResampler, FigureWidgetResampler
from plotly_resampler.figure_resampler.figure_resampler_interface import (
    AbstractFigureAggregator,
)
from functools import wraps

import plotly

WRAPPED_PREFIX = "[Plotly-Resampler]__"
PLOTLY_MODULES = [
    plotly.graph_objs,
    plotly.graph_objects,
]  # wait for this PR https://github.com/plotly/plotly.py/pull/3779
PLOTLY_CONSTRUCTOR_WRAPPER = {
    "Figure": FigureResampler,
    "FigureWidget": FigureWidgetResampler,
}


def _already_wrapped(constr):
    return constr.__name__.startswith(WRAPPED_PREFIX)


def _get_plotly_constr(constr):
    """Return the constructor of the underlying plotly graph object and thus omit the
    possibly wrapped :class:`AbstractFigureAggregator <plotly_resampler.figure_resampler.figure_resampler_interface.AbstractFigureAggregator>`
    instance.

    Parameters
    ----------
    constr : callable
        The constructor of a instantiatedplotly-object.

    Returns
    -------
    callable
        The constructor of a ``go.FigureWidget`` or a ``go.Figure``.
    """
    if _already_wrapped(constr):
        return constr.__wrapped__  # get the original constructor
    return constr


### Registering the wrappers


def _is_ipython_env():
    """Check if we are in an IPython environment (with a kernel)."""
    try:
        from IPython import get_ipython

        return "IPKernelApp" in get_ipython().config
    except (ImportError, AttributeError):
        return False


def _register_wrapper(
    module: type,
    constr_name: str,
    pr_class: AbstractFigureAggregator,
    **aggregator_kwargs,
):
    constr = getattr(module, constr_name)
    constr = _get_plotly_constr(constr)  # get the original plotly constructor

    # print(f"Wrapping {constr_name} with {pr_class}")

    @wraps(constr)
    def wrapped_constr(*args, **kwargs):
        # print(f"Executing constructor wrapper for {constr_name}", constr)
        return pr_class(constr(*args, **kwargs), **aggregator_kwargs)

    wrapped_constr.__name__ = WRAPPED_PREFIX + constr_name
    setattr(module, constr_name, wrapped_constr)


def register_plotly_resampler(mode="auto", **aggregator_kwargs):
    """Register plotly-resampler to plotly.graph_objects.

    This function results in the use of plotly-resampler under the hood.

    .. Note::
        We advise to use mode= ``widget`` when working in an IPython based environment
        as this will just behave as a ``go.FigureWidget``, but with dynamic aggregation.
        When using mode= ``auto`` or ``figure``; most figures will be wrapped as
        :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>`,
        on which
        :func:`show_dash <plotly_resampler.figure_resampler.FigureResampler.show_dash>`
        needs to be called.

    Parameters
    ----------
    mode : str, optional
        The mode of the plotly-resampler.
        Possible values are: 'auto', 'figure', 'widget', None.
        If 'auto' is used, the mode is determined based on the environment; if it is in
        an ipython environment, the mode is 'widget', otherwise it is 'figure'.
        If 'figure' is used, all plotly figures are wrapped as FigureResampler objects.
        If 'widget' is used, all plotly figure widgets are wrapped as
        FigureWidgetResampler objects (we advise to use this mode in ipython environment
        with a kernel).
        If None is used, wrapping is done as expected (go.Figure -> FigureResampler,
        go.FigureWidget -> FigureWidgetResampler).
    aggregator_kwargs : dict, optional
        The keyword arguments to pass to the plotly-resampler decorator its constructor.
        See more details in :class:`FigureResampler <FigureResampler>` and
        :class:`FigureWidgetResampler <FigureWidgetResampler>`.

    """
    for constr_name, pr_class in PLOTLY_CONSTRUCTOR_WRAPPER.items():
        if (mode == "auto" and _is_ipython_env()) or mode == "widget":
            pr_class = FigureWidgetResampler
        elif mode == "figure":
            pr_class = FigureResampler
        # else: default mode -> wrap according to PLOTLY_CONSTRUCTOR_WRAPPER

        for module in PLOTLY_MODULES:
            _register_wrapper(module, constr_name, pr_class, **aggregator_kwargs)


### Unregistering the wrappers


def _unregister_wrapper(module: type, constr_name: str):
    constr = getattr(module, constr_name)
    if _already_wrapped(constr):
        constr = constr.__wrapped__
        setattr(module, constr_name, constr)


def unregister_plotly_resampler():
    """Unregister plotly-resampler from plotly.graph_objects."""
    for constr in PLOTLY_CONSTRUCTOR_WRAPPER.keys():
        for module in PLOTLY_MODULES:
            _unregister_wrapper(module, constr)
