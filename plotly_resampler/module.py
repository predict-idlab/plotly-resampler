__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt, Emiel Deprost"

from plotly_resampler import FigureResampler, FigureWidgetResampler
from plotly_resampler.figure_resampler.figure_resampler_interface import (
    AbstractFigureAggregator,
)
from functools import wraps
from importlib import reload

import plotly

WRAPPED_PREFIX = "[Plotly-Resampler]__"
PLOTLY_MODULE = plotly.graph_objs  # plotly.graph_objects is an alias for this module
PLOTLY_CONSTRUCTOR_WRAPPER = {
    "Figure": FigureResampler,
    "FigureWidget": FigureWidgetResampler,
}


def _already_wrapped(constr):
    return constr.__name__.startswith(WRAPPED_PREFIX)


### Registering the wrappers


def _register_wrapper(
    constr_name: str, pr_class: AbstractFigureAggregator, **aggregator_kwargs
):
    constr = getattr(reload(PLOTLY_MODULE), constr_name)

    if _already_wrapped(constr):
        constr = constr.__wrapped__  # get the original constructor

    @wraps(constr)
    def wrapped_constr(*args, **kwargs):
        # if isinstance(constr, pr_class): # TODO: why did I add this in the first place?
        # return constr(*args, **kwargs)
        return pr_class(constr(*args, **kwargs), **aggregator_kwargs)

    wrapped_constr.__name__ = WRAPPED_PREFIX + constr_name
    setattr(PLOTLY_MODULE, constr_name, wrapped_constr)


def register_plotly_resampler(
    ipython_env=False, **aggregator_kwargs
):  # TODO: better argument names (ipython_env is geen top naam) -> mss zelfs een auto mode ofz???
    """Register plotly-resampler to plotly.graph_objects.

    This function results in the use of plotly-resampler under the hood.
    """
    for constr_name, pr_class in PLOTLY_CONSTRUCTOR_WRAPPER.items():
        if ipython_env:
            # TODO: if ipython_env and on google colab -> apply janky colab fix here as well...
            pr_class = FigureWidgetResampler
        _register_wrapper(constr_name, pr_class, **aggregator_kwargs)


### Unregistering the wrappers


def _unregister_wrapper(constr_name: str):
    constr = getattr(PLOTLY_MODULE, constr_name)
    if _already_wrapped(constr):
        constr = constr.__wrapped__
        setattr(PLOTLY_MODULE, constr_name, constr)


def unregister_plotly_resampler():
    """Unregister plotly-resampler from plotly.graph_objects."""
    for constr in PLOTLY_CONSTRUCTOR_WRAPPER.keys():
        _unregister_wrapper(constr)
