__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt, Emiel Deprost"

from plotly_resampler import FigureResampler, FigureWidgetResampler
from functools import wraps

import plotly
import warnings

WRAPPED_PREFIX = "[Plotly-Resampler]__"
PLOTLY_MODULES = [plotly.graph_objects, plotly.graph_objs]
PLOTLY_CONSTRUCTOR_WRAPPER = {
    "Figure": FigureResampler,
    "FigureWidget": FigureWidgetResampler,
}


def _already_wrapped(constr):
    return constr.__name__.startswith(WRAPPED_PREFIX)


def register_plotly_resampler():
    for mod in PLOTLY_MODULES:
        for constr_name, pr_class in PLOTLY_CONSTRUCTOR_WRAPPER.items():
            constr = getattr(mod, constr_name)
            if _already_wrapped(constr):
                # Stop if plotly-resampler is already registered
                warnings.warn(
                    "plotly-resampler is already registered!",
                    category=UserWarning,
                )
                return

            @wraps(constr)
            def wrapped_constr(*args, **kwargs):
                return FigureResampler(constr(*args, **kwargs), verbose=True)

            wrapped_constr.__name__ = WRAPPED_PREFIX + constr_name.__name__
            setattr(mod, constr_name, wrapped_constr)
