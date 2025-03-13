"""**plotly_resampler**: visualizing large sequences."""

import contextlib

from .aggregation import LTTB, EveryNthPoint, MinMaxLTTB
from .figure_resampler import ASSETS_FOLDER, FigureResampler, FigureWidgetResampler
from .registering import register_plotly_resampler, unregister_plotly_resampler

__docformat__ = "numpy"
__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"
__version__ = "0.11.0rc0"

__all__ = [
    "__version__",
    "FigureResampler",
    "FigureWidgetResampler",
    "ASSETS_FOLDER",
    "MinMaxLTTB",
    "LTTB",
    "EveryNthPoint",
    "register_plotly_resampler",
    "unregister_plotly_resampler",
]


# Enable ipywidgets on google colab!
with contextlib.suppress(ImportError, ModuleNotFoundError):
    import sys

    if "google.colab" in sys.modules:
        from google.colab import output

        output.enable_custom_widget_manager()
