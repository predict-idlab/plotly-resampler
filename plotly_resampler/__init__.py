"""**plotly\_resampler**: visualizing large sequences."""

from .aggregation import LTTB, EfficientLTTB, EveryNthPoint
from .figure_resampler import FigureResampler, FigureWidgetResampler
from .registering import register_plotly_resampler, unregister_plotly_resampler

__docformat__ = "numpy"
__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"
__version__ = "0.8.2"

__all__ = [
    "__version__",
    "FigureResampler",
    "FigureWidgetResampler",
    "EfficientLTTB",
    "LTTB",
    "EveryNthPoint",
    "register_plotly_resampler",
    "unregister_plotly_resampler",
]


try:  # Enable ipywidgets on google colab!
    import sys

    if "google.colab" in sys.modules:
        from google.colab import output

        output.enable_custom_widget_manager()
except (ImportError, ModuleNotFoundError):
    pass
