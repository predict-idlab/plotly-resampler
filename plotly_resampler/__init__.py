"""**plotly\_resampler**: visualizing large sequences."""

from .aggregation import (
    LTTB,
    EfficientLTTB,
    EveryNthPoint,
    FuncAggregator,
    MinMaxOverlapAggregator,
)
from .figure_resampler import FigureResampler, FigureWidgetResampler

__docformat__ = "numpy"
__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"
__version__ = "0.6.4"

__all__ = [
    "__version__",
    "FigureResampler",
    "FigureWidgetResampler",
    "EfficientLTTB",
    "MinMaxOverlapAggregator",
    "LTTB",
    "EveryNthPoint",
    "FuncAggregator",
]
