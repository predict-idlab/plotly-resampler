"""**plotly\_resampler**: visualizing large sequences

"""

from .figure_resampler import FigureResampler
from .aggregation import LTTB, EveryNthPoint, FuncAggregator

__docformat__ = "numpy"
__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"
__version__ = "0.4.0"

__all__ = [
    "__version__",
    "FigureResampler",
    "LTTB",
    "EveryNthPoint",
    "FuncAggregator",
]
