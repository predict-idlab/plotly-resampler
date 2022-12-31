"""
Compatible implementation for various downsample methods and open interface to 
other downsample methods.

"""

__author__ = "Jonas Van Der Donckt"


from .aggregation_interface import AbstractSeriesArgDownsampler
from .aggregators import (  # FuncAggregator,
    LTTB,
    EfficientLTTB,
    EveryNthPoint,
    MinMaxAggregator,
    MinMaxOverlapAggregator,
)

__all__ = [
    "AbstractSeriesArgDownsampler",
    "LTTB",
    "EfficientLTTB",
    "EveryNthPoint",
    # "FuncAggregator",
    "MinMaxAggregator",
    "MinMaxOverlapAggregator",
]
