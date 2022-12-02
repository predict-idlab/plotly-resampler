"""
Compatible implementation for various downsample methods and open interface to 
other downsample methods.

"""

__author__ = "Jonas Van Der Donckt"


from .aggregation_interface import AbstractSeriesAggregator
from .aggregators import (
    LTTB,
    EfficientLTTB,
    EveryNthPoint,
    FuncAggregator,
    MinMaxAggregator,
    MinMaxOverlapAggregator,
)

__all__ = [
    "AbstractSeriesAggregator",
    "LTTB",
    "EfficientLTTB",
    "EveryNthPoint",
    "FuncAggregator",
    "MinMaxAggregator",
    "MinMaxOverlapAggregator",
]
