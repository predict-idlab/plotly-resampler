"""
Compatible implementation for various downsample methods and open interface to 
other downsample methods.

"""

__author__ = "Jonas Van Der Donckt"


from .aggregation_interface import DataPointSelector
from .aggregators import (
    LTTB,
    EveryNthPoint,
    FuncAggregator,
    MinMaxAggregator,
    MinMaxLTTB,
    MinMaxOverlapAggregator,
)

__all__ = [
    "DataPointSelector",
    "LTTB",
    "MinMaxLTTB",
    "EveryNthPoint",
    "FuncAggregator",
    "MinMaxAggregator",
    "MinMaxOverlapAggregator",
]
