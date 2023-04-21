"""
Compatible implementation for various downsample methods and open interface to 
other downsample methods.

"""

__author__ = "Jonas Van Der Donckt"


from .aggregation_interface import AbstractAggregator
from .aggregators import (
    LTTB,
    EveryNthPoint,
    FuncAggregator,
    MinMaxAggregator,
    MinMaxLTTB,
    MinMaxOverlapAggregator,
)
from .gap_handler_interface import AbstractGapHandler
from .gap_handlers import MedDiffGapHandler, NoGapHandler
from .plotly_aggregator_parser import PlotlyAggregatorParser

__all__ = [
    "AbstractAggregator",
    "AbstractGapHandler",
    "PlotlyAggregatorParser",
    "LTTB",
    "MinMaxLTTB",
    "EveryNthPoint",
    "FuncAggregator",
    "MedDiffGapHandler",
    "MinMaxAggregator",
    "MinMaxOverlapAggregator",
    "NoGapHandler",
]
