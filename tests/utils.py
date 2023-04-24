from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from plotly_resampler.aggregation import MedDiffGapHandler, MinMaxLTTB
from plotly_resampler.aggregation.aggregation_interface import (
    DataAggregator,
    DataPointSelector,
)
from plotly_resampler.aggregation.gap_handler_interface import AbstractGapHandler
from plotly_resampler.aggregation.plotly_aggregator_parser import PlotlyAggregatorParser


def not_on_linux():
    """Return True if the current platform is not Linux.

    This is to avoid / alter test bahavior for non-Linux (as browser testing gets
    tricky on other platforms).
    """
    return not sys.platform.startswith("linux")


def construct_hf_data_dict(hf_x, hf_y, **kwargs):
    hf_data_dict = {
        "x": hf_x,
        "y": hf_y,
        "axis_type": "date" if isinstance(hf_x, pd.DatetimeIndex) else "linear",
        "downsampler": MinMaxLTTB(),
        "gap_handler": MedDiffGapHandler(),
        "max_n_samples": 1_000,
    }
    hf_data_dict.update(kwargs)
    return hf_data_dict


def wrap_aggregate(
    hf_x: np.ndarray | None = None,
    hf_y: pd.Series | np.ndarray = None,
    downsampler: DataPointSelector | DataAggregator = None,
    gap_handler: AbstractGapHandler = None,
    n_out: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hf_trace_data = construct_hf_data_dict(
        **{
            "hf_x": hf_x,
            "hf_y": hf_y,
            "downsampler": downsampler,
            "gap_handler": gap_handler,
            "max_n_samples": n_out,
        }
    )
    return PlotlyAggregatorParser.aggregate(hf_trace_data, 0, len(hf_y))


def construct_index(series: pd.Series, index_type: str) -> pd.Index:
    """Construct an index of the given type for the given series.

    series: pd.Series
        The series to construct an index for
    index_type: str
        One of "range", "datetime", "timedelta", "float", or "int"
    """
    if index_type == "range":
        return pd.RangeIndex(len(series))
    if index_type == "datetime":
        return pd.date_range("1/1/2020", periods=len(series), freq="1ms")
    if index_type == "timedelta":
        return pd.timedelta_range(start="0s", periods=len(series), freq="1ms")
    if index_type == "float":
        return pd.Float64Index(np.arange(len(series)))
    if index_type == "int":
        return pd.Int64Index(np.arange(len(series)))
    raise ValueError(f"Unknown index type: {index_type}")
