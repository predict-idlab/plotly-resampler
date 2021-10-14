# -*- coding: utf-8 -*-
"""Compatible implementation for various downsample methods
"""
__author__ = 'Jonas Van Der Donckt'

from ..downsamplers.resampling_interface import AbstractSeriesDownsampler
import pandas as pd

import lttbc
import numpy as np


# TODO -> maybe use decorators for these methods?

class LTTB(AbstractSeriesDownsampler):
    def _downsample(self, s: pd.Series, n_out: int) -> pd.Series:
        idx, downsampled_data = lttbc.downsample(
            np.arange(len(s), dtype=np.uint32), s.values, n_out
        )
        return pd.Series(
            index=s.iloc[idx.astype(int)].index, data=downsampled_data, copy=False
        )


class EveryNthPoint(AbstractSeriesDownsampler):
    def _downsample(self, s: pd.Series, n_out: int) -> pd.Series:
        return s[:: (max(1, len(s) // n_out))]


class AggregationDownsampler(AbstractSeriesDownsampler):
    def __init__(self, aggregation_func, interleave_gaps: bool = True):
        self.aggregation_func = aggregation_func
        super().__init__(interleave_gaps)

    def _downsample(self, s: pd.Series, n_out: int) -> pd.Series:
        if isinstance(s.index, pd.DatetimeIndex):
            t_start, t_end = s.index[::len(s) - 1]
            return s.resample((t_end - t_start) / n_out).apply(
                func=self.aggregation_func).dropna()
        return s.resample(max(1, int(len(s) // n_out))).apply(
            func=self.aggregation_func).dropna()
