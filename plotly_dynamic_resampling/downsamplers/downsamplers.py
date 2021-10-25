# -*- coding: utf-8 -*-
"""Compatible implementation for various downsample methods
"""
__author__ = "Jonas Van Der Donckt"

import math

import lttbc
import numpy as np
import pandas as pd

from ..downsamplers.downsampling_interface import AbstractSeriesDownsampler


class LTTB(AbstractSeriesDownsampler):
    """LTTB downsampler method

    Notes
    -----
    * Mainly designed to work on numerical data as this algorithm uses distance based
      measures within the data.

    """

    def __init__(
        self,
        interleave_gaps: bool = True,
    ):
        super().__init__(
            interleave_gaps,
            dtype_regex_list=[rf"{dtype}\d*" for dtype in ["float", "int", "uint"]] +
                             ['category'],
        )

    # TODO -> check whether these are able to deal with non-int based datatypes
    # Wont work as you work with distances -> no categorical data downsample techniques
    # Maybe create a proxy for categorical data
    # but how can we ensure equidistant things? one-hot? LTTB use when one-hot?
    # create a new C-file for which we support downsampling of categorical data
    # how often would this be useful?
    def _downsample(self, s: pd.Series, n_out: int) -> pd.Series:
        s_v = s.cat.codes.values if str(s.dtype) == 'category' else s.values
        idx, data = lttbc.downsample(np.arange(len(s), dtype="uint32"), s_v, n_out)
        if str(s.dtype) == 'category':
            data = np.vectorize(s.dtype.categories.values.item)(data.astype(s_v.dtype))
        return pd.Series(
            index=s.iloc[idx.astype("uint32")].index.astype(s.index.dtype),
            data=data,
            name=s.name,
            copy=False
        )


class EveryNthPoint(AbstractSeriesDownsampler):
    """Naive (but fast) downsampler method which returns every n'th point."""
    def __init__(self, interleave_gaps: bool = True ):
        # this downsampler supports all pd.Series dtypes
        return super().__init__(interleave_gaps, dtype_regex_list=None)

    def _downsample(self, s: pd.Series, n_out: int) -> pd.Series:
        return s[:: max(1, math.ceil(len(s) / n_out))]


class AggregationDownsampler(AbstractSeriesDownsampler):
    """Downsampler method which uses the passed aggregation func."""
    def __init__(
        self, aggregation_func, interleave_gaps: bool = True, supported_dtypes=None
    ):
        self.aggregation_func = aggregation_func
        super().__init__(interleave_gaps, supported_dtypes)

    def _downsample(self, s: pd.Series, n_out: int) -> pd.Series:
        if isinstance(s.index, pd.DatetimeIndex):
            t_start, t_end = s.index[:: len(s) - 1]
            rate = (t_end - t_start) / n_out
            return s.resample(rate).apply(self.aggregation_func).dropna()

        # no time index -> use the every nth heuristic
        group_size = max(1, np.ceil(len(s) / n_out))
        s_out = (
            s.groupby(
                # create an array of [0, 0, 0, ...., n_out, n_out]
                # where each value is repeated based $len(s)/n_out$ times
                by=np.repeat(np.arange(n_out), group_size)[: len(s)]
            )
            .agg(self.aggregation_func)
            .dropna()
        )
        # Create an index-estimation for real-time data
        # Add one to the index so it's pointed at the end of the window
        # Note: this can be adjusted to .5 to center the data
        # Multiply it with the group size to get the real index-position
        # TODO: add option to select start / middle / end as index
        idx_locs = (np.arange(len(s_out)) + 1) * group_size
        idx_locs[-1] = len(s) - 1
        return pd.Series(
            index=s.iloc[idx_locs.astype(s.index.dtype)].index.astype(s.index.dtype),
            data=s_out.values,
            name=s.name,
            copy=False
        )
