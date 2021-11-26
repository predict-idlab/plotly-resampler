# -*- coding: utf-8 -*-
"""Compatible implementation for various downsample methods."""

__author__ = "Jonas Van Der Donckt"

import math

import lttbc
import numpy as np
import pandas as pd

from ..downsamplers.downsampling_interface import AbstractSeriesDownsampler


class LTTB(AbstractSeriesDownsampler):
    """Largest Triangle Three Bucket (LTTB) downsampler method.

    Notes
    -----
    * This class is mainly designed to operate on numerical data as LTTB calculates
      distances on the values.<br>
      When dealing with categories, the data is encoded into its numeric codes,
      these codes are the indices of the category array.
    * To downsample category data with LTTB, your `pd.Series` must be of dtype
      'category'.<br>
      **pro tip**: If there is an order in your categories, order them that way, LTTB
       uses the ordered category codes values (se bullet above) to calculate distances
       and make downsample decisions.
      >>> s = pd.Series(["a", "b", "c", "a"])
      >>> cat_type = pd.CategoricalDtype(categories=["b", "c", "a"], ordered=True)
      >>> s_cat = s.astype(cat_type)

    """

    def __init__(
        self,
        interleave_gaps: bool = True,
    ):
        super().__init__(
            interleave_gaps,
            dtype_regex_list=[rf"{dtype}\d*" for dtype in ["float", "int", "uint"]] +
                             ['category', 'bool'],
        )

    def _downsample(self, s: pd.Series, n_out: int) -> pd.Series:
        # if we have categorical data, LTTB will convert the categorical values into
        # their numeric codes, i.e., the index position of the category array
        s_v = s.cat.codes.values if str(s.dtype) == 'category' else s.values

        idx, data = lttbc.downsample(np.arange(len(s), dtype="uint32"), s_v, n_out)

        if str(s.dtype) == "category":
            # reconvert the downsampled numeric codes to the category array
            data = np.vectorize(s.dtype.categories.values.item)(data.astype(s_v.dtype))
        else:
            # default case, use the series it's dtype as return type
            data = data.astype(s.dtype)

        return pd.Series(
            index=s.iloc[idx.astype("uint32")].index.astype(s.index.dtype),
            data=data,
            name=s.name,
            copy=False
        )


class EveryNthPoint(AbstractSeriesDownsampler):
    """Naive (but fast) downsampler method which returns every n'th point."""
    def __init__(self, interleave_gaps: bool = True):
        # this downsampler supports all pd.Series dtypes
        super().__init__(interleave_gaps, dtype_regex_list=None)

    def _downsample(self, s: pd.Series, n_out: int) -> pd.Series:
        out = s[:: max(1, math.ceil(len(s) / n_out))]
        return out.astype('uint8') if str(s.dtype) == 'bool' else out


class AggregationDownsampler(AbstractSeriesDownsampler):
    """Downsampler method which uses the passed aggregation func.

    Notes
    -----
    * The user has total control which aggregation_func is passed to this method,
      hence it is the users' responisbility to handle categorical and bool-based
      datatypes.

    """
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
