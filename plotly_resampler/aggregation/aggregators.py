# -*- coding: utf-8 -*-
"""Compatible implementation for various aggregation/downsample methods.

.. |br| raw:: html

   <br>

"""

__author__ = "Jonas Van Der Donckt"

import math

import numpy as np
import pandas as pd

from ..aggregation.aggregation_interface import AbstractSeriesAggregator

try:
    # The efficient c version of the LTTB algorithm
    from .algorithms.lttb_c import LTTB_core_c as LTTB_core
except (ImportError, ModuleNotFoundError):
    import warnings

    warnings.warn("Could not import lttbc; will use a (slower) python alternative.")
    from .algorithms.lttb_py import LTTB_core_py as LTTB_core


class LTTB(AbstractSeriesAggregator):
    """Largest Triangle Three Buckets (LTTB) aggregation method.

    .. Tip::
        `LTTB` doesn't scale super-well when moving to really large datasets, so when
        dealing with more than 1 million samples, you might consider using
        :class:`EffientLTTB <EfficientLTTB>`.

    Note
    ----
    * This class is mainly designed to operate on numerical data as LTTB calculates
      distances on the values. |br|
      When dealing with categories, the data is encoded into its numeric codes,
      these codes are the indices of the category array.
    * To aggregate category data with LTTB, your ``pd.Series`` must be of dtype
      'category'. |br|
      **Tip**: if there is an order in your categories, order them that way, LTTB uses
      the ordered category codes values (se bullet above) to calculate distances and
      make aggregation decisions.
      .. code::
        >>> s = pd.Series(["a", "b", "c", "a"])
        >>> cat_type = pd.CategoricalDtype(categories=["b", "c", "a"], ordered=True)
        >>> s_cat = s.astype(cat_type)

    """

    def __init__(self, interleave_gaps: bool = True, nan_position="end"):
        """
        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.
        nan_position: str, optional
            Indicates where nans must be placed when gaps are detected. \n
            If ``'end'``, the first point after a gap will be replaced with a
            nan-value \n
            If ``'begin'``, the last point before a gap will be replaced with a
            nan-value \n
            If ``'both'``, both the encompassing gap datapoints are replaced with
            nan-values \n
            .. note::
                This parameter only has an effect when ``interleave_gaps`` is set
                to *True*.

        """
        super().__init__(
            interleave_gaps,
            nan_position,
            dtype_regex_list=[rf"{dtype}\d*" for dtype in ["float", "int", "uint"]]
            + ["category", "bool"],
        )

    def _aggregate(self, s: pd.Series, n_out: int) -> pd.Series:
        s_v = s.cat.codes.values if str(s.dtype) == "category" else s.values

        s_i = s.index.values
        s_i = s_i.astype(np.int64) if s_i.dtype.type == np.datetime64 else s_i

        # Use the Core interface to perform the downsampling
        index = LTTB_core.downsample(s_i, s_v, n_out)

        return pd.Series(
            index=s.index[index],
            data=s.values[index],
            name=str(s.name),
            copy=False,
        )


class MinMaxOverlapAggregator(AbstractSeriesAggregator):
    """Aggregation method which performs binned min-max aggregation over 50% overlapping
    windows.

    .. image:: _static/minmax_operator.png

    In the above image, **bin_size**: represents the size of *(len(series) / n_out)*.
    As the windows have 50% overlap and are consecutive, the min & max values are
    calculated on a windows with size (2x bin-size).

    .. note::
        This method is rather efficient when scaling to large data sizes and can be used
        as a data-reduction step before feeding it to the :class:`LTTB <LTTB>`
        algorithm, as :class:`EfficientLTTB <EfficientLTTB>` does.

    """

    def __init__(self, interleave_gaps: bool = True, nan_position="end"):
        """
        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.
        nan_position: str, optional
            Indicates where nans must be placed when gaps are detected. \n
            If ``'end'``, the first point after a gap will be replaced with a
            nan-value \n
            If ``'begin'``, the last point before a gap will be replaced with a
            nan-value \n
            If ``'both'``, both the encompassing gap datapoints are replaced with
            nan-values \n
            .. note::
                This parameter only has an effect when ``interleave_gaps`` is set
                to *True*.

        """
        # this downsampler supports all pd.Series dtypes
        super().__init__(interleave_gaps, nan_position, dtype_regex_list=None)

    def _aggregate(self, s: pd.Series, n_out: int) -> pd.Series:
        # The block size 2x the bin size we also perform the ceil-operation
        # to ensure that the block_size =
        block_size = math.ceil(s.shape[0] / (n_out + 1) * 2)
        argmax_offset = block_size // 2

        # Calculate the offset range which will be added to the argmin and argmax pos
        offset = np.arange(
            0, stop=s.shape[0] - block_size - argmax_offset, step=block_size
        )

        # Calculate the argmin & argmax on the reshaped view of `s` &
        # add the corresponding offset
        argmin = (
            s.values[: block_size * offset.shape[0]]
            .reshape(-1, block_size)
            .argmin(axis=1)
            + offset
        )
        argmax = (
            s.values[argmax_offset : block_size * offset.shape[0] + argmax_offset]
            .reshape(-1, block_size)
            .argmax(axis=1)
            + offset
            + argmax_offset
        )
        # Sort the argmin & argmax (where we append the first and last index item)
        # and then slice the original series on these indexes.
        return s.iloc[np.unique(np.concatenate((argmin, argmax, [0, s.shape[0] - 1])))]


class MinMaxAggregator(AbstractSeriesAggregator):
    """Aggregation method which performs binned min-max aggregation over fully
    overlapping windows.

    .. note::
        This method is rather efficient when scaling to large data sizes and can be used
        as a data-reduction step before feeding it to the :class:`LTTB <LTTB>`
        algorithm, as :class:`EfficientLTTB <EfficientLTTB>` does with the
        :class:`MinMaxOverlapAggregator <MinMaxOverlapAggregator>`.

    """

    def __init__(self, interleave_gaps: bool = True, nan_position="end"):
        """
        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.
        nan_position: str, optional
            Indicates where nans must be placed when gaps are detected. \n
            If ``'end'``, the first point after a gap will be replaced with a
            nan-value \n
            If ``'begin'``, the last point before a gap will be replaced with a
            nan-value \n
            If ``'both'``, both the encompassing gap datapoints are replaced with
            nan-values \n
            .. note::
                This parameter only has an effect when ``interleave_gaps`` is set
                to *True*.
        dtype_regex_list: List[str], optional
            List containing the regex matching the supported datatypes, by default None.
        """
        # this downsampler supports all pd.Series dtypes
        super().__init__(interleave_gaps, nan_position, dtype_regex_list=None)

    def _aggregate(self, s: pd.Series, n_out: int) -> pd.Series:
        # The block size 2x the bin size we also perform the ceil-operation
        # to ensure that the block_size =
        block_size = math.ceil(s.shape[0] / n_out * 2)

        # Calculate the offset range which will be added to the argmin and argmax pos
        offset = np.arange(0, stop=s.shape[0] - block_size, step=block_size)

        # Calculate the argmin & argmax on the reshaped view of `s` &
        # add the corresponding offset
        argmin = (
            s.values[: block_size * offset.shape[0]]
            .reshape(-1, block_size)
            .argmin(axis=1)
            + offset
        )
        argmax = (
            s.values[: block_size * offset.shape[0]]
            .reshape(-1, block_size)
            .argmax(axis=1)
            + offset
        )

        # Note: the implementation below flips the array to search from
        # right-to left (as min or max will always usee the first same minimum item,
        # i.e. the most left item)
        # This however creates a large computational overhead -> we do not use this
        # implementation and suggest using the minmaxaggregator.
        # argmax = (
        #     (block_size - 1)
        #     - np.fliplr(
        #         s[: block_size * offset.shape[0]].values.reshape(-1, block_size)
        #     ).argmax(axis=1)
        # ) + offset

        # Sort the argmin & argmax (where we append the first and last index item)
        # and then slice the original series on these indexes.
        return s.iloc[np.unique(np.concatenate((argmin, argmax, [0, s.shape[0] - 1])))]


class EfficientLTTB(AbstractSeriesAggregator):
    """Efficient version off LTTB by first reducing really large datasets with
    the :class:`MinMaxOverlapAggregator <MinMaxOverlapAggregator>` and then further
    aggregating the reduced result with :class:`LTTB <LTTB>`.
    """

    def __init__(self, interleave_gaps: bool = True, nan_position="end"):
        """
        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.
        nan_position: str, optional
            Indicates where nans must be placed when gaps are detected. \n
            If ``'end'``, the first point after a gap will be replaced with a
            nan-value \n
            If ``'begin'``, the last point before a gap will be replaced with a
            nan-value \n
            If ``'both'``, both the encompassing gap datapoints are replaced with
            nan-values \n
            .. note::
                This parameter only has an effect when ``interleave_gaps`` is set
                to *True*.

        """
        self.lttb = LTTB(interleave_gaps=False)
        self.minmax = MinMaxOverlapAggregator(interleave_gaps=False)
        super().__init__(
            interleave_gaps,
            nan_position,
            dtype_regex_list=[rf"{dtype}\d*" for dtype in ["float", "int", "uint"]]
            + ["category", "bool"],
        )

    def _aggregate(self, s: pd.Series, n_out: int) -> pd.Series:
        size_threshold = 10_000_000
        ratio_threshold = 100

        # TODO -> test this with a move of the .so file
        if LTTB_core.__name__ == "LTTB_core_py":
            size_threshold = 1_000_000

        if s.shape[0] > size_threshold and s.shape[0] / n_out > ratio_threshold:
            s = self.minmax._aggregate(s, n_out * 30)
        return self.lttb._aggregate(s, n_out)


class EveryNthPoint(AbstractSeriesAggregator):
    """Naive (but fast) aggregator method which returns every N'th point."""

    def __init__(self, interleave_gaps: bool = True, nan_position="end"):
        """
        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.
        nan_position: str, optional
            Indicates where nans must be placed when gaps are detected. \n
            If ``'end'``, the first point after a gap will be replaced with a
            nan-value \n
            If ``'begin'``, the last point before a gap will be replaced with a
            nan-value \n
            If ``'both'``, both the encompassing gap datapoints are replaced with
            nan-values \n
            .. note::
                This parameter only has an effect when ``interleave_gaps`` is set
                to *True*.

        """
        # this downsampler supports all pd.Series dtypes
        super().__init__(interleave_gaps, nan_position, dtype_regex_list=None)

    def _aggregate(self, s: pd.Series, n_out: int) -> pd.Series:
        return s[:: max(1, math.ceil(len(s) / n_out))]


class FuncAggregator(AbstractSeriesAggregator):
    """Aggregator instance which uses the passed aggregation func.

    .. attention::
        The user has total control which `aggregation_func` is passed to this method,
        hence it is the users' responsibility to handle categorical and bool-based
        data types.

    """

    def __init__(
        self,
        aggregation_func,
        interleave_gaps: bool = True,
        nan_position="end",
        dtype_regex_list=None,
    ):
        """
        Parameters
        ----------
        aggregation_func: Callable
            The aggregation function which will be applied on each pin.
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.
        nan_position: str, optional
            Indicates where nans must be placed when gaps are detected. \n
            If ``'end'``, the first point after a gap will be replaced with a
            nan-value \n
            If ``'begin'``, the last point before a gap will be replaced with a
            nan-value \n
            If ``'both'``, both the encompassing gap datapoints are replaced with
            nan-values \n
            .. note::
                This parameter only has an effect when ``interleave_gaps`` is set
                to *True*.
        dtype_regex_list: List[str], optional
            List containing the regex matching the supported datatypes, by default None.

        """
        self.aggregation_func = aggregation_func
        super().__init__(interleave_gaps, nan_position, dtype_regex_list)

    def _aggregate(self, s: pd.Series, n_out: int) -> pd.Series:
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
            name=str(s.name),
            copy=False,
        )
