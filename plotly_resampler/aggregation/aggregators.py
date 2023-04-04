# -*- coding: utf-8 -*-
"""Compatible implementation for various aggregation/downsample methods.

.. |br| raw:: html

   <br>

"""

__author__ = "Jonas Van Der Donckt"

import math
from typing import Optional, Tuple

import numpy as np

from ..aggregation.aggregation_interface import DataAggregator, DataPointSelector

try:
    # The efficient c version of the LTTB algorithm
    from .algorithms.lttb_c import LTTB_core_c as LTTB_core
except (ImportError, ModuleNotFoundError):
    import warnings

    warnings.warn("Could not import lttbc; will use a (slower) python alternative.")
    from .algorithms.lttb_py import LTTB_core_py as LTTB_core


class LTTB(DataPointSelector):
    """Largest Triangle Three Buckets (LTTB) aggregation method.

    Thesis: https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf

    .. Tip::
        `LTTB` doesn't scale super-well when moving to really large datasets, so when
        dealing with more than 1 million samples, you might consider using
        :class:`MinMaxLTTB <MinMaxLTTB>`.

    Note
    ----
    * This class is mainly designed to operate on numerical data as LTTB calculates
      distances on the values. |br|
      When dealing with categories, the data is encoded into its numeric codes,
      these codes are the indices of the category array.
    * To aggregate category data with LTTB, your ``pd.Series`` must be of dtype
      'category'. |br|
      **Tip**: if there is an order in your categories, order them that way, LTTB uses
      the ordered category codes values (see bullet above) to calculate distances and
      make aggregation decisions.
      .. code::
        >>> import pandas as pd
        >>> s = pd.Series(["a", "b", "c", "a"])
        >>> cat_type = pd.CategoricalDtype(categories=["b", "c", "a"], ordered=True)
        >>> s_cat = s.astype(cat_type)

    """

    def __init__(self, interleave_gaps: bool = True):
        """
        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.

        """
        super().__init__(
            interleave_gaps,
            y_dtype_regex_list=[rf"{dtype}\d*" for dtype in ("float", "int", "uint")]
            + ["category", "bool"],
        )
        # TODO: when integrating with tsdownsample add x & y dtype regex list

    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
        **_,
    ) -> np.ndarray:
        # Use the Core interface to perform the downsampling
        return LTTB_core.downsample(x, y, n_out)


class MinMaxOverlapAggregator(DataPointSelector):
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

    def __init__(self, interleave_gaps: bool = True):
        """
        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.

        """
        # this downsampler supports all dtypes
        super().__init__(interleave_gaps)

    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
        **kwargs,
    ) -> np.ndarray:
        # The block size 2x the bin size we also perform the ceil-operation
        # to ensure that the block_size * n_out / 2 < len(x)
        block_size = math.ceil(y.shape[0] / (n_out + 1) * 2)
        argmax_offset = block_size // 2

        # Calculate the offset range which will be added to the argmin and argmax pos
        offset = np.arange(
            0, stop=y.shape[0] - block_size - argmax_offset, step=block_size
        )

        # Calculate the argmin & argmax on the reshaped view of `y` &
        # add the corresponding offset
        argmin = (
            y[: block_size * offset.shape[0]].reshape(-1, block_size).argmin(axis=1)
            + offset
        )
        argmax = (
            y[argmax_offset : block_size * offset.shape[0] + argmax_offset]
            .reshape(-1, block_size)
            .argmax(axis=1)
            + offset
            + argmax_offset
        )

        # Sort the argmin & argmax (where we append the first and last index item)
        return np.unique(np.concatenate((argmin, argmax, [0, y.shape[0] - 1])))


class MinMaxAggregator(DataPointSelector):
    """Aggregation method which performs binned min-max aggregation over fully
    overlapping windows.

    .. note::
        This method is rather efficient when scaling to large data sizes and can be used
        as a data-reduction step before feeding it to the :class:`LTTB <LTTB>`
        algorithm, as :class:`EfficientLTTB <EfficientLTTB>` does with the
        :class:`MinMaxOverlapAggregator <MinMaxOverlapAggregator>`.

    """

    def __init__(self, interleave_gaps: bool = True):
        """
        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.

        """
        # this downsampler supports all dtypes
        super().__init__(interleave_gaps)

    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
        **kwargs,
    ) -> np.ndarray:
        # The block size 2x the bin size we also perform the ceil-operation
        # to ensure that the block_size * n_out / 2 < len(x)
        block_size = math.ceil(y.shape[0] / n_out * 2)

        # Calculate the offset range which will be added to the argmin and argmax pos
        offset = np.arange(0, stop=y.shape[0] - block_size, step=block_size)

        # Calculate the argmin & argmax on the reshaped view of `s` &
        # add the corresponding offset
        argmin = (
            y[: block_size * offset.shape[0]].reshape(-1, block_size).argmin(axis=1)
            + offset
        )
        argmax = (
            y[: block_size * offset.shape[0]].reshape(-1, block_size).argmax(axis=1)
            + offset
        )

        # Note: the implementation below flips the array to search from
        # right-to left (as min or max will always use the first same minimum item,
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
        return np.unique(np.concatenate((argmin, argmax, [0, y.shape[0] - 1])))


class MinMaxLTTB(DataPointSelector):
    """Efficient version off LTTB by first reducing really large datasets with
    the :class:`MinMaxOverlapAggregator <MinMaxOverlapAggregator>` and then further
    aggregating the reduced result with :class:`LTTB <LTTB>`.

    Inventor: Jonas & Jeroen Van Der Donckt - 2022
    """

    def __init__(self, interleave_gaps: bool = True):
        """
        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.

        """
        self.lttb = LTTB(interleave_gaps=False)
        self.minmax = MinMaxOverlapAggregator(interleave_gaps=False)
        super().__init__(
            interleave_gaps,
            y_dtype_regex_list=[rf"{dtype}\d*" for dtype in ("float", "int", "uint")]
            + ["category", "bool"],
        )
        # TODO: when integrating with tsdownsample add x & y dtype regex list

    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
        **kwargs,
    ) -> np.ndarray:
        size_threshold = 10_000_000
        ratio_threshold = 100

        # TODO -> test this with a move of the .so file
        if LTTB_core.__name__ == "LTTB_core_py":
            size_threshold = 1_000_000

        if y.shape[0] > size_threshold and y.shape[0] / n_out > ratio_threshold:
            # TODO: add argument for 30 when the paper is published
            idxs = self.minmax._arg_downsample(x, y, n_out * 30)
            y = y[idxs]
            if x is not None:
                x = x[idxs]
        return self.lttb._arg_downsample(x, y, n_out)


class EveryNthPoint(DataPointSelector):
    """Naive (but fast) aggregator method which returns every N'th point."""

    def __init__(self, interleave_gaps: bool = True):
        """
        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.

        """
        # this downsampler supports all dtypes
        super().__init__(interleave_gaps)

    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
        **kwargs,
    ) -> np.ndarray:
        # TODO: check the "-1" below
        return np.arange(step=max(1, math.ceil(len(y) / n_out)), stop=len(y) - 1)


class FuncAggregator(DataAggregator):
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
        x_dtype_regex_list=None,
        y_dtype_regex_list=None,
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

        """
        self.aggregation_func = aggregation_func
        super().__init__(
            interleave_gaps,
            x_dtype_regex_list=x_dtype_regex_list,
            y_dtype_regex_list=y_dtype_regex_list,
        )

    def _aggregate(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Create an index-estimation for real-time data
        # Add one to the index so it's pointed at the end of the window
        # Note: this can be adjusted to .5 to center the data
        # Multiply it with the group size to get the real index-position
        # TODO: add option to select start / middle / end as index
        if x is None:
            # no time index -> use the every nth heuristic
            group_size = max(1, np.ceil(len(y) / n_out))
            idxs = (np.arange(n_out) * group_size).astype(int)
        else:
            x_ = x
            if np.issubdtype(x.dtype, np.datetime64) or np.issubdtype(
                x.dtype, np.timedelta64
            ):
                x_ = x_.view("int64")
            # Thanks to `linspace`, the data is evenly distributed over the index-range
            # The searchsorted function returns the index positions
            idxs = np.searchsorted(x_, np.linspace(x_[0], x_[-1], n_out + 1))

        y_agg = np.array(
            [
                self.aggregation_func(y[t0:t1], **kwargs)
                for t0, t1 in zip(idxs[:-1], idxs[1:])
            ]
        )

        if x is not None:
            x_agg = x[idxs[:-1]]
        else:
            # groupsize * n_out can be larger than the length of the data
            idxs[-1] -= 1
            x_agg = idxs

        return x_agg, y_agg
