# -*- coding: utf-8 -*-
"""Compatible implementation for various aggregation/downsample methods.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt"


import math
from typing import Tuple

import numpy as np
from tsdownsample import (
    EveryNthDownsampler,
    LTTBDownsampler,
    MinMaxDownsampler,
    MinMaxLTTBDownsampler,
    NaNMinMaxDownsampler,
    NaNMinMaxLTTBDownsampler,
)

from ..aggregation.aggregation_interface import DataAggregator, DataPointSelector


def _to_tsdownsample_args(
    x: np.ndarray | None, y: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Converts x & y to the arguments expected by tsdownsample."""
    if x is None:
        return (y,)
    return (x, y)


class LTTB(DataPointSelector):
    """Largest Triangle Three Buckets (LTTB) aggregation method.

    This is arguably the most widely used aggregation method. It is based on the
    effective area of a triangle (inspired from the line simplification domain).
    The algorithm has $O(n)$ complexity, however, for large datasets, it can be much
    slower than other algorithms (e.g. MinMax) due to the higher cost of calculating
    the areas of triangles.

    Thesis: [https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf](https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf) <br/>
    Details on visual representativeness & stability: [https://arxiv.org/abs/2304.00900](https://arxiv.org/abs/2304.00900)

    !!! tip

        `LTTB` doesn't scale super-well when moving to really large datasets, so when
        dealing with more than 1 million samples, you might consider using
        [`MinMaxLTTB`][aggregation.aggregators.MinMaxLTTB].


    !!! note

        * This class is mainly designed to operate on numerical data as LTTB calculates
          distances on the values. <br/>
          When dealing with categories, the data is encoded into its numeric codes,
          these codes are the indices of the category array.
        * To aggregate category data with LTTB, your ``pd.Series`` must be of dtype
          'category'. <br/>

          **tip**:

          if there is an order in your categories, order them that way, LTTB uses
          the ordered category codes values (see bullet above) to calculate distances and
          make aggregation decisions. <br/>
          **code**:
            ```python
                >>> import pandas as pd
                >>> s = pd.Series(["a", "b", "c", "a"])
                >>> cat_type = pd.CategoricalDtype(categories=["b", "c", "a"], ordered=True)
                >>> s_cat = s.astype(cat_type)
            ```
        * `LTTB` has no downsample kwargs, as it cannot be paralellized. Instead, you can
          use the [`MinMaxLTTB`][aggregation.aggregators.MinMaxLTTB] downsampler, which performs
          minmax preselection (in parallel if configured so), followed by LTTB.

    """

    def __init__(self):
        super().__init__(
            y_dtype_regex_list=[rf"{dtype}\d*" for dtype in ("float", "int", "uint")]
            + ["category", "bool"],
        )
        self.downsampler = LTTBDownsampler()

    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
    ) -> np.ndarray:
        return self.downsampler.downsample(*_to_tsdownsample_args(x, y), n_out=n_out)


class MinMaxOverlapAggregator(DataPointSelector):
    """Aggregation method which performs binned min-max aggregation over 50% overlapping
    windows.

    ![minmax operator image](https://github.com/predict-idlab/plotly-resampler/blob/main/mkdocs/static/minmax_operator.png)

    In the above image, **bin_size**: represents the size of *(len(series) / n_out)*.
    As the windows have 50% overlap and are consecutive, the min & max values are
    calculated on a windows with size (2x bin-size).

    This is *very* similar to the MinMaxAggregator, emperical results showed no
    observable difference between both approaches.

    !!! note

        This method is implemented in Python (leveraging numpy for vecotrization), but
        is **significantly slower than the MinMaxAggregator** (which is implemented in
        the tsdownsample toolkit in Rust). <br/>
        As such, this class does not support any downsample kwargs.

    !!! note

        This downsampler supports all dtypes.

    """

    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
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

    This is arguably the most computational efficient downsampling method, as it only
    performs (non-expensive) comparisons on the data in a single pass.

    Details on visual representativeness & stability: [https://arxiv.org/abs/2304.00900](https://arxiv.org/abs/2304.00900)

    !!! note

        This method is rather efficient when scaling to large data sizes and can be used
        as a data-reduction step before feeding it to the [`LTTB`][aggregation.aggregators.LTTB]
        algorithm, as [`MinMaxLTTB`][aggregation.aggregators.MinMaxLTTB] does with the
        [`MinMaxOverlapAggregator`][aggregation.aggregators.MinMaxOverlapAggregator].

    """

    def __init__(self, nan_policy="omit", **downsample_kwargs):
        """
        Parameters
        ----------
        **downsample_kwargs
            Keyword arguments passed to the :class:`MinMaxDownsampler`.
            - The `parallel` argument is set to False by default.
        nan_policy: str, optional
            The policy to handle NaNs. Can be 'omit' or 'keep'. By default, 'omit'.

        """
        # this downsampler supports all dtypes
        super().__init__(**downsample_kwargs)
        if nan_policy not in ("omit", "keep"):
            raise ValueError("nan_policy must be either 'omit' or 'keep'")
        if nan_policy == "omit":
            self.downsampler = MinMaxDownsampler()
        else:
            self.downsampler = NaNMinMaxDownsampler()

    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
    ) -> np.ndarray:
        return self.downsampler.downsample(
            *_to_tsdownsample_args(x, y), n_out=n_out, **self.downsample_kwargs
        )


class MinMaxLTTB(DataPointSelector):
    """Efficient version off LTTB by first reducing really large datasets with
    the [`MinMaxAggregator`][aggregation.aggregators.MinMaxAggregator] and then further aggregating the
    reduced result with [`LTTB`][aggregation.aggregators.LTTB].

    Starting from 10M data points, this method performs the MinMax-prefetching of data
    points to enhance computational efficiency.

    Inventors: Jonas & Jeroen Van Der Donckt - 2022

    Paper: [https://arxiv.org/pdf/2305.00332.pdf](https://arxiv.org/pdf/2305.00332.pdf)
    """

    def __init__(
        self, minmax_ratio: int = 4, nan_policy: str = "omit", **downsample_kwargs
    ):
        """
        Parameters
        ----------
        minmax_ratio: int, optional
            The ratio between the number of data points in the MinMax-prefetching and
            the number of data points that will be outputted by LTTB. By default, 4.
        nan_policy: str, optional
            The policy to handle NaNs. Can be 'omit' or 'keep'. By default, 'omit'.
        **downsample_kwargs
            Keyword arguments passed to the `MinMaxLTTBDownsampler`.
            - The `parallel` argument is set to False by default.
            - The `minmax_ratio` argument is set to 4 by default, which was empirically
              proven to be a good default.

        """
        if nan_policy not in ("omit", "keep"):
            raise ValueError("nan_policy must be either 'omit' or 'keep'")
        if nan_policy == "omit":
            self.minmaxlttb = MinMaxLTTBDownsampler()
        else:
            self.minmaxlttb = NaNMinMaxLTTBDownsampler()

        self.minmax_ratio = minmax_ratio

        super().__init__(
            y_dtype_regex_list=[rf"{dtype}\d*" for dtype in ("float", "int", "uint")]
            + ["category", "bool"],
            **downsample_kwargs,
        )

    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
    ) -> np.ndarray:
        return self.minmaxlttb.downsample(
            *_to_tsdownsample_args(x, y),
            n_out=n_out,
            minmax_ratio=self.minmax_ratio,
            **self.downsample_kwargs,
        )


class EveryNthPoint(DataPointSelector):
    """Naive (but fast) aggregator method which returns every N'th point.

    !!! note
        This downsampler supports all dtypes.
    """

    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
    ) -> np.ndarray:
        return EveryNthDownsampler().downsample(y, n_out=n_out)


class FuncAggregator(DataAggregator):
    """Aggregator instance which uses the passed aggregation func.

    !!! warning

        The user has total control which `aggregation_func` is passed to this method,
        hence the user should be careful to not make copies of the data, nor write to
        the data. Furthermore, the user should beware of performance issues when
        using more complex aggregation functions.

    !!! warning "Attention"

        The user has total control which `aggregation_func` is passed to this method,
        hence it is the users' responsibility to handle categorical and bool-based
        data types.

    """

    def __init__(
        self,
        aggregation_func,
        x_dtype_regex_list=None,
        y_dtype_regex_list=None,
        **downsample_kwargs,
    ):
        """
        Parameters
        ----------
        aggregation_func: Callable
            The aggregation function which will be applied on each pin.

        """
        self.aggregation_func = aggregation_func
        super().__init__(x_dtype_regex_list, y_dtype_regex_list, **downsample_kwargs)

    def _aggregate(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate the data using the object's aggregation function.

        Parameters
        ----------
        x: np.ndarray | None
            The x-values of the data. Can be None if no x-values are available.
        y: np.ndarray
            The y-values of the data.
        n_out: int
            The number of output data points.
        **kwargs
            Additional keyword arguments, which are passed to the aggregation function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The aggregated x & y values.
            If `x` is None, then the indices of the first element of each bin is
            returned as x-values.

        """
        # Create an index-estimation for real-time data
        # Add one to the index so it's pointed at the end of the window
        # Note: this can be adjusted to .5 to center the data
        # Multiply it with the group size to get the real index-position
        # TODO: add option to select start / middle / end as index
        if x is None:
            # equidistant index
            idxs = np.linspace(0, len(y), n_out + 1).astype(int)
        else:
            xdt = x.dtype
            if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
                x = x.view("int64")
            # Thanks to `linspace`, the data is evenly distributed over the index-range
            # The searchsorted function returns the index positions
            idxs = np.searchsorted(x, np.linspace(x[0], x[-1], n_out + 1))

        y_agg = np.array(
            [
                self.aggregation_func(y[t0:t1], **self.downsample_kwargs)
                for t0, t1 in zip(idxs[:-1], idxs[1:])
            ]
        )

        if x is not None:
            x_agg = x[idxs[:-1]]
        else:
            # x is None -> return the indices of the first element of each bin
            x_agg = idxs[:-1]

        return x_agg, y_agg
