"""AbstractSeriesAggregator interface-class, subclassed by concrete aggregators."""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt"

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class AbstractAggregator(ABC):
    def __init__(
        self,
        interleave_gaps: bool = True,
        x_dtype_regex_list: Optional[List[str]] = None,
        y_dtype_regex_list: Optional[List[str]] = None,
    ):
        """Constructor of AbstractSeriesAggregator.

        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. An x-range based approach is used to determine the gaps /
            irregularly sampled data. By default, True.
        x_dtype_regex_list: List[str], optional
            List containing the regex matching the supported datatypes for the x array,
            by default None.
        y_dtype_regex_list: List[str], optional
            List containing the regex matching the supported datatypes for the y array,
            by default None.

        """
        self.interleave_gaps = interleave_gaps
        self.x_dtype_regex_list = x_dtype_regex_list
        self.y_dtype_regex_list = y_dtype_regex_list

    @staticmethod
    def _calc_med_diff(x_agg: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate the median diff of the x_agg array.

        As median is more robust to outliers than the mean, the median is used to define
        the gap threshold.

        This method performs a divide and conquer heuristic to calculate the median;
        1. divide the array into `n_blocks` blocks (with `n_blocks` = 128)
        2. calculate the mean of each block
        3. calculate the median of the means
        => This proves to be a good approximation of the median of the full array, while
              being much faster than calculating the median of the full array.
        """
        # remark: thanks to the prepend -> x_diff.shape === len(s)
        x_diff = np.diff(x_agg, prepend=x_agg[0])

        # To do so - use an approach where we reshape the data
        # into `n_blocks` blocks and calculate the mean and then the median on that
        # Why use `median` instead of a global mean?
        #   => when you have large gaps, they will be represented by a large diff
        #      which will skew the mean way more than the median!
        n_blocks = 128
        if x_agg.shape[0] > 5 * n_blocks:
            blck_size = x_diff.shape[0] // n_blocks

            # convert the index series index diff into a reshaped view (i.e., sid_v)
            sid_v: np.ndarray = x_diff[: blck_size * n_blocks].reshape(n_blocks, -1)

            # calculate the min and max and calculate the median on that
            med_diff = np.median(np.mean(sid_v, axis=1))
        else:
            med_diff = np.median(x_diff)

        return med_diff, x_diff

    @staticmethod
    def _get_gap_mask(x_agg: np.ndarray) -> Optional[np.ndarray]:
        """Return a boolean mask indicating the indices where there are gaps.

        A gap is *currently* defined as a difference between two consecutive x values,
        that is larger than 4 times the median difference between two consecutive x
        values.
        Note: this is a naive approach, but it seems to work well.

        Parameters
        ----------
        x_agg: np.ndarray
            The aggregated x array. This is used to determine the gaps.

        Returns
        -------
        Optional[np.ndarray]
            The boolean mask indicating the indices where there are gaps. If no gaps are
            found, None (i.e., nothing) is returned.

        """
        # ------- INSERT None between gaps / irregularly sampled data -------
        med_diff, s_idx_diff = AbstractAggregator._calc_med_diff(x_agg)

        # TODO: this 4 was revealed to me in a dream, but it seems to work well
        gap_mask = s_idx_diff > 4 * med_diff
        if not any(gap_mask):
            return
        return gap_mask

    @staticmethod
    def insert_none_at_gaps(
        x_agg: np.ndarray,
        y_agg: np.ndarray,
        idxs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Insert None values in the y_agg array when there are gaps.

        Gaps are determined by the x_agg array. The `_get_gap_mask` method is used to
        determine a boolean mask indicating the indices where there are gaps.

        Parameters
        ----------
        x_agg: np.ndarray
            The x array. This is used to determine the gaps.
        y_agg: np.ndarray
            The y array. A copy of this array will be expanded with None values where
            there are gaps.
        idxs: np.ndarray
            The index array. This is relevant aggregators that perform data point
            selection (e.g., max, min, etc.) - this array will be expanded with the
            same indices where there are gaps.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The expanded y_agg array and the expanded idxs array respectively.

        """
        gap_mask = AbstractAggregator._get_gap_mask(x_agg)
        if gap_mask is None:
            # no gaps are found, nothing to do
            return y_agg, idxs

        # An array filled with 1s and 2s, where 2 indicates a large gap mask
        # (i.e., that index will be repeated twice)
        repeats = np.ones(x_agg.shape, dtype="int") + gap_mask

        # use the repeats to expand the idxs, and agg_y array
        idx_exp_nan = np.repeat(idxs, repeats)
        y_agg_exp_nan = np.repeat(y_agg, repeats)

        # only float arrays can contain NaN values
        if issubclass(y_agg_exp_nan.dtype.type, np.integer) or issubclass(
            y_agg_exp_nan.dtype.type, np.bool_
        ):
            y_agg_exp_nan = y_agg_exp_nan.astype("float")

        # Set the NaN values
        # We add the gap index offset (via the np.arange) to the indices to account for
        # the repeats (i.e., expanded y_agg array).
        y_agg_exp_nan[np.where(gap_mask)[0] + np.arange(gap_mask.sum())] = None

        return y_agg_exp_nan, idx_exp_nan

    @staticmethod
    def _check_n_out(n_out: int) -> None:
        """Check if the n_out is valid."""
        assert isinstance(n_out, (int, np.integer))
        assert n_out > 0

    @staticmethod
    def _process_args(*args) -> Tuple[np.ndarray | None, np.ndarray]:
        """Process the args into the x and y arrays.

        If only y is passed, x is set to None.
        """
        assert len(args) in [1, 2], "Must pass either 1 or 2 arrays"
        x, y = (None, args[0]) if len(args) == 1 else args
        if hasattr(y, "values"):
            y = y.values
        # TODO -> what to do with the x index?
        return x, y

    @staticmethod
    def _check_arr(arr: np.ndarray, regex_list: Optional[List[str]] = None):
        """Check if the array is valid."""
        assert isinstance(arr, np.ndarray), f"Expected np.ndarray, got {type(arr)}"
        assert arr.ndim == 1
        AbstractAggregator._supports_dtype(arr, regex_list)

    def _check_x_y(self, x: np.ndarray | None, y: np.ndarray) -> None:
        """Check if the x and y arrays are valid."""
        # Check x (if not None)
        if x is not None:
            self._check_arr(x, self.x_dtype_regex_list)
            assert x.shape == y.shape, "x and y must have the same shape"
        # Check y
        self._check_arr(y, self.y_dtype_regex_list)

    @staticmethod
    def _supports_dtype(arr: np.ndarray, dtype_regex_list: Optional[List[str]] = None):
        # base case
        if dtype_regex_list is None:
            return

        for dtype_regex_str in dtype_regex_list:
            m = re.compile(dtype_regex_str).match(str(arr.dtype))
            if m is not None:  # a match is found
                return
        raise ValueError(
            f"{arr.dtype} doesn't match with any regex in {dtype_regex_list}"
        )


class DataAggregator(AbstractAggregator, ABC):
    """Implementation of the AbstractAggregator interface for data aggregation.

    DataAggregator differs from DataPointSelector in that it doesn't select data points,
    but rather aggregates the data (e.g., mean).
    As such, the `_aggregate` method is responsible for aggregating the data, and thus
    returns a tuple of the aggregated x and y values.

    Concrete implementations of this class must implement the `_aggregate` method, and
    have full responsibility on how they deal with other high-frequency properties, such
    as `hovertext`, `marker_size`, 'marker_color`, etc ...
    """

    @abstractmethod
    def _aggregate(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def aggregate(
        self,
        *args,
        n_out: int,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate the data.

        Parameters
        ----------
        x, y: np.ndarray
            The x and y data of the to-be-aggregated series.
            The x array is optional (i.e., if only 1 array is passed, it is assumed to
            be the y array).
            The array(s) must be 1-dimensional, and have the same length (if x is
            passed).
            These cannot be passed as keyword arguments, as they are positional-only.
        n_out: int
            The number of samples which the downsampled series should contain.
            This should be passed as a keyword argument.
        kwargs : dict
            Additional keyword arguments that are passed to the `_aggregate` method.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The aggregated x and y data, respectively.

        """
        # Check n_out
        assert n_out is not None

        # Get x and y
        x, y = DataPointSelector._process_args(*args)

        # Check x and y
        self._check_x_y(x, y)

        return self._aggregate(x=x, y=y, n_out=n_out, **kwargs)


class DataPointSelector(AbstractAggregator, ABC):
    """Implementation of the AbstractAggregator interface for data point selection.

    DataPointSelector differs from DataAggregator in that they don't aggregate the data
    (e.g., mean) but instead select data points (e.g., first, last, min, max, etc ...).
    As such, the `_arg_downsample` method returns the index positions of the selected
    data points.

    This class utilizes the `arg_downsample` method to compute the index positions.
    """

    @abstractmethod
    def _arg_downsample(
        self,
        x: np.ndarray | None,
        y: np.ndarray,
        n_out: int,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def arg_downsample(
        self,
        *args,
        n_out: int,
        **kwargs,
    ) -> np.ndarray:
        """Compute the index positions for the downsampled representation.

        Parameters
        ----------
        x, y: np.ndarray
            The x and y data of the to-be-aggregated series.
            The x array is optional (i.e., if only 1 array is passed, it is assumed to
            be the y array).
            The array(s) must be 1-dimensional, and have the same length (if x is
            passed).
            These cannot be passed as keyword arguments, as they are positional-only.
        n_out: int
            The number of samples which the downsampled series should contain.
            This should be passed as a keyword argument.
        kwargs : dict
            Additional keyword arguments that are passed to the `_arg_downsample` method.

        Returns
        -------
        np.ndarray
            The index positions of the selected data points.

        """
        # Check n_out
        DataPointSelector._check_n_out(n_out)

        # Get x and y
        x, y = DataPointSelector._process_args(*args)

        # Check x and y
        self._check_x_y(x, y)

        if len(y) <= n_out:
            # Fewer samples than n_out -> return all indices
            return np.arange(len(y))

        # More samples that n_out -> perform data aggregation (with x=None)
        return self._arg_downsample(x=x, y=y, n_out=n_out, **kwargs)
