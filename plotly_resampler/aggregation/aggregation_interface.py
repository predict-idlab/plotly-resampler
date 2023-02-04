"""AbstractSeriesAggregator interface-class, subclassed by concrete aggregators."""

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
        # ----- divide and conquer heuristic to calculate the median diff ------
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
        # ------- INSERT None between gaps / irregularly sampled data -------
        med_diff, s_idx_diff = AbstractAggregator._calc_med_diff(x_agg)

        # TODO: this 4 was revealed to me in a dream, but it seems to work well
        gap_mask = s_idx_diff > 4 * med_diff
        if not any(gap_mask):
            return
        return gap_mask

    @staticmethod
    def insert_gap_none(
        x_agg: np.ndarray,
        y_agg: np.ndarray,
        idxs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Insert None values in the y_agg array when there are gaps.

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

    A data aggregator is an aggregator that aggregates the data, and thus doesn't select
    data points.
    Concrete implementations of this class must implement the `_aggregate` method, and
    have full responsibility on how they deal with other high-frequency properties, such
    as `hovertext`, `marker_size`, 'marker_color`, etc ...
    """

    def _aggregate(
        self,
        x: Optional[np.ndarray] = None,
        y: np.ndarray = None,
        n_out: int = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def aggregate(
        self,
        x: Optional[np.ndarray] = None,
        y: np.ndarray = None,
        n_out: int = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate the data.

        Parameters
        ----------
        x : np.ndarray, optional
            The x-axis data, by default None
        y : np.ndarray
            The y-axis data
        n_out : int, optional
            The number of output datapoints, by default None
        kwargs : dict
            Additional keyword arguments that are passed to the `_aggregate` method.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The aggregated x and y data, respectively

        """
        assert n_out is not None

        # assert isinstance(y, np.ndarray)
        assert y.ndim == 1
        DataAggregator._supports_dtype(y, self.y_dtype_regex_list)

        if x is not None:
            assert x.ndim == 1
            assert x.shape == y.shape
            DataAggregator._supports_dtype(x, self.x_dtype_regex_list)

        return self._aggregate(x, y, n_out=n_out, **kwargs)


class DataPointSelector(AbstractAggregator, ABC):
    """Implementation of the AbstractAggregator interface for data point selection.

    A data point selector is an aggregator that selects data points, and thus doesn't
    aggregate the data.

    This class utilizes the `arg_downsample` method to compute the index positions.
    """

    @abstractmethod
    def _arg_downsample(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        n_out: int = None,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def arg_downsample(
        self,
        x: Optional[np.ndarray] = None,
        y: np.ndarray = None,
        n_out: int = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute the index positions for the downsampled representation.

        Parameters
        ----------
        x: Optional[np.ndarray]
            The time dimension of the to-be-aggregated series
        y: Optional[np.ndarray]
            The value dimension of the to-be-aggregated series.
        n_out: int
            The number of samples which the downsampled series should contain.
        kwargs : dict
            Additional keyword arguments

        Returns
        -------
        np.ndarray
            The index positions of the selected data points.

        """
        assert n_out is not None

        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

        if len(y) <= n_out:
            # Fewer samples than n_out -> no data aggregation need to be performed
            return np.arange(len(y))

        DataAggregator._supports_dtype(y, self.y_dtype_regex_list)

        if x is not None:
            assert x.ndim == 1
            assert x.shape == y.shape
            DataAggregator._supports_dtype(x, self.x_dtype_regex_list)

        # More samples that n_out -> perform data aggregation
        return self._arg_downsample(x, y, n_out=n_out, **kwargs)
