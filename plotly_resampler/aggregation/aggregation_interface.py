"""AbstractSeriesAggregator interface-class, subclassed by concrete aggregators."""

__author__ = "Jonas Van Der Donckt"

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


# NOTE: for now the aggregator is a concrete implementation of
# Create a superclass which is implemented by this class
class AbstractSeriesArgDownsampler(ABC):
    """ """

    def __init__(
        self,
        interleave_gaps: bool = True,
        nan_position: str = "end",
        dtype_regex_list: Optional[List[str]] = None,
        # TODO -> split this in a x and y dtype regex list
        # NOTE: this functionality will be implemented into the `tsdownsample`
    ):
        """Constructor of AbstractSeriesAggregator.

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
        self.interleave_gaps = interleave_gaps
        self.dtype_regex_list = dtype_regex_list
        self.nan_position = nan_position.lower()
        super().__init__()

    @abstractmethod
    def _arg_downsample(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        n_out: int = None,
    ) -> np.ndarray:
        # Again, we can re-use the functionality implemented into tsdownsample
        raise NotImplementedError

    def _supports_ydtype(self, y: np.ndarray):
        # base case
        if self.dtype_regex_list is None:
            return

        for dtype_regex_str in self.dtype_regex_list:
            m = re.compile(dtype_regex_str).match(str(y.dtype))
            if m is not None:  # a match is found
                return
        raise ValueError(
            f"{y.dtype} doesn't match with any regex in {self.dtype_regex_list}"
        )

    def _supports_xdtype(self, x: np.ndarray):
        pass

    @staticmethod
    def _calc_med_diff(x_agg: np.ndarray) -> Tuple[float, np.ndarray]:
        # ----- divide and conquer heuristic to calculate the median diff ------
        # remark: thanks to the prepend -> s_idx_diff.shape === len(s)
        x_diff = np.diff(x_agg, prepend=x_agg[0])

        # To do so - use a quantile-based (median) approach where we reshape the data
        # into `n_blocks` blocks and calculate the min
        n_blcks = 128
        if x_agg.shape[0] > 5 * n_blcks:
            blck_size = x_diff.shape[0] // n_blcks

            # convert the index series index diff into a reshaped view (i.e., sid_v)
            sid_v: np.ndarray = x_diff[: blck_size * n_blcks].reshape(n_blcks, -1)

            # calculate the min and max and calculate the median on that
            med_diff = np.median(np.mean(sid_v, axis=1))
        else:
            med_diff = np.median(x_diff)

        return med_diff, x_diff

    @staticmethod
    def insert_gap_none(
        x_agg: np.ndarray, y_agg: np.ndarray, idxs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: implement this functionality elsewhere
        # ------- INSERT None between gaps / irregularly sampled data -------
        med_diff, s_idx_diff = AbstractSeriesArgDownsampler._calc_med_diff(x_agg)
        if med_diff is None:
            return y_agg, idxs

        # TODO: tweak the nan mask condition
        gap_mask = s_idx_diff > 4 * med_diff
        if not any(gap_mask):
            return y_agg, idxs

        # A an array filled with 1s and 2s, where 2 indicates a large gap mask
        # (i.e., that index will be repeated twice)
        repeats = np.ones(idxs.shape, dtype="int") + gap_mask

        # use the (arange)repeats to expand the index, agg_x, and agg_y array
        idx_exp_nan = np.repeat(idxs, repeats)
        y_agg_exp_nan =  y_agg[np.repeat(np.arange(idxs.shape[0]), repeats)]

        # only float alike array can contain None values
        if issubclass(y_agg_exp_nan.dtype.type, np.integer):
            y_agg_exp_nan = y_agg_exp_nan.astype("float")

        # Set the None values
        y_agg_exp_nan[np.where(gap_mask)[0] + np.arange(gap_mask.sum())] = None

        return y_agg_exp_nan, idx_exp_nan

    def arg_downsample(
        self, x: Optional[np.ndarray] = None, y: np.ndarray = None, n_out: int = None
    ) -> np.ndarray:
        """Compute the index positions for the downsampled representation

        Parameters
        ----------
        x: Optional[np.ndarray]
            The time dimension of the to-be-aggregated series
        y: Optional[np.ndarray]
            The value dimension of the to-be-aggregated series.
        n_out: int
            The number of samples which the downsampled series should contain.

        Returns
        -------
        np.ndarray
            The index positions of the downsample representation

        """
        assert n_out is not None

        assert isinstance(y, np.ndarray)
        assert y.ndim == 1
        self._supports_ydtype(y)

        if x is not None:
            assert x.ndim == 1
            assert x.shape == y.shape
            self._supports_xdtype(x)

        if len(y) > n_out:
            # More samples that n_out -> perform data aggregation
            return self._arg_downsample(x, y, n_out=n_out)
        else:
            # Less samples than n_out -> no data aggregation need to be performed
            return np.arange(len(y))
