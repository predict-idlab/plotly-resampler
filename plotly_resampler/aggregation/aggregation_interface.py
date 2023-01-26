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

    @staticmethod
    def _calc_med_diff(x_agg: np.ndarray) -> Tuple[float, np.ndarray]:
        # ----- divide and conquer heuristic to calculate the median diff ------
        # remark: thanks to the prepend -> s_idx_diff.shape === len(s)
        x_diff = np.diff(x_agg, prepend=x_agg[0])

        # To do so - use a quantile-based (median) approach where we reshape the data
        # into `n_blocks` blocks and calculate the min
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
        # TODO: implement this functionality elsewhere
        # ------- INSERT None between gaps / irregularly sampled data -------
        med_diff, s_idx_diff = AbstractAggregator._calc_med_diff(x_agg)
        if med_diff is None:
            return

        # TODO: tweak the nan mask condition
        gap_mask = s_idx_diff > 4 * med_diff
        if not any(gap_mask):
            return
        return gap_mask

    @staticmethod
    def insert_gap_none(
        x_agg: np.ndarray,
        y_agg: np.ndarray,
        idxs: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        gap_mask = AbstractAggregator._get_gap_mask(x_agg)
        if gap_mask is None:
            # no gaps are found, nothing to do
            return y_agg, idxs

        # An array filled with 1s and 2s, where 2 indicates a large gap mask
        # (i.e., that index will be repeated twice)
        repeats = np.ones(x_agg.shape, dtype="int") + gap_mask

        # use the (arange)repeats to expand the index, agg_x, and agg_y array
        idx_exp_nan = np.repeat(idxs, repeats)
        y_agg_exp_nan = y_agg[np.repeat(np.arange(x_agg.shape[0]), repeats)]

        # only float alike array can contain None values
        if issubclass(y_agg_exp_nan.dtype.type, np.integer):
            y_agg_exp_nan = y_agg_exp_nan.astype("float")

        # Set the None values
        y_agg_exp_nan[np.where(gap_mask)[0] + np.arange(gap_mask.sum())] = None

        return y_agg_exp_nan, idx_exp_nan

    def _supports_y_dtype(self, y: np.ndarray):
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

    def _supports_x_dtype(self, x: np.ndarray):
        pass


class DataAggregator(AbstractAggregator, ABC):
    """Implementation of the AbstractAggregator interface which aggregates the data.

    This implies that no datapoints are selected, but the data is aggregated.
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
            Additional keyword arguments

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The aggregated x and y data, respectively
        """
        # TODO -> a lot of code duplication
        assert n_out is not None

        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

        if len(y) <= n_out:
            # Fewer samples than n_out -> return the original data
            return x, y

        self._supports_y_dtype(y)

        if x is not None:
            assert x.ndim == 1
            assert x.shape == y.shape
            self._supports_x_dtype(x)

        # More samples that n_out -> perform data aggregation
        return self._aggregate(x, y, n_out=n_out, **kwargs)


class DataPointSelector(AbstractAggregator, ABC):
    """Implementation of the AbstractAggregator interface which returns the
    indices of the selected datapoints.

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
        # Again, we can re-use the functionality implemented into tsdownsample
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
            The index positions of the downsample representation

        """
        assert n_out is not None

        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

        if len(y) <= n_out:
            # Fewer samples than n_out -> no data aggregation need to be performed
            return np.arange(len(y))

        self._supports_y_dtype(y)

        if x is not None:
            assert x.ndim == 1
            assert x.shape == y.shape
            self._supports_x_dtype(x)

        # More samples that n_out -> perform data aggregation
        return self._arg_downsample(x, y, n_out=n_out, **kwargs)
