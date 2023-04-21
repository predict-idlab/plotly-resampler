"""AbstractGapHandler interface-class, subclassed by concrete gap handlers."""

from __future__ import annotations

__author__ = "Jeroen Van Der Donckt"

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class AbstractGapHandler(ABC):
    @abstractmethod
    def _get_gap_mask(self, x_agg: np.ndarray) -> Optional[np.ndarray]:
        """Get a boolean mask indicating the indices where there are gaps.

        If you require custom gap handling, you can implement this method to return a
        boolean mask indicating the indices where there are gaps.

        Parameters
        ----------
        x_agg: np.ndarray
            The x array. This is used to determine the gaps.

        Returns
        -------
        Optional[np.ndarray]
            A boolean mask indicating the indices where there are gaps. If there are no
            gaps, None is returned.

        """
        pass

    def insert_none_at_gaps(
        self,
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
        gap_mask = self._get_gap_mask(x_agg)
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


class NoGapHandler(AbstractGapHandler):
    def _get_gap_mask(self, x_agg: np.ndarray) -> Optional[np.ndarray]:
        return


class MedDiffGapHandler(AbstractGapHandler):
    def _calc_med_diff(self, x_agg: np.ndarray) -> Tuple[float, np.ndarray]:
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

    def _get_gap_mask(self, x_agg: np.ndarray) -> Optional[np.ndarray]:
        """Get a boolean mask indicating the indices where there are gaps.

        If you require custom gap handling, you can implement this method to return a
        boolean mask indicating the indices where there are gaps.

        Parameters
        ----------
        x_agg: np.ndarray
            The x array. This is used to determine the gaps.

        Returns
        -------
        Optional[np.ndarray]
            A boolean mask indicating the indices where there are gaps. If there are no
            gaps, None is returned.

        """
        med_diff, x_diff = self._calc_med_diff(x_agg)

        # TODO: this 4 was revealed to me in a dream, but it seems to work well
        gap_mask = x_diff > 4 * med_diff
        if not any(gap_mask):
            return
        return gap_mask
