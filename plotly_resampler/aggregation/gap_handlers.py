# -*- coding: utf-8 -*-
"""Compatible implementation for various gap handling methods."""

from __future__ import annotations

__author__ = "Jeroen Van Der Donckt"

from typing import Optional, Tuple

import numpy as np

from plotly_resampler.aggregation.gap_handler_interface import AbstractGapHandler


class NoGapHandler(AbstractGapHandler):
    """No gap handling."""

    def _get_gap_mask(self, x_agg: np.ndarray) -> Optional[np.ndarray]:
        return


class MedDiffGapHandler(AbstractGapHandler):
    """Gap handling based on the median diff of the x_agg array."""

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

            # calculate the mean fore each block and then the median of those means
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
        # After some consideration, we altered this to a 4.1
        gap_mask = x_diff > 4.1 * med_diff
        if not any(gap_mask):
            return
        return gap_mask
