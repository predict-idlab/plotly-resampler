"""AbstractGapHandler interface-class, subclassed by concrete gap handlers."""

from __future__ import annotations

__author__ = "Jeroen Van Der Donckt"

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class AbstractGapHandler(ABC):
    def __init__(self, fill_value: Optional[float] = None):
        """Constructor of AbstractGapHandler.

        Parameters
        ----------
        fill_value: float, optional
            The value to fill the gaps with, by default None.
            Note that setting this value to 0 for filled area plots is particularly
            useful.

        """
        self.fill_value = fill_value

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

    def insert_fill_value_between_gaps(
        self,
        x_agg: np.ndarray,
        y_agg: np.ndarray,
        idxs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Insert the fill_value in the y_agg array where there are gaps.

        Gaps are determined by the x_agg array. The `_get_gap_mask` method is used to
        determine a boolean mask indicating the indices where there are gaps.

        Parameters
        ----------
        x_agg: np.ndarray
            The x array. This is used to determine the gaps.
        y_agg: np.ndarray
            The y array. A copy of this array will be expanded with fill_values where
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
        y_agg_exp_nan[np.where(gap_mask)[0] + np.arange(gap_mask.sum())] = (
            self.fill_value
        )

        return y_agg_exp_nan, idx_exp_nan
