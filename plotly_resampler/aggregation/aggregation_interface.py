"""AbstractAggregator interface-class, subclassed by concrete aggregators."""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt"

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class AbstractAggregator(ABC):
    def __init__(
        self,
        x_dtype_regex_list: Optional[List[str]] = None,
        y_dtype_regex_list: Optional[List[str]] = None,
        **downsample_kwargs,
    ):
        """Constructor of AbstractSeriesAggregator.

        Parameters
        ----------
        x_dtype_regex_list: List[str], optional
            List containing the regex matching the supported datatypes for the x array,
            by default None.
        y_dtype_regex_list: List[str], optional
            List containing the regex matching the supported datatypes for the y array,
            by default None.
        downsample_kwargs: dict
            Additional kwargs passed to the downsample method.

        """
        self.x_dtype_regex_list = x_dtype_regex_list
        self.y_dtype_regex_list = y_dtype_regex_list
        self.downsample_kwargs = downsample_kwargs

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def aggregate(
        self,
        *args,
        n_out: int,
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

        return self._aggregate(x=x, y=y, n_out=n_out)


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
    ) -> np.ndarray:
        # Note: this method can utilize the self.downsample_kwargs property
        raise NotImplementedError

    def arg_downsample(
        self,
        *args,
        n_out: int,
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

        # More samples that n_out -> perform data aggregation
        return self._arg_downsample(x=x, y=y, n_out=n_out)
