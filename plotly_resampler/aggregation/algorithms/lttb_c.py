"""Interface which selects the appropriate C downsampling method."""


import numpy as np

from .lttbc import (
    downsample_double_double,
    downsample_int_double,
    downsample_int_float,
    downsample_int_int,
)


class LTTB_core_c:
    @staticmethod
    def downsample(x: np.ndarray, y: np.ndarray, n_out: int) -> np.ndarray:
        """Downsample the data using the LTTB algorithm (C implementation).

        The main logic of this method is to select the appropriate C downsampling
        method to do as little datatype casting as possible.

        Parameters
        ----------
        x : np.ndarray
            The time series array.
        y : np.ndarray
            The value series array.
        n_out : int
            The numer of output points.

        Returns
        -------
        np.ndarray
            The indexes of the selected datapoints.
        """
        if x.dtype == np.int64 and y.dtype == np.float64:
            return downsample_int_double(x, y, n_out)
        elif x.dtype == np.int64 and y.dtype == np.int64:
            return downsample_int_int(x, y, n_out)
        elif x.dtype == np.int64 and y.dtype == np.float32:
            return downsample_int_float(x, y, n_out)

        return downsample_double_double(x, y, n_out)
