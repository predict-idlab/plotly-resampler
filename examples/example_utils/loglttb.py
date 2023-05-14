"""An (non-optimized) python implementation of the LTTB algorithm that utilizes 
log-scale buckets.
"""

import numpy as np
from plotly_resampler.aggregation.aggregation_interface import DataPointSelector
from typing import Union


class LogLTTB(DataPointSelector):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        prev_x : float
            The previous selected point is x value.
        prev_y : float
            The previous selected point its y value.
        avg_next_x : float
            The x mean of the next bucket
        avg_next_y : float
            The y mean of the next bucket
        x_bucket : np.ndarray
            All x values in the bucket
        y_bucket : np.ndarray
            All y values in the bucket

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _arg_downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        # We need a valid x array to determing the x-range
        assert x is not None, "x cannot be None for this downsampler"

        # the log function to use
        lf = np.log1p

        offset = np.unique(
            np.searchsorted(
                x, np.exp(np.linspace(lf(x[0]), lf(x[-1]), n_out + 1)).astype(np.int64)
            )
        )

        # Construct the output array
        sampled_x = np.empty(len(offset) + 1, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(len(offset) - 2):
            a = (
                self._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],
                    avg_next_x=np.mean(x[offset[i + 1] : offset[i + 2]]),
                    avg_next_y=y[offset[i + 1] : offset[i + 2]].mean(),
                    x_bucket=x[offset[i] : offset[i + 1]],
                    y_bucket=y[offset[i] : offset[i + 1]],
                )
                + offset[i]
            )
            sampled_x[i + 1] = a

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        sampled_x[-2] = (
            self._argmax_area(
                prev_x=x[a],
                prev_y=y[a],
                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],
                x_bucket=x[offset[-2] : offset[-1]],
                y_bucket=y[offset[-2] : offset[-1]],
            )
            + offset[-2]
        )
        return sampled_x
