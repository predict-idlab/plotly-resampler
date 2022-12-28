"""Wittholds an efficient numpy python implementation of the LTTB algorithm."""

import numpy as np


class LTTB_core_py:
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

    @staticmethod
    def downsample(x: np.ndarray, y: np.ndarray, n_out) -> np.ndarray:
        """Downsample the data using the LTTB algorithm (python implementation).

        Parameters
        ----------
        x : np.ndayarray
            The time series array.
        y : np.ndarray
            The value series array.
        n_out : int
            The number of output points.

        Returns
        -------
        np.array
            The indexes of the selected datapoints.
        """
        # Bucket size. Leave room for start and end data points
        block_size = (y.shape[0] - 2) / (n_out - 2)
        # Note this 'astype' cast must take place after array creation (and not with the
        # aranage() its dtype argument) or it will cast the `block_size` step to an int
        # before the arange array creation
        offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)

        # Construct the output array
        sampled_x = np.empty(n_out, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert y to int if it is boolean
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(n_out - 3):
            a = (
                LTTB_core_py._argmax_area(
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
            LTTB_core_py._argmax_area(
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
