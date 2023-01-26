import bisect
from typing import Tuple, Union

import numpy as np
import pandas as pd

from .aggregation_interface import DataAggregator, DataPointSelector


class PlotlyAggregatorParser:
    @staticmethod
    def to_same_tz(
        ts: Union[pd.Timestamp, None], reference_tz
    ) -> Union[pd.Timestamp, None]:
        """Adjust `ts` its timezone to the `reference_tz`."""
        if ts is None:
            return None
        elif reference_tz is not None:
            if ts.tz is not None:
                assert ts.tz.zone == reference_tz.zone
                return ts
            else:  # localize -> time remains the same
                return ts.tz_localize(reference_tz)
        elif reference_tz is None and ts.tz is not None:
            return ts.tz_localize(None)
        return ts

    @staticmethod
    def get_start_end_indices(hf_trace_data, start, end) -> Tuple[int, int]:
        start = hf_trace_data["x"][0] if start is None else start
        end = hf_trace_data["x"][-1] if end is None else end

        # We can compute the start & end indices directly when it is a RangeIndex
        if isinstance(hf_trace_data["x"], pd.RangeIndex):
            x_start = hf_trace_data["x"].start
            x_step = hf_trace_data["x"].step
            return int((start - x_start) // x_step ), int((end - x_start) // x_step)
        # TODO: this can be performed as-well for a fixed frequency range-index w/ freq

        if hf_trace_data["axis_type"] == "date":
            start, end = pd.to_datetime(start), pd.to_datetime(end)
            # convert start & end to the same timezone
            if isinstance(hf_trace_data["x"], pd.DatetimeIndex):
                tz = hf_trace_data["x"].tz
                start = PlotlyAggregatorParser.to_same_tz(start, tz)
                end = PlotlyAggregatorParser.to_same_tz(end, tz)

        # Search the index-positions
        start_idx = bisect.bisect_left(hf_trace_data["x"], start)
        # TODO: check whether we need to use bisect_left or bisect_right
        end_idx = bisect.bisect_right(hf_trace_data["x"], end)
        return start_idx, end_idx

    @staticmethod
    def aggregate(
        hf_trace_data,
        start_idx: int,
        end_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hf_x = hf_trace_data["x"][start_idx:end_idx]
        hf_y = hf_trace_data["y"][start_idx:end_idx]

        # No downsampling needed ; we show the raw data as is, no gap detection
        if (end_idx - start_idx) <= hf_trace_data["max_n_samples"]:
            return hf_x, hf_y, np.arange(len(hf_y))

        # indicates whether the x is a default pd.RangeIndex
        downsampler = hf_trace_data["downsampler"]

        if isinstance(downsampler, DataPointSelector):
            indices = downsampler.arg_downsample(
                hf_x,
                hf_y,
                hf_trace_data["max_n_samples"],
                **hf_trace_data.get("downsampler_kwargs", {}),
            )
            # we avoid slicing the default pd.RangeIndex
            if isinstance(hf_trace_data["x"], pd.RangeIndex):
                agg_x = (
                    start_idx
                    + hf_trace_data["x"].start
                    + indices
                    + hf_trace_data["x"].step
                )
            else:
                agg_x = hf_x[indices]
            agg_y = hf_y[indices]
        elif isinstance(downsampler, DataAggregator):
            agg_x, agg_y = downsampler.aggregate(
                hf_x,
                hf_y,
                hf_trace_data["max_n_samples"],
                **hf_trace_data.get("downsampler_kwargs", {}),
            )
            # TODO
            indices = np.arange(len(agg_x))
        else:
            raise ValueError("Invalid downsampler instance")

        # TODO check for trace mode (markers, lines, etc.) and only perform the
        # gap insertion methodology when the mode is lines.
        # if trace.get("connectgaps") != True and
        if (
            # rangeIndex | datetimeIndex with freq -> equally spaced x; so no gaps
            not (
                isinstance(hf_trace_data["x"], pd.RangeIndex)
                or (
                    isinstance(hf_trace_data["x"], pd.DatetimeIndex)
                    and hf_trace_data["x"].freq is not None
                )
            )
            and downsampler.interleave_gaps
        ):
            # View the data as an int64 when we have a DatetimeIndex
            # We only want to detect gaps, so we only want to compare values.
            if hf_trace_data["axis_type"] == "date" and isinstance(
                agg_x, pd.DatetimeIndex
            ):
                agg_x = agg_x.view("int64")

            agg_y, indices = downsampler.insert_gap_none(agg_x, agg_y, indices)
            if isinstance(downsampler, DataPointSelector):
                agg_x = hf_x[indices]
            elif isinstance(downsampler, DataAggregator):
                # The indices are in this case a repeat
                agg_x = agg_x[indices]

        return agg_x, agg_y, indices
