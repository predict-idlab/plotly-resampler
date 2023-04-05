import bisect
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from .aggregation_interface import DataAggregator, DataPointSelector


class PlotlyAggregatorParser:
    @staticmethod
    def parse_hf_data(
        hf_data: np.ndarray | pd.Categorical | pd.Series | pd.Index,
    ) -> np.ndarray | pd.Categorical:
        """Parse the high-frequency data to a numpy array."""
        # Categorical data (pandas)
        #   - pd.Series with categorical dtype -> calling .values will returns a
        #       pd.Categorical
        #   - pd.CategoricalIndex -> calling .values returns a pd.Categorical
        #   - pd.Categorical: has no .values attribute -> will not be parsed
        if isinstance(hf_data, (pd.Series, pd.Index)):
            # TODO: range index
            return hf_data.values
        return hf_data

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
        """Get the start & end indices of the high-frequency data."""
        # Base case: no hf data, or both start & end are None
        if not len(hf_trace_data["x"]):
            return 0, 0
        elif start is None and end is None:
            return 0, len(hf_trace_data["x"])

        # NOTE: as we use bisect right for the end index, we do not need to add a
        #      small epsilon to the end value
        start = hf_trace_data["x"][0] if start is None else start
        end = hf_trace_data["x"][-1] if end is None else end

        # We can compute the start & end indices directly when it is a RangeIndex
        if isinstance(hf_trace_data["x"], pd.RangeIndex):
            x_start = hf_trace_data["x"].start
            x_step = hf_trace_data["x"].step
            return max((start - x_start) // x_step, 0), (end - x_start) // x_step
        # NOTE: this can be performed as-well for a fixed frequency range-index w/ freq

        if hf_trace_data["axis_type"] == "date":
            start, end = pd.to_datetime(start), pd.to_datetime(end)
            # convert start & end to the same timezone
            if isinstance(hf_trace_data["x"], pd.DatetimeIndex):
                tz = hf_trace_data["x"].tz
                assert start.tz == end.tz
                start = PlotlyAggregatorParser.to_same_tz(start, tz)
                end = PlotlyAggregatorParser.to_same_tz(end, tz)

        # Search the index-positions
        start_idx = bisect.bisect_left(hf_trace_data["x"], start)
        end_idx = bisect.bisect_right(hf_trace_data["x"], end)
        return start_idx, end_idx

    @staticmethod
    def aggregate(
        hf_trace_data: dict,
        start_idx: int,
        end_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate the data in `hf_trace_data` between `start_idx` and `end_idx`.

        Returns:
            - x: the aggregated x-values
            - y: the aggregated y-values
            - indices: the indices of the hf_data data that were aggregated

        """
        hf_x = hf_trace_data["x"][start_idx:end_idx]
        hf_y = hf_trace_data["y"][start_idx:end_idx]

        # No downsampling needed ; we show the raw data as is, no gap detection
        if (end_idx - start_idx) <= hf_trace_data["max_n_samples"]:
            return hf_x, hf_y, np.arange(len(hf_y))

        downsampler = hf_trace_data["downsampler"]

        hf_x_parsed = PlotlyAggregatorParser.parse_hf_data(hf_x)
        hf_y_parsed = PlotlyAggregatorParser.parse_hf_data(hf_y)

        if isinstance(downsampler, DataPointSelector):
            s_v = hf_y_parsed
            if str(s_v.dtype) == "category":  # pd.Categorical (has no .values)
                s_v = s_v.codes
            indices = downsampler.arg_downsample(
                hf_x_parsed,
                s_v,
                n_out=hf_trace_data["max_n_samples"],
                **hf_trace_data.get("downsampler_kwargs", {}),
            )
            # we avoid slicing the default pd.RangeIndex
            if isinstance(hf_trace_data["x"], pd.RangeIndex):
                agg_x = (
                    start_idx
                    + hf_trace_data["x"].start
                    + indices * hf_trace_data["x"].step
                )
            else:
                agg_x = hf_x[indices]
            agg_y = hf_y[indices]
        elif isinstance(downsampler, DataAggregator):
            agg_x, agg_y = downsampler.aggregate(
                hf_x_parsed,
                hf_y_parsed,
                n_out=hf_trace_data["max_n_samples"],
                **hf_trace_data.get("downsampler_kwargs", {}),
            )
            # The indices are just the range of the aggregated data
            indices = np.arange(len(agg_x))
        else:
            raise ValueError("Invalid downsampler instance")

        # TODO check for trace mode (markers, lines, etc.) and only perform the
        # gap insertion methodology when the mode is lines.
        # if trace.get("connectgaps") != True and
        if (
            not downsampler.interleave_gaps
            # rangeIndex | datetimeIndex with freq -> equally spaced x; so no gaps
            or isinstance(hf_trace_data["x"], pd.RangeIndex)
            or (
                isinstance(hf_trace_data["x"], pd.DatetimeIndex)
                and hf_trace_data["x"].freq is not None
            )
        ):
            return agg_x, agg_y, indices

        # Interleave the gaps`
        # View the data as an int64 when we have a DatetimeIndex
        # We only want to detect gaps, so we only want to compare values.
        agg_x_parsed = PlotlyAggregatorParser.parse_hf_data(agg_x)
        xdt = agg_x_parsed.dtype
        if np.issubdtype(xdt, np.timedelta64) or np.issubdtype(xdt, np.datetime64):
            agg_x_parsed = agg_x_parsed.view("int64")

        agg_y, indices = downsampler.insert_none_at_gaps(agg_x_parsed, agg_y, indices)
        if isinstance(downsampler, DataPointSelector):
            agg_x = hf_x[indices]
        elif isinstance(downsampler, DataAggregator):
            # The indices are in this case a repeat
            agg_x = agg_x[indices]

        return agg_x, agg_y, indices
