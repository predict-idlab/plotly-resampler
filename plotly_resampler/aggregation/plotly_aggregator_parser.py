from __future__ import annotations

import bisect
from typing import Tuple, Union

import numpy as np
import pandas as pd
import pytz

from .aggregation_interface import DataAggregator, DataPointSelector
from .gap_handler_interface import AbstractGapHandler
from .gap_handlers import NoGapHandler


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
        if isinstance(hf_data, pd.RangeIndex):
            return None
        if isinstance(hf_data, (pd.Series, pd.Index)):
            return hf_data.values
        return hf_data

    @staticmethod
    def to_same_tz(
        ts: Union[pd.Timestamp, None], reference_tz: Union[pytz.BaseTzInfo, None]
    ) -> Union[pd.Timestamp, None]:
        """Adjust `ts` its timezone to the `reference_tz`."""
        if ts is None:
            return None
        elif reference_tz is not None:
            if ts.tz is not None:
                # compare if these two have the same timezone / offset
                try:
                    assert ts.tz.__str__() == reference_tz.__str__()
                except AssertionError:
                    assert ts.utcoffset() == reference_tz.utcoffset(ts.tz_convert(None))
                return ts
            else:  # localize -> time remains the same
                return ts.tz_localize(reference_tz)
        elif reference_tz is None and ts.tz is not None:
            return ts.tz_localize(None)
        return ts

    @staticmethod
    def get_start_end_indices(hf_trace_data, axis_type, start, end) -> Tuple[int, int]:
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

        # NOTE: we must verify this before check if the x is a range-index
        if axis_type == "log":
            start, end = 10**start, 10**end

        # We can compute the start & end indices directly when it is a RangeIndex
        if isinstance(hf_trace_data["x"], pd.RangeIndex):
            x_start = hf_trace_data["x"].start
            x_step = hf_trace_data["x"].step
            start_idx = int(max((start - x_start) // x_step, 0))
            end_idx = int((end - x_start) // x_step)
            return start_idx, end_idx
        # TODO: this can be performed as-well for a fixed frequency range-index w/ freq

        if axis_type == "date":
            start, end = pd.to_datetime(start), pd.to_datetime(end)
            # convert start & end to the same timezone
            if isinstance(hf_trace_data["x"], pd.DatetimeIndex):
                tz = hf_trace_data["x"].tz
                try:
                    assert start.tz.__str__() == end.tz.__str__()
                except (TypeError, AssertionError):
                    # This fix is needed for DST (when the timezone is not fixed)
                    assert start.tz_localize(None) == start.tz_convert(tz).tz_localize(
                        None
                    )
                    assert end.tz_localize(None) == end.tz_convert(tz).tz_localize(None)

                start = PlotlyAggregatorParser.to_same_tz(start, tz)
                end = PlotlyAggregatorParser.to_same_tz(end, tz)

        # Search the index-positions
        start_idx = bisect.bisect_left(hf_trace_data["x"], start)
        end_idx = bisect.bisect_right(hf_trace_data["x"], end)
        return start_idx, end_idx

    @staticmethod
    def _handle_gaps(
        hf_trace_data: dict,
        hf_x: np.ndarray,
        agg_x: np.ndarray,
        agg_y: np.ndarray,
        indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Handle the gaps in the aggregated data.

        Returns:
            - agg_x: the aggregated x-values
            - agg_y: the aggregated y-values
            - indices: the indices of the hf_data data that were aggregated

        """
        gap_handler: AbstractGapHandler = hf_trace_data["gap_handler"]
        downsampler = hf_trace_data["downsampler"]

        # TODO check for trace mode (markers, lines, etc.) and only perform the
        # gap insertion methodology when the mode is lines.
        # if trace.get("connectgaps") != True and
        if (
            isinstance(gap_handler, NoGapHandler)
            # rangeIndex | datetimeIndex with freq -> equally spaced x; so no gaps
            or isinstance(hf_trace_data["x"], pd.RangeIndex)
            or (
                isinstance(hf_trace_data["x"], pd.DatetimeIndex)
                and hf_trace_data["x"].freq is not None
            )
        ):
            return agg_x, agg_y, indices

        # Interleave the gaps
        # View the data as an int64 when we have a DatetimeIndex
        # We only want to detect gaps, so we only want to compare values.
        agg_x_parsed = PlotlyAggregatorParser.parse_hf_data(agg_x)
        xdt = agg_x_parsed.dtype
        if np.issubdtype(xdt, np.timedelta64) or np.issubdtype(xdt, np.datetime64):
            agg_x_parsed = agg_x_parsed.view("int64")

        agg_y, indices = gap_handler.insert_fill_value_between_gaps(
            agg_x_parsed, agg_y, indices
        )
        if isinstance(downsampler, DataPointSelector):
            agg_x = hf_x[indices]
        elif isinstance(downsampler, DataAggregator):
            # The indices are in this case a repeat
            agg_x = agg_x[indices]

        return agg_x, agg_y, indices

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

            These indices are useful to select the corresponding hf_data from
            non `x` and `y` data (e.g. `text`, `marker_size`, `marker_color`).

        """
        hf_x = hf_trace_data["x"][start_idx:end_idx]
        hf_y = hf_trace_data["y"][start_idx:end_idx]

        # No downsampling needed ; we show the raw data as is, but with gap-detection
        if (end_idx - start_idx) <= hf_trace_data["max_n_samples"]:
            indices = np.arange(len(hf_y))  # no downsampling - all values are selected
            if len(indices):
                return PlotlyAggregatorParser._handle_gaps(
                    hf_trace_data, hf_x=hf_x, agg_x=hf_x, agg_y=hf_y, indices=indices
                )
            else:
                return hf_x, hf_y, indices

        downsampler = hf_trace_data["downsampler"]

        hf_x_parsed = PlotlyAggregatorParser.parse_hf_data(hf_x)
        hf_y_parsed = PlotlyAggregatorParser.parse_hf_data(hf_y)

        if isinstance(downsampler, DataPointSelector):
            s_v = hf_y_parsed
            if isinstance(s_v, pd.Categorical):  # pd.Categorical (has no .values)
                s_v = s_v.codes
            indices = downsampler.arg_downsample(
                hf_x_parsed,
                s_v,
                n_out=hf_trace_data["max_n_samples"],
                **hf_trace_data.get("downsampler_kwargs", {}),
            )
            if isinstance(hf_trace_data["x"], pd.RangeIndex):
                # we avoid slicing the default pd.RangeIndex (as this is not an
                # in-memory array) - this proves to be faster than slicing the index.
                agg_x = (
                    start_idx
                    + hf_trace_data["x"].start
                    + indices.astype(hf_trace_data["x"].dtype) * hf_trace_data["x"].step
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
            if isinstance(hf_trace_data["x"], pd.RangeIndex):
                # we avoid slicing the default pd.RangeIndex (as this is not an
                # in-memory array) - this proves to be faster than slicing the index.
                agg_x = (
                    start_idx
                    + hf_trace_data["x"].start
                    + agg_x * hf_trace_data["x"].step
                )
            # The indices are just the range of the aggregated data
            indices = np.arange(len(agg_x))
        else:
            raise ValueError(
                "Invalid downsampler instance, must be either a "
                + f"DataAggregator or a DataPointSelector, got {type(downsampler)}"
            )

        return PlotlyAggregatorParser._handle_gaps(
            hf_trace_data, hf_x=hf_x, agg_x=agg_x, agg_y=agg_y, indices=indices
        )
