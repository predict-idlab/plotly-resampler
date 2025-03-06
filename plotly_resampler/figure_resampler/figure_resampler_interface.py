# -*- coding: utf-8 -*-
"""
Abstract ``AbstractFigureAggregator`` interface for the concrete *Resampler* classes.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import itertools
import re
from abc import ABC
from collections import namedtuple
from copy import copy
from typing import Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.io.json._normalize import nested_to_record
from plotly.basedatatypes import BaseFigure, BaseTraceType

from ..aggregation import AbstractAggregator, MedDiffGapHandler, MinMaxLTTB
from ..aggregation.aggregation_interface import DataPointSelector
from ..aggregation.gap_handler_interface import AbstractGapHandler
from ..aggregation.plotly_aggregator_parser import PlotlyAggregatorParser
from .utils import round_number_str, round_td_str

# A high-frequency data container
# NOTE: the attributes must all be valid trace attributes, with attribute levels
# separated by an '_' (e.g., 'marker_color' is valid) as the
# `_hf_data_container._asdict()` function is used in
#  `AbstractFigureAggregator._construct_hf_data_dict`.
_hf_data_container = namedtuple(
    "DataContainer",
    ["x", "y", "text", "hovertext", "marker_size", "marker_color", "customdata"],
)


class AbstractFigureAggregator(BaseFigure, ABC):
    """Abstract interface for data aggregation functionality for plotly figures."""

    _high_frequency_traces = ["scatter", "scattergl"]

    def __init__(
        self,
        figure: BaseFigure,
        convert_existing_traces: bool = True,
        default_n_shown_samples: int = 1000,
        default_downsampler: AbstractAggregator = MinMaxLTTB(),
        default_gap_handler: AbstractGapHandler = MedDiffGapHandler(),
        resampled_trace_prefix_suffix: Tuple[str, str] = (
            '<b style="color:sandybrown">[R]</b> ',
            "",
        ),
        show_mean_aggregation_size: bool = True,
        convert_traces_kwargs: dict | None = None,
        verbose: bool = False,
    ):
        """Instantiate a resampling data mirror.

        Parameters
        ----------
        figure: BaseFigure
            The figure that will be decorated. Can be either an empty figure
            (e.g., ``go.Figure()``, ``make_subplots()``, ``go.FigureWidget``) or an
            existing figure.
        convert_existing_traces: bool
            A bool indicating whether the high-frequency traces of the passed ``figure``
            should be resampled, by default True. Hence, when set to False, the
            high-frequency traces of the passed ``figure`` will not be resampled.
        default_n_shown_samples: int, optional
            The default number of samples that will be shown for each trace,
            by default 1000.\n
            !!! note
                * This can be overridden within the [`add_trace`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.add_trace] method.
                * If a trace withholds fewer datapoints than this parameter,
                  the data will *not* be aggregated.
        default_downsampler: AbstractAggregator
            An instance which implements the AbstractSeriesDownsampler interface and
            will be used as default downsampler, by default ``MinMaxLTTB``. \n
            !!! note
                This can be overridden within the [`add_trace`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.add_trace] method.
        default_gap_handler: GapHandler
            An instance which implements the AbstractGapHandler interface and will be
            used as default gap handler, by default ``MedDiffGapHandler``. \n
            !!! note
                This can be overridden within the [`add_trace`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.add_trace] method.
        resampled_trace_prefix_suffix: str, optional
            A tuple which contains the ``prefix`` and ``suffix``, respectively, which
            will be added to the trace its legend-name when a resampled version of the
            trace is shown. By default, a bold, orange ``[R]`` is shown as prefix
            (no suffix is shown).
        show_mean_aggregation_size: bool, optional
            Whether the mean aggregation bin size will be added as a suffix to the trace
            its legend-name, by default True.
        convert_traces_kwargs: dict, optional
            A dict of kwargs that will be passed to the [`add_traces`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.add_traces] method and
            will be used to convert the existing traces. \n
            !!! note
                This argument is only used when the passed ``figure`` contains data and
                ``convert_existing_traces`` is set to True.
        verbose: bool, optional
            Whether some verbose messages will be printed or not, by default False.

        """
        self._hf_data: Dict[str, dict] = {}
        self._global_n_shown_samples = default_n_shown_samples
        self._print_verbose = verbose
        self._show_mean_aggregation_size = show_mean_aggregation_size

        assert len(resampled_trace_prefix_suffix) == 2
        self._prefix, self._suffix = resampled_trace_prefix_suffix

        self._global_downsampler = default_downsampler
        self._global_gap_handler = default_gap_handler

        # Given figure should always be a BaseFigure that is not wrapped by
        # a plotly-resampler class
        assert isinstance(figure, BaseFigure)
        assert not issubclass(type(figure), AbstractFigureAggregator)
        self._figure_class = figure.__class__

        # Overwrite the passed arguments with the property dict values
        # (this is the case when the PR figure is created from a pickled object)
        if hasattr(figure, "_pr_props"):
            pr_props = figure._pr_props  # a dict of PR properties
            if pr_props is not None:
                # Overwrite the default arguments with the serialized properties
                for k, v in pr_props.items():
                    setattr(self, k, v)
            delattr(figure, "_pr_props")  # should not be stored anymore

        if convert_existing_traces:
            # call __init__ with the correct layout and set the `_grid_ref` of the
            # to-be-converted figure
            f_ = self._figure_class(layout=figure.layout)
            f_._grid_str = figure._grid_str
            f_._grid_ref = figure._grid_ref
            super().__init__(f_)

            if convert_traces_kwargs is None:
                convert_traces_kwargs = {}

            # make sure that the UIDs of these traces do not get adjusted
            self._data_validator.set_uid = False
            self.add_traces(figure.data, **convert_traces_kwargs)
        else:
            super().__init__(figure)
            self._data_validator.set_uid = False

        # A list of al xaxis and yaxis string names
        # e.g., "xaxis", "xaxis2", "xaxis3", .... for _xaxis_list
        self._xaxis_list = self._re_matches(
            re.compile(r"xaxis\d*"), self._layout.keys()
        )
        self._yaxis_list = self._re_matches(
            re.compile(r"yaxis\d*"), self._layout.keys()
        )
        # edge case: an empty `go.Figure()` does not yet contain axes keys
        if not len(self._xaxis_list):
            self._xaxis_list = ["xaxis"]
            self._yaxis_list = ["yaxis"]

        # Make sure to reset the layout its range
        # self.update_layout(
        #     {
        #         axis: {"autorange": None, "range": None}
        #         for axis in self._xaxis_list + self._yaxis_list
        #     }
        # )

    def _print(self, *values):
        """Helper method for printing if ``verbose`` is set to True."""
        if self._print_verbose:
            print(*values)

    def _query_hf_data(self, trace: dict) -> Optional[dict]:
        """Query the internal ``_hf_data`` attribute and returns a match based on
        ``uid``.

        Parameters
        ----------
        trace : dict
            The trace where we want to find a match for.

        Returns
        -------
        Optional[dict]
            The ``hf_data``-trace dict if a match is found, else ``None``.

        """
        uid = trace["uid"]
        hf_trace_data = self._hf_data.get(uid)
        if hf_trace_data is None:
            trace_props = {
                k: trace[k] for k in set(trace.keys()).difference({"x", "y"})
            }
            self._print(f"[W] trace with {trace_props} not found")
        return hf_trace_data

    def _get_current_graph(self) -> dict:
        """Create an efficient copy of the current graph by omitting the "hovertext",
        "x", and "y" properties of each trace.

        Returns
        -------
        dict
            The current graph dict

        See Also
        --------
        https://github.com/plotly/plotly.py/blob/2e7f322c5ea4096ce6efe3b4b9a34d9647a8be9c/packages/python/plotly/plotly/basedatatypes.py#L3278
        """
        return {
            "data": [
                {
                    k: copy(trace[k])
                    # TODO: why not "text" as well? -> we can use _hf_data_container.fields then
                    for k in set(trace.keys()).difference({"x", "y", "hovertext"})
                }
                for trace in self._data
            ],
            "layout": copy(self._layout),
        }

    def _parse_trace_name(
        self, hf_trace_data: dict, slice_len: int, agg_x: np.ndarray
    ) -> str:
        """Parse the trace name.

        Parameters
        ----------
        hf_trace_data : dict
            The high-frequency trace data dict.
        slice_len : int
            The length of the slice.
        agg_x : np.ndarray
            The x-axis values of the aggregated trace.

        Returns
        -------
        str
            The parsed trace name.
            When no downsampling is needed, the original trace name is returned.
            When downsampling is needed, the average bin size (expressed in x-units) is
            added in orange color with a `~` to the trace name.

        """
        if slice_len <= hf_trace_data["max_n_samples"]:  # When no downsampling needed
            return hf_trace_data["name"]

        # The data is downsampled, so we add the downsampling information to the name
        agg_prefix, agg_suffix = ' <i style="color:#fc9944">~', "</i>"
        name = self._prefix + hf_trace_data["name"] + self._suffix

        # Add the mean aggregation bin size to the trace name
        if self._show_mean_aggregation_size:
            # Base case ...
            if len(agg_x) < 2:
                return name

            mean_bin_size = (agg_x[-1] - agg_x[0]) / agg_x.shape[0]  # mean bin size
            if isinstance(mean_bin_size, (np.timedelta64, pd.Timedelta)):
                mean_bin_size = round_td_str(pd.Timedelta(mean_bin_size))
            else:
                mean_bin_size = round_number_str(mean_bin_size)
            name += f"{agg_prefix}{mean_bin_size}{agg_suffix}"
        return name

    def _check_update_trace_data(
        self,
        trace: dict,
        start: Optional[Union[str, float]] = None,
        end: Optional[Union[str, float]] = None,
    ) -> Optional[Union[dict, BaseTraceType]]:
        """Check and update the passed ``trace`` its data properties based on the
        slice range.

        Note
        ----
        This is a pass by reference. The passed trace object will be updated and
        returned if found in ``hf_data``.

        Parameters
        ----------
        trace : BaseTraceType or dict
             - An instances of a trace class from the ``plotly.graph_objects`` (go)
                package (e.g, ``go.Scatter``, ``go.Bar``)
             - or a dict where:

                  - The 'type' property specifies the trace type (e.g.
                    'scatter', 'bar', 'area', etc.). If the dict has no 'type'
                    property then 'scatter' is assumed.
                  - All remaining properties are passed to the constructor
                    of the specified trace type.

        start : Union[float, str], optional
            The start index for which we want resampled data to be updated to,
            by default None,
        end : Union[float, str], optional
            The end index for which we want the resampled data to be updated to,
            by default None

        Returns
        -------
        Optional[Union[dict, BaseTraceType]]
            If the matching ``hf_series`` is found in ``hf_dict``, an (updated) trace
            will be returned, otherwise None.

        Note
        ----
        * If ``start`` and ``stop`` are strings, they most likely represent time-strings
        * ``start`` and ``stop`` will always be of the same type (float / time-string)
           because their underlying axis is the same.

        """
        hf_trace_data = self._query_hf_data(trace)

        if hf_trace_data is None:
            self._print("hf_data not found")
            return None

        # Parse trace data (necessary when updating the trace data)
        for k in _hf_data_container._fields:
            if isinstance(
                hf_trace_data[k], (np.ndarray, pd.RangeIndex, pd.DatetimeIndex)
            ):
                # is faster to escape the loop here than check inside the hasattr if
                continue
            elif pd.DatetimeTZDtype.is_dtype(hf_trace_data[k]):
                # When we use the .values method, timezone information is lost
                # so convert it to pd.DatetimeIndex, which preserves the tz-info
                hf_trace_data[k] = pd.Index(hf_trace_data[k])
            elif hasattr(hf_trace_data[k], "values"):
                # when not a range index or datetime index
                hf_trace_data[k] = hf_trace_data[k].values

        # Also check if the y-data is empty, if so, return an empty trace
        if len(hf_trace_data["y"]) == 0:
            trace["x"] = []
            trace["y"] = []
            trace["name"] = hf_trace_data["name"]
            return trace

        # Leverage the axis type to get the start and end indices
        # Note: the axis type specified in the figure layout takes precedence over the
        # the axis type which is inferred from the data (and stored in hf_trace_data)
        # TODO: verify if we need to use `axis`of anchor as key to determing axis type
        axis = trace.get("xaxis", "x")
        axis_type = self.layout._props.get(axis[:1] + "axis" + axis[1:], {}).get(
            "type", hf_trace_data["axis_type"]
        )
        start_idx, end_idx = PlotlyAggregatorParser.get_start_end_indices(
            hf_trace_data, axis_type, start, end
        )

        # Return an invisible, single-point, trace when the sliced hf_series doesn't
        # contain any data in the current view
        if end_idx == start_idx:
            trace["x"] = [hf_trace_data["x"][0]]
            trace["y"] = [None]
            trace["name"] = hf_trace_data["name"]
            return trace

        agg_x, agg_y, indices = PlotlyAggregatorParser.aggregate(
            hf_trace_data, start_idx, end_idx
        )

        # -------------------- Set the hf_trace_data_props -------------------
        trace["x"] = agg_x
        trace["y"] = agg_y
        trace["name"] = self._parse_trace_name(
            hf_trace_data, end_idx - start_idx, agg_x
        )

        def _nest_dict_rec(k: str, v: any, out: dict) -> None:
            """Recursively nest a dict based on the key whose '_' indicates level."""
            k, *rest = k.split("_", 1)
            if rest:
                _nest_dict_rec(rest[0], v, out.setdefault(k, {}))
            else:
                out[k] = v

        # Check if (hover)text also needs to be downsampled
        for k in ["text", "hovertext", "marker_size", "marker_color", "customdata"]:
            k_val = hf_trace_data.get(k)
            if isinstance(k_val, (np.ndarray, pd.Series)):
                assert isinstance(
                    hf_trace_data["downsampler"], DataPointSelector
                ), "Only DataPointSelector can downsample non-data trace array props."
                _nest_dict_rec(k, k_val[start_idx + indices], trace)
            elif k_val is not None:
                trace[k] = k_val

        return trace

    def _layout_xaxis_to_trace_xaxis_mapping(self) -> Dict[str, List[str]]:
        """Construct a dict which maps the layout xaxis keys to the trace xaxis keys.

        Returns
        -------
        Dict[str, List[str]]
            A dict with the layout xaxis values as keys and the trace its corresponding
            xaxis anchor value.

        """
        # edge case: an empty `go.Figure()` does not yet contain axes keys
        if self._grid_ref is None:
            return {"xaxis": ["x"]}

        mapping_dict = {}
        for sub_plot in itertools.chain.from_iterable(self._grid_ref):  # flattten
            sub_plot = [] if sub_plot is None else sub_plot
            for axes in sub_plot:  # NOTE: you can have multiple axes in a subplot
                layout_xaxes = axes.layout_keys[0]
                trace_xaxes = axes.trace_kwargs["xaxis"]

                # append the trace xaxis to the layout xaxis key its value list
                mapping_dict.setdefault(layout_xaxes, []).append(trace_xaxes)
        return mapping_dict

    def _check_update_figure_dict(
        self,
        figure: dict,
        start: Optional[Union[float, str]] = None,
        stop: Optional[Union[float, str]] = None,
        layout_xaxis_filter: Optional[str] = None,
        updated_trace_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """Check and update the traces within the figure dict.

        hint
        ----
        This method will most likely be used within a ``Dash`` callback to resample the
        view, based on the configured number of parameters.

        Note
        ----
        This is a pass by reference. The passed figure object will be updated.
        No new view of this figure will be created, hence no return!

        Parameters
        ----------
        figure : dict
            The figure dict which will be updated.
        start : Union[float, str], optional
            The start time for the new resampled data view, by default None.
        stop : Union[float, str], optional
            The end time for the new resampled data view, by default None.
        layout_xaxis_filter: str, optional
            Additional layout xaxis filter, e.g. the affected x-axis values by the
            triggered relayout event (e.g. xaxis), by default None.
        updated_trace_indices: List[int], optional
            List of trace indices that already have been updated, by default None.

        Returns
        -------
        List[int]
            A list of indices withholding the trace-data-array-index from the of data
            modalities which are updated.

        """
        if updated_trace_indices is None:
            updated_trace_indices = []

        if layout_xaxis_filter is not None:
            layout_trace_mapping = self._layout_xaxis_to_trace_xaxis_mapping()
            # Retrieve the trace xaxis values that are affected by the relayout event
            trace_xaxis_filter: List[str] = layout_trace_mapping[layout_xaxis_filter]

        for idx, trace in enumerate(figure["data"]):
            # We skip when (i) the trace-idx already has been updated or (ii) when
            # there is a layout_xaxis_filter and the trace xaxis is not in the filter
            if idx in updated_trace_indices or (
                layout_xaxis_filter is not None
                and trace.get("xaxis", "x") not in trace_xaxis_filter
            ):
                continue

            # If we managed to find and update the trace, it will return the trace
            # and thus not None.
            updated_trace = self._check_update_trace_data(trace, start=start, end=stop)
            if updated_trace is not None:
                updated_trace_indices.append(idx)
        return updated_trace_indices

    @staticmethod
    def _get_figure_class(constr: type) -> type:
        """Get the plotly figure class (constructor) for the given class (constructor).

        !!! note
            This method will always return a plotly constructor, even when the given
            `constr` is decorated (after executing the ``register_plotly_resampler``
            function).

        Parameters
        ----------
        constr: type
            The constructor class for which we want to retrieve the plotly constructor.

        Returns
        -------
        type:
            The plotly figure class (constructor) of the given `constr`.

        """
        from ..registering import _get_plotly_constr  # To avoid ImportError

        return _get_plotly_constr(constr)

    @property
    def hf_data(self):
        """Property to adjust the `data` component of the current graph

        !!! note
            The user has full responsibility to adjust ``hf_data`` properly.


        ??? example

            ```python
                >>> from plotly_resampler import FigureResampler
                >>> fig = FigureResampler(go.Figure())
                >>> fig.add_trace(...)
                >>> # Adjust the y property of the above added trace
                >>> fig.hf_data[-1]["y"] = - s ** 2
                >>> fig.hf_data
                [
                    {
                        'max_n_samples': 1000,
                        'x': RangeIndex(start=0, stop=11000000, step=1),
                        'y': array([-0.01339909,  0.01390696,, ...,  0.25051913, 0.55876513]),
                        'axis_type': 'linear',
                        'downsampler': <plotly_resampler.aggregation.aggregators.LTTB at 0x7f786d5a9ca0>,
                        'text': None,
                        'hovertext': None
                    },
                ]
            ```
        """
        return list(self._hf_data.values())

    def _parse_get_trace_props(
        self,
        trace: BaseTraceType,
        hf_x: Iterable = None,
        hf_y: Iterable = None,
        hf_text: Iterable = None,
        hf_hovertext: Iterable = None,
        hf_marker_size: Iterable = None,
        hf_marker_color: Iterable = None,
    ) -> _hf_data_container:
        """Parse and capture the possibly high-frequency trace-props in a datacontainer.

        Parameters
        ----------
        trace : BaseTraceType
            The trace which will be parsed.
        hf_x : Iterable, optional
            High-frequency trace "x" data, overrides the current trace its x-data.
        hf_y : Iterable, optional
            High-frequency trace "y" data, overrides the current trace its y-data.
        hf_text : Iterable, optional
            High-frequency trace "text" data, overrides the current trace its text-data.
        hf_hovertext : Iterable, optional
            High-frequency trace "hovertext" data, overrides the current trace its
            hovertext data.

        Returns
        -------
        _hf_data_container
            A namedtuple which serves as a datacontainer.

        """
        hf_x: np.ndarray | pd.Index = (
            # fmt: off
            (np.asarray(trace["x"]) if trace["x"] is not None else None)
            if hasattr(trace, "x") and hf_x is None
            # If we cast a tz-aware datetime64 array to `.values` we lose the tz-info 
            # and the UTC time will be displayed instead of the tz-localized time, 
            # hence we cast to a pd.DatetimeIndex, which preserves the tz-info
            # As a matter of fact, to resolve #231, we also convert non-tz-aware 
            # datetime64 arrays to an pd.Index
            else pd.Index(hf_x) if pd.core.dtypes.common.is_datetime64_any_dtype(hf_x)
            else hf_x.values if isinstance(hf_x, pd.Series)
            else hf_x if isinstance(hf_x, pd.Index)
            else np.asarray(hf_x)
            # fmt: on
        )
        if pd.core.dtypes.common.is_datetime64_any_dtype(hf_x):
            hf_x = pd.Index(hf_x)

        hf_y = (
            trace["y"]
            if hasattr(trace, "y") and hf_y is None
            else hf_y.values if isinstance(hf_y, (pd.Series, pd.Index)) else hf_y
        )
        # NOTE: the if will not be triggered for a categorical series its values
        if not hasattr(hf_y, "dtype"):
            hf_y: np.ndarray = np.asarray(hf_y)

        hf_text = (
            hf_text
            if hf_text is not None
            else (
                trace["text"]
                if hasattr(trace, "text") and trace["text"] is not None
                else None
            )
        )

        hf_hovertext = (
            hf_hovertext
            if hf_hovertext is not None
            else (
                trace["hovertext"]
                if hasattr(trace, "hovertext") and trace["hovertext"] is not None
                else None
            )
        )

        hf_marker_size = (
            trace["marker"]["size"]
            if (
                hf_marker_size is None
                and hasattr(trace, "marker")
                and "size" in trace["marker"]
            )
            else hf_marker_size
        )

        hf_marker_color = (
            trace["marker"]["color"]
            if (
                hf_marker_color is None
                and hasattr(trace, "marker")
                and "color" in trace["marker"]
            )
            else hf_marker_color
        )

        hf_customdata = trace["customdata"] if hasattr(trace, "customdata") else None

        if trace["type"].lower() in self._high_frequency_traces:
            if hf_x is None:  # if no data as x or hf_x is passed
                if hf_y.ndim != 0:  # if hf_y is an array
                    hf_x = pd.RangeIndex(0, len(hf_y))  # np.arange(len(hf_y))
                else:  # if no data as y or hf_y is passed
                    hf_x = np.asarray([])
                    hf_y = np.asarray([])

            assert hf_y.ndim == np.ndim(hf_x), (
                "plotly-resampler requires scatter data "
                "(i.e., x and y, or hf_x and hf_y) to have the same dimensionality!"
            )
            # When the x or y of a trace has more than 1 dimension, it is not at all
            # straightforward how it should be resampled.
            assert hf_y.ndim <= 1 and np.ndim(hf_x) <= 1, (
                "plotly-resampler requires scatter data "
                "(i.e., x and y, or hf_x and hf_y) to be <= 1 dimensional!"
            )

            # Note: this converts the hf property to a np.ndarray
            if isinstance(hf_text, (tuple, list, np.ndarray, pd.Series)):
                hf_text = np.asarray(hf_text)
            if isinstance(hf_hovertext, (tuple, list, np.ndarray, pd.Series)):
                hf_hovertext = np.asarray(hf_hovertext)
            if isinstance(hf_marker_size, (tuple, list, np.ndarray, pd.Series)):
                hf_marker_size = np.asarray(hf_marker_size)
            if isinstance(hf_marker_color, (tuple, list, np.ndarray, pd.Series)):
                hf_marker_color = np.asarray(hf_marker_color)

            # Try to parse the hf_x data if it is of object type or
            if len(hf_x) and (hf_x.dtype.type is np.str_ or hf_x.dtype == "object"):
                try:
                    # Try to parse to numeric
                    hf_x = pd.to_numeric(hf_x, errors="raise")
                except (ValueError, TypeError):
                    try:
                        # Try to parse to datetime
                        hf_x = pd.to_datetime(hf_x, utc=False, errors="raise")
                        # Will be cast to object array if it contains multiple timezones.
                        if hf_x.dtype == "object":
                            raise ValueError(
                                "The x-data contains multiple timezones, which is not "
                                "supported by plotly-resampler!"
                            )
                    except (ValueError, TypeError):
                        raise ValueError(
                            "plotly-resampler requires the x-data to be numeric or "
                            "datetime-like \nMore details in the stacktrace above."
                        )

            # If the categorical or string-like hf_y data is of type object (happens
            # when y argument is used for the trace constructor instead of hf_y), we
            # transform it to type string as such it will be sent as categorical data
            # to the downsampling algorithm
            if hf_y.dtype == "object" or hf_y.dtype.type == np.str_:
                # But first, we try to parse to a numeric dtype (as this is the
                # behavior that plotly supports)
                # Note that a bool array of type object will remain a bool array (and
                # not will be transformed to an array of ints (0, 1))
                try:
                    hf_y = pd.to_numeric(hf_y, errors="raise")
                except ValueError:
                    hf_y = pd.Categorical(hf_y)  # TODO: ordered=True?
            assert len(hf_x) == len(hf_y), "x and y have different length!"
        else:
            self._print(f"trace {trace['type']} is not a high-frequency trace")

            # hf_x and hf_y have priority over the traces' data
            if hasattr(trace, "x"):
                trace["x"] = hf_x

            if hasattr(trace, "y"):
                trace["y"] = hf_y

            if hasattr(trace, "text"):
                trace["text"] = hf_text

            if hasattr(trace, "hovertext"):
                trace["hovertext"] = hf_hovertext
            if hasattr(trace, "marker"):
                if hasattr(trace.marker, "size"):
                    trace.marker.size = hf_marker_size
                if hasattr(trace.marker, "color"):
                    trace.marker.color = hf_marker_color

        return _hf_data_container(
            hf_x,
            hf_y,
            hf_text,
            hf_hovertext,
            hf_marker_size,
            hf_marker_color,
            hf_customdata,
        )

    def _construct_hf_data_dict(
        self,
        dc: _hf_data_container,
        trace: BaseTraceType,
        downsampler: AbstractAggregator | None,
        gap_handler: AbstractGapHandler | None,
        max_n_samples: int | None,
        offset=0,
    ) -> dict:
        """Create the `hf_data` dict which will be put in the `_hf_data` property.

        Parameters
        ----------
        dc : _hf_data_container
            The hf_data container, withholding the parsed hf-data.
        trace : BaseTraceType
            The trace.
        downsampler : AbstractAggregator | None
            The downsampler which will be used.
        gap_handler : AbstractGapHandler | None
            The gap handler which will be used.
        max_n_samples : int | None
            The max number of output samples.

        Returns
        -------
        dict
            The hf_data dict.
        """
        # Checking this now avoids less interpretable `KeyError` when resampling
        assert_text = (
            "In order to perform time series aggregation, the data must be "
            "sorted in time; i.e., the x-data must be (non-strictly) "
            "monotonically increasing."
        )
        if isinstance(dc.x, pd.Index):
            assert dc.x.is_monotonic_increasing, assert_text
        else:
            assert pd.Series(dc.x).is_monotonic_increasing, assert_text

        # As we support prefix-suffixing of downsampled data, we assure that
        # each trace has a name
        # https://github.com/plotly/plotly.py/blob/ce0ed07d872c487698bde9d52e1f1aadf17aa65f/packages/python/plotly/plotly/basedatatypes.py#L539
        # The link above indicates that the trace index is derived from `data`
        if trace.name is None:
            trace.name = f"trace {len(self.data) + offset}"

        # Determine (1) the axis type and (2) the downsampler instance
        # & (3) store a hf_data entry for the corresponding trace,
        # identified by its UUID
        axis_type = (
            "date"
            if isinstance(dc.x, pd.DatetimeIndex)
            or pd.core.dtypes.common.is_datetime64_any_dtype(dc.x)
            else "linear"
        )

        default_n_samples = False
        if max_n_samples is None:
            default_n_samples = True
            max_n_samples = self._global_n_shown_samples

        default_downsampler = False
        if downsampler is None:
            default_downsampler = True
            downsampler = self._global_downsampler

        default_gap_handler = False
        if gap_handler is None:
            default_gap_handler = True
            gap_handler = self._global_gap_handler

        # TODO -> can't we just store the DC here (might be less duplication of
        #  code knowledge, because now, you need to know all the eligible hf_keys in
        #  dc
        return {
            "max_n_samples": max_n_samples,
            "default_n_samples": default_n_samples,
            "name": trace.name,
            "axis_type": axis_type,
            "downsampler": downsampler,
            "default_downsampler": default_downsampler,
            "gap_handler": gap_handler,
            "default_gap_handler": default_gap_handler,
            **dc._asdict(),
        }

    @staticmethod
    def _add_trace_to_add_traces_kwargs(kwargs: dict) -> dict:
        """Convert the `add_trace` kwargs to the `add_traces` kwargs."""
        # The keywords that need to be converted to a list
        convert_keywords = ["row", "col", "secondary_y"]

        updated_kwargs = {}  # The updated kwargs (from `add_trace` to `add_traces`)
        for keyword in convert_keywords:
            value = kwargs.pop(keyword, None)
            if value is not None:
                updated_kwargs[f"{keyword}s"] = [value]
            else:
                updated_kwargs[f"{keyword}s"] = None

        return {**kwargs, **updated_kwargs}

    def add_trace(
        self,
        trace: Union[BaseTraceType, dict],
        max_n_samples: int = None,
        downsampler: AbstractAggregator = None,
        gap_handler: AbstractGapHandler = None,
        limit_to_view: bool = False,
        # Use these if you want some speedups (and are working with really large data)
        hf_x: Iterable = None,
        hf_y: Iterable = None,
        hf_text: Union[str, Iterable] = None,
        hf_hovertext: Union[str, Iterable] = None,
        hf_marker_size: Union[str, Iterable] = None,
        hf_marker_color: Union[str, Iterable] = None,
        **trace_kwargs,
    ):
        """Add a trace to the figure.

        Parameters
        ----------
        trace : BaseTraceType or dict
            Either:

              - An instances of a trace class from the ``plotly.graph_objects`` (go)
                package (e.g., ``go.Scatter``, ``go.Bar``)
              - or a dict where:

                - The type property specifies the trace type (e.g. scatter, bar,
                  area, etc.). If the dict has no 'type' property then scatter is
                  assumed.
                - All remaining properties are passed to the constructor
                  of the specified trace type.
        max_n_samples : int, optional
            The maximum number of samples that will be shown by the trace.\n
            !!! note
                If this variable is not set; ``_global_n_shown_samples`` will be used.
        downsampler: AbstractAggregator, optional
            The abstract series downsampler method.\n
            !!! note
                If this variable is not set, ``_global_downsampler`` will be used.
        gap_handler: AbstractGapHandler, optional
            The abstract series gap handler method.\n
            !!! note
                If this variable is not set, ``_global_gap_handler`` will be used.
        limit_to_view: boolean, optional
            If set to True, the trace's datapoints will be cut to the corresponding
            front-end view, even if the total number of samples is lower than
            ``max_n_samples``, By default False.\n
            Remark that setting this parameter to True ensures that low frequency traces
            are added to the ``hf_data`` property.
        hf_x: Iterable, optional
            The original high frequency series positions, can be either a time-series or
            an increasing, numerical index. If set, this has priority over the trace its
            data.
        hf_y: Iterable, optional
            The original high frequency values. If set, this has priority over the
            trace its data.
        hf_text: Iterable, optional
            The original high frequency text. If set, this has priority over the trace
            its ``text`` argument.
        hf_hovertext: Iterable, optional
            The original high frequency hovertext. If set, this has priority over the
            trace its ```hovertext`` argument.
        hf_marker_size: Iterable, optional
            The original high frequency marker size. If set, this has priority over the
            trace its ``marker.size`` argument.
        hf_marker_color: Iterable, optional
            The original high frequency marker color. If set, this has priority over the
            trace its ``marker.color`` argument.
        **trace_kwargs: dict
            Additional trace related keyword arguments.
            e.g.: row=.., col=..., secondary_y=...

            !!! info "See Also"
                [`Figure.add_trace`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.add_trace>) docs.

        Returns
        -------
        BaseFigure
            The Figure on which ``add_trace`` was called on; i.e. self.

        !!! note

            Constructing traces with **very large data amounts** really takes some time.
            To speed this up; use this [`add_trace`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.add_trace] method and

            1. Create a trace with no data (empty lists)
            2. pass the high frequency data to this method using the ``hf_x`` and ``hf_y``
               parameters.

            See the example below:
                ```python
                >>> from plotly.subplots import make_subplots
                >>> s = pd.Series()  # a high-frequency series, with more than 1e7 samples
                >>> fig = FigureResampler(go.Figure())
                >>> fig.add_trace(go.Scattergl(x=[], y=[], ...), hf_x=s.index, hf_y=s)
                ```

            !!! todo
                * explain why adding x and y to a trace is so slow
                * check and simplify the example above

        !!! tip

            * If you **do not want to downsample** your data, set ``max_n_samples`` to the
              the number of datapoints of your trace!

        !!! warning

            * The ``NaN`` values in either ``hf_y`` or ``trace.y`` will be omitted! We do
              not allow ``NaN`` values in ``hf_x`` or ``trace.x``.
            * ``hf_x``, ``hf_y``, ``hf_text``, and ``hf_hovertext`` are useful when you deal
              with large amounts of data (as it can increase the speed of this add_trace()
              method with ~30%). These arguments have priority over the trace's data and
              (hover)text attributes.
            * Low-frequency time-series data, i.e. traces that are not resampled, can hinder
              the the automatic-zooming (y-scaling) as these will not be stored in the
              back-end and thus not be scaled to the view.
              To circumvent this, the ``limit_to_view`` argument can be set, resulting in
              also storing the low-frequency series in the back-end.

        """
        # to comply with the plotly data input acceptance behavior
        if isinstance(trace, (list, tuple)):
            raise ValueError("Trace must be either a dict or a BaseTraceType")

        max_out_s = (
            self._global_n_shown_samples if max_n_samples is None else max_n_samples
        )

        # Validate the trace and convert to a trace object
        if not isinstance(trace, BaseTraceType):
            trace = self._data_validator.validate_coerce(trace)[0]

        # First add a UUID, as each (even the non-hf_data traces), must contain this
        # key for comparison. If the trace already has a UUID, we will keep it.
        uuid_str = str(uuid4()) if trace.uid is None else trace.uid
        trace.uid = uuid_str

        # These traces will determine the autoscale its RANGE!
        #   -> so also store when `limit_to_view` is set.
        if trace["type"].lower() in self._high_frequency_traces:
            # construct the hf_data_container
            # TODO in future version -> maybe regex on kwargs which start with `hf_`
            dc = self._parse_get_trace_props(
                trace,
                hf_x,
                hf_y,
                hf_text,
                hf_hovertext,
                hf_marker_size,
                hf_marker_color,
            )

            n_samples = len(dc.x)
            if n_samples > max_out_s or limit_to_view:
                self._print(
                    f"\t[i] DOWNSAMPLE {trace['name']}\t{n_samples}->{max_out_s}"
                )

                self._hf_data[uuid_str] = self._construct_hf_data_dict(
                    dc,
                    trace=trace,
                    downsampler=downsampler,
                    gap_handler=gap_handler,
                    max_n_samples=max_n_samples,
                )

                # Before we update the trace, we create a new pointer to that trace in
                # which the downsampled data will be stored. This way, the original
                # data of the trace to this `add_trace` method will not be altered.
                # We copy (by reference) all the non-data properties of the trace in
                # the new trace.
                trace = trace._props  # convert the trace into a dict
                # NOTE: there is no need to store `marker` property here.
                # If needed, it will be added  to `trace` via `check_update_trace_data`
                trace = {
                    k: trace[k] for k in set(trace.keys()).difference(set(dc._fields))
                }

                # NOTE:
                # If all the raw data needs to be sent to the javascript, and the trace
                # is high-frequency, this would take significant time!
                # Hence, you first downsample the trace.
                trace = self._check_update_trace_data(trace)
                assert trace is not None
                return super(AbstractFigureAggregator, self).add_traces(
                    [trace], **self._add_trace_to_add_traces_kwargs(trace_kwargs)
                )
            else:
                self._print(f"[i] NOT resampling {trace['name']} - len={n_samples}")
                trace._process_kwargs(**{k: getattr(dc, k) for k in dc._fields})
                return super(AbstractFigureAggregator, self).add_traces(
                    [trace], **self._add_trace_to_add_traces_kwargs(trace_kwargs)
                )

        return super().add_traces(
            [trace], **self._add_trace_to_add_traces_kwargs(trace_kwargs)
        )

    def add_traces(
        self,
        data: List[BaseTraceType | dict] | BaseTraceType | Dict,
        max_n_samples: None | List[int] | int = None,
        downsamplers: None | List[AbstractAggregator] | AbstractAggregator = None,
        gap_handlers: None | List[AbstractGapHandler] | AbstractGapHandler = None,
        limit_to_views: List[bool] | bool = False,
        **traces_kwargs,
    ):
        """Add traces to the figure.

        !!! note

            Make sure to look at the [`add_trace`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.add_trace] function for more info about
            **speed optimization**, and dealing with not ``high-frequency`` data, but
            still want to resample / limit the data to the front-end view.

        Parameters
        ----------
        data : List[BaseTraceType  |  dict]
            A list of trace specifications to be added.
            Trace specifications may be either:

              - Instances of trace classes from the plotly.graph_objs
                package (e.g plotly.graph_objs.Scatter, plotly.graph_objs.Bar).
              - Dicts where:

                  - The 'type' property specifies the trace type (e.g.
                    'scatter', 'bar', 'area', etc.). If the dict has no 'type'
                    property then 'scatter' is assumed.
                  - All remaining properties are passed to the constructor
                    of the specified trace type.
        max_n_samples : None | List[int] | int, optional
              The maximum number of samples that will be shown for each trace.
              If a single integer is passed, all traces will use this number. If this
              variable is not set; ``_global_n_shown_samples`` will be used.
        downsamplers : None | List[AbstractAggregator] | AbstractAggregator, optional
            The downsampler that will be used to aggregate the traces. If a single
            aggregator is passed, all traces will use this aggregator.
            If this variable is not set, ``_global_downsampler`` will be used.
        gap_handlers : None | List[AbstractGapHandler] | AbstractGapHandler, optional
            The gap handler that will be used to aggregate the traces. If a single
            gap handler is passed, all traces will use this gap handler.
            If this variable is not set, ``_global_gap_handler`` will be used.
        limit_to_views : None | List[bool] | bool, optional
            List of limit_to_view booleans for the added traces. If set to True the
            trace's datapoints will be cut to the corresponding front-end view, even if
            the total number of samples is lower than ``max_n_samples``.
            If a single boolean is passed, all to be added traces will use this value,
            by default False.\n
            Remark that setting this parameter to True ensures that low frequency traces
            are added to the ``hf_data`` property.
        **traces_kwargs: dict
            Additional trace related keyword arguments.
            e.g.: rows=.., cols=..., secondary_ys=...

            !!! info "See Also"

                [`Figure.add_traces`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.add_traces>) docs.

        Returns
        -------
        BaseFigure
            The Figure on which ``add_traces`` was called on; i.e. self.

        """
        # note: Plotly its add_traces also a allows non list-like input e.g. a scatter
        # object; the code below is an exact copy of their internally applied parsing
        if not isinstance(data, (list, tuple)):
            data = [data]

        # Convert each trace into a BaseTraceType object
        data = [
            (
                self._data_validator.validate_coerce(trace)[0]
                if not isinstance(trace, BaseTraceType)
                else trace
            )
            for trace in data
        ]

        # First add a UUID, as each (even the non-hf_data traces), must contain this
        # key for comparison. If the trace already has a UUID, we will keep it.
        for trace in data:
            uuid_str = str(uuid4()) if trace.uid is None else trace.uid
            trace.uid = uuid_str

        # Convert the data properties
        if isinstance(max_n_samples, (int, np.integer)) or max_n_samples is None:
            max_n_samples = [max_n_samples] * len(data)
        if isinstance(downsamplers, AbstractAggregator) or downsamplers is None:
            downsamplers = [downsamplers] * len(data)
        if isinstance(gap_handlers, AbstractGapHandler) or gap_handlers is None:
            gap_handlers = [gap_handlers] * len(data)
        if isinstance(limit_to_views, bool):
            limit_to_views = [limit_to_views] * len(data)

        zipped = zip(data, max_n_samples, downsamplers, gap_handlers, limit_to_views)
        for i, (trace, max_out, downsampler, gap_handler, limit_to_view) in enumerate(
            zipped
        ):
            if (
                trace.type.lower() not in self._high_frequency_traces
                or self._hf_data.get(trace.uid) is not None
            ):
                continue

            max_out_s = self._global_n_shown_samples if max_out is None else max_out
            if not limit_to_view and (trace.y is None or len(trace.y) <= max_out_s):
                continue

            dc = self._parse_get_trace_props(trace)
            self._hf_data[trace.uid] = self._construct_hf_data_dict(
                dc,
                trace=trace,
                downsampler=downsampler,
                gap_handler=gap_handler,
                max_n_samples=max_out,
                offset=i,
            )

            # convert the trace into a dict, and only withholds the non-hf props
            trace = trace._props
            trace = {k: trace[k] for k in set(trace.keys()).difference(set(dc._fields))}

            # update the trace data with the HF props
            trace = self._check_update_trace_data(trace)
            assert trace is not None
            data[i] = trace

        return super().add_traces(data, **traces_kwargs)

    def _clear_figure(self):
        """Clear the current figure object its data and layout."""
        self._hf_data = {}
        self.data = []
        self._data = []
        self._layout = {}
        self.layout = {}

    def _copy_hf_data(self, hf_data: dict, adjust_default_values: bool = False) -> dict:
        """Copy (i.e. create a new key reference, not a deep copy) of a hf_data dict.

        Parameters
        ----------
        hf_data : dict
            The hf_data dict, having the trace 'uid' as key and the
            hf-data, together with its aggregation properties as dict-values
        adjust_default_values: bool
            Whether the default values (of the downsampler, max # shown samples) will
            be adjusted according to the values of this object, by default False

        Returns
        -------
        dict
            The copied (& default values adjusted) output dict.

        """
        hf_data_cp = {
            uid: {k: hf_dict[k] for k in set(hf_dict.keys())}
            for uid, hf_dict in hf_data.items()
        }

        # Adjust the default arguments to the current argument values
        if adjust_default_values:
            for hf_props in hf_data_cp.values():
                if hf_props.get("default_downsampler", False):
                    hf_props["downsampler"] = self._global_downsampler
                if hf_props.get("default_gap_handler", False):
                    hf_props["gap_handler"] = self._global_gap_handler
                if hf_props.get("default_n_samples", False):
                    hf_props["max_n_samples"] = self._global_n_shown_samples

        return hf_data_cp

    def replace(self, figure: go.Figure, convert_existing_traces: bool = True):
        """Replace the current figure layout with the passed figure object.

        Parameters
        ----------
        figure: go.Figure
            The figure object which will replace the existing figure.
        convert_existing_traces: bool, Optional
            A bool indicating whether the traces of the passed ``figure`` should be
            resampled, by default True.

        """
        self._clear_figure()
        self.__init__(
            figure=figure,
            convert_existing_traces=convert_existing_traces,
            default_n_shown_samples=self._global_n_shown_samples,
            default_downsampler=self._global_downsampler,
            default_gap_handler=self._global_gap_handler,
            resampled_trace_prefix_suffix=(self._prefix, self._suffix),
            show_mean_aggregation_size=self._show_mean_aggregation_size,
            verbose=self._print_verbose,
        )

    def _parse_relayout(self, relayout_dict: dict) -> dict:
        """Update the relayout object so that the autorange will be set to None when
        there are xy-matches.

        Parameters
        ----------
        relayout_dict : dict
            The relayout dictionary.
        """
        # 1. Create a new dict with additional layout updates for the front-end
        extra_layout_updates = {}

        # 1.1. Set autorange to False for each layout item with a specified x-range
        xy_matches = self._re_matches(
            re.compile(r"[xy]axis\d*.range\[\d+]"), relayout_dict.keys()
        )
        for range_change_axis in xy_matches:
            axis = range_change_axis.split(".")[0]
            extra_layout_updates[f"{axis}.autorange"] = None
        return extra_layout_updates

    def _construct_update_data(
        self,
        relayout_data: dict,
    ) -> Union[List[dict], None]:
        """Construct the to-be-updated front-end data, based on the layout change.

        Parameters
        ----------
        relayout_data: dict
            A dict containing the ``relayoutData`` (i.e., the changed layout data) of
            the corresponding front-end graph.

        Returns
        -------
        List[dict]:
            A list of dicts, where each dict-item is a representation of a trace its
            *data* properties which are affected by the front-end layout change. |br|
            In other words, only traces which need to be updated will be sent to the
            front-end. Additionally, each trace-dict withholds the *index* of its
            corresponding position in the ``figure[data]`` array with the ``index``-key
            in each dict.

        """
        current_graph = self._get_current_graph()
        updated_trace_indices, cl_k = [], []
        if relayout_data:
            # flatten the possibly nested dict using '.' as separator
            relayout_data = nested_to_record(relayout_data, sep=".")
            self._print("-" * 100 + "\n", "changed layout", relayout_data)

            cl_k = list(relayout_data.keys())

            # ------------------ HF DATA aggregation ---------------------
            # 1. Base case - there is an x-range specified in the front-end
            start_matches = self._re_matches(re.compile(r"xaxis\d*.range\[0]"), cl_k)
            stop_matches = self._re_matches(re.compile(r"xaxis\d*.range\[1]"), cl_k)

            # related issue: https://github.com/predict-idlab/plotly-resampler/pull/336
            # When the user sets x range via update_xaxes and the number of points in
            # data exceeds the default_n_shown_samples, then after resetting the axes
            # the relayout may only have "xaxis.range", instead of "xaxis.range[0]" and
            # "xaxis.range[1]". If this happens, we need to manually add "xaxis.range[0]"
            # and "xaxis.range[1]", otherwise resetting axes wouldn't work.
            if not (start_matches and stop_matches):
                range_matches = self._re_matches(re.compile(r"xaxis\d*.range"), cl_k)
                for range_match in range_matches:
                    x_range = relayout_data[range_match]
                    start, stop = x_range
                    start_match = range_match + "[0]"
                    stop_match = range_match + "[1]"
                    relayout_data[start_match] = start
                    relayout_data[stop_match] = stop
                    start_matches.append(start_match)
                    stop_matches.append(stop_match)
                    del x_range, start, stop, start_match, stop_match
            if start_matches and stop_matches:  # when both are not empty
                for t_start_key, t_stop_key in zip(start_matches, stop_matches):
                    # Check if the xaxis<NUMB> part of xaxis<NUMB>.[0-1] matches
                    xaxis = t_start_key.split(".")[0]
                    assert xaxis == t_stop_key.split(".")[0]
                    # -> we want to copy the layout on the back-end
                    updated_trace_indices = self._check_update_figure_dict(
                        current_graph,
                        start=relayout_data[t_start_key],
                        stop=relayout_data[t_stop_key],
                        layout_xaxis_filter=xaxis,
                        updated_trace_indices=updated_trace_indices,
                    )

            # 2. The user clicked on either autorange | reset axes
            autorange_matches = self._re_matches(
                re.compile(r"xaxis\d*.autorange"), cl_k
            )
            spike_matches = self._re_matches(re.compile(r"xaxis\d*.showspikes"), cl_k)
            # 2.1 Reset-axes -> autorange & reset to the global data view
            if autorange_matches and spike_matches:  # when both are not empty
                for autorange_key in autorange_matches:
                    if relayout_data[autorange_key]:
                        xaxis = autorange_key.split(".")[0]
                        updated_trace_indices = self._check_update_figure_dict(
                            current_graph,
                            layout_xaxis_filter=xaxis,
                            updated_trace_indices=updated_trace_indices,
                        )
            # 2.1. Autorange -> do nothing, the autorange will be applied on the
            #      current front-end view
            elif (
                autorange_matches and not spike_matches
            ):  # when only autorange is not empty
                # PreventUpdate returns a 204 status code response on the
                # relayout post request
                return dash.no_update

        # If we do not have any traces to be updated, we will return an empty
        # request response
        if not updated_trace_indices:  # when updated_trace_indices is empty
            # PreventUpdate returns a 204 status-code response on the relayout post
            # request
            return dash.no_update

        # -------------------- construct callback data --------------------------
        # 1. Create the layout data for the front-end
        layout_traces_list: List[dict] = [relayout_data]

        # 2. Create the additional trace data for the frond-end
        relevant_keys = list(_hf_data_container._fields) + ["name", "marker"]
        # Note that only updated trace-data will be sent to the client
        for idx in updated_trace_indices:
            trace = current_graph["data"][idx]
            # TODO: check if we can reduce even more
            trace_reduced = {k: trace[k] for k in relevant_keys if k in trace}

            # Store the index into the corresponding to-be-sent trace-data so
            # the client front-end can know which trace needs to be updated
            trace_reduced.update({"index": idx})

            layout_traces_list.append(trace_reduced)
        return layout_traces_list

    @staticmethod
    def _re_matches(regex: re.Pattern, strings: Iterable[str]) -> List[str]:
        """Returns all the items in ``strings`` which regex.match(es) ``regex``."""
        matches = []
        for item in strings:
            m = regex.match(item)
            if m is not None:
                matches.append(m.string)
        return sorted(matches)

    @staticmethod
    def _is_no_update(update_data: Union[List[dict], dash.no_update]) -> bool:
        return update_data is dash.no_update

    # ------------------------------- Magic methods ---------------------------------

    def _get_pr_props_keys(self) -> List[str]:
        """Returns the keys (i.e., the names) of the plotly-resampler properties.

        Note
        ----
        This method is used to serialize the object in the `__reduce__` method.

        """
        return [
            "_hf_data",
            "_global_n_shown_samples",
            "_print_verbose",
            "_show_mean_aggregation_size",
            "_prefix",
            "_suffix",
            "_global_downsampler",
            "_global_gap_handler",
        ]

    def __reduce__(self):
        """Overwrite the reduce method (which is used to support deep copying and
        pickling).

        Note
        ----
        We do not overwrite the `to_dict` method, as this is used to send the figure
        to the frontend (and thus should not capture the plotly-resampler properties).
        """
        _, props = super().__reduce__()
        assert len(props) == 1  # I don't know why this would be > 1
        props = props[0]

        # Add the plotly-resampler properties
        props["pr_props"] = {}
        for k in self._get_pr_props_keys():
            props["pr_props"][k] = getattr(self, k)
        return self.__class__, (props,)  # (props,) to comply with plotly magic
