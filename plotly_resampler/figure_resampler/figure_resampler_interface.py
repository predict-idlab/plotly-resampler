# -*- coding: utf-8 -*-
"""
Abstract ``AbstractFigureAggregator`` interface for the concrete *Resampler* classes.

.. |br| raw:: html

   <br>

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import re
from copy import copy
from typing import Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid4
from collections import namedtuple

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType, BaseFigure

from ..aggregation import AbstractSeriesAggregator, EfficientLTTB
from .utils import round_td_str, round_number_str

from abc import ABC

_hf_data_container = namedtuple("DataContainer", ["x", "y", "text", "hovertext"])


class AbstractFigureAggregator(BaseFigure, ABC):
    """Abstract interface for data aggregation functionality for plotly figures."""

    _high_frequency_traces = ["scatter", "scattergl"]

    def __init__(
        self,
        figure: BaseFigure,
        convert_existing_traces: bool = True,
        default_n_shown_samples: int = 1000,
        default_downsampler: AbstractSeriesAggregator = EfficientLTTB(),
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
            .. note::
                * This can be overridden within the :func:`add_trace` method.
                * If a trace withholds fewer datapoints than this parameter,
                  the data will *not* be aggregated.
        default_downsampler: AbstractSeriesDownsampler
            An instance which implements the AbstractSeriesDownsampler interface and
            will be used as default downsampler, by default ``EfficientLTTB`` with
            _interleave_gaps_ set to True. \n
            .. note:: This can be overridden within the :func:`add_trace` method.
        resampled_trace_prefix_suffix: str, optional
            A tuple which contains the ``prefix`` and ``suffix``, respectively, which
            will be added to the trace its legend-name when a resampled version of the
            trace is shown. By default a bold, orange ``[R]`` is shown as prefix
            (no suffix is shown).
        show_mean_aggregation_size: bool, optional
            Whether the mean aggregation bin size will be added as a suffix to the trace
            its legend-name, by default True.
        convert_traces_kwargs: dict, optional
            A dict of kwargs that will be passed to the :func:`add_traces` method and
            will be used to convert the existing traces. \n
            .. note::
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
        self._xaxis_list = self._re_matches(re.compile("xaxis\d*"), self._layout.keys())
        self._yaxis_list = self._re_matches(re.compile("yaxis\d*"), self._layout.keys())
        # edge case: an empty `go.Figure()` does not yet contain axes keys
        if not len(self._xaxis_list):
            self._xaxis_list = ["xaxis"]
            self._yaxis_list = ["yaxis"]

        # Make sure to reset the layout its range
        self.update_layout(
            {
                axis: {"autorange": None, "range": None}
                for axis in self._xaxis_list + self._yaxis_list
            }
        )

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
                    for k in set(trace.keys()).difference({"x", "y", "hovertext"})
                }
                for trace in self._data
            ],
            "layout": copy(self._layout),
        }

    def _check_update_trace_data(
        self,
        trace: dict,
        start=None,
        end=None,
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
        if hf_trace_data is not None:
            axis_type = hf_trace_data["axis_type"]
            if axis_type == "date":
                start, end = pd.to_datetime(start), pd.to_datetime(end)
                hf_series = self._slice_time(
                    self._to_hf_series(hf_trace_data["x"], hf_trace_data["y"]),
                    start,
                    end,
                )
            else:
                hf_series = self._to_hf_series(hf_trace_data["x"], hf_trace_data["y"])
                if len(hf_series) and (start is not None or end is not None):
                    start = hf_series.index[0] if start is None else start
                    end = hf_series.index[-1] if end is None else end
                    if hf_series.index.is_integer():
                        start = round(start)
                        end = round(end)

                    # Search the index-positions
                    start_idx, end_idx = np.searchsorted(hf_series.index, [start, end])
                    hf_series = hf_series.iloc[start_idx:end_idx]

            # Return an invisible, single-point, trace when the sliced hf_series doesn't
            # contain any data in the current view
            if len(hf_series) == 0:
                trace["x"] = [start]
                trace["y"] = [None]
                trace["text"] = ""
                trace["hovertext"] = ""
                return trace

            # Downsample the data and store it in the trace-fields
            downsampler: AbstractSeriesAggregator = hf_trace_data["downsampler"]
            s_res: pd.Series = downsampler.aggregate(
                hf_series, hf_trace_data["max_n_samples"]
            )
            # Also parse the data types to an orjson compatible format
            # Note this can be removed once orjson supports f16
            trace["x"] = self._parse_dtype_orjson(s_res.index)
            trace["y"] = self._parse_dtype_orjson(s_res.values)
            # todo -> first draft & not MP safe

            agg_prefix, agg_suffix = ' <i style="color:#fc9944">~', "</i>"
            name: str = trace["name"].split(agg_prefix)[0]

            if len(hf_series) > hf_trace_data["max_n_samples"]:
                name = ("" if name.startswith(self._prefix) else self._prefix) + name
                name += self._suffix if not name.endswith(self._suffix) else ""
                # Add the mean aggregation bin size to the trace name
                if self._show_mean_aggregation_size:
                    agg_mean = np.mean(np.diff(s_res.index.values))
                    if isinstance(agg_mean, np.timedelta64):
                        agg_mean = round_td_str(pd.Timedelta(agg_mean))
                    else:
                        agg_mean = round_number_str(agg_mean)
                    name += f"{agg_prefix}{agg_mean}{agg_suffix}"
            else:
                # When not resampled: trim prefix and/or suffix if necessary
                if len(self._prefix) and name.startswith(self._prefix):
                    name = name[len(self._prefix) :]
                if len(self._suffix) and trace["name"].endswith(self._suffix):
                    name = name[: -len(self._suffix)]
            trace["name"] = name

            # Check if text also needs to be resampled
            text = hf_trace_data.get("text")
            if isinstance(text, (np.ndarray, pd.Series)):
                # TODO -> extra logic is necessary for the detection and processing of
                # non data-point selection downsamplers
                trace["text"] = self._to_hf_series(x=hf_trace_data["x"], y=text).loc[
                    s_res.index
                ]
            else:
                trace["text"] = text

            # Check if hovertext also needs to be resampled
            hovertext = hf_trace_data.get("hovertext")
            if isinstance(hovertext, (np.ndarray, pd.Series)):
                trace["hovertext"] = self._to_hf_series(
                    x=hf_trace_data["x"], y=hovertext
                ).loc[s_res.index]
            else:
                trace["hovertext"] = hovertext
            return trace
        else:
            self._print("hf_data not found")
            return None

    def _check_update_figure_dict(
        self,
        figure: dict,
        start: Optional[Union[float, str]] = None,
        stop: Optional[Union[float, str]] = None,
        xaxis_filter: str = None,
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
        xaxis_filter: str, optional
            Additional trace-update subplot filter, by default None.
        updated_trace_indices: List[int], optional
            List of trace indices that already have been updated, by default None.

        Returns
        -------
        List[int]
            A list of indices withholding the trace-data-array-index from the of data
            modalities which are updated.

        """
        xaxis_filter_short = None
        if xaxis_filter is not None:
            xaxis_filter_short = "x" + xaxis_filter.lstrip("xaxis")

        if updated_trace_indices is None:
            updated_trace_indices = []

        for idx, trace in enumerate(figure["data"]):
            # We skip when the trace-idx already has been updated.
            if idx in updated_trace_indices:
                continue

            if xaxis_filter is not None:
                # the x-anchor of the trace is stored in the layout data
                if trace.get("yaxis") is None:
                    # no yaxis -> we make the assumption that yaxis = xaxis_filter_short
                    y_axis = "y" + xaxis_filter[1:]
                else:
                    y_axis = "yaxis" + trace.get("yaxis")[1:]

                # Next to the x-anchor, we also fetch the xaxis which matches the
                # current trace (i.e. if this value is not None, the axis shares the
                # x-axis with one or more traces).
                # This is relevant when e.g. fig.update_traces(xaxis='x...') was called.
                x_anchor_trace = figure["layout"].get(y_axis, {}).get("anchor")
                if x_anchor_trace is not None:
                    xaxis_matches = (
                        figure["layout"]
                        .get("xaxis" + x_anchor_trace.lstrip("x"), {})
                        .get("matches")
                    )
                else:
                    xaxis_matches = figure["layout"].get("xaxis", {}).get("matches")

                # print(
                #     f"x_anchor: {x_anchor_trace} - xaxis_filter: {xaxis_filter} ",
                #     f"- xaxis_matches: {xaxis_matches}"
                # )

                # We skip when:
                # * the change was made on the first row and the trace its anchor is not
                #   in [None, 'x'] and the matching (a.k.a. shared) xaxis is not equal
                #   to the xaxis filter argument.
                #   -> why None: traces without row/col argument and stand on first row
                #      and do not have the anchor property (hence the DICT.get() method)
                # * x_axis_filter_short not in [x_anchor or xaxis matches] for
                #   NON first rows
                if (
                    xaxis_filter_short == "x"
                    and (
                        x_anchor_trace not in [None, "x"]
                        and xaxis_matches != xaxis_filter_short
                    )
                ) or (
                    xaxis_filter_short != "x"
                    and (xaxis_filter_short not in [x_anchor_trace, xaxis_matches])
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

        .. Note::
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

    @staticmethod
    def _slice_time(
        hf_series: pd.Series,
        t_start: Optional[pd.Timestamp] = None,
        t_stop: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """Slice the time-indexed ``hf_series`` for the passed pd.Timestamps.

        Note
        ----
        This returns a **view** of ``hf_series``!

        Parameters
        ----------
        hf_series: pd.Series
            The **datetime-indexed** series, which will be sliced.
        t_start: pd.Timestamp, optional
            The lower-time-bound of the slice, if set to None, no lower-bound threshold
            will be applied, by default None.
        t_stop:  pd.Timestamp, optional
            The upper time-bound of the slice, if set to None, no upper-bound threshold
            will be applied, by default None.

        Returns
        -------
        pd.Series
            The sliced **view** of the series.

        """

        def to_same_tz(
            ts: Union[pd.Timestamp, None], reference_tz=hf_series.index.tz
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

        if t_start is not None and t_stop is not None:
            assert t_start.tz == t_stop.tz

        return hf_series[to_same_tz(t_start) : to_same_tz(t_stop)]

    @property
    def hf_data(self):
        """Property to adjust the `data` component of the current graph

        .. note::
            The user has full responsibility to adjust ``hf_data`` properly.


        Example:
            >>> fig = FigureResampler(go.Figure())
            >>> fig.add_trace(...)
            >>> fig.hf_data[-1]["y"] = - s ** 2  # adjust the y-property of the trace added above
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
        """
        return list(self._hf_data.values())

    @staticmethod
    def _to_hf_series(x: np.ndarray, y: np.ndarray) -> pd.Series:
        """Construct the hf-series.

        By constructing the hf-series this way, users can dynamically adjust the hf
        series argument.

        Parameters
        ----------
        x : np.ndarray
            The hf_series index
        y : np.ndarray
            The hf_series values

        Returns
        -------
        pd.Series
            The constructed hf_series
        """
        # Note this is the same behavior as plotly support
        # i.e. it also used the `values` property of the `x` and `y` parameters when
        # these are pd.Series
        if isinstance(x, pd.Series):
            x = x.values

        if isinstance(y, pd.Series):
            y = y.values

        return pd.Series(
            data=y,
            index=x,
            copy=False,
            name="data",
            dtype="category" if y.dtype.type == np.str_ else y.dtype,
        )

    def _parse_get_trace_props(
        self,
        trace: BaseTraceType,
        hf_x: Iterable = None,
        hf_y: Iterable = None,
        hf_text: Iterable = None,
        hf_hovertext: Iterable = None,
    ) -> _hf_data_container:
        """Parse and capture the possibly high-frequency trace-props in a datacontainer.

        Parameters
        ----------
        trace : BaseTraceType
            The trace which will be parsed.
        hf_x : Iterable, optional
            high-frequency trace "x" data, overrides the current trace its x-data.
        hf_y : Iterable, optional
            high-frequency trace "y" data, overrides the current trace its y-data.
        hf_text : Iterable, optional
            high-frequency trace "text" data, overrides the current trace its text-data.
        hf_hovertext : Iterable, optional
            high-frequency trace "hovertext" data, overrides the current trace its
            hovertext data.

        Returns
        -------
        _hf_data_container
            A namedtuple which serves as a datacontainer.

        """
        hf_x = (
            trace["x"]
            if hasattr(trace, "x") and hf_x is None
            else hf_x.values
            if isinstance(hf_x, pd.Series)
            else hf_x
            if isinstance(hf_x, pd.Index)
            else np.asarray(hf_x)
        )

        hf_y = (
            trace["y"]
            if hasattr(trace, "y") and hf_y is None
            else hf_y.values
            if isinstance(hf_y, (pd.Series, pd.Index))
            else hf_y
        )
        hf_y = np.asarray(hf_y)

        hf_text = (
            hf_text
            if hf_text is not None
            else trace["text"]
            if hasattr(trace, "text") and trace["text"] is not None
            else None
        )

        hf_hovertext = (
            hf_hovertext
            if hf_hovertext is not None
            else trace["hovertext"]
            if hasattr(trace, "hovertext") and trace["hovertext"] is not None
            else None
        )

        if trace["type"].lower() in self._high_frequency_traces:
            if hf_x is None:  # if no data as x or hf_x is passed
                if hf_y.ndim != 0:  # if hf_y is an array
                    hf_x = pd.RangeIndex(0, len(hf_y))  # np.arange(len(hf_y))
                else:  # if no data as y or hf_y is passed
                    hf_x = np.asarray(None)

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

            # Note: this also converts hf_text and hf_hovertext to a np.ndarray
            if isinstance(hf_text, (list, np.ndarray, pd.Series)):
                hf_text = np.asarray(hf_text)
            if isinstance(hf_hovertext, (list, np.ndarray, pd.Series)):
                hf_hovertext = np.asarray(hf_hovertext)

            # Remove NaNs for efficiency (storing less meaningless data)
            # NaNs introduce gaps between enclosing non-NaN data points & might distort
            # the resampling algorithms
            if pd.isna(hf_y).any():
                not_nan_mask = ~pd.isna(hf_y)
                hf_x = hf_x[not_nan_mask]
                hf_y = hf_y[not_nan_mask]
                if isinstance(hf_text, np.ndarray):
                    hf_text = hf_text[not_nan_mask]
                if isinstance(hf_hovertext, np.ndarray):
                    hf_hovertext = hf_hovertext[not_nan_mask]

            # If the categorical or string-like hf_y data is of type object (happens
            # when y argument is used for the trace constructor instead of hf_y), we
            # transform it to type string as such it will be sent as categorical data
            # to the downsampling algorithm
            if hf_y.dtype == "object":
                # But first, we try to parse to a numeric dtype (as this is the
                # behavior that plotly supports)
                # Note that a bool array of type object will remain a bool array (and
                # not will be transformed to an array of ints (0, 1))
                try:
                    hf_y = pd.to_numeric(hf_y, errors="raise")
                except ValueError:
                    hf_y = hf_y.astype("str")

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

        return _hf_data_container(hf_x, hf_y, hf_text, hf_hovertext)

    def _construct_hf_data_dict(
        self,
        dc: _hf_data_container,
        trace: BaseTraceType,
        downsampler: AbstractSeriesAggregator | None,
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
        downsampler : AbstractSeriesAggregator | None
            The downsampler which will be used.
        max_n_samples : int | None
            The max number of output samples.

        Returns
        -------
        dict
            The hf_data dict.
        """
        # We will re-create this each time as hf_x and hf_y withholds
        # high-frequency data and can be adjusted on the fly with the public hf_data
        # property.
        hf_series = self._to_hf_series(x=dc.x, y=dc.y)

        # Checking this now avoids less interpretable `KeyError` when resampling
        assert hf_series.index.is_monotonic_increasing

        # As we support prefix-suffixing of downsampled data, we assure that
        # each trace has a name
        # https://github.com/plotly/plotly.py/blob/ce0ed07d872c487698bde9d52e1f1aadf17aa65f/packages/python/plotly/plotly/basedatatypes.py#L539
        # The link above indicates that the trace index is derived from `data`
        if trace.name is None:
            trace.name = f"trace {len(self.data) + offset}"

        # Determine (1) the axis type and (2) the downsampler instance
        # & (3) store a hf_data entry for the corresponding trace,
        # identified by its UUID
        axis_type = "date" if isinstance(dc.x, pd.DatetimeIndex) else "linear"

        default_n_samples = False
        if max_n_samples is None:
            default_n_samples = True
            max_n_samples = self._global_n_shown_samples

        default_downsampler = False
        if downsampler is None:
            default_downsampler = True
            downsampler = self._global_downsampler

        # TODO -> can't we just store the DC here (might be less duplication of
        #  code knowledge, because now, you need to know all the eligible hf_keys in
        #  dc
        return {
            "max_n_samples": max_n_samples,
            "default_n_samples": default_n_samples,
            "x": dc.x,
            "y": dc.y,
            "axis_type": axis_type,
            "downsampler": downsampler,
            "default_downsampler": default_downsampler,
            "text": dc.text,
            "hovertext": dc.hovertext,
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
        downsampler: AbstractSeriesAggregator = None,
        limit_to_view: bool = False,
        # Use these if you want some speedups (and are working with really large data)
        hf_x: Iterable = None,
        hf_y: Iterable = None,
        hf_text: Union[str, Iterable] = None,
        hf_hovertext: Union[str, Iterable] = None,
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
            .. note::
                If this variable is not set; ``_global_n_shown_samples`` will be used.
        downsampler: AbstractSeriesDownsampler, optional
            The abstract series downsampler method.\n
            .. note::
                If this variable is not set, ``_global_downsampler`` will be used.
        limit_to_view: boolean, optional
            If set to True the trace's datapoints will be cut to the corresponding
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
        **trace_kwargs: dict
            Additional trace related keyword arguments.
            e.g.: row=.., col=..., secondary_y=...

            .. seealso::
                `Figure.add_trace <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.add_trace>`_ docs.

        Returns
        -------
        BaseFigure
            The Figure on which ``add_trace`` was called on; i.e. self.

        Note
        ----
        Constructing traces with **very large data amounts** really takes some time.
        To speed this up; use this :func:`add_trace` method and

        1. Create a trace with no data (empty lists)
        2. pass the high frequency data to this method using the ``hf_x`` and ``hf_y``
           parameters.

        See the example below:

            >>> from plotly.subplots import make_subplots
            >>> s = pd.Series()  # a high-frequency series, with more than 1e7 samples
            >>> fig = FigureResampler(go.Figure())
            >>> fig.add_trace(go.Scattergl(x=[], y=[], ...), hf_x=s.index, hf_y=s)

        .. todo::
            * explain why adding x and y to a trace is so slow
            * check and simplify the example above

        Tip
        ---
        * If you **do not want to downsample** your data, set ``max_n_samples`` to the
          the number of datapoints of your trace!

        Attention
        ---------
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

        # First add an UUID, as each (even the non-hf_data traces), must contain this
        # key for comparison. If the trace already has an UUID, we will keep it.
        uuid_str = str(uuid4()) if trace.uid is None else trace.uid
        trace.uid = uuid_str

        # construct the hf_data_container
        # TODO in future version -> maybe regex on kwargs which start with `hf_`
        dc = self._parse_get_trace_props(trace, hf_x, hf_y, hf_text, hf_hovertext)

        # These traces will determine the autoscale RANGE!
        #   -> so also store when `limit_to_view` is set.
        if trace["type"].lower() in self._high_frequency_traces:
            n_samples = len(dc.x)
            if n_samples > max_out_s or limit_to_view:
                self._print(
                    f"\t[i] DOWNSAMPLE {trace['name']}\t{n_samples}->{max_out_s}"
                )

                self._hf_data[uuid_str] = self._construct_hf_data_dict(
                    dc,
                    trace=trace,
                    downsampler=downsampler,
                    max_n_samples=max_n_samples,
                )

                # Before we update the trace, we create a new pointer to that trace in
                # which the downsampled data will be stored. This way, the original
                # data of the trace to this `add_trace` method will not be altered.
                # We copy (by reference) all the non-data properties of the trace in
                # the new trace.
                trace = trace._props  # convert the trace into a dict
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
                # TODO: can be made more generic
                trace.x = dc.x
                trace.y = dc.y
                trace.text = dc.text
                trace.hovertext = dc.hovertext
                return super(AbstractFigureAggregator, self).add_traces(
                    [trace], **self._add_trace_to_add_traces_kwargs(trace_kwargs)
                )
        return super(AbstractFigureAggregator, self).add_traces(
            [trace], **self._add_trace_to_add_traces_kwargs(trace_kwargs)
        )

    def add_traces(
        self,
        data: List[BaseTraceType | dict] | BaseTraceType | Dict,
        max_n_samples: None | List[int] | int = None,
        downsamplers: None
        | List[AbstractSeriesAggregator]
        | AbstractFigureAggregator = None,
        limit_to_views: List[bool] | bool = False,
        **traces_kwargs,
    ):
        """Add traces to the figure.

        .. note::
            Make sure to look at the :func:`add_trace` function for more info about
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
        downsamplers : None | List[AbstractSeriesAggregator] | AbstractFigureAggregator, optional
            The downsampler that will be used to aggregate the traces. If a single
            aggregator is passed, all traces will use this aggregator.
            If this variable is not set, ``_global_downsampler`` will be used.
        limit_to_views : None | List[bool] | bool, optional
            List of limit_to_view booleans for the added traces.  If set to True
            the trace's datapoints will be cut to the corresponding front-end view,
            even if the total number of samples is lower than ``max_n_samples``. If a
            single boolean is passed, all to be added traces will use this value,
            by default False.\n
            Remark that setting this parameter to True ensures that low frequency traces
            are added to the ``hf_data`` property.
        **traces_kwargs: dict
            Additional trace related keyword arguments.
            e.g.: rows=.., cols=..., secondary_ys=...

            .. seealso::
                `Figure.add_traces <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.add_traces>`_ docs.

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
            self._data_validator.validate_coerce(trace)[0]
            if not isinstance(trace, BaseTraceType)
            else trace
            for trace in data
        ]

        # First add an UUID, as each (even the non-hf_data traces), must contain this
        # key for comparison. If the trace already has an UUID, we will keep it.
        for trace in data:
            uuid_str = str(uuid4()) if trace.uid is None else trace.uid
            trace.uid = uuid_str

        # Convert the data properties
        if isinstance(max_n_samples, (int, np.integer)) or max_n_samples is None:
            max_n_samples = [max_n_samples] * len(data)
        if isinstance(downsamplers, AbstractSeriesAggregator) or downsamplers is None:
            downsamplers = [downsamplers] * len(data)
        if isinstance(limit_to_views, bool):
            limit_to_views = [limit_to_views] * len(data)

        for i, (trace, max_out, downsampler, limit_to_view) in enumerate(
            zip(data, max_n_samples, downsamplers, limit_to_views)
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

        return super(self._figure_class, self).add_traces(data, **traces_kwargs)

    def _clear_figure(self):
        """Clear the current figure object it's data and layout."""
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
            resampled_trace_prefix_suffix=(self._prefix, self._suffix),
        )

    def construct_update_data(
        self,
        relayout_data: dict,
    ) -> Union[List[dict], dash.no_update]:
        """Construct the to-be-updated front-end data, based on the layout change.

        Attention
        ---------
        This method is tightly coupled with Dash app callbacks. It takes the front-end
        figure its ``relayoutData`` as input and returns the data which needs to be
        sent tot the ``TraceUpdater`` its ``updateData`` property for that corresponding
        graph.

        Parameters
        ----------
        relayout_data: dict
            A dict containing the ``relayout``-data (a.k.a. changed layout data) of
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
            self._print("-" * 100 + "\n", "changed layout", relayout_data)

            cl_k = relayout_data.keys()

            # ------------------ HF DATA aggregation ---------------------
            # 1. Base case - there is a x-range specified in the front-end
            start_matches = self._re_matches(re.compile(r"xaxis\d*.range\[0]"), cl_k)
            stop_matches = self._re_matches(re.compile(r"xaxis\d*.range\[1]"), cl_k)
            if len(start_matches) and len(stop_matches):
                for t_start_key, t_stop_key in zip(start_matches, stop_matches):
                    # Check if the xaxis<NUMB> part of xaxis<NUMB>.[0-1] matches
                    xaxis = t_start_key.split(".")[0]
                    assert xaxis == t_stop_key.split(".")[0]
                    # -> we want to copy the layout on the back-end
                    updated_trace_indices = self._check_update_figure_dict(
                        current_graph,
                        start=relayout_data[t_start_key],
                        stop=relayout_data[t_stop_key],
                        xaxis_filter=xaxis,
                        updated_trace_indices=updated_trace_indices,
                    )

            # 2. The user clicked on either autorange | reset axes
            autorange_matches = self._re_matches(
                re.compile(r"xaxis\d*.autorange"), cl_k
            )
            spike_matches = self._re_matches(re.compile(r"xaxis\d*.showspikes"), cl_k)
            # 2.1 Reset-axes -> autorange & reset to the global data view
            if len(autorange_matches) and len(spike_matches):
                for autorange_key in autorange_matches:
                    if relayout_data[autorange_key]:
                        xaxis = autorange_key.split(".")[0]
                        updated_trace_indices = self._check_update_figure_dict(
                            current_graph,
                            xaxis_filter=xaxis,
                            updated_trace_indices=updated_trace_indices,
                        )
            # 2.1. Autorange -> do nothing, the autorange will be applied on the
            #      current front-end view
            elif len(autorange_matches) and not len(spike_matches):
                # PreventUpdate returns a 204 status code response on the
                # relayout post request
                return dash.no_update

        # If we do not have any traces to be updated, we will return an empty
        # request response
        if len(updated_trace_indices) == 0:
            # PreventUpdate returns a 204 status-code response on the relayout post
            # request
            return dash.no_update

        # -------------------- construct callback data --------------------------
        layout_traces_list: List[dict] = []  # the data

        # 1. Create a new dict with additional layout updates for the front-end
        extra_layout_updates = {}

        # 1.1. Set autorange to False for each layout item with a specified x-range
        xy_matches = self._re_matches(re.compile(r"[xy]axis\d*.range\[\d+]"), cl_k)
        for range_change_axis in xy_matches:
            axis = range_change_axis.split(".")[0]
            extra_layout_updates[f"{axis}.autorange"] = None
        layout_traces_list.append(extra_layout_updates)

        # 2. Create the additional trace data for the frond-end
        relevant_keys = ["x", "y", "text", "hovertext", "name"]  # TODO - marker color
        # Note that only updated trace-data will be sent to the client
        for idx in updated_trace_indices:
            trace = current_graph["data"][idx]
            trace_reduced = {k: trace[k] for k in relevant_keys if k in trace}

            # Store the index into the corresponding to-be-sent trace-data so
            # the client front-end can know which trace needs to be updated
            trace_reduced.update({"index": idx})
            layout_traces_list.append(trace_reduced)
        return layout_traces_list

    @staticmethod
    def _parse_dtype_orjson(series: np.ndarray) -> np.ndarray:
        """Verify the orjson compatibility of the series and convert it if needed."""
        # NOTE:
        #    * float16 and float128 aren't supported with latest orjson versions (3.8.1)
        #    * this method assumes that the it will not get a float128 series
        # -> this method can be removed if orjson supports float16
        if series.dtype in [np.float16]:
            return series.astype(np.float32)
        return series

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

    ## Magic methods (to use plotly.py words :grin:)

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
        return (self.__class__, (props,))  # (props,) to comply with plotly magic
