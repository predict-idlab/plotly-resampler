# -*- coding: utf-8 -*-
"""
Wrapper around the the plotly go.Figure class which allows bookkeeping and
back-end based resampling of high-frequency sequential data.

Notes
-----
* The term `high-frequency` actually refers very large amounts of data, see also<br>
  https://www.sciencedirect.com/topics/social-sciences/high-frequency-data

Future work
------------
* Add functionality to let the user define a downsampling method
    -> would use a class based approach here!
    -> better separation of resampling & adding none to gaps & slicing


"""
__author__ = "Jonas Van Der Donckt, Emiel Deprost"

import re
from typing import List, Optional, Union, Iterable, Tuple, Dict
from uuid import uuid4

import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import dcc
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash

from .downsamplers import AbstractSeriesDownsampler, EveryNthPoint


class FigureResampler(go.Figure):
    """Mirrors the go.Figure's `data` attribute to allow resampling in the back-end."""

    def __init__(
        self,
        figure: go.Figure = go.Figure(),
        default_n_shown_samples: int = 1000,
        default_downsampler: AbstractSeriesDownsampler = EveryNthPoint(),
        resampled_trace_prefix_suffix: Tuple[str, str] = (
            '<b style="color:sandybrown">[R]</b> ',
            "",
        ),
        verbose: bool = False,
    ):
        """Instantiate a resampling data mirror.

        Parameters
        ----------
        figure: go.Figure
            The figure that will be decorated.
        default_n_shown_samples: int, optional
            The global set number of samples that will be shown for each trace,
            by default 1000.
        default_downsampler: AbstractSeriesDownsampler
            A instance which implements the AbstractSeriesDownsampler interface. This will
            be used as default downsampler.<br>
            Note, this can be overriden with the `add_trace()` method.
            by default `EveryNthPoint`
        resampled_trace_prefix_suffix: str, optional
            A tuple which contains the `prefix` and `suffix`, respectively, which
            will be added to the trace its name when a resampled version of the trace
            is shown.
        verbose: bool, optional
            Whether some verbose messages will be printed or not, by default False

        """
        self._hf_data: Dict[str, dict] = {}
        self._global_n_shown_samples = default_n_shown_samples
        self._print_verbose = verbose
        assert len(resampled_trace_prefix_suffix) == 2
        self._prefix, self._suffix = resampled_trace_prefix_suffix

        self._global_downsampler = default_downsampler

        super().__init__(figure)

    def _query_hf_data(self, trace: dict) -> Optional[dict]:
        """Query the internal `hf_data` attribute and returns a match based on `uid`.

        Parameters
        ----------
        trace : dict
            The trace where we want to find a match for.

        Returns
        -------
        Optional[dict]
            The `hf_data`-trace dict if a match is found, else `None`.

        """
        trace_data = self._hf_data.get(trace["uid"])
        if trace_data is None:
            trace_props = {
                k: trace[k] for k in set(trace.keys()).difference({"x", "y"})
            }
            self._print(f"[W] trace with {trace_props} not found")
        return trace_data

    def _print(self, *values):
        """Helper method for printing if `verbose` is set to True"""
        if self._print_verbose:
            print(*values)

    def check_update_trace_data(self, trace, start=None, end=None):
        """Check and updates the passed`trace`.

        Note
        ----
        This is a pass by reference. The passed trace object will be updated.
        No new view of this trace will be created!

        Parameters
        ----------
        trace : BaseTraceType or dict
             - An instances of a trace class from the plotly.graph_objs
                package (e.g plotly.graph_objs.Scatter, plotly.graph_objs.Bar)
              - or a dicts where:

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

        Notes
        -----
        * If `start` and `stop` are strings, the most likely represents a time-strings
        * `start` and `stop` will always be of the same type (float / time-string) because the
           underlying axis is the same.

        """
        hf_data = self._query_hf_data(trace)
        if hf_data is not None:
            axis_type = hf_data["axis_type"]
            if axis_type == "date":
                hf_series = self._slice_time(
                    hf_data["hf_series"], pd.to_datetime(start), pd.to_datetime(end)
                )
            else:
                hf_series: pd.Series = hf_data["hf_series"]
                if isinstance(hf_series.index, pd.Int64Index) or isinstance(
                    hf_series.index, pd.UInt64Index
                ):
                    start = round(start) if start is not None else None
                    end = round(end) if start is not None else None

                hf_series = hf_series[start:end]

            # Add a prefix when the original data is downsampled
            name: str = trace["name"]
            if len(hf_series) > hf_data["max_n_samples"]:
                name = (
                    "" if name.startswith(self._prefix) else self._prefix
                ) + name
                name += self._suffix if not name.endswith(self._suffix) else ""
                trace["name"] = name
            else:
                if len(self._prefix) and name.startswith(self._prefix):
                    trace["name"] = name[len(self._prefix) :]
                if len(self._suffix) and name.endswith(self._suffix):
                    trace["name"] = name[: -len(self._suffix)]

            downsampler: AbstractSeriesDownsampler = hf_data["downsampler"]
            s_res: pd.Series = downsampler.downsample(
                hf_series, hf_data["max_n_samples"]
            )
            trace["x"] = s_res.index
            trace["y"] = s_res.values
        else:
            self._print("hf_data not found")

    def check_update_figure_dict(
        self,
        figure: dict,
        start: Optional[Union[float, str]] = None,
        stop: Optional[Union[float, str]] = None,
        xaxis: str = None,
    ):
        """Check and update the traces within the figure dict.

        This method will most likely be used within a `Dash` callback to resample the
        view, based on the configured number of parameters.

        Note
        ----
        This is a pass by reference. The passed trace object will be updated.
        No new view of this trace will be created!

        Parameters
        ----------
        figure : dict
            The figure dict
        start : Union[float, str], optional
            The start time range for which we want resampled data to be updated to,
            by default None,
        stop : Union[float, str], optional
            The end time for which we want the resampled data to be updated to,
            by default None
        xaxis: str, Optional
            Additional trace-update filter

        """
        for trace in figure["data"]:
            if xaxis is not None:
                # the x-anchor of the trace is stored in the layout data
                if trace.get('yaxis') is None:
                    # no yaxis -> we make the assumption that yaxis = xaxis
                    y_axis = 'y' + xaxis[1:]
                else:
                    y_axis = 'yaxis' + trace.get('yaxis')[1:]
                x_anchor = figure["layout"][y_axis].get("anchor")
                # we skip when:
                # * the change was made on the first row and the trace its anchor is not
                #   in [None, 'x']
                #   -> why None: traces without row/col argument and stand on first row
                #      and do not have the anchor property (hence the DICT.get() method)
                # * x-anchor != trace['xaxis'] for NON first rows
                if (
                    x_anchor == "x" and trace.get("xaxis", None) not in [None, "x"]
                ) or (x_anchor != "x" and trace.get("xaxis", None) != x_anchor):
                    continue

            self.check_update_trace_data(trace=trace, start=start, end=stop)

    @staticmethod
    def _slice_time(
        hf_series: pd.Series,
        t_start: Optional[pd.Timestamp] = None,
        t_stop: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        def to_same_tz(
            ts: Union[pd.Timestamp, None], reference_tz=hf_series.index.tz
        ) -> Union[pd.Timestamp, None]:
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

        return hf_series[to_same_tz(t_start):to_same_tz(t_stop)]

    def add_trace(
        self,
        trace,
        # Use this if you have high-frequency data
        orig_x: Iterable = None,
        orig_y: Iterable = None,
        max_n_samples: int = None,
        downsampler: AbstractSeriesDownsampler = None,
        limit_to_view: bool = True,
        **trace_kwargs,
    ):
        """Add a trace to the figure.

        Note
        ----
        As constructing traces with high frequency data really takes a
        long time -> it is preferred to just create an empty trace and pass the
        high frequency to this method, using the `orig_x` and `orig_y` parameters.
        >>> from plotly.subplots import make_subplots
        >>> d = pd.DataFrame()  # a high-frequency dataframe
        >>> fig = FigureResampler())
        >>> fig.add_trace(go.Scattergl(x=[], y=[], ...), orig_x=d.index, orig_y=.d['c'])

        Notes
        -----
        * **Pro tip**: if you do `not want to downsample` your data, set `max_n_samples` to the
            size of your trace!
        * Sparse time-series data (e.g., a scatter of detected peaks), can hinder the
          the automatic-zoom functionality; as these will not be stored in the
          back-end data-mirror and thus not be (re)sampled to the view.<br>
          To circumvent this, the `cut_points_to_view` argument can be set, which forces
          these sparse data-series to be also stored in the database.
        * `orig_x` and `orig_y` have priority over the trace's data.

        Parameters
        ----------
        trace : BaseTraceType or dict
            Either:
              - An instances of a trace classe from the plotly.graph_objs
                package (e.g plotly.graph_objs.Scatter, plotly.graph_objs.Bar)
              - or a dicts where:

                  - The 'type' property specifies the trace type (e.g.
                    'scatter', 'bar', 'area', etc.). If the dict has no 'type'
                    property then 'scatter' is assumed.
                  - All remaining properties are passed to the constructor
                    of the specified trace type.
        orig_x: pd.Series, optional
            The original high frequency series position, can be either a time-series or an
            increasing, numerical index. If set, this has priority over the trace its
            data.
        orig_y: pd.Series, optional
            The original high frequency values. If set, this has priority over the
            trace's data.
        max_n_samples : int, optional
            The maximum number of samples that will be shown by the trace.\n
            .. note::
                If this variable is not set; `_global_n_shown_samples` will be used.
        downsampler: AbstractSeriesDownsampler, optional
            The abstract series downsampler method
        limit_to_view: boolean, optional
            If set to True and the trace it's format is a high-frequency trace type,
            then the trace's datapoints will be cut to the corresponding front-end view,
            even if the total number of samples is lower than `max_n_samples`.
        **trace_kwargs:
            Additional trace related keyword arguments
            e.g.: row=.., col=..., secondary_y=...,
            see trace_docs: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.add_traces

        Returns
        -------
        BaseFigure
            The Figure that add_trace was called on

        """
        if max_n_samples is None:
            max_n_samples = self._global_n_shown_samples

        # first add the trace, as each (even the non hf data traces), must contain this
        # key for comparison
        trace.uid = str(uuid4())

        high_frequency_traces = ["scatter", "scattergl"]
        if trace["type"].lower() in high_frequency_traces:

            orig_x = (
                trace["x"]
                if orig_x is None
                else orig_x.values
                if isinstance(orig_x, pd.Series)
                else orig_x
            )
            orig_y = (
                trace["y"]
                if orig_y is None
                else orig_y.values
                if isinstance(orig_y, pd.Series)
                else orig_y
            )

            # Remove NaNs for efficiency (storing less meaningless data)
            # NaNs introduce gaps between enclosing non-NaN datapoints & might distort
            # the resampling algorithms
            try:
                orig_x = orig_x[~np.isnan(orig_y)]
                orig_y = orig_y[~np.isnan(orig_y)]
            except:
                pass

            assert len(orig_x) > 0
            assert len(orig_x) == len(orig_y)

            numb_samples = len(orig_x)
            # These traces will determine the autoscale RANGE!
            #   -> so also store when limit_to_view` is set.
            if numb_samples > max_n_samples or limit_to_view:
                self._print(
                    f"[i] resample {trace['name']} - {numb_samples}->{max_n_samples}"
                )

                # we will re-create this each time as orig_y and orig_x withholds
                # high-frequency data
                hf_series = pd.Series(data=orig_y, index=orig_x, copy=False)
                hf_series.rename("data", inplace=True)
                hf_series.index.rename("timestamp", inplace=True)

                # Checking this now avoids less interpretable `KeyError` when resampling
                assert hf_series.index.is_monotonic_increasing

                # As we support prefix-suffixing of downsampled data, we assure that
                # each trace has a name
                # https://github.com/plotly/plotly.py/blob/ce0ed07d872c487698bde9d52e1f1aadf17aa65f/packages/python/plotly/plotly/basedatatypes.py#L539
                # The link above indicates that the trace index is derived from `data`
                if trace.name is None:
                    trace.name = f"trace {len(self.data)}"

                # determine (1) the axis type and (2) the downsampler instance
                # & (3) add store a  hf_data entry for the corresponding trace,
                # identified by its UUID
                axis_type = "date" if isinstance(orig_x, pd.DatetimeIndex) else "linear"
                d = self._global_downsampler if downsampler is None else downsampler
                self._hf_data[trace.uid] = {
                    "max_n_samples": max_n_samples,
                    "hf_series": hf_series,
                    "axis_type": axis_type,
                    "downsampler": d,
                }

                # NOTE: if all the raw data need sent to the javascript, and the trace
                #  is truly high-frequency, this would take a lot of time!
                #  hence, you first downsample the trace.
                self.check_update_trace_data(trace)
                super().add_trace(trace=trace, **trace_kwargs)
            else:
                self._print(f"[i] NOT resampling {trace['name']} - len={numb_samples}")
                trace.x = orig_x
                trace.y = orig_y
                return super().add_trace(trace=trace, **trace_kwargs)
        else:
            self._print(f"trace {trace['type']} is not a high-frequency trace")

            # orig_x and orig_y have priority over the traces' data
            trace["x"] = trace["x"] if orig_x is not None else orig_x
            trace["y"] = trace["y"] if orig_y is not None else orig_y
            assert len(trace["x"]) > 0
            assert len(trace["x"] == len(trace["y"]))
            return super().add_trace(trace=trace, **trace_kwargs)

    def show_dash(self, mode=None, **kwargs):
        app = JupyterDash("local_app")
        app.layout = dbc.Container(dcc.Graph(id="resampled-graph", figure=self))

        @app.callback(
            Output("resampled-graph", "figure"),
            Input("resampled-graph", "relayoutData"),
            State("resampled-graph", "figure"),
        )
        def update_graph(changed_layout: dict, current_graph):
            if changed_layout:
                self._print("-" * 100 + "\n", "changed layout", changed_layout)

                # determine the start and end regex matches
                def get_matches(regex: re.Pattern, strings: Iterable[str]) -> List[str]:
                    matches = []
                    for item in strings:
                        m = regex.match(item)
                        if m is not None:
                            matches.append(m.string)
                    return sorted(matches)

                key_list = changed_layout.keys()
                start_matches = get_matches(re.compile(r"xaxis\d*.range\[0]"), key_list)
                stop_matches = get_matches(re.compile(r"xaxis\d*.range\[1]"), key_list)

                if len(start_matches) and len(stop_matches):
                    for t_start_key, t_stop_key in zip(start_matches, stop_matches):
                        # check if the xaxis<NUMB> part of xaxis<NUMB>.[0-1] matches
                        assert t_start_key.split(".")[0] == t_stop_key.split(".")[0]

                        self.check_update_figure_dict(
                            current_graph,
                            start=changed_layout[t_start_key],
                            stop=changed_layout[t_stop_key],
                            xaxis=t_start_key.split(".")[0],
                        )

                elif len(get_matches(re.compile(r"xaxis\d*.autorange"), key_list)):
                    # Autorange is applied on all axes -> no xaxis argument
                    self.check_update_figure_dict(current_graph)
            return current_graph

        additional_kwargs = {}
        if self.layout.height is not None and mode == "inline":
            additional_kwargs = {"height": self.layout.height + 18}
        app.run_server(mode=mode, **kwargs, **additional_kwargs)
