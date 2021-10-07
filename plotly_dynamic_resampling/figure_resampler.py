# -*- coding: utf-8 -*-
"""
Wrapper around the the plotly go.Figure class which allows bookkeeping and
back-end based resampling of high-dimensional sequential data.

Future work
------------
* Add functionality to let the user define a downsampling method
    -> would use a class based approach here!
    -> better separation of resampling & adding none to gaps & slicing


"""
__author__ = "Jonas Van Der Donckt, Emiel Deprost"

import re
from typing import List, Optional, Union, Iterable, Tuple
from uuid import uuid4

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash


class FigureResampler(go.Figure):
    """Mirrors the go.Figure's `data` attribute to allow resampling in the back-end.
    """

    def __init__(
            self,
            figure: go.Figure,
            global_n_shown_samples: int = 1000,
            resampled_trace_prefix_suffix: Tuple[str, str] = (
                    '<b style="color:sandybrown">[R]</b> ', ''),
            verbose: bool = False
    ):
        """Instantiate a resampling data mirror.

        Parameters
        ----------
        figure: go.Figure
            The figure that will be decorated.
        global_n_shown_samples: int, optional
            The global set number of samples that will be shown for each trace,
            by default 1000.
        resampled_trace_prefix_suffix: str, optional
            A tuple which contains the `prefix` and `suffix`, respectively, which
            will be added to the trace its name when a resampled version of the trace
            is shown.
        verbose: bool, optional
            Whether some verbose messages will be printed or not, by default False

        """
        self._hf_data: List[dict] = []
        self._global_n_shown_samples = global_n_shown_samples
        self._print_verbose = verbose
        assert len(resampled_trace_prefix_suffix) == 2
        self._prefix, self._suffix = resampled_trace_prefix_suffix

        self._downsampler = None  # downsampling method, still to be implemented

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
        for trace_data in self._hf_data:
            if trace_data['uid'] == trace['uid']:
                return trace_data

        trace_props = {
            k: trace[k]
            for k in set(trace.keys()).difference({'x', 'y'})
        }
        self._print(f"[W] trace with {trace_props} not found")
        return None

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
            axis_type = hf_data['axis_type']
            if axis_type == 'date':
                df_data = self._slice_time(
                    hf_data['df_hf'],
                    pd.to_datetime(start),
                    pd.to_datetime(end)
                )
            else:
                df_data = hf_data['df_hf']
                if isinstance(df_data.index, pd.Int64Index) or \
                        isinstance(df_data.index, pd.UInt64Index):
                    start = round(start) if start is not None else None
                    end = round(end) if start is not None else None
                df_data = hf_data['df_hf'][start:end]

            # add a prefix when not all data is shown
            if trace['name'] is not None:
                name: str = trace['name']
                if len(df_data) > hf_data["max_n_samples"]:
                    name = ('' if name.startswith(
                        self._prefix) else self._prefix) + name
                    name += self._suffix if not name.endswith(self._suffix) else ''
                    trace['name'] = name
                else:
                    if len(self._prefix) and name.startswith(self._prefix):
                        trace['name'] = name[len(self._prefix):]
                    if len(self._suffix) and name.endswith(self._suffix):
                        trace['name'] = name[:-len(self._suffix)]

            # TODO -> support for other resample methods
            df_res: pd.Series = self._resample_series(df_data, hf_data["max_n_samples"])
            trace["x"] = df_res.index
            trace["y"] = df_res.values
        else:
            self._print('hf_data not found')

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
                # the x-anchor of is stored in the layout data
                y_axis = 'y' + xaxis[1:]
                x_anchor = figure['layout'][y_axis].get('anchor')
                # we skip when:
                # * the change was made on the first row and the trace its anchor is not
                #   in [None, 'x']
                #   -> why None: traces without row/col argument and stand on first row
                #      and do not have the anchor property (hence the DICT.get() method)
                # * x-anchor != trace['xaxis'] for NON first rows
                if ((x_anchor == 'x' and trace.get("xaxis", None) not in [None, 'x']) or
                        (x_anchor != 'x' and trace.get('xaxis', None) != x_anchor)):
                    continue

            self.check_update_trace_data(trace=trace, start=start, end=stop)

    @staticmethod
    def _slice_time(
            df_data: pd.Series,
            t_start: Optional[pd.Timestamp] = None,
            t_stop: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        def to_same_tz(
                ts: Union[pd.Timestamp, None],
                reference_tz=df_data.index.tz
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

        return df_data[to_same_tz(t_start):to_same_tz(t_stop)]

    @staticmethod
    def _resample_series(
            df_data: pd.Series,
            max_n_samples,
    ) -> pd.Series:
        # just use plain simple slicing
        df_res = df_data[:: (max(1, len(df_data) // max_n_samples))]

        # ------- add None where there are gaps / irregularly sampled data
        if isinstance(df_res.index, pd.DatetimeIndex):
            series_index_diff = df_res.index.to_series().diff().dt.total_seconds()
        else:
            series_index_diff = df_res.index.to_series().diff()

        # use a quantile based approach
        min_diff, max_gap_q_s = series_index_diff.quantile(q=[0, 0.95])

        # add None data-points in between the gaps
        df_res_gap = df_res.loc[series_index_diff > max_gap_q_s].copy()
        df_res_gap.loc[:] = None
        if isinstance(df_res.index, pd.DatetimeIndex):
            df_res_gap.index -= pd.Timedelta(seconds=min_diff / 2)
        else:
            df_res_gap.index -= (min_diff / 2)
        index_name = df_res.index.name
        df_res = pd.concat(
            [df_res.reset_index(drop=False), df_res_gap.reset_index(drop=False)]
        ).set_index(index_name).sort_index()
        return df_res['data']

    def add_trace(
            self,
            trace,
            # Use this if you have high-dimensional data
            orig_x: Iterable = None,
            orig_y: Iterable = None,
            max_n_samples: int = None,
            cut_points_to_view: bool = False,
            **trace_kwargs
    ):
        """Add a trace to the figure.

        Note
        ----
        As constructing traces with high dimensional data really takes a
        long time -> it is preferred to just create an empty trace and pass the
        high dimensional to this method, using the `orig_x` and `orig_y` parameters.
        >>> from plotly.subplots import make_subplots
        >>> df = pd.DataFrame()  # a high-dimensional dataframe
        >>> fig = PlotlyDataMirror(make_subplots(...))
        >>> fig.add_trace(go.Scattergl(x=[], y=[], ...), orig_x=df.index, orig_y=.df['c'])

        Note
        ----
        Sparse time-series data (e.g., a scatter of detected peaks), can hinder the
        the automatic-zoom functionality; as these will not be stored in the data-mirror
        and thus not be (re)sampled to the view. To circumvent this, the
        `cut_points_to_view` argument can be set, which forces these sparse data-series
        to be also stored in the database.

        Note
        ----
        `orig_x` and `orig_y` have priority over the trace's data.

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
        cut_points_to_view: boolean, optional
            If set to True and the trace it's format is a high-dimensional trace type,
            then the trace's datapoints will be cut to the corresponding front-end view,
            even if the total number of samples is lower than the MAX amount of samples.
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

        high_dimensional_traces = ["scatter", "scattergl"]
        if trace["type"].lower() in high_dimensional_traces:
            orig_x = trace["x"] if orig_x is None else orig_x
            orig_y = trace["y"] if orig_y is None else orig_y

            assert len(orig_x) > 0
            assert len(orig_x) == len(orig_y)

            numb_samples = len(orig_x)
            # These traces will determine the autoscale
            #   -> so also store when cut_points_to_view` is set.
            if numb_samples > max_n_samples or cut_points_to_view:
                self._print(
                    f"[i] resample {trace['name']} - {numb_samples}->{max_n_samples}")

                # we will re-create this each time as df_hf withholds
                df_hf = pd.Series(data=orig_y, index=orig_x, copy=False)
                df_hf.rename('data', inplace=True)
                df_hf.index.rename('timestamp', inplace=True)

                # Checking this now avoids less interpretable `KeyError` when resampling
                assert df_hf.index.is_monotonic_increasing

                axis_type = "date" if isinstance(orig_x, pd.DatetimeIndex) else "linear"
                self._hf_data.append(
                    {
                        "max_n_samples": max_n_samples,
                        "df_hf": df_hf,
                        "uid": trace.uid,
                        'axis_type': axis_type
                        # "resample_method": "#resample_method,
                    }
                )
                # first resample the high-dim trace b4 converting it into javascript
                self.check_update_trace_data(trace)
                super().add_trace(trace=trace, **trace_kwargs)
            else:
                self._print(
                    f"[i] NOT resampling {trace['name']} - {numb_samples} samples")
                trace.x = orig_x
                trace.y = orig_y
                return super().add_trace(trace=trace, **trace_kwargs)
        else:
            self._print(f"trace {trace['type']} is not a high-dimensional trace")

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
            State("resampled-graph", "figure")
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
                start_matches = get_matches(re.compile(r'xaxis\d*.range\[0]'), key_list)
                stop_matches = get_matches(re.compile(r'xaxis\d*.range\[1]'), key_list)

                if len(start_matches) and len(stop_matches):
                    for t_start_key, t_stop_key in zip(start_matches, stop_matches):
                        # check if the xaxis<NUMB> part of xaxis<NUMB>.[0-1] matches
                        assert t_start_key.split('.')[0] == t_stop_key.split('.')[0]

                        self.check_update_figure_dict(
                            current_graph,
                            start=changed_layout[t_start_key],
                            stop=changed_layout[t_stop_key],
                            xaxis=t_start_key.split('.')[0]
                        )

                elif len(get_matches(re.compile(r'xaxis\d*.autorange'), key_list)):
                    # Autorange is applied on all axes -> no xaxis argument
                    self.check_update_figure_dict(current_graph)
            return current_graph

        app.run_server(mode=mode, **kwargs)
