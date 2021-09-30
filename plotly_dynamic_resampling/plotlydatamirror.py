# -*- coding: utf-8 -*-
"""
Wrapper around the plotly figure to allow bookkeeping and back-end based resampling of
HF data.
"""
__author__ = "Jonas Van Der Donckt"

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
import dash_core_components as dcc
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State
from uuid import uuid4


class PlotlyDataMirror(go.Figure):
    """Mirrors the figures' `data` attribute to allow resampling on the back-end."""

    def __init__(
            self,
            figure: go.Figure,
            global_n_shown_samples: int = 1500,
            verbose: bool = False
    ):
        """Instantiate a data mirror.

        Parameters
        ----------
        figure: go.Figure
            The figure that will be decorated.
        global_n_shown_samples: int
            The global set number of samples that will be shown for each trace.
        verbose: bool
            Whether some verbose messages will be printed or not

        """
        self._hf_data: List[dict] = []
        self._global_n_shown_samples = global_n_shown_samples
        self._print_verbose = verbose

        # downsampling method, optional
        self._downsampler = None

        super().__init__(figure)

    def _query_hf_data(self, trace: dict) -> Optional[dict]:
        """Query the internal `hf_data` attribute and returns a match.

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
        print(f"[W] trace with {trace_props} not found")
        return None

    def _get_hf_data_props(self) -> List[dict]:
        return [
            {k: hf[k] for k in set(hf.keys()).difference({'x', 'y'})}
            for hf in self._hf_data
        ]

    def check_update_trace_data(self, trace, t_start=None, t_stop=None):
        """Check and updates the passed`trace`.

        Note
        ----
        This is a pass by reference. The passed trace object will be updated.
        No new view of this trace will be created!

        Parameters
        ----------
        trace : BaseTraceType or dict
             - An instances of a trace classe from the plotly.graph_objs
                package (e.g plotly.graph_objs.Scatter, plotly.graph_objs.Bar)
              - or a dicts where:

                  - The 'type' property specifies the trace type (e.g.
                    'scatter', 'bar', 'area', etc.). If the dict has no 'type'
                    property then 'scatter' is assumed.
                  - All remaining properties are passed to the constructor
                    of the specified trace type.
        t_start : Optional[pd.Timestamp], optional
            The start time range for which we want resampled data to be updated to,
            by default None,
        t_stop : Optional[pd.Timestamp], optional
            The end time for which we want the resampled data to be updated to,
            by default None

        """
        hf_data = self._query_hf_data(trace)
        if hf_data is not None:
            df_res: pd.Series = self._resample_series(
                df_data=hf_data["df_hf"],
                max_n_samples=hf_data["max_n_samples"],
                t_start=t_start,
                t_stop=t_stop,
            )
            trace["x"] = df_res.index
            trace["y"] = df_res.values

    def check_update_figure_dict(
            self,
            figure: dict,
            t_start: Optional[pd.Timestamp] = None,
            t_stop: Optional[pd.Timestamp] = None,
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
        t_start : Optional[pd.Timestamp], optional
            The start time range for which we want resampled data to be updated to,
            by default None,
        t_stop : Optional[pd.Timestamp], optional
            The end time for which we want the resampled data to be updated to,
            by default None

        """
        for trace in figure["data"]:
            # print(trace.keys())
            self.check_update_trace_data(trace=trace, t_start=t_start, t_stop=t_stop)

    @staticmethod
    def _resample_series(
            df_data: pd.Series,
            max_n_samples: int = 6000,
            t_start: Optional[pd.Timestamp] = None,
            t_stop: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        ts_tzone = df_data.index.tz
        # todo -> maybe add ts_tzone

        if isinstance(t_start, pd.Timestamp):
            if ts_tzone is not None:
                if t_start.tz is not None:
                    assert t_start.tz.zone == ts_tzone.zone
                else:  # localize -> time remains the same
                    t_start = t_start.tz_localize(ts_tzone)
            elif ts_tzone is None and t_start.tz is not None:
                t_start = t_start.tz_localize(None)

            df_data = df_data[t_start:]
                        
        if isinstance(t_stop, pd.Timestamp):
            if ts_tzone is not None:
                if t_stop.tz is not None:
                    assert t_stop.tz.zone == ts_tzone.zone
                else:  # localize -> time remains the same
                    t_stop = t_stop.tz_localize(ts_tzone)
            elif ts_tzone is None and t_stop.tz is not None:
                t_stop = t_stop.tz_localize(None)

            df_data = df_data[:t_stop]

        # idx_series = df_data.index.to_series()
        # t_start, t_stop = idx_series.iloc[0], idx_series.iloc[-1]
        # timedelta_s = (t_stop - t_start).total_seconds()
        # return df_data.resample(f"{(1e3*timedelta_s)//(1.5*max_n_samples)}ms").mean()
        # todo -> check whether copy is necessary
        df_res = df_data[:: (max(1, len(df_data) // max_n_samples))].copy()

        tot_diff_sec_series = df_res.index.to_series().diff().dt.total_seconds()
        # diff = t_curr-t_prev (no diff for first item)

        # todo -> remove the max_gaps seconds param
        # todo -> do we need to shift this with -1 (so the last one of the ..)
        #   because we want to detect large gaps
        #       -> validate
        # if isinstance(max_gap_s, int):
        #     # if ((t_stop - t_start).total_seconds() / len(df_res)) < max_gap_s:
        #     if tot_diff_sec_series.median() < max_gap_s:
        #         df_res.loc[tot_diff_sec_series > max_gap_s] = None

        # use a quantile based approach
        max_gap_q_s = tot_diff_sec_series.quantile(0.95)

        # add None data-points in between the gaps
        df_res_gap = df_res.loc[tot_diff_sec_series > max_gap_q_s].copy()
        df_res_gap.loc[:] = None
        df_res_gap.index -= pd.Timedelta(microseconds=1)
        index_name = df_res.index.name
        df_res = pd.concat(
            [df_res.reset_index(drop=False), df_res_gap.reset_index(drop=False)]
        ).set_index(index_name).sort_index()
        return df_res['data']

    def add_trace(
            self,
            trace,
            row=None,
            col=None,
            secondary_y=None,
            cut_points_to_view=False,
            orig_x=None,  # Use this if you have high-dimensional data
            orig_y=None,  # Use this if you have high-dimensional data
            # resample_method="",
            max_n_samples=None,
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

        row : 'all', int or None (default)
            Subplot row index (starting from 1) for the trace to be
            added. Only valid if figure was created using
            `plotly.tools.make_subplots`.
            If 'all', addresses all rows in the specified column(s).
        col : 'all', int or None (default)
            Subplot col index (starting from 1) for the trace to be
            added. Only valid if figure was created using
            `plotly.tools.make_subplots`.
            If 'all', addresses all columns in the specified row(s).
        secondary_y: boolean or None (default None)
            If True, associate this trace with the secondary y-axis of the
            subplot at the specified row and col. Only valid if all of the
            following conditions are satisfied:
              * The figure was created using `plotly.subplots.make_subplots`.
              * The row and col arguments are not None
              * The subplot at the specified row and col has type xy
                (which is the default) and secondary_y True.  These
                properties are specified in the specs argument to
                make_subplots. See the make_subplots docstring for more info.
              * The trace argument is a 2D cartesian trace
                (scatter, bar, etc.)
        cut_points_to_view: boolean
            If True; the trace's datapoints will be add
        orig_x: pd.Series, optional
            The original high frequency time-index. If set, this has priority over the
            trace's data.
        orig_y: pd.Series, optional
            The original high frequency values. If set, this has priority over the
            trace's data.
        max_n_samples : int, optional
            The maximum number of samples that will be shown by the trace.
            .. note::
                If this variable is not set; `_global_n_shown_samples` will be used.

        Returns
        -------
        BaseFigure
            The Figure that add_trace was called on

        """
        if max_n_samples is None:
            max_n_samples = self._global_n_shown_samples

        def super_add_trace(self):
            return super().add_trace(
                trace=trace,
                row=row,
                col=col,
                secondary_y=secondary_y,
            )

        high_dimensional_traces = ["scatter", "scattergl"]

        # first add the trace, as each (even the non hf data traces), must contain this
        # key for comparison
        trace.uid = str(uuid4())

        if trace["type"] in high_dimensional_traces:
            orig_x = trace["x"] if orig_x is None else orig_x
            orig_y = trace["y"] if orig_y is None else orig_y

            assert len(orig_x) > 0
            assert len(orig_x) == len(orig_y)

            numb_samples = len(orig_x)
            # these traces will determine the autoscale -> so also store when
            # `cut_points_to_view` is set.
            if numb_samples > max_n_samples or cut_points_to_view:
                if self._print_verbose:
                    print(f"[i] resample {trace['name']} - {numb_samples}->{max_n_samples}")

                # we will re-create this each time as df_hf wittholds
                df_hf = pd.Series(data=orig_y, index=pd.to_datetime(orig_x), copy=False)
                df_hf.rename('data', inplace=True)
                df_hf.index.rename('timestamp', inplace=True)
                
                # Checking this now avoids less interpretable `KeyError` when resampling
                assert df_hf.index.is_monotonic_increasing

                if self._downsampler is None:
                    df_res = self._resample_series(
                        df_hf, max_n_samples=max_n_samples
                    )
                    trace.x = df_res.index
                    trace.y = df_res.values
                else:
                    raise NotImplementedError("Resampling method isn't supported!")

                super_add_trace(self)
                self._hf_data.append(
                    {
                        "max_n_samples": max_n_samples,
                        "df_hf": df_hf,
                        "uid": trace.uid

                        # TODO -> add support for trace-based resampling methods
                        # "resample_method": "#resample_method,
                    }
                )
            else:
                if self._print_verbose:
                    print(
                        f"[i] NOT resampling {trace['name']} - {numb_samples} samples")
                trace.x = orig_x
                trace.y = orig_y
                return super_add_trace(self)
        else:
            if self._print_verbose:
                print(f"trace {trace['type']} is not a high-dimensional trace")

            # orig_x and orig_y have priority over the traces' data
            trace["x"] = trace["x"] if orig_x is not None else orig_x
            trace["y"] = trace["y"] if orig_y is not None else orig_y
            assert len(trace["x"]) > 0
            assert len(trace["x"] == len(trace["y"]))
            return super_add_trace(self)

    def show_dash(self, mode=None, **kwargs):
        app = JupyterDash("local_app")
        app.layout = dbc.Container(dcc.Graph(id="resampled-graph", figure=self))

        def to_datetime(time_str: str) -> pd.Timestamp:
            # TODO WOWAWIEAAA what happens here -> what happens when series
            #  timezone is UTC???
            # uses self -> so we need to be able to gather the time-zone if there is any
            # and maybe not convert it to datetime if int-columns are used
            # todo -> also need to be able to perform zooming on selected sub-trace if
            #  uses subplots without shared x-axes
            return pd.to_datetime([time_str], infer_datetime_format=True).tz_localize(
                'Europe/Brussels'
            )[0]

        @app.callback(
            Output("resampled-graph", "figure"),
            Input("resampled-graph", "relayoutData"),
            State("resampled-graph", "figure")
        )
        def update_graph(changed_layout: dict, current_graph):
            if changed_layout:
                if self._print_verbose:
                    print("-" * 100, "\n", "changed layout", changed_layout)

                if "xaxis.range[0]" in changed_layout:
                    t_start = to_datetime(changed_layout["xaxis.range[0]"])
                    t_stop = to_datetime(changed_layout["xaxis.range[1]"])
                    self.check_update_figure_dict(current_graph, t_start, t_stop)
                elif changed_layout.get("xaxis.autorange", False):
                    self.check_update_figure_dict(current_graph)
            return current_graph

        app.run_server(mode=mode, **kwargs)
