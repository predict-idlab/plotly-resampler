"""Minimal auth dashboard
Dash-board

inspiration drawn of:
* https://shiny.rstudio.com/gallery/kmeans-example.html



Note
----
Drawback -> no support for multiple sessions.
    -> partially solved in future dashboard iterations
"""


import base64
import io
import ast
import os
import dash
import datetime
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.FormGroup import FormGroup
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.Div import Div
import dash_uploader as du
import pandas as pd
import plotly.graph_objects as go

from typing import Optional, Dict, List, Union, Tuple
from plotlydatamirror import PlotlyDataMirror
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
from plotly.subplots import make_subplots
from pathlib import Path


class Dashboard:
    def __init__(self) -> None:
        # global figure object
        self.fig: Optional[PlotlyDataMirror] = None

        self.app = JupyterDash(
            __name__, external_stylesheets=[dbc.themes.LUX]  # , server=server
        )  # themes.lux

        self.TIMEZONE = "Europe/Brussels"
        self.CACHE_FOLDER = Path(".cache_datasets")

        self.dataset_dict: Dict[str, pd.DataFrame] = {}

        self.df_traces = pd.DataFrame(
            {
                "trace_key": [],
                "dataset_name": [],
                "column": [],
                "subplot_row": [],
            }
        ).set_index("trace_key")

        du.configure_upload(self.app, self.CACHE_FOLDER, use_upload_id=False)

        self._define_layout()
        self._callbacks()

    def _define_layout(self):
        controls = dbc.Card(
            [
                html.Div(
                    [
                        du.Upload(
                            id="load-dataset",
                            text="Upload a dataset here!",
                            cancel_button=True,
                            filetypes=["parquet"],
                            max_files=1,  # Is buggy, so I set to 1
                            upload_id=None,
                        ),
                        html.Div(id="load-dataset-list"),
                    ]
                ),
                dcc.Upload(
                    dbc.Button("Load Config"),
                    id="load-config",
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("DF selector"),
                        dcc.Dropdown(
                            id="dataset-selector",
                            options=[
                                {"label": k, "value": k}
                                for k in self.dataset_dict.keys()
                            ],
                            multi=False,
                        ),
                        dbc.Label("Columns:"),
                        dcc.Dropdown(
                            id="column-selector",
                            options=[],
                            multi=True,
                        ),
                        dbc.Label("Subplot:"),
                        dcc.RadioItems(
                            id="subplot-selector",
                            options=[{"label": "New subplot", "value": 0}],
                            value=0,
                            labelStyle={"display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id="y-axis-selector",
                            options=[
                                # {"label": "Left Y-axis", "value": "left"},
                                {"label": "Right Y-axis", "value": "secondary"},
                            ],
                            labelStyle={"display": "inline-block"},
                        ),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Button("Add trace(s)", id="add-trace", color="primary"),
                    ],
                    style={"textAlign": "center"},
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Current traces:"),
                        dcc.Dropdown(
                            id="traces-list",
                            options=[],
                            value=[],
                            multi=True,
                        ),
                    ],
                ),
            ]
        )

        self.app.layout = dbc.Container(
            [
                html.H1("ITSDAR", style={"textAlign": "center"}),
                html.Hr(),
                dcc.ConfirmDialog(id="confirm"),
                dbc.Row(
                    [
                        dbc.Col(controls, md=3),
                        dbc.Col(
                            dcc.Graph(
                                id="resampled-graph",
                                figure=PlotlyDataMirror(go.Figure()),
                            ),
                            md=9,
                        ),
                    ],
                    align="center",
                ),
                dbc.Button("Download config", id="btn_config"),
                dcc.Download(id="download-config"),
            ],
            fluid=True,
        )

    # --------------------------------- Visualization ---------------------------------
    def create_figure(
        self, trace_keys: List[str], number_of_samples: Optional[int] = 2000
    ) -> Tuple[go.Figure, int]:
        """
        Create the `PlotlyDataMirror` figure.

        Parameters
        ----------
        df_data : pd.DataFrame
            The dataframe with the actual data.
        col_names : List[str]
            The column names, each column name will be plotted in a new row.
        number_of_samples : int
            The number of samples per trace, more samples will be slower.

        Returns
        -------
        go.Figure
            Returns a plotly figure
        """
        keys_to_drop = set(self.df_traces.index).difference(trace_keys)
        self.df_traces.drop(keys_to_drop, axis=0, inplace=True)

        if self.df_traces.empty:
            return PlotlyDataMirror(go.Figure()), 0

        # 0's mean add it to new row
        self.df_traces["subplot_row"].replace(
            0, int(self.df_traces.subplot_row.max() + 1), inplace=True
        )
        nb_rows = int(self.df_traces.subplot_row.max())

        kwargs = {"vertical_spacing": 0.15 / (nb_rows)}

        subplot_titles = ["" for _ in range(nb_rows)]
        for _, params in self.df_traces.iterrows():
            subplot_titles[int(params.subplot_row) - 1] += str(params.column)

        self.fig = PlotlyDataMirror(
            make_subplots(
                rows=nb_rows,
                cols=1,
                shared_xaxes=True,
                subplot_titles=subplot_titles,
                specs=[[{"secondary_y": True}] for _ in range(nb_rows)],
                **kwargs,
            )
        )

        self.fig.update_layout(
            height=min(1000, 400 * (nb_rows)),
            title_x=0.5,
            legend=dict(
                orientation="h",
            ),
        )

        for key, params in self.df_traces.iterrows():
            df_data = self.dataset_dict[params["dataset_name"]]
            col = params["column"]
            notna_mask = df_data[col].notna().values
            self.fig.add_trace(
                trace=go.Scattergl(
                    x=[],
                    y=[],
                    name=str(key),
                    # legendgroup=int(params.subplot_row)
                ),
                row=int(params.subplot_row),
                col=1,
                cut_points_to_view=True,
                max_n_samples=number_of_samples,
                orig_x=df_data[notna_mask].index,
                orig_y=df_data[notna_mask][col],
                secondary_y=params["secondary_y"],
            )

        return self.fig, nb_rows

    # --------------------------------- DASH code base ---------------------------------
    def _to_datetime(self, time_str: str) -> pd.Timestamp:
        return pd.to_datetime([time_str], infer_datetime_format=True).tz_localize(
            self.TIMEZONE
        )[0]

    def _callbacks(self):
        @self.app.callback(
            Output("download-config", "data"),
            Input("btn_config", "n_clicks"),
            prevent_initial_call=True,
        )
        def download_config(n_clicks):
            return dcc.send_data_frame(
                self.df_traces.to_csv, f"{datetime.datetime.now()}-ITSDAR-config.csv"
            )

        @self.app.callback(
            Output(
                "column-selector", "options"
            ),  # This updates the field end_date in the DatePicker
            [
                Input("dataset-selector", "value"),
            ],
        )
        def update_columns(dataset_name):
            if dataset_name is None:
                return []
            dataset = self.dataset_dict[dataset_name]
            return [
                {"label": str(i), "value": str(i)}
                for i in dataset.columns.to_flat_index()
            ]

        @self.app.callback(
            [
                Output("traces-list", "value"),
                Output("confirm", "displayed"),
                Output("confirm", "message"),
            ],
            [
                Input("add-trace", "n_clicks"),
                Input("load-config", "contents"),
                State("traces-list", "value"),
                State("dataset-selector", "value"),
                State("column-selector", "value"),
                State("subplot-selector", "value"),
                State("y-axis-selector", "value"),
            ],
        )
        def add_traces(
            n_clicks,
            config,
            trace_values,
            dataset_name,
            columns,
            subplot_row,
            secondary_y,
        ):
            ctx = dash.callback_context
            # Checking if callback is triggred from the loading of a new config
            if len(ctx.triggered) and "load-config" in ctx.triggered[0]["prop_id"]:
                content_type, content_string = config.split(",")

                decoded = base64.b64decode(content_string)
                df: pd.DataFrame = pd.read_csv(
                    io.StringIO(decoded.decode("utf-8")),
                    converters={"column": ast.literal_eval},
                )
                df = df.set_index("trace_key")

                # We need to load all the required datasets for those traces, therefore
                # we checkthe dataset cache folder.
                for dataset_name in df["dataset_name"]:
                    if not (dataset_name in self.dataset_dict):
                        try:
                            self.dataset_dict[dataset_name] = pd.read_parquet(
                                f"{self.CACHE_FOLDER}/{dataset_name}"
                            ).tz_convert(self.TIMEZONE)
                        except FileNotFoundError:
                            # Show this message in a confirm dialog
                            return (
                                trace_values,
                                True,
                                f"No dataset named {dataset_name} is loaded and none"
                                " was found in the cache.\n Please upload the dataset"
                                " before uploading this config file.",
                            )
                self.df_traces = df.copy()

                return (list(self.df_traces.index), False, "")

            else:
                # Updating traces list for render
                new_values = trace_values
                if columns is not None:
                    for col in columns:
                        trace_key = f"{dataset_name}_{col}"

                        new_values.append(trace_key)
                        print(secondary_y)
                        self.df_traces = self.df_traces.append(
                            pd.DataFrame(
                                {
                                    "trace_key": [trace_key],
                                    "dataset_name": [dataset_name],
                                    "column": [
                                        ast.literal_eval(col)
                                    ],  # Converts col from str tuple to tuple
                                    "subplot_row": [subplot_row],
                                    "secondary_y": [secondary_y == ["secondary"]],
                                }
                            ).set_index("trace_key")
                        )
                print(self.df_traces)
                return (new_values, False, "")

        @self.app.callback(
            Output("traces-list", "options"), [Input("traces-list", "value")]
        )
        def sync_value_options(values):
            # In order to make the dropdown act as a simple list we always have to sync
            # the values and options field.
            return [{"label": v, "value": v} for v in values]

        @self.app.callback(
            Output("load-dataset-list", "children"),
            [Input("load-dataset", "isCompleted")],
            [State("load-dataset", "fileNames"), State("load-dataset", "upload_id")],
            prevent_initial_call=True,
        )
        def dataset_upload(iscompleted, filenames, upload_id):
            if not iscompleted:
                return

            out = []
            if filenames is not None:
                # Ignoring upload_id, not sure why this does not work
                # if upload_id:
                #     root_folder = CACHE_FOLDER / upload_id
                # else:
                root_folder = self.CACHE_FOLDER
                for filename in filenames:
                    file = root_folder / filename
                    df = pd.read_parquet(file)
                    print(df.index)
                    self.dataset_dict[filename] = df.tz_convert(self.TIMEZONE)

                    out.append(file)
                return html.Ul([html.Li(str(x)) for x in out])

            return html.Div("No Files Uploaded Yet!")

        @self.app.callback(
            Output("dataset-selector", "options"),
            Input("load-dataset-list", "children"),
        )
        def populate_dataset_dropdown(children):
            print(f"following keys in dict: {self.dataset_dict.keys()}")
            return [{"label": k, "value": k} for k in self.dataset_dict.keys()]

        @self.app.callback(
            [
                Output("resampled-graph", "figure"),
                Output("subplot-selector", "options"),
            ],
            [
                Input("traces-list", "value"),
                Input("resampled-graph", "relayoutData"),
                State("resampled-graph", "figure"),
                State("subplot-selector", "options"),
            ],
        )
        def plot_or_update_graph(
            trace_keys, changed_layout: dict, current_graph, subplot_options
        ):
            ctx = dash.callback_context
            if len(ctx.triggered) and "traces-list" in ctx.triggered[0]["prop_id"]:
                new_figure, nrows = self.create_figure(trace_keys)
                subplot_options = [
                    {"label": i, "value": i} for i in range(1, nrows + 1)
                ] + [{"label": "New subplot", "value": 0}]
                return new_figure, subplot_options

            if current_graph is not None and len(current_graph["data"]):
                if changed_layout:

                    if "xaxis.range[0]" in changed_layout:
                        t_start = self._to_datetime(
                            changed_layout["xaxis.range[0]"],
                        )
                        t_stop = self._to_datetime(changed_layout["xaxis.range[1]"])
                        self.fig.check_update_figure_dict(
                            current_graph, t_start, t_stop
                        )
                    elif changed_layout.get("xaxis.autorange", False):
                        self.fig.check_update_figure_dict(current_graph)
                else:
                    t_start, t_stop = current_graph["layout"]["xaxis"]["range"]
                    t_start = self._to_datetime(t_start)
                    t_stop = self._to_datetime(t_stop)
                    self.fig.check_update_figure_dict(current_graph, t_start, t_stop)
                    return current_graph, subplot_options
            return current_graph, subplot_options


if __name__ == "__main__":
    # app.run_server(mode="external", port=8053, debug=False)
    db = Dashboard()
    db.app.run_server(port=8056, debug=True, host="192.168.50.218")
