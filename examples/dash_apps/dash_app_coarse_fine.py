"""Minimal dash app example.

In this usecase, we have dropdowns which allows the end-user to select files, which are
visualized using FigureResampler after clicking on a button.

There a two graphs displayed a coarse and a dynamic graph.
Interactions with the coarse will affect the dynamic graph range.
Note that the autosize of the coarse graph is not linked.

TODO: add an rectangle on the coarse graph

"""

__author__ = "Jonas Van Der Donckt"

import re
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import trace_updater
from pathlib import Path
from typing import List, Union
from dash import Input, Output, State, dcc, html

from plotly_resampler import FigureResampler

from callback_helpers import multiple_folder_file_selector


# Globals
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
fr_fig: FigureResampler = FigureResampler()
# NOTE, this reference to a FigureResampler is essential to preserve throughout the
# whole dash app! If your dash apps want to create a new go.Figure(), you should not
# construct a new FigureResampler object, but replace the figure of this FigureResampler
# object by using the FigureResampler.replace() method.
# Example: see the plot_multiple_files function in this file.


# ------------------------------- File selection logic -------------------------------
name_folder_list = [
    {
        # the key-string below is the title which will be shown in the dash app
        "dash example data": {"folder": Path("../data")},
        "other name same folder": {"folder": Path("../data")},
    },
    # NOTE: A new item om this level creates a new file-selector card.
    # { "PC data": { "folder": Path("/home/jonas/data/wesad/empatica/") } }
    # TODO: change the folder path above to a location where you have some
    # `.parquet` files stored on your machine.
]


# ------------------------------------ DASH logic -------------------------------------
# First we construct the app layout
def serve_layout() -> dbc.Container:
    """Constructs the app's layout.

    Returns
    -------
    dbc.Container
        A Container withholding the layout.

    """
    return dbc.Container(
        [
            dbc.Container(
                html.H1("Data visualization - coarse & dynamic graph"),
                style={"textAlign": "center"},
            ),
            html.Hr(),
            dbc.Row(
                [
                    # Add file selection layout (+ assign callbacks)
                    dbc.Col(
                        multiple_folder_file_selector(
                            app, name_folder_list, multi=False
                        ),
                        md=2,
                    ),
                    # Add the graphs and the trace updater component
                    dbc.Col(
                        [
                            # The coarse graph whose updates will fetch data for the
                            # broad graph
                            dcc.Graph(
                                id="coarse-graph",
                                figure=go.Figure(),
                                config={"modeBarButtonsToAdd": ["drawrect"]},
                            ),
                            dcc.Graph(id="plotly-resampler-graph", figure=go.Figure()),
                            # The broad graph
                            trace_updater.TraceUpdater(
                                id="trace-updater", gdID="plotly-resampler-graph"
                            ),
                        ],
                        md=10,
                    ),
                ],
                align="center",
            ),
        ],
        fluid=True,
    )


app.layout = serve_layout()


# Register the graph update callbacks to the layout
@app.callback(
    Output("trace-updater", "updateData"),
    Input("coarse-graph", "relayoutData"),
    Input("plotly-resampler-graph", "relayoutData"),
    prevent_initial_call=True,
)
def update_dynamic_fig(coarse_grained_relayout, fine_grained_relayout):
    global fr_fig

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0].get("prop_id", "").split(".")[0]

    if trigger_id == "plotly-resampler-graph":
        return fr_fig.construct_update_data(fine_grained_relayout)
    elif trigger_id == "coarse-graph":
        if "shapes" in coarse_grained_relayout:
            print(coarse_grained_relayout)
        cl_k = coarse_grained_relayout.keys()
        # We do not resample when and autorange / autosize event takes place
        matches = fr_fig._re_matches(re.compile(r"xaxis\d*.range\[0]"), cl_k)
        if len(matches):
            return fr_fig.construct_update_data(coarse_grained_relayout)

    return dash.no_update


# ------------------------------ Visualization logic ---------------------------------
def plot_multiple_files(file_list: List[Union[str, Path]]) -> FigureResampler:
    """Code to create the visualizations.

    Parameters
    ----------
    file_list: List[Union[str, Path]]

    Returns
    -------
    FigureResampler
        Returns a view of the existing, global FigureResampler object.

    """
    global fr_fig

    # NOTE, we do not construct a new FigureResampler object, but replace the figure of
    # the figureResampler object. Otherwise the coupled callbacks would be lost and it
    # is not (straightforward) to construct dynamic callbacks in dash.
    fr_fig._global_n_shown_samples = 3000
    fr_fig.replace(go.Figure())
    fr_fig.update_layout(height=min(900, 350 * len(file_list)))

    for f in file_list:
        df = pd.read_parquet(f)  # should be replaced by more generic data loading code
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        for c in df.columns:
            fr_fig.add_trace(go.Scatter(name=c), hf_x=df.index, hf_y=df[c])
    return fr_fig


# Note: the list sum-operations flattens the list
selector_states = list(
    sum(
        [
            (
                State(f"folder-selector{i}", "value"),
                State(f"file-selector{i}", "value"),
            )
            for i in range(1, len(name_folder_list) + 1)
        ],
        (),
    )
)


@app.callback(
    Output("coarse-graph", "figure"),
    Output("plotly-resampler-graph", "figure"),
    [Input("plot-button", "n_clicks"), *selector_states],
    prevent_initial_call=True,
)
def plot_graph(
    n_clicks,
    *folder_list,
):
    it = iter(folder_list)
    file_list: List[Path] = []
    for folder, files in zip(it, it):
        if not all((folder, files)):
            continue
        else:
            files = [files] if not isinstance(files, list) else file_list
            for file in files:
                file_list.append((Path(folder).joinpath(file)))

    ctx = dash.callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        if len(file_list):
            dynamic_fig = plot_multiple_files(file_list)
            coarse_fig = go.Figure(dynamic_fig)
            coarse_fig.update_layout(title="<b>coarse view</b>", height=250)
            coarse_fig.update_layout(margin=dict(l=0, r=0, b=0, t=40, pad=10))
            coarse_fig.update_layout(showlegend=False)
            coarse_fig._config = coarse_fig._config.update(
                {"modeBarButtonsToAdd": ["drawrect"]}
            )

            dynamic_fig._global_n_shown_samples = 1000
            dynamic_fig.update_layout(title="<b>dynamic view<b>", height=450)
            dynamic_fig.update_layout(margin=dict(l=0, r=0, b=40, t=40, pad=10))
            dynamic_fig.update_layout(
                legend=dict(
                    orientation="h", y=-0.11, xanchor="right", x=1, font_size=18
                )
            )

            # coarse_fig['layout'].update(dict(title='coarse view', title_x=0.5, height=250))
            return coarse_fig, dynamic_fig
    else:
        raise dash.exceptions.PreventUpdate()


# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9023)
