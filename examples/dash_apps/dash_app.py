"""Minimal dash app example.

In this usecase, we have dropdowns which allows the end-user to select files, which are
visualized using FigureResampler after clicking on a button.

"""

__author__ = "Jonas Van Der Donckt"

from pathlib import Path
from typing import List, Union

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import trace_updater
from dash import Input, Output, State, dcc, html

from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler

from callback_helpers import multiple_folder_file_selector


# Globals
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
fig: FigureResampler = FigureResampler()


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
                html.H1("Data loading and visualization dashboard"),
                style={"textAlign": "center"},
            ),
            html.Hr(),
            dbc.Row(
                [
                    # Add file selection layout (+ assign callbacks)
                    dbc.Col(multiple_folder_file_selector(app, name_folder_list), md=2),
                    # Add the graph and the trace updater component
                    dbc.Col(
                        [
                            dcc.Graph(id="graph-id", figure=go.Figure()),
                            trace_updater.TraceUpdater(
                                id="trace-updater", gdID="graph-id"
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
fig.register_update_graph_callback(
    app=app, graph_id="graph-id", trace_updater_id="trace-updater"
)


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
    global fig

    # NOTE, we do not construct a new FigureResampler object, but replace the figure of 
    # the figureResampler object. Otherwise the coupled callbacks would be lost and it
    # is not (straightforward) to construct dynamic callbacks in dash.

    fig.replace(make_subplots(rows=len(file_list), shared_xaxes=False))
    fig.update_layout(height=min(900, 350 * len(file_list)))

    for i, f in enumerate(file_list, 1):
        df = pd.read_parquet(f)  # should be replaced by more generic data loading code
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        for c in df.columns:
            fig.add_trace(go.Scattergl(name=c), hf_x=df.index, hf_y=df[c], row=i, col=1)
    return fig


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
    Output("graph-id", "figure"),
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
            for file in files:
                file_list.append((Path(folder).joinpath(file)))

    ctx = dash.callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        if len(file_list):
            return plot_multiple_files(file_list)
    else:
        raise dash.exceptions.PreventUpdate()



# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9023)
