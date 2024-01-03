"""Dash file parquet visualization app example with a coarse and fine-grained view.

In this use case, we have dropdowns which allows end-users to select multiple
parquet files, which are visualized using FigureResampler after clicking on a button.

There a two graphs displayed; a coarse and a dynamic graph. Interactions with the
coarse graph will affect the dynamic graph it's shown range. Note that the autosize
of the coarse graph is not linked.

TODO: add an rectangle on the coarse graph

"""
from __future__ import annotations

__author__ = "Jonas Van Der Donckt"

from pathlib import Path
from typing import List

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update
from dash_extensions.enrich import DashProxy, Serverside, ServersideOutputTransform
from utils.callback_helpers import get_selector_states, multiple_folder_file_selector
from utils.graph_construction import visualize_multiple_files

from plotly_resampler import FigureResampler

# --------------------------------------Globals ---------------------------------------
app = DashProxy(
    __name__,
    suppress_callback_exceptions=False,
    external_stylesheets=[dbc.themes.LUX],
    transforms=[ServersideOutputTransform()],
)

# --------- File selection configurations ---------
name_folder_list = [
    {
        # the key-string below is the title which will be shown in the dash app
        "example data": {"folder": Path(__file__).parent.parent.joinpath("data")},
        "other folder": {"folder": Path(__file__).parent.parent.joinpath("data")},
    },
    # NOTE: A new item om this level creates a new file-selector card.
    # { "PC data": { "folder": Path("/home/jonas/data/wesad/empatica/") } }
    # TODO: change the folder path above to a location where you have some
    # `.parquet` files stored on your machine.
]


# --------- DASH layout logic ---------
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
                    # Add the graphs, the dcc.Store (for serialization) and the
                    dbc.Col(
                        [
                            # The coarse graph whose updates will fetch data for the
                            dcc.Graph(
                                id="coarse-graph",
                                figure=go.Figure(),
                                config={"modeBarButtonsToAdd": ["drawrect"]},
                            ),
                            html.Br(),
                            dcc.Graph(id="plotly-resampler-graph", figure=go.Figure()),
                            dcc.Loading(dcc.Store(id="store")),
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


# ------------------------------------ DASH logic -------------------------------------
# --------- graph construction logic + callback ---------
@app.callback(
    [
        Output("coarse-graph", "figure"),
        Output("plotly-resampler-graph", "figure"),
        Output("store", "data"),
    ],
    [Input("plot-button", "n_clicks"), *get_selector_states(len(name_folder_list))],
    prevent_initial_call=True,
)
def construct_plot_graph(n_clicks, *folder_list):
    it = iter(folder_list)
    file_list: List[Path] = []
    for folder, files in zip(it, it):
        if not all((folder, files)):
            continue
        else:
            files = [files] if not isinstance(files, list) else file_list
            for file in files:
                file_list.append((Path(folder).joinpath(file)))

    ctx = callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        if len(file_list):
            # Create two graphs, a dynamic plotly-resampler graph and a coarse graph
            dynamic_fig: FigureResampler = visualize_multiple_files(file_list)
            coarse_fig: go.Figure = go.Figure(
                FigureResampler(dynamic_fig, default_n_shown_samples=3_000)
            )

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

            return coarse_fig, dynamic_fig, Serverside(dynamic_fig)
    else:
        return no_update


# Register the graph update callbacks to the layout
# As we use the figure again as output, we need to set: allow_duplicate=True
@app.callback(
    Output("plotly-resampler-graph", "figure", allow_duplicate=True),
    Input("coarse-graph", "relayoutData"),
    Input("plotly-resampler-graph", "relayoutData"),
    State("store", "data"),
    prevent_initial_call=True,
)
def update_dynamic_fig(
    coarse_grained_relayout: dict | None,
    fine_grained_relayout: dict | None,
    fr_fig: FigureResampler,
):
    if fr_fig is None:  # When the figure does not exist -> do nothing
        return no_update

    ctx = callback_context
    trigger_id = ctx.triggered[0].get("prop_id", "").split(".")[0]

    if trigger_id == "plotly-resampler-graph":
        return fr_fig.construct_update_data_patch(fine_grained_relayout)
    elif trigger_id == "coarse-graph":
        return fr_fig.construct_update_data_patch(coarse_grained_relayout)

    return no_update


# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9023, use_reloader=False)
