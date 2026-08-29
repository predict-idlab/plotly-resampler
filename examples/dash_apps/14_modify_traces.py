"""Minimal Dash app example with dynamic trace updating based on front-end interactions.

The aim of this example is to illustrate how plotly-resampler can be used to:
* update high-frequency trace data based on front-end interactions
* retain the front-end graph view when doing these updates

Click on a button, and draw a new plotly-resampler graph of two noisy sinusoids.
The sinusoid in the lower graph its shape is determined by the "expansion factor".
When you alter this variable, and re-click on the button, the lower subplot will change
accordingly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, dcc, html, no_update, callback_context
import dash_bootstrap_components as dbc
from dash_extensions.enrich import (
    DashProxy,
    ServersideOutput,
    ServersideOutputTransform,
)
from trace_updater import TraceUpdater
from plotly_resampler import FigureResampler

# Data that will be used for the plotly-resampler figures
x = np.arange(2_000_000)
noisy_sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

# --------------------------------------Globals ---------------------------------------
app = DashProxy(
    __name__,
    transforms=[ServersideOutputTransform()],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

app.layout = html.Div(
    [
        html.H1("Plotly-Resampler: modify trace data", style={"textAlign": "center"}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.H3("Expansion factor:", style={"textAlign": "right"})),
                dbc.Col(
                    dbc.Input(
                        id="expansion-factor",
                        **dict(min=0.98, max=1.02, step=0.005, value=1, type="number"),
                    )
                ),
                dbc.Col(dbc.Button("plot chart", id="plot-button")),
            ],
            justify="start",
        ),
        # The graph and its needed components to serialize and update efficiently
        # Note: we also add a dcc.Store component, which will be used to link the
        #       server side cached FigureResampler object
        dcc.Graph(id="graph-id"),
        dcc.Loading(dcc.Store(id="store")),
        TraceUpdater(id="trace-updater", gdID="graph-id"),
    ]
)


def y_parsing_func(expansion_factor) -> np.ndarray:
    "Dummy function which can be replaced with more advanced signal processing logic."
    return noisy_sin * expansion_factor ** np.sqrt(x)


# ------------------------------------ DASH logic -------------------------------------
@app.callback(
    [Output("graph-id", "figure"), ServersideOutput("store", "data")],
    Input("plot-button", "n_clicks"),
    [
        State("store", "data"),
        State("graph-id", "relayoutData"),
        State("expansion-factor", "value"),
    ],
    prevent_initial_call=True,
)
def plot_update_graph(n_clicks, fig, relayout, expansion_factor):
    ctx = callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        if fig is None:
            # Construct the figure
            fig = FigureResampler(make_subplots(rows=2, shared_xaxes=True))
            fig.add_trace(go.Scattergl(name="orig"), hf_x=x, hf_y=noisy_sin)
            fig.add_trace(
                go.Scattergl(name="parsed"),
                hf_x=x,
                hf_y=y_parsing_func(expansion_factor),
                **dict(row=2, col=1),
            )
            fig.update_layout(template="plotly_white")
        else:
            # 0. Update the data in the back-end
            #    i.e, update the lower trace its RAW data via the `hf_data` property
            fig.hf_data[1]["y"] = y_parsing_func(expansion_factor)

            # NOTE: As we send back the `dcc.Figure` and not the `updateData`, we will
            # construct the `updateData` based on the `relayout` and use it to set the
            # `dcc.Figure` its trace data accordingly

            # 1. Perform a relayout on the figure based on the front-end layout
            #    So that the figure range of the to be returned figure will be correct
            fig.plotly_relayout(relayout)

            # 2. Alter the relayout dict to make sure that `construct_update_data` will
            #    trigger (and thus not return `dash.no_update`)
            for ax in fig.layout.to_plotly_json().keys():
                if not ax.startswith("xaxis"):
                    continue

                # No xaxis range properties -> mimic reset axes for that xaxis
                if f"{ax}.range[0]" not in relayout:
                    relayout.update(
                        {f"{ax}.autorange": True, f"{ax}.showspikes": False}
                    )

            # 3. Construct the updateData
            #    and update the `dcc.Figure` its data with the updateData
            for data in fig.construct_update_data(relayout)[1:]:
                fig.data[data.pop("index")].update(data, overwrite=True)
        return fig, fig
    return no_update


@app.callback(
    Output("trace-updater", "updateData"),
    Input("graph-id", "relayoutData"),
    State("store", "data"),
    prevent_initial_call=True,
)
def update_fig(relayoutdata, fig):
    if fig is None:
        return no_update
    return fig.construct_update_data(relayoutdata)


# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9023)
