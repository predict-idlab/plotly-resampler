"""Minimal dash app example.

Click on a button, and see a plotly-resampler graph of two noisy sinusoids.
No dynamic graph construction / pattern matching callbacks are needed.

This example uses the dash-extensions its ServersideOutput functionality to cache
the FigureResampler per user/session on the server side. This way, no global figure
variable is used and shows the best practice of using plotly-resampler within dash-apps.

"""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update
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
app = DashProxy(__name__, transforms=[ServersideOutputTransform()])

app.layout = html.Div(
    [
        html.H1("plotly-resampler + dash-extensions", style={"textAlign": "center"}),
        html.Button("plot chart", id="plot-button", n_clicks=0),
        html.Hr(),
        # The graph and its needed components to serialize and update efficiently
        # Note: we also add a dcc.Store component, which will be used to link the
        #       server side cached FigureResampler object
        dcc.Graph(id="graph-id"),
        dcc.Loading(dcc.Store(id="store")),
        TraceUpdater(id="trace-updater", gdID="graph-id"),
    ]
)


# ------------------------------------ DASH logic -------------------------------------
# The callback used to construct and store the FigureResampler on the serverside
@app.callback(
    [Output("graph-id", "figure"), ServersideOutput("store", "data")],
    Input("plot-button", "n_clicks"),
    prevent_initial_call=True,
    memoize=True,
)
def plot_graph(n_clicks):
    ctx = callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        fig: FigureResampler = FigureResampler(go.Figure())

        # Figure construction logic
        fig.add_trace(go.Scattergl(name="log"), hf_x=x, hf_y=noisy_sin * 0.9999995**x)
        fig.add_trace(go.Scattergl(name="exp"), hf_x=x, hf_y=noisy_sin * 1.000002**x)

        return fig, fig
    else:
        return no_update


@app.callback(
    Output("trace-updater", "updateData"),
    Input("graph-id", "relayoutData"),
    State("store", "data"),  # The server side cached FigureResampler per session
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata, fig):
    if fig is None:
        return no_update
    return fig.construct_update_data(relayoutdata)


# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9023)
