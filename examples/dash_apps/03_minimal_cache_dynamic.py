"""Minimal dynamic dash app example.

Click on a button, and draw a new plotly-resampler graph of a noisy sinusoid.
This example uses pattern-matching callbacks to update dynamically constructed graphs.
The plotly-resampler graphs themselves are cached on the server side.

The main difference between this example and the dash_app_minimal_cache.py is that here,
we want to cache using a dcc.Store that is not yet available on the client side. As a
result we split up our logic into two callbacks: (1) the callback used to construct the
necessary components and send them to the client-side, and (2) the callback used to
construce the actual plotly-resampler graph and cache it on the server side. These
two callbacks are chained together using the dcc.Interval component.

"""

from uuid import uuid4

import numpy as np
import plotly.graph_objects as go
import dash
from dash import MATCH, Input, Output, State, dcc, html
from dash_extensions.enrich import (
    DashProxy,
    ServersideOutput,
    ServersideOutputTransform,
    Trigger,
    TriggerTransform,
)
from plotly_resampler import FigureResampler
from trace_updater import TraceUpdater

# Data that will be used for the plotly-resampler figures
x = np.arange(2_000_000)
noisy_sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

# --------------------------------------Globals ---------------------------------------
app = DashProxy(__name__, transforms=[ServersideOutputTransform(), TriggerTransform()])

app.layout = html.Div(
    [
        html.Div(children=[html.Button("Add Chart", id="add-chart", n_clicks=0)]),
        html.Div(id="container", children=[]),
    ]
)


# ------------------------------------ DASH logic -------------------------------------
# This method adds the needed components to the front-end, but does not yet contain the
# figureResampler graph construction logic.
@app.callback(
    Output("container", "children"),
    Input("add-chart", "n_clicks"),
    State("container", "children"),
    prevent_initial_call=True,
)
def add_graph_div(n_clicks: int, div_children: list[html.Div]):
    uid = str(uuid4())
    new_child = html.Div(
        children=[
            # The graph and it's needed components to serialize and update efficiently
            # Note: we also add a dcc.Store component, which will be used to link the
            #       server side cached FigureResampler object
            dcc.Graph(id={"type": "dynamic-graph", "index": uid}, figure=go.Figure()),
            dcc.Loading(dcc.Store(id={"type": "store", "index": uid})),
            TraceUpdater(id={"type": "dynamic-updater", "index": uid}, gdID=f"{uid}"),
            # This dcc.(nterval components makes sure that the `construct_display_graph`
            # callback is fired once after these components are added to the session
            # its front-end
            dcc.Interval(
                id={"type": "interval", "index": uid}, max_intervals=1, interval=1
            ),
        ],
    )
    div_children.append(new_child)
    return div_children


# This method constructs the figureResampler graph and caches it on the server side
@app.callback(
    ServersideOutput({"type": "store", "index": MATCH}, "data"),
    Output({"type": "dynamic-graph", "index": MATCH}, "figure"),
    State("add-chart", "n_clicks"),
    Trigger({"type": "interval", "index": MATCH}, "n_intervals"),
    prevent_initial_call=True,
)
def construct_display_graph(n_clicks) -> FigureResampler:
    fig = FigureResampler(go.Figure(), default_n_shown_samples=2_000)

    # Figure construction logic based on a state variable, in our case n_clicks
    sigma = n_clicks * 1e-6
    fig.add_trace(dict(name="log"), hf_x=x, hf_y=noisy_sin * (1 - sigma) ** x)
    fig.add_trace(dict(name="exp"), hf_x=x, hf_y=noisy_sin * (1 + sigma) ** x)
    fig.update_layout(title=f"<b>graph - {n_clicks}</b>", title_x=0.5)

    return fig, fig


@app.callback(
    Output({"type": "dynamic-updater", "index": MATCH}, "updateData"),
    Input({"type": "dynamic-graph", "index": MATCH}, "relayoutData"),
    State({"type": "store", "index": MATCH}, "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata: dict, fig: FigureResampler):
    if fig is not None:
        return fig.construct_update_data(relayoutdata)
    raise dash.exceptions.PreventUpdate()


# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9023)
