"""Dash runtime sine generator app example.

In this example, users can configure parameters of a sine wave and then generate the
sine-wave graph at runtime using the create-new-graph button. There is also an option
to remove the graph.

This app uses server side caching of the FigureResampler object. As it uses the same
concepts of the 03_minimal_cache_dynamic.py example, the runtime graph construction
callback is again split up into two callbacks: (1) the callback used to construct the
necessary components and send them to the front-end and (2) the callback used to
construct the plotly-resampler figure and cache it on the server side.

"""

from uuid import uuid4

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import MATCH, Input, Output, State, callback_context, dcc, html, no_update
from dash_extensions.enrich import (
    DashProxy,
    Serverside,
    ServersideOutputTransform,
    Trigger,
    TriggerTransform,
)
from trace_updater import TraceUpdater

from plotly_resampler import FigureResampler

# --------------------------------------Globals ---------------------------------------
app = DashProxy(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUX],
    transforms=[ServersideOutputTransform(), TriggerTransform()],
)

# -------- Construct the app layout --------
app.layout = html.Div(
    [
        html.Div(html.H1("Exponential sine generator"), style={"textAlign": "center"}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Form(
                        [
                            dbc.Label("#datapoints:", style={"margin-left": "10px"}),
                            html.Br(),
                            dcc.Input(
                                id="nbr-datapoints",
                                placeholder="n",
                                type="number",
                                style={"margin-left": "10px"},
                            ),
                            *([html.Br()] * 2),
                            dbc.Label("exponent:", style={"margin-left": "10px"}),
                            html.Br(),
                            dcc.Input(
                                id="expansion-factor",
                                placeholder="pow",
                                type="number",
                                min=0.95,
                                max=1.00001,
                                style={"margin-left": "10px"},
                            ),
                            *([html.Br()] * 2),
                            dbc.Button(
                                "Create new graph",
                                id="add-graph-btn",
                                color="primary",
                                style={
                                    "textalign": "center",
                                    "width": "max-content",
                                    "margin-left": "10px",
                                },
                            ),
                            *([html.Br()] * 2),
                            dbc.Button(
                                "Remove last graph",
                                id="remove-graph-btn",
                                color="danger",
                                style={
                                    "textalign": "center",
                                    "width": "max-content",
                                    "margin-left": "10px",
                                },
                            ),
                        ],
                    ),
                    style={"align": "top"},
                    md=2,
                ),
                dbc.Col(html.Div(id="graph-container"), md=10),
            ],
        ),
    ]
)


# ------------------------------------ DASH logic -------------------------------------
# This method adds the needed components to the front-end, but does not yet contain the
# FigureResampler graph construction logic.
@app.callback(
    Output("graph-container", "children"),
    Input("add-graph-btn", "n_clicks"),
    Input("remove-graph-btn", "n_clicks"),
    [
        State("nbr-datapoints", "value"),
        State("expansion-factor", "value"),
        State("graph-container", "children"),
    ],
    prevent_initial_call=True,
)
def add_or_remove_graph(add_graph, remove_graph, n, exp, gc_children):
    if (add_graph is None or n is None or exp is None) and (remove_graph is None):
        return no_update

    # Transform the graph data to a figure
    gc_children = [] if gc_children is None else gc_children

    # Check if we need to remove a graph
    clicked_btns = [p["prop_id"] for p in callback_context.triggered]
    if any("remove-graph" in btn_name for btn_name in clicked_btns):
        if not len(gc_children):
            return no_update
        return gc_children[:-1]

    # No graph needs to be removed -> create a new graph
    uid = str(uuid4())
    new_child = html.Div(
        children=[
            # The graph and its needed components to serialize and update efficiently
            # Note: we also add a dcc.Store component, which will be used to link the
            #       server side cached FigureResampler object
            dcc.Graph(id={"type": "dynamic-graph", "index": uid}, figure=go.Figure()),
            dcc.Loading(dcc.Store(id={"type": "store", "index": uid})),
            TraceUpdater(id={"type": "dynamic-updater", "index": uid}, gdID=f"{uid}"),
            # This dcc.Interval components makes sure that the `construct_display_graph`
            # callback is fired once after these components are added to the session
            # its front-end
            dcc.Interval(
                id={"type": "interval", "index": uid}, max_intervals=1, interval=1
            ),
        ],
    )
    gc_children.append(new_child)
    return gc_children


# This method constructs the FigureResampler graph and caches it on the server side
@app.callback(
    Output({"type": "dynamic-graph", "index": MATCH}, "figure"),
    Output({"type": "store", "index": MATCH}, "data"),
    State("nbr-datapoints", "value"),
    State("expansion-factor", "value"),
    State("add-graph-btn", "n_clicks"),
    Trigger({"type": "interval", "index": MATCH}, "n_intervals"),
    prevent_initial_call=True,
)
def construct_display_graph(n, exp, n_added_graphs) -> FigureResampler:
    # Figure construction logic based on state variables
    x = np.arange(n)
    expansion_scaling = exp**x
    y = (
        np.sin(x / 200) * expansion_scaling
        + np.random.randn(n) / 10 * expansion_scaling
    )

    fr = FigureResampler(go.Figure(), verbose=True)
    fr.add_trace(go.Scattergl(name="sin"), hf_x=x, hf_y=y)
    fr.update_layout(
        height=350,
        showlegend=True,
        legend=dict(orientation="h", y=1.12, xanchor="right", x=1),
        template="plotly_white",
        title=f"graph {n_added_graphs} - n={n:,} pow={exp}",
        title_x=0.5,
    )

    return fr, Serverside(fr)


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
    return no_update


# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9023)
