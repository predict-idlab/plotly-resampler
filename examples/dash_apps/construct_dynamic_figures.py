from dash import Dash, dcc, html, Input, Output, State, MATCH
import dash_bootstrap_components as dbc
import numpy as np
from plotly_resampler import FigureResampler
import plotly.graph_objects as go
from trace_updater import TraceUpdater
from uuid import uuid4

from dash.exceptions import PreventUpdate

# The globabl variables
graph_dict = {}
app = Dash(__name__, suppress_callback_exceptions=True) # external_stylesheets=[dbc.themes.COSMO]

# Construct the app layout
app.layout = dbc.Container(
    [
        dbc.Container(html.H1("Sine graph generator"), style={"textAlign": "center"}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Input(id="nbr-datapoints", placeholder="n", type="number"),
                        dcc.Input(
                            id="expansion-factor",
                            placeholder="gamma",
                            type="number",
                        ),
                        html.Button(id="add-graph-btn", children="Add a graph"),
                    ],
                    md=2,
                ),
                dbc.Col(html.Div(id="graph-container"), md=10),
            ],
            align="center",
        ),
    ]
)


# ------------------------------- Callbacks --------------------------------------
@app.callback(
    Output("graph-container", "children"),
    Input("add-graph-btn", "n_clicks"),
    [
        State("nbr-datapoints", "value"),
        State("expansion-factor", "value"),
        State("graph-container", "children"),
    ],
    prevent_initial_call=True,
)
def add_sin(n_clicks, n, gamma, graph_container):
    print(f"n_clicks: {n_clicks}\tnbr_datapoints: {n}\texpansion factor:{gamma}")
    if n_clicks is None or n is None or gamma is None:
        raise PreventUpdate()

    x = np.arange(n)
    expansion_scaling = x ** gamma
    y = np.sin(x / 10) * expansion_scaling + np.random.randn(n) / 10 * gamma

    global graph_dict
    fr = FigureResampler(go.Figure())

    fr.add_trace(go.Scattergl(name="sin"), hf_x=x, hf_y=y)
    fr.update_layout(
        height=300,
        showlegend=True,
        legend=dict(orientation="h"),
        title=f"graph {len(graph_dict) + 1} - n={n} epxansion={gamma}",
        title_x=0.5,
    )

    graph_container = [] if graph_container is None else graph_container

    uid = str(uuid4())
    graph_dict[uid] = fr
    return html.Div(
        [
            html.Div(graph_container),
            # html.H2(f"graph-{len(graph_dict)}"),
            dcc.Graph(figure=fr, id={"type": "dynamic-graph", "index": uid}),
            TraceUpdater(id={"type": "dynamic-updater", "index": uid}, gdID=uid),
        ]
    )


# The generic resampling callback
# see: https://dash.plotly.com/pattern-matching-callbacks for more info
@app.callback(
    Output({"type": "dynamic-updater", "index": MATCH}, "updateData"),
    Input({"type": "dynamic-graph", "index": MATCH}, "relayoutData"),
    State({"type": "dynamic-graph", "index": MATCH}, "id"),
)
def update_figure(relayoutdata, id):
    # fetch the corresponding graph and update it.
    return graph_dict.get(id.get("index"))._update_graph(relayoutdata)


if __name__ == "__main__":
    app.run_server(debug=True)
