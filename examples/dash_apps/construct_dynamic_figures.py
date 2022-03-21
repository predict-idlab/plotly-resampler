from typing import Dict
from uuid import uuid4

import numpy as np
import plotly.graph_objects as go

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, MATCH
from dash.exceptions import PreventUpdate

from trace_updater import TraceUpdater
from plotly_resampler import FigureResampler

# The global variables
graph_dict: Dict[str, FigureResampler] = {}
app = Dash(
    __name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.LUX]
)

# ---------------------------- Construct the app layout ----------------------------
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


# -------------------------------- Callbacks ---------------------------------------
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
def add_or_remove_graph(add_graph, remove_graph, n, exp, gc):
    if (add_graph is None or n is None or exp is None) and (remove_graph is None):
        raise PreventUpdate()

    # Transform the graph data to a figure
    gc = [] if gc is None else gc  # list of existing Graphs and their TraceUpdaters
    if len(gc):
        _gc = []
        for i in range(len(gc) // 2):
            _gc.append(dcc.Graph(**gc[i * 2]["props"]))
            _gc.append(
                TraceUpdater(**{k: gc[i * 2 + 1]["props"][k] for k in ["id", "gdID"]})
            )
        gc = _gc

    # Check if we need to remove a graph
    clicked_btns = [p["prop_id"] for p in dash.callback_context.triggered]
    if any("remove-graph" in btn_name for btn_name in clicked_btns):
        if not len(gc):
            raise PreventUpdate()

        graph_dict.pop(gc[-1].__getattribute__("gdID"))
        return [*gc[:-2]]

    # No graph needs to be removed -> create a new graph
    x = np.arange(n)
    expansion_scaling = exp ** x
    y = np.sin(x / 10) * expansion_scaling + np.random.randn(n) / 10 * expansion_scaling

    fr = FigureResampler(go.Figure(), verbose=True)
    fr.add_trace(go.Scattergl(name="sin"), hf_x=x, hf_y=y)
    fr.update_layout(
        height=350,
        showlegend=True,
        legend=dict(orientation="h", y=1.12, xanchor="right", x=1),
        template="plotly_white",
        title=f"graph {len(graph_dict) + 1} - n={n:,} pow={exp}",
        title_x=0.5,
    )

    # Create a uuid for the graph and add it to the global graph dict,
    uid = str(uuid4())
    graph_dict[uid] = fr

    # Add the graph to the existing output
    return [
        *gc,  # the existing Graphs and their TraceUpdaters
        dcc.Graph(figure=fr, id={"type": "dynamic-graph", "index": uid}),
        TraceUpdater(id={"type": "dynamic-updater", "index": uid}, gdID=uid),
    ]


# The generic resampling callback
# see: https://dash.plotly.com/pattern-matching-callbacks for more info
@app.callback(
    Output({"type": "dynamic-updater", "index": MATCH}, "updateData"),
    Input({"type": "dynamic-graph", "index": MATCH}, "relayoutData"),
    State({"type": "dynamic-graph", "index": MATCH}, "id"),
    prevent_initial_call=True,
)
def update_figure(relayoutdata: dict, graph_id_dict: dict):
    return graph_dict.get(graph_id_dict["index"]).construct_update_data(relayoutdata)


if __name__ == "__main__":
    app.run_server(debug=True)
