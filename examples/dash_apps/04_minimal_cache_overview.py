"""Minimal dash app example.

Click on a button, and see a plotly-resampler graph of two sinusoids.
In addition, another graph is shown, which is an overview of the main graph.
This other graph is bidirectionally linked to the main graph; when you select a region
in the overview graph, the main graph will zoom in on that region and vice versa.

This example uses the dash-extensions its ServersideOutput functionality to cache
the FigureResampler per user/session on the server side. This way, no global figure
variable is used and shows the best practice of using plotly-resampler within dash-apps.

"""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update
from dash_extensions.enrich import DashProxy, Serverside, ServersideOutputTransform

# The overview figure requires clientside callbacks, whose JavaScript code is located
# in the assets folder. We need to tell dash where to find this folder.
from plotly_resampler import ASSETS_FOLDER, FigureResampler

# -------------------------------- Data and constants ---------------------------------
# Data that will be used for the plotly-resampler figures
x = np.arange(2_000_000)
noisy_sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

# The ids of the components used in the app (we put them here to avoid typos)
GRAPH_ID = "graph-id"
OVERVIEW_GRAPH_ID = "overview-graph"
STORE_ID = "store"


# --------------------------------------Globals ---------------------------------------
# NOTE: Remark how the assets folder is passed to the Dash(proxy) application and how
#       the lodash script is included as an external script.
app = DashProxy(
    __name__,
    transforms=[ServersideOutputTransform()],
    assets_folder=ASSETS_FOLDER,
    external_scripts=["https://cdn.jsdelivr.net/npm/lodash/lodash.min.js"],
)

app.layout = html.Div(
    [
        html.H1("plotly-resampler + dash-extensions", style={"textAlign": "center"}),
        html.Button("plot chart", id="plot-button", n_clicks=0),
        html.Hr(),
        # The graph, overview graph, and serverside store for the FigureResampler graph
        dcc.Graph(id=GRAPH_ID),
        dcc.Graph(id=OVERVIEW_GRAPH_ID),
        dcc.Loading(dcc.Store(id=STORE_ID)),
    ]
)


# ------------------------------------ DASH logic -------------------------------------
# --- construct and store the FigureResampler on the serverside ---
@app.callback(
    [
        Output(GRAPH_ID, "figure"),
        Output(OVERVIEW_GRAPH_ID, "figure"),
        Output(STORE_ID, "data"),
    ],
    Input("plot-button", "n_clicks"),
    prevent_initial_call=True,
)
def plot_graph(_):
    global app
    ctx = callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        fig: FigureResampler = FigureResampler(create_overview=True)

        # Figure construction logic
        fig.add_trace(go.Scattergl(name="log"), hf_x=x, hf_y=noisy_sin * 0.9999995**x)
        fig.add_trace(go.Scattergl(name="exp"), hf_x=x, hf_y=noisy_sin * 1.000002**x)

        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_layout(margin=dict(b=10), template="plotly_white")

        coarse_fig = fig._create_overview_figure()
        return fig, coarse_fig, Serverside(fig)
    else:
        return no_update


# --- Clientside callbacks used to bidirectionally link the overview and main graph ---
app.clientside_callback(
    dash.ClientsideFunction(namespace="clientside", function_name="main_to_coarse"),
    dash.Output(OVERVIEW_GRAPH_ID, "id", allow_duplicate=True),
    dash.Input(GRAPH_ID, "relayoutData"),
    [dash.State(OVERVIEW_GRAPH_ID, "id"), dash.State(GRAPH_ID, "id")],
    prevent_initial_call=True,
)

app.clientside_callback(
    dash.ClientsideFunction(namespace="clientside", function_name="coarse_to_main"),
    dash.Output(GRAPH_ID, "id", allow_duplicate=True),
    dash.Input(OVERVIEW_GRAPH_ID, "selectedData"),
    [dash.State(GRAPH_ID, "id"), dash.State(OVERVIEW_GRAPH_ID, "id")],
    prevent_initial_call=True,
)


# --- FigureResampler update callback ---

# The plotly-resampler callback to update the graph after a relayout event (= zoom/pan)
# As we use the figure again as output, we need to set: allow_duplicate=True
@app.callback(
    Output(GRAPH_ID, "figure", allow_duplicate=True),
    Input(GRAPH_ID, "relayoutData"),
    State(STORE_ID, "data"),  # The server side cached FigureResampler per session
    prevent_initial_call=True,
)
def update_fig(relayoutdata: dict, fig: FigureResampler):
    if fig is None:
        return no_update
    return fig.construct_update_data_patch(relayoutdata)


# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9023, use_reloader=False)
