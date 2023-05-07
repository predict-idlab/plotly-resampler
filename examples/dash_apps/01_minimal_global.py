"""Minimal dash app example.

Click on a button, and see a plotly-resampler graph of two noisy sinusoids.
No dynamic graph construction / pattern matching callbacks are needed.

This example uses a global FigureResampler object, which is considered a bad practice.
source: https://dash.plotly.com/sharing-data-between-callbacks: 

    Dash is designed to work in multi-user environments where multiple people view the
    application at the same time and have independent sessions.
    If your app uses and modifies a global variable, then one user's session could set
    the variable to some value which would affect the next user's session.

"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback_context, dcc, html, no_update
from trace_updater import TraceUpdater

from plotly_resampler import FigureResampler

# Data that will be used for the plotly-resampler figures
x = np.arange(2_000_000)
noisy_sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000


# --------------------------------------Globals ---------------------------------------
app = Dash(__name__)
fig: FigureResampler = FigureResampler()
# NOTE: in this example, this reference to a FigureResampler is essential to preserve
# throughout the whole dash app! If your dash app wants to create a new go.Figure(),
# you should not construct a new FigureResampler object, but replace the figure of this
# FigureResampler object by using the FigureResampler.replace() method.

app.layout = html.Div(
    [
        html.H1("plotly-resampler global variable", style={"textAlign": "center"}),
        html.Button("plot chart", id="plot-button", n_clicks=0),
        html.Hr(),
        # The graph and it's needed components to update efficiently
        dcc.Graph(id="graph-id"),
        TraceUpdater(id="trace-updater", gdID="graph-id"),
    ]
)


# ------------------------------------ DASH logic -------------------------------------
# The callback used to construct and store the graph's data on the serverside
@app.callback(
    Output("graph-id", "figure"),
    Input("plot-button", "n_clicks"),
    prevent_initial_call=True,
)
def plot_graph(n_clicks):
    ctx = callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        # Note how the replace method is used here on the global figure object
        global fig
        if len(fig.data):
            # Replace the figure with an empty one to clear the graph
            fig.replace(go.Figure())
        fig.add_trace(go.Scattergl(name="log"), hf_x=x, hf_y=noisy_sin * 0.9999995**x)
        fig.add_trace(go.Scattergl(name="exp"), hf_x=x, hf_y=noisy_sin * 1.000002**x)
        return fig
    else:
        return no_update


# Register the graph update callbacks to the layout
fig.register_update_graph_callback(
    app=app, graph_id="graph-id", trace_updater_id="trace-updater"
)

# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9023)
