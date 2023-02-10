import trace_updater
import dash
import plotly.graph_objs as go
from dash import html, dcc, Input, Output
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import EveryNthPoint
import numpy as np
from trace_updater import TraceUpdater

# Construct a high-frequency signal
n = 1_000_000
x = np.arange(n)
noisy_sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / (n / 10)

# Construct the to-be resampled figure
fig = FigureResampler(
    go.Figure(),
    # show_mean_aggregation_size=False,
    # default_downsampler=EveryNthPoint(interleave_gaps=False),
    default_n_shown_samples=4000,
    resampled_trace_prefix_suffix=("", ""),
)
for i in range(100):
    fig.add_trace(go.Scattergl(name=f"sine-{i}", showlegend=True), hf_x=x, hf_y=noisy_sin + 10 * i)


# Construct app & its layout
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dcc.Store(id="visible-indices", data={"visible": [], "invisible": []}),
        dcc.Graph(id="graph-id", figure=fig),
        TraceUpdater(id="trace-updater", gdID="graph-id"),
    ]
)

# Register the callback
fig.register_update_graph_callback(app, "graph-id", "trace-updater", "visible-indices")


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)