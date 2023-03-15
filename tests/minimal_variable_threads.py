
import numpy as np
import plotly.graph_objs as go
# from dash import Input, Output, dcc, html
# from trace_updater import TraceUpdater

# import sys
# print(sys.path)
# sys.path.append('C:\\Users\\willi\\Documents\\ISIS\\Thesis\\plotly-resampler')


from plotly_resampler.figure_resampler import FigureResampler
from plotly_resampler.aggregation import EveryNthPoint

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--npoints", type=int)
parser.add_argument("-s", "--nsamples", type=int)
parser.add_argument("-t", "--traces", type=int)

args = parser.parse_args()
n = args.npoints
s = args.nsamples
t = args.traces

# print(n)
# print(s)
# print(t)



# # Construct a high-frequency signal
# n=1_000_000
# s=10_000
# t=10

def make_fig(n, s, t):  
    x = np.arange(n)
    noisy_sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / (n / 10)
    print(n/s)
    # Construct the to-be resampled figure
    fig = FigureResampler(
        go.Figure(),
        # show_mean_aggregation_size=False,
        default_downsampler=EveryNthPoint(interleave_gaps=False),
        default_n_shown_samples=s,
        resampled_trace_prefix_suffix=("", ""),
    )
    for i in range(t):
        fig.add_trace(
            go.Scattergl(name=f"sine-{i}", showlegend=True), hf_x=x, hf_y=noisy_sin + 10 * i
        )
    return fig


# Construct app & its layout
# app = dash.Dash(__name__)

# app.layout = html.Div(
#     [
#         dcc.Store(id="visible-indices", data={"visible": [], "invisible": []}),
#         dcc.Graph(id="graph-id", figure=fig),
#         TraceUpdater(id="trace-updater", gdID="graph-id",verbose=True),
#     ]
# )

# n=1_000_000
# s=4000
# t=100

fig = make_fig(n, s, t)
# Register the callback

fig.show_dash(mode='external', testing=True)
# # fig.register_update_graph_callback(app, "graph-id", "trace-updater", "visible-indices")


# if __name__ == "__main__":
#     app.run_server(debug=True, port=8050)
