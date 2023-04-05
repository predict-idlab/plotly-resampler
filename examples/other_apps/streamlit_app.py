"""Minimal streamlit app example.

This example shows how to integrate plotly-resampler in a streamlit app.
The following thee steps are required;
1. use FigureResampler
2. run the visualization (which is a dash app) in a (sub)process on a certain port
3. add as iframe component to streamlit

To run this example execute the following command:
$ streamlit run streamlit_app.py

Note: to have colored traces in the streamlit app, you should always include the
following code: `import plotly.io as pio; pio.templates.default = "plotly"`

"""

__author__ = "Jeroen Van Der Donckt"

# Explicitely set pio.templates in order to have colored traces in the streamlit app!
# -> https://discuss.streamlit.io/t/streamlit-overrides-colours-of-plotly-chart/34943/5
import plotly.io as pio

pio.templates.default = "plotly"

# 0. Create a noisy sine wave
import numpy as np
import plotly.graph_objects as go

from plotly_resampler import FigureResampler

x = np.arange(1_000_000)
noisy_sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

### 1. Use FigureResampler
fig = FigureResampler(default_n_shown_samples=2_000)
fig.add_trace(go.Scattergl(name="noisy sine", showlegend=True), hf_x=x, hf_y=noisy_sin)
fig.update_layout(height=700)

### 2. Run the visualization (which is a dash app) in a (sub)process on a certain port
# Note: starting a process allows executing code after `.show_dash` is called
from multiprocessing import Process

port = 9022
proc = Process(target=fig.show_dash, kwargs=dict(mode="external", port=port)).start()

# Deleting the lines below this and running this file will result in a classic running dash app
# Note: for just a dash app it is not even necessary to execute .show_dash in a (sub)process

### 3. Add as iframe component to streamlit
import streamlit.components.v1 as components

components.iframe(f"http://localhost:{port}", height=700)
