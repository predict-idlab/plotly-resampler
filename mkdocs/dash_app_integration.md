# Dash apps 🤝

This documentation page describes how you can integrate `plotly-resampler` in a [dash](https://dash.plotly.com/) application.

Examples of dash apps with `plotly-resampler` can be found in the
[examples folder](https://github.com/predict-idlab/plotly-resampler/tree/main/examples) of the GitHub repository.

## Registering callbacks in a new dash app

When you add a `FigureResampler` figure in a basic dash app, you should:

- Register the [`FigureResampler`][figure_resampler.FigureResampler] figure its callbacks to the dash app.
      - The id of the [dcc.Graph](https://dash.plotly.com/dash-core-components/graph) component that contains the
      [`FigureResampler`][figure_resampler.FigureResampler] figure should be passed to the
      [`register_update_graph_callback`][figure_resampler.FigureResampler.register_update_graph_callback] method.

**Code illustration**:

```python
# Construct the to-be resampled figure
fig = FigureResampler(px.line(...))

# Construct app & its layout
app = dash.Dash(__name__)
app.layout = html.Div(children=[dcc.Graph(id="graph-id", figure=fig)])

# Register the callback
fig.register_update_graph_callback(app, "graph-id")

# start the app
app.run_server(debug=True)
```

!!! warning

    The above example serves as an illustration, but uses a _global variable_ to store the `FigureResampler` instance;
    this is not a good practice. Ideally you should cache the `FigureResampler` per session on the server side.
    In the [examples folder](https://github.com/predict-idlab/plotly-resampler/tree/main/examples),
    we provide several dash app examples where we perform server side caching of such figures.
