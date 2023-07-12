# FAQ ❓

??? abstract "What does the orange `~time|number` suffix in legend name indicate?"

    This tilde suffix is only shown when the data is aggregated and represents the _mean aggregation bin size_
    which is the mean index-range difference between two consecutive aggregated samples.

    > - for _time-indexed data_: the mean time-range between 2 consecutive (sampled) samples.
    > - for _numeric-indexed data_: the mean numeric range between 2 consecutive (sampled) samples.

    When the index is a range-index; the mean aggregation bin size represents the mean downsample ratio; i.e.,
    the mean number of samples that are aggregated into one sample.

??? abstract "What is the difference between plotly-resampler figures and plain plotly figures?"

    plotly-resampler can be thought of as wrapper around plain plotly figures
    which adds line-chart visualization scalability by dynamically aggregating the data of the figures w.r.t.
    the front-end view. plotly-resampler thus adds dynamic aggregation functionality to plain plotly figures.

    **important to know**:

    - `show` _always_ returns a static html view of the figure, i.e., no dynamic aggregation can be performed on that view.
    - To have dynamic aggregation:
          - with `FigureResampler`, you need to call `show_dash` (or output the object in a cell via `IPython.display`) ->
          which spawns a dash-web app, and the dynamic aggregation is realized with dash callback
          - with `FigureWidgetResampler`, you need to use `IPython.display` on the object,
          which uses widget-events to realize dynamic aggregation (via the running IPython kernel).

    **other changes of plotly-resampler figures w.r.t. vanilla plotly**:

    - double-clicking within a line-chart area does not Reset Axes, as it results in an “Autoscale” event.
    We decided to implement an Autoscale event as updating your y-range such that it shows all the data that
    is in your x-range
         - **Note**: vanilla Plotly figures their Autoscale result in Reset Axes behavior,
         in our opinion this did not make a lot of sense.
         It is therefore that we have overriden this behavior in plotly-resampler.

??? abstract "What does [TraceUpdater](https://github.com/predict-idlab/trace-updater) do?"

    The `TraceUpdater` class is a custom dash component that aids `dcc.Graph` components to
    efficiently send and update (in our case aggregated) data to the front-end.

    For more information on how to use the trace-updater component together with the `FigureResampler`,
    see our dash app [examples](https://github.com/predict-idlab/plotly-resampler/tree/main/examples)
    and look at the [trace-updater](https://github.com/predict-idlab/trace-updater/blob/master/trace_updater/TraceUpdater.py) its documentation.

??? abstract "My `FigureResampler.show_dash` keeps hanging (indefinitely) with the error message: `OSError: Port already in use`"

    !!! info "Disclaimer"
        Since v0.9.0 we use Dash instead of JupyterDash for Jupyter integration which should have resolved this issue!


    Plotly-resampler its `FigureResampler.show_dash` method leverages the [jupyterdash](https://github.com/plotly/jupyter-dash)
    toolkit to easily allow integration of dash apps in notebooks.
    However, there is a [known issue](https://github.com/plotly/jupyter-dash/pull/105) with jupyterDash that causes the `FigureResampler.show_dash`
    method to hang when the port is already in use. In a future Pull-Request they will hopefully fix this issue.
    We internally track this [issue](https://github.com/predict-idlab/plotly-resampler/issues/123) as well -
    please comment there if you want to provide feedback.

    In the meantime, you can use the following workaround (if you do not care about the [Werkzeug security issue](https://github.com/predict-idlab/plotly-resampler/pull/174)):
    `pip install werkzeug==2.1.2`.

??? abstract "What is the difference in approach between plotly-resampler and datashader?"

    [Datashader](https://datashader.org/getting_started/Introduction.html) is a highly scalable
    [open-source](https://github.com/holoviz/datashader) library for analyzing and visualizing large datasets.
    More specifically, datashader _“rasterizes”_ or _“aggregates”_ datasets into regular grids
    that can be analyzed further or viewed as **images**.

    **The main differences are**:

    Datashader can deal with various kinds of data (e.g., location related data, point clouds),
    whereas plotly-resampler is more tailored towards time-series data visualizations.
    Furthermore, datashader outputs a **rasterized image/array** encompassing all traces their data,
    whereas plotly-resampler outputs an **aggregated series** per trace.
    Thus, datashader is more suited for analyzing data where you do not want to pin-out a certain series/trace.

    In our opinion, datashader truly shines (for the time series use case) when:

    - you want a global, overlaying view of all your traces
    - you want to visualize a large number of time series in a single plot (many traces)
    - there is a lot of noise on your high-frequency data and you want to uncover the underlying pattern
    - you want to render all data points in your visualization

    In our opinion, plotly-resampler shines when:

    - you need the capabilities to interact with the traces (e.g., hovering, toggling traces, hovertext per trace)
    - you want to use a less complex (but more restricted) visualization interface (as opposed to holoviews), i.e., plotly
    - you want to make existing plotly time-series figures more scalable and efficient
    - to build scalable Dash apps for time-series data visualization

    Furthermore combined with holoviews, datashader can also be employed in an interactive manner, see the example below.

    ```python
       from holoviews.operation.datashader import datashade
       import datashader as ds
       import holoviews as hv
       import numpy as np
       import pandas as pd
       import panel as pn

       hv.extension("bokeh")
       pn.extension(comms='ipywidgets')

       # Create the dummy dataframe
       n = 1_000_000
       x = np.arange(n)
       noisy_sine = (np.sin(x / 3_000) + (np.random.randn(n) / 10)) * x / 5_000
       df = pd.DataFrame(
          {"ns": noisy_sine, "ns_abs": np.abs(noisy_sine),}
       )

       # Visualize interactively with datashader
       opts = hv.opts.RGB(width=800, height=400)
       ndoverlay = hv.NdOverlay({c:hv.Curve((df.index, df[c])) for c in df.columns})
       datashade(ndoverlay, cnorm='linear', aggregator=ds.count(), line_width=3).opts(opts)
    ```

    ![interactive datashader example](static/datashader.png)

??? abstract "Pandas or numpy datetime works much slower than unix epoch timestamps?"

    This stems from the plotly scatter(gl) constructor being much slower for non-numeric data.
    Plotly performs a different serialization for datetime arrays (which are interpreted as object arrays).
    However, plotly-resampler should not be limited by this - to avoid this issue,
    add your datetime data as _hf_x_ to your plotly-resampler `FigureResampler.add_trace`
    (or `FigureWidgetResampler.add_trace`) method. This avoids adding (& serializing) _all_ the data to the scatter object,
    since plotly-resampler will pass the aggregated data to the scatter object.

    Some illustration:

    ```python
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from plotly_resampler import FigureResampler

    # Create the dummy dataframe
    y = np.arange(1_000_000)
    x = pd.date_range(start="2020-01-01", periods=len(y), freq="1s")

    # Create the plotly-resampler figure
    fig = FigureResampler()
    # fig.add_trace(go.Scatter(x=x, y=y))  # This is slow
    fig.add_trace(go.Scatter(), hf_x=x, hf_y=y)  # This is fast

    # ... (add more traces, etc.)
    ```
