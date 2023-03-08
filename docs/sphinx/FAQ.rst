.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br>


FAQ ❓
======

.. raw:: html

   <details>
   <summary>
      <a><b>What does the orange <b style="color:orange">~ time|number </b> suffix in legend name indicate?</b></a>
   </summary>
   <div style="margin-left:1em">


This tilde suffix is only shown when the data is aggregated and represents the *mean aggregation bin size* which is the mean index-range difference between two consecutive aggregated samples.

 * for *time-indexed data*: the mean time-range between 2 consecutive (sampled) samples.
 * for *numeric-indexed data*: the mean numeric range between 2 consecutive (sampled) samples.

When the index is a range-index; the *mean aggregation bin size* represents the *mean* downsample ratio; i.e., the mean number of samples that are aggregated into one sample.

.. raw:: html

   </div>
   </details>
   <br>
   <details>
   <summary>
      <a><b>What is the difference between plotly-resampler figures and plain plotly figures?</b></a>
   </summary>
   <div style="margin-left:1em">

plotly-resampler can be thought of as wrapper around plain plotly figures which adds line-chart visualization scalability by dynamically aggregating the data of the figures w.r.t. the front-end view. plotly-resampler thus adds dynamic aggregation functionality to plain plotly figures.

**important to know**:

* ``show`` *always* returns a static html view of the figure, i.e., no dynamic aggregation can be performed on that view.
* To have dynamic aggregation:

  * with ``FigureResampler``, you need to call ``show_dash`` (or output the object in a cell via ``IPython.display``) -> which spawns a dash-web app, and the dynamic aggregation is realized with dash callback
  * with ``FigureWidgetResampler``, you need to use ``IPython.display`` on the object, which uses widget-events to realize dynamic aggregation (via the running IPython kernel).

**other changes of plotly-resampler figures w.r.t. vanilla plotly**:

* **double-clicking** within a line-chart area **does not Reset Axes**, as it results in an “Autoscale” event. We decided to implement an Autoscale event as updating your y-range such that it shows all the data that is in your x-range
   * **Note**: vanilla Plotly figures their Autoscale result in Reset Axes behavior, in our opinion this did not make a lot of sense. It is therefore that we have overriden this behavior in plotly-resampler.

.. raw:: html

   </div>
   </details>
   <br>
   <details>
   <summary>
      <a><b>What does <code><a href="https://github.com/predict-idlab/trace-updater" target="_blank">TraceUpdater</a></code> do?</b></a>
   </summary>
   <div style="margin-left:1em">

The ``TraceUpdater`` class is a custom dash component that aids ``dcc.Graph`` components to efficiently send and update (in our case aggregated) data to the front-end.

For more information on how to use the trace-updater component together with the ``FigureResampler``, see our dash app `examples <https://github.com/predict-idlab/plotly-resampler/tree/main/examples>`_` and look at the `trace-updater <https://github.com/predict-idlab/trace-updater/blob/master/trace_updater/TraceUpdater.py>`_ its documentation.

.. raw:: html

   </div>
   </details>
   <br>
   <details>
   <summary>
      <a><b>My <code>FigureResampler.show_dash</code> keeps hanging (indefinitely) with the error message:<br>&nbsp;&nbsp;&nbsp; <code>OSError: Port already in use</code></b></a>
   </summary>
   <div style="margin-left:1em">

Plotly-resampler its ``FigureResampler.show_dash`` method leverages the `jupyterdash <https://github.com/plotly/jupyter-dash>`_ toolkit to easily allow integration of dash apps in notebooks. However, there is a `known issue <https://github.com/plotly/jupyter-dash/pull/105>`_ with jupyterDash that causes the ``FigureResampler.show_dash`` method to hang when the port is already in use. In a future Pull-Request they will hopefully fix this issue. We internally track `this issue <https://github.com/predict-idlab/plotly-resampler/issues/123>` as well - please comment there if you want to provide feedback. 

In the meantime, you can use the following workaround (if you do not care about the `Werkzeug security issue <https://github.com/predict-idlab/plotly-resampler/pull/174>`_): `pip install werkzeug==2.1.2`.

.. raw:: html

   </div>
   </details>
   <br>
      <details>
   <summary>
      <a><b>What is the difference in approach between plotly-resampler and datashader?</b></a>
   </summary>
   <div style="margin-left:1em">


`Datashader <https://datashader.org/getting_started/Introduction.html>`_ is a highly scalable `open-source <https://github.com/holoviz/datashader>`_ library for analyzing and visualizing large datasets. More specifically, datashader *“rasterizes”* or *“aggregates”* datasets into regular grids that can be analyzed further or viewed as **images**. 


**The main differences are**:

Datashader can deal with various kinds of data (e.g., location related data, point clouds), whereas plotly-resampler is more tailored towards time-series data visualizations. 
Furthermore, datashader outputs a **rasterized image/array** encompassing all traces their data, whereas plotly-resampler outputs an **aggregated series** per trace. Thus, datashader is more suited for analyzing data where you do not want to pin-out a certain series/trace.

In our opinion, datashader truly shines (for the time series use case) when:

* you want a global, overlaying view of all your traces
* you want to visualize a large number of time series in a single plot (many traces)
* there is a lot of noise on your high-frequency data and you want to uncover the underlying pattern
* you want to render all data points in your visualization

In our opinion, plotly-resampler shines when:

* you need the capabilities to interact with the traces (e.g., hovering, toggling traces, hovertext per trace)
* you want to use a less complex (but more restricted) visualization interface (as opposed to holoviews), i.e., plotly
* you want to make existing plotly time-series figures more scalable and efficient
* to build scalable Dash apps for time-series data visualization

Furthermore combined with holoviews, datashader can also be employed in an interactive manner, see the example below.

.. code:: python

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

.. image:: _static/datashader.png


.. raw:: html

   </div>
   </details>
   <br>
   <details>
   <summary>
      <a><b>I get errors such as:</b><br><ul><li>
         <code>RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xe</code></li>  
         <li><code>ImportError: numpy.core.multiarray failed to import</code></li>
         </ul>
      </a>
   </summary>
   <div style="margin-left:1em">

   Plotly-resampler uses compiled C code (which uses the NumPy C API) to speed up the LTTB data-aggregation algorithm. This C code gets compiled during the building stage of the package (which might be before you install the package).<br><br>
   If this C extension was build against a more recent NumPy version than your local version, you obtain a 
   <a href="https://numpy.org/devdocs/user/troubleshooting-importerror.html#c-api-incompatibility"><i>NumPy C-API incompatibility</i></a> 
   and the above error will be raised.<br><br>

   These above mentioned errors can thus be resolved by running<br>
   &nbsp;&nbsp;&nbsp;&nbsp;<code>pip install --upgrade numpy</code><br>
   and reinstalling plotly-resampler afterwards.<br><br>

   For more information about compatibility and building upon NumPy, you can consult 
   <a href="https://numpy.org/doc/stable/user/depending_on_numpy.html?highlight=compiled#for-downstream-package-authors">NumPy's docs for downstream package authors</a>.

   We aim to limit this issue as much as possible (by for example using <a href="https://github.com/scipy/oldest-supported-numpy">oldest-supported-numpy</a> in our build.py), 
   but if you still experience issues, please open an issue on <a href="https://github.com/predict-idlab/plotly-resampler/issues">GitHub</a>.

.. raw:: html

   </div>
   </details>
   <br>
