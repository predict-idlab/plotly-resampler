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

 * for *time-indexed data*: the mean time-range which is span between 2 consecutive samples.
 * for *numeric-indexed data*: the mean numeric range which is span between 2 consecutive samples.

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
  * with ``FigureWidgetResampler``, you need to use ``IPython.display`` on the object, which uses widget-events to realize dynamic aggregation.

.. raw:: html

   </div>
   </details>
   <br>
   <details>
   <summary>
      <a><b>What does <code><a href="https://github.com/predict-idlab/trace-updater" target="_blank">TraceUpdater</a></code> do?</b></a>
   </summary>
   <div style="margin-left:1em">

The ``TraceUpdater`` class is a custom dash component that aids ``dcc.Graph`` components to efficiently sent and update (in our case aggregated) data to the front-end.

For more information on how to use the trace-updater component together with the ``FigureResampler``, see our dash app `examples <https://github.com/predict-idlab/plotly-resampler/tree/main/examples>`_` and look at the `trace-updater <https://github.com/predict-idlab/trace-updater/blob/master/trace_updater/TraceUpdater.py>`_ its documentation.

.. raw:: html

   </div>
   </details>
   <br>