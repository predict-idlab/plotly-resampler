.. role:: raw-html(raw)
   :format: html

Getting started 🚀
==================

``plotly-resampler`` serves two main **modules**:

* :py:mod:`figure_resampler <plotly_resampler.figure_resampler>`: a wrapper for *plotly.graph\_objects Figures*,  coupling the dynamic resampling functionality to the *Figure*.
* :py:mod:`aggregation <plotly_resampler.aggregation>`: a module that withholds various data aggregation methods.

Installation ⚙️
---------------

Install via `pip <https://pypi.org/project/plotly-resampler>`_:

.. code:: bash

    pip install plotly-resampler

How to use 📈
-------------

Dynamic resampling callbacks are realized with either:

* `Dash <https://github.com/plotly/dash>`_ callbacks, when a ``go.Figure`` object is wrapped with dynamic aggregation functionality.

  .. note::

      This is especially useful when working with **dash functionality** or when you do **not want to solely operate in jupyter environments**.

  To **add dynamic resampling**, you should:
    1. wrap the plotly Figure with :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>`
    2. call :func:`.show_dash() <plotly_resampler.figure_resampler.FigureResampler.show_dash>` on the Figure

* `FigureWidget.layout.on_change <https://plotly.com/python-api-reference/generated/plotly.html?highlight=on_change#plotly.basedatatypes.BasePlotlyType.on_change>`_ , when a ``go.FigureWidget`` is used within a ``.ipynb`` environment.

  .. note::

      This is especially useful when developing in ``jupyter`` environments and when **you cannot open/forward a network-port**.


  To **add dynamic resampling** using a **FigureWidget**, you should:
    1. wrap your plotly Figure (can be a ``go.Figure``) with :class:`FigureWidgetResampler <plotly_resampler.figure_resampler.FigureWidgetResampler>`
    2. output the ```FigureWidgetResampler`` instance in a cell

.. tip::

  For **significant faster initial loading** of the Figure, we advise to wrap the constructor of the plotly Figure with either :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>` or :class:`FigureWidgetResampler <plotly_resampler.figure_resampler.FigureWidgetResampler>` and add the trace data as ``hf_x`` and ``hf_y``

.. note::

  Any plotly Figure can be wrapped with dynamic aggregation functionality! 🎉 :raw-html:`<br>`
  But, (obviously) only the scatter traces will be resampled.

Working examples ✅
-------------------

.. code:: py

    import plotly.graph_objects as go; import numpy as np
    from plotly_resampler import FigureResampler

    x = np.arange(1_000_000)
    sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

    fig = FigureResampler(go.Figure())
    fig.add_trace(go.Scattergl(name='noisy sine', showlegend=True), hf_x=x, hf_y=sin)

    fig.show_dash(mode='inline')

The gif below demonstrates the example usage of of :class:`FigureWidgetResampler <plotly_resampler.figure_resampler.FigureWidgetResampler>`, where ``JupyterLab`` is used as environment and the ``FigureWidgetResampler`` instance it's output is redirected into a new view. Also note how you are able to dynamically add traces!

.. image:: https://raw.githubusercontent.com/predict-idlab/plotly-resampler/main/docs/sphinx/_static/figurewidget.gif

Important considerations & tips 🚨
----------------------------------

* When running the code on a server, you should forward the port of the :func:`FigureResampler.show_dash <plotly_resampler.figure_resampler.FigureResampler.show_dash>` method to your local machine.
* In general, when using downsampling one should be aware of (possible) `aliasing <https://en.wikipedia.org/wiki/Aliasing>`_ effects. :raw-html:`<br>`
  The :raw-html:`<b><a style="color:orange">[R]</a></b>` in the legend indicates when the corresponding trace is resampled (and thus possibly distorted). :raw-html:`<br>`
  The :raw-html:`<a style="color:orange"><b>~</b> <i>delta</i></a>` suffix in the legend represents the mean index delta for consecutive aggregated data points.


Dynamically adjusting the scatter data 🔩
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The raw high-frequency trace data can be adjusted using the :func:`hf_data <plotly_resampler.figure_resampler.FigureResampler.hf_data>` property of the plotly-resampler Figure instance.

Working example ⬇️:

.. code:: py

    import plotly.graph_objects as go; import numpy as np
    from plotly_resampler import FigureResampler 
    # Note: a FigureWidgetResampler can be used here as well

    # Construct the hf-data
    x = np.arange(1_000_000)
    sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

    fig = FigureResampler(go.Figure())
    fig.add_trace(go.Scattergl(name='noisy sine', showlegend=True), hf_x=x, hf_y=sin)
    fig.show_dash(mode='inline')

    # After some time -> update the hf_data y property of the trace
    # As we only have 1 trace, this needs to be mapped
    fig.hf_data[-1]['y'] = - sin ** 2

.. Note::

    `hf_data` only withholds high-frequency traces (i.e., traces that are aggregated)

.. tip::

    The ``FigureWidgetResampler`` graph will not be automatically redrawn after 
    adjusting the fig its `hf_data` property,. The redrawning can be triggered by 
    manually calling either:

    * :func:`FigureWidgetResampler.reload_data <plotly_resampler.figure_resampler.FigureWidgetResampler.reload_data>`, which keeps the current-graph range.
    * :func:`FigureWidgetResampler.reset_axes <plotly_resampler.figure_resampler.FigureWidgetResampler.reset_axes>`, which performs a graph update.

Plotly-resampler & not high-frequency traces 🔍
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Tip::

  In the *Skin conductance example* of the :raw-html:`<a href="https://github.com/predict-idlab/plotly-resampler/tree/main/examples"><b>basic_example.ipynb</b><a>`, we deal with such low-frequency traces.

The :func:`add_trace <plotly_resampler.figure_resampler.FigureResampler.add_trace>` method allows configuring argument which allows us to deal with low-frequency traces.


Use-cases
"""""""""

* **not resampling** trace data: To achieve this, set:

  * ``max_n_samples`` = len(hf_x)

* **not resampling** trace data, but **slicing to the view**: To achieve this, set:

  * ``max_n_samples`` = len(hf_x)
  * ``limit_to_view`` = True

.. Note::
    For, **irregularly sampled traces** which are **filled** (e.g. *colored background* signal quality trace of the skin conductance example), it is important that you set ``interleave_gaps`` to ``False`` for that trace it's aggregator.

    Otherwise, when you leave ``interleave_gaps`` to ``True``, you may get weird background shapes such as ⬇️:

    .. image:: _static/skin_conductance_interleave_gaps_true.png

    When ``interleave_gaps`` is set to ``False`` you get ⬇️:

    .. image:: _static/skin_conductance_interleave_gaps_false.png
