.. role:: raw-html(raw)
   :format: html

Getting started 🚀
==================


``plotly-resampler`` maintains its interactiveness on large data by applying front-end 
**resampling**.


Users can interact with 2 components:

* :ref:`FigureResampler <FigureResampler>`: a wrapper for *plotly.graph\_objects* that serves the adaptive resampling functionality.
* :ref:`downsamplers <downsamplers>`: this module withholds various downsampling methods.

Installation ⚙️
---------------

Install via :raw-html:`<a href="https://pypi.org/project/plotly-resampler/"><b>pip</b><a>`:

.. code:: bash

    pip install plotly-resampler


How to use 📈
-------------

To **add dynamic resampling to a plotly Figure**, you should;  

  1. wrap the plotly Figure with :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>`
  2. call :func:`.show_dash() <plotly_resampler.figure_resampler.FigureResampler.show_dash>` on the Figure

.. tip::

  For **significant faster initial loading** of the Figure, we advise to wrap the constructor of the plotly Figure with :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>` and add the trace data as ``hf_x`` and ``hf_y``

.. note::

  Any plotly Figure can be wrapped with :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>`! 🎉 :raw-html:`<br>`
  But, (obviously) only the scatter traces will be resampled. 

Working example ✅
------------------

.. code:: py

    import plotly.graph_objects as go; import numpy as np
    from plotly_resampler import FigureResampler

    x = np.arange(1_000_000)
    sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

    fig = FigureResampler(go.Figure())
    fig.add_trace(go.Scattergl(name='noisy sine', showlegend=True), hf_x=x, hf_y=sin)

    fig.show_dash(mode='inline')

Important considerations & tips 🚨
----------------------------------

* When running the code on a server, you should forward the port of the :func:`FigureResampler.show_dash <plotly_resampler.figure_resampler.FigureResampler.show_dash>` method to your local machine.
* In general, when using downsampling one should be aware of (possible) `aliasing <https://en.wikipedia.org/wiki/Aliasing>`_ effects. :raw-html:`<br>`
  The :raw-html:`<b><a style="color:orange">[R]</a></b>` in the legend indicates when the corresponding trace is resampled (and thus possibly distorted).
