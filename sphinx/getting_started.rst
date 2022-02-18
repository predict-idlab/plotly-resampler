Getting started 🚀
==================


*plotly-resampler* maintains its interactiveness on large data by applying front-end 
**resampling**.


Users can interact with 2 components:

* ``FigureResampler``: a wrapper for *plotly.graph\_objects* that serves the adaptive resampling functionality.
* ``downsamplers``: this module withholds various downsampling methods.

How to use 📈
-------------

To **add dynamic resampling to your plotly Figure**, you should;  

  1. wrap the constructor of your plotly Figure with ``FigureResampler``  
  2. call ``.show_dash()`` on the Figure

.. raw:: html

    <p style="color:#545454">[OPTIONAL]  add the trace data as <code>hf_x</code> and <code>hf_y</code> (for faster initial loading)</p>

Working example ✅
------------------

::

    import plotly.graph_objects as go; import numpy as np
    from plotly_resampler import FigureResampler

    x = np.arange(1_000_000)
    sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

    fig = FigureResampler(go.Figure())
    fig.add_trace(go.Scattergl(name='noisy sine', showlegend=True), hf_x=x, hf_y=sin)

    fig.show_dash(mode='inline')

Important considerations & tips 🚨
----------------------------------

* When running the code on a server, you should forward the port of the `FigureResampler.show_dash` method to your local machine.
* In general, when using downsampling one should be aware of (possible) [aliasing](https://en.wikipedia.org/wiki/Aliasing) effects.  
  The <b><a style="color:orange">[R]</a></b> in the legend indicates when the corresponding trace is being resampled (and thus possibly distorted) or not.

