.. role:: raw-html(raw)
   :format: html

Integration with a dash app 🤝
==============================

This documentation page describes how you can integrate ``plotly-resampler`` in a `dash <https://dash.plotly.com/>`_ application.

Examples of dash apps with ``plotly-resampler`` can be found in the `examples folder <https://github.com/predict-idlab/plotly-resampler/tree/main/examples>`_ of the GitHub repository.

Registering callbacks in a new dash app
---------------------------------------
When you add a :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>` figure in a basic dash app, you should;

- add a `trace-updater component <https://github.com/predict-idlab/trace-updater>`_ to the dash app layout
    - it should have as ``gID`` the id of the `dcc.Graph <https://dash.plotly.com/dash-core-components/graph>`_ component that contains the ``FigureResampler`` figure.
- register the :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>` figure its callbacks to the dash app
    - the id of the `dcc.Graph <https://dash.plotly.com/dash-core-components/graph>`_ component that contains the :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>` figure and the id of the trace-updater component should be passed.


**Code illustration**:

.. code :: python

    # Construct the to-be resampled figure
    fig = FigureResampler(px.line(...))

    # Construct app & its layout
    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(id="graph-id", figure=fig),
            trace_updater.TraceUpdater(id="trace-updater", gdID="graph-id"),
        ]
    )

    # Register the callback
    fig.register_update_graph_callback(app, "graph-id", "trace-updater")


.. tip::

    You can make the resampling faster by setting the `TraceUpdater <https://github.com/predict-idlab/trace-updater>`_ its ``sequentialUpdate`` argument to ``False``.


`This example <https://github.com/predict-idlab/trace-updater/blob/master/usage.py>`_ serves as a minimal working example.


Limitations
-----------
It is not straightforward to add callbacks dynamically in a dash app.

**Basic solution**: register the callbacks statically (and not dynamically in another callback function). 
However, it is (often) not possible to register (in a proper manner) callbacks statically.

Another solution, when statical callback registration is not an option, involves pattern matching matching callbacks. 
However, ``plotly-resampler`` currently does not support this (yet).