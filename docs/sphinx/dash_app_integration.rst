.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br>



Integration with a dash app 🤝
==============================

This documentation page describes how you can integrate ``plotly-resampler`` in a `dash <https://dash.plotly.com/>`_ application.

Examples of dash apps with ``plotly-resampler`` can be found in the `examples folder <https://github.com/predict-idlab/plotly-resampler/tree/main/examples>`_ of the GitHub repository.

Registering callbacks in a new dash app
---------------------------------------
When you add a :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>` figure in a basic dash app, you should:

- Add a `trace-updater component <https://github.com/predict-idlab/trace-updater>`_ to the dash app layout.

  - It should have as ``gID`` the id of the `dcc.Graph <https://dash.plotly.com/dash-core-components/graph>`_ component that contains the ``FigureResampler`` figure.

- Register the :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>` figure its callbacks to the dash app.

  - The id of the `dcc.Graph <https://dash.plotly.com/dash-core-components/graph>`_ component that contains the :class:`FigureResampler <plotly_resampler.figure_resampler.FigureResampler>` figure and the id of the trace-updater component should be passed to the :func:`register_update_graph_callback <plotly_resampler.figure_resampler.FigureResampler.register_update_graph_callback>` method.


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

    You can make the resampling faster by ensuring the
    `TraceUpdater <https://github.com/predict-idlab/trace-updater>`_ its
    ``sequentialUpdate`` argument is set to ``False``.


* `This TraceUpdater-example <https://github.com/predict-idlab/trace-updater/blob/master/usage.py>`_ serves as a minimal working example.


Limitations
-----------
plotly_resampler relies on ``TraceUpdater`` to ensure that the *updateData* is sent
efficiently to the front-end.

To enable dynamic-graph-construction, plotly-resampler supports
`pattern matching callbacks <https://dash.plotly.com/pattern-matching-callbacks>`_.
This could only be achieved by performing partial id matching on the graph-div ID within
the TraceUpdater component. This causes the following:

.. attention::

    TraceUpdater will determine the html-graph-div by performing **partial
    matching on the "id" property** (using `gdID`) of all divs with
    classname=\"dash-graph\". |br|
    So if multiple same graph-div IDs are used, or one graph-div-ID is a
    subset of other(s), multiple eligible *graph-divs* will be found and a
    ``SyntaxError`` will be raised.

This can be circumvented by using an ``uuid``-str for each graph-div, as done in this
`dynamic graph construction example <https://github.com/predict-idlab/plotly-resampler/blob/main/examples/dash_apps/construct_dynamic_figures.py>`_.