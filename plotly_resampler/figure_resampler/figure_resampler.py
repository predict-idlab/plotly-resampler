# -*- coding: utf-8 -*-
"""
``FigureResampler`` wrapper around the plotly ``go.Figure`` class.

Creates a web-application and uses ``dash`` callbacks to enable dynamic resampling.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import warnings
from typing import Tuple, List

import uuid
import base64
import dash
import plotly.graph_objects as go
from flask_cors import cross_origin
from jupyter_dash import JupyterDash
from plotly.basedatatypes import BaseFigure
from trace_updater import TraceUpdater

from ..aggregation import AbstractSeriesAggregator, EfficientLTTB
from .figure_resampler_interface import AbstractFigureAggregator
from .utils import is_figure, is_fr


class JupyterDashPersistentInlineOutput(JupyterDash):
    """Extension of the JupyterDash class to support the custom inline output for
    ``FigureResampler`` figures.

    Specifically we embed a div in the notebook to display the figure inline.

     - In this div the figure is shown as an iframe when the server (of the dash app)
       is alive.
     - In this div the figure is shown as an image when the server (of the dash app)
       is dead.

    As the HTML & javascript code is embedded in the notebook output, which is loaded
    each time you open the notebook, the figure is always displayed (either as iframe
    or just an image).
    Hence, this extension enables to maintain always an output in the notebook.

    .. Note::
        This subclass is only used when the mode is set to ``"inline_persistent"`` in
        the :func:`FigureResampler.show_dash <plotly_resampler.figure_resampler.FigureResampler.show_dash>`
        method. However, the mode should be passed as ``"inline"`` since this subclass
        overwrites the inline behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._uid = str(uuid.uuid4())  # A new unique id for each app

        # Mimic the _alive_{token} endpoint but with cors
        @self.server.route(f"/_is_alive_{self._uid}", methods=["GET"])
        @cross_origin(origin=["*"], allow_headers=["Content-Type"])
        def broadcast_alive():
            return "Alive"

    def _display_inline_output(self, dashboard_url, width, height):
        """Display the dash app persistent inline in the notebook.

        The figure is displayed as an iframe in the notebook if the server is reachable,
        otherwise as an image.
        """
        # TODO: check whether an error gets logged in case of crash
        # TODO: add option to opt out of this
        from IPython.display import display

        # Get the image from the dashboard and encode it as base64
        fig = self.layout.children[0].figure  # is stored there in the show_dash method
        f_width = 1000 if fig.layout.width is None else fig.layout.width
        fig_base64 = base64.b64encode(
            fig.to_image(format="png", width=f_width, scale=1, height=fig.layout.height)
        ).decode("utf8")

        # The unique id of this app
        # This id is used to couple the output in the notebook with this app
        # A fetch request is performed to the _is_alive_{uid} endpoint to check if the
        # app is still alive.
        uid = self._uid

        # The html (& javascript) code to display the app / figure
        display(
            {
                "text/html": f"""
                <div id='PR_div__{uid}'></div>
                <script type='text/javascript'>
                """
                + """

                function setOutput(timeout) {
                    """
                +
                # Variables should be in local scope (in the closure)
                f"""
                    var pr_div = document.getElementById('PR_div__{uid}');
                    var url = '{dashboard_url}';
                    var pr_img_src = 'data:image/png;base64, {fig_base64}';
                    var is_alive_suffix = '_is_alive_{uid}';
                    """
                + """

                    if (pr_div.firstChild) return  // return if already loaded

                    const controller = new AbortController();
                    const signal = controller.signal;

                    return fetch(url + is_alive_suffix, {method: 'GET', signal: signal})
                        .then(response => response.text())
                        .then(data =>
                            {
                                if (data == "Alive") {
                                    console.log("Server is alive");
                                    iframeOutput(pr_div, url);
                                } else {
                                    // I think this case will never occur because of CORS
                                    console.log("Server is dead");
                                    imageOutput(pr_div, pr_img_src);
                                }
                            }
                        )
                        .catch(error => {
                            console.log("Server is unreachable");
                            imageOutput(pr_div, pr_img_src);
                        })
                }

                setOutput(350);

                function imageOutput(element, pr_img_src) {
                    console.log('Setting image');
                    var pr_img = document.createElement("img");
                    pr_img.setAttribute("src", pr_img_src)
                    pr_img.setAttribute("alt", 'Server unreachable - using image instead');
                    """
                + f"""
                    pr_img.setAttribute("max-width", '{width}');
                    pr_img.setAttribute("max-height", '{height}');
                    pr_img.setAttribute("width", 'auto');
                    pr_img.setAttribute("height", 'auto');
                    """
                + """
                    element.appendChild(pr_img);
                }

                function iframeOutput(element, url) {
                    console.log('Setting iframe');
                    var pr_iframe = document.createElement("iframe");
                    pr_iframe.setAttribute("src", url);
                    pr_iframe.setAttribute("frameborder", '0');
                    pr_iframe.setAttribute("allowfullscreen", '');
                    """
                + f"""
                    pr_iframe.setAttribute("width", '{width}');
                    pr_iframe.setAttribute("height", '{height}');
                    """
                + """
                    element.appendChild(pr_iframe);
                }
                </script>
                """
            },
            raw=True,
            clear=True,
            display_id=uid,
        )

    def _display_in_jupyter(self, dashboard_url, port, mode, width, height):
        """Override the display method to retain some output when displaying inline
        in jupyter.
        """
        if mode == "inline":
            self._display_inline_output(dashboard_url, width, height)
        else:
            super()._display_in_jupyter(dashboard_url, port, mode, width, height)


class FigureResampler(AbstractFigureAggregator, go.Figure):
    """Data aggregation functionality for ``go.Figures``."""

    def __init__(
        self,
        figure: BaseFigure | dict = None,
        convert_existing_traces: bool = True,
        default_n_shown_samples: int = 1000,
        default_downsampler: AbstractSeriesAggregator = EfficientLTTB(),
        resampled_trace_prefix_suffix: Tuple[str, str] = (
            '<b style="color:sandybrown">[R]</b> ',
            "",
        ),
        show_mean_aggregation_size: bool = True,
        convert_traces_kwargs: dict | None = None,
        verbose: bool = False,
        show_dash_kwargs: dict | None = None,
    ):
        """Initialize a dynamic aggregation data mirror using a dash web app.

        Parameters
        ----------
        figure: BaseFigure
            The figure that will be decorated. Can be either an empty figure
            (e.g., ``go.Figure()``, ``make_subplots()``, ``go.FigureWidget``) or an
            existing figure.
        convert_existing_traces: bool
            A bool indicating whether the high-frequency traces of the passed ``figure``
            should be resampled, by default True. Hence, when set to False, the
            high-frequency traces of the passed ``figure`` will not be resampled.
        default_n_shown_samples: int, optional
            The default number of samples that will be shown for each trace,
            by default 1000.\n
            .. note::
                * This can be overridden within the :func:`add_trace` method.
                * If a trace withholds fewer datapoints than this parameter,
                  the data will *not* be aggregated.
        default_downsampler: AbstractSeriesDownsampler
            An instance which implements the AbstractSeriesDownsampler interface and
            will be used as default downsampler, by default ``EfficientLTTB`` with
            _interleave_gaps_ set to True. \n
            .. note:: This can be overridden within the :func:`add_trace` method.
        resampled_trace_prefix_suffix: str, optional
            A tuple which contains the ``prefix`` and ``suffix``, respectively, which
            will be added to the trace its legend-name when a resampled version of the
            trace is shown. By default a bold, orange ``[R]`` is shown as prefix
            (no suffix is shown).
        show_mean_aggregation_size: bool, optional
            Whether the mean aggregation bin size will be added as a suffix to the trace
            its legend-name, by default True.
        convert_traces_kwargs: dict, optional
            A dict of kwargs that will be passed to the :func:`add_traces` method and
            will be used to convert the existing traces. \n
            .. note::
                This argument is only used when the passed ``figure`` contains data and
                ``convert_existing_traces`` is set to True.
        verbose: bool, optional
            Whether some verbose messages will be printed or not, by default False.
        show_dash_kwargs: dict, optional
            A dict that will be used as default kwargs for the :func:`show_dash` method.
            Note that the passed kwargs will be take precedence over these defaults.

        """
        # Parse the figure input before calling `super`
        if is_figure(figure) and not is_fr(figure):
            # A go.Figure
            # => base case: the figure does not need to be adjusted
            f = figure
        else:
            # Create a new figure object and make sure that the trace uid will not get
            # adjusted when they are added.
            f = self._get_figure_class(go.Figure)()
            f._data_validator.set_uid = False

            if isinstance(figure, BaseFigure):
                # A base figure object, can be;
                # - a go.FigureWidget
                # - a plotly-resampler figure: subclass of AbstractFigureAggregator
                # => we first copy the layout, grid_str and grid ref
                f.layout = figure.layout
                f._grid_str = figure._grid_str
                f._grid_ref = figure._grid_ref
                f.add_traces(figure.data)
            elif isinstance(figure, dict) and (
                "data" in figure or "layout" in figure  # or "frames" in figure  # TODO
            ):
                # A figure as a dict, can be;
                # - a plotly figure as a dict (after calling `fig.to_dict()`)
                # - a pickled (plotly-resampler) figure (after loading a pickled figure)
                # => we first copy the layout, grid_str and grid ref
                f.layout = figure.get("layout")
                f._grid_str = figure.get("_grid_str")
                f._grid_ref = figure.get("_grid_ref")
                f.add_traces(figure.get("data"))
                # `pr_props` is not None when loading a pickled plotly-resampler figure
                f._pr_props = figure.get("pr_props")
                # `f._pr_props`` is an attribute to store properties of a
                # plotly-resampler figure. This attribute is only used to pass
                # information to the super() constructor. Once the super constructor is
                # called, the attribute is removed.

                # f.add_frames(figure.get("frames")) TODO
            elif isinstance(figure, (dict, list)):
                # A single trace dict or a list of traces
                f.add_traces(figure)

        self._show_dash_kwargs = (
            show_dash_kwargs if show_dash_kwargs is not None else {}
        )

        super().__init__(
            f,
            convert_existing_traces,
            default_n_shown_samples,
            default_downsampler,
            resampled_trace_prefix_suffix,
            show_mean_aggregation_size,
            convert_traces_kwargs,
            verbose,
        )

        if isinstance(figure, AbstractFigureAggregator):
            # Copy the `_hf_data` if the previous figure was an AbstractFigureAggregator
            # and adjust the default `max_n_samples` and `downsampler`
            self._hf_data.update(
                self._copy_hf_data(figure._hf_data, adjust_default_values=True)
            )

            # Note: This hack ensures that the this figure object initially uses
            # data of the whole view. More concretely; we create a dict
            # serialization figure and adjust the hf-traces to the whole view
            # with the check-update method (by passing no range / filter args)
            with self.batch_update():
                graph_dict: dict = self._get_current_graph()
                update_indices = self._check_update_figure_dict(graph_dict)
                for idx in update_indices:
                    self.data[idx].update(graph_dict["data"][idx])

        # The FigureResampler needs a dash app
        self._app: JupyterDash | dash.Dash | None = None
        self._port: int | None = None
        self._host: str | None = None

    def show_dash(
        self,
        mode=None,
        config: dict | None = None,
        graph_properties: dict | None = None,
        **kwargs,
    ):
        """Registers the :func:`update_graph` callback & show the figure in a dash app.

        Parameters
        ----------
        mode: str, optional
            Display mode. One of:\n
              * ``"external"``: The URL of the app will be displayed in the notebook
                output cell. Clicking this URL will open the app in the default
                web browser.
              * ``"inline"``: The app will be displayed inline in the notebook output
                cell in an iframe.
              * ``"inline_persistent"``: The app will be displayed inline in the
                notebook output cell in an iframe, if the app is not reachable a static
                image of the figure is shown. Hence this is a persistent version of the
                ``"inline"`` mode, allowing users to see a static figure in other
                environments, browsers, etc.

                .. note::
                    This mode requires the ``kaleido`` package.

              * ``"jupyterlab"``: The app will be displayed in a dedicated tab in the
                JupyterLab interface. Requires JupyterLab and the ``jupyterlab-dash``
                extension.
            By default None, which will result in the same behavior as ``"external"``.
        config: dict, optional
            The configuration options for displaying this figure, by default None.
            This ``config`` parameter is the same as the dict that you would pass as
            ``config`` argument to the `show` method.
            See more https://plotly.com/python/configuration-options/
        graph_properties: dict, optional
            Dictionary of (keyword, value) for the properties that should be passed to
            the dcc.Graph, by default None.
            e.g.: {"style": {"width": "50%"}}
            Note: "config" is not allowed as key in this dict, as there is a distinct
            ``config`` parameter for this property in this method.
            See more https://dash.plotly.com/dash-core-components/graph
        **kwargs: dict
            Additional app.run_server() kwargs. e.g.: port, ...
            Also note that these kwargs take precedence over the ones passed to the
            constructor via the ``show_dash_kwargs`` argument.

        """
        available_modes = ["external", "inline", "inline_persistent", "jupyterlab"]
        assert (
            mode is None or mode in available_modes
        ), f"mode must be one of {available_modes}"
        graph_properties = {} if graph_properties is None else graph_properties
        assert "config" not in graph_properties.keys()  # There is a param for config

        # 0. Check if the traces need to be updated when there is a xrange set
        # This will be the case when the users has set a xrange (via the `update_layout`
        # or `update_xaxes` methods`)
        relayout_dict = {}
        for xaxis_str in self._xaxis_list:
            x_range = self.layout[xaxis_str].range
            if x_range:  # when not None
                relayout_dict[f"{xaxis_str}.range[0]"] = x_range[0]
                relayout_dict[f"{xaxis_str}.range[1]"] = x_range[1]
        if len(relayout_dict):
            update_data = self.construct_update_data(relayout_dict)

            if not self._is_no_update(update_data):  # when there is an update
                with self.batch_update():
                    # First update the layout (first item of update_data)
                    self.layout.update(update_data[0])

                    # Then update the data
                    for updated_trace in update_data[1:]:
                        trace_idx = updated_trace.pop("index")
                        self.data[trace_idx].update(updated_trace)

        # 1. Construct the Dash app layout
        if mode == "inline_persistent":
            # Inline persistent mode: we display a static image of the figure when the
            # app is not reachable
            # Note: this is the "inline" behavior of JupyterDashInlinePersistentOutput
            mode = "inline"
            app = JupyterDashPersistentInlineOutput("local_app")
        else:
            app = JupyterDash("local_app")
        app.layout = dash.html.Div(
            [
                dash.dcc.Graph(
                    id="resample-figure", figure=self, config=config, **graph_properties
                ),
                TraceUpdater(
                    id="trace-updater", gdID="resample-figure", sequentialUpdate=False
                ),
            ]
        )
        self.register_update_graph_callback(app, "resample-figure", "trace-updater")

        # 2. Run the app
        if mode == "inline" and "height" not in kwargs:
            # If app height is not specified -> re-use figure height for inline dash app
            #  Note: default layout height is 450 (whereas default app height is 650)
            #  See: https://plotly.com/python/reference/layout/#layout-height
            fig_height = self.layout.height if self.layout.height is not None else 450
            kwargs["height"] = fig_height + 18

        # kwargs take precedence over the show_dash_kwargs
        kwargs = {**self._show_dash_kwargs, **kwargs}

        # Store the app information, so it can be killed
        self._app = app
        self._host = kwargs.get("host", "127.0.0.1")
        self._port = kwargs.get("port", "8050")

        app.run_server(mode=mode, **kwargs)

    def stop_server(self, warn: bool = True):
        """Stop the running dash-app.

        Parameters
        ----------
        warn: bool
            Whether a warning message will be shown or  not, by default True.

        .. attention::
            This only works if the dash-app was started with :func:`show_dash`.
        """
        if self._app is not None:

            old_server = self._app._server_threads.get((self._host, self._port))
            if old_server:
                old_server.kill()
                old_server.join()
                del self._app._server_threads[(self._host, self._port)]
        elif warn:
            warnings.warn(
                "Could not stop the server, either the \n"
                + "\t- 'show-dash' method was not called, or \n"
                + "\t- the dash-server wasn't started with 'show_dash'"
            )

    def register_update_graph_callback(
        self, app: dash.Dash, graph_id: str, trace_updater_id: str
    ):
        """Register the :func:`construct_update_data` method as callback function to
        the passed dash-app.

        Parameters
        ----------
        app: Union[dash.Dash, JupyterDash]
            The app in which the callback will be registered.
        graph_id:
            The id of the ``dcc.Graph``-component which withholds the to-be resampled
            Figure.
        trace_updater_id
            The id of the ``TraceUpdater`` component. This component is leveraged by
            ``FigureResampler`` to efficiently POST the to-be-updated data to the
            front-end.

        """
        app.callback(
            dash.dependencies.Output(trace_updater_id, "updateData"),
            dash.dependencies.Input(graph_id, "relayoutData"),
            prevent_initial_call=True,
        )(self.construct_update_data)

    def _get_pr_props_keys(self) -> List[str]:
        # Add the additional plotly-resampler properties of this class
        return super()._get_pr_props_keys() + ["_show_dash_kwargs"]

    def _ipython_display_(self):
        # To display the figure inline as a dash app
        self.show_dash(mode="inline")
