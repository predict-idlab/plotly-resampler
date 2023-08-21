# -*- coding: utf-8 -*-
"""
``FigureResampler`` wrapper around the plotly ``go.Figure`` class.

Creates a web-application and uses ``dash`` callbacks to enable dynamic resampling.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import dash
import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure
from trace_updater import TraceUpdater

from ..aggregation import (
    AbstractAggregator,
    AbstractGapHandler,
    MedDiffGapHandler,
    MinMaxLTTB,
)
from .figure_resampler_interface import AbstractFigureAggregator
from .utils import is_figure, is_fr

try:
    from .jupyter_dash_persistent_inline_output import JupyterDashPersistentInlineOutput

    _jupyter_dash_installed = True
except ImportError:
    _jupyter_dash_installed = False


class FigureResampler(AbstractFigureAggregator, go.Figure):
    """Data aggregation functionality for ``go.Figures``."""

    def __init__(
        self,
        figure: BaseFigure | dict = None,
        convert_existing_traces: bool = True,
        default_n_shown_samples: int = 1000,
        default_downsampler: AbstractAggregator = MinMaxLTTB(),
        default_gap_handler: AbstractGapHandler = MedDiffGapHandler(),
        resampled_trace_prefix_suffix: Tuple[str, str] = (
            '<b style="color:sandybrown">[R]</b> ',
            "",
        ),
        show_mean_aggregation_size: bool = True,
        convert_traces_kwargs: dict | None = None,
        verbose: bool = False,
        xaxis_overview_kwargs: dict = {"visible": False},
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
            !!! note
                - This can be overridden within the [`add_trace`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.add_trace] method.
                - If a trace withholds fewer datapoints than this parameter,
                  the data will *not* be aggregated.
        default_downsampler: AbstractAggregator, optional
            An instance which implements the AbstractAggregator interface and
            will be used as default downsampler, by default ``MinMaxLTTB`` with
            ``MinMaxLTTB`` is a heuristic to the LTTB algorithm that uses pre-selection
            of min-max values (default 4 per bin) to speed up LTTB (as now only 4 values
            per bin are considered by LTTB). This min-max ratio of 4 can be changed by
            initializing ``MinMaxLTTB`` with a different value for the ``minmax_ratio``
            parameter. \n
            !!! note
                This can be overridden within the [`add_trace`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.add_trace] method.
        default_gap_handler: AbstractGapHandler, optional
            An instance which implements the AbstractGapHandler interface and
            will be used as default gap handler, by default ``MedDiffGapHandler``.
            ``MedDiffGapHandler`` will determine gaps by first calculating the median
            aggregated x difference and then thresholding the aggregated x delta on a
            multiple of this median difference.  \n
            !!! note
                This can be overridden within the [`add_trace`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.add_trace] method.
        resampled_trace_prefix_suffix: str, optional
            A tuple which contains the ``prefix`` and ``suffix``, respectively, which
            will be added to the trace its legend-name when a resampled version of the
            trace is shown. By default a bold, orange ``[R]`` is shown as prefix
            (no suffix is shown).
        show_mean_aggregation_size: bool, optional
            Whether the mean aggregation bin size will be added as a suffix to the trace
            its legend-name, by default True.
        convert_traces_kwargs: dict, optional
            A dict of kwargs that will be passed to the [`add_trace`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.add_trace] method and
            will be used to convert the existing traces. \n
            !!! note
                This argument is only used when the passed ``figure`` contains data and
                ``convert_existing_traces`` is set to True.
        verbose: bool, optional
            Whether some verbose messages will be printed or not, by default False.
        xaxis_overview_kwargs: dict, optional
            TODO - write docs
        show_dash_kwargs: dict, optional
            A dict that will be used as default kwargs for the [`show_dash`][figure_resampler.figure_resampler.FigureResampler.show_dash] method.
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
            default_gap_handler,
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

        # TODO: find the proper way to keep these kwargs
        self._xaxis_overview_visible = xaxis_overview_kwargs.get("visible", False)

        # array representing the row indices per column (START AT 0) of the subplot
        # that should be linked with the columns corresponding xaxis overview
        # by default, the first row (i.e. index 0) will be utilized for each column
        self._xaxis_overview_linked_subplots = self._check_linked_indices_valid(
            xaxis_overview_kwargs.get("linked_subplots", None)
        )

        # The FigureResampler needs a dash app
        self._app: dash.Dash | None = None
        self._port: int | None = None
        self._host: str | None = None
        # Certain functions will be different when using persistent inline
        # (namely `show_dash` and `stop_callback`)
        self._is_persistent_inline = False

    def _get_subplot_rows_and_cols_from_grid(self) -> Tuple[int, int]:
        """Get the number of rows and columns of the figure's grid.

        Returns
        -------
        Tuple[int, int]
            The number of rows and columns of the figure's grid, respectively.
        """
        if self._grid_ref is None:
            return (1, 1)
        # TODO: not 100% sure whether this is correct
        return (len(self._grid_ref), len(self._grid_ref[0]))

    def _check_linked_indices_valid(self, linked_indices: list = None) -> List[int]:
        """Verify whether the passed linked indices are valid.

        Parameters
        ----------
        linked_indices: list, optional
            A list of integers representing the row indices per column (START AT 0) of
            the figure that should be linked with the columns corresponding xaxis
            overview.

        Returns
        -------
        List[int]
            A list of integers representing the row indices per column (START AT 0)

        """
        n_rows, n_cols = self._get_subplot_rows_and_cols_from_grid()

        # By default, the first row is utilized to set the linked indices
        if linked_indices is None:
            return [0] * n_cols

        # perform some checks on the linked indices
        assert isinstance(linked_indices, list), "linked indices must be a list"
        assert (
            len(linked_indices) == n_cols
        ), "the number of linked indices must be equal to the number of columns"
        assert all(
            [li < n_rows for li in linked_indices]
        ), "all linked indices must be smaller than the number of rows"

        return linked_indices

    # determines which subplot data to take from main and put into coarse
    def _remove_other_axes_for_coarse(self) -> go.Figure:
        fig_dict = self._get_current_graph()
        # base case: no rows and cols to filter
        if self._grid_ref is None:
            # TODO check whether this dict return works
            return fig_dict

        # 'data_indices' -> list of lists of indices of the subplots will be used for
        # the coarse graph
        trace_list, l_xaxis_list, l_yaxis_list, reduced_grid_ref = [], [], [], [[]]
        for col_idx, row_idx in enumerate(self._xaxis_overview_linked_subplots):
            reduced_grid_ref[0].append(self._grid_ref[row_idx][col_idx])
            for subplot in self._grid_ref[row_idx][col_idx]:
                trace_list.append(subplot.trace_kwargs["xaxis"])
                xaxis_key, yaxis_key = subplot.layout_keys
                l_yaxis_list.append(yaxis_key)
                l_xaxis_list.append(xaxis_key)
        print("layout_list", l_xaxis_list, l_yaxis_list)
        print("trace_list", trace_list)

        # copy the data from the relevant subplots
        reduced_fig_dict = {
            "data": [],
            "layout": {"template": fig_dict["layout"]["template"]},
        }
        for i, trace in enumerate(fig_dict["data"]):
            if trace.get("xaxis", "x") in trace_list:
                if "line" not in trace:
                    trace["line"] = {}
                trace["line"]["color"] = (
                    self._layout_obj.template.layout.colorway[i]
                    if self.data[i].line.color is None
                    else self.data[i].line.color
                )
                reduced_fig_dict["data"].append(trace)
        for k, v in fig_dict["layout"].items():
            if k in l_xaxis_list:
                reduced_fig_dict["layout"][k] = v
            elif k in l_yaxis_list:
                v = v.copy()
                v.update({"domain": [0, 1]})
                reduced_fig_dict["layout"][k] = v

        # create a figure object with those
        reduced_fig = go.Figure(layout=reduced_fig_dict["layout"])
        reduced_fig._grid_ref = reduced_grid_ref
        reduced_fig._data_validator.set_uid = False
        reduced_fig.add_traces(reduced_fig_dict["data"])
        return reduced_fig

    def toggle_overview(self, overview_kwargs: dict):
        self._xaxis_overview_visible = overview_kwargs.get("visible", False)
        if self._xaxis_overview_visible:
            linked_subplots = overview_kwargs.get("linked_subplots", None)
            # only make changes in the overviews if they are needed, otherwise keep the previous configuration
            if linked_subplots is not None:
                self._xaxis_overview_linked_subplots = self._check_linked_indices_valid(
                    linked_subplots
                )

    def _create_overview_figure(self) -> go.Figure:
        # create a new coarse fig
        reduced_fig = self._remove_other_axes_for_coarse()

        # Resample the coarse figure
        coarse_fig_hf = FigureResampler(
            reduced_fig,
            default_n_shown_samples=3 * self._global_n_shown_samples,
        )
        import time

        t0 = time.time()
        # NOTE: this way we can alter props without altering the original hf data
        coarse_fig_hf._hf_data = {uid: trc.copy() for uid, trc in self._hf_data.items()}
        print("time to copy", round((time.time() - t0) * 1e6, 2), "us")
        for trace in coarse_fig_hf.hf_data:
            trace["max_n_samples"] *= 3

        coarse_fig_dict = coarse_fig_hf._get_current_graph()
        print(coarse_fig_hf._check_update_figure_dict(coarse_fig_dict))
        print("coarse fig data size", len(coarse_fig_dict["data"][0]["x"]))

        coarse_fig = go.Figure(layout=coarse_fig_dict["layout"])
        coarse_fig._grid_ref = reduced_fig._grid_ref
        coarse_fig._data_validator.set_uid = False
        coarse_fig.add_traces(coarse_fig_dict["data"])
        # coarse_fig.show()

        # TODO -> look into these margin props
        # TODO -> check if we also copy the layout of the main graph
        # coarse_fig.update_layout(margin=dict(l=0, r=0, b=0, t=40, pad=10))
        # height of the overview scales with the height of the dynamic view
        coarse_fig.update_layout(showlegend=False, height=250)
        coarse_fig.update_layout(
            hovermode=False,
            clickmode="event+select",
            dragmode="select",
            activeselection=dict(fillcolor="coral", opacity=0.2),
        )

        for col_idx, row_idx in enumerate(self._xaxis_overview_linked_subplots):
            # we will only use the first grid-ref (as we will otherwsie have multiple overlapping selection boxes
            for subplot in self._grid_ref[row_idx][col_idx][:1]:
                xaxis_key, yaxis_key = subplot.layout_keys

                # set the fixed range to True
                coarse_fig["layout"][xaxis_key]["fixedrange"] = True
                coarse_fig["layout"][yaxis_key]["fixedrange"] = True

        # adds a rangeslider to the coarse graph
        coarse_fig._config = coarse_fig._config.update(
            {"modeBarButtonsToAdd": ["drawrect", "select2d"]}
        )
        return coarse_fig

    def show_dash(
        self,
        mode=None,
        config: dict | None = None,
        graph_properties: dict | None = None,
        **kwargs,
    ):
        """Registers the `update_graph` callback & show the figure in a dash app.

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

                !!! note

                    This mode requires the ``kaleido`` and ``flask_cors`` package.
                    Install them : ``pip install plotly_resampler[inline_persistent]``
                    or ``pip install kaleido flask_cors``.

              * ``"jupyterlab"``: The app will be displayed in a dedicated tab in the
                JupyterLab interface. Requires JupyterLab and the ``jupyterlab-dash``
                extension.
            By default None, which will result in the same behavior as ``"external"``.
        config: dict, optional
            The configuration options for displaying this figure, by default None.
            This ``config`` parameter is the same as the dict that you would pass as
            ``config`` argument to the `show` method.
            See more [https://plotly.com/python/configuration-options/](https://plotly.com/python/configuration-options/)
        graph_properties: dict, optional
            Dictionary of (keyword, value) for the properties that should be passed to
            the dcc.Graph, by default None.
            e.g.: `{"style": {"width": "50%"}}`
            Note: "config" is not allowed as key in this dict, as there is a distinct
            ``config`` parameter for this property in this method.
            See more [https://dash.plotly.com/dash-core-components/graph](https://dash.plotly.com/dash-core-components/graph)
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
        assert "config" not in graph_properties  # There is a param for config

        # 0. Check if the traces need to be updated when there is a xrange set
        # This will be the case when the users has set a xrange (via the `update_layout`
        # or `update_xaxes` methods`)
        relayout_dict = {}
        for xaxis_str in self._xaxis_list:
            x_range = self.layout[xaxis_str].range
            if x_range:  # when not None
                relayout_dict[f"{xaxis_str}.range[0]"] = x_range[0]
                relayout_dict[f"{xaxis_str}.range[1]"] = x_range[1]
        if relayout_dict:  # when not empty
            update_data = self.construct_update_data(relayout_dict)

            if not self._is_no_update(update_data):  # when there is an update
                with self.batch_update():
                    # First update the layout (first item of update_data)
                    self.layout.update(self._parse_relayout(update_data[0]))

                    # Then update the data
                    for updated_trace in update_data[1:]:
                        trace_idx = updated_trace.pop("index")
                        self.data[trace_idx].update(updated_trace)

        # 1. Construct the Dash app layout
        # Create an asset folder relative to current file
        assets_folder = Path(__file__).parent.joinpath("assets").absolute().__str__()
        # print("assets", assets_folder, "\cwd", os.getcwd(), '\n', os.path.relpath(assets_folder, os.getcwd()))
        init_kwargs = {}
        if self._xaxis_overview_visible:
            init_kwargs["assets_folder"] = os.path.relpath(assets_folder, os.getcwd())

        if mode == "inline_persistent":
            mode = "inline"
            if _jupyter_dash_installed:
                # Inline persistent mode: we display a static image of the figure when the
                # app is not reachable
                # Note: this is the "inline" behavior of JupyterDashInlinePersistentOutput
                app = JupyterDashPersistentInlineOutput("local_app", **init_kwargs)
                self._is_persistent_inline = True
            else:
                # If Jupyter Dash is not installed, inline persistent won't work and hence
                # we default to normal inline mode with a normal Dash app
                app = dash.Dash("local_app", **init_kwargs)
                warnings.warn(
                    "'jupyter_dash' is not installed. The persistent inline mode will not work. Defaulting to standard inline mode."
                )
        else:
            # jupyter dash uses a normal Dash app as figure
            app = dash.Dash("local_app", **init_kwargs)

        div = dash.html.Div(
            [
                dash.dcc.Graph(
                    id="resample-figure", figure=self, config=config, **graph_properties
                ),
                TraceUpdater(
                    id="trace-updater", gdID="resample-figure", sequentialUpdate=False
                ),
            ]
        )
        if self._xaxis_overview_visible:
            coarse_fig = self._create_overview_figure()
            print("add coarse figure")
            div.children += [
                # This store contains the linked subplots for which the zoom will occur
                dash.dcc.Graph(
                    id="overview-figure",
                    figure=coarse_fig,
                    config=config,
                    **graph_properties,
                ),
            ]
        app.layout = div

        self.register_update_graph_callback(
            app,
            "resample-figure",
            "trace-updater",
            "overview-figure" if self._xaxis_overview_visible else None,
        )

        height_param = "height" if self._is_persistent_inline else "jupyter_height"

        # 2. Run the app
        if mode == "inline" and height_param not in kwargs:
            # If app height is not specified -> re-use figure height for inline dash app
            #  Note: default layout height is 450 (whereas default app height is 650)
            #  See: https://plotly.com/python/reference/layout/#layout-height
            fig_height = self.layout.height if self.layout.height is not None else 450
            kwargs[height_param] = fig_height + 18

        # kwargs take precedence over the show_dash_kwargs
        kwargs = {**self._show_dash_kwargs, **kwargs}

        # Store the app information, so it can be killed
        self._app = app
        self._host = kwargs.get("host", "127.0.0.1")
        self._port = kwargs.get("port", "8050")

        # function signature is slightly different for the Dash and JupyterDash implementations
        if self._is_persistent_inline:
            app.run(mode=mode, **kwargs)
        else:
            app.run(jupyter_mode=mode, **kwargs)

    def stop_server(self, warn: bool = True):
        """Stop the running dash-app.

        Parameters
        ----------
        warn: bool
            Whether a warning message will be shown or  not, by default True.

        !!! warning

            This only works if the dash-app was started with [`show_dash`][figure_resampler.figure_resampler.FigureResampler.show_dash].
        """
        if self._app is not None:
            servers_dict = (
                self._app._server_threads
                if self._is_persistent_inline
                else dash.jupyter_dash._servers
            )
            old_server = servers_dict.get((self._host, self._port))
            if old_server:
                if self._is_persistent_inline:
                    old_server.kill()
                    old_server.join()
                else:
                    old_server.shutdown()
            del servers_dict[(self._host, self._port)]
        elif warn:
            warnings.warn(
                "Could not stop the server, either the \n"
                + "\t- 'show-dash' method was not called, or \n"
                + "\t- the dash-server wasn't started with 'show_dash'"
            )

    def register_update_graph_callback(
        self,
        app: dash.Dash,
        graph_id: str,
        trace_updater_id: str,
        coarse_graph_id: Optional[str] = None,
    ):
        """Register the [`construct_update_data`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.construct_update_data] method as callback function to
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
        coarse_graph_id: str, optional
            The id of the ``dcc.Graph``-component which withholds the coarse overview
            Figure, by default None.

        """
        if self._xaxis_overview_visible:
            # update pr graph range with overview selection
            app.clientside_callback(
                dash.ClientsideFunction(
                    namespace="clientside", function_name="coarse_to_main"
                ),
                dash.Output(graph_id, "id", allow_duplicate=True),
                dash.Input(coarse_graph_id, "selectedData"),
                dash.State(graph_id, "id"),
                dash.State(coarse_graph_id, "id"),
                prevent_initial_call=True,
            )

            # add external scripts to the app
            # TODO - check if this is really necessary
            app.config.external_scripts.append(
                {
                    "src": "https://raw.githubusercontent.com/lodash/lodash/4.17.15-npm/core.js"
                }
            )
            app.scripts.serve_locally = True

            # update selectbox with clientside callback
            app.clientside_callback(
                dash.ClientsideFunction(
                    namespace="clientside", function_name="main_to_coarse"
                ),
                dash.Output(coarse_graph_id, "id", allow_duplicate=True),
                dash.Input(graph_id, "relayoutData"),
                dash.State(coarse_graph_id, "id"),
                dash.State(graph_id, "id"),
                prevent_initial_call=True,
            )

        app.callback(
            dash.Output(trace_updater_id, "updateData"),
            dash.Input(graph_id, "relayoutData"),
            prevent_initial_call=True,
        )(self.construct_update_data)

    def _get_pr_props_keys(self) -> List[str]:
        # Add the additional plotly-resampler properties of this class
        return super()._get_pr_props_keys() + ["_show_dash_kwargs"]

    def _ipython_display_(self):
        # To display the figure inline as a dash app
        self.show_dash(mode="inline")
