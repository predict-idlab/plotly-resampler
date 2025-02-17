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
from typing import List, Optional, Tuple, Union

import dash
import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure

from ..aggregation import (
    AbstractAggregator,
    AbstractGapHandler,
    MedDiffGapHandler,
    MinMaxLTTB,
)
from .figure_resampler_interface import AbstractFigureAggregator
from .jupyter_dash_persistent_inline_output import JupyterDashPersistentInlineOutput
from .utils import is_figure, is_fr

# Default arguments for the Figure overview
ASSETS_FOLDER = Path(__file__).parent.joinpath("assets").absolute().__str__()
_DEFAULT_OVERVIEW_LAYOUT_KWARGS = {
    "showlegend": False,
    "height": 120,
    "activeselection": dict(fillcolor="#96C291", opacity=0.3),
    "margin": {"t": 0, "b": 0},
}


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
        create_overview: bool = False,
        overview_row_idxs: list = None,
        overview_kwargs: dict = {},
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
        create_overview: bool, optional
            Whether an overview will be added to the figure (also known as rangeslider),
            by default False. An overview is a bidirectionally linked figure that is
            placed below the FigureResampler figure and shows a coarse version on which
            the current view of the FigureResampler figure is highlighted. The overview
            can be used to quickly navigate through the data by dragging the selection
            box.
            !!! note
                - In the case of subplots, the overview will be created for each subplot
                  column. Only a single subplot row can be captured in the overview,
                  this is by default the first row. If you want to customize this
                  behavior, you can use the `overview_row_idxs` argument.
                - This functionality is not yet extensively validated. Please report any
                  issues you encounter on GitHub.
        overview_row_idxs: list, optional
            A list of integers corresponding to the row indices (START AT 0) of the
            subplots columns that should be linked with the column its corresponding
            overview. By default None, which will result in the first row being utilized
            for each column.
        overview_kwargs: dict, optional
            A dict of kwargs that will be passed to the `update_layout` method of the
            overview figure, by default {}, which will result in utilizing the
            [`default`][_DEFAULT_OVERVIEW_LAYOUT_KWARGS] overview layout kwargs.
        verbose: bool, optional
            Whether some verbose messages will be printed or not, by default False.
        show_dash_kwargs: dict, optional
            A dict that will be used as default kwargs for the [`show_dash`][figure_resampler.figure_resampler.FigureResampler.show_dash] method.
            !!! note
                The passed kwargs to the [`show_dash`][figure_resampler.figure_resampler.FigureResampler.show_dash] method will take precedence over these defaults.

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

        self._create_overview = create_overview
        # update the overview layout
        overview_layout_kwargs = _DEFAULT_OVERVIEW_LAYOUT_KWARGS.copy()
        overview_layout_kwargs.update(overview_kwargs)
        self._overview_layout_kwargs = overview_layout_kwargs

        # array representing the row indices per column (START AT 0) of the subplot
        # that should be linked with the columns corresponding overview.
        # By default, the first row (i.e. index 0) will be utilized for each column
        self._overview_row_idxs = self._parse_subplot_row_indices(overview_row_idxs)

        # The FigureResampler needs a dash app
        self._app: dash.Dash | None = None
        self._port: int | None = None
        self._host: str | None = None
        # Certain functions will be different when using persistent inline
        # (namely `show_dash` and `stop_callback`)

    def _get_subplot_rows_and_cols_from_grid(self) -> Tuple[int, int]:
        """Get the number of rows and columns of the figure's grid.

        Returns
        -------
        Tuple[int, int]
            The number of rows and columns of the figure's grid, respectively.
        """
        if self._grid_ref is None:  # case: go.Figure (no subplots)
            return (1, 1)
        # TODO: not 100% sure whether this is correct
        return (len(self._grid_ref), len(self._grid_ref[0]))

    def _parse_subplot_row_indices(self, row_indices: list = None) -> List[int]:
        """Verify whether the passed row indices are valid.

        Parameters
        ----------
        row_indices: list, optional
            A list of integers representing the row indices for which the overview
            should be created. The length of the list should be equal to the number of
            columns of the figure. Each element of the list should be smaller than the
            number of rows of the figure (thus note that the row indices start at 0). By
            default None, which will result in the first row being utilized for each
            column.
            !!! note
                When you do not want to use an overview of a certain column (because
                a certain subplot spans more than 1 column), you can specify this by
                setting that respecive row_index value to `None`.

                For instance, the sbuplot on row 2, col 1 spans two coloms. So when you
                intend to utilize that subplot within the overview, you want to specify
                the row_indices as: `[1, None, ...]`

        Returns
        -------
        List[int]
            A list of integers representing the row indices per subplot column.

        """
        n_rows, n_cols = self._get_subplot_rows_and_cols_from_grid()

        # By default, the first row is utilized to set the row indices
        if row_indices is None:
            return [0] * n_cols

        # perform some checks on the row indices
        assert isinstance(row_indices, list), "row indices must be a list"
        assert (
            len(row_indices) == n_cols
        ), "the number of row indices must be equal to the number of columns"
        assert all(
            [(li is None) or (0 <= li < n_rows) for li in row_indices]
        ), "row indices must be smaller than the number of rows"

        return row_indices

    # determines which subplot data to take from main and put into coarse
    def _remove_other_axes_for_coarse(self) -> go.Figure:
        # base case: no rows and cols to filter
        if self._grid_ref is None:  # case: go.Figure (no subplots)
            return self

        # Create the grid specification for the overview figure (in `reduced_grid_ref`)
        # The trace_list and the 2 axis lists are 1D arrays holding track of the traces
        # and axes to track.
        reduced_grid_ref = [[]]

        # Store the xaxis keys (e.g., x2) of the traces to keep
        trace_list = []
        # Store the xaxis and yaxis layout keys of the traces to keep (e.g., xaxis2)
        layout_xaxis_list, layout_yaxis_list = [], []
        for col_idx, row_idx in enumerate(self._overview_row_idxs):
            if row_idx is None:  # skip None value
                continue

            overview_grid_ref = self._grid_ref[row_idx][col_idx]
            reduced_grid_ref[0].append(overview_grid_ref)  # [0] bc 1 row in overview
            for subplot in overview_grid_ref:
                trace_list.append(subplot.trace_kwargs["xaxis"])

                # store the layout keys so that we can retain the exact layout
                xaxis_key, yaxis_key = subplot.layout_keys
                layout_yaxis_list.append(yaxis_key)
                layout_xaxis_list.append(xaxis_key)
        # print("layout_list", l_xaxis_list, l_yaxis_list)
        # print("trace_list", trace_list)

        fig_dict = self._get_current_graph()  # a copy of the current graph

        # copy the data from the relevant overview subplots
        reduced_fig_dict = {
            "data": [],
            "layout": {"template": fig_dict["layout"]["template"]},
        }
        # NOTE: we enumerate over the data of the full figure so that we can utilize the
        # trace index to mimic the colorway.
        for i, trace in enumerate(fig_dict["data"]):
            # NOTE: the interplay between line_color and marker_color seems to work in
            # this implementation - a more thorough investigation might be needed
            if trace.get("xaxis", "x") in trace_list:
                if "line" not in trace:
                    trace["line"] = {}
                # Ensure that the same color is utilized
                trace["line"]["color"] = (
                    self._layout_obj.template.layout.colorway[i]
                    if self.data[i].line.color is None
                    else self.data[i].line.color
                )
                # add the trace to the reduced figure
                reduced_fig_dict["data"].append(trace)

        # Add the relevant layout keys to the reduced figure
        for k, v in fig_dict["layout"].items():
            if k in layout_xaxis_list:
                reduced_fig_dict["layout"][k] = v
            elif k in layout_yaxis_list:
                v = v.copy()
                # set the domain to [0, 1] to ensure that the overview figure has the
                # global y-axis range
                v.update({"domain": [0, 1]})
                reduced_fig_dict["layout"][k] = v

        # Create a figure object using the reduced figure dict
        reduced_fig = go.Figure(layout=reduced_fig_dict["layout"])
        reduced_fig._grid_ref = reduced_grid_ref
        # Ensure that the trace uid is not adjusted, this must be set prior to adding
        # the trace data. Otherwise, data aggregation will not work.
        reduced_fig._data_validator.set_uid = False
        reduced_fig.add_traces(reduced_fig_dict["data"])
        return reduced_fig

    def _create_overview_figure(self) -> go.Figure:
        # create a new coarse fig
        reduced_fig = self._remove_other_axes_for_coarse()

        # Resample the coarse figure using 3x the default aggregation size to ensure
        # that it contains sufficient details
        coarse_fig_hf = FigureResampler(
            reduced_fig,
            default_n_shown_samples=3 * self._global_n_shown_samples,
        )

        # NOTE: this way we can alter props without altering the original hf data
        # NOTE: this also copies the default aggregation functionality to the coarse figure
        coarse_fig_hf._hf_data = {uid: trc.copy() for uid, trc in self._hf_data.items()}
        for trace in coarse_fig_hf.hf_data:
            trace["max_n_samples"] *= 3

        coarse_fig_dict = coarse_fig_hf._get_current_graph()
        # add the 3x max_n_samples coarse figure data to the coarse_fig_dict
        coarse_fig_hf._check_update_figure_dict(coarse_fig_dict)
        del coarse_fig_hf

        coarse_fig = go.Figure(layout=coarse_fig_dict["layout"])
        coarse_fig._grid_ref = reduced_fig._grid_ref
        coarse_fig._data_validator.set_uid = False
        coarse_fig.add_traces(coarse_fig_dict["data"])
        # remove any update menus for the coarse figure
        coarse_fig.layout.pop("updatemenus", None)
        # remove the `rangeselector` options for all 'axis' keys in the layout of the
        # coarse figure
        for k, v in coarse_fig.layout._props.items():
            if "axis" in k:
                v.pop("rangeselector", None)

        # height of the overview scales with the height of the dynamic view
        coarse_fig.update_layout(
            **self._overview_layout_kwargs,
            hovermode=False,
            clickmode="event+select",
            dragmode="select",
        )
        # Hide the grid
        hide_kwrgs = dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            title_text=None,
            mirror=True,
            ticks="",
            showline=False,
            linecolor="black",
        )
        coarse_fig.update_yaxes(**hide_kwrgs)
        coarse_fig.update_xaxes(**hide_kwrgs)

        vrect_props = dict(
            **dict(line_width=0, x0=0, x1=1),
            **dict(fillcolor="lightblue", opacity=0.25, layer="above"),
        )

        if self._grid_ref is None:  # case: go.Figure (no subplots)
            # set the fixed range to True
            coarse_fig["layout"]["xaxis"]["fixedrange"] = True
            coarse_fig["layout"]["yaxis"]["fixedrange"] = True

            # add a shading to the overview
            coarse_fig.add_vrect(xref="x domain", **vrect_props)
            return coarse_fig

        col_idx_overview = 0
        for col_idx, row_idx in enumerate(self._overview_row_idxs):
            if row_idx is None:  # skip the None value
                continue

            # we will only use the first grid-ref (as we will otherwise have multiple
            # overlapping selection boxes)
            for subplot in self._grid_ref[row_idx][col_idx][:1]:
                xaxis_key, yaxis_key = subplot.layout_keys

                # set the fixed range to True
                coarse_fig["layout"][xaxis_key]["fixedrange"] = True
                coarse_fig["layout"][yaxis_key]["fixedrange"] = True

                # add a shading to the overview
                coarse_fig.add_vrect(
                    col=col_idx_overview + 1,
                    xref=f"{subplot.trace_kwargs['xaxis']} domain",
                    **vrect_props,
                )

            col_idx_overview += 1  # only increase the index when not None

        return coarse_fig

    def show_dash(
        self,
        mode=None,
        config: dict | None = None,
        init_dash_kwargs: dict | None = None,
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
        init_dash_kwargs: dict, optional
            Keyword arguments for the Dash app constructor.
            !!! note
                This variable is of special interest when working in a jupyterhub +
                kubernetes environment. In this case, user notebook servers are spawned
                as separate pods and user access to those servers are proxied via
                jupyterhub. Dash requires the `requests_pathname_prefix` to be set on
                __init__ - which can be done via this `init_dash_kwargs` argument.
                Note that you should also pass the `jupyter_server_url` to the
                `show_dash` method.
                More details: https://github.com/predict-idlab/plotly-resampler/issues/265
        graph_properties: dict, optional
            Dictionary of (keyword, value) for the properties that should be passed to
            the dcc.Graph, by default None.
            e.g.: `{"style": {"width": "50%"}}`
            Note: "config" is not allowed as key in this dict, as there is a distinct
            ``config`` parameter for this property in this method.
            See more [https://dash.plotly.com/dash-core-components/graph](https://dash.plotly.com/dash-core-components/graph)
        **kwargs: dict
            kwargs for the ``app.run_server()`` method, e.g., port=8037.
            !!! note
                These kwargs take precedence over the ones that are passed to the
                constructor via the ``show_dash_kwargs`` argument.

        """
        available_modes = list(dash._jupyter.JupyterDisplayMode.__args__) + [
            "inline_persistent"
        ]
        assert (
            mode is None or mode in available_modes
        ), f"mode must be one of {available_modes}"
        graph_properties = {} if graph_properties is None else graph_properties
        assert "config" not in graph_properties  # There is a param for config
        if self["layout"]["autosize"] is True and self["layout"]["height"] is None:
            graph_properties.setdefault("style", {}).update({"height": "100%"})

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
            update_data = self._construct_update_data(relayout_dict)

            if not self._is_no_update(update_data):  # when there is an update
                with self.batch_update():
                    # First update the layout (first item of update_data)
                    self.layout.update(self._parse_relayout(update_data[0]))

                    # Then update the data
                    for updated_trace in update_data[1:]:
                        trace_idx = updated_trace.pop("index")
                        self.data[trace_idx].update(updated_trace)

        # 1. Construct the Dash app layout
        init_dash_kwargs = {} if init_dash_kwargs is None else init_dash_kwargs
        if self._create_overview:
            # fmt: off
            # Add the assets folder to the init_dash_kwargs
            init_dash_kwargs["assets_folder"] = os.path.relpath(ASSETS_FOLDER, os.getcwd())
            # Also include the lodash script, as the client-side callbacks uses this
            init_dash_kwargs["external_scripts"] = ["https://cdn.jsdelivr.net/npm/lodash/lodash.min.js" ]
            # fmt: on

        # fmt: off
        div = dash.html.Div(
            children=[
                dash.dcc.Graph(
                    id="resample-figure", figure=self, config=config, **graph_properties
                )
            ],
            style={
                "display": "flex", "flex-flow": "column",
                "height": "95vh", "width": "100%",
            },
        )
        # fmt: on
        if self._create_overview:
            overview_config = config.copy() if config is not None else {}
            overview_config["displayModeBar"] = False
            coarse_fig = self._create_overview_figure()
            div.children += [
                dash.dcc.Graph(
                    id="overview-figure",
                    figure=coarse_fig,
                    config=overview_config,
                    **graph_properties,
                ),
            ]

        # Create the app, populate the layout and register the resample callback
        app = dash.Dash("local_app", **init_dash_kwargs)
        app.layout = div
        self.register_update_graph_callback(
            app,
            "resample-figure",
            "overview-figure" if self._create_overview else None,
        )

        # 2. Run the app
        height_param = "height" if mode == "inline_persistent" else "jupyter_height"
        if "inline" in mode and height_param not in kwargs:
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

        # function signatures are slightly different for the (Jupyter)Dash and the
        # JupyterDashInlinePersistent implementations
        if mode == "inline_persistent":
            jpi = JupyterDashPersistentInlineOutput(self)
            jpi.run_app(app=app, **kwargs)
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
            servers_dict = dash.jupyter_dash._servers
            old_server = servers_dict.get((self._host, self._port))
            if old_server:
                old_server.shutdown()
            del servers_dict[(self._host, self._port)]
        elif warn:
            warnings.warn(
                "Could not stop the server, either the \n"
                + "\t- 'show-dash' method was not called, or \n"
                + "\t- the dash-server wasn't started with 'show_dash'"
            )

    def construct_update_data_patch(
        self, relayout_data: dict
    ) -> Union[dash.Patch, dash.no_update]:
        """Construct the Patch of the to-be-updated front-end data, based on the layout
        change.

        Attention
        ---------
        This method is tightly coupled with Dash app callbacks. It takes the front-end
        figure its ``relayoutData`` as input and returns the ``dash.Patch`` which needs
        to be sent to the ``figure`` property for the corresponding ``dcc.Graph``.

        Parameters
        ----------
        relayout_data: dict
            A dict containing the ``relayoutData`` (i.e., the changed layout data) of
            the corresponding front-end graph.

        Returns
        -------
        dash.Patch:
            The Patch object containing the figure updates which needs to be sent to
            the front-end.

        """
        update_data = self._construct_update_data(relayout_data)
        if not isinstance(update_data, list) or len(update_data) <= 1:
            return dash.no_update

        patched_figure = dash.Patch()  # create patch
        for trace in update_data[1:]:  # skip first item as it contains the relayout
            trace_index = trace.pop("index")  # the index of the corresponding trace
            # All the other items are the trace properties which needs to be updated
            for k, v in trace.items():
                # NOTE: we need to use the `patched_figure` as a dict, and not
                # `patched_figure.data` as the latter will replace **all** the
                # data for the corresponding trace, and we just want to update the
                # specific trace its properties.
                patched_figure["data"][trace_index][k] = v
        return patched_figure

    def register_update_graph_callback(
        self,
        app: dash.Dash,
        graph_id: str,
        coarse_graph_id: Optional[str] = None,
    ):
        """Register the [`construct_update_data_patch`][figure_resampler.figure_resampler_interface.AbstractFigureAggregator.construct_update_data_patch]
        method as callback function to the passed dash-app.

        Parameters
        ----------
        app: Union[dash.Dash, JupyterDash]
            The app in which the callback will be registered.
        graph_id:
            The id of the ``dcc.Graph``-component which withholds the to-be resampled
            Figure.
        coarse_graph_id: str, optional
            The id of the ``dcc.Graph``-component which withholds the coarse overview
            Figure, by default None.

        """
        # As we use the figure again as output, we need to set: allow_duplicate=True

        if coarse_graph_id is not None:
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
            dash.Output(graph_id, "figure", allow_duplicate=True),
            dash.Input(graph_id, "relayoutData"),
            prevent_initial_call=True,
        )(self.construct_update_data_patch)

    def _get_pr_props_keys(self) -> List[str]:
        # Add the additional plotly-resampler properties of this class
        return super()._get_pr_props_keys() + ["_show_dash_kwargs"]

    def _ipython_display_(self):
        # To display the figure inline as a dash app
        self.show_dash(mode="inline")
