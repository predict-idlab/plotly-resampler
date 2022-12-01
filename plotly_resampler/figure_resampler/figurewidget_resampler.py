# -*- coding: utf-8 -*-
"""
``FigureWidgetResampler`` wrapper around the plotly ``go.FigureWidget`` class.

Utilizes the ``fig.layout.on_change`` method to enable dynamic resampling.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

from typing import Tuple

import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure

from ..aggregation import AbstractSeriesAggregator, EfficientLTTB
from .figure_resampler_interface import AbstractFigureAggregator


class _FigureWidgetResamplerM(type(AbstractFigureAggregator), type(go.FigureWidget)):
    # MetaClass for the FigureWidgetResampler
    pass


class FigureWidgetResampler(
    AbstractFigureAggregator, go.FigureWidget, metaclass=_FigureWidgetResamplerM
):
    """Data aggregation functionality wrapper for ``go.FigureWidgets``.

    .. attention::

        * This wrapper only works within ``jupyter``-based environments.
        * The ``.show()`` method returns a **static figure** on which the
          **dynamic resampling cannot be performed**. To allow dynamic resampling,
          you should just output the ``FigureWidgetResampler`` object in a cell.

    """

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
    ):
        # Parse the figure input before calling `super`
        f = self._get_figure_class(go.FigureWidget)()
        f._data_validator.set_uid = False

        if isinstance(figure, BaseFigure):
            # A base figure object, can be;
            # - a base plotly figure: go.Figure or go.FigureWidget
            # - a plotly-resampler figure: subclass of AbstractFigureAggregator
            # => we first copy the layout, grid_str and grid ref
            f.layout = figure.layout
            f._grid_str = figure._grid_str
            f._grid_ref = figure._grid_ref
            f.add_traces(figure.data)
        elif isinstance(figure, dict) and (
            "data" in figure or "layout" in figure # or "frames" in figure  # TODO
        ):
            # A figure as a dict, can be;
            # - a plotly figure as a dict (after calling `fig.to_dict()`)
            # - a pickled (plotly-resampler) figure (after loading a pickled figure)
            f.layout = figure.get("layout")
            f._grid_str = figure.get("_grid_str")
            f._grid_ref = figure.get("_grid_ref")
            f.add_traces(figure.get("data"))
            # `pr_props` is not None when loading a pickled plotly-resampler figure
            f._pr_props = figure.get("pr_props")
            # `f._pr_props`` is an attribute to store properties of a plotly-resampler
            # figure. This attribute is only used to pass information to the super()
            # constructor. Once the super constructor is called, the attribute is
            # removed.

            # f.add_frames(figure.get("frames")) TODO
        elif isinstance(figure, (dict, list)):
            # A single trace dict or a list of traces
            f.add_traces(figure)

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
            # And adjust the default max_n_samples and
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

        self._prev_layout = None  # Contains the previous xaxis layout configuration

        # used for logging purposes to save a history of layout changes
        self._relayout_hist = []

        # Assign the the update-methods to the corresponding classes
        showspike_keys = [f"{xaxis}.showspikes" for xaxis in self._xaxis_list]
        self.layout.on_change(self._update_spike_ranges, *showspike_keys)

        x_relayout_keys = [f"{xaxis}.range" for xaxis in self._xaxis_list]
        self.layout.on_change(self._update_x_ranges, *x_relayout_keys)

    def _update_x_ranges(self, layout, *x_ranges, force_update: bool = False):
        """Update the the go.Figure data based on changed x-ranges.

        Parameters
        ----------
        layout : go.Layout
            The figure's (i.e, self) layout object. Remark that this is a reference,
            so if we change self.layout (same object reference), this object will
            change.
        *x_ranges: iterable
            A iterable list of current x-ranges, where each x-range is a tuple of two
            items, indicating the current/new (if changed) left-right x-range,
            respectively.
        fore_update: bool
            Whether an update of all traces will be forced, by default False.
        """
        relayout_dict = {}  # variable in which we aim to reconstruct the relayout
        # serialize the layout in a new dict object
        layout = {
            xaxis_str: layout[xaxis_str].to_plotly_json()
            for xaxis_str in self._xaxis_list
        }
        if self._prev_layout is None:
            self._prev_layout = layout

        for xaxis_str, x_range in zip(self._xaxis_list, x_ranges):
            # We also check whether "range" is within the xaxis its layout otherwise
            # It is most-likely an autorange check
            if (
                "range" in layout[xaxis_str]
                and self._prev_layout[xaxis_str].get("range", []) != x_range
                or (force_update and x_range is not None)
            ):
                # a change took place -> add to the relayout dict
                relayout_dict[f"{xaxis_str}.range[0]"] = x_range[0]
                relayout_dict[f"{xaxis_str}.range[1]"] = x_range[1]

                # An update will take place for that trace
                # -> save current xaxis range to _prev_layout
                self._prev_layout[xaxis_str]["range"] = x_range

        if len(relayout_dict):
            # Construct the update data
            update_data = self.construct_update_data(relayout_dict)

            if self._is_no_update(update_data):
                # Return when no data update
                return

            if self._print_verbose:
                self._relayout_hist.append(dict(zip(self._xaxis_list, x_ranges)))
                self._relayout_hist.append(layout)
                self._relayout_hist.append(["xaxis-range-update", len(update_data) - 1])
                self._relayout_hist.append("-" * 30)

            with self.batch_update():
                # First update the layout (first item of update_data)
                self.layout.update(update_data[0])

                for xaxis_str in self._xaxis_list:
                    if "showspikes" in layout[xaxis_str]:
                        self.layout[xaxis_str].pop("showspikes")

                # Then update the data
                for updated_trace in update_data[1:]:
                    trace_idx = updated_trace.pop("index")
                    self.data[trace_idx].update(updated_trace)

    def _update_spike_ranges(self, layout, *showspikes, force_update=False):
        """Update the go.Figure based on the changed spike-ranges.

        Parameters
        ----------
        layout : go.Layout
            The figure's (i.e, self) layout object. Remark that this is a reference,
            so if we change self.layout (same object reference), this object will
            change.
        *showspikes: iterable
            A iterable where each item is a bool, indicating  whether showspikes is set
            to true/false for the corresponding xaxis in ``self._xaxis_list``.
        force_update: bool
            Bool indicating whether the range updates need to take place. This is
            especially useful when you have recently updated the figure its data (with
            the hf_data property) and want to perform an autoscale, independent from
            the current figure-layout.
        """
        relayout_dict = {}  # variable in which we aim to reconstruct the relayout
        # serialize the layout in a new dict object
        layout = {
            xaxis_str: layout[xaxis_str].to_plotly_json()
            for xaxis_str in self._xaxis_list
        }

        if self._prev_layout is None:
            self._prev_layout = layout

        for xaxis_str, showspike in zip(self._xaxis_list, showspikes):
            if (
                force_update
                or
                # autorange key must be set to True
                (
                    layout[xaxis_str].get("autorange", False)
                    # we only perform updates for traces which have 'range' property,
                    # as we do need to reconstruct the update-data for these traces
                    and self._prev_layout[xaxis_str].get("range", None) is not None
                )
            ):
                relayout_dict[f"{xaxis_str}.autorange"] = True
                relayout_dict[f"{xaxis_str}.showspikes"] = showspike
                # autorange -> we pop the xaxis range
                if "range" in layout[xaxis_str]:
                    del layout[xaxis_str]["range"]

        if len(relayout_dict):
            # An update will take place, save current layout to _prev_layout
            self._prev_layout = layout

            # Construct the update data
            update_data = self.construct_update_data(relayout_dict)
            if self._print_verbose:
                self._relayout_hist.append(layout)
                self._relayout_hist.append(["showspikes-update", len(update_data) - 1])
                self._relayout_hist.append("-" * 30)

            with self.batch_update():
                # First update the layout (first item of update_data)
                if not force_update:
                    self.layout.update(update_data[0])

                # Also:  Remove the showspikes from the layout, otherwise the autorange
                # will not work as intended (it will not be triggered again)
                # Note: this removal causes a second trigger of this method
                # which will go in the "else" part below.
                for xaxis_str in self._xaxis_list:
                    self.layout[xaxis_str].pop("showspikes")

                # Then, update the data
                for updated_trace in update_data[1:]:
                    trace_idx = updated_trace.pop("index")
                    self.data[trace_idx].update(updated_trace)
        elif self._print_verbose:
            self._relayout_hist.append(["showspikes", "initial call or showspikes"])
            self._relayout_hist.append("-" * 40)

    def reset_axes(self):
        """Reset the axes of the FigureWidgetResampler.

        This is useful when adjusting the `hf_data` properties of the
        ``FigureWidgetResampler``.
        """
        self._update_spike_ranges(
            self.layout, *[False] * len(self._xaxis_list), force_update=True
        )
        # Reset the layout
        self.update_layout(
            {
                axis: {"autorange": None, "range": None}
                for axis in self._xaxis_list + self._yaxis_list
            }
        )

    def reload_data(self):
        """Reload all the data of FigureWidgetResampler for the current range-view.

        This is useful when adjusting the `hf_data` properties of the
        ``FigureWidgetResampler``.
        """
        if all(
            self.layout[xaxis].autorange
            or (
                self.layout[xaxis].autorange is None
                and self.layout[xaxis].range is None
            )
            for xaxis in self._xaxis_list
        ):
            self._update_spike_ranges(
                self.layout, *[False] * len(self._xaxis_list), force_update=True
            )
        else:
            # Resample the data for the current range-view
            self._update_x_ranges(
                self.layout,
                # Pass the current view to trigger a resample operation
                *[self.layout[xaxis_str]["range"] for xaxis_str in self._xaxis_list],
                force_update=True,
            )
            # TODO: when we know which traces have changed we can use
            # a new -> `update_xaxis_str` argument.
