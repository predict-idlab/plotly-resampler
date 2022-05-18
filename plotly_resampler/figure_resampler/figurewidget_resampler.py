# -*- coding: utf-8 -*-
"""
``FigureWidgetResampler`` wrapper around the plotly ``go.FigureWidget`` class.

Utilizes the ``fig.layout.on_change`` method to enable dynamic resampling.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import re
from typing import Tuple

import plotly.graph_objects as go

from .figure_resampler import AbstractFigureAggregator
from ..aggregation import AbstractSeriesAggregator, EfficientLTTB


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
        figure: go.FigureWidget | go.Figure = None,
        convert_existing_traces: bool = True,
        default_n_shown_samples: int = 1000,
        default_downsampler: AbstractSeriesAggregator = EfficientLTTB(),
        resampled_trace_prefix_suffix: Tuple[str, str] = (
            '<b style="color:sandybrown">[R]</b> ',
            "",
        ),
        show_mean_aggregation_size: bool = True,
        verbose: bool = False,
    ):
        if figure is None:
            figure = go.FigureWidget()

        if not isinstance(figure, go.FigureWidget):
            figure = go.FigureWidget(figure)

        super().__init__(
            figure,
            convert_existing_traces,
            default_n_shown_samples,
            default_downsampler,
            resampled_trace_prefix_suffix,
            show_mean_aggregation_size,
            verbose,
        )

        self._prev_layout = None  # Contains the previous xaxis layout configuration

        # used for logging purposes to save a history of layout changes
        self._relayout_hist = []

        # A list of al xaxis and yaxis string names
        # e.g., "xaxis", "xaxis2", "xaxis3", .... for _xaxis_list
        self._xaxis_list = self._re_matches(re.compile("xaxis\d*"), self._layout.keys())
        self._yaxis_list = self._re_matches(re.compile("yaxis\d*"), self._layout.keys())
        # edge case: an empty `go.Figure()` does not yet contain axes keys
        if not len(self._xaxis_list):
            self._xaxis_list = ["xaxis"]
            self._yaxis_list = ["yaxis"]

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
                axis: {"autorange": True, "range": None}
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
