"""Code which tests the FigureResampler functionaliteis"""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"


import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.downsamplers import LTTB, EveryNthPoint
from .utils import float_series, bool_series, cat_series


def test_add_trace(float_series, bool_series, cat_series):
    # see: https://plotly.com/python/subplots/#custom-sized-subplot-with-subplot-titles
    base_fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    kwarg_space_list = [
        {},
        {
            "default_downsampler": LTTB(interleave_gaps=True),
            "resampled_trace_prefix_suffix": tuple(["<b>[r]</b>", "~~"]),
            "verbose": True,
        },
    ]
    for kwarg_space in kwarg_space_list:
        fig = FigureResampler(base_fig, **kwarg_space)

        fig.add_trace(
            go.Scatter(x=float_series.index, y=float_series, name="float_series"),
            row=1,
            col=1,
            limit_to_view=False,
            hf_hovertext="text"
        )

        fig.add_trace(
            go.Scatter(x=[], y=[], text="text", name="bool_series"),
            hf_x=bool_series.index,
            hf_y=bool_series,
            row=1,
            col=2,
            limit_to_view=True,
        )

        fig.add_trace(
            go.Scattergl(x=[], y=[], text="text", name="cat_series"),
            row=2,
            col=1,
            downsampler=EveryNthPoint(interleave_gaps=True),
            hf_x=cat_series.index,
            hf_y=cat_series,
            limit_to_view=True,
        )

