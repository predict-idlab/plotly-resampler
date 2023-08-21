import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from plotly_resampler import FigureResampler, FigureWidgetResampler
from plotly_resampler.aggregation import MinMaxLTTB


@pytest.mark.parametrize("fig_type", [FigureResampler, FigureWidgetResampler])
def test_multiple_axes_figure(fig_type):
    # Generate some data
    x = np.arange(200_000)
    sin = 3 + np.sin(x / 200) + np.random.randn(len(x)) / 30

    fig = fig_type(
        default_n_shown_samples=2000, default_downsampler=MinMaxLTTB(parallel=True)
    )

    # all traces will be plotted against the same x-axis
    # note: the first added trace its yaxis will be used as reference
    fig.add_trace(go.Scatter(name="orig", yaxis="y1", line_width=1), hf_x=x, hf_y=sin)
    fig.add_trace(
        go.Scatter(name="negative", yaxis="y2", line_width=1), hf_x=x, hf_y=-sin
    )
    fig.add_trace(
        go.Scatter(name="sqrt(orig)", yaxis="y3", line_width=1),
        hf_x=x,
        hf_y=np.sqrt(sin * 10),
    )
    fig.add_trace(
        go.Scatter(name="orig**2", yaxis="y4", line_width=1),
        hf_x=x,
        hf_y=(sin - 3) ** 2,
    )

    # in order for autoshift to work, you need to set x-anchor to free
    fig.update_layout(
        # NOTE: you can use the domain key to set the x-axis range (if you want to display)
        # the legend on the right instead of the top as done here
        xaxis=dict(domain=[0, 1]),
        # Add a title to the y-axis
        yaxis=dict(title="orig"),
        # by setting anchor=free, overlaying, and autoshift, the axis will be placed
        # automatically, without overlapping any other axes
        yaxis2=dict(
            title="negative",
            anchor="free",
            overlaying="y1",
            side="left",
            autoshift=True,
        ),
        yaxis3=dict(
            title="sqrt(orig)",
            anchor="free",
            overlaying="y1",
            side="right",
            autoshift=True,
        ),
        yaxis4=dict(
            title="orig ** 2",
            anchor="free",
            overlaying="y1",
            side="right",
            autoshift=True,
        ),
    )

    # Update layout properties
    fig.update_layout(
        title_text="multiple y-axes example",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        template="plotly_white",
    )

    # Test: check whether a single update triggers all traces to be updated
    out = fig.construct_update_data({"xaxis.range[0]": 0, "xaxis.range[1]": 50_000})
    assert len(out) == 5
    # fig.show_dash


@pytest.mark.parametrize("fig_type", [FigureResampler, FigureWidgetResampler])
def test_multiple_axes_subplot_rows(fig_type):
    # Generate some data
    x = np.arange(200_000)
    sin = 3 + np.sin(x / 200) + np.random.randn(len(x)) / 30

    # create a figure with 2 rows and 1 column
    # NOTE: instead of the above methods, we don't add the "yaxis" argument to the
    #       scatter object
    fig = fig_type(make_subplots(rows=2, cols=1, shared_xaxes=True))
    fig.add_trace(go.Scatter(name="orig"), hf_x=x, hf_y=sin, row=2, col=1)
    fig.add_trace(go.Scatter(name="-orig"), hf_x=x, hf_y=-sin, row=2, col=1)
    fig.add_trace(go.Scatter(name="sqrt"), hf_x=x, hf_y=np.sqrt(sin * 10), row=2, col=1)
    fig.add_trace(go.Scatter(name="orig**2"), hf_x=x, hf_y=(sin - 3) ** 2, row=2, col=1)

    # NOTE: because of the row and col specification, the yaxis is automatically set to y2
    for i, data in enumerate(fig.data[1:], 3):
        data.update(yaxis=f"y{i}")

    # add the original signal to the first row subplot
    fig.add_trace(go.Scatter(name="<b>orig</b>"), row=1, col=1, hf_x=x, hf_y=sin)

    # in order for autoshift to work, you need to set x-anchor to free
    fig.update_layout(
        xaxis2=dict(domain=[0, 1], anchor="y2"),
        yaxis2=dict(title="orig"),
        yaxis3=dict(
            title="-orig",
            anchor="free",
            overlaying="y2",
            side="left",
            autoshift=True,
        ),
        yaxis4=dict(
            title="sqrt(orig)",
            anchor="free",
            overlaying="y2",
            side="right",
            autoshift=True,
        ),
        yaxis5=dict(
            title="orig ** 2",
            anchor="free",
            overlaying="y2",
            side="right",
            autoshift=True,
        ),
    )

    # Update layout properties
    fig.update_layout(
        title_text="multiple y-axes example",
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        template="plotly_white",
    )

    # Test: check whether a single update triggers all traces to be updated
    out = fig.construct_update_data(
        {
            "xaxis.range[0]": 0,
            "xaxis.range[1]": 50_000,
            "xaxis2.range[0]": 0,
            "xaxis2.range[1]": 50_000,
        }
    )
    assert len(out) == 6


@pytest.mark.parametrize("fig_type", [FigureResampler, FigureWidgetResampler])
def test_multiple_axes_subplot_cols(fig_type):
    x = np.arange(200_000)
    sin = 3 + np.sin(x / 200) + np.random.randn(len(x)) / 30

    # Create a figure with 1 row and 2 columns
    fig = fig_type(make_subplots(rows=1, cols=2))
    fig.add_trace(go.Scatter(name="orig"), hf_x=x, hf_y=sin, row=1, col=2)
    fig.add_trace(go.Scatter(name="-orig"), hf_x=x, hf_y=-sin, row=1, col=2)
    fig.add_trace(go.Scatter(name="sqrt"), hf_x=x, hf_y=np.sqrt(sin * 10), row=1, col=2)
    fig.add_trace(go.Scatter(name="orig**2"), hf_x=x, hf_y=(sin - 3) ** 2, row=1, col=2)

    # NOTE: because of the row & col specification, the yaxis is automatically set to y2
    for i, data in enumerate(fig.data[1:], 3):
        data.update(yaxis=f"y{i}")

    fig.add_trace(go.Scatter(name="<b>orig</b>"), row=1, col=1, hf_x=x, hf_y=sin)

    # In order for autoshift to work, you need to set x-anchor to free
    fig.update_layout(
        xaxis=dict(domain=[0, 0.4]),
        xaxis2=dict(domain=[0.56, 1]),
        yaxis2=dict(title="orig"),
        yaxis3=dict(
            title="-orig",
            anchor="free",
            overlaying="y2",
            side="left",
            autoshift=True,
        ),
        yaxis4=dict(
            title="sqrt(orig)",
            anchor="free",
            overlaying="y2",
            side="right",
            autoshift=True,
        ),
        yaxis5=dict(
            title="orig ** 2",
            anchor="free",
            overlaying="y2",
            side="right",
            autoshift=True,
        ),
    )

    # Update layout properties
    fig.update_layout(
        title_text="multiple y-axes example",
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        template="plotly_white",
    )

    out = fig.construct_update_data(
        {
            "xaxis.range[0]": 0,
            "xaxis.range[1]": 50_000,
        }
    )
    assert len(out) == 2

    out = fig.construct_update_data(
        {
            "xaxis2.range[0]": 0,
            "xaxis2.range[1]": 50_000,
        }
    )
    assert len(out) == 5
