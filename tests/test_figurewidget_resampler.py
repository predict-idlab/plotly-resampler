"""Code which tests the FigureWidgetResampler functionalities"""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"


import datetime
from copy import copy
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from plotly_resampler import EveryNthPoint, FigureWidgetResampler, MinMaxLTTB


def test_add_trace_kwarg_space(float_series, bool_series, cat_series):
    # see: https://plotly.com/python/subplots/#custom-sized-subplot-with-subplot-titles
    base_fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    kwarg_space_list = [
        {},
        {
            "default_downsampler": MinMaxLTTB(),
            "resampled_trace_prefix_suffix": tuple(["<b>[r]</b>", "~~"]),
            "verbose": True,
        },
    ]
    for kwarg_space in kwarg_space_list:
        fig = FigureWidgetResampler(base_fig, **kwarg_space)

        fig.add_trace(
            go.Scatter(x=float_series.index, y=float_series),
            row=1,
            col=1,
            limit_to_view=False,
            hf_text="text",
            hf_hovertext="hovertext",
        )

        fig.add_trace(
            go.Scatter(text="text", name="bool_series"),
            hf_x=bool_series.index,
            hf_y=bool_series,
            row=1,
            col=2,
            limit_to_view=True,
        )

        fig.add_trace(
            go.Scattergl(text="text", name="cat_series"),
            row=2,
            col=1,
            downsampler=EveryNthPoint(),
            hf_x=cat_series.index,
            hf_y=cat_series,
            limit_to_view=True,
        )


def test_add_trace_not_resampling(float_series):
    # see: https://plotly.com/python/subplots/#custom-sized-subplot-with-subplot-titles
    base_fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    fig = FigureWidgetResampler(base_fig, default_n_shown_samples=1000)

    fig.add_trace(
        go.Scatter(
            x=float_series.index[:800], y=float_series[:800], name="float_series"
        ),
        row=1,
        col=1,
        hf_text="text",
        hf_hovertext="hovertext",
    )

    fig.add_trace(
        go.Scatter(name="float_series"),
        limit_to_view=False,
        row=1,
        col=1,
        hf_x=float_series.index[-800:],
        hf_y=float_series[-800:],
        hf_text="text",
        hf_hovertext="hovertext",
    )


def test_add_scatter_trace_no_data():
    fig = FigureWidgetResampler(default_n_shown_samples=1000)

    # no x and y data
    fig.add_trace(go.Scatter())


def test_add_scatter_trace_no_x():
    fig = FigureWidgetResampler(go.Figure(), default_n_shown_samples=1000)

    # no x data
    fig.add_trace(go.Scatter(y=[2, 1, 4, 3], name="s1"))
    fig.add_trace(go.Scatter(name="s2"), hf_y=[2, 1, 4, 3])


def test_add_not_a_hf_trace(float_series):
    # see: https://plotly.com/python/subplots/#custom-sized-subplot-with-subplot-titles
    base_fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    fig = FigureWidgetResampler(base_fig, default_n_shown_samples=1000, verbose=True)

    fig.add_trace(
        go.Scatter(
            x=float_series.index[:800], y=float_series[:800], name="float_series"
        ),
        row=1,
        col=1,
        hf_text="text",
        hf_hovertext="hovertext",
    )

    # add a not hf-trace
    fig.add_trace(
        go.Histogram(
            x=float_series,
            name="float_series",
        ),
        row=2,
        col=1,
    )


def test_box_histogram(float_series):
    base_fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    fig = FigureWidgetResampler(base_fig, default_n_shown_samples=1000, verbose=True)

    fig.add_trace(
        go.Scattergl(x=float_series.index, y=float_series, name="float_series"),
        row=1,
        col=1,
        hf_text="text",
        hf_hovertext="hovertext",
    )

    fig.add_trace(go.Box(x=float_series.values, name="float_series"), row=1, col=2)
    fig.add_trace(
        go.Box(x=float_series.values**2, name="float_series**2"), row=1, col=2
    )

    # add a not hf-trace
    fig.add_trace(
        go.Histogram(
            x=float_series,
            name="float_series",
        ),
        row=2,
        col=1,
    )


def test_cat_box_histogram(float_series):
    # Create a categorical series, with mostly a's, but a few sparse b's and c's
    cats_list = np.array(list("aaaaaaaaaa" * 1000))
    cats_list[np.random.choice(len(cats_list), 100, replace=False)] = "b"
    cats_list[np.random.choice(len(cats_list), 50, replace=False)] = "c"
    cat_series = pd.Series(cats_list, dtype="category")

    base_fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )
    fig = FigureWidgetResampler(base_fig, default_n_shown_samples=1000, verbose=True)

    fig.add_trace(
        go.Scattergl(name="cat_series", x=cat_series.index, y=cat_series),
        row=1,
        col=1,
        hf_text="text",
        hf_hovertext="hovertext",
    )

    fig.add_trace(go.Box(x=float_series.values, name="float_box_pow"), row=1, col=2)
    fig.add_trace(
        go.Box(x=float_series.values**2, name="float_box_pow_2"), row=1, col=2
    )

    # add a not hf-trace
    fig.add_trace(
        go.Histogram(
            x=float_series,
            name="float_hist",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=700)


def test_replace_figure(float_series):
    # see: https://plotly.com/python/subplots/#custom-sized-subplot-with-subplot-titles
    base_fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    fr_fig = FigureWidgetResampler(base_fig, default_n_shown_samples=1000)

    go_fig = go.Figure()
    go_fig.add_trace(go.Scattergl(x=float_series.index, y=float_series, name="fs"))

    fr_fig.replace(go_fig, convert_existing_traces=False)
    # assert len(fr_fig.data) == 1
    assert len(fr_fig.data[0]["x"]) == len(float_series)
    # the orig float series data must still be the orig shape (we passed a view so
    # we must check this)
    assert len(go_fig.data[0]["x"]) == len(float_series)

    fr_fig.replace(go_fig, convert_existing_traces=True)
    # assert len(fr_fig.data) == 1
    assert len(fr_fig.data[0]["x"]) == 1000

    # the orig float series data must still be the orig shape (we passed a view so
    # we must check this)
    assert len(go_fig.data[0]["x"]) == len(float_series)


def test_nan_removed_input(float_series):
    # see: https://plotly.com/python/subplots/#custom-sized-subplot-with-subplot-titles
    base_fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    fig = FigureWidgetResampler(
        base_fig,
        default_n_shown_samples=1000,
        resampled_trace_prefix_suffix=(
            '<b style="color:sandybrown">[R]</b>',
            '<b style="color:sandybrown">[R]</b>',
        ),
    )

    float_series = float_series.copy()
    float_series.iloc[np.random.choice(len(float_series), 100, replace=False)] = np.nan
    fig.add_trace(
        go.Scatter(x=float_series.index, y=float_series, name="float_series"),
        row=1,
        col=1,
        hf_text="text",
        hf_hovertext="hovertext",
    )
    # Check the desired behavior
    assert len(fig.hf_data[0]["y"]) == len(float_series) - 100
    assert ~pd.isna(fig.hf_data[0]["y"]).any()

    # here we test whether we are able to deal with not-nan output
    float_series.iloc[np.random.choice(len(float_series), 100)] = np.nan
    fig.add_trace(
        go.Scatter(
            x=float_series.index, y=float_series
        ),  # we explicitly do not add a name
        hf_hovertext="mean" + float_series.rolling(10).mean().round(2).astype("str"),
        row=2,
        col=1,
    )

    float_series.iloc[np.random.choice(len(float_series), 100)] = np.nan
    fig.add_trace(
        go.Scattergl(
            x=float_series.index,
            y=float_series,
            text="mean" + float_series.rolling(10).mean().round(2).astype("str"),
        ),
        row=1,
        col=2,
    )


def test_nan_removed_input_check_nans_false(float_series):
    # see: https://plotly.com/python/subplots/#custom-sized-subplot-with-subplot-titles
    base_fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    fig = FigureWidgetResampler(
        base_fig,
        default_n_shown_samples=1000,
        default_downsampler=EveryNthPoint(),
        resampled_trace_prefix_suffix=(
            '<b style="color:sandybrown">[R]</b>',
            '<b style="color:sandybrown">[R]</b>',
        ),
    )

    float_series = float_series.copy()
    float_series.iloc[np.random.choice(len(float_series), 100)] = np.nan
    fig.add_trace(
        go.Scatter(x=float_series.index, y=float_series, name="float_series"),
        row=1,
        col=1,
        hf_text="text",
        hf_hovertext="hovertext",
        check_nans=False,
    )
    # Check the undesired behavior
    assert len(fig.hf_data[0]["y"]) == len(float_series)
    assert pd.isna(fig.hf_data[0]["y"]).any()


def test_hf_text():
    y = np.arange(10_000)

    fig = FigureWidgetResampler()
    fig.add_trace(
        go.Scatter(name="blabla", text=y.astype(str)),
        hf_y=y,
    )

    assert np.all(fig.hf_data[0]["text"] == y.astype(str))
    assert fig.hf_data[0]["hovertext"] is None

    assert len(fig.data[0].y) < 5_000
    assert np.all(fig.data[0].text == fig.data[0].y.astype(int).astype(str))
    assert fig.data[0].hovertext is None

    fig = FigureWidgetResampler()
    fig.add_trace(go.Scatter(name="blabla"), hf_y=y, hf_text=y.astype(str))

    assert np.all(fig.hf_data[0]["text"] == y.astype(str))
    assert fig.hf_data[0]["hovertext"] is None

    assert len(fig.data[0].y) < 5_000
    assert np.all(fig.data[0].text == fig.data[0].y.astype(int).astype(str))
    assert fig.data[0].hovertext is None


def test_hf_hovertext():
    y = np.arange(10_000)

    fig = FigureWidgetResampler()
    fig.add_trace(
        go.Scatter(name="blabla", hovertext=y.astype(str)),
        hf_y=y,
    )

    assert np.all(fig.hf_data[0]["hovertext"] == y.astype(str))
    assert fig.hf_data[0]["text"] is None

    assert len(fig.data[0].y) < 5_000
    assert np.all(fig.data[0].hovertext == fig.data[0].y.astype(int).astype(str))
    assert fig.data[0].text is None

    fig = FigureWidgetResampler()
    fig.add_trace(go.Scatter(name="blabla"), hf_y=y, hf_hovertext=y.astype(str))

    assert np.all(fig.hf_data[0]["hovertext"] == y.astype(str))
    assert fig.hf_data[0]["text"] is None

    assert len(fig.data[0].y) < 5_000
    assert np.all(fig.data[0].hovertext == fig.data[0].y.astype(int).astype(str))
    assert fig.data[0].text is None


def test_hf_text_and_hf_hovertext():
    y = np.arange(10_000)

    fig = FigureWidgetResampler()
    fig.add_trace(
        go.Scatter(name="blabla", text=y.astype(str), hovertext=y.astype(str)[::-1]),
        hf_y=y,
    )

    assert np.all(fig.hf_data[0]["text"] == y.astype(str))
    assert np.all(fig.hf_data[0]["hovertext"] == y.astype(str)[::-1])

    assert len(fig.data[0].y) < 5_000
    assert np.all(fig.data[0].text == fig.data[0].y.astype(int).astype(str))
    assert np.all(
        fig.data[0].hovertext == (9_999 - fig.data[0].y).astype(int).astype(str)
    )

    fig = FigureWidgetResampler()
    fig.add_trace(
        go.Scatter(name="blabla"),
        hf_y=y,
        hf_text=y.astype(str),
        hf_hovertext=y.astype(str)[::-1],
    )

    assert np.all(fig.hf_data[0]["text"] == y.astype(str))
    assert np.all(fig.hf_data[0]["hovertext"] == y.astype(str)[::-1])

    assert len(fig.data[0].y) < 5_000
    assert np.all(fig.data[0].text == fig.data[0].y.astype(int).astype(str))
    assert np.all(
        fig.data[0].hovertext == (9_999 - fig.data[0].y).astype(int).astype(str)
    )


def test_hf_text_and_hf_marker_color():
    # Test for https://github.com/predict-idlab/plotly-resampler/issues/224
    fig = FigureWidgetResampler(default_n_shown_samples=1_000)

    x = pd.date_range("1-1-2000", "1-1-2001", periods=2_000)
    y = np.sin(100 * np.arange(len(x)) / len(x))
    text = [f'text: {yi}, color:{"black" if yi>=0.99 else "blue"}' for yi in y]
    marker_color = ["black" if yi >= 0.99 else "blue" for yi in y]
    trace = go.Scatter(
        x=x,
        y=y,
        marker={"color": marker_color},
        text=text,
    )
    fig.add_trace(trace)

    # Check correct data types
    assert not isinstance(fig.hf_data[0]["text"], (tuple, list))
    assert fig.hf_data[0]["hovertext"] is None
    assert not isinstance(fig.hf_data[0]["marker_color"], (tuple, list))
    assert fig.hf_data[0]["marker_size"] is None

    # Check correct hf values
    assert np.all(list(fig.hf_data[0]["text"]) == text)
    assert np.all(list(fig.hf_data[0]["marker_color"]) == marker_color)

    # Check correct trace values
    assert len(fig.data[0].y) == len(fig.data[0].text)
    assert len(fig.data[0].y) == len(fig.data[0].marker.color)
    y_color = ["black" if yi >= 0.99 else "blue" for yi in fig.data[0].y]
    assert np.all(list(fig.data[0].marker.color) == y_color)
    y_text = [f"text: {yi}, color:{ci}" for yi, ci in zip(fig.data[0].y, y_color)]
    assert np.all(list(fig.data[0].text) == y_text)


def test_hf_text_and_hf_hovertext_and_hf_marker_size_nans():
    y_orig = np.arange(10_000).astype(float)
    y = y_orig.copy()
    y[::101] = np.nan

    y_nonan = y[~np.isnan(y)]

    fig = FigureWidgetResampler()
    fig.add_trace(
        go.Scatter(
            name="blabla",
            text=y.astype(str),
            hovertext=y.astype(str)[::-1],
            marker={"size": y_orig},
        ),
        hf_y=y,
    )

    assert np.all(fig.hf_data[0]["text"] == y_nonan.astype(str))
    assert np.all(fig.hf_data[0]["hovertext"] == y_nonan.astype(str)[::-1])
    assert np.all(fig.hf_data[0]["marker_size"] == y_nonan)

    fig = FigureWidgetResampler()
    fig.add_trace(
        go.Scatter(name="blabla"),
        hf_y=y,
        hf_text=y.astype(str),
        hf_hovertext=y.astype(str)[::-1],
        hf_marker_size=y_orig,
    )

    assert np.all(fig.hf_data[0]["text"] == y_nonan.astype(str))
    assert np.all(fig.hf_data[0]["hovertext"] == y_nonan.astype(str)[::-1])
    assert np.all(fig.hf_data[0]["marker_size"] == y_nonan)


def test_multiple_timezones():
    n = 5_050

    dr = pd.date_range("2022-02-14", freq="s", periods=n, tz="UTC")
    dr_v = np.random.randn(n)

    cs = [
        dr,
        dr.tz_localize(None).tz_localize("Europe/Amsterdam"),
        dr.tz_convert("Europe/Brussels"),
        dr.tz_convert("Australia/Perth"),
        dr.tz_convert("Australia/Canberra"),
    ]

    plain_plotly_fig = make_subplots(rows=len(cs), cols=1, shared_xaxes=True)
    plain_plotly_fig.update_layout(height=min(300, 250 * len(cs)))

    fr_fig = FigureWidgetResampler(
        make_subplots(rows=len(cs), cols=1, shared_xaxes=True),
        default_n_shown_samples=500,
        convert_existing_traces=False,
        verbose=True,
    )
    fr_fig.update_layout(height=min(300, 250 * len(cs)))

    for i, date_range in enumerate(cs, 1):
        name = date_range.dtype.name.split(", ")[-1][:-1]
        plain_plotly_fig.add_trace(
            go.Scattergl(x=date_range, y=dr_v, name=name), row=i, col=1
        )
        fr_fig.add_trace(
            go.Scattergl(name=name),
            hf_x=date_range,
            hf_y=dr_v,
            row=i,
            col=1,
        )
        # Assert that the time parsing is exactly the same
        assert plain_plotly_fig.data[0].x[0] == fr_fig.data[0].x[0]


def test_datetime_hf_x_no_index():
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2020-01-01", "2020-01-02", freq="1s")}
    )
    df["value"] = np.random.randn(len(df))

    # add via hf_x kwargs
    fr = FigureWidgetResampler()
    fr.add_trace({}, hf_x=df.timestamp, hf_y=df.value)
    output = fr.construct_update_data(
        {
            "xaxis.range[0]": "2020-01-01 00:00:00",
            "xaxis.range[1]": "2020-01-01 00:00:20",
        }
    )
    assert len(output) == 2

    # add via scatter kwargs
    fr = FigureWidgetResampler()
    fr.add_trace(go.Scatter(x=df.timestamp, y=df.value))
    output = fr.construct_update_data(
        {
            "xaxis.range[0]": "2020-01-01 00:00:00",
            "xaxis.range[1]": "2020-01-01 00:00:20",
        }
    )
    assert len(output) == 2


def test_multiple_timezones_in_single_x_index__datetimes_and_timestamps():
    # TODO: can be improved with pytest parametrize
    y = np.arange(20)

    index1 = pd.date_range("2018-01-01", periods=10, freq="H", tz="US/Eastern")
    index2 = pd.date_range("2018-01-02", periods=10, freq="H", tz="Asia/Dubai")
    index_timestamps = index1.append(index2)
    assert all(isinstance(x, pd.Timestamp) for x in index_timestamps)
    index_datetimes = pd.Index([x.to_pydatetime() for x in index_timestamps])
    assert not any(isinstance(x, pd.Timestamp) for x in index_datetimes)
    assert all(isinstance(x, datetime.datetime) for x in index_datetimes)

    ## Test why we throw ValueError if array is still of object type after
    ## successful pd.to_datetime call
    # String array of datetimes with same tz -> NOT object array
    assert not pd.to_datetime(index1.astype("str")).dtype == "object"
    # String array of datetimes with multiple tz -> object array
    assert pd.to_datetime(index_timestamps.astype("str")).dtype == "object"
    assert pd.to_datetime(index_datetimes.astype("str")).dtype == "object"

    for index in [index_timestamps, index_datetimes]:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=index, y=y))
        with pytest.raises(ValueError):
            fr_fig = FigureWidgetResampler(fig, default_n_shown_samples=10)
        # Add as hf_x as index
        fr_fig = FigureWidgetResampler(default_n_shown_samples=10)
        with pytest.raises(ValueError):
            fr_fig.add_trace(go.Scattergl(), hf_x=index, hf_y=y)
        # Add as hf_x as object array of datetime values
        fr_fig = FigureWidgetResampler(default_n_shown_samples=10)
        with pytest.raises(ValueError):
            fr_fig.add_trace(go.Scattergl(), hf_x=index.values.astype("object"), hf_y=y)
        # Add as hf_x as string array
        fr_fig = FigureWidgetResampler(default_n_shown_samples=10)
        with pytest.raises(ValueError):
            fr_fig.add_trace(go.Scattergl(), hf_x=index.astype(str), hf_y=y)
        # Add as hf_x as object array of strings
        fr_fig = FigureWidgetResampler(default_n_shown_samples=10)
        with pytest.raises(ValueError):
            fr_fig.add_trace(
                go.Scattergl(), hf_x=index.astype(str).astype("object"), hf_y=y
            )

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=index.astype("object"), y=y))
        with pytest.raises(ValueError):
            fr_fig = FigureWidgetResampler(fig, default_n_shown_samples=10)

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=index.astype("str"), y=y))
        with pytest.raises(ValueError):
            fr_fig = FigureWidgetResampler(fig, default_n_shown_samples=10)


def test_proper_copy_of_wrapped_fig(float_series):
    plotly_fig = go.Figure()
    plotly_fig.add_trace(
        go.Scatter(
            x=float_series.index,
            y=float_series,
        )
    )

    plotly_resampler_fig = FigureWidgetResampler(
        plotly_fig, default_n_shown_samples=500
    )

    assert len(plotly_fig.data) == 1
    assert all(plotly_fig.data[0].x == float_series.index)
    assert all(plotly_fig.data[0].y == float_series.values)
    assert (len(plotly_fig.data[0].x) > 500) & (len(plotly_fig.data[0].y) > 500)

    assert len(plotly_resampler_fig.data) == 1
    assert len(plotly_resampler_fig.data[0].x) == 500
    assert len(plotly_resampler_fig.data[0].y) == 500


def test_2d_input_y():
    # Create some dummy dataframe with a nan
    df = pd.DataFrame(
        index=np.arange(5_000), data={"a": np.arange(5_000), "b": np.arange(5_000)}
    )
    df.iloc[42] = np.nan

    plotly_fig = go.Figure()
    plotly_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[["a"]],  # (100, 1) shape
        )
    )

    with pytest.raises(AssertionError) as e_info:
        _ = FigureWidgetResampler(  # does not alter plotly_fig
            plotly_fig,
            default_n_shown_samples=500,
        )
        assert "1 dimensional" in e_info


def test_hf_x_object_array():
    y = np.random.randn(100)

    ## Object array of datetime
    ### Should be parsed to a pd.DatetimeIndex (is more efficient than object array)
    x = pd.date_range("2020-01-01", freq="s", periods=100).astype("object")
    assert x.dtype == "object"
    assert isinstance(x[0], pd.Timestamp)
    # Add in the scatter
    fig = FigureWidgetResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla", x=x, y=y))
    assert isinstance(fig.hf_data[0]["x"], pd.DatetimeIndex)
    assert isinstance(fig.hf_data[0]["x"][0], pd.Timestamp)
    # Add as hf_x
    fig = FigureWidgetResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla"), hf_x=x, hf_y=y)
    assert isinstance(fig.hf_data[0]["x"], pd.DatetimeIndex)
    assert isinstance(fig.hf_data[0]["x"][0], pd.Timestamp)

    ## Object array of datetime strings
    ### Should be parsed to a pd.DatetimeIndex (is more efficient than object array)
    x = pd.date_range("2020-01-01", freq="s", periods=100).astype(str).astype("object")
    assert x.dtype == "object"
    assert isinstance(x[0], str)
    # Add in the scatter
    fig = FigureWidgetResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla", x=x, y=y))
    assert isinstance(fig.hf_data[0]["x"], pd.DatetimeIndex)
    assert isinstance(fig.hf_data[0]["x"][0], pd.Timestamp)
    # Add as hf_x
    fig = FigureWidgetResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla"), hf_x=x, hf_y=y)
    assert isinstance(fig.hf_data[0]["x"], pd.DatetimeIndex)
    assert isinstance(fig.hf_data[0]["x"][0], pd.Timestamp)

    ## Object array of ints
    ### Should be parsed to an int array (is more efficient than object array)
    x = np.arange(100).astype("object")
    assert x.dtype == "object"
    assert isinstance(x[0], int)
    # Add in the scatter
    fig = FigureWidgetResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla", x=x, y=y))
    assert np.issubdtype(fig.hf_data[0]["x"].dtype, np.integer)
    # Add as hf_x
    fig = FigureWidgetResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla"), hf_x=x, hf_y=y)
    assert np.issubdtype(fig.hf_data[0]["x"].dtype, np.integer)

    ## Object array of ints as strings
    ### Should be an integer array where the values are int objects
    x = np.arange(100).astype(str).astype("object")
    assert x.dtype == "object"
    assert isinstance(x[0], str)
    # Add in the scatter
    fig = FigureWidgetResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla", x=x, y=y))
    assert np.issubdtype(fig.hf_data[0]["x"].dtype, np.integer)
    # Add as hf_x
    fig = FigureWidgetResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla"), hf_x=x, hf_y=y)
    assert np.issubdtype(fig.hf_data[0]["x"].dtype, np.integer)

    ## Object array of strings
    x = np.array(["x", "y"] * 50).astype("object")
    assert x.dtype == "object"
    # Add in the scatter
    with pytest.raises(ValueError):
        fig = FigureWidgetResampler(default_n_shown_samples=50)
        fig.add_trace(go.Scatter(name="blabla", x=x, y=y))
    # Add as hf_x
    with pytest.raises(ValueError):
        fig = FigureWidgetResampler(default_n_shown_samples=50)
        fig.add_trace(go.Scatter(name="blabla"), hf_x=x, hf_y=y)


def test_check_update_figure_dict():
    # mostly written to test the check_update_figure_dict with
    # "updated_trace_indices" = None
    fr = FigureWidgetResampler(go.Figure())
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)
    fr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)
    fr._check_update_figure_dict(fr.to_dict())


def test_hf_data_property():
    fwr = FigureWidgetResampler(go.Figure(), default_n_shown_samples=2_000)
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)
    assert len(fwr.hf_data) == 0
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)
    assert len(fwr.hf_data) == 1
    assert len(fwr.hf_data[0]["x"]) == n
    fwr.hf_data[0]["x"] = x
    fwr.hf_data[0]["y"] = -2 * y
    assert np.all(fwr.hf_data[0]["x"] == x)
    assert np.all(fwr.hf_data[0]["y"] == y * -2)


def test_hf_data_property_reset_axes():
    fwr = FigureWidgetResampler(go.Figure(), default_n_shown_samples=2_000)
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)

    assert len(fwr.hf_data) == 0
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)

    fwr.layout.update(
        {"xaxis": {"range": [10_000, 20_000]}, "yaxis": {"range": [-20, 3]}},
        overwrite=False,
    )

    assert len(fwr.hf_data) == 1
    assert len(fwr.hf_data[0]["x"]) == n
    new_y = -2 * y
    fwr.hf_data[0]["y"] = new_y

    assert np.all(fwr.hf_data[0]["y"] == new_y)

    fwr.reset_axes()
    assert (fwr.data[0]["x"][0] <= 100) & (fwr.data[0]["x"][-1] >= 99_900)
    assert np.all(fwr.data[0]["y"] == new_y[fwr.data[0]["x"]])
    assert fwr.layout["yaxis"].range is None or fwr.layout["yaxis"].range[0] < -100


def test_hf_data_property_reload_data():
    fwr = FigureWidgetResampler(go.Figure(), default_n_shown_samples=2_000)
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)

    assert len(fwr.hf_data) == 0
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)

    fwr.layout.update(
        {"xaxis": {"range": [10_000, 20_000]}, "yaxis": {"range": [-20, 3]}},
        overwrite=False,
    )

    assert len(fwr.hf_data) == 1
    assert len(fwr.hf_data[0]["x"]) == n
    new_y = -2 * y
    fwr.hf_data[0]["y"] = new_y

    assert np.all(fwr.hf_data[0]["y"] == new_y)

    fwr.reload_data()
    assert (fwr.data[0]["x"][0] >= 10_000) & (fwr.data[0]["x"][-1] <= 20_000)
    assert np.all(fwr.data[0]["y"] == new_y[fwr.data[0]["x"]])
    assert (fwr.layout["yaxis"].range[0] == -20) & (fwr.layout["yaxis"].range[-1] == 3)


def test_hf_data_property_subplots_reset_axes():
    fwr = FigureWidgetResampler(make_subplots(rows=2, cols=1, shared_xaxes=False))
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)

    assert len(fwr.hf_data) == 0
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y, row=1, col=1)
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y, row=2, col=1)

    fwr.layout.update(
        {
            "xaxis": {"range": [10_000, 20_000]},
            "yaxis": {"range": [-20, 3]},
            "xaxis2": {"range": [40_000, 60_000]},
            "yaxis2": {"range": [-10, 3]},
        },
        overwrite=False,
    )

    assert len(fwr.hf_data) == 2
    assert len(fwr.hf_data[0]["x"]) == n
    assert len(fwr.hf_data[1]["x"]) == n
    new_y = -2 * y
    fwr.hf_data[0]["y"] = new_y
    fwr.hf_data[1]["y"] = new_y

    assert np.all(fwr.hf_data[0]["y"] == new_y)
    assert np.all(fwr.hf_data[0]["y"] == new_y)

    fwr.reset_axes()
    assert (fwr.data[0]["x"][0] <= 100) & (fwr.data[0]["x"][-1] >= 99_900)
    assert (fwr.data[1]["x"][0] <= 100) & (fwr.data[1]["x"][-1] >= 99_900)
    assert np.all(fwr.data[0]["y"] == new_y[fwr.data[0]["x"]])
    assert np.all(fwr.data[1]["y"] == new_y[fwr.data[1]["x"]])
    assert fwr.layout["yaxis"].range is None or fwr.layout["yaxis"].range[0] < -100
    assert fwr.layout["yaxis2"].range is None or fwr.layout["yaxis2"].range[0] < -100


def test_hf_data_property_subplots_reload_data():
    fwr = FigureWidgetResampler(make_subplots(rows=2, cols=1, shared_xaxes=False))
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)

    assert len(fwr.hf_data) == 0
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y, row=1, col=1)
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y, row=2, col=1)

    fwr.layout.update(
        {
            "xaxis": {"range": [10_000, 20_000]},
            "yaxis": {"range": [-20, 3]},
            "xaxis2": {"range": [40_000, 60_000]},
            "yaxis2": {"range": [-10, 3]},
        },
        overwrite=False,
    )

    assert len(fwr.hf_data) == 2
    assert len(fwr.hf_data[0]["x"]) == n
    assert len(fwr.hf_data[1]["x"]) == n
    new_y = -2 * y
    fwr.hf_data[0]["y"] = new_y
    fwr.hf_data[1]["y"] = new_y

    assert np.all(fwr.hf_data[0]["y"] == new_y)
    assert np.all(fwr.hf_data[0]["y"] == new_y)

    fwr.reload_data()
    assert (fwr.data[0]["x"][0] >= 10_000) & (fwr.data[0]["x"][-1] <= 20_000)
    assert (fwr.data[1]["x"][0] >= 40_000) & (fwr.data[1]["x"][-1] <= 60_000)
    assert np.all(fwr.data[0]["y"] == new_y[fwr.data[0]["x"]])
    assert np.all(fwr.data[1]["y"] == new_y[fwr.data[1]["x"]])
    assert (fwr.layout["yaxis"].range[0] == -20) & (fwr.layout["yaxis"].range[-1] == 3)
    assert (fwr.layout["yaxis2"].range[0] == -10) & (
        fwr.layout["yaxis2"].range[-1] == 3
    )


def test_hf_data_subplots_non_shared_xaxes():
    fwr = FigureWidgetResampler(make_subplots(rows=2, cols=1, shared_xaxes=False))
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)

    assert len(fwr.hf_data) == 0
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y, row=1, col=1)
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y, row=2, col=1)

    fwr.layout.update(
        {
            "xaxis2": {"range": [40_000, 60_000]},
            "yaxis2": {"range": [-10, 3]},
        },
        overwrite=False,
    )
    x_0 = fwr.data[0]["x"]
    assert 0 <= x_0[0] <= (n / 1000)
    assert (n - 1000) <= x_0[-1] <= n - 1
    x_1 = fwr.data[1]["x"]
    assert 40_000 <= x_1[0] <= 40_000 + (20_000 / 1000)
    assert (60_000 - 20_000 / 1_000) <= x_1[-1] <= 60_000


def test_hf_data_subplots_non_shared_xaxes_row_col_none():
    fwr = FigureWidgetResampler(make_subplots(rows=2, cols=1, shared_xaxes=False))
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)

    assert len(fwr.hf_data) == 0
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)
    fwr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y, row=2, col=1)

    fwr.layout.update(
        {
            "xaxis2": {"range": [40_000, 60_000]},
            "yaxis2": {"range": [-10, 3]},
        },
        overwrite=False,
    )
    x_0 = fwr.data[0]["x"]
    assert 0 <= x_0[0] <= (n / 1000)
    assert (n - 1000) <= x_0[-1] <= n - 1
    x_1 = fwr.data[1]["x"]
    assert 40_000 <= x_1[0] <= 40_000 + (20_000 / 1000)
    assert (60_000 - 20_000 / 1_000) <= x_1[-1] <= 60_000


def test_updates_two_traces():
    n = 1_000_000
    X = np.arange(n)
    Y = np.random.rand(n) / 5 + np.sin(np.arange(n) / 10000)

    fw_fig = FigureWidgetResampler(
        make_subplots(rows=2, shared_xaxes=False), verbose=True
    )
    fw_fig.update_layout(height=400, showlegend=True)

    fw_fig.add_trace(go.Scattergl(), hf_x=X, hf_y=(Y + 90) * X / 2000, row=1, col=1)
    fw_fig.add_trace(go.Scattergl(), hf_x=X, hf_y=(Y + 3) * 0.99999**X, row=2, col=1)

    # we do not want to have an relayout update
    assert len(fw_fig._relayout_hist) == 0

    # zoom in on both traces
    fw_fig.layout.update(
        {"xaxis": {"range": [10_000, 200_000]}, "xaxis2": {"range": [0, 200_000]}},
        overwrite=False,
    )

    # check whether the two traces were updated with the xaxis-range method
    assert ["xaxis-range-update", 2] in fw_fig._relayout_hist
    assert sum([["xaxis-range-update", 2] == rh for rh in fw_fig._relayout_hist]) == 1
    # check whether the showspikes update was did not enter the update state
    assert (
        sum(
            [
                "showspikes-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )

    # apply an autorange, see whether an update takes place
    fw_fig._relayout_hist.clear()
    fw_fig.layout.update({"xaxis": {"autorange": True}})
    fw_fig.layout.update({"xaxis2": {"autorange": True}})

    assert len(fw_fig._relayout_hist) == 0

    # Perform a reset axis update
    fw_fig.layout.update(
        {
            "xaxis": {"autorange": True, "showspikes": False},
            "xaxis2": {"autorange": True, "showspikes": False},
        }
    )

    # check whether the two traces were updated with the showspike method
    assert ["showspikes-update", 2] in fw_fig._relayout_hist
    assert sum([["showspikes-update", 2] == rh for rh in fw_fig._relayout_hist]) == 1
    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "xaxis-range-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )

    # RE-perform a reset axis update
    fw_fig._relayout_hist.clear()
    fw_fig.layout.update(
        {
            "xaxis": {"autorange": True, "showspikes": False},
            "xaxis2": {"autorange": True, "showspikes": False},
        }
    )

    # check whether none of the traces we updated with the showspike method
    assert ["showspikes-update", 1] not in fw_fig._relayout_hist
    assert ["showspikes-update", 2] not in fw_fig._relayout_hist
    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "xaxis-range-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )


def test_updates_two_traces_single_trace_adjust():
    n = 1_000_000
    X = np.arange(n)
    Y = np.random.rand(n) / 5 + np.sin(np.arange(n) / 10000)

    fw_fig = FigureWidgetResampler(
        make_subplots(rows=2, shared_xaxes=False), verbose=True
    )
    fw_fig.update_layout(height=400, showlegend=True)

    fw_fig.add_trace(go.Scattergl(), hf_x=X, hf_y=(Y + 90) * X / 2000, row=1, col=1)
    fw_fig.add_trace(go.Scattergl(), hf_x=X, hf_y=(Y + 3) * 0.99999**X, row=2, col=1)

    # we do not want to have an relayout update
    assert len(fw_fig._relayout_hist) == 0

    # zoom in on both traces
    fw_fig.layout.update(
        {"xaxis2": {"range": [0, 200_000]}},
        overwrite=False,
    )

    # check whether the single traces were updated with the xaxis-range method
    assert ["xaxis-range-update", 1] in fw_fig._relayout_hist
    assert ["xaxis-range-update", 2] not in fw_fig._relayout_hist
    assert sum([["xaxis-range-update", 1] == rh for rh in fw_fig._relayout_hist]) == 1

    # check whether the showspikes update was did not enter the update state
    assert (
        sum(
            [
                "showspikes-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )

    fw_fig._relayout_hist.clear()

    # apply an autorange, see whether an update takes place
    fw_fig.layout.update({"xaxis": {"autorange": True}})
    fw_fig.layout.update({"xaxis2": {"autorange": True}})

    assert len(fw_fig._relayout_hist) == 0

    # Perform a reset axis update
    fw_fig.layout.update(
        {
            "xaxis": {"autorange": True, "showspikes": False},
            "xaxis2": {"autorange": True, "showspikes": False},
        }
    )

    # check whether the single traces was updated with the showspike method
    assert ["showspikes-update", 1] in fw_fig._relayout_hist
    assert ["showspikes-update", 2] not in fw_fig._relayout_hist
    assert sum([["showspikes-update", 1] == rh for rh in fw_fig._relayout_hist]) == 1
    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "xaxis-range-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )

    fw_fig._relayout_hist.clear()

    # RE-perform a reset axis update
    #
    fw_fig.layout.update(
        {
            "xaxis": {"autorange": True, "showspikes": False},
            "xaxis2": {"autorange": True, "showspikes": False},
        }
    )

    # check whether none of the traces we updated with the showspike method
    assert ["showspikes-update", 1] not in fw_fig._relayout_hist
    assert ["showspikes-update", 2] not in fw_fig._relayout_hist
    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "xaxis-range-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )


def test_update_direct_reset_axis():
    n = 1_000_000
    X = np.arange(n)
    Y = np.random.rand(n) / 5 + np.sin(np.arange(n) / 10000)

    fw_fig = FigureWidgetResampler(
        make_subplots(rows=2, shared_xaxes=False), verbose=True
    )
    fw_fig.update_layout(height=400, showlegend=True)

    fw_fig.add_trace(go.Scattergl(), hf_x=X, hf_y=(Y + 90) * X / 2000, row=1, col=1)
    fw_fig.add_trace(go.Scattergl(), hf_x=X, hf_y=(Y + 3) * 0.99999**X, row=2, col=1)

    # we do not want to have an relayout update
    assert len(fw_fig._relayout_hist) == 0

    # Perform a reset_axis
    fw_fig.layout.update(
        {
            "xaxis": {"autorange": True, "showspikes": False},
            "xaxis2": {"autorange": True, "showspikes": False},
        }
    )

    # check whether the two traces was updated with the showspike method
    assert ["showspikes-update", 1] not in fw_fig._relayout_hist
    assert ["showspikes-update", 2] not in fw_fig._relayout_hist
    assert sum([["showspikes-update", 1] == rh for rh in fw_fig._relayout_hist]) == 0
    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "xaxis-range-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )


def test_bare_update_methods():
    n = 1_000_000
    X = np.arange(n)
    Y = np.random.rand(n) / 5 + np.sin(np.arange(n) / 10000)

    fw_fig = FigureWidgetResampler(
        make_subplots(rows=2, shared_xaxes=False), verbose=True
    )
    fw_fig.update_layout(height=400, showlegend=True)

    fw_fig.add_trace(go.Scattergl(), hf_x=X, hf_y=(Y + 90) * X / 2000, row=1, col=1)
    fw_fig.add_trace(go.Scattergl(), hf_x=X, hf_y=(Y + 3) * 0.99999**X, row=2, col=1)

    # equivalent of calling the reset-axis dict update
    fw_fig._update_spike_ranges(fw_fig.layout, False, False)
    fw_fig._update_spike_ranges(fw_fig.layout, False, False)

    assert ["showspikes-update", 1] not in fw_fig._relayout_hist
    assert ["showspikes-update", 2] not in fw_fig._relayout_hist
    assert sum([["showspikes-update", 1] == rh for rh in fw_fig._relayout_hist]) == 0

    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "xaxis-range-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )

    fw_fig._relayout_hist.clear()

    # Zoom in on the xaxis2
    fw_fig._update_x_ranges(
        copy(fw_fig.layout).update(
            {"xaxis2": {"range": [0, 200_000]}},
            overwrite=True,
        ),
        (0, len(X)),
        (0, 200_000),
    )

    # check whether the single traces were updated with the xaxis-range method
    assert ["xaxis-range-update", 1] in fw_fig._relayout_hist
    assert ["xaxis-range-update", 2] not in fw_fig._relayout_hist
    assert sum([["xaxis-range-update", 1] == rh for rh in fw_fig._relayout_hist]) == 1

    # check whether the showspikes update was did not enter the update state
    assert (
        sum(
            [
                "showspikes-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )

    # check whether the new update call (on the same range) does nothing
    fw_fig._relayout_hist.clear()
    fw_fig._update_x_ranges(
        copy(fw_fig.layout).update(
            {"xaxis2": {"range": [0, 200_000]}},
            overwrite=True,
        ),
        (0, len(X)),
        (0, 200_000),
    )

    # check whether none of the traces we updated with the showspike method
    assert ["showspikes-update", 1] not in fw_fig._relayout_hist
    assert ["showspikes-update", 2] not in fw_fig._relayout_hist
    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "xaxis-range-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )

    # Perform an autorange update -> assert that the range i
    fw_fig._relayout_hist.clear()
    fw_fig.layout.update({"xaxis2": {"autorange": True}, "yaxis2": {"autorange": True}})
    assert len(fw_fig._relayout_hist) == 0

    fw_fig.layout.update({"yaxis2": {"range": [0, 2]}})
    assert len(fw_fig._relayout_hist) == 0

    # perform an reset axis
    fw_fig._relayout_hist.clear()
    layout = fw_fig.layout.update(
        {
            "xaxis": {"autorange": True, "showspikes": False},
            "xaxis2": {"autorange": True, "showspikes": False},
        },
        overwrite=True,  # by setting this to true -> the update call will not takte clear
    )
    fw_fig._update_spike_ranges(layout, False, False)

    # Assert that only a single trace was updated
    assert ["showspikes-update", 1] in fw_fig._relayout_hist
    assert ["showspikes-update", 2] not in fw_fig._relayout_hist
    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "xaxis-range-update" in rh if isinstance(rh, list) else False
                for rh in fw_fig._relayout_hist
            ]
        )
        == 0
    )


def test_fwr_add_empty_trace():
    fig = FigureWidgetResampler(go.FigureWidget())
    fig.add_trace(go.Scattergl(name="Test"), limit_to_view=True)

    assert len(fig.hf_data) == 1
    assert len(fig.hf_data[0]["x"]) == 0
    assert len(fig.hf_data[0]["y"]) == 0


def test_fwr_update_trace_data_zoom():
    k = 50_000
    fig = FigureWidgetResampler(
        go.FigureWidget(make_subplots(rows=2, cols=1)), verbose=True
    )
    fig.add_trace(
        go.Scattergl(name="A", line_color="red"), limit_to_view=True, row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(name="B", line_color="green"), limit_to_view=True, row=2, col=1
    )

    fig._relayout_hist.clear()

    A = np.random.randn(k)
    fig.hf_data[0]["x"] = np.arange(k)
    fig.hf_data[0]["y"] = np.arange(k) + A * 300 * 20

    fig.hf_data[1]["x"] = fig.hf_data[0]["x"]
    fig.hf_data[1]["y"] = -np.arange(k) + A * 300 * 40
    fig.reload_data()

    # In the current implementation -> reload data will update all traces
    # Since there was no zoom-in event -> the `showspikes-update` will be called
    assert ["showspikes-update", 2] in fig._relayout_hist

    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "xaxis-range-update" in rh if isinstance(rh, list) else False
                for rh in fig._relayout_hist
            ]
        )
        == 0
    )

    # zoom in on the first row it's xaxis and perform a layout update
    layout = fig.layout.update(
        {"xaxis": {"range": [0, 100_000]}},
        overwrite=True,
    )
    fig._update_x_ranges(
        layout, (0, 100_000), (fig.hf_data[1]["x"][0], fig.hf_data[1]["x"][-1])
    )

    fig._relayout_hist.clear()

    fig.hf_data[1]["x"] = fig.hf_data[0]["x"]
    fig.hf_data[1]["y"] = -np.arange(k) + A * 300 * 10
    fig.reload_data()

    # In the current implementation -> reload data will update all traces
    # As we have performed a zoom event -> the`update showspikes` `will be called
    # TODO -> for some reason this assert does not succeed when not showing the graph
    # data
    # assert ["xaxis-range-update", len(fig.hf_data)] in fig._relayout_hist

    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "showspikes-update" in rh if isinstance(rh, list) else False
                for rh in fig._relayout_hist
            ]
        )
        == 0
    )

    # zoom in on the second row it's xaxis and perform a layout update
    layout = fig.layout.update(
        {"xaxis2": {"range": [200_000, 500_000]}},
        overwrite=True,
    )
    fig._update_x_ranges(layout, (0, 100_000), (200_000, 500_000))

    fig._relayout_hist.clear()
    fig.hf_data[0]["x"] = fig.hf_data[0]["x"]
    fig.hf_data[0]["y"] = -np.arange(k) + A * 300 * 39
    fig.hf_data[1]["x"] = fig.hf_data[0]["x"]
    fig.hf_data[1]["y"] = -np.arange(k) + A * 300 * 25
    fig.reload_data()

    # In the current implementation -> reload data will update all traces
    # As we have performed a zoom event -> the`update showspikes` `will be called
    assert ["xaxis-range-update", len(fig.hf_data)] in fig._relayout_hist

    # check whether the xaxis-range-update was did not enter the update state
    assert (
        sum(
            [
                "showspikes-update" in rh if isinstance(rh, list) else False
                for rh in fig._relayout_hist
            ]
        )
        == 0
    )


def test_fwr_text_update():
    k = 10_000
    fig = FigureWidgetResampler(default_n_shown_samples=1000, verbose=True)
    fig.add_trace(go.Scattergl(name="A", line_color="red"), limit_to_view=True)

    fig._relayout_hist.clear()

    A = np.random.randn(k)
    fig.hf_data[0]["x"] = np.arange(k)
    fig.hf_data[0]["y"] = np.arange(k) + A * 300 * 20
    fig.hf_data[0]["text"] = (-A * 20).astype(int).astype(str)
    fig.reload_data()

    assert ["showspikes-update", 1] in fig._relayout_hist
    assert ["xaxis-range-update", 1] not in fig._relayout_hist

    text = fig.data[0]["text"].astype(int)
    hovertext = fig.data[0]["hovertext"]

    assert len(text) == 1000
    assert hovertext is None


def test_fwr_hovertext_update():
    k = 10_000
    fig = FigureWidgetResampler(default_n_shown_samples=1000, verbose=True)
    fig.add_trace(go.Scattergl(name="B", line_color="red"), limit_to_view=True)

    fig._relayout_hist.clear()

    with fig.batch_update():
        A = np.random.randn(k)
        fig.hf_data[0]["x"] = np.arange(k)
        fig.hf_data[0]["y"] = np.arange(k) + A * 300 * 20
        fig.hf_data[0]["hovertext"] = (-A * 20).astype(int).astype(str)
        fig.reload_data()

    assert ["showspikes-update", 1] in fig._relayout_hist
    assert ["xaxis-range-update", 1] not in fig._relayout_hist

    text = fig.data[0]["text"]
    hovertext = fig.data[0]["hovertext"].astype(int)

    assert len(hovertext) == 1000
    assert text is None


def test_fwr_text_hovertext_update():
    k = 10_000
    fig = FigureWidgetResampler(default_n_shown_samples=1000, verbose=True)
    fig.add_trace(go.Scattergl(name="B", line_color="red"), limit_to_view=True)

    fig._relayout_hist.clear()

    with fig.batch_update():
        A = np.random.randn(k)
        fig.hf_data[0]["x"] = np.arange(k)
        fig.hf_data[0]["y"] = np.arange(k) + A * 300 * 20
        fig.hf_data[0]["text"] = (-A * 20).astype(int).astype(str)
        fig.hf_data[0]["hovertext"] = (A * 20).astype(int).astype(str)
        fig.reload_data()

    assert ["showspikes-update", 1] in fig._relayout_hist
    assert ["xaxis-range-update", 1] not in fig._relayout_hist

    text = fig.data[0]["text"].astype(int)
    hovertext = fig.data[0]["hovertext"].astype(int)

    assert len(hovertext) == 1000
    assert len(text) == 1000

    # text === -hovertext -> so the sum should their length
    assert (text == -hovertext).sum() == 1000


def test_fwr_adjust_text_unequal_length():
    k = 10_000
    fig = FigureWidgetResampler(default_n_shown_samples=1000, verbose=True)
    fig.add_trace(go.Scattergl(name="A", line_color="red"), limit_to_view=True)

    fig._relayout_hist.clear()

    with pytest.raises(ValueError):
        A = np.random.randn(k)
        fig.hf_data[0]["x"] = np.arange(k + 100)
        fig.hf_data[0]["y"] = np.arange(k + 100) + A * 300 * 20
        fig.hf_data[0]["text"] = (-A * 20).astype(int).astype(str)
        fig.reload_data()


def test_fwr_hovertext_adjust_unequal_length():
    k = 10_000
    fig = FigureWidgetResampler(default_n_shown_samples=1000, verbose=True)
    fig.add_trace(go.Scattergl(name="A", line_color="red"), limit_to_view=True)

    fig._relayout_hist.clear()

    with pytest.raises(ValueError):
        A = np.random.randn(k)
        fig.hf_data[0]["x"] = np.arange(k - 500)
        fig.hf_data[0]["y"] = np.arange(k - 500) + A * 300 * 20
        fig.hf_data[0]["hovertext"] = (-A * 20).astype(int).astype(str)
        fig.reload_data()


def test_fwr_adjust_series_input():
    k = 10_000
    a_k = np.arange(k)
    fig = FigureWidgetResampler(default_n_shown_samples=1000, verbose=True)
    fig.add_trace(go.Scattergl(name="A", line_color="red"), limit_to_view=True)

    fig._relayout_hist.clear()

    with fig.batch_update():
        A = np.random.randn(k)
        fig.hf_data[0]["x"] = pd.Series(index=a_k + 1_000_000, data=a_k - 2000)
        fig.hf_data[0]["y"] = pd.Series(index=a_k - 9999, data=a_k + 5 + np.abs(A) * 50)
        fig.reload_data()

    assert ["showspikes-update", 1] in fig._relayout_hist
    assert ["xaxis-range-update", 1] not in fig._relayout_hist

    x = fig.data[0]["x"]
    y = fig.data[0]["y"]

    # assert that hf x and y its values are used and not its index
    assert x[0] == -2000
    assert y[0] >= 5


def test_fwr_adjust_series_text_input():
    k = 10_000
    a_k = np.arange(k)
    fig = FigureWidgetResampler(default_n_shown_samples=1000, verbose=True)
    fig.add_trace(go.Scattergl(name="A", line_color="red"), limit_to_view=True)

    fig._relayout_hist.clear()

    with fig.batch_update():
        A = np.random.randn(k)
        fig.hf_data[0]["x"] = pd.Series(index=a_k + 10_000, data=a_k - 2000)
        fig.hf_data[0]["y"] = pd.Series(index=a_k, data=a_k + 10 + np.abs(A) * 50)
        fig.hf_data[0]["hovertext"] = pd.Series(
            index=a_k - 1_000_000, data=(-A * 20).astype(int).astype(str)
        )
        fig.hf_data[0]["text"] = pd.Series(
            index=a_k + 1_000_000, data=(A * 20).astype(int).astype(str)
        )
        fig.reload_data()

    assert ["showspikes-update", 1] in fig._relayout_hist
    assert ["xaxis-range-update", 1] not in fig._relayout_hist

    x = fig.data[0]["x"]
    y = fig.data[0]["y"]

    # assert that hf x and y its values are used and not its index
    assert x[0] == -2000
    assert y[0] >= 10

    text = fig.data[0]["text"].astype(int)
    hovertext = fig.data[0]["hovertext"].astype(int)

    assert len(hovertext) == 1000
    assert len(text) == 1000

    # text === -hovertext -> so the sum should their length
    assert (text == -hovertext).sum() == 1000


def test_fwr_time_based_data_ns():
    n = 100_000
    fig = FigureWidgetResampler(
        default_n_shown_samples=1000, verbose=True, default_downsampler=MinMaxLTTB()
    )

    for i in range(3):
        s = pd.Series(
            index=pd.date_range(
                datetime.datetime.now(),
                freq=f"{np.random.randint(5,100_000)}ns",
                periods=n,
            ),
            data=np.arange(n),
        )

        fig.add_trace(
            go.Scatter(name="hf_text"),
            hf_x=s.index,
            hf_y=s,
            hf_text=s.astype(str),
            hf_hovertext=(-s).astype(str),
        )

        x = fig.data[i]["x"]
        y = fig.data[i]["y"]

        assert len(x) == 1000
        assert len(y) == 1000

        text = fig.data[i]["text"].astype(int)
        hovertext = fig.data[i]["hovertext"].astype(int)

        assert len(hovertext) == 1000
        assert len(text) == 1000

        # text === -hovertext -> so the sum should their length
        assert (text == -hovertext).sum() == 1000


def test_fwr_time_based_data_us():
    n = 100_000
    fig = FigureWidgetResampler(
        default_n_shown_samples=1000, verbose=True, default_downsampler=MinMaxLTTB()
    )

    for i in range(3):
        s = pd.Series(
            index=pd.date_range(
                datetime.datetime.now(),
                freq=f"{np.random.randint(5,100_000)}us",
                periods=n,
            ),
            data=np.arange(n),
        )

        fig.add_trace(
            go.Scatter(name="hf_text"),
            hf_x=s.index,
            hf_y=s,
            hf_text=s.astype(str),
            hf_hovertext=(-s).astype(str),
        )

        x = fig.data[i]["x"]
        y = fig.data[i]["y"]

        assert len(x) == 1000
        assert len(y) == 1000

        text = fig.data[i]["text"].astype(int)
        hovertext = fig.data[i]["hovertext"].astype(int)

        assert len(hovertext) == 1000
        assert len(text) == 1000

        # text === -hovertext -> so the sum should their length
        assert (text == -hovertext).sum() == 1000


def test_fwr_time_based_data_ms():
    n = 100_000
    fig = FigureWidgetResampler(
        default_n_shown_samples=1000, verbose=True, default_downsampler=MinMaxLTTB()
    )

    for i in range(3):
        s = pd.Series(
            index=pd.date_range(
                datetime.datetime.now(),
                freq=f"{np.random.randint(5,10_000)}ms",
                periods=n,
            ),
            data=np.arange(n),
        )

        fig.add_trace(
            go.Scatter(name="hf_text"),
            hf_x=s.index,
            hf_y=s,
            hf_text=s.astype(str),
            hf_hovertext=(-s).astype(str),
        )

        x = fig.data[i]["x"]
        y = fig.data[i]["y"]

        assert len(x) == 1000
        assert len(y) == 1000

        text = fig.data[i]["text"].astype(int)
        hovertext = fig.data[i]["hovertext"].astype(int)

        assert len(hovertext) == 1000
        assert len(text) == 1000

        # text === -hovertext -> so the sum should their length
        assert (text == -hovertext).sum() == 1000


def test_fwr_time_based_data_s():
    # See: https://github.com/predict-idlab/plotly-resampler/issues/93
    n = 100_000
    fig = FigureWidgetResampler(
        default_n_shown_samples=1000, verbose=True, default_downsampler=MinMaxLTTB()
    )

    for i in range(3):
        s = pd.Series(
            index=pd.date_range(
                datetime.datetime.now(),
                freq=pd.Timedelta(f"{round(np.abs(np.random.randn()) * 1000, 4)}s"),
                periods=n,
            ),
            data=np.arange(n),
        )

        fig.add_trace(
            go.Scatter(name="hf_text"),
            hf_x=s.index,
            hf_y=s,
            hf_text=s.astype(str),
            hf_hovertext=(-s).astype(str),
        )

        x = fig.data[i]["x"]
        y = fig.data[i]["y"]

        assert len(x) == 1000
        assert len(y) == 1000

        text = fig.data[i]["text"].astype(int)
        hovertext = fig.data[i]["hovertext"].astype(int)

        assert len(hovertext) == 1000
        assert len(text) == 1000

        # text === -hovertext -> so the sum should their length
        assert (text == -hovertext).sum() == 1000


def test_fwr_from_trace_dict():
    y = np.array([1] * 10_000)
    base_fig = {
        "type": "scatter",
        "y": y,
    }

    fwr_fig = FigureWidgetResampler(base_fig, default_n_shown_samples=1000)
    assert len(fwr_fig.hf_data) == 1
    assert (fwr_fig.hf_data[0]["y"] == y).all()
    assert len(fwr_fig.data) == 1
    assert len(fwr_fig.data[0]["x"]) == 1_000
    assert (fwr_fig.data[0]["x"][0] >= 0) & (fwr_fig.data[0]["x"][-1] < 10_000)
    assert (fwr_fig.data[0]["y"] == [1] * 1_000).all()

    # assert that all the uuids of data and hf_data match
    # this is a proxy for assuring that the dynamic aggregation should work
    assert fwr_fig.data[0].uid in fwr_fig._hf_data


def test_fwr_from_figure_dict():
    y = np.array([1] * 10_000)
    base_fig = go.Figure()
    base_fig.add_trace(go.Scatter(y=y))

    fwr_fig = FigureWidgetResampler(base_fig.to_dict(), default_n_shown_samples=1000)
    assert len(fwr_fig.hf_data) == 1
    assert (fwr_fig.hf_data[0]["y"] == y).all()
    assert len(fwr_fig.data) == 1
    assert len(fwr_fig.data[0]["x"]) == 1_000
    assert (fwr_fig.data[0]["x"][0] >= 0) & (fwr_fig.data[0]["x"][-1] < 10_000)
    assert (fwr_fig.data[0]["y"] == [1] * 1_000).all()

    # assert that all the uuids of data and hf_data match
    # this is a proxy for assuring that the dynamic aggregation should work
    assert fwr_fig.data[0].uid in fwr_fig._hf_data


def test_fwr_empty_list():
    # and empty list -> so no concrete traces were added
    fr_fig = FigureWidgetResampler([], default_n_shown_samples=1000)
    assert len(fr_fig.hf_data) == 0
    assert len(fr_fig.data) == 0


def test_fwr_empty_dict():
    # a dict is a concrete trace so 1 trace should be added
    fr_fig = FigureWidgetResampler({}, default_n_shown_samples=1000)
    assert len(fr_fig._hf_data) == 0
    assert len(fr_fig.data) == 1


def test_fwr_wrong_keys(float_series):
    base_fig = [
        {"ydata": float_series.values + 2, "name": "sp2"},
    ]
    with pytest.raises(ValueError):
        FigureWidgetResampler(base_fig, default_n_shown_samples=1000)


def test_fwr_from_list_dict(float_series):
    base_fig: List[dict] = [
        {"y": float_series.values + 2, "name": "sp2"},
        {"y": float_series.values, "name": "s"},
    ]

    fr_fig = FigureWidgetResampler(base_fig, default_n_shown_samples=1000)
    assert len(fr_fig.hf_data) == 2
    assert (fr_fig.hf_data[0]["y"] == float_series + 2).all()
    assert (fr_fig.hf_data[1]["y"] == float_series).all()
    assert len(fr_fig.data) == 2
    assert len(fr_fig.data[0]["x"]) == 1_000
    assert (fr_fig.data[0]["x"][0] >= 0) & (fr_fig.data[0]["x"][-1] < 10_000)
    assert (fr_fig.data[1]["x"][0] >= 0) & (fr_fig.data[1]["x"][-1] < 10_000)

    # assert that all the uuids of data and hf_data match
    assert fr_fig.data[0].uid in fr_fig._hf_data
    assert fr_fig.data[1].uid in fr_fig._hf_data

    # redo the exercise with a new low-freq trace
    base_fig.append({"y": float_series[:1000], "name": "s_no_agg"})
    fr_fig = FigureWidgetResampler(base_fig, default_n_shown_samples=1000)
    assert len(fr_fig.hf_data) == 2
    assert len(fr_fig.data) == 3


def test_fwr_list_dict_add_trace(float_series):
    fr_fig = FigureWidgetResampler(default_n_shown_samples=1000)

    traces: List[dict] = [
        {"y": float_series.values + 2, "name": "sp2"},
        {"y": float_series.values, "name": "s"},
    ]
    for trace in traces:
        fr_fig.add_trace(trace)

    # both traces are HF traces so should be aggregated
    assert len(fr_fig.hf_data) == 2
    assert (fr_fig.hf_data[0]["y"] == float_series + 2).all()
    assert (fr_fig.hf_data[1]["y"] == float_series).all()
    assert len(fr_fig.data) == 2
    assert len(fr_fig.data[0]["x"]) == 1_000
    assert (fr_fig.data[0]["x"][0] >= 0) & (fr_fig.data[0]["x"][-1] < 10_000)
    assert (fr_fig.data[1]["x"][0] >= 0) & (fr_fig.data[1]["x"][-1] < 10_000)

    # assert that all the uuids of data and hf_data match
    assert fr_fig.data[0].uid in fr_fig._hf_data
    assert fr_fig.data[1].uid in fr_fig._hf_data

    # redo the exercise with a new low-freq trace
    fr_fig.add_trace({"y": float_series[:1000], "name": "s_no_agg"})
    assert len(fr_fig.hf_data) == 2
    assert len(fr_fig.data) == 3

    # add low-freq trace but set limit_to_view to True
    fr_fig.add_trace({"y": float_series[:100], "name": "s_agg"}, limit_to_view=True)
    assert len(fr_fig.hf_data) == 3
    assert len(fr_fig.data) == 4

    # add a low-freq trace but adjust max_n_samples
    lf_series = {"y": float_series[:1000], "name": "s_agg"}
    # plotly its default behavior raises a ValueError when a list or tuple is passed
    # to add_trace
    with pytest.raises(ValueError):
        fr_fig.add_trace([lf_series], max_n_samples=999)
    with pytest.raises(ValueError):
        fr_fig.add_trace((lf_series,), max_n_samples=999)

    fr_fig.add_trace(lf_series, max_n_samples=999)
    assert len(fr_fig.hf_data) == 4
    assert len(fr_fig.data) == 5


def test_fwr_list_dict_add_traces(float_series):
    fr_fig = FigureWidgetResampler(default_n_shown_samples=1000)

    traces: List[dict] = [
        {"y": float_series.values + 2, "name": "sp2"},
        {"y": float_series.values, "name": "s"},
    ]
    fr_fig.add_traces(traces)
    # both traces are HF traces so should be aggregated
    assert len(fr_fig.hf_data) == 2
    assert (fr_fig.hf_data[0]["y"] == float_series + 2).all()
    assert (fr_fig.hf_data[1]["y"] == float_series).all()
    assert len(fr_fig.data) == 2
    assert len(fr_fig.data[0]["x"]) == 1_000
    assert (fr_fig.data[0]["x"][0] >= 0) & (fr_fig.data[0]["x"][-1] < 10_000)
    assert (fr_fig.data[1]["x"][0] >= 0) & (fr_fig.data[1]["x"][-1] < 10_000)

    # assert that all the uuids of data and hf_data match
    assert fr_fig.data[0].uid in fr_fig._hf_data
    assert fr_fig.data[1].uid in fr_fig._hf_data

    # redo the exercise with a new low-freq trace
    # plotly also allows a dict or a scatter object as input
    fr_fig.add_traces({"y": float_series[:1000], "name": "s_no_agg"})
    assert len(fr_fig.hf_data) == 2
    assert len(fr_fig.data) == 3

    # add low-freq trace but set limit_to_view to True
    fr_fig.add_traces([{"y": float_series[:100], "name": "s_agg"}], limit_to_views=True)
    assert len(fr_fig.hf_data) == 3
    assert len(fr_fig.data) == 4

    # add a low-freq trace but adjust max_n_samples
    # note that we use tuple as input
    fr_fig.add_traces(({"y": float_series[:1000], "name": "s_agg"},), max_n_samples=999)
    assert len(fr_fig.hf_data) == 4
    assert len(fr_fig.data) == 5


def test_fwr_list_scatter_add_traces(float_series):
    fr_fig = FigureWidgetResampler(default_n_shown_samples=1000)

    traces: List[dict] = [
        go.Scattergl({"y": float_series.values + 2, "name": "sp2"}),
        go.Scatter({"y": float_series.values, "name": "s"}),
    ]
    fr_fig.add_traces(tuple(traces))
    # both traces are HF traces so should be aggregated
    assert len(fr_fig.hf_data) == 2
    assert (fr_fig.hf_data[0]["y"] == float_series + 2).all()
    assert (fr_fig.hf_data[1]["y"] == float_series).all()
    assert len(fr_fig.data) == 2
    assert len(fr_fig.data[0]["x"]) == 1_000
    assert (fr_fig.data[0]["x"][0] >= 0) & (fr_fig.data[0]["x"][-1] < 10_000)
    assert (fr_fig.data[1]["x"][0] >= 0) & (fr_fig.data[1]["x"][-1] < 10_000)

    # assert that all the uuids of data and hf_data match
    assert fr_fig.data[0].uid in fr_fig._hf_data
    assert fr_fig.data[1].uid in fr_fig._hf_data

    # redo the exercise with a new low-freq trace
    fr_fig.add_traces([go.Scattergl({"y": float_series[:1000], "name": "s_no_agg"})])
    assert len(fr_fig.hf_data) == 2
    assert len(fr_fig.data) == 3

    # add low-freq trace but set limit_to_view to True
    # note how the scatter object is not encapsulated within a list
    fr_fig.add_traces(go.Scattergl(), limit_to_views=True)
    assert len(fr_fig.hf_data) == 3
    assert len(fr_fig.data) == 4

    # add a low-freq trace but adjust max_n_samples
    fr_fig.add_traces(
        go.Scatter({"y": float_series[:1000], "name": "s_agg"}), max_n_samples=999
    )
    assert len(fr_fig.hf_data) == 4
    assert len(fr_fig.data) == 5


def test_fwr_add_scatter():
    # Checks whether the add_scatter method works as expected
    # .add_scatter calls `add_traces` under the hood
    fw_orig = go.FigureWidget().add_scatter(y=np.arange(2_000))
    fw_pr = FigureWidgetResampler().add_scatter(y=np.arange(2_000))

    assert len(fw_orig.data) == 1
    assert (len(fw_pr.data) == 1) & (len(fw_pr.hf_data) == 1)
    assert len(fw_orig.data[0].y) == 2_000
    assert len(fw_pr.data[0]["y"]) == 1_000
    assert np.all(fw_orig.data[0].y == fw_pr.hf_data[0]["y"])


def test_fwr_object_hf_data(
    float_series,
):
    float_series_o = float_series.astype(object)

    fig = FigureWidgetResampler()
    fig.add_trace({"name": "s0"}, hf_y=float_series_o)
    assert float_series_o.dtype == object
    assert len(fig.hf_data) == 1
    assert fig.hf_data[0]["y"].dtype == "float64"
    assert fig.data[0]["y"].dtype == "float64"


def test_fwr_object_bool_data(bool_series):
    # First try with the original non-object bool series
    fig = FigureWidgetResampler()
    fig.add_trace({"name": "s0"}, hf_y=bool_series)
    assert len(fig.hf_data) == 1
    assert fig.hf_data[0]["y"].dtype == "bool"
    # plotly internally ocnverts this to object
    assert fig.data[0]["y"].dtype == "object"

    # Now try with the object bool series
    bool_series_o = bool_series.astype(object)

    fig = FigureWidgetResampler()
    fig.add_trace({"name": "s0"}, hf_y=bool_series_o)
    assert bool_series_o.dtype == object
    assert len(fig.hf_data) == 1
    assert fig.hf_data[0]["y"].dtype == "bool"
    # plotly internally ocnverts this to object
    assert fig.data[0]["y"].dtype == "object"


def test_fwr_object_binary_data():
    binary_series = np.array(
        [0, 1] * 20, dtype="int32"
    )  # as this is << max_n_samples -> limit_to_view

    # First try with the original non-object binary series
    fig = FigureWidgetResampler()
    fig.add_trace({"name": "s0"}, hf_y=binary_series, limit_to_view=True)
    assert len(fig.hf_data) == 1
    assert fig.hf_data[0]["y"].dtype == "int32"
    assert str(fig.data[0]["y"].dtype).startswith("int")
    assert np.all(fig.data[0]["y"] == binary_series)

    # Now try with the object binary series
    binary_series_o = binary_series.astype(object)

    fig = FigureWidgetResampler()
    fig.add_trace({"name": "s0"}, hf_y=binary_series_o, limit_to_view=True)
    assert binary_series_o.dtype == object
    assert len(fig.hf_data) == 1
    assert (fig.hf_data[0]["y"].dtype == "int32") or (
        fig.hf_data[0]["y"].dtype == "int64"
    )
    assert str(fig.data[0]["y"].dtype).startswith("int")
    assert np.all(fig.data[0]["y"] == binary_series)


def test_fwr_update_layout_axes_range():
    nb_datapoints = 2_000
    n_shown = 500  # < nb_datapoints

    # Checks whether the update_layout method works as expected
    f_orig = go.Figure().add_scatter(y=np.arange(nb_datapoints))
    f_pr = FigureWidgetResampler(default_n_shown_samples=n_shown).add_scatter(
        y=np.arange(nb_datapoints)
    )

    def check_data(fwr: FigureWidgetResampler, min_v=0, max_v=nb_datapoints - 1):
        # closure for n_shown and nb_datapoints
        assert len(fwr.data[0]["y"]) == min(n_shown, nb_datapoints)
        assert len(fwr.data[0]["x"]) == min(n_shown, nb_datapoints)
        assert fwr.data[0]["y"][0] == min_v
        assert fwr.data[0]["y"][-1] == max_v
        assert fwr.data[0]["x"][0] == min_v
        assert fwr.data[0]["x"][-1] == max_v

    # Check the initial data
    check_data(f_pr)

    # The xaxis (auto)range should be the same for both figures

    assert f_orig.layout.xaxis.range is None
    assert f_pr.layout.xaxis.range is None
    assert f_orig.layout.xaxis.autorange is None
    assert f_pr.layout.xaxis.autorange is None

    f_orig.update_layout(xaxis_range=[100, 1000])
    f_pr.update_layout(xaxis_range=[100, 1000])

    assert f_orig.layout.xaxis.range == (100, 1000)
    assert f_pr.layout.xaxis.range == (100, 1000)
    assert f_orig.layout.xaxis.autorange is None
    assert f_pr.layout.xaxis.autorange is None

    # The yaxis (auto)range should be the same for both figures

    assert f_orig.layout.yaxis.range is None
    assert f_pr.layout.yaxis.range is None
    assert f_orig.layout.yaxis.autorange is None
    assert f_pr.layout.yaxis.autorange is None

    f_orig.update_layout(yaxis_range=[100, 1000])
    f_pr.update_layout(yaxis_range=[100, 1000])

    assert list(f_orig.layout.yaxis.range) == [100, 1000]
    assert list(f_pr.layout.yaxis.range) == [100, 1000]
    assert f_orig.layout.yaxis.autorange is None
    assert f_pr.layout.yaxis.autorange is None

    # Now the f_pr contains the data of the selected xrange (downsampled to 500 samples)
    check_data(f_pr, 100, 1_000 - 1)


def test_fwr_update_layout_axes_range_no_update():
    nb_datapoints = 2_000
    n_shown = 20_000  # > nb. datapoints

    # Checks whether the update_layout method works as expected
    f_orig = go.Figure().add_scatter(y=np.arange(nb_datapoints))
    f_pr = FigureWidgetResampler(default_n_shown_samples=n_shown).add_scatter(
        y=np.arange(nb_datapoints)
    )

    def check_data(fwr: FigureWidgetResampler, min_v=0, max_v=nb_datapoints - 1):
        # closure for n_shown and nb_datapoints
        assert len(fwr.data[0]["y"]) == min(n_shown, nb_datapoints)
        assert len(fwr.data[0]["x"]) == min(n_shown, nb_datapoints)
        assert fwr.data[0]["y"][0] == min_v
        assert fwr.data[0]["y"][-1] == max_v
        assert fwr.data[0]["x"][0] == min_v
        assert fwr.data[0]["x"][-1] == max_v

    # Check the initial data
    check_data(f_pr)

    # The xaxis (auto)range should be the same for both figures

    assert f_orig.layout.xaxis.range is None
    assert f_pr.layout.xaxis.range is None
    assert f_orig.layout.xaxis.autorange is None
    assert f_pr.layout.xaxis.autorange is None

    f_orig.update_layout(xaxis_range=[100, 1000])
    f_pr.update_layout(xaxis_range=[100, 1000])

    assert f_orig.layout.xaxis.range == (100, 1000)
    assert f_pr.layout.xaxis.range == (100, 1000)
    assert f_orig.layout.xaxis.autorange is None
    assert f_pr.layout.xaxis.autorange is None

    # The yaxis (auto)range should be the same for both figures

    assert f_orig.layout.yaxis.range is None
    assert f_pr.layout.yaxis.range is None
    assert f_orig.layout.yaxis.autorange is None
    assert f_pr.layout.yaxis.autorange is None

    f_orig.update_layout(yaxis_range=[100, 1000])
    f_pr.update_layout(yaxis_range=[100, 1000])

    assert list(f_orig.layout.yaxis.range) == [100, 1000]
    assert list(f_pr.layout.yaxis.range) == [100, 1000]
    assert f_orig.layout.yaxis.autorange is None
    assert f_pr.layout.yaxis.autorange is None

    # Now the f_pr still contains the full original data (not downsampled)
    # Even after updating the axes ranges
    check_data(f_pr)


def test_fwr_copy_grid():
    # Checks whether _grid_ref and _grid_str are correctly maintained

    f = make_subplots(rows=2, cols=1)
    f.add_scatter(y=np.arange(2_000), row=1, col=1)
    f.add_scatter(y=np.arange(2_000), row=2, col=1)

    ## go.Figure
    assert isinstance(f, go.Figure)
    assert f._grid_ref is not None
    assert f._grid_str is not None
    fwr = FigureWidgetResampler(f)
    assert fwr._grid_ref is not None
    assert fwr._grid_ref == f._grid_ref
    assert fwr._grid_str is not None
    assert fwr._grid_str == f._grid_str

    ## go.FigureWidget
    fw = go.FigureWidget(f)
    assert fw._grid_ref is not None
    assert fw._grid_str is not None
    assert isinstance(fw, go.FigureWidget)
    fwr = FigureWidgetResampler(fw)
    assert fwr._grid_ref is not None
    assert fwr._grid_ref == fw._grid_ref
    assert fwr._grid_str is not None
    assert fwr._grid_str == fw._grid_str

    ## FigureWidgetResampler
    fwr_ = FigureWidgetResampler(f)
    assert fwr_._grid_ref is not None
    assert fwr_._grid_str is not None
    assert isinstance(fwr_, FigureWidgetResampler)
    fwr = FigureWidgetResampler(fwr_)
    assert fwr._grid_ref is not None
    assert fwr._grid_ref == fwr_._grid_ref
    assert fwr._grid_str is not None
    assert fwr._grid_str == fwr_._grid_str

    ## FigureResampler
    from plotly_resampler import FigureResampler

    fr = FigureResampler(f)
    assert fr._grid_ref is not None
    assert fr._grid_str is not None
    assert isinstance(fr, FigureResampler)
    fwr = FigureWidgetResampler(fr)
    assert fwr._grid_ref is not None
    assert fwr._grid_ref == fr._grid_ref
    assert fwr._grid_str is not None
    assert fwr._grid_str == fr._grid_str

    ## dict (with no _grid_ref & no _grid_str)
    f_dict = f.to_dict()
    assert isinstance(f_dict, dict)
    assert f_dict.get("_grid_ref") is None
    assert f_dict.get("_grid_str") is None
    fwr = FigureWidgetResampler(f_dict)
    assert fwr._grid_ref is f_dict.get("_grid_ref")  # both are None
    assert fwr._grid_str is f_dict.get("_grid_str")  # both are None

    ## dict (with _grid_ref & _grid_str)
    f_dict = f.to_dict()
    f_dict["_grid_ref"] = f._grid_ref
    f_dict["_grid_str"] = f._grid_str
    assert isinstance(f_dict, dict)
    assert f_dict.get("_grid_ref") is not None
    fwr = FigureWidgetResampler(f_dict)
    assert fwr._grid_ref is not None
    assert fwr._grid_ref == f_dict.get("_grid_ref")
    assert fwr._grid_str is not None
    assert fwr._grid_str == f_dict.get("_grid_str")
