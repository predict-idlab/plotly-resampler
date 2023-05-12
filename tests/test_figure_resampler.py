"""Code which tests the FigureResampler functionalities"""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"


import datetime
import multiprocessing
import subprocess
import time
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots
from selenium.webdriver.common.by import By

from plotly_resampler import LTTB, EveryNthPoint, FigureResampler
from plotly_resampler.aggregation import NoGapHandler, PlotlyAggregatorParser

# Note: this will be used to skip / alter behavior when running browser tests on
# non-linux platforms.
from .utils import construct_hf_data_dict, not_on_linux


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
            "default_downsampler": LTTB(),
            "resampled_trace_prefix_suffix": tuple(["<b>[r]</b>", "~~"]),
            "verbose": True,
        },
    ]
    for kwarg_space in kwarg_space_list:
        fig = FigureResampler(base_fig, **kwarg_space)

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

    fig = FigureResampler(base_fig, default_n_shown_samples=1000)

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


def test_various_dtypes(float_series):
    # List of dtypes supported by orjson >= 3.8
    valid_dtype_list = [
        np.bool_,
        # ---- uints
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        # -------- ints
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        # -------- floats
        np.float16,  # currently not supported by orjson
        np.float32,
        np.float64,
    ]
    for dtype in valid_dtype_list:
        fig = FigureResampler(go.Figure(), default_n_shown_samples=1000)
        # nb. datapoints > default_n_shown_samples
        fig.add_trace(
            go.Scatter(name="float_series"),
            hf_x=float_series.index,
            hf_y=float_series.astype(dtype),
        )
        fig.full_figure_for_development()

    # List of dtypes not supported by orjson >= 3.8
    invalid_dtype_list = [np.float16]
    for invalid_dtype in invalid_dtype_list:
        fig = FigureResampler(go.Figure(), default_n_shown_samples=1000)
        # nb. datapoints < default_n_shown_samples
        with pytest.raises(TypeError):
            # if this test fails -> orjson supports f16 => remove casting frome code
            fig.add_trace(
                go.Scatter(name="float_series"),
                hf_x=float_series.index[:500],
                hf_y=float_series.astype(invalid_dtype)[:500],
            )
            fig.full_figure_for_development()


def test_max_n_samples(float_series):
    s = float_series[:5000]

    fig = FigureResampler()
    fig.add_trace(
        go.Scattergl(name="test"), hf_x=s.index, hf_y=s, max_n_samples=len(s) + 1
    )
    # make sure that there is not hf_data
    assert len(fig.hf_data) == 0
    assert len(fig.data[0]["x"]) == len(s)


def test_add_scatter_trace_no_data():
    fig = FigureResampler(default_n_shown_samples=1000)

    # no x and y data
    fig.add_trace(go.Scatter())


def test_add_scatter_trace_no_x():
    fig = FigureResampler(go.Figure(), default_n_shown_samples=1000)

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

    fig = FigureResampler(base_fig, default_n_shown_samples=1000, verbose=True)

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

    fig = FigureResampler(base_fig, default_n_shown_samples=1000, verbose=True)

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


def test_add_traces_from_other_figure():
    labels = ["Investing", "Liquid", "Real Estate", "Retirement"]
    values = [324643.4435821581, 112238.37140194925, 2710711.06, 604360.2864262027]

    changes_section = FigureResampler(
        make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Asset Allocation", "Changes in last 12 hours"),
            specs=[[{"type": "pie"}, {"type": "xy"}]],
        )
    )

    # First create a pie chart Figure
    pie_total = go.Figure(data=[go.Pie(labels=labels, values=values)])

    # Add the pie chart traces to the changes_section figure
    for trace in pie_total.data:
        changes_section.add_trace(trace, row=1, col=1)


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
    fig = FigureResampler(base_fig, default_n_shown_samples=1000, verbose=True)

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

    fr_fig = FigureResampler(base_fig, default_n_shown_samples=1000)

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


def test_replace_properties(float_series):
    resampled_trace_prefix_suffix = ("a", "b")
    verbose = True
    default_n_shown_samples = 1050
    default_gap_handler = NoGapHandler()
    default_downsampler = EveryNthPoint()
    fr_fig = FigureResampler(
        default_n_shown_samples=default_n_shown_samples,
        verbose=verbose,
        resampled_trace_prefix_suffix=resampled_trace_prefix_suffix,
        default_gap_handler=default_gap_handler,
        default_downsampler=default_downsampler,
    )

    fr_fig.add_trace(go.Scattergl(x=float_series.index, y=float_series, name="fs"))
    fr_fig.replace(go.Figure())

    assert fr_fig._global_n_shown_samples == default_n_shown_samples
    assert fr_fig._print_verbose == verbose
    assert (fr_fig._prefix, fr_fig._suffix) == resampled_trace_prefix_suffix
    assert fr_fig._global_gap_handler == default_gap_handler
    assert fr_fig._global_downsampler == default_downsampler


def test_nan_removed_input(float_series):
    # see: https://plotly.com/python/subplots/#custom-sized-subplot-with-subplot-titles
    base_fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    fig = FigureResampler(
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

    fig = FigureResampler(
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


def test_hf_text():
    y = np.arange(10_000)

    fig = FigureResampler()
    fig.add_trace(
        go.Scatter(name="blabla", text=y.astype(str)),
        hf_y=y,
    )

    assert np.all(fig.hf_data[0]["text"] == y.astype(str))
    assert fig.hf_data[0]["hovertext"] is None

    assert len(fig.data[0].y) < 5_000
    assert np.all(fig.data[0].text == fig.data[0].y.astype(int).astype(str))
    assert fig.data[0].hovertext is None

    fig = FigureResampler()
    fig.add_trace(go.Scatter(name="blabla"), hf_y=y, hf_text=y.astype(str))

    assert np.all(fig.hf_data[0]["text"] == y.astype(str))
    assert fig.hf_data[0]["hovertext"] is None

    assert len(fig.data[0].y) < 5_000
    assert np.all(fig.data[0].text == fig.data[0].y.astype(int).astype(str))
    assert fig.data[0].hovertext is None


def test_hf_hovertext():
    y = np.arange(10_000)

    fig = FigureResampler()
    fig.add_trace(
        go.Scatter(name="blabla", hovertext=y.astype(str)),
        hf_y=y,
    )

    assert np.all(fig.hf_data[0]["hovertext"] == y.astype(str))
    assert fig.hf_data[0]["text"] is None

    assert len(fig.data[0].y) < 5_000
    assert np.all(fig.data[0].hovertext == fig.data[0].y.astype(int).astype(str))
    assert fig.data[0].text is None

    fig = FigureResampler()
    fig.add_trace(go.Scatter(name="blabla"), hf_y=y, hf_hovertext=y.astype(str))

    assert np.all(fig.hf_data[0]["hovertext"] == y.astype(str))
    assert fig.hf_data[0]["text"] is None

    assert len(fig.data[0].y) < 5_000
    assert np.all(fig.data[0].hovertext == fig.data[0].y.astype(int).astype(str))
    assert fig.data[0].text is None


def test_hf_text_and_hf_hovertext():
    y = np.arange(10_000)

    fig = FigureResampler()
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

    fig = FigureResampler()
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


def test_multiple_timezones():
    n = 5_050

    # NOTE: date-range returns a (tz-aware) DatetimeIndex
    dr = pd.date_range("2022-02-14", freq="s", periods=n, tz="UTC")
    dr_v = np.random.randn(n)

    cs = [
        dr,
        dr.tz_localize(None).tz_localize("Europe/Amsterdam"),
        dr.tz_convert("Europe/Brussels"),
        dr.tz_convert("Australia/Perth"),
        dr.tz_convert("Australia/Canberra"),
        # NOTE: this pd.Series tests the functionality of a Pandas series with (tz-aware) DatetimeIndex
        pd.Series(dr),
        pd.Series(dr.tz_localize(None).tz_localize("Europe/Amsterdam")),
        pd.Series(dr.tz_convert("Europe/Brussels")),
        pd.Series(dr.tz_convert("Australia/Perth")),
        pd.Series(dr.tz_convert("Australia/Canberra")),
    ]

    plain_plotly_fig = make_subplots(rows=len(cs), cols=1, shared_xaxes=True)
    plain_plotly_fig.update_layout(height=min(300, 250 * len(cs)))

    fr_fig = FigureResampler(
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
        assert plain_plotly_fig.data[i - 1].x[0] == fr_fig.data[i - 1].x[0]


def test_set_hfx_tz_aware_series():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01", "2020-01-02", freq="1s"
            ).tz_localize("Asia/Seoul")
        }
    )
    df["value"] = np.random.randn(len(df))

    fr = FigureResampler()
    fr.add_trace({}, hf_x=pd.Index(df.timestamp), hf_y=df.value)
    assert isinstance(fr.hf_data[0]["x"], pd.DatetimeIndex)
    # Now we set the pd.Series as hf_x
    fr.hf_data[0]["x"] = df.timestamp
    assert not isinstance(fr.hf_data[0]["x"], pd.DatetimeIndex)
    # perform an update
    out = fr.construct_update_data({"xaxis.autorange": True, "xaxis.showspikes": False})
    assert len(out) == 2
    # assert that the update was performed correctly
    assert isinstance(fr.hf_data[0]["x"], pd.DatetimeIndex)
    assert all(fr.hf_data[0]["x"] == pd.DatetimeIndex(df.timestamp))


def test_datetime_hf_x_no_index_():
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2020-01-01", "2020-01-02", freq="1s")}
    )
    df["value"] = np.random.randn(len(df))

    # add via hf_x kwargs
    fr = FigureResampler()
    fr.add_trace({}, hf_x=df.timestamp, hf_y=df.value)
    output = fr.construct_update_data(
        {
            "xaxis.range[0]": "2020-01-01 00:00:00",
            "xaxis.range[1]": "2020-01-01 00:00:20",
        }
    )
    assert len(output) == 2

    # add via scatter kwargs
    fr = FigureResampler()
    fr.add_trace(go.Scatter(x=df.timestamp, y=df.value))
    output = fr.construct_update_data(
        {
            "xaxis.range[0]": "2020-01-01 00:00:00",
            "xaxis.range[1]": "2020-01-01 00:00:20",
        }
    )
    assert len(output) == 2


def test_datetime_hf_x_no_index():
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2020-01-01", "2020-01-02", freq="1s")}
    )
    df["value"] = np.random.randn(len(df))

    # add via hf_x kwargs
    fr = FigureResampler()
    fr.add_trace({}, hf_x=df.timestamp, hf_y=df.value)
    output = fr.construct_update_data(
        {
            "xaxis.range[0]": "2020-01-01 00:00:00",
            "xaxis.range[1]": "2020-01-01 00:00:20",
        }
    )
    assert len(output) == 2

    # add via scatter kwargs
    fr = FigureResampler()
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
    index1_datetimes = pd.Index([x.to_pydatetime() for x in index1])
    index_datetimes = pd.Index([x.to_pydatetime() for x in index_timestamps])
    assert not any(isinstance(x, pd.Timestamp) for x in index_datetimes)
    assert all(isinstance(x, datetime.datetime) for x in index_datetimes)

    ## Test why we throw ValueError if array is still of object type after
    ## successful pd.to_datetime call
    # String array of datetimes with same tz -> NOT object array
    assert not pd.to_datetime(index1.astype("str")).dtype == "object"
    assert not pd.to_datetime(index1_datetimes.astype("str")).dtype == "object"
    # String array of datetimes with multiple tz -> object array
    assert pd.to_datetime(index_timestamps.astype("str")).dtype == "object"
    assert pd.to_datetime(index_datetimes.astype("str")).dtype == "object"

    for index in [index_timestamps, index_datetimes]:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=index, y=y))
        with pytest.raises(ValueError):
            fr_fig = FigureResampler(fig, default_n_shown_samples=10)
        # Add as hf_x as index
        fr_fig = FigureResampler(default_n_shown_samples=10)
        with pytest.raises(ValueError):
            fr_fig.add_trace(go.Scattergl(), hf_x=index, hf_y=y)
        # Add as hf_x as object array of datetime values
        fr_fig = FigureResampler(default_n_shown_samples=10)
        with pytest.raises(ValueError):
            fr_fig.add_trace(go.Scattergl(), hf_x=index.values.astype("object"), hf_y=y)
        # Add as hf_x as string array
        fr_fig = FigureResampler(default_n_shown_samples=10)
        with pytest.raises(ValueError):
            fr_fig.add_trace(go.Scattergl(), hf_x=index.astype(str), hf_y=y)
        # Add as hf_x as object array of strings
        fr_fig = FigureResampler(default_n_shown_samples=10)
        with pytest.raises(ValueError):
            fr_fig.add_trace(
                go.Scattergl(), hf_x=index.astype(str).astype("object"), hf_y=y
            )

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=index.astype("object"), y=y))
        with pytest.raises(ValueError):
            fr_fig = FigureResampler(fig, default_n_shown_samples=10)

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=index.astype("str"), y=y))
        with pytest.raises(ValueError):
            fr_fig = FigureResampler(fig, default_n_shown_samples=10)


def test_proper_copy_of_wrapped_fig(float_series):
    plotly_fig = go.Figure()
    plotly_fig.add_trace(
        go.Scatter(
            x=float_series.index,
            y=float_series,
        )
    )

    plotly_resampler_fig = FigureResampler(plotly_fig, default_n_shown_samples=500)

    assert len(plotly_fig.data) == 1
    assert all(plotly_fig.data[0].x == float_series.index)
    assert all(plotly_fig.data[0].y == float_series.values)
    assert (len(plotly_fig.data[0].x) > 500) & (len(plotly_fig.data[0].y) > 500)

    assert len(plotly_resampler_fig.data) == 1
    assert len(plotly_resampler_fig.data[0].x) == 500
    assert len(plotly_resampler_fig.data[0].y) == 500


def test_low_dim_input():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="a"))

    fig = FigureResampler(go.Figure())
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="a"))
    fig.add_trace(go.Scatter(), hf_x=[1, 2, 3], hf_y=[1, 2, 3])


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
        _ = FigureResampler(  # does not alter plotly_fig
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
    fig = FigureResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla", x=x, y=y))
    assert isinstance(fig.hf_data[0]["x"], pd.DatetimeIndex)
    assert isinstance(fig.hf_data[0]["x"][0], pd.Timestamp)
    # Add as hf_x
    fig = FigureResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla"), hf_x=x, hf_y=y)
    assert isinstance(fig.hf_data[0]["x"], pd.DatetimeIndex)
    assert isinstance(fig.hf_data[0]["x"][0], pd.Timestamp)

    ## Object array of datetime strings
    ### Should be parsed to a pd.DatetimeIndex (is more efficient than object array)
    x = pd.date_range("2020-01-01", freq="s", periods=100).astype(str).astype("object")
    assert x.dtype == "object"
    assert isinstance(x[0], str)
    # Add in the scatter
    fig = FigureResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla", x=x, y=y))
    assert isinstance(fig.hf_data[0]["x"], pd.DatetimeIndex)
    assert isinstance(fig.hf_data[0]["x"][0], pd.Timestamp)
    # Add as hf_x
    fig = FigureResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla"), hf_x=x, hf_y=y)
    assert isinstance(fig.hf_data[0]["x"], pd.DatetimeIndex)
    assert isinstance(fig.hf_data[0]["x"][0], pd.Timestamp)

    ## Object array of ints
    ### Should be parsed to an int array (is more efficient than object array)
    x = np.arange(100).astype("object")
    assert x.dtype == "object"
    assert isinstance(x[0], int)
    # Add in the scatter
    fig = FigureResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla", x=x, y=y))
    assert np.issubdtype(fig.hf_data[0]["x"].dtype, np.integer)
    # Add as hf_x
    fig = FigureResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla"), hf_x=x, hf_y=y)
    assert np.issubdtype(fig.hf_data[0]["x"].dtype, np.integer)

    ## Object array of ints as strings
    ### Should be an integer array where the values are int objects
    x = np.arange(100).astype(str).astype("object")
    assert x.dtype == "object"
    assert isinstance(x[0], str)
    # Add in the scatter
    fig = FigureResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla", x=x, y=y))
    assert np.issubdtype(fig.hf_data[0]["x"].dtype, np.integer)
    # Add as hf_x
    fig = FigureResampler(default_n_shown_samples=50)
    fig.add_trace(go.Scatter(name="blabla"), hf_x=x, hf_y=y)
    assert np.issubdtype(fig.hf_data[0]["x"].dtype, np.integer)

    ## Object array of strings
    x = np.array(["x", "y"] * 50).astype("object")
    assert x.dtype == "object"
    # Add in the scatter
    with pytest.raises(ValueError):
        fig = FigureResampler(default_n_shown_samples=50)
        fig.add_trace(go.Scatter(name="blabla", x=x, y=y))
    # Add as hf_x
    with pytest.raises(ValueError):
        fig = FigureResampler(default_n_shown_samples=50)
        fig.add_trace(go.Scatter(name="blabla"), hf_x=x, hf_y=y)


def test_time_tz_slicing():
    n = 5050
    dr = pd.Series(
        index=pd.date_range("2022-02-14", freq="s", periods=n, tz="UTC"),
        data=np.random.randn(n),
    )

    cs = [
        dr,
        dr.tz_localize(None),
        dr.tz_localize(None).tz_localize("Europe/Amsterdam"),
        dr.tz_convert("Europe/Brussels"),
        dr.tz_convert("Australia/Perth"),
        dr.tz_convert("Australia/Canberra"),
    ]

    for s in cs:
        t_start, t_stop = sorted(s.iloc[np.random.randint(0, n, 2)].index)
        hf_data_dict = construct_hf_data_dict(s.index, s.values)
        start_idx, end_idx = PlotlyAggregatorParser.get_start_end_indices(
            hf_data_dict, hf_data_dict["axis_type"], t_start, t_stop
        )
        assert (s.index[start_idx] - t_start) <= pd.Timedelta(seconds=1)
        assert (s.index[min(end_idx, n - 1)] - t_stop) <= pd.Timedelta(seconds=1)


def test_time_tz_slicing_different_timestamp():
    # construct a time indexed series with UTC timezone
    n = 60 * 60 * 24 * 3
    dr = pd.Series(
        index=pd.date_range("2022-02-14", freq="s", periods=n, tz="UTC"),
        data=np.random.randn(n),
    )

    # create multiple other time zones
    cs = [
        dr,
        dr.tz_localize(None).tz_localize("Europe/Amsterdam"),
        dr.tz_convert("Europe/Brussels"),
        dr.tz_convert("Australia/Perth"),
        dr.tz_convert("Australia/Canberra"),
    ]

    for i, s in enumerate(cs):
        t_start, t_stop = sorted(s.iloc[np.random.randint(0, n, 2)].index)
        t_start = t_start.tz_convert(cs[(i + 1) % len(cs)].index.tz)
        t_stop = t_stop.tz_convert(cs[(i + 1) % len(cs)].index.tz)

        # As each timezone in CS tz aware, using other timezones in `t_start` & `t_stop`
        # will raise an AssertionError
        with pytest.raises(AssertionError):
            hf_data_dict = construct_hf_data_dict(s.index, s.values)
            start_idx, end_idx = PlotlyAggregatorParser.get_start_end_indices(
                hf_data_dict, hf_data_dict["axis_type"], t_start, t_stop
            )


def test_different_tz_no_tz_series_slicing():
    n = 60 * 60 * 24 * 3
    dr = pd.Series(
        index=pd.date_range("2022-02-14", freq="s", periods=n, tz="UTC"),
        data=np.random.randn(n),
    )

    cs = [
        dr,
        dr.tz_localize(None),
        dr.tz_localize(None).tz_localize("Europe/Amsterdam"),
        dr.tz_convert("Europe/Brussels"),
        dr.tz_convert("Australia/Perth"),
        dr.tz_convert("Australia/Canberra"),
    ]

    for i, s in enumerate(cs):
        t_start, t_stop = sorted(
            s.tz_localize(None).iloc[np.random.randint(n / 2, n, 2)].index
        )
        # both timestamps now have the same tz
        t_start = t_start.tz_localize(cs[(i + 1) % len(cs)].index.tz)
        t_stop = t_stop.tz_localize(cs[(i + 1) % len(cs)].index.tz)

        # the s has no time-info -> assumption is made that s has the same time-zone
        # the timestamps
        hf_data_dict = construct_hf_data_dict(s.tz_localize(None).index, s.values)
        start_idx, end_idx = PlotlyAggregatorParser.get_start_end_indices(
            hf_data_dict, hf_data_dict["axis_type"], t_start, t_stop
        )
        assert (
            s.tz_localize(None).index[start_idx].tz_localize(t_start.tz) - t_start
        ) <= pd.Timedelta(seconds=1)
        assert (
            s.tz_localize(None).index[end_idx].tz_localize(t_stop.tz) - t_stop
        ) <= pd.Timedelta(seconds=1)


def test_multiple_tz_no_tz_series_slicing():
    n = 60 * 60 * 24 * 3
    dr = pd.Series(
        index=pd.date_range("2022-02-14", freq="s", periods=n, tz="UTC"),
        data=np.random.randn(n),
    )

    cs = [
        dr,
        dr.tz_localize(None),
        dr.tz_localize(None).tz_localize("Europe/Amsterdam"),
        dr.tz_convert("Europe/Brussels"),
        dr.tz_convert("Australia/Perth"),
        dr.tz_convert("Australia/Canberra"),
    ]

    for i, s in enumerate(cs):
        t_start, t_stop = sorted(
            s.tz_localize(None).iloc[np.random.randint(n / 2, n, 2)].index
        )
        # both timestamps now have the a different tz
        t_start = t_start.tz_localize(cs[(i + 1) % len(cs)].index.tz)
        t_stop = t_stop.tz_localize(cs[(i + 2) % len(cs)].index.tz)

        # Now the assumption cannot be made that s has the same time-zone as the
        # timestamps -> AssertionError will be raised.
        with pytest.raises(AssertionError):
            hf_data_dict = construct_hf_data_dict(s.tz_localize(None).index, s.values)
            PlotlyAggregatorParser.get_start_end_indices(
                hf_data_dict, hf_data_dict["axis_type"], t_start, t_stop
            )


def test_check_update_figure_dict():
    # mostly written to test the check_update_figure_dict with
    # "updated_trace_indices" = None
    fr = FigureResampler(go.Figure())
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)
    fr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)
    fr._check_update_figure_dict(fr.to_dict())


def test_stop_server_inline():
    # mostly written to test the check_update_figure_dict whether the inline + height
    # line option triggers
    fr = FigureResampler(go.Figure())
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)
    fr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)
    fr.update_layout(height=900)
    fr.stop_server()
    proc = multiprocessing.Process(target=fr.show_dash, kwargs=dict(mode="inline"))
    proc.start()

    time.sleep(3)
    fr.stop_server()
    proc.terminate()


def test_stop_server_inline_persistent():
    # mostly written to test the check_update_figure_dict whether the inline + height
    # line option triggers
    fr = FigureResampler(go.Figure())
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)
    fr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)
    fr.update_layout(height=900)
    fr.stop_server()
    proc = multiprocessing.Process(
        target=fr.show_dash, kwargs=dict(mode="inline_persistent")
    )
    proc.start()

    time.sleep(3)
    fr.stop_server()
    proc.terminate()


def test_showdash_not_hanging_when_port_in_use():
    if not_on_linux():
        pytest.skip("This test is currently only supported on linux")

    port = 8032

    start_fig = "from plotly_resampler import FigureResampler; "
    start_fig += (
        f"FigureResampler().show_dash(mode='external', port={port}, debug=False)"
    )

    # Start the first figure in another python interpreter
    p1 = subprocess.Popen(["python", "-c", start_fig])
    # Wait a little bit
    time.sleep(3)

    # Start the second figure in the current python interpreter
    with pytest.raises(SystemExit):
        # Start the second figure
        FigureResampler().show_dash(mode="external", port=port)
        # Wait a little bit
        time.sleep(3)

    # Stop the first figure
    p1.kill()


def test_manual_jupyterdashpersistentinline():
    # Manually call the JupyterDashPersistentInline its method
    # This requires some gimmicky stuff to mimmick the behaviour of a jupyter notebook.

    fr = FigureResampler(go.Figure())
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)
    fr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)

    # no need to start the app (we just need the FigureResampler object)

    import dash

    from plotly_resampler.figure_resampler.figure_resampler import (
        JupyterDashPersistentInlineOutput,
    )

    app = JupyterDashPersistentInlineOutput("manual_app")
    assert hasattr(app, "_uid")

    # Mimmick what happens in the .show_dash method
    # note: this is necessary because the figure gets accessed in the J
    # JupyterDashPersistentInline its _display_inline_output method (to create the img)
    app.layout = dash.html.Div(
        [
            dash.dcc.Graph(id="resample-figure", figure=fr),
            # no need to add traceupdater for this dummy app
        ]
    )

    # call the method (as it would normally be called)
    app._display_in_jupyter("", port="", mode="inline", width="100%", height=500)
    # call with a different mode (as it normally never would be called)
    app._display_in_jupyter("", port="", mode="external", width="100%", height=500)


def test_stop_server_external():
    fr = FigureResampler(go.Figure())
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)
    fr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)
    fr.update_layout(height=900)
    fr.stop_server()
    proc = multiprocessing.Process(target=fr.show_dash, kwargs=dict(mode="external"))
    proc.start()

    time.sleep(3)
    fr.stop_server()
    proc.terminate()


def test_hf_data_property():
    fr = FigureResampler(go.Figure(), default_n_shown_samples=2_000)
    n = 100_000
    x = np.arange(n)
    y = np.sin(x)
    assert len(fr.hf_data) == 0
    fr.add_trace(go.Scattergl(name="test"), hf_x=x, hf_y=y)
    assert len(fr.hf_data) == 1
    assert len(fr.hf_data[0]["x"]) == n
    fr.hf_data[0] = -2 * y


def test_fr_add_empty_trace():
    fig = FigureResampler(go.Figure())
    fig.add_trace(go.Scattergl(name="Test"), limit_to_view=True)

    assert len(fig.hf_data) == 1
    assert len(fig.hf_data[0]["x"]) == 0
    assert len(fig.hf_data[0]["y"]) == 0


def test_fr_from_trace_dict():
    y = np.array([1] * 10_000)
    base_fig = {
        "type": "scatter",
        "y": y,
    }

    fr_fig = FigureResampler(base_fig, default_n_shown_samples=1000)
    assert len(fr_fig.hf_data) == 1
    assert (fr_fig.hf_data[0]["y"] == y).all()
    assert len(fr_fig.data) == 1
    assert len(fr_fig.data[0]["x"]) == 1_000
    assert (fr_fig.data[0]["x"][0] >= 0) & (fr_fig.data[0]["x"][-1] < 10_000)
    assert (fr_fig.data[0]["y"] == [1] * 1_000).all()

    # assert that all the uuids of data and hf_data match
    # this is a proxy for assuring that the dynamic aggregation should work
    assert fr_fig.data[0].uid in fr_fig._hf_data


def test_fr_from_figure_dict():
    y = np.array([1] * 10_000)
    base_fig = go.Figure()
    base_fig.add_trace(go.Scatter(y=y))

    fr_fig = FigureResampler(base_fig.to_dict(), default_n_shown_samples=1000)
    assert len(fr_fig.hf_data) == 1
    assert (fr_fig.hf_data[0]["y"] == y).all()
    assert len(fr_fig.data) == 1
    assert len(fr_fig.data[0]["x"]) == 1_000
    assert (fr_fig.data[0]["x"][0] >= 0) & (fr_fig.data[0]["x"][-1] < 10_000)
    assert (fr_fig.data[0]["y"] == [1] * 1_000).all()

    # assert that all the uuids of data and hf_data match
    # this is a proxy for assuring that the dynamic aggregation should work
    assert fr_fig.data[0].uid in fr_fig._hf_data


def test_fr_empty_list():
    # and empty list -> so no concrete traces were added
    fr_fig = FigureResampler([], default_n_shown_samples=1000)
    assert len(fr_fig.hf_data) == 0
    assert len(fr_fig.data) == 0


def test_fr_empty_dict():
    # a dict is a concrete trace so 1 trace should be added
    fr_fig = FigureResampler({}, default_n_shown_samples=1000)
    assert len(fr_fig.hf_data) == 0
    assert len(fr_fig.data) == 1


def test_fr_wrong_keys(float_series):
    base_fig = [
        {"ydata": float_series.values + 2, "name": "sp2"},
    ]
    with pytest.raises(ValueError):
        FigureResampler(base_fig, default_n_shown_samples=1000)


def test_fr_from_list_dict(float_series):
    base_fig: List[dict] = [
        {"y": float_series.values + 2, "name": "sp2"},
        {"y": float_series.values, "name": "s"},
    ]

    fr_fig = FigureResampler(base_fig, default_n_shown_samples=1000)
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
    base_fig.append({"y": float_series[:1000], "name": "s_no_agg"})
    fr_fig = FigureResampler(base_fig, default_n_shown_samples=1000)
    assert len(fr_fig.hf_data) == 2
    assert len(fr_fig.data) == 3


def test_fr_list_dict_add_traces(float_series):
    fr_fig = FigureResampler(default_n_shown_samples=1000)

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
    fr_fig.add_traces({"y": float_series[:1000], "name": "s_no_agg"})
    assert len(fr_fig.hf_data) == 2
    assert len(fr_fig.data) == 3

    # add low-freq trace but set limit_to_view to True
    fr_fig.add_traces([{"y": float_series[:100], "name": "s_agg"}], limit_to_views=True)
    assert len(fr_fig.hf_data) == 3
    assert len(fr_fig.data) == 4

    # add a low-freq trace but adjust max_n_samples
    # note that we use a tuple as input here
    fr_fig.add_traces(({"y": float_series[:1000], "name": "s_agg"},), max_n_samples=999)
    assert len(fr_fig.hf_data) == 4
    assert len(fr_fig.data) == 5


def test_fr_list_dict_add_trace(float_series):
    fr_fig = FigureResampler(default_n_shown_samples=1000)

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


def test_fr_list_scatter_add_traces(float_series):
    fr_fig = FigureResampler(default_n_shown_samples=1000)

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
    fr_fig.add_traces(go.Scattergl(), limit_to_views=True)
    assert len(fr_fig.hf_data) == 3
    assert len(fr_fig.data) == 4

    # add a low-freq trace but adjust max_n_samples
    fr_fig.add_traces(
        go.Scatter({"y": float_series[:1000], "name": "s_agg"}), max_n_samples=999
    )
    assert len(fr_fig.hf_data) == 4
    assert len(fr_fig.data) == 5


def test_fr_add_scatter():
    # Checks whether the add_scatter method works as expected
    # .add_scatter calls `add_traces` under the hood
    f_orig = go.Figure().add_scatter(y=np.arange(2_000))
    f_pr = FigureResampler().add_scatter(y=np.arange(2_000))

    assert len(f_orig.data) == 1
    assert (len(f_pr.data) == 1) & (len(f_pr.hf_data) == 1)
    assert len(f_orig.data[0].y) == 2_000
    assert len(f_pr.data[0]["y"]) == 1_000
    assert np.all(f_orig.data[0].y == f_pr.hf_data[0]["y"])


def test_fr_copy_hf_data(float_series):
    fr_fig = FigureResampler(default_n_shown_samples=2000)
    traces: List[dict] = [
        go.Scattergl({"y": float_series.values + 2, "name": "sp2"}),
        go.Scatter({"y": float_series.values, "name": "s"}),
    ]
    fr_fig.add_traces(tuple(traces))

    hf_data_cp = FigureResampler()._copy_hf_data(fr_fig._hf_data)
    uid = list(hf_data_cp.keys())[0]

    hf_data_cp[uid]["x"] = np.arange(1000)
    hf_data_cp[uid]["y"] = float_series[:1000]

    assert len(fr_fig.hf_data[0]["x"]) == 10_000
    assert len(fr_fig.hf_data[0]["y"]) == 10_000
    assert len(fr_fig.hf_data[1]["x"]) == 10_000
    assert len(fr_fig.hf_data[1]["y"]) == 10_000


def test_fr_object_hf_data(float_series):
    float_series_o = float_series.astype(object)

    fig = FigureResampler()
    fig.add_trace({"name": "s0"}, hf_y=float_series_o)
    assert float_series_o.dtype == object
    assert len(fig.hf_data) == 1
    assert fig.hf_data[0]["y"].dtype == "float64"
    assert fig.data[0]["y"].dtype == "float64"


def test_fr_object_bool_data(bool_series):
    # First try with the original non-object bool series
    fig = FigureResampler()
    fig.add_trace({"name": "s0"}, hf_y=bool_series)
    assert len(fig.hf_data) == 1
    assert fig.hf_data[0]["y"].dtype == "bool"
    # plotly internally ocnverts this to object
    assert fig.data[0]["y"].dtype == "object"

    # Now try with the object bool series
    bool_series_o = bool_series.astype(object)

    fig = FigureResampler()
    fig.add_trace({"name": "s0"}, hf_y=bool_series_o)
    assert bool_series_o.dtype == object
    assert len(fig.hf_data) == 1
    assert fig.hf_data[0]["y"].dtype == "bool"
    # plotly internally converts this to object
    assert fig.data[0]["y"].dtype == "object"


def test_fr_object_binary_data():
    binary_series = np.array(
        [0, 1] * 20, dtype="int32"
    )  # as this is << max_n_samples -> limit_to_view

    # First try with the original non-object binary series
    fig = FigureResampler()
    fig.add_trace({"name": "s0"}, hf_y=binary_series, limit_to_view=True)
    assert len(fig.hf_data) == 1
    assert fig.hf_data[0]["y"].dtype == "int32"
    assert str(fig.data[0]["y"].dtype).startswith("int")
    assert np.all(fig.data[0]["y"] == binary_series)

    # Now try with the object binary series
    binary_series_o = binary_series.astype(object)

    fig = FigureResampler()
    fig.add_trace({"name": "s0"}, hf_y=binary_series_o, limit_to_view=True)
    assert binary_series_o.dtype == object
    assert len(fig.hf_data) == 1
    assert (fig.hf_data[0]["y"].dtype == "int32") or (
        fig.hf_data[0]["y"].dtype == "int64"
    )
    assert str(fig.data[0]["y"].dtype).startswith("int")
    assert np.all(fig.data[0]["y"] == binary_series)


def test_fr_update_layout_axes_range(driver):
    nb_datapoints = 2_000
    n_shown = 500  # < nb_datapoints

    # Checks whether the update_layout method works as expected
    f_orig = go.Figure().add_scatter(y=np.arange(nb_datapoints))
    f_pr = FigureResampler(default_n_shown_samples=n_shown).add_scatter(
        y=np.arange(nb_datapoints)
    )

    def check_data(fr: FigureResampler, min_v=0, max_v=nb_datapoints - 1):
        # closure for n_shown and nb_datapoints
        assert len(fr.data[0]["y"]) == min(n_shown, nb_datapoints)
        assert len(fr.data[0]["x"]) == min(n_shown, nb_datapoints)
        assert fr.data[0]["y"][0] == min_v
        assert fr.data[0]["y"][-1] == max_v
        assert fr.data[0]["x"][0] == min_v
        assert fr.data[0]["x"][-1] == max_v

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

    # Before showing the figure, the f_pr contains the full original data (downsampled to 500 samples)
    # Even after updating the axes ranges
    check_data(f_pr)

    if not_on_linux():
        # TODO: eventually we should run this test on Windows & MacOS too
        pytest.skip("This test is currently only run on Linux")

    f_pr.stop_server()
    proc = multiprocessing.Process(target=f_pr.show_dash, kwargs=dict(mode="external"))
    proc.start()
    try:
        time.sleep(1)
        driver.get("http://localhost:8050")
        time.sleep(3)
        # Get the data property from the front-end figure
        el = driver.find_element(by=By.ID, value="resample-figure")
        el = el.find_element(by=By.CLASS_NAME, value="js-plotly-plot")
        f_pr_data = el.get_property("data")
        f_pr_layout = el.get_property("layout")

        # After showing the figure, the f_pr contains the data of the selected xrange (downsampled to 500 samples)
        assert len(f_pr_data[0]["y"]) == 500
        assert len(f_pr_data[0]["x"]) == 500
        assert f_pr_data[0]["y"][0] >= 100 and f_pr_data[0]["y"][-1] <= 1000
        assert f_pr_data[0]["x"][0] >= 100 and f_pr_data[0]["x"][-1] <= 1000
        # Check the front-end layout
        assert list(f_pr_layout["xaxis"]["range"]) == [100, 1000]
        assert list(f_pr_layout["yaxis"]["range"]) == [100, 1000]
    except Exception as e:
        raise e
    finally:
        proc.terminate()
        f_pr.stop_server()


def test_fr_update_layout_axes_range_no_update(driver):
    nb_datapoints = 2_000
    n_shown = 20_000  # > nb. datapoints

    # Checks whether the update_layout method works as expected
    f_orig = go.Figure().add_scatter(y=np.arange(nb_datapoints))
    f_pr = FigureResampler(default_n_shown_samples=n_shown).add_scatter(
        y=np.arange(nb_datapoints)
    )

    def check_data(fr: FigureResampler, min_v=0, max_v=nb_datapoints - 1):
        # closure for n_shown and nb_datapoints
        assert len(fr.data[0]["y"]) == min(n_shown, nb_datapoints)
        assert len(fr.data[0]["x"]) == min(n_shown, nb_datapoints)
        assert fr.data[0]["y"][0] == min_v
        assert fr.data[0]["y"][-1] == max_v
        assert fr.data[0]["x"][0] == min_v
        assert fr.data[0]["x"][-1] == max_v

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

    # Before showing the figure, the f_pr contains the full original data (not downsampled)
    # Even after updating the axes ranges
    check_data(f_pr)

    if not_on_linux():
        # TODO: eventually we should run this test on Windows & MacOS too
        pytest.skip("This test is currently only run on Linux")

    f_pr.stop_server()
    proc = multiprocessing.Process(target=f_pr.show_dash, kwargs=dict(mode="external"))
    proc.start()
    try:
        time.sleep(1)
        driver.get("http://localhost:8050")
        time.sleep(3)
        # Get the data & layout property from the front-end figure
        el = driver.find_element(by=By.ID, value="resample-figure")
        el = el.find_element(by=By.CLASS_NAME, value="js-plotly-plot")
        f_pr_data = el.get_property("data")
        f_pr_layout = el.get_property("layout")

        # After showing the figure, the f_pr contains the original data (not downsampled), but shown xrange is [100, 1000]
        assert len(f_pr_data[0]["y"]) == 2_000
        assert len(f_pr_data[0]["x"]) == 2_000
        assert f_pr.data[0]["y"][0] == 0
        assert f_pr.data[0]["y"][-1] == 1999
        assert f_pr.data[0]["x"][0] == 0
        assert f_pr.data[0]["x"][-1] == 1999
        # Check the front-end layout
        assert list(f_pr_layout["xaxis"]["range"]) == [100, 1000]
        assert list(f_pr_layout["yaxis"]["range"]) == [100, 1000]
    except Exception as e:
        raise e
    finally:
        proc.terminate()
        f_pr.stop_server()


def test_fr_copy_grid():
    # Checks whether _grid_ref and _grid_str are correctly maintained

    f = make_subplots(rows=2, cols=1)
    f.add_scatter(y=np.arange(2_000), row=1, col=1)
    f.add_scatter(y=np.arange(2_000), row=2, col=1)

    ## go.Figure
    assert isinstance(f, go.Figure)
    assert f._grid_ref is not None
    assert f._grid_str is not None
    fr = FigureResampler(f)
    assert fr._grid_ref is not None
    assert fr._grid_ref == f._grid_ref
    assert fr._grid_str is not None
    assert fr._grid_str == f._grid_str

    ## go.FigureWidget
    fw = go.FigureWidget(f)
    assert fw._grid_ref is not None
    assert fw._grid_str is not None
    assert isinstance(fw, go.FigureWidget)
    fr = FigureResampler(fw)
    assert fr._grid_ref is not None
    assert fr._grid_ref == fw._grid_ref
    assert fr._grid_str is not None
    assert fr._grid_str == fw._grid_str

    ## FigureResampler
    fr_ = FigureResampler(f)
    assert fr_._grid_ref is not None
    assert fr_._grid_str is not None
    assert isinstance(fr_, FigureResampler)
    fr = FigureResampler(fr_)
    assert fr._grid_ref is not None
    assert fr._grid_ref == fr_._grid_ref
    assert fr._grid_str is not None
    assert fr._grid_str == fr_._grid_str

    ## FigureWidgetResampler
    from plotly_resampler import FigureWidgetResampler

    fwr = FigureWidgetResampler(f)
    assert fwr._grid_ref is not None
    assert fwr._grid_str is not None
    assert isinstance(fwr, FigureWidgetResampler)
    fr = FigureResampler(fwr)
    assert fr._grid_ref is not None
    assert fr._grid_ref == fwr._grid_ref
    assert fr._grid_str is not None
    assert fr._grid_str == fwr._grid_str

    ## dict (with no _grid_ref & no _grid_str)
    f_dict = f.to_dict()
    assert isinstance(f_dict, dict)
    assert f_dict.get("_grid_ref") is None
    assert f_dict.get("_grid_str") is None
    fr = FigureResampler(f_dict)
    assert fr._grid_ref is f_dict.get("_grid_ref")  # both are None
    assert fr._grid_str is f_dict.get("_grid_str")  # both are None

    ## dict (with _grid_ref & _grid_str)
    f_dict = f.to_dict()
    f_dict["_grid_ref"] = f._grid_ref
    f_dict["_grid_str"] = f._grid_str
    assert isinstance(f_dict, dict)
    assert f_dict.get("_grid_ref") is not None
    assert f_dict.get("_grid_str") is not None
    fr = FigureResampler(f_dict)
    assert fr._grid_ref is not None
    assert fr._grid_ref == f_dict.get("_grid_ref")
    assert fr._grid_str is not None
    assert fr._grid_str == f_dict.get("_grid_str")


# Testing HF marker_size and color arguments
def test_hf_marker_size_hf_args():
    # create dummy data
    n = 100_000
    y = np.sin(np.arange(n) / 2_000) + np.random.randn(n) / 10

    # construct the figure via hf kwargs
    fr = FigureResampler()
    fr.add_trace(
        go.Scattergl(mode="markers"),
        hf_y=y,
        hf_marker_size=(3 + 20 * np.abs(y)).astype(int),
        hf_marker_color=np.abs(y) / np.max(np.abs(y)),
    )

    # Perform asserts on the hf_data part of the figure
    hf_trace = fr.hf_data[0]
    assert "marker_size" in hf_trace
    assert "marker_color" in hf_trace

    assert len(hf_trace["marker_size"] == len(y))
    assert len(hf_trace["marker_color"] == len(y))

    # perform some asserts on the to-be constructed update data
    update_trace = fr.construct_update_data(
        {"xaxis.autorange": True, "xaxis.showspikes": True}
    )[1]

    assert all(k in update_trace for k in ["x", "y", "name", "marker", "index"])

    # check whether the marker size and marker color are available
    assert all(k in update_trace["marker"] for k in ["size", "color"])
    assert len(update_trace["marker"]["size"]) == len(update_trace["x"])
    assert np.allclose(
        update_trace["marker"]["color"],
        (np.abs(update_trace["y"]) / np.max(np.abs(y))),
        rtol=1e-3,
    )


def test_hf_marker_size_plotly_args():
    # create dummy data
    n = 100_000
    y = np.sin(np.arange(n) / 2_000) + np.random.randn(n) / 10

    # construct the figure via hf kwargs
    fr = FigureResampler()
    fr.add_trace(
        go.Scattergl(
            mode="markers",
            marker_size=(3 + 20 * np.abs(y)).astype(int),
            marker_color=np.abs(y) / np.max(np.abs(y)),
        ),
        hf_y=y,
    )

    # Perform asserts on the hf_data part of the figure
    hf_trace = fr.hf_data[0]
    assert "marker_size" in hf_trace
    assert "marker_color" in hf_trace

    assert len(hf_trace["marker_size"] == len(y))
    assert len(hf_trace["marker_color"] == len(y))

    # perform some asserts on the to-be constructed update data
    update_trace = fr.construct_update_data(
        {"xaxis.autorange": True, "xaxis.showspikes": True}
    )[1]

    assert all(k in update_trace for k in ["x", "y", "name", "marker", "index"])

    # check whether the marker size and marker color are available
    assert all(k in update_trace["marker"] for k in ["size", "color"])
    assert len(update_trace["marker"]["size"]) == len(update_trace["x"])
    assert np.allclose(
        update_trace["marker"]["color"],
        (np.abs(update_trace["y"]) / np.max(np.abs(y))),
        rtol=1e-3,
    )
