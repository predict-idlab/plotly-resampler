"""Code which tests the FigureWidgetResampler functionalities"""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"


from copy import copy
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots
from plotly_resampler import EfficientLTTB, EveryNthPoint, FigureWidgetResampler


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
            "default_downsampler": EfficientLTTB(interleave_gaps=True),
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
            downsampler=EveryNthPoint(interleave_gaps=True),
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
    float_series.iloc[np.random.choice(len(float_series), 100)] = np.nan
    fig.add_trace(
        go.Scatter(x=float_series.index, y=float_series, name="float_series"),
        row=1,
        col=1,
        hf_text="text",
        hf_hovertext="hovertext",
    )

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

    fr_fig = FigureWidgetResampler(
        make_subplots(rows=len(cs), cols=1, shared_xaxes=True),
        default_n_shown_samples=500,
        convert_existing_traces=False,
        verbose=True,
    )
    fr_fig.update_layout(height=min(300, 250 * len(cs)))

    for i, date_range in enumerate(cs, 1):
        fr_fig.add_trace(
            go.Scattergl(name=date_range.dtype.name.split(", ")[-1]),
            hf_x=date_range,
            hf_y=dr_v,
            row=i,
            col=1,
        )


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

    fig = FigureWidgetResampler(go.Figure())

    for s in cs:
        t_start, t_stop = sorted(s.iloc[np.random.randint(0, n, 2)].index)
        out = fig._slice_time(s, t_start, t_stop)
        assert (out.index[0] - t_start) <= pd.Timedelta(seconds=1)
        assert (out.index[-1] - t_stop) <= pd.Timedelta(seconds=1)


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

    fig = FigureWidgetResampler(go.Figure())
    for i, s in enumerate(cs):
        t_start, t_stop = sorted(s.iloc[np.random.randint(0, n, 2)].index)
        t_start = t_start.tz_convert(cs[(i + 1) % len(cs)].index.tz)
        t_stop = t_stop.tz_convert(cs[(i + 1) % len(cs)].index.tz)

        # As each timezone in CS tz aware, using other timezones in `t_start` & `t_stop`
        # will raise an AssertionError
        with pytest.raises(AssertionError):
            fig._slice_time(s, t_start, t_stop)


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

    fig = FigureWidgetResampler(go.Figure())

    for i, s in enumerate(cs):
        t_start, t_stop = sorted(
            s.tz_localize(None).iloc[np.random.randint(n / 2, n, 2)].index
        )
        # both timestamps now have the same tz
        t_start = t_start.tz_localize(cs[(i + 1) % len(cs)].index.tz)
        t_stop = t_stop.tz_localize(cs[(i + 1) % len(cs)].index.tz)

        # the s has no time-info -> assumption is made that s has the same time-zone
        # the timestamps
        out = fig._slice_time(s.tz_localize(None), t_start, t_stop)
        assert (out.index[0].tz_localize(t_start.tz) - t_start) <= pd.Timedelta(
            seconds=1
        )
        assert (out.index[-1].tz_localize(t_stop.tz) - t_stop) <= pd.Timedelta(
            seconds=1
        )


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

    fig = FigureWidgetResampler(go.Figure())

    for i, s in enumerate(cs):
        t_start, t_stop = sorted(
            s.tz_localize(None).iloc[np.random.randint(n / 2, n, 2)].index
        )
        # both timestamps now have the a different tz
        t_start = t_start.tz_localize(cs[(i + 1) % len(cs)].index.tz)
        t_stop = t_stop.tz_localize(cs[(i + 2) % len(cs)].index.tz)

        # Now the assumpton cannot be made that s ahd the same time-zone as the
        # timestamps -> Assertionerror will be raised.
        with pytest.raises(AssertionError):
            fig._slice_time(s.tz_localize(None), t_start, t_stop)


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
    assert not ["showspikes-update", 2] in fw_fig._relayout_hist
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

    # Perform an autorange udpate -> assert that the range i
    fw_fig._relayout_hist.clear()
    fw_fig.layout.update({"xaxis2": {"autorange": True}, "yaxis2": {"autorange": True}})
    assert len(fw_fig._relayout_hist) == 0

    fw_fig.layout.update({"yaxis2": {"range": [0, 2]}})
    assert len(fw_fig._relayout_hist) == 0

    # perform an reset axis
    fw_fig._relayout_hist.clear()
    l = fw_fig.layout.update(
        {
            "xaxis": {"autorange": True, "showspikes": False},
            "xaxis2": {"autorange": True, "showspikes": False},
        },
        overwrite=True,  # by setting this to true -> the update call will not takte clear
    )
    fw_fig._update_spike_ranges(l, False, False)

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


def test_fwr_updata_trace_data_zoom():
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
    l = fig.layout.update(
        {"xaxis": {"range": [0, 100_000]}},
        overwrite=True,
    )
    fig._update_x_ranges(
        l, (0, 100_000), (fig.hf_data[1]["x"][0], fig.hf_data[1]["x"][-1])
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
    l = fig.layout.update(
        {"xaxis2": {"range": [200_000, 500_000]}},
        overwrite=True,
    )
    fig._update_x_ranges(l, (0, 100_000), (200_000, 500_000))

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


def test_fwr_hovertext_adjust_unequal_length():
    k = 10_000
    fig = FigureWidgetResampler(default_n_shown_samples=1000, verbose=True)
    fig.add_trace(go.Scattergl(name="A", line_color="red"), limit_to_view=True)

    fig._relayout_hist.clear()

    with pytest.raises(ValueError):
        A = np.random.randn(k)
        fig.hf_data[0]["x"] = pd.Series(np.arange(k - 500))
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

    # asser that hf x and y its values are used and not its index
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

    # asser that hf x and y its values are used and not its index
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
        default_n_shown_samples=1000, verbose=True, default_downsampler=EfficientLTTB()
    )

    for i in range(3):
        s = pd.Series(
            index=pd.date_range(
                datetime.now(), freq=f"{np.random.randint(5,100_000)}ns", periods=n
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
        default_n_shown_samples=1000, verbose=True, default_downsampler=EfficientLTTB()
    )

    for i in range(3):
        s = pd.Series(
            index=pd.date_range(
                datetime.now(), freq=f"{np.random.randint(5,100_000)}us", periods=n
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
        default_n_shown_samples=1000, verbose=True, default_downsampler=EfficientLTTB()
    )

    for i in range(3):
        s = pd.Series(
            index=pd.date_range(
                datetime.now(), freq=f"{np.random.randint(5,10_000)}ms", periods=n
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
    n = 100_000
    fig = FigureWidgetResampler(
        default_n_shown_samples=1000, verbose=True, default_downsampler=EfficientLTTB()
    )

    for i in range(3):
        s = pd.Series(
            index=pd.date_range(
                datetime.now(),
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


def test_fwr_from_dict():
    y = np.array([1] * 10_000)
    base_fig = {
        "type": "scatter",
        "y": y,
    }

    fr_fig = FigureWidgetResampler(base_fig, default_n_shown_samples=1000)
    assert len(fr_fig.hf_data) == 1
    assert (fr_fig.hf_data[0]["y"] == y).all()
    assert len(fr_fig.data) == 1
    assert len(fr_fig.data[0]["x"]) == 1_000
    assert (fr_fig.data[0]["x"][0] >= 0) & (fr_fig.data[0]["x"][-1] < 10_000)
    assert (fr_fig.data[0]["y"] == [1] * 1_000).all()
