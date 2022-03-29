"""Fixtures and helper functions for testing"""


from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from plotly_resampler import FigureResampler, LTTB, EveryNthPoint

# hyperparameters
_nb_samples = 10_000
data_dir = "examples/data/"
headless = True
TESTING_LOCAL = False  # SET THIS TO TRUE IF YOU ARE TESTING LOCALLY


@pytest.fixture
def driver():
    from seleniumwire import webdriver
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.utils import ChromeType

    options = Options()
    if not TESTING_LOCAL:
        if headless:
            options.add_argument("--headless")
        # options.add_argument("--no=sandbox")
        driver = webdriver.Chrome(
            ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install(),
            options=options,
        )
    else:
        options.add_argument("--remote-debugging-port=9222")
        driver = webdriver.Chrome(
            options=options,
            # executable_path="/home/jonas/git/gIDLaB/plotly-dynamic-resampling/chromedriver",
        )
        # driver = webdriver.Firefox(executable_path='/home/jonas/git/gIDLaB/plotly-dynamic-resampling/geckodriver')
    return driver


@pytest.fixture
def float_series() -> pd.Series:
    x = np.arange(_nb_samples).astype(np.uint32)
    y = np.sin(x / 300).astype(np.float32) + np.random.randn(_nb_samples) / 5
    return pd.Series(index=x, data=y)


@pytest.fixture
def cat_series() -> pd.Series:
    cats_list = ["a", "b", "b", "b", "c", "c", "a", "d", "a"]
    return pd.Series(cats_list * (_nb_samples // len(cats_list)), dtype="category")


@pytest.fixture
def bool_series() -> pd.Series:
    bool_list = [True, False, True, True, True, True]
    return pd.Series(bool_list * (_nb_samples // len(bool_list)), dtype="bool")


@pytest.fixture
def example_figure() -> FigureResampler:
    df_gusb = pd.read_parquet(f"{data_dir}df_gusb.parquet")
    df_data_pc = pd.read_parquet(f"{data_dir}df_pc_test.parquet")

    n = 110_000  # _000
    np_series = np.array(
        (3 + np.sin(np.arange(n) / 200_000) + np.random.randn(n) / 10)
        * np.arange(n)
        / 100_000,
        dtype=np.float32,
    )
    x = np.arange(len(np_series))

    fig = FigureResampler(
        make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
            subplot_titles=(
                "GUSB swimming pool",
                "Generated sine",
                "Power consumption",
            ),
            vertical_spacing=0.12,
        ),
        default_n_shown_samples=1_000,
        verbose=False,
    )

    # ------------ swimming pool data -----------
    df_gusb_pool = df_gusb["zwembad"].last("4D").dropna()
    fig.add_trace(
        go.Scattergl(
            x=df_gusb_pool.index,
            y=df_gusb_pool.astype("uint16"),
            mode="markers",
            marker_size=5,
            name="occupancy",
            showlegend=True,
        ),
        hf_hovertext="mean last hour: "
        + df_gusb_pool.rolling("1h").mean().astype(int).astype(str),
        downsampler=EveryNthPoint(interleave_gaps=False),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Occupancy", row=1, col=1)

    # ----------------- generated sine -----------
    fig.add_trace(
        go.Scattergl(name="sin", line_color="#26b2e0"),
        hf_x=x,
        hf_y=np_series,
        row=1,
        col=2,
    )

    # ------------- Power consumption data -------------
    df_data_pc = df_data_pc.last("190D")
    for i, c in enumerate(df_data_pc.columns):
        fig.add_trace(
            go.Scattergl(
                name=f"room {i+1}",
            ),
            hf_x=df_data_pc.index,
            hf_y=df_data_pc[c],
            row=2,
            col=1,
            downsampler=LTTB(interleave_gaps=True),
        )

    fig.update_layout(height=600)
    fig.update_yaxes(title_text="Watt/hour", row=2, col=1)
    fig.update_layout(
        title="<b>Plotly-Resampler demo</b>",
        title_x=0.5,
        legend_traceorder="normal",
    )
    return fig


@pytest.fixture
def example_figure_fig() -> go.Figure:
    df_gusb = pd.read_parquet(f"{data_dir}df_gusb.parquet")
    df_data_pc = pd.read_parquet(f"{data_dir}df_pc_test.parquet")

    n = 110_000  # _000
    np_series = np.array(
        (3 + np.sin(np.arange(n) / 200_000) + np.random.randn(n) / 10)
        * np.arange(n)
        / 100_000,
        dtype=np.float32,
    )
    x = np.arange(len(np_series))

    # construct a normal figure object instead of a figureResample object
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("GUSB swimming pool", "Generated sine", "Power consumption"),
        vertical_spacing=0.12,
    )

    # ------------ swimming pool data -----------
    df_gusb_pool = df_gusb["zwembad"].last("4D").dropna()
    fig.add_trace(
        go.Scattergl(
            x=df_gusb_pool.index,
            y=df_gusb_pool,  # .astype("uint16"),
            mode="markers",
            marker_size=5,
            name="occupancy",
            showlegend=True,
            hovertext="mean last hour: "
            + df_gusb_pool.rolling("1h").mean().astype(int).astype(str),
        ),
        # downsampler=EveryNthPoint(interleave_gaps=False),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Occupancy", row=1, col=1)

    # ----------------- generated sine -----------
    fig.add_trace(
        go.Scattergl(name="sin", line_color="#26b2e0", x=x, y=np_series),
        row=1,
        col=2,
    )

    # ------------- Power consumption data -------------
    df_data_pc = df_data_pc.last("190D")
    for i, c in enumerate(df_data_pc.columns):
        fig.add_trace(
            go.Scattergl(
                name=f"room {i+1}",
                x=df_data_pc.index,
                y=df_data_pc[c],
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=600)
    fig.update_yaxes(title_text="Watt/hour", row=2, col=1)
    fig.update_layout(
        title="<b>Plotly-Resampler demo - fig base</b>",
        title_x=0.5,
        legend_traceorder="normal",
    )
    return fig


@pytest.fixture
def gsr_figure() -> FigureResampler:
    def groupby_consecutive(
        df: Union[pd.Series, pd.DataFrame], col_name: str = None
    ) -> pd.DataFrame:
        """Merges consecutive `column_name` values in a single dataframe.

        This is especially useful if you want to represent sparse data in a more
        compact format.

        Parameters
        ----------
        df : Union[pd.Series, pd.DataFrame]
            Must be time-indexed!
        col_name : str, optional
            If a dataFrame is passed, you will need to specify the `col_name` on which
            the consecutive-grouping will need to take plase.

        Returns
        -------
        pd.DataFrame
            A new `DataFrame` view, with columns:
            [`start`, `end`, `n_consecutive`, `col_name`], representing the
            start- and endtime of the consecutive range, the number of consecutive samples,
            and the col_name's consecutive values.
        """
        if type(df) == pd.Series:
            col_name = df.name
            df = df.to_frame()

        assert col_name in df.columns

        df_cum = (
            (df[col_name].diff(1) != 0)
            .astype("int")
            .cumsum()
            .rename("value_grp")
            .to_frame()
        )
        df_cum["sequence_idx"] = df.index
        df_cum[col_name] = df[col_name]

        df_grouped = pd.DataFrame(
            {
                "start": df_cum.groupby("value_grp")["sequence_idx"].first(),
                "end": df_cum.groupby("value_grp")["sequence_idx"].last(),
                "n_consecutive": df_cum.groupby("value_grp").size(),
                col_name: df_cum.groupby("value_grp")[col_name].first(),
            }
        ).reset_index(drop=True)
        df_grouped["next_start"] = df_grouped.start.shift(-1).fillna(df_grouped["end"])
        return df_grouped

    df_gsr = pd.read_parquet(f"{data_dir}processed_gsr.parquet")

    fig = FigureResampler(
        make_subplots(
            rows=2,
            cols=1,
            specs=[[{"secondary_y": True}], [{}]],
            shared_xaxes=True,
        ),
        default_n_shown_samples=1_000,
        resampled_trace_prefix_suffix=(
            '<b style="color:blue">[R]</b> ',
            ' <b style="color:red">[R]</b>',
        ),
        verbose=False,
        show_mean_aggregation_size=True
    )
    fig.update_layout(height=700)

    for c in ["EDA", "EDA_lf_cleaned", "EDA_lf_cleaned_tonic"]:
        fig.add_trace(
            go.Scattergl(name=c), hf_x=df_gsr.index, hf_y=df_gsr[c], row=1, col=1
        )

    df_peaks = df_gsr[df_gsr["SCR_Peaks_neurokit_reduced_acc"] == 1]
    fig.add_trace(
        trace=go.Scattergl(
            x=df_peaks.index,
            y=df_peaks["EDA_lf_cleaned"],
            visible="legendonly",
            mode="markers",
            marker_symbol="cross",
            marker_size=15,
            marker_color="red",
            name="SCR peaks",
        ),
        limit_to_view=True,
    )

    df_grouped = groupby_consecutive(df_gsr["EDA_SQI"])
    df_grouped["EDA_SQI"] = df_grouped["EDA_SQI"].map(bool)
    df_grouped["good_sqi"] = df_grouped["EDA_SQI"].map(int)
    df_grouped["bad_sqi"] = (~df_grouped["EDA_SQI"]).map(int)
    for sqi_col, col_or in [
        ("good_sqi", "#2ca02c"),
        ("bad_sqi", "#d62728"),
    ]:
        fig.add_trace(
            go.Scattergl(
                x=df_grouped["start"],
                y=df_grouped[sqi_col],
                mode="lines",
                line_width=0,
                fill="tozeroy",
                fillcolor=col_or,
                opacity=0.1 if "good" in sqi_col else 0.2,
                line_shape="hv",
                name=sqi_col,
                showlegend=False,
            ),
            max_n_samples=len(df_grouped) + 1,
            downsampler=EveryNthPoint(interleave_gaps=False),
            limit_to_view=True,
            secondary_y=True,
        )

    fig.add_trace(
        go.Scattergl(name="EDA_Phasic", visible="legendonly"),
        hf_x=df_gsr.index,
        hf_y=df_gsr["EDA_Phasic"],
        row=2,
        col=1,
    )

    return fig


@pytest.fixture
def multiple_tz_figure() -> FigureResampler:
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

    fr_fig = FigureResampler(
        make_subplots(rows=len(cs), cols=1, shared_xaxes=True),
        default_n_shown_samples=500,
        convert_existing_traces=False,
        verbose=True,
        show_mean_aggregation_size=False,
    )
    fr_fig.update_layout(height=min(700, 250 * len(cs)))

    for i, date_range in enumerate(cs, 1):
        fr_fig.add_trace(
            go.Scattergl(name=date_range.dtype.name.split(", ")[-1]),
            hf_x=date_range,
            hf_y=dr_v,
            row=i,
            col=1,
        )
    return fr_fig


@pytest.fixture
def cat_series_box_hist_figure() -> FigureResampler:
    # Create a categorical series, with mostly a's, but a few sparse b's and c's
    cats_list = np.array(list("aaaaaaaaaa" * 1000))
    cats_list[np.random.choice(len(cats_list), 100, replace=False)] = "b"
    cats_list[np.random.choice(len(cats_list), 50, replace=False)] = "c"
    cat_series = pd.Series(cats_list, dtype="category")

    x = np.arange(_nb_samples).astype(np.uint32)
    y = np.sin(x / 300).astype(np.float32) + np.random.randn(_nb_samples) / 5
    float_series = pd.Series(index=x, data=y)

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
        hf_hovertext="text",
    )

    fig.add_trace(go.Box(x=float_series.values, name="float_series"), row=1, col=2)
    fig.add_trace(
        go.Box(x=float_series.values ** 2, name="float_series**2"), row=1, col=2
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
    return fig


@pytest.fixture
def shared_hover_figure() -> FigureResampler:
    fig = FigureResampler(make_subplots(rows=3, cols=1, shared_xaxes=True), verbose=1)

    x = np.array(range(100_000))
    y = np.sin(x / 120) + np.random.random(len(x)) / 10

    for i in range(1, 4):
        fig.add_trace(go.Scatter(x=[], y=[]), hf_x=x, hf_y=y, row=i, col=1)

    fig.update_layout(template="plotly_white", height=900)
    fig.update_traces(xaxis="x3")
    fig.update_xaxes(spikemode="across", showspikes=True)

    return fig


@pytest.fixture
def multi_trace_go_figure() -> FigureResampler:
    fig = FigureResampler(go.Figure())

    n = 500_000  # nb points per sensor
    x = np.arange(n)
    y = np.sin(x / 20) + np.random.random(len(x)) / 10

    for i in range(30):  # 30 sensors
        fig.add_trace(
            go.Scattergl(name=str(i)),
            hf_x=x,
            hf_y=y + i,
        )
    fig.update_layout(height=800)
    return fig
