"""Fixtures and helper functions for testing"""


import pytest
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import selenium
from plotly_resampler import FigureResampler
from plotly_resampler.downsamplers import LTTB, EveryNthPoint


# hyperparameters
_nb_samples = 10_000
data_dir = 'examples/data/'
headless = True


@pytest.fixture
def driver():
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    if headless:
        options = Options()
        options.headless = True
        return webdriver.Firefox(options=options)
    else:
        web_driver = webdriver.Firefox()
        return web_driver


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
    df_gusb = pd.read_parquet(f"{data_dir}df_gusb.parquet", engine='fastparquet')
    df_data_pc = pd.read_parquet(f"{data_dir}df_pc_test.parquet", engine='fastparquet')

    n = 110_000#_000
    np_series = np.array(
        (3 + np.sin(np.arange(n) / 200_000) + np.random.randn(n) / 10) * np.arange(n) / 100_000,
        dtype=np.float32,
    )
    x = np.arange(len(np_series))
    
    fig = FigureResampler(
    make_subplots(
        rows=2, cols=2, 
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("GUSB swimming pool", "Generated sine", "Power consumption"),
        vertical_spacing=0.12,
    ), default_n_shown_samples=1_000, verbose=False)


    # ------------ swimming pool data -----------
    df_gusb_pool = df_gusb[df_gusb.zone == "zwembad"]
    df_gusb_pool = df_gusb_pool[df_gusb_pool["aantal aanwezigen"] < 3_000].last('4D')
    fig.add_trace(
        go.Scattergl(
            x=df_gusb_pool.index, 
            y=df_gusb_pool["aantal aanwezigen"].astype('uint16'), 
            mode='markers',
            marker_size=5,
            name="occupancy",
            showlegend=True
        ),
        hf_hovertext='mean last hour: ' + df_gusb_pool["aantal aanwezigen"].rolling('1h').mean().astype(int).astype(str),
        downsampler=EveryNthPoint(interleave_gaps=False),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text='Occupancy', row=1, col=1)


    # ----------------- generated sine -----------
    fig.add_trace(
        go.Scattergl(name="sin", line_color='#26b2e0'),
        hf_x=x,
        hf_y=np_series,
        row=1,
        col=2,
    )

    # ------------- Power consumption data -------------
    df_data_pc = df_data_pc.last('190D')
    for i, c in enumerate(df_data_pc.columns):
        fig.add_trace(
            go.Scattergl(name=f"room {i+1}",),
            hf_x=df_data_pc.index, hf_y=df_data_pc[c],
            row=2, col=1,
            downsampler=LTTB(interleave_gaps=True)
        )

    fig.update_layout(height=600)
    fig.update_yaxes(title_text='Watt/hour', row=2, col=1)
    fig.update_layout(
        title='<b>Plotly-Resampler demo</b>', title_x=0.5, legend_traceorder='normal',
    )
    return fig