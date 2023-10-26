"""Code which tests the overview functionality."""

__author__ = "Jonas Van Der Donckt"

import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots
from pytest_lazyfixture import lazy_fixture as lf

from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import (
    EveryNthPoint,
    MedDiffGapHandler,
    MinMaxLTTB,
    NoGapHandler,
)


@pytest.mark.parametrize("figure_class", [go.Figure, make_subplots])
@pytest.mark.parametrize(
    "series", [lf("float_series"), lf("cat_series"), lf("bool_series")]
)
def test_overview_figure_type(figure_class, series):
    """Test the overview functionality (i.e., whether the overview figure can be
    constructed)"""
    # Create a figure with a scatter plot
    fig = FigureResampler(figure_class(), create_overview=True)
    fig.add_trace(go.Scatter(x=series.index, y=series))
    fig.add_trace({}, hf_x=series.index, hf_y=series)

    fig._create_overview_figure()
    # fig.write_image(f"test_{figure_class.__name__}_{series.name}.png")


@pytest.mark.parametrize("n_cols", [1, 2, 3])
def test_valid_row_indices_subplots(n_cols):
    fig = FigureResampler(
        make_subplots(rows=3, cols=n_cols, shared_xaxes="columns"),
        create_overview=True,
        overview_row_idxs=None,
    )
    fig._create_overview_figure()
    assert fig._overview_row_idxs == [0] * n_cols

    # this should not crash
    fig = FigureResampler(
        make_subplots(rows=3, cols=n_cols, shared_xaxes="columns"),
        create_overview=True,
        overview_row_idxs=[np.random.randint(0, 2) for _ in range(n_cols)],
    )
    fig._create_overview_figure()


@pytest.mark.parametrize("n_cols", [1, 2, 3])
def test_invalid_row_indices_subplots(n_cols):
    with pytest.raises(AssertionError):
        FigureResampler(
            make_subplots(rows=3, cols=n_cols, shared_xaxes="columns"),
            create_overview=True,
            overview_row_idxs=[3 for _ in range(n_cols)],
        )

    with pytest.raises(AssertionError):
        FigureResampler(
            make_subplots(rows=3, cols=n_cols, shared_xaxes="columns"),
            create_overview=True,
            overview_row_idxs=[0 for _ in range(n_cols - 1)],
        )


@pytest.mark.parametrize("overview_kwargs", [{"height": 80}])
@pytest.mark.parametrize("series", [lf("float_series")])
def test_overview_kwargs(overview_kwargs, series):
    fig = FigureResampler(
        go.Figure(),
        create_overview=True,
        overview_kwargs=overview_kwargs,
    )
    fig.add_trace(go.Scatter(x=series.index, y=series))

    overview_fig = fig._create_overview_figure()
    for key, value in overview_kwargs.items():
        assert overview_fig.layout[key] == value


@pytest.mark.parametrize("figure_class", [go.Figure, make_subplots])
@pytest.mark.parametrize(
    "series", [lf("float_series"), lf("cat_series"), lf("bool_series")]
)
@pytest.mark.parametrize("default_n_samples", [500, 1000, 1500])
def test_coarse_figure_aggregation(figure_class, series, default_n_samples):
    """Test whether the coarse figure aggregation works as expected"""
    # Create a figure with a scatter plot
    fig = FigureResampler(
        figure_class(), create_overview=True, default_n_shown_samples=default_n_samples
    )
    fig.add_trace(go.Scatter(x=series.index, y=series))
    fig.add_trace({}, hf_x=series.index, hf_y=series)

    overview_fig = fig._create_overview_figure()
    for trace in overview_fig.data:
        assert len(trace.y) == 3 * default_n_samples


@pytest.mark.parametrize("aggregator", [MinMaxLTTB, EveryNthPoint])
def test_overview_figure_gap_handler_similarity(aggregator):
    """Test whether the same gap handlers as those used in the figure are used in the
    overview figure"""
    fig = FigureResampler(create_overview=True, default_downsampler=aggregator())

    # create uneven data which contains gaps
    N = 20_000
    x = np.arange(N)
    for idx in np.random.randint(0, N, size=4):
        x[idx:] += np.random.randint(N / 10, N / 5)
    y = np.random.normal(size=N)

    fig.add_trace(go.Scatter(x=x, y=y), gap_handler=NoGapHandler())
    fig.add_trace({}, hf_x=x, hf_y=y, gap_handler=MedDiffGapHandler())
    fig.add_trace({}, hf_x=x, hf_y=y, gap_handler=MedDiffGapHandler(fill_value=42))

    overview_fig = fig._create_overview_figure()
    assert len(overview_fig.data) == 3
    assert np.isnan(overview_fig.data[0]["y"]).sum() == 0
    assert np.isnan(overview_fig.data[1]["y"]).sum() == 4
    assert (overview_fig.data[2]["y"] == 42).sum() == 4
