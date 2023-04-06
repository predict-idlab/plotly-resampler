import numpy as np
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture as lf

from plotly_resampler.aggregation import (
    LTTB,
    EveryNthPoint,
    FuncAggregator,
    MinMaxAggregator,
    MinMaxLTTB,
    MinMaxOverlapAggregator,
)
from plotly_resampler.aggregation.algorithms.lttb_c import LTTB_core_c
from plotly_resampler.aggregation.algorithms.lttb_py import LTTB_core_py

from .utils import construct_index, wrap_aggregate


# ------------------------------- DatapointSelector ----------------------------------
@pytest.mark.parametrize(
    "downsampler",
    [
        # NOTE:-> the current LTTB based aggregators need an `x`
        # LTTB,  MinMaxLTTB,
        EveryNthPoint,
        MinMaxAggregator,
        MinMaxOverlapAggregator,
    ],
)
# NOTE: -> categorical series it's values is not of dtype array; but the
# PlotlyAggregatorParser is able to deal with this
@pytest.mark.parametrize("series", [lf("float_series"), lf("bool_series")])
@pytest.mark.parametrize("interleave_gaps", [True, False])
def test_arg_downsample_no_x(series, downsampler, interleave_gaps):
    for n in np.random.randint(100, len(series), 6):
        # make sure n is even (required for MinMax downsampler)
        n = n - (n % 2)
        indices = downsampler(interleave_gaps=interleave_gaps).arg_downsample(
            series.values, n_out=n
        )
        assert len(indices) <= n + (n % 2)


@pytest.mark.parametrize(
    "downsampler",
    [
        LTTB,
        MinMaxLTTB,
        EveryNthPoint,
        MinMaxAggregator,
        MinMaxOverlapAggregator,
    ],
)
@pytest.mark.parametrize("series", [lf("float_series"), lf("bool_series")])
@pytest.mark.parametrize("interleave_gaps", [True, False])
@pytest.mark.parametrize("index_type", ["datetime", "timedelta", "float", "int"])
def test_arg_downsample_x(series, downsampler, interleave_gaps, index_type):
    series = series.copy()
    series.index = construct_index(series, index_type)
    for n in np.random.randint(100, len(series), 6):
        # make sure n is even (required for MinMax downsampler)
        n = n - (n % 2)
        indices = downsampler(interleave_gaps=interleave_gaps).arg_downsample(
            series.index.values, series.values, n_out=n
        )
        assert len(indices) <= n + (n % 2)


@pytest.mark.parametrize(
    "downsampler",
    [EveryNthPoint, LTTB, MinMaxAggregator, MinMaxLTTB, MinMaxOverlapAggregator],
)
@pytest.mark.parametrize("series", [lf("float_series"), lf("bool_series")])
@pytest.mark.parametrize("interleave_gaps", [True, False])
@pytest.mark.parametrize("index_type", ["datetime", "timedelta", "float", "int"])
def test_arg_downsample_empty_series(downsampler, series, interleave_gaps, index_type):
    series = series.copy()
    series.index = construct_index(series, index_type)
    empty_series = series.iloc[0:0]
    idxs = downsampler(interleave_gaps=interleave_gaps).arg_downsample(
        empty_series.index.values, empty_series.values, n_out=1_000
    )
    assert len(idxs) == 0


@pytest.mark.parametrize(
    "downsampler",
    [EveryNthPoint, MinMaxAggregator, MinMaxLTTB, MinMaxOverlapAggregator],
)
@pytest.mark.parametrize("series", [lf("float_series"), lf("bool_series")])
@pytest.mark.parametrize("interleave_gaps", [True, False])
def test_arg_downsample_no_x_empty_series(downsampler, series, interleave_gaps):
    empty_series = series.iloc[0:0]
    idxs = downsampler(interleave_gaps=interleave_gaps).arg_downsample(
        empty_series.values, n_out=1_000
    )
    assert len(idxs) == 0


@pytest.mark.parametrize(
    "downsampler",
    [EveryNthPoint, LTTB, MinMaxAggregator, MinMaxLTTB, MinMaxOverlapAggregator],
)
@pytest.mark.parametrize(
    "series",
    [lf("float_series"), lf("cat_series"), lf("bool_series")],
)
@pytest.mark.parametrize("interleave_gaps", [True, False])
@pytest.mark.parametrize("index_type", ["datetime", "timedelta", "float", "int"])
def test_wrap_aggregate(downsampler, series, interleave_gaps, index_type):
    series = series.copy()
    series.index = construct_index(series, index_type)
    for n in np.random.randint(100, len(series), 6):
        # make sure n is even (required for MinMax downsampler)
        n = n - (n % 2)
        x_agg, y_agg, indices = wrap_aggregate(
            hf_x=series.index,
            hf_y=series.values,
            downsampler=downsampler(interleave_gaps=interleave_gaps),
            n_out=n,
        )
        assert not pd.Series(y_agg).isna().any()
        assert len(x_agg) == len(y_agg) == len(indices)
        assert len(y_agg) <= n + (n % 2)


@pytest.mark.parametrize(
    "downsampler",
    [EveryNthPoint, LTTB, MinMaxAggregator, MinMaxLTTB, MinMaxOverlapAggregator],
)
@pytest.mark.parametrize("series", [lf("float_series"), lf("bool_series")])
@pytest.mark.parametrize("interleave_gaps", [True, False])
@pytest.mark.parametrize("index_type", ["datetime", "timedelta", "float", "int"])
def test_wrap_aggregate_empty_series(downsampler, series, interleave_gaps, index_type):
    empty_series = series.copy()
    empty_series.index = construct_index(empty_series, index_type)
    empty_series = empty_series.iloc[0:0]
    x_agg, y_agg, indices = wrap_aggregate(
        hf_x=empty_series.index,
        hf_y=empty_series.values,
        downsampler=downsampler(interleave_gaps=interleave_gaps),
        n_out=1000,
    )
    assert len(x_agg) == len(y_agg) == len(indices) == 0


@pytest.mark.parametrize(
    "downsampler",
    [EveryNthPoint, LTTB, MinMaxAggregator, MinMaxLTTB, MinMaxOverlapAggregator],
)
@pytest.mark.parametrize(
    "series", [lf("float_series"), lf("bool_series"), lf("cat_series")]
)
def test_wrap_aggregate_x_gaps(downsampler, series):
    series = series.copy()
    # Create a range-index with some gaps in it
    idx = np.arange(len(series))
    idx[1000:] += 1000
    idx[2000:] += 1500
    idx[8000:] += 2500
    series.index = idx

    x_agg, y_agg, indices = wrap_aggregate(
        hf_x=series.index,
        hf_y=series.values,
        downsampler=downsampler(interleave_gaps=True),
        n_out=100,
    )
    assert len(x_agg) == len(y_agg) == len(indices)
    assert pd.Series(y_agg).isna().sum() == 3


# # ------------------------------- DataAggregator -------------------------------
@pytest.mark.parametrize("agg_func", [np.mean])  # np.median, sum])
@pytest.mark.parametrize("series", [lf("float_series"), lf("bool_series")])
@pytest.mark.parametrize("interleave_gaps", [False, True])
@pytest.mark.parametrize("index_type", ["datetime", "timedelta", "float", "int"])
def test_func_aggregator_float_time_data(series, interleave_gaps, index_type, agg_func):
    # TIME indexed data -> resampled output should be same size as n_out
    series = series.copy()
    series.index = construct_index(series, index_type)
    for n in np.random.randint(100, len(series), 6):
        x_agg, y_agg, indices = wrap_aggregate(
            hf_x=series.index,
            hf_y=series.values,
            downsampler=FuncAggregator(
                interleave_gaps=interleave_gaps, aggregation_func=agg_func
            ),
            n_out=100,
        )
        assert not pd.Series(y_agg).isna().any()
        assert len(x_agg) == len(y_agg) == len(indices)
        assert len(y_agg) <= n + (n % 2)


def test_func_aggregator_categorical_time_data(cat_series):
    # TIME indexed data -> resampled output should be same size as n_out
    cat_series.index = pd.date_range("1/1/2020", periods=len(cat_series), freq="1s")

    def cat_count(x):
        return len(np.unique(x))

    for n in np.random.randint(100, len(cat_series), 3):
        agg_x, agg_y = FuncAggregator(
            interleave_gaps=False, aggregation_func=cat_count
        ).aggregate(cat_series.index.values, cat_series.values.codes, n_out=n)
        assert not np.isnan(agg_y).any()
        assert len(agg_x) <= n + 1


def test_func_aggregator_invalid_input_data(cat_series):
    # note: it is the user's responsibility to ensure that the input data is valid
    def treat_string_as_numeric_data(x):
        return np.mean(x)

    n = np.random.randint(100, len(cat_series) / 3)
    with pytest.raises(TypeError):
        FuncAggregator(
            interleave_gaps=True, aggregation_func=treat_string_as_numeric_data
        ).aggregate(cat_series.index.values, cat_series.to_numpy(), n_out=n)


def test_funcAggregator_no_x():
    n = 1_000_000
    x = np.arange(n)
    y = np.sin(x / (x.shape[0] / 30)) + np.random.randn(n)

    fa = FuncAggregator(np.mean, interleave_gaps=False)
    for n_out in np.random.randint(500, 1_000, size=3):
        fa.aggregate(y, n_out=n_out)


# ------------------------------- MinMaxLTTB -------------------------------
def test_MinMaxLTTB_size():
    # This test was made to certainly trigger the threshold for the MinMaxLTTB algorithm
    n = 12_000_000
    x = np.arange(n)
    y = np.sin(x / (x.shape[0] / 30)) + np.random.randn(n)
    mmltb = MinMaxLTTB(interleave_gaps=False)
    for n_out in np.random.randint(500, 1_000, size=3):
        assert n_out == mmltb._arg_downsample(x, y, n_out).shape[0]


# # ------------------------------- LTTB_Bindings -------------------------------
def test_lttb_bindings():
    # Test whether both algorithms produce the same results with different types of
    # input data
    n = np.random.randint(low=1_000_000, high=2_000_000)
    x_int = np.arange(n, dtype="int64")
    x_double = x_int.astype("float64")
    y_double = np.sin(x_int / 300) + np.random.randn(n)
    y_float = y_double.astype("float32")
    y_int = (100 * y_double).astype("int64")
    y_bool = (x_int % 250).astype("bool")

    for n_out in np.random.randint(500, 2000, size=3):
        sampled_x_c = LTTB_core_c.downsample(x_int, y_double, n_out)
        sampled_x_py = LTTB_core_py.downsample(x_int, y_double, n_out)
        assert sum(sampled_x_c == sampled_x_py) / len(sampled_x_c) > 0.995

        sampled_x_c = LTTB_core_c.downsample(x_int, y_float, n_out)
        sampled_x_py = LTTB_core_py.downsample(x_int, y_float, n_out)
        assert sum(sampled_x_c == sampled_x_py) / len(sampled_x_c) > 0.995

        sampled_x_c = LTTB_core_c.downsample(x_int, y_int, n_out)
        sampled_x_py = LTTB_core_py.downsample(x_int, y_int, n_out)
        assert sum(sampled_x_c == sampled_x_py) / len(sampled_x_c) > 0.995

        sampled_x_c = LTTB_core_c.downsample(x_int, y_bool, n_out)
        sampled_x_py = LTTB_core_py.downsample(x_int, y_bool, n_out)
        assert sum(sampled_x_c == sampled_x_py) / len(sampled_x_c) > 0.995

        sampled_x_c = LTTB_core_c.downsample(x_double, y_double, n_out)
        sampled_x_py = LTTB_core_py.downsample(x_double, y_double, n_out)
        assert sum(sampled_x_c == sampled_x_py) / len(sampled_x_c) > 0.995
