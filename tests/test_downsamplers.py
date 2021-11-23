from plotly_resampler.downsamplers import EveryNthPoint, AggregationDownsampler, LTTB
import pandas as pd
import numpy as np
import pytest

from .utils import float_series, cat_series, bool_series


# --------------------------------- EveryNthPoint ------------------------------------
def test_every_nth_point_float_time_data(float_series):
    float_series.index = pd.date_range(
        "1/1/2020", periods=len(float_series), freq="1ms"
    )
    for n in np.random.randint(100, len(float_series), 3):
        out = EveryNthPoint(interleave_gaps=False).downsample(float_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n

    for n in np.random.randint(100, len(float_series) / 3, 3):
        out = EveryNthPoint(interleave_gaps=True).downsample(float_series, n_out=n)
        assert sum(out.notna()) <= n


def test_every_nth_point_float_sequence_data(float_series):
    float_series.index = np.arange(len(float_series), dtype="uint32")
    for n in np.random.randint(100, len(float_series), 3):
        out = EveryNthPoint(interleave_gaps=False).downsample(float_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n

    for n in np.random.randint(100, len(float_series) / 3, 3):
        out = EveryNthPoint(interleave_gaps=True).downsample(float_series, n_out=n)
        assert sum(out.notna()) <= n


def test_every_nth_point_categorical_time_data(cat_series):
    cat_series.index = pd.date_range("1/1/2023", periods=len(cat_series), freq="10us")
    for n in np.random.randint(100, len(cat_series), 3):
        out = EveryNthPoint(interleave_gaps=False).downsample(cat_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n

    for n in np.random.randint(100, len(cat_series) / 3, 3):
        out = EveryNthPoint(interleave_gaps=True).downsample(cat_series, n_out=n)
        assert sum(out.notna()) <= n


def test_every_nth_point_categorical_sequence_data(cat_series):
    cat_series.index = np.arange(len(cat_series), dtype="uint32")
    for n in np.random.randint(100, len(cat_series), 3):
        out = EveryNthPoint(interleave_gaps=False).downsample(cat_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n

    for n in np.random.randint(100, len(cat_series) / 3, 3):
        out = EveryNthPoint(interleave_gaps=True).downsample(cat_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n


def test_every_nth_point_bool_time_data(bool_series):
    bool_series.index = pd.date_range("1/1/2020", periods=len(bool_series), freq="1ms")
    for n in np.random.randint(100, len(bool_series), 3):
        out = EveryNthPoint(interleave_gaps=False).downsample(bool_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n

    for n in np.random.randint(100, len(bool_series) / 3, 3):
        out = EveryNthPoint(interleave_gaps=True).downsample(bool_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n


def test_every_nth_point_bool_sequence_data(bool_series):
    bool_series.index = np.arange(len(bool_series), dtype="uint32")
    for n in np.random.randint(100, len(bool_series), 3):
        out = EveryNthPoint(interleave_gaps=False).downsample(bool_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n

    for n in np.random.randint(100, len(bool_series) / 3, 3):
        out = EveryNthPoint(interleave_gaps=True).downsample(bool_series, n_out=n)
        assert sum(out.notna()) <= n


# -------------------------------------- LTTB --------------------------------------
def test_lttb_float_time_data(float_series):
    float_series.index = pd.date_range(
        "1/1/2020", periods=len(float_series), freq="1ms"
    )
    for n in np.random.randint(100, len(float_series), 3):
        out = LTTB(interleave_gaps=False).downsample(float_series, n_out=n)
        assert not out.isna().any()
        assert len(out) == n

    for n in np.random.randint(100, len(float_series) / 3, 3):
        out = LTTB(interleave_gaps=True).downsample(float_series, n_out=n)
        assert sum(out.notna()) == n


def test_lttb_float_sequence_data(float_series):
    float_series.index = np.arange(len(float_series), dtype="uint32")
    for n in np.random.randint(100, len(float_series), 3):
        out = LTTB(interleave_gaps=False).downsample(float_series, n_out=n)
        assert not out.isna().any()
        assert len(out) == n

    for n in np.random.randint(100, len(float_series) / 3, 3):
        out = LTTB(interleave_gaps=True).downsample(float_series, n_out=n)
        assert sum(out.notna()) == n


def test_lttb_categorical_time_data(cat_series):
    cat_series.index = pd.date_range("1/5/2022", periods=len(cat_series), freq="10s")
    for n in np.random.randint(100, len(cat_series), 3):
        out = LTTB(interleave_gaps=False).downsample(cat_series, n_out=n)
        assert not out.isna().any()
        assert len(out) == n

    for n in np.random.randint(100, len(cat_series) / 3, 3):
        out = LTTB(interleave_gaps=True).downsample(cat_series, n_out=n)
        assert sum(out.notna()) == n


def test_lttb_categorical_sequence_data(cat_series):
    cat_series.index = np.arange(len(cat_series), dtype="uint32")
    for n in np.random.randint(100, len(cat_series), 3):
        out = LTTB(interleave_gaps=False).downsample(cat_series, n_out=n)
        assert not out.isna().any()
        assert len(out) == n

    for n in np.random.randint(100, len(cat_series) / 3, 3):
        out = LTTB(interleave_gaps=True).downsample(cat_series, n_out=n)
        assert sum(out.notna()) == n


def test_lttb_bool_time_data(bool_series):
    bool_series.index = pd.date_range("1/1/2020", periods=len(bool_series), freq="1s")
    for n in np.random.randint(100, len(bool_series), 3):
        out = LTTB(interleave_gaps=False).downsample(bool_series, n_out=n)
        assert not out.isna().any()
        assert len(out) == n

    for n in np.random.randint(100, len(bool_series) / 3, 3):
        out = LTTB(interleave_gaps=True).downsample(bool_series, n_out=n)
        assert sum(out.notna()) == n


def test_lttb_bool_sequence_data(bool_series):
    bool_series.index = np.arange(len(bool_series), dtype="uint32")
    for n in np.random.randint(100, len(bool_series), 3):
        out = LTTB(interleave_gaps=False).downsample(bool_series, n_out=n)
        assert not out.isna().any()
        assert len(out) == n

    for n in np.random.randint(100, len(bool_series) / 3, 3):
        out = LTTB(interleave_gaps=True).downsample(bool_series, n_out=n)
        assert sum(out.notna()) == n


def test_lttb_invalid_input_data():
    # string data
    nb_samples = 10_000
    string_arr = ["a", "bdc", "ef", "gh", "ijklkm", "nopq"]
    s = pd.Series(data=string_arr * (nb_samples // len(string_arr)))
    with pytest.raises(ValueError):
        LTTB(interleave_gaps=False).downsample(s, n_out=100)

    # time data
    time_arr = pd.date_range("1/1/2020", periods=nb_samples, freq="1s")
    s = pd.Series(data=time_arr)
    with pytest.raises(ValueError):
        LTTB(interleave_gaps=False).downsample(s, n_out=100)


# TODO - also tests for time series, time-series with gaps

# ------------------------------- AggregationDownsampler -------------------------------
def test_aggregation_downsampler_float_time_data(float_series):
    # TIME indexed data -> resampled output should be same size as n_out
    float_series.index = pd.date_range("1/1/2020", periods=len(float_series), freq="1s")
    for n in np.random.randint(100, len(float_series), 3):
        out = AggregationDownsampler(
            interleave_gaps=False, aggregation_func=sum
        ).downsample(float_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n + 1

    for n in np.random.randint(100, len(float_series) / 3, 3):
        out = AggregationDownsampler(
            interleave_gaps=True, aggregation_func=np.mean
        ).downsample(float_series, n_out=n)
        assert sum(out.notna()) <= n + 1


def test_aggregation_downsampler_float_sequence_data(float_series):
    # No time-index => we use every nth heuristic
    float_series.index = np.arange(len(float_series), dtype="uint32")
    for n in np.random.randint(100, len(float_series), 3):
        out = AggregationDownsampler(
            interleave_gaps=False, aggregation_func=sum
        ).downsample(float_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n + 1

    for n in np.random.randint(100, len(float_series) / 3, 3):
        out = AggregationDownsampler(
            interleave_gaps=True, aggregation_func=np.mean
        ).downsample(float_series, n_out=n)
        assert sum(out.notna()) <= n + 1


def test_aggregation_downsampler_categorical_time_data(cat_series):
    # TIME indexed data -> resampled output should be same size as n_out
    cat_series.index = pd.date_range("1/1/2020", periods=len(cat_series), freq="1s")

    def cat_count(x):
        return len(np.unique(x))

    for n in np.random.randint(100, len(cat_series), 3):
        out = AggregationDownsampler(
            interleave_gaps=False, aggregation_func=cat_count
        ).downsample(cat_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n + 1

    for n in np.random.randint(100, len(cat_series) / 3, 3):
        out = AggregationDownsampler(
            interleave_gaps=True, aggregation_func=cat_count
        ).downsample(cat_series, n_out=n)
        assert sum(out.notna()) <= n + 1


def test_aggregation_downsampler_categorical_sequence_data(cat_series):
    # TIME indexed data -> resampled output should be same size as n_out
    cat_series.index = np.arange(len(cat_series), dtype="uint32")
    cat_series = cat_series[: len(cat_series) // 4]
    # note this method takes a long time - so we only test a small number of samples
    def most_common(x):
        return x.value_counts().index[0]

    for n in np.random.randint(100, len(cat_series), 3):
        out = AggregationDownsampler(
            interleave_gaps=False, aggregation_func=most_common
        ).downsample(cat_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n + 1

    for n in np.random.randint(100, len(cat_series) / 3, 3):
        out = AggregationDownsampler(
            interleave_gaps=True, aggregation_func=most_common
        ).downsample(cat_series, n_out=n)
        assert sum(out.notna()) <= n + 1


def test_aggregation_downsampler_bool_time_data(bool_series):
    bool_series.index = pd.date_range("1/1/2020", periods=len(bool_series), freq="1s")

    def most_common(x):
        return sum(x) / len(x) >= 0.5

    for n in np.random.randint(100, len(bool_series) / 2, 3):
        out = AggregationDownsampler(
            interleave_gaps=False, aggregation_func=most_common
        ).downsample(bool_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n + 1

    for n in np.random.randint(100, len(bool_series) / 3, 3):
        out = AggregationDownsampler(
            interleave_gaps=True, aggregation_func=most_common
        ).downsample(bool_series, n_out=n)
        assert sum(out.notna()) <= n + 1


def test_aggregation_downsampler_bool_sequence_data(bool_series):
    bool_series.index = np.arange(len(bool_series), step=1, dtype="uint32")

    def most_common(x):
        return sum(x) / len(x) >= 0.5

    for n in np.random.randint(100, len(bool_series) / 2, 3):
        out = AggregationDownsampler(
            interleave_gaps=False, aggregation_func=most_common
        ).downsample(bool_series, n_out=n)
        assert not out.isna().any()
        assert len(out) <= n + 1

    for n in np.random.randint(100, len(bool_series) / 3, 3):
        out = AggregationDownsampler(
            interleave_gaps=True, aggregation_func=most_common
        ).downsample(bool_series, n_out=n)
        assert sum(out.notna()) <= n + 1


def test_aggregation_downsampler_invalid_input_data(cat_series):
    # note: it is the user's responsibility to ensure that the input data is valid
    def treat_string_as_numeric_data(x):
        return np.sum(x)

    n = np.random.randint(100, len(cat_series) / 3)
    with pytest.raises(TypeError):
        AggregationDownsampler(
            interleave_gaps=True, aggregation_func=treat_string_as_numeric_data
        ).downsample(cat_series, n_out=n)
