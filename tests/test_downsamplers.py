from plotly_resampler.downsamplers import EveryNthPoint, AggregationDownsampler, LTTB
import pandas as pd
import numpy as np
import pytest

# ----------------------------------- LTTB -----------------------------------
def test_lttb_float_data():
    # TODO -fixtures
    nb_samples = 10_000
    x = np.arange(nb_samples).astype(np.uint32)
    y = np.sin(x / 300).astype(np.float32) + np.random.randn(nb_samples) / 5
    s = pd.Series(index=x, data=y)

    for n in np.random.randint(100, len(s), 3):
        out = LTTB(interleave_gaps=False).downsample(s, n_out=n)
        assert not out.isna().any()
        assert len(out) == n

    for n in np.random.randint(100, len(s) / 3, 3):
        out = LTTB(interleave_gaps=True).downsample(s, n_out=n)
        assert sum(out.notna()) == n


def test_lttb_categorical_data():
    cats = pd.Series(["a", "b", "b", "b", "c", "c", "c"] * 1_000, dtype="category")
    for n in np.random.randint(100, len(cats), 3):
        out = LTTB(interleave_gaps=False).downsample(cats, n_out=n)
        assert not out.isna().any()
        assert len(out) == n

    for n in np.random.randint(100, len(cats) / 3, 3):
        out = LTTB(interleave_gaps=True).downsample(cats, n_out=n)
        assert sum(out.notna()) == n


def test_lttb_bool_data():
    bools = pd.Series([True, False, True, True, True, True] * 1_000)
    for n in np.random.randint(100, len(bools), 3):
        out = LTTB(interleave_gaps=False).downsample(bools, n_out=n)
        assert not out.isna().any()
        assert len(out) == n

    for n in np.random.randint(100, len(bools) / 3, 3):
        out = LTTB(interleave_gaps=True).downsample(bools, n_out=n)
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
