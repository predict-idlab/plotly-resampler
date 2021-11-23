"""Fixtures and helper functions for testing"""


import pytest
import pandas as pd
import numpy as np

_nb_samples = 10_000


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

