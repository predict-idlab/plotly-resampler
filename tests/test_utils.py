import pandas
from plotly_resampler.utils import timedelta_to_str, round_td_str, round_number_str
import pandas as pd


def test_timedelta_to_str():
    assert (round_td_str(pd.Timedelta('1W'))) == '7D'
    assert (timedelta_to_str(pd.Timedelta('1W'))) == '7D'
    assert (timedelta_to_str(pd.Timedelta('1W') * -1)) == 'NEG7D'
    assert timedelta_to_str(pd.Timedelta('1s 114ms')) == '1.114s'
    assert round_td_str(pd.Timedelta('14.4ms')) == '14ms'
    assert round_td_str(pd.Timedelta('501ms')) == '501ms'
    assert round_td_str(pd.Timedelta('951ms')) == '1s'
    assert round_td_str(pd.Timedelta('950ms')) == '950ms'
    assert round_td_str(pd.Timedelta('949ms')) == '949ms'
    assert round_td_str(pd.Timedelta('500ms')) == '500ms'
    assert round_td_str(pd.Timedelta('14.4ms')) == '14ms'
    assert round_td_str(pd.Timedelta('14.6ms')) == '15ms'
    assert round_td_str(pd.Timedelta('1h 14.4us')) == '1h'
    assert round_td_str(pd.Timedelta('1128.9us')) == '1ms'
    assert round_td_str(pd.Timedelta('128.9us')) == '129us'
    assert round_td_str((pd.Timedelta('14ns'))) == '14ns'


def test_round_int_str():
    assert round_number_str(0.951) == '1'
    assert round_number_str(0.95) == '0.9'
    assert round_number_str(0.949) == '0.9'
    assert round_number_str(0.00949) == '0.009'
    assert round_number_str(0.00950) == '0.009'
    assert round_number_str(0.00951) == '0.01'
    assert round_number_str(0.0044) == '0.004'
    assert round_number_str(0.00451) == '0.005'
    assert round_number_str(0.0001) == '0.0001'
    assert round_number_str(0.00001) == '1e-05'
    assert round_number_str(0.000000321) == '3e-07'
    assert round_number_str(12_000) == '12k'
    assert round_number_str(13_340) == '13k'
    assert round_number_str(13_540) == '14k'
    assert round_number_str(559_540) == '560k'
    assert round_number_str(949_000) == '949k'
    assert round_number_str(950_000) == '950k'
    assert round_number_str(950_001) == '1M'
    assert round_number_str(1_950_001) == '2M'
    assert round_number_str(111_950_001) == '112M'