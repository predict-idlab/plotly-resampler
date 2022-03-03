import pandas
from plotly_resampler.utils import timedelta_to_str, round_td_str
import pandas as pd


def test_timedelta_to_str():
    assert (round_td_str(pd.Timedelta('1W'))) == '7D'
    assert (timedelta_to_str(pd.Timedelta('1W'))) == '7D'
    assert (timedelta_to_str(pd.Timedelta('1W') * -1)) == 'NEG7D'
    assert timedelta_to_str(pd.Timedelta('1s 114ms')) == '1.114s'
    assert round_td_str(pd.Timedelta('14.4ms')) == '14ms'
    assert round_td_str(pd.Timedelta('501ms')) == '1s'
    assert round_td_str(pd.Timedelta('500ms')) == '500ms'
    assert round_td_str(pd.Timedelta('14.4ms')) == '14ms'
    assert round_td_str(pd.Timedelta('14.6ms')) == '15ms'
    assert round_td_str(pd.Timedelta('1h 14.4us')) == '1h'
    assert round_td_str(pd.Timedelta('1128.9us')) == '1ms'
    assert round_td_str(pd.Timedelta('128.9us')) == '129us'
    assert round_td_str((pd.Timedelta('14ns'))) == '14ns'