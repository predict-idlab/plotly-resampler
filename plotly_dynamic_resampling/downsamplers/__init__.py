__author__ = 'Jonas Van Der Donckt'


from .resampling_interface import AbstractSeriesDownsampler
from .downsamplers import LTTB, EveryNthPoint, AggregationDownsampler
