__author__ = 'Jonas Van Der Donckt'


from .downsampling_interface import AbstractSeriesDownsampler
from .downsamplers import LTTB, EveryNthPoint, AggregationDownsampler
