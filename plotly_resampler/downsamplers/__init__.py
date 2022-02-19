"""
Compatible implementation for various downsample methods and open interface to 
other downsample methods.

"""

__author__ = 'Jonas Van Der Donckt'


from .downsampling_interface import AbstractSeriesDownsampler
from .downsamplers import LTTB, EveryNthPoint, AggregationDownsampler
