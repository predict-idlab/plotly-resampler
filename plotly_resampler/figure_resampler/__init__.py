# -*- coding: utf-8 -*-
"""
Module withholding wrappers for the plotly ``go.Figure`` and ``go.FigureWidget`` class 
which allows bookkeeping and back-end based resampling of high-frequency sequential
data.

Tip
---
The term `high-frequency` actually refers very large amounts of sequential data.

"""

from .figure_resampler import FigureResampler
from .figurewidget_resampler import FigureWidgetResampler

__all__ = [
    "FigureResampler",
    "FigureWidgetResampler",
]
