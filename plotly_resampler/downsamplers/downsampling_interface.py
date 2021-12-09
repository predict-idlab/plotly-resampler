"""AbstractSeriesDownsampler interface-class, subclassed by concrete downsamplers."""

__author__ = "Jonas Van Der Donckt"

import re
from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class AbstractSeriesDownsampler(ABC):
    """"""

    def __init__(
        self, interleave_gaps: bool = True, dtype_regex_list: List[str] = None
    ):
        """Constructor of AbstractSeriesDownsampler.

        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly 
            sampled data. A quantile based approach is used to determine the gaps /
            irregularly sampled data. By default True.
        dtype_regex_list: List[str], optional
            List containing the regex matching the supported datatypes, by default None.
        
        """
        self.interleave_gaps = interleave_gaps
        self.dtype_regex_list = dtype_regex_list
        self.max_gap_q = 0.95
        super().__init__()

    @abstractmethod
    def _downsample(self, s: pd.Series, n_out: int) -> pd.Series:
        pass

    def _supports_dtype(self, s: pd.Series):
        # base case
        if self.dtype_regex_list is None:
            return

        for dtype_regex_str in self.dtype_regex_list:
            m = re.compile(dtype_regex_str).match(str(s.dtype))
            if m is not None:  # a match is found
                return
        raise ValueError(
            f"{s.dtype} doesn't match with any regex in {self.dtype_regex_list}"
        )

    def _interleave_gaps_none(self, s: pd.Series):
        # ------- add None where there are gaps / irregularly sampled data
        if isinstance(s.index, pd.DatetimeIndex):
            series_index_diff = s.index.to_series().diff().dt.total_seconds()
        else:
            series_index_diff = s.index.to_series().diff()

        # use a quantile based approach
        med_gap_s, max_q_gap_s = series_index_diff.quantile(q=[0.5, self.max_gap_q])

        # add None data-points in between the gaps
        if med_gap_s is not None and max_q_gap_s is not None:
            max_q_gap_s = max(2 * med_gap_s, max_q_gap_s)
            df_res_gap = s.loc[series_index_diff > max_q_gap_s ].copy()
            if len(df_res_gap):
                df_res_gap.loc[:] = None
                # Note:
                #  * the order of pd.concat is important for correct visualization
                #  * we also need a stable algorithm for sorting, i.e., the equal-index
                #    data-entries their order will be maintained.
                return pd.concat([df_res_gap, s], ignore_index=False).sort_index(
                    kind="mergesort"
                )
        return s

    def downsample(self, s: pd.Series, n_out: int) -> pd.Series:
        # base case: the passed series is empty
        if s.empty:
            return s

        self._supports_dtype(s)

        # convert the bool values to uint8 (as we will display them on a y-axis)
        if str(s.dtype) == "bool":
            s = s.astype("uint8")

        if len(s) > n_out:
            s = self._downsample(s, n_out=n_out)

        if self.interleave_gaps:
            s = self._interleave_gaps_none(s)

        return s
