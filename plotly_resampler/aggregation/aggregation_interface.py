"""AbstractSeriesAggregator interface-class, subclassed by concrete aggregators."""

__author__ = "Jonas Van Der Donckt"

import re
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class AbstractSeriesAggregator(ABC):
    """"""

    def __init__(
        self,
        interleave_gaps: bool = True,
        dtype_regex_list: List[str] = None,
        max_gap_detection_data_size: int = 25_000,
    ):
        """Constructor of AbstractSeriesAggregator.

        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.
        dtype_regex_list: List[str], optional
            List containing the regex matching the supported datatypes, by default None.
        max_gap_detection_data_size: int, optional
            The maximum raw-data size on which gap detection is performed. If the
            raw data size exceeds this value, gap detection will be performed on
            the aggregated (a.k.a. downsampled) series.

            .. note::
                This parameter only has an effect if ``interleave_gaps`` is set to True.

        """
        self.interleave_gaps = interleave_gaps
        self.dtype_regex_list = dtype_regex_list
        self.max_gap_q = 0.975
        self.max_gap_data_size = max_gap_detection_data_size
        super().__init__()

    @abstractmethod
    def _aggregate(self, s: pd.Series, n_out: int) -> pd.Series:
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

    def _get_gap_df(self, s: pd.Series) -> Optional[pd.Series]:
        # ------- add None where there are gaps / irregularly sampled data
        if isinstance(s.index, pd.DatetimeIndex):
            series_index_diff = s.index.to_series().diff().dt.total_seconds()
        else:
            series_index_diff = s.index.to_series().diff()

        # use a quantile-based approach
        med_gap_s, max_q_gap_s = series_index_diff.quantile(q=[0.5, self.max_gap_q])

        # add None data-points in between the gaps
        if med_gap_s is not None and max_q_gap_s is not None:
            max_q_gap_s = max(2 * med_gap_s, max_q_gap_s)
            df_res_gap = s.loc[series_index_diff > max_q_gap_s].copy()
            if len(df_res_gap):
                df_res_gap.loc[:] = None
                return df_res_gap
        return None

    def aggregate(self, s: pd.Series, n_out: int) -> pd.Series:
        """Aggregate (downsample) the given input series to the given n_out samples.

        Parameters
        ----------
        s: pd.Series
            The series that has to be aggregated.
        n_out: int
            The number of samples that the downsampled series should contain.

        Returns
        -------
        pd.Series
            The aggregated series.

        """
        # base case: the passed series is empty
        if s.empty:
            return s

        self._supports_dtype(s)

        # convert the bool values to uint8 (as we will display them on a y-axis)
        if str(s.dtype) == "bool":
            s = s.astype("uint8")

        gaps = None
        raw_slice_size = s.shape[0]
        if self.interleave_gaps and raw_slice_size < self.max_gap_data_size:
            # if the raw-data slice is not too large -> gaps are detected on the raw
            # data
            gaps = self._get_gap_df(s)

        if len(s) > n_out:
            s = self._aggregate(s, n_out=n_out)

        if self.interleave_gaps and raw_slice_size >= self.max_gap_data_size:
            # if the raw-data slice is too large -> gaps are detected on the
            # aggregated data
            gaps = self._get_gap_df(s)

        if gaps is not None:
            # Note:
            #  * the order of pd.concat is important for correct visualization
            #  * we also need a stable algorithm for sorting, i.e., the equal-index
            #    data-entries their order will be maintained.
            return pd.concat([gaps, s], ignore_index=False).sort_index(kind="mergesort")
        return s
