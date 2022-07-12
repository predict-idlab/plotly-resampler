"""AbstractSeriesAggregator interface-class, subclassed by concrete aggregators."""

__author__ = "Jonas Van Der Donckt"

import re
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd


class AbstractSeriesAggregator(ABC):
    """"""

    def __init__(
        self,
        interleave_gaps: bool = True,
        nan_position: str = "end",
        dtype_regex_list: List[str] = None,
    ):
        """Constructor of AbstractSeriesAggregator.

        Parameters
        ----------
        interleave_gaps: bool, optional
            Whether None values should be added when there are gaps / irregularly
            sampled data. A quantile-based approach is used to determine the gaps /
            irregularly sampled data. By default, True.
        nan_position: str, optional
            Indicates where nans must be placed when gaps are detected. \n
            If ``'end'``, the first point after a gap will be replaced with a 
            nan-value \n
            If ``'begin'``, the last point before a gap will be replaced with a 
            nan-value \n
            If ``'both'``, both the encompassing gap datapoints are replaced with 
            nan-values \n
            .. note::
                This parameter only has an effect when ``interleave_gaps`` is set 
                to *True*.
        dtype_regex_list: List[str], optional
            List containing the regex matching the supported datatypes, by default None.

        """
        self.interleave_gaps = interleave_gaps
        self.dtype_regex_list = dtype_regex_list
        self.nan_position = nan_position.lower()
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

    @staticmethod
    def _calc_med_diff(s: pd.Series) -> Tuple[float, np.ndarray]:
        # ----- divide and conquer heuristic to calculate the median diff ------
        # remark: thanks to the prepend -> s_idx_diff.shape === len(s)
        siv = s.index.values
        s_idx_diff = np.diff(s.index.values, prepend=siv[0])

        # To do so - use a quantile-based (median) approach where we reshape the data
        # into `n_blocks` blocks and calculate the min
        n_blcks = 128
        if s.shape[0] > 5 * n_blcks:
            blck_size = s_idx_diff.shape[0] // n_blcks

            # convert the index series index diff into a reshaped view (i.e., sid_v)
            sid_v: np.ndarray = s_idx_diff[: blck_size * n_blcks].reshape(n_blcks, -1)

            # calculate the min and max and calculate the median on that
            med_diff = np.median(np.mean(sid_v, axis=1))
        else:
            med_diff = np.median(s_idx_diff)

        return med_diff, s_idx_diff

    def _insert_gap_none(self, s: pd.Series) -> pd.Series:
        # ------- INSERT None between gaps / irregularly sampled data -------
        med_diff, s_idx_diff = self._calc_med_diff(s)
        # add None data-points in-between the gaps
        if med_diff is not None:
            df_gap_idx = s.index.values[s_idx_diff > 3 * med_diff]
            if len(df_gap_idx):
                df_res_gap = pd.Series(
                    index=df_gap_idx, data=None, name=s.name, copy=False, dtype=s.dtype
                )

                if isinstance(df_res_gap.index, pd.DatetimeIndex):
                    # Due to the s.index`.values` cast, df_res_gap has lost
                    # time-information, so now we restore it
                    df_res_gap.index = df_res_gap.index.tz_localize("UTC").tz_convert(
                        s.index.tz
                    )

                # Note:
                #  * the order of pd.concat is important for correct visualization
                #  * we also need a stable algorithm for sorting, i.e., the equal-index
                #    data-entries their order will be maintained.
                s = pd.concat([df_res_gap, s], ignore_index=False).sort_index(
                    kind="mergesort"
                )
        return s

    def _replace_gap_end_none(self, s: pd.Series) -> pd.Series:
        # ------- REPLACE None where a gap ends -------
        med_diff, s_idx_diff = self._calc_med_diff(s)
        if med_diff is not None:
            # Replace data-points with None where the gaps occur
            # The default is the end of a gap
            nan_mask = s_idx_diff > 4 * med_diff
            if self.nan_position == "begin":
                # Replace the last non-gap datapoint (begin of gap) with Nan
                nan_mask = np.roll(nan_mask, -1)
            elif self.nan_position == "both":
                # Replace the encompassing gap datapoints with Nan
                nan_mask |= np.roll(nan_mask, -1)
            s.loc[nan_mask] = None
        return s

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

        if len(s) > n_out:
            # More samples that n_out -> perform data aggregation
            s = self._aggregate(s, n_out=n_out)

            # When data aggregation is performed -> we do not "insert" gaps but replace
            # The end of gap periods (i.e. the first non-gap sample) with None to
            # induce such gaps
            if self.interleave_gaps:
                s = self._replace_gap_end_none(s)
        else:
            # Less samples than n_out -> no data aggregation need to be performed

            # on the raw data -> gaps are inserted instead of replaced; i.e., we show
            # all data points and do not omit data-points with None
            if self.interleave_gaps:
                s = self._insert_gap_none(s)

        return s
