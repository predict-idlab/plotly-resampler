from pathlib import Path
from typing import List, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly_resampler import FigureResampler


# --------- graph construction logic + callback ---------
def visualize_multiple_files(file_list: List[Union[str, Path]]) -> FigureResampler:
    """Create FigureResampler where each subplot row represents all signals from a file.

    Parameters
    ----------
    file_list: List[Union[str, Path]]

    Returns
    -------
    FigureResampler
        Returns a view of the existing, global FigureResampler object.

    """
    fig = FigureResampler(make_subplots(rows=len(file_list), shared_xaxes=False))
    fig.update_layout(height=min(900, 350 * len(file_list)))

    for i, f in enumerate(file_list, 1):
        df = pd.read_parquet(f)  # should be replaced by more generic data loading code
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        for c in df.columns[::-1]:
            print(df[c].dtype)
            fig.add_trace(go.Scattergl(name=c), hf_x=df.index, hf_y=df[c], row=i, col=1)
    return fig
