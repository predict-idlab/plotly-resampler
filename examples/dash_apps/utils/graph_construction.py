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
    fig = FigureResampler(make_subplots(cols=len(file_list), rows=3 , shared_xaxes=False))
    fig.update_layout(height=min(2000, 300 * 3))
    

    for i, f in enumerate(file_list, 1):
        #TODO: remove this for loop. was for testing subplots
        for j in range(3):
            df = pd.read_parquet(f)  # TODO: replace with more generic data loading code
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")

            for c in df.columns[::-1]:
                fig.add_trace(go.Scatter(name=c), hf_x=df.index, hf_y=df[c], row=j+1, col=i)
    return fig

# TODO: Adjust function to work with a list of linkedIndices
# => (determines which subplots to take from main and put into coarse)
def remove_other_axes_for_coarse(figure):
    rows, cols = get_total_rows_and_cols(figure)
    # axes are numbered from left to right, top to bottom.
    # subplot in row 1, col 1 will have axes xy, subplot in row 1, col 2 will have axes x2y2, and so on...
    # to obtain the first subplot of every column, 
    # one must simply take the axes xy until xmym (m being the number of columns)

    first_y_indices = [('y' if r == 0 else f'y{r+1}') for r in range(cols)]
    first_x_indices = [('x' if r == 0 else f'x{r+1}') for r in range(cols)]

    # print(first_y_indices)
    # print(first_x_indices)
    
    first_subplot = []
    for i, d in enumerate(figure.data):
        if (d['xaxis'] in first_x_indices) & (d['yaxis'] in first_y_indices):
            first_subplot.append(d)
    fig_tmp = go.Figure(data=first_subplot)
    fig = make_subplots(rows=1, cols=cols, figure=fig_tmp)
    return fig

def get_total_rows_and_cols(figure):
    x_axes = set()
    y_axes = set()

    for key, value in figure.layout.to_plotly_json().items():
        if key.startswith("xaxis"):
            x_axes.add(tuple(value["domain"]))
        elif key.startswith("yaxis"):
            y_axes.add(tuple(value["domain"]))

    num_rows = len(y_axes)
    num_columns = len(x_axes)
    return (num_rows,num_columns)
