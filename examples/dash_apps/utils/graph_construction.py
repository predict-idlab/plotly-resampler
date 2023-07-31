from pathlib import Path
from typing import List, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy

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
    fig.update_layout(height=min(750, 250 * 3))
    

    for i, f in enumerate(file_list, 1):
        #TODO: remove this for loop. was for testing subplots
        for j in range(3):
            df = pd.read_parquet(f)  # TODO: replace with more generic data loading code
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")

            for c in df.columns[::-1]:
                fig.add_trace(go.Scatter(name=c), hf_x=df.index, hf_y=df[c], row=j+1, col=i)
    return fig

# determines which subplot data to take from main and put into coarse
def remove_other_axes_for_coarse(figure, linked_indices):
    _, cols = get_total_rows_and_cols(figure)
    # axes are numbered from left to right, top to bottom.
    # subplot in row 1, col 1 will have axes xy, subplot in row 1, col 2 will have axes x2y2, and so on...
    # to obtain the first subplot of every column, 
    # one must simply take the axes xy until xmym (m being the number of columns)

    main_graph_indices = [len(linked_indices) * d + i for (i, d) in enumerate(linked_indices)]
    # for (i , d)  in enumerate(linked_indices):
    #     main_graph_indices.append(len(linked_indices) * d + i)

    y_indices = [('y' if r == 0 else f'y{r+1}') for r in main_graph_indices]
    x_indices = [('x' if r == 0 else f'x{r+1}') for r in main_graph_indices]
    

    first_subplot = []
    figdata = copy.deepcopy(figure.data)
    for i, d in enumerate(figdata):
        if (d['xaxis'] in x_indices) & (d['yaxis'] in y_indices):
            col = x_indices.index(d['xaxis'])
            d['xaxis'] = 'x' if col == 0 else f'x{col+1}'
            d['yaxis'] = 'y' if col == 0 else f'y{col+1}'
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
