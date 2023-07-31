"""Dash file parquet visualization app example with a coarse and fine-grained view.

In this use case, we have dropdowns which allows end-users to select multiple
parquet files, which are visualized using FigureResampler after clicking on a button.

There a two graphs displayed; a coarse and a dynamic graph. Interactions with the
coarse graph will affect the dynamic graph it's shown range. Note that the autosize
of the coarse graph is not linked.

TODO: add an rectangle on the coarse graph

"""

__author__ = "Jonas Van Der Donckt"

import time
from pathlib import Path
from typing import List

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import (
    ClientsideFunction,
    Input,
    Output,
    State,
    callback_context,
    dcc,
    html,
    no_update,
)
from dash_extensions.enrich import (
    DashProxy,
    ServersideOutput,
    ServersideOutputTransform,
)
from trace_updater import TraceUpdater
from utils.callback_helpers import get_selector_states, multiple_folder_file_selector
from utils.graph_construction import visualize_multiple_files, remove_other_axes_for_coarse, get_total_rows_and_cols

from plotly_resampler import FigureResampler

# --------------------------------------Globals ---------------------------------------
app = DashProxy(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUX],
    transforms=[ServersideOutputTransform()],
)

# --------- File selection configurations ---------
name_folder_list = [
    {
        # the key-string below is the title which will be shown in the dash app
        "example data": {"folder": Path(__file__).parent.parent.joinpath("data")},
        "other folder": {"folder": Path(__file__).parent.parent.joinpath("data")},
    },
    {
        # the key-string below is the title which will be shown in the dash app
        "example data": {"folder": Path(__file__).parent.parent.joinpath("data")},
        "other folder": {"folder": Path(__file__).parent.parent.joinpath("data")},
    },
    # NOTE: A new item om this level creates a new file-selector card.
    # (we're gonna use this to create subplots! to test our callback functionality)
    # { "PC data": { "folder": Path("/home/jonas/data/wesad/empatica/") } }
    # TODO: change the folder path above to a location where you have some
    # `.parquet` files stored on your machine.
]

rangeslider_state = {}

global_linked_indices = [2,1,3]

# --------- DASH layout logic ---------
def serve_layout() -> dbc.Container:
    """Constructs the app's layout.

    Returns
    -------
    dbc.Container
        A Container withholding the layout.

    """
    return dbc.Container(
        [   
            dcc.Store(id='linked-subplots', data=global_linked_indices),
            dbc.Container(
                html.H1("Data visualization - coarse & dynamic graph"),
                style={"textAlign": "center"},
            ),
            html.Hr(),
            dbc.Row(
                [
                    # Add file selection layout (+ assign callbacks)
                    dbc.Col(
                        multiple_folder_file_selector(
                            app, name_folder_list, multi=False
                        ),
                        md=2,
                    ),
                    # Add the graphs, the dcc.Store (for serialization) and the
                    # TraceUpdater (for efficient data updating) components
                    dbc.Col(
                        [
                            dcc.Graph(id="plotly-resampler-graph", figure=go.Figure()),
                            dcc.Loading(dcc.Store(id="store")),
                            TraceUpdater(
                                id="trace-updater", gdID="plotly-resampler-graph"
                            ),
                            html.Br(),
                            # The coarse graph whose updates will fetch data for the
                            dcc.Graph(
                                id="coarse-graph",
                                figure=go.Figure(
                                    layout=go.Layout(
                                        clickmode="select",
                                        dragmode="select",
                                    )
                                ),
                            ),
                        ],
                        md=10,
                    ),
                ],
                align="center",
            ),
        ],
        fluid=True,
    )


app.layout = serve_layout()

def create_overview_figure(dynamic_fig, linked_indices):
    dyn_rows, dyn_cols = get_total_rows_and_cols(dynamic_fig)

    print([dyn_rows, dyn_cols])

    # adjust the linked_indices if they are set wrong by the user,
    # bring it to the closest possible indices on the dynamic fig
    if dyn_cols > 0 and len(linked_indices) > dyn_cols:
        linked_indices = linked_indices[:dyn_cols]
    elif dyn_cols == 0:
        linked_indices = [linked_indices[0]]

    # take the absolute value of the linked index, 
    # linked_indices = [item if dyn_rows == 0 else min(abs(item), dyn_rows - 1) for item in linked_indices]
    
    #alternative: take the modulo of the linked index
    # TODO_idea: let the user know the indices are out of range and that it has been wrapped
    linked_indices = [item%dyn_rows for item in linked_indices]

    print(linked_indices)

    coarse_fig: go.Figure = go.Figure(
        FigureResampler(remove_other_axes_for_coarse(dynamic_fig, linked_indices), default_n_shown_samples=3000)
    )

    # coarse_fig.update_layout(margin=dict(l=0, r=0, b=0, t=40, pad=10))
    # height of the overview scales with the height of the dynamic view
    coarse_fig.update_layout(showlegend=False, template="plotly_white", height=max(dynamic_fig.layout.height/5, 125))
    coarse_fig.update_layout(
        hovermode=False,
        clickmode="event+select",
        dragmode="select",
        activeselection=dict(fillcolor="coral", opacity=0.2),
    )

    coarse_layout = coarse_fig.layout
    for c in range(dyn_cols):
        coarse_layout[f'xaxis{c+1}']["fixedrange"] = True
        coarse_layout[f'yaxis{c+1}']["fixedrange"] = True
    
    # not needed
    # coarse_fig.update_layout(coarse_layout)


    coarse_fig._config = coarse_fig._config.update(
        {"modeBarButtonsToAdd": ["drawrect", "select2d"]}
        # adds a rangeslider to the coarse graph
    )
    return coarse_fig
            



# ------------------------------------ DASH logic -------------------------------------
# --------- graph construction logic + callback ---------
@app.callback(
    [
        Output("coarse-graph", "figure"),
        Output("plotly-resampler-graph", "figure"),
        ServersideOutput("store", "data"),
    ],
    [Input("plot-button", "n_clicks"), State("linked-subplots", 'data'), *get_selector_states(len(name_folder_list))],
    prevent_initial_call=True,
)
def construct_plot_graph(_, linked_indices, *folder_list):
    it = iter(folder_list)
    file_list: List[Path] = []
    for folder, files in zip(it, it):
        if not all((folder, files)):
            continue
        else:
            files = [files] if not isinstance(files, list) else file_list
            for file in files:
                file_list.append((Path(folder).joinpath(file)))

    ctx = callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        if len(file_list):
            # Create two graphs, a dynamic plotly-resampler graph and a coarse graph
            dynamic_fig: FigureResampler = visualize_multiple_files(file_list)

            print(get_total_rows_and_cols(dynamic_fig))
            coarse_fig = create_overview_figure(dynamic_fig, linked_indices)
            if False:
                coarse_fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list(
                                [
                                    dict(
                                        count=1,
                                        label="1d",
                                        step="day",
                                        stepmode="backward",
                                    ),
                                    dict(
                                        count=7,
                                        label="1w",
                                        step="day",
                                        stepmode="backward",
                                    ),
                                    dict(step="all"),
                                ]
                            )
                        ),
                        # rangeslider=dict(visible=True),
                        type="date",
                    )
                )

            coarse_fig.update_layout(margin=dict(l=0, r=0, b=0, t=40, pad=10))

            dynamic_fig._global_n_shown_samples = 1000
            dynamic_fig.update_layout(title="<b>dynamic view<b>")
            dynamic_fig.update_layout(template="plotly_white")
            dynamic_fig.update_layout(margin=dict(l=0, r=0, b=40, t=40, pad=10))
            dynamic_fig.update_layout(
                legend=dict(
                    orientation="h", y=1.01, xanchor="right", x=1, font_size=15
                ),
            )
            return coarse_fig, dynamic_fig, dynamic_fig
    else:
        return no_update

# update range with clientside callback
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="set_coarse_range"),
    Output("plotly-resampler-graph", "id", allow_duplicate=True),
    Input("coarse-graph", "figure"),
    State("plotly-resampler-graph", "id"),
    State("coarse-graph", "id"),
    State("linked-subplots", 'data'),
    prevent_initial_call=True,
)

# update range with clientside callback
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="coarse_to_main"),
    Output("plotly-resampler-graph", "id", allow_duplicate=True),
    Input("coarse-graph", "selectedData"),
    State("plotly-resampler-graph", "id"),
    State("coarse-graph", "id"),
    State("linked-subplots", 'data'),
    prevent_initial_call=True,
)

# update selectbox with clientside callback
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="main_to_coarse"),
    Output("coarse-graph", "id"),  # , allow_duplicate=True),
    Input("plotly-resampler-graph", "relayoutData"),
    State("coarse-graph", "id"),
    State("plotly-resampler-graph", "id"),
    State("linked-subplots", 'data'),
    prevent_initial_call=True,
)

# Register the graph update callbacks to the layout
@app.callback(
    Output("trace-updater", "updateData"),
    Input("plotly-resampler-graph", "relayoutData"),
    State("store", "data"),
    prevent_initial_call=True,
)
def construct_update_data(relayout, fr_fig):
    if fr_fig is None:  # When the figure does not exist -> do nothing
        return no_update

    print("update_dynamic_fig")
    # print(relayout)
    t_start = time.time()
    output = fr_fig.construct_update_data(relayout)
    if output != no_update:
        print("duration: ", round(1000 * (time.time() - t_start), 2), "ms")
    return output


# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9024)
