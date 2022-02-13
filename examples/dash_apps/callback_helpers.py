"""Dash helper functions for constructing a file seelector
"""

__author__ = "Jonas Van Der Donckt"

from pathlib import Path
from typing import Dict, List

import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from functional import seq


def _update_file_widget(folder):
    if folder is None:
        return []
    return [
        {"label": filename, "value": filename}
        for filename in sorted(
            set(
                list(
                    seq(Path(folder).iterdir())
                    .filter(lambda x: x.is_file() and x.name.endswith("parquet"))
                    .map(lambda x: x.name)
                )
            )
        )
    ]


def _register_selection_callbacks(app, ids=None):
    if ids is None:
        ids = [""]

    for id in ids:

        app.callback(
            Output(f"file-selector{id}", "options"),
            [Input(f"folder-selector{id}", "value")],
        )(_update_file_widget)


def multiple_folder_file_selector(
    app, name_folders_list: List[Dict[str, dict]]
) -> dbc.Card:
    """Constructs a folder user date selector

    Creates a `dbc.Card` component which can be

    Parameters
    ----------
    app:
        The dash application.
    name_folders_list:List[Dict[str, Union[Path, str]]]
         A dict with key, the display-key and values the correspondign path.

    Returns
    -------
    A bootstrap card component
    """
    selector = dbc.Card(
        [
            dbc.Card(
                [
                    dbc.Col(
                        [
                            dbc.Label("folder"),
                            dcc.Dropdown(
                                id=f"folder-selector{i}",
                                options=[
                                    {"label": l, "value": str(f["folder"])}
                                    for (l, f) in name_folders.items()
                                ],
                                clearable=False,
                            ),
                            dbc.Label("file"),
                            dcc.Dropdown(
                                id=f"file-selector{i}",
                                options=[],
                                clearable=True,
                                multi=True,
                            ),
                            html.Br(),
                        ]
                    ),
                ]
            )
            for i, name_folders in enumerate(name_folders_list, 1)
        ]
        + [
            dbc.Card(
                dbc.Col(
                    [
                        html.Br(),
                        dbc.Button(
                            "create figure",
                            id="plot-button",
                            color="primary",
                        ),
                    ],
                    style={"textAlign": "center"},
                ),
            )
        ],
        body=True,
    )

    _register_selection_callbacks(app=app, ids=range(1, len(name_folders_list) + 1))
    return selector
