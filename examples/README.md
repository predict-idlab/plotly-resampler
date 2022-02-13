# plotly-resampler examples

This directory withholds several examples, indicating the applicability of plotly-resampler in various use cases.

## 0. basic example

The testing CI/CD of plotly resampler uses _selenium_ and _selenium-wire_ to test the interactiveness of various figures. All these figures are shown in the [basic-example notebook](basic_example.ipynb)

## 1. Dash apps
The [dash_apps](dash_apps/dash_app.py) folder contains example dash apps in which `plotly-resampler` is integrated


| app-name | description |
| --- | --- |
| [file visualization](dash_apps/dash_app.py) | load and visualize multiple `.parquet` files with plotly-resampler |