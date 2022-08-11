# plotly-resampler examples

This directory withholds several examples, highlighting the applicability of plotly-resampler for various use cases.


## Prerequisites

To successfully run these examples, make sure that you've installed all the [requirements](requirements.txt) by running:
```bash
pip install -r requirements.txt
```

## 1. Example notebooks
### 1.1 basic examples

The [basic example notebook](basic_example.ipynb) covers most use-cases in which plotly resampler will be employed. It servers as an ideal starting point for data-scientists who want to use plotly-resampler in their day-to-day jupyter environments.

Additionally, this notebook also shows some more advanced functionalities, such as:
* Retaining (a static) plotly-resampler figure in your notebook
* Adjusting trace data of plotly-resampler figures at runtime
* The flexibility of configuring different aggregation-algorithms and number of shown samples per trace


### 1.2 Figurewidget example

The [figurewidget example notebook](figurewidget_example.ipynb) utilizes the `FigureWidgetResampler` wrapper to create a `go.FigureWidget` with dynamic aggregation functionality. A major advantage of this approach is that this does not create a web application, avoiding starting an application on a port (and forwarding that port when working remotely).

Additionally, this notebook highlights how to use the `FigureWidget` its on-click callback to utilize plotly for large **time series annotation**.

## 2. Dash apps

The [dash_apps](dash_apps/) folder contains example dash apps in
which `plotly-resampler` is integrated

|                                                          | description                                                                                                                                                                                                                                                                         |
|------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **minimal examples** |                                                                                                                                                                                                                                                                                     |
| [global variable](dash_apps/01_minimal_global.py) | *bad practice*: minimal example in which a global `FigureResampler` variable is used                                                                                                                                                                                                |
| [server side caching](dash_apps/02_minimal_cache.py) | *good practice*: minimal example in which we perform server side caching of the `FigureResampler` variable                                                                                                                                                                          |
| [runtime graph construction](dash_apps/03_minimal_cache_dynamic.py) | minimal example where graphs are constructed based on user interactions at runtime. [Pattern matching callbacks](https://dash.plotly.com/pattern-matching-callbacks) are used construct these plotly-resampler graphs dynamically. Again, server side caching is performed.         |
| **advanced apps** |                                                                                                                                                                                                                                                                                     |
| [dynamic sine generator](dash_apps/11_sine_generator.py) | exponential sine generator which uses [pattern matching callbacks](https://dash.plotly.com/pattern-matching-callbacks) to remove and construct plotly-resampler graphs dynamically                                                                                                  |
| [file visualization](dash_apps/12_file_selector.py) | load and visualize multiple `.parquet` files with plotly-resampler                                                                                                                                                                                                                  |
| [dynamic static graph](dash_apps/13_coarse_fine.py) | Visualization dashboard in which a dynamic (i.e., plotly-resampler graph) and a coarse, static graph (i.e., go.Figure) are shown (made for [this issue](https://github.com/predict-idlab/plotly-resampler/issues/56)). Graph interaction events on the coarse graph update the dynamic graph. |

## 3. Other apps

The [other_apps](other_apps/) folder contains examples of `plotly-resampler` being *integrated* in other apps / frameworks

| app-name | description |
| --- | --- |
| [streamlit integration](other_apps/streamlit_app.py) | visualize a large noisy sine in a [streamlit](https://streamlit.io/) app |