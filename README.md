<p align="center">
    <a href="#readme">
        <img alt="Plotly-Resampler logo" src="https://raw.githubusercontent.com/predict-idlab/plotly-resampler/main/docs/sphinx/_static/logo.svg" width=65%>
    </a>
</p>

[![PyPI Latest Release](https://img.shields.io/pypi/v/plotly-resampler.svg)](https://pypi.org/project/plotly-resampler/)
[![support-version](https://img.shields.io/pypi/pyversions/plotly-resampler)](https://img.shields.io/pypi/pyversions/plotly-resampler)
[![codecov](https://img.shields.io/codecov/c/github/predict-idlab/plotly-resampler?logo=codecov)](https://codecov.io/gh/predict-idlab/plotly-resampler)
[![Code quality](https://img.shields.io/lgtm/grade/python/github/predict-idlab/plotly-resampler?label=code%20quality&logo=lgtm)](https://lgtm.com/projects/g/predict-idlab/plotly-resampler/context:python)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?)](http://makeapullrequest.com)
[![Documentation](https://github.com/predict-idlab/plotly-resampler/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/predict-idlab/plotly-resampler/actions/workflows/deploy-docs.yml)
[![Testing](https://github.com/predict-idlab/plotly-resampler/actions/workflows/test.yml/badge.svg)](https://github.com/predict-idlab/plotly-resampler/actions/workflows/test.yml)


<!-- [![Downloads](https://pepy.tech/badge/plotly-resampler)](https://pepy.tech/project/plotly-resampler) -->

> `plotly_resampler`: visualize large sequential data by **adding resampling functionality to Plotly figures**

[Plotly](https://github.com/plotly/plotly.py) is an awesome interactive visualization library, however it can get pretty slow when a lot of data points are visualized (100 000+ datapoints). This library solves this by downsampling the data respective to the view and then plotting the downsampled points. When you interact with the plot (panning, zooming, ...), [dash](https://github.com/plotly/dash) callbacks are used to resample and redraw the figures. 

<p align="center">
    <a href="#readme">
        <img alt="example demo" src="https://github.com/predict-idlab/plotly-resampler/blob/main/docs/sphinx/_static/basic_example.gif" width=95%>
    </a>
</p>

In [this Plotly-Resampler demo](https://github.com/predict-idlab/plotly-resampler/blob/main/examples/basic_example.ipynb) over `110,000,000` data points are visualized! 

<!-- #### Useful links -->

<!-- - [Documentation]() work in progress 🚧  -->
<!-- - [Example notebooks](https://github.com/predict-idlab/plotly-resampler/tree/main/examples/) -->

### Installation

| [**pip**](https://pypi.org/project/plotly_resampler/) | `pip install plotly-resampler` | 
| ---| ----|
<!-- | [**conda**](https://anaconda.org/conda-forge/plotly_resampler/) | `conda install -c conda-forge plotly_resampler` | -->


## Usage

To **add dynamic resampling to your plotly Figure**, you should;
1. wrap the plotly Figure with `FigureResampler`
2. call `.show_dash()` on the Figure

> **Note**:  
> Any plotly Figure can be wrapped with FigureResampler! 🎉  
> But, (obviously) only the scatter traces will be resampled.

> **Tip** 💡:  
> For significant faster initial loading of the Figure, we advise to wrap the constructor of the plotly Figure with `FigureResampler` and add the trace data as `hf_x` and `hf_y`

### Minimal example

```python
import plotly.graph_objects as go; import numpy as np
from plotly_resampler import FigureResampler

x = np.arange(1_000_000)
noisy_sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

fig = FigureResampler(go.Figure())
fig.add_trace(go.Scattergl(name='noisy sine', showlegend=True), hf_x=x, hf_y=noisy_sin)

fig.show_dash(mode='inline')
```

### Features

* **Convenient** to use:
  * just add the `FigureResampler` decorator around a plotly Figure and call `.show_dash()`
  * allows all other plotly figure construction flexibility to be used!
* **Environment-independent** 
  * can be used in Jupyter, vscode-notebooks, Pycharm-notebooks, Google Colab, and even as application (on a server)
* Interface for **various downsampling algorithms**:
  * ability to define your preferred sequence aggregation method


### Important considerations & tips

* When running the code on a server, you should forward the port of the `FigureResampler.show_dash()` method to your local machine.
* In general, when using downsampling one should be aware of (possible) [aliasing](https://en.wikipedia.org/wiki/Aliasing) effects.  
  The <b><a style="color:orange">[R]</a></b> in the legend indicates when the corresponding trace is being resampled (and thus possibly distorted) or not.

## Future work 🔨

* Support `.add_traces()` (currently only `.add_trace` is supported)

<br>

---

<p align="center">
👤 <i>Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost</i>
</p>
