<p align="center">
    <a href="#readme">
        <img width=65% alt="Plotly-Resampler logo" src="docs/_static/logo.png">
    </a>
</p>

---

`plotly_resampler` enables visualizing large sequential data by adding resampling functionality to plotly figures.

<p align="center">
    <a href="#readme">
        <img width=95% alt="example demo" src="docs/_static/basic_example.gif">
    </a>
</p>

#### Useful links

- [Documentation]()
- [Example notebooks]()

## Installation

| | command|
|:--------------|:--------------|
| [**pip**](https://pypi.org/project/plotly_resampler/) | `pip install plotly_resampler` | 
| [**conda**](https://anaconda.org/conda-forge/plotly_resampler/) | `conda install -c conda-forge plotly_resampler` |


## Usage

To **add dynamic resampling to your plotly Figure**, you should;
1. wrap the constructor of your plotly Figure with `FigureResampler`
2. call `.show_dash()` on the Figure

(OPTIONAL) add the trace data as `hf_x` and `hf_y` (for faster initial loading)

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
  * just add the `FigureResampler` decorator around a plotly Figure consructor and call `.show_dash()`
  * allows all other ploty figure construction flexibility to be used!
* **Environment-independent** 
  * can be used in Jupyter, vscode-notebooks, Pycharm-notebooks, as application (on a server)
* Interface for **various downsampling algorithms**:
  * ability to define your preffered sequence aggregation method

<br>

---

<p align="center">
👤 <i>Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost</i>
</p>
