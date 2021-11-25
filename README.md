<p align="center">
    <a href="#readme">
        <img width=65% alt="Plotly-Resampler logo" src="docs/_static/logo.png">
    </a>
</p>

---

`plotly_resampler` enables snappy, interactive visualiziations of large sequences of data by decorating plotly figures.

<p align="center">
    <a href="#readme">
        <img width=85% alt="plotly resamplign logo" src="docs/_static/basic_example.gif">
    </a>
</p>

#### Useful links

- [Documentation]()
- [Example notebooks]()

## Installation

| | command|
|:--------------|:--------------|
| [**pip**](https://pypi.org/project/figure_resampler/) | `pip install figure_resampler` | 
| [**conda**](https://anaconda.org/conda-forge/figure_resampler/) | `conda install -c conda-forge figure_resampler` |


## Usage

### Minimalistic example

```python
import plotly.graph_objects as go; import numpy as np
from plotly_resampler import FigureResampler

n = 1_000_000
x = np.arange(n)
noisy_sin = (3 + np.sin(x / 200) + np.random.randn(n) / 10) * x / 1_000
fig = FigureResampler(go.Figure())
fig.add_trace(go.Scattergl(name='noisy sine', showlegend=True), hf_x=x, hf_y=noisy_sin)
fig.show_dash(mode='inline')
```

### Features

* Environment-independent (Jupyter, vscode-notebooks, browser, Pycharm-notebooks)
* Convenient to use<br>
  just add the `FigureResampler` decorator around plotly go.Figure and call `show_dash()`
  * All the figure-construction fleixibility is thus maintained
* Interface for various downsampling algorithms<br>
  user can define its preffered sequence aggregation method
* `:wip:` tests


<br>
---

<p align="center">
👤 <i>Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost</i>
</p>
