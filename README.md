<p align="center">
    <a href="#readme">
        <img width=65% alt="Plotly-Resampler logo" src="docs/_static/logo.png">
    </a>
</p>

---

`plotly_resampler` enables interactive visualizations of large sequential data by decorating plotly figures.

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

To add dynamic resampling to your plotly Figure, you should;
- Wrap the constructor of your plotly Figure with `FigureResampler`
- [OPTIONAL] Add the trace data as `hf_x` and `hf_y` (for faster initial loading)
- Call `.show_dash()` on the Figure

### Minimalistic example

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
