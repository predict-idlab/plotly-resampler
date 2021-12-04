This is the documentation of [**plotly-resampler**]([htt](https://github.com/predict-idlab/plotly-resampler)); a plotly wrapper for Figures to visualize large time-series data.


<link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.2/css/all.min.css" integrity="sha512-HK5fgLBL+xu6dm/Ii3z4xhlSUyZgTT9tuc/hSrtw6uzJOvgRr2a9jyxxT1ely+B+xFAmJKVSTbpM/CuL7qxO8w==" crossorigin>

<div class="container" style="text-align: center">
        <h3><strong>Installation</strong></h3><br>
        <a title="plotly_resampler on PyPI" href="https://pypi.org/project/plotly_resampler/" style="margin-right:.8em; background-color: #48c774; border-color: transparent; color: #fff; padding: 0.75rem; border-radius: 4px;"
                   itemprop="downloadUrl" data-ga-event-category="PyPI">
                    <span class="icon"><i class="fa fa-download"></i></span>
                    <span><b>PyPI</b></span>
                </a> &nbsp;
                <a title="plotly_resampler on GitHub" href="https://github.com/predict-idlab/plotly-resampler" style="color: #4a4a4a; background-color: #f5f5f5 !important; font-size: 1em; font-weight: 400; line-height: 1.5; border-radius: 4px; padding: 0.75rem; "
                   data-ga-event-category="GitHub">
                    <span class="icon"><i class="fab fa-github"></i></span>
                    <span><b>GitHub</b></span>
                </a>
</div>
<br>
<hr style="height: 1px; border: none; border-top: 1px solid darkgrey;">

<!-- <div style="text-align: center"> -->
<h3><b><a href="#header-submodules">Jump to API reference</a></b></h3>
<!-- </div> -->

<p align="center">
    <a href="#readme">
        <img width=100% alt="example demo" src="https://raw.githubusercontent.com/predict-idlab/plotly-resampler/main/docs/_static/basic_example.gif">
    </a>
</p>

## Getting started 🚀

_plotly-resampler_ maintains its interactiveness on large data by applying front-end **resampling**.


Users can interact with 2 components:

* `FigureResampler`: a wrapper for _plotly.graph\_ojbects_  which serves the adaptive resampling functionality.
* `downsamplers`: this module withholds various downsampling methods.

### Working example ✅

```python
import plotly.graph_objects as go; import numpy as np
from plotly_resampler import FigureResampler

x = np.arange(1_000_000)
sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

fig = FigureResampler(go.Figure())
fig.add_trace(go.Scattergl(name='noisy sine', showlegend=True), hf_x=x, hf_y=sin)

fig.show_dash(mode='inline')
```