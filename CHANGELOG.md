
# v0.9.1
## Major changes:
Support for multiple axes. 

The `.GIF` below demonstrates how multiple axes on a subplots can be used to enhance the number of visible traces, without using more (vertical) screen space 🔥!

Make sure to take a look at our [examples](https://github.com/predict-idlab/plotly-resampler/blob/main/examples/other_examples.ipynb)

![Peek 2023-07-13 10-24](https://github.com/predict-idlab/plotly-resampler/assets/38005924/aa5d278f-7baf-4251-91b5-7445eb7d53d0)

## What's Changed (generated)
* :fire: multiple y-axes support by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/244
 

 # v0.9.0
## Major changes:
### Even faster aggregation 🐎
We switched our aggregation backend to [tsdownsample](https://github.com/predict-idlab/tsdownsample), which alleviates the need to compile our C code on non-supported devices, and has parallelization capabilities. 
`tsdownsample` leverages the [argminmax](https://github.com/jvdd/argminmax) crate, which has SIMD-optimized instruction to find vertical extrema really fast! 

With parallelization enabled, you should clearly see a bump in perfomance when visualizing (multiple) large traces! 🐎

### Versioned docs! :party:
We restyled our [documentation](https://predict-idlab.github.io/plotly-resampler/latest) and added versioning! 🎉

https://predict-idlab.github.io/plotly-resampler/latest/

Go check it out! :point_up:

### Other Features
- Support for **log-scale** axes (and thus log-bin-based aggregators) - check [this pull-request](https://github.com/predict-idlab/plotly-resampler/pull/207)
![](https://cdn.discordapp.com/attachments/372491075153166338/1129004610472906782/image.png)

> The above image shows how the `log` aggregator (row2) will use log-scale bins. This can be seen in the 1-1000 range when comparing both subplots. <br>*Note: the shown data has a fixed delta-x of 1. Hence, here are no exact equally spaced bins for the left part of the LogLTTB.*

- Add a fill-value option to gap handlers
![](https://cdn.discordapp.com/attachments/372491075153166338/1129004638016897045/image.png)

> The above image shows how the `fill_value` option can be used to fill gaps with a specific value.<br> This can be of greate use, when you use the `fill='tozeroy'` option in plotly **and gaps occur in your data**, as this will, combined with `line_shape='vh'`, fill the area between the trace and the x-axis and gaps will be a flat zero-line.
### Bugfixes
- support for pandas2.0 intricacies

## What's Changed (generated)
* fix: handle bool dtype for x in LTTB_core_py by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/183
* fix: add colors to streamlit example :art: by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/187
* docs: describe solution in FAQ for slow datetime arrays by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/184
* Rework aggregator interface  by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/186
* :rocket: integrate with tsdownsample by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/191
* refactor: use composition for gap handling by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/199
* ✨ np.array interface implementation by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/154
* 🧹  fix typo in docstring + remove LTTB from MinMaxLTTB + remove interleave_gaps by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/201
* chore: use ruff instead of isort by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/200
* 🌈 adding marker props by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/148
* Datetime bugfix by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/209
* Fixes #210 by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/211
* Log support by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/207
* Datetime range by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/213
* :sparkles: add fill_value option to gap handlers by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/218
* :sparkles: fix `limit_to_view=True` but no gaps inserted bug by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/220
* :bug: convert trace props to array + check for nan removal by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/225
* Figurewidget datetime bug by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/232
* ♻️ deprecate JupyterDash in favor for updated Dash version by @NielsPraet in https://github.com/predict-idlab/plotly-resampler/pull/233
* :eyes: comment out reset layout by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/228
* Docs/versioned docs (#236) by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/237

# v 0.8.0

## Major changes

### Faster aggregation 🐎 
the `lttbc` dependency is removed; and we added our own (faster) lttb C implementation. Additionally we provide a Python fallback when this lttb-C building fails. In the near future, we will look into CIBuildWheels to build the wheels for the major OS & Python matrix versions.  
A well deserved s/o to [dgoeris/lttbc](https://github.com/dgoeries/lttbc), who heavily inspired our implementation!

### Figure Output serialization 📸 
Plotly-resampler now also has the option to store the output figure as an Image in notebook output. As long the notebook is connected, the interactive plotly-resampler figure is shown; but once the figure / notebook isn't connected anymore, a static image will be rendered in the notebook output.

## What's Changed (generated)
* :bug: return self when calling add_traces by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/75
* :fire: add streamlit integration example by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/80
* ✨ adding `convert_traces_kwargs` by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/81
* Fix numeric `hf_y` input as dtype object by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/90
* :fire: add support for figure dict input + propagate _grid_str by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/92
* :pray: fix tests for all OS by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/95
* Add python3dot10 by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/96
* :sunrise: FigureResampler display improvements by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/97
* :package: serialization support +  :level_slider: update OS & python version in test-matrix by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/87
* Lttbv2 🍒 ⛏️  branch by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/103
* :robot: hack together output retention in notebooks by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/105
* :package: improve docs by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/104

& some other minor bug fixes :see_no_evil: 

**Full Changelog**: https://github.com/predict-idlab/plotly-resampler/compare/v0.7.0...v0.8.0

---

# V0.7.0

## What's Changed

**You can register plotly_resampler**; this adds dynamic resampling functionality *under the hood* to plotly.py! 🥳
As a result, you can stop wrapping plotly figures with a plotly-resampler decorator (as this all happens automatically) 
> You only need to call the `register_plotly_resampler` method and all plotly figures will be wrapped (under the hood) according to that method's configuration.

-> More info in the [README](https://github.com/predict-idlab/plotly-resampler#usage) and [docs](https://predict-idlab.github.io/plotly-resampler/getting_started.html#how-to-use)!

Aditionally, all resampler Figures are now composable; implying that they can be decorated by themselves and all other types of plotly-(resampler) figures. This eases the switching from a FigureResampler to FigureWidgetResampler and vice-versa.


## What's Changed (PR's)
* 🦌 Adding reset-axes functionality by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/48
* 🐛 Small bugfixes by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/52
* 🔍  investigating gap-detection methodology by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/53
* :mag: fix float index problem of #63 by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/64
* :wrench: hotfix for rounding error by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/66
* 🗳️ Compose figs by @jonasvdd in https://github.com/predict-idlab/plotly-resampler/pull/72
* :sparkles: register plotly-resampler by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/70
* :robot: update dependencies + new release by @jvdd in https://github.com/predict-idlab/plotly-resampler/pull/74


**Full Changelog**: https://github.com/predict-idlab/plotly-resampler/compare/v0.6.0...v0.7.0