

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