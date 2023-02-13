import copy
import pickle
from hashlib import sha1
from inspect import isfunction

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly_resampler import FigureResampler, FigureWidgetResampler
from plotly_resampler.registering import (
    _get_plotly_constr,
    register_plotly_resampler,
    unregister_plotly_resampler,
)

from .conftest import pickle_figure, registering_cleanup

#### PICKLING

## Test basic pickling


def test_pickle_figure_resampler(pickle_figure):
    nb_traces = 3
    nb_samples = 5_007

    fig = FigureResampler(default_n_shown_samples=50, show_dash_kwargs=dict(port=8051))
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))
    assert fig._show_dash_kwargs["port"] == 8051

    pickle.dump(fig, open(pickle_figure, "wb"))
    fig_pickle = pickle.load(open(pickle_figure, "rb"))

    assert isinstance(fig_pickle, FigureResampler)
    assert fig_pickle._show_dash_kwargs["port"] == 8051
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Test for figure with subplots (check non-pickled private properties)
    fig = FigureResampler(
        make_subplots(rows=2, cols=1, shared_xaxes=True),
        default_n_shown_samples=50,
        show_dash_kwargs=dict(port=8051),
    )
    for i in range(nb_traces):
        fig.add_trace(
            go.Scattergl(name=f"trace--{i}"),
            hf_y=np.arange(nb_samples),
            row=(i % 2) + 1,
            col=1,
        )
    assert fig._global_n_shown_samples == 50
    assert fig._show_dash_kwargs["port"] == 8051
    assert fig._figure_class == go.Figure
    assert fig._xaxis_list == ["xaxis", "xaxis2"]
    assert fig._yaxis_list == ["yaxis", "yaxis2"]

    pickle.dump(fig, open(pickle_figure, "wb"))
    fig_pickle = pickle.load(open(pickle_figure, "rb"))

    assert isinstance(fig_pickle, FigureResampler)
    assert fig_pickle._global_n_shown_samples == 50
    assert fig_pickle._show_dash_kwargs["port"] == 8051
    assert fig_pickle._figure_class == go.Figure
    assert fig_pickle._xaxis_list == ["xaxis", "xaxis2"]
    assert fig_pickle._yaxis_list == ["yaxis", "yaxis2"]
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))


def test_pickle_figurewidget_resampler(pickle_figure):
    nb_traces = 2
    nb_samples = 4_123

    fig = FigureWidgetResampler(default_n_shown_samples=50)
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))

    pickle.dump(fig, open(pickle_figure, "wb"))
    fig_pickle = pickle.load(open(pickle_figure, "rb"))

    assert isinstance(fig_pickle, FigureWidgetResampler)
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Test for figure with subplots (check non-pickled private properties)
    fig = FigureWidgetResampler(
        make_subplots(rows=2, cols=1, shared_xaxes=True),
        default_n_shown_samples=50,
    )
    for i in range(nb_traces):
        fig.add_trace(
            go.Scattergl(name=f"trace--{i}"),
            hf_y=np.arange(nb_samples),
            row=(i % 2) + 1,
            col=1,
        )
    assert fig._global_n_shown_samples == 50
    assert fig._figure_class == go.FigureWidget
    assert fig._xaxis_list == ["xaxis", "xaxis2"]
    assert fig._yaxis_list == ["yaxis", "yaxis2"]

    pickle.dump(fig, open(pickle_figure, "wb"))
    fig_pickle = pickle.load(open(pickle_figure, "rb"))

    assert isinstance(fig_pickle, FigureWidgetResampler)
    assert fig_pickle._global_n_shown_samples == 50
    assert fig_pickle._figure_class == go.FigureWidget
    assert fig_pickle._xaxis_list == ["xaxis", "xaxis2"]
    assert fig_pickle._yaxis_list == ["yaxis", "yaxis2"]
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))


## Test pickling when registered


def test_pickle_figure_resampler_registered(registering_cleanup, pickle_figure):
    nb_traces = 4
    nb_samples = 5_043

    register_plotly_resampler(
        mode="figure", default_n_shown_samples=50, show_dash_kwargs=dict(port=8051)
    )

    fig = go.Figure()
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))
    assert isinstance(fig, FigureResampler)
    assert not isinstance(fig, FigureWidgetResampler)
    assert fig._show_dash_kwargs["port"] == 8051

    pickle.dump(fig, open(pickle_figure, "wb"))

    # Loading with PR registered
    assert isinstance(go.Figure(), FigureResampler)
    fig_pickle = pickle.load(open(pickle_figure, "rb"))
    assert isinstance(fig_pickle, FigureResampler)
    assert fig_pickle._show_dash_kwargs["port"] == 8051
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Loading with PR registered as FigureWidgetResampler (& other nb default samples)
    register_plotly_resampler(mode="widget", default_n_shown_samples=75)
    assert isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    fig_pickle = pickle.load(open(pickle_figure, "rb"))
    assert isinstance(fig_pickle, FigureResampler)
    assert fig_pickle._show_dash_kwargs["port"] == 8051
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Loading with PR NOT registered
    unregister_plotly_resampler()
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    fig_pickle = pickle.load(open(pickle_figure, "rb"))
    assert isinstance(fig_pickle, FigureResampler)
    assert fig_pickle._show_dash_kwargs["port"] == 8051
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Pickling and loading with PR NOT registered
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    pickle.dump(fig, open(pickle_figure, "wb"))
    fig_pickle = pickle.load(open(pickle_figure, "rb"))
    assert isinstance(fig_pickle, FigureResampler)
    assert fig_pickle._show_dash_kwargs["port"] == 8051
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))


def test_pickle_figurewidget_resampler_registered(registering_cleanup, pickle_figure):
    nb_traces = 3
    nb_samples = 3_643

    register_plotly_resampler(mode="widget", default_n_shown_samples=50)

    fig = go.Figure()
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))
    assert isinstance(fig, FigureWidgetResampler)
    assert not isinstance(fig, FigureResampler)

    pickle.dump(fig, open(pickle_figure, "wb"))

    # Loading with PR registered
    assert isinstance(go.Figure(), FigureWidgetResampler)
    fig_pickle = pickle.load(open(pickle_figure, "rb"))
    assert isinstance(fig_pickle, FigureWidgetResampler)
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Loading with PR registered as FigureResampler (& other nb default samples)
    register_plotly_resampler(mode="figure", default_n_shown_samples=75)
    assert isinstance(go.Figure(), FigureResampler)
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    fig_pickle = pickle.load(open(pickle_figure, "rb"))
    assert isinstance(fig_pickle, FigureWidgetResampler)
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Loading with PR NOT registered
    unregister_plotly_resampler()
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    fig_pickle = pickle.load(open(pickle_figure, "rb"))
    assert isinstance(fig_pickle, FigureWidgetResampler)
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Pickling and loading with PR NOT registered
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    pickle.dump(fig, open(pickle_figure, "wb"))
    fig_pickle = pickle.load(open(pickle_figure, "rb"))
    assert isinstance(fig_pickle, FigureWidgetResampler)
    assert len(fig_pickle.data) == nb_traces
    assert len(fig_pickle.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_pickle.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_pickle.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))


#### (DEEP)COPY

## Test basic (deep)copy


def test_copy_and_deepcopy_figure_resampler():
    nb_traces = 3
    nb_samples = 3_243

    fig = FigureResampler(default_n_shown_samples=50, show_dash_kwargs=dict(port=8051))
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))
    assert fig._show_dash_kwargs["port"] == 8051

    fig_copy = copy.copy(fig)

    assert isinstance(fig_copy, FigureResampler)
    assert fig_copy._show_dash_kwargs["port"] == 8051
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    fig_copy = copy.deepcopy(fig)

    assert isinstance(fig_copy, FigureResampler)
    assert fig_copy._show_dash_kwargs["port"] == 8051
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))


def test_copy_and_deepcopy_figurewidget_resampler():
    nb_traces = 3
    nb_samples = 3_243

    fig = FigureWidgetResampler(default_n_shown_samples=50)
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))

    fig_copy = copy.copy(fig)

    assert isinstance(fig_copy, FigureWidgetResampler)
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    fig_copy = copy.deepcopy(fig)

    assert isinstance(fig_copy, FigureWidgetResampler)
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))


## Test basic (deep)copy with PR registered


def test_copy_figure_resampler_registered():
    nb_traces = 3
    nb_samples = 4_069

    register_plotly_resampler(
        mode="figure", default_n_shown_samples=50, show_dash_kwargs=dict(port=8051)
    )

    fig = go.Figure()
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))
    assert isinstance(fig, FigureResampler)
    assert not isinstance(fig, FigureWidgetResampler)
    assert fig._show_dash_kwargs["port"] == 8051

    # Copy with PR registered
    fig_copy = copy.copy(fig)
    assert isinstance(go.Figure(), FigureResampler)
    assert isinstance(fig_copy, FigureResampler)
    assert fig_copy._show_dash_kwargs["port"] == 8051
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Copy with PR registered as FigureWidgetResampler (& other nb default samples)
    register_plotly_resampler(mode="widget", default_n_shown_samples=75)
    assert isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    fig_copy = copy.copy(fig)
    assert isinstance(fig_copy, FigureResampler)
    assert fig_copy._show_dash_kwargs["port"] == 8051
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Copy with PR NOT registered
    unregister_plotly_resampler()
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    fig_copy = copy.copy(fig)
    assert isinstance(fig_copy, FigureResampler)
    assert fig_copy._show_dash_kwargs["port"] == 8051
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))


def test_deepcopy_figure_resampler_registered():
    nb_traces = 4
    nb_samples = 3_169

    register_plotly_resampler(
        mode="figure", default_n_shown_samples=50, show_dash_kwargs=dict(port=8051)
    )

    fig = go.Figure()
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))
    assert isinstance(fig, FigureResampler)
    assert not isinstance(fig, FigureWidgetResampler)
    assert fig._show_dash_kwargs["port"] == 8051

    # Copy with PR registered
    fig_copy = copy.deepcopy(fig)
    assert isinstance(go.Figure(), FigureResampler)
    assert isinstance(fig_copy, FigureResampler)
    assert fig_copy._show_dash_kwargs["port"] == 8051
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Copy with PR registered as FigureWidgetResampler (& other nb default samples)
    register_plotly_resampler(mode="widget", default_n_shown_samples=75)
    assert isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    fig_copy = copy.deepcopy(fig)
    assert isinstance(fig_copy, FigureResampler)
    assert fig_copy._show_dash_kwargs["port"] == 8051
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Copy with PR NOT registered
    unregister_plotly_resampler()
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    fig_copy = copy.deepcopy(fig)
    assert isinstance(fig_copy, FigureResampler)
    assert fig_copy._show_dash_kwargs["port"] == 8051
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))


def test_copy_figurewidget_resampler_registered():
    nb_traces = 5
    nb_samples = 3_012

    register_plotly_resampler(mode="widget", default_n_shown_samples=50)

    fig = go.Figure()
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))
    assert isinstance(fig, FigureWidgetResampler)
    assert not isinstance(fig, FigureResampler)

    # Copy with PR registered
    fig_copy = copy.copy(fig)
    assert isinstance(go.Figure(), FigureWidgetResampler)
    assert isinstance(fig_copy, FigureWidgetResampler)
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Copy with PR registered as FigureResampler (& other nb default samples)
    register_plotly_resampler(mode="figure", default_n_shown_samples=75)
    assert isinstance(go.Figure(), FigureResampler)
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    fig_copy = copy.copy(fig)
    assert isinstance(fig_copy, FigureWidgetResampler)
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Copy with PR NOT registered
    unregister_plotly_resampler()
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    fig_copy = copy.copy(fig)
    assert isinstance(fig_copy, FigureWidgetResampler)
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))


def test_deepcopy_figurewidget_resampler_registered():
    nb_traces = 5
    nb_samples = 3_012

    register_plotly_resampler(mode="widget", default_n_shown_samples=50)

    fig = go.Figure()
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))
    assert isinstance(fig, FigureWidgetResampler)
    assert not isinstance(fig, FigureResampler)

    # Copy with PR registered
    fig_copy = copy.deepcopy(fig)
    assert isinstance(go.Figure(), FigureWidgetResampler)
    assert isinstance(fig_copy, FigureWidgetResampler)
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Copy with PR registered as FigureResampler (& other nb default samples)
    register_plotly_resampler(mode="figure", default_n_shown_samples=75)
    assert isinstance(go.Figure(), FigureResampler)
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    fig_copy = copy.deepcopy(fig)
    assert isinstance(fig_copy, FigureWidgetResampler)
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))

    # Copy with PR NOT registered
    unregister_plotly_resampler()
    assert not isinstance(go.Figure(), FigureWidgetResampler)
    assert not isinstance(go.Figure(), FigureResampler)
    fig_copy = copy.deepcopy(fig)
    assert isinstance(fig_copy, FigureWidgetResampler)
    assert len(fig_copy.data) == nb_traces
    assert len(fig_copy.hf_data) == nb_traces
    for i in range(nb_traces):
        trace = fig_copy.data[i]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.y) == 50
        assert f"trace--{i}" in trace.name
        hf_trace = fig_copy.hf_data[i]
        assert len(hf_trace["y"]) == nb_samples
        assert np.all(hf_trace["y"] == np.arange(nb_samples))
