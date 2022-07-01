import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pickle

from plotly_resampler import FigureResampler, FigureWidgetResampler
from plotly_resampler.registering import (
    register_plotly_resampler,
    unregister_plotly_resampler,
    _get_plotly_constr,
)

from .conftest import registering_cleanup, pickle_figure
from inspect import isfunction


## Test basic pickling

def test_pickle_figure_resampler(pickle_figure):
    nb_traces = 3
    nb_samples = 5_007

    fig = FigureResampler(default_n_shown_samples=50)
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))

    pickle.dump(fig, open(pickle_figure, "wb"))
    fig_pickle = pickle.load(open(pickle_figure, "rb"))

    assert isinstance(fig_pickle, FigureResampler)
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


## Test pickling when registered

def test_pickle_figure_resampler_registered(registering_cleanup, pickle_figure):
    nb_traces = 4
    nb_samples = 5_043

    register_plotly_resampler(mode="figure", default_n_shown_samples=50)
    
    fig = go.Figure()
    for i in range(nb_traces):
        fig.add_trace(go.Scattergl(name=f"trace--{i}"), hf_y=np.arange(nb_samples))
    assert isinstance(fig, FigureResampler)
    assert not isinstance(fig, FigureWidgetResampler)

    pickle.dump(fig, open(pickle_figure, "wb"))

    # Loading with PR registered
    assert isinstance(go.Figure(), FigureResampler)
    fig_pickle = pickle.load(open(pickle_figure, "rb"))
    assert isinstance(fig_pickle, FigureResampler)
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
