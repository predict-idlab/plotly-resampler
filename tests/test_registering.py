import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from plotly_resampler import FigureResampler, FigureWidgetResampler
from plotly_resampler.figure_resampler.figure_resampler_interface import (
    AbstractFigureAggregator,
)
from plotly_resampler.registering import (
    register_plotly_resampler,
    unregister_plotly_resampler,
    _get_plotly_constr,
)

from .conftest import registering_cleanup
from inspect import isfunction


def test_get_plotly_const(registering_cleanup):
    # Check the basi(c)s
    assert issubclass(FigureResampler, AbstractFigureAggregator)
    assert issubclass(FigureWidgetResampler, AbstractFigureAggregator)

    # Is unregistered now
    assert not (isfunction(go.Figure) or isfunction(go.FigureWidget))
    assert not issubclass(go.Figure, AbstractFigureAggregator)
    assert not issubclass(go.FigureWidget, AbstractFigureAggregator)
    assert not issubclass(_get_plotly_constr(go.Figure), AbstractFigureAggregator)
    assert not issubclass(_get_plotly_constr(go.FigureWidget), AbstractFigureAggregator)

    register_plotly_resampler()
    assert isfunction(go.Figure) and isfunction(go.FigureWidget)
    assert isinstance(go.Figure(), AbstractFigureAggregator)
    assert isinstance(go.FigureWidget(), AbstractFigureAggregator)
    assert issubclass(FigureResampler, AbstractFigureAggregator)
    assert issubclass(FigureWidgetResampler, AbstractFigureAggregator)
    assert not issubclass(_get_plotly_constr(go.Figure), AbstractFigureAggregator)
    assert not issubclass(_get_plotly_constr(go.FigureWidget), AbstractFigureAggregator)

    unregister_plotly_resampler()
    assert not (isfunction(go.Figure) or isfunction(go.FigureWidget))
    assert not issubclass(go.Figure, AbstractFigureAggregator)
    assert not issubclass(go.FigureWidget, AbstractFigureAggregator)
    assert not issubclass(_get_plotly_constr(go.Figure), AbstractFigureAggregator)
    assert not issubclass(_get_plotly_constr(go.FigureWidget), AbstractFigureAggregator)


def test_register_and_unregister_graph_objects(registering_cleanup):
    import plotly.graph_objects as go_

    # Is unregistered now
    assert not (isfunction(go_.Figure) or isfunction(go_.FigureWidget))
    fig = go_.Figure()
    assert not isinstance(fig, AbstractFigureAggregator)
    fig = go_.FigureWidget()
    assert not isinstance(fig, AbstractFigureAggregator)

    register_plotly_resampler()
    assert isfunction(go_.Figure) and isfunction(go_.FigureWidget)
    fig = go_.Figure()
    assert isinstance(fig, AbstractFigureAggregator)
    assert isinstance(fig, FigureResampler)
    assert not isinstance(fig, FigureWidgetResampler)
    fig = go_.FigureWidget()
    assert isinstance(fig, AbstractFigureAggregator)
    assert isinstance(fig, FigureWidgetResampler)
    assert not isinstance(fig, FigureResampler)

    unregister_plotly_resampler()
    assert not (isfunction(go_.Figure) or isfunction(go_.FigureWidget))
    fig = go_.Figure()
    assert not isinstance(fig, AbstractFigureAggregator)
    fig = go_.FigureWidget()
    assert not isinstance(fig, AbstractFigureAggregator)


def test_register_and_unregister_graph_objs(registering_cleanup):
    import plotly.graph_objs as go_

    # Is unregistered now
    assert not (isfunction(go_.Figure) or isfunction(go_.FigureWidget))
    fig = go_.Figure()
    assert not isinstance(fig, AbstractFigureAggregator)
    fig = go_.FigureWidget()
    assert not isinstance(fig, AbstractFigureAggregator)

    register_plotly_resampler()
    assert isfunction(go_.Figure) and isfunction(go_.FigureWidget)
    fig = go_.Figure()
    assert isinstance(fig, AbstractFigureAggregator)
    assert isinstance(fig, FigureResampler)
    assert not isinstance(fig, FigureWidgetResampler)
    fig = go_.FigureWidget()
    assert isinstance(fig, AbstractFigureAggregator)
    assert isinstance(fig, FigureWidgetResampler)
    assert not isinstance(fig, FigureResampler)

    unregister_plotly_resampler()
    assert not (isfunction(go_.Figure) or isfunction(go_.FigureWidget))
    fig = go_.Figure()
    assert not isinstance(fig, AbstractFigureAggregator)
    fig = go_.FigureWidget()
    assert not isinstance(fig, AbstractFigureAggregator)


def test_registering_modes(registering_cleanup):
    register_plotly_resampler(mode="auto")
    # Should be default
    assert isinstance(go.Figure(), FigureResampler)
    assert isinstance(go.FigureWidget(), FigureWidgetResampler)

    register_plotly_resampler(mode="figure")
    # Should be all FigureResampler
    assert isinstance(go.Figure(), FigureResampler)
    assert isinstance(go.FigureWidget(), FigureResampler)

    register_plotly_resampler(mode="widget")
    # Should be all FigureWidgetResampler
    assert isinstance(go.Figure(), FigureWidgetResampler)
    assert isinstance(go.FigureWidget(), FigureWidgetResampler)


def test_registering_plotly_express_and_kwargs(registering_cleanup):
    # Is unregistered now
    fig = px.scatter(y=np.arange(500))
    assert not isinstance(fig, AbstractFigureAggregator)
    assert len(fig.data) == 1
    assert len(fig.data[0].y) == 500

    register_plotly_resampler(
        default_n_shown_samples=50, show_dash_kwargs=dict(mode="inline", port=8051)
    )
    fig = px.scatter(y=np.arange(500))
    assert isinstance(fig, FigureResampler)
    assert fig._show_dash_kwargs == dict(mode="inline", port=8051)
    assert len(fig.data) == 1
    assert len(fig.data[0].y) == 50
    assert len(fig.hf_data) == 1
    assert len(fig.hf_data[0]["y"]) == 500

    register_plotly_resampler()
    fig = px.scatter(y=np.arange(5000))
    assert isinstance(fig, FigureResampler)
    assert fig._show_dash_kwargs == dict()
    assert len(fig.data) == 1
    assert len(fig.data[0].y) == 1000
    assert len(fig.hf_data) == 1
    assert len(fig.hf_data[0]["y"]) == 5000

    unregister_plotly_resampler()
    fig = px.scatter(y=np.arange(500))
    assert not isinstance(fig, AbstractFigureAggregator)
    assert len(fig.data) == 1
    assert len(fig.data[0].y) == 500


def test_compasibility_when_registered(registering_cleanup):
    fr = FigureResampler
    fwr = FigureWidgetResampler

    fig_orig_1 = px.scatter(y=np.arange(1_005))
    fig_orig_2 = go.FigureWidget({"type": "scatter", "y": np.arange(1_005)})
    for fig in [fig_orig_1, fig_orig_2]:
        fig1 = fr(fig)
        fig2 = fr(fwr(fig))
        fig3 = fr(fr(fr(fr(fwr(fwr(fr(fwr(fr(fig)))))))))
        for f in [fig1, fig2, fig3]:
            assert isinstance(f, FigureResampler)
            assert len(f.data) == 1
            assert len(f.data[0].y) == 1000
            assert len(f.hf_data) == 1
            assert len(f.hf_data[0]["y"]) == 1005

        fig1 = fwr(fig)
        fig2 = fwr(fr(fig))
        fig3 = fwr(fwr(fwr(fwr(fr(fr(fwr(fr(fwr(fig)))))))))
        for f in [fig1, fig2, fig3]:
            assert isinstance(f, FigureWidgetResampler)
            assert len(f.data) == 1
            assert len(f.data[0].y) == 1000
            assert len(f.hf_data) == 1
            assert len(f.hf_data[0]["y"]) == 1005

    register_plotly_resampler()

    fig_orig_1 = px.scatter(y=np.arange(1_005))
    fig_orig_2 = go.FigureWidget({"type": "scatter", "y": np.arange(1_005)})
    for fig in [fig_orig_1, fig_orig_2]:
        fig1 = fr(fig)
        fig2 = fr(fwr(fig))
        fig3 = fr(fr(fr(fr(fwr(fwr(fr(fwr(fr(fig)))))))))
        for f in [fig1, fig2, fig3]:
            assert isinstance(f, FigureResampler)
            assert len(f.data) == 1
            assert len(f.data[0].y) == 1000
            assert len(f.hf_data) == 1
            assert len(f.hf_data[0]["y"]) == 1005

        fig1 = fwr(fig)
        fig2 = fwr(fr(fig))
        fig3 = fwr(fwr(fwr(fwr(fr(fr(fwr(fr(fwr(fig)))))))))
        for f in [fig1, fig2, fig3]:
            assert isinstance(f, FigureWidgetResampler)
            assert len(f.data) == 1
            assert len(f.data[0].y) == 1000
            assert len(f.hf_data) == 1
            assert len(f.hf_data[0]["y"]) == 1005
