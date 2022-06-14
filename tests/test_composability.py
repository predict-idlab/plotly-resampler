import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler, FigureWidgetResampler


# ----------------------- Figure as Base -----------------------
if True:
    # -------- All scatters
    def test_fr_f_scatter_agg(float_series, bool_series, cat_series):
        base_fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureResampler object from a go.Figure
        # 1. All scatters are aggregated
        fr_f = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_f.data) == 3
        assert len(fr_f.hf_data) == 3
        for trace in fr_f.data:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_f._hf_data
            assert len(trace["y"]) == 2_000

        # 2. No scatters are aggregated
        fr_f = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fr_f.data) == 3
        assert len(fr_f.hf_data) == 0
        for trace in fr_f.data:
            assert trace.uid not in fr_f._hf_data
            assert len(trace["y"]) == 10_000

    def test_fwr_f_scatter_agg(float_series, bool_series, cat_series):
        base_fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
        )
        base_fig.add_trace(go.Scatter(y=cat_series), row=1, col=1)
        base_fig.add_trace(dict(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureWidgetResampler object from a go.Figure
        # 1. All scatters are aggregated
        fwr_f = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fwr_f.data) == 3
        assert len(fwr_f.hf_data) == 3
        for trace in fwr_f.data:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fwr_f._hf_data
            assert len(trace["y"]) == 2_000

        # 2. No scatters are aggregated
        fwr_f = FigureWidgetResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fwr_f.data) == 3
        assert len(fwr_f.hf_data) == 0
        for trace in fwr_f.data:
            assert trace.uid not in fwr_f._hf_data
            assert len(trace["y"]) == 10_000

    # ---- Must not be aggregated
    def test_fr_f_scatter_not_all_agg(float_series, bool_series, cat_series):
        base_fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series[:1500]), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series[:800]), row=2, col=1)

        # Create FigureResampler object from a go.Figure
        fr_f = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_f.data) == 3
        assert len(fr_f.hf_data) == 1
        # Only the fist trace will be aggregated
        for trace in fr_f.data[:1]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_f._hf_data
            assert len(trace["y"]) == 2_000

        for trace in fr_f.data[1:]:
            assert trace.uid not in fr_f._hf_data
            assert len(trace["y"]) != 2_000

    def test_fwr_f_scatter_not_all_agg(float_series, bool_series, cat_series):
        base_fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
        )
        base_fig.add_trace(go.Scatter(y=cat_series), row=1, col=1)
        base_fig.add_trace(dict(y=bool_series[:1500]), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series[:800]), row=2, col=1)

        # Create FigureResampler object from a go.Figure
        fwr_f = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fwr_f.data) == 3
        assert len(fwr_f.hf_data) == 1
        # Only the fist trace will be aggregated
        for trace in fwr_f.data[:1]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fwr_f._hf_data
            assert len(trace["y"]) == 2_000

        for trace in fwr_f.data[1:]:
            assert trace.uid not in fwr_f._hf_data
            assert len(trace["y"]) != 2_000

    # ------- Mixed
    def test_fr_f_mixed_agg(float_series):
        base_fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fr_f = FigureResampler(base_fig, default_n_shown_samples=1_000)
        assert len(fr_f.data) == 3
        assert len(fr_f.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fr_f.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_f._hf_data
            assert len(trace["y"]) == 1_000

        for trace in fr_f.data[:1] + fr_f.data[2:]:
            assert trace.uid not in fr_f._hf_data
            assert trace.y is None  # these traces don't even have a y value

    def test_fwr_f_mixed_agg(float_series):
        base_fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fwr_f = FigureWidgetResampler(base_fig, default_n_shown_samples=1_000)
        assert len(fwr_f.data) == 3
        assert len(fwr_f.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fwr_f.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fwr_f._hf_data
            assert len(trace["y"]) == 1_000

        for trace in fwr_f.data[:1] + fwr_f.data[2:]:
            assert trace.uid not in fwr_f._hf_data
            assert trace.y is None  # these traces don't even have a y value

    # ---- Must not (all) be aggregated
    def test_fr_f_mixed_no_agg(float_series):
        base_fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fr_f = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fr_f.data) == 3
        assert len(fr_f.hf_data) == 0
        assert len(fr_f.data[1]["y"]) == 10_000

        fwr_f = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fwr_f.data) == 3
        assert len(fwr_f.hf_data) == 0
        assert len(fwr_f.data[1]["y"]) == 10_000


# ----------------------- FigureWidget as Base -----------------------
if True:
    # -------- All scatters
    def test_fr_fw_scatter_agg(float_series, bool_series, cat_series):
        base_fig = go.FigureWidget(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            )
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureResampler object from a go.Figure
        # 1. All scatters are aggregated
        fr_fw = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fw.data) == 3
        assert len(fr_fw.hf_data) == 3
        for trace in fr_fw.data:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fw._hf_data
            assert len(trace["y"]) == 2_000

        # 2. No scatters are aggregated
        fr_fw = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fr_fw.data) == 3
        assert len(fr_fw.hf_data) == 0
        for trace in fr_fw.data:
            assert trace.uid not in fr_fw._hf_data
            assert len(trace["y"]) == 10_000

    def test_fwr_fw_scatter_agg(float_series, bool_series, cat_series):
        base_fig = go.FigureWidget(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            )
        )
        base_fig.add_trace(go.Scatter(y=cat_series), row=1, col=1)
        base_fig.add_trace(dict(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureWidgetResampler object from a go.Figure
        # 1. All scatters are aggregated
        fwr_fw = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fwr_fw.data) == 3
        assert len(fwr_fw.hf_data) == 3
        for trace in fwr_fw.data:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fwr_fw._hf_data
            assert len(trace["y"]) == 2_000

        # 2. No scatters are aggregated
        fwr_fw = FigureWidgetResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fwr_fw.data) == 3
        assert len(fwr_fw.hf_data) == 0
        for trace in fwr_fw.data:
            assert trace.uid not in fwr_fw._hf_data
            assert len(trace["y"]) == 10_000

    # ---- Must not be aggregated
    def test_fr_fw_scatter_not_all_agg(float_series, bool_series, cat_series):
        base_fig = go.FigureWidget(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            )
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series[:1500]), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series[:800]), row=2, col=1)

        # Create FigureResampler object from a go.Figure
        fr_fw = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fw.data) == 3
        assert len(fr_fw.hf_data) == 1
        # Only the fist trace will be aggregated
        for trace in fr_fw.data[:1]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fw._hf_data
            assert len(trace["y"]) == 2_000

        for trace in fr_fw.data[1:]:
            assert trace.uid not in fr_fw._hf_data
            assert len(trace["y"]) != 2_000

    def test_fwr_fw_scatter_not_all_agg(float_series, bool_series, cat_series):
        base_fig = go.FigureWidget(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            )
        )
        base_fig.add_trace(go.Scatter(y=cat_series), row=1, col=1)
        base_fig.add_trace(dict(y=bool_series[:1500]), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series[:800]), row=2, col=1)

        # Create FigureResampler object from a go.Figure
        fwr_fw = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fwr_fw.data) == 3
        assert len(fwr_fw.hf_data) == 1
        # Only the fist trace will be aggregated
        for trace in fwr_fw.data[:1]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fwr_fw._hf_data
            assert len(trace["y"]) == 2_000

        for trace in fwr_fw.data[1:]:
            assert trace.uid not in fwr_fw._hf_data
            assert len(trace["y"]) != 2_000

    # ------- Mixed
    def test_fr_fw_mixed_agg(float_series):
        base_fig = go.FigureWidget(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            )
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fr_fw = FigureResampler(base_fig, default_n_shown_samples=1_000)
        assert len(fr_fw.data) == 3
        assert len(fr_fw.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fr_fw.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fw._hf_data
            assert len(trace["y"]) == 1_000

        for trace in fr_fw.data[:1] + fr_fw.data[2:]:
            assert trace.uid not in fr_fw._hf_data
            assert trace.y is None  # these traces don't even have a y value

    def test_fwr_fw_mixed_agg(float_series):
        base_fig = go.FigureWidget(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            )
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fwr_fw = FigureWidgetResampler(base_fig, default_n_shown_samples=1_000)
        assert len(fwr_fw.data) == 3
        assert len(fwr_fw.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fwr_fw.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fwr_fw._hf_data
            assert len(trace["y"]) == 1_000

        for trace in fwr_fw.data[:1] + fwr_fw.data[2:]:
            assert trace.uid not in fwr_fw._hf_data
            assert trace.y is None  # these traces don't even have a y value

    # ---- Must not (all) be aggregated
    def test_fr_fw_mixed_no_agg(float_series):
        base_fig = go.FigureWidget(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            )
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fr_fw = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fr_fw.data) == 3
        assert len(fr_fw.hf_data) == 0
        assert len(fr_fw.data[1]["y"]) == 10_000

        fwr_fw = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fwr_fw.data) == 3
        assert len(fwr_fw.hf_data) == 0
        assert len(fwr_fw.data[1]["y"]) == 10_000


# ----------------------- FigureResampler As base -----------------------
if True:
    # -------- All scatters
    def test_fr_fr_scatter_agg(float_series, bool_series, cat_series):
        base_fig = FigureResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fr_fr = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fr.data) == 3
        assert len(fr_fr.hf_data) == 3
        for trace in fr_fr.data:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fr._hf_data
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

        # 2. No scatters are aggregated
        fr_fr = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fr_fr.data) == 3
        assert len(fr_fr.hf_data) == 3
        for trace in fr_fr.data:
            assert len(trace["y"]) == 10_000

    def test_fr_fr_scatter_no_agg_agg(float_series, bool_series, cat_series):
        # This initial figure object does not contain any aggregated data as
        # default_n_shown samples >= the input data
        base_fig = FigureResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=10_000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        assert len(base_fig.hf_data) == 0

        # Create FigureResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fr_fr = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fr.data) == 3
        assert len(fr_fr.hf_data) == 3
        for trace in fr_fr.data:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fr._hf_data
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

        # 2. No scatters are aggregated
        fr_fr = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fr_fr.data) == 3
        assert len(fr_fr.hf_data) == 0
        for trace in fr_fr.data:
            assert len(trace["y"]) == 10_000

    def test_fr_fr_scatter_agg_limit_to_view(float_series, bool_series, cat_series):
        # we test whether the to view limited LF series will also get copied.
        base_fig = FigureResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(
            go.Scatter(y=bool_series[:800]), limit_to_view=True, row=1, col=2
        )
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fr_fr = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fr.data) == 3
        assert len(fr_fr.hf_data) == 3
        for trace in fr_fr.data[:1] + fr_fr.data[2:]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fr._hf_data
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000
        assert len(fr_fr.data[1]["y"]) == 800

        # 2. No scatters are aggregated
        fr_fr = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fr_fr.data) == 3
        assert len(fr_fr.hf_data) == 3
        for trace in fr_fr.data[:1] + fr_fr.data[2:]:
            assert len(trace["y"]) == 10_000
        assert len(fr_fr.data[1]["y"]) == 800

    def test_fwr_fr_scatter_agg(float_series, bool_series, cat_series):
        base_fig = FigureResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureWidgetResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fw_fr = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fw_fr.data) == 3
        assert len(fw_fr.hf_data) == 3
        for trace in fw_fr.data:
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

        # 2. No scatters are aggregated
        fw_fr = FigureWidgetResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fw_fr.data) == 3
        # NOTE: the hf_data gets copied so the lenght will be the same length as the
        # original figure
        assert len(fw_fr.hf_data) == 3
        for trace in fw_fr.data:
            assert len(trace["y"]) == 10_000

    def test_fwr_fr_scatter_no_agg_agg(float_series, bool_series, cat_series):
        # This initial figure object does not contain any aggregated data as
        # default_n_shown samples >= the input data
        base_fig = FigureResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=10_000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        assert len(base_fig.hf_data) == 0

        # Create FigureResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fwr_fr = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fwr_fr.data) == 3
        assert len(fwr_fr.hf_data) == 3
        for trace in fwr_fr.data:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fwr_fr._hf_data
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

        # 2. No scatters are aggregated
        fr_fr = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fr_fr.data) == 3
        assert len(fr_fr.hf_data) == 0
        for trace in fr_fr.data:
            assert len(trace["y"]) == 10_000

    def test_fwr_fr_scatter_agg_limit_to_view(float_series, bool_series, cat_series):
        # we test whether the to view limited LF series will also get copied.
        base_fig = FigureResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(
            go.Scatter(y=bool_series[:800]), limit_to_view=True, row=1, col=2
        )
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fwr_fr = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fwr_fr.data) == 3
        assert len(fwr_fr.hf_data) == 3
        for trace in fwr_fr.data[:1] + fwr_fr.data[2:]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fwr_fr._hf_data
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000
        assert len(fwr_fr.data[1]["y"]) == 800

        # 2. No scatters are aggregated
        fwr_fr = FigureWidgetResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fwr_fr.data) == 3
        assert len(fwr_fr.hf_data) == 3
        for trace in fwr_fr.data[:1] + fwr_fr.data[2:]:
            assert len(trace["y"]) == 10_000
        assert len(fwr_fr.data[1]["y"]) == 800

    def test_fr_fr_scatter_agg_no_default(float_series, bool_series, cat_series):
        base_fig = FigureResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2, max_n_samples=1000)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureREsampler object from a FigureResampler
        # 1. All scatters are aggregated
        fr_fr = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fr.data) == 3
        assert len(fr_fr.hf_data) == 3
        for trace in fr_fr.data[:1] + fr_fr.data[2:]:
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

        # this was not a default value, so it remains its original value; i.e 1000
        assert len(fr_fr.data[1]["y"]) == 1000

    def test_fwr_fr_scatter_agg_no_default(float_series, bool_series, cat_series):
        base_fig = FigureResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2, max_n_samples=1000)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureWidgetResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fwr_fr = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fwr_fr.data) == 3
        assert len(fwr_fr.hf_data) == 3
        for trace in fwr_fr.data[:1] + fwr_fr.data[2:]:
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

        # this was not a default value, so it remains its original value; i.e 1000
        assert len(fwr_fr.data[1]["y"]) == 1000

    # -------- Mixed
    def test_fr_fr_mixed_agg(float_series):
        base_fig = FigureResampler(
            go.FigureWidget(
                make_subplots(
                    rows=2,
                    cols=2,
                    specs=[[{}, {}], [{"colspan": 2}, None]],
                )
            ),
            default_n_shown_samples=999,
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fr_fr_mixed = FigureResampler(base_fig, default_n_shown_samples=1_020)
        assert len(fr_fr_mixed.data) == 3
        assert len(fr_fr_mixed.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fr_fr_mixed.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fr_mixed._hf_data
            assert len(trace["y"]) == 1_020

        for trace in fr_fr_mixed.data[:1] + fr_fr_mixed.data[2:]:
            assert trace.uid not in fr_fr_mixed._hf_data

    def test_fr_fr_mixed_no_default_agg(float_series):
        base_fig = FigureResampler(
            go.FigureWidget(
                make_subplots(
                    rows=2,
                    cols=2,
                    specs=[[{}, {}], [{"colspan": 2}, None]],
                )
            )
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2, max_n_samples=1054)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fr_fr_mixed = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fr_mixed.data) == 3
        assert len(fr_fr_mixed.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fr_fr_mixed.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fr_mixed._hf_data
            assert len(trace["y"]) == 1054

        for trace in fr_fr_mixed.data[:1] + fr_fr_mixed.data[2:]:
            assert trace.uid not in fr_fr_mixed._hf_data

    def test_fw_fr_mixed_agg(float_series):
        base_fig = FigureResampler(
            go.FigureWidget(
                make_subplots(
                    rows=2,
                    cols=2,
                    specs=[[{}, {}], [{"colspan": 2}, None]],
                )
            ),
            default_n_shown_samples=999,
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fw_fr_mixed = FigureWidgetResampler(base_fig, default_n_shown_samples=1_020)
        assert len(fw_fr_mixed.data) == 3
        assert len(fw_fr_mixed.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fw_fr_mixed.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fw_fr_mixed._hf_data
            assert len(trace["y"]) == 1_020

        for trace in fw_fr_mixed.data[:1] + fw_fr_mixed.data[2:]:
            assert trace.uid not in fw_fr_mixed._hf_data

    def test_fw_fr_mixed_no_default_agg(float_series):
        base_fig = FigureResampler(
            go.FigureWidget(
                make_subplots(
                    rows=2,
                    cols=2,
                    specs=[[{}, {}], [{"colspan": 2}, None]],
                )
            )
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2, max_n_samples=1054)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fw_fr_mixed = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fw_fr_mixed.data) == 3
        assert len(fw_fr_mixed.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fw_fr_mixed.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fw_fr_mixed._hf_data
            assert len(trace["y"]) == 1054

        for trace in fw_fr_mixed.data[:1] + fw_fr_mixed.data[2:]:
            assert trace.uid not in fw_fr_mixed._hf_data


# ----------------------- FigureWidgetResampler As base -----------------------
if True:
    # -------- All scatters
    def test_fr_fwr_scatter_agg(float_series, bool_series, cat_series):
        base_fig = FigureWidgetResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fr_fw = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fw.data) == 3
        assert len(fr_fw.hf_data) == 3
        for trace in fr_fw.data:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fw._hf_data
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

        # 2. No scatters are aggregated
        fr_fw = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fr_fw.data) == 3
        assert len(fr_fw.hf_data) == 3
        for trace in fr_fw.data:
            assert len(trace["y"]) == 10_000

    def test_fr_fwr_scatter_no_agg_agg(float_series, bool_series, cat_series):
        # This inital figure object does not contain any aggregated data as
        # default_n_shown samples >= the input data
        base_fig = FigureWidgetResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=10_000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        assert len(base_fig.hf_data) == 0

        # Create FigureResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fr_fwr = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fwr.data) == 3
        assert len(fr_fwr.hf_data) == 3
        for trace in fr_fwr.data:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fwr._hf_data
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

    def test_fr_fwr_scatter_agg_limit_to_view(float_series, bool_series, cat_series):
        # we test whether the to view limited LF series will also get copied.
        base_fig = FigureWidgetResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(
            go.Scatter(y=bool_series[:800]), limit_to_view=True, row=1, col=2
        )
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fr_fw = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fw.data) == 3
        assert len(fr_fw.hf_data) == 3
        for trace in fr_fw.data[:1] + fr_fw.data[2:]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fw._hf_data
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000
        assert len(fr_fw.data[1]["y"]) == 800

        # 2. No scatters are aggregated
        fr_fw = FigureResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fr_fw.data) == 3
        assert len(fr_fw.hf_data) == 3
        for trace in fr_fw.data[:1] + fr_fw.data[2:]:
            assert len(trace["y"]) == 10_000
        assert len(fr_fw.data[1]["y"]) == 800

    def test_fw_fwr_scatter_agg(float_series, bool_series, cat_series):
        base_fig = FigureWidgetResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureWidgetResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fw_fw = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fw_fw.data) == 3
        assert len(fw_fw.hf_data) == 3
        for trace in fw_fw.data:
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

        # 2. No scatters are aggregated
        fw_fw = FigureWidgetResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fw_fw.data) == 3
        # NOTE: the hf_data gets copied so the lenght will be the same length as the
        # original figure
        assert len(fw_fw.hf_data) == 3
        for trace in fw_fw.data:
            assert len(trace["y"]) == 10_000

    def test_fwr_fwr_scatter_no_agg_agg(float_series, bool_series, cat_series):
        # This inital figure object does not contain any aggregated data as
        # default_n_shown samples >= the input data
        base_fig = FigureWidgetResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=10_000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        assert len(base_fig.hf_data) == 0

        # Create FigureResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fwr_fwr = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fwr_fwr.data) == 3
        assert len(fwr_fwr.hf_data) == 3
        for trace in fwr_fwr.data:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fwr_fwr._hf_data
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

    def test_fwr_fwr_scatter_agg_limit_to_view(float_series, bool_series, cat_series):
        # we test whether the to view limited LF series will also get copied.
        base_fig = FigureWidgetResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(
            go.Scatter(y=bool_series[:800]), limit_to_view=True, row=1, col=2
        )
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fwr_fw = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fwr_fw.data) == 3
        assert len(fwr_fw.hf_data) == 3
        for trace in fwr_fw.data[:1] + fwr_fw.data[2:]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fwr_fw._hf_data
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000
        assert len(fwr_fw.data[1]["y"]) == 800

        # 2. No scatters are aggregated
        fwr_fw = FigureWidgetResampler(base_fig, default_n_shown_samples=10_000)
        assert len(fwr_fw.data) == 3
        assert len(fwr_fw.hf_data) == 3
        for trace in fwr_fw.data[:1] + fwr_fw.data[2:]:
            assert len(trace["y"]) == 10_000
        assert len(fwr_fw.data[1]["y"]) == 800

    def test_fr_fwr_scatter_agg_no_default(float_series, bool_series, cat_series):
        base_fig = FigureWidgetResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2, max_n_samples=1000)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureREsampler object from a FigureResampler
        # 1. All scatters are aggregated
        fr_fw = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fw.data) == 3
        assert len(fr_fw.hf_data) == 3
        for trace in fr_fw.data[:1] + fr_fw.data[2:]:
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

        # this was not a default value, so it remains its original value; i.e 1000
        assert len(fr_fw.data[1]["y"]) == 1000

    def test_fw_fwr_scatter_agg_no_default(float_series, bool_series, cat_series):
        base_fig = FigureWidgetResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2, max_n_samples=1000)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        # Create FigureWidgetResampler object from a go.Scatter
        # 1. All scatters are aggregated
        fw_fw = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fw_fw.data) == 3
        assert len(fw_fw.hf_data) == 3
        for trace in fw_fw.data[:1] + fw_fw.data[2:]:
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

        # this was not a default value, so it remains its original value; i.e 1000
        assert len(fw_fw.data[1]["y"]) == 1000

    # -------- Midex
    def test_fr_fwr_mixed_agg(float_series):
        base_fig = FigureWidgetResampler(
            go.FigureWidget(
                make_subplots(
                    rows=2,
                    cols=2,
                    specs=[[{}, {}], [{"colspan": 2}, None]],
                )
            ),
            default_n_shown_samples=999,
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fr_fw_mixed = FigureResampler(base_fig, default_n_shown_samples=1_020)
        assert len(fr_fw_mixed.data) == 3
        assert len(fr_fw_mixed.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fr_fw_mixed.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fw_mixed._hf_data
            assert len(trace["y"]) == 1_020

        for trace in fr_fw_mixed.data[:1] + fr_fw_mixed.data[2:]:
            assert trace.uid not in fr_fw_mixed._hf_data

    def test_fr_fwr_mixed_no_default_agg(float_series):
        base_fig = FigureWidgetResampler(
            go.FigureWidget(
                make_subplots(
                    rows=2,
                    cols=2,
                    specs=[[{}, {}], [{"colspan": 2}, None]],
                )
            )
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2, max_n_samples=1054)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fr_fw_mixed = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fw_mixed.data) == 3
        assert len(fr_fw_mixed.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fr_fw_mixed.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fr_fw_mixed._hf_data
            assert len(trace["y"]) == 1054

        for trace in fr_fw_mixed.data[:1] + fr_fw_mixed.data[2:]:
            assert trace.uid not in fr_fw_mixed._hf_data

    def test_fw_fwr_mixed_agg(float_series):
        base_fig = FigureWidgetResampler(
            go.FigureWidget(
                make_subplots(
                    rows=2,
                    cols=2,
                    specs=[[{}, {}], [{"colspan": 2}, None]],
                )
            ),
            default_n_shown_samples=999,
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fw_fw_mixed = FigureWidgetResampler(base_fig, default_n_shown_samples=1_020)
        assert len(fw_fw_mixed.data) == 3
        assert len(fw_fw_mixed.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fw_fw_mixed.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fw_fw_mixed._hf_data
            assert len(trace["y"]) == 1_020

        for trace in fw_fw_mixed.data[:1] + fw_fw_mixed.data[2:]:
            assert trace.uid not in fw_fw_mixed._hf_data

    def test_fw_fwr_mixed_no_default_agg(float_series):
        base_fig = FigureWidgetResampler(
            go.Figure(
                make_subplots(
                    rows=2,
                    cols=2,
                    specs=[[{}, {}], [{"colspan": 2}, None]],
                )
            )
        )
        base_fig.add_trace(go.Box(x=float_series), row=1, col=1)
        base_fig.add_trace(dict(y=float_series), row=1, col=2, max_n_samples=1054)
        base_fig.add_trace(go.Histogram(x=float_series), row=2, col=1)

        fw_fw_mixed = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fw_fw_mixed.data) == 3
        assert len(fw_fw_mixed.hf_data) == 1  # Only the second trace will be aggregated
        for trace in fw_fw_mixed.data[1:2]:
            # ensure that all uids are in the `_hf_data` property
            assert trace.uid in fw_fw_mixed._hf_data
            assert len(trace["y"]) == 1054

        for trace in fw_fw_mixed.data[:1] + fw_fw_mixed.data[2:]:
            assert trace.uid not in fw_fw_mixed._hf_data


# =========================================================
# Performing zoom events on widgets
if True:

    def test_fr_fwr_scatter_agg_zoom(cat_series, bool_series, float_series):
        base_fig = FigureWidgetResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2, max_n_samples=1000)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        base_fig.layout.update(
            {
                "xaxis": {"range": [10_000, 20_000]},
                "yaxis": {"range": [-20, 3]},
                "xaxis2": {"range": [40_000, 60_000]},
                "yaxis2": {"range": [-10, 3]},
            },
            overwrite=False,
        )

        fr_fwr = FigureResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fr_fwr.data) == 3
        assert len(fr_fwr.hf_data) == 3
        for trace in fr_fwr.data[:1] + fr_fwr.data[2:]:
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

            # Verify whether the zoom did not affect antyhign
            assert trace["x"][0] == 0
            assert trace["x"][-1] == 9999

        # this was not a default value, so it remains its original value; i.e 1000
        assert len(fr_fwr.data[1]["y"]) == 1000
        assert fr_fwr.data[1]["x"][0] == 0
        assert fr_fwr.data[1]["x"][-1] == 9999

    def test_fwr_fwr_scatter_agg_zoom(cat_series, bool_series, float_series):
        base_fig = FigureWidgetResampler(
            make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
            ),
            default_n_shown_samples=1000,
        )
        base_fig.add_trace(dict(y=cat_series), row=1, col=1)
        base_fig.add_trace(go.Scatter(y=bool_series), row=1, col=2, max_n_samples=1000)
        base_fig.add_trace(go.Scattergl(y=float_series), row=2, col=1)

        base_fig.layout.update(
            {
                "xaxis": {"range": [10_000, 20_000]},
                "yaxis": {"range": [-20, 3]},
                "xaxis2": {"range": [40_000, 60_000]},
                "yaxis2": {"range": [-10, 3]},
            },
            overwrite=False,
        )

        fwr_fwr = FigureWidgetResampler(base_fig, default_n_shown_samples=2_000)
        assert len(fwr_fwr.data) == 3
        assert len(fwr_fwr.hf_data) == 3
        for trace in fwr_fwr.data[:1] + fwr_fwr.data[2:]:
            # NOTE: default arguments are overtaken, so the default number of samples
            # of the wrapped `FigureResampler` traces are overriden by the default
            # number of samples of this class
            assert len(trace["y"]) == 2_000

            # Verify whether the zoom did not affect antyhign
            assert trace["x"][0] == 0
            assert trace["x"][-1] == 9999

        # this was not a default value, so it remains its original value; i.e 1000
        assert len(fwr_fwr.data[1]["y"]) == 1000
        assert fwr_fwr.data[1]["x"][0] == 0
        assert fwr_fwr.data[1]["x"][-1] == 9999
