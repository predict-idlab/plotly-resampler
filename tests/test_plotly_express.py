import numpy as np
import pandas as pd
import plotly.express as px

from plotly_resampler import register_plotly_resampler, unregister_plotly_resampler


def test_px_hoverlabel_figureResampler():
    labels = list(range(0, 3))
    N = 60_000
    x = np.arange(N)
    y = np.random.normal(size=N)
    label = np.random.randint(low=labels[0], high=labels[-1] + 1, size=N).astype(str)
    description = np.random.randint(low=3, high=5, size=N)

    df = pd.DataFrame.from_dict(
        {"x": x, "y": y, "label": label, "description": description}
    )

    x_label = "x"
    y_label = "y"
    label_label = "label"
    df = df.sort_values(by=[x_label])

    # Without resampler, shows correct hover data
    fig = px.scatter(
        df,
        x=x_label,
        y=y_label,
        color=label_label,
        title="Without resampler",
        hover_data=["description"],
    )

    # With resampler, shows incorrect hover data
    register_plotly_resampler(mode="auto", default_n_shown_samples=1000)
    fig2 = px.scatter(
        df,
        x=x_label,
        y=y_label,
        color=label_label,
        title="With resampler",
        hover_data=["description"],
    )

    # verify whether the selected has the same y and customdata as the original
    for idx in range(len(fig.data)):
        trc_orig = fig.data[idx]
        trc_res = fig2.data[idx]

        agg_indices = np.searchsorted(trc_orig["x"], trc_res["x"]).ravel()
        for k in ["customdata", "y"]:
            assert all(trc_orig[k].ravel()[agg_indices] == trc_res[k].ravel())

    unregister_plotly_resampler()
