"""Tests for the dynamic property handling architecture."""

import numpy as np
import plotly.graph_objects as go
import pytest

from plotly_resampler import FigureResampler, FigureWidgetResampler


def test_marker_symbol_support():
    """Test that marker_symbol is now supported as a downsamplable property."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    symbols = ["circle", "square", "diamond", "cross", "x"] * (n // 5)

    # Add trace with marker_symbol
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(symbol=symbols)))

    # Check that marker_symbol is in hf_data
    assert "marker_symbol" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_symbol"]) == n

    # Check that the downsampled trace has marker_symbol
    assert hasattr(fig.data[0].marker, "symbol")
    assert len(fig.data[0].marker.symbol) == 100  # default_n_shown_samples


def test_hf_marker_symbol_parameter():
    """Test that hf_marker_symbol parameter works."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    symbols = ["circle", "square", "diamond", "cross", "x"] * (n // 5)

    # Add trace with hf_marker_symbol parameter
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers"), hf_marker_symbol=symbols)

    # Check that marker_symbol is in hf_data
    assert "marker_symbol" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_symbol"]) == n

    # Check that the downsampled trace has marker_symbol
    assert hasattr(fig.data[0].marker, "symbol")
    assert len(fig.data[0].marker.symbol) == 100


def test_multiple_dynamic_properties():
    """Test multiple dynamic properties together."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    text = [f"Point {i}" for i in range(n)]
    colors = ["red" if i % 2 == 0 else "blue" for i in range(n)]
    symbols = ["circle", "square"] * (n // 2)
    sizes = np.random.randint(5, 15, n)

    # Add trace with multiple dynamic properties
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers"),
        hf_text=text,
        hf_marker_color=colors,
        hf_marker_symbol=symbols,
        hf_marker_size=sizes,
    )

    # Check all properties are in hf_data
    hf_trace = fig.hf_data[0]
    assert "text" in hf_trace
    assert "marker_color" in hf_trace
    assert "marker_symbol" in hf_trace
    assert "marker_size" in hf_trace

    # Check lengths
    assert len(hf_trace["text"]) == n
    assert len(hf_trace["marker_color"]) == n
    assert len(hf_trace["marker_symbol"]) == n
    assert len(hf_trace["marker_size"]) == n

    # Check downsampled trace
    trace = fig.data[0]
    assert len(trace.text) == 100
    assert len(trace.marker.color) == 100
    assert len(trace.marker.symbol) == 100
    assert len(trace.marker.size) == 100


def test_aggregation_contains_downsampled_properties():
    """Test that aggregation contains the downsampled properties using MinMaxAggregator."""
    # Highly similar to the usecase in issue #354
    from plotly_resampler.aggregation import MinMaxAggregator

    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data similar to the example
    n = 2000
    x = np.arange(n)
    y = (3 + np.sin(x / (x.shape[0] / 100)) + np.random.randn(len(x)) / 10) * x / 1_000

    # Prepare the marker symbol and color arrays
    marker_symbol = np.where(x % 2 == 0, "triangle-up", "triangle-down")
    marker_color = np.where(x % 3 == 0, "green", np.where(x % 3 == 1, "red", "blue"))

    # Add the trace with hf_marker_symbol and hf_marker_color
    fig.add_trace(
        go.Scatter(name="foobar", showlegend=True, mode="markers", marker={"size": 20}),
        hf_x=x,
        hf_y=y,
        hf_marker_symbol=marker_symbol,
        hf_marker_color=marker_color,
        downsampler=MinMaxAggregator(),
    )

    # Check that properties are in hf_data
    hf_trace = fig.hf_data[0]
    assert "marker_symbol" in hf_trace
    assert "marker_color" in hf_trace
    assert len(hf_trace["marker_symbol"]) == n
    assert len(hf_trace["marker_color"]) == n

    # Check that the downsampled trace has the properties
    trace = fig.data[0]
    assert hasattr(trace.marker, "symbol")
    assert hasattr(trace.marker, "color")
    assert len(trace.marker.symbol) == 100
    assert len(trace.marker.color) == 100

    # Check that the downsampled properties maintain the pattern
    downsampled_symbols = trace.marker.symbol
    downsampled_colors = trace.marker.color
    downsampled_x = trace.x

    # Verify that the downsampled symbols are still valid
    assert np.all(downsampled_symbols == marker_symbol[downsampled_x])
    assert np.all(downsampled_colors == marker_color[downsampled_x])


def test_aggregation_property_consistency():
    """Test that aggregated properties are consistent with the original data pattern."""
    # Highly similar to the usecase in issue #354
    from plotly_resampler.aggregation import MinMaxAggregator

    fig = FigureResampler(default_n_shown_samples=50)

    # Create test data with a clear pattern
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)

    # Create alternating patterns for easy verification
    marker_symbol = np.where(
        x % 4 == 0,
        "circle",
        np.where(x % 4 == 1, "square", np.where(x % 4 == 2, "diamond", "cross")),
    )

    marker_color = np.where(x % 3 == 0, "red", np.where(x % 3 == 1, "green", "blue"))

    # Add trace with clear patterns
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers"),
        hf_marker_symbol=marker_symbol,
        hf_marker_color=marker_color,
        downsampler=MinMaxAggregator(),
    )

    # Check that the downsampled properties maintain the original values
    # (not aggregated/transformed values)
    trace = fig.data[0]
    downsampled_symbols = trace.marker.symbol
    downsampled_colors = trace.marker.color
    downsampled_x = trace.x

    # The length should match the downsampled data
    assert len(downsampled_symbols) == 50
    assert len(downsampled_colors) == 50
    assert len(downsampled_x) == 50

    # Assert the the downsampled data matches the original data
    assert np.all(downsampled_symbols == marker_symbol[downsampled_x])
    assert np.all(downsampled_colors == marker_color[downsampled_x])


def test_figurewidget_resampler_support():
    """Test that FigureWidgetResampler also supports dynamic properties."""
    fig = FigureWidgetResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    symbols = (["circle", "square", "diamond"] * ((n // 3) + 1))[:n]

    # Add trace with marker_symbol
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(symbol=symbols)))

    # Check that marker_symbol is in hf_data
    assert "marker_symbol" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_symbol"]) == n


def test_property_priority():
    """Test that hf_* parameters have priority over trace properties."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)

    # Create different symbols for trace vs hf parameter
    trace_symbols = ["circle"] * n
    hf_symbols = ["square"] * n

    # Add trace with both trace property and hf parameter
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(symbol=trace_symbols)),
        hf_marker_symbol=hf_symbols,
    )

    # Check that hf parameter took priority
    assert np.all(fig.hf_data[0]["marker_symbol"] == hf_symbols)


def test_none_properties():
    """Test that None properties are handled correctly."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)

    # Add trace with None marker_symbol
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
        ),
        hf_marker_symbol=None,
    )

    # Check that marker_symbol is None in hf_data
    assert fig.hf_data[0]["marker_symbol"] is None


def test_mixed_property_types():
    """Test mixing different property types."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)

    # Mix different property types
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            text=["static text"] * n,  # From trace
        ),
        hf_marker_symbol=["circle", "square"] * (n // 2),  # From hf parameter
        hf_marker_color=None,  # Explicitly None
    )

    # Check properties
    hf_trace = fig.hf_data[0]
    assert "text" in hf_trace
    assert "marker_symbol" in hf_trace
    assert "marker_color" in hf_trace

    assert len(hf_trace["text"]) == n
    assert len(hf_trace["marker_symbol"]) == n
    assert hf_trace["marker_color"] is None


def test_data_point_selector_requirement():
    """Test that DataPointSelector is required for array properties."""
    from plotly_resampler.aggregation import FuncAggregator

    fig = FigureResampler(
        default_n_shown_samples=100,
        default_downsampler=FuncAggregator(aggregation_func=np.mean),
    )

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    symbols = ["circle", "square"] * (n // 2)

    # This should raise an error because FuncAggregator is not a DataPointSelector
    with pytest.raises(AssertionError, match="Only DataPointSelector can downsample"):
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(symbol=symbols)))


def test_extensibility():
    """Test that the architecture is extensible by checking the configuration."""
    from plotly_resampler.figure_resampler.figure_resampler_interface import (
        DOWNSAMPLABLE_PROPERTIES,
    )

    # Check that marker_symbol is in the configuration
    property_names = [prop[0] for prop in DOWNSAMPLABLE_PROPERTIES]
    assert "marker_symbol" in property_names

    # Check that all expected properties are there
    expected_properties = [
        "text",
        "hovertext",
        "marker_size",
        "marker_color",
        "marker_symbol",
        "customdata",
    ]
    for prop in expected_properties:
        assert prop in property_names


def test_property_path_handling():
    """Test that nested property paths are handled correctly."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)

    # Test with deeply nested property (marker.symbol)
    # Make sure the symbols array has exactly n elements
    symbols = (["circle", "square", "diamond"] * ((n // 3) + 1))[:n]

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(symbol=symbols)))

    # Check that the nested property was handled correctly
    assert "marker_symbol" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_symbol"]) == n

    # Check that the downsampled trace has the correct nested structure
    trace = fig.data[0]
    assert hasattr(trace.marker, "symbol")
    assert len(trace.marker.symbol) == 100
