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
    angles = np.linspace(0, 360, n)
    opacities = np.linspace(0.1, 1.0, n)

    # Add trace with multiple dynamic properties
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers"),
        hf_text=text,
        hf_marker_color=colors,
        hf_marker_symbol=symbols,
        hf_marker_size=sizes,
        hf_marker_angle=angles,
        hf_marker_opacity=opacities,
    )

    # Check all properties are in hf_data
    hf_trace = fig.hf_data[0]
    assert "text" in hf_trace
    assert "marker_color" in hf_trace
    assert "marker_symbol" in hf_trace
    assert "marker_size" in hf_trace
    assert "marker_angle" in hf_trace
    assert "marker_opacity" in hf_trace

    # Check lengths
    assert len(hf_trace["text"]) == n
    assert len(hf_trace["marker_color"]) == n
    assert len(hf_trace["marker_symbol"]) == n
    assert len(hf_trace["marker_size"]) == n
    assert len(hf_trace["marker_angle"]) == n
    assert len(hf_trace["marker_opacity"]) == n

    # Check downsampled trace
    trace = fig.data[0]
    assert len(trace.text) == 100
    assert len(trace.marker.color) == 100
    assert len(trace.marker.symbol) == 100
    assert len(trace.marker.size) == 100
    assert len(trace.marker.angle) == 100
    assert len(trace.marker.opacity) == 100


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
        "marker_angle",
        "marker_opacity",
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


def test_marker_angle_support():
    """Test that marker_angle is supported as a downsamplable property."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    angles = np.linspace(0, 360, n)  # Angles from 0 to 360 degrees

    # Add trace with marker_angle
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(angle=angles)))

    # Check that marker_angle is in hf_data
    assert "marker_angle" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_angle"]) == n

    # Check that the downsampled trace has marker_angle
    assert hasattr(fig.data[0].marker, "angle")
    assert len(fig.data[0].marker.angle) == 100  # default_n_shown_samples

    # Check that the downsampled data matches the original data
    downsampled_x = fig.data[0].x
    downsampled_angles = fig.data[0].marker.angle

    # Create a regular Plotly trace with the downsampled data to get expected normalized angles
    expected_trace = go.Scatter(x=x, y=y, mode="markers", marker=dict(angle=angles))
    expected_angles = expected_trace.marker.angle

    assert np.allclose(downsampled_angles, expected_angles[downsampled_x])


def test_hf_marker_angle_parameter():
    """Test that hf_marker_angle parameter works."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    angles = np.random.uniform(0, 360, n)  # Random angles

    # Add trace with hf_marker_angle parameter
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers"), hf_marker_angle=angles)

    # Check that marker_angle is in hf_data
    assert "marker_angle" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_angle"]) == n

    # Check that the downsampled trace has marker_angle
    assert hasattr(fig.data[0].marker, "angle")
    assert len(fig.data[0].marker.angle) == 100

    # Check that the downsampled data matches the original data
    downsampled_x = fig.data[0].x
    downsampled_angles = fig.data[0].marker.angle

    # Create a regular Plotly trace with the downsampled data to get expected normalized angles
    expected_trace = go.Scatter(x=x, y=y, mode="markers", marker=dict(angle=angles))
    expected_angles = expected_trace.marker.angle

    assert np.allclose(downsampled_angles, expected_angles[downsampled_x])


def test_marker_opacity_support():
    """Test that marker_opacity is supported as a downsamplable property."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    opacities = np.linspace(0.1, 1.0, n)  # Opacities from 0.1 to 1.0

    # Add trace with marker_opacity
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(opacity=opacities)))

    # Check that marker_opacity is in hf_data
    assert "marker_opacity" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_opacity"]) == n

    # Check that the downsampled trace has marker_opacity
    assert hasattr(fig.data[0].marker, "opacity")
    assert len(fig.data[0].marker.opacity) == 100  # default_n_shown_samples

    # Check that the downsampled data matches the original data
    downsampled_x = fig.data[0].x
    downsampled_opacities = fig.data[0].marker.opacity
    assert np.all(downsampled_opacities == opacities[downsampled_x])


def test_hf_marker_opacity_parameter():
    """Test that hf_marker_opacity parameter works."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    opacities = np.random.uniform(0.0, 1.0, n)  # Random opacities

    # Add trace with hf_marker_opacity parameter
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers"), hf_marker_opacity=opacities)

    # Check that marker_opacity is in hf_data
    assert "marker_opacity" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_opacity"]) == n

    # Check that the downsampled trace has marker_opacity
    assert hasattr(fig.data[0].marker, "opacity")
    assert len(fig.data[0].marker.opacity) == 100

    # Check that the downsampled data matches the original data
    downsampled_x = fig.data[0].x
    downsampled_opacities = fig.data[0].marker.opacity
    assert np.all(downsampled_opacities == opacities[downsampled_x])


def test_marker_angle_and_opacity_together():
    """Test marker_angle and marker_opacity properties together."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    angles = np.linspace(0, 360, n)
    opacities = np.linspace(0.1, 1.0, n)

    # Add trace with both marker_angle and marker_opacity
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers"),
        hf_marker_angle=angles,
        hf_marker_opacity=opacities,
    )

    # Check both properties are in hf_data
    hf_trace = fig.hf_data[0]
    assert "marker_angle" in hf_trace
    assert "marker_opacity" in hf_trace

    # Check lengths
    assert len(hf_trace["marker_angle"]) == n
    assert len(hf_trace["marker_opacity"]) == n

    # Check downsampled trace
    trace = fig.data[0]
    assert len(trace.marker.angle) == 100
    assert len(trace.marker.opacity) == 100

    # Check that the downsampled data matches the original data
    downsampled_x = fig.data[0].x
    downsampled_angles = fig.data[0].marker.angle
    downsampled_opacities = fig.data[0].marker.opacity
    original_angles = angles[downsampled_x]

    # Create a regular Plotly trace with the downsampled data to get expected normalized angles
    expected_trace = go.Scatter(
        x=downsampled_x,
        y=fig.data[0].y,
        mode="markers",
        marker=dict(angle=original_angles),
    )
    expected_angles = expected_trace.marker.angle

    assert np.allclose(downsampled_angles, expected_angles)
    assert np.all(downsampled_opacities == opacities[downsampled_x])


def test_marker_angle_opacity_aggregation_consistency():
    """Test that aggregated marker_angle and marker_opacity are consistent with original data."""
    from plotly_resampler.aggregation import MinMaxAggregator

    fig = FigureResampler(default_n_shown_samples=50)

    # Create test data with clear patterns
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)

    # Create alternating patterns for easy verification
    marker_angle = np.where(x % 2 == 0, 0, 90)  # Alternating 0 and 90 degrees
    marker_opacity = np.where(x % 3 == 0, 0.3, np.where(x % 3 == 1, 0.6, 0.9))

    # Add trace with clear patterns
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers"),
        hf_marker_angle=marker_angle,
        hf_marker_opacity=marker_opacity,
        downsampler=MinMaxAggregator(),
    )

    # Check that the downsampled properties maintain the original values
    trace = fig.data[0]
    downsampled_angles = trace.marker.angle
    downsampled_opacities = trace.marker.opacity
    downsampled_x = trace.x

    # The length should match the downsampled data
    assert len(downsampled_angles) == 50
    assert len(downsampled_opacities) == 50
    assert len(downsampled_x) == 50

    # Assert that the downsampled data matches the original data
    assert np.all(downsampled_opacities == marker_opacity[downsampled_x])
    # Plotly normalizes angles (angles > 180 become negative)
    expected_trace = go.Scatter(
        x=x, y=y, mode="markers", marker=dict(angle=marker_angle)
    )
    expected_angles = expected_trace.marker.angle
    assert np.allclose(downsampled_angles, expected_angles[downsampled_x])


def test_marker_angle_opacity_with_other_properties():
    """Test marker_angle and marker_opacity with other dynamic properties."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    angles = np.linspace(0, 360, n)
    opacities = np.linspace(0.1, 1.0, n)
    colors = ["red" if i % 2 == 0 else "blue" for i in range(n)]
    sizes = np.random.randint(5, 15, n)

    # Add trace with multiple properties including angle and opacity
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers"),
        hf_marker_angle=angles,
        hf_marker_opacity=opacities,
        hf_marker_color=colors,
        hf_marker_size=sizes,
    )

    # Check all properties are in hf_data
    hf_trace = fig.hf_data[0]
    assert "marker_angle" in hf_trace
    assert "marker_opacity" in hf_trace
    assert "marker_color" in hf_trace
    assert "marker_size" in hf_trace

    # Check lengths
    assert len(hf_trace["marker_angle"]) == n
    assert len(hf_trace["marker_opacity"]) == n
    assert len(hf_trace["marker_color"]) == n
    assert len(hf_trace["marker_size"]) == n

    # Check downsampled trace
    trace = fig.data[0]
    assert len(trace.marker.angle) == 100
    assert len(trace.marker.opacity) == 100
    assert len(trace.marker.color) == 100
    assert len(trace.marker.size) == 100

    # Check that the downsampled data matches the original data
    downsampled_x = fig.data[0].x
    downsampled_angles = fig.data[0].marker.angle
    downsampled_opacities = fig.data[0].marker.opacity
    downsampled_colors = fig.data[0].marker.color
    downsampled_sizes = fig.data[0].marker.size

    # Create a regular Plotly trace with the downsampled data to get expected normalized angles
    expected_trace = go.Scatter(x=x, y=y, mode="markers", marker=dict(angle=angles))
    expected_angles = expected_trace.marker.angle

    assert np.allclose(downsampled_angles, expected_angles[downsampled_x])
    assert np.all(downsampled_opacities == opacities[downsampled_x])
    assert np.all(downsampled_colors == np.array(colors)[downsampled_x])
    assert np.all(downsampled_sizes == sizes[downsampled_x])


def test_marker_angle_opacity_none_values():
    """Test that None values for marker_angle and marker_opacity are handled correctly."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)

    # Add trace with None marker_angle and marker_opacity
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers"),
        hf_marker_angle=None,
        hf_marker_opacity=None,
    )

    # Check that properties are None in hf_data
    assert fig.hf_data[0]["marker_angle"] is None
    assert fig.hf_data[0]["marker_opacity"] is None


def test_marker_angle_opacity_priority():
    """Test that hf_marker_angle and hf_marker_opacity parameters have priority over trace properties."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)

    # Create different values for trace vs hf parameters
    trace_angles = np.zeros(n)
    hf_angles = np.ones(n) * 90
    trace_opacities = np.ones(n) * 0.5
    hf_opacities = np.ones(n) * 0.8

    # Add trace with both trace properties and hf parameters
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(angle=trace_angles, opacity=trace_opacities),
        ),
        hf_marker_angle=hf_angles,
        hf_marker_opacity=hf_opacities,
    )

    # Check that hf parameters took priority
    assert np.all(fig.hf_data[0]["marker_angle"] == hf_angles)
    assert np.all(fig.hf_data[0]["marker_opacity"] == hf_opacities)

    # Check that the downsampled data matches the original data
    downsampled_x = fig.data[0].x
    downsampled_angles = fig.data[0].marker.angle
    downsampled_opacities = fig.data[0].marker.opacity
    # Plotly normalizes angles (angles > 180 become negative)
    # So we need to normalize the original angles the same way
    original_angles = hf_angles[downsampled_x]
    # Normalize angles to match Plotly's behavior
    normalized_original = np.where(
        original_angles > 180, original_angles - 360, original_angles
    )
    assert np.allclose(downsampled_angles, normalized_original)
    assert np.all(downsampled_opacities == hf_opacities[downsampled_x])


def test_marker_angle_opacity_edge_cases():
    """Test edge cases for marker_angle and marker_opacity."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)

    # Test edge cases: negative angles, angles > 360, opacities at boundaries
    angles = np.array([-90, 0, 90, 180, 270, 450, 720])  # Various angle values
    opacities = np.array([0.0, 0.1, 0.5, 0.9, 1.0])  # Valid opacity values

    # Repeat arrays to match data length
    angles = np.tile(angles, (n // len(angles)) + 1)[:n]
    opacities = np.tile(opacities, (n // len(opacities)) + 1)[:n]

    # Use Plotly to get the normalized angles for comparison
    temp_trace = go.Scatter(x=x, y=y, mode="markers", marker=dict(angle=angles))
    expected_normalized_angles = temp_trace.marker.angle

    # Add trace with edge case values
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers"),
        hf_marker_angle=angles,
        hf_marker_opacity=opacities,
    )

    # Check that properties are stored correctly
    assert "marker_angle" in fig.hf_data[0]
    assert "marker_opacity" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_angle"]) == n
    assert len(fig.hf_data[0]["marker_opacity"]) == n

    # Check that downsampled trace has the properties
    trace = fig.data[0]
    assert hasattr(trace.marker, "angle")
    assert hasattr(trace.marker, "opacity")
    assert len(trace.marker.angle) == 100
    assert len(trace.marker.opacity) == 100

    # Check that the downsampled data matches the expected normalized data
    downsampled_x = fig.data[0].x
    downsampled_angles = fig.data[0].marker.angle
    downsampled_opacities = fig.data[0].marker.opacity
    expected_angles = expected_normalized_angles[downsampled_x]
    assert np.allclose(downsampled_angles, expected_angles)
    assert np.all(downsampled_opacities == opacities[downsampled_x])


def test_marker_angle_opacity_data_types():
    """Test different data types for marker_angle and marker_opacity."""
    fig = FigureResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)

    # Test with different data types - make sure arrays have correct length
    angles_list = list(range(0, n))  # List of integers with correct length
    opacities_array = np.random.random(n)  # NumPy array of floats

    # Add trace with different data types
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers"),
        hf_marker_angle=angles_list,
        hf_marker_opacity=opacities_array,
    )

    # Check that properties are stored correctly
    assert "marker_angle" in fig.hf_data[0]
    assert "marker_opacity" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_angle"]) == n
    assert len(fig.hf_data[0]["marker_opacity"]) == n

    # Check that downsampled trace has the properties
    trace = fig.data[0]
    assert hasattr(trace.marker, "angle")
    assert hasattr(trace.marker, "opacity")
    assert len(trace.marker.angle) == 100
    assert len(trace.marker.opacity) == 100

    # Check that the downsampled data matches the original data
    downsampled_x = fig.data[0].x
    downsampled_angles = fig.data[0].marker.angle
    downsampled_opacities = fig.data[0].marker.opacity

    # Create a regular Plotly trace with the downsampled data to get expected normalized angles
    expected_trace = go.Scatter(
        x=x, y=y, mode="markers", marker=dict(angle=angles_list)
    )
    expected_angles = expected_trace.marker.angle

    assert np.allclose(downsampled_angles, np.array(expected_angles)[downsampled_x])
    assert np.all(downsampled_opacities == opacities_array[downsampled_x])


def test_marker_angle_opacity_figurewidget():
    """Test that FigureWidgetResampler supports marker_angle and marker_opacity."""
    fig = FigureWidgetResampler(default_n_shown_samples=100)

    # Create test data
    n = 1000
    x = np.arange(n)
    y = np.sin(x / 100)
    angles = np.linspace(0, 360, n)
    opacities = np.linspace(0.1, 1.0, n)

    # Add trace with marker_angle and marker_opacity
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers"),
        hf_marker_angle=angles,
        hf_marker_opacity=opacities,
    )

    # Check that properties are in hf_data
    assert "marker_angle" in fig.hf_data[0]
    assert "marker_opacity" in fig.hf_data[0]
    assert len(fig.hf_data[0]["marker_angle"]) == n
    assert len(fig.hf_data[0]["marker_opacity"]) == n


def test_marker_angle_opacity_configuration():
    """Test that marker_angle and marker_opacity are properly configured in DOWNSAMPLABLE_PROPERTIES."""
    from plotly_resampler.figure_resampler.figure_resampler_interface import (
        DOWNSAMPLABLE_PROPERTIES,
    )

    # Check that marker_angle and marker_opacity are in the configuration
    property_names = [prop[0] for prop in DOWNSAMPLABLE_PROPERTIES]
    assert "marker_angle" in property_names
    assert "marker_opacity" in property_names

    # Check that the configuration has the correct structure
    for prop_name, trace_path, hf_param_name in DOWNSAMPLABLE_PROPERTIES:
        if prop_name in ["marker_angle", "marker_opacity"]:
            assert (
                len(trace_path) == 2
            )  # Should be ["marker", "angle"] or ["marker", "opacity"]
            assert trace_path[0] == "marker"
            assert hf_param_name.startswith("hf_marker_")
