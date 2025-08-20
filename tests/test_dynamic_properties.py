"""Tests for the dynamic property handling architecture."""

import numpy as np
import plotly.graph_objects as go
import pytest

from plotly_resampler import FigureResampler, FigureWidgetResampler


class TestDynamicProperties:
    """Test the dynamic property handling architecture."""

    def test_marker_symbol_support(self):
        """Test that marker_symbol is now supported as a downsamplable property."""
        fig = FigureResampler(default_n_shown_samples=100)

        # Create test data
        n = 1000
        x = np.arange(n)
        y = np.sin(x / 100)
        symbols = ["circle", "square", "diamond", "cross", "x"] * (n // 5)

        # Add trace with marker_symbol
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(symbol=symbols),
            )
        )

        # Check that marker_symbol is in hf_data
        assert "marker_symbol" in fig.hf_data[0]
        assert len(fig.hf_data[0]["marker_symbol"]) == n

        # Check that the downsampled trace has marker_symbol
        assert hasattr(fig.data[0].marker, "symbol")
        assert len(fig.data[0].marker.symbol) == 100  # default_n_shown_samples

    def test_hf_marker_symbol_parameter(self):
        """Test that hf_marker_symbol parameter works."""
        fig = FigureResampler(default_n_shown_samples=100)

        # Create test data
        n = 1000
        x = np.arange(n)
        y = np.sin(x / 100)
        symbols = ["circle", "square", "diamond", "cross", "x"] * (n // 5)

        # Add trace with hf_marker_symbol parameter
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
            ),
            hf_marker_symbol=symbols,
        )

        # Check that marker_symbol is in hf_data
        assert "marker_symbol" in fig.hf_data[0]
        assert len(fig.hf_data[0]["marker_symbol"]) == n

        # Check that the downsampled trace has marker_symbol
        assert hasattr(fig.data[0].marker, "symbol")
        assert len(fig.data[0].marker.symbol) == 100

    def test_multiple_dynamic_properties(self):
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
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
            ),
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

    def test_figurewidget_resampler_support(self):
        """Test that FigureWidgetResampler also supports dynamic properties."""
        fig = FigureWidgetResampler(default_n_shown_samples=100)

        # Create test data
        n = 1000
        x = np.arange(n)
        y = np.sin(x / 100)
        symbols = (["circle", "square", "diamond"] * ((n // 3) + 1))[:n]

        # Add trace with marker_symbol
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(symbol=symbols),
            )
        )

        # Check that marker_symbol is in hf_data
        assert "marker_symbol" in fig.hf_data[0]
        assert len(fig.hf_data[0]["marker_symbol"]) == n

    def test_property_priority(self):
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
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(symbol=trace_symbols),
            ),
            hf_marker_symbol=hf_symbols,
        )

        # Check that hf parameter took priority
        assert np.all(fig.hf_data[0]["marker_symbol"] == hf_symbols)

    def test_none_properties(self):
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

    def test_mixed_property_types(self):
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

    def test_data_point_selector_requirement(self):
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
        with pytest.raises(
            AssertionError, match="Only DataPointSelector can downsample"
        ):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(symbol=symbols),
                )
            )

    def test_extensibility(self):
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

    def test_property_path_handling(self):
        """Test that nested property paths are handled correctly."""
        fig = FigureResampler(default_n_shown_samples=100)

        # Create test data
        n = 1000
        x = np.arange(n)
        y = np.sin(x / 100)

        # Test with deeply nested property (marker.symbol)
        # Make sure the symbols array has exactly n elements
        symbols = (["circle", "square", "diamond"] * ((n // 3) + 1))[:n]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(symbol=symbols),
            )
        )

        # Check that the nested property was handled correctly
        assert "marker_symbol" in fig.hf_data[0]
        assert len(fig.hf_data[0]["marker_symbol"]) == n

        # Check that the downsampled trace has the correct nested structure
        trace = fig.data[0]
        assert hasattr(trace.marker, "symbol")
        assert len(trace.marker.symbol) == 100
