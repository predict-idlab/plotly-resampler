import multiprocessing
import time

from plotly_resampler.figure_resampler import FigureResampler

from .fr_selenium import FigureResamplerGUITests, RequestParser


def test_multiple_tz(driver, multiple_tz_figure):
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()

    port = 9012
    proc = multiprocessing.Process(
        target=multiple_tz_figure.show_dash, kwargs=dict(mode="external", port=port)
    )
    proc.start()
    try:
        time.sleep(1)
        fr = FigureResamplerGUITests(driver, port=port)
        time.sleep(1)
        fr.go_to_page()

        # First, apply some box based zooms
        fr.drag_and_zoom("xy", x0=0.25, x1=0.5, y0=0.25, y1=0.5)
        fr.drag_and_zoom("x2y2", x0=0.3, x1=0.7, y0=0.1, y1=1)

        # Clear the requests till now
        fr.clear_requests(sleep_time_s=1)
        # Perform a zoom operation, and capture the request output
        fr.drag_and_zoom("x2y2", x0=0.25, x1=0.5, y0=0.2, y1=0.2)
        time.sleep(1)
        # 1. Verify the fetch data request
        RequestParser.browser_independent_single_callback_request_assert(
            fr, relayout_keys=["xaxis2.range[0]", "xaxis2.range[1]"], n_updated_traces=5
        )

        # The reset axes autoscales AND resets tot he global data view -> all data
        # will be updated.
        fr.clear_requests()
        fr.reset_axes()
        time.sleep(1)
        RequestParser.browser_independent_single_callback_request_assert(
            fr,
            relayout_keys=[
                "xaxis.autorange",
                "xaxis2.autorange",
                "xaxis3.autorange",
                "xaxis4.autorange",
                "xaxis5.autorange",
                "yaxis.autorange",
                "yaxis2.autorange",
                "yaxis3.autorange",
                "yaxis4.autorange",
                "yaxis5.autorange",
            ],
            n_updated_traces=5,
        )

        # we autoscale to the current front-end view, no updated dat will be sent from
        # the server to the front-end, however, a callback will still be made, but
        # will return a 204 response status (no content), as we do not need new data
        # to autoscale to the current front-end view.
        fr.clear_requests(sleep_time_s=1)
        fr.autoscale()
        time.sleep(1)
        autoscale_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(autoscale_requests) == 1
        assert autoscale_requests[0].response.status_code == 204

        if len(driver.get_log("browser")) > 0:  # Check no errors in the browser
            for entry in driver.get_log("browser"):
                print(entry)
                if not entry["level"] == "INFO":
                    # Only WebGL warnings are allowed
                    assert entry["level"] == "warning"
                    assert entry["message"].contains("WebGL")
    except Exception as e:
        raise e
    finally:
        proc.terminate()
        multiple_tz_figure.stop_server()


def test_basic_example_gui(driver, example_figure):
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()

    port = 9012
    config = {"displayModeBar": True}
    proc = multiprocessing.Process(
        target=example_figure.show_dash,
        kwargs=dict(mode="external", port=port, config=config),
    )
    proc.start()
    try:
        time.sleep(1)
        fr = FigureResamplerGUITests(driver, port=port)
        time.sleep(1)
        fr.go_to_page()

        # First, apply some box based zooms
        fr.drag_and_zoom("xy", x0=0.25, x1=0.5, y0=0.25, y1=0.5)
        fr.drag_and_zoom("x2y2", x0=0.3, x1=0.7, y0=0.1, y1=1)

        # Clear the requests till now
        fr.clear_requests(sleep_time_s=1)
        # Perform a zoom operation, and capture the request output
        fr.drag_and_zoom("x2y2", x0=0.25, x1=0.5, y0=0.2, y1=0.2)
        time.sleep(1)
        # 1. Verify the fetch data request
        RequestParser.browser_independent_single_callback_request_assert(
            fr, relayout_keys=["xaxis2.range[0]", "xaxis2.range[1]"], n_updated_traces=1
        )

        # A legend toggle operation
        # This does not trigger the relayout callback, no new requests should be made
        fr.clear_requests()
        fr.click_legend_item("room 3")
        time.sleep(1)
        assert len(RequestParser.filter_callback_requests(fr.get_requests())) == 0

        # y remains the same - zoom horizontally
        fr.drag_and_zoom("x2y2", x0=0.25, x1=0.5, y0=0.3, y1=0.3)

        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("x3y3", x0=0.4, x1=0.5, y0=0.5, y1=0.5)
        time.sleep(1)
        RequestParser.browser_independent_single_callback_request_assert(
            fr, relayout_keys=["xaxis3.range[0]", "xaxis3.range[1]"], n_updated_traces=3
        )

        fr.drag_and_zoom("x2y2", x0=0.3, x1=0.7, y0=0.1, y1=1)

        # y remains the same - zoom horizontally
        fr.click_legend_item("room 3")
        fr.click_legend_item("room 2")
        fr.drag_and_zoom("x3y3", x0=0.4, x1=0.5, y0=0.5, y1=0.5)
        fr.drag_and_zoom("x3y3", x0=0.95, x1=0.5, y0=0.95, y1=0.95)
        fr.drag_and_zoom("x3y3", x0=0.05, x1=0.5, y0=0.95, y1=0.95)
        fr.drag_and_zoom("x2y2", x0=0.05, x1=0.5, y0=0.95, y1=0.95)

        # scale vertically -
        # This will trigger a relayout callback, however as only y-values are updated,
        # no new data points will be send to the front-end and a - NO CONTENT response
        # will be returned.
        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("x2y2", x0=0.5, x1=0.5, y0=0.1, y1=0.5)
        time.sleep(1)
        vertical_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(vertical_requests) == 1
        assert vertical_requests[0].response.status_code == 204

        # we autoscale to the current front-end view, no updated dat will be sent from
        # the server to the front-end, however, a callback will still be made, but
        # will return a 204 response status (no content), as we do not need new data
        # to autoscale to the current front-end view.
        fr.clear_requests(sleep_time_s=1)
        fr.autoscale()
        time.sleep(1)
        autoscale_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(autoscale_requests) == 1
        assert autoscale_requests[0].response.status_code == 204

        # The reset axes autoscales AND resets tot he global data view -> all data
        # will be updated.
        fr.clear_requests()
        fr.reset_axes()
        time.sleep(1)
        RequestParser.browser_independent_single_callback_request_assert(
            fr,
            relayout_keys=[
                "xaxis.autorange",
                "xaxis2.autorange",
                "xaxis3.autorange",
                "xaxis.showspikes",
            ],
            n_updated_traces=5,
        )

        if len(driver.get_log("browser")) > 0:  # Check no errors in the browser
            for entry in driver.get_log("browser"):
                print(entry)
                if not entry["level"] == "INFO":
                    # Only WebGL warnings are allowed
                    assert entry["level"] == "warning"
                    assert entry["message"].contains("WebGL")
    except Exception as e:
        raise e
    finally:
        proc.terminate()


def test_basic_example_gui_existing(driver, example_figure_fig):
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()

    fig = FigureResampler(
        resampled_trace_prefix_suffix=(
            '<b style="color:sandybrown">[R]</b>',
            '<b style="color:darkcyan">[R]</b>',
        )
    )
    fig.replace(example_figure_fig)

    port = 9012
    proc = multiprocessing.Process(
        target=fig.show_dash, kwargs=dict(mode="external", port=port)
    )
    proc.start()
    try:
        time.sleep(1)
        fr = FigureResamplerGUITests(driver, port=port)
        time.sleep(1)
        fr.go_to_page()

        # First, apply some box based zooms
        fr.drag_and_zoom("xy", x0=0.25, x1=0.5, y0=0.25, y1=0.5)
        fr.drag_and_zoom("x2y2", x0=0.3, x1=0.7, y0=0.1, y1=1)

        # Clear the requests till now
        fr.clear_requests(sleep_time_s=1)
        # Perform a zoom operation, and capture the request output
        fr.drag_and_zoom("x2y2", x0=0.25, x1=0.5, y0=0.2, y1=0.2)
        time.sleep(1)
        # 1. Verify the fetch data request
        RequestParser.browser_independent_single_callback_request_assert(
            fr, relayout_keys=["xaxis2.range[0]", "xaxis2.range[1]"], n_updated_traces=1
        )

        # A legend toggle operation
        # This does not trigger the relayout callback, no new requests should be made
        fr.clear_requests()
        fr.click_legend_item("room 3")
        time.sleep(1)
        assert len(RequestParser.filter_callback_requests(fr.get_requests())) == 0

        # y remains the same - zoom horizontally
        fr.drag_and_zoom("x2y2", x0=0.25, x1=0.5, y0=0.3, y1=0.3)

        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("x3y3", x0=0.4, x1=0.5, y0=0.5, y1=0.5)
        time.sleep(1)
        RequestParser.browser_independent_single_callback_request_assert(
            fr, relayout_keys=["xaxis3.range[0]", "xaxis3.range[1]"], n_updated_traces=3
        )

        fr.drag_and_zoom("x2y2", x0=0.3, x1=0.7, y0=0.1, y1=1)

        # y remains the same - zoom horizontally
        fr.click_legend_item("room 3")
        fr.click_legend_item("room 2")
        fr.drag_and_zoom("x3y3", x0=0.4, x1=0.5, y0=0.5, y1=0.5)
        fr.drag_and_zoom("x3y3", x0=0.05, x1=0.5, y0=0.95, y1=0.95)

        # scale vertically -
        # This will trigger a relayout callback, however as only y-values are updated,
        # no new data points will be send to the front-end and a - NO CONTENT response
        # will be returned.
        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("x2y2", x0=0.5, x1=0.5, y0=0.1, y1=0.5)
        time.sleep(1)
        vertical_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(vertical_requests) == 1
        assert vertical_requests[0].response.status_code == 204

        # we autoscale to the current front-end view, no updated dat will be sent from
        # the server to the front-end, however, a callback will still be made, but
        # will return a 204 response status (no content), as we do not need new data
        # to autoscale to the current front-end view.
        fr.clear_requests(sleep_time_s=1)
        fr.autoscale()
        time.sleep(1)
        autoscale_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(autoscale_requests) == 1
        assert autoscale_requests[0].response.status_code == 204

        # The reset axes autoscales AND resets tot he global data view -> all data
        # will be updated.
        fr.clear_requests()
        fr.reset_axes()
        time.sleep(1)
        RequestParser.browser_independent_single_callback_request_assert(
            fr,
            relayout_keys=[
                "xaxis.autorange",
                "xaxis2.autorange",
                "xaxis3.autorange",
                "xaxis.showspikes",
            ],
            n_updated_traces=5,
        )

        if len(driver.get_log("browser")) > 0:  # Check no errors in the browser
            for entry in driver.get_log("browser"):
                print(entry)
                if not entry["level"] == "INFO":
                    # Only WebGL warnings are allowed
                    assert entry["level"] == "warning"
                    assert entry["message"].contains("WebGL")
    except Exception as e:
        raise e
    finally:
        proc.terminate()


def test_gsr_gui(driver, gsr_figure):
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()

    port = 9032
    proc = multiprocessing.Process(
        target=gsr_figure.show_dash, kwargs=dict(mode="external", port=port)
    )
    proc.start()

    try:
        time.sleep(1)
        fr = FigureResamplerGUITests(driver, port=port)
        time.sleep(1)
        fr.go_to_page()

        # box based zooms
        fr.drag_and_zoom("xy", x0=0.25, x1=0.5, y0=0.25, y1=0.5)
        fr.drag_and_zoom("x2y3", x0=0.3, x1=0.7, y0=0.1, y1=1)

        # Note: we have shared-xaxes so all traces will be updated using this command
        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("x2y3", x0=0.25, x1=0.5, y0=0.2, y1=0.2)
        time.sleep(1)
        RequestParser.browser_independent_single_callback_request_assert(
            fr=fr,
            relayout_keys=[
                "xaxis.range[0]",
                "xaxis.range[1]",
                "xaxis2.range[0]",
                "xaxis2.range[1]",
            ],
            n_updated_traces=7,
        )

        # A toggle operation
        # This does not trigger the relayout callback, no new requests should be made
        fr.clear_requests(sleep_time_s=1)
        fr.click_legend_item("EDA_Phasic")
        time.sleep(0.2)
        fr.click_legend_item("SCR peaks")
        time.sleep(1)
        assert len(RequestParser.filter_callback_requests(fr.get_requests())) == 0

        # A reset axes operation resets the front-end view to the global data view
        fr.clear_requests(sleep_time_s=1)
        fr.reset_axes()
        time.sleep(1)
        RequestParser.browser_independent_single_callback_request_assert(
            fr=fr,
            relayout_keys=[
                "xaxis.autorange",
                "xaxis2.autorange",
                "xaxis.showspikes",
                "xaxis2.showspikes",
            ],
            n_updated_traces=7,
        )

        # y remains the same - zoom horizontally
        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("x2y3", x0=0.25, x1=0.5, y0=0.3, y1=0.3)

        # y remains the same - zom horizontally
        fr.drag_and_zoom("xy", x0=0.4, x1=0.5, y0=0.5, y1=0.5)
        fr.drag_and_zoom("x2y3", x0=0.95, x1=0.5, y0=0.95, y1=0.95)
        fr.click_legend_item("EDA_lf_cleaned_tonic")
        fr.drag_and_zoom("xy", x0=0.05, x1=0.5, y0=0.95, y1=0.95)
        fr.drag_and_zoom("x2y3", x0=0.05, x1=0.5, y0=0.95, y1=0.95)

        # scale vertically
        # This will trigger a relayout callback, however as only y-values are updated,
        # no new data points will be send to the front-end and a - NO CONTENT response
        # will be returned.
        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("x2y3", x0=0.2, x1=0.2, y0=0.1, y1=0.5)
        time.sleep(1)
        vertical_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(vertical_requests) == 1
        assert vertical_requests[0].response.status_code == 204

        # autoscale
        # we autoscale to the current front-end view, no updated dat will be sent from
        # the server to the front-end, however, a callback will still be made, but
        # will return a 204 response status (no content), as we do not need new data
        # to autoscale to the current front-end view.
        fr.clear_requests(sleep_time_s=1)
        fr.autoscale()
        time.sleep(1)
        autoscale_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(autoscale_requests) == 1
        assert autoscale_requests[0].response.status_code == 204

        fr.reset_axes()
        time.sleep(0.2)

        if len(driver.get_log("browser")) > 0:  # Check no errors in the browser
            for entry in driver.get_log("browser"):
                print(entry)
                if not entry["level"] == "INFO":
                    # Only WebGL warnings are allowed
                    assert entry["level"] == "warning"
                    assert entry["message"].contains("WebGL")
    except Exception as e:
        raise e
    finally:
        proc.terminate()


def test_cat_gui(driver, cat_series_box_hist_figure):
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()

    port = 9034
    proc = multiprocessing.Process(
        target=cat_series_box_hist_figure.show_dash,
        kwargs=dict(mode="external", port=port),
    )
    proc.start()
    try:
        time.sleep(1)
        fr = FigureResamplerGUITests(driver, port=port)
        time.sleep(1)
        fr.go_to_page()

        # First, apply some horizontal based zooms
        fr.drag_and_zoom("xy", x0=0.1, x1=0.5, y0=0.5, y1=0.5)
        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("xy", x0=0.1, x1=0.5, y0=0.5, y1=0.5)
        time.sleep(1)
        RequestParser.browser_independent_single_callback_request_assert(
            fr=fr,
            relayout_keys=["xaxis.range[0]", "xaxis.range[1]"],
            n_updated_traces=1,
        )

        # The right top subplot withholds a boxplot, which is NOT a high-freq trace
        # so no new data should be send from the server to the client -> 204 status code
        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("x2y2", x0=0.3, x1=0.7, y0=0.2, y1=0.8)
        time.sleep(1)
        vertical_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(vertical_requests) == 1
        assert vertical_requests[0].response.status_code == 204

        fr.clear_requests(sleep_time_s=1)
        fr.autoscale()
        time.sleep(1)
        autoscale_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(autoscale_requests) == 1
        assert autoscale_requests[0].response.status_code == 204

        # Note: as there is only 1 hf-scatter-trace, the reset axes command will only
        # update a single trace
        fr.clear_requests(sleep_time_s=1)
        fr.reset_axes()
        time.sleep(1)
        RequestParser.browser_independent_single_callback_request_assert(
            fr=fr,
            relayout_keys=[
                "xaxis.autorange",
                "xaxis.showspikes",
                "xaxis2.autorange",
                "xaxis2.showspikes",
                "xaxis3.autorange",
                "xaxis3.showspikes",
                "yaxis.autorange",
                "yaxis.showspikes",
                "yaxis2.autorange",
                "yaxis2.showspikes",
                "yaxis3.autorange",
                "yaxis3.showspikes",
            ],
            n_updated_traces=1,
        )

        if len(driver.get_log("browser")) > 0:  # Check no errors in the browser
            for entry in driver.get_log("browser"):
                print(entry)
                if not entry["level"] == "INFO":
                    # Only WebGL warnings are allowed
                    assert entry["level"] == "warning"
                    assert entry["message"].contains("WebGL")
    except Exception as e:
        raise e
    finally:
        proc.terminate()


def test_shared_hover_gui(driver, shared_hover_figure):
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()

    port = 9038
    proc = multiprocessing.Process(
        target=shared_hover_figure.show_dash,
        kwargs=dict(mode="external", port=port),
    )
    proc.start()
    try:
        time.sleep(1)
        fr = FigureResamplerGUITests(driver, port=port)
        time.sleep(1)
        fr.go_to_page()

        # First, apply some horizontal based zooms
        fr.drag_and_zoom("x3y", x0=0.1, x1=0.5, y0=0.5, y1=0.5)
        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("x3y", x0=0.1, x1=0.5, y0=0.5, y1=0.5)
        time.sleep(1)
        # As all axes are shared, we expect at least 3 updated
        RequestParser.browser_independent_single_callback_request_assert(
            fr=fr,
            relayout_keys=["xaxis3.range[0]", "xaxis3.range[1]"],
            n_updated_traces=3,
        )

        # Note: as there is only 1 hf-scatter-trace, the reset axes command will only
        # update a single trace
        fr.clear_requests(sleep_time_s=1)
        fr.reset_axes()
        time.sleep(1)
        RequestParser.browser_independent_single_callback_request_assert(
            fr=fr,
            relayout_keys=[
                "xaxis3.autorange",
                "xaxis3.showspikes",
                "yaxis.autorange",
                "yaxis.showspikes",
                "yaxis2.autorange",
                "yaxis2.showspikes",
                "yaxis3.autorange",
                "yaxis3.showspikes",
            ],
            n_updated_traces=3,
        )

        fr.drag_and_zoom("x3y2", x0=0.1, x1=0.5, y0=0.5, y1=0.5)

       # First, apply some horizontal based zooms
        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("x3y3", x0=0.1, x1=0.5, y0=0.5, y1=0.5)
        time.sleep(1)
        # As all axes are shared, we expect at least 3 updated
        RequestParser.browser_independent_single_callback_request_assert(
            fr=fr,
            relayout_keys=["xaxis3.range[0]", "xaxis3.range[1]"],
            n_updated_traces=3,
        )

        fr.clear_requests(sleep_time_s=1)
        fr.autoscale()
        time.sleep(1)
        autoscale_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(autoscale_requests) == 1
        assert autoscale_requests[0].response.status_code == 204

        if len(driver.get_log("browser")) > 0:  # Check no errors in the browser
            for entry in driver.get_log("browser"):
                print(entry)
                if not entry["level"] == "INFO":
                    # Only WebGL warnings are allowed
                    assert entry["level"] == "warning"
                    assert entry["message"].contains("WebGL")
    except Exception as e:
        raise e
    finally:
        proc.terminate()


def test_multi_trace_go_figure(driver, multi_trace_go_figure):
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()

    port = 9038
    proc = multiprocessing.Process(
        target=multi_trace_go_figure.show_dash,
        kwargs=dict(mode="external", port=port),
    )
    proc.start()
    try:
        time.sleep(1)
        fr = FigureResamplerGUITests(driver, port=port)
        time.sleep(1)
        fr.go_to_page()

        # First, apply some horizontal based zooms
        fr.drag_and_zoom("xy", x0=0.1, x1=0.5, y0=0.5, y1=0.5)
        fr.clear_requests(sleep_time_s=3)
        fr.drag_and_zoom("xy", x0=0.1, x1=0.5, y0=0.5, y1=0.5)
        time.sleep(3)
        # As all axes are shared, we expect at least 3 updated
        RequestParser.browser_independent_single_callback_request_assert(
            fr=fr,
            relayout_keys=["xaxis.range[0]", "xaxis.range[1]"],
            n_updated_traces=30,
        )

        # Note: as there is only 1 hf-scatter-trace, the reset axes command will only
        # update a single trace
        fr.clear_requests(sleep_time_s=1)
        fr.reset_axes()
        time.sleep(3)
        RequestParser.browser_independent_single_callback_request_assert(
            fr=fr,
            relayout_keys=[
                "xaxis.autorange",
                "xaxis.showspikes",
                "yaxis.autorange",
                "yaxis.showspikes",
            ],
            n_updated_traces=30,
        )

        fr.drag_and_zoom("xy", x0=0.1, x1=0.3, y0=0.6, y1=0.9)
        fr.clear_requests(sleep_time_s=3)

       # First, apply some horizontal based zooms
        fr.clear_requests(sleep_time_s=1)
        fr.drag_and_zoom("xy", x0=0.1, x1=0.2, y0=0.5, y1=0.5)
        time.sleep(3)
        # As all axes are shared, we expect at least 3 updated
        RequestParser.browser_independent_single_callback_request_assert(
            fr=fr,
            relayout_keys=["xaxis.range[0]", "xaxis.range[1]"],
            n_updated_traces=30,
        )

        fr.clear_requests(sleep_time_s=3)
        fr.autoscale()
        time.sleep(3)
        autoscale_requests = RequestParser.filter_callback_requests(fr.get_requests())
        assert len(autoscale_requests) == 1
        assert autoscale_requests[0].response.status_code == 204

        if len(driver.get_log("browser")) > 0:  # Check no errors in the browser
            for entry in driver.get_log("browser"):
                print(entry)
                if not entry["level"] == "INFO":
                    # Only WebGL warnings are allowed
                    assert entry["level"] == "warning"
                    assert entry["message"].contains("WebGL")
    except Exception as e:
        raise e
    finally:
        proc.terminate()


def test_multi_trace_go_figure_updated_xrange(driver, multi_trace_go_figure):
    # This test checks that the xaxis range is updated when the xaxis range is set
    # Notet hat this test just hits the .show_dash() method
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()

    multi_trace_go_figure.update_xaxes(range=[100, 200_000])

    port = 9038
    proc = multiprocessing.Process(
        target=multi_trace_go_figure.show_dash,
        kwargs=dict(mode="external", port=port),
    )
    proc.start()
    try:
        # Just hit the code of the .show_dash method when an x-range is set
        time.sleep(1)
        fr = FigureResamplerGUITests(driver, port=port)
        time.sleep(1)
        fr.go_to_page()

    except Exception as e:
        raise e
    finally:
        proc.terminate()
