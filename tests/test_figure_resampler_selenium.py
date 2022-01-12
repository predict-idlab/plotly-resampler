import multiprocessing
import time

import json

from seleniumwire.request import Response

from .fr_selenium import FigureResamplerGUITests, RequestParser
import warnings


def test_basic_example_gui(driver, example_figure):
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()

    port = 9012
    proc = multiprocessing.Process(
        target=example_figure.show_dash, kwargs=dict(mode="external", port=port)
    )
    proc.start()
    try:
        time.sleep(1)
        fr = FigureResamplerGUITests(driver, port=port)

        # First, apply some box based zooms
        fr.drag_and_zoom("xy", x0=0.25, x1=0.5, y0=0.25, y1=0.5)
        fr.drag_and_zoom("x2y2", x0=0.3, x1=0.7, y0=0.1, y1=1)
        time.sleep(1)

        # Clear the requests till now
        fr.clear_requests()
        # Perform a zoom operation, and capture the request output
        fr.drag_and_zoom("x2y2", x0=0.25, x1=0.5, y0=0.2, y1=0.2)
        time.sleep(1)
        requests = RequestParser.filter_callback_requests(fr.get_requests())
        # There are 2 requests which are send
        # 1. first: changed-layout to server -> new data to back-end request
        # 2. the front-end relayout request
        assert len(requests) == 2
        fetch_data_request, relayout_request = requests
        # 1. Verify the fetch data request
        fetch_data_body = json.loads(fetch_data_request.body)
        assert "inputs" in fetch_data_body and len(fetch_data_body["inputs"]) == 1
        assert fetch_data_body["inputs"][0]["id"] == "resample-figure"
        assert fetch_data_body["inputs"][0]["property"] == "relayoutData"
        assert all(
            k in fetch_data_body["inputs"][0]["value"]
            for k in ["xaxis2.range[0]", "xaxis2.range[1]"]
        )
        fetch_data_response_body = json.loads(fetch_data_request.response.body)[
            "response"
        ]
        updateData = fetch_data_response_body["trace-updater"]["updateData"]
        # in this case; the length of updateData is 2 as
        # (1) only the sin-wave is updated ÁND (2) relayout data is always sent first
        assert len(updateData) == 2
        # verify that the layout update does not contain trace props
        assert "x" not in updateData[0]
        assert "y" not in updateData[0]

        # verify that the trace-update does not contain layout update props
        assert "x" in updateData[1]
        assert "y" in updateData[1]
        # As identifier, we always send the trace-index
        # (i.e. the the position of the trace in the `trace-list`)
        assert "index" in updateData[1]

        # 2. Verify the relayout request
        relayout_body = json.loads(relayout_request.body)
        assert "inputs" in relayout_body and len(relayout_body["inputs"]) == 1
        assert relayout_body["inputs"][0]["id"] == "resample-figure"
        assert relayout_body["inputs"][0]["property"] == "relayoutData"
        assert all(
            k in relayout_body["inputs"][0]["value"]
            for k in ["annotations", "template", "title", "legend", "xaxis", "yaxis"]
        )

        relayout_response_body = json.loads(relayout_request.response.body)["response"]
        # the relayout response its updateData should be an empty dict
        # { "response": { "trace-updater": { "updateData": [ {} ] } } }
        updateData = relayout_response_body["trace-updater"]["updateData"]
        assert len(updateData) == 1
        assert updateData[0] == {}

        # just a legend toggle operation, no new requests should be made
        fr.clear_requests()
        fr.click_legend_item("room 3")
        time.sleep(1)
        assert len(RequestParser.filter_callback_requests(fr.get_requests())) == 0

        # y remains the same - zoom horizontally
        fr.drag_and_zoom("x2y2", x0=0.25, x1=0.5, y0=0.3, y1=0.3)
        fr.drag_and_zoom("x3y3", x0=0.4, x1=0.5, y0=0.5, y1=0.5)

        fr.drag_and_zoom("x2y2", x0=0.3, x1=0.7, y0=0.1, y1=1)

        # y remains the same - zom horizontally
        fr.click_legend_item("room 3")
        fr.click_legend_item("room 2")
        fr.drag_and_zoom("x3y3", x0=0.4, x1=0.5, y0=0.5, y1=0.5)
        fr.drag_and_zoom("x3y3", x0=0.95, x1=0.5, y0=0.95, y1=0.95)
        fr.drag_and_zoom("x3y3", x0=0.05, x1=0.5, y0=0.95, y1=0.95)
        fr.drag_and_zoom("x2y2", x0=0.05, x1=0.5, y0=0.95, y1=0.95)
        # scale vertically
        fr.drag_and_zoom("x2y2", x0=0.01, x1=0.01, y0=0.01, y1=0.8)

        # check whether the autoscale command updates the data
        fr.autoscale()
        time.sleep(0.2)

        fr.reset_axes()
        time.sleep(0.2)
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

        # box based zooms
        fr.drag_and_zoom("xy", x0=0.25, x1=0.5, y0=0.25, y1=0.5)
        fr.drag_and_zoom("x2y3", x0=0.3, x1=0.7, y0=0.1, y1=1)
        fr.drag_and_zoom("x2y3", x0=0.25, x1=0.5, y0=0.2, y1=0.2)

        fr.click_legend_item("EDA_Phasic")
        time.sleep(0.1)
        fr.click_legend_item("SCR peaks")
        fr.reset_axes()

        # y remains the same - zoom horizontally
        fr.drag_and_zoom("x2y3", x0=0.25, x1=0.5, y0=0.3, y1=0.3)

        # y remains the same - zom horizontally
        fr.drag_and_zoom("xy", x0=0.4, x1=0.5, y0=0.5, y1=0.5)
        fr.drag_and_zoom("x2y3", x0=0.95, x1=0.5, y0=0.95, y1=0.95)
        fr.click_legend_item('EDA_lf_cleaned_tonic')
        fr.drag_and_zoom("xy", x0=0.05, x1=0.5, y0=0.95, y1=0.95)
        fr.drag_and_zoom("x2y3", x0=0.05, x1=0.5, y0=0.95, y1=0.95)
        # scale vertically
        fr.drag_and_zoom("x2y3", x0=0.01, x1=0.01, y0=0.01, y1=0.8)
        fr.autoscale()
        time.sleep(0.2)

        fr.reset_axes()
        time.sleep(0.2)
    except Exception as e:
        raise e
    finally:
        proc.terminate()
