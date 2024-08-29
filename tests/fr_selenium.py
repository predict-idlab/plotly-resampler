# -*- coding: utf-8 -*-
"""
Selenium wrapper class withholding methods for testing the plolty-figureResampler.

.. note::
    Headless mode is enabled by default.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt"

import json
import time
from typing import List, Union

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from seleniumwire import webdriver
from seleniumwire.request import Request

# Note: this will be used to add more waiting time to windows & mac os tests as
# - on these OS's serialization of the figure is necessary (to start the dash app in a
#    multiprocessing.Process)
#    https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
# - on linux, the browser (i.e., sending & getting requests) goes a lot faster
from .utils import not_on_linux

# https://www.blazemeter.com/blog/improve-your-selenium-webdriver-tests-with-pytest
# and create a parameterized driver.get method


class RequestParser:
    @staticmethod
    def filter_callback_requests(requests: List[Request]) -> List[Request]:
        valid_requests = []
        for r in requests:
            if r.method.upper() != "POST":
                # note; the `_reload_hash` GET request will thus be filtered out
                continue

            if not r.url.endswith("_dash-update-component"):
                continue

            valid_requests.append(r)
        return valid_requests

    def assert_fetch_data_request(
        data_request: Request, relayout_keys: List[str], n_updated_traces: int
    ):
        """Withholds checks for the relayout-data fetch request

        Parameters
        ----------
        data_request : Request
            The relayout data fetch request, with
            * Request body: the relayout changes
            * Response body: a list of dicts with first tiem
        relayout_keys : List[str]
            The expected keys to be found in the relayout command
        n_updated_traces : int
            The expected amount of traces which will be updated.

        """
        fetch_data_body = json.loads(data_request.body)
        assert "inputs" in fetch_data_body and len(fetch_data_body["inputs"]) == 1
        # verify that the request is triggered by the relayoutData
        figure_id = "resample-figure"
        assert fetch_data_body["inputs"][0]["id"] == figure_id
        assert fetch_data_body["inputs"][0]["property"] == "relayoutData"
        assert all(k in fetch_data_body["inputs"][0]["value"] for k in relayout_keys)
        # verify that the response is a list of dicts
        fetch_data_response_body = json.loads(data_request.response.body)["response"]
        # convert the updateData to a list of dicts
        updateData = fetch_data_response_body[figure_id]["figure"]["operations"]
        updated_traces = list(set(d["location"][1] for d in updateData))

        updated_x_keys = set(
            map(
                lambda d: d["location"][1],
                (filter(lambda x: x["location"][-1] == "x", updateData)),
            )
        )
        updated_y_keys = set(
            map(
                lambda d: d["location"][1],
                (filter(lambda x: x["location"][-1] == "y", updateData)),
            )
        )

        assert n_updated_traces == len(updated_traces)

        # verify that there are x and y updates for each trace
        assert len(updated_x_keys) == len(updated_y_keys) == n_updated_traces

    def assert_front_end_relayout_request(relayout_request: Request):
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

    def browser_independent_single_callback_request_assert(
        fr: FigureResamplerGUITests, relayout_keys: List[str], n_updated_traces: int
    ):
        """Verifies the callback requests on a browser-independent manner

        fr: FigureResamplerGUITests
            used for determining the browser-type.
        requests: List[Request]
            The captured requests of a SINGLE INTERACTION CALLBACK
        relayout_keys : List[str]
            The expected keys to be found in the relayout command
        n_updated_traces : int
            The expected amount of traces which will be updated.

        """
        # First, filter the requests to only retain the relevant ones
        requests = RequestParser.filter_callback_requests(fr.get_requests())

        browser_name = fr.driver.capabilities["browserName"]
        if "firefox" in browser_name:
            # There are 2 requests which are send
            # 1. first: changed-layout to server -> new data to back-end request
            # 2. the front-end relayout request
            assert len(requests) >= 1
            if len(requests) == 2:
                fetch_data_request, relayout_request = requests
                # RequestParser.assert_front_end_relayout_request(relayout_request)
            else:
                fetch_data_request = requests[0]

        elif "chrome" in browser_name:
            # for some, yet unknown reason, chrome does not seem to capture the
            # second front-end request.
            assert len(requests) == 1, f"len(requests) = {len(requests)}"
            fetch_data_request = requests[0]
        else:
            raise ValueError(f"invalid browser name {browser_name}")

        # Validate the update-data-callback request
        RequestParser.assert_fetch_data_request(
            fetch_data_request,
            relayout_keys=relayout_keys,
            n_updated_traces=n_updated_traces,
        )


class FigureResamplerGUITests:
    """Wrapper for performing figure-resampler GUI."""

    def __init__(self, driver: webdriver, port: int):
        """Construct an instance of A firefox selenium driver to fetch wearable data.

        Parameters
        ----------
        username : str
            The e4connect login username.
        password : str
            The e4connect password.
        save_dir : str
            The directory in which the data elements will be saved.
        headless: bool, default: True
            If set to `True` the driver will be ran in a headless mode.

        """
        self.port = port
        self.driver: Union[webdriver.Firefox, webdriver.Chrome] = driver
        self.on_page = False

    def go_to_page(self):
        """Navigate to FigureResampler page."""
        time.sleep(1)
        self.driver.get("http://localhost:{}".format(self.port))
        self.on_page = True
        if not_on_linux():
            time.sleep(7)  # bcs serialization of multiprocessing
        max_nb_tries = 3
        for _ in range(max_nb_tries):
            try:
                self.driver.find_element_by_id("resample-figure")
                break
            except Exception:
                time.sleep(5)

    def clear_requests(self, sleep_time_s=1):
        time.sleep(sleep_time_s)
        del self.driver.requests

    def get_requests(self, delete: bool = True):
        if not_on_linux():
            time.sleep(2)  # bcs slower browser
        requests = self.driver.requests
        if delete:
            self.clear_requests()

        return requests

    def drag_and_zoom(self, div_classname, x0=0.25, x1=0.5, y0=0.25, y1=0.5):
        """
        Drags and zooms the div with the given classname.

        Parameters
        ----------
        div_classname : str
            The classname of the div to be dragged and zoomed.
        x0 : float, default: 0.5
            The relative x-coordinate of the upper left corner of the div.
        x1 : float, default: 0.5
            The relative x-coordinate of the lower right corner of the div.
        y0 : float, default: 0.5
            The relative y-coordinate of the upper left corner of the div.
        y1 : float, default: 0.5
            The relative y-coordinate of the lower right corner of the div.

        """
        if not self.on_page:
            self.go_to_page()

        WebDriverWait(self.driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, div_classname))
        )

        subplot = self.driver.find_element(By.CLASS_NAME, div_classname)
        size = subplot.size
        w, h = size["width"], size["height"]

        actions = ActionChains(self.driver)
        actions.move_to_element_with_offset(subplot, xoffset=w * x0, yoffset=h * y0)
        actions.click_and_hold()
        actions.pause(0.2)
        actions.move_by_offset(xoffset=w * (x1 - x0), yoffset=h * (y1 - y0))
        actions.pause(0.2)
        actions.release()
        actions.pause(0.2)
        actions.perform()

    def _get_modebar_btns(self):
        if not self.on_page:
            self.go_to_page()

        WebDriverWait(self.driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, "modebar-group"))
        )
        return self.driver.find_elements(By.CLASS_NAME, "modebar-btn")

    def autoscale(self):
        for btn in self._get_modebar_btns():
            data_title = btn.get_attribute("data-title")
            if data_title == "Autoscale":
                ActionChains(self.driver).move_to_element(btn).click().perform()
                return

    def reset_axes(self):
        for btn in self._get_modebar_btns():
            data_title = btn.get_attribute("data-title")
            if data_title == "Reset axes":
                ActionChains(self.driver).move_to_element(btn).click().perform()
                return

    def click_legend_item(self, legend_name):
        WebDriverWait(self.driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, "modebar-group"))
        )
        for legend_item in self.driver.find_elements(By.CLASS_NAME, "legendtext"):
            if legend_name in legend_item.get_attribute("data-unformatted"):
                # move to the center of the item and click it
                (
                    ActionChains(self.driver)
                    .move_to_element(legend_item)
                    .pause(0.1)
                    .click()
                    .perform()
                )
                return

    # ------------------------------ DATA MODEL METHODS  ------------------------------
    def __del__(self):
        self.driver.close()
