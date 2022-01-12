# -*- coding: utf-8 -*-
"""
Selenium wrapper class withholding methods for testing the plolty-figureResampler.

.. note::
    Headless mode is enabled by default.

"""

__author__ = "Jonas Van Der Donckt"

from os import stat
import time
from datetime import datetime, timedelta
from typing import List, Union

from seleniumwire import webdriver
from seleniumwire.request import Request
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# https://www.blazemeter.com/blog/improve-your-selenium-webdriver-tests-with-pytest
# and credate a parameterized driver.get method


class RequestParser:
    @staticmethod
    def filter_callback_requests(requests: List[Request]) -> List[Request]:
        valid_requests = []
        for r in requests:
            if r.method.upper() != 'POST':
                # note; the `_reload_hash` GET request will thus be filtered out
                continue

            if not r.url.endswith("_dash-update-component"):
                continue

            valid_requests.append(r)
        return valid_requests


class FigureResamplerGUITests:
    """Fetches data from Empatica's
    `E4Connect <https://www.empatica.com/connect/login.php>`_ platform
    """

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

    # def setup_create_driver(self):
    #     """Witholds logic to create and set up the driver"""
    #     # custom profile so we won't get a download prompt :)
    #     # self.profile = webdriver.FirefoxProfile()
    #     if self.headless:
    #         opts = Options()
    #         opts.headless = True
    #         print("Headless mode enabled")
    #         self.driver = webdriver.Firefox(
    #             options=opts,
    #             executable_path="/home/jonas/git/gIDLaB/plotly-dynamic-resampling/examples/geckodriver",
    #         )
    #     else:
    #         print("Headless mode disabled")
    #         self.driver = webdriver.Firefox(
    #             executable_path="./home/jonas/git/gIDLaB/plotly-dynamic-resampling/examples/geckodriver",
    #         )  # firefox_profile=self.profile)list_packages

    def go_to_page(self):
        """Navigate to FigureResampler page."""
        time.sleep(3)
        # if self.driver is None:
        #     self.setup_create_driver()
        self.driver.get("http://localhost:{}".format(self.port))
        self.on_page = True

    def clear_requests(self):
        del self.driver.requests

    def get_requests(self, delete: bool = True):
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
        actions.move_by_offset(xoffset=w * (x1 - x0), yoffset=h * (y1 - y0))
        actions.pause(0.1)
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
