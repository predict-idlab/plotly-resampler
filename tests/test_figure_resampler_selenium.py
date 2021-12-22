from .fr_selenium import FigureResamplerGUITests
import multiprocessing
import time



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
        # box based zooms
        fr.drag_and_zoom("xy", x0=0.25, x1=0.5, y0=0.25, y1=0.5)
        fr.drag_and_zoom("x2y2", x0=0.3, x1=0.7, y0=0.1, y1=1)
        time.sleep(0.5)
        del fr.driver.requests # clear the requests
        fr.drag_and_zoom("x2y2", x0=0.25, x1=0.5, y0=0.2, y1=0.2)
        time.sleep(0.5)
        fr.parse_requests(warn=True, delete=True)
        fr.click_legend_item('room 3')
        fr.parse_requests(warn=True, delete=True)


        # y remains the same - zoom horizontally
        fr.drag_and_zoom("x2y2", x0=0.25, x1=0.5, y0=0.3, y1=0.3)
        fr.drag_and_zoom("x3y3", x0=0.4, x1=0.5, y0=0.5, y1=0.5)

        fr.drag_and_zoom("x2y2", x0=0.3, x1=0.7, y0=0.1, y1=1)

        # y remains the same - zom horizontally
        fr.click_legend_item('room 3')
        fr.click_legend_item('room 2')
        fr.drag_and_zoom("x3y3", x0=0.4, x1=0.5, y0=0.5, y1=0.5)
        fr.drag_and_zoom("x3y3", x0=0.95, x1=0.5, y0=0.95, y1=0.95)
        fr.drag_and_zoom("x3y3", x0=0.05, x1=0.5, y0=0.95, y1=0.95)
        fr.drag_and_zoom("x2y2", x0=0.05, x1=0.5, y0=0.95, y1=0.95)
        # scale vertically
        fr.drag_and_zoom("x2y2", x0=0.01, x1=0.01, y0=0.01, y1=0.8)

        fr.autoscale()
        time.sleep(0.2)

        fr.reset_axes()
        time.sleep(0.2)
    except Exception as e:
        raise e
    finally:
        proc.terminate()
        driver.close()


# def test_gsr_gui(driver, gsr_figure):
#     from pytest_cov.embed import cleanup_on_sigterm

#     cleanup_on_sigterm()

#     port = 9032
#     proc = multiprocessing.Process(
#         target=gsr_figure.show_dash, kwargs=dict(mode="external", port=port)
#     )
#     proc.start()

#     try:
#         time.sleep(1)
#         fr = FigureResamplerGUITests(driver, port=port)

#         # box based zooms
#         fr.drag_and_zoom("xy", x0=0.25, x1=0.5, y0=0.25, y1=0.5)
#         fr.drag_and_zoom("x2y3", x0=0.3, x1=0.7, y0=0.1, y1=1)
#         fr.drag_and_zoom("x2y3", x0=0.25, x1=0.5, y0=0.2, y1=0.2)

#         fr.click_legend_item("EDA_Phasic")
#         time.sleep(0.1)
#         fr.click_legend_item("SCR peaks")
#         fr.reset_axes()

#         # y remains the same - zoom horizontally
#         fr.drag_and_zoom("x2y3", x0=0.25, x1=0.5, y0=0.3, y1=0.3)

#         # y remains the same - zom horizontally
#         fr.drag_and_zoom("xy", x0=0.4, x1=0.5, y0=0.5, y1=0.5)
#         fr.drag_and_zoom("x2y3", x0=0.95, x1=0.5, y0=0.95, y1=0.95)
#         fr.click_legend_item('EDA_lf_cleaned_tonic')
#         fr.drag_and_zoom("xy", x0=0.05, x1=0.5, y0=0.95, y1=0.95)
#         fr.drag_and_zoom("x2y3", x0=0.05, x1=0.5, y0=0.95, y1=0.95)
#         # scale vertically
#         fr.drag_and_zoom("x2y3", x0=0.01, x1=0.01, y0=0.01, y1=0.8)
#         fr.autoscale()
#         time.sleep(0.2)

#         fr.reset_axes()
#         time.sleep(0.2)
#     except Exception as e:
#         raise e
#     finally:
#         proc.terminate()
#         driver.close()

