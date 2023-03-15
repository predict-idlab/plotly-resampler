import multiprocessing
import time
import subprocess as sp
import os
import signal
import psutil as ps

import json
import numpy as np

import plotly.graph_objects as go
from plotly_resampler import FigureResampler
from fr_selenium import FigureResamplerGUITests



from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from seleniumwire import webdriver
from webdriver_manager.chrome import ChromeDriverManager, ChromeType
from selenium.webdriver.chrome.service import Service as ChromeService


# create a test for each value of n_traces, n_datapoints and shown_datapoints
        # open new page
        # loop over a range of percentages (% of shown traces)
            # start timer (in front end via selenium? performance library js)
            # apply 50% range zoom
            # stop timer when visible update
            # start another timer for invisible 
            # stop timer when invisible renders
            # return to original range (may trigger timer in front end... prevent this!!)
        # extract logs from this iteration into a file
        # close page! 

# d = driver()
options = Options()
d = DesiredCapabilities.CHROME
d["goog:loggingPrefs"] = {"browser": "ALL"}
driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()),
            # service_args=["--verbose", "--log-path=C:\\Users\\willi\\Documents\\ISIS\\Thesis\\plotly-resampler\\logs"],
            options=options,
            desired_capabilities=d,
        )
port = 8050
fr = FigureResamplerGUITests(driver, port=port)

percentages_hidden = np.array([0, 0.2, 0.5, 0.8, 0.9])
n_traces = [10, 20, 50]
n_datapoints = [
                 100_000,
                 1_000_000,
                 10_000_000
                 ]
n_shown_datapoints = [
                      100,
                      1000,
                      4000
                    ]

try:      
    for t in n_traces:
        for n in n_datapoints:
            for s in n_shown_datapoints:
                time.sleep(2)
                proc = sp.Popen(['poetry','run','python','./tests/minimal_variable_threads.py', '-n', str(n), '-s', str(s), '-t', str(t)], 
                                # creationflags=sp.CREATE_NEW_CONSOLE
                                )
                print(f'n_traces: {t}')
                print(f'n_datapoints: {n}')
                print(f'n_shown_datapoints: {s}')
                try:
                    time.sleep(20)
                    fr.go_to_page()
                    
                    time.sleep(1)

                    # determine the number of traces that will be hidden corresponding to each percentage
                    n_traces_hidden = np.unique(np.ceil(t*percentages_hidden)).astype(int)
                    # TODO: get final list of percentages (visible!) and print to console

                    # print(n_traces_hidden)
                    last = t
                    for idx, j in enumerate(n_traces_hidden):
                        if idx == 0:
                            previous_n_hidden = 0
                        else:
                            previous_n_hidden = n_traces_hidden[idx-1]
                            # hide r traces from the last hidden trace
                        driver.execute_script(f'console.log("{100-((j/t)*100)}%")')
                        print(previous_n_hidden) 
                        residual = n_traces_hidden[idx]-previous_n_hidden
                        print(residual)
                        residual_indices = [int(last-(i+1)) for i in range(residual)]
                        last -= residual
                        if residual_indices != []:
                            fr.hide_legend_restyle(residual_indices)

                        # after hiding the traces, (start the timer,) zoom in, then reset the axes for the next iteration 
                        fr.drag_and_zoom("xy", x0=0.25, x1=0.75, y0=0.5, y1=0.5, testing=True)
                        #start timer
                        # fr.start_timer('zoom')

                        time.sleep(5)
                        fr.reset_axes(testing=True)
                        # fr.start_timer('reset')
                        time.sleep(5)
                    with open(f'./logs/n{n}_s{s}_t{t}_everynth.json', 'w') as logfile:
                        for log in driver.get_log('browser'):
                            logfile.write(json.dumps(log)) 
                    print('done saving log')  
                    # print(logs)
                    # print(type(logs))
                except Exception as e:
                        raise e
                finally:
                    # print(proc.pid)
                    # p = ps.Process(proc.pid)
                    # print(f'pid {proc.pid}')
                    # print(f'process is running {p.is_running()}')
                    # proc.send_signal(signal.CTRL_C_EVENT)
                    
                    #this works with windows! add if clause for Linux version! (proc.kill works?)
                    os.system("TASKKILL /F /T /PID " + str(proc.pid))
                    os.system('killport 8050 --view-only')
                    # p.kill()
                    
                    # os.kill(proc.pid, signal.SIGKILL)
                    
except Exception as ex:
    raise ex
finally:
    print('closing driver')
    # driver.close()
    print(driver is None)
    # driver.quit()
