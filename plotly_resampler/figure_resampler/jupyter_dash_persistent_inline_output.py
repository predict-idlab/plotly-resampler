import base64
import contextlib
import logging
import os
import queue
import threading
import uuid
import warnings

import requests

try:
    from IPython.display import HTML, display
except ImportError:
    pass

from dash._jupyter import JupyterDash, _jupyter_config, make_server, retry
from plotly.graph_objects import Figure


class JupyterDashPersistentInlineOutput:
    """Extension of the JupyterDash class to support the custom inline output for
    ``FigureResampler`` figures.

    Specifically we embed a div in the notebook to display the figure inline.

     - In this div the figure is shown as an iframe when the server (of the dash app)
       is alive.
     - In this div the figure is shown as an image when the server (of the dash app)
       is dead.

    As the HTML & javascript code is embedded in the notebook output, which is loaded
    each time you open the notebook, the figure is always displayed (either as iframe
    or just an image).
    Hence, this extension enables to maintain always an output in the notebook.

    .. Note::
        This subclass is only used when the mode is set to ``"inline_persistent"`` in
        the :func:`FigureResampler.show_dash <plotly_resampler.figure_resampler.FigureResampler.show_dash>`
        method. However, the mode should be passed as ``"inline"`` since this subclass
        overwrites the inline behavior.

    .. Note::
        This subclass utilizes the optional ``flask_cors`` package to detect whether the
        server is alive or not.

    """

    def __init__(self, fig: Figure) -> None:
        super().__init__()
        self.fig = fig

        # The unique id of this app
        # This id is used to couple the output in the notebook with this app
        # A fetch request is performed to the _is_alive_{uid} endpoint to check if the
        # app is still alive.
        self.uid = str(uuid.uuid4())

    def _display_inline_output(self, dashboard_url, width, height):
        """Display the dash app persistent inline in the notebook.

        The figure is displayed as an iframe in the notebook if the server is reachable,
        otherwise as an image.
        """
        # TODO: check whether an error gets logged in case of crash
        # TODO: add option to opt out of this
        from IPython.display import display

        try:
            import flask_cors  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            warnings.warn(
                "'flask_cors' is not installed. The persistent inline output will "
                + " not be able to detect whether the server is alive or not."
            )

        # Get the image from the dashboard and encode it as base64
        fig = self.fig  # is stored in the show_dash method
        f_width = 1000 if fig.layout.width is None else fig.layout.width
        fig_base64 = base64.b64encode(
            fig.to_image(format="png", width=f_width, scale=1, height=fig.layout.height)
        ).decode("utf8")

        # The html (& javascript) code to display the app / figure
        uid = self.uid
        display(
            {
                "text/html": f"""
                <div id='PR_div__{uid}'></div>
                <script type='text/javascript'>
                """
                + """

                function setOutput(timeout) {
                    """
                +
                # Variables should be in local scope (in the closure)
                f"""
                    var pr_div = document.getElementById('PR_div__{uid}');
                    var url = '{dashboard_url}';
                    var pr_img_src = 'data:image/png;base64, {fig_base64}';
                    var is_alive_suffix = '_is_alive_{uid}';
                    """
                + """

                    if (pr_div.firstChild) return  // return if already loaded

                    const controller = new AbortController();
                    const signal = controller.signal;

                    return fetch(url + is_alive_suffix, {method: 'GET', signal: signal})
                        .then(response => response.text())
                        .then(data =>
                            {
                                if (data == "Alive") {
                                    console.log("Server is alive");
                                    iframeOutput(pr_div, url);
                                } else {
                                    // I think this case will never occur because of CORS
                                    console.log("Server is dead");
                                    imageOutput(pr_div, pr_img_src);
                                }
                            }
                        )
                        .catch(error => {
                            console.log("Server is unreachable");
                            imageOutput(pr_div, pr_img_src);
                        })
                }

                setOutput(350);

                function imageOutput(element, pr_img_src) {
                    console.log('Setting image');
                    var pr_img = document.createElement("img");
                    pr_img.setAttribute("src", pr_img_src)
                    pr_img.setAttribute("alt", 'Server unreachable - using image instead');
                    """
                + f"""
                    pr_img.setAttribute("max-width", '{width}');
                    pr_img.setAttribute("max-height", '{height}');
                    pr_img.setAttribute("width", 'auto');
                    pr_img.setAttribute("height", 'auto');
                    """
                + """
                    element.appendChild(pr_img);
                }

                function iframeOutput(element, url) {
                    console.log('Setting iframe');
                    var pr_iframe = document.createElement("iframe");
                    pr_iframe.setAttribute("src", url);
                    pr_iframe.setAttribute("frameborder", '0');
                    pr_iframe.setAttribute("allowfullscreen", '');
                    """
                + f"""
                    pr_iframe.setAttribute("width", '{width}');
                    pr_iframe.setAttribute("height", '{height}');
                    """
                + """
                    element.appendChild(pr_iframe);
                }
                </script>
                """
            },
            raw=True,
            clear=True,
            display_id=uid,
        )

    # NOTE: Minimally adatped version from dash._jupyter.JupyterDash.run_server
    def run_app(
        self,
        app,
        width="100%",
        height=650,
        host="127.0.0.1",
        port=8050,
        server_url=None,
    ):
        """Run the inline persistent dash app in the notebook.

        Parameters
        ----------
        app : dash.Dash
            A Dash application instance
        width : str, optional
            Width of app when displayed using mode="inline", by default "100%"
        height : int, optional
            Height of app when displayed using mode="inline", by default 650
        host : str, optional
            Host of the server, by default "127.0.0.1"
        port : int, optional
            Port used by the server, by default 8050
        server_url : str, optional
            Use if a custom url is required to display the app, by default None

        """
        # Terminate any existing server using this port
        old_server = JupyterDash._servers.get((host, port))
        if old_server:
            old_server.shutdown()
            del JupyterDash._servers[(host, port)]

        # Configure pathname prefix
        if "base_subpath" in _jupyter_config:
            requests_pathname_prefix = (
                _jupyter_config["base_subpath"].rstrip("/") + "/proxy/{port}/"
            )
        else:
            requests_pathname_prefix = app.config.get("requests_pathname_prefix", None)

        if requests_pathname_prefix is not None:
            requests_pathname_prefix = requests_pathname_prefix.format(port=port)
        else:
            requests_pathname_prefix = "/"

        # FIXME Move config initialization to main dash __init__
        # low-level setter to circumvent Dash's config locking
        # normally it's unsafe to alter requests_pathname_prefix this late, but
        # Jupyter needs some unusual behavior.
        dict.__setitem__(
            app.config, "requests_pathname_prefix", requests_pathname_prefix
        )

        # # Compute server_url url
        if server_url is None:
            if "server_url" in _jupyter_config:
                server_url = _jupyter_config["server_url"].rstrip("/")
            else:
                domain_base = os.environ.get("DASH_DOMAIN_BASE", None)
                if domain_base:
                    # Dash Enterprise sets DASH_DOMAIN_BASE environment variable
                    server_url = "https://" + domain_base
                else:
                    server_url = f"http://{host}:{port}"
        else:
            server_url = server_url.rstrip("/")

        # server_url = "http://{host}:{port}".format(host=host, port=port)

        dashboard_url = f"{server_url}{requests_pathname_prefix}"

        # prevent partial import of orjson when it's installed and mode=jupyterlab
        # TODO: why do we need this? Why only in this mode? Importing here in
        # all modes anyway, in case there's a way it can pop up in another mode
        try:
            # pylint: disable=C0415,W0611
            import orjson  # noqa: F401
        except ImportError:
            pass

        err_q = queue.Queue()

        server = make_server(host, port, app.server, threaded=True, processes=0)
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        # ---------------------------------------------------------------------
        # NOTE: added this code to mimic the _alive_{token} endpoint but with cors
        with contextlib.suppress(ImportWarning, ModuleNotFoundError):
            from flask_cors import cross_origin

            @app.server.route(f"/_is_alive_{self.uid}", methods=["GET"])
            @cross_origin(origins=["*"], allow_headers=["Content-Type"])
            def broadcast_alive():
                return "Alive"

        # ---------------------------------------------------------------------

        @retry(
            stop_max_attempt_number=15,
            wait_exponential_multiplier=100,
            wait_exponential_max=1000,
        )
        def run():
            try:
                server.serve_forever()
            except SystemExit:
                pass
            except Exception as error:
                err_q.put(error)
                raise error

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

        JupyterDash._servers[(host, port)] = server

        # Wait for server to start up
        alive_url = f"http://{host}:{port}/_alive_{JupyterDash.alive_token}"

        def _get_error():
            try:
                err = err_q.get_nowait()
                if err:
                    raise err
            except queue.Empty:
                pass

        # Wait for app to respond to _alive endpoint
        @retry(
            stop_max_attempt_number=15,
            wait_exponential_multiplier=10,
            wait_exponential_max=1000,
        )
        def wait_for_app():
            _get_error()
            try:
                req = requests.get(alive_url)
                res = req.content.decode()
                if req.status_code != 200:
                    raise Exception(res)

                if res != "Alive":
                    url = f"http://{host}:{port}"
                    raise OSError(
                        f"Address '{url}' already in use.\n"
                        "    Try passing a different port to run_server."
                    )
            except requests.ConnectionError as err:
                _get_error()
                raise err

        try:
            wait_for_app()
            self._display_inline_output(dashboard_url, width=width, height=height)

        except Exception as final_error:  # pylint: disable=broad-except
            msg = str(final_error)
            if msg.startswith("<!"):
                display(HTML(msg))
            else:
                raise final_error
