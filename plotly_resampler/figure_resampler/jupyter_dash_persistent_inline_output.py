import base64
import contextlib
import uuid
import warnings

from jupyter_dash import JupyterDash


class JupyterDashPersistentInlineOutput(JupyterDash):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._uid = str(uuid.uuid4())  # A new unique id for each app

        with contextlib.suppress(ImportWarning, ModuleNotFoundError):
            from flask_cors import cross_origin

            # Mimic the _alive_{token} endpoint but with cors
            @self.server.route(f"/_is_alive_{self._uid}", methods=["GET"])
            @cross_origin(origins=["*"], allow_headers=["Content-Type"])
            def broadcast_alive():
                return "Alive"

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
        fig = self.layout.children[0].figure  # is stored there in the show_dash method
        f_width = 1000 if fig.layout.width is None else fig.layout.width
        fig_base64 = base64.b64encode(
            fig.to_image(format="png", width=f_width, scale=1, height=fig.layout.height)
        ).decode("utf8")

        # The unique id of this app
        # This id is used to couple the output in the notebook with this app
        # A fetch request is performed to the _is_alive_{uid} endpoint to check if the
        # app is still alive.
        uid = self._uid

        # The html (& javascript) code to display the app / figure
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

    def _display_in_jupyter(self, dashboard_url, port, mode, width, height):
        """Override the display method to retain some output when displaying inline
        in jupyter.
        """
        if mode == "inline":
            self._display_inline_output(dashboard_url, width, height)
        else:
            super()._display_in_jupyter(dashboard_url, port, mode, width, height)
