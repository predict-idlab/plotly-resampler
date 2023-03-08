# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../plotly_resampler"))


# -- Project information -----------------------------------------------------

project = "plotly-resampler"
copyright = "2023, Jonas Van Der Donckt"
author = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

# The full version, including alpha/beta/rc tags
# -> extract it from the pyproject.toml file
release: str = None
with open("../../pyproject.toml", "r") as f:
    for l in f.readlines():
        if l.startswith("version"):
            release = l.strip().split("=")[1].strip()
            continue
assert release is not None, "Was not able to extract the release number!"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # load napoleon b4 sphinx autodoc typehints
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    # 'sphinx.ext.githubpages',
]


# NApoleon conf
# napoleon_include_init_with_doc = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

todo_include_todos = True

autoclass_content = "both"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True

## This snipped can be used to adjust the signature-typehints
# def fix_sig(app, what, name, obj, options, signature, return_annotation):
#     return ("", "")

# def setup(app):
#     app.connect("autodoc-process-signature", fix_sig)

# typehints_defaults = 'comma'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"
html_favicon = "_static/icon.png"
language = "en"

html_theme_options = {
    # "show_nav_level": 2,
    # "collapse_navigation": True,
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/predict-idlab/plotly-resampler",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fab fa-github-square",
            # Whether icon should be a FontAwesome class, or a local file
            "type": "fontawesome",  # Default is fontawesome
        }
    ],
    "pygment_light_style": "tango",  # tango
    "pygment_dark_style": "native",
    "navbar_end": [
        "theme-switcher.html",
        "navbar-icon-links.html",
        "search-field.html",
    ],
}

html_sidebars = {
    "figure_resampler*": [],
    "aggregation*": [],
    "_autosummary*": [],
    "*": [],
}
# html_sidebars = {"figure_resampler*": [], "aggregation*": []}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
