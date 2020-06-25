# configuration file for the Sphinx documentation builder.
#
# modified by Derek Huang
#
# full configuration list:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Changelog:
#
# 06-25-2020
#
# change pygments highlighting style to emacs.
#
# 06-24-2020
#
# initial creation. change log added. added autodoc, autosummary, intersphinx,
# and guzzle theme configuration. changed theme to RTD theme + add delimiter
# check for correct sys.path paths (import local build, not installed
# site-packages build, which led to builds not being updated).

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
from os.path import dirname, abspath
import sys
# change delimiter style based on whether we have NT or POSIX
_delim = "/"
if os.name == "nt": _delim = "\\"
sys.path.insert(0, dirname(abspath(__file__)) + _delim + ".." + _delim + "..")

# -- Project information -----------------------------------------------------

project = "shizuka"
copyright = "2020, Derek Huang"
author = "Derek Huang"

# The full version, including alpha/beta/rc tags
release = "0.0.1"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your own.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- autodoc configuration ---------------------------------------------------

# set default options for autodoc directives. include __repr__ special member
# and any private members (names prepended with _), show class inheritance,
# but leave out any class members that are not documented with a docstring.
#
# note: since ignore-module-all is not set, only the members in __all__ in
# __init__.py will be looked for and their order will be maintained.
autodoc_default_options = {
    "members": True,
    "private-members": True,
    "show-inheritance": True,
    "special-members": "__repr__"
}

# -- autosummary configuration -----------------------------------------------

# set to True to generate stub files for any modules named in a file's
# autosummary directive(s). so far, only index.rst should have autosummary.
autosummary_generate = True

# -- intersphinx configuration -----------------------------------------------

# determines which external package documentations to link to. thanks to the
# link here: https://gist.github.com/bskinn/0e164963428d4b51017cebdb6cda5209
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None)
}

# -- Options for HTML output -------------------------------------------------

# name of the HTML theme. note that we wouldn't need all this other stuff if
# we were using a builtin theme like nature, etc.
html_theme = "sphinx_rtd_theme"

# add theme to extensions so sphinx can go find it
extensions.append(html_theme)

# HTML theme options
html_theme_options = {}

# pygments style to use for syntax highlighting
pygments_style = "emacs"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
