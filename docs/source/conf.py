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
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../src/'))

from eureka import __version__

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'Eureka!'
copyright = '2022, Eureka! pipeline developers'
author = 'Eureka! pipeline developers'

version = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx_rtd_theme', 'sphinx.ext.todo', 'sphinx.ext.viewcode',
              'sphinx.ext.autodoc', 'numpydoc', 'nbsphinx', 'myst_parser',
              'sphinx.ext.autosectionlabel', 'sphinx.ext.napoleon']

master_doc = 'index'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = 'sphinx'

# nbsphinx settings
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. note::  `Download the full notebook for this tutorial here <https://github.com/kevin218/Eureka/tree/main/docs/source/{{ docname }}>`_
"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

import sphinx_rtd_theme
# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/rtd_dark.css"]
html_logo = "../media/Eureka_logo.png"

# Ignoring duplicated section warnings in api file
suppress_warnings = ['autosectionlabel.*']

# Remove stub file not found warnings
numpydoc_class_members_toctree = False
