# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))  # adjust if needed

project = "Ymago"
copyright = "2025, Andrej Mäsiar"
author = "Andrej Mäsiar"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google/NumPy docstrings
    "sphinx_autodoc_typehints",  # Nice type hints in docs
    "sphinxcontrib.mermaid",  # Optional diagrams
]


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}
extensions.append("sphinx.ext.autodoc.typehints")
typehints_fully_qualified = False


html_theme = "furo"

# Allow Markdown with MyST
myst_enable_extensions = [
    "linkify",
    "colon_fence",
]

html_static_path = ["_static"]
