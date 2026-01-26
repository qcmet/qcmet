"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys

import qcmet

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "QCMet"
copyright = "2026, QCMet Contributors"
author = "QCMet Contributors"
release = qcmet.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Include notebook files
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# myst-nb settings - don't execute notebooks during build
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "deflist"]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__init__",
}

# Autosummary settings - automatically generate API docs
autosummary_generate = True
autosummary_imported_members = False

# Show full module path but trim it to the importable path
python_use_unqualified_type_names = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "gitlab_url": "https://gitlab.npl.co.uk/qc-metrics-and-benchmarks/qcmet",
    "header_links_before_dropdown": 10,  # Show all main nav items
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "search_bar_text": "Search the docs...",
}

html_logo = "./_static/qcmet-logo.png"
html_favicon = None

html_context = {"default_mode": "light"}

html_sidebars = {
    "index": [],
    "tutorials/01_installation": [],
    "tutorials/02_quickstart": [],
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}
