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

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'Optimization'
copyright = '2022, Mendonça et al'
author = 'Mendonça et al.'

# The full version, including alpha/beta/rc tags
release = '0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.napoleon",
        "sphinx.ext.autosummary",
        "sphinx.ext.viewcode",
        "sphinx.ext.intersphinx"
]

autoclass_content = "both"  # include both class docstring and __init__
autodoc_default_options = {
        "members":           True,
        "inherited-members": True,
        "private-members":   False,
        "show-inheritance":  True,
}

intersphinx_mapping = {
        'numpy':      ('https://numpy.org/doc/stable/', None),
        'python':     ('https://docs.python.org/3/', None),
        'pydagogue':  ('https://matthew-brett.github.io/pydagogue/', None),
        'matplotlib': ('https://matplotlib.org/', None),
        'scipy':      ('https://docs.scipy.org/doc/scipy/reference/', None),
        'pandas':     ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

autosummary_generate = True  # Make _autosummary files and include them
napoleon_numpy_docstring = False  # Force consistency, leave only Google
napoleon_use_rtype = False  # More legible

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
        "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

# If false, no module index is generated.
html_domain_indices = True

# The master toctree document.
master_doc = 'index'

# Theme configuration
html_theme_options = {
        'analytics_id':                'G-CKGHPS9QNL',
        'display_version':             False,
        'style_nav_header_background': '#008b00'
}
