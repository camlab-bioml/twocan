# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Twocan'
copyright = '2024, Caitlin F. Harrigan'
author = 'Caitlin F. Harrigan'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_nb',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# MyST-NB settings for notebook execution
nb_execution_mode = "off"  # Don't execute notebooks during build
nb_execution_timeout = 60
myst_nb_render_priority = {
    "html": ("application/vnd.jupyter.widget-view+json", "text/html", "image/png", "text/plain"),
    "latex": ("text/latex", "image/png", "text/plain")
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'opencv': ('https://docs.opencv.org/4.x/', None),
    'optuna': ('https://optuna.readthedocs.io/en/stable/', None),
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

autosummary_generate = True 