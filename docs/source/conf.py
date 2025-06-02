# Import mock configuration
try:
    import conf_mock
except ImportError:
    pass

"""
Configuration file for the Sphinx documentation builder.
"""

import os
import sys
from datetime import datetime

# Add src directory to Python path for autodoc
sys.path.insert(0, os.path.abspath('../../'))

# Project information
project = 'ML Training Framework'
copyright = f'{datetime.now().year}, Your Name'
author = 'Your Name'

# Import package to get the version
import src.utils.version
version = src.utils.version.__version__
release = version

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_togglebutton',
    'sphinxcontrib.mermaid',
    'sphinx_tabs.tabs',
    'myst_parser',
    'nbsphinx',
]

# Add any paths that contain templates
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    'examples/**',  # Exclude example notebooks to avoid PandocMissing errors
    '*.ipynb',      # Skip any remaining notebooks
]

# Mock heavy or optional imports to avoid import failures during autodoc
autodoc_mock_imports = [
    'torch',
    'lightning',
    'transformers',
    'src.tasks.clm_training.orchestrator',
    'src.tasks.clm_training.fabric.distributed',
]

# The theme to use for HTML and HTML Help pages
html_theme = 'pydata_sphinx_theme'

# Theme options
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/gplsi/continual-pretraining-framework/tree/last-llama-fsdp",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "text": "ML Training Framework",
    },
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "show_toc_level": 3,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "footer_items": ["copyright", "sphinx-version", "theme-version"],
    "switcher": {
        "json_url": "https://yourusername.github.io/workspace/_static/switcher.json",
        "version_match": version,
    },
    "pygment_light_style": "github-light",
    "pygment_dark_style": "github-dark",
}

# HTML context for GitHub links
html_context = {
    "github_user": "gplsi",
    "github_repo": "continual-pretraining-framework",
    "github_version": "last-llama-fsdp",
    "doc_path": "docs/source",
}

# Add any paths that contain custom static files
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_js_files = [
    'js/custom.js',
]

# Favicon
html_favicon = '_static/favicon.ico'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autoclass_content = 'both'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'transformers': ('https://huggingface.co/docs/transformers/main', None),
    'lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
}

# MyST parser settings
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]
myst_heading_anchors = 3
myst_dmath_double_inline = True

# NBSphinx settings
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True