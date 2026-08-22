# Configuration file for the Sphinx documentation builder.
import pathlib
import sys

# Resolve relative to THIS FILE, not the cwd: conf.py lives at docs/source/, so the
# package root is two levels up. (Sphinx does chdir to confdir, which would make
# "../../src" work too, but __file__-relative cannot silently break if that changes
# or if conf.py is ever imported by tooling.)
_HERE = pathlib.Path(__file__).resolve().parent          # <repo>/docs/source
sys.path.insert(0, str(_HERE.parents[1] / "src"))        # <repo>/src
# The package is NOT pip-installed on the RTD builder: its `full` extra pulls
# ps_1loop_jax, which is a local sibling checkout and not a PyPI distribution.

project = "jaxPTPolyPol"
author = "Minh Nguyen"
copyright = "2026, Minh Nguyen"
release = "0.1"
version = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]

# Deliberate layout: the Sphinx source root is docs/source/, so docs/design/ (the
# measurement log + convention maps) and the git-ignored docs/plans/ and
# docs/superpowers/ are OUTSIDE the source tree and invisible to the glob. Do not
# move conf.py up to docs/ -- that would pull ~19 non-page .md files into the build
# and fail_on_warning would reject every one of them as "not included in any toctree".
exclude_patterns = ["Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Unavailable on PyPI (ps_1loop_jax, candl_data) or too heavy for the builder.
autodoc_mock_imports = [
    "ps_1loop_jax",
    "cosmopower_jax",
    "candl",
    "candl_data",
    "clipy",
    "blackjax",
    "quadax",
    "numdifftools",
    "corner",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
