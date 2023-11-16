# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import numpy.typing as npt

project = "Clinamen2"
copyright = "2023, The Clinamen2 contributors"
author = "Ralf Wanzenböck, Florian Buchner, Péter Kovács, Georg K. H. Madsen, Jesús Carrete"
release = "2023.11.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]
autodoc_typehints = "description"


def custom_typehints_formatter(annotation, config):
    """Small custom formatter taking care of npt.ArrayLike"""
    # TODO: Remove this quick hack as soon as a better solution is
    # available for aliases (that works with typehints).
    if annotation == npt.ArrayLike:
        return ":py:class:`numpy.typing.ArrayLike`"
    return None


typehints_formatter = custom_typehints_formatter


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "pydata_sphinx_theme"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"collapse_navigation": True, "sticky_navigation": True}
