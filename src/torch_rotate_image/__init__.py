"""A package to rotate images with torch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-rotate-image")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Spencer J Rothfuss"
__email__ = "spencer.j.rothfuss@vanderbilt.edu"
