"""A package to rotate images with torch."""

from importlib.metadata import PackageNotFoundError, version

from .rotate_image_2d import rotate_image_2d
from .rotate_image_3d import rotate_image_3d

try:
    __version__ = version("torch-rotate-image")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Spencer J Rothfuss"
__email__ = "spencer.j.rothfuss@vanderbilt.edu"
