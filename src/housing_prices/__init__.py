"""Utilities for the California housing portfolio project."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("housing-prices")
except PackageNotFoundError:
    __version__ = "0.1.0"
