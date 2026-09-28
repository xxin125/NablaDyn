"""Supervised fitting of a potential to reference energies, forces and virials.

A ``Dataset`` holds the labeled frames; ``fit`` trains the parameters and returns a ``Report``.
"""

from ._dataset import Dataset
from ._fit import Report, fit

__all__ = ("Dataset", "Report", "fit")
