"""Potentials: models whose one scalar energy defines forces and virial."""

from ._api import angular_schnet, lennard_jones, opls, schnet, sum, tabulated
from ._base import Potential
from ._lennard_jones import pair_index

__all__ = (
    "Potential",
    "angular_schnet",
    "lennard_jones",
    "opls",
    "pair_index",
    "schnet",
    "sum",
    "tabulated",
)
