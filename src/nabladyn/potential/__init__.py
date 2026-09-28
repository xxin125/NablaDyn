"""Potentials: models whose one scalar energy defines forces and virial."""

from ._api import lennard_jones, opls, schnet, tabulated
from ._base import Potential
from ._lennard_jones import pair_index

__all__ = ("Potential", "lennard_jones", "opls", "pair_index", "schnet", "tabulated")
