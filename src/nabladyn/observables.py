"""Observables of a frame: kinetic energy, temperature, pressure and the RDF.

Plain tensor functions, for a single frame, a simulation or training. Units nm, ps, Da,
kJ/mol and K.
"""

import math

import torch
from torch import Tensor

from . import space
from .neighbors import NeighborPairs
from .potential import pair_index
from .units import BOLTZMANN


def kinetic_energy(velocities: Tensor, masses: Tensor) -> Tensor:
    """Return K = sum m v^2 / 2 in kJ/mol for velocities [N, 3] in nm/ps and masses [N] in Da."""
    return 0.5 * (masses[:, None] * velocities.square()).sum()


def temperature(kinetic: Tensor, degrees_of_freedom: int) -> Tensor:
    """Return T = 2K / (g k_B) in K, for K in kJ/mol and g degrees of freedom.

    With zero total momentum, g = 3N - 3.
    """
    return 2.0 * kinetic / (degrees_of_freedom * BOLTZMANN)


def pressure(kinetic: Tensor, virial: Tensor, cell: Tensor) -> Tensor:
    """Return the instantaneous pressure P = (2K + trace W) / (3V) in kJ/mol/nm^3.

    ``kinetic`` K and ``virial`` W [3, 3] are in kJ/mol; ``cell`` [3] holds the edge lengths
    in nm. Divide by ``units.BAR`` for bar.
    """
    return (2.0 * kinetic + virial.trace()) / (3.0 * cell.prod())


def _check_edges(edges: Tensor, pairs: NeighborPairs) -> None:
    """Raise ``ValueError`` unless the last edge lies within the pairs' cutoff."""
    if float(edges[-1]) > pairs.cutoff:
        raise ValueError(
            f"the last edge, {float(edges[-1])} nm, lies beyond the pairs' cutoff "
            f"{pairs.cutoff} nm, so pairs beyond it would be missing",
        )


def rdf(
    positions: Tensor,
    cell: Tensor,
    pairs: NeighborPairs,
    edges: Tensor,
    types: Tensor | None = None,
) -> Tensor:
    """Return the radial distribution function of one frame, [C, B].

    A histogram over bins [r_l, r_u) with ``edges`` [B + 1] in nm from 0:

        g_c(b) = V n_c(b) / (P_c (4 pi / 3)(r_u^3 - r_l^3))

    where n_c(b) counts the pairs of channel c in bin b, V is the cell volume and P_c is the
    channel's number of atom pairs: N_a N_b for two types, N_a (N_a - 1) / 2 for a type with
    itself, so an ideal gas reads 1. The channels are the unordered type pairs of ``types``
    [N] in the order (0, 0), (0, 1), ..., (1, 1), ..., or one channel without types.
    ``positions`` [N, 3] may lie outside the cell. The result has no derivative in the
    positions. Syncs with the host, to check the edges and, with ``types``, to count them.

    Raises ``ValueError`` if the last edge lies beyond the cutoff of ``pairs``.
    """
    _check_edges(edges, pairs)
    positions, _ = space.wrap(positions, cell)
    first, second = pairs.index[0], pairs.index[1]
    vectors = space._displacement_wrapped(positions[first], positions[second], cell)
    distance = vectors.norm(dim=-1)[:, None]
    inside = (first != second)[:, None] & (distance >= edges[:-1]) & (distance < edges[1:])
    inside = inside.to(positions.dtype)
    num_atoms = positions.shape[0]
    if types is None:
        sums = inside.sum(dim=0, keepdim=True)
        counts = positions.new_tensor([num_atoms * (num_atoms - 1) / 2])
    else:
        num_types = int(types.max()) + 1
        channel = pair_index(types[first], types[second], num_types)
        sums = inside.new_zeros((num_types * (num_types + 1) // 2, inside.shape[1]))
        sums = sums.index_add(0, channel, inside)
        per_type = torch.bincount(types, minlength=num_types).to(inside.dtype)
        low, high = torch.triu_indices(num_types, num_types, device=types.device)
        counts = torch.where(
            low == high,
            per_type[low] * (per_type[low] - 1) / 2,
            per_type[low] * per_type[high],
        )
    shells = (4.0 * math.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    return cell.prod() * sums / (counts[:, None] * shells)


__all__ = ("kinetic_energy", "pressure", "rdf", "temperature")
