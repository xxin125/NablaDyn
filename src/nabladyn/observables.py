"""Observables of a frame: kinetic energy, temperature, pressure, the RDF and the ADF.

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
    r"""Return the kinetic energy in kJ/mol for velocities [N, 3] in nm/ps and masses [N] in Da.

    $`K = \sum m v^2 / 2`$.
    """
    return 0.5 * (masses[:, None] * velocities.square()).sum()


def temperature(kinetic: Tensor, degrees_of_freedom: int) -> Tensor:
    """Return $`T = 2K / (g k_B)`$ in K, for K in kJ/mol and g degrees of freedom.

    With zero total momentum, $`g = 3N - 3`$.
    """
    return 2.0 * kinetic / (degrees_of_freedom * BOLTZMANN)


def pressure(kinetic: Tensor, virial: Tensor, cell: Tensor) -> Tensor:
    r"""Return the instantaneous pressure in kJ/mol/nm^3.

    $`P = (2K + \mathrm{tr}(W)) / (3V)`$, with ``kinetic`` K and ``virial`` W [3, 3] in
    kJ/mol and V the volume of ``cell`` [3], its edge lengths in nm. Divide by ``units.BAR``
    for bar.
    """
    return (2.0 * kinetic + virial.trace()) / (3.0 * cell.prod())


def _check_edges(edges: Tensor, pairs: NeighborPairs) -> None:
    """Raise ``ValueError`` unless the last edge lies within the pairs' cutoff."""
    if float(edges[-1]) > pairs.cutoff:
        raise ValueError(
            f"the last edge, {float(edges[-1])} nm, lies beyond the pairs' cutoff "
            f"{pairs.cutoff} nm, so pairs beyond it would be missing",
        )


def _check_types(types: Tensor | None, positions: Tensor) -> None:
    """Raise ``ValueError`` unless the types, when given, hold one entry per atom."""
    if types is not None and tuple(types.shape) != (positions.shape[0],):
        raise ValueError(
            f"types must hold one entry per atom, [{positions.shape[0]}], not "
            f"{list(types.shape)}",
        )


def rdf(
    positions: Tensor,
    cell: Tensor,
    pairs: NeighborPairs,
    edges: Tensor,
    types: Tensor | None = None,
) -> Tensor:
    r"""Return the radial distribution function of one frame, [C, B].

    A histogram over bins $`[r_l, r_u)`$ with ``edges`` [B + 1] in nm from 0:

    ```math
    g_c(b) = V n_c(b) / (P_c (4\pi/3)(r_u^3 - r_l^3))
    ```

    where $`n_c(b)`$ counts the pairs of channel c in bin b, V is the cell volume and $`P_c`$
    is the channel's number of atom pairs: $`N_a N_b`$ for two types, $`N_a (N_a - 1) / 2`$
    for a type with itself, so an ideal gas reads 1. The channels are the unordered type
    pairs of ``types`` [N] in the order (0, 0), (0, 1), ..., (1, 1), ..., or one channel
    without types. ``positions`` [N, 3] may lie outside the cell. The result has no derivative
    in the positions. Syncs with the host, to check the edges and, with ``types``, to count
    them.

    Raises ``ValueError`` if the last edge lies beyond the cutoff of ``pairs``, or if
    ``types`` does not hold one entry per atom.
    """
    _check_edges(edges, pairs)
    _check_types(types, positions)
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


def _check_r_max(r_max: float, pairs: NeighborPairs) -> None:
    """Raise ``ValueError`` unless r_max lies within the pairs' cutoff."""
    if r_max > pairs.cutoff:
        raise ValueError(
            f"r_max, {r_max} nm, lies beyond the pairs' cutoff {pairs.cutoff} nm, so "
            "neighbors beyond it would be missing",
        )


def adf(
    positions: Tensor,
    cell: Tensor,
    pairs: NeighborPairs,
    edges: Tensor,
    *,
    r_max: float,
) -> Tensor:
    r"""Return the angle distribution of one frame's neighbors closer than r_max, [B].

    For every atom i and every unordered pair of its neighbors j and k, atoms closer to i
    than ``r_max`` in nm by minimum image, the angle $`\theta_{jik}`$ between $`r_j - r_i`$
    and $`r_k - r_i`$, binned on ``edges`` [B + 1] in degrees, rising within [0, 180], each
    bin $`[l, u)`$ and the last closed; per degree:

    ```math
    P(b) = n(b) / (n (u - l))
    ```

    with $`n(b)`$ of the frame's n binned angles in bin b, so the distribution has unit area; a
    frame with no angle gives zeros. ``positions`` [N, 3] may lie outside the cell. The
    result has no derivative in the positions. With ``r_max`` at the g(r)'s first minimum,
    it is the first shell's angle distribution. Syncs with the host, to check ``r_max`` and
    to size the table of each atom's neighbors.

    Raises ``ValueError`` if ``r_max`` lies beyond the cutoff of ``pairs``.
    """
    _check_r_max(r_max, pairs)
    positions, _ = space.wrap(positions, cell)
    first, second = pairs.index[0].long(), pairs.index[1].long()
    vectors = space._displacement_wrapped(positions[first], positions[second], cell)
    distance = vectors.norm(dim=-1)
    near = (first != second) & (distance < r_max)
    # A pair closer than r_max is a neighbor of both its atoms; vectors run from second to
    # first, so first sees second along -vectors.
    centers = torch.cat((first[near], second[near]))
    units = torch.cat((-vectors[near], vectors[near])) / distance[near].repeat(2)[:, None]
    num_atoms, bins = positions.shape[0], edges.numel() - 1
    counts = torch.bincount(centers, minlength=num_atoms)
    most = int(counts.max())
    if most < 2:
        return positions.new_zeros(bins)
    # Each atom's neighbors fill the first slots of its row of a table, the rest stay empty.
    order = torch.argsort(centers, stable=True)
    centers, units = centers[order], units[order]
    starts = torch.cumsum(counts, dim=0) - counts
    slots = torch.arange(centers.numel(), device=centers.device) - starts[centers]
    table = units.new_zeros((num_atoms, most, 3)).index_put((centers, slots), units)
    filled = torch.zeros((num_atoms, most), dtype=torch.bool, device=centers.device)
    filled = filled.index_put((centers, slots), torch.ones_like(slots, dtype=torch.bool))
    left, right = torch.triu_indices(most, most, offset=1, device=centers.device)
    cosine = (table[:, left] * table[:, right]).sum(dim=-1).clamp(-1.0, 1.0)
    theta = torch.rad2deg(torch.arccos(cosine[filled[:, left] & filled[:, right]]))
    edges = edges.to(theta)
    which = torch.bucketize(theta, edges, right=True) - 1
    which = torch.where(theta == edges[-1], bins - 1, which)
    inside = (which >= 0) & (which < bins)
    binned = torch.bincount(which[inside], minlength=bins).to(positions.dtype)
    return binned / (binned.sum().clamp(min=1.0) * edges.diff())


__all__ = ("adf", "kinetic_energy", "pressure", "rdf", "temperature")
