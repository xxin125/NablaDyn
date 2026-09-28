"""OPLS nonbonded pairs: exclusions and 1-4 pairs from the bonds, mixing, pair energies."""

import math

import torch
from torch import Tensor

from ... import space, units
from .._lennard_jones._potential import _pair_energy, _pair_rows


def pair_classes(bonds: Tensor, num_atoms: int) -> tuple[Tensor, Tensor]:
    """Return the excluded [E, 2] and 1-4 [P, 2] pairs of ``bonds`` [B, 2].

    Atoms 1 or 2 bonds apart by the shortest path are excluded; 3 bonds apart, a 1-4 pair.
    Rows (i, j) have i < j and are sorted; int64 on the CPU. Syncs with the host.
    """
    bonded = [[] for _ in range(num_atoms)]
    for left, right in bonds.tolist():
        bonded[left].append(right)
        bonded[right].append(left)
    excluded, one_four = [], []
    for source in range(num_atoms):
        depths, frontier = {source: 0}, [source]
        for depth in (1, 2, 3):
            reached = []
            for atom in frontier:
                for partner in bonded[atom]:
                    if partner not in depths:
                        depths[partner] = depth
                        reached.append(partner)
            frontier = reached
        for atom, depth in depths.items():
            if atom > source:
                (excluded if depth < 3 else one_four).append((source, atom))
    return (
        torch.tensor(sorted(excluded), dtype=torch.int64).reshape(-1, 2),
        torch.tensor(sorted(one_four), dtype=torch.int64).reshape(-1, 2),
    )


def partner_table(excluded: Tensor, one_four: Tensor, num_atoms: int) -> Tensor:
    """Return [N, M] int32: each atom's excluded and 1-4 partners, padded with its own index.

    M is the most partners any atom has, at least 1. Syncs with the host.
    """
    partners = [[] for _ in range(num_atoms)]
    for left, right in torch.cat([excluded, one_four]).tolist():
        partners[left].append(right)
        partners[right].append(left)
    width = max([1] + [len(row) for row in partners])
    rows = [row + [atom] * (width - len(row)) for atom, row in enumerate(partners)]
    return torch.tensor(rows, dtype=torch.int32).reshape(num_atoms, width)


def geometric_rows(values: Tensor, first: Tensor, second: Tensor) -> Tensor:
    """Return sqrt(v_a v_b) for the type pairs ``first`` and ``second``, 0 where v_a v_b is 0.

    ``values`` [T] >= 0 per type; the type pairs are rows in ``pair_index`` order.
    """
    product = values[first] * values[second]
    positive = product > 0
    # sqrt has an infinite slope at 0: a zero product takes the root of 1 and is then
    # replaced by 0, so neither type of the pair gets a NaN gradient.
    return torch.where(positive, torch.where(positive, product, 1.0).sqrt(), 0.0)


def _listed_squares(
    positions: Tensor,
    cell: Tensor,
    strain: Tensor | None,
    pairs: Tensor,
) -> Tensor:
    """Return the squared strained minimum-image distances of listed pairs [P, 2], in nm^2."""
    vectors = space._displacement_wrapped(positions[pairs[:, 0]], positions[pairs[:, 1]], cell)
    return space.strained(vectors, strain).square().sum(dim=-1)


def ordinary_energy(
    positions: Tensor,
    cell: Tensor,
    strain: Tensor | None,
    index: Tensor,
    partners: Tensor,
    cutoff: float,
    lj_types: Tensor,
    num_lj_types: int,
    sigma_rows: Tensor,
    epsilon_rows: Tensor,
    charges: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """Return the ordinary pairs' Lennard-Jones and real-space Coulomb energies, in kJ/mol.

    ``index`` [2, C] holds the neighbor pairs and ``partners`` [N, M] each atom's excluded
    and 1-4 partners, which are skipped. Every other pair closer than ``cutoff`` adds
    u(r) - u(rc), from ``sigma_rows`` and ``epsilon_rows`` in ``pair_index`` order, and
    C q_i q_j [erfc(alpha r)/r - erfc(alpha rc)/rc], from ``charges`` [N].
    """
    first, second = index[0], index[1]
    vectors = space._displacement_wrapped(positions[first], positions[second], cell)
    squared = space.strained(vectors, strain).square().sum(dim=-1)
    # Padding columns are self-pairs, and partners are excluded or counted as 1-4 pairs.
    partner = (partners[first] == second[:, None]).any(dim=1)
    valid = (first != second) & ~partner & (squared < cutoff**2)
    # Masked columns take r^2 = 1, so no infinity reaches the energy or its gradient.
    safe = torch.where(valid, squared, 1.0)
    cutoff_energy = _pair_energy(cutoff**2, sigma_rows, epsilon_rows)
    sigma, epsilon, shift = _pair_rows(
        lj_types,
        first,
        second,
        num_lj_types,
        sigma_rows,
        epsilon_rows,
        cutoff_energy,
    )
    lennard_jones = _pair_energy(safe, sigma, epsilon) - shift
    distance = safe.sqrt()
    products = charges[first] * charges[second]
    coulomb_shift = math.erfc(alpha * cutoff) / cutoff
    coulomb = products * (torch.erfc(alpha * distance) / distance - coulomb_shift)
    return (
        torch.where(valid, lennard_jones, 0.0).sum(),
        units.COULOMB * torch.where(valid, coulomb, 0.0).sum(),
    )


def one_four_energy(
    positions: Tensor,
    cell: Tensor,
    strain: Tensor | None,
    one_four: Tensor,
    lj_types: Tensor,
    num_lj_types: int,
    sigma_rows: Tensor,
    epsilon_rows: Tensor,
    charges: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """Return the Lennard-Jones and Coulomb energies of the 1-4 pairs [P, 2], in kJ/mol.

    Each pair adds 0.5 u(r) and 0.5 C q_i q_j / r, unshifted and at any distance. The mesh
    already holds C q_i q_j erf(alpha r)/r, so the Coulomb term here is
    C q_i q_j (0.5 - erf(alpha r))/r.
    """
    left, right = one_four[:, 0], one_four[:, 1]
    squared = _listed_squares(positions, cell, strain, one_four)
    sigma, epsilon = _pair_rows(lj_types, left, right, num_lj_types, sigma_rows, epsilon_rows)
    distance = squared.sqrt()
    coulomb = charges[left] * charges[right] * (0.5 - torch.erf(alpha * distance)) / distance
    return 0.5 * _pair_energy(squared, sigma, epsilon).sum(), units.COULOMB * coulomb.sum()


def excluded_energy(
    positions: Tensor,
    cell: Tensor,
    strain: Tensor | None,
    excluded: Tensor,
    charges: Tensor,
    alpha: float,
) -> Tensor:
    """Return -C q_i q_j erf(alpha r)/r over the excluded pairs [E, 2], in kJ/mol.

    Excluded pairs do not interact; this removes their share of the mesh energy.
    """
    distance = _listed_squares(positions, cell, strain, excluded).sqrt()
    products = charges[excluded[:, 0]] * charges[excluded[:, 1]]
    return -units.COULOMB * (products * torch.erf(alpha * distance) / distance).sum()


__all__ = (
    "excluded_energy",
    "geometric_rows",
    "one_four_energy",
    "ordinary_energy",
    "pair_classes",
    "partner_table",
)
