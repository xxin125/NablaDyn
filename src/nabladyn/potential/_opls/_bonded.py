"""Bonded OPLS terms: harmonic bonds and angles, periodic dihedrals."""

import torch
from torch import Tensor

from ... import space


def _vectors(
    positions: Tensor,
    cell: Tensor,
    strain: Tensor | None,
    left: Tensor,
    right: Tensor,
) -> Tensor:
    """Return the strained minimum-image vectors ``positions[left] - positions[right]``."""
    vectors = space._displacement_wrapped(positions[left], positions[right], cell)
    return space.strained(vectors, strain)


def bond_energy(
    positions: Tensor,
    cell: Tensor,
    strain: Tensor | None,
    bonds: Tensor,
    r0: Tensor,
    k: Tensor,
) -> Tensor:
    """Return the sum of 0.5 k (r - r0)^2 over ``bonds`` [B, 2], in kJ/mol.

    ``r0`` [B] in nm and ``k`` [B] in kJ/mol/nm^2 hold each bond's values.
    """
    vectors = _vectors(positions, cell, strain, bonds[:, 0], bonds[:, 1])
    length = torch.linalg.vector_norm(vectors, dim=-1)
    return (0.5 * k * (length - r0).square()).sum()


def angle_energy(
    positions: Tensor,
    cell: Tensor,
    strain: Tensor | None,
    angles: Tensor,
    theta0: Tensor,
    k: Tensor,
) -> Tensor:
    """Return the sum of 0.5 k (theta - theta0)^2 over ``angles`` [A, 3], in kJ/mol.

    Rows are (i, j, k) with the vertex j; theta = atan2(|a x b|, a . b), a = r_i - r_j and
    b = r_k - r_j. ``theta0`` [A] in rad and ``k`` [A] in kJ/mol/rad^2 are per angle.
    """
    first = _vectors(positions, cell, strain, angles[:, 0], angles[:, 1])
    second = _vectors(positions, cell, strain, angles[:, 2], angles[:, 1])
    sine = torch.linalg.vector_norm(torch.linalg.cross(first, second, dim=-1), dim=-1)
    theta = torch.atan2(sine, (first * second).sum(dim=-1))
    return (0.5 * k * (theta - theta0).square()).sum()


def dihedral_energy(
    positions: Tensor,
    cell: Tensor,
    strain: Tensor | None,
    dihedrals: Tensor,
    k: Tensor,
    periodicity: Tensor,
    phase: Tensor,
) -> Tensor:
    """Return the sum of k [1 + cos(n phi - phase)] over ``dihedrals`` [D, 4], in kJ/mol.

    phi is the IUPAC dihedral of each row (i, j, k, l), in (-pi, pi]: 0 cis, pi trans,
    positive when bond j-i turns clockwise onto k-l seen along j -> k. ``k`` [D] in kJ/mol,
    the integer ``periodicity`` n [D] and ``phase`` [D] in rad are per dihedral.
    """
    first = _vectors(positions, cell, strain, dihedrals[:, 1], dihedrals[:, 0])
    middle = _vectors(positions, cell, strain, dihedrals[:, 2], dihedrals[:, 1])
    last = _vectors(positions, cell, strain, dihedrals[:, 3], dihedrals[:, 2])
    near = torch.linalg.cross(first, middle, dim=-1)
    far = torch.linalg.cross(middle, last, dim=-1)
    # phi = atan2(|middle| first . far, near . far) keeps the sign and stays accurate near 0
    # and pi.
    sine = torch.linalg.vector_norm(middle, dim=-1) * (first * far).sum(dim=-1)
    phi = torch.atan2(sine, (near * far).sum(dim=-1))
    return (k * (1.0 + torch.cos(periodicity * phi - phase))).sum()


__all__ = ("angle_energy", "bond_energy", "dihedral_energy")
