"""Energy, forces and virial from one scalar energy.

Each function binds an energy, any callable ``energy(positions, cell, pairs, strain=None)``
such as a Potential, and returns a function of ``(positions, cell, pairs)`` that computes
all of its results from one evaluation, by ``torch.func``:

- E is a scalar in kJ/mol;
- F = -dE/dpositions is [N, 3] in kJ/mol/nm;
- W = -dE/dstrain at zero strain is [3, 3] in kJ/mol, with no kinetic term and no
  impulsive term from a hard cutoff. The pressure is (2K + trace W) / (3V).
"""

from collections.abc import Callable

import torch
from torch import Tensor

from .neighbors import NeighborPairs


def _with_strain(energy_fn: Callable[..., Tensor]):
    """Return ``energy_fn`` with the strain as its second argument, for ``torch.func``."""

    def strained_energy(
        positions: Tensor,
        strain: Tensor,
        cell: Tensor,
        pairs: NeighborPairs,
    ) -> Tensor:
        return energy_fn(positions, cell, pairs, strain=strain)

    return strained_energy


def force(energy_fn: Callable[..., Tensor]):
    """Bind F, the forces [N, 3] in kJ/mol/nm."""
    gradient = torch.func.grad(energy_fn, argnums=0)

    def evaluate(positions: Tensor, cell: Tensor, pairs: NeighborPairs) -> Tensor:
        return -gradient(positions, cell, pairs)

    return evaluate


def energy_and_force(energy_fn: Callable[..., Tensor]):
    """Bind (E, F), the energy in kJ/mol and the forces [N, 3] in kJ/mol/nm."""
    gradient_and_value = torch.func.grad_and_value(energy_fn, argnums=0)

    def evaluate(positions: Tensor, cell: Tensor, pairs: NeighborPairs) -> tuple[Tensor, Tensor]:
        gradient, energy = gradient_and_value(positions, cell, pairs)
        return energy, -gradient

    return evaluate


def force_virial(energy_fn: Callable[..., Tensor]):
    """Bind (F, W), the forces [N, 3] in kJ/mol/nm and the virial [3, 3] in kJ/mol."""
    gradient = torch.func.grad(_with_strain(energy_fn), argnums=(0, 1))

    def evaluate(positions: Tensor, cell: Tensor, pairs: NeighborPairs) -> tuple[Tensor, Tensor]:
        position_grad, strain_grad = gradient(positions, positions.new_zeros((3, 3)), cell, pairs)
        return -position_grad, -strain_grad

    return evaluate


def virial(energy_fn: Callable[..., Tensor]):
    """Bind W, the virial [3, 3] in kJ/mol."""
    gradient = torch.func.grad(_with_strain(energy_fn), argnums=1)

    def evaluate(positions: Tensor, cell: Tensor, pairs: NeighborPairs) -> Tensor:
        return -gradient(positions, positions.new_zeros((3, 3)), cell, pairs)

    return evaluate


def energy_virial(energy_fn: Callable[..., Tensor]):
    """Bind (E, W), the energy in kJ/mol and the virial [3, 3] in kJ/mol."""
    gradient_and_value = torch.func.grad_and_value(_with_strain(energy_fn), argnums=1)

    def evaluate(positions: Tensor, cell: Tensor, pairs: NeighborPairs) -> tuple[Tensor, Tensor]:
        strain_grad, energy = gradient_and_value(
            positions,
            positions.new_zeros((3, 3)),
            cell,
            pairs,
        )
        return energy, -strain_grad

    return evaluate


def energy_force_virial(energy_fn: Callable[..., Tensor]):
    r"""Bind (E, F, W): the energy and virial [3, 3] in kJ/mol, forces [N, 3] in kJ/mol/nm.

    ``energy_fn`` is a model, or any energy with its call signature. The result is a
    function of ``(positions, cell, pairs)``, with ``pairs`` from a ``NeighborList``, that
    returns all three from one evaluation. $`F = -\partial E / \partial r`$ in the positions
    r; $`W = -\partial E / \partial \varepsilon`$ in the strain $`\varepsilon`$, at zero
    strain, with no kinetic term, so the pressure is $`(2K + \mathrm{tr}(W)) / (3V)`$.
    """
    gradient_and_value = torch.func.grad_and_value(_with_strain(energy_fn), argnums=(0, 1))

    def evaluate(
        positions: Tensor,
        cell: Tensor,
        pairs: NeighborPairs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        (position_grad, strain_grad), energy = gradient_and_value(
            positions,
            positions.new_zeros((3, 3)),
            cell,
            pairs,
        )
        return energy, -position_grad, -strain_grad

    return evaluate


__all__ = (
    "force",
    "energy_and_force",
    "force_virial",
    "virial",
    "energy_virial",
    "energy_force_virial",
)
