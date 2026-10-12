"""The Potential contract: a model whose one scalar energy defines forces and virial."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .. import space
from ..neighbors import NeighborPairs


def _check_tensor(
    name: str,
    value: Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Raise ``ValueError`` unless an input is a tensor of this shape, dtype and device."""
    if not isinstance(value, Tensor) or value.shape != shape:
        raise ValueError(f"{name} must be a tensor of shape {list(shape)}")
    if value.dtype != dtype or value.device != device:
        raise ValueError(f"{name} must be {dtype} on {device} to match the model")


def _check_pairs(pairs: NeighborPairs, num_atoms: int, cutoff: float, device: torch.device) -> None:
    """Raise ``ValueError`` unless the pairs and their inner lists fit these atoms and device.

    The list must reach the cutoff.
    """
    for listed in (pairs, *pairs.inner):
        index = listed.index
        if index.ndim != 2 or index.shape[0] != 2 or index.device != device:
            raise ValueError(f"pairs.index must be a [2, C] tensor on {device}")
        if listed.num_atoms != num_atoms:
            raise ValueError(
                f"the pairs are for {listed.num_atoms} atoms, the model for {num_atoms}",
            )
    # A list is complete only within its own cutoff; beyond it, pairs drop out as atoms move.
    if pairs.cutoff < cutoff:
        raise ValueError(
            f"the neighbor list's cutoff {pairs.cutoff} nm is below the model's {cutoff} nm",
        )


class Potential(nn.Module, ABC):
    """A model of a fixed set of atoms with one scalar energy in kJ/mol.

    It owns parameters and a topology, never frames or neighbor lists. Calling it,
    ``model(positions, cell, pairs)``, checks the inputs, wraps the positions into the cell
    and returns the energy. Forces and virial come from ``nabladyn.quantity``.
    """

    @property
    @abstractmethod
    def cutoff(self) -> float:
        """The interaction cutoff in nm."""

    @property
    @abstractmethod
    def num_atoms(self) -> int:
        """The number of atoms in the model's topology."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The floating dtype the model computes in."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device the model computes on."""

    @property
    def _list_cutoffs(self) -> tuple[float, ...]:
        """The cutoffs in nm of the neighbor lists the model reads, ascending; the last is its own.

        MD and fit build a list at each; a sum reads one per cutoff of its models.
        """
        return (self.cutoff,)

    def forward(
        self,
        positions: Tensor,
        cell: Tensor,
        pairs: NeighborPairs,
        strain: Tensor | None = None,
    ) -> Tensor:
        """Return the energy in kJ/mol of positions [N, 3] in nm, which may lie outside the cell.

        ``cell`` [3] holds the cell lengths in nm. ``pairs`` must come from a neighbor list
        updated to these positions and cell, with a cutoff at least the model's. The
        optional ``strain`` [3, 3] deforms positions and cell together, x -> (I + strain) x;
        the virial is the derivative at zero strain. Positions, cell and strain share the
        model's dtype and device. No host sync.

        Raises ``ValueError`` for inputs of another shape, dtype or device, or for pairs of
        another system or a shorter cutoff.
        """
        _check_tensor("positions", positions, (self.num_atoms, 3), self.dtype, self.device)
        _check_tensor("cell", cell, (3,), self.dtype, self.device)
        if strain is not None:
            _check_tensor("strain", strain, (3, 3), self.dtype, self.device)
        _check_pairs(pairs, self.num_atoms, self.cutoff, self.device)
        wrapped, _ = space.wrap(positions, cell)
        return self._energy_wrapped(wrapped, cell, pairs, strain)

    @abstractmethod
    def _energy_wrapped(
        self,
        positions: Tensor,
        cell: Tensor,
        pairs: NeighborPairs,
        strain: Tensor | None = None,
    ) -> Tensor:
        """Return ``forward``'s energy for positions already in [0, L), without checks."""

    def _trained_masks(self) -> dict[str, Tensor]:
        """Return a bool mask for each parameter that trains only some of its entries.

        A mask broadcasts to its parameter and is True on the trained entries; a fit holds
        the others. The default, no masks, trains whole parameters.
        """
        return {}


__all__ = ("Potential",)
