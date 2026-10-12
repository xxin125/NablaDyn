"""The sum of models: one energy, the sum of the parts' energies."""

import torch
from torch import Tensor, nn

from ...neighbors import NeighborPairs
from .._base import Potential


def _check_matching(models) -> None:
    """Raise ``ValueError`` unless the models share their atom count, dtype and device."""
    first = models[0]
    shared = (first.num_atoms, first.dtype, first.device)
    for model in models[1:]:
        if (model.num_atoms, model.dtype, model.device) != shared:
            raise ValueError(
                f"the models of a sum must match: {first.num_atoms} atoms in {first.dtype} on "
                f"{first.device}, against {model.num_atoms} in {model.dtype} on "
                f"{model.device}; convert a sum as a whole, not one of its models",
            )


def _check_models(models: tuple[Potential, ...]) -> None:
    """Raise ``TypeError`` or ``ValueError`` unless there are two or more matching Potentials."""
    if not all(isinstance(model, Potential) for model in models):
        raise TypeError("every model of a sum must be a Potential")
    if len(models) < 2:
        raise ValueError(f"a sum needs at least two models, not {len(models)}")
    _check_matching(models)


def _own_pairs(pairs: NeighborPairs, cutoff: float) -> NeighborPairs:
    """Return the first list, ascending, that reaches ``cutoff``, with the shorter ones inside.

    The lists are ``pairs.inner`` and then ``pairs``, which reaches every part's cutoff.
    """
    lists = (*pairs.inner, pairs)
    own = [listed for listed in lists if listed.cutoff >= cutoff][0]
    shorter = tuple(listed for listed in lists if listed.cutoff < own.cutoff)
    return NeighborPairs(own.index, own.num_atoms, own.cutoff, shorter)


class Sum(Potential):
    """Models whose energies add, E = sum_k E_k.

    Built by ``potential.sum``; see its docstring.
    """

    def __init__(self, *models: Potential):
        super().__init__()
        _check_models(models)
        self._cutoff = max(model.cutoff for model in models)
        self.parts = nn.ModuleList(models)

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def num_atoms(self) -> int:
        return self.parts[0].num_atoms

    # A model converted alone after the sum was built, with ``to``, would compute the sum in
    # mixed precision or across devices; every reader of dtype and device checks first.
    @property
    def dtype(self) -> torch.dtype:
        _check_matching(self.parts)
        return self.parts[0].dtype

    @property
    def device(self) -> torch.device:
        _check_matching(self.parts)
        return self.parts[0].device

    @property
    def _list_cutoffs(self) -> tuple[float, ...]:
        return tuple(sorted({value for part in self.parts for value in part._list_cutoffs}))

    def _trained_masks(self) -> dict[str, Tensor]:
        return {
            f"parts.{index}.{name}": mask
            for index, part in enumerate(self.parts)
            for name, mask in part._trained_masks().items()
        }

    def _energy_wrapped(
        self,
        positions: Tensor,
        cell: Tensor,
        pairs: NeighborPairs,
        strain: Tensor | None = None,
    ) -> Tensor:
        # Each part reads the shortest given list that reaches its cutoff and drops the pairs
        # beyond it, so a list per cutoff changes the cost, never the energy.
        energies = [
            part._energy_wrapped(positions, cell, _own_pairs(pairs, part.cutoff), strain)
            for part in self.parts
        ]
        return torch.stack(energies).sum()


__all__ = ("Sum",)
