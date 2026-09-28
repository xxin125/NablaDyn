"""The supervised loss: predicted energies, forces and virials, and their scaled errors.

The trained parameters are explicit inputs, so the loss is differentiable in them.
"""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .. import quantity
from ..neighbors import NeighborPairs

CHANNELS = ("energy", "forces", "virial")
# The quantity function of each set of channels; it returns them in channel order.
_QUANTITIES = {
    ("forces",): quantity.force,
    ("virial",): quantity.virial,
    ("energy", "forces"): quantity.energy_and_force,
    ("energy", "virial"): quantity.energy_virial,
    ("forces", "virial"): quantity.force_virial,
    ("energy", "forces", "virial"): quantity.energy_force_virial,
}


def scales(
    energy: Tensor | None = None,
    forces: Tensor | None = None,
    virial: Tensor | None = None,
) -> dict[str, float]:
    """Return each given channel's scale from its training labels.

    Energy [F] and virial [F, 3, 3], in kJ/mol: the RMS about the mean over frames. Forces
    [F, N, 3], in kJ/mol/nm: the plain RMS. Labels constant over frames fall back to the
    plain RMS, and all-zero labels to 1. Syncs with the host.
    """
    result = {}
    channels = (("energy", energy, True), ("forces", forces, False), ("virial", virial, True))
    for name, labels, about_mean in channels:
        if labels is not None:
            rms = float(labels.square().mean().sqrt())
            spread = rms
            if about_mean:
                spread = float((labels - labels.mean(dim=0)).square().mean().sqrt())
            result[name] = spread or rms or 1.0
    return result


def frame_channels(energy, channels: tuple[str, ...], num_atoms: int, cutoff: float):
    """Return a one-frame function giving ``channels`` of ``energy``, such as a Potential.

    ``channels`` is a nonempty subsequence of ``CHANNELS``. The function takes positions
    [N, 3], a cell [3] and a pair index [2, C] and returns a tuple in ``CHANNELS`` order.
    """
    evaluate = energy if channels == ("energy",) else _QUANTITIES[channels](energy)

    def one(positions: Tensor, cell: Tensor, index: Tensor) -> tuple[Tensor, ...]:
        values = evaluate(positions, cell, NeighborPairs(index, num_atoms, cutoff))
        return values if isinstance(values, tuple) else (values,)

    return one


def predict(
    model: nn.Module,
    parameters: Mapping[str, Tensor],
    positions: Tensor,
    cells: Tensor,
    index: Tensor,
    channels: tuple[str, ...],
) -> dict[str, Tensor]:
    """Return the channels of B frames by name, vmapped over the frames.

    Takes positions [B, N, 3], cells [B, 3] and pair indices [B, 2, C], padded with
    self-pairs that the model masks; gives energy [B], forces [B, N, 3], virial [B, 3, 3].
    ``parameters`` replaces the named model parameters, so the channels are differentiable
    in them. ``channels`` is a nonempty subsequence of ``CHANNELS``.
    """

    def energy(
        positions: Tensor,
        cell: Tensor,
        pairs: NeighborPairs,
        strain: Tensor | None = None,
    ) -> Tensor:
        return torch.func.functional_call(
            model,
            parameters,
            (positions, cell, pairs),
            {"strain": strain},
        )

    one = frame_channels(energy, channels, model.num_atoms, model.cutoff)
    return dict(zip(channels, torch.func.vmap(one)(positions, cells, index)))


def loss(
    predicted: Mapping[str, Tensor],
    labels: Mapping[str, Tensor],
    weights: Mapping[str, float | Tensor],
    scales: Mapping[str, float | Tensor],
    coefficients: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return a batch's loss, the weighted sum of its channel terms, and the terms by name.

    A channel's term is each frame's mean of ((X - X*) / s_X)^2 over its components, times
    the frame's coefficient [B] (1/B when ``coefficients`` is None), summed over the B frames.
    """
    terms = {}
    for name, value in predicted.items():
        squares = ((value - labels[name]) / scales[name]).square()
        per_frame = squares.reshape(squares.shape[0], -1).mean(dim=1)
        if coefficients is None:
            terms[name] = per_frame.mean()
        else:
            terms[name] = (coefficients * per_frame).sum()
    return sum(weights[name] * term for name, term in terms.items()), terms


__all__ = ()
