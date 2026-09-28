"""The state of a simulation: one frame, detached from the simulation that made it."""

from typing import NamedTuple

from torch import Tensor


class _Frame(NamedTuple):
    """The resident frame of a simulation, updated in place by each step.

    Tensor addresses stay fixed for the CUDA Graph. ``energy`` [] is U at ``positions``;
    ``virial`` [3, 3] is W there when the ensemble needs it.
    """

    positions: Tensor
    velocities: Tensor
    forces: Tensor
    energy: Tensor
    virial: Tensor
    cell: Tensor
    images: Tensor


class ChainState(NamedTuple):
    """A Nose-Hoover chain: positions [3], dimensionless, and momenta [3] in kJ ps/mol."""

    positions: Tensor
    momenta: Tensor


class BarostatState(NamedTuple):
    """An isotropic barostat: the box momentum [] in kJ ps/mol, float64, and its chain."""

    momentum: Tensor
    chain: ChainState


class State(NamedTuple):
    """A copy of a simulation's frame, in nm, nm/ps and kJ/mol/nm.

    ``positions`` [N, 3] lie in [0, L); ``images`` [N, 3], int64, count lattice crossings,
    so the unwrapped positions are ``positions + images * cell``. ``forces`` [N, 3] act at
    ``positions``; ``cell`` is [3]. ``step`` counts the steps since the first frame.
    ``thermostat`` is the chain of an NVT or NPT run, ``barostat`` the barostat of an NPT
    run. Changing the tensors changes no simulation. To continue a run, pass ``positions``,
    ``velocities``, ``cell``, ``images`` and ``step``, with ``thermostat`` (NVT, NPT) and
    ``barostat`` (NPT), to ``nve``, ``nvt`` or ``npt``.
    """

    positions: Tensor
    velocities: Tensor
    forces: Tensor
    cell: Tensor
    images: Tensor
    step: int
    thermostat: ChainState | None = None
    barostat: BarostatState | None = None


__all__ = ("BarostatState", "ChainState", "State")
