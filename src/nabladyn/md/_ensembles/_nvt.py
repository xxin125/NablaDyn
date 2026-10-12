"""NVT: velocity Verlet between half steps of a Nose-Hoover chain."""

import functools
import math
from numbers import Integral, Real

import torch
from torch import Tensor

from ...observables import kinetic_energy
from ...potential import Potential
from ...units import BOLTZMANN
from .._simulation import Simulation
from .._state import ChainState, _Frame
from ._chain import chain_masses, half_step, heat_bath_energy, own_chain
from ._nve import NVE


def _check_settings(temperature: Real, thermostat_timescale: Real) -> None:
    """Raise ``ValueError`` unless the temperature and the timescale are finite and positive."""
    settings = (
        ("temperature", temperature),
        ("thermostat_timescale", thermostat_timescale),
    )
    for name, value in settings:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")


def _check_atom_count(masses: Tensor) -> None:
    """Raise ``ValueError`` unless there are at least 2 atoms, for 3N - 3 degrees of freedom."""
    if masses.numel() < 2:
        raise ValueError("NVT needs at least 2 atoms, for 3N - 3 degrees of freedom")


class NVT:
    """The halves of a step at constant temperature.

    First half: a chain half step, then the Verlet first half. Second half: the Verlet
    second half, after the force. Closing: a chain half step. The chain counts g = 3N - 3.

    The chain works in float64 whatever the frame's dtype: its scale factor is within about
    1e-3 of 1, and float32 rounding makes the conserved energy drift. It reads K carried
    from the previous program, keeping its serial scalar work out of the atom-sum kernels.

    Raises ``ValueError`` for fewer than 2 atoms or a chain state that is not [3] float64
    on the masses' device.
    """

    fields = ("conserved_energy",)
    needs_virial = False

    def __init__(
        self,
        timestep: float,
        masses: Tensor,
        frame: _Frame,
        *,
        temperature: float,
        timescale: float,
        chain: ChainState | None,
    ):
        _check_atom_count(masses)
        self.settings = {
            "ensemble": "NVT",
            "temperature": temperature,
            "thermostat_timescale": timescale,
        }
        self._verlet = NVE(timestep, masses, frame)
        self._masses = masses
        self._degrees_of_freedom = 3 * masses.numel() - 3
        wide = {"dtype": torch.float64, "device": masses.device}
        self._timestep = torch.tensor(timestep, **wide)
        self._kT = torch.tensor(BOLTZMANN * temperature, **wide)
        self._chain_masses = chain_masses(
            self._degrees_of_freedom,
            self._kT,
            torch.tensor(timescale, **wide),
        )
        self._positions, self._momenta = own_chain(chain, masses.device, "thermostat")
        self._kinetic = kinetic_energy(frame.velocities, masses).double()  # K of the frame

    def first_half(self, frame: _Frame) -> None:
        self._chain_half_step(frame)
        self._verlet.first_half(frame)

    def second_half(self, frame: _Frame) -> None:
        self._verlet.second_half(frame)
        self._kinetic.copy_(kinetic_energy(frame.velocities, self._masses))

    def closing(self, frame: _Frame) -> None:
        self._chain_half_step(frame)

    def tensors(self) -> tuple[Tensor, ...]:
        return self._positions, self._momenta, self._kinetic

    def state(self) -> dict:
        return {"thermostat": ChainState(self._positions.clone(), self._momenta.clone())}

    def panel(self, frame: _Frame, energy: Tensor, kinetic: Tensor) -> tuple[Tensor, ...]:
        bath = heat_bath_energy(
            self._positions,
            self._momenta,
            self._chain_masses,
            self._degrees_of_freedom,
            self._kT,
        )
        return ((energy.double() + kinetic.double() + bath).to(energy.dtype),)

    def _chain_half_step(self, frame: _Frame) -> None:
        """Advance the chain by dt/2 against the carried K; scale velocities and K."""
        factor, positions, momenta = half_step(
            self._kinetic,
            self._positions,
            self._momenta,
            self._chain_masses,
            self._degrees_of_freedom,
            self._kT,
            self._timestep,
        )
        frame.velocities.mul_(factor.to(frame.velocities.dtype))
        self._positions.copy_(positions)
        self._momenta.copy_(momenta)
        self._kinetic.copy_(self._kinetic * factor.square())


def nvt(
    model: Potential,
    positions: Tensor,
    velocities: Tensor,
    masses: Tensor,
    cell: Tensor,
    *,
    timestep: Real,
    temperature: Real,
    thermostat_timescale: Real,
    skin: Real = 0.1,
    capacity_multiplier: Real = 1.25,
    images: Tensor | None = None,
    step: Integral = 0,
    thermostat: ChainState | None = None,
) -> Simulation:
    r"""Build a simulation at constant temperature, by velocity Verlet and a Nose-Hoover chain.

    Takes the arguments of ``nve``, plus ``temperature`` in K, ``thermostat_timescale``
    $`\tau`$ in ps and, to continue a run, the ``thermostat`` state (float64). The 3 chain
    masses are $`g k_B T \tau^2`$, $`k_B T \tau^2`$ and $`k_B T \tau^2`$, $`g = 3N - 3`$. The
    panel of samples (``FIELDS``, see ``Simulation.run``) adds ``conserved_energy``: $`U + K`$
    plus the chain's energy.

    Raises ``ValueError`` unless the temperature and the timescale are finite and positive,
    for fewer than 2 atoms or a ``thermostat`` that is not [3] float64 on the model's
    device, and as ``nve`` does.
    """
    _check_settings(temperature, thermostat_timescale)
    ensemble = functools.partial(
        NVT,
        temperature=float(temperature),
        timescale=float(thermostat_timescale),
        chain=thermostat,
    )
    return Simulation(
        model,
        positions,
        velocities,
        masses,
        cell,
        ensemble=ensemble,
        timestep=timestep,
        skin=skin,
        capacity_multiplier=capacity_multiplier,
        images=images,
        step=step,
    )


__all__ = ("nvt",)
