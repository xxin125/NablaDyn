"""NPT: isotropic Martyna-Tobias-Klein (MTK) with Nose-Hoover chains.

The chains act on the particles and on the barostat, in the operator order of Tuckerman,
Alejandre, Lopez-Rendon, Jochim and Martyna (2006). The barostat's drag is split from the
force's kick, and the box grows in two halves around the drift.
"""

import functools
import math
from numbers import Integral, Real

import torch
from torch import Tensor

from ... import space
from ...observables import kinetic_energy
from ...potential import Potential
from ...units import BOLTZMANN
from .._simulation import Simulation
from .._state import BarostatState, ChainState, _Frame
from ._chain import chain_masses, half_step, heat_bath_energy, own_chain
from ._verlet import half_kick


def _check_box_momentum(momentum: Tensor, device: torch.device) -> None:
    """Raise ``ValueError`` unless the box momentum is a [] float64 tensor on the device."""
    if (
        not isinstance(momentum, Tensor)
        or momentum.shape != ()
        or momentum.dtype != torch.float64
        or momentum.device != device
    ):
        raise ValueError(f"barostat.momentum must be a [] float64 tensor on {device}")


def _check_settings(
    temperature: Real,
    pressure: Real,
    thermostat_timescale: Real,
    barostat_timescale: Real,
) -> None:
    """Raise ``ValueError`` unless the settings are finite, and all but the pressure positive."""
    positive = (
        ("temperature", temperature),
        ("thermostat_timescale", thermostat_timescale),
        ("barostat_timescale", barostat_timescale),
    )
    for name, value in positive:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(pressure, bool) or not isinstance(pressure, Real) or not math.isfinite(pressure):
        raise ValueError("pressure must be a finite number")


def _check_atom_count(masses: Tensor) -> None:
    """Raise ``ValueError`` unless there are at least 2 atoms, for 3N - 3 degrees of freedom."""
    if masses.numel() < 2:
        raise ValueError("NPT needs at least 2 atoms, for 3N - 3 degrees of freedom")


class NPT:
    """The halves of a step at constant pressure and temperature.

    First half: the barostat's and the particles' chains, then the box momentum, each over
    dt/2; the drag and the half kick; then the drift, with box and positions grown by
    exp(v_eps dt/2) before and after it. Second half, after the force and virial: the half
    kick and the drag. Closing: the box momentum, then the particles' and the barostat's
    chains, each over dt/2.

    The cell scales isotropically about the origin, L = L_0 e^eps, at the rate
    v_eps = p_eps / M.
    - The drive is G = alpha 2K + trace W - 3 P V, with W the virial and alpha = 1 + 1/N.
    - The drag scales velocities by exp(-alpha v_eps dt/2).
    - The box mass is M = 3 (N + 1) kT tau_b^2.
    - The barostat's chain has one degree of freedom and masses kT tau_b^2.
    - The particles' chain counts g = 3N - 3.
    - The barostat, the chains and a master copy of the cell work in float64.

    Raises ``ValueError`` for fewer than 2 atoms, a chain that is not [3] float64 or a box
    momentum that is not [] float64 on the masses' device.
    """

    fields = ("volume", "conserved_energy")
    needs_virial = True

    def __init__(
        self,
        timestep: float,
        masses: Tensor,
        frame: _Frame,
        *,
        temperature: float,
        pressure: float,
        thermostat_timescale: float,
        barostat_timescale: float,
        thermostat: ChainState | None,
        barostat: BarostatState | None,
    ):
        _check_atom_count(masses)
        self.settings = {
            "ensemble": "NPT",
            "temperature": temperature,
            "pressure": pressure,
            "thermostat_timescale": thermostat_timescale,
            "barostat_timescale": barostat_timescale,
        }
        wide = {"dtype": torch.float64, "device": masses.device}
        num_atoms = masses.numel()
        self._masses = masses
        self._inverse_masses = 1.0 / masses[:, None]
        self._degrees_of_freedom = degrees = 3 * num_atoms - 3
        self._alpha = 1.0 + 1.0 / num_atoms
        self._timestep = torch.tensor(timestep, **wide)
        self._half_timestep = torch.tensor(0.5 * timestep, **wide)
        # In the frame's dtype, for the kicks and the drift.
        self._frame_timestep = masses.new_tensor(timestep)
        self._frame_half_timestep = masses.new_tensor(0.5 * timestep)
        self._kT = torch.tensor(BOLTZMANN * temperature, **wide)
        self._pressure = torch.tensor(pressure, **wide)
        barostat_tau = torch.tensor(barostat_timescale, **wide)
        self._box_mass = 3 * (num_atoms + 1) * self._kT * barostat_tau.square()
        self._chain_masses = chain_masses(
            degrees,
            self._kT,
            torch.tensor(thermostat_timescale, **wide),
        )
        self._box_chain_masses = chain_masses(1, self._kT, barostat_tau)
        self._positions, self._momenta = own_chain(thermostat, masses.device, "thermostat")
        if barostat is None:
            barostat = BarostatState(torch.zeros((), **wide), None)
        _check_box_momentum(barostat.momentum, masses.device)
        self._box_momentum = barostat.momentum.detach().clone()
        self._box_positions, self._box_momenta = own_chain(
            barostat.chain,
            masses.device,
            "barostat.chain",
        )
        self._kinetic = kinetic_energy(frame.velocities, masses).double()  # K of the frame
        self._cell = frame.cell.double().clone()

    def first_half(self, frame: _Frame) -> None:
        self._box_chain_half_step()
        self._chain_half_step(frame)
        self._box_momentum.add_(self._half_timestep * self._box_force(frame))
        rate = self._box_momentum / self._box_mass
        drag = torch.exp(-self._alpha * rate * self._half_timestep).to(frame.velocities.dtype)
        velocities = half_kick(
            frame.velocities * drag,
            frame.forces,
            self._inverse_masses,
            self._frame_half_timestep,
        )
        growth = torch.exp(rate * self._half_timestep)
        scale = growth.to(frame.positions.dtype)
        moved = (frame.positions * scale + self._frame_timestep * velocities) * scale
        self._cell.mul_(growth).mul_(growth)
        frame.cell.copy_(self._cell)
        positions, crossings = space.wrap(moved, frame.cell)
        frame.velocities.copy_(velocities)
        frame.positions.copy_(positions)
        frame.images.add_(crossings)

    def second_half(self, frame: _Frame) -> None:
        rate = self._box_momentum / self._box_mass
        drag = torch.exp(-self._alpha * rate * self._half_timestep).to(frame.velocities.dtype)
        velocities = half_kick(
            frame.velocities,
            frame.forces,
            self._inverse_masses,
            self._frame_half_timestep,
        )
        frame.velocities.copy_(velocities * drag)
        self._kinetic.copy_(kinetic_energy(frame.velocities, self._masses))

    def closing(self, frame: _Frame) -> None:
        self._box_momentum.add_(self._half_timestep * self._box_force(frame))
        self._chain_half_step(frame)
        self._box_chain_half_step()

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self._positions,
            self._momenta,
            self._kinetic,
            self._box_momentum,
            self._box_positions,
            self._box_momenta,
            self._cell,
        )

    def state(self) -> dict:
        return {
            "thermostat": ChainState(self._positions.clone(), self._momenta.clone()),
            "barostat": BarostatState(
                self._box_momentum.clone(),
                ChainState(self._box_positions.clone(), self._box_momenta.clone()),
            ),
        }

    def panel(self, frame: _Frame, energy: Tensor, kinetic: Tensor) -> tuple[Tensor, ...]:
        volume = self._cell.prod()
        thermostat_bath = heat_bath_energy(
            self._positions,
            self._momenta,
            self._chain_masses,
            self._degrees_of_freedom,
            self._kT,
        )
        barostat_bath = heat_bath_energy(
            self._box_positions,
            self._box_momenta,
            self._box_chain_masses,
            1,
            self._kT,
        )
        baths = thermostat_bath + barostat_bath
        conserved = (
            energy.double()
            + kinetic.double()
            + baths
            + self._pressure * volume
            + self._box_momentum.square() / (2.0 * self._box_mass)
        )
        return volume.to(energy.dtype), conserved.to(energy.dtype)

    def _box_force(self, frame: _Frame) -> Tensor:
        """Return the box force G = alpha 2K + trace W - 3 P V in kJ/mol, with the carried K."""
        return (
            self._alpha * 2.0 * self._kinetic
            + frame.virial.trace().double()
            - 3.0 * self._pressure * self._cell.prod()
        )

    def _chain_half_step(self, frame: _Frame) -> None:
        """Advance the particles' chain by dt/2 against the carried K; scale velocities and K."""
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

    def _box_chain_half_step(self) -> None:
        """Advance the barostat's chain by dt/2 against the box's kinetic energy; scale p_eps."""
        kinetic = self._box_momentum.square() / (2.0 * self._box_mass)
        factor, positions, momenta = half_step(
            kinetic,
            self._box_positions,
            self._box_momenta,
            self._box_chain_masses,
            1,
            self._kT,
            self._timestep,
        )
        self._box_momentum.mul_(factor)
        self._box_positions.copy_(positions)
        self._box_momenta.copy_(momenta)


def npt(
    model: Potential,
    positions: Tensor,
    velocities: Tensor,
    masses: Tensor,
    cell: Tensor,
    *,
    timestep: Real,
    temperature: Real,
    pressure: Real,
    thermostat_timescale: Real,
    barostat_timescale: Real,
    skin: Real = 0.1,
    capacity_multiplier: Real = 1.25,
    images: Tensor | None = None,
    step: Integral = 0,
    thermostat: ChainState | None = None,
    barostat: BarostatState | None = None,
) -> Simulation:
    r"""Build a simulation at constant pressure and temperature, by isotropic MTK.

    MTK is Martyna-Tobias-Klein dynamics. Takes the arguments of ``nvt``, plus ``pressure``
    in kJ/mol/nm^3 (``units.BAR`` converts), ``barostat_timescale`` $`\tau_b`$ in ps and, to
    continue a run, the ``barostat`` state. Nose-Hoover chains of 3 thermostats act on the
    particles and on the barostat; the box mass is $`3(N + 1) k_B T \tau_b^2`$. The cell
    keeps its aspect ratio and scales about the origin.

    The panel of samples (``FIELDS``, see ``Simulation.run``) adds ``volume`` in nm^3 and
    ``conserved_energy``: $`U + K`$ plus $`PV`$, the box's kinetic energy and both chains'
    energies. Started far from ``pressure``, the box oscillates, as with any extended-system
    barostat.

    Raises ``ValueError`` unless the pressure is finite and the temperature and timescales
    finite and positive; for a ``barostat`` whose momentum is not [] float64 or whose chain
    is not [3] float64 on the model's device; and as ``nvt`` does.
    """
    _check_settings(temperature, pressure, thermostat_timescale, barostat_timescale)
    ensemble = functools.partial(
        NPT,
        temperature=float(temperature),
        pressure=float(pressure),
        thermostat_timescale=float(thermostat_timescale),
        barostat_timescale=float(barostat_timescale),
        thermostat=thermostat,
        barostat=barostat,
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


__all__ = ("npt",)
