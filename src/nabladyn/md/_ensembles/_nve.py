"""NVE: velocity Verlet at constant energy."""

from numbers import Integral, Real

from torch import Tensor

from ...potential import Potential
from .._simulation import Simulation
from .._state import _Frame
from ._verlet import drift, half_kick


class NVE:
    """The halves of a velocity Verlet step around the force.

    A half kick and a drift come before the force at the new positions, a half kick after.

    Every ensemble is built from the timestep, the masses and the first frame, and offers:
    - ``first_half`` and ``second_half``, which update the frame in place, the second after
      the force (and the virial, when ``needs_virial``);
    - ``closing``, None or a last part of the step, run as its own program;
    - ``tensors()``, its own state that a step updates, and ``state()``, that state for
      ``State``;
    - ``fields`` and ``panel``, its additions to the sampled panel;
    - ``settings``, its name and set points for sample files.
    """

    fields = ()
    closing = None
    needs_virial = False

    def __init__(self, timestep: float, masses: Tensor, frame: _Frame):
        self.settings = {"ensemble": "NVE"}
        # Tensors, not floats, so one compiled program serves every timestep.
        self._timestep = masses.new_tensor(timestep)
        self._half_timestep = masses.new_tensor(0.5 * timestep)
        self._inverse_masses = 1.0 / masses[:, None]

    def first_half(self, frame: _Frame) -> None:
        velocities = half_kick(
            frame.velocities,
            frame.forces,
            self._inverse_masses,
            self._half_timestep,
        )
        positions, crossings = drift(frame.positions, velocities, frame.cell, self._timestep)
        frame.velocities.copy_(velocities)
        frame.positions.copy_(positions)
        frame.images.add_(crossings)

    def second_half(self, frame: _Frame) -> None:
        velocities = half_kick(
            frame.velocities,
            frame.forces,
            self._inverse_masses,
            self._half_timestep,
        )
        frame.velocities.copy_(velocities)

    def tensors(self) -> tuple[Tensor, ...]:
        return ()

    def state(self) -> dict:
        return {}

    def panel(self, frame: _Frame, energy: Tensor, kinetic: Tensor) -> tuple[Tensor, ...]:
        return ()


def nve(
    model: Potential,
    positions: Tensor,
    velocities: Tensor,
    masses: Tensor,
    cell: Tensor,
    *,
    timestep: Real,
    skin: Real = 0.1,
    capacity_multiplier: Real = 1.25,
    images: Tensor | None = None,
    step: Integral = 0,
) -> Simulation:
    """Build a simulation at constant energy, by velocity Verlet.

    ``positions`` [N, 3] in nm, which may lie outside the cell; ``velocities`` [N, 3] in
    nm/ps; ``masses`` [N] in Da; ``cell`` [3] in nm; all in the model's dtype and on its
    device. ``timestep`` is in ps. The neighbor list takes the model's cutoff, ``skin``
    in nm and ``capacity_multiplier``. ``images`` [N, 3], int64, and ``step`` continue an
    earlier run's count. The center-of-mass velocity is removed before the first step.
    Syncs with the host.

    Raises ``TypeError`` for a model that is not a ``Potential``, and ``ValueError`` for
    inputs of the wrong shape, dtype, device or value.
    """
    return Simulation(
        model,
        positions,
        velocities,
        masses,
        cell,
        ensemble=NVE,
        timestep=timestep,
        skin=skin,
        capacity_multiplier=capacity_multiplier,
        images=images,
        step=step,
    )


__all__ = ("nve",)
