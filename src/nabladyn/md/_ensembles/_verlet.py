"""The velocity Verlet half steps, shared by every ensemble."""

from torch import Tensor

from ... import space


def half_kick(
    velocities: Tensor,
    forces: Tensor,
    inverse_masses: Tensor,
    half_timestep: Tensor,
) -> Tensor:
    """Return the velocities [N, 3] in nm/ps after half a step of the forces [N, 3] in kJ/mol/nm.

    The result is v + (dt/2) F/m, with ``inverse_masses`` [N, 1] in 1/Da.
    """
    return velocities + half_timestep * forces * inverse_masses


def drift(
    positions: Tensor,
    velocities: Tensor,
    cell: Tensor,
    timestep: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the positions [N, 3] after a step at constant velocity, wrapped into the cell.

    Also returns this step's lattice crossings [N, 3], int64.
    """
    return space.wrap(positions + timestep * velocities, cell)


__all__ = ()
