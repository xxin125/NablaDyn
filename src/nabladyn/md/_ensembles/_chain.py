"""The Nose-Hoover chain of three thermostats and its half step.

Integrated as in Martyna, Tuckerman, Tobias and Klein (1996), one chain step per half step
of the dynamics.
"""

import torch
from torch import Tensor


def _check_chain(chain, device: torch.device, name: str) -> None:
    """Raise ``ValueError`` unless a chain state holds two [3] float64 tensors on the device."""
    for field, value in zip(("positions", "momenta"), chain):
        if (
            not isinstance(value, Tensor)
            or value.shape != (3,)
            or value.dtype != torch.float64
            or value.device != device
        ):
            raise ValueError(f"{name}.{field} must be a [3] float64 tensor on {device}")


def chain_masses(degrees_of_freedom: int, kT: Tensor, timescale: Tensor) -> Tensor:
    """Return the chain masses Q [3] in kJ ps^2/mol: g kT tau^2, then kT tau^2 twice."""
    base = kT * timescale.square()
    return torch.stack((degrees_of_freedom * base, base, base))


def half_step(
    kinetic: Tensor,
    positions: Tensor,
    momenta: Tensor,
    masses: Tensor,
    degrees_of_freedom: int,
    kT: Tensor,
    timestep: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Advance the chain by half a step against the kinetic energy K it thermostats.

    Returns the velocity scale factor [] and the chain's new positions and momenta [3].
    From the last thermostat down, each momentum gets a kick (dt/4) G_k, damped before and
    after by exp(-(dt/8) p_(k+1) / Q_(k+1)); the velocities then scale by
    exp(-(dt/2) p_1 / Q_1) and the positions advance by (dt/2) p / Q; then the kicks
    repeat from the first thermostat up, with the scaled K.
    """
    p, mass = list(momenta.unbind()), masses.unbind()
    last = len(p) - 1
    quarter, eighth = 0.25 * timestep, 0.125 * timestep

    def driving(index: int, kinetic: Tensor) -> Tensor:
        """Return G_k: 2K - g kT for the first thermostat, else p^2 / Q - kT of the one below."""
        if index == 0:
            return 2.0 * kinetic - degrees_of_freedom * kT
        return p[index - 1].square() / mass[index - 1] - kT

    # A damping factor uses the thermostat above after its first kick; that momentum stays
    # fixed until the second kick below it, so one factor serves both kicks.
    damping: list[Tensor | None] = [None] * last
    for index in range(last, -1, -1):
        if index == last:
            p[index] = p[index] + quarter * driving(index, kinetic)
        else:
            kick = quarter * driving(index, kinetic)
            p[index] = damping[index] * (damping[index] * p[index] + kick)
        if index > 0:
            damping[index - 1] = torch.exp(-eighth * p[index] / mass[index])
    factor = torch.exp(-0.5 * timestep * p[0] / mass[0])
    kinetic = kinetic * factor.square()
    positions = positions + 0.5 * timestep * torch.stack(p) / masses
    for index in range(last + 1):
        if index == last:
            p[index] = p[index] + quarter * driving(index, kinetic)
        else:
            kick = quarter * driving(index, kinetic)
            p[index] = damping[index] * (damping[index] * p[index] + kick)
    return factor, positions, torch.stack(p)


def own_chain(chain, device: torch.device, name: str) -> tuple[Tensor, Tensor]:
    """Return copies of a chain state's positions and momenta, [3] float64; zeros for None.

    Raises ``ValueError`` as ``_check_chain`` does.
    """
    if chain is None:
        return (
            torch.zeros(3, dtype=torch.float64, device=device),
            torch.zeros(3, dtype=torch.float64, device=device),
        )
    _check_chain(chain, device, name)
    return chain[0].detach().clone(), chain[1].detach().clone()


def heat_bath_energy(
    positions: Tensor,
    momenta: Tensor,
    masses: Tensor,
    degrees_of_freedom: int,
    kT: Tensor,
) -> Tensor:
    """Return the chain's energy in kJ/mol: sum p_k^2 / (2 Q_k) + g kT q_1 + kT (q_2 + q_3)."""
    return (
        (momenta.square() / (2.0 * masses)).sum()
        + degrees_of_freedom * kT * positions[0]
        + kT * positions[1:].sum()
    )


__all__ = ()
