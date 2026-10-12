"""Initial velocities."""

import math
from numbers import Integral, Real

import torch
from torch import Tensor

from ..units import BOLTZMANN


def _check_masses(masses: Tensor) -> None:
    """Raise ``ValueError`` unless masses is a finite positive float vector of at least 2 atoms."""
    if (
        not isinstance(masses, Tensor)
        or masses.ndim != 1
        or masses.numel() < 2
        or masses.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("masses must be a float32 or float64 vector of at least 2 atoms")
    if not bool((torch.isfinite(masses) & (masses > 0)).all()):
        raise ValueError("masses must be finite and positive")


def _check_temperature(temperature: Real) -> None:
    """Raise ``ValueError`` unless the temperature is a finite nonnegative number."""
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be a finite nonnegative number")


def _check_seed(seed: Integral) -> None:
    """Raise ``ValueError`` unless the seed is an integer."""
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")


def maxwell_boltzmann(masses: Tensor, temperature: Real, seed: Integral) -> Tensor:
    """Draw velocities from the Maxwell-Boltzmann distribution at ``temperature`` in K.

    Returns velocities [N, 3] in nm/ps for ``masses`` [N] in Da, in their dtype and on their
    device, with zero total momentum and scaled to exactly ``temperature`` over $`3N - 3`$
    degrees of freedom. ``seed`` fixes the draw on any device. Syncs with the host.

    Raises ``ValueError`` unless ``masses`` is a finite positive float32 or float64 vector
    of at least 2 atoms, ``temperature`` a finite nonnegative number and ``seed`` an
    integer.
    """
    _check_masses(masses)
    _check_temperature(temperature)
    _check_seed(seed)
    generator = torch.Generator().manual_seed(int(seed))
    host = masses.detach().to("cpu", torch.float64)
    velocities = torch.randn((host.numel(), 3), generator=generator, dtype=torch.float64)
    velocities *= torch.sqrt(BOLTZMANN * float(temperature) / host)[:, None]
    velocities -= (host[:, None] * velocities).sum(dim=0) / host.sum()
    kinetic = 0.5 * float((host[:, None] * velocities.square()).sum())
    target = 0.5 * (3 * host.numel() - 3) * BOLTZMANN * float(temperature)
    if kinetic > 0:
        velocities *= math.sqrt(target / kinetic)
    return velocities.to(masses.device, masses.dtype)


__all__ = ("maxwell_boltzmann",)
