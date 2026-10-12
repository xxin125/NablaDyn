"""Smooth particle-mesh Ewald with order-5 B-splines: the mesh plan and the mesh energy."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from ... import units

ORDER = 5  # cardinal B-splines of order 5: quartic pieces over 5 mesh points per axis
# M5 at 1, 2, 3 and 4, the values the spline moduli sum.
_SPLINE_AT_INTEGERS = (1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0)


def _smooth_size(count: int) -> int:
    """Return the smallest integer >= count whose only prime factors are 2, 3 and 5."""
    while True:
        rest = count
        for prime in (2, 3, 5):
            while rest % prime == 0:
                rest //= prime
        if rest == 1:
            return count
        count += 1


def plan(
    cutoff: float,
    tolerance: float,
    cell: Sequence[float],
) -> tuple[float, tuple[int, int, int]]:
    """Return alpha in nm^-1 and the mesh (K1, K2, K3) for a cutoff and a cell [3] in nm.

    alpha = sqrt(-ln(2 delta)) / rc and K_d = max(6, ceil(2 alpha L_d / (3 delta^(1/5)))),
    rounded up to the next integer with no prime factor above 5; the tolerance delta is
    about the relative force error.
    """
    alpha = math.sqrt(-math.log(2.0 * tolerance)) / cutoff
    bound = 3.0 * tolerance**0.2
    sizes = (_smooth_size(max(6, math.ceil(2.0 * alpha * length / bound))) for length in cell)
    return alpha, tuple(sizes)


def _spline_moduli(size: int) -> list[float]:
    """Return |sum_j M5(j + 1) exp(2 pi i k j / K)|^2 for the modes k in [0, K), on the host."""
    moduli = []
    for mode in range(size):
        angles = [2.0 * math.pi * mode * j / size for j in range(len(_SPLINE_AT_INTEGERS))]
        real = sum(value * math.cos(angle) for value, angle in zip(_SPLINE_AT_INTEGERS, angles))
        imag = sum(value * math.sin(angle) for value, angle in zip(_SPLINE_AT_INTEGERS, angles))
        moduli.append(real * real + imag * imag)
    # Order 5 vanishes at k = K/2 on an even axis; that mode takes its neighbors' mean.
    for mode in range(size):
        if moduli[mode] < 1e-7:
            moduli[mode] = 0.5 * (moduli[mode - 1] + moduli[(mode + 1) % size])
    return moduli


def influence_table(
    mesh: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return [K1, K2, K3//2 + 1]: each mode's Hermitian weight over its spline moduli.

    The weight is 2 for 0 < k3 < K3/2, whose conjugates the ``rfftn`` half spectrum omits,
    and 1 otherwise. Computed in float64, returned in ``dtype`` on ``device``.
    """
    first, second, third = (
        torch.tensor(_spline_moduli(size), dtype=torch.float64) for size in mesh
    )
    third = third[: mesh[2] // 2 + 1]
    modes = torch.arange(mesh[2] // 2 + 1)
    weights = torch.where((modes > 0) & (2 * modes < mesh[2]), 2.0, 1.0).double()
    table = weights / (first[:, None, None] * second[None, :, None] * third[None, None, :])
    return table.to(dtype=dtype, device=device)


def signed_modes(
    mesh: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the integer modes k of each axis of the ``rfftn`` half spectrum.

    [K1] and [K2] signed, in ``fftfreq`` order, and [K3//2 + 1] in ``rfftfreq`` order.
    """
    return (
        torch.fft.fftfreq(mesh[0], 1.0 / mesh[0], dtype=dtype, device=device),
        torch.fft.fftfreq(mesh[1], 1.0 / mesh[1], dtype=dtype, device=device),
        torch.fft.rfftfreq(mesh[2], 1.0 / mesh[2], dtype=dtype, device=device),
    )


def _spline_weights(fractions: Tensor) -> Tensor:
    """Return M5(w + j), j = 0..4, as [..., 3, 5] for fractions w [..., 3] in [0, 1).

    Uses M_n(x) = [x M_(n-1)(x) + (n - x) M_(n-1)(x - 1)] / (n - 1), whose terms are >= 0
    on the support, so nothing cancels.
    """
    w = fractions.unsqueeze(-1)
    j = torch.arange(ORDER, dtype=w.dtype, device=w.device)
    weights = (j == 0).to(w.dtype).expand(*w.shape[:-1], ORDER)
    for order in range(2, ORDER + 1):
        previous = torch.cat([torch.zeros_like(weights[..., :1]), weights[..., :-1]], dim=-1)
        weights = ((w + j) * weights + (order - w - j) * previous) / (order - 1)
    return weights


def spread(charges: Tensor, positions: Tensor, cell: Tensor, mesh: tuple[int, int, int]) -> Tensor:
    """Return the charge mesh [K1, K2, K3] of charges [N] at positions [N, 3] in [0, L).

    Each charge adds q M5(u - k) at the 5 x 5 x 5 points k = floor(u) - j, j = 0..4, mod K,
    of its scaled position u = K x / L.
    """
    # The mesh sizes stay Python numbers: a tensor of them would copy from the host each call.
    scaled = torch.stack(
        [positions[:, axis] / cell[axis] * mesh[axis] for axis in range(3)],
        dim=-1,
    )
    base = torch.floor(scaled)
    weights = _spline_weights(scaled - base)
    offsets = torch.arange(ORDER, device=positions.device)
    first, second, third = (
        (base[:, axis].long().unsqueeze(-1) - offsets) % mesh[axis] for axis in range(3)
    )
    flat = (
        first[:, :, None, None] * mesh[1] + second[:, None, :, None]
    ) * mesh[2] + third[:, None, None, :]
    values = (
        charges[:, None, None, None]
        * weights[:, 0, :, None, None]
        * weights[:, 1, None, :, None]
        * weights[:, 2, None, None, :]
    )
    # A fit's vmap over frames maps scatter_add as one sum, index_add one frame at a time; out of
    # place, since the in-place form fails under vmap.
    empty = values.new_zeros(mesh[0] * mesh[1] * mesh[2])
    return empty.scatter_add(0, flat.reshape(-1), values.reshape(-1)).view(mesh)


def _reciprocal_squares(
    modes: tuple[Tensor, Tensor, Tensor],
    cell: Tensor,
    strain: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Return |m|^2 [K1, K2, K3//2 + 1] of the modes and the volume, in the strained cell.

    m = (I + strain)^-T (k / L) and V = det(I + strain) L1 L2 L3; the inverse and the
    determinant are closed forms, so nothing syncs.
    """
    scaled = (
        (modes[0] / cell[0])[:, None, None],
        (modes[1] / cell[1])[None, :, None],
        (modes[2] / cell[2])[None, None, :],
    )
    volume = cell.prod()
    if strain is None:
        return sum(component.square() for component in scaled), volume
    rows = torch.eye(3, dtype=strain.dtype, device=strain.device) + strain
    # The rows of the inverse transpose are the cofactor rows over the determinant.
    cofactors = torch.stack(
        [
            torch.linalg.cross(rows[1], rows[2], dim=-1),
            torch.linalg.cross(rows[2], rows[0], dim=-1),
            torch.linalg.cross(rows[0], rows[1], dim=-1),
        ],
    )
    determinant = (rows[0] * cofactors[0]).sum()
    squares = sum(
        (
            cofactors[axis, 0] * scaled[0]
            + cofactors[axis, 1] * scaled[1]
            + cofactors[axis, 2] * scaled[2]
        ).square()
        for axis in range(3)
    )
    return squares / determinant.square(), volume * determinant


def mesh_energy(
    charges: Tensor,
    positions: Tensor,
    cell: Tensor,
    strain: Tensor | None,
    alpha: float,
    mesh: tuple[int, int, int],
    influence: Tensor,
    modes: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    """Return the reciprocal, self and background Ewald energies of charges [N], in kJ/mol.

    E_rec = C / (2 pi V) sum_(m != 0) exp(-pi^2 m^2 / alpha^2) / m^2 B(m) |F(Q)(m)|^2,
    E_self = -C alpha / sqrt(pi) sum q^2 and E_background = -C pi Q^2 / (2 alpha^2 V) for
    the net charge Q. ``influence`` comes from ``influence_table`` and ``modes`` from
    ``signed_modes``. The strain changes m and V, not the fractional coordinates.
    """
    structure = torch.fft.rfftn(spread(charges, positions, cell, mesh))
    power = torch.view_as_real(structure).square().sum(dim=-1)
    squares, volume = _reciprocal_squares(modes, cell, strain)
    # The zero mode is masked before its division.
    nonzero = squares > 0
    safe = torch.where(nonzero, squares, 1.0)
    factor = torch.where(nonzero, torch.exp(-((math.pi / alpha) ** 2) * safe) / safe, 0.0)
    reciprocal = (influence * factor * power).sum() / (2.0 * math.pi * volume)
    self_term = alpha / math.sqrt(math.pi) * charges.square().sum()
    background = math.pi * charges.sum().square() / (2.0 * alpha**2 * volume)
    return units.COULOMB * (reciprocal - self_term - background)


__all__ = ("influence_table", "mesh_energy", "plan", "signed_modes", "spread")
