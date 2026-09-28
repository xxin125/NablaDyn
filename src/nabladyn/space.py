"""Orthorhombic periodic geometry on [..., 3] tensors.

``cell`` [3] holds the edge lengths in nm. Every function is elementwise, so a compiled
program fuses each into one kernel. Derivatives follow the selected periodic image.
"""

import torch
from torch import Tensor


def _displacement_wrapped(left: Tensor, right: Tensor, cell: Tensor) -> Tensor:
    """Return the minimum-image vector ``left - right`` for coordinates already in [0, L)."""
    raw = left - right
    half_cell = 0.5 * cell
    # Shift the coordinate near the far face by L before subtracting: the shift is exact,
    # so a short vector across a face is rounded only once.
    left_shift = torch.where(raw > half_cell, cell, 0.0)
    right_shift = torch.where(raw < -half_cell, cell, 0.0)
    return (left - left_shift) - (right - right_shift)


def wrap(positions: Tensor, cell: Tensor) -> tuple[Tensor, Tensor]:
    """Wrap positions [..., 3] into the cell, each coordinate in [0, L).

    Returns ``(wrapped, crossings)``, with crossings [..., 3] int64 such that ``positions``
    equals ``wrapped + crossings * cell`` up to rounding.
    """
    crossings = torch.floor(positions / cell)
    wrapped = positions - crossings * cell
    # The floor of a rounded quotient can be off by one, and a tiny negative plus L can
    # round to L. The two fixes, in this order, keep every result in [0, L).
    below = wrapped < 0
    wrapped = torch.where(below, wrapped + cell, wrapped)
    crossings = torch.where(below, crossings - 1, crossings)
    above = wrapped >= cell
    wrapped = torch.where(above, wrapped - cell, wrapped)
    crossings = torch.where(above, crossings + 1, crossings)
    return wrapped, crossings.to(torch.int64)


def shift(positions: Tensor, delta: Tensor, cell: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``positions + delta`` [..., 3] wrapped into the cell, with ``wrap``'s crossings."""
    return wrap(positions + delta, cell)


def displacement(left: Tensor, right: Tensor, cell: Tensor) -> Tensor:
    """Return the minimum-image vector ``left - right`` [..., 3] in nm, for positions anywhere."""
    return _displacement_wrapped(wrap(left, cell)[0], wrap(right, cell)[0], cell)


def strained(vectors: Tensor, strain: Tensor | None) -> Tensor:
    """Deform ``vectors`` [..., 3] by (I + strain); ``None`` returns ``vectors`` itself.

    ``strain`` is dimensionless, [3, 3]. Select periodic images before straining.
    """
    if strain is None:
        return vectors
    # Broadcast products, so a compiled program fuses them into one kernel.
    return (
        vectors
        + vectors[..., 0:1] * strain[:, 0]
        + vectors[..., 1:2] * strain[:, 1]
        + vectors[..., 2:3] * strain[:, 2]
    )


__all__ = ("wrap", "shift", "displacement", "strained")
