"""A pair potential tabulated per unordered type pair, 0 at the cutoff."""

import math
from collections.abc import Mapping
from numbers import Real

import torch
from torch import Tensor, nn

from ... import space
from ...neighbors import NeighborPairs
from .._base import Potential
from .._lennard_jones._potential import _check_atom_types, _check_cutoff, _row_mask, pair_index


def _check_trainable_names(trainable: tuple[str, ...] | Mapping[str, list[int]]) -> None:
    """Raise ``ValueError`` unless ``trainable`` names only "table"."""
    if isinstance(trainable, str) or not set(trainable) <= {"table"}:
        raise ValueError('trainable may name only "table"')


def _check_table(table: Tensor) -> None:
    """Raise ``ValueError`` unless the table is a finite float [P, K >= 4], 0 in its last column."""
    if (
        not isinstance(table, Tensor)
        or table.ndim != 2
        or table.shape[1] < 4
        or table.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("table must be a float32 or float64 matrix of at least 4 columns")
    if not bool(table.isfinite().all()):
        raise ValueError("table must be finite")
    if bool((table[:, -1] != 0).any()):
        raise ValueError("table must hold 0 in its last column, at the cutoff")


def _num_types(rows: int) -> int:
    """Return T from the table's T(T+1)/2 rows, one per unordered type pair.

    Raises ``ValueError`` unless there are T(T+1)/2 rows for some T >= 1.
    """
    num_types = math.isqrt(2 * rows)
    if num_types == 0 or num_types * (num_types + 1) // 2 != rows:
        raise ValueError("table and start need T(T+1)/2 rows, one per type pair")
    return num_types


def _check_start(start: Tensor, table: Tensor, cutoff: Real) -> None:
    """Raise ``ValueError`` unless start holds one value in [0, cutoff) per table row."""
    if (
        not isinstance(start, Tensor)
        or (start.shape, start.dtype, start.device) != (table.shape[:1], table.dtype, table.device)
    ):
        raise ValueError("start must hold one value per table row, in the table's dtype and device")
    if not bool(((start >= 0) & (start < cutoff)).all()):
        raise ValueError(f"start must lie in [0, {cutoff})")


def _hermite_energy(
    r: Tensor,
    rows: Tensor,
    table: Tensor,
    start: Tensor,
    spacing: Tensor,
) -> Tensor:
    """Return u(r) in kJ/mol at pair distances ``r`` [C] in nm, each pair from its table row.

    ``rows`` [C], int64, picks each pair's row of ``table`` [P, K]; row p's knot k lies at
    start_p + k spacing_p, and its last knot reads as 0. Each segment is the Catmull-Rom
    cubic of its four nearest knots; virtual knots u_(-1) = 3u_0 - 3u_1 + u_2 and
    u_K = 3u_(K-1) - 3u_(K-2) + u_(K-3) make the end segments exact for quadratics too.
    u is C1; below the first knot a row follows its tangent there.
    """
    num_knots = table.shape[1]
    values = torch.cat([table[:, :-1], torch.zeros_like(table[:, :1])], dim=1)
    before = 3 * values[:, :1] - 3 * values[:, 1:2] + values[:, 2:3]
    after = 3 * values[:, -1:] - 3 * values[:, -2:-1] + values[:, -3:-2]
    padded = torch.cat([before, values, after], dim=1).reshape(-1)
    x = (r - start[rows]) / spacing[rows]
    segment = x.floor().clamp(0, num_knots - 2)
    t = x - segment
    # The segment's knots k-1 ... k+2 sit at k ... k+3 of its padded row.
    first = rows * (num_knots + 2) + segment.long()
    u0, u1, u2, u3 = (padded[first + k] for k in range(4))
    s = t.clamp(min=0)
    cubic = u1 + 0.5 * s * (
        (u2 - u0) + s * ((2 * u0 - 5 * u1 + 4 * u2 - u3) + s * (3 * (u1 - u2) + u3 - u0))
    )
    # t < 0 only below the first knot. At t = 0 the cubic carries the slope, so the tangent
    # term must pass no gradient there.
    return cubic + torch.where(t < 0, 0.5 * (u2 - u0) * t, 0.0)


class Tabulated(Potential):
    """A pair potential tabulated per unordered type pair, 0 at the cutoff.

    Built by ``potential.tabulated``; see its docstring.
    """

    def __init__(
        self,
        *,
        atom_types: Tensor,
        table: Tensor,
        start: Tensor,
        cutoff: Real,
        trainable: tuple[str, ...] | Mapping[str, list[int]] = (),
    ):
        super().__init__()
        _check_trainable_names(trainable)
        _check_cutoff(cutoff)
        _check_table(table)
        num_types = _num_types(table.shape[0])
        _check_start(start, table, cutoff)
        _check_atom_types(atom_types, num_types)
        self._cutoff = float(cutoff)
        self._num_types = num_types
        self.table = nn.Parameter(table.detach().clone(), requires_grad="table" in trainable)
        self.register_buffer("start", start.detach().clone())
        self.register_buffer("spacing", (cutoff - start.detach()) / (table.shape[1] - 1))
        self.register_buffer("atom_types", atom_types.detach().to(table.device, copy=True))
        mask = _row_mask(trainable, "table", table.shape[0], table.device)
        # [P, 1]: a row's mask spans its knots.
        rows = None if mask is None else mask[:, None]
        self.register_buffer("_table_rows", rows, persistent=False)

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def num_atoms(self) -> int:
        return self.atom_types.numel()

    @property
    def dtype(self) -> torch.dtype:
        return self.table.dtype

    @property
    def device(self) -> torch.device:
        return self.table.device

    def _trained_masks(self) -> dict[str, Tensor]:
        return {} if self._table_rows is None else {"table": self._table_rows}

    def _energy_wrapped(
        self,
        positions: Tensor,
        cell: Tensor,
        pairs: NeighborPairs,
        strain: Tensor | None = None,
    ) -> Tensor:
        first, second = pairs.index[0], pairs.index[1]
        vectors = space._displacement_wrapped(positions[first], positions[second], cell)
        squared = space.strained(vectors, strain).square().sum(dim=-1)
        # Padding self-pairs and list pairs beyond the cutoff count nothing; they take
        # r = cutoff, a finite point of every row, before the root.
        valid = (first != second) & (squared < self._cutoff**2)
        r = torch.where(valid, squared, self._cutoff**2).sqrt()
        rows = pair_index(self.atom_types[first], self.atom_types[second], self._num_types)
        energy = _hermite_energy(r, rows, self.table, self.start, self.spacing)
        return torch.where(valid, energy, 0.0).sum()


__all__ = ("Tabulated",)
