"""The neighbor list: pairs of atoms closer than cutoff + skin, kept on the GPU."""

import math
from dataclasses import dataclass, fields
from numbers import Real

import torch
from torch import Tensor

from .. import space
from ._extension import ops
from ._pairs import NeighborPairs

_INT32_MAX = 2**31 - 1
_INVALID_CELL, _NON_FINITE, _OVERFLOW = 1, 2, 8


@dataclass(slots=True)
class _Buffers:
    """The device tensors a build writes, in the operators' argument order."""

    dims: Tensor
    widths: Tensor
    atom_bin: Tensor
    bin_counts: Tensor
    bin_offsets: Tensor
    unordered_ids: Tensor
    sorted_ids: Tensor
    sorted_positions: Tensor
    scan_temp: Tensor
    pair_counts: Tensor
    pair_offsets: Tensor
    pair_total: Tensor
    status: Tensor
    pairs: Tensor
    bits: Tensor

    def args(self) -> tuple[Tensor, ...]:
        """Return the tensors themselves, not copies, in argument order."""
        return tuple(getattr(self, field.name) for field in fields(self))


@dataclass(slots=True)
class _Reuse:
    """The device tensors of the reuse check.

    The frame of the last rebuild, the first grid's bins per axis, the threshold and scales,
    and the rebuild flag and count.
    """

    reference_positions: Tensor
    reference_cell: Tensor
    limits: Tensor
    parameters: Tensor
    rebuild: Tensor
    rebuild_count: Tensor


def _check_settings(cutoff: Real, skin: Real, capacity_multiplier: Real) -> None:
    """Raise ``ValueError`` unless the settings are finite numbers in their ranges."""
    settings = (
        ("cutoff", cutoff),
        ("skin", skin),
        ("capacity_multiplier", capacity_multiplier),
    )
    for name, value in settings:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if cutoff <= 0 or skin < 0 or capacity_multiplier < 1:
        raise ValueError(
            "cutoff must be positive, skin nonnegative and capacity_multiplier at least 1",
        )


def _check_inner_cutoffs(inner_cutoffs: tuple[Real, ...], cutoff: Real) -> None:
    """Raise ``ValueError`` unless the inner cutoffs are distinct numbers between 0 and cutoff."""
    if not isinstance(inner_cutoffs, (tuple, list)):
        raise ValueError("inner_cutoffs must be a tuple of cutoffs in nm")
    for value in inner_cutoffs:
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value < cutoff:
            raise ValueError("every inner cutoff must be a number between 0 and the cutoff")
    if len(set(inner_cutoffs)) != len(inner_cutoffs):
        raise ValueError("the inner cutoffs must be distinct")


def _check_frame(positions: Tensor, cell: Tensor) -> None:
    """Raise ``TypeError`` or ``ValueError`` unless positions [N, 3] and cell [3] fit a list."""
    for name, value in (("positions", positions), ("cell", cell)):
        if not isinstance(value, Tensor) or value.layout is not torch.strided:
            raise TypeError(f"{name} must be a dense tensor")
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not 1 <= positions.shape[0] <= _INT32_MAX
    ):
        raise ValueError("positions must have shape [N, 3] with 1 <= N < 2**31")
    if positions.dtype not in (torch.float32, torch.float64) or not positions.is_cuda:
        raise TypeError("positions must be a CUDA float32 or float64 tensor")
    if cell.shape != (3,) or cell.dtype != positions.dtype or cell.device != positions.device:
        raise ValueError("cell must be a [3] tensor with the positions' dtype and device")


def _check_cell_lengths(lengths: list[float], cutoff: float) -> None:
    """Raise ``ValueError`` unless every cell length in nm is finite and above twice the cutoff."""
    if not all(math.isfinite(length) and length > 2 * cutoff for length in lengths):
        raise ValueError("every cell length must be finite and above twice the cutoff")


def _check_finite(positions: Tensor) -> None:
    """Raise ``ValueError`` unless every coordinate is finite."""
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("positions must be finite")


def _grid(lengths: list[float], radius: float, dtype: torch.dtype) -> list[int]:
    """Return the bins per axis, each wider than the list radius.

    A margin of 4 eps L absorbs the rounding of the binning, so every pair closer than the
    radius lies in neighboring bins.
    """
    eps = torch.finfo(dtype).eps
    return [max(1, math.floor(length / (radius + 4 * eps * length))) for length in lengths]


def _check_bin_count(num_bins: int) -> None:
    """Raise ``ValueError`` unless the bin count plus one fits int32."""
    if num_bins >= _INT32_MAX:
        raise ValueError("the grid must have fewer than 2**31 - 1 bins")


def _all_pairs_bits(num_atoms: int, dims: list[int], device: torch.device) -> Tensor:
    """Return the all-pairs bit rows: N rows of ceil(N / 32) int32 words, flattened.

    Only a list whose first grid has at most three bins per axis gets them, and it keeps this
    path because a rebuild never uses more bins than the first grid. Any other list gets an
    empty tensor and searches the grid.
    """
    if max(dims) > 3:
        return torch.empty(0, dtype=torch.int32, device=device)
    return torch.zeros(num_atoms * ((num_atoms + 31) // 32), dtype=torch.int32, device=device)


def _check_capacity(capacity: int, count: int) -> None:
    """Raise ``ValueError`` unless the pair capacity fits int32."""
    if capacity > _INT32_MAX:
        raise ValueError(
            f"the pair capacity {capacity}, capacity_multiplier times {count} pairs, "
            "exceeds the int32 range",
        )


def _check_allocated_like(positions: Tensor, allocated: Tensor) -> None:
    """Raise ``ValueError`` unless positions have the allocated shape, dtype and device."""
    if (
        positions.shape != allocated.shape
        or positions.dtype != allocated.dtype
        or positions.device != allocated.device
    ):
        raise ValueError("positions must match the allocated shape, dtype and device")


class NeighborList:
    """Pairs of atoms closer than the list radius, cutoff + skin, in nm.

    ``allocate`` sizes the list to ``capacity_multiplier`` times a first frame's pair count
    and builds it. ``update`` reuses the pairs until an atom has moved far enough that a pair
    within the cutoff could be missing, then rebuilds. ``tensors()`` returns the pairs. Cells
    are orthorhombic and periodic; positions may lie outside the cell.

    For each of ``inner_cutoffs``, smaller cutoffs in nm, the list also keeps an inner list:
    the pairs within that cutoff + skin, with the same skin and capacity rule, allocated,
    updated and checked with this one. ``tensors()`` returns them in ``NeighborPairs.inner``;
    a sum of models gives each model the list at its own cutoff.

    Raises ``ValueError`` unless ``cutoff`` is positive, ``skin`` nonnegative and
    ``capacity_multiplier`` at least 1, all finite, and the inner cutoffs distinct numbers
    between 0 and ``cutoff``.
    """

    def __init__(
        self,
        cutoff: Real,
        *,
        skin: Real = 0.0,
        capacity_multiplier: Real = 1.25,
        inner_cutoffs: tuple[Real, ...] = (),
    ):
        _check_settings(cutoff, skin, capacity_multiplier)
        _check_inner_cutoffs(inner_cutoffs, cutoff)
        self._cutoff = float(cutoff)
        self._skin = float(skin)
        self._capacity_multiplier = float(capacity_multiplier)
        self._inner = tuple(
            NeighborList(value, skin=skin, capacity_multiplier=capacity_multiplier)
            for value in sorted(inner_cutoffs)
        )
        self._buffers: _Buffers | None = None
        self._reuse: _Reuse | None = None
        self._failure: str | None = None

    @property
    def cutoff(self) -> float:
        """The cutoff in nm."""
        return self._cutoff

    @property
    def radius(self) -> float:
        """The list radius, cutoff + skin, in nm."""
        return self._cutoff + self._skin

    def allocate(self, positions: Tensor, cell: Tensor) -> None:
        """Count the pairs of positions [N, 3] in the cell [3], size the list and build it.

        Positions and cell are CUDA float32 or float64 tensors in nm. The inner lists are
        allocated after the list; one that cannot be leaves the list failed until ``allocate``
        succeeds. It also recovers a list that failed an update. Syncs with the host.

        Raises ``TypeError`` or ``ValueError`` for invalid positions or cell, and
        ``ValueError`` when the grid or the pairs do not fit int32.
        """
        _check_frame(positions, cell)
        lengths = [float(length) for length in cell.tolist()]
        _check_cell_lengths(lengths, self._cutoff)
        _check_finite(positions)
        dims = _grid(lengths, self.radius, positions.dtype)
        num_atoms, num_bins = positions.shape[0], math.prod(dims)
        _check_bin_count(num_bins)
        device, dtype = positions.device, positions.dtype
        integers = {"dtype": torch.int32, "device": device}
        scan_bytes = max(
            ops().scan_temp_bytes(positions, num_bins + 1),
            ops().scan_temp_bytes(positions, num_atoms),
        )
        buffers = _Buffers(
            dims=torch.tensor(dims, **integers),
            widths=torch.tensor(
                [length / count for length, count in zip(lengths, dims)],
                dtype=dtype,
                device=device,
            ),
            atom_bin=torch.empty(num_atoms, **integers),
            bin_counts=torch.zeros(num_bins + 1, **integers),
            bin_offsets=torch.empty(num_bins + 1, **integers),
            unordered_ids=torch.empty(num_atoms, **integers),
            sorted_ids=torch.empty(num_atoms, **integers),
            sorted_positions=torch.empty((num_atoms, 3), dtype=dtype, device=device),
            scan_temp=torch.empty(scan_bytes, dtype=torch.uint8, device=device),
            pair_counts=torch.empty(num_atoms, **integers),
            pair_offsets=torch.empty(num_atoms, **integers),
            pair_total=torch.zeros(1, **integers),
            status=torch.zeros(2, **integers),
            pairs=torch.empty((2, 1), **integers),  # sized after the count
            bits=_all_pairs_bits(num_atoms, dims, device),
        )
        wrapped, _ = space.wrap(positions.detach(), cell.detach())
        cell = cell.detach().contiguous()
        ops().count_first_frame(wrapped, cell, self.radius, *buffers.args())
        count = int(buffers.pair_counts.sum(dtype=torch.int64))
        capacity = max(8, math.ceil(self._capacity_multiplier * count))
        _check_capacity(capacity, count)
        buffers.pairs = torch.empty((2, capacity), **integers)
        # The capacity differs per list; a dynamic size lets one compiled program serve them all.
        torch._dynamo.mark_dynamic(buffers.pairs, 1)
        ops().fill_first_frame(wrapped, cell, self.radius, *buffers.args())
        self._reuse = _Reuse(
            reference_positions=wrapped.clone(),
            reference_cell=cell.clone(),
            limits=torch.tensor(dims, **integers),
            parameters=torch.zeros(4, dtype=dtype, device=device),
            rebuild=torch.zeros(1, **integers),
            rebuild_count=torch.ones(1, **integers),
        )
        self._buffers = buffers
        # Failed until every inner list is built: it must never return its new pairs with an
        # inner list's old ones.
        self._failure = "an inner list was not allocated"
        for inner in self._inner:
            inner.allocate(positions, cell)
        self._failure = None

    def update(self, positions: Tensor, cell: Tensor) -> None:
        r"""Reuse the pairs for positions [N, 3] in the cell [3], or rebuild them.

        The pairs are reused while no atom has moved more than $`(R s_{\min} - r_c) / 2`$ from
        its position at the last rebuild, scaled with the cell; R is the list radius,
        $`r_c`$ the cutoff and $`s_{\min}`$ the smallest ratio of a cell length to its length
        then. In a fixed cell that is half the skin, so zero skin rebuilds every time. Each
        inner list follows the same rule. Syncs with the host.

        Raises ``TypeError`` or ``ValueError`` for positions or a cell of the wrong kind or
        unlike the allocated ones. Raises ``ValueError``, and fails the list until
        ``allocate``, for a cell length not finite or not above twice the cutoff, a coordinate
        that is not finite, or more pairs than the capacity of the list or an inner list.
        Raises ``RuntimeError`` before ``allocate`` or after a failure.
        """
        buffers = self._live_buffers()
        _check_frame(positions, cell)
        _check_allocated_like(positions, buffers.sorted_positions)
        wrapped, _ = space.wrap(positions.detach(), cell.detach())
        self._update_wrapped(wrapped, cell.detach().contiguous())
        self._check_status()

    def _update_wrapped(self, positions: Tensor, cell: Tensor) -> None:
        """Reuse or rebuild the pairs, and the inner lists', for contiguous positions in [0, L).

        Runs no checks, so a CUDA Graph can capture it; read the status later with
        ``_check_status``. No host sync.
        """
        reuse = self._reuse
        ops().update(
            positions,
            cell,
            self._cutoff,
            self.radius,
            reuse.reference_positions,
            reuse.reference_cell,
            reuse.limits,
            reuse.parameters,
            reuse.rebuild,
            reuse.rebuild_count,
            *self._buffers.args(),
        )
        for inner in self._inner:
            inner._update_wrapped(positions, cell)

    def _rebuild_wrapped(self, positions: Tensor, cell: Tensor) -> None:
        """Rebuild the pairs for contiguous positions already in [0, L), whatever the reuse rule.

        An infinite reference cell forces the rebuild, which stores this frame as the reference;
        the inner lists rebuild too. Runs no checks; read the status later with
        ``_check_status``. No host sync.
        """
        for listed in (self, *self._inner):
            listed._reuse.reference_cell.fill_(math.inf)
        self._update_wrapped(positions, cell)

    def _check_status(self) -> None:
        """Raise ``ValueError`` and fail the list if an update of it or an inner list failed.

        Syncs with the host.
        """
        flags, needed = self._live_buffers().status.tolist()
        if flags:
            causes = []
            if flags & _INVALID_CELL:
                causes.append("every cell length must be finite and above twice the cutoff")
            if flags & _NON_FINITE:
                causes.append("positions must be finite")
            if flags & _OVERFLOW:
                causes.append(
                    f"the pair capacity {self.pair_capacity} is too small: the frame has "
                    f"{needed} pairs within {self.radius} nm; raise capacity_multiplier",
                )
            self._failure = "; ".join(causes)
            raise ValueError(self._failure)
        try:
            for inner in self._inner:
                inner._check_status()
        except ValueError as error:
            self._failure = str(error)
            raise

    def tensors(self) -> NeighborPairs:
        """Return the pairs of the last build, the ``NeighborPairs`` that models take.

        The inner lists' pairs are in its ``inner``, ascending. Raises ``RuntimeError`` before
        ``allocate`` or after a failure.
        """
        buffers = self._live_buffers()
        inner = tuple(listed.tensors() for listed in self._inner)
        return NeighborPairs(buffers.pairs, buffers.sorted_ids.numel(), self._cutoff, inner)

    @property
    def pair_capacity(self) -> int:
        """C, the columns of the list's own pair index; 0 before ``allocate``."""
        return self._buffers.pairs.shape[1] if self._buffers is not None else 0

    @property
    def num_pairs(self) -> int:
        """The number of the list's own pairs at the last build.

        Syncs with the host. Raises ``RuntimeError`` before ``allocate`` or after a failure.
        """
        return int(self._live_buffers().pair_total.item())

    @property
    def _rebuild_count(self) -> int:
        """The builds since allocate, counting allocate's own.

        Syncs with the host.
        """
        return int(self._reuse.rebuild_count.item())

    def _live_buffers(self) -> _Buffers:
        """Return the buffers.

        Raises ``RuntimeError`` before ``allocate`` or after a failure.
        """
        if self._buffers is None:
            raise RuntimeError("call allocate() first")
        if self._failure is not None:
            raise RuntimeError(
                f"the neighbor list failed ({self._failure}); call allocate() again",
            )
        return self._buffers


__all__ = ("NeighborList",)
