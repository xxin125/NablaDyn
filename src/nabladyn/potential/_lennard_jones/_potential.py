"""12-6 Lennard-Jones shifted to zero at the cutoff, one sigma and epsilon per type pair."""

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import torch
from torch import Tensor, nn

from ... import space
from ...neighbors import NeighborPairs
from .._base import Potential


def _check_trainable_names(trainable: tuple[str, ...] | Mapping[str, list[int]]) -> None:
    """Raise ``ValueError`` unless ``trainable`` names parameters among "sigma" and "epsilon"."""
    if isinstance(trainable, str) or not set(trainable) <= {"sigma", "epsilon"}:
        raise ValueError('trainable must name parameters among "sigma" and "epsilon"')


def _check_cutoff(cutoff: Real) -> None:
    """Raise ``ValueError`` unless the cutoff is a finite positive real number."""
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, Real)
        or not math.isfinite(cutoff)
        or cutoff <= 0
    ):
        raise ValueError("cutoff must be a finite positive number")


def _check_parameter_vectors(sigma: Tensor, epsilon: Tensor) -> None:
    """Raise ``ValueError`` unless sigma and epsilon are matching float32 or float64 vectors."""
    for name, value in (("sigma", sigma), ("epsilon", epsilon)):
        if (
            not isinstance(value, Tensor)
            or value.ndim != 1
            or value.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError(f"{name} must be a float32 or float64 vector")
    shared = (sigma.shape, sigma.dtype, sigma.device)
    if (epsilon.shape, epsilon.dtype, epsilon.device) != shared:
        raise ValueError("sigma and epsilon must share shape, dtype and device")


def _num_types(sigma: Tensor) -> int:
    """Return T, the number of atom types, from sigma's T(T+1)/2 rows, one per type pair.

    Raises ``ValueError`` unless sigma holds T(T+1)/2 rows for some T >= 1.
    """
    num_types = math.isqrt(2 * sigma.numel())
    if num_types == 0 or num_types * (num_types + 1) // 2 != sigma.numel():
        raise ValueError("sigma and epsilon need T(T+1)/2 values, one per type pair")
    return num_types


def _check_parameter_values(sigma: Tensor, epsilon: Tensor) -> None:
    """Raise ``ValueError`` unless sigma and epsilon are finite and positive."""
    values = torch.cat([sigma, epsilon])
    if not bool((values.isfinite() & (values > 0)).all()):
        raise ValueError("sigma and epsilon must be finite and positive")


def _check_atom_types(atom_types: Tensor, num_types: int) -> None:
    """Raise ``ValueError`` unless atom_types is a nonempty int64 vector in [0, T)."""
    if (
        not isinstance(atom_types, Tensor)
        or atom_types.ndim != 1
        or atom_types.dtype != torch.int64
        or atom_types.numel() == 0
    ):
        raise ValueError("atom_types must be a nonempty int64 vector")
    if not bool(((atom_types >= 0) & (atom_types < num_types)).all()):
        raise ValueError(f"atom_types must lie in [0, {num_types})")


def _check_trainable_rows(name: str, rows: Sequence[int], count: int) -> None:
    """Raise ``ValueError`` unless ``rows`` is a nonempty sequence of integer rows in [0, count)."""
    if (
        isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or not rows
        or any(
            isinstance(row, bool) or not isinstance(row, Integral) or not 0 <= row < count
            for row in rows
        )
    ):
        raise ValueError(f"trainable[{name!r}] must list rows in [0, {count})")


def _row_mask(
    trainable: tuple[str, ...] | Mapping[str, list[int]],
    name: str,
    count: int,
    device: torch.device,
) -> Tensor | None:
    """Return a bool mask [count], True on the trained rows of parameter ``name``, or None.

    None means the whole parameter trains or none of it: ``trainable`` is a tuple, lacks
    ``name``, or lists every row. Raises ``ValueError`` as ``_check_trainable_rows`` does.
    """
    if not isinstance(trainable, Mapping) or name not in trainable:
        return None
    rows = trainable[name]
    _check_trainable_rows(name, rows, count)
    if len(set(rows)) == count:
        return None
    mask = torch.zeros(count, dtype=torch.bool, device=device)
    mask[[int(row) for row in rows]] = True
    return mask


def _pair_energy(squared: Tensor | float, sigma: Tensor, epsilon: Tensor) -> Tensor:
    """Return the unshifted 12-6 energy 4 eps [(sigma/r)^12 - (sigma/r)^6] in kJ/mol.

    ``squared`` holds r^2 in nm^2, a tensor or a number, and broadcasts against ``sigma``
    (nm) and ``epsilon`` (kJ/mol).
    """
    power6 = (sigma.square() / squared).pow(3)
    return 4.0 * epsilon * (power6.square() - power6)


def _pair_rows(
    atom_types: Tensor,
    first: Tensor,
    second: Tensor,
    num_types: int,
    *tables: Tensor,
) -> tuple[Tensor, ...]:
    """Return each table's row for each pair of atoms ``first`` and ``second``.

    Each table holds T(T+1)/2 rows in ``pair_index`` order, T = ``num_types``. Each result
    has the shape of ``first``, or is the single row when T is 1.
    """
    if num_types == 1:
        # A broadcast row's gradient is a sum reduction. Read by index, every pair would add
        # atomically into one word, which is slow and loses digits in float32.
        return tuple(table[0] for table in tables)
    rows = pair_index(atom_types[first], atom_types[second], num_types)
    return tuple(table[rows] for table in tables)


def pair_index(first: Tensor, second: Tensor, num_types: int) -> Tensor:
    """Return the parameter row of each unordered type pair.

    ``first`` and ``second`` are integer tensors of one shape with types in [0, T),
    T = ``num_types``; the rows share that shape and dtype. Rows run over the row-major upper
    triangle (0, 0), (0, 1), ..., (0, T-1), (1, 1), ...: for a <= b, row a (2T - a - 1) / 2 + b.
    """
    low, high = torch.minimum(first, second), torch.maximum(first, second)
    return low * (2 * num_types - low - 1) // 2 + high


class LennardJones(Potential):
    """12-6 Lennard-Jones shifted to zero at the cutoff, one sigma and epsilon per type pair.

    Built by ``potential.lennard_jones``; see its docstring.
    """

    def __init__(
        self,
        *,
        atom_types: Tensor,
        sigma: Tensor,
        epsilon: Tensor,
        cutoff: Real,
        trainable: tuple[str, ...] | Mapping[str, list[int]] = (),
    ):
        super().__init__()
        _check_trainable_names(trainable)
        _check_cutoff(cutoff)
        _check_parameter_vectors(sigma, epsilon)
        num_types = _num_types(sigma)
        _check_parameter_values(sigma, epsilon)
        _check_atom_types(atom_types, num_types)
        self._cutoff = float(cutoff)
        self._num_types = num_types
        # Only trained parameters take gradients; energy, forces and virial need none.
        self.sigma = nn.Parameter(sigma.detach().clone(), requires_grad="sigma" in trainable)
        self.epsilon = nn.Parameter(
            epsilon.detach().clone(),
            requires_grad="epsilon" in trainable,
        )
        self.register_buffer("atom_types", atom_types.detach().to(sigma.device, copy=True))
        for name in ("sigma", "epsilon"):
            mask = _row_mask(trainable, name, sigma.numel(), sigma.device)
            self.register_buffer(f"_{name}_rows", mask, persistent=False)

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def num_atoms(self) -> int:
        return self.atom_types.numel()

    @property
    def dtype(self) -> torch.dtype:
        return self.sigma.dtype

    @property
    def device(self) -> torch.device:
        return self.sigma.device

    def _trained_masks(self) -> dict[str, Tensor]:
        masks = {"sigma": self._sigma_rows, "epsilon": self._epsilon_rows}
        return {name: mask for name, mask in masks.items() if mask is not None}

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
        # Padding columns are self-pairs; list pairs beyond the cutoff lie in the skin.
        valid = (first != second) & (squared < self._cutoff**2)
        # Masked columns take r^2 = 1, so neither the energy nor its gradient meets 1/0.
        safe = torch.where(valid, squared, 1.0)
        cutoff_energy = _pair_energy(self._cutoff**2, self.sigma, self.epsilon)
        sigma, epsilon, shift = _pair_rows(
            self.atom_types,
            first,
            second,
            self._num_types,
            self.sigma,
            self.epsilon,
            cutoff_energy,
        )
        energy = _pair_energy(safe, sigma, epsilon) - shift
        return torch.where(valid, energy, 0.0).sum()


__all__ = ("LennardJones", "pair_index")
