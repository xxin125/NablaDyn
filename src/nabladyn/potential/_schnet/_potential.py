"""SchNet: continuous-filter convolutions over the pairs of a neighbor list."""

import math
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn.utils import skip_init

from ... import space
from ...neighbors import NeighborPairs
from .._base import Potential
from .._lennard_jones._potential import _check_atom_types, _check_cutoff


def _check_sizes(num_types: int, features: int, interactions: int, gaussians: int) -> None:
    """Raise ``ValueError`` unless the sizes are integers in range and features is even."""
    sizes = (
        ("num_types", num_types, 1),
        ("features", features, 2),
        ("interactions", interactions, 1),
        ("gaussians", gaussians, 2),
    )
    for name, value, least in sizes:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < least:
            raise ValueError(f"{name} must be an integer of at least {least}")
    if features % 2:
        raise ValueError("features must be even: the readout halves it")


def _check_flag_and_seed(trainable: bool, seed: int) -> None:
    """Raise ``ValueError`` unless trainable is a bool and seed an integer."""
    if not isinstance(trainable, bool):
        raise ValueError("trainable must be True or False")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")


def _check_dtype(dtype: torch.dtype) -> None:
    """Raise ``ValueError`` unless dtype is float32 or float64."""
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")


def _shifted_softplus(x: Tensor) -> Tensor:
    """Return ln(1 + e^x) - ln 2, which is 0 at x = 0."""
    return nn.functional.softplus(x) - math.log(2.0)


def _linear(inputs: int, outputs: int, bias: bool = True) -> nn.Linear:
    """Return a float64 CPU linear layer whose values are set later, from the model's seed."""
    return skip_init(nn.Linear, inputs, outputs, bias=bias, dtype=torch.float64)


class _Interaction(nn.Module):
    """One interaction block: a continuous-filter convolution and an atom-wise update."""

    def __init__(self, features: int, gaussians: int):
        super().__init__()
        self.atoms = _linear(features, features, bias=False)
        self.filter1 = _linear(gaussians, features)
        self.filter2 = _linear(features, features)
        self.update1 = _linear(features, features)
        self.update2 = _linear(features, features)

    def forward(
        self,
        x: Tensor,
        basis: Tensor,
        envelope: Tensor,
        first: Tensor,
        second: Tensor,
    ) -> Tensor:
        """Return the atoms' features [N, F] after this block, from the pairs' basis [C, K]."""
        values = self.atoms(x)
        filters = self.filter2(_shifted_softplus(self.filter1(basis))) * envelope[:, None]
        # Each pair is listed once and its filter is symmetric, so it sends to both atoms.
        # Out of place, since in-place index_add_ fails under the fit's vmap over frames.
        messages = (
            x.new_zeros(x.shape)
            .index_add(0, first, filters * values[second])
            .index_add(0, second, filters * values[first])
        )
        return x + self.update2(_shifted_softplus(self.update1(messages)))


class SchNet(Potential):
    """SchNet with fixed Gaussians, a cosine envelope and the shifted softplus.

    Built by ``potential.schnet``; see its docstring.
    """

    def __init__(
        self,
        *,
        atom_types: Tensor,
        num_types: int,
        cutoff: float,
        features: int,
        interactions: int,
        gaussians: int,
        trainable: bool,
        seed: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        _check_sizes(num_types, features, interactions, gaussians)
        _check_cutoff(cutoff)
        _check_flag_and_seed(trainable, seed)
        _check_dtype(dtype)
        _check_atom_types(atom_types, num_types)
        self._cutoff = float(cutoff)
        self.embedding = nn.Parameter(torch.empty(num_types, features, dtype=torch.float64))
        self.blocks = nn.ModuleList(
            _Interaction(features, gaussians) for _ in range(interactions)
        )
        self.readout1 = _linear(features, features // 2)
        self.readout2 = _linear(features // 2, 1)
        # Drawn in float64 from the seed alone, so a seed gives one network in both dtypes.
        generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name == "embedding":
                    values = torch.randn(parameter.shape, generator=generator, dtype=torch.float64)
                elif name.endswith("weight"):
                    bound = math.sqrt(6.0 / sum(parameter.shape))
                    uniform = torch.rand(parameter.shape, generator=generator, dtype=torch.float64)
                    values = (2.0 * uniform - 1.0) * bound
                else:
                    values = torch.zeros_like(parameter)
                parameter.copy_(values)
        self.to(device=atom_types.device, dtype=dtype)
        self.requires_grad_(trainable)
        self.register_buffer("atom_types", atom_types.detach().clone(), persistent=False)
        centres = torch.linspace(0.0, self._cutoff, gaussians, dtype=dtype, device=self.device)
        self.register_buffer("_centres", centres, persistent=False)

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def num_atoms(self) -> int:
        return self.atom_types.numel()

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding.dtype

    @property
    def device(self) -> torch.device:
        return self.embedding.device

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
        # Masked columns take r^2 = 1 before the root, whose gradient at a self-pair's r = 0
        # would be infinite even behind torch.where.
        distance = torch.where(valid, squared, 1.0).sqrt()
        # The envelope and its slope vanish at the cutoff and it is 0 on masked columns, so
        # every filter, and with it E and F, goes smoothly to 0.
        cosine = 0.5 * (torch.cos(distance * (math.pi / self._cutoff)) + 1.0)
        envelope = torch.where(valid, cosine, 0.0)
        width = self._cutoff / (self._centres.numel() - 1)
        basis = torch.exp(-0.5 * ((distance[:, None] - self._centres) / width).square())
        x = self.embedding[self.atom_types]
        for block in self.blocks:
            x = block(x, basis, envelope, first, second)
        return self.readout2(_shifted_softplus(self.readout1(x))).sum()


__all__ = ("SchNet",)
