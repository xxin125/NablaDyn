"""A set of labeled frames."""

import numpy as np
import torch
from torch import Tensor

from ._loss import CHANNELS


def _host_copy(value) -> Tensor:
    """Return a detached CPU copy of a NumPy array or a tensor."""
    tensor = value if isinstance(value, Tensor) else torch.from_numpy(np.asarray(value))
    return tensor.detach().to("cpu", copy=True)


def _check_positions(positions: Tensor) -> None:
    """Raise ``ValueError`` unless the positions are [F, N, 3] with at least one frame."""
    if positions.ndim != 3 or positions.shape[0] == 0 or positions.shape[2] != 3:
        raise ValueError("positions must be [F, N, 3] with at least one frame")


def _check_fields(fields: dict[str, Tensor]) -> None:
    """Raise ``ValueError`` for a wrong shape or dtype, a non-finite value or a nonpositive cell."""
    positions = fields["positions"]
    frames, atoms = positions.shape[:2]
    shapes = {
        "positions": (frames, atoms, 3),
        "cells": (frames, 3),
        "energy": (frames,),
        "forces": (frames, atoms, 3),
        "virial": (frames, 3, 3),
    }
    for name, value in fields.items():
        if tuple(value.shape) != shapes[name]:
            raise ValueError(f"{name} must be {list(shapes[name])}, not {list(value.shape)}")
        if value.dtype != positions.dtype or value.dtype not in (torch.float32, torch.float64):
            raise ValueError("every field must share one dtype, float32 or float64")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    if not bool((fields["cells"] > 0).all()):
        raise ValueError("cells must be positive")


class Dataset:
    """Labeled frames of one topology, copied to the CPU.

    ``positions`` [F, N, 3] and ``cells`` [F, 3] in nm; optional labels ``energy`` [F] in
    kJ/mol, ``forces`` [F, N, 3] in kJ/mol/nm and ``virial`` [F, 3, 3] in kJ/mol. All are
    NumPy arrays or tensors of one dtype, float32 or float64. Indexing with an integer, a
    slice or an integer array returns those frames as a new Dataset.

    Raises ``ValueError`` for a wrong shape or dtype, a non-finite value or a nonpositive cell.
    """

    def __init__(
        self,
        positions: np.ndarray | Tensor,
        cells: np.ndarray | Tensor,
        *,
        energy: np.ndarray | Tensor | None = None,
        forces: np.ndarray | Tensor | None = None,
        virial: np.ndarray | Tensor | None = None,
    ):
        given = {
            "positions": positions,
            "cells": cells,
            "energy": energy,
            "forces": forces,
            "virial": virial,
        }
        fields = {name: _host_copy(value) for name, value in given.items() if value is not None}
        _check_positions(fields["positions"])
        _check_fields(fields)
        self._fields = fields

    def __len__(self) -> int:
        return self._fields["positions"].shape[0]

    def __getitem__(self, key) -> "Dataset":
        if isinstance(key, slice):
            index = torch.arange(len(self))[key]
        else:
            index = torch.as_tensor(key).cpu()
        index = index.reshape(-1)
        selected = Dataset.__new__(Dataset)
        selected._fields = {name: value[index] for name, value in self._fields.items()}
        return selected

    @property
    def positions(self) -> Tensor:
        """The positions [F, N, 3] in nm."""
        return self._fields["positions"]

    @property
    def cells(self) -> Tensor:
        """The cell lengths [F, 3] in nm."""
        return self._fields["cells"]

    @property
    def labels(self) -> dict[str, Tensor]:
        """The labels present, by channel name: ``energy``, ``forces``, ``virial``."""
        return {name: self._fields[name] for name in CHANNELS if name in self._fields}


__all__ = ("Dataset",)
