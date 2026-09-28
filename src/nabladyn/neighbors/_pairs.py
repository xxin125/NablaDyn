"""The pairs a neighbor list hands to an energy."""

from typing import NamedTuple

from torch import Tensor


class NeighborPairs(NamedTuple):
    """The pairs of one frame, each pair once.

    ``index`` [2, C] int32: the first columns hold each pair closer than the list radius at
    the last build, in no set orientation; the rest are self-pairs (k mod N, k mod N), which
    an energy drops by keeping only ``index[0] != index[1]``. ``num_atoms`` is N; ``cutoff``
    is the list's cutoff in nm.
    """

    index: Tensor
    num_atoms: int
    cutoff: float


__all__ = ("NeighborPairs",)
