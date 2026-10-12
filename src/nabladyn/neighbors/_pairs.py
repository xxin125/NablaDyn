"""The pairs a neighbor list hands to an energy."""

from typing import NamedTuple

from torch import Tensor


class NeighborPairs(NamedTuple):
    """The pairs of one frame, each pair once.

    ``index`` [2, C] int32: the first columns hold each pair closer than the list radius at
    the last build, in no set orientation; the rest are self-pairs (k mod N, k mod N), which
    an energy drops by keeping only ``index[0] != index[1]``. ``num_atoms`` is N; ``cutoff``
    is the list's cutoff in nm. ``inner`` holds the same frame's lists at smaller cutoffs,
    ascending, each without inner lists of its own; a sum of models gives each model the list
    at its own cutoff.
    """

    index: Tensor
    num_atoms: int
    cutoff: float
    inner: tuple["NeighborPairs", ...] = ()


__all__ = ("NeighborPairs",)
