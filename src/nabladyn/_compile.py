"""Compile policy for NablaDyn's numerical programs."""

import functools
import math

import torch
from torch._inductor.choices import InductorChoices
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.virtualized import V

# A program compiles once per system (atom count, dtype, ensemble). Past torch's default
# of 8 compiles per program, a full-graph program fails, so the limit is raised while
# NablaDyn's programs compile.
_RECOMPILE_LIMIT = 64
# The most values one thread block sums, as Inductor splits sums below compute capability 10.
_SPLIT_ABOVE = 8192


class _Choices(InductorChoices):
    """Inductor's heuristics, with every sum over more than 8,192 values split across blocks.

    From compute capability 10, Inductor keeps a sum over up to 524,288 values in one thread
    block and fuses the work that produces the values into it. A pair energy then runs its
    whole pair computation serially: the MD step of 4,096 argon atoms is about 3x slower.
    """

    @staticmethod
    def reduction_split_factor(device, reduction_numel_hint, numel_hint, inner_reduction):
        split = InductorChoices.reduction_split_factor(
            device,
            reduction_numel_hint,
            numel_hint,
            inner_reduction,
        )
        # With this many sums, each needs no more than one block to fill the GPU.
        many = 2 * DeviceProperties.create(device).multi_processor_count
        if split == 1 and inner_reduction and numel_hint < many:
            return max(1, math.ceil(reduction_numel_hint / _SPLIT_ABOVE))
        return split

    def uuid(self):
        """Name these heuristics in Inductor's cache key."""
        return f"nabladyn-split-above-{_SPLIT_ABOVE}"


def _compile(program):
    """Compile ``program`` with Inductor for its caller to capture in a CUDA Graph.

    One full graph with static shapes; Inductor's own CUDA Graphs and shape padding are
    off, and sums are split as ``_Choices`` says. The returned function has the program's
    signature.
    """
    compiled = torch.compile(
        program,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={
            "triton.cudagraphs": False,
            "shape_padding": False,
            "inductor_choices_class": _Choices,  # keys the cache; the handler below applies it
        },
    )

    @functools.wraps(program)
    def call(*args, **kwargs):
        with (
            torch._dynamo.config.patch(recompile_limit=_RECOMPILE_LIMIT),
            V.set_choices_handler(_Choices()),
        ):
            return compiled(*args, **kwargs)

    return call


__all__ = ()
