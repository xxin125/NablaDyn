"""Loading of the CUDA neighbor operators."""

from pathlib import Path
from threading import Lock

import torch

from ._build import CXX_FLAGS, NVCC_FLAGS, SOURCES

_ROOT = Path(__file__).resolve().parent
_LOCK = Lock()
_loaded = False


def _load() -> None:
    """Load the installed extension, or build it from the sources and load it.

    Raises ``RuntimeError`` when a build finds no CUDA_HOME.
    """
    installed = sorted(_ROOT.glob("_C*.so"))
    if installed:
        torch.ops.load_library(str(installed[0]))
        return
    # Imported here: only a source checkout needs the build tools. The build is cached in
    # build/jit at the project root and redone only when the sources or flags change.
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError(
            "building the neighbor extension needs CUDA_HOME, a CUDA toolkit matching "
            "torch.version.cuda (see the README, Installation)",
        )
    build = _ROOT.parents[2] / "build" / "jit"
    build.mkdir(parents=True, exist_ok=True)
    load(
        name="nabladyn_neighbors",
        sources=[str(_ROOT / source) for source in SOURCES],
        extra_cflags=CXX_FLAGS,
        extra_cuda_cflags=NVCC_FLAGS,
        extra_include_paths=[
            str(_ROOT / "csrc"),
            str(Path(CUDA_HOME) / "include" / "cccl"),
        ],
        build_directory=str(build),
        is_python_module=False,
    )


def ops():
    """Return the ``torch.ops`` namespace of the neighbor operators, loaded once per process.

    In a source checkout the first call builds the extension, which takes minutes, and raises
    ``RuntimeError`` when it finds no CUDA_HOME.
    """
    global _loaded
    if not _loaded:
        with _LOCK:
            if not _loaded:
                _load()
                _loaded = True
    return torch.ops.nabladyn_neighbors


__all__ = ("ops",)
