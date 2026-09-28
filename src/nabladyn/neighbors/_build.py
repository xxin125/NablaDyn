"""Build settings of the CUDA neighbor extension, shared by setup.py and the JIT loader."""

SOURCES = ("csrc/ops.cpp", "csrc/runtime/entry_points.cu")

# IEEE arithmetic: no fast math and no flush-to-zero.
CXX_FLAGS = ["-O3", "-std=c++17"]
NVCC_FLAGS = ["-O3", "-std=c++17", "--ftz=false"]

__all__ = ("CXX_FLAGS", "NVCC_FLAGS", "SOURCES")
