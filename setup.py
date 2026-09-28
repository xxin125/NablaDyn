"""NablaDyn with its CUDA neighbor extension, compiled at installation.

pip install --no-build-isolation .
"""

import runpy
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent
NEIGHBORS = Path("src/nabladyn/neighbors")
BUILD = runpy.run_path(str(HERE / NEIGHBORS / "_build.py"))
# CUB ships under include/cccl in pip-installed CUDA toolkits.
CCCL = [str(Path(CUDA_HOME) / "include" / "cccl")] if CUDA_HOME else []

setup(
    ext_modules=[
        CUDAExtension(
            "nabladyn.neighbors._C",
            sources=[str(NEIGHBORS / source) for source in BUILD["SOURCES"]],
            # Absolute: the compiler runs inside the build directory.
            include_dirs=[str(HERE / NEIGHBORS / "csrc"), *CCCL],
            extra_compile_args={"cxx": BUILD["CXX_FLAGS"], "nvcc": BUILD["NVCC_FLAGS"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
