import os

from setuptools import setup


build_cub = os.environ.get("HAWKES_BUILD_CUB", "0")
if build_cub not in ("0", "1"):
    raise RuntimeError("HAWKES_BUILD_CUB must be 0 or 1.")

extensions, commands = [], {}
if build_cub == "1":
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except ImportError as exc:
        raise RuntimeError(
            "Install CUDA-enabled PyTorch first, then install HawkesTorch with "
            "HAWKES_BUILD_CUB=1 and pip --no-build-isolation."
        ) from exc

    if torch.version.cuda is None or CUDA_HOME is None:
        raise RuntimeError(
            "Building CUB requires CUDA-enabled PyTorch and a compatible CUDA "
            "toolkit. Set CUDA_HOME to the toolkit directory."
        )

    extensions = [
        CUDAExtension(
            "hawkes.utils._cub_scan_cuda",
            ["src/hawkes/utils/csrc/affine_scan.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ]
    commands["build_ext"] = BuildExtension.with_options(use_ninja=False)

setup(ext_modules=extensions, cmdclass=commands)
