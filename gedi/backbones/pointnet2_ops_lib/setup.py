from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os
import glob
import os.path as osp
import torch

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
cpp_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp"))
cuda_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cu"))
include_dirs = [osp.join(this_dir, _ext_src_root, "include")]

use_cuda = torch.cuda.is_available() and torch.utils.cpp_extension.CUDA_HOME is not None

define_macros = [("WITH_CUDA", None)] if use_cuda else []

if use_cuda:
    ext = CUDAExtension(
        name="pointnet2_ops._ext",
        sources=cpp_sources + cuda_sources,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
        },
        include_dirs=include_dirs,
        define_macros=define_macros,
    )
else:
    ext = CppExtension(
        name="pointnet2_ops._ext",
        sources=cpp_sources,
        extra_compile_args={"cxx": ["-O3"]},
        include_dirs=include_dirs,
        define_macros=define_macros,
    )

setup(
    name="pointnet2_ops",
    version="3.1.0",
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=["torch>=1.4"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)