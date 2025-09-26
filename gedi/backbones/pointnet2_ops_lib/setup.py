from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os
import glob
import os.path as osp
import torch
from pathlib import Path

this_dir = Path(__file__).parent.resolve()
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")

# -----------------------------
# Collect Pixi dependencies
# -----------------------------
def collect_pixi_deps():
    conda_prefix = os.environ.get("CONDA_PREFIX", None)
    include_dirs, library_dirs, libraries = [], [], []

    if conda_prefix:
        if os.name == "nt":
            include_dirs.append(os.path.join(conda_prefix, "Library", "include"))
            library_dirs.append(os.path.join(conda_prefix, "Library", "lib"))
        else:
            include_dirs.append(os.path.join(conda_prefix, "include"))
            library_dirs.append(os.path.join(conda_prefix, "lib"))

        # Check for binary libraries in lib/
        libdir = library_dirs[0] if library_dirs else None
        if libdir and osp.isdir(libdir):
            libfiles = os.listdir(libdir)

            def has_lib(name):
                return any(name in f for f in libfiles)

            if has_lib("tbb"):
                libraries.append("tbb")
            if has_lib("fmt"):
                libraries.append("fmt")

        # Eigen and nanoflann: header-only → only include_dirs
        include_dirs.append(os.path.join(conda_prefix, "include", "eigen3"))
        include_dirs.append(os.path.join(conda_prefix, "include")) 

    return include_dirs, library_dirs, libraries

all_cpp = glob.glob(osp.join(_ext_src_root, "src", "*.cpp"))
all_cu = glob.glob(osp.join(_ext_src_root, "src", "*.cu"))
exclude_cpp = {"RadiusSearchOps.cpp", "RadiusSearchOpKernel.cpp"}
cpp_sources = [s for s in all_cpp if osp.basename(s) not in exclude_cpp]
cuda_sources = list(all_cu)

use_cuda = torch.cuda.is_available() and torch.utils.cpp_extension.CUDA_HOME is not None

define_macros = [("WITH_CUDA", None)] if use_cuda else []

pixi_includes, pixi_libdirs, pixi_libs = collect_pixi_deps()

pointnet_includes = [osp.join(this_dir.as_posix(), _ext_src_root, "include")]
o3d_includes = [osp.join(this_dir.as_posix(), _ext_src_root, "include", "open3d")]

# -----------------------------
# Pointnet2 extension
# -----------------------------
if use_cuda and cuda_sources:
    torch_ext = CUDAExtension(
        name="pointnet2_ops._ext",
        sources=cpp_sources + cuda_sources,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
        },
        include_dirs=pointnet_includes + pixi_includes,
        library_dirs=pixi_libdirs,
        define_macros=define_macros,
    )
else:
    torch_ext = CppExtension(
        name="pointnet2_ops._ext",
        sources=cpp_sources,
        extra_compile_args={"cxx": ["-O3"]},
        include_dirs=pointnet_includes + pixi_includes,
        library_dirs=pixi_libdirs,
        define_macros=define_macros,
    )

# -----------------------------
# RadiusSearch extension
# -----------------------------
radius_search_ext = CppExtension(
    name="pointnet2_ops.gedi_radius_search_op",
    sources=[
        osp.join(_ext_src_root, "src", "RadiusSearchOps.cpp"),
        osp.join(_ext_src_root, "src", "RadiusSearchOpKernel.cpp"),
    ],
    include_dirs=o3d_includes + pixi_includes,
    library_dirs=pixi_libdirs,
    libraries=pixi_libs,
    extra_compile_args = {"cxx": ["/std:c++17", "/utf-8", "/MD"]},
    define_macros=[],
)

setup(
    name="pointnet2_ops",
    version="3.1.0",
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=["torch>=1.4"],
    ext_modules=[torch_ext, radius_search_ext],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
    package_data={
        "pointnet2_ops": ["*.dll", "*.so", "*.dylib"],
    },
)