import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *
import os

try:
    from . import _ext as _ext
except ImportError as e:
    # Prebuilt extension missing → fall back to JIT compilation automatically.
    warnings.warn(
        "pointnet2_ops prebuilt extension not found; running JIT build. "
        "This is only for development purpose and not intended to be used with prebuilt wheels."
    )

    # JIT compile the extension (as done in setup.py)
    from torch.utils.cpp_extension import load, CppExtension, CUDAExtension
    import glob
    import os.path as osp
    from pathlib import Path

    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.5;8.6;8.9"

    this_dir = Path(__file__).parent.resolve()
    _ext_src_root = osp.join(this_dir, "_ext-src")

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

    # -----------------------------
    # Pointnet2 extension
    # -----------------------------
    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-Xfatbin", "-compress-all"]}
    extra_cflags = extra_compile_args.get("cxx", ["-O3"])
    extra_cuda_cflags = extra_compile_args.get("nvcc", ["-O3", "-Xfatbin", "-compress-all"])
    
    define_macros = []
    if use_cuda:
        define_macros.append(("WITH_CUDA", None))

    if define_macros:
        cflags_macros = [f"-D{name}" if value is None else f"-D{name}={value}" for name, value in define_macros]
        extra_cflags.extend(cflags_macros)
        if use_cuda:
            extra_cuda_cflags.extend(cflags_macros)

    pixi_includes, pixi_libdirs, pixi_libs = collect_pixi_deps()
    pointnet_includes = [osp.join(_ext_src_root, "include")]
    o3d_includes = [osp.join(_ext_src_root, "include", "open3d")]

    _ext = load(
        name="gedi_pointnet2",
        sources=cpp_sources + cuda_sources if use_cuda else cpp_sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=pointnet_includes + pixi_includes,
        with_cuda=use_cuda,
        verbose=True
    )

    # -----------------------------
    # RadiusSearch extension
    # -----------------------------
    prefix = "/LIBPATH:" if os.name == "nt" else "-L"
    extra_ldflags = [f"{prefix}{d}" for d in pixi_libdirs if d]

    extra_ldflags_rs = list(extra_ldflags)

    if os.name == "nt":
        extra_compile_args_rs = {"cxx": ["/utf-8"]}
        extra_ldflags_rs.extend(f"{lib}.lib" for lib in pixi_libs if lib)
    else:
        extra_compile_args_rs = {"cxx": ["-finput-charset=UTF-8", "-fexec-charset=UTF-8"]}
        extra_ldflags_rs.extend(f"-l{lib}" for lib in pixi_libs if lib)

    gedi_radius_search_op = load(
        name="gedi_radius_search_op",
        sources=[
            osp.join(_ext_src_root, "src", "RadiusSearchOps.cpp"),
            osp.join(_ext_src_root, "src", "RadiusSearchOpKernel.cpp"),
        ],
        extra_cflags=extra_compile_args_rs.get("cxx", []),
        extra_include_paths=o3d_includes + pixi_includes,
        extra_ldflags=extra_ldflags_rs,
        verbose=True
    )

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = _ext.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        ctx.save_for_backward(idx, features)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
    # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        ctx.save_for_backward(idx, weight, features)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        ctx.save_for_backward(idx, features)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        output = _ext.ball_query(new_xyz, xyz, radius, nsample)

        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
    # type: (QueryAndGroup, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
