// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "Dtype.h"
#include "NeighborSearchCommon.h"
#include "TorchHelper.h"
#include "Helper.h"
#include "torch/script.h"
#include "torch/extension.h"

using namespace open3d::core::nns;

template <class T, class TIndex>
void RadiusSearchCPU(const torch::Tensor &points,
                     const torch::Tensor &queries,
                     const torch::Tensor &radii,
                     const torch::Tensor &points_row_splits,
                     const torch::Tensor &queries_row_splits,
                     const Metric metric,
                     const bool ignore_query_point,
                     const bool return_distances,
                     const bool normalize_distances,
                     torch::Tensor &neighbors_index,
                     torch::Tensor &neighbors_row_splits,
                     torch::Tensor &neighbors_distance);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MultiRadiusSearch(
    torch::Tensor points,
    torch::Tensor queries,
    torch::Tensor radii,
    torch::Tensor points_row_splits,
    torch::Tensor queries_row_splits,
    torch::ScalarType index_dtype,
    const std::string &metric_str,
    const bool ignore_query_point,
    const bool return_distances,
    const bool normalize_distances)
{
    Metric metric = L2;
    if (metric_str == "L1")
    {
        metric = L1;
    }
    else if (metric_str == "L2")
    {
        metric = L2;
    }
    else
    {
        TORCH_CHECK(false,
                    "metric must be one of (L1, L2) but got " + metric_str);
    }
    CHECK_TYPE(points_row_splits, kInt64);
    CHECK_TYPE(queries_row_splits, kInt64);
    CHECK_SAME_DTYPE(points, queries, radii);
    CHECK_SAME_DEVICE_TYPE(points, queries, radii);
    TORCH_CHECK(index_dtype == torch::kInt32 || index_dtype == torch::kInt64,
                "index_dtype must be int32 or int64");
    // ensure that these are on the cpu
    points_row_splits = points_row_splits.to(torch::kCPU);
    queries_row_splits = queries_row_splits.to(torch::kCPU);
    points = points.contiguous();
    queries = queries.contiguous();
    radii = radii.contiguous();
    points_row_splits = points_row_splits.contiguous();
    queries_row_splits = queries_row_splits.contiguous();

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_points("num_points");
    Dim num_queries("num_queries");
    Dim batch_size("batch_size");
    Dim num_cells("num_cells");
    CHECK_SHAPE(points, num_points, 3);
    CHECK_SHAPE(queries, num_queries, 3);
    CHECK_SHAPE(radii, num_queries);
    CHECK_SHAPE(points_row_splits, batch_size + 1);
    CHECK_SHAPE(queries_row_splits, batch_size + 1);

    const auto &point_type = points.dtype();

    auto device = points.device().type();
    auto device_idx = points.device().index();

    torch::Tensor neighbors_index;
    torch::Tensor neighbors_row_splits = torch::empty(
        {queries.size(0) + 1},
        torch::dtype(ToTorchDtype<int64_t>()).device(device, device_idx));
    torch::Tensor neighbors_distance;

#define FN_PARAMETERS                                                      \
    points, queries, radii, points_row_splits, queries_row_splits, metric, \
        ignore_query_point, return_distances, normalize_distances,         \
        neighbors_index, neighbors_row_splits, neighbors_distance

    if (points.is_cuda())
    {
        TORCH_CHECK(false, "MultiRadiusSearch does not support CUDA")
    }
    else
    {
        if (CompareTorchDtype<float>(point_type))
        {
            if (index_dtype == torch::kInt32)
            {
                RadiusSearchCPU<float, int32_t>(FN_PARAMETERS);
            }
            else
            {
                RadiusSearchCPU<float, int64_t>(FN_PARAMETERS);
            }
        }
        else
        {
            if (index_dtype == torch::kInt32)
            {
                RadiusSearchCPU<double, int32_t>(FN_PARAMETERS);
            }
            else
            {
                RadiusSearchCPU<double, int64_t>(FN_PARAMETERS);
            }
        }
        return std::make_tuple(neighbors_index, neighbors_row_splits,
                               neighbors_distance);
    }
    TORCH_CHECK(false, "MultiRadiusSearch does not support " +
                           points.toString() + " as input for points")
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>();
}

PYBIND11_MODULE(gedi_radius_search_op, m)
{
    m.def("radius_search", [](torch::Tensor points, torch::Tensor queries, torch::Tensor radii, torch::Tensor points_row_splits, torch::Tensor queries_row_splits, int index_dtype_int = 3, const std::string &metric_str = "L2", bool ignore_query_point = false, bool return_distances = true, bool normalize_distances = false)
          {
            c10::ScalarType index_dtype = static_cast<c10::ScalarType>(index_dtype_int);
            return MultiRadiusSearch(points, queries, radii,
                                     points_row_splits, queries_row_splits,
                                     index_dtype, metric_str,
                                     ignore_query_point,
                                     return_distances,
                                     normalize_distances); }, "Multi-radius search function");
}
