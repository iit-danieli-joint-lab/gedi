#include "ball_query.h"
#include "ball_query_cpu.h"
#include "utils.h"

#ifdef WITH_CUDA
void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx);
#endif

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample)
{
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  if (new_xyz.is_cuda())
  {
    CHECK_CUDA(xyz);

#ifdef WITH_CUDA
    at::Tensor idx =
        torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                     at::device(new_xyz.device()).dtype(at::ScalarType::Int));

    query_ball_point_kernel_wrapper(
        xyz.size(0), xyz.size(1), new_xyz.size(1),
        radius, nsample,
        new_xyz.data_ptr<float>(), xyz.data_ptr<float>(), idx.data_ptr<int>());

    return idx;
#else
    TORCH_CHECK(false, "Not compiled with CUDA support");
#endif
  }
  else
  {
    at::Tensor idx =
        torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                     at::device(new_xyz.device()).dtype(at::ScalarType::Int));

    query_ball_point_cpu(xyz.size(0), xyz.size(1), new_xyz.size(1),
                         radius, nsample,
                         new_xyz.data_ptr<float>(), xyz.data_ptr<float>(), idx.data_ptr<int>());

    return idx;
  }
}
