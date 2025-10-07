#include "sampling.h"
#include "sampling_cpu.h"
#include "utils.h"

#ifdef WITH_CUDA
void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);
void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);
#endif

at::Tensor gather_points(at::Tensor points, at::Tensor idx)
{
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.is_cuda())
  {
    CHECK_CUDA(idx);

#ifdef WITH_CUDA
    at::Tensor output =
        torch::zeros({points.size(0), points.size(1), idx.size(1)},
                     at::device(points.device()).dtype(at::ScalarType::Float));

    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points.data_ptr<float>(),
                                 idx.data_ptr<int>(), output.data_ptr<float>());

    return output;
#else
    TORCH_CHECK(false, "Not compiled with CUDA support");
#endif
  }
  else
  {
    at::Tensor output =
        torch::zeros({points.size(0), points.size(1), idx.size(1)},
                     at::device(points.device()).dtype(at::ScalarType::Float));

    gather_points_cpu_kernel(points.size(0), points.size(1), points.size(2),
                             idx.size(1), points.data_ptr<float>(),
                             idx.data_ptr<int>(), output.data_ptr<float>());

    return output;
  }
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n)
{
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.is_cuda())
  {
    CHECK_CUDA(idx);

#ifdef WITH_CUDA
    at::Tensor output =
        torch::zeros({grad_out.size(0), grad_out.size(1), n},
                     at::device(grad_out.device()).dtype(at::ScalarType::Float));

    gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), grad_out.data_ptr<float>(),
                                      idx.data_ptr<int>(),
                                      output.data_ptr<float>());

    return output;
#else
    TORCH_CHECK(false, "Not compiled with CUDA support");
#endif
  }
  else
  {
    at::Tensor output =
        torch::zeros({grad_out.size(0), grad_out.size(1), n},
                     at::device(grad_out.device()).dtype(at::ScalarType::Float));

    gather_points_grad_cpu_kernel(grad_out.size(0), grad_out.size(1), n,
                                  idx.size(1), grad_out.data_ptr<float>(),
                                  idx.data_ptr<int>(), output.data_ptr<float>());

    return output;
  }
}

at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples)
{
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.is_cuda())
  {
#ifdef WITH_CUDA
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points.data_ptr<float>(),
        tmp.data_ptr<float>(), output.data_ptr<int>());
#else
    TORCH_CHECK(false, "Not compiled with CUDA support");
#endif
  }
  else
  {
    furthest_point_sampling_cpu_kernel(
        points.size(0), points.size(1), nsamples, points.data_ptr<float>(),
        tmp.data_ptr<float>(), output.data_ptr<int>());
  }

  return output;
}
