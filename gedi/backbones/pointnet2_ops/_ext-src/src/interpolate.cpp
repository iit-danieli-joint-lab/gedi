#include "interpolate.h"
#include "interpolate_cpu.h"
#include "utils.h"

#ifdef WITH_CUDA
void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx);
void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out);
void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points);
#endif

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows)
{
  CHECK_CONTIGUOUS(unknowns);
  CHECK_CONTIGUOUS(knows);
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);

  if (unknowns.is_cuda())
  {
    CHECK_CUDA(knows);

#ifdef WITH_CUDA
    at::Tensor idx =
        torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                     at::device(unknowns.device()).dtype(at::ScalarType::Int));
    at::Tensor dist2 =
        torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                     at::device(unknowns.device()).dtype(at::ScalarType::Float));

    three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                            unknowns.data_ptr<float>(), knows.data_ptr<float>(),
                            dist2.data_ptr<float>(), idx.data_ptr<int>());

    return {dist2, idx};
#else
    TORCH_CHECK(false, "Not compiled with CUDA support");
#endif
  }
  else
  {
    at::Tensor idx =
        torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                     at::device(unknowns.device()).dtype(at::ScalarType::Int));
    at::Tensor dist2 =
        torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                     at::device(unknowns.device()).dtype(at::ScalarType::Float));

    three_nn_cpu(unknowns.size(0), unknowns.size(1), knows.size(1),
                 unknowns.data_ptr<float>(), knows.data_ptr<float>(),
                 dist2.data_ptr<float>(), idx.data_ptr<int>());

    return {dist2, idx};
  }
}

at::Tensor three_interpolate(at::Tensor points, at::Tensor idx,
                             at::Tensor weight)
{
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (points.is_cuda())
  {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);

#ifdef WITH_CUDA
    at::Tensor output =
        torch::zeros({points.size(0), points.size(1), idx.size(1)},
                     at::device(points.device()).dtype(at::ScalarType::Float));

    three_interpolate_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), idx.size(1),
        points.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(),
        output.data_ptr<float>());

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

    three_interpolate_cpu(points.size(0), points.size(1), points.size(2), idx.size(1),
                          points.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(),
                          output.data_ptr<float>());

    return output;
  }
}

at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m)
{
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (grad_out.is_cuda())
  {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);

#ifdef WITH_CUDA
    at::Tensor output =
        torch::zeros({grad_out.size(0), grad_out.size(1), m},
                     at::device(grad_out.device()).dtype(at::ScalarType::Float));

    three_interpolate_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
        grad_out.data_ptr<float>(), idx.data_ptr<int>(),
        weight.data_ptr<float>(), output.data_ptr<float>());

    return output;
#else
    TORCH_CHECK(false, "Not compiled with CUDA support");
#endif
  }
  else
  {
    at::Tensor output =
        torch::zeros({grad_out.size(0), grad_out.size(1), m},
                     at::device(grad_out.device()).dtype(at::ScalarType::Float));

    three_interpolate_grad_cpu(grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
                               grad_out.data_ptr<float>(), idx.data_ptr<int>(),
                               weight.data_ptr<float>(), output.data_ptr<float>());

    return output;
  }
}
