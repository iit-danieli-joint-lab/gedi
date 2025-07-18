#pragma once
#include <torch/extension.h>

#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x)                                    \
  do                                                     \
  {                                                      \
    AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)
#else
#define CHECK_CUDA(x)                                                                     \
  do                                                                                      \
  {                                                                                       \
    TORCH_CHECK(false, #x " WARNING: Requested CUDA but not compiled with CUDA support"); \
  } while (0)
#endif

#define CHECK_CONTIGUOUS(x)                                          \
  do                                                                 \
  {                                                                  \
    AT_ASSERT(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                               \
  do                                                  \
  {                                                   \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Int, \
              #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                               \
  do                                                    \
  {                                                     \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Float, \
              #x " must be a float tensor");            \
  } while (0)
