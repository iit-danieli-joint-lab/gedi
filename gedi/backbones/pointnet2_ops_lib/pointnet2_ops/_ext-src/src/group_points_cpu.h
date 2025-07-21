#include "group_points.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
void group_points_cpu(int b, int c, int n, int npoints, int nsample,
                      const float *points, const int *idx, float *out)
{
  for (int bs = 0; bs < b; ++bs)
  {
    const float *cur_points = points + bs * c * n;
    const int *cur_idx = idx + bs * npoints * nsample;
    float *cur_out = out + bs * c * npoints * nsample;

    for (int i = 0; i < c * npoints; ++i)
    {
      int l = i / npoints;
      int j = i % npoints;

      for (int k = 0; k < nsample; ++k)
      {
        int ii = cur_idx[j * nsample + k];
        cur_out[(l * npoints + j) * nsample + k] = cur_points[l * n + ii];
      }
    }
  }
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
void group_points_grad_cpu(int b, int c, int n, int npoints, int nsample,
                           const float *grad_out, const int *idx, float *grad_points)
{
  for (int bs = 0; bs < b; ++bs)
  {
    const float *cur_grad_out = grad_out + bs * c * npoints * nsample;
    const int *cur_idx = idx + bs * npoints * nsample;
    float *cur_grad_points = grad_points + bs * c * n;

    for (int i = 0; i < c * npoints; ++i)
    {
      int l = i / npoints;
      int j = i % npoints;

      for (int k = 0; k < nsample; ++k)
      {
        int ii = cur_idx[j * nsample + k];
        cur_grad_points[l * n + ii] += cur_grad_out[(l * npoints + j) * nsample + k];
      }
    }
  }
}