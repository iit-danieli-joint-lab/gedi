#include "sampling.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
void gather_points_cpu_kernel(int b, int c, int n, int m,
                              const float *points,
                              const int *idx,
                              float *out)
{
  for (int i = 0; i < b; ++i)
  {
    for (int l = 0; l < c; ++l)
    {
      for (int j = 0; j < m; ++j)
      {
        int a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
void gather_points_grad_cpu_kernel(int b, int c, int n, int m,
                                   const float *grad_out,
                                   const int *idx,
                                   float *grad_points)
{
  for (int i = 0; i < b; ++i)
  {
    for (int l = 0; l < c; ++l)
    {
      for (int j = 0; j < m; ++j)
      {
        int a = idx[i * m + j];
        grad_points[(i * c + l) * n + a] += grad_out[(i * c + l) * m + j];
      }
    }
  }
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
void furthest_point_sampling_cpu_kernel(int b, int n, int m,
                                        const float *dataset,
                                        float *temp,
                                        int *idxs)
{
  for (int batch_index = 0; batch_index < b; ++batch_index)
  {
    const float *cur_dataset = dataset + batch_index * n * 3;
    float *cur_temp = temp + batch_index * n;
    int *cur_idxs = idxs + batch_index * m;

    int old = 0;
    cur_idxs[0] = old;

    for (int j = 1; j < m; ++j)
    {
      int besti = 0;
      float best = -1.0f;

      float x1 = cur_dataset[old * 3 + 0];
      float y1 = cur_dataset[old * 3 + 1];
      float z1 = cur_dataset[old * 3 + 2];

      for (int k = 0; k < n; ++k)
      {
        float x2 = cur_dataset[k * 3 + 0];
        float y2 = cur_dataset[k * 3 + 1];
        float z2 = cur_dataset[k * 3 + 2];

        float mag = x2 * x2 + y2 * y2 + z2 * z2;
        if (mag <= 1e-3)
          continue;

        float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

        float d2 = std::min(d, cur_temp[k]);
        cur_temp[k] = d2;

        if (d2 > best)
        {
          best = d2;
          besti = k;
        }
      }

      old = besti;
      cur_idxs[j] = old;
    }
  }
}