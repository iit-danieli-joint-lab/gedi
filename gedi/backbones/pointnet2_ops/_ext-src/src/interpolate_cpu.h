#include <interpolate.h>
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace at;

// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
void three_nn_cpu(int b, int n, int m,
                  const float *unknown, const float *known,
                  float *dist2, int *idx)
{
  for (int batch = 0; batch < b; ++batch)
  {
    const float *cur_unknown = unknown + batch * n * 3;
    const float *cur_known = known + batch * m * 3;
    float *cur_dist2 = dist2 + batch * n * 3;
    int *cur_idx = idx + batch * n * 3;

    for (int j = 0; j < n; ++j)
    {
      float ux = cur_unknown[j * 3 + 0];
      float uy = cur_unknown[j * 3 + 1];
      float uz = cur_unknown[j * 3 + 2];

      double best1 = 1e40, best2 = 1e40, best3 = 1e40;
      int besti1 = 0, besti2 = 0, besti3 = 0;

      for (int k = 0; k < m; ++k)
      {
        float x = cur_known[k * 3 + 0];
        float y = cur_known[k * 3 + 1];
        float z = cur_known[k * 3 + 2];

        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);

        if (d < best1)
        {
          best3 = best2;
          besti3 = besti2;
          best2 = best1;
          besti2 = besti1;
          best1 = d;
          besti1 = k;
        }
        else if (d < best2)
        {
          best3 = best2;
          besti3 = besti2;
          best2 = d;
          besti2 = k;
        }
        else if (d < best3)
        {
          best3 = d;
          besti3 = k;
        }
      }

      cur_dist2[j * 3 + 0] = static_cast<float>(best1);
      cur_dist2[j * 3 + 1] = static_cast<float>(best2);
      cur_dist2[j * 3 + 2] = static_cast<float>(best3);

      cur_idx[j * 3 + 0] = besti1;
      cur_idx[j * 3 + 1] = besti2;
      cur_idx[j * 3 + 2] = besti3;
    }
  }
}

// input: points(b, c, m), idx(b, n, 3), weight(b, n, 3)
// output: out(b, c, n)
void three_interpolate_cpu(int b, int c, int m, int n,
                           const float *points,
                           const int *idx,
                           const float *weight,
                           float *out)
{
  for (int batch = 0; batch < b; ++batch)
  {
    const float *cur_points = points + batch * m * c;
    const int *cur_idx = idx + batch * n * 3;
    const float *cur_weight = weight + batch * n * 3;
    float *cur_out = out + batch * n * c;

    for (int i = 0; i < c * n; ++i)
    {
      int l = i / n;
      int j = i % n;

      float w1 = cur_weight[j * 3 + 0];
      float w2 = cur_weight[j * 3 + 1];
      float w3 = cur_weight[j * 3 + 2];

      int i1 = cur_idx[j * 3 + 0];
      int i2 = cur_idx[j * 3 + 1];
      int i3 = cur_idx[j * 3 + 2];

      cur_out[i] = cur_points[l * m + i1] * w1 +
                   cur_points[l * m + i2] * w2 +
                   cur_points[l * m + i3] * w3;
    }
  }
}

// input: grad_out(b, c, n), idx(b, n, 3), weight(b, n, 3)
// output: grad_points(b, c, m)
void three_interpolate_grad_cpu(int b, int c, int n, int m,
                                const float *grad_out,
                                const int *idx,
                                const float *weight,
                                float *grad_points)
{
  for (int batch = 0; batch < b; ++batch)
  {
    const float *cur_grad_out = grad_out + batch * n * c;
    const int *cur_idx = idx + batch * n * 3;
    const float *cur_weight = weight + batch * n * 3;
    float *cur_grad_points = grad_points + batch * m * c;

    for (int i = 0; i < c * n; ++i)
    {
      int l = i / n;
      int j = i % n;

      float w1 = cur_weight[j * 3 + 0];
      float w2 = cur_weight[j * 3 + 1];
      float w3 = cur_weight[j * 3 + 2];

      int i1 = cur_idx[j * 3 + 0];
      int i2 = cur_idx[j * 3 + 1];
      int i3 = cur_idx[j * 3 + 2];

      cur_grad_points[l * m + i1] += cur_grad_out[i] * w1;
      cur_grad_points[l * m + i2] += cur_grad_out[i] * w2;
      cur_grad_points[l * m + i3] += cur_grad_out[i] * w3;
    }
  }
}