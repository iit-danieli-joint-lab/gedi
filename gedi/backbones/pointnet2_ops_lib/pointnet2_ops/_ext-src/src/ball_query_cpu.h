#include "ball_query.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
void query_ball_point_cpu(
    int b, int n, int m, float radius, int nsample,
    const float *new_xyz, const float *xyz, int *idx)
{
  float radius2 = radius * radius;

  for (int bs_idx = 0; bs_idx < b; ++bs_idx)
  {
    const float *cur_new_xyz = new_xyz + bs_idx * m * 3;
    const float *cur_xyz = xyz + bs_idx * n * 3;
    int *cur_idx = idx + bs_idx * m * nsample;

    for (int j = 0; j < m; ++j)
    {
      float new_x = cur_new_xyz[j * 3 + 0];
      float new_y = cur_new_xyz[j * 3 + 1];
      float new_z = cur_new_xyz[j * 3 + 2];

      int cnt = 0;
      for (int k = 0; k < n && cnt < nsample; ++k)
      {
        float x = cur_xyz[k * 3 + 0];
        float y = cur_xyz[k * 3 + 1];
        float z = cur_xyz[k * 3 + 2];

        float d2 = (new_x - x) * (new_x - x) +
                   (new_y - y) * (new_y - y) +
                   (new_z - z) * (new_z - z);

        if (d2 < radius2)
        {
          if (cnt == 0)
          {
            std::fill(cur_idx + j * nsample, cur_idx + (j + 1) * nsample, k);
          }
          cur_idx[j * nsample + cnt] = k;
          ++cnt;
        }
      }
    }
  }
}