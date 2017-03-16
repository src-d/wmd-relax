#pragma once

#include <cstdint>
#include <algorithm>

template <typename T>
T emd_approx_relaxed (const T* w1, const T* w2, const T* dist, uint32_t size)
{
  static int idxes[1000*1000+7]; //to avoid any malloc in code
  for (size_t i=0; i<size; i++)
    idxes[i] = i;

  T my_score = 0;
  for (int c=0; c<2; c++)
  {
    T acc = 0;

    //stuff
    for (size_t i=0; i<size; i++)
      if (w1[i] != T(0))
      {
        std::sort(
          idxes,
          idxes+size, 
          [&](const int a, const int b){return dist[i*size + a]<dist[i*size + b];}
        );
        T remaining = w1[i];
        for (size_t j=0; j<size; j++)
        {
          int w = idxes[j];
          if (remaining < w2[w])
          {
            acc += remaining * dist[i*size + w];
            break;
          }
          else
          {
            remaining -= w2[w];
            acc += w2[w]*dist[i*size + w];
          }
        }
      }

    my_score = std::max(my_score, acc);
    std::swap(w1, w2);
  }
  return my_score;
}
