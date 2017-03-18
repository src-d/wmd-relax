#pragma once

#include <cstdint>
#include <algorithm>
#include <memory>

template <typename T>
T emd_approx_relaxed(
  const T* w1,
  const T* w2,
  const T* dist,
  uint32_t size,
  std::shared_ptr<int32_t> cache_shared  // at least size elements
) {
  int32_t* cache = cache_shared.get();
  for (size_t i = 0; i < size; i++) {
    cache[i] = i;
  }

  T score = 0;
  for (size_t c = 0; c < 2; c++) {
    T acc = 0;
    for (size_t i = 0; i < size; i++) {
      if (w1[i] != T(0)) {
        std::sort(
          cache,
          cache+size,
          [&](const int a, const int b) {
            return dist[i*size + a] < dist[i*size + b];
          });

        T remaining = w1[i];
        for (size_t j = 0; j < size; j++) {
          int w = cache[j];
          if (remaining < w2[w]) {
            acc += remaining * dist[i*size + w];
            break;
          } else {
            remaining -= w2[w];
            acc += w2[w]*dist[i*size + w];
          }
        }
      }
    }
    score = std::max(score, acc);
    std::swap(w1, w2);
  }
  return score;
}
