#pragma once

#include <cstdint>
#include <algorithm>

/// @author Wojciech Jabłoński <wj359634@students.mimuw.edu.pl>
template <typename T>
T emd_relaxed(const T *__restrict__ w1, const T *__restrict__ w2,
              const T *__restrict__ dist, uint32_t size,
              int32_t *__restrict__ cache) {  // at least size elements
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
          cache + size,
          [&](const int a, const int b) {
            return dist[i*size + a] < dist[i * size + b];
          });

        T remaining = w1[i];
        for (size_t j = 0; j < size; j++) {
          int w = cache[j];
          if (remaining < w2[w]) {
            acc += remaining * dist[i * size + w];
            break;
          } else {
            remaining -= w2[w];
            acc += w2[w] * dist[i * size + w];
          }
        }
      }
    }
    score = std::max(score, acc);
    std::swap(w1, w2);
  }
  return score;
}
