#pragma once

#include <cstdint>
#include <algorithm>

#include "cache.h"


namespace {

/// The cache for emd_relaxed().
class EMDRelaxedCache : public wmd::Cache {
 public:
  int32_t* boilerplate() const noexcept {
    return boilerplate_.get();
  }

 protected:
  void _allocate() override {
    boilerplate_.reset(new int32_t[size_]);
  }

  void _reset() noexcept override {
    boilerplate_.reset();
  }

 private:
  mutable std::unique_ptr<int32_t[]> boilerplate_;
};

}


/// Solves the relaxed EMD problem. The relaxation is tighter than the original
/// one from the WMD paper. We set f_{ij} &le; Q_j instead of dropping it; also,
/// because the problem is symmetric, we obtain the solution on inputs and
/// swapped inputs and choose the maximum of the scores.
/// @param w1 The first array with weights of length `size`.
/// @param w2 The second array with weights of length `size`.
/// @param dist The costs matrix of shape `size` x `size`.
/// @param size The dimensionality of the problem.
/// @param cache The cache to use. It should be initialized with at least `size`
///              elements.
/// @author Wojciech Jabłoński <wj359634@students.mimuw.edu.pl>
template <typename T>
T emd_relaxed(const T *__restrict__ w1, const T *__restrict__ w2,
              const T *__restrict__ dist, uint32_t size,
              const EMDRelaxedCache& cache) {
  std::lock_guard<std::mutex> _(cache.enter(size));
  auto boilerplate = cache.boilerplate();
  for (size_t i = 0; i < size; i++) {
    boilerplate[i] = i;
  }

  T score = 0;
  for (size_t c = 0; c < 2; c++) {
    T acc = 0;
    for (size_t i = 0; i < size; i++) {
      if (w1[i] != 0) {
        std::sort(
          boilerplate,
          boilerplate + size,
          [&](const int a, const int b) {
            return dist[i * size + a] < dist[i * size + b];
          });

        T remaining = w1[i];
        for (size_t j = 0; j < size; j++) {
          int w = boilerplate[j];
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
