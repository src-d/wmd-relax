#include <cstdint>
#include <cmath>
#include <algorithm>

#include "graph/min_cost_flow.h"


/// @author Wojciech Jabłoński <wj359634@students.mimuw.edu.pl>

namespace {


const int64_t MASS_MULT = 1000 * 1000 * 1000;   // weights quantization constant
const int64_t COST_MULT = 1000 * 1000;          // costs quantization constant


class EMDCache {
 public:
  enum AllocationError {
    kAllocationErrorSuccess = 0,
    /// Can't allocate empty cache.
    kAllocationErrorInvalidSize,
    /// You have to deallocate the cache first before allocating again.
    kAllocationErrorDeallocationNeeded
  };

  EMDCache() : size_(0) {}

  bool* side() const noexcept {
    return side_.get();
  }

  int64_t* demand() const noexcept {
    return demand_.get();
  }

  int64_t* cost() const noexcept {
    return cost_.get();
  }

  size_t get_size() const noexcept {
    return size_;
  }

  operations_research::SimpleMinCostFlow& min_cost_flow() const noexcept {
    return min_cost_flow_;
  }

  AllocationError allocate(size_t size) {
    if (size == 0) {
      return kAllocationErrorInvalidSize;
    }
    if (size_ != 0) {
      return kAllocationErrorDeallocationNeeded;
    }
    size_ = size;
    side_.reset(new bool[size]);
    demand_.reset(new int64_t[size]);
    cost_.reset(new int64_t[size * size]);
    return kAllocationErrorSuccess;
  }

  void reset() noexcept {
    size_ = 0;
    side_.reset();
    demand_.reset();
    cost_.reset();
    min_cost_flow_.Reset();
  }

 private:
  mutable std::unique_ptr<bool[]> side_;
  mutable std::unique_ptr<int64_t[]> demand_;
  mutable std::unique_ptr<int64_t[]> cost_;
  size_t size_;
  mutable operations_research::SimpleMinCostFlow min_cost_flow_;

  EMDCache(const EMDCache&) = delete;
  EMDCache& operator=(const EMDCache&) = delete;
};


template <typename T>
void convert_weights(const T*__restrict__ in, bool sign,
                     int64_t*__restrict__ out, size_t size) {
  assert(in && out);
  assert(size > 0);
  int64_t sum = 0;
  double old_s = 0, new_s = 0;
  double mult = (sign ? -1 : 1);
  #pragma omp simd
  for (size_t i = 0; i < size; i++) {
    old_s = new_s;
    new_s = old_s + in[i];
    int64_t w = round(new_s * MASS_MULT) - round(old_s * MASS_MULT);
    sum += w;
    out[i] += w * mult;
  }
  if (sum != MASS_MULT) {
    if (abs(sum - MASS_MULT + 0.) / MASS_MULT > 0.0000001) {
#ifndef NDEBUG
      assert(sum == MASS_MULT && "Masses on one side not sufficiently normalized.");
#else
      fprintf(stderr, "wmd: weights are not normalized: %li != %li\n",
              sum, MASS_MULT);
#endif
    } else {
      // compensate for the rounding error
      out[0] += (sign ? 1 : -1) * (sum - MASS_MULT);
    }
  }
}


template <typename T>
void convert_costs(const T*__restrict__ in, const bool*__restrict__ side,
                   int64_t*__restrict__ out, size_t size) {
  #pragma omp simd
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      out[i * size + j] = round(in[i * size + j] * COST_MULT);
    }
  }

  #pragma omp simd
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if (side[i] && !side[j]) {
        out[i * size + j] = -out[j * size + i];
      }
    }
  }
}

}   // namespace


template <typename T>
T emd(const T*__restrict__ w1, const T*__restrict__ w2,
      const T*__restrict__ dist, uint32_t size, const EMDCache& cache) {
  assert(w1 && w2 && dist);
  assert(size > 0);
#ifndef NDEBUG
  assert(cache.get_size() >= size && "EMDCache not big enough.");
#else
  if (cache.get_size() < size) {
    fprintf(stderr, "emd: cache size is too small: %zu < %u\n",
            cache.get_size(), size);
    return -1;
  }
#endif
  bool* side = cache.side();
  int64_t* demand = cache.demand();
  int64_t* cost = cache.cost();

  memset(demand, 0, size * sizeof(demand[0]));
  convert_weights(w1, false, demand, size);
  convert_weights(w2, true,  demand, size);

  #pragma omp simd
  for (size_t i = 0; i < size; i++) {
    side[i] = (demand[i] < 0);
  }
  convert_costs(dist, side, cost, size);

  auto& min_cost_flow = cache.min_cost_flow();
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if (!side[i] && side[j]) {
        min_cost_flow.AddArcWithCapacityAndUnitCost(
            i, j, std::min(demand[i], -demand[j]), cost[i * size + j]);
      }
    }
  }
  for (size_t i = 0; i < size; i++) {
    min_cost_flow.SetNodeSupply(i, demand[i]);
  }
  auto status = min_cost_flow.Solve();
  double result = min_cost_flow.OptimalCost();
  min_cost_flow.Reset();
#ifndef NDEBUG
  assert(status == operations_research::SimpleMinCostFlow::OPTIMAL);
#else
  if (status != operations_research::SimpleMinCostFlow::OPTIMAL) {
    fprintf(stderr, "wmd: status is %d\n", status);
    return -status;
  }
#endif
  return T((result / MASS_MULT) / COST_MULT);
}
