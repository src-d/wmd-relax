// fix windows compile
#if defined(_MSC_VER) && !defined(__restrict__)
#define __restrict__ __restrict
#endif
// fix inttypes for GCC
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <cinttypes>
// fix for the fix - it conflicts with numpy
#undef __STDC_FORMAT_MACROS
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>

#include "cache.h"
#include "graph/min_cost_flow.h"


/*! @mainpage libwmdrelax
 *
 * @section s0 Description
 * This library allows to efficinetly solve the Earth Mover's Distance
 * problem (http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/RUBNER/emd.htm).
 * It also solves the relaxed approximation suitable for calculating the
 * Word Mover's Distance (http://www.cs.cornell.edu/~kilian/papers/wmd_metric.pdf),
 * hence the name.
 *
 * Project: https://github.com/src-d/wmd-relax
 *
 * README: @ref ignore_this_doxygen_anchor
 *
 * @section s1 C/C++ API
 * - emd() solves the original Earth Mover's Distance problem.
 * - emd_relaxed() solves the relaxed problem - one of the two sums is replaced
 *   with the maximum element.
 * - EMDCache and EMDRelaxedCache are the caches to prevent from dynamic memory
 *   allocation.
 *
 * Although C/C++ API is complete and totally usable as-is, python.cc provides
 * the Python 3 API.
 *
 * @section s2 Python 3 API
 *
 * - emd_relaxed()
 * - emd_relaxed_cache_init() creates the cache object for emd_relaxed()
 * - emd_relaxed_cache_fini() destroys the cache object for emd_relaxed()
 * - emd()
 * - emd_cache_init() creates the cache object for emd()
 * - emd_cache_fini() destroys the cache object for emd()
 *
 * @section s3 Building
 *
 * Normally, the library is built with setup.py as a part of the python package.
 * Besides, it can be built with cmake. In the latter case, ensure that you've
 * cloned or-tools submodule:
 * @code{.unparsed}
 * git submodule update --init
 * @endcode
 */


namespace {


const int64_t MASS_MULT = 1000 * 1000 * 1000;   // weights quantization constant
const int64_t COST_MULT = 1000 * 1000;          // costs quantization constant


/// The cache for emd().
class EMDCache : public wmd::Cache {
 public:
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

 protected:
  void _allocate() override {
    side_.reset(new bool[size_]);
    demand_.reset(new int64_t[size_]);
    cost_.reset(new int64_t[size_ * size_]);

    // warmup min_cost_flow_
    for (size_t i = 0; i < size_; i++) {
      for (size_t j = 0; j < size_; j++) {
        min_cost_flow_.AddArcWithCapacityAndUnitCost(i, j, 1, 1);
      }
    }
    for (size_t i = 0; i < size_; i++) {
      min_cost_flow_.SetNodeSupply(i, 1);
    }
    min_cost_flow_.Reset();
  }

  void _reset() noexcept override {
    side_.reset();
    demand_.reset();
    cost_.reset();
    min_cost_flow_.Reset();
  }

 private:
  mutable std::unique_ptr<bool[]> side_;
  mutable std::unique_ptr<int64_t[]> demand_;
  mutable std::unique_ptr<int64_t[]> cost_;
  mutable operations_research::SimpleMinCostFlow min_cost_flow_;
  mutable std::mutex lock_;
};


/// Used by emd() to convert the problem to min cost flow.
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
    if (fabs(sum - MASS_MULT + 0.) / MASS_MULT > 0.000001) {
#ifndef NDEBUG
      assert(sum == MASS_MULT && "Masses on one side not sufficiently normalized.");
#else
      fprintf(stderr,
              "wmd: weights are not normalized: %" PRId64 " != %" PRId64 "\n",
              sum, MASS_MULT);
#endif
    } else {
      // compensate for the rounding error
      out[0] += (sign ? 1 : -1) * (sum - MASS_MULT);
    }
  }
}


/// Used by emd() to convert the problem to min cost flow.
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


/// Solves the exact EMD problem. Internally, it converts the conditions to
/// a min cost flow statement and calls operations_research::SimpleMinCostFlow.
/// @param w1 The first array with weights of length `size`.
/// @param w2 The second array with weights of length `size`.
/// @param dist The costs matrix of shape `size` x `size`.
/// @param size The dimensionality of the problem.
/// @param cache The cache to use. It should be initialized with at least `size`
///              elements.
/// @author Wojciech Jabłoński <wj359634@students.mimuw.edu.pl>
template <typename T>
T emd(const T*__restrict__ w1, const T*__restrict__ w2,
      const T*__restrict__ dist, uint32_t size, const EMDCache& cache) {
  assert(w1 && w2 && dist);
  assert(size > 0);
  std::lock_guard<std::mutex> _(cache.enter(size));
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
