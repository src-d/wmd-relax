#include <cstdint>
#include <cmath>
#include <algorithm>
#include "graph/min_cost_flow.h"




/// @author Wojciech Jabłoński <wj359634@students.mimuw.edu.pl>

namespace {


using LL = int64_t;
const LL MASS_MULT  = 1000*1000*1000;   // weights quantization constant
const LL COST_MULT  = 1000*1000;        // costs quantization constant


struct buffer_t {
  bool* side;
  LL*   demand;
  LL*   cost;

  buffer_t()
  : side(nullptr)
  , demand(nullptr)
  , cost(nullptr)
  , size(0)
  {}

  size_t get_size() const {
    return size;
  }

  void allocate(size_t s) {
    if (s == 0) {
      throw "Can't allocate empty buffer!";
    }
    if (size != 0) {
      throw "You have to deallocate the buffer first before allocating again!";
    }
    size = s;
    side    = new bool[size];
    demand  = new LL[size];
    cost    = new LL[size * size];
  }

  void deallocate() {
    if (size == 0) {
      throw "DO NOT free already empty buffer!";
    }
    size = 0;
  }

  private:
  size_t size;

  buffer_t(const buffer_t&) = delete;
  buffer_t(const buffer_t&&) = delete;
  buffer_t& operator=(const buffer_t&) = delete;
  buffer_t& operator=(const buffer_t&&) = delete;
};


template <typename T>
void convert_weights(
  const T* in,
  const bool sid,
  LL* out,
  size_t size
) {
  LL sum = 0;
  double old_s = 0, new_s = 0;
  for (size_t i = 0; i < size; i++) {
    old_s = new_s;
    new_s = old_s + in[i];
    LL w = round(new_s * MASS_MULT) - round(old_s * MASS_MULT);
    sum += w;
    out[i] += w * (sid ? -1 : 1);
  }
  if (sum != MASS_MULT) {
    throw "masses on one side not sufficiently normalized!";
  }
}


template <typename T>
void convert_costs(
  const T* in,
  const bool* side,
  LL* out,
  const size_t size
) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      out[i*size + j] = round(in[i*size + j]*COST_MULT);
    }
  }

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if (side[i] && !side[j])
        out[i * size + j] = -out[j * size + i];
    }
  }
}


}   // namespace



template <typename T>
T emd(
  const T* w1,
  const T* w2,
  const T* dist,
  const buffer_t& buffer,
  uint32_t size
) {
  if (buffer.get_size() < size)
    throw "Buffer not big enough!";
  bool* side    = buffer.side;
  LL*   demand  = buffer.demand;
  LL*   cost    = buffer.cost;

  for (size_t i = 0; i < size; i++) {
    demand[i] = 0;
  }

  convert_weights<T>(w1, 0, demand, size);
  convert_weights<T>(w2, 1, demand, size);

  for (size_t i = 0; i < size; i++) {
    side[i] = (demand[i] < 0);
  }

  convert_costs<T>(dist, side, cost, size);

  operations_research::SimpleMinCostFlow min_cost_flow;
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if (!side[i] && side[j]) {
        min_cost_flow.AddArcWithCapacityAndUnitCost(
          i,
          j,
          std::min(demand[i], -demand[j]),
          cost[i*size + j]);
      }
    }
  }

  for (size_t i = 0; i < size; i++) {
    min_cost_flow.SetNodeSupply(i, demand[i]);
  }

  min_cost_flow.Solve();
  double result = min_cost_flow.OptimalCost();

  return T((result/MASS_MULT)/COST_MULT);
}
