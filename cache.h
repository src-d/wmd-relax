#ifndef WMDRELAX_CACHE_H_
#define WMDRELAX_CACHE_H_

#include <cstdio>
#include <mutex>

namespace wmd {

/// This is supposed to be the base class for all the other caches.
/// "Cache" here means the carrier of reusable buffers which should eliminate
/// memory allocations. It should be used as follows:
///
/// Cache instance;
/// instance.allocate(100500);
/// // thread safety
/// {
///   // the problem size is 100
///   std::lock_guard<std::mutex> _(instance.enter(100));
///   auto whatever = instance.whatever();
///   // ... use whatever ...
/// }
class Cache {
 public:
  enum AllocationError {
    kAllocationErrorSuccess = 0,
    /// Can't allocate empty cache.
    kAllocationErrorInvalidSize,
    /// You have to deallocate the cache first before allocating again.
    kAllocationErrorDeallocationNeeded
  };

  Cache() : size_(0) {}
  virtual ~Cache() {}

  AllocationError allocate(size_t size) {
    if (size == 0) {
      return kAllocationErrorInvalidSize;
    }
    if (size_ != 0) {
      return kAllocationErrorDeallocationNeeded;
    }
    size_ = size;
    _allocate();
    return kAllocationErrorSuccess;
  }

  void reset() noexcept {
    _reset();
    size_ = 0;
  }

  std::mutex& enter(size_t size) const {
#ifndef NDEBUG
  assert(size_ >= size && "the cache is too small");
#else
  if (size_ < size) {
    fprintf(stderr, "emd: cache size is too small: %zu < %zu\n",
            size_, size);
    throw "the cache is too small";
  }
#endif
    return lock_;
  }

 protected:
  virtual void _allocate() = 0;
  virtual void _reset() noexcept = 0;

  size_t size_;

 private:
  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;

  mutable std::mutex lock_;
};

}

#endif  // WMDRELAX_CACHE_H_
