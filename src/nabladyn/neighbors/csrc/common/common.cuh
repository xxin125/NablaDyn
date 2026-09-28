// Shared types, status bits, launch sizes and the rebuild gate of the neighbor kernels.

#pragma once

#include <cstdint>

namespace nabladyn::neighbors {

using index_t = std::int32_t;

constexpr int kThreads = 256;

// Status bits. They stay set until the list is allocated again.
constexpr index_t kInvalidCell = 1;  // a cell length is not finite or at most twice the cutoff
constexpr index_t kNonFinite = 2;    // a coordinate is not finite
constexpr index_t kOverflow = 8;     // the pairs exceed the capacity

inline unsigned int blocks_for(std::int64_t items)
{
    return static_cast<unsigned int>((items + kThreads - 1) / kThreads);
}

// Whether a rebuild kernel runs. A build from allocate passes no flag and always runs; an
// update runs it only when the reuse check asked for a rebuild and no error is set.
__device__ __forceinline__ bool rebuilding(const index_t* rebuild, const index_t* status)
{
    return rebuild == nullptr || (rebuild[0] != 0 && status[0] == 0);
}

}  // namespace nabladyn::neighbors
