// The all-pairs path, for a list whose first grid has at most three bins on every axis:
// every pair is tested once, and every pair is (i, j) with i < j.

#pragma once

#include "common/common.cuh"
#include "common/traversal.cuh"

namespace nabladyn::neighbors {

// With at most three bins per axis the 27 neighboring bins cover the whole box, so the grid
// is not built. One warp per atom tests every atom with a higher id, 32 at a time, and keeps
// each round's hits as one word of the atom's bit row; the fill reads the bits, so each
// distance is computed once.
constexpr int kWarp = 32;
constexpr unsigned int kFullWarp = 0xffffffffu;
// Atom i tests N - 1 - i others, so the first atoms carry the most work. Two warps per
// block spread those atoms over every multiprocessor.
constexpr int kAllPairsThreads = 2 * kWarp;

// Words per bit row: one bit for every atom.
__host__ __device__ __forceinline__ std::int64_t bit_words(index_t num_atoms)
{
    return (static_cast<std::int64_t>(num_atoms) + kWarp - 1) / kWarp;
}

// Tests each atom against every atom with a higher id, stores the hits and counts them. Row
// words before the atom's first candidate are neither written nor read.
template <typename scalar_t>
__global__ void test_all_pairs_kernel(
    const scalar_t* __restrict__ positions,
    const scalar_t* __restrict__ cell,
    scalar_t radius_squared,
    index_t num_atoms,
    std::uint32_t* __restrict__ bits,
    index_t* __restrict__ pair_counts,
    const index_t* __restrict__ rebuild,
    const index_t* __restrict__ status
)
{
    const std::int64_t thread = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t center = thread / kWarp;
    if (center >= num_atoms || !rebuilding(rebuild, status))
    {
        return;  // the whole warp returns together
    }
    const int lane = static_cast<int>(thread % kWarp);
    const std::int64_t words = bit_words(num_atoms);
    const scalar_t cx = positions[3 * center];
    const scalar_t cy = positions[3 * center + 1];
    const scalar_t cz = positions[3 * center + 2];
    index_t count = 0;
    for (std::int64_t word = (center + 1) / kWarp; word < words; ++word)
    {
        const std::int64_t neighbor = word * kWarp + lane;
        bool hit = false;
        if (neighbor > center && neighbor < num_atoms)
        {
            const scalar_t* p = positions + 3 * neighbor;
            const scalar_t dx = minimum_image(cx, p[0], cell[0]);
            const scalar_t dy = minimum_image(cy, p[1], cell[1]);
            const scalar_t dz = minimum_image(cz, p[2], cell[2]);
            hit = dx * dx + dy * dy + dz * dz < radius_squared;
        }
        const unsigned int hits = __ballot_sync(kFullWarp, hit);
        if (lane == 0)
        {
            bits[center * words + word] = hits;
        }
        count += __popc(hits);
    }
    if (lane == 0)
    {
        pair_counts[center] = count;
    }
}

// Writes each atom's pairs from its bit row, from its first column on, each word's hits
// in lane order. Columns at or past the capacity are counted but not written.
__global__ void fill_all_pairs_kernel(
    const std::uint32_t* __restrict__ bits,
    const index_t* __restrict__ pair_offsets,
    index_t num_atoms,
    index_t capacity,
    index_t* __restrict__ pairs,
    const index_t* __restrict__ rebuild,
    const index_t* __restrict__ status
)
{
    const std::int64_t thread = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t center = thread / kWarp;
    if (center >= num_atoms || !rebuilding(rebuild, status))
    {
        return;
    }
    const int lane = static_cast<int>(thread % kWarp);
    const unsigned int lanes_before = (1u << lane) - 1u;
    const std::int64_t words = bit_words(num_atoms);
    std::int64_t column = pair_offsets[center];
    for (std::int64_t word = (center + 1) / kWarp; word < words; ++word)
    {
        const unsigned int hits = bits[center * words + word];
        if ((hits >> lane) & 1u)
        {
            const std::int64_t slot = column + __popc(hits & lanes_before);
            if (slot < capacity)
            {
                pairs[slot] = static_cast<index_t>(center);
                pairs[capacity + slot] = static_cast<index_t>(word * kWarp + lane);
            }
        }
        column += __popc(hits);
    }
}

}  // namespace nabladyn::neighbors
