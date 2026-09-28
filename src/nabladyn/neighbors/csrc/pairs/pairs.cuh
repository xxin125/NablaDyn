// The pair kernels: count each atom's pairs, fill them from its first column, and pad.
// Each atom lists the neighbors after it in bin order, so each pair is found once.

#pragma once

#include "common/common.cuh"
#include "common/traversal.cuh"

namespace nabladyn::neighbors {

// Counts the pairs a center lists.
struct CountPairs
{
    index_t count;

    __device__ __forceinline__ void operator()(index_t)
    {
        ++count;
    }
};

// Writes (center, neighbor) as atom ids from the center's first column; columns past the
// capacity are counted but not written.
struct WritePairs
{
    const index_t* __restrict__ sorted_ids;
    index_t center_id;
    std::int64_t column;
    std::int64_t capacity;
    index_t* __restrict__ pairs;

    __device__ __forceinline__ void operator()(index_t neighbor)
    {
        if (column < capacity)
        {
            pairs[column] = center_id;
            pairs[capacity + column] = sorted_ids[neighbor];
        }
        ++column;
    }
};

template <typename scalar_t>
__global__ void count_pairs_kernel(
    const scalar_t* __restrict__ sorted_positions,
    const index_t* __restrict__ sorted_ids,
    const index_t* __restrict__ atom_bin,
    const index_t* __restrict__ bin_offsets,
    const index_t* __restrict__ dims,
    const scalar_t* __restrict__ cell,
    scalar_t radius_squared,
    index_t num_atoms,
    index_t* __restrict__ pair_counts,
    const index_t* __restrict__ rebuild,
    const index_t* __restrict__ status
)
{
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= num_atoms || !rebuilding(rebuild, status))
    {
        return;
    }
    CountPairs visit{0};
    visit_neighbors(
        sorted_positions,
        sorted_ids,
        atom_bin,
        bin_offsets,
        dims,
        cell,
        radius_squared,
        static_cast<index_t>(center),
        visit
    );
    pair_counts[center] = visit.count;
}

template <typename scalar_t>
__global__ void fill_pairs_kernel(
    const scalar_t* __restrict__ sorted_positions,
    const index_t* __restrict__ sorted_ids,
    const index_t* __restrict__ atom_bin,
    const index_t* __restrict__ bin_offsets,
    const index_t* __restrict__ dims,
    const scalar_t* __restrict__ cell,
    const index_t* __restrict__ pair_offsets,
    scalar_t radius_squared,
    index_t num_atoms,
    index_t capacity,
    index_t* __restrict__ pairs,
    const index_t* __restrict__ rebuild,
    const index_t* __restrict__ status
)
{
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= num_atoms || !rebuilding(rebuild, status))
    {
        return;
    }
    WritePairs visit{sorted_ids, sorted_ids[center], pair_offsets[center], capacity, pairs};
    visit_neighbors(
        sorted_positions,
        sorted_ids,
        atom_bin,
        bin_offsets,
        dims,
        cell,
        radius_squared,
        static_cast<index_t>(center),
        visit
    );
}

// One thread: the pair total and, above the capacity, the overflow bit and the pairs needed.
__global__ void total_pairs_kernel(
    const index_t* __restrict__ pair_counts,
    const index_t* __restrict__ pair_offsets,
    index_t num_atoms,
    index_t capacity,
    index_t* __restrict__ pair_total,
    index_t* __restrict__ status,
    const index_t* __restrict__ rebuild
)
{
    if (!rebuilding(rebuild, status))
    {
        return;
    }
    const index_t total = pair_offsets[num_atoms - 1] + pair_counts[num_atoms - 1];
    pair_total[0] = total;
    if (total > capacity)
    {
        status[0] |= kOverflow;
        status[1] = total > status[1] ? total : status[1];
    }
}

// The columns past the pairs are padding: column k holds the self-pair (k mod N, k mod N).
// Spreading them over the atoms keeps their zero gradients from piling onto one atom.
__global__ void pad_pairs_kernel(
    const index_t* __restrict__ pair_total,
    index_t num_atoms,
    index_t capacity,
    index_t* __restrict__ pairs,
    const index_t* __restrict__ rebuild,
    const index_t* __restrict__ status
)
{
    const std::int64_t column = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= capacity || column < pair_total[0] || !rebuilding(rebuild, status))
    {
        return;
    }
    const index_t atom = static_cast<index_t>(column % num_atoms);
    pairs[column] = atom;
    pairs[capacity + column] = atom;
}

}  // namespace nabladyn::neighbors
