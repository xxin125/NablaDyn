// The grid: bins at least one list radius wide, and the atoms sorted by bin, by atom id
// within a bin so that every build is deterministic.

#pragma once

#include "common/common.cuh"

namespace nabladyn::neighbors {

template <typename scalar_t>
__device__ __forceinline__ index_t axis_bin(scalar_t x, scalar_t width, index_t count)
{
    // x is in [0, L); the quotient can still round up to count at the far face.
    const index_t bin = static_cast<index_t>(floor(x / width));
    return bin < 0 ? 0 : (bin >= count ? count - 1 : bin);
}

__device__ __forceinline__ index_t flat_bin(index_t x, index_t y, index_t z, const index_t* dims)
{
    return x + dims[0] * (y + dims[1] * z);
}

template <typename scalar_t>
__global__ void assign_bins_kernel(
    const scalar_t* __restrict__ positions,
    const index_t* __restrict__ dims,
    const scalar_t* __restrict__ widths,
    index_t num_atoms,
    index_t* __restrict__ atom_bin,
    index_t* __restrict__ bin_counts,
    const index_t* __restrict__ rebuild,
    const index_t* __restrict__ status
)
{
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (atom >= num_atoms || !rebuilding(rebuild, status))
    {
        return;
    }
    const scalar_t* p = positions + 3 * atom;
    const index_t bin = flat_bin(
        axis_bin(p[0], widths[0], dims[0]),
        axis_bin(p[1], widths[1], dims[1]),
        axis_bin(p[2], widths[2], dims[2]),
        dims
    );
    atom_bin[atom] = bin;
    atomicAdd(bin_counts + bin, 1);
}

// Places each atom in its bin's slice, in the arbitrary order of the atomics. The counts
// return to zero, ready for the next build.
__global__ void scatter_atoms_kernel(
    const index_t* __restrict__ atom_bin,
    const index_t* __restrict__ bin_offsets,
    index_t num_atoms,
    index_t* __restrict__ bin_counts,
    index_t* __restrict__ unordered_ids,
    const index_t* __restrict__ rebuild,
    const index_t* __restrict__ status
)
{
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (atom >= num_atoms || !rebuilding(rebuild, status))
    {
        return;
    }
    const index_t bin = atom_bin[atom];
    unordered_ids[bin_offsets[bin] + atomicSub(bin_counts + bin, 1) - 1] =
        static_cast<index_t>(atom);
}

// Orders each bin's slice by atom id: every atom counts the smaller ids in its bin and moves
// to that rank.
__global__ void rank_atoms_kernel(
    const index_t* __restrict__ unordered_ids,
    const index_t* __restrict__ atom_bin,
    const index_t* __restrict__ bin_offsets,
    index_t num_atoms,
    index_t* __restrict__ sorted_ids,
    const index_t* __restrict__ rebuild,
    const index_t* __restrict__ status
)
{
    const std::int64_t slot = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= num_atoms || !rebuilding(rebuild, status))
    {
        return;
    }
    const index_t id = unordered_ids[slot];
    const index_t bin = atom_bin[id];
    const index_t begin = bin_offsets[bin];
    const index_t end = bin_offsets[bin + 1];
    index_t rank = 0;
    for (index_t other = begin; other < end; ++other)
    {
        rank += unordered_ids[other] < id;
    }
    sorted_ids[begin + rank] = id;
}

template <typename scalar_t>
__global__ void gather_positions_kernel(
    const scalar_t* __restrict__ positions,
    const index_t* __restrict__ sorted_ids,
    index_t num_atoms,
    scalar_t* __restrict__ sorted_positions,
    const index_t* __restrict__ rebuild,
    const index_t* __restrict__ status
)
{
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= 3LL * num_atoms || !rebuilding(rebuild, status))
    {
        return;
    }
    sorted_positions[index] = positions[3LL * sorted_ids[index / 3] + index % 3];
}

}  // namespace nabladyn::neighbors
