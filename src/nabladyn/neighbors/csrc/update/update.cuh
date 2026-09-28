// The update: check the cell, decide whether the pairs can be reused, and after a rebuild
// store the frame it was built from.

#pragma once

#include "common/common.cuh"
#include "common/traversal.cuh"

namespace nabladyn::neighbors {

// Every kernel runs at every update, so one CUDA Graph holds the whole update; a device flag
// decides whether the rebuild kernels do any work.
//
// Reuse rule. A rebuild stores the wrapped positions x0 and the cell L0. For positions x in
// the cell L, with s = L / L0 per axis, atom i has moved d_i = minimum image of x_i - s x0_i.
// A pair now closer than the cutoff was closer than (cutoff + 2 max |d_i|) / min(s) at the
// rebuild, so the pairs are reused while max |d_i| < (R min(s) - cutoff) / 2, less a
// rounding guard, where R is the list radius.

// Entries of the reuse parameters: the squared threshold, then s per axis.
constexpr int kThresholdSquared = 0;
constexpr int kScale = 1;

// One thread: checks the cell, chooses the grid for it (at most `limits`, the first grid's
// bins, each wider than the list radius) and sets the reuse threshold. A list with an
// error stays as it is until it is allocated again.
template <typename scalar_t>
__global__ void prepare_update_kernel(
    const scalar_t* __restrict__ cell,
    const scalar_t* __restrict__ reference_cell,
    scalar_t cutoff,
    scalar_t radius,
    scalar_t epsilon,
    const index_t* __restrict__ limits,
    index_t* __restrict__ dims,
    scalar_t* __restrict__ widths,
    scalar_t* __restrict__ reuse,
    index_t* __restrict__ rebuild,
    index_t* __restrict__ status
)
{
    rebuild[0] = 0;
    if (status[0] != 0)
    {
        return;
    }
    scalar_t smallest_scale = cell[0] / reference_cell[0];
    scalar_t longest = cell[0];
    for (int axis = 0; axis < 3; ++axis)
    {
        const scalar_t length = cell[axis];
        if (!(isfinite(length) && length > 2 * cutoff))
        {
            status[0] |= kInvalidCell;
            return;
        }
        const scalar_t scale = length / reference_cell[axis];
        reuse[kScale + axis] = scale;
        smallest_scale = scale < smallest_scale ? scale : smallest_scale;
        longest = length > longest ? length : longest;
        // Binning rounds, so each bin keeps a margin of a few roundings of L.
        const scalar_t fit = floor(length / (radius + 4 * epsilon * length));
        const index_t count =
            fit < 1 ? 1 : (fit > limits[axis] ? limits[axis] : static_cast<index_t>(fit));
        dims[axis] = count;
        widths[axis] = length / count;
    }
    const scalar_t allowance = radius * smallest_scale - cutoff - 32 * epsilon * longest;
    if (allowance <= 0)
    {
        rebuild[0] = 1;
        return;
    }
    reuse[kThresholdSquared] = allowance * allowance / 4;
}

// Asks for a rebuild when an atom has moved past the threshold, and flags a coordinate
// that is not finite.
template <typename scalar_t>
__global__ void check_reuse_kernel(
    const scalar_t* __restrict__ positions,
    const scalar_t* __restrict__ reference_positions,
    const scalar_t* __restrict__ cell,
    const scalar_t* __restrict__ reuse,
    index_t num_atoms,
    index_t* __restrict__ rebuild,
    index_t* __restrict__ status
)
{
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (atom >= num_atoms || status[0] != 0)
    {
        return;
    }
    const scalar_t* x = positions + 3 * atom;
    const scalar_t* x0 = reference_positions + 3 * atom;
    if (!(isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2])))
    {
        atomicOr(status, kNonFinite);
        return;
    }
    if (rebuild[0] != 0)
    {
        return;
    }
    scalar_t moved = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        const scalar_t d = minimum_image(x[axis], reuse[kScale + axis] * x0[axis], cell[axis]);
        moved += d * d;
    }
    if (moved >= reuse[kThresholdSquared])
    {
        rebuild[0] = 1;
    }
}

// After a rebuild, stores the frame it was built from and counts the rebuild.
template <typename scalar_t>
__global__ void commit_rebuild_kernel(
    const scalar_t* __restrict__ positions,
    const scalar_t* __restrict__ cell,
    index_t num_atoms,
    scalar_t* __restrict__ reference_positions,
    scalar_t* __restrict__ reference_cell,
    index_t* __restrict__ rebuild_count,
    const index_t* __restrict__ rebuild,
    const index_t* __restrict__ status
)
{
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= 3LL * num_atoms || !rebuilding(rebuild, status))
    {
        return;
    }
    reference_positions[index] = positions[index];
    if (index < 3)
    {
        reference_cell[index] = cell[index];
    }
    if (index == 0)
    {
        ++rebuild_count[0];
    }
}

}  // namespace nabladyn::neighbors
