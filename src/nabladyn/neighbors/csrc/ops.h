// The four neighbor operators: defined in runtime/entry_points.cu, registered in ops.cpp.

#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace nabladyn::neighbors {

// The scratch bytes of an exclusive scan over `items` int32 values.
std::int64_t scan_temp_bytes(
    const at::Tensor& reference,
    std::int64_t items
);

// allocate, part one: the grid and each atom's pair count, so the host can size the pairs.
void count_first_frame(
    const at::Tensor& positions,
    const at::Tensor& cell,
    double radius,
    at::Tensor& dims,
    at::Tensor& widths,
    at::Tensor& atom_bin,
    at::Tensor& bin_counts,
    at::Tensor& bin_offsets,
    at::Tensor& unordered_ids,
    at::Tensor& sorted_ids,
    at::Tensor& sorted_positions,
    at::Tensor& scan_temp,
    at::Tensor& pair_counts,
    at::Tensor& pair_offsets,
    at::Tensor& pair_total,
    at::Tensor& status,
    at::Tensor& pairs,
    at::Tensor& bits
);

// allocate, part two: the pairs, into the capacity the host chose.
void fill_first_frame(
    const at::Tensor& positions,
    const at::Tensor& cell,
    double radius,
    at::Tensor& dims,
    at::Tensor& widths,
    at::Tensor& atom_bin,
    at::Tensor& bin_counts,
    at::Tensor& bin_offsets,
    at::Tensor& unordered_ids,
    at::Tensor& sorted_ids,
    at::Tensor& sorted_positions,
    at::Tensor& scan_temp,
    at::Tensor& pair_counts,
    at::Tensor& pair_offsets,
    at::Tensor& pair_total,
    at::Tensor& status,
    at::Tensor& pairs,
    at::Tensor& bits
);

// The update: check reuse, rebuild behind a device flag, store the new reference. It never
// syncs or allocates, so a CUDA Graph can capture it.
void update(
    const at::Tensor& positions,
    const at::Tensor& cell,
    double cutoff,
    double radius,
    at::Tensor& reference_positions,
    at::Tensor& reference_cell,
    const at::Tensor& limits,
    at::Tensor& reuse,
    at::Tensor& rebuild,
    at::Tensor& rebuild_count,
    at::Tensor& dims,
    at::Tensor& widths,
    at::Tensor& atom_bin,
    at::Tensor& bin_counts,
    at::Tensor& bin_offsets,
    at::Tensor& unordered_ids,
    at::Tensor& sorted_ids,
    at::Tensor& sorted_positions,
    at::Tensor& scan_temp,
    at::Tensor& pair_counts,
    at::Tensor& pair_offsets,
    at::Tensor& pair_total,
    at::Tensor& status,
    at::Tensor& pairs,
    at::Tensor& bits
);

}  // namespace nabladyn::neighbors
