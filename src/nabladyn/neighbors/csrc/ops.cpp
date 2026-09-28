// Registers the neighbor operators. A build bins and sorts the atoms (grid/), then counts,
// scans, fills and pads the pairs (pairs/, all_pairs.cuh on small grids); update/ gates a
// build on reuse. runtime/entry_points.cu, the one CUDA file, launches every kernel.

#include <torch/library.h>

#include <string>

#include "ops.h"

TORCH_LIBRARY(nabladyn_neighbors, m)
{
    const std::string buffers =
        "Tensor(a!) dims, Tensor(b!) widths, Tensor(c!) atom_bin, Tensor(d!) bin_counts, "
        "Tensor(e!) bin_offsets, Tensor(f!) unordered_ids, Tensor(g!) sorted_ids, "
        "Tensor(h!) sorted_positions, Tensor(i!) scan_temp, Tensor(j!) pair_counts, "
        "Tensor(k!) pair_offsets, Tensor(l!) pair_total, Tensor(m!) status, Tensor(n!) pairs, "
        "Tensor(t!) bits";
    m.def("scan_temp_bytes(Tensor reference, int items) -> int");
    const std::string first_frame =
        "(Tensor positions, Tensor cell, float radius, " + buffers + ") -> ()";
    m.def(("count_first_frame" + first_frame).c_str());
    m.def(("fill_first_frame" + first_frame).c_str());
    const std::string update_schema =
        "update(Tensor positions, Tensor cell, float cutoff, float radius, "
        "Tensor(o!) reference_positions, Tensor(p!) reference_cell, Tensor limits, "
        "Tensor(q!) reuse, Tensor(r!) rebuild, Tensor(s!) rebuild_count, " + buffers + ") -> ()";
    m.def(update_schema.c_str());
}

TORCH_LIBRARY_IMPL(nabladyn_neighbors, CUDA, m)
{
    m.impl("count_first_frame", &nabladyn::neighbors::count_first_frame);
    m.impl("fill_first_frame", &nabladyn::neighbors::fill_first_frame);
    m.impl("update", &nabladyn::neighbors::update);
}

TORCH_LIBRARY_IMPL(nabladyn_neighbors, CompositeExplicitAutograd, m)
{
    m.impl("scan_temp_bytes", &nabladyn::neighbors::scan_temp_bytes);
}
