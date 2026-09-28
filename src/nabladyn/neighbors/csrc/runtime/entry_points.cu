// The host entry points of the neighbor operators: check the buffers and launch the kernels.
// This is the one CUDA translation unit, since the headers define non-template kernels.

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cub/device/device_scan.cuh>

#include <limits>
#include <utility>

#include "ops.h"
#include "common/common.cuh"
#include "grid/grid.cuh"
#include "pairs/all_pairs.cuh"
#include "pairs/pairs.cuh"
#include "update/update.cuh"

namespace nabladyn::neighbors {

void check(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    at::ScalarType dtype,
    const char* name
)
{
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == reference.device(),
        name,
        " must be on the positions' CUDA device"
    );
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
}

// The buffers of a list, as the operators receive them. bits is empty on the grid path.
// rebuild is null for a build from allocate, which always runs.
struct Buffers
{
    at::Tensor dims, widths, atom_bin, bin_counts, bin_offsets, unordered_ids, sorted_ids;
    at::Tensor sorted_positions, scan_temp, pair_counts, pair_offsets, pair_total, status, pairs;
    at::Tensor bits;
    const index_t* rebuild = nullptr;

    bool all_pairs() const
    {
        return bits.numel() > 0;
    }

    void check_against(const at::Tensor& positions) const
    {
        const auto dtype = positions.scalar_type();
        TORCH_CHECK(
            dtype == at::kFloat || dtype == at::kDouble,
            "positions must be float32 or float64"
        );
        TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "positions must be [N, 3]");
        const std::int64_t n = positions.size(0);
        check(positions, positions, dtype, "positions");
        check(widths, positions, dtype, "widths");
        check(sorted_positions, positions, dtype, "sorted_positions");
        const std::pair<const at::Tensor*, const char*> integers[] = {
            {&dims, "dims"},
            {&atom_bin, "atom_bin"},
            {&bin_counts, "bin_counts"},
            {&bin_offsets, "bin_offsets"},
            {&unordered_ids, "unordered_ids"},
            {&sorted_ids, "sorted_ids"},
            {&pair_counts, "pair_counts"},
            {&pair_offsets, "pair_offsets"},
            {&pair_total, "pair_total"},
            {&status, "status"},
            {&pairs, "pairs"},
            {&bits, "bits"},
        };
        for (const auto& [tensor, name] : integers)
        {
            check(*tensor, positions, at::kInt, name);
        }
        check(scan_temp, positions, at::kByte, "scan_temp");
        TORCH_CHECK(
            dims.numel() == 3 && widths.numel() == 3 && atom_bin.numel() == n &&
                unordered_ids.numel() == n && sorted_ids.numel() == n &&
                sorted_positions.sizes() == positions.sizes() &&
                bin_counts.numel() >= 2 && bin_offsets.numel() == bin_counts.numel() &&
                pair_counts.numel() == n && pair_offsets.numel() == n &&
                pair_total.numel() == 1 && status.numel() == 2,
            "the list's buffers do not match the positions"
        );
        TORCH_CHECK(
            pairs.dim() == 2 && pairs.size(0) == 2 && pairs.size(1) >= 1 &&
                pairs.size(1) <= INT32_MAX,
            "pairs must be [2, C] with int32 C"
        );
        TORCH_CHECK(
            bits.numel() == 0 || bits.numel() == n * bit_words(static_cast<index_t>(n)),
            "bits must be empty or hold N rows of ceil(N / 32) words"
        );
    }
};

void exclusive_scan(
    const at::Tensor& input,
    const at::Tensor& output,
    const at::Tensor& scan_temp,
    cudaStream_t stream
)
{
    std::size_t bytes = static_cast<std::size_t>(scan_temp.numel());
    C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_temp.data_ptr(),
        bytes,
        input.data_ptr<index_t>(),
        output.data_ptr<index_t>(),
        static_cast<index_t>(input.numel()),
        stream
    ));
}

// Bins, scans the bin counts, scatters, ranks and gathers. The scan is a library call
// and cannot be gated: when nothing rebuilds it scans counts that are all zero, and no
// kernel reads the result before the next rebuild writes it again.
template <typename scalar_t>
void launch_grid(const at::Tensor& positions, const Buffers& b, cudaStream_t stream)
{
    const auto n = static_cast<index_t>(positions.size(0));
    const index_t* status = b.status.data_ptr<index_t>();
    assign_bins_kernel<scalar_t><<<blocks_for(n), kThreads, 0, stream>>>(
        positions.data_ptr<scalar_t>(),
        b.dims.data_ptr<index_t>(),
        b.widths.data_ptr<scalar_t>(),
        n,
        b.atom_bin.data_ptr<index_t>(),
        b.bin_counts.data_ptr<index_t>(),
        b.rebuild,
        status
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    exclusive_scan(b.bin_counts, b.bin_offsets, b.scan_temp, stream);
    scatter_atoms_kernel<<<blocks_for(n), kThreads, 0, stream>>>(
        b.atom_bin.data_ptr<index_t>(),
        b.bin_offsets.data_ptr<index_t>(),
        n,
        b.bin_counts.data_ptr<index_t>(),
        b.unordered_ids.data_ptr<index_t>(),
        b.rebuild,
        status
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    rank_atoms_kernel<<<blocks_for(n), kThreads, 0, stream>>>(
        b.unordered_ids.data_ptr<index_t>(),
        b.atom_bin.data_ptr<index_t>(),
        b.bin_offsets.data_ptr<index_t>(),
        n,
        b.sorted_ids.data_ptr<index_t>(),
        b.rebuild,
        status
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gather_positions_kernel<scalar_t><<<blocks_for(3LL * n), kThreads, 0, stream>>>(
        positions.data_ptr<scalar_t>(),
        b.sorted_ids.data_ptr<index_t>(),
        n,
        b.sorted_positions.data_ptr<scalar_t>(),
        b.rebuild,
        status
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Counts each atom's pairs, on the list's path, and scans the counts into first columns.
// When nothing rebuilds, the scan repeats the last rebuild's result.
template <typename scalar_t>
void launch_count(
    const at::Tensor& positions,
    const at::Tensor& cell,
    double radius,
    const Buffers& b,
    cudaStream_t stream
)
{
    const auto n = static_cast<index_t>(b.sorted_ids.numel());
    const auto r = static_cast<scalar_t>(radius);
    if (b.all_pairs())
    {
        const auto blocks = static_cast<unsigned int>(n * kWarp / kAllPairsThreads + 1);
        test_all_pairs_kernel<scalar_t><<<blocks, kAllPairsThreads, 0, stream>>>(
            positions.data_ptr<scalar_t>(),
            cell.data_ptr<scalar_t>(),
            r * r,
            n,
            reinterpret_cast<std::uint32_t*>(b.bits.data_ptr<index_t>()),
            b.pair_counts.data_ptr<index_t>(),
            b.rebuild,
            b.status.data_ptr<index_t>()
        );
    }
    else
    {
        count_pairs_kernel<scalar_t><<<blocks_for(n), kThreads, 0, stream>>>(
            b.sorted_positions.data_ptr<scalar_t>(),
            b.sorted_ids.data_ptr<index_t>(),
            b.atom_bin.data_ptr<index_t>(),
            b.bin_offsets.data_ptr<index_t>(),
            b.dims.data_ptr<index_t>(),
            cell.data_ptr<scalar_t>(),
            r * r,
            n,
            b.pair_counts.data_ptr<index_t>(),
            b.rebuild,
            b.status.data_ptr<index_t>()
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    exclusive_scan(b.pair_counts, b.pair_offsets, b.scan_temp, stream);
}

// Fills the pairs, on the list's path, checks the total against the capacity, and pads
// the remaining columns.
template <typename scalar_t>
void launch_fill(const at::Tensor& cell, double radius, const Buffers& b, cudaStream_t stream)
{
    const auto n = static_cast<index_t>(b.sorted_ids.numel());
    const auto capacity = static_cast<index_t>(b.pairs.size(1));
    const auto r = static_cast<scalar_t>(radius);
    index_t* status = b.status.data_ptr<index_t>();
    if (b.all_pairs())
    {
        const auto blocks = static_cast<unsigned int>(n * kWarp / kAllPairsThreads + 1);
        fill_all_pairs_kernel<<<blocks, kAllPairsThreads, 0, stream>>>(
            reinterpret_cast<const std::uint32_t*>(b.bits.data_ptr<index_t>()),
            b.pair_offsets.data_ptr<index_t>(),
            n,
            capacity,
            b.pairs.data_ptr<index_t>(),
            b.rebuild,
            status
        );
    }
    else
    {
        fill_pairs_kernel<scalar_t><<<blocks_for(n), kThreads, 0, stream>>>(
            b.sorted_positions.data_ptr<scalar_t>(),
            b.sorted_ids.data_ptr<index_t>(),
            b.atom_bin.data_ptr<index_t>(),
            b.bin_offsets.data_ptr<index_t>(),
            b.dims.data_ptr<index_t>(),
            cell.data_ptr<scalar_t>(),
            b.pair_offsets.data_ptr<index_t>(),
            r * r,
            n,
            capacity,
            b.pairs.data_ptr<index_t>(),
            b.rebuild,
            status
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    total_pairs_kernel<<<1, 1, 0, stream>>>(
        b.pair_counts.data_ptr<index_t>(),
        b.pair_offsets.data_ptr<index_t>(),
        n,
        capacity,
        b.pair_total.data_ptr<index_t>(),
        status,
        b.rebuild
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    pad_pairs_kernel<<<blocks_for(capacity), kThreads, 0, stream>>>(
        b.pair_total.data_ptr<index_t>(),
        n,
        capacity,
        b.pairs.data_ptr<index_t>(),
        b.rebuild,
        status
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::int64_t scan_temp_bytes(
    const at::Tensor& reference,
    std::int64_t items
)
{
    TORCH_CHECK(items > 0 && items <= INT32_MAX, "the scan length must fit int32");
    const c10::cuda::CUDAGuard guard(reference.device());
    std::size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr,
        bytes,
        static_cast<const index_t*>(nullptr),
        static_cast<index_t*>(nullptr),
        static_cast<index_t>(items),
        c10::cuda::getCurrentCUDAStream()
    ));
    return static_cast<std::int64_t>(bytes);
}

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
)
{
    const Buffers b{
        dims,
        widths,
        atom_bin,
        bin_counts,
        bin_offsets,
        unordered_ids,
        sorted_ids,
        sorted_positions,
        scan_temp,
        pair_counts,
        pair_offsets,
        pair_total,
        status,
        pairs,
        bits,
    };
    b.check_against(positions);
    check(cell, positions, positions.scalar_type(), "cell");
    const c10::cuda::CUDAGuard guard(positions.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "count_first_frame", [&] {
        if (!b.all_pairs())
        {
            launch_grid<scalar_t>(positions, b, stream);
        }
        launch_count<scalar_t>(positions, cell, radius, b, stream);
    });
}

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
)
{
    const Buffers b{
        dims,
        widths,
        atom_bin,
        bin_counts,
        bin_offsets,
        unordered_ids,
        sorted_ids,
        sorted_positions,
        scan_temp,
        pair_counts,
        pair_offsets,
        pair_total,
        status,
        pairs,
        bits,
    };
    b.check_against(positions);
    check(cell, positions, positions.scalar_type(), "cell");
    const c10::cuda::CUDAGuard guard(positions.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "fill_first_frame", [&] {
        launch_fill<scalar_t>(cell, radius, b, stream);
    });
}

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
)
{
    const auto dtype = positions.scalar_type();
    Buffers b{
        dims,
        widths,
        atom_bin,
        bin_counts,
        bin_offsets,
        unordered_ids,
        sorted_ids,
        sorted_positions,
        scan_temp,
        pair_counts,
        pair_offsets,
        pair_total,
        status,
        pairs,
        bits,
    };
    b.check_against(positions);
    check(cell, positions, dtype, "cell");
    check(reference_positions, positions, dtype, "reference_positions");
    check(reference_cell, positions, dtype, "reference_cell");
    check(limits, positions, at::kInt, "limits");
    check(reuse, positions, dtype, "reuse");
    check(rebuild, positions, at::kInt, "rebuild");
    check(rebuild_count, positions, at::kInt, "rebuild_count");
    TORCH_CHECK(
        cell.numel() == 3 && reference_cell.numel() == 3 && limits.numel() == 3 &&
            reference_positions.sizes() == positions.sizes() && reuse.numel() == 4 &&
            rebuild.numel() == 1 && rebuild_count.numel() == 1,
        "the list's update buffers do not match the positions"
    );
    b.rebuild = rebuild.data_ptr<index_t>();
    const auto n = static_cast<index_t>(positions.size(0));
    const c10::cuda::CUDAGuard guard(positions.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(dtype, "update", [&] {
        prepare_update_kernel<scalar_t><<<1, 1, 0, stream>>>(
            cell.data_ptr<scalar_t>(),
            reference_cell.data_ptr<scalar_t>(),
            static_cast<scalar_t>(cutoff),
            static_cast<scalar_t>(radius),
            std::numeric_limits<scalar_t>::epsilon(),
            limits.data_ptr<index_t>(),
            dims.data_ptr<index_t>(),
            widths.data_ptr<scalar_t>(),
            reuse.data_ptr<scalar_t>(),
            rebuild.data_ptr<index_t>(),
            status.data_ptr<index_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        check_reuse_kernel<scalar_t><<<blocks_for(n), kThreads, 0, stream>>>(
            positions.data_ptr<scalar_t>(),
            reference_positions.data_ptr<scalar_t>(),
            cell.data_ptr<scalar_t>(),
            reuse.data_ptr<scalar_t>(),
            n,
            rebuild.data_ptr<index_t>(),
            status.data_ptr<index_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (!b.all_pairs())
        {
            launch_grid<scalar_t>(positions, b, stream);
        }
        launch_count<scalar_t>(positions, cell, radius, b, stream);
        launch_fill<scalar_t>(cell, radius, b, stream);
        commit_rebuild_kernel<scalar_t><<<blocks_for(3LL * n), kThreads, 0, stream>>>(
            positions.data_ptr<scalar_t>(),
            cell.data_ptr<scalar_t>(),
            n,
            reference_positions.data_ptr<scalar_t>(),
            reference_cell.data_ptr<scalar_t>(),
            rebuild_count.data_ptr<index_t>(),
            rebuild.data_ptr<index_t>(),
            status.data_ptr<index_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

}  // namespace nabladyn::neighbors
