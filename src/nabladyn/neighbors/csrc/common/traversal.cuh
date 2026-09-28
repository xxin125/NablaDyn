// The neighbor visit: which bins a center searches, and each candidate's periodic image.

#pragma once

#include "common/common.cuh"

namespace nabladyn::neighbors {

// The difference center - (other + shift) along one axis, shift in {-L, 0, L} per bin. The
// coordinate near the far face moves by L first, which is exact, so the difference is rounded
// once: (center - L) - other for shift = L, center - (other - L) for shift = -L.
template <typename scalar_t>
struct AxisImage
{
    scalar_t ahead;
    scalar_t behind;

    __device__ __forceinline__ AxisImage(scalar_t center, scalar_t shift)
        : ahead(shift > scalar_t{0} ? center - shift : center),
          behind(shift < scalar_t{0} ? shift : scalar_t{0})
    {
    }

    __device__ __forceinline__ scalar_t difference(scalar_t other) const
    {
        return ahead - (other + behind);
    }
};

// The same difference with the image chosen per candidate: the minimum image.
template <typename scalar_t>
__device__ __forceinline__ scalar_t minimum_image(scalar_t center, scalar_t other, scalar_t length)
{
    const scalar_t raw = center - other;
    return raw > scalar_t{0.5} * length ? (center - length) - other
         : raw < scalar_t{-0.5} * length ? center - (other - length)
         : raw;
}

// The three bins around the home bin along one axis, and the image shift of their atoms.
__device__ __forceinline__ index_t neighbor_bin(index_t home, int step, index_t count)
{
    const index_t bin = home + step;
    return bin < 0 ? bin + count : (bin >= count ? bin - count : bin);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t neighbor_shift(
    index_t home,
    int step,
    index_t count,
    scalar_t length
)
{
    const index_t bin = home + step;
    return bin < 0 ? -length : (bin >= count ? length : scalar_t{0});
}

// Calls visit(neighbor) for every atom after sorted atom `center` in bin order that lies
// closer than the list radius. With three or more bins per axis the center visits its 27
// neighboring bins, each with a fixed image. Otherwise it visits every bin along an axis
// with fewer than three bins, and each candidate takes its minimum image.
template <typename scalar_t, typename Visit>
__device__ __forceinline__ void visit_neighbors(
    const scalar_t* __restrict__ sorted_positions,
    const index_t* __restrict__ sorted_ids,
    const index_t* __restrict__ atom_bin,
    const index_t* __restrict__ bin_offsets,
    const index_t* __restrict__ dims,
    const scalar_t* __restrict__ cell,
    scalar_t radius_squared,
    index_t center,
    Visit& visit
)
{
    const scalar_t cx = sorted_positions[3LL * center];
    const scalar_t cy = sorted_positions[3LL * center + 1];
    const scalar_t cz = sorted_positions[3LL * center + 2];
    const index_t nx = dims[0], ny = dims[1], nz = dims[2];
    const index_t home = atom_bin[sorted_ids[center]];
    const index_t hx = home % nx, hy = (home / nx) % ny, hz = home / (nx * ny);
    if (nx >= 3 && ny >= 3 && nz >= 3)
    {
        for (int sz = -1; sz <= 1; ++sz)
        {
            const AxisImage<scalar_t> iz(cz, neighbor_shift(hz, sz, nz, cell[2]));
            const index_t bz = neighbor_bin(hz, sz, nz);
            for (int sy = -1; sy <= 1; ++sy)
            {
                const AxisImage<scalar_t> iy(cy, neighbor_shift(hy, sy, ny, cell[1]));
                const index_t by = neighbor_bin(hy, sy, ny);
                for (int sx = -1; sx <= 1; ++sx)
                {
                    const AxisImage<scalar_t> ix(cx, neighbor_shift(hx, sx, nx, cell[0]));
                    const index_t bin = neighbor_bin(hx, sx, nx) + nx * (by + ny * bz);
                    const index_t first = bin_offsets[bin];
                    const index_t end = bin_offsets[bin + 1];
                    for (index_t neighbor = first > center ? first : center + 1;
                         neighbor < end;
                         ++neighbor)
                    {
                        const scalar_t* p = sorted_positions + 3LL * neighbor;
                        const scalar_t dx = ix.difference(p[0]);
                        const scalar_t dy = iy.difference(p[1]);
                        const scalar_t dz = iz.difference(p[2]);
                        if (dx * dx + dy * dy + dz * dz < radius_squared)
                        {
                            visit(neighbor);
                        }
                    }
                }
            }
        }
        return;
    }
    // Along an axis with three or more bins only the three around home can hold a
    // neighbor; along a shorter axis every bin can.
    const int span_x = nx >= 3 ? 3 : nx, span_y = ny >= 3 ? 3 : ny, span_z = nz >= 3 ? 3 : nz;
    for (int iz = 0; iz < span_z; ++iz)
    {
        const index_t bz = nz >= 3 ? neighbor_bin(hz, iz - 1, nz) : iz;
        for (int iy = 0; iy < span_y; ++iy)
        {
            const index_t by = ny >= 3 ? neighbor_bin(hy, iy - 1, ny) : iy;
            for (int ix = 0; ix < span_x; ++ix)
            {
                const index_t bin =
                    (nx >= 3 ? neighbor_bin(hx, ix - 1, nx) : ix) + nx * (by + ny * bz);
                const index_t first = bin_offsets[bin];
                const index_t end = bin_offsets[bin + 1];
                for (index_t neighbor = first > center ? first : center + 1;
                     neighbor < end;
                     ++neighbor)
                {
                    const scalar_t* p = sorted_positions + 3LL * neighbor;
                    const scalar_t dx = minimum_image(cx, p[0], cell[0]);
                    const scalar_t dy = minimum_image(cy, p[1], cell[1]);
                    const scalar_t dz = minimum_image(cz, p[2], cell[2]);
                    if (dx * dx + dy * dy + dz * dz < radius_squared)
                    {
                        visit(neighbor);
                    }
                }
            }
        }
    }
}

}  // namespace nabladyn::neighbors
