#pragma once
#include "utils.h"
#include "namespace_config.h"
#include <cute/tensor.hpp>

namespace FLASH_NAMESPACE {
using namespace cute;

// Cooperative load: 2KB bitset from global to shared via uint4 (128-bit)
__forceinline__ __device__ void load_bitset_to_smem(
    uint32_t *__restrict__ smem_bits,           // [512] in shared mem
    const uint32_t *__restrict__ global_bits,   // [512] in global mem
    const int tidx) {
    // 2048 bytes / 16 bytes per uint4 = 128 loads
    // Kernel has 128 or 256 threads; use first 128

    #ifdef DEBUG
    assert(reinterpret_cast<uintptr_t>(smem_bits) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(global_bits) % 16 == 0);
    #endif
    
    uint4 *dst = reinterpret_cast<uint4 *>(smem_bits);
    const uint4 *src = reinterpret_cast<const uint4 *>(global_bits);
    if (tidx < 128) {
        dst[tidx] = src[tidx];
    }
}

// Check single bit: row in [0, kBlockM), col in [0, kBlockN)
__forceinline__ __device__ bool has_edge_bit(
    const uint32_t *smem_bits, int row, int col, int kBlockN) {
    int bit_idx = row * kBlockN + col;
    return (smem_bits[bit_idx >> 5] >> (bit_idx & 31)) & 1u;
}

// Apply edge bias to acc_s (MMA=4, MMA_M, MMA_N) format
// Follows mask.h Mask::apply_mask pattern (lines 166-207)
template <int kBlockN, typename Engine, typename Layout>
__forceinline__ __device__ void apply_edge_bias(
    Tensor<Engine, Layout> &tensor_,
    const uint32_t *__restrict__ smem_bits,
    const float bias_val,
    const int col_idx_offset_,
    const int row_idx_offset,
    const int warp_row_stride) {
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    // Reshape from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    Tensor tensor = make_tensor(tensor_.data(),
        FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    if (has_edge_bit(smem_bits, row_idx, col_idx, kBlockN)) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) += bias_val;
                    }
                }
            }
        }
    }
}

} // namespace FLASH_NAMESPACE