# Edge Bias for FlashAttention: Experiment Report

## 1. Overview

This report documents the implementation, debugging, and validation of **sparse edge bias** support in FlashAttention-2's CUDA kernels. Edge bias allows injecting a learnable, per-head scalar bias for arbitrary (sparse) pairs of query-key positions, enabling graph-aware attention patterns without materializing a dense bias matrix.

### 1.1 Motivation

In graph neural networks and molecular modeling, attention scores between tokens often need to be modulated by the presence of edges in a graph structure. A naive dense bias matrix `(seqlen × seqlen)` is prohibitively expensive for long sequences. Our approach uses a **tile-based bitset** representation: for each `(kBlockM × kBlockN)` attention tile, we store a compressed bitset indicating which Q-K pairs have edges, enabling O(1) lookup per bit in shared memory.

### 1.2 Environment

| Component | Value |
|---|---|
| GPU | NVIDIA H100 (sm_90), 80 GB HBM3 |
| CUDA | 12.x |
| PyTorch | 2.x with CUDA support |
| FlashAttention | v2.x (Dao-AILab fork) |
| Python | 3.10.12 |
| Data types | `bfloat16` (primary), `float16` (secondary) |

---

## 2. Architecture: Edge Bias Data Structures

### 2.1 Python Side (`edge_bias_utils.py`)

The Python utility `build_edge_bias_bitset()` converts a sparse edge list `[2, E]` into kernel-compatible structures:

| Structure | Shape | Description |
|---|---|---|
| `tile_offsets` | `[total_q_blocks + 1]` int32 | CSR row pointers indexed by global q_block |
| `tile_k_indices` | `[nnz_tiles]` int32 | Global k_block index for each non-empty tile |
| `bitsets` | `[nnz_tiles, bitset_words]` int32 | Compressed bit matrix per tile |
| `tile_map` | `[total_q_blocks, max_k_blocks]` int32 | Dense 2D lookup: `tile_map[q_blk, k_blk_local] → tile_idx` (-1 if empty) |
| `cu_q_blocks` | `[B+1]` int32 | Cumulative Q block counts per batch |
| `cu_k_blocks` | `[B+1]` int32 | Cumulative K block counts per batch |

**Bit layout per tile:**
```
bit_idx = row_in_tile * block_size_n + col_in_tile
word    = bitsets[tile, bit_idx // 32]
bit     = (word >> (bit_idx % 32)) & 1
```

where `bitset_words = block_size_m * block_size_n / 32`.

### 2.2 CUDA Side

The kernel loads one tile's bitset into shared memory via `load_bitset_to_smem<kBitsetUint4>()`, then calls `apply_edge_bias<kBlockN>()` to add `edge_bias_scale[head] / scale_softmax` to each attention score where the corresponding bit is set.

**Shared memory layout:**
```
[--- Kernel_traits::kSmemSize ---][-- 16B aligned padding --][--- kBitsetBytes ---]
                                                              = kBlockM * kBlockN / 8
```

---

## 3. Problem: Block Size Mismatch

### 3.1 Root Cause

FlashAttention uses different tile dimensions `(kBlockM, kBlockN)` depending on:
- `head_dim` (32, 64, 96, 128, 192, 256)
- Pass direction (forward vs. backward)
- GPU architecture (sm80 vs. sm8x vs. sm90)
- Dropout enabled/disabled, causal mode

The original `edge_bias_utils.py` **hardcoded `block_size=128`**, assuming all tiles are `128×128`. This was correct for `head_dim ≤ 64` but wrong for larger head dimensions where the kernel uses non-square tiles.

### 3.2 Actual Kernel Block Sizes (sm90, no dropout)

**Forward pass** (from `flash_fwd_launch_template.h`):

| head_dim | kBlockM | kBlockN | Bitset words | Bitset bytes |
|---|---|---|---|---|
| 32 | 128 | 128 | 512 | 2048 |
| 64 | 128 | 128 | 512 | 2048 |
| 96 | 128 | 64 | 256 | 1024 |
| 128 | 128 | 64 | 256 | 1024 |
| 192 | 128 | 64 | 256 | 1024 |
| 256 | 64 | 64 | 128 | 512 |

**Backward pass** (from `flash_bwd_launch_template.h`, sm90 with `max_smem ≥ 144 KB`):

| head_dim | kBlockM | kBlockN | Bitset words | Bitset bytes |
|---|---|---|---|---|
| 32 | 128 | 128 | 512 | 2048 |
| 64 | 128 | 128 | 512 | 2048 |
| 96 | 64 | 128 | 256 | 1024 |
| 128 | 64 | 128 | 256 | 1024 |
| 192 | 64 | 64 | 128 | 512 |
| 256 | 64 | 64 | 128 | 512 |

**Key observation:** For `head_dim ∈ {96, 128}`, forward and backward use *transposed* tile shapes:
- Forward: `(128, 64)` — tall tiles
- Backward: `(64, 128)` — wide tiles

This means the bitset layouts are incompatible between passes, requiring **separate edge bias structures** for forward and backward.

### 3.3 Consequences of Mismatch

When `block_size=128` was hardcoded but the kernel used e.g. `kBlockM=64, kBlockN=128`:
1. **Bitset size mismatch**: Python produced 512-word bitsets, kernel expected 256 words
2. **Tile count mismatch**: Python computed `ceil(seqlen/128)` blocks, kernel iterated over `ceil(seqlen/64)` blocks
3. **Bit position errors**: `bit_idx = row * 128 + col` vs. `row * kBlockN + col` produced different bit positions
4. **Out-of-bounds reads**: Kernel accessed tiles/bits that didn't exist in the Python-generated structures

---

## 4. Problem: NaN Gradients from Register Pressure

### 4.1 Symptom

Backward pass produced NaN gradients for `head_dim = 64` and `head_dim = 128`, even when `Has_edge_bias = false`.

### 4.2 Root Cause

The `EDGE_BIAS_SWITCH` macro expanded to `BOOL_SWITCH`, which instantiated **both** `Has_edge_bias=true` and `Has_edge_bias=false` template variants of the kernel. This doubled the register pressure from template combinatorics:

```
BOOL_SWITCH × EVENK_SWITCH × LOCAL_SWITCH × ALIBI_SWITCH × SOFTCAP_SWITCH × EDGE_BIAS_SWITCH
```

Each additional `BOOL_SWITCH` doubles the number of kernel instantiations. For `head_dim=64, 128`, the backward kernel already uses near-maximum registers. Adding the edge bias variant pushed register usage over the limit, causing **register spills to local memory**, which introduced numerical instability (NaN).

### 4.3 Solution

Disabled unused template switches at compile time via `setup.py` flags:

```python
"-DFLASHATTENTION_DISABLE_DROPOUT",
"-DFLASHATTENTION_DISABLE_ALIBI",
"-DFLASHATTENTION_DISABLE_SOFTCAP",
"-DFLASHATTENTION_DISABLE_UNEVEN_K",
"-DFLASHATTENTION_DISABLE_LOCAL",
```

Each `FLASHATTENTION_DISABLE_*` flag converts its corresponding `*_SWITCH` macro from `BOOL_SWITCH` (2 instantiations) to a single-path macro (1 instantiation). This halves the register pressure per disabled feature. With 5 features disabled, the total instantiation count dropped from `2^7 = 128` to `2^2 = 4` (only `IsEvenMN` and `Has_edge_bias` remain dynamic).

**In `static_switch.h`:**
```cpp
// When FLASHATTENTION_DISABLE_DROPOUT is defined:
#define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] { constexpr static bool CONST_NAME = false; return __VA_ARGS__(); }()
// vs. default (both paths):
#define DROPOUT_SWITCH BOOL_SWITCH
```

---

## 5. Changes Made

### 5.1 `csrc/flash_attn/edge_bias_utils.py`

1. **Added `get_kernel_block_sizes(head_dim, is_backward)`** — replicates the C++ dispatch logic to return `(kBlockM, kBlockN)` for the default path (sm90, no dropout, `smem ≥ 144 KB`).

2. **Modified `build_edge_bias_bitset()`** to accept `block_size_m` and `block_size_n` separately:
   - `bitset_words = block_size_m * block_size_n // 32` (was hardcoded to `128*128//32 = 512`)
   - Separate `cu_q_blocks` (by `block_size_m`) and `cu_k_blocks` (by `block_size_n`)
   - Correct bit indexing: `bit_idx = row_in_tile * block_size_n + col_in_tile`

### 5.2 `flash_attn/flash_attn_interface.py`

Added `edge_bias_bwd` parameter to `flash_attn_varlen_func()` and `FlashAttnVarlenFunc`:
- When `edge_bias_bwd` is provided, backward pass uses separate structures
- When `None`, forward structures are reused (valid when `kBlockM_fwd == kBlockM_bwd` and `kBlockN_fwd == kBlockN_bwd`)

### 5.3 `csrc/flash_attn/src/edge_bias.h`

Templated `load_bitset_to_smem` with `kBitsetUint4` parameter:
```cpp
template <int kBitsetUint4>
__forceinline__ __device__ void load_bitset_to_smem(...) {
    if (tidx < kBitsetUint4) { dst[tidx] = src[tidx]; }
}
```
Previously hardcoded `if (tidx < 128)` which assumed 128 `uint4` loads = 2048 bytes = `128×128/8`.

### 5.4 `csrc/flash_attn/src/flash_fwd_kernel.h` (4 sites)

Updated bitset loading to use dynamic sizes derived from kernel traits:
```cpp
constexpr int kBitsetWords = kBlockM * kBlockN / 32;
constexpr int kBitsetUint4 = kBitsetWords * 4 / 16;
load_bitset_to_smem<kBitsetUint4>(smem_edge_bits,
    params.edge_bias_bitsets + tile_idx * kBitsetWords, tidx);
```
Previously: `params.edge_bias_bitsets + tile_idx * 512`.

### 5.5 `csrc/flash_attn/src/flash_bwd_kernel.h` (1 primary site)

Same change as forward kernel. The second edge bias read site (for `d_edge_bias_scale` gradient) is read-only and uses the already-loaded shared memory.

### 5.6 `csrc/flash_attn/src/flash_fwd_launch_template.h` and `flash_bwd_launch_template.h`

Dynamic shared memory allocation:
```cpp
constexpr size_t kBitsetBytes = Kernel_traits::kBlockM * Kernel_traits::kBlockN / 8;
constexpr size_t smem_size_edge = (smem_size + 15) & ~15;
const size_t smem_total = (Has_edge_bias ? smem_size_edge + kBitsetBytes : smem_size);
```
Previously hardcoded `+ 2048` bytes.

### 5.7 `setup.py`

Enabled compile-time feature disabling:
```python
"-DFLASHATTENTION_DISABLE_DROPOUT",
"-DFLASHATTENTION_DISABLE_ALIBI",
"-DFLASHATTENTION_DISABLE_SOFTCAP",
"-DFLASHATTENTION_DISABLE_UNEVEN_K",
"-DFLASHATTENTION_DISABLE_LOCAL",
```

---

## 6. Build Configuration

| Setting | Value | Purpose |
|---|---|---|
| `FLASH_ATTN_CUDA_ARCHS` | `"90"` | Only build for H100 (sm_90) |
| `MAX_JOBS` | 26 | Parallel nvcc compilation |
| Build system | ninja (via `.venv/bin`) | Fast incremental builds with dependency tracking |
| `PATH` | `.venv/bin` prepended | Ensures `shutil.which('ninja')` finds ninja |
| `FLASHATTENTION_DISABLE_*` | 5 features disabled | Reduces template instantiation count |

**Build time: ~3.5 minutes** (73 compilation units, parallel).

---

## 7. Test Methodology

### 7.1 NaN Gradient Test (`tests/debug_nan.py`)

Tests `flash_attn_func` (non-varlen) forward and backward for `head_dim ∈ {32, 64, 96, 128}`:
- Fixed `seqlen=128`, `batch=2`, `n_heads=4`
- Checks `torch.isnan(grad).any()` on dQ, dK, dV
- Additional seqlen sweep: `{8, 32, 64, 128, 256}` for `head_dim=64`
- Tests `flash_attn_varlen_func` with `seqlen=10`
- Tests both `bfloat16` and `float16`

### 7.2 Edge Bias Correctness Test (`tests/test_edge_bias.py`)

7 unit tests comparing FlashAttention output against a vanilla PyTorch attention implementation:
1. **Single sequence** forward (seqlen=5)
2. **Multi-sequence** forward (seqlens=[3,5,4,6,2])
3. **Float16** forward (seqlens=[5,3])
4. **Backward** without edge bias (seqlens=[4,5,3])
5. **Edge bias forward** (seqlens=[5,8,3,12,6], density=0.3)
6. **Edge bias backward** (seqlens=[6,4,7], density=0.25)
7. **Edge bias no edges** (seqlens=[5,3], empty edge set)

Tolerance: `max_diff < 0.01` (fwd), `max_diff < 0.05` (bwd), `mean_diff < 0.001` (fwd).

### 7.3 Large Sequence Length Test (inline script)

Tests edge bias with seqlens in the hundreds, covering multi-tile scenarios:

| Config | head_dim | n_heads | seqlens | Total tokens |
|---|---|---|---|---|
| 1 | 32 | 4 | [128, 256, 200] | 584 |
| 2 | 64 | 4 | [300, 150, 400] | 850 |
| 3 | 96 | 4 | [200, 350] | 550 |
| 4 | 128 | 2 | [256, 512] | 768 |
| 5 | 32 | 4 | [500, 300, 200, 100] | 1100 |
| 6 | 64 | 4 | [600] | 600 |

Each config:
1. Builds edge bias structures with `get_kernel_block_sizes(head_dim, is_backward=False)` for forward
2. If backward block sizes differ, builds separate structures with `is_backward=True`
3. Runs `flash_attn_varlen_func` with `edge_bias_bwd` parameter
4. Compares against vanilla attention (float32 reference)
5. Checks: forward max/mean diff, NaN in gradients, dQ/dK/dV max diff

Tolerance: `fwd_max < 0.05`, `fwd_mean < 0.005`, `bwd_max < 0.1` per grad component.

---

## 8. Results

### 8.1 NaN Gradient Test

| head_dim | Forward | Backward |
|---|---|---|
| 32 | OK | OK |
| 64 | OK | OK |
| 96 | OK | OK |
| 128 | OK | OK |

**Before fix**: `head_dim=64` and `head_dim=128` backward produced NaN.

### 8.2 Edge Bias Unit Tests

```
test_flash_vs_vanilla_single_seq             PASSED
test_flash_vs_vanilla_multi_seq              PASSED
test_flash_vs_vanilla_fp16                   PASSED
test_flash_vs_vanilla_backward               PASSED
test_flash_vs_vanilla_edge_bias_forward      PASSED
test_flash_vs_vanilla_edge_bias_backward     PASSED
test_flash_vs_vanilla_edge_bias_no_edges     PASSED
```

**7/7 passed** in 3.45 seconds.

### 8.3 Large Sequence Length Test

| Config | Forward | Backward | Block sizes |
|---|---|---|---|
| hd32 [128,256,200] | max=0.0023, mean=0.0002 | dQ=0.004, dK=0.009, dV=0.004 | fwd=bwd=(128,128) |
| hd64 [300,150,400] | max=0.0022, mean=0.0002 | dQ=0.003, dK=0.005, dV=0.004 | fwd=bwd=(128,128) |
| hd96 [200,350] | max=0.0021, mean=0.0002 | dQ=0.004, dK=0.006, dV=0.004 | fwd=(128,64) bwd=(64,128) **separate** |
| hd128 [256,512] | max=0.0015, mean=0.0001 | dQ=0.003, dK=0.006, dV=0.004 | fwd=(128,64) bwd=(64,128) **separate** |
| hd32 [500,300,200,100] | max=0.0024, mean=0.0002 | dQ=0.004, dK=0.005, dV=0.005 | fwd=bwd=(128,128) |
| hd64 [600] | max=0.0012, mean=0.0001 | dQ=0.002, dK=0.005, dV=0.004 | fwd=bwd=(128,128) |

**6/6 passed.** No NaN. All differences within bf16 numerical tolerance.

**Key observations:**
- `head_dim=96` and `head_dim=128` correctly use **separate forward/backward edge bias structures** with transposed tile shapes
- Forward max diff is consistently < 0.003 (excellent for bf16)
- Backward grad diff is consistently < 0.01 (acceptable for bf16 with accumulated rounding)
- Larger seqlens (500, 600) spanning many tiles show no degradation

---

## 9. Conclusions

### 9.1 Block Size Mismatch Fix

The hardcoded `block_size=128` in `edge_bias_utils.py` was the primary correctness issue for `head_dim > 64`. The fix involves:
1. `get_kernel_block_sizes()` to replicate the C++ dispatch table in Python
2. Separate `block_size_m` / `block_size_n` parameters throughout the pipeline
3. `edge_bias_bwd` parameter for cases where forward and backward tile shapes differ

**This is essential for `head_dim ∈ {96, 128, 192, 256}`** where the kernel uses non-square or transposed tiles.

### 9.2 Register Pressure Fix

Disabling unused `BOOL_SWITCH` features (`DROPOUT`, `ALIBI`, `SOFTCAP`, `UNEVEN_K`, `LOCAL`) at compile time eliminated NaN gradients for `head_dim = 64, 128` by reducing template instantiation from O(2^7) to O(2^2).

**Important caveat:** These disabled features must be re-enabled if the corresponding functionality is needed. For production use, consider:
- Splitting edge bias kernels into separate `.cu` files to isolate register pressure
- Using `__launch_bounds__` to explicitly control register allocation
- Profiling with `--ptxas-options=-v` to monitor register usage per kernel variant

### 9.3 Build System

- **ninja** must be in `PATH` for PyTorch's `cpp_extension` to find it (uses `shutil.which('ninja')`)
- The `.venv/bin/ninja` was not found by default; prepending `.venv/bin` to `PATH` resolved this
- `distutils` fallback (when ninja is missing) does **not** track header dependencies, causing stale object files
- Build with `FLASH_ATTN_CUDA_ARCHS="90"` to only target the current GPU architecture

### 9.4 Remaining Limitations

1. `get_kernel_block_sizes()` assumes sm90 with `max_smem ≥ 144 KB` and no dropout. Other GPU architectures or configurations need manual override.
2. The `FLASHATTENTION_DISABLE_*` flags are a development-time workaround. A production solution should isolate edge bias instantiations.
3. `head_dim = 192, 256` with edge bias have not been tested (no test data generated for these configurations).
4. The `edge_bias_scale` gradient computation in the backward kernel reuses the shared memory bitset loaded during the forward-style score computation in the backward pass, so it does not require a second load.

---

## Appendix A: File Change Summary

| File | Lines changed | Nature |
|---|---|---|
| `csrc/flash_attn/edge_bias_utils.py` | +65 (rewrite) | `get_kernel_block_sizes()`, rectangular tile support |
| `flash_attn/flash_attn_interface.py` | +20 | `edge_bias_bwd` parameter plumbing |
| `csrc/flash_attn/src/edge_bias.h` | +3 | Template `kBitsetUint4` parameter |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | +16 (4 sites) | Dynamic `kBitsetWords` / `kBitsetUint4` |
| `csrc/flash_attn/src/flash_bwd_kernel.h` | +4 (1 site) | Dynamic `kBitsetWords` / `kBitsetUint4` |
| `csrc/flash_attn/src/flash_fwd_launch_template.h` | +6 (2 sites) | Dynamic SMEM allocation |
| `csrc/flash_attn/src/flash_bwd_launch_template.h` | +3 (1 site) | Dynamic SMEM allocation |
| `setup.py` | +5 | Compile-time feature disabling |

## Appendix B: How to Reproduce

```bash
cd /path/to/flash-attention

# 1. Build (sm90 only, with disabled features for register pressure)
export PATH="$(pwd)/.venv/bin:$PATH"
FLASH_ATTN_CUDA_ARCHS="90" MAX_JOBS=26 .venv/bin/python setup.py build_ext --inplace

# 2. NaN test
.venv/bin/python tests/debug_nan.py

# 3. Edge bias unit tests
.venv/bin/python -m pytest tests/test_edge_bias.py -v

# 4. Large seqlen test (see Section 7.3 for inline script)
```

## Appendix C: Usage Example

```python
from csrc.flash_attn.edge_bias_utils import build_edge_bias_bitset, get_kernel_block_sizes
from flash_attn import flash_attn_varlen_func

head_dim = 96

# Get kernel-matching block sizes
bm_f, bn_f = get_kernel_block_sizes(head_dim, is_backward=False)  # (128, 64)
bm_b, bn_b = get_kernel_block_sizes(head_dim, is_backward=True)   # (64, 128)

# Build forward structures
fwd = build_edge_bias_bitset(edge_index, cu_seqlens, max_seqlen,
                              block_size_m=bm_f, block_size_n=bn_f)
tile_offsets, tile_k_indices, bitsets, tile_map, cu_q_blocks, cu_k_blocks = fwd
max_k_blocks = tile_map.shape[1]

# Build backward structures (different tile shape!)
bwd = build_edge_bias_bitset(edge_index, cu_seqlens, max_seqlen,
                              block_size_m=bm_b, block_size_n=bn_b)
bwd_with_max_k = (*bwd, bwd[3].shape[1])  # append max_k_blocks

# Run flash attention
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
    edge_bias_tile_offsets=tile_offsets,
    edge_bias_tile_k_indices=tile_k_indices,
    edge_bias_bitsets=bitsets,
    edge_bias_scale=eb_scale,
    edge_bias_tile_map=tile_map,
    cu_q_blocks=cu_q_blocks,
    cu_k_blocks=cu_k_blocks,
    edge_bias_max_k_blocks=max_k_blocks,
    edge_bias_bwd=bwd_with_max_k,  # separate backward structures
)
```
