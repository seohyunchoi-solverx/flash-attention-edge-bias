import torch
import numpy as np
from typing import Tuple


def build_edge_bias_bitset(
    edge_index: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sparse tile-based edge bias structures for flash_attn_varlen_func.

    Converts a sparse edge list into CSR-indexed 128x128 bitset tiles, plus a
    2D dense tile_map for O(1) kernel lookup.

    Args:
        edge_index: [2, E] int64 tensor in packed-tensor coordinates.
            edge_index[0] = source positions (Q side),
            edge_index[1] = destination positions (K side).
        cu_seqlens: [B+1] int32, cumulative sequence lengths (same for Q and K
            in self-attention).
        max_seqlen: Maximum sequence length in the batch.  Used to compute
            ``max_k_blocks = ceil(max_seqlen / block_size)``.
        block_size: Tile dimension, must equal kernel's kBlockM == kBlockN.
            Default 128.

    Returns:
        tile_offsets:   [total_q_blocks + 1] int32  — CSR row pointers by q_block
        tile_k_indices: [nnz_tiles] int32           — global k_block per tile
        bitsets:        [nnz_tiles, 512] int32       — 128x128 bit matrix (512 words)
        tile_map:       [total_q_blocks, max_k_blocks] int32
                        — 2D dense lookup: tile_map[q_block_global, k_block_local]
                          = tile_idx (-1 means no tile).
                        Kernel indexes as:
                          ``tile_map[q_block_global * max_k_blocks + n_block]``
        cu_q_blocks:    [B+1] int32 — cumulative Q block counts
        cu_k_blocks:    [B+1] int32 — cumulative K block counts

    Bit layout per tile:
        ``bit_idx = row_in_tile * block_size + col_in_tile``
        stored at ``bitsets[tile, bit_idx // 32]``, bit ``bit_idx % 32``.
    """
    device = edge_index.device
    edge_cpu = edge_index.cpu().to(torch.int64)
    cu_cpu = cu_seqlens.cpu().to(torch.int64)

    batch_size = cu_cpu.numel() - 1
    seqlens = cu_cpu[1:] - cu_cpu[:-1]
    n_blocks_per_seq = (seqlens + block_size - 1) // block_size

    cu_blocks = torch.zeros(batch_size + 1, dtype=torch.int64)
    cu_blocks[1:] = torch.cumsum(n_blocks_per_seq, dim=0)
    total_q_blocks = int(cu_blocks[-1].item())
    max_k_blocks = (max_seqlen + block_size - 1) // block_size

    n_edges = edge_cpu.size(1) if edge_cpu.dim() == 2 else 0

    cu_blocks_i32 = cu_blocks.to(torch.int32)
    if n_edges == 0 or total_q_blocks == 0:
        return (
            torch.zeros(total_q_blocks + 1, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, 512, dtype=torch.int32, device=device),
            torch.full((total_q_blocks, max(max_k_blocks, 1)), -1, dtype=torch.int32, device=device),
            cu_blocks_i32.to(device),
            cu_blocks_i32.clone().to(device),
        )

    src = edge_cpu[0]                                        # [E]
    dst = edge_cpu[1]                                        # [E]

    batch_idx = torch.searchsorted(cu_cpu, src, right=True) - 1
    seq_start = cu_cpu[batch_idx]
    local_src = src - seq_start
    local_dst = dst - seq_start

    q_blk_local = local_src // block_size
    k_blk_local = local_dst // block_size
    q_blk_g = cu_blocks[batch_idx] + q_blk_local             # global q_block
    k_blk_g = cu_blocks[batch_idx] + k_blk_local             # global k_block

    row_in_tile = local_src % block_size
    col_in_tile = local_dst % block_size
    bit_idx = row_in_tile * block_size + col_in_tile
    word_idx = (bit_idx // 32).numpy().astype(np.intp)
    bit_pos = (bit_idx % 32).numpy().astype(np.int32)

    # --- unique tiles, sorted by (q_block_global, k_block_global) ---
    tile_keys = q_blk_g * (total_q_blocks + 1) + k_blk_g
    unique_keys, inverse = torch.unique(tile_keys, sorted=True, return_inverse=True)
    nnz = int(unique_keys.numel())
    tile_q = (unique_keys // (total_q_blocks + 1)).to(torch.int32)   # [nnz]
    tile_k = (unique_keys % (total_q_blocks + 1)).to(torch.int32)    # [nnz]

    # --- bitsets via numpy unbuffered bitwise-or ---
    bitsets_np = np.zeros((nnz, 512), dtype=np.int32)
    inv_np = inverse.numpy().astype(np.intp)
    bit_val = np.left_shift(np.int32(1), bit_pos)
    np.bitwise_or.at(bitsets_np, (inv_np, word_idx), bit_val)
    bitsets = torch.from_numpy(bitsets_np)

    # --- CSR row pointers keyed by q_block_global ---
    tile_offsets = torch.zeros(total_q_blocks + 1, dtype=torch.int32)
    counts = torch.zeros(total_q_blocks, dtype=torch.int32)
    counts.scatter_add_(0, tile_q.long(), torch.ones(nnz, dtype=torch.int32))
    tile_offsets[1:] = counts.cumsum(0).to(torch.int32)

    # --- 2D tile_map: [total_q_blocks, max_k_blocks] ---
    # Indexed by (q_block_global, k_block_local_within_sequence)
    # Recover each tile's local k_block from its global k_block
    batch_of_tile = torch.searchsorted(cu_blocks, tile_q.long(), right=True) - 1
    k_blk_local_of_tile = tile_k.long() - cu_blocks[batch_of_tile]

    tile_map = torch.full((total_q_blocks, max_k_blocks), -1, dtype=torch.int32)
    tile_map[tile_q.long(), k_blk_local_of_tile] = torch.arange(nnz, dtype=torch.int32)

    return (
        tile_offsets.to(device),
        tile_k.to(device),
        bitsets.to(device),
        tile_map.to(device),
        cu_blocks_i32.to(device),
        cu_blocks_i32.clone().to(device),
    )
