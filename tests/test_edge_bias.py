"""
Unit tests for flash_attn_varlen_func with edge bias.

Tests:
  - Baseline tests (no edge bias): verify flash matches vanilla attention.
  - Edge bias forward test: verify flash + edge bias matches vanilla + edge bias.
  - Edge bias backward test: verify gradients (dQ, dK, dV, d_edge_bias_scale).
"""

import sys
import os
import random

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc", "flash_attn"))
from edge_bias_utils import build_edge_bias_bitset


def vanilla_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    edge_index: torch.Tensor = None,
    edge_bias_scale: torch.Tensor = None,
) -> torch.Tensor:
    """
    Vanilla (non-flash) attention for variable-length sequences,
    with optional per-pair edge bias.

    Args:
        q, k, v: (total_tokens, n_heads, head_dim)  — should be float32
        cu_seqlens: (n_seqs + 1,) int32
        max_seqlen: max sequence length
        softmax_scale: 1/sqrt(head_dim) typically
        edge_index: [2, E] int64 in packed-tensor coordinates (optional)
        edge_bias_scale: [n_heads] float32 (optional, requires edge_index)

    Returns:
        out: (total_tokens, n_heads, head_dim)
    """
    n_seqs = cu_seqlens.shape[0] - 1
    n_heads = q.shape[1]
    out = torch.zeros_like(q)

    for i in range(n_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seqlen = end - start

        qi = q[start:end].permute(1, 0, 2)   # (n_heads, seqlen, head_dim)
        ki = k[start:end].permute(1, 0, 2)
        vi = v[start:end].permute(1, 0, 2)

        scores = torch.matmul(qi, ki.transpose(-2, -1)) * softmax_scale

        if edge_index is not None and edge_bias_scale is not None:
            mask = (edge_index[0] >= start) & (edge_index[0] < end)
            seq_src = edge_index[0, mask] - start
            seq_dst = edge_index[1, mask] - start
            bias_matrix = torch.zeros(seqlen, seqlen, device=q.device, dtype=q.dtype)
            bias_matrix[seq_src, seq_dst] = 1.0
            # scores shape: (n_heads, seqlen, seqlen)
            # edge_bias_scale shape: (n_heads,)
            scores = scores + bias_matrix.unsqueeze(0) * edge_bias_scale[:, None, None]

        attn = F.softmax(scores, dim=-1)
        oi = torch.matmul(attn, vi)

        out[start:end] = oi.permute(1, 0, 2)

    return out


def make_random_edges(cu_seqlens, density=0.3, seed=123):
    """Generate random edges within each sequence (packed-tensor coordinates).

    Returns:
        edge_index: [2, E] int64 on CPU
    """
    rng = random.Random(seed)
    src_list, dst_list = [], []

    n_seqs = cu_seqlens.shape[0] - 1
    for i in range(n_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seqlen = end - start
        n_edges = max(1, int(seqlen * seqlen * density))
        for _ in range(n_edges):
            r = rng.randint(0, seqlen - 1)
            c = rng.randint(0, seqlen - 1)
            src_list.append(start + r)
            dst_list.append(start + c)

    return torch.tensor([src_list, dst_list], dtype=torch.int64)


def make_toy_data(
    seqlens: list[int],
    n_heads: int = 4,
    head_dim: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Create toy Q, K, V data and cu_seqlens for flash_attn_varlen_func.

    Args:
        seqlens: list of sequence lengths, e.g. [3, 5, 4]
        n_heads: number of attention heads
        head_dim: dimension per head
        dtype: bf16 or fp16
        device: cuda device

    Returns:
        q, k, v: (total_tokens, n_heads, head_dim)
        cu_seqlens: (n_seqs + 1,) int32
        max_seqlen: int
    """
    torch.manual_seed(seed)
    total_tokens = sum(seqlens)
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, n_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, n_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, n_heads, head_dim, device=device, dtype=dtype)

    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seqlens), dim=0)),
        device=device,
        dtype=torch.int32,
    )

    return q, k, v, cu_seqlens, max_seqlen


def test_flash_vs_vanilla_single_seq():
    """Single short sequence: verify flash matches vanilla."""
    q, k, v, cu_seqlens, max_seqlen = make_toy_data(seqlens=[5])
    scale = q.shape[-1] ** -0.5

    out_flash = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=False,
    )

    out_vanilla = vanilla_attention_varlen(
        q.float(), k.float(), v.float(), cu_seqlens, max_seqlen, scale,
    )

    out_flash_f32 = out_flash.float()
    out_vanilla_f32 = out_vanilla.float()

    max_diff = (out_flash_f32 - out_vanilla_f32).abs().max().item()
    mean_diff = (out_flash_f32 - out_vanilla_f32).abs().mean().item()

    print(f"[single_seq] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert max_diff < 1e-2, f"max_diff too large: {max_diff}"
    assert mean_diff < 1e-3, f"mean_diff too large: {mean_diff}"
    print("[single_seq] PASSED")


def test_flash_vs_vanilla_multi_seq():
    """Multiple sequences with varying lengths."""
    q, k, v, cu_seqlens, max_seqlen = make_toy_data(seqlens=[3, 5, 4, 6, 2])
    scale = q.shape[-1] ** -0.5

    out_flash = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=False,
    )

    out_vanilla = vanilla_attention_varlen(
        q.float(), k.float(), v.float(), cu_seqlens, max_seqlen, scale,
    )

    out_flash_f32 = out_flash.float()
    out_vanilla_f32 = out_vanilla.float()

    max_diff = (out_flash_f32 - out_vanilla_f32).abs().max().item()
    mean_diff = (out_flash_f32 - out_vanilla_f32).abs().mean().item()

    print(f"[multi_seq]  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert max_diff < 1e-2, f"max_diff too large: {max_diff}"
    assert mean_diff < 1e-3, f"mean_diff too large: {mean_diff}"

    # Also verify cross-sequence isolation: sequences should not attend to each other
    # The vanilla implementation handles this by construction (separate loops).
    # Flash handles it via cu_seqlens boundaries.
    print("[multi_seq]  PASSED")


def test_flash_vs_vanilla_fp16():
    """Same test with float16 instead of bfloat16."""
    q, k, v, cu_seqlens, max_seqlen = make_toy_data(
        seqlens=[5, 3], dtype=torch.float16,
    )
    scale = q.shape[-1] ** -0.5

    out_flash = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=False,
    )

    out_vanilla = vanilla_attention_varlen(
        q.float(), k.float(), v.float(), cu_seqlens, max_seqlen, scale,
    )

    max_diff = (out_flash.float() - out_vanilla.float()).abs().max().item()
    mean_diff = (out_flash.float() - out_vanilla.float()).abs().mean().item()

    print(f"[fp16]       max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert max_diff < 1e-2, f"max_diff too large: {max_diff}"
    assert mean_diff < 1e-3, f"mean_diff too large: {mean_diff}"
    print("[fp16]       PASSED")


def test_flash_vs_vanilla_backward():
    """Verify gradients match between flash and vanilla."""
    seqlens = [4, 5, 3]
    q, k, v, cu_seqlens, max_seqlen = make_toy_data(seqlens=seqlens)
    scale = q.shape[-1] ** -0.5

    q_flash, k_flash, v_flash = [x.clone().detach().requires_grad_(True) for x in (q, k, v)]
    q_vanilla, k_vanilla, v_vanilla = [x.clone().detach().float().requires_grad_(True) for x in (q, k, v)]

    out_flash = flash_attn_varlen_func(
        q_flash, k_flash, v_flash,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=False,
    )

    out_vanilla = vanilla_attention_varlen(
        q_vanilla, k_vanilla, v_vanilla, cu_seqlens, max_seqlen, scale,
    )

    loss_flash = out_flash.float().sum()
    loss_vanilla = out_vanilla.sum()

    loss_flash.backward()
    loss_vanilla.backward()

    for name, grad_flash, grad_vanilla in [
        ("dQ", q_flash.grad, q_vanilla.grad),
        ("dK", k_flash.grad, k_vanilla.grad),
        ("dV", v_flash.grad, v_vanilla.grad),
    ]:
        max_diff = (grad_flash.float() - grad_vanilla.float()).abs().max().item()
        mean_diff = (grad_flash.float() - grad_vanilla.float()).abs().mean().item()
        print(f"[backward {name}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 5e-2, f"{name} max_diff too large: {max_diff}"
        assert mean_diff < 1e-2, f"{name} mean_diff too large: {mean_diff}"

    print("[backward]   PASSED")


def test_flash_vs_vanilla_edge_bias_forward():
    """Forward: flash + edge bias must match vanilla + edge bias."""
    seqlens = [5, 8, 3, 12, 6]
    n_heads = 4
    head_dim = 32
    q, k, v, cu_seqlens, max_seqlen = make_toy_data(
        seqlens, n_heads=n_heads, head_dim=head_dim, seed=77,
    )
    scale = head_dim ** -0.5

    edge_index_cpu = make_random_edges(cu_seqlens, density=0.3, seed=42)
    edge_index = edge_index_cpu.to(q.device)

    eb_scale = torch.randn(n_heads, device="cuda", dtype=torch.float32)

    tile_offsets, tile_k_indices, bitsets, tile_map, cu_q_blocks, cu_k_blocks = \
        build_edge_bias_bitset(edge_index, cu_seqlens, max_seqlen)
    max_k_blocks = tile_map.shape[1]

    out_flash = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=False,
        edge_bias_tile_offsets=tile_offsets,
        edge_bias_tile_k_indices=tile_k_indices,
        edge_bias_bitsets=bitsets,
        edge_bias_scale=eb_scale,
        edge_bias_tile_map=tile_map,
        cu_q_blocks=cu_q_blocks,
        cu_k_blocks=cu_k_blocks,
        edge_bias_max_k_blocks=max_k_blocks,
    )

    out_vanilla = vanilla_attention_varlen(
        q.float(), k.float(), v.float(),
        cu_seqlens, max_seqlen, scale,
        edge_index=edge_index.long(),
        edge_bias_scale=eb_scale,
    )

    max_diff = (out_flash.float() - out_vanilla).abs().max().item()
    mean_diff = (out_flash.float() - out_vanilla).abs().mean().item()
    print(f"[edge_bias fwd]  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert max_diff < 1e-2, f"max_diff too large: {max_diff}"
    assert mean_diff < 1e-3, f"mean_diff too large: {mean_diff}"
    print("[edge_bias fwd]  PASSED")


def test_flash_vs_vanilla_edge_bias_backward():
    """Backward: verify dQ, dK, dV, d_edge_bias_scale match vanilla."""
    seqlens = [6, 4, 7]
    n_heads = 4
    head_dim = 32
    q, k, v, cu_seqlens, max_seqlen = make_toy_data(
        seqlens, n_heads=n_heads, head_dim=head_dim, seed=99,
    )
    scale = head_dim ** -0.5

    edge_index_cpu = make_random_edges(cu_seqlens, density=0.25, seed=55)
    edge_index = edge_index_cpu.to(q.device)

    tile_offsets, tile_k_indices, bitsets, tile_map, cu_q_blocks, cu_k_blocks = \
        build_edge_bias_bitset(edge_index, cu_seqlens, max_seqlen)
    max_k_blocks = tile_map.shape[1]

    # --- Flash path ---
    q_f = q.clone().detach().requires_grad_(True)
    k_f = k.clone().detach().requires_grad_(True)
    v_f = v.clone().detach().requires_grad_(True)
    eb_scale_f = torch.randn(n_heads, device="cuda", dtype=torch.float32, requires_grad=True)

    out_flash = flash_attn_varlen_func(
        q_f, k_f, v_f,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=False,
        edge_bias_tile_offsets=tile_offsets,
        edge_bias_tile_k_indices=tile_k_indices,
        edge_bias_bitsets=bitsets,
        edge_bias_scale=eb_scale_f,
        edge_bias_tile_map=tile_map,
        cu_q_blocks=cu_q_blocks,
        cu_k_blocks=cu_k_blocks,
        edge_bias_max_k_blocks=max_k_blocks,
    )
    loss_flash = out_flash.float().sum()
    loss_flash.backward()

    # --- Vanilla path ---
    q_v = q.clone().detach().float().requires_grad_(True)
    k_v = k.clone().detach().float().requires_grad_(True)
    v_v = v.clone().detach().float().requires_grad_(True)
    eb_scale_v = eb_scale_f.detach().clone().requires_grad_(True)

    out_vanilla = vanilla_attention_varlen(
        q_v, k_v, v_v,
        cu_seqlens, max_seqlen, scale,
        edge_index=edge_index.long(),
        edge_bias_scale=eb_scale_v,
    )
    loss_vanilla = out_vanilla.sum()
    loss_vanilla.backward()

    # --- Compare ---
    for name, gf, gv in [
        ("dQ", q_f.grad, q_v.grad),
        ("dK", k_f.grad, k_v.grad),
        ("dV", v_f.grad, v_v.grad),
    ]:
        max_diff = (gf.float() - gv.float()).abs().max().item()
        mean_diff = (gf.float() - gv.float()).abs().mean().item()
        print(f"[edge_bias bwd {name}]  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 5e-2, f"{name} max_diff too large: {max_diff}"
        assert mean_diff < 1e-2, f"{name} mean_diff too large: {mean_diff}"

    max_diff_scale = (eb_scale_f.grad - eb_scale_v.grad).abs().max().item()
    mean_diff_scale = (eb_scale_f.grad - eb_scale_v.grad).abs().mean().item()
    print(f"[edge_bias bwd dScale]  max_diff={max_diff_scale:.6f}, mean_diff={mean_diff_scale:.6f}")
    assert max_diff_scale < 5e-2, f"d_edge_bias_scale max_diff too large: {max_diff_scale}"

    print("[edge_bias bwd]  PASSED")


def test_flash_vs_vanilla_edge_bias_no_edges():
    """Edge bias with zero edges should match baseline (no bias)."""
    seqlens = [5, 3]
    n_heads = 4
    head_dim = 32
    q, k, v, cu_seqlens, max_seqlen = make_toy_data(
        seqlens, n_heads=n_heads, head_dim=head_dim, seed=11,
    )
    scale = head_dim ** -0.5

    edge_index = torch.zeros(2, 0, dtype=torch.int64, device="cuda")
    eb_scale = torch.randn(n_heads, device="cuda", dtype=torch.float32)

    tile_offsets, tile_k_indices, bitsets, tile_map, cu_q_blocks, cu_k_blocks = \
        build_edge_bias_bitset(edge_index, cu_seqlens, max_seqlen)
    max_k_blocks = tile_map.shape[1]

    out_with_bias = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=False,
        edge_bias_tile_offsets=tile_offsets,
        edge_bias_tile_k_indices=tile_k_indices,
        edge_bias_bitsets=bitsets,
        edge_bias_scale=eb_scale,
        edge_bias_tile_map=tile_map,
        cu_q_blocks=cu_q_blocks,
        cu_k_blocks=cu_k_blocks,
        edge_bias_max_k_blocks=max_k_blocks,
    )

    out_no_bias = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=False,
    )

    max_diff = (out_with_bias.float() - out_no_bias.float()).abs().max().item()
    print(f"[edge_bias no_edges]  max_diff={max_diff:.6f}")
    assert max_diff < 1e-6, f"Outputs differ with zero edges: {max_diff}"
    print("[edge_bias no_edges]  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Flash Attention varlen tests")
    print("=" * 60)
    print()

    print("--- Baseline tests (no edge bias) ---")
    test_flash_vs_vanilla_single_seq()
    print()
    test_flash_vs_vanilla_multi_seq()
    print()
    test_flash_vs_vanilla_fp16()
    print()
    test_flash_vs_vanilla_backward()
    print()

    print("--- Edge bias tests ---")
    test_flash_vs_vanilla_edge_bias_no_edges()
    print()
    test_flash_vs_vanilla_edge_bias_forward()
    print()
    test_flash_vs_vanilla_edge_bias_backward()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
