"""
Benchmark: flash_attn_varlen_func with and without edge bias.

Generates a batch of sequences with total ~1M tokens, seqlens in [1000, 2000].
Measures forward-only and forward+backward wall-clock time using CUDA events.
"""

import sys
import os
import random
import torch

from flash_attn import flash_attn_varlen_func

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc", "flash_attn"))
from edge_bias_utils import build_edge_bias_bitset, get_kernel_block_sizes


def make_batch(target_tokens=1_000_000, min_sl=1000, max_sl=2000,
               n_heads=8, head_dim=64, dtype=torch.bfloat16, seed=42):
    rng = random.Random(seed)
    seqlens = []
    total = 0
    while total < target_tokens:
        sl = rng.randint(min_sl, max_sl)
        seqlens.append(sl)
        total += sl

    total_tokens = sum(seqlens)
    max_seqlen = max(seqlens)
    print(f"Batch: {len(seqlens)} seqs, total_tokens={total_tokens}, "
          f"max_seqlen={max_seqlen}, avg_seqlen={total_tokens/len(seqlens):.0f}")

    torch.manual_seed(seed)
    q = torch.randn(total_tokens, n_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_tokens, n_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_tokens, n_heads, head_dim, device="cuda", dtype=dtype)

    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
                       dtype=torch.int32, device="cuda")
    return q, k, v, cu, max_seqlen, seqlens


def make_edges(cu_seqlens, density=0.05, seed=123):
    """Generate random edges with given density per sequence."""
    rng = random.Random(seed)
    cu_cpu = cu_seqlens.cpu()
    n_seqs = cu_cpu.numel() - 1
    src_list, dst_list = [], []
    for i in range(n_seqs):
        start = cu_cpu[i].item()
        end = cu_cpu[i + 1].item()
        sl = end - start
        n_edges = max(1, int(sl * sl * density))
        for _ in range(n_edges):
            r = rng.randint(0, sl - 1)
            c = rng.randint(0, sl - 1)
            src_list.append(start + r)
            dst_list.append(start + c)
    return torch.tensor([src_list, dst_list], dtype=torch.int64, device="cuda")


def bench_fn(fn, warmup=5, repeats=20):
    """Time a function using CUDA events. Returns (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def main():
    n_heads = 8
    head_dim = 64
    density = 0.05
    target_tokens = 1_000_000

    print("=" * 70)
    print("Edge Bias Benchmark")
    print(f"  n_heads={n_heads}, head_dim={head_dim}, density={density}")
    print(f"  target_tokens={target_tokens}")
    print("=" * 70)

    q, k, v, cu, max_sl, seqlens = make_batch(
        target_tokens=target_tokens, n_heads=n_heads, head_dim=head_dim)
    scale = head_dim ** -0.5

    edge_index = make_edges(cu, density=density)
    total_edges = edge_index.shape[1]
    total_tokens = q.shape[0]
    total_pairs = sum(s * s for s in seqlens)
    print(f"Total edges: {total_edges} "
          f"(effective density: {total_edges/total_pairs:.4f})")

    fwd_bm, fwd_bn = get_kernel_block_sizes(head_dim, is_backward=False)
    bwd_bm, bwd_bn = get_kernel_block_sizes(head_dim, is_backward=True)

    fwd_to, fwd_ki, fwd_bs, fwd_tm, fwd_cq, fwd_ck = \
        build_edge_bias_bitset(edge_index, cu, max_sl,
                               block_size_m=fwd_bm, block_size_n=fwd_bn)

    bwd_to, bwd_ki, bwd_bs, bwd_tm, bwd_cq, bwd_ck = \
        build_edge_bias_bitset(edge_index, cu, max_sl,
                               block_size_m=bwd_bm, block_size_n=bwd_bn)

    eb_scale = torch.randn(n_heads, device="cuda", dtype=torch.float32)
    eb_scale_grad = eb_scale.clone().requires_grad_(True)

    print(f"\nForward tile sizes: kBlockM={fwd_bm}, kBlockN={fwd_bn}")
    print(f"Backward tile sizes: kBlockM={bwd_bm}, kBlockN={bwd_bn}")
    print(f"FWD tiles: {fwd_to.shape[0]-1} q_blocks, {fwd_ki.shape[0]} nnz_tiles")
    print(f"BWD tiles: {bwd_to.shape[0]-1} q_blocks, {bwd_ki.shape[0]} nnz_tiles")

    # ---- Forward only: no edge bias ----
    def fwd_no_bias():
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_sl, max_seqlen_k=max_sl,
            softmax_scale=scale, causal=False,
        )

    # ---- Forward only: with edge bias ----
    def fwd_with_bias():
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_sl, max_seqlen_k=max_sl,
            edge_bias_scale=eb_scale,
            softmax_scale=scale, causal=False,
            edge_bias_tile_offsets=fwd_to,
            edge_bias_tile_k_indices=fwd_ki,
            edge_bias_bitsets=fwd_bs,
            edge_bias_tile_map=fwd_tm,
            cu_q_blocks=fwd_cq, cu_k_blocks=fwd_ck,
        )

    # ---- Forward + Backward: no edge bias ----
    def fwd_bwd_no_bias():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out = flash_attn_varlen_func(
            q_, k_, v_,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_sl, max_seqlen_k=max_sl,
            softmax_scale=scale, causal=False,
        )
        out.sum().backward()

    # ---- Forward + Backward: with edge bias ----
    def fwd_bwd_with_bias():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        ebs = eb_scale.detach().requires_grad_(True)
        out = flash_attn_varlen_func(
            q_, k_, v_,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_sl, max_seqlen_k=max_sl,
            edge_bias_scale=ebs,
            softmax_scale=scale, causal=False,
            edge_bias_tile_offsets=fwd_to,
            edge_bias_tile_k_indices=fwd_ki,
            edge_bias_bitsets=fwd_bs,
            edge_bias_tile_map=fwd_tm,
            cu_q_blocks=fwd_cq, cu_k_blocks=fwd_ck,
            edge_bias_bwd=(bwd_to, bwd_ki, bwd_bs, bwd_tm, bwd_cq, bwd_ck),
        )
        out.sum().backward()

    print("\n" + "=" * 70)
    print("Benchmarking (warmup=5, repeats=20)")
    print("=" * 70)

    mean, std = bench_fn(fwd_no_bias)
    print(f"\n[Forward] No edge bias:   {mean:8.2f} ms  (± {std:.2f})")

    mean_eb, std_eb = bench_fn(fwd_with_bias)
    print(f"[Forward] With edge bias: {mean_eb:8.2f} ms  (± {std_eb:.2f})")
    overhead = (mean_eb - mean) / mean * 100
    print(f"  → overhead: {mean_eb - mean:+.2f} ms ({overhead:+.1f}%)")

    mean, std = bench_fn(fwd_bwd_no_bias)
    print(f"\n[Fwd+Bwd] No edge bias:   {mean:8.2f} ms  (± {std:.2f})")

    mean_eb, std_eb = bench_fn(fwd_bwd_with_bias)
    print(f"[Fwd+Bwd] With edge bias: {mean_eb:8.2f} ms  (± {std_eb:.2f})")
    overhead = (mean_eb - mean) / mean * 100
    print(f"  → overhead: {mean_eb - mean:+.2f} ms ({overhead:+.1f}%)")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
