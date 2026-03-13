"""Debug script for NaN in backward pass for head_dim >= 64."""
import json, time, os

LOG_PATH = "/home/seohyun/flash-attention/.cursor/debug-085910.log"

def log(hypothesis_id, location, message, data=None):
    entry = {
        "sessionId": "085910",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

import torch
from flash_attn import flash_attn_func

def test_head_dim(hd):
    torch.manual_seed(42)
    b, s, h = 2, 16, 4
    q = torch.randn(b, s, h, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(b, s, h, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(b, s, h, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    # Hypothesis C: check forward output
    out = flash_attn_func(q, k, v, softmax_scale=hd**-0.5, causal=False)
    out_nan = out.isnan().any().item()
    out_inf = out.isinf().any().item()
    out_max = out.abs().max().item()
    log("C", f"debug_nan.py:fwd:hd{hd}", "Forward output check", {
        "head_dim": hd, "out_nan": out_nan, "out_inf": out_inf, "out_max": out_max
    })

    # Check softmax_lse from the internal state
    # (not directly accessible, but NaN in forward would propagate)

    if out_nan or out_inf:
        log("C", f"debug_nan.py:fwd:hd{hd}", "Forward FAILED - skipping backward", {
            "head_dim": hd
        })
        return

    # Backward pass
    loss = out.float().sum()
    loss.backward()

    dq_nan = q.grad.isnan().any().item()
    dk_nan = k.grad.isnan().any().item()
    dv_nan = v.grad.isnan().any().item()
    dq_inf = q.grad.isinf().any().item()
    dq_max = q.grad.float().abs().max().item() if not dq_nan else float('inf')

    log("A", f"debug_nan.py:bwd:hd{hd}", "Backward gradient check", {
        "head_dim": hd,
        "dq_nan": dq_nan, "dk_nan": dk_nan, "dv_nan": dv_nan,
        "dq_inf": dq_inf, "dq_max": dq_max
    })

    # Hypothesis D: Check if params struct alignment is off
    # by testing with different batch sizes and seq lengths
    if dq_nan and hd == 64:
        for test_s in [8, 32, 64, 128, 256]:
            q2 = torch.randn(1, test_s, 1, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            k2 = torch.randn(1, test_s, 1, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            v2 = torch.randn(1, test_s, 1, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            out2 = flash_attn_func(q2, k2, v2, softmax_scale=hd**-0.5, causal=False)
            out2.float().sum().backward()
            dq2_nan = q2.grad.isnan().any().item()
            log("D", f"debug_nan.py:bwd:hd{hd}:s{test_s}", "Varying seqlen test", {
                "head_dim": hd, "seqlen": test_s, "dq_nan": dq2_nan
            })

    # Also test with flash_attn_varlen_func (no edge bias)
    if dq_nan and hd == 64:
        from flash_attn import flash_attn_varlen_func
        total = 20
        cu = torch.tensor([0, 10, 20], device='cuda', dtype=torch.int32)
        q3 = torch.randn(total, h, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        k3 = torch.randn(total, h, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v3 = torch.randn(total, h, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        out3 = flash_attn_varlen_func(q3, k3, v3, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                       max_seqlen_q=10, max_seqlen_k=10,
                                       softmax_scale=hd**-0.5, causal=False)
        out3.float().sum().backward()
        dq3_nan = q3.grad.isnan().any().item()
        log("B", f"debug_nan.py:varlen_bwd:hd{hd}", "Varlen backward (no edge bias)", {
            "head_dim": hd, "dq_nan": dq3_nan
        })

    # Test with fp16 instead of bf16
    if dq_nan and hd == 64:
        q4 = torch.randn(b, s, h, hd, device='cuda', dtype=torch.float16, requires_grad=True)
        k4 = torch.randn(b, s, h, hd, device='cuda', dtype=torch.float16, requires_grad=True)
        v4 = torch.randn(b, s, h, hd, device='cuda', dtype=torch.float16, requires_grad=True)
        out4 = flash_attn_func(q4, k4, v4, softmax_scale=hd**-0.5, causal=False)
        out4.float().sum().backward()
        dq4_nan = q4.grad.isnan().any().item()
        log("A", f"debug_nan.py:fp16_bwd:hd{hd}", "FP16 backward check", {
            "head_dim": hd, "dq_nan": dq4_nan
        })


if __name__ == "__main__":
    print("Testing head_dim=32...")
    test_head_dim(32)
    print("Testing head_dim=64...")
    test_head_dim(64)
    print("Testing head_dim=96...")
    test_head_dim(96)
    print("Testing head_dim=128...")
    test_head_dim(128)
    print("Done. Check log at:", LOG_PATH)
