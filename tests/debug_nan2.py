"""Confirm Is_V_in_regs correlation with NaN."""
import json, time

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

torch.manual_seed(42)
b, s, h = 2, 64, 4

for hd in [32, 64, 96, 128]:
    for dropout_p in [0.0, 0.5]:
        q = torch.randn(b, s, h, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(b, s, h, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(b, s, h, hd, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=hd**-0.5,
                              causal=False, return_attn_probs=(dropout_p > 0))
        if isinstance(out, tuple):
            out = out[0]

        fwd_nan = out.isnan().any().item()
        loss = out.float().sum()
        loss.backward()
        dq_nan = q.grad.isnan().any().item()

        msg = f"hd={hd:3d} dropout={dropout_p:.1f} fwd_nan={fwd_nan} dq_nan={dq_nan}"
        print(msg)
        log("V", f"debug_nan2.py:hd{hd}:dp{dropout_p}", msg, {
            "head_dim": hd, "dropout": dropout_p, "fwd_nan": fwd_nan, "dq_nan": dq_nan
        })

print("Done.")
