# Edge Bias 사용 가이드

FlashAttention에 **edge bias**를 적용하여 그래프 구조의 sparse한 pair-wise bias를 attention score에 더하는 방법을 설명합니다.

## 개요

Edge bias는 GNN-Transformer 하이브리드 모델 등에서, 특정 (query, key) 쌍에만 learnable한 bias를 더하고 싶을 때 사용합니다. 일반적인 dense bias matrix 대신 **sparse edge list**를 tile 기반 bitset 구조로 변환하여 CUDA 커널에 전달합니다.

**수식:**

```
Attention(Q, K, V) = softmax( QK^T / √d + edge_bias_scale · M ) V
```

여기서 `M[i,j] = 1`이면 edge `(i, j)`가 존재, `0`이면 미존재입니다.
`edge_bias_scale`은 head별 learnable scalar로, gradient가 흐릅니다.

## 기본 사용법

### 1. 필요한 import

```python
import torch
from flash_attn import flash_attn_varlen_func

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "csrc", "flash_attn"))
from edge_bias_utils import build_edge_bias_bitset, get_kernel_block_sizes
```

### 2. 입력 데이터 준비

```python
# 시퀀스 길이 (variable length batching)
seqlens = [200, 150, 300]
total_tokens = sum(seqlens)
max_seqlen = max(seqlens)
n_heads = 8
head_dim = 64

# Q, K, V 텐서 (packed format)
q = torch.randn(total_tokens, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
k = torch.randn(total_tokens, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
v = torch.randn(total_tokens, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)

# cumulative sequence lengths
cu_seqlens = torch.tensor(
    [0] + list(torch.cumsum(torch.tensor(seqlens), dim=0)),
    device="cuda", dtype=torch.int32
)
```

### 3. Edge index 생성

Edge index는 `[2, E]` 형태의 int64 텐서로, **packed-tensor 좌표계**를 사용합니다.
즉, 각 시퀀스의 local 좌표가 아닌, batch 전체에서의 global 위치입니다.

```python
# 예: 첫 번째 시퀀스(offset=0)에서 position 3 → position 7 연결
# 두 번째 시퀀스(offset=200)에서 position 10 → position 25 연결
edge_index = torch.tensor([
    [3, 210],    # source (Q side)
    [7, 225],    # destination (K side)
], dtype=torch.int64, device="cuda")
```

실제로는 그래프의 edge list에서 `cu_seqlens` offset을 더해 packed 좌표로 변환합니다.

### 4. Edge bias scale 생성

```python
# head별 learnable scalar (gradient 흐름)
edge_bias_scale = torch.randn(n_heads, device="cuda", dtype=torch.float32, requires_grad=True)
```

### 5. Bitset 구조 빌드

커널의 실제 tile 크기를 가져와서 bitset을 빌드합니다.
**Forward와 backward에서 tile 크기가 다를 수 있으므로** 각각 별도로 빌드해야 합니다.

```python
# 커널의 실제 tile 크기 조회
fwd_bm, fwd_bn = get_kernel_block_sizes(head_dim, is_backward=False)
bwd_bm, bwd_bn = get_kernel_block_sizes(head_dim, is_backward=True)

# Forward용 bitset
fwd_result = build_edge_bias_bitset(
    edge_index, cu_seqlens, max_seqlen,
    block_size_m=fwd_bm, block_size_n=fwd_bn,
)
fwd_tile_offsets, fwd_tile_k_indices, fwd_bitsets, fwd_tile_map, fwd_cu_q, fwd_cu_k = fwd_result

# Backward용 bitset (tile 크기가 다를 수 있음)
bwd_result = build_edge_bias_bitset(
    edge_index, cu_seqlens, max_seqlen,
    block_size_m=bwd_bm, block_size_n=bwd_bn,
)
bwd_tile_offsets, bwd_tile_k_indices, bwd_bitsets, bwd_tile_map, bwd_cu_q, bwd_cu_k = bwd_result
```

### 6. FlashAttention 호출

```python
output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    softmax_scale=head_dim ** -0.5,
    causal=False,
    # Edge bias (forward)
    edge_bias_tile_offsets=fwd_tile_offsets,
    edge_bias_tile_k_indices=fwd_tile_k_indices,
    edge_bias_bitsets=fwd_bitsets,
    edge_bias_scale=edge_bias_scale,
    edge_bias_tile_map=fwd_tile_map,
    cu_q_blocks=fwd_cu_q,
    cu_k_blocks=fwd_cu_k,
    # Edge bias (backward) — tile 크기가 같으면 생략 가능
    edge_bias_bwd=(
        bwd_tile_offsets, bwd_tile_k_indices, bwd_bitsets,
        bwd_tile_map, bwd_cu_q, bwd_cu_k,
    ),
)

# Backward
loss = output.float().sum()
loss.backward()

# edge_bias_scale.grad 로 gradient 확인 가능
```

## API 레퍼런스

### `get_kernel_block_sizes(head_dim, is_backward=False)`

CUDA 커널이 사용하는 실제 tile 크기 `(kBlockM, kBlockN)`을 반환합니다.

| head_dim | Forward (M, N) | Backward (M, N) |
|----------|----------------|-----------------|
| 32       | (128, 128)     | (128, 128)      |
| 64       | (128, 128)     | (128, 128)      |
| 96       | (128, 64)      | (64, 128)       |
| 128      | (128, 64)      | (64, 128)       |
| 192      | (128, 64)      | (64, 64)        |
| 256      | (64, 64)       | (64, 64)        |

> 위 값은 sm90 (H100), no dropout, smem >= 144KB 기준입니다.
> GPU나 dropout 설정에 따라 달라질 수 있습니다.

### `build_edge_bias_bitset(edge_index, cu_seqlens, max_seqlen, block_size_m, block_size_n)`

**인자:**

| 인자 | 타입 | 설명 |
|------|------|------|
| `edge_index` | `[2, E]` int64 | packed-tensor 좌표의 edge list. `[0]`=source(Q), `[1]`=dest(K) |
| `cu_seqlens` | `[B+1]` int32 | cumulative sequence lengths |
| `max_seqlen` | int | 배치 내 최대 시퀀스 길이 |
| `block_size_m` | int | Q 차원 tile 크기 (커널의 kBlockM과 일치해야 함) |
| `block_size_n` | int | K 차원 tile 크기 (커널의 kBlockN과 일치해야 함) |

**반환값 (6개 텐서):**

| 반환값 | Shape | 설명 |
|--------|-------|------|
| `tile_offsets` | `[total_q_blocks+1]` int32 | CSR 형식의 row pointer (Q block 기준) |
| `tile_k_indices` | `[nnz_tiles]` int32 | 각 tile의 global K block index |
| `bitsets` | `[nnz_tiles, words]` int32 | tile별 bit matrix (M×N bits) |
| `tile_map` | `[total_q_blocks, max_k_blocks]` int32 | 2D dense lookup (-1이면 tile 없음) |
| `cu_q_blocks` | `[B+1]` int32 | cumulative Q block counts |
| `cu_k_blocks` | `[B+1]` int32 | cumulative K block counts |

### `flash_attn_varlen_func` edge bias 관련 파라미터

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `edge_bias_tile_offsets` | Tensor | `build_edge_bias_bitset` 반환값의 첫 번째 |
| `edge_bias_tile_k_indices` | Tensor | 두 번째 반환값 |
| `edge_bias_bitsets` | Tensor | 세 번째 반환값 |
| `edge_bias_scale` | `[n_heads]` float32 | head별 bias scale (learnable) |
| `edge_bias_tile_map` | Tensor | 네 번째 반환값 |
| `cu_q_blocks` | Tensor | 다섯 번째 반환값 |
| `cu_k_blocks` | Tensor | 여섯 번째 반환값 |
| `edge_bias_bwd` | tuple or None | backward용 6-tuple (아래 참조) |

`edge_bias_bwd`는 backward pass에서 사용할 별도의 bitset 구조입니다.
forward와 backward의 tile 크기가 다를 때 반드시 지정해야 합니다.

```python
edge_bias_bwd = (
    bwd_tile_offsets,      # [total_q_blocks+1] int32
    bwd_tile_k_indices,    # [nnz_tiles] int32
    bwd_bitsets,           # [nnz_tiles, words] int32
    bwd_tile_map,          # [total_q_blocks, max_k_blocks] int32
    bwd_cu_q_blocks,       # [B+1] int32
    bwd_cu_k_blocks,       # [B+1] int32
)
```

> `edge_bias_max_k_blocks`(CUDA 커널의 tile_map stride)는 `tile_map.shape[1]`에서 자동으로 추출되므로 별도로 전달할 필요가 없습니다.

## 다른 기능과의 조합

Edge bias는 다음 기능들과 함께 사용할 수 있습니다:

- **Causal masking** (`causal=True`)
- **Sliding window** (`window_size=(left, right)`)
- **ALiBi** (`alibi_slopes=...`)
- **Softcap** (`softcap=...`)
- **Dropout** (`dropout_p=...`)
- **MQA/GQA** (K, V의 head 수가 Q보다 적은 경우)

## 지원하는 head_dim

`head_dim`은 32, 64, 96, 128, 192, 256을 지원합니다.

## 주의사항

1. **block_size 일치**: `build_edge_bias_bitset`의 `block_size_m`/`block_size_n`은 반드시 커널의 실제 `kBlockM`/`kBlockN`과 일치해야 합니다. `get_kernel_block_sizes()`를 사용하세요.
2. **`get_kernel_block_sizes()` 가정 조건**: 현재 구현은 **sm90 (H100) + no dropout + smem >= 144KB** 환경을 가정합니다. 이 조건에 해당하지 않는 GPU(예: A100에서 smem < 144KB인 경우, sm86/sm89, 또는 dropout 사용 시)에서는 커널이 다른 block size를 선택하므로, `get_kernel_block_sizes()`의 반환값을 해당 환경에 맞게 직접 수정해야 합니다. 실제 커널의 dispatch 로직은 `flash_fwd_launch_template.h`와 `flash_bwd_launch_template.h`를 참고하세요.
3. **Packed 좌표**: `edge_index`는 시퀀스별 local 좌표가 아닌, batch 전체의 global packed 좌표입니다.
4. **Forward/Backward 분리**: `head_dim`에 따라 forward/backward의 tile 크기가 다를 수 있으므로, 각각 별도로 bitset을 빌드하고 `edge_bias_bwd` 파라미터로 backward용을 전달하세요.
5. **dtype**: Q, K, V는 `bfloat16` 또는 `float16`이어야 합니다. `edge_bias_scale`은 `float32`입니다.
6. **GPU**: sm80 (A100) 이상을 지원합니다.

## 전체 예제

### 단순 forward/backward

```python
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "csrc", "flash_attn"))
from edge_bias_utils import build_edge_bias_bitset, get_kernel_block_sizes
from flash_attn import flash_attn_varlen_func

# --- 데이터 준비 ---
seqlens = [200, 150, 300]
total = sum(seqlens)
max_sl = max(seqlens)
n_heads, head_dim = 8, 64

q = torch.randn(total, n_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(total, n_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(total, n_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), dim=0)),
                   device="cuda", dtype=torch.int32)

# --- Edge list (예: 각 시퀀스 내에서 랜덤 edge 생성) ---
import random
random.seed(42)
src_list, dst_list = [], []
for i in range(len(seqlens)):
    offset = cu[i].item()
    sl = seqlens[i]
    for _ in range(int(sl * sl * 0.1)):
        src_list.append(offset + random.randint(0, sl - 1))
        dst_list.append(offset + random.randint(0, sl - 1))
edge_index = torch.tensor([src_list, dst_list], dtype=torch.int64, device="cuda")

# --- Learnable scale ---
eb_scale = torch.randn(n_heads, device="cuda", dtype=torch.float32, requires_grad=True)

# --- Bitset 빌드 ---
fwd_bm, fwd_bn = get_kernel_block_sizes(head_dim, is_backward=False)
bwd_bm, bwd_bn = get_kernel_block_sizes(head_dim, is_backward=True)

fwd_to, fwd_ki, fwd_bs, fwd_tm, fwd_cq, fwd_ck = build_edge_bias_bitset(
    edge_index, cu, max_sl, block_size_m=fwd_bm, block_size_n=fwd_bn)

bwd_to, bwd_ki, bwd_bs, bwd_tm, bwd_cq, bwd_ck = build_edge_bias_bitset(
    edge_index, cu, max_sl, block_size_m=bwd_bm, block_size_n=bwd_bn)

# --- Forward + Backward ---
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=max_sl, max_seqlen_k=max_sl,
    softmax_scale=head_dim ** -0.5,
    causal=False,
    edge_bias_tile_offsets=fwd_to,
    edge_bias_tile_k_indices=fwd_ki,
    edge_bias_bitsets=fwd_bs,
    edge_bias_scale=eb_scale,
    edge_bias_tile_map=fwd_tm,
    cu_q_blocks=fwd_cq,
    cu_k_blocks=fwd_ck,
    edge_bias_bwd=(bwd_to, bwd_ki, bwd_bs, bwd_tm, bwd_cq, bwd_ck),
)

loss = out.float().sum()
loss.backward()

print(f"output shape: {out.shape}")
print(f"dQ has NaN: {q.grad.isnan().any().item()}")
print(f"edge_bias_scale grad: {eb_scale.grad}")
```

### 학습 루프에서의 사용 패턴

`edge_bias_scale`은 fp32 `nn.Parameter`로 선언합니다. CUDA 커널 내부에서 attention score
accumulator가 fp32이므로, scale도 fp32로 받아야 정밀도 손실이 없습니다.
backward에서 반환되는 `d_edge_bias_scale` gradient 역시 fp32입니다.

```python
import torch
import torch.nn as nn

class GraphAttentionLayer(nn.Module):
    def __init__(self, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        # fp32 — 커널이 fp32 accumulator에서 직접 사용하므로
        self.edge_bias_scale = nn.Parameter(torch.zeros(n_heads, dtype=torch.float32))

    def forward(self, q, k, v, cu_seqlens, max_seqlen, edge_index):
        fwd_bm, fwd_bn = get_kernel_block_sizes(self.head_dim, is_backward=False)
        bwd_bm, bwd_bn = get_kernel_block_sizes(self.head_dim, is_backward=True)

        fwd_to, fwd_ki, fwd_bs, fwd_tm, fwd_cq, fwd_ck = build_edge_bias_bitset(
            edge_index, cu_seqlens, max_seqlen,
            block_size_m=fwd_bm, block_size_n=fwd_bn)
        bwd_to, bwd_ki, bwd_bs, bwd_tm, bwd_cq, bwd_ck = build_edge_bias_bitset(
            edge_index, cu_seqlens, max_seqlen,
            block_size_m=bwd_bm, block_size_n=bwd_bn)

        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
            edge_bias_tile_offsets=fwd_to,
            edge_bias_tile_k_indices=fwd_ki,
            edge_bias_bitsets=fwd_bs,
            edge_bias_scale=self.edge_bias_scale,
            edge_bias_tile_map=fwd_tm,
            cu_q_blocks=fwd_cq,
            cu_k_blocks=fwd_ck,
            edge_bias_bwd=(bwd_to, bwd_ki, bwd_bs, bwd_tm, bwd_cq, bwd_ck),
        )

# --- 학습 루프 ---
model = GraphAttentionLayer(n_heads=8, head_dim=64).cuda()

# edge_bias_scale은 fp32이므로, optimizer에 넘길 때 별도 param group으로 분리하면
# mixed-precision 학습에서도 fp32 master weight 없이 바로 업데이트 가능
optimizer = torch.optim.Adam([
    {"params": [p for n, p in model.named_parameters() if n != "edge_bias_scale"]},
    {"params": [model.edge_bias_scale], "lr": 1e-2},  # scale용 별도 lr 설정 가능
], lr=1e-4)

for step in range(100):
    optimizer.zero_grad()
    out = model(q, k, v, cu, max_sl, edge_index)
    loss = out.float().sum()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"step={step}, loss={loss.item():.4f}, "
              f"eb_scale={model.edge_bias_scale.data.tolist()}")
```
