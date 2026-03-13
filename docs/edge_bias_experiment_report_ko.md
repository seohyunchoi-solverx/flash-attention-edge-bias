# FlashAttention Edge Bias: 실험 보고서

## 1. 개요

본 보고서는 FlashAttention-2의 CUDA 커널에 **희소 edge bias** 기능을 구현, 디버깅, 검증한 과정을 문서화한다. Edge bias는 임의의 (희소한) query-key 위치 쌍에 대해 학습 가능한 head별 스칼라 바이어스를 주입할 수 있게 해주며, 밀집 바이어스 행렬을 실체화하지 않고도 그래프 인지 어텐션 패턴을 구현할 수 있다.

### 1.1 동기

그래프 신경망(GNN)이나 분자 모델링에서는 토큰 간 어텐션 스코어를 그래프 구조의 엣지 존재 여부에 따라 조절해야 하는 경우가 많다. 단순한 밀집 바이어스 행렬 `(seqlen × seqlen)`은 긴 시퀀스에서 메모리와 연산 비용이 지나치게 크다. 본 접근법은 **타일 기반 bitset** 표현을 사용한다: 각 `(kBlockM × kBlockN)` 어텐션 타일마다 어떤 Q-K 쌍에 엣지가 있는지를 나타내는 압축 bitset을 저장하여, shared memory에서 비트당 O(1) 조회를 가능하게 한다.

### 1.2 실험 환경

| 구성 요소 | 값 |
|---|---|
| GPU | NVIDIA H100 (sm_90), 80 GB HBM3 |
| CUDA | 12.x |
| PyTorch | 2.x (CUDA 지원) |
| FlashAttention | v2.x (Dao-AILab 포크) |
| Python | 3.10.12 |
| 데이터 타입 | `bfloat16` (주), `float16` (부) |

---

## 2. 아키텍처: Edge Bias 자료구조

### 2.1 Python 측 (`edge_bias_utils.py`)

Python 유틸리티 `build_edge_bias_bitset()`는 희소 엣지 리스트 `[2, E]`를 커널 호환 구조체로 변환한다:

| 구조체 | 형태 | 설명 |
|---|---|---|
| `tile_offsets` | `[total_q_blocks + 1]` int32 | 전역 q_block 기준 CSR 행 포인터 |
| `tile_k_indices` | `[nnz_tiles]` int32 | 비어있지 않은 타일별 전역 k_block 인덱스 |
| `bitsets` | `[nnz_tiles, bitset_words]` int32 | 타일별 압축 비트 행렬 |
| `tile_map` | `[total_q_blocks, max_k_blocks]` int32 | 밀집 2D 조회: `tile_map[q_blk, k_blk_local] → tile_idx` (비어있으면 -1) |
| `cu_q_blocks` | `[B+1]` int32 | 배치별 누적 Q 블록 수 |
| `cu_k_blocks` | `[B+1]` int32 | 배치별 누적 K 블록 수 |

**타일 내 비트 레이아웃:**
```
bit_idx = row_in_tile * block_size_n + col_in_tile
word    = bitsets[tile, bit_idx // 32]
bit     = (word >> (bit_idx % 32)) & 1
```

여기서 `bitset_words = block_size_m * block_size_n / 32`.

### 2.2 CUDA 측

커널은 `load_bitset_to_smem<kBitsetUint4>()`를 통해 한 타일의 bitset을 shared memory로 로드한 후, `apply_edge_bias<kBlockN>()`를 호출하여 해당 비트가 설정된 각 어텐션 스코어에 `edge_bias_scale[head] / scale_softmax` 값을 더한다.

**Shared memory 레이아웃:**
```
[--- Kernel_traits::kSmemSize ---][-- 16B 정렬 패딩 --][--- kBitsetBytes ---]
                                                         = kBlockM * kBlockN / 8
```

---

## 3. 문제 1: 블록 크기 불일치

### 3.1 근본 원인

FlashAttention은 다음 조건에 따라 서로 다른 타일 크기 `(kBlockM, kBlockN)`을 사용한다:
- `head_dim` (32, 64, 96, 128, 192, 256)
- 패스 방향 (forward vs. backward)
- GPU 아키텍처 (sm80 vs. sm8x vs. sm90)
- Dropout 활성화 여부, causal 모드

기존 `edge_bias_utils.py`는 **`block_size=128`로 하드코딩**되어 있어, 모든 타일이 `128×128`이라고 가정했다. 이는 `head_dim ≤ 64`에서는 올바르지만, 커널이 비정방형 타일을 사용하는 더 큰 head dimension에서는 틀렸다.

### 3.2 실제 커널 블록 크기 (sm90, dropout 없음)

**Forward 패스** (출처: `flash_fwd_launch_template.h`):

| head_dim | kBlockM | kBlockN | Bitset 워드 수 | Bitset 바이트 |
|---|---|---|---|---|
| 32 | 128 | 128 | 512 | 2048 |
| 64 | 128 | 128 | 512 | 2048 |
| 96 | 128 | 64 | 256 | 1024 |
| 128 | 128 | 64 | 256 | 1024 |
| 192 | 128 | 64 | 256 | 1024 |
| 256 | 64 | 64 | 128 | 512 |

**Backward 패스** (출처: `flash_bwd_launch_template.h`, sm90, `max_smem ≥ 144 KB`):

| head_dim | kBlockM | kBlockN | Bitset 워드 수 | Bitset 바이트 |
|---|---|---|---|---|
| 32 | 128 | 128 | 512 | 2048 |
| 64 | 128 | 128 | 512 | 2048 |
| 96 | 64 | 128 | 256 | 1024 |
| 128 | 64 | 128 | 256 | 1024 |
| 192 | 64 | 64 | 128 | 512 |
| 256 | 64 | 64 | 128 | 512 |

**핵심 관찰:** `head_dim ∈ {96, 128}`의 경우, forward와 backward가 *전치된* 타일 형태를 사용한다:
- Forward: `(128, 64)` — 세로로 긴 타일
- Backward: `(64, 128)` — 가로로 긴 타일

이는 패스 간 bitset 레이아웃이 호환되지 않음을 의미하며, forward와 backward에 대해 **별도의 edge bias 구조체**가 필요하다.

### 3.3 불일치의 결과

`block_size=128`로 하드코딩되어 있는데 커널이 예를 들어 `kBlockM=64, kBlockN=128`을 사용하는 경우:
1. **Bitset 크기 불일치**: Python은 512워드 bitset을 생성하지만, 커널은 256워드를 기대
2. **타일 수 불일치**: Python은 `ceil(seqlen/128)` 블록을 계산하지만, 커널은 `ceil(seqlen/64)` 블록을 순회
3. **비트 위치 오류**: `bit_idx = row * 128 + col` vs. `row * kBlockN + col`이 서로 다른 비트 위치를 생성
4. **범위 초과 읽기**: 커널이 Python이 생성한 구조체에 존재하지 않는 타일/비트에 접근

---

## 4. 문제 2: 레지스터 압박으로 인한 NaN 그래디언트

### 4.1 증상

Backward 패스에서 `head_dim = 64`와 `head_dim = 128`일 때, `Has_edge_bias = false`인 경우에도 NaN 그래디언트가 발생했다.

### 4.2 근본 원인

`EDGE_BIAS_SWITCH` 매크로가 `BOOL_SWITCH`로 확장되면서, `Has_edge_bias=true`와 `Has_edge_bias=false` **양쪽** 템플릿 변형을 모두 인스턴스화했다. 이로 인해 템플릿 조합에 의한 레지스터 압박이 두 배로 증가했다:

```
BOOL_SWITCH × EVENK_SWITCH × LOCAL_SWITCH × ALIBI_SWITCH × SOFTCAP_SWITCH × EDGE_BIAS_SWITCH
```

`BOOL_SWITCH`가 하나 추가될 때마다 커널 인스턴스 수가 두 배가 된다. `head_dim=64, 128`의 backward 커널은 이미 레지스터를 거의 최대로 사용하고 있었다. Edge bias 변형이 추가되면서 레지스터 사용량이 한계를 초과하여 **레지스터가 로컬 메모리로 스필(spill)**되었고, 이것이 수치적 불안정(NaN)을 초래했다.

### 4.3 해결 방법

`setup.py` 플래그를 통해 컴파일 시점에 미사용 템플릿 스위치를 비활성화했다:

```python
"-DFLASHATTENTION_DISABLE_DROPOUT",
"-DFLASHATTENTION_DISABLE_ALIBI",
"-DFLASHATTENTION_DISABLE_SOFTCAP",
"-DFLASHATTENTION_DISABLE_UNEVEN_K",
"-DFLASHATTENTION_DISABLE_LOCAL",
```

각 `FLASHATTENTION_DISABLE_*` 플래그는 해당 `*_SWITCH` 매크로를 `BOOL_SWITCH` (2개 인스턴스)에서 단일 경로 매크로(1개 인스턴스)로 변환한다. 이로써 비활성화된 기능당 레지스터 압박이 절반으로 줄어든다. 5개 기능을 비활성화하면 총 인스턴스 수가 `2^7 = 128`에서 `2^2 = 4`로 감소한다 (동적으로 남는 것은 `IsEvenMN`과 `Has_edge_bias`뿐).

**`static_switch.h`에서:**
```cpp
// FLASHATTENTION_DISABLE_DROPOUT이 정의된 경우:
#define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] { constexpr static bool CONST_NAME = false; return __VA_ARGS__(); }()
// vs. 기본값 (양쪽 경로 모두):
#define DROPOUT_SWITCH BOOL_SWITCH
```

---

## 5. 변경 사항

### 5.1 `csrc/flash_attn/edge_bias_utils.py`

1. **`get_kernel_block_sizes(head_dim, is_backward)` 추가** — C++ 디스패치 로직을 복제하여 기본 경로(sm90, dropout 없음, `smem ≥ 144 KB`)의 `(kBlockM, kBlockN)`을 반환한다.

2. **`build_edge_bias_bitset()` 수정**: `block_size_m`과 `block_size_n`을 별도로 받도록 변경:
   - `bitset_words = block_size_m * block_size_n // 32` (기존에는 `128*128//32 = 512`로 하드코딩)
   - `cu_q_blocks`(by `block_size_m`)와 `cu_k_blocks`(by `block_size_n`)를 분리
   - 올바른 비트 인덱싱: `bit_idx = row_in_tile * block_size_n + col_in_tile`

### 5.2 `flash_attn/flash_attn_interface.py`

`flash_attn_varlen_func()`과 `FlashAttnVarlenFunc`에 `edge_bias_bwd` 파라미터를 추가:
- `edge_bias_bwd`가 제공되면, backward 패스에서 별도의 구조체를 사용
- `None`이면, forward 구조체를 재사용 (`kBlockM_fwd == kBlockM_bwd`이고 `kBlockN_fwd == kBlockN_bwd`일 때 유효)

### 5.3 `csrc/flash_attn/src/edge_bias.h`

`load_bitset_to_smem`에 `kBitsetUint4` 템플릿 파라미터 추가:
```cpp
template <int kBitsetUint4>
__forceinline__ __device__ void load_bitset_to_smem(...) {
    if (tidx < kBitsetUint4) { dst[tidx] = src[tidx]; }
}
```
기존에는 `if (tidx < 128)`로 하드코딩되어 있었으며, 이는 128번의 `uint4` 로드 = 2048 바이트 = `128×128/8`을 가정한 것이었다.

### 5.4 `csrc/flash_attn/src/flash_fwd_kernel.h` (4곳)

커널 traits에서 파생된 동적 크기를 사용하도록 bitset 로딩을 업데이트:
```cpp
constexpr int kBitsetWords = kBlockM * kBlockN / 32;
constexpr int kBitsetUint4 = kBitsetWords * 4 / 16;
load_bitset_to_smem<kBitsetUint4>(smem_edge_bits,
    params.edge_bias_bitsets + tile_idx * kBitsetWords, tidx);
```
기존: `params.edge_bias_bitsets + tile_idx * 512`.

### 5.5 `csrc/flash_attn/src/flash_bwd_kernel.h` (주요 1곳)

Forward 커널과 동일한 변경. 두 번째 edge bias 읽기 지점(`d_edge_bias_scale` 그래디언트용)은 읽기 전용이며, 이미 로드된 shared memory를 재사용한다.

### 5.6 `csrc/flash_attn/src/flash_fwd_launch_template.h` 및 `flash_bwd_launch_template.h`

동적 shared memory 할당:
```cpp
constexpr size_t kBitsetBytes = Kernel_traits::kBlockM * Kernel_traits::kBlockN / 8;
constexpr size_t smem_size_edge = (smem_size + 15) & ~15;
const size_t smem_total = (Has_edge_bias ? smem_size_edge + kBitsetBytes : smem_size);
```
기존에는 `+ 2048` 바이트로 하드코딩되어 있었다.

### 5.7 `setup.py`

컴파일 시점 기능 비활성화 활성화:
```python
"-DFLASHATTENTION_DISABLE_DROPOUT",
"-DFLASHATTENTION_DISABLE_ALIBI",
"-DFLASHATTENTION_DISABLE_SOFTCAP",
"-DFLASHATTENTION_DISABLE_UNEVEN_K",
"-DFLASHATTENTION_DISABLE_LOCAL",
```

---

## 6. 빌드 구성

| 설정 | 값 | 목적 |
|---|---|---|
| `FLASH_ATTN_CUDA_ARCHS` | `"90"` | H100 (sm_90)만 빌드 |
| `MAX_JOBS` | 26 | nvcc 병렬 컴파일 |
| 빌드 시스템 | ninja (`.venv/bin` 경유) | 빠른 증분 빌드 + 의존성 추적 |
| `PATH` | `.venv/bin` 선행 추가 | `shutil.which('ninja')`가 ninja를 찾을 수 있도록 |
| `FLASHATTENTION_DISABLE_*` | 5개 기능 비활성화 | 템플릿 인스턴스 수 감소 |

**빌드 시간: ~3.5분** (73개 컴파일 유닛, 병렬).

---

## 7. 테스트 방법론

### 7.1 NaN 그래디언트 테스트 (`tests/debug_nan.py`)

`flash_attn_func` (non-varlen)의 forward와 backward를 `head_dim ∈ {32, 64, 96, 128}`에서 테스트:
- 고정 `seqlen=128`, `batch=2`, `n_heads=4`
- dQ, dK, dV에 대해 `torch.isnan(grad).any()` 검사
- `head_dim=64`에 대한 추가 seqlen 스윕: `{8, 32, 64, 128, 256}`
- `flash_attn_varlen_func`을 `seqlen=10`으로 테스트
- `bfloat16`과 `float16` 모두 테스트

### 7.2 Edge Bias 정확도 테스트 (`tests/test_edge_bias.py`)

FlashAttention 출력을 vanilla PyTorch 어텐션 구현과 비교하는 7개 단위 테스트:
1. **단일 시퀀스** forward (seqlen=5)
2. **다중 시퀀스** forward (seqlens=[3,5,4,6,2])
3. **Float16** forward (seqlens=[5,3])
4. **Backward** edge bias 없이 (seqlens=[4,5,3])
5. **Edge bias forward** (seqlens=[5,8,3,12,6], 밀도=0.3)
6. **Edge bias backward** (seqlens=[6,4,7], 밀도=0.25)
7. **Edge bias 엣지 없음** (seqlens=[5,3], 빈 엣지 집합)

허용 오차: `max_diff < 0.01` (fwd), `max_diff < 0.05` (bwd), `mean_diff < 0.001` (fwd).

### 7.3 대형 시퀀스 길이 테스트 (인라인 스크립트)

수백 단위 seqlen으로 edge bias를 테스트하여 다중 타일 시나리오를 검증:

| 구성 | head_dim | n_heads | seqlens | 총 토큰 수 |
|---|---|---|---|---|
| 1 | 32 | 4 | [128, 256, 200] | 584 |
| 2 | 64 | 4 | [300, 150, 400] | 850 |
| 3 | 96 | 4 | [200, 350] | 550 |
| 4 | 128 | 2 | [256, 512] | 768 |
| 5 | 32 | 4 | [500, 300, 200, 100] | 1100 |
| 6 | 64 | 4 | [600] | 600 |

각 구성에 대해:
1. `get_kernel_block_sizes(head_dim, is_backward=False)`로 forward용 edge bias 구조체 빌드
2. Backward 블록 크기가 다른 경우 `is_backward=True`로 별도 구조체 빌드
3. `edge_bias_bwd` 파라미터와 함께 `flash_attn_varlen_func` 실행
4. Vanilla 어텐션(float32 참조)과 비교
5. 검사 항목: forward max/mean 차이, 그래디언트 NaN 여부, dQ/dK/dV max 차이

허용 오차: `fwd_max < 0.05`, `fwd_mean < 0.005`, `bwd_max < 0.1` (그래디언트 성분별).

---

## 8. 결과

### 8.1 NaN 그래디언트 테스트

| head_dim | Forward | Backward |
|---|---|---|
| 32 | OK | OK |
| 64 | OK | OK |
| 96 | OK | OK |
| 128 | OK | OK |

**수정 전**: `head_dim=64`와 `head_dim=128`의 backward에서 NaN 발생.

### 8.2 Edge Bias 단위 테스트

```
test_flash_vs_vanilla_single_seq             PASSED
test_flash_vs_vanilla_multi_seq              PASSED
test_flash_vs_vanilla_fp16                   PASSED
test_flash_vs_vanilla_backward               PASSED
test_flash_vs_vanilla_edge_bias_forward      PASSED
test_flash_vs_vanilla_edge_bias_backward     PASSED
test_flash_vs_vanilla_edge_bias_no_edges     PASSED
```

**7/7 통과**, 소요 시간 3.45초.

### 8.3 대형 시퀀스 길이 테스트

| 구성 | Forward | Backward | 블록 크기 |
|---|---|---|---|
| hd32 [128,256,200] | max=0.0023, mean=0.0002 | dQ=0.004, dK=0.009, dV=0.004 | fwd=bwd=(128,128) |
| hd64 [300,150,400] | max=0.0022, mean=0.0002 | dQ=0.003, dK=0.005, dV=0.004 | fwd=bwd=(128,128) |
| hd96 [200,350] | max=0.0021, mean=0.0002 | dQ=0.004, dK=0.006, dV=0.004 | fwd=(128,64) bwd=(64,128) **별도** |
| hd128 [256,512] | max=0.0015, mean=0.0001 | dQ=0.003, dK=0.006, dV=0.004 | fwd=(128,64) bwd=(64,128) **별도** |
| hd32 [500,300,200,100] | max=0.0024, mean=0.0002 | dQ=0.004, dK=0.005, dV=0.005 | fwd=bwd=(128,128) |
| hd64 [600] | max=0.0012, mean=0.0001 | dQ=0.002, dK=0.005, dV=0.004 | fwd=bwd=(128,128) |

**6/6 통과.** NaN 없음. 모든 차이가 bf16 수치 허용 범위 이내.

**핵심 관찰:**
- `head_dim=96`과 `head_dim=128`은 전치된 타일 형태를 가진 **별도의 forward/backward edge bias 구조체**를 올바르게 사용
- Forward max 차이는 일관적으로 < 0.003 (bf16 기준 우수)
- Backward 그래디언트 차이는 일관적으로 < 0.01 (누적 반올림을 고려하면 bf16 기준 허용 가능)
- 여러 타일에 걸치는 큰 seqlen (500, 600)에서도 성능 저하 없음

---

## 9. 결론

### 9.1 블록 크기 불일치 수정

`edge_bias_utils.py`에서 `block_size=128`로 하드코딩된 것이 `head_dim > 64`에서의 주요 정확도 문제였다. 수정 내용은 다음과 같다:
1. C++ 디스패치 테이블을 Python에서 복제하는 `get_kernel_block_sizes()`
2. 파이프라인 전체에서 `block_size_m` / `block_size_n` 파라미터를 분리
3. Forward와 backward 타일 형태가 다른 경우를 위한 `edge_bias_bwd` 파라미터

**이는 커널이 비정방형 또는 전치된 타일을 사용하는 `head_dim ∈ {96, 128, 192, 256}`에서 필수적이다.**

### 9.2 레지스터 압박 수정

미사용 `BOOL_SWITCH` 기능 (`DROPOUT`, `ALIBI`, `SOFTCAP`, `UNEVEN_K`, `LOCAL`)을 컴파일 시점에 비활성화하여, 템플릿 인스턴스화를 O(2^7)에서 O(2^2)로 줄임으로써 `head_dim = 64, 128`에서의 NaN 그래디언트를 제거했다.

**중요한 주의사항:** 해당 기능이 필요한 경우 이 비활성화 플래그를 다시 활성화해야 한다. 프로덕션 사용을 위해 다음을 고려할 수 있다:
- Edge bias 커널을 별도 `.cu` 파일로 분리하여 레지스터 압박을 격리
- `__launch_bounds__`를 사용하여 레지스터 할당을 명시적으로 제어
- `--ptxas-options=-v`로 커널 변형별 레지스터 사용량을 프로파일링

### 9.3 빌드 시스템

- **ninja**가 `PATH`에 있어야 PyTorch의 `cpp_extension`이 이를 찾을 수 있다 (`shutil.which('ninja')` 사용)
- `.venv/bin/ninja`는 기본적으로 발견되지 않았으며, `.venv/bin`을 `PATH`에 선행 추가하여 해결
- `distutils` 폴백(ninja 없을 때)은 헤더 의존성을 추적**하지 않아** 오래된 오브젝트 파일이 남는 문제 발생
- 현재 GPU 아키텍처만 대상으로 하려면 `FLASH_ATTN_CUDA_ARCHS="90"`으로 빌드

### 9.4 남은 제한사항

1. `get_kernel_block_sizes()`는 sm90, `max_smem ≥ 144 KB`, dropout 없음을 가정한다. 다른 GPU 아키텍처나 구성에서는 수동 오버라이드가 필요하다.
2. `FLASHATTENTION_DISABLE_*` 플래그는 개발 시점의 임시 방편이다. 프로덕션 솔루션에서는 edge bias 인스턴스를 격리해야 한다.
3. `head_dim = 192, 256`에서의 edge bias는 테스트되지 않았다 (해당 구성의 테스트 데이터가 생성되지 않음).
4. Backward 커널에서 `edge_bias_scale` 그래디언트 계산은 backward 패스의 forward 스타일 스코어 계산 중 로드된 shared memory bitset을 재사용하므로, 두 번째 로드가 필요하지 않다.

---

## 부록 A: 파일 변경 요약

| 파일 | 변경 줄 수 | 성격 |
|---|---|---|
| `csrc/flash_attn/edge_bias_utils.py` | +65 (재작성) | `get_kernel_block_sizes()`, 직사각형 타일 지원 |
| `flash_attn/flash_attn_interface.py` | +20 | `edge_bias_bwd` 파라미터 전달 |
| `csrc/flash_attn/src/edge_bias.h` | +3 | 템플릿 `kBitsetUint4` 파라미터 |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | +16 (4곳) | 동적 `kBitsetWords` / `kBitsetUint4` |
| `csrc/flash_attn/src/flash_bwd_kernel.h` | +4 (1곳) | 동적 `kBitsetWords` / `kBitsetUint4` |
| `csrc/flash_attn/src/flash_fwd_launch_template.h` | +6 (2곳) | 동적 SMEM 할당 |
| `csrc/flash_attn/src/flash_bwd_launch_template.h` | +3 (1곳) | 동적 SMEM 할당 |
| `setup.py` | +5 | 컴파일 시점 기능 비활성화 |

## 부록 B: 재현 방법

```bash
cd /path/to/flash-attention

# 1. 빌드 (sm90만, 레지스터 압박 해소를 위한 기능 비활성화 포함)
export PATH="$(pwd)/.venv/bin:$PATH"
FLASH_ATTN_CUDA_ARCHS="90" MAX_JOBS=26 .venv/bin/python setup.py build_ext --inplace

# 2. NaN 테스트
.venv/bin/python tests/debug_nan.py

# 3. Edge bias 단위 테스트
.venv/bin/python -m pytest tests/test_edge_bias.py -v

# 4. 대형 seqlen 테스트 (7.3절의 인라인 스크립트 참조)
```

## 부록 C: 사용 예시

```python
from csrc.flash_attn.edge_bias_utils import build_edge_bias_bitset, get_kernel_block_sizes
from flash_attn import flash_attn_varlen_func

head_dim = 96

# 커널과 일치하는 블록 크기 가져오기
bm_f, bn_f = get_kernel_block_sizes(head_dim, is_backward=False)  # (128, 64)
bm_b, bn_b = get_kernel_block_sizes(head_dim, is_backward=True)   # (64, 128)

# Forward 구조체 빌드
fwd = build_edge_bias_bitset(edge_index, cu_seqlens, max_seqlen,
                              block_size_m=bm_f, block_size_n=bn_f)
tile_offsets, tile_k_indices, bitsets, tile_map, cu_q_blocks, cu_k_blocks = fwd
max_k_blocks = tile_map.shape[1]

# Backward 구조체 빌드 (타일 형태가 다름!)
bwd = build_edge_bias_bitset(edge_index, cu_seqlens, max_seqlen,
                              block_size_m=bm_b, block_size_n=bn_b)
bwd_with_max_k = (*bwd, bwd[3].shape[1])  # max_k_blocks 추가

# Flash attention 실행
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
    edge_bias_bwd=bwd_with_max_k,  # 별도 backward 구조체
)
```
