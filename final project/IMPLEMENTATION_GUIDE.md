# 🚀 Speculative Decoding Fused Sampling Kernel 實作指南

## 📋 專案概述

本專案目標是實作一個 **Fused CUDA Kernel** 來優化 Speculative Decoding 的 Rejection Sampling 階段，將原本 O(K) 的 kernel launch overhead 降低為 O(1)。

### 核心問題

在 Speculative Decoding 中，draft model 產生 K 個候選 tokens 後，需要用 target model 驗證並決定接受/拒絕。目前的實作存在以下瓶頸：

```
問題：Python for loop 導致 O(K) 次 kernel launch
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Launch 1│──▶│ Launch 2│──▶│ Launch K│  ← 每次 launch 都有 overhead
└─────────┘   └─────────┘   └─────────┘
     ↓             ↓             ↓
   驗證 t₁       驗證 t₂       驗證 tₖ
```

### 解決方案

```
目標：單一 Fused Kernel，O(1) launch
┌──────────────────────────────────────┐
│         Fused CUDA Kernel            │  ← 只有 1 次 launch overhead
│  ┌────┐ ┌────┐ ┌────┐     ┌────┐   │
│  │ t₁ │→│ t₂ │→│ t₃ │→...→│ tₖ │   │  ← 所有驗證在 GPU 內完成
│  └────┘ └────┘ └────┘     └────┘   │
└──────────────────────────────────────┘
```

---

## 📊 三級實作架構

| Level | 名稱 | 實作方式 | 預期效能 | 用途 |
|-------|------|----------|----------|------|
| **L1** | Baseline | PyTorch for loop | O(K) | 正確性標準 |
| **L2** | Competitor | `@torch.compile` | O(K) | 展示編譯器限制 |
| **L3** | Contribution | Fused CUDA Kernel | **O(1)** | 最終成果 |

---

## 🔬 Rejection Sampling 演算法詳解

### 數學原理（來自論文 [1]）

給定：
- `p(x)` = draft model 的機率分布
- `q(x)` = target model 的機率分布
- `x̂` = draft model 提出的 token

**Accept/Reject 規則：**
```
r ~ Uniform(0, 1)
if r < q(x̂) / p(x̂):
    ACCEPT x̂
else:
    REJECT, resample from: q'(x) = norm(max(0, q(x) - p(x)))
```

### 演算法流程圖

```
                    ┌─────────────────┐
                    │  開始驗證 K tokens │
                    └────────┬────────┘
                             ▼
              ┌──────────────────────────┐
              │  for k = 0 to K-1:       │
              │    token = draft[k]      │
              │    p = draft_prob[token] │
              │    q = target_prob[token]│
              └────────────┬─────────────┘
                           ▼
                  ┌────────────────┐
                  │  r < q/p ?     │
                  └───────┬────────┘
                    ┌─────┴─────┐
                    ▼           ▼
              ┌─────────┐ ┌──────────────┐
              │ ACCEPT  │ │   REJECT     │
              │ token   │ │ + Resample   │
              └────┬────┘ └──────┬───────┘
                   │             │
                   ▼             ▼
              ┌─────────┐ ┌──────────────┐
              │ 繼續下一個│ │  BREAK 迴圈  │
              └────┬────┘ └──────┬───────┘
                   │             │
                   └──────┬──────┘
                          ▼
              ┌─────────────────────────┐
              │  若全部 ACCEPT：        │
              │  從 target 採樣 bonus   │
              └─────────────────────────┘
```

---

## 📁 專案結構

```
pp25_final_project/
├── docs/
│   └── IMPLEMENTATION_GUIDE.md     # 本文件
│
├── spec_decode/                     # vLLM 參考實作（唯讀）
│   ├── eagle.py                     # EAGLE proposer
│   ├── medusa.py                    # Medusa proposer
│   └── ...
│
├── src/
│   ├── __init__.py
│   │
│   ├── baseline/                    # Level 1: Baseline
│   │   ├── __init__.py
│   │   └── rejection_sampler.py     # PyTorch for loop 實作
│   │
│   ├── compiled/                    # Level 2: torch.compile
│   │   ├── __init__.py
│   │   └── rejection_sampler.py     # @torch.compile 版本
│   │
│   └── cuda/                        # Level 3: CUDA Kernel
│       ├── __init__.py
│       ├── fused_sampler.cu         # CUDA kernel 實作
│       ├── fused_sampler.cpp        # PyTorch C++ bindings
│       └── setup.py                 # 編譯腳本
│
├── tests/
│   ├── __init__.py
│   ├── test_correctness.py          # 正確性測試（Golden Standard）
│   └── conftest.py                  # pytest fixtures
│
├── benchmarks/
│   ├── benchmark.py                 # 效能測試主程式
│   ├── plot_results.py              # 繪製 "Money Slide" 圖表
│   └── results/                     # 測試結果輸出
│
├── analysis/
│   ├── nsys_traces/                 # Nsight Systems 分析結果
│   └── compiler_analysis.md         # torch.compile 失敗分析
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 📝 實作步驟詳解

### 第一階段：環境設置與 Baseline（Week 1-2）

#### Step 1.1: 環境配置

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate

# 安裝依賴
pip install torch numpy pytest matplotlib

# 確認 CUDA 可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Step 1.2: 實作 Level 1 Baseline

**檔案**: `src/baseline/rejection_sampler.py`

```python
"""
Level 1 Baseline: Naive PyTorch Implementation
==============================================
這是正確性的「黃金標準」，但效能最差（O(K) kernel launches）
"""

import torch
from typing import Tuple

def rejection_sample_baseline(
    draft_probs: torch.Tensor,      # [batch_size, K, vocab_size]
    target_probs: torch.Tensor,     # [batch_size, K, vocab_size]
    draft_token_ids: torch.Tensor,  # [batch_size, K]
    bonus_probs: torch.Tensor,      # [batch_size, vocab_size] - 用於 bonus token
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    執行 Rejection Sampling 驗證 draft tokens
    
    Args:
        draft_probs: Draft model 在每個位置的機率分布
        target_probs: Target model 在每個位置的機率分布
        draft_token_ids: Draft model 產生的 token IDs
        bonus_probs: 用於採樣 bonus token 的機率分布
    
    Returns:
        accepted_tokens: [batch_size, K+1] 接受的 tokens（含 bonus）
        num_accepted: [batch_size] 每個 batch 接受的 token 數量
    """
    batch_size, K, vocab_size = draft_probs.shape
    device = draft_probs.device
    
    # 輸出 buffers
    accepted_tokens = torch.zeros(batch_size, K + 1, dtype=torch.long, device=device)
    num_accepted = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # 對每個 batch element 獨立處理
    for b in range(batch_size):
        n_accepted = 0
        all_accepted = True
        
        # 驗證 K 個 draft tokens
        for k in range(K):
            token_id = draft_token_ids[b, k].item()
            
            # 取得該 token 的 draft/target 機率
            p_draft = draft_probs[b, k, token_id].item()
            p_target = target_probs[b, k, token_id].item()
            
            # 避免除以零
            if p_draft < 1e-10:
                p_draft = 1e-10
            
            # Accept/Reject 決策
            r = torch.rand(1, device=device).item()
            acceptance_prob = min(1.0, p_target / p_draft)
            
            if r < acceptance_prob:
                # ACCEPT: 加入已接受列表
                accepted_tokens[b, n_accepted] = token_id
                n_accepted += 1
            else:
                # REJECT: 從調整後的分布 resample
                adjusted_probs = torch.clamp(
                    target_probs[b, k] - draft_probs[b, k], 
                    min=0.0
                )
                
                # 正規化
                prob_sum = adjusted_probs.sum()
                if prob_sum > 1e-10:
                    adjusted_probs = adjusted_probs / prob_sum
                else:
                    # Fallback: 使用 target distribution
                    adjusted_probs = target_probs[b, k]
                
                # Resample
                resampled_token = torch.multinomial(adjusted_probs, 1).item()
                accepted_tokens[b, n_accepted] = resampled_token
                n_accepted += 1
                all_accepted = False
                break  # ⚠️ EARLY EXIT - 這是關鍵的動態控制流！
        
        # 若全部接受，採樣 bonus token
        if all_accepted:
            bonus_token = torch.multinomial(bonus_probs[b], 1).item()
            accepted_tokens[b, n_accepted] = bonus_token
            n_accepted += 1
        
        num_accepted[b] = n_accepted
    
    return accepted_tokens, num_accepted
```

#### Step 1.3: 建立測試套件（Golden Standard）

**檔案**: `tests/test_correctness.py`

```python
"""
Correctness Tests (Golden Standard)
===================================
所有實作都必須通過這些測試
"""

import pytest
import torch
from src.baseline.rejection_sampler import rejection_sample_baseline

# 固定隨機種子以確保可重現性
SEED = 42

@pytest.fixture
def sample_data():
    """生成測試用的模擬資料"""
    torch.manual_seed(SEED)
    
    batch_size = 4
    K = 8
    vocab_size = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 生成隨機 logits 並轉換為機率
    draft_logits = torch.randn(batch_size, K, vocab_size, device=device)
    target_logits = torch.randn(batch_size, K, vocab_size, device=device)
    
    draft_probs = torch.softmax(draft_logits, dim=-1)
    target_probs = torch.softmax(target_logits, dim=-1)
    
    # 從 draft distribution 採樣 token ids
    draft_token_ids = torch.stack([
        torch.multinomial(draft_probs[:, k, :], 1).squeeze(-1)
        for k in range(K)
    ], dim=1)
    
    # Bonus probs
    bonus_probs = torch.softmax(
        torch.randn(batch_size, vocab_size, device=device), 
        dim=-1
    )
    
    return {
        "draft_probs": draft_probs,
        "target_probs": target_probs,
        "draft_token_ids": draft_token_ids,
        "bonus_probs": bonus_probs,
        "batch_size": batch_size,
        "K": K,
        "vocab_size": vocab_size,
    }


class TestBaseline:
    """測試 Level 1 Baseline"""
    
    def test_output_shape(self, sample_data):
        """確認輸出形狀正確"""
        result, num_accepted = rejection_sample_baseline(
            sample_data["draft_probs"],
            sample_data["target_probs"],
            sample_data["draft_token_ids"],
            sample_data["bonus_probs"],
        )
        
        batch_size = sample_data["batch_size"]
        K = sample_data["K"]
        
        assert result.shape == (batch_size, K + 1)
        assert num_accepted.shape == (batch_size,)
    
    def test_num_accepted_range(self, sample_data):
        """確認接受數量在合理範圍內"""
        _, num_accepted = rejection_sample_baseline(
            sample_data["draft_probs"],
            sample_data["target_probs"],
            sample_data["draft_token_ids"],
            sample_data["bonus_probs"],
        )
        
        K = sample_data["K"]
        
        # 至少接受 1 個（reject 後的 resample）
        # 最多接受 K+1 個（全部 accept + bonus）
        assert (num_accepted >= 1).all()
        assert (num_accepted <= K + 1).all()
    
    def test_accepted_tokens_valid(self, sample_data):
        """確認接受的 tokens 都是有效的 vocab indices"""
        result, num_accepted = rejection_sample_baseline(
            sample_data["draft_probs"],
            sample_data["target_probs"],
            sample_data["draft_token_ids"],
            sample_data["bonus_probs"],
        )
        
        vocab_size = sample_data["vocab_size"]
        
        # 只檢查實際接受的 tokens
        for b in range(sample_data["batch_size"]):
            n = num_accepted[b].item()
            valid_tokens = result[b, :n]
            assert (valid_tokens >= 0).all()
            assert (valid_tokens < vocab_size).all()
    
    def test_deterministic_with_seed(self, sample_data):
        """確認固定種子時結果可重現"""
        torch.manual_seed(SEED)
        result1, num1 = rejection_sample_baseline(
            sample_data["draft_probs"],
            sample_data["target_probs"],
            sample_data["draft_token_ids"],
            sample_data["bonus_probs"],
        )
        
        torch.manual_seed(SEED)
        result2, num2 = rejection_sample_baseline(
            sample_data["draft_probs"],
            sample_data["target_probs"],
            sample_data["draft_token_ids"],
            sample_data["bonus_probs"],
        )
        
        assert torch.equal(num1, num2)


class TestCompareImplementations:
    """比較不同實作的正確性"""
    
    def test_compiled_matches_baseline(self, sample_data):
        """Level 2 應與 Level 1 結果一致（統計上）"""
        # TODO: 實作 Level 2 後啟用
        pass
    
    def test_cuda_matches_baseline(self, sample_data):
        """Level 3 應與 Level 1 結果一致（統計上）"""
        # TODO: 實作 Level 3 後啟用
        pass
```

---

### 第二階段：torch.compile 分析（Week 2）

#### Step 2.1: 實作 Level 2

**檔案**: `src/compiled/rejection_sampler.py`

```python
"""
Level 2: torch.compile Version
==============================
展示 SOTA 編譯器在動態控制流上的限制
"""

import torch
from typing import Tuple

@torch.compile(mode="reduce-overhead")
def rejection_sample_compiled(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    bonus_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    與 baseline 相同邏輯，但加上 @torch.compile
    
    預期結果：torch.compile 無法有效融合這個函數，
    因為存在 data-dependent 的 break 控制流
    """
    batch_size, K, vocab_size = draft_probs.shape
    device = draft_probs.device
    
    accepted_tokens = torch.zeros(batch_size, K + 1, dtype=torch.long, device=device)
    num_accepted = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        n_accepted = 0
        all_accepted = True
        
        for k in range(K):
            token_id = draft_token_ids[b, k]
            p_draft = draft_probs[b, k, token_id]
            p_target = target_probs[b, k, token_id]
            
            p_draft = torch.clamp(p_draft, min=1e-10)
            r = torch.rand(1, device=device)
            acceptance_prob = torch.minimum(
                torch.ones(1, device=device), 
                p_target / p_draft
            )
            
            if r < acceptance_prob:
                accepted_tokens[b, n_accepted] = token_id
                n_accepted += 1
            else:
                adjusted_probs = torch.clamp(
                    target_probs[b, k] - draft_probs[b, k], 
                    min=0.0
                )
                prob_sum = adjusted_probs.sum()
                adjusted_probs = torch.where(
                    prob_sum > 1e-10,
                    adjusted_probs / prob_sum,
                    target_probs[b, k]
                )
                resampled_token = torch.multinomial(adjusted_probs, 1)
                accepted_tokens[b, n_accepted] = resampled_token.squeeze()
                n_accepted += 1
                all_accepted = False
                break  # ⚠️ 這個 break 會導致 graph break!
        
        if all_accepted:
            bonus_token = torch.multinomial(bonus_probs[b], 1)
            accepted_tokens[b, n_accepted] = bonus_token.squeeze()
            n_accepted += 1
        
        num_accepted[b] = n_accepted
    
    return accepted_tokens, num_accepted
```

#### Step 2.2: 分析 torch.compile 的限制

**檔案**: `analysis/compiler_analysis.md`

```markdown
# torch.compile 編譯器分析

## 問題：Graph Breaks

當使用 `TORCH_LOGS="graph_breaks" python benchmark.py` 執行時，
會看到類似以下的輸出：

```
[graph_break] Dynamic control flow: data-dependent break statement
  File "rejection_sampler.py", line XX
    break  # ⚠️ 這個 break 會導致 graph break!
```

## 為什麼 torch.compile 無法處理？

1. **Data-dependent Control Flow**: 
   - `if r < acceptance_prob` 的結果在編譯時未知
   - `break` 語句會根據運行時數據提前退出

2. **Graph 分裂**:
   - 每次 `break` 都會產生一個新的 graph
   - 導致實際上仍然是 O(K) 次 kernel launch

3. **無法向量化**:
   - 不同 batch elements 可能在不同位置 reject
   - 無法簡單地用 SIMD 處理

## 結論

手動 CUDA kernel 是必要的，因為：
- 可以在單一 kernel 內處理所有動態邏輯
- 每個 thread 獨立處理一個 batch element
- 使用 on-device RNG 避免 CPU 往返
```

---

### 第三階段：CUDA Kernel 實作（Week 3-4）

#### Step 3.1: CUDA Kernel 設計

**檔案**: `src/cuda/fused_sampler.cu`

```cpp
/*
 * Level 3: Fused CUDA Kernel for Rejection Sampling
 * ==================================================
 * 
 * 設計目標：
 * 1. 單一 kernel launch 處理整個 batch
 * 2. 每個 thread 處理一個 batch element
 * 3. On-device RNG (curand)
 * 4. 正確處理 variable-length output
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA 錯誤檢查 macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 初始化 RNG states
__global__ void init_rng_kernel(
    curandState* states,
    unsigned long long seed,
    int num_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// 主要的 Fused Rejection Sampling Kernel
__global__ void fused_rejection_sample_kernel(
    const float* __restrict__ draft_probs,      // [batch, K, vocab]
    const float* __restrict__ target_probs,     // [batch, K, vocab]
    const int64_t* __restrict__ draft_token_ids, // [batch, K]
    const float* __restrict__ bonus_probs,      // [batch, vocab]
    int64_t* __restrict__ accepted_tokens,      // [batch, K+1] output
    int64_t* __restrict__ num_accepted,         // [batch] output
    curandState* __restrict__ rng_states,
    const int batch_size,
    const int K,
    const int vocab_size
) {
    // 每個 thread 處理一個 batch element
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    // 載入 local RNG state
    curandState local_rng = rng_states[batch_idx];
    
    // 計算 base offsets
    const int prob_batch_offset = batch_idx * K * vocab_size;
    const int token_batch_offset = batch_idx * K;
    const int output_batch_offset = batch_idx * (K + 1);
    const int bonus_batch_offset = batch_idx * vocab_size;
    
    int n_accepted = 0;
    bool all_accepted = true;
    
    // ============================================
    // 核心邏輯：驗證 K 個 draft tokens
    // ============================================
    for (int k = 0; k < K; k++) {
        const int token_id = static_cast<int>(draft_token_ids[token_batch_offset + k]);
        
        // 取得該 token 的機率
        const int prob_offset = prob_batch_offset + k * vocab_size + token_id;
        float p_draft = draft_probs[prob_offset];
        float p_target = target_probs[prob_offset];
        
        // 避免除以零
        p_draft = fmaxf(p_draft, 1e-10f);
        
        // Accept/Reject 決策
        float r = curand_uniform(&local_rng);
        float acceptance_prob = fminf(1.0f, p_target / p_draft);
        
        if (r < acceptance_prob) {
            // ACCEPT
            accepted_tokens[output_batch_offset + n_accepted] = token_id;
            n_accepted++;
        } else {
            // REJECT: Resample from adjusted distribution
            // 計算 adjusted_probs = max(0, target - draft)
            
            const int k_prob_offset = prob_batch_offset + k * vocab_size;
            float prob_sum = 0.0f;
            
            // 第一遍：計算 sum
            for (int v = 0; v < vocab_size; v++) {
                float adj = fmaxf(0.0f, target_probs[k_prob_offset + v] 
                                      - draft_probs[k_prob_offset + v]);
                prob_sum += adj;
            }
            
            // Multinomial sampling from adjusted distribution
            float u = curand_uniform(&local_rng) * prob_sum;
            float cumsum = 0.0f;
            int resampled_token = 0;
            
            for (int v = 0; v < vocab_size; v++) {
                float adj = fmaxf(0.0f, target_probs[k_prob_offset + v] 
                                      - draft_probs[k_prob_offset + v]);
                cumsum += adj;
                if (cumsum >= u) {
                    resampled_token = v;
                    break;
                }
            }
            
            accepted_tokens[output_batch_offset + n_accepted] = resampled_token;
            n_accepted++;
            all_accepted = false;
            break;  // EARLY EXIT - 在 GPU 內自然處理！
        }
    }
    
    // ============================================
    // 若全部 accept，採樣 bonus token
    // ============================================
    if (all_accepted) {
        // Multinomial sampling from bonus_probs
        float u = curand_uniform(&local_rng);
        float cumsum = 0.0f;
        int bonus_token = 0;
        
        for (int v = 0; v < vocab_size; v++) {
            cumsum += bonus_probs[bonus_batch_offset + v];
            if (cumsum >= u) {
                bonus_token = v;
                break;
            }
        }
        
        accepted_tokens[output_batch_offset + n_accepted] = bonus_token;
        n_accepted++;
    }
    
    // 儲存接受數量
    num_accepted[batch_idx] = n_accepted;
    
    // 儲存 RNG state
    rng_states[batch_idx] = local_rng;
}


// ============================================
// PyTorch C++ Extension Interface
// ============================================

class FusedRejectionSampler {
public:
    FusedRejectionSampler(int max_batch_size, unsigned long long seed = 42) 
        : max_batch_size_(max_batch_size), initialized_(false) {
        
        // 分配 RNG states
        CUDA_CHECK(cudaMalloc(&rng_states_, max_batch_size * sizeof(curandState)));
        
        // 初始化 RNG
        int threads = 256;
        int blocks = (max_batch_size + threads - 1) / threads;
        init_rng_kernel<<<blocks, threads>>>(rng_states_, seed, max_batch_size);
        CUDA_CHECK(cudaGetLastError());
        
        initialized_ = true;
    }
    
    ~FusedRejectionSampler() {
        if (initialized_) {
            cudaFree(rng_states_);
        }
    }
    
    std::tuple<torch::Tensor, torch::Tensor> sample(
        torch::Tensor draft_probs,
        torch::Tensor target_probs,
        torch::Tensor draft_token_ids,
        torch::Tensor bonus_probs
    ) {
        const int batch_size = draft_probs.size(0);
        const int K = draft_probs.size(1);
        const int vocab_size = draft_probs.size(2);
        
        // 確保輸入在 GPU 上
        TORCH_CHECK(draft_probs.is_cuda(), "draft_probs must be on CUDA");
        TORCH_CHECK(target_probs.is_cuda(), "target_probs must be on CUDA");
        TORCH_CHECK(draft_token_ids.is_cuda(), "draft_token_ids must be on CUDA");
        TORCH_CHECK(bonus_probs.is_cuda(), "bonus_probs must be on CUDA");
        
        // 確保 contiguous
        draft_probs = draft_probs.contiguous();
        target_probs = target_probs.contiguous();
        draft_token_ids = draft_token_ids.contiguous();
        bonus_probs = bonus_probs.contiguous();
        
        // 分配輸出 tensors
        auto options_long = torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(draft_probs.device());
        
        torch::Tensor accepted_tokens = torch::zeros({batch_size, K + 1}, options_long);
        torch::Tensor num_accepted = torch::zeros({batch_size}, options_long);
        
        // Launch kernel
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        
        fused_rejection_sample_kernel<<<blocks, threads>>>(
            draft_probs.data_ptr<float>(),
            target_probs.data_ptr<float>(),
            draft_token_ids.data_ptr<int64_t>(),
            bonus_probs.data_ptr<float>(),
            accepted_tokens.data_ptr<int64_t>(),
            num_accepted.data_ptr<int64_t>(),
            rng_states_,
            batch_size,
            K,
            vocab_size
        );
        
        CUDA_CHECK(cudaGetLastError());
        
        return std::make_tuple(accepted_tokens, num_accepted);
    }

private:
    int max_batch_size_;
    curandState* rng_states_;
    bool initialized_;
};


// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<FusedRejectionSampler>(m, "FusedRejectionSampler")
        .def(pybind11::init<int, unsigned long long>(),
             pybind11::arg("max_batch_size"),
             pybind11::arg("seed") = 42)
        .def("sample", &FusedRejectionSampler::sample);
}
```

#### Step 3.2: 編譯腳本

**檔案**: `src/cuda/setup.py`

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_sampler',
    ext_modules=[
        CUDAExtension(
            name='fused_sampler',
            sources=['fused_sampler.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

編譯指令：
```bash
cd src/cuda
python setup.py install
```

---

### 第四階段：效能測試與分析（Week 5-6）

#### Step 4.1: Benchmark 腳本

**檔案**: `benchmarks/benchmark.py`

```python
"""
Performance Benchmark
=====================
比較三個實作層級的效能
"""

import torch
import time
import json
import argparse
from pathlib import Path

# Import implementations
from src.baseline.rejection_sampler import rejection_sample_baseline
from src.compiled.rejection_sampler import rejection_sample_compiled
import fused_sampler  # CUDA extension


def generate_test_data(batch_size: int, K: int, vocab_size: int, device: str):
    """生成測試資料"""
    draft_logits = torch.randn(batch_size, K, vocab_size, device=device)
    target_logits = torch.randn(batch_size, K, vocab_size, device=device)
    
    draft_probs = torch.softmax(draft_logits, dim=-1)
    target_probs = torch.softmax(target_logits, dim=-1)
    
    draft_token_ids = torch.stack([
        torch.multinomial(draft_probs[:, k, :], 1).squeeze(-1)
        for k in range(K)
    ], dim=1)
    
    bonus_probs = torch.softmax(
        torch.randn(batch_size, vocab_size, device=device),
        dim=-1
    )
    
    return draft_probs, target_probs, draft_token_ids, bonus_probs


def benchmark_function(fn, args, warmup: int = 10, iterations: int = 100):
    """測量函數執行時間"""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        fn(*args)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event) / iterations
    return elapsed_ms * 1000  # Convert to µs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/benchmark.json")
    args = parser.parse_args()
    
    device = "cuda"
    K_values = [2, 4, 8, 16]
    
    results = {
        "config": {
            "batch_size": args.batch_size,
            "vocab_size": args.vocab_size,
            "iterations": args.iterations,
        },
        "data": []
    }
    
    # 初始化 CUDA sampler
    cuda_sampler = fused_sampler.FusedRejectionSampler(args.batch_size)
    
    print("\n" + "=" * 70)
    print("📊 SPECULATIVE DECODING REJECTION SAMPLER BENCHMARK")
    print("=" * 70)
    print(f"Configuration: batch_size={args.batch_size}, vocab_size={args.vocab_size}")
    print("-" * 70)
    print(f"{'K':<6} {'L1 Baseline (µs)':<20} {'L2 Compile (µs)':<20} {'L3 CUDA (µs)':<20}")
    print("-" * 70)
    
    for K in K_values:
        # 生成測試資料
        data = generate_test_data(args.batch_size, K, args.vocab_size, device)
        
        # Level 1: Baseline
        t1 = benchmark_function(
            rejection_sample_baseline, 
            data, 
            iterations=args.iterations
        )
        
        # Level 2: torch.compile
        t2 = benchmark_function(
            rejection_sample_compiled,
            data,
            iterations=args.iterations
        )
        
        # Level 3: CUDA Kernel
        t3 = benchmark_function(
            cuda_sampler.sample,
            data,
            iterations=args.iterations
        )
        
        print(f"{K:<6} {t1:<20.2f} {t2:<20.2f} {t3:<20.2f}")
        
        results["data"].append({
            "K": K,
            "L1_baseline_us": t1,
            "L2_compile_us": t2,
            "L3_cuda_us": t3,
            "speedup_vs_baseline": t1 / t3,
            "speedup_vs_compile": t2 / t3,
        })
    
    print("-" * 70)
    
    # 儲存結果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")
    
    # 顯示 speedup 摘要
    print("\n📈 SPEEDUP SUMMARY:")
    print("-" * 40)
    for entry in results["data"]:
        print(f"K={entry['K']}: "
              f"{entry['speedup_vs_baseline']:.1f}x vs baseline, "
              f"{entry['speedup_vs_compile']:.1f}x vs compile")


if __name__ == "__main__":
    main()
```

#### Step 4.2: 繪製結果圖表（Money Slide）

**檔案**: `benchmarks/plot_results.py`

```python
"""
Generate "Money Slide" Performance Graph
========================================
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_results(results_path: str, output_path: str = "results/performance.png"):
    with open(results_path) as f:
        results = json.load(f)
    
    data = results["data"]
    K_values = [d["K"] for d in data]
    baseline = [d["L1_baseline_us"] for d in data]
    compiled = [d["L2_compile_us"] for d in data]
    cuda = [d["L3_cuda_us"] for d in data]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(K_values, baseline, 'o-', linewidth=2, markersize=8, 
             label='L1: PyTorch Baseline', color='#e74c3c')
    plt.plot(K_values, compiled, 's-', linewidth=2, markersize=8,
             label='L2: torch.compile', color='#f39c12')
    plt.plot(K_values, cuda, '^-', linewidth=2, markersize=8,
             label='L3: Fused CUDA Kernel', color='#27ae60')
    
    plt.xlabel('K (Number of Draft Tokens)', fontsize=12)
    plt.ylabel('Latency (µs)', fontsize=12)
    plt.title('Rejection Sampling Performance: O(K) vs O(1)', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(K_values)
    
    # 標註 speedup
    for i, k in enumerate(K_values):
        speedup = baseline[i] / cuda[i]
        plt.annotate(f'{speedup:.1f}x', 
                    xy=(k, cuda[i]), 
                    xytext=(k + 0.3, cuda[i] + 50),
                    fontsize=9, color='#27ae60')
    
    plt.tight_layout()
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"✅ Plot saved to {output}")


if __name__ == "__main__":
    plot_results("results/benchmark.json")
```

---

## ✅ 檢查清單

### Week 1-2
- [ ] 環境設置完成
- [ ] Level 1 Baseline 實作完成
- [ ] Unit tests (Golden Standard) 通過
- [ ] Level 2 torch.compile 版本實作
- [ ] 確認 torch.compile 有 graph breaks

### Week 3-4
- [ ] CUDA kernel 架構設計
- [ ] curand RNG 整合
- [ ] Kernel 編譯成功
- [ ] 通過 correctness tests
- [ ] 處理 edge cases

### Week 5-6
- [ ] Benchmark 腳本完成
- [ ] 效能數據收集
- [ ] Money Slide 圖表產生
- [ ] nsys trace 分析
- [ ] 報告撰寫完成

---

## 📚 參考資源

1. **論文**: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
2. **vLLM 原始碼**: `spec_decode/` 資料夾
3. **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
4. **PyTorch C++ Extension**: https://pytorch.org/tutorials/advanced/cpp_extension.html
5. **cuRAND Library**: https://docs.nvidia.com/cuda/curand/

---

## 🤔 常見問題

### Q: 為什麼不能用 `torch.compile` 解決這個問題？
A: `torch.compile` 無法處理 data-dependent control flow（如 `break`）。當遇到這種情況時，它會產生 "graph break"，導致仍然是 O(K) 次 kernel launch。

### Q: CUDA kernel 如何處理 variable-length output？
A: 每個 thread 使用自己的 counter (`n_accepted`)，最後寫入 `num_accepted[batch_idx]`。呼叫者根據這個值知道每個 batch element 實際產生了多少 tokens。

### Q: 如何確保 correctness？
A: 使用固定隨機種子，確保 baseline 和 CUDA kernel 在統計上產生相同分布的輸出。具體通過 chi-squared test 或 KL divergence 驗證。

---

*最後更新: 2025年11月*
