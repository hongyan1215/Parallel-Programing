# CUDA Kernel Optimization Report

## 優化版本 V2 性能摘要

### 測試環境
- **GPU**: NVIDIA GeForce RTX 4090
- **CUDA**: 12.8
- **PyTorch**: 2.8.0

### 性能成果

| 指標 | 數值 |
|------|------|
| **平均加速比** | 75.12x |
| **最小加速比** | 23.82x (小 vocab 場景) |
| **最大加速比** | 199.45x (大 batch + 大 vocab) |
| **V2 穩定延遲** | ~26μs |

### 詳細結果

#### 不同模型 Vocab 場景

| 模型類型 | Vocab Size | Batch | 原版 (ms) | V2 (ms) | 加速比 |
|---------|-----------|-------|----------|---------|-------|
| GPT-2 | 50,257 | 1 | 1.01 | 0.026 | 39x |
| TinyLlama | 32,000 | 8 | 0.71 | 0.026 | 28x |
| Llama-3 | 128,256 | 8 | 2.83 | 0.027 | 105x |
| Qwen2.5 | 151,936 | 32 | 6.11 | 0.031 | **199x** |

### 關鍵優化技術

#### 1. 並行架構改進
- **原版**: 每個 thread 處理一整個 batch element，對 vocab 做 O(vocab_size) 的序列 argmax
- **V2**: 每個 block 處理一個 batch，threads 並行掃描 vocab，使用 reduction 找 argmax

#### 2. Warp-Level Primitives
```cuda
__device__ __forceinline__ void warpReduceArgMax(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}
```
- 使用 `__shfl_down_sync` 實現無 shared memory 的 warp 內 reduction
- 比 shared memory reduction 更快，減少 memory barrier

#### 3. Shared Memory 使用最小化
- 只使用 ~256 bytes shared memory (用於 warp 間 reduction)
- 避免了原本 V2 第一版試圖緩存整個 vocab 導致的問題

#### 4. 記憶體訪問優化
- `__restrict__` 指針讓編譯器優化
- Strided access pattern 保持 memory coalescing
- 連續的 thread 存取連續的 vocab indices

#### 5. Early Exit 策略
- 一旦遇到 rejection，立即進入 recovery 階段
- 避免不必要的計算

### 複雜度分析

| 操作 | 原版 | V2 |
|------|------|-----|
| Accept/Reject | O(1) | O(1) |
| Argmax (recovery) | O(vocab_size) | O(vocab_size / num_threads) |
| 總延遲 | ~0.6-6.1ms | ~26μs |

### 影響分析

對於 Speculative Decoding 整體：
- Rejection Sampling 佔 LLM 推理的 ~2%
- 原版 kernel 約 0.6-6ms
- V2 kernel 約 26μs
- 整體推理時間改善約 **0.5-6ms per step**

對於高 throughput 場景 (如 batch=32, Qwen2.5):
- 原版: 6.1ms → V2: 0.03ms
- 節省 6.07ms per decoding step
- 假設 100 tokens 輸出 = 節省 **607ms**

## 文件結構

```
src/cuda/csrc/
├── fused_rejection_kernel.cu      # 原版 kernel (保留作為 fallback)
├── fused_rejection_kernel_v2.cu   # 優化版 V2
├── fused_rejection.cpp            # PyTorch binding
└── setup.py                       # Build script
```

## 使用方式

```python
import fused_rejection_cuda

# 原版 (較慢)
output = fused_rejection_cuda.fused_rejection_sample(
    draft_probs, target_probs, draft_token_ids, 
    bonus_token_ids, cu_num_draft, uniform_samples, max_spec_len
)

# V2 優化版 (推薦)
output = fused_rejection_cuda.fused_rejection_sample_v2(
    draft_probs, target_probs, draft_token_ids, 
    bonus_token_ids, cu_num_draft, uniform_samples, max_spec_len
)
```

## 結論

V2 kernel 成功將 rejection sampling 的延遲降低到 ~26μs，實現了：
- **平均 75x 加速**
- **最高 199x 加速** (大 vocab + 大 batch)
- **穩定的亞毫秒延遲**，幾乎不受 vocab size 和 batch size 影響

這使得 rejection sampling 不再是 speculative decoding 的瓶頸。
