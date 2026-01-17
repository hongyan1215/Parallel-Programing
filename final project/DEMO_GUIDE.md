# 期末專題展示指南
# ===================

## 🎯 三種展示方案

### 方案 1: 快速 Demo（推薦！不需下載模型）
```powershell
# 使用合成資料，立即展示 CUDA kernel 效能
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;" + $env:PATH
python quick_demo.py
```

**優點**：
- ✅ 不需下載任何模型
- ✅ 執行超快（30秒內完成）
- ✅ 完整展示 CUDA kernel 功能和效能
- ✅ 適合現場 demo

**展示內容**：
- 功能正確性驗證
- 三種實作比較（Baseline vs PyTorch vs CUDA）
- 效能提升數據

---

### 方案 2: 真實 LLM Demo（更有說服力）

#### Step 1: 安裝 transformers
```powershell
pip install transformers accelerate sentencepiece
```

#### Step 2: 執行 Demo（會自動下載模型）
```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;" + $env:PATH
python demo_with_real_llm.py
```

**注意**：
- 首次執行會下載 TinyLlama-1.1B（約 2GB）
- 需要約 5-10 分鐘下載時間
- RTX 3060 12GB 完全夠用

**展示內容**：
- 真實的 Speculative Decoding
- Draft model 猜測 → Target model 驗證
- 顯示 acceptance rate
- 實際生成文字

---

### 方案 3: 完整 Benchmark（期末報告數據）

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;" + $env:PATH
python benchmark_cuda_comparison.py
```

**展示內容**：
- 不同 batch size 的效能比較
- 不同 spec_len 的效能比較
- 平均加速比和最大加速比

---

## 📊 期末報告建議結構

### 1. 問題描述（5 分鐘）
- **Speculative Decoding 是什麼**
  - Draft model (小模型) 猜測多個 tokens
  - Target model (大模型) 驗證
  - Rejection Sampling 確保正確性
  
- **為什麼需要優化**
  - Baseline 使用 Python for loop
  - O(K) 次 kernel launches
  - CPU-GPU 同步是瓶頸

### 2. 解決方案（10 分鐘）

#### 方案 1: PyTorch 向量化
```python
# Before: O(K) kernel launches
for k in range(spec_len):
    accept = check_acceptance(k)  # Kernel 1
    if accept:
        output[k] = draft[k]      # Kernel 2
    else:
        break                     # CPU sync!

# After: O(1) kernel launches
accepts = check_all_acceptances()  # Single kernel
first_reject = find_first_false()  # Single kernel
output = gather_accepted()         # Single kernel
```

**結果**: 平均 9.5x 加速

#### 方案 2: CUDA C++ Kernel（你的貢獻！）
```cuda
// 單一 kernel，每個 thread 處理一個 batch
__global__ void fused_rejection_sample_kernel(...) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int k = 0; k < n_draft; k++) {
        if (accepted) {
            output[n_accepted++] = draft[k];
        } else {
            // Resample from adjusted distribution
            output[n_accepted++] = argmax(adjusted);
            break;  // ✅ Break 在 GPU 內！
        }
    }
}
```

**關鍵創新**:
- ✅ 單一 kernel launch
- ✅ Early exit 在 GPU 內完成
- ✅ 無 CPU-GPU 同步

**結果**: 平均 11.95x 加速，最高 35.57x

### 3. 實驗結果（5 分鐘）

**展示 benchmark 圖表**:
```
Batch Size | Baseline | PyTorch | CUDA  | CUDA Speedup
-----------|----------|---------|-------|-------------
1          | 1.48ms   | 1.17ms  | 0.04ms| 35.57x 🔥
4          | 3.21ms   | 1.05ms  | 0.94ms| 3.41x
16         | 14.74ms  | 1.22ms  | 2.11ms| 6.97x
64         | 66.92ms  | 3.04ms  | 3.21ms| 20.87x
```

**重點說明**:
- 小 batch 時 CUDA kernel 優勢明顯（35x!）
- 大 batch 時兩者都很快（kernel overhead 相對不重要）
- 證明了 CPU-GPU 同步確實是瓶頸

### 4. Demo 展示（5 分鐘）

**選項 A: Quick Demo**
```powershell
python quick_demo.py
```
- 展示三種實作的輸出一致性
- 展示效能差異
- 解釋為什麼 CUDA 更快

**選項 B: Real LLM Demo**（如果時間夠）
```powershell
python demo_with_real_llm.py
```
- 展示真實的 Speculative Decoding
- 顯示 acceptance rate
- 實際生成文字

### 5. 結論（2 分鐘）

**成果**:
- ✅ 實作了三種版本的 Rejection Sampler
- ✅ 最高達到 35.57x 加速
- ✅ 真正的 CUDA C++ kernel（不是 Python wrapper）

**學到的**:
- CPU-GPU 同步是效能殺手
- 向量化可以大幅減少 kernel launches
- CUDA kernel 能消除所有同步

**未來改進**:
- 支援更複雜的 sampling 策略（top-k, nucleus）
- 優化記憶體存取模式（shared memory）
- 支援更大的 batch size

---

## 🎬 Demo 腳本建議

### 開場（30 秒）
"大家好，今天要展示的是 Speculative Decoding 中的 Rejection Sampling 優化。
這是 LLM 推理加速的重要技術。"

### 問題說明（1 分鐘）
"Baseline 實作使用 Python for loop，每次迴圈都要 launch kernel 並同步 CPU-GPU。
這造成嚴重的效能瓶頸。"

[展示 baseline 程式碼片段]

### 解決方案（2 分鐘）
"我們開發了兩個解決方案：
1. PyTorch 向量化 - 消除 Python loop
2. CUDA C++ Kernel - 真正的 GPU 融合"

[展示 CUDA kernel 程式碼]

"關鍵是把 early exit 移到 GPU 內部執行"

### 實際展示（2 分鐘）
```powershell
python quick_demo.py
```

"可以看到：
- 三種實作輸出完全一致 ✅
- PyTorch 快了 9.5 倍
- CUDA kernel 快了 11.95 倍，在 batch=1 時甚至達到 35 倍！"

### Q&A 預期問題

**Q: 為什麼不是所有情況都比 PyTorch 快？**
A: 在大 batch size 時，計算量變大，kernel launch overhead 相對不重要，
   所以兩者效能接近。但 CUDA kernel 在小 batch 時優勢明顯。

**Q: 能用在真實的 LLM 嗎？**
A: 可以！我們也實作了真實 LLM 版本（demo_with_real_llm.py），
   使用 TinyLlama-1.1B 進行實際的 Speculative Decoding。

**Q: 這個技術的實際應用是什麼？**
A: Speculative Decoding 被用在很多 LLM serving 系統中，
   例如 vLLM、TensorRT-LLM 等，可以讓大模型推理快 2-3 倍。

---

## 🚀 最後建議

**現場 Demo 用**: `quick_demo.py`
- 不需下載模型
- 執行快速
- 結果清晰

**報告截圖用**: `benchmark_cuda_comparison.py`
- 完整的效能數據
- 可以做成圖表

**加分項**: `demo_with_real_llm.py`
- 如果評審問"能用真實模型嗎"
- 可以說"可以！我們也做了"
- 增加專案完整度

**重點**: 強調你寫的是**真正的 CUDA C++ code**，不是 PyTorch wrapper！

祝你期末報告順利！🎉
