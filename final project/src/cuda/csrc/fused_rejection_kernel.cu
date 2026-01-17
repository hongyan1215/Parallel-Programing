/*
 * Fused CUDA Kernel for Rejection Sampling
 * =========================================
 * 
 * 真正的單一 CUDA Kernel 實作，O(1) kernel launch
 * 
 * 每個 CUDA thread 處理一個 batch element，完全避免：
 * - Python for loop
 * - CPU-GPU 同步
 * - 多次 kernel launch
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Constants
#define PLACEHOLDER_TOKEN_ID -1
#define EPS 1e-10f

// ============================================================================
// CUDA Kernel: Fused Rejection Sampling
// ============================================================================

__global__ void fused_rejection_sample_kernel(
    // Inputs (flattened)
    const float* __restrict__ draft_probs,      // [num_tokens, vocab_size]
    const float* __restrict__ target_probs,     // [num_tokens, vocab_size]
    const int64_t* __restrict__ draft_token_ids, // [num_tokens]
    const int64_t* __restrict__ bonus_token_ids, // [batch_size]
    const int64_t* __restrict__ cu_num_draft,   // [batch_size + 1] cumulative
    const float* __restrict__ uniform_samples,   // [num_tokens]
    
    // Outputs
    int32_t* __restrict__ output_token_ids,     // [batch_size, max_spec_len + 1]
    int32_t* __restrict__ num_accepted,         // [batch_size]
    int32_t* __restrict__ accepted_counts,      // [batch_size]
    int32_t* __restrict__ recovered_counts,     // [batch_size]
    int32_t* __restrict__ bonus_counts,         // [batch_size]
    
    // Dimensions
    const int batch_size,
    const int max_spec_len,
    const int vocab_size
) {
    // 每個 thread 處理一個 batch element
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    // 計算這個 batch 的 token 範圍
    const int64_t start_idx = cu_num_draft[batch_idx];
    const int64_t end_idx = cu_num_draft[batch_idx + 1];
    const int n_draft = end_idx - start_idx;
    
    // 輸出 offset
    const int output_offset = batch_idx * (max_spec_len + 1);
    
    // 初始化輸出為 placeholder
    for (int i = 0; i <= max_spec_len; i++) {
        output_token_ids[output_offset + i] = PLACEHOLDER_TOKEN_ID;
    }
    
    int n_accepted_tokens = 0;
    bool all_accepted = true;
    int first_reject_idx = -1;
    
    // ========================================
    // 核心邏輯：驗證每個 draft token
    // ========================================
    for (int k = 0; k < n_draft; k++) {
        const int token_idx = start_idx + k;
        const int64_t draft_token = draft_token_ids[token_idx];
        
        // 計算機率索引
        const int64_t prob_idx = token_idx * vocab_size + draft_token;
        
        // 取得 draft 和 target 機率
        float p_draft = draft_probs[prob_idx];
        float p_target = target_probs[prob_idx];
        
        // 避免除以零
        p_draft = fmaxf(p_draft, EPS);
        
        // Accept/Reject 決策
        float r = uniform_samples[token_idx];
        float acceptance_prob = fminf(1.0f, p_target / p_draft);
        
        if (r < acceptance_prob) {
            // ✅ ACCEPT
            output_token_ids[output_offset + n_accepted_tokens] = (int32_t)draft_token;
            n_accepted_tokens++;
        } else {
            // ❌ REJECT
            all_accepted = false;
            first_reject_idx = token_idx;
            break;  // Early exit - GPU 內自然處理！
        }
    }
    
    // ========================================
    // 處理 Rejection 或 Bonus
    // ========================================
    if (!all_accepted && first_reject_idx >= 0) {
        // 從 adjusted distribution 重新採樣 (greedy: argmax)
        // adjusted_probs = max(0, target - draft)
        const int64_t base_idx = first_reject_idx * vocab_size;
        
        float max_adj_prob = -1.0f;
        int recovered_token = 0;
        
        // 找 argmax of adjusted probs
        for (int v = 0; v < vocab_size; v++) {
            float adj = fmaxf(0.0f, target_probs[base_idx + v] - draft_probs[base_idx + v]);
            if (adj > max_adj_prob) {
                max_adj_prob = adj;
                recovered_token = v;
            }
        }
        
        output_token_ids[output_offset + n_accepted_tokens] = recovered_token;
        n_accepted_tokens++;
        
        recovered_counts[batch_idx] = 1;
        bonus_counts[batch_idx] = 0;
    } else {
        // 全部 accept，加入 bonus token
        output_token_ids[output_offset + n_accepted_tokens] = (int32_t)bonus_token_ids[batch_idx];
        n_accepted_tokens++;
        
        recovered_counts[batch_idx] = 0;
        bonus_counts[batch_idx] = 1;
    }
    
    // 儲存統計
    num_accepted[batch_idx] = n_accepted_tokens;
    accepted_counts[batch_idx] = all_accepted ? n_draft : (n_accepted_tokens - 1);
}


// ============================================================================
// CUDA Kernel: 使用 cuRAND 的版本（自帶隨機數生成）
// ============================================================================

__global__ void fused_rejection_sample_kernel_with_rng(
    // Inputs
    const float* __restrict__ draft_probs,
    const float* __restrict__ target_probs,
    const int64_t* __restrict__ draft_token_ids,
    const int64_t* __restrict__ bonus_token_ids,
    const int64_t* __restrict__ cu_num_draft,
    curandState* __restrict__ rng_states,
    
    // Outputs
    int32_t* __restrict__ output_token_ids,
    int32_t* __restrict__ num_accepted,
    int32_t* __restrict__ accepted_counts,
    int32_t* __restrict__ recovered_counts,
    int32_t* __restrict__ bonus_counts,
    
    // Dimensions
    const int batch_size,
    const int max_spec_len,
    const int vocab_size
) {
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    // 載入 local RNG state
    curandState local_rng = rng_states[batch_idx];
    
    const int64_t start_idx = cu_num_draft[batch_idx];
    const int64_t end_idx = cu_num_draft[batch_idx + 1];
    const int n_draft = end_idx - start_idx;
    const int output_offset = batch_idx * (max_spec_len + 1);
    
    // 初始化輸出
    for (int i = 0; i <= max_spec_len; i++) {
        output_token_ids[output_offset + i] = PLACEHOLDER_TOKEN_ID;
    }
    
    int n_accepted_tokens = 0;
    bool all_accepted = true;
    int first_reject_idx = -1;
    
    for (int k = 0; k < n_draft; k++) {
        const int token_idx = start_idx + k;
        const int64_t draft_token = draft_token_ids[token_idx];
        const int64_t prob_idx = token_idx * vocab_size + draft_token;
        
        float p_draft = fmaxf(draft_probs[prob_idx], EPS);
        float p_target = target_probs[prob_idx];
        
        // 使用 cuRAND 生成隨機數
        float r = curand_uniform(&local_rng);
        float acceptance_prob = fminf(1.0f, p_target / p_draft);
        
        if (r < acceptance_prob) {
            output_token_ids[output_offset + n_accepted_tokens] = (int32_t)draft_token;
            n_accepted_tokens++;
        } else {
            all_accepted = false;
            first_reject_idx = token_idx;
            break;
        }
    }
    
    if (!all_accepted && first_reject_idx >= 0) {
        const int64_t base_idx = first_reject_idx * vocab_size;
        float max_adj = -1.0f;
        int recovered_token = 0;
        
        for (int v = 0; v < vocab_size; v++) {
            float adj = fmaxf(0.0f, target_probs[base_idx + v] - draft_probs[base_idx + v]);
            if (adj > max_adj) {
                max_adj = adj;
                recovered_token = v;
            }
        }
        
        output_token_ids[output_offset + n_accepted_tokens] = recovered_token;
        n_accepted_tokens++;
        recovered_counts[batch_idx] = 1;
        bonus_counts[batch_idx] = 0;
    } else {
        output_token_ids[output_offset + n_accepted_tokens] = (int32_t)bonus_token_ids[batch_idx];
        n_accepted_tokens++;
        recovered_counts[batch_idx] = 0;
        bonus_counts[batch_idx] = 1;
    }
    
    num_accepted[batch_idx] = n_accepted_tokens;
    accepted_counts[batch_idx] = all_accepted ? n_draft : (n_accepted_tokens - 1);
    
    // 儲存 RNG state
    rng_states[batch_idx] = local_rng;
}


// ============================================================================
// 初始化 RNG states
// ============================================================================

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


// ============================================================================
// C++ Wrapper Functions (供 PyTorch 調用)
// ============================================================================

std::vector<torch::Tensor> fused_rejection_sample_cuda(
    torch::Tensor draft_probs,          // [num_tokens, vocab_size]
    torch::Tensor target_probs,         // [num_tokens, vocab_size]
    torch::Tensor draft_token_ids,      // [num_tokens]
    torch::Tensor bonus_token_ids,      // [batch_size]
    torch::Tensor cu_num_draft,         // [batch_size + 1]
    torch::Tensor uniform_samples,      // [num_tokens]
    int max_spec_len
) {
    // 確保 CUDA tensors
    TORCH_CHECK(draft_probs.is_cuda(), "draft_probs must be CUDA tensor");
    TORCH_CHECK(target_probs.is_cuda(), "target_probs must be CUDA tensor");
    TORCH_CHECK(draft_token_ids.is_cuda(), "draft_token_ids must be CUDA tensor");
    TORCH_CHECK(bonus_token_ids.is_cuda(), "bonus_token_ids must be CUDA tensor");
    TORCH_CHECK(cu_num_draft.is_cuda(), "cu_num_draft must be CUDA tensor");
    TORCH_CHECK(uniform_samples.is_cuda(), "uniform_samples must be CUDA tensor");
    
    // 確保 contiguous
    draft_probs = draft_probs.contiguous();
    target_probs = target_probs.contiguous();
    draft_token_ids = draft_token_ids.contiguous();
    bonus_token_ids = bonus_token_ids.contiguous();
    cu_num_draft = cu_num_draft.contiguous();
    uniform_samples = uniform_samples.contiguous();
    
    const int batch_size = cu_num_draft.size(0) - 1;
    const int vocab_size = draft_probs.size(1);
    
    auto device = draft_probs.device();
    
    // 分配輸出 tensors
    auto output_token_ids = torch::full(
        {batch_size, max_spec_len + 1}, 
        PLACEHOLDER_TOKEN_ID,
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
    auto num_accepted = torch::zeros(
        {batch_size}, 
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
    auto accepted_counts = torch::zeros(
        {batch_size}, 
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
    auto recovered_counts = torch::zeros(
        {batch_size}, 
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
    auto bonus_counts = torch::zeros(
        {batch_size}, 
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    fused_rejection_sample_kernel<<<blocks, threads>>>(
        draft_probs.data_ptr<float>(),
        target_probs.data_ptr<float>(),
        draft_token_ids.data_ptr<int64_t>(),
        bonus_token_ids.data_ptr<int64_t>(),
        cu_num_draft.data_ptr<int64_t>(),
        uniform_samples.data_ptr<float>(),
        output_token_ids.data_ptr<int32_t>(),
        num_accepted.data_ptr<int32_t>(),
        accepted_counts.data_ptr<int32_t>(),
        recovered_counts.data_ptr<int32_t>(),
        bonus_counts.data_ptr<int32_t>(),
        batch_size,
        max_spec_len,
        vocab_size
    );
    
    return {output_token_ids, num_accepted, accepted_counts, recovered_counts, bonus_counts};
}
