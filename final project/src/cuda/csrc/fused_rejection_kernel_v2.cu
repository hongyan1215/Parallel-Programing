/*
 * Optimized Fused CUDA Kernel for Rejection Sampling - Version 2
 * ===============================================================
 * 
 * 優化策略：
 * 1. 每個 block 處理一個 batch element（vs 原版每個 thread 處理一個）
 * 2. 並行 reduction 找 argmax（使用 warp-level primitives）
 * 3. Shared memory 用於 warp 間的 reduction
 * 4. 減少 global memory 的隨機訪問
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constants
#define PLACEHOLDER_TOKEN_ID -1
#define EPS 1e-10f
#define WARP_SIZE 32

// ============================================================================
// Utility: Warp-level reduction for finding argmax
// ============================================================================
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

// ============================================================================
// Optimized Kernel V2: 每個 block 處理一個 batch，threads 並行處理 vocab
// ============================================================================
template <int BLOCK_SIZE>
__global__ void fused_rejection_sample_v2_kernel(
    const float* __restrict__ draft_probs,      // [num_tokens, vocab_size]
    const float* __restrict__ target_probs,     // [num_tokens, vocab_size]
    const int64_t* __restrict__ draft_token_ids, // [num_tokens]
    const int64_t* __restrict__ bonus_token_ids, // [batch_size]
    const int64_t* __restrict__ cu_num_draft,   // [batch_size + 1]
    const float* __restrict__ uniform_samples,   // [num_tokens]
    
    int32_t* __restrict__ output_token_ids,     // [batch_size, max_spec_len + 1]
    int32_t* __restrict__ num_accepted,         // [batch_size]
    int32_t* __restrict__ accepted_counts,      // [batch_size]
    int32_t* __restrict__ recovered_counts,     // [batch_size]
    int32_t* __restrict__ bonus_counts,         // [batch_size]
    
    const int batch_size,
    const int max_spec_len,
    const int vocab_size
) {
    // 每個 block 處理一個 batch element
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    constexpr int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // Shared memory for reduction (只需要很小的空間)
    __shared__ float s_max_vals[32];
    __shared__ int s_max_idxs[32];
    __shared__ int s_accepted_count;
    __shared__ bool s_early_exit;
    __shared__ int s_recovered_token;
    
    // 計算這個 batch 的 token 範圍
    const int64_t start_idx = cu_num_draft[batch_idx];
    const int64_t end_idx = cu_num_draft[batch_idx + 1];
    const int n_draft = end_idx - start_idx;
    const int output_offset = batch_idx * (max_spec_len + 1);
    
    // 初始化 (只有 thread 0)
    if (tid == 0) {
        s_accepted_count = 0;
        s_early_exit = false;
        // 初始化輸出為 placeholder
        for (int i = 0; i <= max_spec_len; i++) {
            output_token_ids[output_offset + i] = PLACEHOLDER_TOKEN_ID;
        }
    }
    __syncthreads();
    
    bool all_accepted = true;
    int first_reject_idx = -1;
    
    // ========================================
    // Phase 1: 順序處理每個 draft token (Accept/Reject)
    // ========================================
    for (int k = 0; k < n_draft && !s_early_exit; k++) {
        __syncthreads();
        
        const int token_idx = start_idx + k;
        const int64_t draft_token = draft_token_ids[token_idx];
        const int64_t prob_idx = token_idx * vocab_size + draft_token;
        
        // Thread 0 執行 accept/reject 決策
        if (tid == 0) {
            float p_draft = fmaxf(draft_probs[prob_idx], EPS);
            float p_target = target_probs[prob_idx];
            float r = uniform_samples[token_idx];
            float acceptance_prob = fminf(1.0f, p_target / p_draft);
            
            if (r < acceptance_prob) {
                // ACCEPT
                output_token_ids[output_offset + s_accepted_count] = (int32_t)draft_token;
                s_accepted_count++;
            } else {
                // REJECT - 需要 resample
                s_early_exit = true;
            }
        }
        __syncthreads();
        
        // 更新 local 變數
        if (s_early_exit) {
            all_accepted = false;
            first_reject_idx = token_idx;
            break;
        }
    }
    __syncthreads();
    
    // ========================================
    // Phase 2: 處理 Rejection (並行找 argmax)
    // ========================================
    if (!all_accepted && s_early_exit) {
        const int64_t base_idx = first_reject_idx * vocab_size;
        
        // 每個 thread 處理一部分 vocab，找 local max
        float local_max = -1.0f;
        int local_idx = 0;
        
        for (int v = tid; v < vocab_size; v += BLOCK_SIZE) {
            float adj = fmaxf(0.0f, target_probs[base_idx + v] - draft_probs[base_idx + v]);
            if (adj > local_max) {
                local_max = adj;
                local_idx = v;
            }
        }
        
        // Warp-level reduction
        warpReduceArgMax(local_max, local_idx);
        
        // 存儲每個 warp 的結果
        if (lane == 0) {
            s_max_vals[warp_id] = local_max;
            s_max_idxs[warp_id] = local_idx;
        }
        __syncthreads();
        
        // 第一個 warp 做最終 reduction
        if (warp_id == 0) {
            local_max = (lane < num_warps) ? s_max_vals[lane] : -1.0f;
            local_idx = (lane < num_warps) ? s_max_idxs[lane] : 0;
            
            warpReduceArgMax(local_max, local_idx);
            
            if (lane == 0) {
                s_recovered_token = local_idx;
            }
        }
        __syncthreads();
        
        // 寫入結果
        if (tid == 0) {
            output_token_ids[output_offset + s_accepted_count] = s_recovered_token;
            s_accepted_count++;
            recovered_counts[batch_idx] = 1;
            bonus_counts[batch_idx] = 0;
        }
    } else {
        // 全部 accept，加入 bonus token
        if (tid == 0) {
            output_token_ids[output_offset + s_accepted_count] = (int32_t)bonus_token_ids[batch_idx];
            s_accepted_count++;
            recovered_counts[batch_idx] = 0;
            bonus_counts[batch_idx] = 1;
        }
    }
    __syncthreads();
    
    // 儲存統計
    if (tid == 0) {
        num_accepted[batch_idx] = s_accepted_count;
        accepted_counts[batch_idx] = all_accepted ? n_draft : (s_accepted_count - 1);
    }
}

// ============================================================================
// C++ Wrapper: 優化版本 V2
// ============================================================================
std::vector<torch::Tensor> fused_rejection_sample_v2_cuda(
    torch::Tensor draft_probs,
    torch::Tensor target_probs,
    torch::Tensor draft_token_ids,
    torch::Tensor bonus_token_ids,
    torch::Tensor cu_num_draft,
    torch::Tensor uniform_samples,
    int max_spec_len
) {
    TORCH_CHECK(draft_probs.is_cuda(), "draft_probs must be CUDA tensor");
    
    draft_probs = draft_probs.contiguous();
    target_probs = target_probs.contiguous();
    draft_token_ids = draft_token_ids.contiguous();
    bonus_token_ids = bonus_token_ids.contiguous();
    cu_num_draft = cu_num_draft.contiguous();
    uniform_samples = uniform_samples.contiguous();
    
    const int batch_size = cu_num_draft.size(0) - 1;
    const int vocab_size = draft_probs.size(1);
    auto device = draft_probs.device();
    
    auto output_token_ids = torch::full(
        {batch_size, max_spec_len + 1}, 
        PLACEHOLDER_TOKEN_ID,
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
    auto num_accepted = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto accepted_counts = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto recovered_counts = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto bonus_counts = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    constexpr int BLOCK_SIZE = 256;
    const int num_blocks = batch_size;
    
    // 統一使用 v2 kernel（不需要大量 shared memory）
    fused_rejection_sample_v2_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
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
        batch_size, max_spec_len, vocab_size
    );
    
    return {output_token_ids, num_accepted, accepted_counts, recovered_counts, bonus_counts};
}
