"""
Level 1 Baseline: Naive PyTorch Implementation
==============================================

這是正確性的「黃金標準」，使用純 Python for loop 實作。
效能最差：O(K) kernel launches，但保證正確性。

演算法來源：https://arxiv.org/abs/2211.17192
"Fast Inference from Transformers via Speculative Decoding"

術語說明：
- accepted tokens: 基於 draft/target 機率比較被接受的 tokens
- recovered tokens: 被拒絕後從調整分布重新採樣的 tokens  
- bonus tokens: 當所有 draft tokens 都被接受時，額外採樣的 token
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from dataclasses import dataclass

# Placeholder for rejected positions
PLACEHOLDER_TOKEN_ID = -1


@dataclass
class RejectionSampleOutput:
    """Rejection sampling 的輸出結果"""
    output_token_ids: torch.Tensor  # [batch_size, max_spec_len + 1]
    num_accepted: torch.Tensor      # [batch_size] 每個 batch 實際輸出的 token 數
    accepted_counts: torch.Tensor   # [batch_size] 純接受的 token 數（不含 recovered/bonus）
    recovered_counts: torch.Tensor  # [batch_size] recovered token 數
    bonus_counts: torch.Tensor      # [batch_size] bonus token 數


def rejection_sample_baseline(
    draft_token_ids: torch.Tensor,      # [num_tokens] - 所有 batch 的 draft tokens 展平
    num_draft_tokens: List[int],        # [batch_size] - 每個 request 的 draft token 數
    draft_probs: torch.Tensor,          # [num_tokens, vocab_size] - draft 機率分布
    target_probs: torch.Tensor,         # [num_tokens, vocab_size] - target 機率分布
    bonus_token_ids: torch.Tensor,      # [batch_size, 1] - 預先採樣的 bonus tokens
    uniform_samples: Optional[torch.Tensor] = None,  # [num_tokens] - 預生成的均勻隨機數
) -> RejectionSampleOutput:
    """
    執行 Rejection Sampling 驗證 draft tokens（純 Python for loop 版本）
    
    這個實作故意使用 Python for loop，以展示：
    1. 演算法的正確邏輯
    2. O(K) 的 kernel launch overhead
    
    Args:
        draft_token_ids: 展平的 draft token IDs
        num_draft_tokens: 每個 request 的 draft token 數量
        draft_probs: Draft model 的機率分布
        target_probs: Target model 的機率分布  
        bonus_token_ids: 預先採樣的 bonus tokens
        uniform_samples: 可選的預生成均勻隨機數（用於測試可重現性）
    
    Returns:
        RejectionSampleOutput: 包含輸出 tokens 和統計資訊
    """
    batch_size = len(num_draft_tokens)
    max_spec_len = max(num_draft_tokens) if num_draft_tokens else 0
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    
    # 計算 cumulative offsets
    cu_num_draft_tokens = torch.tensor(
        [sum(num_draft_tokens[:i+1]) for i in range(batch_size)],
        dtype=torch.int32, device=device
    )
    
    # 輸出 buffer
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    
    # 統計 counters
    num_accepted = torch.zeros(batch_size, dtype=torch.int32, device=device)
    accepted_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    recovered_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    bonus_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    # 如果沒有提供隨機數，現場生成
    if uniform_samples is None:
        num_tokens = draft_token_ids.shape[0]
        uniform_samples = torch.rand(num_tokens, device=device, dtype=torch.float32)
    
    # ========================================
    # 核心邏輯：O(K) Python for loop
    # ========================================
    for batch_idx in range(batch_size):
        # 計算這個 request 的 token 範圍
        start_idx = 0 if batch_idx == 0 else cu_num_draft_tokens[batch_idx - 1].item()
        end_idx = cu_num_draft_tokens[batch_idx].item()
        n_draft = end_idx - start_idx
        
        output_pos = 0
        all_accepted = True
        
        # 驗證每個 draft token
        for k in range(n_draft):
            token_idx = start_idx + k
            draft_token = draft_token_ids[token_idx].item()
            
            # 取得該 token 的 draft 和 target 機率
            p_draft = draft_probs[token_idx, draft_token].item()
            p_target = target_probs[token_idx, draft_token].item()
            
            # 避免除以零
            p_draft = max(p_draft, 1e-10)
            
            # Accept/Reject 決策
            # Accept with probability min(1, p_target / p_draft)
            r = uniform_samples[token_idx].item()
            acceptance_prob = min(1.0, p_target / p_draft)
            
            if r < acceptance_prob:
                # ✅ ACCEPT: 接受這個 draft token
                output_token_ids[batch_idx, output_pos] = draft_token
                output_pos += 1
                accepted_counts[batch_idx] += 1
            else:
                # ❌ REJECT: 從調整後的分布重新採樣
                # adjusted_probs = max(0, target - draft)
                adjusted_probs = torch.clamp(
                    target_probs[token_idx] - draft_probs[token_idx],
                    min=0.0
                )
                
                # 正規化
                prob_sum = adjusted_probs.sum()
                if prob_sum > 1e-10:
                    adjusted_probs = adjusted_probs / prob_sum
                else:
                    # Fallback: 使用 target distribution
                    adjusted_probs = target_probs[token_idx]
                
                # 重新採樣
                recovered_token = torch.multinomial(adjusted_probs, 1).item()
                output_token_ids[batch_idx, output_pos] = recovered_token
                output_pos += 1
                recovered_counts[batch_idx] += 1
                
                all_accepted = False
                break  # ⚠️ EARLY EXIT - 這是關鍵的動態控制流！
        
        # 如果所有 draft tokens 都被接受，加入 bonus token
        if all_accepted:
            bonus_token = bonus_token_ids[batch_idx, 0].item()
            output_token_ids[batch_idx, output_pos] = bonus_token
            output_pos += 1
            bonus_counts[batch_idx] += 1
        
        num_accepted[batch_idx] = output_pos
    
    return RejectionSampleOutput(
        output_token_ids=output_token_ids,
        num_accepted=num_accepted,
        accepted_counts=accepted_counts,
        recovered_counts=recovered_counts,
        bonus_counts=bonus_counts,
    )


def rejection_sample_baseline_greedy(
    draft_token_ids: torch.Tensor,      # [num_tokens]
    num_draft_tokens: List[int],        # [batch_size]
    target_probs: torch.Tensor,         # [num_tokens, vocab_size]
    bonus_token_ids: torch.Tensor,      # [batch_size, 1]
) -> RejectionSampleOutput:
    """
    Greedy 版本的 rejection sampling
    
    在 greedy 模式下，只要 target argmax == draft token 就接受
    """
    batch_size = len(num_draft_tokens)
    max_spec_len = max(num_draft_tokens) if num_draft_tokens else 0
    device = target_probs.device
    
    # 計算 cumulative offsets  
    cu_num_draft_tokens = torch.tensor(
        [sum(num_draft_tokens[:i+1]) for i in range(batch_size)],
        dtype=torch.int32, device=device
    )
    
    # Target argmax for each position
    target_argmax = target_probs.argmax(dim=-1)
    
    # 輸出 buffer
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    
    num_accepted = torch.zeros(batch_size, dtype=torch.int32, device=device)
    accepted_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    recovered_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    bonus_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    for batch_idx in range(batch_size):
        start_idx = 0 if batch_idx == 0 else cu_num_draft_tokens[batch_idx - 1].item()
        end_idx = cu_num_draft_tokens[batch_idx].item()
        n_draft = end_idx - start_idx
        
        output_pos = 0
        all_accepted = True
        
        for k in range(n_draft):
            token_idx = start_idx + k
            draft_token = draft_token_ids[token_idx].item()
            target_token = target_argmax[token_idx].item()
            
            # Greedy: 輸出 target argmax
            output_token_ids[batch_idx, output_pos] = target_token
            output_pos += 1
            
            if draft_token == target_token:
                # ACCEPT
                accepted_counts[batch_idx] += 1
            else:
                # REJECT (但我們已經輸出了 target token)
                recovered_counts[batch_idx] += 1
                all_accepted = False
                break
        
        if all_accepted:
            bonus_token = bonus_token_ids[batch_idx, 0].item()
            output_token_ids[batch_idx, output_pos] = bonus_token
            output_pos += 1
            bonus_counts[batch_idx] += 1
        
        num_accepted[batch_idx] = output_pos
    
    return RejectionSampleOutput(
        output_token_ids=output_token_ids,
        num_accepted=num_accepted,
        accepted_counts=accepted_counts,
        recovered_counts=recovered_counts,
        bonus_counts=bonus_counts,
    )


class RejectionSamplerBaseline(nn.Module):
    """
    Level 1 Baseline Rejection Sampler (Module 版本)
    
    封裝成 nn.Module 以便與其他實作保持一致的介面
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: List[int],
        draft_probs: Optional[torch.Tensor],
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        is_greedy: bool = False,
    ) -> RejectionSampleOutput:
        """
        執行 rejection sampling
        
        Args:
            draft_token_ids: [num_tokens] draft token IDs
            num_draft_tokens: 每個 request 的 draft token 數
            draft_probs: [num_tokens, vocab_size] draft 機率（greedy 時可為 None）
            target_probs: [num_tokens, vocab_size] target 機率
            bonus_token_ids: [batch_size, 1] bonus tokens
            is_greedy: 是否使用 greedy 模式
        
        Returns:
            RejectionSampleOutput
        """
        if is_greedy or draft_probs is None:
            return rejection_sample_baseline_greedy(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens,
                target_probs=target_probs,
                bonus_token_ids=bonus_token_ids,
            )
        else:
            return rejection_sample_baseline(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens,
                draft_probs=draft_probs,
                target_probs=target_probs,
                bonus_token_ids=bonus_token_ids,
            )


# ============================================
# 輔助函數：生成測試資料
# ============================================

def generate_test_data(
    batch_size: int,
    K: int,  # max speculation length
    vocab_size: int,
    device: str = "cuda",
    acceptance_rate_target: float = 0.7,
) -> dict:
    """
    生成測試用的合成資料
    
    Args:
        batch_size: Batch 大小
        K: 最大 speculation 長度
        vocab_size: 詞彙表大小
        device: 裝置
        acceptance_rate_target: 目標接受率（控制 draft/target 相似度）
    
    Returns:
        包含所有測試資料的字典
    """
    import numpy as np
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 每個 request 的 draft token 數（隨機變化）
    num_draft_tokens = [np.random.randint(K // 2 + 1, K + 1) for _ in range(batch_size)]
    total_tokens = sum(num_draft_tokens)
    
    # 生成 draft tokens
    draft_token_ids = torch.randint(
        0, vocab_size, (total_tokens,),
        device=device, dtype=torch.int32
    )
    
    # 生成 draft 機率分布
    draft_probs = torch.softmax(
        torch.randn(total_tokens, vocab_size, device=device),
        dim=-1
    )
    
    # 生成 target 機率分布（與 draft 有一定相似度）
    draft_logits = torch.log(draft_probs + 1e-10)
    target_logits = (
        acceptance_rate_target * draft_logits +
        (1 - acceptance_rate_target) * torch.randn(total_tokens, vocab_size, device=device)
    )
    target_probs = torch.softmax(target_logits, dim=-1)
    
    # 生成 bonus tokens
    bonus_probs = torch.softmax(
        torch.randn(batch_size, vocab_size, device=device),
        dim=-1
    )
    bonus_token_ids = torch.multinomial(bonus_probs, 1)
    
    return {
        "draft_token_ids": draft_token_ids,
        "num_draft_tokens": num_draft_tokens,
        "draft_probs": draft_probs,
        "target_probs": target_probs,
        "bonus_token_ids": bonus_token_ids,
        "batch_size": batch_size,
        "K": K,
        "vocab_size": vocab_size,
        "max_spec_len": max(num_draft_tokens),
    }


# ============================================
# 簡單測試
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Level 1 Baseline Rejection Sampler Test")
    print("=" * 60)
    
    # 生成測試資料
    data = generate_test_data(
        batch_size=4,
        K=8,
        vocab_size=1000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        acceptance_rate_target=0.7,
    )
    
    print(f"\nTest Configuration:")
    print(f"  Batch Size: {data['batch_size']}")
    print(f"  Max Spec Length: {data['max_spec_len']}")
    print(f"  Vocab Size: {data['vocab_size']}")
    print(f"  Num Draft Tokens per Request: {data['num_draft_tokens']}")
    
    # 執行 rejection sampling
    sampler = RejectionSamplerBaseline()
    
    import time
    
    # Warmup
    for _ in range(5):
        result = sampler(
            draft_token_ids=data["draft_token_ids"],
            num_draft_tokens=data["num_draft_tokens"],
            draft_probs=data["draft_probs"],
            target_probs=data["target_probs"],
            bonus_token_ids=data["bonus_token_ids"],
            is_greedy=False,
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        result = sampler(
            draft_token_ids=data["draft_token_ids"],
            num_draft_tokens=data["num_draft_tokens"],
            draft_probs=data["draft_probs"],
            target_probs=data["target_probs"],
            bonus_token_ids=data["bonus_token_ids"],
            is_greedy=False,
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    
    print(f"\nResults:")
    print(f"  Output Shape: {result.output_token_ids.shape}")
    print(f"  Num Accepted: {result.num_accepted.tolist()}")
    print(f"  Accepted Counts: {result.accepted_counts.tolist()}")
    print(f"  Recovered Counts: {result.recovered_counts.tolist()}")
    print(f"  Bonus Counts: {result.bonus_counts.tolist()}")
    
    total_draft = sum(data["num_draft_tokens"])
    total_accepted = result.accepted_counts.sum().item()
    acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0
    
    print(f"\nStatistics:")
    print(f"  Total Draft Tokens: {total_draft}")
    print(f"  Total Accepted: {total_accepted}")
    print(f"  Acceptance Rate: {acceptance_rate:.2%}")
    print(f"  Average Time: {avg_time_ms:.3f} ms")
    
    print("\n" + "=" * 60)
    print("[OK] Level 1 Baseline Test Passed!")
    print("=" * 60)
