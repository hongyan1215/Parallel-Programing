"""
Level 2: torch.compile Version
==============================

這個版本使用 @torch.compile 裝飾器，展示 SOTA 編譯器在動態控制流上的限制。

預期結果：
- torch.compile 無法有效融合這個函數
- 因為存在 data-dependent 的 break 控制流
- 會產生 "graph breaks"，導致仍然是 O(K) 次 kernel launch

使用方式：
    TORCH_LOGS="graph_breaks" python -c "from src.compiled import rejection_sample_compiled"
    
可以看到類似以下的輸出：
    [graph_break] Dynamic control flow: data-dependent break statement
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
    output_token_ids: torch.Tensor
    num_accepted: torch.Tensor
    accepted_counts: torch.Tensor
    recovered_counts: torch.Tensor
    bonus_counts: torch.Tensor


# ============================================
# 嘗試 1: 直接編譯整個函數
# 預期：會有 graph breaks
# ============================================

@torch.compile(mode="reduce-overhead", fullgraph=False)
def _rejection_sample_inner_compiled(
    draft_token_ids: torch.Tensor,
    num_draft_tokens_tensor: torch.Tensor,  # 改用 tensor 而非 list
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    uniform_samples: torch.Tensor,
    max_spec_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    torch.compile 版本的 rejection sampling
    
    注意：由於 data-dependent control flow，這個函數無法完全融合
    """
    batch_size = num_draft_tokens_tensor.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    
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
    
    # ========================================
    # 核心邏輯：與 baseline 相同，但加上 @torch.compile
    # ========================================
    for batch_idx in range(batch_size):
        start_idx = 0 if batch_idx == 0 else cu_num_draft_tokens[batch_idx - 1]
        end_idx = cu_num_draft_tokens[batch_idx]
        n_draft = end_idx - start_idx
        
        output_pos = 0
        all_accepted = True
        
        # 驗證每個 draft token
        for k in range(max_spec_len):  # 使用固定上界
            if k >= n_draft:
                break
                
            token_idx = start_idx + k
            draft_token = draft_token_ids[token_idx]
            
            # 取得機率
            p_draft = draft_probs[token_idx, draft_token]
            p_target = target_probs[token_idx, draft_token]
            
            # 避免除以零
            p_draft = torch.clamp(p_draft, min=1e-10)
            
            # Accept/Reject 決策
            r = uniform_samples[token_idx]
            acceptance_prob = torch.minimum(
                torch.ones(1, device=device),
                p_target / p_draft
            )
            
            # ⚠️ 這個 if 會導致 graph break！
            if r < acceptance_prob:
                # ACCEPT
                output_token_ids[batch_idx, output_pos] = draft_token
                output_pos = output_pos + 1
                accepted_counts[batch_idx] = accepted_counts[batch_idx] + 1
            else:
                # REJECT + Resample
                adjusted_probs = torch.clamp(
                    target_probs[token_idx] - draft_probs[token_idx],
                    min=0.0
                )
                prob_sum = adjusted_probs.sum()
                adjusted_probs = torch.where(
                    prob_sum > 1e-10,
                    adjusted_probs / prob_sum,
                    target_probs[token_idx]
                )
                
                recovered_token = torch.multinomial(adjusted_probs.unsqueeze(0), 1).squeeze()
                output_token_ids[batch_idx, output_pos] = recovered_token
                output_pos = output_pos + 1
                recovered_counts[batch_idx] = recovered_counts[batch_idx] + 1
                all_accepted = False
                break  # ⚠️ DATA-DEPENDENT BREAK - 導致 graph break！
        
        # Bonus token
        if all_accepted:
            output_token_ids[batch_idx, output_pos] = bonus_token_ids[batch_idx, 0]
            output_pos = output_pos + 1
            bonus_counts[batch_idx] = bonus_counts[batch_idx] + 1
        
        num_accepted[batch_idx] = output_pos
    
    return output_token_ids, num_accepted, accepted_counts, recovered_counts, bonus_counts


def rejection_sample_compiled(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: List[int],
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    uniform_samples: Optional[torch.Tensor] = None,
) -> RejectionSampleOutput:
    """
    torch.compile 版本的 rejection sampling
    
    這個版本展示：
    1. @torch.compile 嘗試優化 Python 迴圈
    2. 但由於 data-dependent break，會產生 graph breaks
    3. 實際效能可能比 baseline 差（編譯開銷）
    """
    batch_size = len(num_draft_tokens)
    max_spec_len = max(num_draft_tokens) if num_draft_tokens else 0
    device = target_probs.device
    
    # 轉換為 tensor
    num_draft_tokens_tensor = torch.tensor(num_draft_tokens, dtype=torch.int32, device=device)
    cu_num_draft_tokens = torch.cumsum(num_draft_tokens_tensor, dim=0)
    
    # 生成隨機數
    if uniform_samples is None:
        num_tokens = draft_token_ids.shape[0]
        uniform_samples = torch.rand(num_tokens, device=device, dtype=torch.float32)
    
    # 呼叫編譯版本
    output_token_ids, num_accepted, accepted_counts, recovered_counts, bonus_counts = \
        _rejection_sample_inner_compiled(
            draft_token_ids=draft_token_ids,
            num_draft_tokens_tensor=num_draft_tokens_tensor,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_probs=draft_probs,
            target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
            uniform_samples=uniform_samples,
            max_spec_len=max_spec_len,
        )
    
    return RejectionSampleOutput(
        output_token_ids=output_token_ids,
        num_accepted=num_accepted,
        accepted_counts=accepted_counts,
        recovered_counts=recovered_counts,
        bonus_counts=bonus_counts,
    )


# ============================================
# 嘗試 2: 向量化版本（避免 Python loop）
# 但仍然無法處理動態長度輸出
# ============================================

@torch.compile(mode="reduce-overhead")
def _compute_acceptance_mask_compiled(
    draft_token_ids: torch.Tensor,      # [num_tokens]
    draft_probs: torch.Tensor,          # [num_tokens, vocab_size]
    target_probs: torch.Tensor,         # [num_tokens, vocab_size]
    uniform_samples: torch.Tensor,      # [num_tokens]
) -> torch.Tensor:
    """
    向量化計算 acceptance mask
    
    這部分可以被 torch.compile 有效優化，因為沒有動態控制流
    """
    num_tokens = draft_token_ids.shape[0]
    
    # 取得每個 token 的 draft/target 機率
    token_indices = torch.arange(num_tokens, device=draft_token_ids.device)
    p_draft = draft_probs[token_indices, draft_token_ids]
    p_target = target_probs[token_indices, draft_token_ids]
    
    # 避免除以零
    p_draft = torch.clamp(p_draft, min=1e-10)
    
    # 計算 acceptance probability
    acceptance_prob = torch.minimum(
        torch.ones_like(p_draft),
        p_target / p_draft
    )
    
    # Acceptance mask
    accepted = uniform_samples < acceptance_prob
    
    return accepted


class RejectionSamplerCompiled(nn.Module):
    """
    Level 2 torch.compile Rejection Sampler
    """
    
    def __init__(self):
        super().__init__()
        self._compiled = False
    
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
        執行 rejection sampling (torch.compile 版本)
        """
        if is_greedy or draft_probs is None:
            # Greedy 模式：使用 baseline 的 greedy 實作
            from src.baseline.rejection_sampler import rejection_sample_baseline_greedy
            return rejection_sample_baseline_greedy(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens,
                target_probs=target_probs,
                bonus_token_ids=bonus_token_ids,
            )
        else:
            return rejection_sample_compiled(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens,
                draft_probs=draft_probs,
                target_probs=target_probs,
                bonus_token_ids=bonus_token_ids,
            )


# ============================================
# 測試與分析
# ============================================

if __name__ == "__main__":
    import time
    import os
    import sys
    
    # 添加專案根目錄到 path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # 啟用 graph break 日誌
    os.environ["TORCH_LOGS"] = "graph_breaks,recompiles"
    
    print("=" * 60)
    print("Level 2 torch.compile Rejection Sampler Test")
    print("=" * 60)
    print("\n[INFO] TORCH_LOGS=graph_breaks enabled")
    print("[INFO] Watch for 'graph break' messages below...\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 生成測試資料
    from src.baseline.rejection_sampler import generate_test_data
    data = generate_test_data(
        batch_size=4,
        K=8,
        vocab_size=1000,
        device=device,
        acceptance_rate_target=0.7,
    )
    
    print(f"Test Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch Size: {data['batch_size']}")
    print(f"  Max Spec Length: {data['max_spec_len']}")
    print(f"  Vocab Size: {data['vocab_size']}")
    
    sampler = RejectionSamplerCompiled()
    
    # 第一次執行（觸發編譯）
    print("\n[Compilation Phase] First run (triggering compilation)...")
    start = time.perf_counter()
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
    compile_time = time.perf_counter() - start
    print(f"  Compilation time: {compile_time:.3f} s")
    
    # Warmup
    print("\n[Warmup Phase]...")
    for _ in range(10):
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
    print("\n[Benchmark Phase]...")
    iterations = 100
    start = time.perf_counter()
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
    
    # 比較與 baseline
    print("\n" + "-" * 40)
    print("Comparing with L1 Baseline...")
    print("-" * 40)
    
    from src.baseline.rejection_sampler import RejectionSamplerBaseline
    baseline_sampler = RejectionSamplerBaseline()
    
    # Warmup baseline
    for _ in range(10):
        baseline_result = baseline_sampler(
            draft_token_ids=data["draft_token_ids"],
            num_draft_tokens=data["num_draft_tokens"],
            draft_probs=data["draft_probs"],
            target_probs=data["target_probs"],
            bonus_token_ids=data["bonus_token_ids"],
            is_greedy=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark baseline
    start = time.perf_counter()
    for _ in range(iterations):
        baseline_result = baseline_sampler(
            draft_token_ids=data["draft_token_ids"],
            num_draft_tokens=data["num_draft_tokens"],
            draft_probs=data["draft_probs"],
            target_probs=data["target_probs"],
            bonus_token_ids=data["bonus_token_ids"],
            is_greedy=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    baseline_time_ms = (time.perf_counter() - start) / iterations * 1000
    
    print(f"  L1 Baseline: {baseline_time_ms:.3f} ms")
    print(f"  L2 Compiled: {avg_time_ms:.3f} ms")
    print(f"  Speedup: {baseline_time_ms / avg_time_ms:.2f}x")
    
    print("\n" + "=" * 60)
    print("[OK] Level 2 torch.compile Test Completed!")
    print("=" * 60)
    print("\n[NOTE] If you see 'graph_break' messages above,")
    print("       it confirms that torch.compile cannot fully fuse")
    print("       the rejection sampling due to dynamic control flow.")
