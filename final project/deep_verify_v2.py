#!/usr/bin/env python3
"""
深入驗證 V2 kernel 的正確性和實際工作量
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'cuda', 'csrc'))

import torch
import fused_rejection_cuda

def test_recovery_distribution():
    """測試 recovery sampling 是否正確工作"""
    print("=" * 70)
    print("Test 1: Recovery Distribution Correctness")
    print("=" * 70)
    
    device = torch.device('cuda')
    batch_size = 4
    spec_len = 8
    vocab_size = 32000
    
    torch.manual_seed(42)
    
    num_tokens = batch_size * spec_len
    
    # 創建會強制 rejection 的情境
    # draft_probs 在某個 token 上很高，target_probs 在不同 token 上很高
    draft_probs = torch.zeros(num_tokens, vocab_size, device=device, dtype=torch.float32)
    target_probs = torch.zeros(num_tokens, vocab_size, device=device, dtype=torch.float32)
    
    # 對於每個 token，設置不同的分佈
    for i in range(num_tokens):
        # Draft 在 token 100 上有高機率
        draft_probs[i, 100] = 0.9
        draft_probs[i, :100] = 0.001
        draft_probs[i, 101:200] = 0.001
        # Normalize
        draft_probs[i] = draft_probs[i] / draft_probs[i].sum()
        
        # Target 在 token 500+i 上有高機率
        target_token = 500 + i
        target_probs[i, target_token] = 0.9
        target_probs[i, :target_token] = 0.0001
        target_probs[i, target_token+1:] = 0.0001
        target_probs[i] = target_probs[i] / target_probs[i].sum()
    
    # Draft tokens 都是 100 (會被 reject 因為 target 更偏好其他 token)
    draft_token_ids = torch.full((num_tokens,), 100, device=device, dtype=torch.int64)
    bonus_token_ids = torch.zeros(batch_size, device=device, dtype=torch.int64)
    cu_num_draft = torch.arange(0, batch_size + 1, device=device, dtype=torch.int64) * spec_len
    
    # 設置 uniform samples 使得一定會 reject
    # acceptance_prob = min(1, target_probs[100] / draft_probs[100])
    # target_probs[100] 很小，所以 acceptance_prob 很小
    uniform_samples = torch.ones(num_tokens, device=device, dtype=torch.float32) * 0.5  # 中等值會 reject
    
    # Run V1
    result_v1 = fused_rejection_cuda.fused_rejection_sample(
        draft_probs, target_probs, draft_token_ids,
        bonus_token_ids, cu_num_draft, uniform_samples, spec_len
    )
    
    # Run V2
    result_v2 = fused_rejection_cuda.fused_rejection_sample_v2(
        draft_probs, target_probs, draft_token_ids,
        bonus_token_ids, cu_num_draft, uniform_samples, spec_len
    )
    
    print(f"\nDraft token: 100 (high prob in draft, low in target)")
    print(f"Expected recovery token: 500-507 (high prob in target)")
    
    print(f"\nV1 output tokens: {result_v1[0][0, :3].tolist()}")
    print(f"V2 output tokens: {result_v2[0][0, :3].tolist()}")
    print(f"V1 num_accepted: {result_v1[1].tolist()}")
    print(f"V2 num_accepted: {result_v2[1].tolist()}")
    
    # 檢查 recovery token 是否在 500-507 範圍
    v1_first_token = result_v1[0][0, 0].item()
    v2_first_token = result_v2[0][0, 0].item()
    
    print(f"\nV1 recovered token: {v1_first_token}")
    print(f"V2 recovered token: {v2_first_token}")
    
    # Recovery 應該選擇 target_probs - draft_probs 最大的位置
    # 對於 batch 0, token 0: 這應該是 500 (因為 target_probs[500] ≈ 0.9, draft_probs[500] ≈ 0)
    expected_recovery = 500
    
    v1_correct = v1_first_token == expected_recovery
    v2_correct = v2_first_token == expected_recovery
    
    print(f"\nExpected recovery token: {expected_recovery}")
    print(f"V1 correct: {v1_correct}")
    print(f"V2 correct: {v2_correct}")
    
    return v1_correct and v2_correct


def test_acceptance_path():
    """測試全部接受的情境"""
    print("\n" + "=" * 70)
    print("Test 2: Full Acceptance Path")
    print("=" * 70)
    
    device = torch.device('cuda')
    batch_size = 2
    spec_len = 4
    vocab_size = 1000
    
    num_tokens = batch_size * spec_len
    
    # 創建相同的分佈（100% acceptance）
    draft_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1).float()
    target_probs = draft_probs.clone()  # 相同分佈
    
    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int64)
    bonus_token_ids = torch.full((batch_size,), 999, device=device, dtype=torch.int64)
    cu_num_draft = torch.arange(0, batch_size + 1, device=device, dtype=torch.int64) * spec_len
    uniform_samples = torch.zeros(num_tokens, device=device, dtype=torch.float32)  # Always accept
    
    result_v1 = fused_rejection_cuda.fused_rejection_sample(
        draft_probs, target_probs, draft_token_ids,
        bonus_token_ids, cu_num_draft, uniform_samples, spec_len
    )
    
    result_v2 = fused_rejection_cuda.fused_rejection_sample_v2(
        draft_probs, target_probs, draft_token_ids,
        bonus_token_ids, cu_num_draft, uniform_samples, spec_len
    )
    
    print(f"\nWith identical distributions and uniform=0 (always accept):")
    print(f"V1 num_accepted: {result_v1[1].tolist()} (expected: [{spec_len+1}, {spec_len+1}])")
    print(f"V2 num_accepted: {result_v2[1].tolist()}")
    print(f"V1 bonus_counts: {result_v1[4].tolist()} (expected: [1, 1])")
    print(f"V2 bonus_counts: {result_v2[4].tolist()}")
    
    v1_correct = result_v1[1].tolist() == [spec_len + 1, spec_len + 1]
    v2_correct = result_v2[1].tolist() == [spec_len + 1, spec_len + 1]
    
    print(f"\nV1 correct: {v1_correct}")
    print(f"V2 correct: {v2_correct}")
    
    return v1_correct and v2_correct


def test_random_correctness():
    """隨機測試大量案例"""
    print("\n" + "=" * 70)
    print("Test 3: Random Correctness (1000 cases)")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    mismatches = 0
    total_tests = 1000
    
    for i in range(total_tests):
        batch_size = torch.randint(1, 17, (1,)).item()
        spec_len = torch.randint(1, 17, (1,)).item()
        vocab_size = torch.randint(1000, 50001, (1,)).item()
        
        num_tokens = batch_size * spec_len
        
        draft_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1).float()
        target_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1).float()
        draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int64)
        bonus_token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)
        cu_num_draft = torch.arange(0, batch_size + 1, device=device, dtype=torch.int64) * spec_len
        uniform_samples = torch.rand(num_tokens, device=device, dtype=torch.float32)
        
        result_v1 = fused_rejection_cuda.fused_rejection_sample(
            draft_probs, target_probs, draft_token_ids,
            bonus_token_ids, cu_num_draft, uniform_samples, spec_len
        )
        
        result_v2 = fused_rejection_cuda.fused_rejection_sample_v2(
            draft_probs, target_probs, draft_token_ids,
            bonus_token_ids, cu_num_draft, uniform_samples, spec_len
        )
        
        if not torch.all(result_v1[0] == result_v2[0]).item():
            mismatches += 1
            if mismatches <= 3:  # 只打印前3個錯誤
                print(f"\nMismatch at test {i}:")
                print(f"  Config: batch={batch_size}, spec_len={spec_len}, vocab={vocab_size}")
                print(f"  V1: {result_v1[0].flatten()[:10].tolist()}")
                print(f"  V2: {result_v2[0].flatten()[:10].tolist()}")
    
    print(f"\nTotal mismatches: {mismatches}/{total_tests}")
    return mismatches == 0


def test_timing_consistency():
    """測試時間是否真的隨 vocab 變化"""
    print("\n" + "=" * 70)
    print("Test 4: Timing Consistency Analysis")
    print("=" * 70)
    
    device = torch.device('cuda')
    batch_size = 8
    spec_len = 8
    
    vocab_sizes = [1000, 10000, 32000, 100000, 150000]
    
    print(f"\n{'Vocab Size':<15} {'V1 (ms)':<12} {'V2 (ms)':<12} {'V2/V1 ratio':<12}")
    print("-" * 55)
    
    for vocab_size in vocab_sizes:
        num_tokens = batch_size * spec_len
        
        draft_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1).float()
        target_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1).float()
        draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int64)
        bonus_token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)
        cu_num_draft = torch.arange(0, batch_size + 1, device=device, dtype=torch.int64) * spec_len
        uniform_samples = torch.rand(num_tokens, device=device, dtype=torch.float32)
        
        # Warmup
        for _ in range(20):
            fused_rejection_cuda.fused_rejection_sample(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids, cu_num_draft, uniform_samples, spec_len
            )
            fused_rejection_cuda.fused_rejection_sample_v2(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids, cu_num_draft, uniform_samples, spec_len
            )
        
        torch.cuda.synchronize()
        
        # V1 timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            fused_rejection_cuda.fused_rejection_sample(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids, cu_num_draft, uniform_samples, spec_len
            )
        end.record()
        torch.cuda.synchronize()
        v1_time = start.elapsed_time(end) / 100
        
        # V2 timing
        start.record()
        for _ in range(100):
            fused_rejection_cuda.fused_rejection_sample_v2(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids, cu_num_draft, uniform_samples, spec_len
            )
        end.record()
        torch.cuda.synchronize()
        v2_time = start.elapsed_time(end) / 100
        
        ratio = v2_time / v1_time
        print(f"{vocab_size:<15} {v1_time:<12.4f} {v2_time:<12.4f} {ratio:<12.4f}")
    
    return True


def main():
    print("=" * 70)
    print("Deep Verification of CUDA Kernel V2")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    results = []
    
    results.append(("Recovery Distribution", test_recovery_distribution()))
    results.append(("Acceptance Path", test_acceptance_path()))
    results.append(("Random Correctness", test_random_correctness()))
    test_timing_consistency()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")


if __name__ == '__main__':
    main()
