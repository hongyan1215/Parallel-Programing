#!/usr/bin/env python3
"""
確認 V2 kernel 真的在做 argmax 操作
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'cuda', 'csrc'))

import torch
import fused_rejection_cuda

def test_argmax_work():
    """確保 V2 真的在遍歷 vocab"""
    print("=" * 70)
    print("Test: Verify V2 is actually doing argmax over vocab")
    print("=" * 70)
    
    device = torch.device('cuda')
    batch_size = 1
    spec_len = 1
    vocab_size = 100000
    
    num_tokens = batch_size * spec_len
    
    # 創建一個明確的 recovery 場景
    # target_probs 在 vocab 的最後一個位置 (99999) 有最高值
    draft_probs = torch.ones(num_tokens, vocab_size, device=device, dtype=torch.float32) / vocab_size
    target_probs = torch.ones(num_tokens, vocab_size, device=device, dtype=torch.float32) / vocab_size
    
    # 讓 draft 在 token 0 有高機率
    draft_probs[0, 0] = 0.99
    draft_probs[0, 1:] = 0.01 / (vocab_size - 1)
    
    # 讓 target 在 token 99999 有最高機率
    target_probs[0, vocab_size - 1] = 0.9
    target_probs[0, :vocab_size - 1] = 0.1 / (vocab_size - 1)
    
    # Normalize
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    
    # Draft token = 0 (will be rejected because target prefers 99999)
    draft_token_ids = torch.zeros(num_tokens, device=device, dtype=torch.int64)
    bonus_token_ids = torch.zeros(batch_size, device=device, dtype=torch.int64)
    cu_num_draft = torch.arange(0, batch_size + 1, device=device, dtype=torch.int64) * spec_len
    uniform_samples = torch.ones(num_tokens, device=device, dtype=torch.float32) * 0.99  # Force rejection
    
    # The recovery token should be argmax of (target - draft)
    # Since target[99999] = 0.9 and draft[99999] ≈ 0, recovery should be 99999
    
    result_v1 = fused_rejection_cuda.fused_rejection_sample(
        draft_probs, target_probs, draft_token_ids,
        bonus_token_ids, cu_num_draft, uniform_samples, spec_len
    )
    
    result_v2 = fused_rejection_cuda.fused_rejection_sample_v2(
        draft_probs, target_probs, draft_token_ids,
        bonus_token_ids, cu_num_draft, uniform_samples, spec_len
    )
    
    v1_token = result_v1[0][0, 0].item()
    v2_token = result_v2[0][0, 0].item()
    
    expected = vocab_size - 1  # 99999
    
    print(f"\nVocab size: {vocab_size}")
    print(f"Draft token: 0 (high prob in draft)")
    print(f"Expected recovery: {expected} (highest in target-draft)")
    print(f"\nV1 recovered: {v1_token}")
    print(f"V2 recovered: {v2_token}")
    
    print(f"\nV1 correct: {v1_token == expected}")
    print(f"V2 correct: {v2_token == expected}")
    
    if v2_token == expected:
        print("\n✓ V2 is correctly scanning the entire vocab and finding argmax!")
    else:
        print(f"\n✗ V2 found wrong token. Expected {expected}, got {v2_token}")
        
        # Debug: find the actual argmax
        diff = target_probs[0] - draft_probs[0]
        actual_argmax = diff.argmax().item()
        print(f"   Actual argmax of (target-draft): {actual_argmax}")
        print(f"   diff[{expected}] = {diff[expected].item():.6f}")
        print(f"   diff[{v2_token}] = {diff[v2_token].item():.6f}")


def test_timing_with_forced_rejection():
    """測試強制 rejection 時的時間"""
    print("\n" + "=" * 70)
    print("Test: Timing with forced rejection (worst case for V2)")
    print("=" * 70)
    
    device = torch.device('cuda')
    batch_size = 8
    spec_len = 8
    
    vocab_sizes = [10000, 50000, 100000, 150000]
    
    print(f"\n{'Vocab Size':<15} {'V1 (ms)':<12} {'V2 (ms)':<12} {'Speedup':<12}")
    print("-" * 55)
    
    for vocab_size in vocab_sizes:
        num_tokens = batch_size * spec_len
        
        # Create distributions that force rejection at first token
        draft_probs = torch.ones(num_tokens, vocab_size, device=device, dtype=torch.float32) / vocab_size
        target_probs = torch.ones(num_tokens, vocab_size, device=device, dtype=torch.float32) / vocab_size
        
        # Draft likes token 0, target likes token vocab_size-1
        for i in range(num_tokens):
            draft_probs[i, 0] = 0.99
            target_probs[i, vocab_size - 1] = 0.99
        
        draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
        
        draft_token_ids = torch.zeros(num_tokens, device=device, dtype=torch.int64)  # All draft token 0
        bonus_token_ids = torch.zeros(batch_size, device=device, dtype=torch.int64)
        cu_num_draft = torch.arange(0, batch_size + 1, device=device, dtype=torch.int64) * spec_len
        uniform_samples = torch.ones(num_tokens, device=device, dtype=torch.float32) * 0.99  # Force rejection
        
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
        
        # Timing
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
        
        start.record()
        for _ in range(100):
            fused_rejection_cuda.fused_rejection_sample_v2(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids, cu_num_draft, uniform_samples, spec_len
            )
        end.record()
        torch.cuda.synchronize()
        v2_time = start.elapsed_time(end) / 100
        
        speedup = v1_time / v2_time
        print(f"{vocab_size:<15} {v1_time:<12.4f} {v2_time:<12.4f} {speedup:<12.1f}x")


if __name__ == '__main__':
    test_argmax_work()
    test_timing_with_forced_rejection()
