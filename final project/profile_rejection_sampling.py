#!/usr/bin/env python
"""
詳細分析 Rejection Sampling 性能瓶頸
比較 CUDA kernel vs Baseline 的各個步驟耗時
"""

import os
import sys
import torch
import time

# Add DLL directory for CUDA (Windows only)
if sys.platform == 'win32':
    try:
        os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin')
    except (AttributeError, OSError):
        pass

sys.path.insert(0, 'src/cuda/csrc')
sys.path.insert(0, 'src')

try:
    import fused_rejection_cuda
    CUDA_EXT_AVAILABLE = True
    print("✓ CUDA extension loaded")
except ImportError:
    CUDA_EXT_AVAILABLE = False
    print("✗ CUDA extension not available")

from baseline.rejection_sampler import rejection_sample_baseline


def benchmark_rejection_sampling(batch_size=1, spec_len=4, vocab_size=128256, n_iter=100):
    """
    詳細測試 rejection sampling 的各個步驟
    """
    device = 'cuda'
    
    print("\n" + "="*80)
    print("  Detailed Rejection Sampling Performance Analysis")
    print("="*80)
    print(f"Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Spec Length: {spec_len}")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Iterations: {n_iter}")
    print()
    
    # 準備數據
    draft_token_ids = torch.randint(0, vocab_size, (batch_size * spec_len,), device=device)
    draft_probs = torch.rand(batch_size * spec_len, vocab_size, device=device)
    target_probs = torch.rand(batch_size * spec_len, vocab_size, device=device)
    draft_probs = draft_probs / draft_probs.sum(dim=1, keepdim=True)
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
    bonus_token_ids = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    num_draft_tokens = [spec_len] * batch_size
    
    # =========================================================================
    # Test 1: Data Preparation Overhead
    # =========================================================================
    print("1. Testing Data Preparation...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        # Simulate data prep
        _ = draft_probs.float()
        _ = target_probs.float()
    torch.cuda.synchronize()
    prep_time = (time.time() - start) / n_iter * 1000
    print(f"   Data prep: {prep_time:.3f} ms")
    
    # =========================================================================
    # Test 2: Baseline Rejection Sampling
    # =========================================================================
    print("\n2. Testing Baseline (Python loop)...")
    uniform_samples = torch.rand(batch_size * spec_len, device=device)
    
    # Warmup
    for _ in range(10):
        _ = rejection_sample_baseline(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=draft_probs,
            target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
            uniform_samples=uniform_samples,
        )
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        result = rejection_sample_baseline(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=draft_probs,
            target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
            uniform_samples=uniform_samples,
        )
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / n_iter * 1000
    print(f"   Baseline: {baseline_time:.3f} ms")
    
    # =========================================================================
    # Test 3: CUDA Kernel (if available)
    # =========================================================================
    if CUDA_EXT_AVAILABLE:
        print("\n3. Testing CUDA Fused Kernel...")
        
        # Prepare cumulative format
        cu_num_draft = torch.zeros(batch_size + 1, dtype=torch.int64, device=device)
        for i in range(batch_size):
            cu_num_draft[i + 1] = cu_num_draft[i] + num_draft_tokens[i]
        
        # Prepare uniform samples
        uniform_samples = torch.rand(batch_size * spec_len, device=device)
        
        # Warmup
        for _ in range(10):
            _ = fused_rejection_cuda.fused_rejection_sample(
                draft_probs,
                target_probs,
                draft_token_ids,
                bonus_token_ids,
                cu_num_draft,
                uniform_samples,
                spec_len
            )
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            output_tokens, num_accepted, accepted_counts, recovered_counts, bonus_counts = \
                fused_rejection_cuda.fused_rejection_sample(
                    draft_probs,
                    target_probs,
                    draft_token_ids,
                    bonus_token_ids,
                    cu_num_draft,
                    uniform_samples,
                    spec_len
                )
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / n_iter * 1000
        print(f"   CUDA Kernel: {cuda_time:.3f} ms")
        
        # =========================================================================
        # Test 4: CUDA Kernel with Data Prep
        # =========================================================================
        print("\n4. Testing CUDA Kernel + Data Preparation...")
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            # Simulate full pipeline
            cu_num_draft = torch.zeros(batch_size + 1, dtype=torch.int64, device=device)
            for i in range(batch_size):
                cu_num_draft[i + 1] = cu_num_draft[i] + num_draft_tokens[i]
            uniform_samples = torch.rand(batch_size * spec_len, device=device)
            
            _ = fused_rejection_cuda.fused_rejection_sample(
                draft_probs,
                target_probs,
                draft_token_ids,
                bonus_token_ids,
                cu_num_draft,
                uniform_samples,
                spec_len
            )
        torch.cuda.synchronize()
        cuda_full_time = (time.time() - start) / n_iter * 1000
        print(f"   CUDA + Prep: {cuda_full_time:.3f} ms")
        
        # =========================================================================
        # Summary
        # =========================================================================
        print("\n" + "="*80)
        print("  PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Data Preparation:        {prep_time:.3f} ms")
        print(f"Baseline (Python):       {baseline_time:.3f} ms")
        print(f"CUDA Kernel (pure):      {cuda_time:.3f} ms  ({baseline_time/cuda_time:.2f}x faster)")
        print(f"CUDA Kernel + Prep:      {cuda_full_time:.3f} ms  ({baseline_time/cuda_full_time:.2f}x faster)")
        print()
        print("Analysis:")
        print(f"  - Pure kernel speedup:   {baseline_time/cuda_time:.2f}x")
        print(f"  - With prep overhead:    {baseline_time/cuda_full_time:.2f}x")
        print(f"  - Prep overhead:         {cuda_full_time - cuda_time:.3f} ms ({(cuda_full_time - cuda_time)/cuda_full_time*100:.1f}%)")
        
        if cuda_time > baseline_time:
            print(f"\n⚠️  CUDA kernel is slower than baseline!")
            print(f"    This suggests the overhead dominates for this problem size.")
        else:
            print(f"\n✓  CUDA kernel is faster than baseline!")


if __name__ == '__main__':
    # Test different configurations
    print("\n" + "#"*80)
    print("#  Configuration 1: Small (typical real-world)")
    print("#"*80)
    benchmark_rejection_sampling(batch_size=1, spec_len=4, vocab_size=128256, n_iter=100)
    
    print("\n" + "#"*80)
    print("#  Configuration 2: Larger batch")
    print("#"*80)
    benchmark_rejection_sampling(batch_size=8, spec_len=4, vocab_size=128256, n_iter=100)
    
    print("\n" + "#"*80)
    print("#  Configuration 3: Longer speculation")
    print("#"*80)
    benchmark_rejection_sampling(batch_size=1, spec_len=16, vocab_size=128256, n_iter=100)
