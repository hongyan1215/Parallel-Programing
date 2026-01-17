#!/usr/bin/env python
"""
Quick Demo: CUDA Fused Kernel with Synthetic Data
==================================================

不需要下載大型 LLM，使用合成資料快速展示 CUDA kernel 功能
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
    print("✅ CUDA Extension loaded successfully!")
except ImportError as e:
    CUDA_EXT_AVAILABLE = False
    print(f"⚠️ CUDA extension not available: {e}")
    print("   Will use PyTorch vectorized version")

from cuda.fused_sampler import rejection_sample_fused_kernel
from baseline.rejection_sampler import rejection_sample_baseline


def demo_cuda_kernel():
    """展示 CUDA Kernel 的功能和效能"""
    print("\n" + "=" * 80)
    print("   🎓 期末專題 Demo: CUDA Fused Rejection Sampler")
    print("=" * 80)
    print()
    
    # 模擬參數（類似真實 LLM）
    batch_size = 4
    spec_len = 8
    vocab_size = 32000  # Llama 的 vocab size
    device = 'cuda'
    
    print(f"Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Speculation Length: {spec_len}")
    print(f"  Vocabulary Size: {vocab_size:,}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print()
    
    # 準備測試資料
    print("Preparing test data...")
    draft_probs = torch.rand(batch_size * spec_len, vocab_size, device=device)
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    
    target_probs = torch.rand(batch_size * spec_len, vocab_size, device=device)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    
    draft_token_ids = torch.randint(0, vocab_size, (batch_size * spec_len,), 
                                    device=device, dtype=torch.int64)
    bonus_token_ids = torch.randint(0, vocab_size, (batch_size, 1), 
                                    device=device, dtype=torch.int64)
    uniform_samples = torch.rand(batch_size * spec_len, device=device)
    
    num_draft_tokens = [spec_len] * batch_size
    
    print("✅ Data prepared\n")
    
    # =========================================================================
    # Test 1: 功能測試
    # =========================================================================
    print("=" * 80)
    print("  Test 1: Functional Correctness")
    print("=" * 80)
    
    # Baseline
    print("\n1️⃣ Running Baseline (Python for loop)...")
    result_baseline = rejection_sample_baseline(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        draft_probs=draft_probs,
        target_probs=target_probs,
        bonus_token_ids=bonus_token_ids,
        uniform_samples=uniform_samples,
    )
    print(f"   Output shape: {result_baseline.output_token_ids.shape}")
    print(f"   Accepted: {result_baseline.num_accepted.tolist()}")
    
    # PyTorch Vectorized
    print("\n2️⃣ Running PyTorch Vectorized...")
    result_pytorch = rejection_sample_fused_kernel(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        draft_probs=draft_probs,
        target_probs=target_probs,
        bonus_token_ids=bonus_token_ids,
        uniform_samples=uniform_samples,
    )
    print(f"   Output shape: {result_pytorch.output_token_ids.shape}")
    print(f"   Accepted: {result_pytorch.num_accepted.tolist()}")
    
    # CUDA Kernel
    if CUDA_EXT_AVAILABLE:
        print("\n3️⃣ Running CUDA C++ Kernel...")
        cu_num_draft = torch.arange(0, (batch_size + 1) * spec_len, spec_len,
                                    device=device, dtype=torch.int64)
        
        output_tokens, num_accepted, accepted_counts, recovered_counts, bonus_counts = \
            fused_rejection_cuda.fused_rejection_sample(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids.squeeze(-1), cu_num_draft, uniform_samples, spec_len
            )
        print(f"   Output shape: {output_tokens.shape}")
        print(f"   Accepted: {num_accepted.tolist()}")
        
        # 驗證一致性
        print("\n✅ Consistency Check:")
        matches = (result_baseline.num_accepted == result_pytorch.num_accepted).all()
        print(f"   Baseline vs PyTorch: {'✅ MATCH' if matches else '❌ MISMATCH'}")
        if CUDA_EXT_AVAILABLE:
            cuda_matches = (result_baseline.num_accepted.cpu() == num_accepted.cpu()).all()
            print(f"   Baseline vs CUDA: {'✅ MATCH' if cuda_matches else '❌ MISMATCH'}")
    
    # =========================================================================
    # Test 2: 效能測試
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Test 2: Performance Benchmark")
    print("=" * 80)
    print()
    
    n_warmup = 10
    n_iter = 100
    
    # Warmup
    print("Warming up GPU...")
    for _ in range(n_warmup):
        rejection_sample_baseline(
            draft_token_ids, num_draft_tokens, draft_probs, 
            target_probs, bonus_token_ids, uniform_samples
        )
    torch.cuda.synchronize()
    
    # Baseline
    print("Benchmarking Baseline...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        rejection_sample_baseline(
            draft_token_ids, num_draft_tokens, draft_probs,
            target_probs, bonus_token_ids, uniform_samples
        )
    end.record()
    torch.cuda.synchronize()
    time_baseline = start.elapsed_time(end) / n_iter
    
    # PyTorch Vectorized
    print("Benchmarking PyTorch Vectorized...")
    start.record()
    for _ in range(n_iter):
        rejection_sample_fused_kernel(
            draft_token_ids, num_draft_tokens, draft_probs,
            target_probs, bonus_token_ids, uniform_samples
        )
    end.record()
    torch.cuda.synchronize()
    time_pytorch = start.elapsed_time(end) / n_iter
    
    # CUDA Kernel
    time_cuda = None
    if CUDA_EXT_AVAILABLE:
        print("Benchmarking CUDA C++ Kernel...")
        start.record()
        for _ in range(n_iter):
            fused_rejection_cuda.fused_rejection_sample(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids.squeeze(-1), cu_num_draft, uniform_samples, spec_len
            )
        end.record()
        torch.cuda.synchronize()
        time_cuda = start.elapsed_time(end) / n_iter
    
    # 結果
    print("\n" + "=" * 80)
    print("  📊 Performance Results")
    print("=" * 80)
    print()
    print(f"{'Method':<25} | {'Time (ms)':>10} | {'Speedup':>10}")
    print("-" * 80)
    print(f"{'Baseline (Python loop)':<25} | {time_baseline:>10.3f} | {'1.00x':>10}")
    print(f"{'PyTorch Vectorized':<25} | {time_pytorch:>10.3f} | {time_baseline/time_pytorch:>9.2f}x")
    if time_cuda is not None:
        print(f"{'CUDA C++ Kernel':<25} | {time_cuda:>10.3f} | {time_baseline/time_cuda:>9.2f}x")
    
    print()
    print("=" * 80)
    print("  🎯 Key Takeaways for Your Presentation")
    print("=" * 80)
    print()
    print("1️⃣ Problem: Python for loop causes O(K) kernel launches")
    print("   → CPU-GPU synchronization is the bottleneck")
    print()
    print("2️⃣ Solution 1: PyTorch Vectorization")
    print(f"   → {time_baseline/time_pytorch:.1f}x speedup by eliminating Python loops")
    print()
    if time_cuda is not None:
        print("3️⃣ Solution 2: True CUDA C++ Kernel")
        print(f"   → {time_baseline/time_cuda:.1f}x speedup with single kernel launch")
        print("   → Each GPU thread independently handles one batch")
        print("   → Early exit happens inside GPU (no CPU sync needed!)")
    else:
        print("3️⃣ Solution 2: True CUDA C++ Kernel")
        print("   → Compile the extension to see CUDA kernel performance!")
    print()
    print("✅ Demo complete!")


if __name__ == '__main__':
    demo_cuda_kernel()
