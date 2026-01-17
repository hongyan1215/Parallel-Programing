#!/usr/bin/env python
"""
LLM Rejection Sampling Benchmark
=================================

只測量 rejection sampling 部分的效能，排除模型推理時間
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
except ImportError:
    CUDA_EXT_AVAILABLE = False

from cuda.fused_sampler import rejection_sample_fused_kernel
from baseline.rejection_sampler import rejection_sample_baseline


def benchmark_rejection_sampling_only():
    """只測量 rejection sampling 的效能（使用真實 LLM 的 vocab size）"""
    
    print("=" * 80)
    print("  LLM Rejection Sampling Benchmark")
    print("  (Measuring rejection sampling overhead only)")
    print("=" * 80)
    print()
    
    # 使用真實 LLM 的參數
    batch_size = 1  # 模擬單一生成請求
    spec_len = 4    # 每次猜4個tokens
    vocab_size = 32000  # TinyLlama vocab size
    device = 'cuda'
    n_iter = 100
    
    print(f"Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Speculation Length: {spec_len}")
    print(f"  Vocabulary Size: {vocab_size:,}")
    print(f"  Iterations: {n_iter}")
    print()
    
    # 準備資料（模擬 LLM 的輸出）
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
    
    # Warmup
    print("Warming up...")
    for _ in range(20):
        rejection_sample_baseline(
            draft_token_ids, num_draft_tokens, draft_probs,
            target_probs, bonus_token_ids, uniform_samples
        )
    torch.cuda.synchronize()
    
    # Benchmark Baseline
    print("Benchmarking Baseline (Python for loop)...")
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
    
    # Benchmark PyTorch Vectorized
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
    
    # Benchmark CUDA Kernel
    time_cuda = None
    if CUDA_EXT_AVAILABLE:
        print("Benchmarking CUDA C++ Kernel...")
        cu_num_draft = torch.arange(0, (batch_size + 1) * spec_len, spec_len,
                                    device=device, dtype=torch.int64)
        
        start.record()
        for _ in range(n_iter):
            fused_rejection_cuda.fused_rejection_sample(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids.squeeze(-1), cu_num_draft, uniform_samples, spec_len
            )
        end.record()
        torch.cuda.synchronize()
        time_cuda = start.elapsed_time(end) / n_iter
    
    # Results
    print("\n" + "=" * 80)
    print("  Results: Pure Rejection Sampling Overhead")
    print("=" * 80)
    print()
    print(f"{'Method':<30} | {'Time (ms)':>12} | {'Speedup':>10}")
    print("-" * 80)
    print(f"{'Baseline (Python loop)':<30} | {time_baseline:>12.4f} | {'1.00x':>10}")
    print(f"{'PyTorch Vectorized':<30} | {time_pytorch:>12.4f} | {time_baseline/time_pytorch:>9.2f}x")
    if time_cuda is not None:
        print(f"{'CUDA C++ Kernel':<30} | {time_cuda:>12.4f} | {time_baseline/time_cuda:>9.2f}x")
    
    print()
    print("=" * 80)
    print("  Analysis")
    print("=" * 80)
    print()
    print("In real LLM scenarios:")
    print(f"  • Rejection sampling overhead: ~{time_baseline:.2f}ms per step")
    print(f"  • Model forward pass: ~150-300ms (TinyLlama-1.1B)")
    print(f"  • Total time per step: ~{150 + time_baseline:.0f}-{300 + time_baseline:.0f}ms")
    print()
    print(f"Speedup contribution:")
    if time_cuda is not None:
        saved_time = time_baseline - time_cuda
        print(f"  • CUDA saves {saved_time:.2f}ms per step")
        print(f"  • For 30 tokens: {saved_time * 30 / 1000:.2f}s total savings")
        print(f"  • Overall speedup: {100 * saved_time / (150 + time_baseline):.1f}% - {100 * saved_time / (300 + time_baseline):.1f}% faster")
    else:
        saved_time = time_baseline - time_pytorch
        print(f"  • PyTorch saves {saved_time:.2f}ms per step")
        print(f"  • For 30 tokens: {saved_time * 30 / 1000:.2f}s total savings")
    
    print()
    print("Why LLM demo shows ~1.0x speedup:")
    print("  • Model inference dominates total time (95%+)")
    print("  • Rejection sampling is only 1-5% of total time")
    print("  • But in high-throughput serving with many parallel requests,")
    print("    this optimization becomes significant!")


if __name__ == '__main__':
    benchmark_rejection_sampling_only()
