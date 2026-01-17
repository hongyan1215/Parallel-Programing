#!/usr/bin/env python
"""
CUDA Kernel vs PyTorch Vectorized vs Baseline Benchmark
========================================================

完整比較三種 Rejection Sampling 實作的效能

Usage:
    python benchmark_cuda_comparison.py
"""

import os
import sys

# Add DLL directory for CUDA 12.4 (Windows only)
if sys.platform == 'win32':
    try:
        os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin')
    except (AttributeError, OSError):
        pass

# Add paths
sys.path.insert(0, 'src/cuda/csrc')
sys.path.insert(0, 'src')

import torch
import time
from typing import List

# Import CUDA extension
try:
    import fused_rejection_cuda
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CUDA extension not available: {e}")
    CUDA_AVAILABLE = False

# Import implementations
from baseline.rejection_sampler import rejection_sample_baseline
from cuda.fused_sampler import rejection_sample_fused_kernel


def prepare_test_data(batch_size: int, spec_len: int, vocab_size: int = 32000, device: str = 'cuda'):
    """準備測試資料，使用統一接口"""
    # Draft probs (flattened)
    num_tokens = batch_size * spec_len
    draft_probs = torch.rand(num_tokens, vocab_size, device=device)
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    
    # Target probs (flattened)
    target_probs = torch.rand(num_tokens, vocab_size, device=device)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    
    # Draft token IDs (flattened)
    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int64)
    
    # Bonus tokens [batch, 1]
    bonus_token_ids = torch.randint(0, vocab_size, (batch_size, 1), device=device, dtype=torch.int64)
    
    # num_draft_tokens list
    num_draft_tokens = [spec_len] * batch_size
    
    # Uniform samples
    uniform_samples = torch.rand(num_tokens, device=device)
    
    return {
        'draft_probs': draft_probs,
        'target_probs': target_probs,
        'draft_token_ids': draft_token_ids,
        'bonus_token_ids': bonus_token_ids,
        'num_draft_tokens': num_draft_tokens,
        'uniform_samples': uniform_samples,
        'batch_size': batch_size,
        'spec_len': spec_len,
        'vocab_size': vocab_size,
    }


def benchmark_baseline(data: dict, n_iter: int = 50) -> float:
    """Benchmark baseline (Python for loop)"""
    # Warmup
    for _ in range(5):
        rejection_sample_baseline(
            draft_token_ids=data['draft_token_ids'],
            num_draft_tokens=data['num_draft_tokens'],
            draft_probs=data['draft_probs'],
            target_probs=data['target_probs'],
            bonus_token_ids=data['bonus_token_ids'],
            uniform_samples=data['uniform_samples'],
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        rejection_sample_baseline(
            draft_token_ids=data['draft_token_ids'],
            num_draft_tokens=data['num_draft_tokens'],
            draft_probs=data['draft_probs'],
            target_probs=data['target_probs'],
            bonus_token_ids=data['bonus_token_ids'],
            uniform_samples=data['uniform_samples'],
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def benchmark_pytorch_vectorized(data: dict, n_iter: int = 50) -> float:
    """Benchmark PyTorch Vectorized (no Python loop)"""
    # Warmup
    for _ in range(5):
        rejection_sample_fused_kernel(
            draft_token_ids=data['draft_token_ids'],
            num_draft_tokens=data['num_draft_tokens'],
            draft_probs=data['draft_probs'],
            target_probs=data['target_probs'],
            bonus_token_ids=data['bonus_token_ids'],
            uniform_samples=data['uniform_samples'],
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        rejection_sample_fused_kernel(
            draft_token_ids=data['draft_token_ids'],
            num_draft_tokens=data['num_draft_tokens'],
            draft_probs=data['draft_probs'],
            target_probs=data['target_probs'],
            bonus_token_ids=data['bonus_token_ids'],
            uniform_samples=data['uniform_samples'],
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def benchmark_cuda_kernel(data: dict, n_iter: int = 50) -> float:
    """Benchmark TRUE CUDA C++ Kernel"""
    if not CUDA_AVAILABLE:
        return float('nan')
    
    batch_size = data['batch_size']
    spec_len = data['spec_len']
    vocab_size = data['vocab_size']
    
    # Prepare CUDA kernel format
    cu_num_draft = torch.arange(0, (batch_size + 1) * spec_len, spec_len, device='cuda', dtype=torch.int64)
    bonus_flat = data['bonus_token_ids'].squeeze(-1).contiguous()  # [batch]
    
    # Warmup
    for _ in range(5):
        fused_rejection_cuda.fused_rejection_sample(
            data['draft_probs'],
            data['target_probs'],
            data['draft_token_ids'],
            bonus_flat,
            cu_num_draft,
            data['uniform_samples'],
            spec_len
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fused_rejection_cuda.fused_rejection_sample(
            data['draft_probs'],
            data['target_probs'],
            data['draft_token_ids'],
            bonus_flat,
            cu_num_draft,
            data['uniform_samples'],
            spec_len
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def main():
    print("=" * 80)
    print("   🚀 CUDA Fused Rejection Sampler - Full Benchmark Comparison")
    print("=" * 80)
    print()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"CUDA Extension loaded: {CUDA_AVAILABLE}")
    print()
    
    # GPU warmup
    if torch.cuda.is_available():
        _ = torch.rand(2000, 2000, device='cuda') @ torch.rand(2000, 2000, device='cuda')
        torch.cuda.synchronize()
    
    # =========================================================================
    # Benchmark 1: Varying batch size
    # =========================================================================
    print("=" * 80)
    print("  Benchmark 1: Varying Batch Size (spec_len=8, vocab=32000)")
    print("=" * 80)
    print()
    print(f"{'Batch':>6} | {'Baseline':>12} | {'PyTorch Vec':>12} | {'CUDA Kernel':>12} | {'Vec vs Base':>12} | {'CUDA vs Base':>12}")
    print("-" * 80)
    
    results_batch = []
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        data = prepare_test_data(batch_size, spec_len=8)
        
        t_baseline = benchmark_baseline(data)
        t_pytorch = benchmark_pytorch_vectorized(data)
        t_cuda = benchmark_cuda_kernel(data)
        
        speedup_pytorch = t_baseline / t_pytorch if t_pytorch > 0 else 0
        speedup_cuda = t_baseline / t_cuda if t_cuda > 0 else 0
        
        results_batch.append({
            'batch': batch_size,
            't_baseline': t_baseline,
            't_pytorch': t_pytorch,
            't_cuda': t_cuda,
            'speedup_pytorch': speedup_pytorch,
            'speedup_cuda': speedup_cuda,
        })
        
        print(f"{batch_size:>6} | {t_baseline:>10.3f}ms | {t_pytorch:>10.3f}ms | {t_cuda:>10.3f}ms | {speedup_pytorch:>10.2f}x | {speedup_cuda:>11.2f}x")
    
    # =========================================================================
    # Benchmark 2: Varying speculation length
    # =========================================================================
    print()
    print("=" * 80)
    print("  Benchmark 2: Varying Spec Length (batch=4, vocab=32000)")
    print("=" * 80)
    print()
    print(f"{'SpecLen':>7} | {'Baseline':>12} | {'PyTorch Vec':>12} | {'CUDA Kernel':>12} | {'Vec vs Base':>12} | {'CUDA vs Base':>12}")
    print("-" * 80)
    
    for spec_len in [4, 8, 16, 32]:
        data = prepare_test_data(batch_size=4, spec_len=spec_len)
        
        t_baseline = benchmark_baseline(data)
        t_pytorch = benchmark_pytorch_vectorized(data)
        t_cuda = benchmark_cuda_kernel(data)
        
        speedup_pytorch = t_baseline / t_pytorch if t_pytorch > 0 else 0
        speedup_cuda = t_baseline / t_cuda if t_cuda > 0 else 0
        
        print(f"{spec_len:>7} | {t_baseline:>10.3f}ms | {t_pytorch:>10.3f}ms | {t_cuda:>10.3f}ms | {speedup_pytorch:>10.2f}x | {speedup_cuda:>11.2f}x")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 80)
    print("  📊 Summary")
    print("=" * 80)
    print()
    print("Implementations:")
    print("  • Baseline:       Python for loop (O(K) kernel launches)")
    print("  • PyTorch Vec:    100% vectorized PyTorch ops (O(1) kernel launches)")
    print("  • CUDA Kernel:    TRUE CUDA C++ kernel (single kernel launch) 🔥")
    print()
    
    if results_batch:
        max_speedup_pytorch = max(r['speedup_pytorch'] for r in results_batch)
        max_speedup_cuda = max(r['speedup_cuda'] for r in results_batch)
        avg_speedup_pytorch = sum(r['speedup_pytorch'] for r in results_batch) / len(results_batch)
        avg_speedup_cuda = sum(r['speedup_cuda'] for r in results_batch) / len(results_batch)
        
        print(f"PyTorch Vectorized speedup: avg {avg_speedup_pytorch:.2f}x, max {max_speedup_pytorch:.2f}x")
        print(f"CUDA C++ Kernel speedup:    avg {avg_speedup_cuda:.2f}x, max {max_speedup_cuda:.2f}x")
    
    print()
    print("✅ Benchmark complete!")


if __name__ == '__main__':
    main()
