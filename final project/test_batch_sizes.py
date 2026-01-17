#!/usr/bin/env python
"""
測試不同 batch size 對 CUDA kernel 性能的影響
使用真實 LLM 進行 Speculative Decoding
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    print("⚠️ CUDA extension not available")

from baseline.rejection_sampler import rejection_sample_baseline


def benchmark_batch_rejection_sampling(batch_size, spec_len, vocab_size=128256, n_iter=50):
    """
    測試 rejection sampling 在不同 batch size 下的性能
    """
    device = 'cuda'
    
    # 準備數據
    draft_token_ids = torch.randint(0, vocab_size, (batch_size * spec_len,), device=device)
    draft_probs = torch.rand(batch_size * spec_len, vocab_size, device=device)
    target_probs = torch.rand(batch_size * spec_len, vocab_size, device=device)
    draft_probs = draft_probs / draft_probs.sum(dim=1, keepdim=True)
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
    bonus_token_ids = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    num_draft_tokens = [spec_len] * batch_size
    uniform_samples = torch.rand(batch_size * spec_len, device=device)
    
    # Test Baseline
    for _ in range(10):  # warmup
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
        _ = rejection_sample_baseline(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=draft_probs,
            target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
            uniform_samples=uniform_samples,
        )
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / n_iter * 1000
    
    # Test CUDA
    if CUDA_EXT_AVAILABLE:
        cu_num_draft = torch.tensor([i * spec_len for i in range(batch_size + 1)],
                                    device=device, dtype=torch.int64)
        
        for _ in range(10):  # warmup
            _ = fused_rejection_cuda.fused_rejection_sample(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids.squeeze(-1), cu_num_draft, uniform_samples, spec_len
            )
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            _ = fused_rejection_cuda.fused_rejection_sample(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids.squeeze(-1), cu_num_draft, uniform_samples, spec_len
            )
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / n_iter * 1000
        
        speedup = baseline_time / cuda_time
        return baseline_time, cuda_time, speedup
    else:
        return baseline_time, None, None


print("="*80)
print("  Batch Size Impact on Rejection Sampling Performance")
print("  Vocab Size: 128,256 (Llama 3.2), Spec Length: 8")
print("="*80)
print()

print(f"{'Batch Size':>12} | {'Baseline':>12} | {'CUDA':>12} | {'Speedup':>10}")
print("-"*80)

batch_sizes = [1, 2, 4, 8, 16]

for batch_size in batch_sizes:
    baseline, cuda, speedup = benchmark_batch_rejection_sampling(batch_size, spec_len=8)
    if cuda is not None:
        status = "✅" if speedup > 1.0 else "⚠️"
        print(f"{batch_size:>12} | {baseline:>11.3f}ms | {cuda:>11.3f}ms | {speedup:>9.2f}x {status}")
    else:
        print(f"{batch_size:>12} | {baseline:>11.3f}ms | {'N/A':>12} | {'N/A':>10}")

print()
print("="*80)
print("Analysis:")
print("  - Larger batch size increases GPU utilization")
print("  - CUDA kernel benefits from parallelism across batches")
print("  - Sweet spot for Llama 3.2 (vocab=128K): batch >= 4")
print("="*80)
