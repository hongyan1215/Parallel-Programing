#!/usr/bin/env python
"""
對比不同 vocab size 和 batch size 下的性能差異
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

import fused_rejection_cuda
from baseline.rejection_sampler import rejection_sample_baseline

def benchmark_config(batch_size, spec_len, vocab_size, n_iter=100):
    """測試特定配置"""
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


print("="*80)
print("  Vocab Size & Batch Size Impact on CUDA Kernel Performance")
print("="*80)
print()

configs = [
    # (batch, spec_len, vocab_size, description)
    (4, 8, 32000, "Quick Demo Config (batch=4, vocab=32K)"),
    (1, 8, 32000, "Single batch, vocab=32K"),
    (1, 8, 128256, "Real LLM Config (batch=1, vocab=128K)"),
    (4, 8, 128256, "Larger batch, vocab=128K"),
]

print(f"{'Configuration':<45} | {'Baseline':>10} | {'CUDA':>10} | {'Speedup':>10}")
print("-"*80)

for batch, spec_len, vocab, desc in configs:
    baseline, cuda, speedup = benchmark_config(batch, spec_len, vocab)
    print(f"{desc:<45} | {baseline:>9.3f}ms | {cuda:>9.3f}ms | {speedup:>9.2f}x")

print()
print("="*80)
print("Key Findings:")
print("  1. Larger vocab size (128K) increases memory access overhead")
print("  2. Batch size = 1 cannot utilize GPU parallelism effectively")
print("  3. CUDA kernel shines with: smaller vocab + larger batch")
print("="*80)
