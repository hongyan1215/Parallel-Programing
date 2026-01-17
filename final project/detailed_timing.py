#!/usr/bin/env python3
"""Detailed timing comparison with CUDA events for accurate measurement"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'cuda', 'csrc'))

import torch
import fused_rejection_cuda

def benchmark_with_cuda_events(func, warmup=50, iterations=200):
    """Use CUDA events for precise GPU timing"""
    # Warmup
    for _ in range(warmup):
        func()
    
    torch.cuda.synchronize()
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Measure
    start_event.record()
    for _ in range(iterations):
        func()
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event) / iterations  # ms
    return elapsed

def prepare_data(batch_size, spec_len, vocab_size, device):
    num_tokens = batch_size * spec_len
    draft_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1)
    target_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1)
    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int64)
    bonus_token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)
    cu_num_draft = torch.arange(0, batch_size + 1, device=device, dtype=torch.int64) * spec_len
    uniform_samples = torch.rand(num_tokens, device=device)
    return draft_probs, target_probs, draft_token_ids, bonus_token_ids, cu_num_draft, uniform_samples, spec_len

def main():
    print("=" * 80)
    print("Precise CUDA Event Timing Comparison")
    print("=" * 80)
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    configs = [
        (1, 8, 32000),
        (4, 8, 32000),
        (8, 8, 32000),
        (16, 8, 32000),
        (32, 8, 32000),
        (64, 8, 32000),
        (1, 8, 128256),
        (8, 8, 128256),
        (1, 8, 151936),
        (8, 8, 151936),
    ]
    
    print(f"\n{'Config':<25} {'Original (ms)':<15} {'V2 (ms)':<15} {'Speedup':<10}")
    print("-" * 65)
    
    results = []
    for batch, spec_len, vocab in configs:
        data = prepare_data(batch, spec_len, vocab, device)
        max_spec_len = spec_len
        
        def run_original():
            return fused_rejection_cuda.fused_rejection_sample(
                data[0], data[1], data[2], data[3], data[4], data[5], max_spec_len
            )
        
        def run_v2():
            return fused_rejection_cuda.fused_rejection_sample_v2(
                data[0], data[1], data[2], data[3], data[4], data[5], max_spec_len
            )
        
        t_orig = benchmark_with_cuda_events(run_original)
        t_v2 = benchmark_with_cuda_events(run_v2)
        speedup = t_orig / t_v2
        
        config_str = f"b={batch}, s={spec_len}, v={vocab//1000}K"
        print(f"{config_str:<25} {t_orig:<15.4f} {t_v2:<15.4f} {speedup:.2f}x")
        
        results.append((config_str, t_orig, t_v2, speedup))
    
    print("-" * 65)
    
    avg_speedup = sum(r[3] for r in results) / len(results)
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    
    # Print more detail about the fastest case
    print("\n" + "=" * 80)
    print("Latency Analysis")
    print("=" * 80)
    
    # Test with minimal config
    data = prepare_data(1, 8, 32000, device)
    t_orig = benchmark_with_cuda_events(lambda: fused_rejection_cuda.fused_rejection_sample(
        data[0], data[1], data[2], data[3], data[4], data[5], 8
    ), warmup=100, iterations=500)
    t_v2 = benchmark_with_cuda_events(lambda: fused_rejection_cuda.fused_rejection_sample_v2(
        data[0], data[1], data[2], data[3], data[4], data[5], 8
    ), warmup=100, iterations=500)
    
    print(f"\nMinimal config (1, 8, 32K) with 500 iterations:")
    print(f"  Original kernel: {t_orig:.4f} ms ({t_orig*1000:.2f} μs)")
    print(f"  V2 kernel:       {t_v2:.4f} ms ({t_v2*1000:.2f} μs)")
    print(f"  Speedup:         {t_orig/t_v2:.2f}x")

if __name__ == '__main__':
    main()
