#!/usr/bin/env python3
"""
Batch LLM Inference Test - 展示 V2 kernel 在高 throughput 場景的優勢
"""

import os
import sys
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'cuda', 'csrc'))

import fused_rejection_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_rejection_in_context(batch_size, spec_len, vocab_size, device, iterations=100):
    """模擬真實 LLM 推理中的 rejection sampling"""
    
    num_tokens = batch_size * spec_len
    
    # 模擬 LLM 輸出的 logits (需要先 softmax)
    draft_logits = torch.randn(num_tokens, vocab_size, device=device, dtype=torch.float16)
    target_logits = torch.randn(num_tokens, vocab_size, device=device, dtype=torch.float16)
    
    # Convert to probs (模擬溫度=1)
    draft_probs = torch.softmax(draft_logits, dim=-1).float().contiguous()
    target_probs = torch.softmax(target_logits, dim=-1).float().contiguous()
    
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
    for _ in range(iterations):
        fused_rejection_cuda.fused_rejection_sample(
            draft_probs, target_probs, draft_token_ids,
            bonus_token_ids, cu_num_draft, uniform_samples, spec_len
        )
    end.record()
    torch.cuda.synchronize()
    v1_time = start.elapsed_time(end) / iterations
    
    # V2 timing
    start.record()
    for _ in range(iterations):
        fused_rejection_cuda.fused_rejection_sample_v2(
            draft_probs, target_probs, draft_token_ids,
            bonus_token_ids, cu_num_draft, uniform_samples, spec_len
        )
    end.record()
    torch.cuda.synchronize()
    v2_time = start.elapsed_time(end) / iterations
    
    return v1_time, v2_time


def main():
    print("=" * 80)
    print("Batch Throughput Analysis: V1 vs V2 CUDA Kernel")
    print("=" * 80)
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 測試不同 batch size 和 vocab size
    vocab_sizes = [32000, 128256, 151936]  # TinyLlama, Llama-3, Qwen2.5
    batch_sizes = [1, 2, 4, 8, 16, 32]
    spec_len = 8
    
    print("\n" + "-" * 90)
    print(f"{'Vocab Size':<12} {'Batch':<8} {'V1 (ms)':<12} {'V2 (ms)':<12} {'Speedup':<10} {'Saved (ms)':<12}")
    print("-" * 90)
    
    results = []
    
    for vocab_size in vocab_sizes:
        for batch_size in batch_sizes:
            v1_time, v2_time = benchmark_rejection_in_context(
                batch_size, spec_len, vocab_size, device
            )
            speedup = v1_time / v2_time
            saved = v1_time - v2_time
            
            vocab_str = f"{vocab_size//1000}K"
            print(f"{vocab_str:<12} {batch_size:<8} {v1_time:<12.4f} {v2_time:<12.4f} {speedup:<10.2f}x {saved:<12.4f}")
            
            results.append({
                'vocab': vocab_size,
                'batch': batch_size,
                'v1': v1_time,
                'v2': v2_time,
                'speedup': speedup,
                'saved': saved
            })
        print()
    
    # 計算在不同場景下的節省
    print("=" * 80)
    print("Impact Analysis: Time Saved per Token Generation")
    print("=" * 80)
    
    # 假設平均每個 spec decode step 生成 6 tokens
    avg_accepted = 6
    
    print(f"\n假設: spec_len={spec_len}, 平均接受 {avg_accepted} tokens/step")
    print(f"       生成 1000 tokens 需要 ~{1000 // avg_accepted} 個 spec decode steps\n")
    
    for vocab_size in vocab_sizes:
        vocab_str = f"{vocab_size//1000}K"
        print(f"\n{vocab_str} Vocab (e.g., {'TinyLlama' if vocab_size==32000 else 'Llama-3' if vocab_size==128256 else 'Qwen2.5'}):")
        
        for batch_size in [1, 8, 32]:
            res = next(r for r in results if r['vocab'] == vocab_size and r['batch'] == batch_size)
            steps_for_1000 = 1000 // avg_accepted
            total_saved = res['saved'] * steps_for_1000
            
            print(f"  Batch={batch_size:<4}: V1={res['v1']:.3f}ms, V2={res['v2']:.3f}ms")
            print(f"         Per 1000 tokens: {total_saved:.1f}ms saved ({res['speedup']:.1f}x rejection speedup)")
    
    # Real LLM forward pass context
    print("\n" + "=" * 80)
    print("Context: LLM Forward Pass Timing (for reference)")
    print("=" * 80)
    
    print("\nLoading Qwen2.5-0.5B for forward pass timing...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    model.eval()
    
    # Measure forward pass time
    input_ids = torch.randint(0, 151936, (1, 128), device=device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start) / 20 * 1000
    
    print(f"\nForward pass time (batch=1, seq=128): {forward_time:.2f} ms")
    
    # Compare with rejection sampling
    r_32k = next(r for r in results if r['vocab'] == 151936 and r['batch'] == 1)
    print(f"\nRejection Sampling Comparison (Qwen vocab, batch=1):")
    print(f"  LLM Forward Pass: {forward_time:.2f} ms")
    print(f"  Rejection V1:     {r_32k['v1']:.4f} ms ({r_32k['v1']/forward_time*100:.2f}% of forward)")
    print(f"  Rejection V2:     {r_32k['v2']:.4f} ms ({r_32k['v2']/forward_time*100:.2f}% of forward)")
    print(f"  Speedup:          {r_32k['speedup']:.1f}x")
    
    # High throughput scenario
    print("\n" + "=" * 80)
    print("High Throughput Scenario: Batch=32, Qwen2.5 Vocab")
    print("=" * 80)
    
    r_high = next(r for r in results if r['vocab'] == 151936 and r['batch'] == 32)
    
    print(f"\nRejection Sampling Time:")
    print(f"  V1: {r_high['v1']:.3f} ms")
    print(f"  V2: {r_high['v2']:.3f} ms")
    print(f"  Speedup: {r_high['speedup']:.1f}x")
    print(f"  Time Saved: {r_high['saved']:.3f} ms per step")
    
    # For 1000 tokens per request, 32 requests in batch
    steps = 1000 // avg_accepted
    total_saved_batch = r_high['saved'] * steps
    print(f"\n  For generating 1000 tokens each (batch of 32 requests):")
    print(f"    Steps needed: {steps}")
    print(f"    Total time saved: {total_saved_batch:.1f} ms")
    print(f"    Per-request savings: {total_saved_batch/32:.1f} ms")


if __name__ == '__main__':
    main()
