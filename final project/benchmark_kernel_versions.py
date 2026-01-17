#!/usr/bin/env python3
"""
Benchmark 比較原始版和優化版 CUDA kernel
=========================================

測試不同配置下的性能差異
"""

import os
import sys
import torch
import time
import argparse

# 設定路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'cuda', 'csrc'))


def benchmark_kernel(func, inputs, warmup=10, iterations=100):
    """執行 benchmark"""
    # Warmup
    for _ in range(warmup):
        _ = func(*inputs)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = func(*inputs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # ms


def prepare_test_data(batch_size, spec_len, vocab_size, device):
    """準備測試數據"""
    num_tokens = batch_size * spec_len
    
    # 生成隨機機率分佈
    draft_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1)
    target_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1)
    
    # Draft tokens
    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int64)
    
    # Bonus tokens
    bonus_token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)
    
    # Cumulative num draft
    cu_num_draft = torch.arange(0, batch_size + 1, device=device, dtype=torch.int64) * spec_len
    
    # Uniform samples
    uniform_samples = torch.rand(num_tokens, device=device)
    
    return (draft_probs, target_probs, draft_token_ids, bonus_token_ids, cu_num_draft, uniform_samples)


def run_comparison(batch_size, spec_len, vocab_size, device, iterations=100):
    """比較不同版本的性能"""
    print(f"\n{'='*70}")
    print(f"Config: batch={batch_size}, spec_len={spec_len}, vocab={vocab_size:,}")
    print(f"{'='*70}")
    
    # 準備數據
    inputs = prepare_test_data(batch_size, spec_len, vocab_size, device)
    max_spec_len = spec_len
    
    results = {}
    
    # 1. Baseline (Python loop)
    try:
        from baseline.rejection_sampler import RejectionSamplerBaseline
        baseline_sampler = RejectionSamplerBaseline().to(device)
        
        # 重新格式化數據給 baseline
        # 將資料轉成 baseline 期望的格式
        draft_probs_flat = inputs[0]                    # [num_tokens, vocab_size]
        target_probs_flat = inputs[1]                   # [num_tokens, vocab_size]
        draft_tokens_flat = inputs[2].to(torch.int32)   # [num_tokens]
        bonus_token_ids_2d = inputs[3].view(batch_size, 1).to(torch.int32)
        num_draft_tokens = [spec_len] * batch_size      # 每個 batch 固定 spec_len
        
        def baseline_func():
            return baseline_sampler(
                draft_token_ids=draft_tokens_flat,
                num_draft_tokens=num_draft_tokens,
                draft_probs=draft_probs_flat,
                target_probs=target_probs_flat,
                bonus_token_ids=bonus_token_ids_2d,
                is_greedy=False,
            )
        
        baseline_time = benchmark_kernel(lambda: baseline_func(), [], warmup=5, iterations=min(20, iterations))
        results['Baseline (Python)'] = baseline_time
        print(f"  Baseline (Python loop):    {baseline_time:10.3f} ms")
    except Exception as e:
        print(f"  Baseline failed: {e}")
    
    # 2. CUDA Original
    try:
        import fused_rejection_cuda
        
        def cuda_original():
            return fused_rejection_cuda.fused_rejection_sample(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], max_spec_len
            )
        
        cuda_time = benchmark_kernel(lambda: cuda_original(), [], iterations=iterations)
        results['CUDA Original'] = cuda_time
        print(f"  CUDA Kernel (Original):    {cuda_time:10.3f} ms")
    except Exception as e:
        print(f"  CUDA Original failed: {e}")
    
    # 3. CUDA Optimized V2
    try:
        import fused_rejection_cuda
        
        if hasattr(fused_rejection_cuda, 'fused_rejection_sample_v2'):
            def cuda_v2():
                return fused_rejection_cuda.fused_rejection_sample_v2(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], max_spec_len
                )
            
            cuda_v2_time = benchmark_kernel(lambda: cuda_v2(), [], iterations=iterations)
            results['CUDA Optimized V2'] = cuda_v2_time
            print(f"  CUDA Kernel (Optimized):   {cuda_v2_time:10.3f} ms")
        else:
            print(f"  CUDA V2 not available (no fused_rejection_sample_v2)")
    except Exception as e:
        print(f"  CUDA Optimized failed: {e}")
    
    # 計算加速比
    print(f"\n  Speedup Analysis:")
    if 'Baseline (Python)' in results:
        baseline = results['Baseline (Python)']
        for name, t in results.items():
            if name != 'Baseline (Python)':
                speedup = baseline / t
                print(f"    {name}: {speedup:.2f}x vs Baseline")
    
    if 'CUDA Original' in results and 'CUDA Optimized V2' in results:
        speedup = results['CUDA Original'] / results['CUDA Optimized V2']
        status = "✅" if speedup > 1.0 else "⚠️"
        print(f"    V2 vs Original: {speedup:.2f}x {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark CUDA kernel versions')
    parser.add_argument('--gpu', type=int, default=4, help='GPU index to use')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    args = parser.parse_args()
    
    print("="*70)
    print("CUDA Kernel Optimization Benchmark")
    print("="*70)
    
    # 設置 GPU
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device("cuda:0")
    print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(0)}")
    
    # 測試配置
    configs = [
        # (batch_size, spec_len, vocab_size)
        (1, 8, 32000),      # TinyLlama: small batch, small vocab
        (4, 8, 32000),      # Quick demo config
        (8, 8, 32000),      # Medium batch
        (16, 8, 32000),     # Large batch
        (32, 8, 32000),     # Very large batch
        (64, 8, 32000),     # Stress test
        (1, 8, 128256),     # Llama 3.2: large vocab
        (8, 8, 128256),     # Large vocab + batch
        (1, 8, 151936),     # Qwen2.5: very large vocab
        (8, 8, 151936),     # Very large vocab + batch
    ]
    
    all_results = {}
    for batch_size, spec_len, vocab_size in configs:
        try:
            results = run_comparison(batch_size, spec_len, vocab_size, device, args.iterations)
            all_results[(batch_size, spec_len, vocab_size)] = results
        except Exception as e:
            print(f"  Config failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 總結表格
    print("\n" + "="*90)
    print("Summary Table")
    print("="*90)
    print(f"{'Config':<25} | {'Baseline':>12} | {'CUDA Orig':>12} | {'CUDA V2':>12} | {'V2/Orig':>10}")
    print("-"*90)
    
    for (batch, spec, vocab), results in all_results.items():
        config_str = f"b={batch}, s={spec}, v={vocab//1000}K"
        baseline = results.get('Baseline (Python)', float('nan'))
        cuda_orig = results.get('CUDA Original', float('nan'))
        cuda_v2 = results.get('CUDA Optimized V2', float('nan'))
        
        if cuda_orig > 0 and cuda_v2 > 0:
            speedup = cuda_orig / cuda_v2
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{config_str:<25} | {baseline:>10.2f}ms | {cuda_orig:>10.2f}ms | {cuda_v2:>10.2f}ms | {speedup_str:>10}")
    
    print("="*90)
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
