#!/usr/bin/env python3
"""Verify correctness of CUDA kernel V2 by comparing with original kernel"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'cuda', 'csrc'))

import torch
import torch.nn.functional as F

# Load CUDA extension
try:
    import fused_rejection_cuda
    print("✓ Loaded CUDA extension")
    print(f"  Available functions: {[x for x in dir(fused_rejection_cuda) if not x.startswith('_')]}")
except ImportError as e:
    print(f"✗ Failed to load CUDA extension: {e}")
    sys.exit(1)

def generate_test_data(batch_size, spec_len, vocab_size, acceptance_rate=0.7, device='cuda'):
    """Generate realistic test data matching the kernel's expected format"""
    torch.manual_seed(42)
    
    num_tokens = batch_size * spec_len
    
    # Generate draft and target probabilities (num_tokens, vocab_size)
    draft_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1)
    target_probs = torch.softmax(torch.randn(num_tokens, vocab_size, device=device), dim=-1)
    
    # Draft tokens (num_tokens,)
    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int64)
    
    # Bonus tokens (batch_size,)
    bonus_token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)
    
    # Cumulative num draft
    cu_num_draft = torch.arange(0, batch_size + 1, device=device, dtype=torch.int64) * spec_len
    
    # Uniform samples (num_tokens,)
    uniform_samples = torch.rand(num_tokens, device=device)
    
    return draft_probs, target_probs, draft_token_ids, bonus_token_ids, cu_num_draft, uniform_samples, spec_len

def test_config(batch_size, spec_len, vocab_size, verbose=True):
    """Test one configuration and compare results"""
    if verbose:
        print(f"\nTesting: batch={batch_size}, spec_len={spec_len}, vocab={vocab_size:,}")
    
    # Generate test data
    draft_probs, target_probs, draft_token_ids, bonus_token_ids, cu_num_draft, uniform_samples, max_spec_len = generate_test_data(
        batch_size, spec_len, vocab_size
    )
    
    # Run original CUDA kernel
    try:
        results_orig = fused_rejection_cuda.fused_rejection_sample(
            draft_probs, target_probs, draft_token_ids, bonus_token_ids, cu_num_draft, uniform_samples, max_spec_len
        )
        torch.cuda.synchronize()
        output_orig, accepted_orig = results_orig[0], results_orig[1]
    except Exception as e:
        print(f"  ✗ Original kernel failed: {e}")
        return False
    
    # Run optimized CUDA kernel
    try:
        results_v2 = fused_rejection_cuda.fused_rejection_sample_v2(
            draft_probs, target_probs, draft_token_ids, bonus_token_ids, cu_num_draft, uniform_samples, max_spec_len
        )
        torch.cuda.synchronize()
        output_v2, accepted_v2 = results_v2[0], results_v2[1]
    except Exception as e:
        print(f"  ✗ V2 kernel failed: {e}")
        return False
    
    # Compare results
    tokens_match = torch.all(output_orig == output_v2).item()
    counts_match = torch.all(accepted_orig == accepted_v2).item()
    
    if verbose:
        print(f"  Original output: {output_orig.flatten()[:15].tolist()}...")
        print(f"  V2 output:       {output_v2.flatten()[:15].tolist()}...")
        print(f"  Original counts: {accepted_orig.tolist()[:10]}...")
        print(f"  V2 counts:       {accepted_v2.tolist()[:10]}...")
    
    if tokens_match and counts_match:
        if verbose:
            print(f"  ✓ Results match!")
        return True
    else:
        if verbose:
            if not tokens_match:
                diff_mask = output_orig != output_v2
                print(f"  ✗ Token mismatch at {diff_mask.sum().item()} positions")
                diff_indices = torch.where(diff_mask.flatten())[0][:5]
                for idx in diff_indices:
                    print(f"    Position {idx.item()}: orig={output_orig.flatten()[idx].item()}, v2={output_v2.flatten()[idx].item()}")
            if not counts_match:
                diff_mask = accepted_orig != accepted_v2
                print(f"  ✗ Count mismatch at {diff_mask.sum().item()} positions")
                diff_indices = torch.where(diff_mask)[0][:5]
                for idx in diff_indices:
                    print(f"    Batch {idx.item()}: orig={accepted_orig[idx].item()}, v2={accepted_v2[idx].item()}")
        return False

def main():
    print("=" * 70)
    print("CUDA Kernel Correctness Verification")
    print("=" * 70)
    
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Test configurations
    test_cases = [
        # (batch_size, spec_len, vocab_size)
        (1, 4, 1000),          # Small test
        (4, 8, 32000),         # TinyLlama-like
        (8, 8, 32000),         # Larger batch
        (2, 8, 128256),        # Llama 3-like
        (4, 8, 151936),        # Qwen-like
        (1, 16, 32000),        # Long speculation
        (16, 8, 32000),        # Large batch
    ]
    
    results = []
    for batch_size, spec_len, vocab_size in test_cases:
        passed = test_config(batch_size, spec_len, vocab_size)
        results.append((batch_size, spec_len, vocab_size, passed))
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    all_passed = True
    for batch_size, spec_len, vocab_size, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: batch={batch_size}, spec_len={spec_len}, vocab={vocab_size:,}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! V2 kernel produces correct results.")
    else:
        print("\n✗ Some tests failed. V2 kernel has bugs.")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
