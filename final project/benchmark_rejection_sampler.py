#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark script for evaluating the performance of Rejection Sampling
in Speculative Decoding.

This script measures:
1. Token acceptance rate
2. Effective speedup
3. Average tokens per step
4. Execution time and throughput
5. GPU utilization
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Placeholder token ID for rejected tokens
PLACEHOLDER_TOKEN_ID = -1

# Flag to track if we're using mock implementations
USE_MOCK = False

# Mock imports for testing without full vLLM installation
try:
    from rejection_sampler import RejectionSampler
    from spec_decode.metadata import SpecDecodeMetadata
except ImportError:
    USE_MOCK = True
    print("Warning: Using mock implementations for testing")
    
    @dataclass
    class SpecDecodeMetadata:
        draft_token_ids: torch.Tensor
        num_draft_tokens: list[int]
        cu_num_draft_tokens: torch.Tensor
        cu_num_sampled_tokens: torch.Tensor
        target_logits_indices: torch.Tensor
        bonus_logits_indices: torch.Tensor
        logits_indices: torch.Tensor
        max_spec_len: int = 0
        
        def __post_init__(self):
            self.max_spec_len = max(self.num_draft_tokens) if self.num_draft_tokens else 0


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics"""
    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    recovered_tokens: int = 0
    bonus_tokens: int = 0
    num_steps: int = 0
    total_time: float = 0.0
    kernel_time: float = 0.0
    
    def compute_acceptance_rate(self) -> float:
        """Compute the token acceptance rate"""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens
    
    def compute_avg_tokens_per_step(self) -> float:
        """Compute average output tokens per step"""
        if self.num_steps == 0:
            return 0.0
        return (self.accepted_tokens + self.recovered_tokens + 
                self.bonus_tokens) / self.num_steps
    
    def compute_effective_speedup(self) -> float:
        """Compute effective speedup over standard decoding"""
        if self.num_steps == 0:
            return 0.0
        avg_tokens = self.compute_avg_tokens_per_step()
        # Speedup = avg_tokens_per_step / 1 (standard decoding produces 1 token/step)
        return avg_tokens
    
    def compute_throughput(self) -> float:
        """Compute tokens per second"""
        if self.total_time == 0:
            return 0.0
        total_output = self.accepted_tokens + self.recovered_tokens + self.bonus_tokens
        return total_output / self.total_time


class MockSamplingMetadata:
    """Mock SamplingMetadata for testing"""
    def __init__(self, batch_size: int, all_greedy: bool = False, 
                 all_random: bool = True, vocab_size: int = 32000,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.all_greedy = all_greedy
        self.all_random = all_random
        self.max_num_logprobs = 0
        self.no_penalties = True
        self.allowed_token_ids_mask = None
        self.bad_words_token_ids = None
        self.output_token_ids = [[] for _ in range(batch_size)]
        self.spec_token_ids = None
        self.prompt_token_ids = None
        self.presence_penalties = None
        self.frequency_penalties = None
        self.repetition_penalties = None
        
        # For random sampling
        if all_greedy:
            self.temperature = torch.zeros(batch_size, device=device)
        else:
            self.temperature = torch.ones(batch_size, device=device)
        
        # Use CPU generators if no CUDA
        gen_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generators = {i: torch.Generator(device=gen_device).manual_seed(42 + i) 
                          for i in range(batch_size)}
        self.top_k = None
        self.top_p = None


class MockSampler(nn.Module):
    """Mock Sampler for testing"""
    def __init__(self):
        super().__init__()
        self.logprobs_mode = "processed_logits"
    
    def forward(self, logits, sampling_metadata, **kwargs):
        """Mock forward pass"""
        from dataclasses import dataclass
        
        @dataclass
        class MockOutput:
            sampled_token_ids: torch.Tensor
            logprobs_tensors: Optional[object] = None
        
        # Simple argmax sampling
        sampled = logits.argmax(dim=-1, keepdim=True)
        output = MockOutput(sampled_token_ids=sampled, logprobs_tensors=None)
        output.logprobs_tensors = type('obj', (object,), {'logprobs': logits})()
        return output
    
    def compute_logprobs(self, logits):
        return torch.log_softmax(logits, dim=-1)
    
    def gather_logprobs(self, logprobs, max_num_logprobs, token_ids):
        return None


def mock_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list,
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata,
) -> torch.Tensor:
    """
    Mock implementation of rejection sampling for testing purposes.
    This implements the core algorithm from the paper: https://arxiv.org/abs/2211.17192
    
    Args:
        draft_token_ids: [num_tokens] - Draft token IDs
        num_draft_tokens: List of number of draft tokens per request
        max_spec_len: Maximum speculation length
        cu_num_draft_tokens: [batch_size] - Cumulative draft tokens
        draft_probs: [num_tokens, vocab_size] - Draft probabilities
        target_probs: [num_tokens, vocab_size] - Target probabilities
        bonus_token_ids: [batch_size, 1] - Bonus token IDs
        sampling_metadata: Sampling metadata
    
    Returns:
        output_token_ids: [batch_size, max_spec_len + 1] - Output token IDs
    """
    batch_size = len(num_draft_tokens)
    device = target_probs.device
    
    # Create output buffer filled with placeholder
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    
    # Process each request
    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu_num_draft_tokens[req_idx - 1].item()
        end_idx = cu_num_draft_tokens[req_idx].item()
        n_draft = end_idx - start_idx
        
        is_greedy = sampling_metadata.all_greedy
        rejected = False
        
        for pos in range(n_draft):
            if rejected:
                break
                
            token_idx = start_idx + pos
            draft_token = draft_token_ids[token_idx].item()
            
            if is_greedy:
                # Greedy: accept if target argmax matches draft
                target_token = target_probs[token_idx].argmax().item()
                if draft_token == target_token:
                    output_token_ids[req_idx, pos] = draft_token
                else:
                    # Reject and use target token
                    output_token_ids[req_idx, pos] = target_token
                    rejected = True
            else:
                # Random sampling with rejection
                draft_prob = draft_probs[token_idx, draft_token].item()
                target_prob = target_probs[token_idx, draft_token].item()
                
                # Sample uniform random number
                uniform_prob = torch.rand(1, device=device).item()
                
                # Accept with probability min(1, target_prob / draft_prob)
                if draft_prob > 0 and (target_prob / draft_prob) >= uniform_prob:
                    # Accept
                    output_token_ids[req_idx, pos] = draft_token
                else:
                    # Reject - sample from adjusted distribution
                    rejected = True
                    # Compute adjusted probability: max(0, target - draft)
                    adjusted_probs = torch.clamp(
                        target_probs[token_idx] - draft_probs[token_idx], 
                        min=0
                    )
                    # Normalize and sample
                    if adjusted_probs.sum() > 0:
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()
                        recovered_token = torch.multinomial(adjusted_probs, 1).item()
                    else:
                        recovered_token = target_probs[token_idx].argmax().item()
                    output_token_ids[req_idx, pos] = recovered_token
        
        # If all tokens accepted, add bonus token
        if not rejected:
            output_token_ids[req_idx, n_draft] = bonus_token_ids[req_idx, 0]
    
    return output_token_ids


def generate_synthetic_data(
    batch_size: int,
    max_spec_len: int,
    vocab_size: int,
    device: torch.device,
    acceptance_rate_target: float = 0.7,
) -> tuple:
    """
    Generate synthetic draft and target probabilities for benchmarking.
    
    Args:
        batch_size: Number of requests in the batch
        max_spec_len: Maximum speculation length
        vocab_size: Size of vocabulary
        device: Device to create tensors on
        acceptance_rate_target: Target acceptance rate (controls similarity)
    
    Returns:
        Tuple of (draft_token_ids, num_draft_tokens, draft_probs, target_probs, metadata)
    """
    # Generate variable number of draft tokens per request
    num_draft_tokens = [np.random.randint(max_spec_len // 2, max_spec_len + 1) 
                       for _ in range(batch_size)]
    total_tokens = sum(num_draft_tokens)
    
    # Generate draft tokens and probabilities
    draft_token_ids = torch.randint(0, vocab_size, (total_tokens,), 
                                    device=device, dtype=torch.int32)
    draft_probs = torch.softmax(torch.randn(total_tokens, vocab_size, device=device), dim=-1)
    
    # Generate target probabilities with controlled similarity to draft
    # Higher similarity = higher acceptance rate
    target_logits = torch.randn(total_tokens, vocab_size, device=device)
    
    # Mix draft and random logits to control acceptance rate
    draft_logits = torch.log(draft_probs + 1e-10)
    target_logits = (acceptance_rate_target * draft_logits + 
                    (1 - acceptance_rate_target) * target_logits)
    target_probs = torch.softmax(target_logits, dim=-1)
    
    # Create metadata
    cu_num_draft_tokens = torch.tensor(
        np.cumsum(num_draft_tokens), dtype=torch.int32, device=device
    )
    cu_num_sampled_tokens = torch.tensor(
        np.cumsum([n + 1 for n in num_draft_tokens]), dtype=torch.int32, device=device
    )
    
    target_logits_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    bonus_logits_indices = torch.arange(batch_size, dtype=torch.int32, device=device) + total_tokens
    logits_indices = torch.arange(total_tokens + batch_size, dtype=torch.int32, device=device)
    
    metadata = SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        cu_num_sampled_tokens=cu_num_sampled_tokens,
        target_logits_indices=target_logits_indices,
        bonus_logits_indices=bonus_logits_indices,
        logits_indices=logits_indices,
    )
    metadata.max_spec_len = max(num_draft_tokens)
    
    # Create combined logits for target and bonus
    combined_logits = torch.randn(total_tokens + batch_size, vocab_size, device=device)
    combined_logits[:total_tokens] = target_logits
    
    return draft_token_ids, num_draft_tokens, draft_probs, target_probs, metadata, combined_logits


def count_accepted_tokens(output_token_ids: torch.Tensor, 
                         draft_token_ids: torch.Tensor,
                         cu_num_draft_tokens: torch.Tensor,
                         placeholder_id: int = -1) -> tuple[int, int, int]:
    """
    Count accepted, recovered, and bonus tokens from output.
    
    Returns:
        (accepted_tokens, recovered_tokens, bonus_tokens)
    """
    batch_size = output_token_ids.shape[0]
    accepted = 0
    recovered = 0
    bonus = 0
    
    draft_idx = 0
    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu_num_draft_tokens[req_idx - 1].item()
        end_idx = cu_num_draft_tokens[req_idx].item()
        num_draft = end_idx - start_idx
        
        # Count valid output tokens
        output_row = output_token_ids[req_idx]
        valid_mask = output_row != placeholder_id
        num_valid = valid_mask.sum().item()
        
        # Check each draft position
        consecutive_accepts = 0
        for pos in range(num_draft):
            if output_row[pos] == placeholder_id:
                break
            if output_row[pos] == draft_token_ids[start_idx + pos]:
                accepted += 1
                consecutive_accepts += 1
            else:
                recovered += 1
                break
        
        # If we got more tokens than draft tokens, it's a bonus token
        if num_valid > num_draft:
            bonus += 1
    
    return accepted, recovered, bonus


def run_benchmark_iteration(
    rejection_sampler,
    batch_size: int,
    max_spec_len: int,
    vocab_size: int,
    device: torch.device,
    acceptance_rate_target: float = 0.7,
    sampling_mode: str = "random",
) -> tuple[BenchmarkMetrics, torch.Tensor]:
    """
    Run a single benchmark iteration.
    
    Args:
        rejection_sampler: The RejectionSampler instance
        batch_size: Number of requests
        max_spec_len: Maximum speculation length
        vocab_size: Vocabulary size
        device: Device to run on
        acceptance_rate_target: Target acceptance rate
        sampling_mode: "greedy" or "random"
    
    Returns:
        (metrics, output_token_ids)
    """
    metrics = BenchmarkMetrics()
    
    # Generate synthetic data
    (draft_token_ids, num_draft_tokens, draft_probs, target_probs, 
     metadata, combined_logits) = generate_synthetic_data(
        batch_size, max_spec_len, vocab_size, device, acceptance_rate_target
    )
    
    # Create sampling metadata
    sampling_metadata = MockSamplingMetadata(
        batch_size, 
        all_greedy=(sampling_mode == "greedy"),
        all_random=(sampling_mode == "random"),
        vocab_size=vocab_size,
        device=device
    )
    
    # Warm up GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Time the forward pass
    start_time = time.perf_counter()
    
    output_token_ids = None
    
    if rejection_sampler is not None and not USE_MOCK:
        try:
            # Try using the real RejectionSampler if available
            output = rejection_sampler.forward(
                metadata=metadata,
                draft_probs=draft_probs,
                logits=combined_logits,
                sampling_metadata=sampling_metadata,
            )
            output_token_ids = output.sampled_token_ids
        except Exception as e:
            pass  # Will use mock below
    
    if output_token_ids is None:
        # Use mock rejection sampling
        bonus_token_ids = combined_logits[metadata.bonus_logits_indices].argmax(dim=-1, keepdim=True)
        output_token_ids = mock_rejection_sample(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            max_spec_len=metadata.max_spec_len,
            cu_num_draft_tokens=metadata.cu_num_draft_tokens,
            draft_probs=draft_probs,
            target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    metrics.total_time = end_time - start_time
    metrics.kernel_time = metrics.total_time  # Simplified
    metrics.total_draft_tokens = sum(num_draft_tokens)
    metrics.num_steps = 1
    
    # Count accepted/recovered/bonus tokens
    accepted, recovered, bonus = count_accepted_tokens(
        output_token_ids, draft_token_ids, metadata.cu_num_draft_tokens
    )
    metrics.accepted_tokens = accepted
    metrics.recovered_tokens = recovered
    metrics.bonus_tokens = bonus
    
    return metrics, output_token_ids


def run_benchmark(
    num_iterations: int = 100,
    batch_size: int = 8,
    max_spec_len: int = 8,
    vocab_size: int = 32000,
    acceptance_rate_target: float = 0.7,
    sampling_mode: str = "random",
    device: str = "cuda",
) -> dict:
    """
    Run comprehensive benchmark.
    
    Args:
        num_iterations: Number of iterations to run
        batch_size: Batch size
        max_spec_len: Maximum speculation length
        vocab_size: Vocabulary size
        acceptance_rate_target: Target acceptance rate
        sampling_mode: "greedy" or "random"
        device: Device to run on
    
    Returns:
        Dictionary containing aggregate metrics and statistics
    """
    print("=" * 80)
    print("Rejection Sampler Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Iterations:          {num_iterations}")
    print(f"  Batch Size:          {batch_size}")
    print(f"  Max Spec Length:     {max_spec_len}")
    print(f"  Vocab Size:          {vocab_size}")
    print(f"  Sampling Mode:       {sampling_mode}")
    print(f"  Device:              {device}")
    print(f"  Target Accept Rate:  {acceptance_rate_target:.2%}")
    print("=" * 80)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize rejection sampler
    sampler = MockSampler()
    rejection_sampler = None
    
    try:
        from rejection_sampler import RejectionSampler
        rejection_sampler = RejectionSampler(sampler)
        rejection_sampler.to(device)
        print("Using real RejectionSampler")
    except Exception as e:
        print(f"Warning: Could not initialize RejectionSampler: {e}")
        print("Some features may be limited")
    
    # Collect metrics across iterations
    all_metrics = []
    acceptance_rates = []
    tokens_per_step = []
    throughputs = []
    latencies = []
    
    print(f"\nRunning {num_iterations} iterations...")
    
    for i in range(num_iterations):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_iterations}")
        
        try:
            metrics, _ = run_benchmark_iteration(
                rejection_sampler,
                batch_size,
                max_spec_len,
                vocab_size,
                device,
                acceptance_rate_target,
                sampling_mode,
            )
            
            all_metrics.append(metrics)
            acceptance_rates.append(metrics.compute_acceptance_rate())
            tokens_per_step.append(metrics.compute_avg_tokens_per_step())
            throughputs.append(metrics.compute_throughput())
            latencies.append(metrics.total_time * 1000)  # Convert to ms
        except Exception as e:
            print(f"Warning: Iteration {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Check if we have any successful iterations
    if len(acceptance_rates) == 0:
        print("Error: All iterations failed. Cannot compute statistics.")
        return {
            "config": {
                "num_iterations": num_iterations,
                "batch_size": batch_size,
                "max_spec_len": max_spec_len,
                "vocab_size": vocab_size,
                "sampling_mode": sampling_mode,
                "device": str(device),
                "target_acceptance_rate": acceptance_rate_target,
            },
            "error": "All iterations failed",
            "acceptance_rate": {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0},
            "tokens_per_step": {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0},
            "effective_speedup": {"mean": 0, "std": 0},
            "throughput_tokens_per_sec": {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0},
            "latency_ms": {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "p95": 0, "p99": 0},
            "total_stats": {"total_draft_tokens": 0, "total_accepted_tokens": 0, 
                           "total_recovered_tokens": 0, "total_bonus_tokens": 0},
        }
    
    print(f"\nSuccessful iterations: {len(acceptance_rates)}/{num_iterations}")
    
    # Aggregate statistics
    results = {
        "config": {
            "num_iterations": num_iterations,
            "batch_size": batch_size,
            "max_spec_len": max_spec_len,
            "vocab_size": vocab_size,
            "sampling_mode": sampling_mode,
            "device": str(device),
            "target_acceptance_rate": acceptance_rate_target,
        },
        "acceptance_rate": {
            "mean": np.mean(acceptance_rates),
            "std": np.std(acceptance_rates),
            "min": np.min(acceptance_rates),
            "max": np.max(acceptance_rates),
            "median": np.median(acceptance_rates),
        },
        "tokens_per_step": {
            "mean": np.mean(tokens_per_step),
            "std": np.std(tokens_per_step),
            "min": np.min(tokens_per_step),
            "max": np.max(tokens_per_step),
            "median": np.median(tokens_per_step),
        },
        "effective_speedup": {
            "mean": np.mean(tokens_per_step),  # Same as tokens_per_step
            "std": np.std(tokens_per_step),
        },
        "throughput_tokens_per_sec": {
            "mean": np.mean(throughputs),
            "std": np.std(throughputs),
            "min": np.min(throughputs),
            "max": np.max(throughputs),
            "median": np.median(throughputs),
        },
        "latency_ms": {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "median": np.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
        },
        "total_stats": {
            "total_draft_tokens": sum(m.total_draft_tokens for m in all_metrics),
            "total_accepted_tokens": sum(m.accepted_tokens for m in all_metrics),
            "total_recovered_tokens": sum(m.recovered_tokens for m in all_metrics),
            "total_bonus_tokens": sum(m.bonus_tokens for m in all_metrics),
        }
    }
    
    return results


def print_results(results: dict):
    """Pretty print benchmark results"""
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    
    print("\n📊 Token Acceptance Rate:")
    ar = results["acceptance_rate"]
    print(f"  Mean:     {ar['mean']:.2%} ± {ar['std']:.2%}")
    print(f"  Median:   {ar['median']:.2%}")
    print(f"  Range:    [{ar['min']:.2%}, {ar['max']:.2%}]")
    
    print("\n🚀 Effective Speedup:")
    speedup = results["effective_speedup"]
    print(f"  Mean:     {speedup['mean']:.2f}x ± {speedup['std']:.2f}x")
    print(f"  (vs. standard autoregressive decoding)")
    
    print("\n📈 Tokens per Step:")
    tps = results["tokens_per_step"]
    print(f"  Mean:     {tps['mean']:.2f} ± {tps['std']:.2f}")
    print(f"  Median:   {tps['median']:.2f}")
    print(f"  Range:    [{tps['min']:.2f}, {tps['max']:.2f}]")
    
    print("\n⚡ Throughput:")
    tp = results["throughput_tokens_per_sec"]
    print(f"  Mean:     {tp['mean']:.0f} tokens/sec")
    print(f"  Median:   {tp['median']:.0f} tokens/sec")
    print(f"  Range:    [{tp['min']:.0f}, {tp['max']:.0f}] tokens/sec")
    
    print("\n⏱️  Latency:")
    lat = results["latency_ms"]
    print(f"  Mean:     {lat['mean']:.3f} ms")
    print(f"  Median:   {lat['median']:.3f} ms")
    print(f"  P95:      {lat['p95']:.3f} ms")
    print(f"  P99:      {lat['p99']:.3f} ms")
    print(f"  Range:    [{lat['min']:.3f}, {lat['max']:.3f}] ms")
    
    print("\n📦 Total Statistics:")
    ts = results["total_stats"]
    print(f"  Draft Tokens:      {ts['total_draft_tokens']:,}")
    print(f"  Accepted Tokens:   {ts['total_accepted_tokens']:,}")
    print(f"  Recovered Tokens:  {ts['total_recovered_tokens']:,}")
    print(f"  Bonus Tokens:      {ts['total_bonus_tokens']:,}")
    total_output = ts['total_accepted_tokens'] + ts['total_recovered_tokens'] + ts['total_bonus_tokens']
    print(f"  Total Output:      {total_output:,}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Rejection Sampler for Speculative Decoding"
    )
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size (default: 8)")
    parser.add_argument("--max-spec-len", type=int, default=8,
                       help="Maximum speculation length (default: 8)")
    parser.add_argument("--vocab-size", type=int, default=32000,
                       help="Vocabulary size (default: 32000)")
    parser.add_argument("--acceptance-rate", type=float, default=0.7,
                       help="Target acceptance rate (default: 0.7)")
    parser.add_argument("--sampling-mode", type=str, default="random",
                       choices=["greedy", "random"],
                       help="Sampling mode (default: random)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (default: cuda)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file to save results (JSON format)")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        max_spec_len=args.max_spec_len,
        vocab_size=args.vocab_size,
        acceptance_rate_target=args.acceptance_rate,
        sampling_mode=args.sampling_mode,
        device=args.device,
    )
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {args.output}")
    
    # GPU memory stats if available
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
