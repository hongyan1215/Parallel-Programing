#!/usr/bin/env python3
"""
Compare benchmark results from different runs.
Useful for analyzing performance across different configurations.
"""

import argparse
import json
import sys
from typing import Dict, List


def load_results(filepath: str) -> Dict:
    """Load benchmark results from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def compare_two_results(baseline: Dict, current: Dict, threshold: float = 0.95):
    """
    Compare two benchmark results.
    
    Args:
        baseline: Baseline benchmark results
        current: Current benchmark results
        threshold: Performance degradation threshold (0.95 = 5% tolerance)
    
    Returns:
        Boolean indicating if performance is acceptable
    """
    print("=" * 80)
    print("Benchmark Comparison")
    print("=" * 80)
    
    # Compare acceptance rate
    baseline_ar = baseline["acceptance_rate"]["mean"]
    current_ar = current["acceptance_rate"]["mean"]
    ar_ratio = current_ar / baseline_ar if baseline_ar > 0 else 0
    
    print(f"\n[Acceptance Rate] Token Acceptance Rate:")
    print(f"  Baseline: {baseline_ar:.2%}")
    print(f"  Current:  {current_ar:.2%}")
    print(f"  Ratio:    {ar_ratio:.2%}")
    print(f"  Status:   {'[PASS]' if ar_ratio >= threshold else '[FAIL]'}")
    
    # Compare speedup
    baseline_speedup = baseline["effective_speedup"]["mean"]
    current_speedup = current["effective_speedup"]["mean"]
    speedup_ratio = current_speedup / baseline_speedup if baseline_speedup > 0 else 0
    
    print(f"\n[Speedup] Effective Speedup:")
    print(f"  Baseline: {baseline_speedup:.2f}x")
    print(f"  Current:  {current_speedup:.2f}x")
    print(f"  Ratio:    {speedup_ratio:.2%}")
    print(f"  Status:   {'[PASS]' if speedup_ratio >= threshold else '[FAIL]'}")
    
    # Compare throughput
    baseline_tp = baseline["throughput_tokens_per_sec"]["mean"]
    current_tp = current["throughput_tokens_per_sec"]["mean"]
    tp_ratio = current_tp / baseline_tp if baseline_tp > 0 else 0
    
    print(f"\n[Throughput] Throughput:")
    print(f"  Baseline: {baseline_tp:.0f} tokens/sec")
    print(f"  Current:  {current_tp:.0f} tokens/sec")
    print(f"  Ratio:    {tp_ratio:.2%}")
    print(f"  Status:   {'[PASS]' if tp_ratio >= threshold else '[FAIL]'}")
    
    # Compare latency (lower is better)
    baseline_lat = baseline["latency_ms"]["mean"]
    current_lat = current["latency_ms"]["mean"]
    lat_ratio = current_lat / baseline_lat if baseline_lat > 0 else 0
    
    print(f"\n[Latency] Latency:")
    print(f"  Baseline: {baseline_lat:.3f} ms")
    print(f"  Current:  {current_lat:.3f} ms")
    print(f"  Ratio:    {lat_ratio:.2%}")
    print(f"  Status:   {'[PASS]' if lat_ratio <= (2 - threshold) else '[FAIL]'}")
    
    # Overall status
    print("\n" + "=" * 80)
    passed = (ar_ratio >= threshold and 
              speedup_ratio >= threshold and 
              tp_ratio >= threshold and 
              lat_ratio <= (2 - threshold))
    
    if passed:
        print("[PASS] Overall: PASS - Performance is acceptable")
    else:
        print("[FAIL] Overall: FAIL - Performance degradation detected")
    
    print("=" * 80)
    
    return passed


def compare_multiple_results(result_files: List[str]):
    """Compare multiple benchmark results"""
    print("=" * 80)
    print("Multi-Configuration Comparison")
    print("=" * 80)
    
    results = []
    for filepath in result_files:
        result = load_results(filepath)
        results.append({
            "file": filepath,
            "config": result["config"],
            "metrics": result
        })
    
    # Print comparison table
    print("\n" + "=" * 120)
    print(f"{'Configuration':<30} {'Accept Rate':<15} {'Speedup':<12} {'Throughput':<20} {'Latency':<15}")
    print("=" * 120)
    
    for r in results:
        config = r["config"]
        metrics = r["metrics"]
        
        config_str = f"B{config['batch_size']}_S{config['max_spec_len']}_{config['sampling_mode'][:3]}"
        ar = metrics["acceptance_rate"]["mean"]
        speedup = metrics["effective_speedup"]["mean"]
        tp = metrics["throughput_tokens_per_sec"]["mean"]
        lat = metrics["latency_ms"]["mean"]
        
        print(f"{config_str:<30} {ar:>6.2%} ± {metrics['acceptance_rate']['std']:>5.2%} "
              f"{speedup:>6.2f}x ± {metrics['effective_speedup']['std']:>4.2f}x "
              f"{tp:>10.0f} tok/s "
              f"{lat:>8.3f} ms")
    
    print("=" * 120)
    
    # Find best configuration
    best_speedup = max(results, key=lambda r: r["metrics"]["effective_speedup"]["mean"])
    best_throughput = max(results, key=lambda r: r["metrics"]["throughput_tokens_per_sec"]["mean"])
    best_latency = min(results, key=lambda r: r["metrics"]["latency_ms"]["mean"])
    
    print("\n[BEST] Best Configurations:")
    print(f"  Highest Speedup:    {best_speedup['file']} ({best_speedup['metrics']['effective_speedup']['mean']:.2f}x)")
    print(f"  Highest Throughput: {best_throughput['file']} ({best_throughput['metrics']['throughput_tokens_per_sec']['mean']:.0f} tok/s)")
    print(f"  Lowest Latency:     {best_latency['file']} ({best_latency['metrics']['latency_ms']['mean']:.3f} ms)")


def export_comparison_csv(result_files: List[str], output_file: str):
    """Export comparison results to CSV"""
    import csv
    
    results = [load_results(f) for f in result_files]
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'file', 'batch_size', 'max_spec_len', 'sampling_mode',
            'acceptance_rate_mean', 'acceptance_rate_std',
            'speedup_mean', 'speedup_std',
            'throughput_mean', 'throughput_std',
            'latency_mean', 'latency_p95', 'latency_p99'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for filepath, result in zip(result_files, results):
            writer.writerow({
                'file': filepath,
                'batch_size': result['config']['batch_size'],
                'max_spec_len': result['config']['max_spec_len'],
                'sampling_mode': result['config']['sampling_mode'],
                'acceptance_rate_mean': result['acceptance_rate']['mean'],
                'acceptance_rate_std': result['acceptance_rate']['std'],
                'speedup_mean': result['effective_speedup']['mean'],
                'speedup_std': result['effective_speedup']['std'],
                'throughput_mean': result['throughput_tokens_per_sec']['mean'],
                'throughput_std': result['throughput_tokens_per_sec']['std'],
                'latency_mean': result['latency_ms']['mean'],
                'latency_p95': result['latency_ms']['p95'],
                'latency_p99': result['latency_ms']['p99'],
            })
    
    print(f"[OK] Comparison exported to: {output_file}")


def plot_comparison(result_files: List[str], output_file: str = "comparison_plot.png"):
    """Generate comparison plots"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Cannot generate plots.")
        return
    
    results = [load_results(f) for f in result_files]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Benchmark Comparison', fontsize=16, fontweight='bold')
    
    labels = [f"B{r['config']['batch_size']}_S{r['config']['max_spec_len']}" 
              for r in results]
    x = np.arange(len(labels))
    width = 0.35
    
    # Acceptance Rate
    ax = axes[0, 0]
    acceptance_rates = [r['acceptance_rate']['mean'] for r in results]
    acceptance_stds = [r['acceptance_rate']['std'] for r in results]
    ax.bar(x, acceptance_rates, width, yerr=acceptance_stds, capsize=5)
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Token Acceptance Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Speedup
    ax = axes[0, 1]
    speedups = [r['effective_speedup']['mean'] for r in results]
    speedup_stds = [r['effective_speedup']['std'] for r in results]
    ax.bar(x, speedups, width, yerr=speedup_stds, capsize=5, color='orange')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Effective Speedup')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.axhline(y=1, color='r', linestyle='--', label='Baseline (1x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Throughput
    ax = axes[1, 0]
    throughputs = [r['throughput_tokens_per_sec']['mean'] for r in results]
    ax.bar(x, throughputs, width, color='green')
    ax.set_ylabel('Tokens/Second')
    ax.set_title('Throughput')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Latency
    ax = axes[1, 1]
    latencies = [r['latency_ms']['mean'] for r in results]
    latencies_p95 = [r['latency_ms']['p95'] for r in results]
    ax.bar(x - width/2, latencies, width, label='Mean', color='red')
    ax.bar(x + width/2, latencies_p95, width, label='P95', color='darkred')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results from different runs"
    )
    parser.add_argument("files", nargs="+",
                       help="Benchmark result JSON files to compare")
    parser.add_argument("--baseline", type=str, default=None,
                       help="Baseline file for comparison (first file if not specified)")
    parser.add_argument("--threshold", type=float, default=0.95,
                       help="Performance degradation threshold (default: 0.95)")
    parser.add_argument("--export-csv", type=str, default=None,
                       help="Export comparison to CSV file")
    parser.add_argument("--plot", type=str, default=None,
                       help="Generate comparison plots (PNG file)")
    
    args = parser.parse_args()
    
    if len(args.files) < 1:
        print("Error: At least one result file is required")
        sys.exit(1)
    
    if len(args.files) == 2 and args.baseline is None:
        # Two-way comparison
        baseline = load_results(args.files[0])
        current = load_results(args.files[1])
        passed = compare_two_results(baseline, current, args.threshold)
        sys.exit(0 if passed else 1)
    
    elif len(args.files) >= 2:
        # Multi-way comparison
        compare_multiple_results(args.files)
        
        if args.export_csv:
            export_comparison_csv(args.files, args.export_csv)
        
        if args.plot:
            plot_comparison(args.files, args.plot)
    
    else:
        print("Error: Need at least 2 files for comparison")
        sys.exit(1)


if __name__ == "__main__":
    main()
