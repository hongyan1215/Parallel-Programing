#!/bin/bash
# Automated benchmark suite for testing different configurations
# This script runs benchmarks with various parameters and generates comparison reports

set -e  # Exit on error

# Configuration
OUTPUT_DIR="benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

# Create output directory
mkdir -p "${RESULTS_DIR}"

echo "================================================================"
echo "Rejection Sampler Benchmark Suite"
echo "================================================================"
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Function to run a single benchmark
run_benchmark() {
    local name=$1
    local args=$2
    local output="${RESULTS_DIR}/${name}.json"
    
    echo "Running: ${name}"
    echo "  Args: ${args}"
    python benchmark_rejection_sampler.py ${args} --output "${output}"
    echo "  ✅ Completed: ${output}"
    echo ""
}

# Benchmark Suite 1: Different Speculation Lengths
echo "================================================================"
echo "Suite 1: Testing Different Speculation Lengths"
echo "================================================================"
echo ""

for spec_len in 4 8 16; do
    run_benchmark \
        "spec_len_${spec_len}" \
        "--iterations 50 --batch-size 8 --max-spec-len ${spec_len} --acceptance-rate 0.7"
done

# Benchmark Suite 2: Different Batch Sizes
echo "================================================================"
echo "Suite 2: Testing Different Batch Sizes"
echo "================================================================"
echo ""

for batch in 2 4 8 16; do
    run_benchmark \
        "batch_${batch}" \
        "--iterations 50 --batch-size ${batch} --max-spec-len 8 --acceptance-rate 0.7"
done

# Benchmark Suite 3: Different Acceptance Rates
echo "================================================================"
echo "Suite 3: Testing Different Acceptance Rates"
echo "================================================================"
echo ""

for rate in 0.5 0.6 0.7 0.8 0.9; do
    run_benchmark \
        "acc_rate_${rate}" \
        "--iterations 50 --batch-size 8 --max-spec-len 8 --acceptance-rate ${rate}"
done

# Benchmark Suite 4: Greedy vs Random Sampling
echo "================================================================"
echo "Suite 4: Comparing Sampling Modes"
echo "================================================================"
echo ""

run_benchmark \
    "sampling_greedy" \
    "--iterations 50 --batch-size 8 --max-spec-len 8 --acceptance-rate 0.7 --sampling-mode greedy"

run_benchmark \
    "sampling_random" \
    "--iterations 50 --batch-size 8 --max-spec-len 8 --acceptance-rate 0.7 --sampling-mode random"

# Generate comparison reports
echo "================================================================"
echo "Generating Comparison Reports"
echo "================================================================"
echo ""

# Compare speculation lengths
echo "Comparing speculation lengths..."
python compare_benchmarks.py \
    "${RESULTS_DIR}/spec_len_4.json" \
    "${RESULTS_DIR}/spec_len_8.json" \
    "${RESULTS_DIR}/spec_len_16.json" \
    --export-csv "${RESULTS_DIR}/comparison_spec_len.csv" \
    --plot "${RESULTS_DIR}/comparison_spec_len.png" \
    > "${RESULTS_DIR}/comparison_spec_len.txt"

# Compare batch sizes
echo "Comparing batch sizes..."
python compare_benchmarks.py \
    "${RESULTS_DIR}/batch_2.json" \
    "${RESULTS_DIR}/batch_4.json" \
    "${RESULTS_DIR}/batch_8.json" \
    "${RESULTS_DIR}/batch_16.json" \
    --export-csv "${RESULTS_DIR}/comparison_batch.csv" \
    --plot "${RESULTS_DIR}/comparison_batch.png" \
    > "${RESULTS_DIR}/comparison_batch.txt"

# Compare acceptance rates
echo "Comparing acceptance rates..."
python compare_benchmarks.py \
    "${RESULTS_DIR}/acc_rate_0.5.json" \
    "${RESULTS_DIR}/acc_rate_0.6.json" \
    "${RESULTS_DIR}/acc_rate_0.7.json" \
    "${RESULTS_DIR}/acc_rate_0.8.json" \
    "${RESULTS_DIR}/acc_rate_0.9.json" \
    --export-csv "${RESULTS_DIR}/comparison_acc_rate.csv" \
    --plot "${RESULTS_DIR}/comparison_acc_rate.png" \
    > "${RESULTS_DIR}/comparison_acc_rate.txt"

# Compare sampling modes
echo "Comparing sampling modes..."
python compare_benchmarks.py \
    "${RESULTS_DIR}/sampling_greedy.json" \
    "${RESULTS_DIR}/sampling_random.json" \
    --export-csv "${RESULTS_DIR}/comparison_sampling.csv" \
    --plot "${RESULTS_DIR}/comparison_sampling.png" \
    > "${RESULTS_DIR}/comparison_sampling.txt"

# Generate summary report
echo "Generating summary report..."
cat > "${RESULTS_DIR}/SUMMARY.md" << EOF
# Benchmark Summary Report

**Generated**: $(date)
**Results Directory**: ${RESULTS_DIR}

## Overview

This benchmark suite evaluated the Rejection Sampler performance across different configurations:

1. **Speculation Lengths**: 4, 8, 16
2. **Batch Sizes**: 2, 4, 8, 16
3. **Acceptance Rates**: 0.5, 0.6, 0.7, 0.8, 0.9
4. **Sampling Modes**: Greedy, Random

## Results Files

### Individual Benchmarks
- Speculation Length Tests: \`spec_len_*.json\`
- Batch Size Tests: \`batch_*.json\`
- Acceptance Rate Tests: \`acc_rate_*.json\`
- Sampling Mode Tests: \`sampling_*.json\`

### Comparison Reports
- Speculation Length Comparison: \`comparison_spec_len.*\`
- Batch Size Comparison: \`comparison_batch.*\`
- Acceptance Rate Comparison: \`comparison_acc_rate.*\`
- Sampling Mode Comparison: \`comparison_sampling.*\`

## Key Findings

### Speculation Length Impact
See: \`comparison_spec_len.txt\` and \`comparison_spec_len.png\`

- Longer speculation lengths can provide higher speedup if acceptance rate is maintained
- Trade-off between potential speedup and computational overhead

### Batch Size Impact
See: \`comparison_batch.txt\` and \`comparison_batch.png\`

- Larger batches improve throughput
- Smaller batches reduce latency
- Choose based on use case (throughput vs latency)

### Acceptance Rate Impact
See: \`comparison_acc_rate.txt\` and \`comparison_acc_rate.png\`

- Higher acceptance rates lead to better speedup
- Draft model quality is critical for performance
- Diminishing returns above certain thresholds

### Sampling Mode Comparison
See: \`comparison_sampling.txt\` and \`comparison_sampling.png\`

- Greedy sampling: More deterministic, potentially faster
- Random sampling: More diverse outputs, flexibility for different use cases

## Recommendations

1. **For Maximum Throughput**: Use larger batch sizes (16+)
2. **For Low Latency**: Use smaller batch sizes (2-4)
3. **For Best Speedup**: Optimize draft model to maximize acceptance rate
4. **Speculation Length**: Start with 8, adjust based on acceptance rate

## Next Steps

1. Profile individual kernels for hotspots
2. Test with real model workloads
3. Optimize memory usage for larger vocabularies
4. Implement adaptive speculation length based on acceptance rate

---

For detailed results, see the comparison files in this directory.
EOF

echo "✅ Summary report generated: ${RESULTS_DIR}/SUMMARY.md"
echo ""

# Print summary
echo "================================================================"
echo "Benchmark Suite Completed!"
echo "================================================================"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Files generated:"
echo "  - Individual benchmark results: *.json"
echo "  - Comparison reports: comparison_*.txt"
echo "  - Comparison plots: comparison_*.png"
echo "  - CSV exports: comparison_*.csv"
echo "  - Summary: SUMMARY.md"
echo ""
echo "View the summary report:"
echo "  cat ${RESULTS_DIR}/SUMMARY.md"
echo ""
echo "View comparison plots (if matplotlib is installed):"
echo "  open ${RESULTS_DIR}/comparison_*.png"
echo ""
