@echo off
REM Automated benchmark suite for testing different configurations
REM This script runs benchmarks with various parameters and generates comparison reports

setlocal enabledelayedexpansion

REM Configuration
set OUTPUT_DIR=benchmark_results
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set DATESTAMP=%%c%%a%%b
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set TIMESTAMP=%%a%%b
set RESULTS_DIR=%OUTPUT_DIR%\%DATESTAMP%_%TIMESTAMP%

REM Create output directory
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

echo ================================================================
echo Rejection Sampler Benchmark Suite
echo ================================================================
echo Results will be saved to: %RESULTS_DIR%
echo.

REM Benchmark Suite 1: Different Speculation Lengths
echo ================================================================
echo Suite 1: Testing Different Speculation Lengths
echo ================================================================
echo.

for %%s in (4 8 16) do (
    echo Running: spec_len_%%s
    echo   Args: --iterations 50 --batch-size 8 --max-spec-len %%s --acceptance-rate 0.7
    python benchmark_rejection_sampler.py --iterations 50 --batch-size 8 --max-spec-len %%s --acceptance-rate 0.7 --output "%RESULTS_DIR%\spec_len_%%s.json"
    echo   Completed: %RESULTS_DIR%\spec_len_%%s.json
    echo.
)

REM Benchmark Suite 2: Different Batch Sizes
echo ================================================================
echo Suite 2: Testing Different Batch Sizes
echo ================================================================
echo.

for %%b in (2 4 8 16) do (
    echo Running: batch_%%b
    echo   Args: --iterations 50 --batch-size %%b --max-spec-len 8 --acceptance-rate 0.7
    python benchmark_rejection_sampler.py --iterations 50 --batch-size %%b --max-spec-len 8 --acceptance-rate 0.7 --output "%RESULTS_DIR%\batch_%%b.json"
    echo   Completed: %RESULTS_DIR%\batch_%%b.json
    echo.
)

REM Benchmark Suite 3: Different Acceptance Rates
echo ================================================================
echo Suite 3: Testing Different Acceptance Rates
echo ================================================================
echo.

for %%r in (0.5 0.6 0.7 0.8 0.9) do (
    echo Running: acc_rate_%%r
    echo   Args: --iterations 50 --batch-size 8 --max-spec-len 8 --acceptance-rate %%r
    python benchmark_rejection_sampler.py --iterations 50 --batch-size 8 --max-spec-len 8 --acceptance-rate %%r --output "%RESULTS_DIR%\acc_rate_%%r.json"
    echo   Completed: %RESULTS_DIR%\acc_rate_%%r.json
    echo.
)

REM Benchmark Suite 4: Greedy vs Random Sampling
echo ================================================================
echo Suite 4: Comparing Sampling Modes
echo ================================================================
echo.

echo Running: sampling_greedy
python benchmark_rejection_sampler.py --iterations 50 --batch-size 8 --max-spec-len 8 --acceptance-rate 0.7 --sampling-mode greedy --output "%RESULTS_DIR%\sampling_greedy.json"
echo   Completed: %RESULTS_DIR%\sampling_greedy.json
echo.

echo Running: sampling_random
python benchmark_rejection_sampler.py --iterations 50 --batch-size 8 --max-spec-len 8 --acceptance-rate 0.7 --sampling-mode random --output "%RESULTS_DIR%\sampling_random.json"
echo   Completed: %RESULTS_DIR%\sampling_random.json
echo.

REM Generate comparison reports
echo ================================================================
echo Generating Comparison Reports
echo ================================================================
echo.

REM Compare speculation lengths
echo Comparing speculation lengths...
python compare_benchmarks.py "%RESULTS_DIR%\spec_len_4.json" "%RESULTS_DIR%\spec_len_8.json" "%RESULTS_DIR%\spec_len_16.json" --export-csv "%RESULTS_DIR%\comparison_spec_len.csv" --plot "%RESULTS_DIR%\comparison_spec_len.png" > "%RESULTS_DIR%\comparison_spec_len.txt"

REM Compare batch sizes
echo Comparing batch sizes...
python compare_benchmarks.py "%RESULTS_DIR%\batch_2.json" "%RESULTS_DIR%\batch_4.json" "%RESULTS_DIR%\batch_8.json" "%RESULTS_DIR%\batch_16.json" --export-csv "%RESULTS_DIR%\comparison_batch.csv" --plot "%RESULTS_DIR%\comparison_batch.png" > "%RESULTS_DIR%\comparison_batch.txt"

REM Compare acceptance rates
echo Comparing acceptance rates...
python compare_benchmarks.py "%RESULTS_DIR%\acc_rate_0.5.json" "%RESULTS_DIR%\acc_rate_0.6.json" "%RESULTS_DIR%\acc_rate_0.7.json" "%RESULTS_DIR%\acc_rate_0.8.json" "%RESULTS_DIR%\acc_rate_0.9.json" --export-csv "%RESULTS_DIR%\comparison_acc_rate.csv" --plot "%RESULTS_DIR%\comparison_acc_rate.png" > "%RESULTS_DIR%\comparison_acc_rate.txt"

REM Compare sampling modes
echo Comparing sampling modes...
python compare_benchmarks.py "%RESULTS_DIR%\sampling_greedy.json" "%RESULTS_DIR%\sampling_random.json" --export-csv "%RESULTS_DIR%\comparison_sampling.csv" --plot "%RESULTS_DIR%\comparison_sampling.png" > "%RESULTS_DIR%\comparison_sampling.txt"

REM Generate summary report
echo Generating summary report...

(
echo # Benchmark Summary Report
echo.
echo **Generated**: %date% %time%
echo **Results Directory**: %RESULTS_DIR%
echo.
echo ## Overview
echo.
echo This benchmark suite evaluated the Rejection Sampler performance across different configurations:
echo.
echo 1. **Speculation Lengths**: 4, 8, 16
echo 2. **Batch Sizes**: 2, 4, 8, 16
echo 3. **Acceptance Rates**: 0.5, 0.6, 0.7, 0.8, 0.9
echo 4. **Sampling Modes**: Greedy, Random
echo.
echo ## Results Files
echo.
echo ### Individual Benchmarks
echo - Speculation Length Tests: `spec_len_*.json`
echo - Batch Size Tests: `batch_*.json`
echo - Acceptance Rate Tests: `acc_rate_*.json`
echo - Sampling Mode Tests: `sampling_*.json`
echo.
echo ### Comparison Reports
echo - Speculation Length Comparison: `comparison_spec_len.*`
echo - Batch Size Comparison: `comparison_batch.*`
echo - Acceptance Rate Comparison: `comparison_acc_rate.*`
echo - Sampling Mode Comparison: `comparison_sampling.*`
echo.
echo ## Key Findings
echo.
echo ### Speculation Length Impact
echo See: `comparison_spec_len.txt` and `comparison_spec_len.png`
echo.
echo - Longer speculation lengths can provide higher speedup if acceptance rate is maintained
echo - Trade-off between potential speedup and computational overhead
echo.
echo ### Batch Size Impact
echo See: `comparison_batch.txt` and `comparison_batch.png`
echo.
echo - Larger batches improve throughput
echo - Smaller batches reduce latency
echo - Choose based on use case ^(throughput vs latency^)
echo.
echo ### Acceptance Rate Impact
echo See: `comparison_acc_rate.txt` and `comparison_acc_rate.png`
echo.
echo - Higher acceptance rates lead to better speedup
echo - Draft model quality is critical for performance
echo - Diminishing returns above certain thresholds
echo.
echo ### Sampling Mode Comparison
echo See: `comparison_sampling.txt` and `comparison_sampling.png`
echo.
echo - Greedy sampling: More deterministic, potentially faster
echo - Random sampling: More diverse outputs, flexibility for different use cases
echo.
echo ## Recommendations
echo.
echo 1. **For Maximum Throughput**: Use larger batch sizes ^(16+^)
echo 2. **For Low Latency**: Use smaller batch sizes ^(2-4^)
echo 3. **For Best Speedup**: Optimize draft model to maximize acceptance rate
echo 4. **Speculation Length**: Start with 8, adjust based on acceptance rate
echo.
echo ---
echo.
echo For detailed results, see the comparison files in this directory.
) > "%RESULTS_DIR%\SUMMARY.md"

echo Summary report generated: %RESULTS_DIR%\SUMMARY.md
echo.

REM Print summary
echo ================================================================
echo Benchmark Suite Completed!
echo ================================================================
echo.
echo Results saved to: %RESULTS_DIR%
echo.
echo Files generated:
echo   - Individual benchmark results: *.json
echo   - Comparison reports: comparison_*.txt
echo   - Comparison plots: comparison_*.png
echo   - CSV exports: comparison_*.csv
echo   - Summary: SUMMARY.md
echo.
echo View the summary report:
echo   type %RESULTS_DIR%\SUMMARY.md
echo.

endlocal
pause
