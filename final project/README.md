# Speculative Decoding CUDA Fused Rejection Sampler

NTU Parallel Programming final project. This repository implements and evaluates a fused CUDA kernel that accelerates the rejection sampling step in speculative decoding for LLM inference. The optimized V2 kernel uses warp-level parallel reduction to make the argmax step efficient at large vocab sizes.

## What is included

- CUDA extension (C++/CUDA + PyTorch binding) with V1 and V2 kernels
- Python baseline and torch.compile versions for comparison
- Benchmark scripts, reports, and utilities
- Demo scripts for quick local runs and real LLM integration
- Reference speculative decoding components used for context and testing

## Project structure (main parts)

```
final project/
├── src/
│   ├── baseline/                 # Python baseline sampler
│   ├── compiled/                 # torch.compile version
│   └── cuda/                     # CUDA extension (Python wrapper + csrc)
│       └── csrc/                 # .cu/.cpp sources + setup.py
├── rejection_sampler/            # Standalone sampler (teaching / reference)
├── spec_decode/                  # Speculative decoding components (reference)
├── benchmark_results/            # Generated JSON outputs
├── benchmark_kernel_versions.py  # V1 vs V2 latency comparison
├── generate_performance_report.py# Summary report generator
├── verify_kernel_correctness.py  # Correctness checks
├── deep_verify_v2.py             # Larger randomized validation
├── test_llm_inference_v2.py       # Real LLM inference test
├── quick_demo.py                 # Quick local demo
├── demo_with_real_llm.py          # Real LLM demo
├── LLM_DEMO.md                   # LLM demo instructions
├── DEMO_GUIDE.md                 # Step-by-step demo guide
├── KERNEL_OPTIMIZATION_REPORT.md # Kernel design + performance analysis
└── IMPLEMENTATION_GUIDE.md       # Implementation notes
```

## Requirements

- Python 3.11+
- PyTorch 2.8.0+ with CUDA 12.x
- NVIDIA GPU
- CUDA Toolkit 12.4+

## Build the CUDA extension

```bash
cd src/cuda/csrc
python setup.py build_ext --inplace
```

## Typical workflow

```bash
# 1) Validate correctness
python verify_kernel_correctness.py

# 2) Compare V1 vs V2
python benchmark_kernel_versions.py

# 3) Generate a summary report (JSON outputs in benchmark_results/)
python generate_performance_report.py
```

## Demos

- `quick_demo.py` for a local quick run without extra setup.
- `demo_with_real_llm.py` + `LLM_DEMO.md` for real LLM integration.
- `DEMO_GUIDE.md` for a guided walkthrough of the demo flow.

## Benchmark automation

- `run_benchmark_suite.sh` (Linux/macOS)
- `run_benchmark_suite.bat` (Windows)
- `setup_demo.ps1` (PowerShell helper)

## Documents

- `KERNEL_OPTIMIZATION_REPORT.md`: kernel design, optimization, and results.
- `IMPLEMENTATION_GUIDE.md`: design notes and implementation steps.

## License

Apache-2.0
