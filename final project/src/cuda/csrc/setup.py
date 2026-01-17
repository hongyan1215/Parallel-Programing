"""
Setup script for building the CUDA extension
=============================================

使用方法:
    cd src/cuda/csrc
    python setup.py install

或使用 JIT 編譯:
    python -c "from torch.utils.cpp_extension import load; load(...)"
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 獲取當前目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='fused_rejection_sampler',
    version='2.0.0',
    author='PP25 Final Project',
    description='Fused CUDA Kernel for Rejection Sampling in Speculative Decoding',
    ext_modules=[
        CUDAExtension(
            name='fused_rejection_cuda',
            sources=[
                os.path.join(current_dir, 'fused_rejection.cpp'),
                os.path.join(current_dir, 'fused_rejection_kernel.cu'),
                os.path.join(current_dir, 'fused_rejection_kernel_v2.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--use_fast_math',
                    '-lineinfo',
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                ],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
)
