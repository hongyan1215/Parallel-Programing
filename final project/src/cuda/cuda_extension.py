"""
CUDA C++ Extension Wrapper for Fused Rejection Sampling
========================================================

這個模組提供真正的 CUDA kernel 實作。

使用前需要先編譯:
    cd src/cuda/csrc
    python setup.py install

或使用 JIT 編譯（第一次會較慢）
"""

import torch
import torch.nn as nn
from typing import List, Optional
from dataclasses import dataclass
import os

# Placeholder token ID
PLACEHOLDER_TOKEN_ID: int = -1


@dataclass
class RejectionSampleOutput:
    """Rejection sampling 的輸出結果"""
    output_token_ids: torch.Tensor
    num_accepted: torch.Tensor
    accepted_counts: torch.Tensor
    recovered_counts: torch.Tensor
    bonus_counts: torch.Tensor


# ============================================================================
# 嘗試載入編譯好的 CUDA extension
# ============================================================================

_cuda_ext = None
_cuda_ext_available = False

def _load_cuda_extension():
    """嘗試載入 CUDA extension"""
    global _cuda_ext, _cuda_ext_available
    
    if _cuda_ext is not None:
        return _cuda_ext_available
    
    # 方法 1: 嘗試載入已安裝的 extension
    try:
        import fused_rejection_cuda
        _cuda_ext = fused_rejection_cuda
        _cuda_ext_available = True
        print("[INFO] Loaded pre-compiled CUDA extension")
        return True
    except ImportError:
        pass
    
    # 方法 2: JIT 編譯
    try:
        from torch.utils.cpp_extension import load
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csrc_dir = os.path.join(current_dir, 'csrc')
        
        if os.path.exists(os.path.join(csrc_dir, 'fused_rejection_kernel.cu')):
            print("[INFO] JIT compiling CUDA extension (this may take a minute)...")
            _cuda_ext = load(
                name='fused_rejection_cuda',
                sources=[
                    os.path.join(csrc_dir, 'fused_rejection.cpp'),
                    os.path.join(csrc_dir, 'fused_rejection_kernel.cu'),
                ],
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                verbose=True,
            )
            _cuda_ext_available = True
            print("[INFO] CUDA extension compiled successfully!")
            return True
    except Exception as e:
        print(f"[WARNING] Failed to compile CUDA extension: {e}")
    
    _cuda_ext_available = False
    return False


def is_cuda_ext_available() -> bool:
    """檢查 CUDA extension 是否可用"""
    return _load_cuda_extension()


# ============================================================================
# CUDA Kernel Wrapper
# ============================================================================

def rejection_sample_cuda_kernel(
    draft_token_ids: torch.Tensor,      # [num_tokens]
    num_draft_tokens: List[int],        # [batch_size]
    draft_probs: torch.Tensor,          # [num_tokens, vocab_size]
    target_probs: torch.Tensor,         # [num_tokens, vocab_size]
    bonus_token_ids: torch.Tensor,      # [batch_size] or [batch_size, 1]
    uniform_samples: Optional[torch.Tensor] = None,
) -> RejectionSampleOutput:
    """
    使用真正的 CUDA kernel 執行 rejection sampling
    
    這是 O(1) kernel launch 的實作！
    """
    if not _load_cuda_extension():
        raise RuntimeError(
            "CUDA extension not available. Please compile it first:\n"
            "  cd src/cuda/csrc && python setup.py install"
        )
    
    batch_size = len(num_draft_tokens)
    device = target_probs.device
    num_tokens = draft_token_ids.shape[0]
    max_spec_len = max(num_draft_tokens) if num_draft_tokens else 0
    
    # 處理邊界情況
    if batch_size == 0 or num_tokens == 0:
        return RejectionSampleOutput(
            output_token_ids=torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID, 
                                         dtype=torch.int32, device=device),
            num_accepted=torch.zeros(batch_size, dtype=torch.int32, device=device),
            accepted_counts=torch.zeros(batch_size, dtype=torch.int32, device=device),
            recovered_counts=torch.zeros(batch_size, dtype=torch.int32, device=device),
            bonus_counts=torch.zeros(batch_size, dtype=torch.int32, device=device),
        )
    
    # 準備 cumulative num_draft_tokens
    num_draft_tensor = torch.tensor(num_draft_tokens, dtype=torch.int64, device=device)
    cu_num_draft = torch.zeros(batch_size + 1, dtype=torch.int64, device=device)
    cu_num_draft[1:] = num_draft_tensor.cumsum(0)
    
    # 確保 bonus_token_ids 是 1D
    if bonus_token_ids.ndim == 2:
        bonus_token_ids = bonus_token_ids[:, 0]
    bonus_token_ids = bonus_token_ids.to(torch.int64)
    
    # 生成隨機數
    if uniform_samples is None:
        uniform_samples = torch.rand(num_tokens, device=device, dtype=torch.float32)
    
    # 確保資料類型正確
    draft_probs = draft_probs.float().contiguous()
    target_probs = target_probs.float().contiguous()
    draft_token_ids = draft_token_ids.to(torch.int64).contiguous()
    uniform_samples = uniform_samples.float().contiguous()
    
    # 呼叫 CUDA kernel
    outputs = _cuda_ext.fused_rejection_sample(
        draft_probs,
        target_probs,
        draft_token_ids,
        bonus_token_ids,
        cu_num_draft,
        uniform_samples,
        max_spec_len,
    )
    
    return RejectionSampleOutput(
        output_token_ids=outputs[0],
        num_accepted=outputs[1],
        accepted_counts=outputs[2],
        recovered_counts=outputs[3],
        bonus_counts=outputs[4],
    )


# ============================================================================
# Module Class
# ============================================================================

class CUDAFusedRejectionSampler(nn.Module):
    """
    使用真正 CUDA C++ Extension 的 Rejection Sampler
    
    特點:
    - 單一 CUDA kernel launch
    - 每個 thread 處理一個 batch element
    - 完全在 GPU 上執行，無 CPU-GPU 同步
    """
    
    def __init__(self):
        super().__init__()
        
        # 確認 CUDA extension 可用
        if not is_cuda_ext_available():
            raise RuntimeError(
                "CUDA extension not available. Please compile it first:\n"
                "  cd src/cuda/csrc && python setup.py install\n"
                "Or use FusedRejectionSampler (PyTorch vectorized version) instead."
            )
    
    def forward(
        self,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: List[int],
        draft_probs: Optional[torch.Tensor],
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        is_greedy: bool = False,
    ) -> RejectionSampleOutput:
        """
        執行 rejection sampling
        
        Args:
            draft_token_ids: [num_tokens] - 所有 draft tokens
            num_draft_tokens: [batch_size] - 每個 request 的 draft token 數
            draft_probs: [num_tokens, vocab_size] - draft 機率分布
            target_probs: [num_tokens, vocab_size] - target 機率分布
            bonus_token_ids: [batch_size] - bonus tokens
            is_greedy: 是否使用 greedy mode
        """
        if is_greedy or draft_probs is None:
            # Greedy mode fallback to baseline
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from src.baseline.rejection_sampler import rejection_sample_baseline_greedy
            return rejection_sample_baseline_greedy(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens,
                target_probs=target_probs,
                bonus_token_ids=bonus_token_ids,
            )
        
        return rejection_sample_cuda_kernel(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=draft_probs,
            target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
        )


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import time
    import sys
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    print("=" * 70)
    print("CUDA C++ Extension Test")
    print("=" * 70)
    
    # 檢查 extension 是否可用
    print("\n[Checking CUDA extension availability...]")
    if is_cuda_ext_available():
        print("[SUCCESS] CUDA extension is available!")
    else:
        print("[FAILED] CUDA extension not available.")
        print("\nTo compile, run:")
        print("  cd src/cuda/csrc")
        print("  python setup.py install")
        exit(1)
    
    device = "cuda"
    
    # 生成測試資料
    from src.baseline.rejection_sampler import generate_test_data
    
    data = generate_test_data(
        batch_size=8,
        K=8,
        vocab_size=32000,
        device=device,
        acceptance_rate_target=0.7,
    )
    
    print(f"\nTest Configuration:")
    print(f"  Batch Size: {data['batch_size']}")
    print(f"  Max Spec Length: {data['max_spec_len']}")
    print(f"  Vocab Size: {data['vocab_size']}")
    
    # 測試 CUDA kernel
    print("\n[Testing CUDA kernel...]")
    result = rejection_sample_cuda_kernel(
        draft_token_ids=data["draft_token_ids"],
        num_draft_tokens=data["num_draft_tokens"],
        draft_probs=data["draft_probs"],
        target_probs=data["target_probs"],
        bonus_token_ids=data["bonus_token_ids"],
    )
    
    print(f"\nResults:")
    print(f"  Output Shape: {result.output_token_ids.shape}")
    print(f"  Accepted Counts: {result.accepted_counts.tolist()}")
    print(f"  Recovered Counts: {result.recovered_counts.tolist()}")
    print(f"  Bonus Counts: {result.bonus_counts.tolist()}")
    
    # Benchmark
    print("\n[Benchmarking...]")
    iterations = 200
    
    # Warmup
    for _ in range(20):
        rejection_sample_cuda_kernel(
            draft_token_ids=data["draft_token_ids"],
            num_draft_tokens=data["num_draft_tokens"],
            draft_probs=data["draft_probs"],
            target_probs=data["target_probs"],
            bonus_token_ids=data["bonus_token_ids"],
        )
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        rejection_sample_cuda_kernel(
            draft_token_ids=data["draft_token_ids"],
            num_draft_tokens=data["num_draft_tokens"],
            draft_probs=data["draft_probs"],
            target_probs=data["target_probs"],
            bonus_token_ids=data["bonus_token_ids"],
        )
    torch.cuda.synchronize()
    cuda_ms = (time.perf_counter() - start) / iterations * 1000
    
    print(f"\nCUDA Kernel Time: {cuda_ms:.3f} ms")
    print("=" * 70)
