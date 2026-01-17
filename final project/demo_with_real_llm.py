#!/usr/bin/env python
"""
Real LLM Demo with CUDA Fused Rejection Sampler
================================================

使用真實的小型 LLM 展示 Speculative Decoding + CUDA Fused Kernel

推薦模型組合:
- Draft: meta-llama/Llama-3.2-1B (1B)
- Target: meta-llama/Llama-3.2-3B (3B parameters)

使用同系列模型可以獲得更高的 acceptance rate，展示真正的 Speculative Decoding 優勢
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Add DLL directory for CUDA (Windows only)
if sys.platform == 'win32':
    try:
        os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin')
    except (AttributeError, OSError):
        pass

sys.path.insert(0, 'src/cuda/csrc')
sys.path.insert(0, 'src')

try:
    import fused_rejection_cuda
    CUDA_EXT_AVAILABLE = True
except ImportError:
    CUDA_EXT_AVAILABLE = False
    print("⚠️ CUDA extension not available, will use PyTorch vectorized version")

from cuda.fused_sampler import rejection_sample_fused_kernel
from baseline.rejection_sampler import rejection_sample_baseline


class SpeculativeDecoder:
    """
    Speculative Decoding with CUDA Fused Rejection Sampler
    """
    def __init__(self, draft_model_name: str, target_model_name: str = None, device: str = 'cuda'):
        """
        Args:
            draft_model_name: Draft model (小模型，如 TinyLlama-1.1B)
            target_model_name: Target model (大模型)，None 表示使用同一個模型
            device: 'cuda' or 'cpu'
        """
        print("=" * 80)
        print("Loading Models...")
        print("=" * 80)
        
        self.device = device
        
        # Load draft model
        print(f"\nLoading Draft Model: {draft_model_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.draft_model.eval()
        
        # Load target model (或使用同一個模型展示)
        if target_model_name is None:
            print(f"Using same model as target (for demo purposes)")
            self.target_model = self.draft_model
        else:
            print(f"\nLoading Target Model: {target_model_name}")
            self.target_model = AutoModelForCausalLM.from_pretrained(
                target_model_name,
                torch_dtype=torch.float16,
                device_map=device,
            )
            self.target_model.eval()
        
        # Load tokenizer (使用 target model 的 tokenizer)
        tokenizer_model = target_model_name if target_model_name else draft_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 預分配常用張量以減少開銷
        self._cu_num_draft_cache = {}
        self._uniform_samples_cache = {}
        
        print(f"\nModels loaded successfully!")
        print(f"   Draft model params: {sum(p.numel() for p in self.draft_model.parameters()) / 1e9:.2f}B")
        print(f"   Target model params: {sum(p.numel() for p in self.target_model.parameters()) / 1e9:.2f}B")
        print(f"   Vocab size - Draft: {self.draft_model.config.vocab_size}, Target: {self.target_model.config.vocab_size}")
    
    @torch.no_grad()
    def generate_speculative(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        spec_len: int = 4,
        temperature: float = 1.0,
        use_cuda_kernel: bool = True,
    ):
        """
        使用 Speculative Decoding 生成文字
        
        Args:
            prompt: 輸入文字
            max_new_tokens: 最多生成幾個 token
            spec_len: Speculation length (draft model 一次猜幾個)
            temperature: 採樣溫度
            use_cuda_kernel: True=使用CUDA kernel, False=使用baseline
        """
        method_name = "CUDA Fused Kernel" if use_cuda_kernel else "Baseline (Python loop)"
        print("\n" + "=" * 80)
        print(f"Speculative Decoding Generation - {method_name}")
        print("=" * 80)
        print(f"Prompt: {prompt}")
        print(f"Spec Length: {spec_len}, Max Tokens: {max_new_tokens}")
        print("-" * 80)
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids.clone()
        
        total_accepted = 0
        total_drafts = 0
        start_time = time.time()
        
        for step in range(max_new_tokens // spec_len):
            # 1. Draft model 生成 spec_len 個 tokens
            draft_ids = self._draft_generate(generated_ids, spec_len)
            
            # 2. Target model 驗證
            if use_cuda_kernel:
                accepted_ids, n_accepted = self._verify_with_cuda(
                    generated_ids, draft_ids, spec_len, temperature
                )
            else:
                accepted_ids, n_accepted = self._verify_with_baseline(
                    generated_ids, draft_ids, spec_len, temperature
                )
            
            # 3. 更新
            generated_ids = torch.cat([generated_ids, accepted_ids], dim=1)
            total_accepted += n_accepted
            total_drafts += spec_len
            
            # 即時顯示
            print(f"\rStep {step+1}: Accepted {n_accepted}/{spec_len} | Total: {generated_ids.shape[1]} tokens", end='')
            
            if generated_ids.shape[1] >= input_ids.shape[1] + max_new_tokens:
                break
        
        end_time = time.time()
        
        # 結果
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        acceptance_rate = total_accepted / total_drafts if total_drafts > 0 else 0
        
        print(f"\n\n{'='*80}")
        print("Results")
        print("=" * 80)
        print(f"Generated Text:\n{output_text}\n")
        print("-" * 80)
        print(f"Total Tokens: {generated_ids.shape[1] - input_ids.shape[1]}")
        print(f"Acceptance Rate: {acceptance_rate:.1%} ({total_accepted}/{total_drafts})")
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Speed: {(generated_ids.shape[1] - input_ids.shape[1]) / (end_time - start_time):.1f} tokens/s")
        
        return output_text, end_time - start_time
    
    @torch.no_grad()
    def generate_autoregressive(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ):
        """
        標準的 autoregressive 生成（沒有 speculative decoding）
        
        Args:
            prompt: 輸入文字
            max_new_tokens: 最多生成幾個 token
            temperature: 採樣溫度
        """
        print("\n" + "=" * 80)
        print("Standard Autoregressive Generation (No Speculative Decoding)")
        print("=" * 80)
        print(f"Prompt: {prompt}")
        print(f"Max Tokens: {max_new_tokens}")
        print("-" * 80)
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids.clone()
        
        start_time = time.time()
        
        # Standard autoregressive generation - one token at a time
        for step in range(max_new_tokens):
            outputs = self.target_model(generated_ids)
            probs = torch.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            print(f"\rStep {step+1}/{max_new_tokens} | Total: {generated_ids.shape[1]} tokens", end='')
        
        end_time = time.time()
        
        # 結果
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"\n\n{'='*80}")
        print("Results")
        print("=" * 80)
        print(f"Generated Text:\n{output_text}\n")
        print("-" * 80)
        print(f"Total Tokens: {generated_ids.shape[1] - input_ids.shape[1]}")
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Speed: {(generated_ids.shape[1] - input_ids.shape[1]) / (end_time - start_time):.1f} tokens/s")
        
        return output_text, end_time - start_time
    
    def _draft_generate(self, input_ids: torch.Tensor, spec_len: int) -> torch.Tensor:
        """Draft model 生成 spec_len 個 tokens"""
        draft_ids = []
        current_ids = input_ids
        
        for _ in range(spec_len):
            outputs = self.draft_model(current_ids)
            # Use sampling instead of greedy
            probs = torch.softmax(outputs.logits[:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            draft_ids.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        return torch.cat(draft_ids, dim=1)
    
    def _verify_with_cuda(
        self,
        prefix_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        spec_len: int,
        temperature: float,
    ):
        """使用 CUDA Fused Kernel 驗證 draft tokens"""
        batch_size = prefix_ids.shape[0]
        vocab_size = self.target_model.config.vocab_size
        
        # 取得 draft 和 target 的 logits
        full_ids = torch.cat([prefix_ids, draft_ids], dim=1)
        
        # Draft logits (pad to target vocab size if needed)
        with torch.no_grad():
            draft_outputs = self.draft_model(full_ids[:, :-1])
            draft_logits = draft_outputs.logits[:, -spec_len:, :]  # [batch, spec_len, draft_vocab]
            if draft_logits.shape[-1] < vocab_size:
                padding = torch.full(
                    (draft_logits.shape[0], draft_logits.shape[1], vocab_size - draft_logits.shape[-1]),
                    float('-inf'), device=draft_logits.device, dtype=draft_logits.dtype
                )
                draft_logits = torch.cat([draft_logits, padding], dim=-1)
        
        # Target logits
        with torch.no_grad():
            target_outputs = self.target_model(full_ids[:, :-1])
            target_logits = target_outputs.logits[:, -spec_len:, :]  # [batch, spec_len, vocab]
        
        # Convert to probs (ensure float32 for CUDA kernel)
        draft_probs = torch.softmax(draft_logits / temperature, dim=-1).float()
        target_probs = torch.softmax(target_logits / temperature, dim=-1).float()
        
        # Flatten for rejection sampling
        draft_probs_flat = draft_probs.reshape(-1, vocab_size)
        target_probs_flat = target_probs.reshape(-1, vocab_size)
        draft_token_ids_flat = draft_ids.reshape(-1)
        
        # Bonus token (從最後的 target distribution 採樣)
        bonus_token_ids = torch.multinomial(target_probs[:, -1, :], num_samples=1)
        
        # CUDA Rejection Sampling
        if CUDA_EXT_AVAILABLE:
            result = self._cuda_rejection_sample(
                draft_probs_flat, target_probs_flat, draft_token_ids_flat,
                bonus_token_ids, batch_size, spec_len
            )
        else:
            result = rejection_sample_fused_kernel(
                draft_token_ids=draft_token_ids_flat,
                num_draft_tokens=[spec_len] * batch_size,
                draft_probs=draft_probs_flat,
                target_probs=target_probs_flat,
                bonus_token_ids=bonus_token_ids,
            )
        
        # Extract accepted tokens
        accepted_ids = result.output_token_ids[0, :result.num_accepted[0]]
        accepted_ids = accepted_ids[accepted_ids >= 0]  # Remove placeholders
        
        return accepted_ids.unsqueeze(0), len(accepted_ids)
    
    def _cuda_rejection_sample(self, draft_probs, target_probs, draft_token_ids,
                               bonus_token_ids, batch_size, spec_len):
        """Call CUDA extension - optimized with caching"""
        # 使用緩存的張量以減少分配開銷
        cache_key = (batch_size, spec_len)
        if cache_key not in self._cu_num_draft_cache:
            self._cu_num_draft_cache[cache_key] = torch.tensor(
                [i * spec_len for i in range(batch_size + 1)],
                device=self.device, dtype=torch.int64
            )
        cu_num_draft = self._cu_num_draft_cache[cache_key]
        
        # 重用 uniform samples 張量
        total_samples = batch_size * spec_len
        if total_samples not in self._uniform_samples_cache:
            self._uniform_samples_cache[total_samples] = torch.empty(
                total_samples, device=self.device, dtype=torch.float32
            )
        uniform_samples = self._uniform_samples_cache[total_samples].uniform_()
        
        output_tokens, num_accepted, accepted_counts, recovered_counts, bonus_counts = \
            fused_rejection_cuda.fused_rejection_sample(
                draft_probs, target_probs, draft_token_ids,
                bonus_token_ids.squeeze(-1), cu_num_draft, uniform_samples, spec_len
            )
        
        # Convert to RejectionSampleOutput format
        from dataclasses import dataclass
        
        @dataclass
        class Result:
            output_token_ids: torch.Tensor
            num_accepted: torch.Tensor
            accepted_counts: torch.Tensor
            recovered_counts: torch.Tensor
            bonus_counts: torch.Tensor
        
        return Result(output_tokens, num_accepted, accepted_counts, recovered_counts, bonus_counts)
    
    def _verify_with_baseline(
        self,
        prefix_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        spec_len: int,
        temperature: float,
    ):
        """使用 Baseline Rejection Sampler 驗證 draft tokens"""
        batch_size = prefix_ids.shape[0]
        vocab_size = self.target_model.config.vocab_size
        
        # 取得 draft 和 target 的 logits
        full_ids = torch.cat([prefix_ids, draft_ids], dim=1)
        
        # Draft logits (pad to target vocab size if needed)
        with torch.no_grad():
            draft_outputs = self.draft_model(full_ids[:, :-1])
            draft_logits = draft_outputs.logits[:, -spec_len:, :]  # [batch, spec_len, draft_vocab]
            if draft_logits.shape[-1] < vocab_size:
                padding = torch.full(
                    (draft_logits.shape[0], draft_logits.shape[1], vocab_size - draft_logits.shape[-1]),
                    float('-inf'), device=draft_logits.device, dtype=draft_logits.dtype
                )
                draft_logits = torch.cat([draft_logits, padding], dim=-1)
        
        # Target logits
        with torch.no_grad():
            target_outputs = self.target_model(full_ids[:, :-1])
            target_logits = target_outputs.logits[:, -spec_len:, :]  # [batch, spec_len, vocab]
        
        # Convert to probs (ensure float32)
        draft_probs = torch.softmax(draft_logits / temperature, dim=-1).float()
        target_probs = torch.softmax(target_logits / temperature, dim=-1).float()
        
        # Flatten for rejection sampling
        draft_probs_flat = draft_probs.reshape(-1, vocab_size)
        target_probs_flat = target_probs.reshape(-1, vocab_size)
        draft_token_ids_flat = draft_ids.reshape(-1)
        
        # Bonus token
        bonus_token_ids = torch.multinomial(target_probs[:, -1, :], num_samples=1)
        
        # Baseline Rejection Sampling
        uniform_samples = torch.rand(batch_size * spec_len, device=self.device)
        result = rejection_sample_baseline(
            draft_token_ids=draft_token_ids_flat,
            num_draft_tokens=[spec_len] * batch_size,
            draft_probs=draft_probs_flat,
            target_probs=target_probs_flat,
            bonus_token_ids=bonus_token_ids,
            uniform_samples=uniform_samples,
        )
        
        # Extract accepted tokens
        accepted_ids = result.output_token_ids[0, :result.num_accepted[0]]
        accepted_ids = accepted_ids[accepted_ids >= 0]  # Remove placeholders
        
        return accepted_ids.unsqueeze(0), len(accepted_ids)


def main():
    """
    Demo: 三方比較 - No Spec Decode vs Baseline vs CUDA Kernel
    """
    print("\n" + "=" * 80)
    print("   Final Project Demo: Speculative Decoding with Real LLM")
    print("   Three-Way Performance Comparison")
    print("=" * 80)
    print()
    
    # 模型選擇 - 使用公開模型 Qwen2.5 (不需要授權)
    draft_model = "Qwen/Qwen2.5-1.5B"      # 1.5B 參數作為 draft model
    target_model = "Qwen/Qwen2.5-3B"       # 3B 參數作為 target model
    
    # 初始化
    decoder = SpeculativeDecoder(
        draft_model_name=draft_model,
        target_model_name=target_model,
        device='cuda'
    )
    
    # 測試 prompt - 使用多個 prompt 來增加 batch size
    test_prompts = [
        "Once upon a time, in a small village",
        "In a distant land, there lived",
        "The story begins with a young",
        "Long ago, in a magical kingdom"
    ]
    # 只用第一個 prompt 進行測試（batch=1）
    test_prompt = test_prompts[0]
    max_tokens = 30
    spec_len = 8  # 增加 speculation length
    n_runs = 3
    
    print("\n" + "=" * 80)
    print("  Three-Way Performance Test")
    print("=" * 80)
    print(f"Prompt: {test_prompt}")
    print(f"Max Tokens: {max_tokens}, Spec Length: {spec_len}")
    print(f"Runs per method: {n_runs}")
    print()
    
    # 預熱
    print("Warming up GPU...")
    _, _ = decoder.generate_speculative(
        prompt=test_prompt,
        max_new_tokens=max_tokens,
        spec_len=spec_len,
        temperature=0.8,
        use_cuda_kernel=True,
    )
    
    # =========================================================================
    # Test 1: 標準 Autoregressive（沒有 Spec Decode）
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Test 1: Standard Autoregressive (No Speculative Decoding)")
    print("=" * 80)
    
    autoregressive_times = []
    for i in range(n_runs):
        print(f"\nRun {i+1}/{n_runs}...")
        _, run_time = decoder.generate_autoregressive(
            prompt=test_prompt,
            max_new_tokens=max_tokens,
            temperature=0.8,
        )
        autoregressive_times.append(run_time)
    
    avg_autoregressive_time = sum(autoregressive_times) / len(autoregressive_times)
    
    # =========================================================================
    # Test 2: Speculative Decoding with Baseline Rejection Sampler
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Test 2: Spec Decode + Baseline Rejection Sampler (Python loop)")
    print("=" * 80)
    
    baseline_times = []
    for i in range(n_runs):
        print(f"\nRun {i+1}/{n_runs}...")
        _, run_time = decoder.generate_speculative(
            prompt=test_prompt,
            max_new_tokens=max_tokens,
            spec_len=spec_len,
            temperature=0.8,
            use_cuda_kernel=False,
        )
        baseline_times.append(run_time)
    
    avg_baseline_time = sum(baseline_times) / len(baseline_times)
    
    # =========================================================================
    # Test 3: Speculative Decoding with CUDA Fused Kernel
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Test 3: Spec Decode + CUDA Fused Rejection Sampler")
    print("=" * 80)
    
    cuda_times = []
    for i in range(n_runs):
        print(f"\nRun {i+1}/{n_runs}...")
        _, run_time = decoder.generate_speculative(
            prompt=test_prompt,
            max_new_tokens=max_tokens,
            spec_len=spec_len,
            temperature=0.8,
            use_cuda_kernel=True,
        )
        cuda_times.append(run_time)
    
    avg_cuda_time = sum(cuda_times) / len(cuda_times)
    
    # =========================================================================
    # 結果比較
    # =========================================================================
    print("\n" + "=" * 80)
    print("  FINAL PERFORMANCE COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Method':<50} | {'Avg Time (s)':>12} | {'Speedup':>10}")
    print("-" * 80)
    print(f"{'1. No Spec Decode (Standard Autoregressive)':<50} | {avg_autoregressive_time:>12.3f} | {'1.00x':>10}")
    print(f"{'2. Spec Decode + Baseline (Python loop)':<50} | {avg_baseline_time:>12.3f} | {avg_autoregressive_time/avg_baseline_time:>9.2f}x")
    print(f"{'3. Spec Decode + CUDA Fused Kernel':<50} | {avg_cuda_time:>12.3f} | {avg_autoregressive_time/avg_cuda_time:>9.2f}x")
    print()
    print("Key Insights:")
    print(f"  - Speculative Decoding speedup: {avg_autoregressive_time/avg_baseline_time:.2f}x (baseline) / {avg_autoregressive_time/avg_cuda_time:.2f}x (CUDA)")
    print(f"  - CUDA Kernel vs Baseline: {avg_baseline_time/avg_cuda_time:.2f}x faster")
    print(f"  - Total speedup over no optimization: {avg_autoregressive_time/avg_cuda_time:.2f}x")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
