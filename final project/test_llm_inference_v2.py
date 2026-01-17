#!/usr/bin/env python3
"""
Real LLM Inference Test with CUDA Kernel V1 vs V2
==================================================

實際使用 LLM 進行推理，比較原版和優化版 CUDA kernel 的效果
"""

import os
import sys
import torch
import time
from dataclasses import dataclass

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'cuda', 'csrc'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    import fused_rejection_cuda
    print("✓ CUDA extension loaded")
    print(f"  Available: {[x for x in dir(fused_rejection_cuda) if not x.startswith('_')]}")
except ImportError as e:
    print(f"✗ Failed to load CUDA extension: {e}")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SamplingResult:
    output_token_ids: torch.Tensor
    num_accepted: torch.Tensor


def cuda_rejection_sample_v1(draft_probs, target_probs, draft_token_ids, 
                              bonus_token_ids, cu_num_draft, uniform_samples, max_spec_len):
    """Original CUDA kernel"""
    return fused_rejection_cuda.fused_rejection_sample(
        draft_probs, target_probs, draft_token_ids,
        bonus_token_ids, cu_num_draft, uniform_samples, max_spec_len
    )


def cuda_rejection_sample_v2(draft_probs, target_probs, draft_token_ids, 
                              bonus_token_ids, cu_num_draft, uniform_samples, max_spec_len):
    """Optimized CUDA kernel V2"""
    return fused_rejection_cuda.fused_rejection_sample_v2(
        draft_probs, target_probs, draft_token_ids,
        bonus_token_ids, cu_num_draft, uniform_samples, max_spec_len
    )


class SpeculativeDecoderBenchmark:
    def __init__(self, model_name: str, device: str = 'cuda'):
        print(f"\nLoading model: {model_name}")
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        
        self.vocab_size = self.model.config.vocab_size
        print(f"✓ Model loaded: vocab_size={self.vocab_size:,}")
        
    def speculative_decode_step(self, input_ids, spec_len=8, temperature=1.0, use_v2=False):
        """單步 speculative decoding"""
        batch_size = input_ids.shape[0]
        
        # 1. Draft: 自回歸生成 spec_len tokens
        draft_ids = []
        current_ids = input_ids
        draft_logits_list = []
        
        with torch.no_grad():
            for _ in range(spec_len):
                outputs = self.model(current_ids)
                logits = outputs.logits[:, -1:, :]  # [batch, 1, vocab]
                draft_logits_list.append(logits)
                
                # Greedy sampling for draft
                next_token = logits.argmax(dim=-1)  # [batch, 1]
                draft_ids.append(next_token)
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        draft_ids = torch.cat(draft_ids, dim=1)  # [batch, spec_len]
        draft_logits = torch.cat(draft_logits_list, dim=1)  # [batch, spec_len, vocab]
        
        # 2. Target: 單次前向驗證所有 draft tokens
        full_ids = torch.cat([input_ids, draft_ids], dim=1)
        with torch.no_grad():
            target_outputs = self.model(full_ids)
            target_logits = target_outputs.logits[:, -(spec_len+1):, :]
        
        # 只取前 spec_len 個用於驗證
        target_verify_logits = target_logits[:, :-1, :]
        bonus_logits = target_logits[:, -1:, :]
        
        # 3. 準備 rejection sampling 的輸入
        draft_probs = torch.softmax(draft_logits / temperature, dim=-1).float()
        target_probs = torch.softmax(target_verify_logits / temperature, dim=-1).float()
        
        # Flatten
        draft_probs_flat = draft_probs.reshape(-1, self.vocab_size).contiguous()
        target_probs_flat = target_probs.reshape(-1, self.vocab_size).contiguous()
        draft_token_ids_flat = draft_ids.reshape(-1).contiguous()
        
        # Bonus token (sample from last position)
        bonus_token_ids = bonus_logits.argmax(dim=-1).squeeze(-1).contiguous()
        
        # Cumulative draft counts
        cu_num_draft = torch.arange(0, batch_size + 1, device=self.device, dtype=torch.int64) * spec_len
        
        # Uniform samples
        uniform_samples = torch.rand(batch_size * spec_len, device=self.device, dtype=torch.float32)
        
        # 4. Rejection Sampling
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        if use_v2:
            result = cuda_rejection_sample_v2(
                draft_probs_flat, target_probs_flat, draft_token_ids_flat,
                bonus_token_ids, cu_num_draft, uniform_samples, spec_len
            )
        else:
            result = cuda_rejection_sample_v1(
                draft_probs_flat, target_probs_flat, draft_token_ids_flat,
                bonus_token_ids, cu_num_draft, uniform_samples, spec_len
            )
        
        torch.cuda.synchronize()
        rejection_time = (time.perf_counter() - start) * 1000  # ms
        
        output_tokens = result[0]  # [batch, max_spec_len+1]
        num_accepted = result[1]   # [batch]
        
        return output_tokens, num_accepted, rejection_time
    
    def generate_with_timing(self, prompt: str, max_tokens: int = 50, spec_len: int = 8, 
                             use_v2: bool = False, temperature: float = 1.0):
        """Generate with detailed timing"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids.clone()
        
        total_rejection_time = 0.0
        total_steps = 0
        total_accepted = 0
        
        start_time = time.perf_counter()
        
        while generated_ids.shape[1] - input_ids.shape[1] < max_tokens:
            output_tokens, num_accepted, rejection_time = self.speculative_decode_step(
                generated_ids, spec_len=spec_len, temperature=temperature, use_v2=use_v2
            )
            
            total_rejection_time += rejection_time
            total_steps += 1
            
            # 取得接受的 tokens
            n_accepted = num_accepted[0].item()
            total_accepted += n_accepted
            
            valid_tokens = output_tokens[0, :n_accepted]
            valid_tokens = valid_tokens[valid_tokens >= 0]
            
            if len(valid_tokens) == 0:
                break
            
            generated_ids = torch.cat([
                generated_ids, 
                valid_tokens.unsqueeze(0).long()
            ], dim=1)
            
            # Check for EOS
            if self.tokenizer.eos_token_id in valid_tokens.tolist():
                break
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return {
            'text': generated_text,
            'total_time_ms': total_time,
            'rejection_time_ms': total_rejection_time,
            'rejection_pct': (total_rejection_time / total_time) * 100,
            'num_steps': total_steps,
            'avg_accepted': total_accepted / max(total_steps, 1),
            'tokens_generated': generated_ids.shape[1] - input_ids.shape[1],
        }


def main():
    print("=" * 80)
    print("Real LLM Inference: CUDA Kernel V1 vs V2 Comparison")
    print("=" * 80)
    
    # 使用小模型進行測試
    model_name = "Qwen/Qwen2.5-0.5B"  # 小模型，快速測試
    
    device = torch.device('cuda')
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    
    decoder = SpeculativeDecoderBenchmark(model_name, device='cuda')
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The key to successful machine learning is",
    ]
    
    spec_lens = [4, 8, 12]
    
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    
    for spec_len in spec_lens:
        print(f"\n--- Speculation Length: {spec_len} ---")
        print(f"{'Kernel':<10} {'Total(ms)':<12} {'Reject(ms)':<12} {'Reject%':<10} {'Tokens/s':<10}")
        print("-" * 60)
        
        for prompt in prompts[:1]:  # Just one prompt for speed
            # Warmup
            _ = decoder.generate_with_timing(prompt, max_tokens=20, spec_len=spec_len, use_v2=False)
            _ = decoder.generate_with_timing(prompt, max_tokens=20, spec_len=spec_len, use_v2=True)
            
            # V1 benchmark
            result_v1 = decoder.generate_with_timing(prompt, max_tokens=50, spec_len=spec_len, use_v2=False)
            tokens_per_sec_v1 = result_v1['tokens_generated'] / (result_v1['total_time_ms'] / 1000)
            
            # V2 benchmark
            result_v2 = decoder.generate_with_timing(prompt, max_tokens=50, spec_len=spec_len, use_v2=True)
            tokens_per_sec_v2 = result_v2['tokens_generated'] / (result_v2['total_time_ms'] / 1000)
            
            print(f"{'V1':<10} {result_v1['total_time_ms']:<12.2f} {result_v1['rejection_time_ms']:<12.4f} {result_v1['rejection_pct']:<10.2f} {tokens_per_sec_v1:<10.1f}")
            print(f"{'V2':<10} {result_v2['total_time_ms']:<12.2f} {result_v2['rejection_time_ms']:<12.4f} {result_v2['rejection_pct']:<10.2f} {tokens_per_sec_v2:<10.1f}")
            
            speedup = result_v1['rejection_time_ms'] / max(result_v2['rejection_time_ms'], 0.001)
            overall_speedup = result_v1['total_time_ms'] / result_v2['total_time_ms']
            
            print(f"\n  Rejection Sampling Speedup: {speedup:.2f}x")
            print(f"  Overall Throughput Change: {overall_speedup:.3f}x")
    
    # Final comparison with longer generation
    print("\n" + "=" * 80)
    print("Extended Generation Test (100 tokens)")
    print("=" * 80)
    
    prompt = "The development of large language models has revolutionized"
    
    for use_v2, name in [(False, 'V1 (Original)'), (True, 'V2 (Optimized)')]:
        result = decoder.generate_with_timing(prompt, max_tokens=100, spec_len=8, use_v2=use_v2)
        print(f"\n{name}:")
        print(f"  Total Time: {result['total_time_ms']:.2f} ms")
        print(f"  Rejection Sampling: {result['rejection_time_ms']:.4f} ms ({result['rejection_pct']:.2f}%)")
        print(f"  Tokens Generated: {result['tokens_generated']}")
        print(f"  Throughput: {result['tokens_generated'] / (result['total_time_ms'] / 1000):.1f} tokens/sec")
        print(f"  Avg Accepted per Step: {result['avg_accepted']:.2f}")


if __name__ == '__main__':
    main()
