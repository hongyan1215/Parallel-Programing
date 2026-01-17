# SPDX-License-Identifier: Apache-2.0
# Rejection Sampler module for Speculative Decoding

# Note: The main implementation requires vLLM dependencies.
# For standalone testing, the benchmark script uses mock implementations.

try:
    from .Rejection_Sampler import RejectionSampler, rejection_sample
    __all__ = ['RejectionSampler', 'rejection_sample']
except ImportError as e:
    # vLLM dependencies not available
    import warnings
    warnings.warn(f"Could not import RejectionSampler: {e}. Using mock implementations.")
    RejectionSampler = None
    rejection_sample = None
