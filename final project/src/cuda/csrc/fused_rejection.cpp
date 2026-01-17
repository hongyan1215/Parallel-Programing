/*
 * PyTorch C++ Extension Binding
 * ==============================
 * 
 * 將 CUDA kernel 暴露給 Python
 */

#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA functions
std::vector<torch::Tensor> fused_rejection_sample_cuda(
    torch::Tensor draft_probs,
    torch::Tensor target_probs,
    torch::Tensor draft_token_ids,
    torch::Tensor bonus_token_ids,
    torch::Tensor cu_num_draft,
    torch::Tensor uniform_samples,
    int max_spec_len
);

// 優化版本 V2
std::vector<torch::Tensor> fused_rejection_sample_v2_cuda(
    torch::Tensor draft_probs,
    torch::Tensor target_probs,
    torch::Tensor draft_token_ids,
    torch::Tensor bonus_token_ids,
    torch::Tensor cu_num_draft,
    torch::Tensor uniform_samples,
    int max_spec_len
);


// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_rejection_sample",
        &fused_rejection_sample_cuda,
        "Fused Rejection Sampling CUDA Kernel (Original)",
        py::arg("draft_probs"),
        py::arg("target_probs"),
        py::arg("draft_token_ids"),
        py::arg("bonus_token_ids"),
        py::arg("cu_num_draft"),
        py::arg("uniform_samples"),
        py::arg("max_spec_len")
    );
    
    m.def(
        "fused_rejection_sample_v2",
        &fused_rejection_sample_v2_cuda,
        "Fused Rejection Sampling CUDA Kernel (Optimized V2)",
        py::arg("draft_probs"),
        py::arg("target_probs"),
        py::arg("draft_token_ids"),
        py::arg("bonus_token_ids"),
        py::arg("cu_num_draft"),
        py::arg("uniform_samples"),
        py::arg("max_spec_len")
    );
}
