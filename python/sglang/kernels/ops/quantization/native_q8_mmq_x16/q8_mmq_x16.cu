// Controller research only: original Q8_1 activation quantizer and Q8_0 MMQ,
// with CUDA MMQ_X 16 instead of 4. One BF16 specialization, no wheel override.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <climits>

#include "../native_q8_mmq_x8/vendor/ggml-common.h"
#include "../native_q8_mmq_x8/vendor/vecdotq.cuh"
#include "../native_q8_mmq_x8/vendor/quantize_q8_1.cuh"
#include "mmq_x16_core.cuh"

namespace tierengine_q8_mmq_x16_v1 {

at::Tensor run(const at::Tensor& weight, const at::Tensor& x) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda(), "X16 requires CUDA tensors");
  TORCH_CHECK(weight.device() == x.device(), "X16 device mismatch");
  TORCH_CHECK(weight.scalar_type() == at::kByte && x.scalar_type() == at::kBFloat16,
              "X16 requires packed Q8_0 bytes and BF16 activations");
  TORCH_CHECK(weight.dim() == 2 && x.dim() == 2 && weight.is_contiguous() && x.is_contiguous(),
              "X16 requires contiguous matrices");
  const int64_t batch = x.size(0), col = x.size(1), row = weight.size(0);
  TORCH_CHECK(batch >= 32 && batch <= 8192, "X16 batch outside research scope");
  TORCH_CHECK((col == 2048 || col == 6656) && row > 0 && row <= 131072,
              "X16 matrix dimensions outside fixed-model scope");
  static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36,
                "Unexpected GGUF block packing");
  TORCH_CHECK(weight.size(1) == col / QK8_0 * int64_t(sizeof(block_q8_0)),
              "X16 packed Q8_0 row width mismatch");
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  TORCH_CHECK(batch * padded <= INT_MAX && row * batch <= INT_MAX,
              "X16 original int-index arithmetic would overflow");
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty({batch, row}, x.options());
  auto quant_x = at::empty({batch, padded / 32 * 9},
                          x.options().dtype(at::kInt));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  // Same source quantizer and same scalar type as the original AT_DISPATCH BF16 route.
  quantize_row_q8_1_cuda<at::BFloat16>(
      x.data_ptr<at::BFloat16>(), quant_x.data_ptr(), int(col), int(batch), stream);
  ggml_mul_mat_q8_0_research_x16_q8_1_cuda<at::BFloat16>(
      weight.data_ptr(), quant_x.data_ptr(), y.data_ptr<at::BFloat16>(),
      int(col), int(row), int(batch), int(padded), int(row), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

}  // namespace tierengine_q8_mmq_x16_v1

TORCH_LIBRARY(tierengine_q8_mmq_x16_v1, m) {
  m.def("mmq(Tensor weight, Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(tierengine_q8_mmq_x16_v1, CUDA, m) {
  m.impl("mmq", &tierengine_q8_mmq_x16_v1::run);
}
