// Controller research only: int8 tensor-core (mma.sync m16n8k16) MMQ for the
// three fixed B8192 Spark Q8_0 MLP GEMMs, fed by one-time repacked weight
// panels. The dp4a vec_dot path stays compiled beside it as the exact
// reference for validation. Host wrapper structure mirrors
// native_q8_mmq_x64_y64_w4 unchanged; only the inner loop differs.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <initializer_list>

#include "../native_q8_mmq_x8/vendor/ggml-common.h"
#include "../native_q8_mmq_x8/vendor/vecdotq.cuh"
#include "../native_q8_mmq_x8/vendor/quantize_q8_1.cuh"
#include "../native_q8_mmq_x64_y64_w4/mmq_x64_y64_w4_core.cuh"
#include "imma_m16n8k16_core.cuh"

namespace tierengine_q8_imma_m16n8k16_v1 {

namespace {

using q8_imma_m16n8k16::kBlockCols;
using q8_imma_m16n8k16::kBlockRows;

void require_panel(const at::Tensor& panel_s8, const at::Tensor& panel_d,
                   const at::Tensor& x, int64_t rows, int64_t k) {
  TORCH_CHECK(panel_s8.is_cuda() && panel_d.is_cuda() && x.is_cuda(),
              "IMMA requires CUDA tensors");
  TORCH_CHECK(panel_s8.device() == x.device() && panel_d.device() == x.device(),
              "IMMA device mismatch");
  TORCH_CHECK(panel_s8.scalar_type() == at::kChar && panel_d.scalar_type() == at::kHalf
                  && x.scalar_type() == at::kBFloat16,
              "IMMA requires int8 panels, fp16 scales, BF16 activations");
  TORCH_CHECK(panel_s8.is_contiguous() && panel_d.is_contiguous() && x.is_contiguous()
                  && panel_s8.dim() == 4 && panel_d.dim() == 2 && x.dim() == 2,
              "IMMA requires contiguous panel [rows/16,k/16,16,16], scales [rows,k/32], x [batch,k]");
  TORCH_CHECK(panel_s8.size(0) == rows / 16 && panel_s8.size(1) == k / 16
                  && panel_s8.size(2) == 16 && panel_s8.size(3) == 16,
              "IMMA s8 panel tile geometry mismatch");
  TORCH_CHECK(panel_d.size(0) == rows && panel_d.size(1) == k / 32,
              "IMMA scale table geometry mismatch");
  TORCH_CHECK(x.size(1) == k, "IMMA activation width mismatch");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(panel_s8.data_ptr()) % 16 == 0
                  && reinterpret_cast<std::uintptr_t>(panel_d.data_ptr()) % 2 == 0,
              "IMMA panels must be 16B/2B aligned");
}

// Fixed-model admission: (rows, k) in {(6656,2048), (2048,6656)}, batch a
// multiple of 64 in [32,8192] so every mma tile is full (no predication).
int64_t checked_rows_k(const at::Tensor& x, int64_t rows, int64_t k) {
  TORCH_CHECK((rows == 6656 && k == 2048) || (rows == 2048 && k == 6656),
              "IMMA matrix dimensions outside fixed-model scope");
  const int64_t batch = x.size(0);
  TORCH_CHECK(batch >= 32 && batch <= 8192, "IMMA batch outside research scope");
  TORCH_CHECK(batch % kBlockCols == 0 && rows % kBlockRows == 0,
              "IMMA full-tile admission: batch % 64 == 0 required");
  static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36,
                "Unexpected GGUF block packing");
  return batch;
}

}  // namespace

// Generic repacked GEMM: y = x @ w^T for gate/up (6656x2048) and down (2048x6656).
at::Tensor mmq_imma(const at::Tensor& panel_s8, const at::Tensor& panel_d,
                    const at::Tensor& x) {
  const int64_t rows = panel_d.size(0), k = panel_d.size(1) * 32;
  require_panel(panel_s8, panel_d, x, rows, k);
  const int64_t batch = checked_rows_k(x, rows, k);
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty({batch, rows}, x.options());
  auto quant_x = at::empty({batch, k / 32 * 9}, x.options().dtype(at::kInt));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  quantize_row_q8_1_cuda<at::BFloat16>(
      x.data_ptr<at::BFloat16>(), quant_x.data_ptr(), int(k), int(batch), stream);
  q8_imma_m16n8k16::launch_mul_mat_q8_0_imma<at::BFloat16>(
      panel_s8.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(panel_d.data_ptr<at::Half>()), quant_x.data_ptr(),
      y.data_ptr<at::BFloat16>(), int(k), int(rows), int(batch), int(rows), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Paired gate/up with the same host contract as the W4 run_gate_up wrapper:
// two launches into one [8192,13312] output, activation quantized once.
at::Tensor mmq_gate_up_imma(const at::Tensor& gate_s8, const at::Tensor& gate_d,
                            const at::Tensor& up_s8, const at::Tensor& up_d,
                            const at::Tensor& x) {
  constexpr int64_t batch = 8192, col = 2048, rows = 6656, output_stride = 2 * rows;
  require_panel(gate_s8, gate_d, x, rows, col);
  require_panel(up_s8, up_d, x, rows, col);
  TORCH_CHECK(x.size(0) == batch, "IMMA paired gate/up requires batch 8192");
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty({batch, output_stride}, x.options());
  auto quant_x = at::empty({batch, col / 32 * 9}, x.options().dtype(at::kInt));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  quantize_row_q8_1_cuda<at::BFloat16>(
      x.data_ptr<at::BFloat16>(), quant_x.data_ptr(), int(col), int(batch), stream);
  q8_imma_m16n8k16::launch_mul_mat_q8_0_imma<at::BFloat16, int(output_stride)>(
      gate_s8.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(gate_d.data_ptr<at::Half>()), quant_x.data_ptr(),
      y.data_ptr<at::BFloat16>(), int(col), int(rows), int(batch), int(rows), stream);
  q8_imma_m16n8k16::launch_mul_mat_q8_0_imma<at::BFloat16, int(output_stride)>(
      up_s8.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(up_d.data_ptr<at::Half>()), quant_x.data_ptr(),
      y.data_ptr<at::BFloat16>() + rows, int(col), int(rows), int(batch), int(rows),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

namespace {

constexpr int kProbeBatch = 8192, kDownInput = 6656, kDownOutput = 2048;
static_assert(kDownInput % 512 == 0 && kDownInput % CUDA_QUANTIZE_BLOCK_SIZE == 0);
static_assert(CUDA_QUANTIZE_BLOCK_SIZE == CUDA_DEQUANTIZE_BLOCK_SIZE);
static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36);

void require_probe_activation(const at::Tensor& input, int width) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.is_contiguous()
                  && input.dim() == 2 && input.size(0) == kProbeBatch && input.size(1) == width
                  && reinterpret_cast<std::uintptr_t>(input.data_ptr()) % alignof(at::BFloat16) == 0,
              "IMMA GELU probe requires exact aligned CUDA BF16 geometry");
}

// Identical activation SASS-mirrored math to the W4 fused probe.
static __device__ __forceinline__ float gelu_tanh_product_bf16(float gate, float up) {
  float product;
  asm("{\n\t"
      ".reg .f32 t, u;\n\t"
      "mul.rn.ftz.f32 t, %1, 0f3d372713;\n\t"
      "mul.rn.ftz.f32 u, %1, t;\n\t"
      "fma.rn.ftz.f32 t, %1, u, %1;\n\t"
      "mul.rn.ftz.f32 u, t, 0f3f4c422a;\n\t"
      "tanh.approx.f32 t, u;\n\t"
      "add.rn.ftz.f32 u, t, 0f3f800000;\n\t"
      "mul.rn.ftz.f32 t, u, 0f3f000000;\n\t"
      "mul.rn.ftz.f32 u, %1, t;\n\t"
      "mul.rn.ftz.f32 %0, %2, u;\n\t"
      "}"
      : "=f"(product) : "f"(gate), "f"(up));
  return __bfloat162float(__float2bfloat16_rn(product));
}

static __global__ void quantize_gelu_tanh_q8_1(
    const at::BFloat16* __restrict__ gate_up, block_q8_1* __restrict__ out) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * kDownInput + ix;
  const int gate_index = blockIdx.y * (2 * kDownInput) + ix;
  const float gate = static_cast<float>(gate_up[gate_index]);
  const float up = static_cast<float>(gate_up[gate_index + kDownInput]);
  quantize_q8_1_store(gelu_tanh_product_bf16(gate, up), out, i / QK8_1, i % QK8_1);
}

at::Tensor launch_probe_gelu_quant(const at::Tensor& gate_up, cudaStream_t stream) {
  auto quant = at::empty({kProbeBatch, kDownInput / QK8_1 * int(sizeof(block_q8_1) / sizeof(int32_t))},
                         gate_up.options().dtype(at::kInt));
  const dim3 grid(kDownInput / CUDA_QUANTIZE_BLOCK_SIZE, kProbeBatch, 1);
  const dim3 block(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
  quantize_gelu_tanh_q8_1<<<grid, block, 0, stream>>>(
      gate_up.data_ptr<at::BFloat16>(), static_cast<block_q8_1*>(quant.data_ptr()));
  return quant;
}

void require_down_packed_weight(const at::Tensor& weight, const at::Tensor& ref) {
  constexpr int packed_width = kDownInput / QK8_0 * sizeof(block_q8_0);
  TORCH_CHECK(weight.device() == ref.device() && weight.scalar_type() == at::kByte
                  && weight.is_contiguous() && weight.dim() == 2
                  && weight.size(0) == kDownOutput && weight.size(1) == packed_width
                  && reinterpret_cast<std::uintptr_t>(weight.data_ptr()) % alignof(block_q8_0) == 0,
              "IMMA GELU down probe requires aligned Q8_0 [2048,7072] on the input device");
}

}  // namespace

// Fused GELU-tanh product, Q8_1 quantizer, and IMMA down GEMM from the panel.
at::Tensor mmq_gelu_down_probe_imma(const at::Tensor& panel_s8, const at::Tensor& panel_d,
                                    const at::Tensor& gate_up) {
  require_probe_activation(gate_up, 2 * kDownInput);
  require_panel(panel_s8, panel_d, gate_up, kDownOutput, kDownInput);
  const c10::cuda::CUDAGuard guard(gate_up.device());
  auto out = at::empty({kProbeBatch, kDownOutput}, gate_up.options());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(gate_up.get_device()).stream();
  auto quant = launch_probe_gelu_quant(gate_up, stream);
  q8_imma_m16n8k16::launch_mul_mat_q8_0_imma<at::BFloat16>(
      panel_s8.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(panel_d.data_ptr<at::Half>()), quant.data_ptr(),
      out.data_ptr<at::BFloat16>(), kDownInput, kDownOutput, kProbeBatch, kDownOutput,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---- dp4a exact-reference path (unchanged math, same quantizer) ------------

at::Tensor mmq_reference(const at::Tensor& weight, const at::Tensor& x) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda(), "reference requires CUDA tensors");
  TORCH_CHECK(weight.device() == x.device(), "reference device mismatch");
  TORCH_CHECK(weight.scalar_type() == at::kByte && x.scalar_type() == at::kBFloat16,
              "reference requires packed Q8_0 bytes and BF16 activations");
  TORCH_CHECK(weight.dim() == 2 && x.dim() == 2 && weight.is_contiguous() && x.is_contiguous(),
              "reference requires contiguous matrices");
  const int64_t batch = x.size(0), col = x.size(1), row = weight.size(0);
  TORCH_CHECK(batch >= 32 && batch <= 8192, "reference batch outside research scope");
  TORCH_CHECK((col == 2048 || col == 6656) && row > 0 && row <= 131072,
              "reference dimensions outside fixed-model scope");
  TORCH_CHECK(row % 64 == 0, "reference full-tile admission: rows % 64 == 0");
  TORCH_CHECK(weight.size(1) == col / QK8_0 * int64_t(sizeof(block_q8_0)),
              "reference packed Q8_0 row width mismatch");
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty({batch, row}, x.options());
  auto quant_x = at::empty({batch, padded / 32 * 9}, x.options().dtype(at::kInt));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  quantize_row_q8_1_cuda<at::BFloat16>(
      x.data_ptr<at::BFloat16>(), quant_x.data_ptr(), int(col), int(batch), stream);
  ggml_mul_mat_q8_0_research_x64_y64_w4_q8_1_cuda<at::BFloat16>(
      weight.data_ptr(), quant_x.data_ptr(), y.data_ptr<at::BFloat16>(),
      int(col), int(row), int(batch), int(padded), int(row), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor mmq_gelu_down_probe_reference(const at::Tensor& weight, const at::Tensor& gate_up) {
  require_probe_activation(gate_up, 2 * kDownInput);
  require_down_packed_weight(weight, gate_up);
  const c10::cuda::CUDAGuard guard(gate_up.device());
  auto out = at::empty({kProbeBatch, kDownOutput}, gate_up.options());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(gate_up.get_device()).stream();
  auto quant = launch_probe_gelu_quant(gate_up, stream);
  ggml_mul_mat_q8_0_research_x64_y64_w4_q8_1_cuda<at::BFloat16>(
      weight.data_ptr(), quant.data_ptr(), out.data_ptr<at::BFloat16>(),
      kDownInput, kDownOutput, kProbeBatch, kDownInput, kDownOutput, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace tierengine_q8_imma_m16n8k16_v1

TORCH_LIBRARY(tierengine_q8_imma_m16n8k16_v1, m) {
  m.def("mmq_imma(Tensor panel_s8, Tensor panel_d, Tensor x) -> Tensor");
  m.def("mmq_gate_up_imma(Tensor gate_s8, Tensor gate_d, Tensor up_s8, Tensor up_d, Tensor x) -> Tensor");
  m.def("mmq_gelu_down_probe_imma(Tensor panel_s8, Tensor panel_d, Tensor gate_up) -> Tensor");
  m.def("mmq_reference(Tensor weight, Tensor x) -> Tensor");
  m.def("mmq_gelu_down_probe_reference(Tensor weight, Tensor gate_up) -> Tensor");
}
TORCH_LIBRARY_IMPL(tierengine_q8_imma_m16n8k16_v1, CUDA, m) {
  m.impl("mmq_imma", &tierengine_q8_imma_m16n8k16_v1::mmq_imma);
  m.impl("mmq_gate_up_imma", &tierengine_q8_imma_m16n8k16_v1::mmq_gate_up_imma);
  m.impl("mmq_gelu_down_probe_imma", &tierengine_q8_imma_m16n8k16_v1::mmq_gelu_down_probe_imma);
  m.impl("mmq_reference", &tierengine_q8_imma_m16n8k16_v1::mmq_reference);
  m.impl("mmq_gelu_down_probe_reference", &tierengine_q8_imma_m16n8k16_v1::mmq_gelu_down_probe_reference);
}
