// Controller research only: original Q8_1 activation quantizer and Q8_0 MMQ,
// with CUDA MMQ_X 8 instead of 4. One BF16 specialization, no wheel override.
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
#include "mmq_x64_y64_w4_core.cuh"

namespace tierengine_q8_mmq_x64_y64_w4_v1 {

at::Tensor run(const at::Tensor& weight, const at::Tensor& x) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda(), "X8 requires CUDA tensors");
  TORCH_CHECK(weight.device() == x.device(), "X8 device mismatch");
  TORCH_CHECK(weight.scalar_type() == at::kByte && x.scalar_type() == at::kBFloat16,
              "X8 requires packed Q8_0 bytes and BF16 activations");
  TORCH_CHECK(weight.dim() == 2 && x.dim() == 2 && weight.is_contiguous() && x.is_contiguous(),
              "X8 requires contiguous matrices");
  const int64_t batch = x.size(0), col = x.size(1), row = weight.size(0);
  TORCH_CHECK(batch >= 32 && batch <= 8192, "X8 batch outside research scope");
  TORCH_CHECK((col == 2048 || col == 6656) && row > 0 && row <= 131072,
              "X8 matrix dimensions outside fixed-model scope");
  TORCH_CHECK(row % 64 == 0,
              "x64_y64_w4 full-tile admission: weight rows must be divisible by 64");
  static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36,
                "Unexpected GGUF block packing");
  TORCH_CHECK(weight.size(1) == col / QK8_0 * int64_t(sizeof(block_q8_0)),
              "X8 packed Q8_0 row width mismatch");
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  TORCH_CHECK(batch * padded <= INT_MAX && row * batch <= INT_MAX,
              "X8 original int-index arithmetic would overflow");
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty({batch, row}, x.options());
  auto quant_x = at::empty({batch, padded / 32 * 9},
                          x.options().dtype(at::kInt));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  // Same source quantizer and same scalar type as the original AT_DISPATCH BF16 route.
  quantize_row_q8_1_cuda<at::BFloat16>(
      x.data_ptr<at::BFloat16>(), quant_x.data_ptr(), int(col), int(batch), stream);
  ggml_mul_mat_q8_0_research_x64_y64_w4_q8_1_cuda<at::BFloat16>(
      weight.data_ptr(), quant_x.data_ptr(), y.data_ptr<at::BFloat16>(),
      int(col), int(row), int(batch), int(padded), int(row), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor run_gate_up(const at::Tensor& gate, const at::Tensor& up, const at::Tensor& x) {
  constexpr int batch = 8192, col = 2048, rows = 6656, output_stride = 2 * rows;
  constexpr int packed_width = col / QK8_0 * sizeof(block_q8_0);
  static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36);
  static_assert(col % 512 == 0 && rows % MMQ_Y_Q8_0 == 0 && batch % MMQ_X_Q8_0 == 0);
  static_assert(batch * col <= INT_MAX && batch * output_stride <= INT_MAX);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() &&
                  x.dim() == 2 && x.size(0) == batch && x.size(1) == col &&
                  reinterpret_cast<std::uintptr_t>(x.data_ptr()) % alignof(at::BFloat16) == 0,
              "W4 gate/up requires contiguous CUDA BF16 [8192,2048] input");
  for (const at::Tensor* weight : {&gate, &up}) {
    TORCH_CHECK(weight->device() == x.device() && weight->scalar_type() == at::kByte &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == rows && weight->size(1) == packed_width &&
                    reinterpret_cast<std::uintptr_t>(weight->data_ptr()) % alignof(block_q8_0) == 0,
                "W4 gate/up requires two aligned packed Q8_0 [6656,2176] matrices on the input device");
  }
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty({batch, output_stride}, x.options());
  auto quant_x = at::empty({batch, col / 32 * 9}, x.options().dtype(at::kInt));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  quantize_row_q8_1_cuda<at::BFloat16>(
      x.data_ptr<at::BFloat16>(), quant_x.data_ptr(), col, batch, stream);

  const dim3 grid(rows / MMQ_Y_Q8_0, batch / MMQ_X_Q8_0, 1);
  const dim3 block(WARP_SIZE_GGUF, NWARPS_Q8_0, 1);
  auto* out = y.data_ptr<at::BFloat16>();
  // Each launch retains the ordinary 6656-row logical bound and owns one half.
  mul_mat_q8_0_research_x64_y64_w4<at::BFloat16, false, output_stride>
      <<<grid, block, 0, stream>>>(gate.data_ptr(), quant_x.data_ptr(), out,
                                   col, rows, batch, col, rows);
  mul_mat_q8_0_research_x64_y64_w4<at::BFloat16, false, output_stride>
      <<<grid, block, 0, stream>>>(up.data_ptr(), quant_x.data_ptr(), out + rows,
                                   col, rows, batch, col, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

namespace {

constexpr int kProbeBatch = 8192, kDownInput = 6656, kDownOutput = 2048;
static_assert(kDownInput % 512 == 0 && kDownInput % CUDA_QUANTIZE_BLOCK_SIZE == 0);
static_assert(CUDA_QUANTIZE_BLOCK_SIZE == CUDA_DEQUANTIZE_BLOCK_SIZE);
static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36);

void require_probe_activation(const at::Tensor& input, int width) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.is_contiguous() &&
                  input.dim() == 2 && input.size(0) == kProbeBatch && input.size(1) == width &&
                  reinterpret_cast<std::uintptr_t>(input.data_ptr()) % alignof(at::BFloat16) == 0,
              "W4 GELU probe requires exact aligned CUDA BF16 geometry");
}

static __device__ __forceinline__ float gelu_tanh_product_bf16(float gate, float up) {
  float product;
  // Order and FTZ flags follow the captured SM121a BF16 JIT activation SASS.
  // Keep fast math local to this activation; quantization uses the W4 flags.
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

at::Tensor probe_quant_buffer(const at::Tensor& input) {
  return at::empty({kProbeBatch, kDownInput / QK8_1 * int(sizeof(block_q8_1) / sizeof(int32_t))},
                   input.options().dtype(at::kInt));
}

at::Tensor launch_probe_gelu_quant(const at::Tensor& gate_up, cudaStream_t stream) {
  auto quant = probe_quant_buffer(gate_up);
  const dim3 grid(kDownInput / CUDA_QUANTIZE_BLOCK_SIZE, kProbeBatch, 1);
  const dim3 block(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
  quantize_gelu_tanh_q8_1<<<grid, block, 0, stream>>>(
      gate_up.data_ptr<at::BFloat16>(), static_cast<block_q8_1*>(quant.data_ptr()));
  return quant;
}

}  // namespace

at::Tensor quant_probe(const at::Tensor& x) {
  require_probe_activation(x, kDownInput);
  const c10::cuda::CUDAGuard guard(x.device());
  auto quant = probe_quant_buffer(x);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  quantize_row_q8_1_cuda<at::BFloat16>(
      x.data_ptr<at::BFloat16>(), quant.data_ptr(), kDownInput, kProbeBatch, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return quant;
}

at::Tensor gelu_quant_probe(const at::Tensor& gate_up) {
  require_probe_activation(gate_up, 2 * kDownInput);
  const c10::cuda::CUDAGuard guard(gate_up.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(gate_up.get_device()).stream();
  auto quant = launch_probe_gelu_quant(gate_up, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return quant;
}

at::Tensor mmq_gelu_down_probe(const at::Tensor& weight, const at::Tensor& gate_up) {
  require_probe_activation(gate_up, 2 * kDownInput);
  constexpr int packed_width = kDownInput / QK8_0 * sizeof(block_q8_0);
  TORCH_CHECK(weight.device() == gate_up.device() && weight.scalar_type() == at::kByte &&
                  weight.is_contiguous() && weight.dim() == 2 &&
                  weight.size(0) == kDownOutput && weight.size(1) == packed_width &&
                  reinterpret_cast<std::uintptr_t>(weight.data_ptr()) % alignof(block_q8_0) == 0,
              "W4 GELU down probe requires aligned Q8_0 [2048,7072] on the input device");
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

}  // namespace tierengine_q8_mmq_x64_y64_w4_v1

TORCH_LIBRARY(tierengine_q8_mmq_x64_y64_w4_v1, m) {
  m.def("mmq(Tensor weight, Tensor x) -> Tensor");
  m.def("mmq_gate_up(Tensor gate, Tensor up, Tensor x) -> Tensor");
  m.def("quant_probe(Tensor x) -> Tensor");
  m.def("gelu_quant_probe(Tensor gate_up) -> Tensor");
  m.def("mmq_gelu_down_probe(Tensor weight, Tensor gate_up) -> Tensor");
}
TORCH_LIBRARY_IMPL(tierengine_q8_mmq_x64_y64_w4_v1, CUDA, m) {
  m.impl("mmq", &tierengine_q8_mmq_x64_y64_w4_v1::run);
  m.impl("mmq_gate_up", &tierengine_q8_mmq_x64_y64_w4_v1::run_gate_up);
  m.impl("quant_probe", &tierengine_q8_mmq_x64_y64_w4_v1::quant_probe);
  m.impl("gelu_quant_probe", &tierengine_q8_mmq_x64_y64_w4_v1::gelu_quant_probe);
  m.impl("mmq_gelu_down_probe", &tierengine_q8_mmq_x64_y64_w4_v1::mmq_gelu_down_probe);
}
