#pragma once
#if defined(USE_ROCM) || defined(USE_MUSA)
#error "CUDA-only research IMMA variant"
#endif
// Research core: replaces the dp4a vec_dot inner loop of mul_mat_q with
// int8 tensor-core mma.sync.m16n8k16 over one-time repacked Q8_0 panels.
//
// Panel layout (produced byte-exactly by gguf_q8_imma_m16n8k16.repack_q8_0_panel):
//   panel_s8 : int8 [rows/16, k/16, 16, 16]
//     tile (r16, k16) holds weight bytes w[r16*16+i][k16*16+j] row-major with
//     16-byte rows -> exactly two ldmatrix m8n8.b16 matrices per k16 step.
//   panel_d : fp16 [rows, k/32] per-32-K-block Q8_0 scales in row-major order.
//
// PTX fragment mappings used below (PTX ISA, 8-bit m16n8k16, lane l):
//   A {a0,a1}: a0 = A[l/4][(l%4)*4 .. +3], a1 = A[l/4+8][(l%4)*4 .. +3]
//   B {b0}   : b0 = B col l/4, k (l%4)*4 .. +3  (k-contiguous per n)
//   C {c0..c3}: c0=(row l/4, col (l%4)*2), c1=(+1), c2=(row l/4+8, col (l%4)*2), c3=(+1)
// ldmatrix.x2 on tile rows (lanes 0..15) yields a0/a1 directly; ldmatrix.x4 over
// four n-octets (lane l -> octet l/8 row l%8) yields the four b0 fragments.
//
// Numerics: each 32-K quant block accumulates exact int32 partials (the same 32
// s8*s8 products the dp4a reference sums, |sum| <= 32*127*127 << INT32_MAX, so
// the mma and dp4a int32 partials are bit-identical), then folds into f32 as
// (d_w*d_a)*float(partial) exactly like vec_dot_q8_0_q8_1_impl's
// d8_0 * d8_1 * sumi. Only f32 summation order across chunks differs.

// Requires cuda_fp16.h and ggml-common.h (WARP_SIZE_GGUF, block_q8_1, __half)
// to be included beforehand, in the same order the dp4a reference uses.
#include "../native_q8_mmq_x8/vendor/ggml-common.h"

namespace q8_imma_m16n8k16 {

constexpr int kImmaK = 16;            // mma.sync m16n8k16 K step
constexpr int kBlockRows = 32;        // weight rows per block (2 warps x 16)
constexpr int kBlockCols = 64;        // batch columns per block (2 warps x 32)
constexpr int kWarpsY = 2;
constexpr int kWarpsX = 2;
constexpr int kNumWarps = kWarpsY * kWarpsX;
constexpr int kOctets = 4;            // 32 batch columns per warp = 4 n-octets

__device__ __forceinline__ void ldmatrix_x2_b16(int& r0, int& r1, const void* addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(static_cast<unsigned>(__cvta_generic_to_shared(addr))));
}

__device__ __forceinline__ void ldmatrix_x4_b16(
    int& r0, int& r1, int& r2, int& r3, const void* addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(static_cast<unsigned>(__cvta_generic_to_shared(addr))));
}

__device__ __forceinline__ void mma_m16n8k16_s8(
    int& d0, int& d1, int& d2, int& d3, const int a0, const int a1, const int b0,
    const int c0, const int c1, const int c2, const int c3) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
}

// dst[n * stride + r] = sum over k of w[r][k] * x[n][k], with w from the
// repacked panels and x from the standard quantize_row_q8_1 buffer.
// Admission (host-checked before launch): rows % 32 == 0, batch % 64 == 0,
// k % 32 == 0; every tile is therefore full and no bounds checks are needed.
template <typename scalar_t, int output_stride = 0>
static __global__ __launch_bounds__(WARP_SIZE_GGUF * kNumWarps) void mul_mat_q8_0_imma_q8_1(
    const int8_t* __restrict__ panel_s8,  // [rows/16, k/16, 16, 16]
    const __half* __restrict__ panel_d,   // [rows, k/32]
    const block_q8_1* __restrict__ vy,    // [batch, k/32]
    scalar_t* __restrict__ dst,
    const int k,
    const int rows,
    const int batch,
    const int nrows_dst) {
  static_assert(kBlockRows % 16 == 0 && kBlockCols % 64 == 0, "tile shape");
  const int nblk = k / 32;
  const int k16 = k / kImmaK;
  const int warp_id = threadIdx.x / WARP_SIZE_GGUF;
  const int lane = threadIdx.x % WARP_SIZE_GGUF;
  const int wy = warp_id / kWarpsX;
  const int wx = warp_id % kWarpsX;
  const int row0 = blockIdx.x * kBlockRows + wy * 16;
  const int col0 = blockIdx.y * kBlockCols + wx * 32;
  const int tid = threadIdx.x;

  __shared__ __align__(16) int8_t a_s[kWarpsY][2][16][16];  // 2 k-steps
  __shared__ __align__(16) int8_t b_s[kWarpsX][32][32];     // 32 n x 32 k bytes
  __shared__ float dw_s[kBlockRows];
  __shared__ float da_s[kBlockCols];

  const int* __restrict__ panel32 = reinterpret_cast<const int*>(panel_s8);
  float sum[kOctets][4] = {{0.0f}};

  for (int kb = 0; kb < nblk; ++kb) {
    __syncthreads();
    // Stage A (256 int32): a_s[wy][ks][r][0:16] from panel tiles (kb*2+ks).
    {
      int* a32 = reinterpret_cast<int*>(&a_s[0][0][0][0]);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int idx = tid + i * (WARP_SIZE_GGUF * kNumWarps);
        const int awy = idx >> 7, arem = idx & 127;
        const int aks = arem >> 6, ar = (arem >> 2) & 15, akq = arem & 3;
        const int row = blockIdx.x * kBlockRows + awy * 16 + ar;
        a32[idx] = panel32[((row >> 4) * k16 + (kb << 1) + aks) * 64 + ((row & 15) << 2) + akq];
      }
    }
    // Stage B (512 int32): b_s[wx][n][0:32] from vy[n * nblk + kb].qs.
    {
      int* b32 = reinterpret_cast<int*>(&b_s[0][0][0]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int idx = tid * 4 + i;
        const int n = ((idx >> 8) << 5) + ((idx >> 3) & 31);
        const int kq = idx & 7;
        b32[idx] = reinterpret_cast<const int*>(vy[n * nblk + kb].qs)[kq];
      }
    }
    if (tid < kBlockRows) {
      dw_s[tid] = __half2float(panel_d[(blockIdx.x * kBlockRows + tid) * nblk + kb]);
    }
    if (tid < kBlockCols) {
      da_s[tid] = __low2float(vy[(blockIdx.y * kBlockCols + tid) * nblk + kb].ds);
    }
    __syncthreads();

    int c[kOctets][4];
#pragma unroll
    for (int o = 0; o < kOctets; ++o) {
#pragma unroll
      for (int f = 0; f < 4; ++f) c[o][f] = 0;
    }
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      int a0, a1, b[kOctets];
      ldmatrix_x2_b16(a0, a1, &a_s[wy][ks][lane & 15][0]);
      ldmatrix_x4_b16(b[0], b[1], b[2], b[3], &b_s[wx][lane][ks * kImmaK]);
#pragma unroll
      for (int o = 0; o < kOctets; ++o) {
        mma_m16n8k16_s8(c[o][0], c[o][1], c[o][2], c[o][3], a0, a1, b[o],
                        c[o][0], c[o][1], c[o][2], c[o][3]);
      }
    }
    // Fold exact per-block s32 partials: sum += (d_w*d_a) * float(partial).
#pragma unroll
    for (int o = 0; o < kOctets; ++o) {
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        const float d = dw_s[wy * 16 + (lane >> 2) + (f >> 1) * 8] *
                        da_s[wx * 32 + o * 8 + (lane & 3) * 2 + (f & 1)];
        sum[o][f] += d * static_cast<float>(c[o][f]);
      }
    }
  }

  const int stride = output_stride == 0 ? nrows_dst : output_stride;
#pragma unroll
  for (int o = 0; o < kOctets; ++o) {
#pragma unroll
    for (int f = 0; f < 4; ++f) {
      const int r = row0 + (lane >> 2) + (f >> 1) * 8;
      const int n = col0 + o * 8 + (lane & 3) * 2 + (f & 1);
      dst[n * stride + r] = static_cast<scalar_t>(sum[o][f]);
    }
  }
}

template <typename scalar_t, int output_stride = 0>
static void launch_mul_mat_q8_0_imma(
    const int8_t* panel_s8, const __half* panel_d, const void* quant_x,
    scalar_t* dst, const int k, const int rows, const int batch,
    const int nrows_dst, cudaStream_t stream) {
  const dim3 grid(rows / kBlockRows, batch / kBlockCols, 1);
  const dim3 block(WARP_SIZE_GGUF * kNumWarps, 1, 1);
  mul_mat_q8_0_imma_q8_1<scalar_t, output_stride><<<grid, block, 0, stream>>>(
      panel_s8, panel_d, static_cast<const block_q8_1*>(quant_x), dst, k, rows,
      batch, nrows_dst);
}

}  // namespace q8_imma_m16n8k16
