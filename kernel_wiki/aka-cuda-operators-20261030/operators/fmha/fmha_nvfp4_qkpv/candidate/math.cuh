// Device math helpers derived from NVIDIA Cake (Apache-2.0).
#pragma once

#include <cuda_fp16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

// BF16 values / positive E4M3 scale, immediately rounded to FP4. FP4 midpoint
// products are exactly representable in BF16. Rounding the reciprocal product
// to FP16 restores these ties without moving a neighboring BF16 value across
// them. Evidence exhausts all finite BF16/E4M3 pairs, including signed zeros.
// This preserves FP4 codes; it is not a general FP32 division implementation.
__device__ __forceinline__ unsigned pack_nvfp4_8(const float* x, float inverse) {
    unsigned half_pairs[4];
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        float2 normalized = make_float2(x[2*pair], x[2*pair+1]);
        mul_f32x2_inplace(&normalized, make_float2(inverse, inverse));
        const __half2 h = __float22half2_rn(normalized);
        half_pairs[pair] = *reinterpret_cast<const unsigned*>(&h);
    }
    unsigned packed;
    asm("{ .reg .b8 a,b,c,d;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 a,%1;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 b,%2;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 c,%3;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 d,%4;\n"
        "mov.b32 %0,{a,b,c,d}; }"
        : "=r"(packed) : "r"(half_pairs[0]), "r"(half_pairs[1]),
          "r"(half_pairs[2]), "r"(half_pairs[3]));
    return packed;
}
