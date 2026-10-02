// SPDX-License-Identifier: MIT
// Native CUDA quantized matvec for Rbitnet (#22 Gate E).
// Layouts match crates/bitnet-core/src/ggml/dequant.rs (llama.cpp-compatible).

#include "rbitnet_cuda_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace {

constexpr int QK4_0 = 32;
constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int Q4_0_BLOCK = 18;
constexpr int Q8_0_BLOCK = 34;
constexpr int Q4_K_BLOCK = 144;
constexpr int Q6_K_BLOCK = 210;

__device__ __forceinline__ float fp16_bits_to_f32(uint16_t bits) {
    __half h;
    *reinterpret_cast<uint16_t *>(&h) = bits;
    return __half2float(h);
}

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

__device__ float dot_row_q4_0(const uint8_t *row, const float *x, size_t x_len) {
    if (x_len % QK4_0 != 0) {
        return 0.0f;
    }
    const size_t nb = x_len / QK4_0;
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t *blk = row + b * Q4_0_BLOCK;
        const float d = fp16_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        const float *xb = x + b * QK4_0;
        for (int j = 0; j < 16; ++j) {
            const uint8_t q = blk[2 + j];
            const float x0 = ((q & 0x0f) - 8) * d;
            const float x1 = ((q >> 4) - 8) * d;
            acc = fmaf(x0, xb[j], acc);
            acc = fmaf(x1, xb[j + 16], acc);
        }
    }
    return acc;
}

__device__ float dot_row_q8_0(const uint8_t *row, const float *x, size_t x_len) {
    if (x_len % QK8_0 != 0) {
        return 0.0f;
    }
    const size_t nb = x_len / QK8_0;
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t *blk = row + b * Q8_0_BLOCK;
        const float d = fp16_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        const float *xb = x + b * QK8_0;
        for (int j = 0; j < QK8_0; ++j) {
            const float q = (float)(int8_t)blk[2 + j];
            acc = fmaf(q * d, xb[j], acc);
        }
    }
    return acc;
}

__device__ float dot_row_q4_k(const uint8_t *row, const float *x, size_t x_len) {
    if (x_len % QK_K != 0) {
        return 0.0f;
    }
    const size_t nb = x_len / QK_K;
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t *blk = row + b * Q4_K_BLOCK;
        const float d = fp16_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        const float minv = fp16_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
        const uint8_t *scales = blk + 4;
        const uint8_t *qs = blk + 16;
        const float *xb = x + b * QK_K;
        int is = 0;
        int qo = 0;
        int y_off = 0;
        for (int g = 0; g < 4; ++g) {
            uint8_t sc, m, sc2, m2;
            get_scale_min_k4(is, scales, &sc, &m);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            const float d1 = d * (float)sc;
            const float m1 = minv * (float)m;
            const float d2 = d * (float)sc2;
            const float m2v = minv * (float)m2;
            for (int l = 0; l < 32; ++l) {
                const float w0 = d1 * (float)(qs[qo + l] & 0x0F) - m1;
                const float w1 = d2 * (float)(qs[qo + l] >> 4) - m2v;
                acc = fmaf(w0, xb[y_off + l], acc);
                acc = fmaf(w1, xb[y_off + 32 + l], acc);
            }
            qo += 32;
            y_off += 64;
            is += 2;
        }
    }
    return acc;
}

__device__ float dot_row_q6_k(const uint8_t *row, const float *x, size_t x_len) {
    if (x_len % QK_K != 0) {
        return 0.0f;
    }
    const size_t nb = x_len / QK_K;
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t *blk = row + b * Q6_K_BLOCK;
        const float d = fp16_bits_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t *sc = reinterpret_cast<const int8_t *>(blk + 192);
        const float *xb = x + b * QK_K;
        int ql_o = 0;
        int qh_o = 0;
        int sc_o = 0;
        int yp = 0;
        for (int pass = 0; pass < 2; ++pass) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 =
                    ((ql[ql_o + l] & 0xF) | (((qh[qh_o + l] >> 0) & 3) << 4)) - 32;
                const int q2 =
                    ((ql[ql_o + l + 32] & 0xF) | (((qh[qh_o + l] >> 2) & 3) << 4)) - 32;
                const int q3 =
                    ((ql[ql_o + l] >> 4) | (((qh[qh_o + l] >> 4) & 3) << 4)) - 32;
                const int q4 =
                    ((ql[ql_o + l + 32] >> 4) | (((qh[qh_o + l] >> 6) & 3) << 4)) - 32;
                acc = fmaf(d * (float)sc[sc_o + is + 0] * (float)q1, xb[yp + l], acc);
                acc = fmaf(d * (float)sc[sc_o + is + 2] * (float)q2, xb[yp + l + 32], acc);
                acc = fmaf(d * (float)sc[sc_o + is + 4] * (float)q3, xb[yp + l + 64], acc);
                acc = fmaf(d * (float)sc[sc_o + is + 6] * (float)q4, xb[yp + l + 96], acc);
            }
            yp += 128;
            ql_o += 64;
            qh_o += 32;
            sc_o += 8;
        }
    }
    return acc;
}

enum class QuantKind { Q4_0, Q8_0, Q4_K, Q6_K };

__global__ void quant_matvec_kernel(
    QuantKind kind,
    const uint8_t *d_w,
    size_t row_bytes,
    const float *d_x,
    size_t x_len,
    size_t ne1,
    float *d_y) {
    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ne1) {
        return;
    }
    const uint8_t *row_ptr = d_w + row * row_bytes;
    float acc = 0.0f;
    switch (kind) {
    case QuantKind::Q4_0:
        acc = dot_row_q4_0(row_ptr, d_x, x_len);
        break;
    case QuantKind::Q8_0:
        acc = dot_row_q8_0(row_ptr, d_x, x_len);
        break;
    case QuantKind::Q4_K:
        acc = dot_row_q4_k(row_ptr, d_x, x_len);
        break;
    case QuantKind::Q6_K:
        acc = dot_row_q6_k(row_ptr, d_x, x_len);
        break;
    }
    d_y[row] = acc;
}

struct Scratch {
    float *d_x = nullptr;
    float *d_y = nullptr;
    uint8_t *d_w = nullptr;
    size_t cap_x = 0;
    size_t cap_y = 0;
    size_t cap_w = 0;
};

Scratch g_scratch;

bool ensure(size_t need, void **ptr, size_t *cap) {
    if (*cap >= need && *ptr != nullptr) {
        return true;
    }
    if (*ptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
        *cap = 0;
    }
    if (cudaMalloc(ptr, need) != cudaSuccess) {
        *ptr = nullptr;
        *cap = 0;
        return false;
    }
    *cap = need;
    return true;
}

int launch_device_w(
    QuantKind kind,
    const void *d_w,
    size_t row_bytes,
    const float *x,
    size_t x_len,
    size_t ne1,
    float *y) {
    if (!d_w || !x || !y || x_len == 0 || ne1 == 0 || row_bytes == 0) {
        return 1;
    }
    if (!ensure(x_len * sizeof(float), reinterpret_cast<void **>(&g_scratch.d_x), &g_scratch.cap_x)) {
        return 2;
    }
    if (!ensure(ne1 * sizeof(float), reinterpret_cast<void **>(&g_scratch.d_y), &g_scratch.cap_y)) {
        return 3;
    }
    if (cudaMemcpy(g_scratch.d_x, x, x_len * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return 4;
    }
    const int threads = 128;
    const int blocks = (int)((ne1 + threads - 1) / threads);
    quant_matvec_kernel<<<blocks, threads>>>(
        kind,
        reinterpret_cast<const uint8_t *>(d_w),
        row_bytes,
        g_scratch.d_x,
        x_len,
        ne1,
        g_scratch.d_y);
    if (cudaGetLastError() != cudaSuccess) {
        return 5;
    }
    if (cudaMemcpy(y, g_scratch.d_y, ne1 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 6;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 7;
    }
    return 0;
}

int launch_host_w(
    QuantKind kind,
    const void *w,
    size_t row_bytes,
    const float *x,
    size_t x_len,
    size_t ne1,
    float *y) {
    if (!w || !x || !y || x_len == 0 || ne1 == 0 || row_bytes == 0) {
        return 1;
    }
    const size_t w_bytes = row_bytes * ne1;
    if (!ensure(w_bytes, reinterpret_cast<void **>(&g_scratch.d_w), &g_scratch.cap_w)) {
        return 8;
    }
    if (cudaMemcpy(g_scratch.d_w, w, w_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return 9;
    }
    return launch_device_w(kind, g_scratch.d_w, row_bytes, x, x_len, ne1, y);
}

} // namespace

extern "C" {

int rbitnet_cuda_q4_0_matvec(
    const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_host_w(QuantKind::Q4_0, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_q8_0_matvec(
    const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_host_w(QuantKind::Q8_0, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_q4_k_matvec(
    const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_host_w(QuantKind::Q4_K, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_q6_k_matvec(
    const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_host_w(QuantKind::Q6_K, w, row_bytes, x, x_len, ne1, y);
}

int rbitnet_cuda_q4_0_matvec_device(
    const void *d_w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_device_w(QuantKind::Q4_0, d_w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_q8_0_matvec_device(
    const void *d_w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_device_w(QuantKind::Q8_0, d_w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_q4_k_matvec_device(
    const void *d_w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_device_w(QuantKind::Q4_K, d_w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_q6_k_matvec_device(
    const void *d_w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_device_w(QuantKind::Q6_K, d_w, row_bytes, x, x_len, ne1, y);
}

} // extern "C"
