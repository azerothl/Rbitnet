/* SPDX-License-Identifier: MIT
 * Optional native ABI for Rbitnet quantized matvec (#22 Gate E).
 *
 * Host entrypoints take host pointers for W/x/y.
 * *_matvec_device entrypoints take device-resident W and host x/y (upload x, download y).
 *
 * Return 0 on success, non-zero on failure.
 */
#ifndef RBITNET_CUDA_QUANT_H
#define RBITNET_CUDA_QUANT_H

#include <stddef.h>

#ifdef _WIN32
#define RBITNET_CUDA_API __declspec(dllexport)
#else
#define RBITNET_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

RBITNET_CUDA_API int rbitnet_cuda_q4_0_matvec(
    const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_q8_0_matvec(
    const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_q4_k_matvec(
    const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_q6_k_matvec(
    const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);

RBITNET_CUDA_API int rbitnet_cuda_q4_0_matvec_device(
    const void *d_w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_q8_0_matvec_device(
    const void *d_w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_q4_k_matvec_device(
    const void *d_w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_q6_k_matvec_device(
    const void *d_w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);

#ifdef __cplusplus
}
#endif

#endif /* RBITNET_CUDA_QUANT_H */
