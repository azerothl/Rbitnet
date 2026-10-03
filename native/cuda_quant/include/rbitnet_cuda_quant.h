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

RBITNET_CUDA_API int rbitnet_cuda_f32_matvec(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_f32_matvec_device(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_q5_0_matvec(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_q5_0_matvec_device(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_q5_k_matvec(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_q5_k_matvec_device(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_mxfp4_matvec(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);
RBITNET_CUDA_API int rbitnet_cuda_mxfp4_matvec_device(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y);

RBITNET_CUDA_API int rbitnet_cuda_quant_matvec_batch_device(unsigned ty, const void *w, size_t row_bytes,
    const float *x, size_t cols, size_t rows, size_t batches, float *y);

RBITNET_CUDA_API void *rbitnet_cuda_attention_create(size_t capacity,size_t kv_heads,size_t key_dim,size_t value_dim,size_t heads);
RBITNET_CUDA_API void rbitnet_cuda_attention_destroy(void *context);
RBITNET_CUDA_API void rbitnet_cuda_attention_reset(void *context);
RBITNET_CUDA_API int rbitnet_cuda_attention_step(void *context,const float *q,const float *k,const float *v,size_t pos,size_t first,float scale,const float *sinks,float *y);

#ifdef __cplusplus
}
#endif

#endif /* RBITNET_CUDA_QUANT_H */
