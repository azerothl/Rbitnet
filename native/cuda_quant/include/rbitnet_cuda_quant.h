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

/* Fully resident dense Llama token graph. Weight pointers are device pointers;
 * normalization/frequency/input/output pointers are host pointers. The caller
 * retains the weights until the context is destroyed. */
typedef struct {
    const void *weights;
    size_t row_bytes;
    unsigned type, cols, rows;
} RbitnetLlamaMatrix;
typedef struct {
    RbitnetLlamaMatrix q, k, v, out, gate, up, down;
    const float *attn_norm, *ffn_norm;
} RbitnetLlamaLayer;
typedef struct {
    unsigned embd, ffn, vocab, layers, heads, kv_heads, head_dim, rotary, capacity, window, graphs;
    float epsilon;
} RbitnetLlamaConfig;
RBITNET_CUDA_API void *rbitnet_cuda_llama_create(const RbitnetLlamaConfig*, const RbitnetLlamaLayer*, const RbitnetLlamaMatrix*, const float *output_norm, const float *rope_frequency);
RBITNET_CUDA_API void rbitnet_cuda_llama_destroy(void*);
/* mode: 0=transform only; 1=download F32 logits; 2=download greedy token only.
 * pos=0 starts a fresh sequence; only sequential positions are accepted. */
RBITNET_CUDA_API int rbitnet_cuda_llama_step(void*, const float *embedding, unsigned pos, unsigned mode, float *logits, unsigned *token);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef struct { unsigned embd, ffn, experts, used, oai; } RbitnetMoeConfig;
RBITNET_CUDA_API void *rbitnet_cuda_moe_create(const RbitnetMoeConfig *cfg,
    const RbitnetLlamaMatrix *gate, const RbitnetLlamaMatrix *up, const RbitnetLlamaMatrix *down,
    const float *gate_bias, const float *up_bias, const float *down_bias);
RBITNET_CUDA_API void rbitnet_cuda_moe_destroy(void *context);
RBITNET_CUDA_API int rbitnet_cuda_moe_step(void *context,const float *input,
    const unsigned *experts,const float *probabilities,float *output);

/* A complete dense Qwen3.5 recurrent block, including both residuals and FFN.
 * Matrices, in order: qkv, z, beta, alpha, ssm_out, ffn_gate, ffn_up, ffn_down.
 * Matrix weights are device pointers retained by the caller. All other pointers
 * are host arrays copied during creation; ssm_norm has head*num_v entries. */
typedef struct {
    unsigned embd, ffn, head, num_k, num_v, conv, graphs;
    float epsilon;
} RbitnetQwenRecurrentConfig;
RBITNET_CUDA_API void *rbitnet_cuda_qwen_recurrent_create(const RbitnetQwenRecurrentConfig*,
    const RbitnetLlamaMatrix *matrices,const float *attn_norm,const float *ffn_norm,
    const float *conv_weights,const float *dt_bias,const float *a,const float *ssm_norm);
RBITNET_CUDA_API void rbitnet_cuda_qwen_recurrent_destroy(void *context);
/* pos=0 clears convolution history and recurrent state. Other positions must
 * be sequential. Failures are reported, never silently restarted on CPU. */
RBITNET_CUDA_API int rbitnet_cuda_qwen_recurrent_step(void *context,const float *input,
    unsigned pos,float *output);

/* Shared resident output RMSNorm, quantized head and optional greedy reduction.
 * mode=0 downloads logits, mode=1 downloads only the chosen token. */
RBITNET_CUDA_API void *rbitnet_cuda_head_create(const RbitnetLlamaMatrix*,const float *norm,float epsilon);
RBITNET_CUDA_API void rbitnet_cuda_head_destroy(void *context);
RBITNET_CUDA_API int rbitnet_cuda_head_step(void *context,const float *input,unsigned mode,float *logits,unsigned *token);
#ifdef __cplusplus
}
#endif

#endif /* RBITNET_CUDA_QUANT_H */
