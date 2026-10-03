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

// Shared-weight SIMT quantized GEMM. All three pointers are device addresses;
// input is [tokens, columns], output [tokens, rows]. Synchronizes on return.
RBITNET_CUDA_API int rbitnet_cuda_quant_gemm_device(unsigned type, const void *weights,
    size_t row_bytes, const float *input, unsigned columns, unsigned rows, unsigned tokens, float *output);

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
// Optional SIMT quantized GEMM prefill, 1..128 tokens, causal attention.
RBITNET_CUDA_API int rbitnet_cuda_llama_prefill(void *context, const float *embeddings,
    unsigned position, unsigned count, unsigned mode, float *logits, unsigned *token);
/* Optional per-position verification, 1..16 tokens. Mode 1 downloads
 * count*vocab logits; mode 2 downloads count argmax IDs. Reject mode 0.
 * truncate only adjusts valid dense KV length after a synchronized verification. */
RBITNET_CUDA_API int rbitnet_cuda_llama_verify(void*,const float*,unsigned,unsigned,unsigned,float*,unsigned*);
RBITNET_CUDA_API int rbitnet_cuda_llama_truncate(void*,unsigned length);
/* mode: 0=transform only; 1=download F32 logits; 2=download greedy token only.
 * pos=0 starts a fresh sequence; only sequential positions are accepted. */
RBITNET_CUDA_API int rbitnet_cuda_llama_step(void*, const float *embedding, unsigned pos, unsigned mode, float *logits, unsigned *token);
/* Immutable device snapshots contain the first length tokens, not unused capacity.
 * Restore may truncate an attention snapshot. The caller owns/destroys the snapshot
 * independently of its source context and keeps it scoped to the same model. */
RBITNET_CUDA_API void *rbitnet_cuda_llama_snapshot(void*, unsigned length);
RBITNET_CUDA_API void rbitnet_cuda_llama_snapshot_destroy(void*);
RBITNET_CUDA_API int rbitnet_cuda_llama_restore(void*, const void *snapshot, unsigned length);

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
// Optional expert-cache API. Descriptors retain full-bank shapes, but weight
// addresses are supplied per selected expert to each synchronous step.
RBITNET_CUDA_API void *rbitnet_cuda_moe_dynamic_create(const RbitnetMoeConfig *cfg,
    const RbitnetLlamaMatrix *gate, const RbitnetLlamaMatrix *up, const RbitnetLlamaMatrix *down,
    const float *gate_bias, const float *up_bias, const float *down_bias);
RBITNET_CUDA_API int rbitnet_cuda_moe_dynamic_step(void *context, const float *input,
    const unsigned *experts, const float *probabilities, float *output, const void *const *selected);
RBITNET_CUDA_API int rbitnet_cuda_moe_step(void *context,const float *input,
    const unsigned *experts,const float *probabilities,float *output);

/* Optional fully resident GPT-OSS token graph. Expert contexts and matrix
 * weights are borrowed, exclusive to this runtime, and must outlive it. Host
 * norm/bias/frequency arrays are copied during creation. Fixed expert banks only.
 * Top-k uses Rust total_cmp order and the lower expert ID wins ties. */
typedef struct {
    unsigned embd,vocab,layers,heads,kv_heads,head_dim,rotary,capacity,window,experts,used,graphs,split,ordered;
    float epsilon,rope_magnitude,weight_scale;
} RbitnetGptConfig;
typedef struct {
    RbitnetLlamaMatrix q,k,v,out,router;
    const float *attn_norm,*ffn_norm,*q_bias,*k_bias,*v_bias,*out_bias,*router_bias,*selection_bias,*sinks;
    void *moe;
} RbitnetGptLayer;
RBITNET_CUDA_API void *rbitnet_cuda_gpt_full_create(const RbitnetGptConfig*,const RbitnetGptLayer*,
    const RbitnetLlamaMatrix *head,const float *output_norm,const float *frequency);
RBITNET_CUDA_API void rbitnet_cuda_gpt_full_destroy(void*);
RBITNET_CUDA_API int rbitnet_cuda_gpt_full_step(void*,const float*,unsigned position,unsigned mode,float*,unsigned*);
/* Diagnostic host-array APIs share the production enqueue/router kernels. */
RBITNET_CUDA_API int rbitnet_cuda_gpt_full_hidden_check(void*,const float*,unsigned position,float*);
/* Output per layer: embd hidden values, experts raw router logits, used IDs as F32. */
RBITNET_CUDA_API int rbitnet_cuda_gpt_full_layers_check(void*,const float*,unsigned position,float*);
RBITNET_CUDA_API int rbitnet_cuda_gpt_router_check(const float*,const float*,unsigned,unsigned,float,unsigned*,float*);

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
RBITNET_CUDA_API void *rbitnet_cuda_qwen_recurrent_snapshot(void *context);
RBITNET_CUDA_API void rbitnet_cuda_qwen_recurrent_snapshot_destroy(void *snapshot);
RBITNET_CUDA_API int rbitnet_cuda_qwen_recurrent_restore(void *context, const void *snapshot, unsigned length);
/* pos=0 clears convolution history and recurrent state. Other positions must
 * be sequential. Failures are reported, never silently restarted on CPU. */
RBITNET_CUDA_API int rbitnet_cuda_qwen_recurrent_step(void *context,const float *input,
    unsigned pos,float *output);

/* Optional dense Qwen full-attention block and whole-token pipeline. Matrix
 * order: q, k, v, out, gate, up, down. Norms/frequency are host arrays.
 * A pipeline borrows its ordered layer contexts (0=recurrent, 1=attention).
 * The caller retains contexts and weights until after pipeline destruction. */
typedef struct {
    unsigned embd, ffn, heads, kv_heads, head_dim, rotary, capacity, gated, graphs;
    float epsilon, scale;
} RbitnetQwenAttentionConfig;
typedef struct { unsigned kind; void *context; } RbitnetQwenFullLayer;
RBITNET_CUDA_API void *rbitnet_cuda_qwen_full_attention_create(const RbitnetQwenAttentionConfig*,
    const RbitnetLlamaMatrix*,const float *attn_norm,const float *ffn_norm,
    const float *q_norm,const float *k_norm,const float *frequency);
RBITNET_CUDA_API void rbitnet_cuda_qwen_full_attention_destroy(void*);
RBITNET_CUDA_API int rbitnet_cuda_qwen_full_attention_step(void*,const float*,unsigned,float*);
RBITNET_CUDA_API void *rbitnet_cuda_qwen_full_attention_snapshot(void*);
RBITNET_CUDA_API void rbitnet_cuda_qwen_full_attention_snapshot_destroy(void*);
RBITNET_CUDA_API int rbitnet_cuda_qwen_full_attention_restore(void*,const void*,unsigned);
RBITNET_CUDA_API void *rbitnet_cuda_qwen_full_create(unsigned embd,unsigned vocab,unsigned capacity,
    unsigned layers,unsigned graphs,const RbitnetQwenFullLayer*,const RbitnetLlamaMatrix *head,
    const float *norm,float epsilon);
RBITNET_CUDA_API void rbitnet_cuda_qwen_full_destroy(void*);
/* mode=0 no output; 1 logits; 2 argmax. One synchronization per token. */
RBITNET_CUDA_API int rbitnet_cuda_qwen_full_step(void*,const float*,unsigned,unsigned,float*,unsigned*);
/* Optional causal dense Qwen block prefill. Configure once before inference.
 * A failed workspace allocation leaves the serial pipeline usable. The last
 * token alone produces logits/argmax; recurrent state is advanced exactly count. */
RBITNET_CUDA_API int rbitnet_cuda_qwen_configure_prefill(void*,unsigned enabled,unsigned tensor);
RBITNET_CUDA_API unsigned rbitnet_cuda_qwen_prefill_capacity(void*);
RBITNET_CUDA_API unsigned rbitnet_cuda_qwen_tensor_gemm_calls(void*);
RBITNET_CUDA_API int rbitnet_cuda_qwen_full_prefill(void*,const float*,unsigned position,unsigned count,unsigned mode,float*,unsigned*);
/* Diagnostic host-array layer block runner, sharing production enqueue kernels.
 * Temporary workspace allocation is for numerical oracles, not a hot path. */
RBITNET_CUDA_API int rbitnet_cuda_qwen_recurrent_prefill_check(void*,const float*,unsigned,unsigned,float*);
RBITNET_CUDA_API int rbitnet_cuda_qwen_attention_prefill_check(void*,const float*,unsigned,unsigned,float*);
RBITNET_CUDA_API int rbitnet_cuda_qwen_full_restored(void*,unsigned length);
/* Number of layers actually configured with the optional split-KV kernels. */
RBITNET_CUDA_API unsigned rbitnet_cuda_llama_split_attention_layers(void*);
/* Number of Tensor Core GEMM launches in the last successful block operation. */
RBITNET_CUDA_API unsigned rbitnet_cuda_llama_tensor_gemm_calls(void*);
/* Configure before first block/capture. Explicit ABI avoids a DLL CRT getenv
 * snapshot diverging from Rust SetEnvironmentVariable on Windows. */
RBITNET_CUDA_API int rbitnet_cuda_llama_configure_tensor_prefill(void*,unsigned enabled);
RBITNET_CUDA_API unsigned rbitnet_cuda_qwen_split_attention_layers(void*);

/* Diagnostic split-KV oracle runner, using host arrays. The same captured
 * kernels replay each device position; outputs are [step, token, head, dim].
 * Hot model paths use resident scratch and never call this helper. */
RBITNET_CUDA_API int rbitnet_cuda_split_attention_check(const float *k,const float *v,const float *q,
    unsigned capacity,unsigned kv_heads,unsigned heads,unsigned dim,unsigned window,float scale,
    unsigned count,const unsigned *positions,unsigned steps,unsigned graphs,float *out);

/* Diagnostic host-array GGUF GEMM, with same hot kernels and CUDA-event timing.
 * tf32x3=1 requires compute capability >=8.0; unsupported returns 2. */
RBITNET_CUDA_API int rbitnet_cuda_quant_gemm_check(unsigned type,const void *weights,size_t row_bytes,
    const float *input,unsigned columns,unsigned rows,unsigned tokens,unsigned tf32x3,
    unsigned repeats,float *output,float *elapsed_ms);

/* Shared resident output RMSNorm, quantized head and optional greedy reduction.
 * mode=0 downloads logits, mode=1 downloads only the chosen token. */
RBITNET_CUDA_API void *rbitnet_cuda_head_create(const RbitnetLlamaMatrix*,const float *norm,float epsilon);
RBITNET_CUDA_API void rbitnet_cuda_head_destroy(void *context);
RBITNET_CUDA_API int rbitnet_cuda_head_step(void *context,const float *input,unsigned mode,float *logits,unsigned *token);
#ifdef __cplusplus
}
#endif

#endif /* RBITNET_CUDA_QUANT_H */
