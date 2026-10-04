// SPDX-License-Identifier: MIT
// Native CUDA quantized matvec for Rbitnet (#22 Gate E).
// Layouts match crates/bitnet-core/src/ggml/dequant.rs (llama.cpp-compatible).

#include "rbitnet_cuda_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>

#include <cstdint>
#include <cstring>
#include "device_memory.cuh"
#include "native_completion.cuh"

namespace {

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

enum class QuantKind { F32, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4 };

// Each warp cooperates on one output row instead of serializing all columns in one thread.
template<QuantKind kind>
__device__ __forceinline__ float weight_at(const uint8_t *row, size_t i) {
    if (kind == QuantKind::F32) return reinterpret_cast<const float *>(row)[i];
    if (kind == QuantKind::Q4_0 || kind == QuantKind::Q5_0 || kind == QuantKind::Q8_0 || kind == QuantKind::MXFP4) {
        int j = i % 32;
        size_t bytes = kind == QuantKind::Q4_0 ? 18 : kind == QuantKind::Q5_0 ? 22 : kind == QuantKind::Q8_0 ? 34 : 17;
        const uint8_t *b = row + (i/32)*bytes;
        if (kind == QuantKind::MXFP4) {
            uint32_t scale = b[0] < 2 ? (0x00200000u << b[0]) : ((uint32_t(b[0])-1) << 23);
            int q = (b[1+j%16] >> (j/16*4)) & 15;
            // Packed E2M1 magnitudes avoid a dynamically indexed thread-local array.
            // nvcc otherwise stores/reloads 64 bytes for every decoded weight.
            int magnitude=(0xC8643210u >> ((q&7)*4)) & 15;
            return __uint_as_float(scale)*float((q&8)?-magnitude:magnitude);
        }
        float d = fp16_bits_to_f32(uint16_t(b[0]) | (uint16_t(b[1]) << 8));
        if (kind == QuantKind::Q8_0) return d*float(int8_t(b[2+j]));
        int offset = kind == QuantKind::Q5_0 ? 6 : 2;
        int q = (b[offset+j%16] >> (j/16*4)) & 15;
        if (kind == QuantKind::Q5_0) {
            uint32_t high = uint32_t(b[2]) | (uint32_t(b[3]) << 8) | (uint32_t(b[4]) << 16) | (uint32_t(b[5]) << 24);
            q |= ((high >> j) & 1) << 4;
        }
        return d*(q - (kind == QuantKind::Q5_0 ? 16 : 8));
    }
    size_t bytes = kind == QuantKind::Q4_K ? 144 : kind == QuantKind::Q5_K ? 176 : 210;
    const uint8_t *b = row + (i/256)*bytes;
    int j = i%256;
    if (kind == QuantKind::Q6_K) {
        int pass = j/128, quarter = (j%128)/32, lane = j%32;
        int ql = b[pass*64 + (quarter%2)*32 + lane];
        int low = (ql >> (quarter/2*4)) & 15;
        int high = (b[128+pass*32+lane] >> (quarter*2)) & 3;
        float d = fp16_bits_to_f32(uint16_t(b[208]) | (uint16_t(b[209]) << 8));
        return d*float(int8_t(b[192+pass*8+quarter*2+lane/16]))*float((low | (high<<4))-32);
    }
    float d = fp16_bits_to_f32(uint16_t(b[0]) | (uint16_t(b[1]) << 8));
    float minv = fp16_bits_to_f32(uint16_t(b[2]) | (uint16_t(b[3]) << 8));
    uint8_t sc, m; get_scale_min_k4(j/32, b+4, &sc, &m);
    int pair = j/64, lane = j%32, half = (j%64)/32;
    int qoff = kind == QuantKind::Q5_K ? 48 : 16;
    int q = (b[qoff+pair*32+lane] >> (half*4)) & 15;
    if (kind == QuantKind::Q5_K) q += ((b[16+lane] >> (pair*2+half)) & 1) << 4;
    // Keep dequantization rounding consistent with the CPU reference.
    return __fsub_rn(__fmul_rn(__fmul_rn(d,float(sc)),float(q)), __fmul_rn(minv,float(m)));
}

template<QuantKind kind>
__device__ __forceinline__ float quant_row_dot(const uint8_t *weights,const float *x,size_t x_len) {
    const int lane=threadIdx.x&31;
    float sum = 0.0f;
    if constexpr(kind==QuantKind::Q4_K || kind==QuantKind::Q5_K || kind==QuantKind::Q6_K) {
        constexpr size_t bytes=kind==QuantKind::Q4_K?144:kind==QuantKind::Q5_K?176:210;
        for(size_t block=0;block<x_len/256;block++) {
            const uint8_t *b=weights+block*bytes;
            float d=fp16_bits_to_f32(*reinterpret_cast<const uint16_t*>(b+(kind==QuantKind::Q6_K?208:0)));
            [[maybe_unused]] float minv=kind==QuantKind::Q6_K?0.0f:fp16_bits_to_f32(*reinterpret_cast<const uint16_t*>(b+2));
            #pragma unroll
            for(int step=0;step<2;step++) {
                const int group=lane/8+step*4,j=(lane%8)*4;
                const float4 input=*reinterpret_cast<const float4*>(x+block*256+group*32+j);
                uint32_t packed; [[maybe_unused]] uint32_t high=0;float scale; [[maybe_unused]] float minimum=0.0f;
                if constexpr(kind==QuantKind::Q6_K) {
                    const int pass=group/4,quarter=group%4;
                    const uint8_t *ql=b+pass*64+(quarter%2)*32+j;
                    const uint8_t *qh=b+128+pass*32+j;
                    // Q6_K rows are aligned to two bytes, not necessarily four.
                    packed=uint32_t(*reinterpret_cast<const uint16_t*>(ql))|(uint32_t(*reinterpret_cast<const uint16_t*>(ql+2))<<16);
                    high=uint32_t(*reinterpret_cast<const uint16_t*>(qh))|(uint32_t(*reinterpret_cast<const uint16_t*>(qh+2))<<16);
                    packed=(packed>>(quarter/2*4))&0x0f0f0f0fu;
                    high=(high>>(quarter*2))&0x03030303u;
                    scale=d*float(int8_t(b[192+pass*8+quarter*2+j/16]));
                } else {
                    uint8_t sc,m;get_scale_min_k4(group,b+4,&sc,&m);
                    packed=*reinterpret_cast<const uint32_t*>(b+(kind==QuantKind::Q5_K?48:16)+(group/2)*32+j);
                    packed=(packed>>(group%2*4))&0x0f0f0f0fu;
                    if constexpr(kind==QuantKind::Q5_K) high=(*reinterpret_cast<const uint32_t*>(b+16+j)>>group)&0x01010101u;
                    scale=d*float(sc);minimum=minv*float(m);
                }
                #pragma unroll
                for(int k=0;k<4;k++) {
                    int q=(packed>>(8*k))&15;
                    if constexpr(kind==QuantKind::Q6_K)q|=((high>>(8*k))&3)<<4;
                    if constexpr(kind==QuantKind::Q5_K)q|=((high>>(8*k))&1)<<4;
                    float value;
                    if constexpr(kind==QuantKind::Q6_K)value=scale*float(q-32);
                    else value=__fsub_rn(__fmul_rn(scale,float(q)),minimum);
                    float xi=k==0?input.x:k==1?input.y:k==2?input.z:input.w;
                    sum=fmaf(value,xi,sum);
                }
            }
        }
    } else if constexpr(kind==QuantKind::Q8_0 || kind==QuantKind::MXFP4) {
        constexpr size_t bytes=kind==QuantKind::Q8_0?34:17;
        constexpr int groups=kind==QuantKind::Q8_0?4:8;
        const int group=kind==QuantKind::Q8_0?lane/8:lane/4;
        const int j=(kind==QuantKind::Q8_0?lane%8:lane%4)*4;
        for(size_t block=0;block<x_len/32;block+=groups) {
            size_t qb=block+group;
            if(qb>=x_len/32)continue;
            const uint8_t *b=weights+qb*bytes;
            float d;uint32_t packed;
            if constexpr(kind==QuantKind::Q8_0) {
                d=fp16_bits_to_f32(*reinterpret_cast<const uint16_t*>(b));
                packed=uint32_t(*reinterpret_cast<const uint16_t*>(b+2+j))|(uint32_t(*reinterpret_cast<const uint16_t*>(b+4+j))<<16);
            } else {
                d=__uint_as_float(b[0]<2?(0x00200000u<<b[0]):((uint32_t(b[0])-1)<<23));
                // MXFP4 has odd-byte rows, so do not assume aligned word loads.
                packed=uint32_t(b[1+j])|(uint32_t(b[2+j])<<8)|(uint32_t(b[3+j])<<16)|(uint32_t(b[4+j])<<24);
            }
            const float4 first=*reinterpret_cast<const float4*>(x+qb*32+j);
            #pragma unroll
            for(int half=0;half<(kind==QuantKind::Q8_0?1:2);half++) {
                const float4 input=half?*reinterpret_cast<const float4*>(x+qb*32+16+j):first;
                #pragma unroll
                for(int k=0;k<4;k++) {
                    int q;
                    if constexpr(kind==QuantKind::Q8_0)q=int8_t(packed>>(k*8));
                    else {
                        int nibble=(packed>>(k*8+half*4))&15;
                        int magnitude=(0xC8643210u>>((nibble&7)*4))&15;
                        q=(nibble&8)?-magnitude:magnitude;
                    }
                    float xi=k==0?input.x:k==1?input.y:k==2?input.z:input.w;
                    sum=fmaf(d*float(q),xi,sum);
                }
            }
        }
    } else {
        for(size_t i=lane;i<x_len;i+=32)sum=fmaf(weight_at<kind>(weights,i),x[i],sum);
    }
    for (int shift=16; shift>0; shift/=2) sum += __shfl_down_sync(0xffffffff,sum,shift);
    return sum;
}
template<QuantKind kind>
__global__ void quant_matvec_kernel(const uint8_t *w,size_t row_bytes,
    const float *x,size_t x_len,size_t ne1,size_t rows_per_batch,float *y) {
    const size_t row=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    if(row>=ne1)return;
    float sum=quant_row_dot<kind>(w+row*row_bytes,x+(row/rows_per_batch)*x_len,x_len);
    if((threadIdx.x&31)==0)y[row]=sum;
}

// Format selection belongs on the host, rather than inside every decoded coefficient.
void launch_quant_kernel(QuantKind kind, const void *w, size_t row_bytes,
    const float *x, size_t cols, size_t rows, size_t rows_per_batch, float *y,
    cudaStream_t stream = nullptr) {
    const int threads = 256;
    const int blocks = int((rows + 7) / 8);
#define LAUNCH_QUANT(K) case QuantKind::K: quant_matvec_kernel<QuantKind::K><<<blocks,threads,0,stream>>>(static_cast<const uint8_t*>(w),row_bytes,x,cols,rows,rows_per_batch,y); break
    switch(kind) {
        LAUNCH_QUANT(F32); LAUNCH_QUANT(Q4_0); LAUNCH_QUANT(Q5_0); LAUNCH_QUANT(Q8_0);
        LAUNCH_QUANT(Q4_K); LAUNCH_QUANT(Q5_K); LAUNCH_QUANT(Q6_K); LAUNCH_QUANT(MXFP4);
    }
#undef LAUNCH_QUANT
}

struct Attention {
    float *k=nullptr, *v=nullptr, *q=nullptr, *y=nullptr, *sinks=nullptr;
    size_t capacity, kv_heads, key_dim, value_dim, heads, filled=0;
    ~Attention() { if(k)cudaFree(k);if(v)cudaFree(v);if(q)cudaFree(q);if(y)cudaFree(y);if(sinks)cudaFree(sinks); }
};
__global__ void attention_kernel(const float *k, const float *v, const float *q, const float *sinks,
    size_t seq, size_t first, size_t kv_heads, size_t heads, size_t key_dim, size_t value_dim, float scale, float *y) {
    extern __shared__ float scores[];
    __shared__ float reductions[4];
    const int tid=threadIdx.x, lane=tid&31, warp=tid/32;
    const size_t head=blockIdx.x, kv_head=head/(heads/kv_heads);
    for(size_t p=first+warp;p<seq;p+=4) {
        float score=0;
        for(size_t i=lane;i<key_dim;i+=32) score=fmaf(q[head*key_dim+i],k[p*kv_heads*key_dim+kv_head*key_dim+i],score);
        for(int shift=16;shift>0;shift/=2)score+=__shfl_down_sync(0xffffffff,score,shift);
        if(lane==0)scores[p-first]=score*scale;
    }
    __syncthreads();
    float maximum=sinks?sinks[head]:-CUDART_INF_F;
    for(size_t i=tid;i<seq-first;i+=128)maximum=fmaxf(maximum,scores[i]);
    for(int shift=16;shift>0;shift/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,shift));
    if(lane==0)reductions[warp]=maximum;
    __syncthreads();maximum=fmaxf(fmaxf(reductions[0],reductions[1]),fmaxf(reductions[2],reductions[3]));
    __syncthreads(); // All warps must read maxima before reusing reductions for sums.
    float sum=0;
    for(size_t i=tid;i<seq-first;i+=128) {scores[i]=expf(scores[i]-maximum);sum+=scores[i];}
    for(int shift=16;shift>0;shift/=2)sum+=__shfl_down_sync(0xffffffff,sum,shift);
    if(lane==0)reductions[warp]=sum;
    __syncthreads();sum=reductions[0]+reductions[1]+reductions[2]+reductions[3]+(sinks?expf(sinks[head]-maximum):0.0f);
    for(size_t i=tid;i<value_dim;i+=128) {
        float output=0;
        for(size_t p=first;p<seq;p++)output=fmaf(scores[p-first]/sum,v[p*kv_heads*value_dim+kv_head*value_dim+i],output);
        y[head*value_dim+i]=output;
    }
}

struct Scratch {
    float *d_x = nullptr;
    float *d_y = nullptr;
    uint8_t *d_w = nullptr;
    size_t cap_x = 0;
    size_t cap_y = 0;
    size_t cap_w = 0;
    ~Scratch() { if (d_x) cudaFree(d_x); if (d_y) cudaFree(d_y); if (d_w) cudaFree(d_w); }
};

thread_local Scratch g_scratch;

bool ensure(size_t need, void **ptr, size_t *cap) {
    MemoryCategoryScope category(MemoryScratch);
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
    float *y, size_t batches = 1) {
    if (!d_w || !x || !y || x_len == 0 || ne1 == 0 || row_bytes == 0) {
        return 1;
    }
    if (!ensure(x_len * batches * sizeof(float), reinterpret_cast<void **>(&g_scratch.d_x), &g_scratch.cap_x)) {
        return 2;
    }
    if (!ensure(ne1 * batches * sizeof(float), reinterpret_cast<void **>(&g_scratch.d_y), &g_scratch.cap_y)) {
        return 3;
    }
    if (cudaMemcpy(g_scratch.d_x, x, x_len * batches * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return 4;
    }
    launch_quant_kernel(kind,d_w,row_bytes,g_scratch.d_x,x_len,ne1*batches,ne1,g_scratch.d_y);
    if (cudaGetLastError() != cudaSuccess) {
        return 5;
    }
    if (cudaMemcpy(y, g_scratch.d_y, ne1 * batches * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
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

int rbitnet_cuda_f32_matvec(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_host_w(QuantKind::F32, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_f32_matvec_device(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_device_w(QuantKind::F32, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_q5_0_matvec(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_host_w(QuantKind::Q5_0, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_q5_0_matvec_device(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_device_w(QuantKind::Q5_0, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_q5_k_matvec(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_host_w(QuantKind::Q5_K, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_q5_k_matvec_device(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_device_w(QuantKind::Q5_K, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_mxfp4_matvec(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_host_w(QuantKind::MXFP4, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_mxfp4_matvec_device(const void *w, size_t row_bytes, const float *x, size_t x_len, size_t ne1, float *y) {
    return launch_device_w(QuantKind::MXFP4, w, row_bytes, x, x_len, ne1, y);
}
int rbitnet_cuda_quant_matvec_batch_device(unsigned ty, const void *w, size_t row_bytes,
    const float *x, size_t cols, size_t rows, size_t batches, float *y) {
    QuantKind kind;
    switch (ty) {
        case 0: kind=QuantKind::F32;break; case 2: kind=QuantKind::Q4_0;break;
        case 6: kind=QuantKind::Q5_0;break; case 8: kind=QuantKind::Q8_0;break;
        case 12: kind=QuantKind::Q4_K;break; case 13: kind=QuantKind::Q5_K;break;
        case 14: kind=QuantKind::Q6_K;break; case 39: kind=QuantKind::MXFP4;break;
        default: return 10;
    }
    if (batches==0) return 11;
    return launch_device_w(kind,w,row_bytes,x,cols,rows,y,batches);
}
void *rbitnet_cuda_attention_create(size_t capacity,size_t kv_heads,size_t key_dim,size_t value_dim,size_t heads) {
    if(!capacity || capacity>8192 || !kv_heads || !heads || heads%kv_heads || !key_dim || !value_dim) return nullptr;
    auto *a=new Attention; a->capacity=capacity;a->kv_heads=kv_heads;a->key_dim=key_dim;a->value_dim=value_dim;a->heads=heads;
    if(memory_allocate(reinterpret_cast<void**>(&a->k),capacity*kv_heads*key_dim*sizeof(float),MemoryKv)!=cudaSuccess
        ||memory_allocate(reinterpret_cast<void**>(&a->v),capacity*kv_heads*value_dim*sizeof(float),MemoryKv)!=cudaSuccess
        ||cudaMalloc(reinterpret_cast<void**>(&a->q),heads*key_dim*sizeof(float))!=cudaSuccess
        ||cudaMalloc(reinterpret_cast<void**>(&a->y),heads*value_dim*sizeof(float))!=cudaSuccess
        ||cudaMalloc(reinterpret_cast<void**>(&a->sinks),heads*sizeof(float))!=cudaSuccess) {delete a;return nullptr;}
    return a;
}
void rbitnet_cuda_attention_destroy(void *context) {delete static_cast<Attention*>(context);}
void rbitnet_cuda_attention_reset(void *context) {if(context)static_cast<Attention*>(context)->filled=0;}
int rbitnet_cuda_attention_step(void *context,const float *q,const float *k,const float *v,size_t pos,size_t first,float scale,const float *sinks,float *y) {
    auto *a=static_cast<Attention*>(context);
    if(!a || !q || !k || !v || !y || pos>=a->capacity || first>pos) return 1;
    size_t begin=pos>=a->filled?a->filled:pos;
    size_t key_stride=a->kv_heads*a->key_dim,value_stride=a->kv_heads*a->value_dim;
    if(cudaMemcpy(a->k+begin*key_stride,k+begin*key_stride,(pos+1-begin)*key_stride*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess
        ||cudaMemcpy(a->v+begin*value_stride,v+begin*value_stride,(pos+1-begin)*value_stride*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess
        ||cudaMemcpy(a->q,q,a->heads*a->key_dim*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess) return 2;
    if(sinks && cudaMemcpy(a->sinks,sinks,a->heads*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess) return 3;
    attention_kernel<<<a->heads,128,(pos+1-first)*sizeof(float)>>>(a->k,a->v,a->q,sinks?a->sinks:nullptr,pos+1,first,a->kv_heads,a->heads,a->key_dim,a->value_dim,scale,a->y);
    if(cudaGetLastError()!=cudaSuccess) return 4;
    if(cudaMemcpy(y,a->y,a->heads*a->value_dim*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess) return 5;
    a->filled=pos+1;return 0;
}
} // extern "C"

#include "prefill_quant.cuh"
#include "llama_resident.cuh"

extern "C" int rbitnet_cuda_split_attention_check(const float *k,const float *v,const float *q,
    unsigned capacity,unsigned kv_heads,unsigned heads,unsigned dim,unsigned window,float scale,
    unsigned count,const unsigned *positions,unsigned steps,unsigned graphs,float *out) {
    if(!k || !v || !q || !positions || !out || !capacity || capacity>8192 || !kv_heads
        || !heads || heads>128 || heads%kv_heads || !dim || dim>512 || !count || count>128 || !steps)return 1;
    for(unsigned i=0;i<steps;i++)if(positions[i]>=capacity || count>capacity-positions[i])return 1;
    ResidentLlama buffers;
    if(cudaStreamCreateWithFlags(&buffers.stream,cudaStreamNonBlocking)!=cudaSuccess)return 2;
    size_t kv=size_t(capacity)*kv_heads*dim,qs=size_t(count)*heads*dim;
    float *dk=nullptr,*dv=nullptr,*dq=nullptr,*scratch=nullptr,*dy=nullptr;unsigned *dp=nullptr;
    if(!buffers.alloc(dk,kv,k) || !buffers.alloc(dv,kv,v) || !buffers.alloc(dq,qs,q)
        || !buffers.alloc(scratch,size_t(count)*heads*((capacity+attention_tile-1)/attention_tile)*(dim+2))
        || !buffers.alloc(dy,qs) || !buffers.alloc(dp,1))return 3;
    if(graphs && cudaStreamBeginCapture(buffers.stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 4;
    if(graphs)launch_split_attention(dk,dv,dq,dp,kv_heads,heads,dim,window,scale,capacity,count,scratch,dy,buffers.stream);
    if(graphs && (cudaStreamEndCapture(buffers.stream,&buffers.graphs[0])!=cudaSuccess
        || cudaGraphInstantiate(&buffers.executable[0],buffers.graphs[0],nullptr,nullptr,0)!=cudaSuccess))return 4;
    for(unsigned i=0;i<steps;i++) {
        if(cudaMemcpyAsync(dp,positions+i,sizeof(unsigned),cudaMemcpyHostToDevice,buffers.stream)!=cudaSuccess)return 5;
        if(graphs) {if(cudaGraphLaunch(buffers.executable[0],buffers.stream)!=cudaSuccess)return 5;}
        else launch_split_attention(dk,dv,dq,dp,kv_heads,heads,dim,window,scale,capacity,count,scratch,dy,buffers.stream);
        if(cudaMemcpyAsync(out+size_t(i)*qs,dy,qs*sizeof(float),cudaMemcpyDeviceToHost,buffers.stream)!=cudaSuccess
            || cudaStreamSynchronize(buffers.stream)!=cudaSuccess)return 5;
    }
    return 0;
}

#include "moe_resident.cuh"

#include "qwen_recurrent.cuh"

#include "output_head.cuh"

#include "qwen_full.cuh"
#include "gpt_full.cuh"
#include "mla_full.cuh"
#include "native_completion_check.cuh"

extern "C" int rbitnet_cuda_quant_gemm_device(unsigned type,const void *weights,size_t row_bytes,
    const float *input,unsigned columns,unsigned rows,unsigned tokens,float *output) {
    RbitnetLlamaMatrix m={weights,row_bytes,type,columns,rows};QuantKind kind;
    if(!input || !output || !tokens || !columns || !rows || !qwen_matrix_valid(m,columns,rows) || !resident_kind(type,kind))return 1;
    launch_prefill_gemm(kind,weights,row_bytes,input,columns,rows,tokens,output,nullptr);
    return cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess ? 0 : 2;
}

extern "C" int rbitnet_cuda_quant_gemm_check(unsigned type,const void *weights,size_t row_bytes,
    const float *input,unsigned columns,unsigned rows,unsigned tokens,unsigned tf32x3,
    unsigned repeats,float *output,float *elapsed_ms) {
    RbitnetLlamaMatrix m={weights,row_bytes,type,columns,rows};QuantKind kind;
    if(!input || !output || !elapsed_ms || !weights || tokens<2 || tokens>128 || !columns || !rows
        || tf32x3>1 || !repeats || repeats>1000 || !qwen_matrix_valid(m,columns,rows) || !resident_kind(type,kind))return 1;
    if(tf32x3 && !tf32_prefill_supported())return 2;
    ResidentLlama buffers;
    if(cudaStreamCreateWithFlags(&buffers.stream,cudaStreamNonBlocking)!=cudaSuccess)return 3;
    uint8_t *dw=nullptr;float *dx=nullptr,*dy=nullptr;
    if(!buffers.alloc(dw,size_t(rows)*row_bytes,static_cast<const uint8_t*>(weights))
        || !buffers.alloc(dx,size_t(tokens)*columns,input) || !buffers.alloc(dy,size_t(tokens)*rows))return 4;
    struct Events {cudaEvent_t begin=nullptr,end=nullptr;~Events(){if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);}} events;
    if(cudaEventCreate(&events.begin)!=cudaSuccess || cudaEventCreate(&events.end)!=cudaSuccess)return 5;
    for(unsigned i=0;i<2;i++)launch_prefill_gemm(kind,dw,row_bytes,dx,columns,rows,tokens,dy,buffers.stream,tf32x3!=0);
    if(cudaGetLastError()!=cudaSuccess || cudaEventRecord(events.begin,buffers.stream)!=cudaSuccess)return 6;
    for(unsigned i=0;i<repeats;i++)launch_prefill_gemm(kind,dw,row_bytes,dx,columns,rows,tokens,dy,buffers.stream,tf32x3!=0);
    if(cudaGetLastError()!=cudaSuccess || cudaEventRecord(events.end,buffers.stream)!=cudaSuccess
        || cudaEventSynchronize(events.end)!=cudaSuccess || cudaEventElapsedTime(elapsed_ms,events.begin,events.end)!=cudaSuccess)return 6;
    *elapsed_ms/=repeats;
    return cudaMemcpyAsync(output,dy,size_t(tokens)*rows*sizeof(float),cudaMemcpyDeviceToHost,buffers.stream)==cudaSuccess
        && cudaStreamSynchronize(buffers.stream)==cudaSuccess ? 0 : 7;
}
