// SPDX-License-Identifier: MIT
// SIMT Wq * X GEMM: dequantize each shared weight tile once for 16 tokens.
// FP32 accumulation; no full floating-point copy of model weights.
#include <cstdlib>
namespace {
bool tf32_prefill_supported();
bool tf32_prefill_requested() {
    const char *flag=std::getenv("RBITNET_CUDA_PREFILL_TF32X3");
    return flag && flag[0]=='1' && flag[1]=='\0' && tf32_prefill_supported();
}
bool use_tf32_prefill(unsigned cols,unsigned rows,unsigned tokens,bool enabled) {
    // Small grids lose against SIMT on RTX 4080 SUPER; keep those projections
    // on the reference kernel, including short speculative verification blocks.
    return enabled && tokens>=64 && cols>=1024 && size_t(rows)*tokens>=131072;
}
__device__ __forceinline__ float round_tf32(float value) {
#if __CUDA_ARCH__ >= 800
    unsigned bits;
    // RNA conversion is available from SM80; TF32 RN is SM90-only.
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(value));
    unsigned original=__float_as_uint(value);
    if((original&0x7fffffffu)<0x7f800000u && (bits&0x7fffffffu)==0x7f800000u)bits=original&0xffffe000u;
    return __uint_as_float(bits);
#else
    return value;
#endif
}
// Three Tensor Core products: Whi*Xhi + Whi*Xlo + Wlo*Xhi.
// Accumulate each K=32 subtotal outside the Tensor Core in FP32 RN, rather
// than carrying a growing accumulator through Tensor Core RZ additions.
// This omits Wlo*Xlo and is not bit-exact SGEMM or arbitrary-input FP32 emulation.
template<QuantKind kind>
__global__ void quant_prefill_tf32x3(const uint8_t *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y) {
#if __CUDA_ARCH__ >= 800
    namespace wm=nvcuda::wmma;
    // 32x32 output tile: share each dequantized row across twice as many
    // tokens as 64x16. Skew strides by 8 floats to reduce shared-bank conflicts.
    __shared__ __align__(32) float wh[32][40],wl[32][40],xh[32][40],xl[32][40],output[32][32];
    unsigned tid=threadIdx.x,warp=tid/32;
    wm::fragment<wm::accumulator,16,16,8,float> total,main,correction;
    wm::fill_fragment(total,0.0f);
    for(unsigned base=0;base<cols;base+=32) {
        for(unsigned i=tid;i<32*32;i+=128) {
            unsigned r=i/32,k=i%32,row=blockIdx.x*32+r;
            float value=(row<rows && base+k<cols)?weight_at<kind>(w+size_t(row)*row_bytes,base+k):0;
            float high=round_tf32(value);wh[r][k]=high;wl[r][k]=round_tf32(__fsub_rn(value,high));
        }
        for(unsigned i=tid;i<32*32;i+=128) {
            unsigned t=i/32,k=i%32,token=blockIdx.y*32+t;
            float value=(token<tokens && base+k<cols)?x[size_t(token)*cols+base+k]:0;
            float high=round_tf32(value);xh[t][k]=high;xl[t][k]=round_tf32(__fsub_rn(value,high));
        }
        __syncthreads();wm::fill_fragment(main,0.0f);wm::fill_fragment(correction,0.0f);
        #pragma unroll
        for(unsigned k=0;k<32;k+=8) {
            wm::fragment<wm::matrix_a,16,16,8,wm::precision::tf32,wm::row_major> ah,al;
            wm::fragment<wm::matrix_b,16,16,8,wm::precision::tf32,wm::col_major> bh,bl;
            wm::load_matrix_sync(ah,&wh[(warp/2)*16][k],40);wm::load_matrix_sync(al,&wl[(warp/2)*16][k],40);
            wm::load_matrix_sync(bh,&xh[(warp%2)*16][k],40);wm::load_matrix_sync(bl,&xl[(warp%2)*16][k],40);
            wm::mma_sync(main,ah,bh,main);
            wm::mma_sync(correction,ah,bl,correction);wm::mma_sync(correction,al,bh,correction);
        }
        for(unsigned i=0;i<total.num_elements;i++)total.x[i]=__fadd_rn(total.x[i],__fadd_rn(main.x[i],correction.x[i]));
        __syncthreads();
    }
    wm::store_matrix_sync(&output[(warp/2)*16][(warp%2)*16],total,32,wm::mem_row_major);
    __syncthreads();
    for(unsigned i=tid;i<32*32;i+=128) {
        unsigned r=i/32,t=i%32,row=blockIdx.x*32+r,token=blockIdx.y*32+t;
        if(row<rows && token<tokens)y[size_t(token)*rows+row]=output[r][t];
    }
#endif
}
bool tf32_prefill_supported() {
    cudaFuncAttributes attributes;
    auto status=cudaFuncGetAttributes(&attributes,quant_prefill_tf32x3<QuantKind::F32>);
    if(status!=cudaSuccess) {cudaGetLastError();return false;}
    // A custom compute_75 PTX build can JIT on SM80+, but its preprocessor
    // excluded WMMA TF32. Check the actual image, not just the physical GPU.
    return attributes.binaryVersion>=80 && attributes.ptxVersion>=80;
}
template<QuantKind kind>
__global__ void quant_prefill_gemm(const uint8_t *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y) {
    __shared__ float weights[16][33],inputs[16][33];
    unsigned tid=threadIdx.x,r=tid/16,t=tid%16;
    unsigned row=blockIdx.x*16+r,token=blockIdx.y*16+t;
    float sum=0;
    for(unsigned base=0;base<cols;base+=32) {
        for(unsigned i=tid;i<512;i+=256) {
            unsigned a=i/32,k=i%32,wr=blockIdx.x*16+a,xt=blockIdx.y*16+a;
            weights[a][k]=(wr<rows && base+k<cols)?weight_at<kind>(w+size_t(wr)*row_bytes,base+k):0;
            inputs[a][k]=(xt<tokens && base+k<cols)?x[size_t(xt)*cols+base+k]:0;
        }
        __syncthreads();
        #pragma unroll
        for(unsigned k=0;k<32;k++)sum=fmaf(weights[r][k],inputs[t][k],sum);
        __syncthreads();
    }
    if(row<rows && token<tokens)y[size_t(token)*rows+row]=sum;
}
void launch_prefill_gemm(QuantKind kind,const void *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y,cudaStream_t stream,bool tf32x3=false) {
    if(tokens==1) {launch_quant_kernel(kind,w,row_bytes,x,cols,rows,rows,y,stream);return;}
    if(tf32x3) {
        dim3 grid((rows+31)/32,(tokens+31)/32);
#define PREFILL_TF32(K) case QuantKind::K: quant_prefill_tf32x3<QuantKind::K><<<grid,128,0,stream>>>(static_cast<const uint8_t*>(w),row_bytes,x,cols,rows,tokens,y);break
        switch(kind) {PREFILL_TF32(F32);PREFILL_TF32(Q4_0);PREFILL_TF32(Q5_0);PREFILL_TF32(Q8_0);PREFILL_TF32(Q4_K);PREFILL_TF32(Q5_K);PREFILL_TF32(Q6_K);PREFILL_TF32(MXFP4);}
#undef PREFILL_TF32
        return;
    }
    dim3 grid((rows+15)/16,(tokens+15)/16);
#define PREFILL_QUANT(K) case QuantKind::K: quant_prefill_gemm<QuantKind::K><<<grid,256,0,stream>>>(static_cast<const uint8_t*>(w),row_bytes,x,cols,rows,tokens,y);break
    switch(kind) {PREFILL_QUANT(F32);PREFILL_QUANT(Q4_0);PREFILL_QUANT(Q5_0);PREFILL_QUANT(Q8_0);PREFILL_QUANT(Q4_K);PREFILL_QUANT(Q5_K);PREFILL_QUANT(Q6_K);PREFILL_QUANT(MXFP4);}
#undef PREFILL_QUANT
}
struct LlamaBlock {
    static constexpr unsigned capacity=128;
    static constexpr unsigned verify_capacity=16;
    float *p[9]={};
    float *verify_logits=nullptr,*verify_maxima=nullptr;
    unsigned *verify_ids=nullptr,*verify_tokens=nullptr;
    ~LlamaBlock() {for(auto ptr:p)if(ptr)cudaFree(ptr);if(verify_logits)cudaFree(verify_logits);if(verify_maxima)cudaFree(verify_maxima);if(verify_ids)cudaFree(verify_ids);if(verify_tokens)cudaFree(verify_tokens);}
    bool init_verify(unsigned vocab) {
        if(verify_logits)return true;
        unsigned blocks=(vocab+255)/256;
        float *logits=nullptr,*maxima=nullptr;unsigned *ids=nullptr,*tokens=nullptr;
        bool ok=cudaMalloc(reinterpret_cast<void**>(&logits),size_t(verify_capacity)*vocab*sizeof(float))==cudaSuccess
            && cudaMalloc(reinterpret_cast<void**>(&maxima),size_t(verify_capacity)*blocks*sizeof(float))==cudaSuccess
            && cudaMalloc(reinterpret_cast<void**>(&ids),size_t(verify_capacity)*blocks*sizeof(unsigned))==cudaSuccess
            && cudaMalloc(reinterpret_cast<void**>(&tokens),verify_capacity*sizeof(unsigned))==cudaSuccess;
        if(!ok) {if(logits)cudaFree(logits);if(maxima)cudaFree(maxima);if(ids)cudaFree(ids);if(tokens)cudaFree(tokens);return false;}
        verify_logits=logits;verify_maxima=maxima;verify_ids=ids;verify_tokens=tokens;return true;
    }
    bool init(unsigned embd,unsigned ffn,unsigned stride) {
        unsigned widths[]={embd,embd,embd,stride,stride,embd,embd,ffn,ffn};
        for(unsigned i=0;i<9;i++)if(cudaMalloc(reinterpret_cast<void**>(&p[i]),size_t(widths[i])*capacity*sizeof(float))!=cudaSuccess)return false;
        return true;
    }
};
}
