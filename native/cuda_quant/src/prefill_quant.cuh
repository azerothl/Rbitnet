// SPDX-License-Identifier: MIT
// TF32 Wq * X GEMM. Q4_K is expanded once into a reused FP32 workspace, then
// each 32-row by 32-token tile accumulates in registers.
#include <cublas_v2.h>
#include <cstdio>
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
__device__ void q4k_group_scale(const uint8_t *row,unsigned col,float &scale,float &minimum) {
    const uint8_t *b=row+(size_t(col)/256)*144;
    int group=(col/32)%8;
    float d=fp16_bits_to_f32(uint16_t(b[0])|(uint16_t(b[1])<<8));
    float minv=fp16_bits_to_f32(uint16_t(b[2])|(uint16_t(b[3])<<8));
    uint8_t sc,m;get_scale_min_k4(group,b+4,&sc,&m);
    scale=__fmul_rn(d,float(sc));
    minimum=__fmul_rn(minv,float(m));
}
__device__ float q4k_group_value(const uint8_t *row,unsigned col,unsigned lane,float scale,float minimum) {
    const uint8_t *b=row+(size_t(col)/256)*144;
    int group=(col/32)%8;
    int q=(b[16+(group/2)*32+lane]>>((group&1)*4))&15;
    return __fsub_rn(__fmul_rn(scale,float(q)),minimum);
}
// Four Q4_K codes from one aligned word. `sub` is 0..7. Writes `dst4[0..3]`.
__device__ void q4k_store4(const uint8_t *row,unsigned col,int sub,float scale,float minimum,float *dst4) {
    const uint8_t *b=row+(size_t(col)/256)*144;
    int group=(col/32)%8;
    uint32_t packed=*reinterpret_cast<const uint32_t*>(b+16+(group/2)*32+sub*4);
    packed=(packed>>((group&1)*4))&0x0f0f0f0fu;
    #pragma unroll
    for(int k=0;k<4;k++) {
        int q=(packed>>(8*k))&15;
        dst4[k]=__fsub_rn(__fmul_rn(scale,float(q)),minimum);
    }
}
__device__ void q4k_store4_tf32(const uint8_t *row,unsigned col,int sub,float scale,float minimum,float *hi,float *lo) {
    float values[4];
    q4k_store4(row,col,sub,scale,minimum,values);
    #pragma unroll
    for(int k=0;k<4;k++) {
        float high=round_tf32(values[k]);
        hi[sub*4+k]=high;
        lo[sub*4+k]=round_tf32(__fsub_rn(values[k],high));
    }
}

template<QuantKind kind>
__global__ void quant_prefill_tf32x3(const uint8_t *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y) {
#if __CUDA_ARCH__ >= 800
    namespace wm=nvcuda::wmma;
    __shared__ float q4_scale[32],q4_min[32];
    // 32x32 output tile: share each dequantized row across twice as many
    // tokens as 64x16. Skew strides by 8 floats to reduce shared-bank conflicts.
    __shared__ __align__(32) float wh[32][40],wl[32][40],xh[32][40],xl[32][40],output[32][32];
    unsigned tid=threadIdx.x,warp=tid/32;
    wm::fragment<wm::accumulator,16,16,8,float> total,main,correction;
    wm::fill_fragment(total,0.0f);
    for(unsigned base=0;base<cols;base+=32) {
        if constexpr(kind==QuantKind::Q4_K) {
            if(tid<32) {
                unsigned row=blockIdx.x*32+tid;
                if(row<rows && base<cols) q4k_group_scale(w+size_t(row)*row_bytes,base,q4_scale[tid],q4_min[tid]);
                else {q4_scale[tid]=0;q4_min[tid]=0;}
            }
            __syncthreads();
            for(int pass=0;pass<2;pass++) {
                unsigned job=tid+pass*128,r=job/8,sub=job%8,row=blockIdx.x*32+r;
                if(row<rows && base<cols) q4k_store4_tf32(w+size_t(row)*row_bytes,base,int(sub),q4_scale[r],q4_min[r],wh[r],wl[r]);
                else for(int k=0;k<4;k++) {wh[r][sub*4+k]=0;wl[r][sub*4+k]=0;}
            }
        } else {
        for(unsigned i=tid;i<32*32;i+=128) {
            unsigned r=i/32,k=i%32,row=blockIdx.x*32+r;
            float value=(row<rows && base+k<cols)?weight_at<kind>(w+size_t(row)*row_bytes,base+k):0;
            float high=round_tf32(value);wh[r][k]=high;wl[r][k]=round_tf32(__fsub_rn(value,high));
        }
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
    __shared__ float q4_scale[16],q4_min[16];
    unsigned tid=threadIdx.x,r=tid/16,t=tid%16;
    unsigned row=blockIdx.x*16+r,token=blockIdx.y*16+t;
    float sum=0;
    for(unsigned base=0;base<cols;base+=32) {
        if constexpr(kind==QuantKind::Q4_K) {
            if(tid<16) {
                unsigned wr=blockIdx.x*16+tid;
                if(wr<rows && base<cols) q4k_group_scale(w+size_t(wr)*row_bytes,base,q4_scale[tid],q4_min[tid]);
                else {q4_scale[tid]=0;q4_min[tid]=0;}
            }
            __syncthreads();
        }
        for(unsigned i=tid;i<512;i+=256) {
            unsigned a=i/32,k=i%32,wr=blockIdx.x*16+a,xt=blockIdx.y*16+a;
            float value=0;
            if(wr<rows && base+k<cols) {
                if constexpr(kind==QuantKind::Q4_K) value=q4k_group_value(w+size_t(wr)*row_bytes,base,k,q4_scale[a],q4_min[a]);
                else value=weight_at<kind>(w+size_t(wr)*row_bytes,base+k);
            }
            weights[a][k]=value;
            inputs[a][k]=(xt<tokens && base+k<cols)?x[size_t(xt)*cols+base+k]:0;
        }
        __syncthreads();
        #pragma unroll
        for(unsigned k=0;k<32;k++)sum=fmaf(weights[r][k],inputs[t][k],sum);
        __syncthreads();
    }
    if(row<rows && token<tokens)y[size_t(token)*rows+row]=sum;
}
__global__ void dequant_q4k_rows(const uint8_t *w,size_t row_bytes,unsigned cols,unsigned rows,float *dst) {
    unsigned row=blockIdx.x*blockDim.y+threadIdx.y;
    if(row>=rows) return;
    const uint8_t *rowp=w+size_t(row)*row_bytes;
    float *out=dst+size_t(row)*cols;
    for(unsigned base=threadIdx.x*32;base<cols;base+=blockDim.x*32) {
        float scale,minimum;
        q4k_group_scale(rowp,base,scale,minimum);
        float unpacked[32];
        #pragma unroll
        for(int sub=0;sub<8;sub++) q4k_store4(rowp,base,sub,scale,minimum,unpacked+sub*4);
        #pragma unroll
        for(int k=0;k<32;k++) out[base+k]=unpacked[k];
    }
}
__global__ void dequant_q8_rows(const uint8_t *w,size_t row_bytes,unsigned cols,unsigned rows,float *dst) {
    unsigned row=blockIdx.x;
    if(row>=rows) return;
    const uint8_t *rowp=w+size_t(row)*row_bytes;
    float *out=dst+size_t(row)*cols;
    for(unsigned i=threadIdx.x;i<cols;i+=blockDim.x) {
        unsigned lane=i%32;
        const uint8_t *block=rowp+size_t(i/32)*34;
        float scale=fp16_bits_to_f32(uint16_t(block[0])|(uint16_t(block[1])<<8));
        out[i]=scale*float(int8_t(block[lane+2]));
    }
}
float *q4k_f32_workspace(size_t n) {
    static float *buf=nullptr;
    static size_t cap=0;
    if(n<=cap) return buf;
    if(buf) cudaFree(buf);
    buf=nullptr;cap=0;
    if(cudaMalloc(reinterpret_cast<void**>(&buf),n*sizeof(float))!=cudaSuccess) return nullptr;
    cap=n;
    return buf;
}
cublasHandle_t prefill_cublas() {
    static cublasHandle_t handle=nullptr;
    if(!handle && cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS) {handle=nullptr;return nullptr;}
    return handle;
}
template<QuantKind kind>
__global__ void quant_prefill_tf32(const uint8_t *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y) {
#if __CUDA_ARCH__ >= 800
    namespace wm=nvcuda::wmma;
    __shared__ float q4_scale[32],q4_min[32];
    __shared__ __align__(32) float wh[32][40],xh[32][40],output[32][32];
    unsigned tid=threadIdx.x,warp=tid/32;
    wm::fragment<wm::accumulator,16,16,8,float> total;
    wm::fill_fragment(total,0.0f);
    for(unsigned base=0;base<cols;base+=32) {
        if constexpr(kind==QuantKind::Q4_K) {
            if(tid<32) {
                unsigned row=blockIdx.x*32+tid;
                if(row<rows && base<cols) q4k_group_scale(w+size_t(row)*row_bytes,base,q4_scale[tid],q4_min[tid]);
                else {q4_scale[tid]=0;q4_min[tid]=0;}
            }
            __syncthreads();
            for(int pass=0;pass<2;pass++) {
                unsigned job=tid+pass*128,r=job/8,sub=job%8,row=blockIdx.x*32+r;
                if(row<rows && base<cols) {
                    float values[4];
                    q4k_store4(w+size_t(row)*row_bytes,base,int(sub),q4_scale[r],q4_min[r],values);
                    #pragma unroll
                    for(int k=0;k<4;k++) wh[r][sub*4+k]=round_tf32(values[k]);
                } else for(int k=0;k<4;k++) wh[r][sub*4+k]=0;
            }
        } else {
            for(unsigned i=tid;i<32*32;i+=128) {
                unsigned r=i/32,k=i%32,row=blockIdx.x*32+r;
                float value=(row<rows && base+k<cols)?weight_at<kind>(w+size_t(row)*row_bytes,base+k):0;
                wh[r][k]=round_tf32(value);
            }
        }
        for(unsigned i=tid;i<32*32;i+=128) {
            unsigned t=i/32,k=i%32,token=blockIdx.y*32+t;
            float value=(token<tokens && base+k<cols)?x[size_t(token)*cols+base+k]:0;
            xh[t][k]=round_tf32(value);
        }
        __syncthreads();
        #pragma unroll
        for(unsigned k=0;k<32;k+=8) {
            wm::fragment<wm::matrix_a,16,16,8,wm::precision::tf32,wm::row_major> a;
            wm::fragment<wm::matrix_b,16,16,8,wm::precision::tf32,wm::col_major> b;
            wm::load_matrix_sync(a,&wh[(warp/2)*16][k],40);
            wm::load_matrix_sync(b,&xh[(warp%2)*16][k],40);
            wm::mma_sync(total,a,b,total);
        }
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
void launch_prefill_tf32(QuantKind kind,const void *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y,cudaStream_t stream) {
    const void *weights=w;
    size_t used_row_bytes=row_bytes;
    QuantKind used=kind;
    if(kind==QuantKind::Q4_K) {
        float *ws=q4k_f32_workspace(size_t(rows)*cols);
        cublasHandle_t handle=prefill_cublas();
        if(ws && handle) {
            dim3 block(32,4);
            cublasSetStream(handle,stream);
            cublasSetMathMode(handle,CUBLAS_TF32_TENSOR_OP_MATH);
            float alpha=1.0f,beta=0.0f;
            static int timing=-1;
            if(timing<0) {const char *flag=std::getenv("RBITNET_PREFILL_TIMING");timing=flag && flag[0]=='1' && flag[1]=='\0';}
            static cudaEvent_t ev0,ev1,ev2;static int events=0,calls=0;static float dequant_ms=0,gemm_ms=0;
            if(timing && !stream) {
                if(!events) {cudaEventCreate(&ev0);cudaEventCreate(&ev1);cudaEventCreate(&ev2);events=1;}
                else {
                    float a=0,b=0;cudaEventElapsedTime(&a,ev0,ev1);cudaEventElapsedTime(&b,ev1,ev2);
                    dequant_ms+=a;gemm_ms+=b;calls++;
                    if(calls==251) std::fprintf(stderr,"qwen3 device dequant_ms=%.0f cublas_ms=%.0f\n",dequant_ms,gemm_ms);
                }
                cudaEventRecord(ev0,stream);
            }
            dequant_q4k_rows<<<(rows+3)/4,block,0,stream>>>(static_cast<const uint8_t*>(w),row_bytes,cols,rows,ws);
            if(timing && !stream) cudaEventRecord(ev1,stream);
            cublasStatus_t st=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,
                int(rows),int(tokens),int(cols),&alpha,ws,int(cols),x,int(cols),&beta,y,int(rows));
            if(timing && !stream) cudaEventRecord(ev2,stream);
            static int logged=0;
            if(!logged) {std::fprintf(stderr,"qwen3 cublasSgemm status=%d rows=%u cols=%u tokens=%u\n",int(st),rows,cols,tokens);logged=1;}
            if(st==CUBLAS_STATUS_SUCCESS) return;
        }
    }
    dim3 grid((rows+31)/32,(tokens+31)/32);
    const uint8_t *bytes=static_cast<const uint8_t*>(weights);
#define PREFILL_TF32_1(K) case QuantKind::K: quant_prefill_tf32<QuantKind::K><<<grid,128,0,stream>>>(bytes,used_row_bytes,x,cols,rows,tokens,y);break
    switch(used) {PREFILL_TF32_1(F32);PREFILL_TF32_1(Q4_0);PREFILL_TF32_1(Q5_0);PREFILL_TF32_1(Q8_0);PREFILL_TF32_1(Q4_K);PREFILL_TF32_1(Q5_K);PREFILL_TF32_1(Q6_K);PREFILL_TF32_1(MXFP4);}
#undef PREFILL_TF32_1
}
bool launch_prefill_q8_cublas(const void *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y,cudaStream_t stream) {
    float *ws=q4k_f32_workspace(size_t(rows)*cols);
    cublasHandle_t handle=prefill_cublas();
    if(!ws || !handle) return false;
    dequant_q8_rows<<<rows,256,0,stream>>>(static_cast<const uint8_t*>(w),row_bytes,cols,rows,ws);
    cublasSetStream(handle,stream);
    cublasSetMathMode(handle,CUBLAS_TF32_TENSOR_OP_MATH);
    float alpha=1.0f,beta=0.0f;
    cublasStatus_t st=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,
        int(rows),int(tokens),int(cols),&alpha,ws,int(cols),x,int(cols),&beta,y,int(rows));
    return st==CUBLAS_STATUS_SUCCESS;
}
void launch_prefill_gemm(QuantKind kind,const void *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y,cudaStream_t stream,bool tf32x3=false) {
    if(tokens==1) {launch_quant_kernel(kind,w,row_bytes,x,cols,rows,rows,y,stream);return;}
    if(tf32x3 && kind==QuantKind::Q8_0 && tokens>=64 && cols>=1024
        && launch_prefill_q8_cublas(w,row_bytes,x,cols,rows,tokens,y,stream)) return;
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
