// SPDX-License-Identifier: MIT
// SIMT Wq * X GEMM: dequantize each shared weight tile once for 16 tokens.
// FP32 accumulation; no full floating-point copy of model weights.
namespace {
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
    unsigned cols,unsigned rows,unsigned tokens,float *y,cudaStream_t stream) {
    if(tokens==1) {launch_quant_kernel(kind,w,row_bytes,x,cols,rows,rows,y,stream);return;}
    dim3 grid((rows+15)/16,(tokens+15)/16);
#define PREFILL_QUANT(K) case QuantKind::K: quant_prefill_gemm<QuantKind::K><<<grid,256,0,stream>>>(static_cast<const uint8_t*>(w),row_bytes,x,cols,rows,tokens,y);break
    switch(kind) {PREFILL_QUANT(F32);PREFILL_QUANT(Q4_0);PREFILL_QUANT(Q5_0);PREFILL_QUANT(Q8_0);PREFILL_QUANT(Q4_K);PREFILL_QUANT(Q5_K);PREFILL_QUANT(Q6_K);PREFILL_QUANT(MXFP4);}
#undef PREFILL_QUANT
}
struct LlamaBlock {
    static constexpr unsigned capacity=128;
    float *p[9]={};
    ~LlamaBlock() {for(auto ptr:p)if(ptr)cudaFree(ptr);}
    bool init(unsigned embd,unsigned ffn,unsigned stride) {
        unsigned widths[]={embd,embd,embd,stride,stride,embd,embd,ffn,ffn};
        for(unsigned i=0;i<9;i++)if(cudaMalloc(reinterpret_cast<void**>(&p[i]),size_t(widths[i])*capacity*sizeof(float))!=cudaSuccess)return false;
        return true;
    }
};
}
