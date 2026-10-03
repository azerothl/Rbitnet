// Exact ordered quantized projections. Include after llama_resident.cuh,
// where quant_row_dot and resident_kind exist; retain original GGUF bytes.
namespace {
#include "mxfp4_ordered_gemm.cuh"

template<QuantKind kind>
__global__ void ordered_warp_gemm(const uint8_t *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y) {
    unsigned row=blockIdx.x*8+threadIdx.x/32,token=blockIdx.y;
    if(row>=rows || token>=tokens)return;
    float value=quant_row_dot<kind>(w+size_t(row)*row_bytes,x+size_t(token)*cols,cols);
    if(!(threadIdx.x&31))y[size_t(token)*rows+row]=value;
}
void launch_ordered_gemm(QuantKind kind,const uint8_t *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y,cudaStream_t stream,unsigned tile=0) {
    if(kind==QuantKind::MXFP4 && tile==1) {
        dim3 grid((rows+3)/4,(tokens+3)/4);
        mxfp4_ordered_gemm<4,4><<<grid,512,0,stream>>>(w,row_bytes,x,cols,rows,tokens,y);return;
    }
    if(kind==QuantKind::MXFP4 && tile==2) {
        dim3 grid((rows+1)/2,(tokens+7)/8);
        mxfp4_ordered_gemm<2,8><<<grid,512,0,stream>>>(w,row_bytes,x,cols,rows,tokens,y);return;
    }
    // Every lane retains exactly the original format-specific dot/reduction.
    // Separate token blocks share immutable weight addresses through cache.
    dim3 grid((rows+7)/8,tokens);
#define ORDERED_GEMM(K) case QuantKind::K: ordered_warp_gemm<QuantKind::K><<<grid,256,0,stream>>>(w,row_bytes,x,cols,rows,tokens,y);break
    switch(kind) {ORDERED_GEMM(F32);ORDERED_GEMM(Q4_0);ORDERED_GEMM(Q5_0);ORDERED_GEMM(Q8_0);ORDERED_GEMM(Q4_K);ORDERED_GEMM(Q5_K);ORDERED_GEMM(Q6_K);ORDERED_GEMM(MXFP4);}
#undef ORDERED_GEMM
}

struct OrderedGemmDiagnostic {
    cudaStream_t stream=nullptr;cudaEvent_t start=nullptr,stop=nullptr;
    uint8_t *w=nullptr;float *x=nullptr,*y=nullptr,*reference=nullptr;
    ~OrderedGemmDiagnostic() {
        if(stream)cudaStreamSynchronize(stream);
        if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);
        if(w)cudaFree(w);if(x)cudaFree(x);if(y)cudaFree(y);if(reference)cudaFree(reference);
        if(stream)cudaStreamDestroy(stream);
    }
};
}

extern "C" {
// mode0=ordered warps; 1=MXFP4 4rows*4tokens; 2=MXFP4 2rows*8tokens;
// mode3=the original token GEMV launches. Always exports that baseline too.
int rbitnet_cuda_ordered_gemm_check(unsigned type,const void *weights,size_t row_bytes,
    const float *input,unsigned cols,unsigned rows,unsigned tokens,unsigned mode,
    unsigned repeats,float *out,float *reference,float *elapsed_ms) {
    QuantKind kind;
    if(!weights || !input || !out || !reference || !elapsed_ms || !resident_kind(type,kind)
        || !cols || cols>32768 || !rows || rows>65536 || !tokens || tokens>128
        || size_t(rows)*tokens>1048576 || mode>3 || ((mode==1 || mode==2) && type!=39)
        || !repeats || repeats>1000)return 1;
    unsigned block=type==0?1:(type==12 || type==13 || type==14)?256:32;
    unsigned bytes=type==0?4:type==2?18:type==6?22:type==8?34:type==12?144:type==13?176:type==14?210:17;
    if(cols%block || row_bytes!=size_t(cols/block)*bytes || size_t(rows)*row_bytes>256u*1024u*1024u)return 1;
    OrderedGemmDiagnostic d;
    const size_t weight_bytes=size_t(rows)*row_bytes,input_bytes=size_t(tokens)*cols*sizeof(float),output_bytes=size_t(tokens)*rows*sizeof(float);
    if(cudaStreamCreateWithFlags(&d.stream,cudaStreamNonBlocking)!=cudaSuccess
        || cudaEventCreate(&d.start)!=cudaSuccess || cudaEventCreate(&d.stop)!=cudaSuccess)return 2;
    {MemoryCategoryScope category(MemoryWeights);if(cudaMalloc(reinterpret_cast<void**>(&d.w),weight_bytes)!=cudaSuccess)return 2;}
    if(cudaMalloc(reinterpret_cast<void**>(&d.x),input_bytes)!=cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&d.y),output_bytes)!=cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&d.reference),output_bytes)!=cudaSuccess
        || cudaMemcpyAsync(d.w,weights,weight_bytes,cudaMemcpyHostToDevice,d.stream)!=cudaSuccess
        || cudaMemcpyAsync(d.x,input,input_bytes,cudaMemcpyHostToDevice,d.stream)!=cudaSuccess)return 2;
    auto launch=[&] {
        if(mode==3)for(unsigned t=0;t<tokens;t++)launch_quant_kernel(kind,d.w,row_bytes,d.x+size_t(t)*cols,cols,rows,rows,d.y+size_t(t)*rows,d.stream);
        else launch_ordered_gemm(kind,d.w,row_bytes,d.x,cols,rows,tokens,d.y,d.stream,mode);
    };
    launch(); // One warmup, excluded from the CUDA event interval.
    if(cudaGetLastError()!=cudaSuccess || cudaEventRecord(d.start,d.stream)!=cudaSuccess)return 3;
    for(unsigned i=0;i<repeats;i++)launch();
    if(cudaGetLastError()!=cudaSuccess || cudaEventRecord(d.stop,d.stream)!=cudaSuccess
        || cudaEventSynchronize(d.stop)!=cudaSuccess || cudaEventElapsedTime(elapsed_ms,d.start,d.stop)!=cudaSuccess)return 3;
    *elapsed_ms/=float(repeats);
    for(unsigned t=0;t<tokens;t++)launch_quant_kernel(kind,d.w,row_bytes,d.x+size_t(t)*cols,cols,rows,rows,d.reference+size_t(t)*rows,d.stream);
    if(cudaGetLastError()!=cudaSuccess
        || cudaMemcpyAsync(out,d.y,output_bytes,cudaMemcpyDeviceToHost,d.stream)!=cudaSuccess
        || cudaMemcpyAsync(reference,d.reference,output_bytes,cudaMemcpyDeviceToHost,d.stream)!=cudaSuccess
        || cudaStreamSynchronize(d.stream)!=cudaSuccess)return 3;
    return 0;
}
}
