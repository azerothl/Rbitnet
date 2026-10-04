// SPDX-License-Identifier: MIT
// Diagnostic ABI for independent optional CUDA fixtures; not used by inference.
namespace {
struct GptOrderedCheckBuffers {
    cudaStream_t stream=nullptr;std::vector<void*> owners;
    ~GptOrderedCheckBuffers(){if(stream)cudaStreamSynchronize(stream);for(void *p:owners)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    bool init(){return cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess;}
    template<typename T> bool alloc(T *&p,size_t count,const T *host=nullptr) {
        if(!count || count>std::numeric_limits<size_t>::max()/sizeof(T))return false;
        if(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T))!=cudaSuccess)return false;
        try {owners.push_back(p);}catch(const std::bad_alloc&){cudaFree(p);p=nullptr;return false;}
        return !host || cudaMemcpyAsync(p,host,count*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess;
    }
    bool download(float *host,const float *device,size_t count){return cudaMemcpyAsync(host,device,count*sizeof(float),cudaMemcpyDeviceToHost,stream)==cudaSuccess;}
};
}
extern "C" RBITNET_CUDA_API int rbitnet_cuda_gpt_ordered_norm_check(const float *x,const float *residual,const float *weights,
    unsigned n,unsigned count,float epsilon,float *reference,float *staged,float *reference_x,float *staged_x) {
    if(!x || !weights || !reference || !staged || !reference_x || !staged_x || !n || n>65536 || !count || count>32 || !(epsilon>0) || !std::isfinite(epsilon))return -1;
    MemoryCategoryScope category(MemoryScratch);GptOrderedCheckBuffers b;
    float *left=nullptr,*right=nullptr,*norm=nullptr,*add=nullptr,*before=nullptr,*after=nullptr;const size_t size=size_t(n)*count;
    if(!b.init() || !b.alloc(left,size,x) || !b.alloc(right,size,x) || !b.alloc(norm,n,weights)
        || (residual && !b.alloc(add,size,residual)) || !b.alloc(before,size) || !b.alloc(after,size))return -2;
    gpt_norm_ordered<<<count,256,0,b.stream>>>(left,norm,epsilon,n,before,add);
    // Direct oracle compares the new implementation even in a disabled ablation.
    if(n<=8192)gpt_norm_staged<<<count,256,size_t(n)*sizeof(float),b.stream>>>(right,norm,epsilon,n,after,add);
    else gpt_norm_ordered<<<count,256,0,b.stream>>>(right,norm,epsilon,n,after,add);
    if(cudaGetLastError()!=cudaSuccess || !b.download(reference,before,size) || !b.download(staged,after,size)
        || !b.download(reference_x,left,size) || !b.download(staged_x,right,size) || cudaStreamSynchronize(b.stream)!=cudaSuccess)return -3;
    return 0;
}
