// SPDX-License-Identifier: MIT
// F32 checkpoint transport. Model/tokenizer/config identities and disk
// checksums belong to the Rust sealed envelope; these APIs never trust a file.
extern "C" RBITNET_CUDA_API int rbitnet_cuda_portable_device_key(unsigned char *out,size_t bytes) {
    if(!out || bytes!=32)return 1;
    int device=0,driver=0;cudaDeviceProp properties{};
    if(cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&properties,device)!=cudaSuccess
        || cudaDriverGetVersion(&driver)!=cudaSuccess)return 2;
    std::memcpy(out,properties.uuid.bytes,16);
    const uint32_t details[4]={1u,uint32_t(properties.major),uint32_t(properties.minor),uint32_t(driver)};
    std::memcpy(out+16,details,sizeof(details));return 0;
}
namespace {
size_t portable_llama_bytes(const ResidentLlama *r,unsigned length) {
    if(!r || !r->cfg.layers || !length || length>r->cfg.capacity)return 0;
    size_t stride=size_t(r->cfg.kv_heads)*r->cfg.head_dim;
    if(!stride || stride>SIZE_MAX/length || stride*length>SIZE_MAX/r->cfg.layers
       || stride*length*r->cfg.layers>SIZE_MAX/(2*sizeof(float)))return 0;
    return stride*length*r->cfg.layers*2*sizeof(float);
}
bool portable_finite(const float *host,size_t bytes) {
    if(!host || bytes%sizeof(float))return false;
    for(size_t i=0;i<bytes/sizeof(float);i++)if(!isfinite(host[i]))return false;
    return true;
}
int portable_llama_transfer(ResidentLlama *r,float *host,unsigned length,bool importing) {
    const size_t stride=size_t(r->cfg.kv_heads)*r->cfg.head_dim;
    const size_t span=stride*length,plane=span*r->cfg.layers;
    NativeCallCompletion completion(r->stream);
    cudaMemcpyKind direction=importing?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost;
    for(unsigned which=0;which<2;which++)for(unsigned layer=0;layer<r->cfg.layers;layer++) {
        float *cpu=host+which*plane+layer*span;
        if(r->paged) {
            for(unsigned p=0;p<length;) {
                unsigned page=p/llama_page_tokens,n=std::min(length-p,llama_page_tokens-p%llama_page_tokens);
                if(page>=r->paged->active.size() || !r->paged->active[page])return completion.complete(1,3);
                auto &physical=*r->paged->active[page];
                float *gpu=(which?physical.v:physical.k)+(size_t(layer)*llama_page_tokens+p%llama_page_tokens)*stride;
                if(cudaMemcpyAsync(importing?gpu:cpu+p*stride,importing?cpu+p*stride:gpu,size_t(n)*stride*sizeof(float),direction,r->stream)!=cudaSuccess)
                    return completion.complete(1,3);
                p+=n;
            }
        } else {
            float *gpu=(which?r->kv_v:r->kv_k)+size_t(layer)*r->cfg.capacity*stride;
            if(cudaMemcpyAsync(importing?gpu:cpu,importing?cpu:gpu,span*sizeof(float),direction,r->stream)!=cudaSuccess)
                return completion.complete(1,3);
        }
    }
    return completion.complete(0,3);
}
size_t portable_qwen_bytes(const ResidentQwenFull *r,unsigned length) {
    if(!r || !length || length>r->capacity)return 0;
    size_t bytes=0;
    for(const auto &layer:r->layers) {
        size_t elements=0;
        if(layer.kind==0) {
            const auto &c=static_cast<ResidentQwenRecurrent*>(layer.context)->cfg;
            elements=size_t(c.num_v)*c.head*c.head+size_t(2*c.num_k+c.num_v)*c.head*c.conv;
        } else {
            const auto &c=static_cast<ResidentQwenAttention*>(layer.context)->cfg;
            elements=2*size_t(length)*c.kv_heads*c.head_dim;
        }
        if(elements>SIZE_MAX/sizeof(float) || bytes>SIZE_MAX-elements*sizeof(float))return 0;
        bytes+=elements*sizeof(float);
    }
    return bytes;
}
int portable_qwen_transfer(ResidentQwenFull *r,float *host,unsigned length,bool importing) {
    NativeCallCompletion completion(r->stream);
    cudaMemcpyKind direction=importing?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost;
    for(const auto &layer:r->layers) {
        float *first=nullptr,*second=nullptr;size_t a=0,b=0;
        if(layer.kind==0) {
            auto *block=static_cast<ResidentQwenRecurrent*>(layer.context);const auto &c=block->cfg;
            first=block->state;second=block->history;
            a=size_t(c.num_v)*c.head*c.head;b=size_t(2*c.num_k+c.num_v)*c.head*c.conv;
        } else {
            auto *block=static_cast<ResidentQwenAttention*>(layer.context);const auto &c=block->cfg;
            first=block->kv_k;second=block->kv_v;a=b=size_t(length)*c.kv_heads*c.head_dim;
        }
        if(cudaMemcpyAsync(importing?first:host,importing?host:first,a*sizeof(float),direction,r->stream)!=cudaSuccess
           || cudaMemcpyAsync(importing?second:host+a,importing?host+a:second,b*sizeof(float),direction,r->stream)!=cudaSuccess)
            return completion.complete(1,3);
        host+=a+b;
    }
    return completion.complete(0,3);
}
}
extern "C" RBITNET_CUDA_API size_t rbitnet_cuda_llama_portable_bytes(void *context,unsigned length) {
    return portable_llama_bytes(static_cast<ResidentLlama*>(context),length);
}
extern "C" RBITNET_CUDA_API int rbitnet_cuda_llama_portable_export(void *context,unsigned length,float *host,size_t bytes) {
    auto *r=static_cast<ResidentLlama*>(context);size_t expected=portable_llama_bytes(r,length);
    if(!expected || !host || bytes!=expected || length>r->filled)return 1;
    if(r->paged && !llama_paged_variant(r))return 2;
    return portable_llama_transfer(r,host,length,false);
}
extern "C" RBITNET_CUDA_API int rbitnet_cuda_llama_portable_import(void *context,unsigned length,const float *host,size_t bytes) {
    auto *r=static_cast<ResidentLlama*>(context);size_t expected=portable_llama_bytes(r,length);
    if(!expected || bytes!=expected || !portable_finite(host,bytes))return 1;
    if(r->paged && !llama_paged_variant(r))return 2;
    // Imports replace the logical state. A refused/failed copy leaves length
    // zero, so continuation cannot read a partially overwritten checkpoint.
    r->filled=0;
    if(r->paged && (!r->paged->reset(r->stream) || !r->paged->prepare(0,length,r->stream)))return 3;
    int status=portable_llama_transfer(r,const_cast<float*>(host),length,true);
    if(!status)r->filled=length;return status;
}
extern "C" RBITNET_CUDA_API size_t rbitnet_cuda_qwen_portable_bytes(void *context,unsigned length) {
    return portable_qwen_bytes(static_cast<ResidentQwenFull*>(context),length);
}
extern "C" RBITNET_CUDA_API int rbitnet_cuda_qwen_portable_export(void *context,unsigned length,float *host,size_t bytes) {
    auto *r=static_cast<ResidentQwenFull*>(context);size_t expected=portable_qwen_bytes(r,length);
    // GDN/convolution is an exact checkpoint: no truncation to an earlier token.
    if(!expected || !host || bytes!=expected || length!=r->filled)return 1;
    for(const auto &layer:r->layers) {
        unsigned filled=layer.kind==0?static_cast<ResidentQwenRecurrent*>(layer.context)->filled:static_cast<ResidentQwenAttention*>(layer.context)->filled;
        if(filled!=length)return 2;
    }
    return portable_qwen_transfer(r,host,length,false);
}
extern "C" RBITNET_CUDA_API int rbitnet_cuda_qwen_portable_import(void *context,unsigned length,const float *host,size_t bytes) {
    auto *r=static_cast<ResidentQwenFull*>(context);size_t expected=portable_qwen_bytes(r,length);
    if(!expected || bytes!=expected || !portable_finite(host,bytes))return 1;
    r->advance(0);
    int status=portable_qwen_transfer(r,const_cast<float*>(host),length,true);
    if(!status)r->advance(length);return status;
}
