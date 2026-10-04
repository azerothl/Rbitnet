// SPDX-License-Identifier: MIT
// Private test-only ABI for the actual encoded-cache attention kernels.
namespace {
struct EncodedAttentionFixture {
    cudaStream_t stream=nullptr;std::vector<void*> allocations;
    ~EncodedAttentionFixture() {if(stream)cudaStreamSynchronize(stream);for(auto p:allocations)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<typename T> bool allocate(T *&p,size_t count) {
        if(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(p);return true;
    }
};
template<unsigned Format>
__global__ void encoded_attention_fixture_decode(EncodedKvView<Format,false> view,float *k,float *v) {
    size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at<view.elements()) {
        unsigned token=at/(view.heads*view.dim),head=(at/view.dim)%view.heads,i=at%view.dim;
        k[at]=view.read(false,token,head,i);v[at]=view.read(true,token,head,i);
    }
}
template<unsigned Format>
void enqueue_encoded_attention_fixture(EncodedKvView<Format,false> view,const float *k,const float *v,const float *q,
    unsigned *position,unsigned *invalid,unsigned heads,unsigned window,bool split,float *scratch,float *out,
    float *decoded_k,float *decoded_v,cudaStream_t stream,unsigned *last) {
    cudaMemsetAsync(decoded_k,0,size_t(view.capacity)*view.heads*view.dim*sizeof(float),stream);
    encoded_kv_fused_rope_store<<<dim3(view.heads,view.capacity),128,0,stream>>>(decoded_k,const_cast<float*>(k),v,view,nullptr,position,view.heads,0,invalid);
    size_t elements=size_t(view.capacity)*view.heads*view.dim;
    encoded_attention_fixture_decode<<<(elements+255)/256,256,0,stream>>>(view,decoded_k,decoded_v);
    cudaMemcpyAsync(position,last,sizeof(unsigned),cudaMemcpyHostToDevice,stream);
    float scale=1.0f/sqrtf(float(view.dim));
    if(split) {
        unsigned parts=(view.capacity+attention_tile-1)/attention_tile;
        encoded_attention_partials<<<dim3(heads,parts,1),128,0,stream>>>(view,q,position,view.heads,heads,view.dim,window,scale,parts,scratch);
        attention_merge<<<heads,128,parts*sizeof(float),stream>>>(scratch,heads,view.dim,parts,out);
    } else encoded_resident_attention<<<heads,128,view.capacity*sizeof(float),stream>>>(view,q,position,view.heads,heads,view.dim,window,scale,out);
}
}
extern "C" RBITNET_CUDA_API int rbitnet_cuda_encoded_attention_oracle(unsigned format,unsigned seq,unsigned heads,unsigned kv_heads,unsigned dim,unsigned window,unsigned split,
    const float *q,const float *k,const float *v,float *out,float *decoded_k,float *decoded_v,float *scales_k,float *scales_v,size_t *bytes) {
    if(format<1||format>2||!seq||seq>2048||!heads||heads>32||!kv_heads||heads%kv_heads||dim%32||!dim||dim>256||split>1
        ||!q||!k||!v||!out||!decoded_k||!decoded_v||!scales_k||!scales_v||!bytes)return 1;
    EncodedAttentionFixture f;MemoryCategoryScope category(MemoryScratch);
    float *dq=nullptr,*dk=nullptr,*dv=nullptr,*pk=nullptr,*pv=nullptr,*dout=nullptr,*decode_k=nullptr,*decode_v=nullptr,*scratch=nullptr;
    unsigned *position=nullptr,*invalid=nullptr;size_t elements=size_t(seq)*kv_heads*dim,plane=encoded_kv_plane_bytes(format,elements,dim);
    unsigned parts=(seq+attention_tile-1)/attention_tile;
    if(cudaStreamCreateWithFlags(&f.stream,cudaStreamNonBlocking)!=cudaSuccess
        ||!f.allocate(dq,size_t(heads)*dim)||!f.allocate(dk,elements)||!f.allocate(dv,elements)
        ||!f.allocate(pk,plane/4)||!f.allocate(pv,plane/4)||!f.allocate(dout,size_t(heads)*dim)
        ||!f.allocate(decode_k,elements)||!f.allocate(decode_v,elements)||!f.allocate(scratch,size_t(heads)*parts*(dim+2))
        ||!f.allocate(position,1)||!f.allocate(invalid,1))return 2;
    NativeCallCompletion completion(f.stream);unsigned zero=0,last=seq-1,host_invalid=0;
    if(cudaMemcpyAsync(dq,q,size_t(heads)*dim*4,cudaMemcpyHostToDevice,f.stream)!=cudaSuccess
        ||cudaMemcpyAsync(dk,k,elements*4,cudaMemcpyHostToDevice,f.stream)!=cudaSuccess
        ||cudaMemcpyAsync(dv,v,elements*4,cudaMemcpyHostToDevice,f.stream)!=cudaSuccess
        ||cudaMemcpyAsync(position,&zero,sizeof(unsigned),cudaMemcpyHostToDevice,f.stream)!=cudaSuccess
        ||cudaMemsetAsync(invalid,0,sizeof(unsigned),f.stream)!=cudaSuccess)return completion.complete(3,3);
    if(format==1)enqueue_encoded_attention_fixture(EncodedKvView<1,false>{pk,pv,nullptr,nullptr,0,1,seq,kv_heads,dim},dk,dv,dq,position,invalid,heads,window,split,scratch,dout,decode_k,decode_v,f.stream,&last);
    else enqueue_encoded_attention_fixture(EncodedKvView<2,false>{pk,pv,nullptr,nullptr,0,1,seq,kv_heads,dim},dk,dv,dq,position,invalid,heads,window,split,scratch,dout,decode_k,decode_v,f.stream,&last);
    int status=cudaGetLastError()!=cudaSuccess
        ||cudaMemcpyAsync(out,dout,size_t(heads)*dim*4,cudaMemcpyDeviceToHost,f.stream)!=cudaSuccess
        ||cudaMemcpyAsync(decoded_k,decode_k,elements*4,cudaMemcpyDeviceToHost,f.stream)!=cudaSuccess
        ||cudaMemcpyAsync(decoded_v,decode_v,elements*4,cudaMemcpyDeviceToHost,f.stream)!=cudaSuccess
        ||cudaMemcpyAsync(&host_invalid,invalid,sizeof(unsigned),cudaMemcpyDeviceToHost,f.stream)!=cudaSuccess;
    if(format==2)status=status
        ||cudaMemcpyAsync(scales_k,reinterpret_cast<char*>(pk)+elements,size_t(seq)*kv_heads*4,cudaMemcpyDeviceToHost,f.stream)!=cudaSuccess
        ||cudaMemcpyAsync(scales_v,reinterpret_cast<char*>(pv)+elements,size_t(seq)*kv_heads*4,cudaMemcpyDeviceToHost,f.stream)!=cudaSuccess;
    status=completion.complete(status,4);if(status)return status;*bytes=plane*2;
    return host_invalid?5:0;
}
