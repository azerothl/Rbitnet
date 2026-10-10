// SPDX-License-Identifier: MIT
// One dense Qwen3 prefill chunk: every layer stays on the device stream.
#include <cmath>
#include <cstdio>
#include <mutex>
#include <vector>
namespace {
struct Qwen3ChunkWs {
    cudaStream_t stream=nullptr;
    float *x=nullptr,*h=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*attn=nullptr,*proj=nullptr;
    float *gate=nullptr,*up=nullptr,*scratch=nullptr,*freq=nullptr;
    float *attn_norm=nullptr,*q_norm=nullptr,*k_norm=nullptr,*ffn_norm=nullptr;
    unsigned *pos=nullptr;
    size_t cx=0,ch=0,cproj=0,cq=0,cattn=0,ck=0,cv=0,cg=0,cu=0,cs=0,cfreq=0,can=0,cqn=0,ckn=0,cfn=0;
    struct Plane {float *k=nullptr,*v=nullptr;size_t cap=0;unsigned filled=0;};
    std::vector<Plane> kv;
    bool grow(float *&p,size_t &cap,size_t n) {
        if(cap>=n && p) return true;
        if(p) cudaFree(p);
        p=nullptr;cap=0;
        if(!n || cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(float))!=cudaSuccess) return false;
        cap=n;return true;
    }
};
Qwen3ChunkWs &qwen3_chunk_ws() {static Qwen3ChunkWs ws;return ws;}
void qwen3_chunk_gemm(unsigned type,const void *w,size_t row_bytes,const float *x,unsigned cols,unsigned rows,
    unsigned tokens,float *y,cudaStream_t stream) {
    QuantKind kind;resident_kind(type,kind);
    launch_prefill_tf32(kind,w,row_bytes,x,cols,rows,tokens,y,stream);
}
}
extern "C" RBITNET_CUDA_API int rbitnet_cuda_qwen3_chunk(float *xs,RbitnetQwen3Layer *layers,unsigned n_layers,
    unsigned n_tokens,unsigned base_pos,unsigned hidden,unsigned n_head,unsigned n_kv_head,
    unsigned head_dim,unsigned n_ff,float eps,float rope_theta) {
    unsigned end=base_pos+n_tokens,stride=n_kv_head*head_dim,n_q=n_head*head_dim,half=head_dim/2;
    unsigned parts=(end+255)/256;
    size_t scratch_n=size_t(n_tokens)*n_head*parts*(size_t(head_dim)+2);
    if(!xs || !layers || !n_layers || n_layers>128 || !n_tokens || n_tokens>2048 || end<n_tokens || end>8192
        || !hidden || !n_head || !n_kv_head || n_head%n_kv_head || !head_dim || head_dim>256 || (head_dim&1)
        || !n_ff || scratch_n>(size_t(256)<<20)) return 1;
    for(unsigned i=0;i<n_layers;i++) {
        auto &l=layers[i];
        QuantKind kind;
        if(!l.q || !l.k || !l.v || !l.o || !l.gate || !l.up || !l.down || !l.attn_norm || !l.q_norm
            || !l.k_norm || !l.ffn_norm || !l.k_cache || !l.v_cache
            || !resident_kind(l.q_type,kind) || !resident_kind(l.k_type,kind) || !resident_kind(l.v_type,kind)
            || !resident_kind(l.o_type,kind) || !resident_kind(l.gate_type,kind) || !resident_kind(l.up_type,kind)
            || !resident_kind(l.down_type,kind)) return 1;
    }
    static std::mutex guard;
    std::lock_guard<std::mutex> lock(guard);
    auto &ws=qwen3_chunk_ws();
    if(!ws.stream && cudaStreamCreate(&ws.stream)!=cudaSuccess) return 2;
    size_t xn=size_t(n_tokens)*hidden,qn=size_t(n_tokens)*n_q,kn=size_t(n_tokens)*stride,fn=size_t(n_tokens)*n_ff;
    if(!ws.grow(ws.x,ws.cx,xn) || !ws.grow(ws.h,ws.ch,xn) || !ws.grow(ws.proj,ws.cproj,xn)
        || !ws.grow(ws.q,ws.cq,qn) || !ws.grow(ws.attn,ws.cattn,qn)
        || !ws.grow(ws.k,ws.ck,kn) || !ws.grow(ws.v,ws.cv,kn)
        || !ws.grow(ws.gate,ws.cg,fn) || !ws.grow(ws.up,ws.cu,fn)
        || !ws.grow(ws.scratch,ws.cs,scratch_n) || !ws.grow(ws.freq,ws.cfreq,half)
        || !ws.grow(ws.attn_norm,ws.can,hidden) || !ws.grow(ws.q_norm,ws.cqn,head_dim)
        || !ws.grow(ws.k_norm,ws.ckn,head_dim) || !ws.grow(ws.ffn_norm,ws.cfn,hidden)) return 2;
    if(!ws.pos && cudaMalloc(reinterpret_cast<void**>(&ws.pos),sizeof(unsigned))!=cudaSuccess) return 2;
    if(ws.kv.size()<n_layers) ws.kv.resize(n_layers);
    size_t kv_n=size_t(end)*stride;
    for(auto &plane:ws.kv) if(plane.cap<kv_n) {
        if(plane.k) cudaFree(plane.k);
        if(plane.v) cudaFree(plane.v);
        plane.k=plane.v=nullptr;plane.cap=0;plane.filled=0;
        if(cudaMalloc(reinterpret_cast<void**>(&plane.k),kv_n*sizeof(float))!=cudaSuccess
            || cudaMalloc(reinterpret_cast<void**>(&plane.v),kv_n*sizeof(float))!=cudaSuccess) return 2;
        plane.cap=kv_n;
    }
    std::vector<float> freq(half);
    for(unsigned i=0;i<half;i++) freq[i]=1.0f/powf(rope_theta,2.0f*i/head_dim);
    cudaStream_t stream=ws.stream;
    if(cudaMemcpyAsync(ws.x,xs,xn*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess
        || cudaMemcpyAsync(ws.freq,freq.data(),half*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess
        || cudaMemcpyAsync(ws.pos,&base_pos,sizeof(unsigned),cudaMemcpyHostToDevice,stream)!=cudaSuccess) return 3;
    float scale=1.0f/sqrtf(float(head_dim));
    for(unsigned il=0;il<n_layers;il++) {
        auto &l=layers[il];auto &plane=ws.kv[il];
        if(base_pos==0) plane.filled=0;
        if(plane.filled<base_pos) {
            size_t prefix=size_t(base_pos)*stride*sizeof(float);
            if(cudaMemcpyAsync(plane.k,l.k_cache,prefix,cudaMemcpyHostToDevice,stream)!=cudaSuccess
                || cudaMemcpyAsync(plane.v,l.v_cache,prefix,cudaMemcpyHostToDevice,stream)!=cudaSuccess) return 3;
        }
        if(cudaMemcpyAsync(ws.attn_norm,l.attn_norm,hidden*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess
            || cudaMemcpyAsync(ws.q_norm,l.q_norm,head_dim*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess
            || cudaMemcpyAsync(ws.k_norm,l.k_norm,head_dim*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess
            || cudaMemcpyAsync(ws.ffn_norm,l.ffn_norm,hidden*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess) return 3;
        resident_norm<<<n_tokens,256,0,stream>>>(ws.x,ws.attn_norm,eps,hidden,ws.h);
        qwen3_chunk_gemm(l.q_type,l.q,l.q_bytes,ws.h,hidden,n_q,n_tokens,ws.q,stream);
        qwen3_chunk_gemm(l.k_type,l.k,l.k_bytes,ws.h,hidden,stride,n_tokens,ws.k,stream);
        qwen3_chunk_gemm(l.v_type,l.v,l.v_bytes,ws.h,hidden,stride,n_tokens,ws.v,stream);
        qwen_head_norm<<<dim3(n_head,n_tokens),128,0,stream>>>(ws.q,ws.q_norm,head_dim,head_dim,eps,ws.q,n_head);
        qwen_head_norm<<<dim3(n_kv_head,n_tokens),128,0,stream>>>(ws.k,ws.k_norm,head_dim,head_dim,eps,ws.k,n_kv_head);
        qwen_neox_rope<<<dim3(((n_head+n_kv_head)*half+255)/256,n_tokens),256,0,stream>>>(ws.q,ws.k,ws.freq,ws.pos,n_head,n_kv_head,head_dim,head_dim);
        qwen_write_kv<<<dim3((stride+255)/256,n_tokens),256,0,stream>>>(ws.k,ws.v,plane.k,plane.v,ws.pos,stride);
        launch_split_attention(plane.k,plane.v,ws.q,ws.pos,n_kv_head,n_head,head_dim,0,scale,end,n_tokens,ws.scratch,ws.attn,stream);
        qwen3_chunk_gemm(l.o_type,l.o,l.o_bytes,ws.attn,n_q,hidden,n_tokens,ws.proj,stream);
        resident_norm<<<n_tokens,256,0,stream>>>(ws.x,ws.ffn_norm,eps,hidden,ws.h,ws.proj);
        qwen3_chunk_gemm(l.gate_type,l.gate,l.gate_bytes,ws.h,hidden,n_ff,n_tokens,ws.gate,stream);
        qwen3_chunk_gemm(l.up_type,l.up,l.up_bytes,ws.h,hidden,n_ff,n_tokens,ws.up,stream);
        resident_silu<<<(fn+255)/256,256,0,stream>>>(ws.gate,ws.up,unsigned(fn));
        qwen3_chunk_gemm(l.down_type,l.down,l.down_bytes,ws.gate,n_ff,hidden,n_tokens,ws.proj,stream);
        resident_add<<<(xn+255)/256,256,0,stream>>>(ws.x,ws.proj,unsigned(xn));
        plane.filled=end;
    }
    if(cudaGetLastError()!=cudaSuccess) return 4;
    if(cudaMemcpyAsync(xs,ws.x,xn*sizeof(float),cudaMemcpyDeviceToHost,stream)!=cudaSuccess) return 3;
    for(unsigned il=0;il<n_layers;il++) {
        auto &l=layers[il];auto &plane=ws.kv[il];
        size_t off=size_t(base_pos)*stride,bytes=size_t(n_tokens)*stride*sizeof(float);
        if(cudaMemcpyAsync(l.k_cache+off,plane.k+off,bytes,cudaMemcpyDeviceToHost,stream)!=cudaSuccess
            || cudaMemcpyAsync(l.v_cache+off,plane.v+off,bytes,cudaMemcpyDeviceToHost,stream)!=cudaSuccess) return 3;
    }
    if(cudaStreamSynchronize(stream)!=cudaSuccess || cudaGetLastError()!=cudaSuccess) return 4;
    static int once=0;
    if(!once) {std::fprintf(stderr,"qwen3 device chunk tokens=%u end=%u layers=%u\n",n_tokens,end,n_layers);once=1;}
    return 0;
}
