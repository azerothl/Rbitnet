// SPDX-License-Identifier: MIT
// Private first vertical. The Rust session exclusively borrows all contexts.
#include <array>
#include <cmath>
namespace {
struct BatchLlamaView {
    float *k,*v;float **pages_k,**pages_v;
    unsigned position,capacity;
};
template<bool Paged>
__global__ void batch_llama_rope_kv(float *q,float *k,const float *v,const BatchLlamaView *views,unsigned layer,
    const float *frequency,unsigned heads,unsigned kv_heads,unsigned dim,unsigned rotary) {
    const auto view=views[blockIdx.y];
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,pos=view.position,pairs=dim/2;
    q+=size_t(blockIdx.y)*heads*dim;k+=size_t(blockIdx.y)*kv_heads*dim;v+=size_t(blockIdx.y)*kv_heads*dim;
    if(i<(heads+kv_heads)*pairs) {
        unsigned head=i/pairs,j=i%pairs;
        float *src=head<heads?q+head*dim:k+(head-heads)*dim;
        float a=src[2*j],b=src[2*j+1];
        if(2*j<rotary) {float angle=pos*frequency[j],s=sinf(angle),c=cosf(angle);src[2*j]=a*c-b*s;src[2*j+1]=a*s+b*c;}
        if(head>=heads) {
            float *cache_k,*cache_v;size_t offset;
            if constexpr(Paged) {
                cache_k=view.pages_k[pos/llama_page_tokens];cache_v=view.pages_v[pos/llama_page_tokens];
                offset=(size_t(layer)*llama_page_tokens+pos%llama_page_tokens)*kv_heads*dim+(head-heads)*dim+2*j;
            } else {
                cache_k=view.k;cache_v=view.v;
                offset=(size_t(layer)*view.capacity+pos)*kv_heads*dim+(head-heads)*dim+2*j;
            }
            cache_k[offset]=src[2*j];cache_k[offset+1]=src[2*j+1];
            cache_v[offset]=v[(head-heads)*dim+2*j];cache_v[offset+1]=v[(head-heads)*dim+2*j+1];
        }
    }
}
template<bool Paged>
__global__ void batch_llama_attention(const BatchLlamaView *views,unsigned layer,const float *q,
    unsigned kv_heads,unsigned heads,unsigned dim,unsigned window,float scale,float *y) {
    const auto view=views[blockIdx.y];
    const float *k=view.k;
    if constexpr(!Paged)k+=size_t(layer)*view.capacity*kv_heads*dim;
    const float *v=view.v;
    if constexpr(!Paged)v+=size_t(layer)*view.capacity*kv_heads*dim;
    extern __shared__ float scores[];
    __shared__ float reductions[4];
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid/32;
    const unsigned head=blockIdx.x,kh=head/(heads/kv_heads),seq=view.position+1;
    q+=size_t(blockIdx.y)*heads*dim;y+=size_t(blockIdx.y)*heads*dim;
    const unsigned first=window && seq>window?seq-window:0;
    for(unsigned p=first+warp;p<seq;p+=4) {
        float s=0;
        for(unsigned i=lane;i<dim;i+=32)s=fmaf(q[head*dim+i],(Paged?view.pages_k[p/llama_page_tokens][(size_t(layer)*llama_page_tokens+p%llama_page_tokens)*kv_heads*dim+kh*dim+i]:k[size_t(p)*kv_heads*dim+kh*dim+i]),s);
        for(int shift=16;shift;shift/=2)s+=__shfl_down_sync(0xffffffff,s,shift);
        if(lane==0)scores[p-first]=s*scale;
    }
    __syncthreads();
    float maximum=-CUDART_INF_F;
    for(unsigned i=tid;i<seq-first;i+=128)maximum=fmaxf(maximum,scores[i]);
    for(int shift=16;shift;shift/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,shift));
    if(lane==0)reductions[warp]=maximum;
    __syncthreads();maximum=fmaxf(fmaxf(reductions[0],reductions[1]),fmaxf(reductions[2],reductions[3]));
    __syncthreads();
    float sum=0;
    for(unsigned i=tid;i<seq-first;i+=128) {scores[i]=expf(scores[i]-maximum);sum+=scores[i];}
    for(int shift=16;shift;shift/=2)sum+=__shfl_down_sync(0xffffffff,sum,shift);
    if(lane==0)reductions[warp]=sum;
    __syncthreads();sum=reductions[0]+reductions[1]+reductions[2]+reductions[3];
    for(unsigned i=tid;i<dim;i+=128) {
        float output=0;
        for(unsigned p=first;p<seq;p++)output=fmaf(scores[p-first]/sum,(Paged?view.pages_v[p/llama_page_tokens][(size_t(layer)*llama_page_tokens+p%llama_page_tokens)*kv_heads*dim+kh*dim+i]:v[size_t(p)*kv_heads*dim+kh*dim+i]),output);
        y[head*dim+i]=output;
    }
}


template<QuantKind kind>
__global__ void batch_row_interleaved_gemm(const uint8_t *w,size_t row_bytes,const float *x,
    unsigned cols,unsigned rows,unsigned tokens,float *y) {
    unsigned row=(blockIdx.x/tokens)*8+threadIdx.x/32,token=blockIdx.x%tokens;
    if(row>=rows)return;
    float value=quant_row_dot<kind>(w+size_t(row)*row_bytes,x+size_t(token)*cols,cols);
    if(!(threadIdx.x&31))y[size_t(token)*rows+row]=value;
}
struct BatchLlama {
    static constexpr unsigned limit=8;
    ResidentLlama buffers;
    RbitnetLlamaConfig cfg;
    std::vector<unsigned char> model_key;
    BatchLlamaView *views=nullptr;
    const unsigned maximum;
    unsigned ordering;
    uint64_t waves=0,rows=0,shared_gemms=0;
    bool paged;
    explicit BatchLlama(unsigned n):maximum(n) {}
    bool init(const ResidentLlama &owner,unsigned mode) {
        MemoryCategoryScope category(MemoryScratch);
        cfg=owner.cfg;model_key=owner.batch_model_key;ordering=mode;paged=bool(owner.paged);
        if(cudaStreamCreateWithFlags(&buffers.stream,cudaStreamNonBlocking)!=cudaSuccess)return false;
        const size_t n=maximum,stride=size_t(cfg.kv_heads)*cfg.head_dim;
        return buffers.alloc(buffers.x,n*cfg.embd)&&buffers.alloc(buffers.h,n*cfg.embd)
            &&buffers.alloc(buffers.q,n*cfg.embd)&&buffers.alloc(buffers.k,n*stride)&&buffers.alloc(buffers.v,n*stride)
            &&buffers.alloc(buffers.attn,n*cfg.embd)&&buffers.alloc(buffers.projection,n*cfg.embd)
            &&buffers.alloc(buffers.gate,n*cfg.ffn)&&buffers.alloc(buffers.up,n*cfg.ffn)
            &&buffers.alloc(buffers.logits,n*cfg.vocab)&&buffers.alloc(buffers.maxima,n*((cfg.vocab+255)/256))
            &&buffers.alloc(buffers.ids,n*((cfg.vocab+255)/256))&&buffers.alloc(buffers.maximum,n)&&buffers.alloc(buffers.token,n)
            &&buffers.alloc(views,n);
    }
    void matrix(const RbitnetLlamaMatrix &m,const float *x,float *y,unsigned n) {
        QuantKind kind;resident_kind(m.type,kind);
        if(!ordering) {launch_ordered_gemm(kind,static_cast<const uint8_t*>(m.weights),m.row_bytes,x,m.cols,m.rows,n,y,buffers.stream,0);return;}
        dim3 grid(((m.rows+7)/8)*n);
#define BATCH_KIND(K) case QuantKind::K: batch_row_interleaved_gemm<QuantKind::K><<<grid,256,0,buffers.stream>>>(static_cast<const uint8_t*>(m.weights),m.row_bytes,x,m.cols,m.rows,n,y);break
        switch(kind) {BATCH_KIND(F32);BATCH_KIND(Q4_0);BATCH_KIND(Q5_0);BATCH_KIND(Q8_0);BATCH_KIND(Q4_K);BATCH_KIND(Q5_K);BATCH_KIND(Q6_K);BATCH_KIND(MXFP4);}
#undef BATCH_KIND
    }
    void enqueue(const ResidentLlama &owner,unsigned n,unsigned capacity,unsigned mode) {
        auto &b=buffers;auto stream=b.stream;auto &c=cfg;
        resident_norm<<<n,256,0,stream>>>(b.x,owner.layers[0].attn_norm,c.epsilon,c.embd,b.h);
        for(unsigned il=0;il<c.layers;il++) {
            const auto &l=owner.layers[il];
            matrix(l.q,b.h,b.q,n);matrix(l.k,b.h,b.k,n);matrix(l.v,b.h,b.v,n);
            dim3 rope(((c.heads+c.kv_heads)*(c.head_dim/2)+255)/256,n);
            if(paged) {
                batch_llama_rope_kv<true><<<rope,256,0,stream>>>(b.q,b.k,b.v,views,il,owner.frequency,c.heads,c.kv_heads,c.head_dim,c.rotary);
                batch_llama_attention<true><<<dim3(c.heads,n),128,capacity*sizeof(float),stream>>>(views,il,b.q,c.kv_heads,c.heads,c.head_dim,c.window,1.0f/sqrtf(float(c.head_dim)),b.attn);
            } else {
                batch_llama_rope_kv<false><<<rope,256,0,stream>>>(b.q,b.k,b.v,views,il,owner.frequency,c.heads,c.kv_heads,c.head_dim,c.rotary);
                batch_llama_attention<false><<<dim3(c.heads,n),128,capacity*sizeof(float),stream>>>(views,il,b.q,c.kv_heads,c.heads,c.head_dim,c.window,1.0f/sqrtf(float(c.head_dim)),b.attn);
            }
            matrix(l.out,b.attn,b.projection,n);
            resident_norm<<<n,256,0,stream>>>(b.x,l.ffn_norm,c.epsilon,c.embd,b.h,b.projection);
            matrix(l.gate,b.h,b.gate,n);matrix(l.up,b.h,b.up,n);
            resident_silu<<<(n*c.ffn+255)/256,256,0,stream>>>(b.gate,b.up,n*c.ffn);
            matrix(l.down,b.gate,b.projection,n);
            if(il+1<c.layers)resident_norm<<<n,256,0,stream>>>(b.x,owner.layers[il+1].attn_norm,c.epsilon,c.embd,b.h,b.projection);
            else if(mode)resident_norm<<<n,256,0,stream>>>(b.x,owner.out_norm,c.epsilon,c.embd,b.h,b.projection);
            else resident_add<<<(n*c.embd+255)/256,256,0,stream>>>(b.x,b.projection,n*c.embd);
        }
        if(mode)matrix(owner.output,b.h,b.logits,n);
        if(mode==2) {
            unsigned blocks=(c.vocab+255)/256;
            resident_argmax<<<dim3(blocks,n),256,0,stream>>>(b.logits,nullptr,c.vocab,b.maxima,b.ids);
            resident_argmax<<<dim3(1,n),256,0,stream>>>(b.maxima,b.ids,blocks,b.maximum,b.token);
        }
    }
};
struct BatchLlamaCompletion {
    BatchLlama &batch;const std::array<ResidentLlama*,8> &contexts;unsigned count;bool done=false;
    ~BatchLlamaCompletion() {if(!done) {cudaStreamSynchronize(batch.buffers.stream);for(unsigned i=0;i<count;i++)contexts[i]->filled=0;}}
};
}
extern "C" {
RBITNET_CUDA_API void *rbitnet_cuda_llama_batch_create(const void *peer,unsigned maximum,unsigned ordering) {
    auto *r=static_cast<const ResidentLlama*>(peer);
    if(!r||!maximum||maximum>8||ordering>1||r->split_kv||r->tf32_prefill||r->batch_model_key.empty())return nullptr;
    auto *b=new(std::nothrow) BatchLlama(maximum);if(!b)return nullptr;
    try {if(!b->init(*r,ordering)) {delete b;return nullptr;}}catch(const std::bad_alloc&) {delete b;return nullptr;}
    return b;
}
RBITNET_CUDA_API void rbitnet_cuda_llama_batch_destroy(void *batch) {delete static_cast<BatchLlama*>(batch);}
RBITNET_CUDA_API int rbitnet_cuda_llama_batch_stats(const void *batch,uint64_t *values,unsigned count) {
    auto *b=static_cast<const BatchLlama*>(batch);if(!b||!values||count!=3)return 1;
    values[0]=b->waves;values[1]=b->rows;values[2]=b->shared_gemms;return 0;
}
RBITNET_CUDA_API int rbitnet_cuda_llama_batch_step(void *batch,void *const *contexts,const unsigned *positions,const float *embeddings,
    unsigned count,unsigned mode,float *logits,unsigned *tokens) {
    auto *b=static_cast<BatchLlama*>(batch);
    if(!b||!contexts||!positions||!embeddings||!count||count>b->maximum||mode>2||(mode==1&&!logits)||(mode==2&&!tokens))return 1;
    std::array<BatchLlamaView,8> views{};std::array<ResidentLlama*,8> owners{};
    unsigned capacity=0;
    for(unsigned i=0;i<count;i++) {
        auto *r=static_cast<ResidentLlama*>(contexts[i]);
        if(!r||r->split_kv||r->tf32_prefill||bool(r->paged)!=b->paged||r->batch_model_key!=b->model_key
            ||positions[i]>=r->cfg.capacity||(positions[i]&&positions[i]!=r->filled))return 3;
        for(unsigned j=0;j<i;j++)if(owners[j]==r)return 4;
        if(b->paged&&i&&r->paged->pool.get()!=owners[0]->paged->pool.get())return 5;
        owners[i]=r;capacity=std::max(capacity,r->cfg.capacity);
        for(unsigned j=0;j<r->cfg.embd;j++)if(!std::isfinite(embeddings[size_t(i)*r->cfg.embd+j]))return 6;
    }
    BatchLlamaCompletion completion{*b,owners,count};
    for(unsigned i=0;i<count;i++) {
        auto *r=owners[i];
        if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 7;
        if(!llama_paged_variant(r))return 8;
        if(!positions[i]) {if(r->paged&&!r->paged->reset(r->stream))return 9;r->filled=0;}
        if(r->paged&&!r->paged->prepare(positions[i],1,r->stream))return 10;
        // Admission/COW copies update this owner's device page tables on its
        // own stream. Publish those tables before the batch stream reads them.
        if(r->paged&&cudaStreamSynchronize(r->stream)!=cudaSuccess)return 10;
        views[i]={r->kv_k,r->kv_v,r->paged?r->paged->table_k:nullptr,r->paged?r->paged->table_v:nullptr,positions[i],r->cfg.capacity};
    }
    auto stream=b->buffers.stream;
    if(cudaMemcpyAsync(b->buffers.x,embeddings,size_t(count)*b->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess
        ||cudaMemcpyAsync(b->views,views.data(),size_t(count)*sizeof(BatchLlamaView),cudaMemcpyHostToDevice,stream)!=cudaSuccess)return 11;
    b->enqueue(*owners[0],count,capacity,mode);
    if(cudaGetLastError()!=cudaSuccess)return 12;
    if(mode==1&&cudaMemcpyAsync(logits,b->buffers.logits,size_t(count)*b->cfg.vocab*sizeof(float),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)return 13;
    if(mode==2&&cudaMemcpyAsync(tokens,b->buffers.token,size_t(count)*sizeof(unsigned),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)return 14;
    if(cudaStreamSynchronize(stream)!=cudaSuccess)return 15;
    for(unsigned i=0;i<count;i++)owners[i]->filled=positions[i]+1;
    b->waves++;b->rows+=count;b->shared_gemms+=7*b->cfg.layers+(mode?1:0);
    completion.done=true;return 0;
}
}
