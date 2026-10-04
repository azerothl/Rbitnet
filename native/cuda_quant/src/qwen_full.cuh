// SPDX-License-Identifier: MIT
// Dense Qwen: gated full attention plus all recurrent/attention blocks on one stream.
namespace {
__global__ void qwen_head_norm(const float *input,const float *weights,unsigned dim,unsigned stride,
    float epsilon,float *out,unsigned heads=0) {
    unsigned tid=threadIdx.x,lane=tid&31,warp=tid/32;
    input+=size_t(blockIdx.y)*heads*stride;out+=size_t(blockIdx.y)*heads*dim;
    const float *x=input+size_t(blockIdx.x)*stride;out+=size_t(blockIdx.x)*dim;
    __shared__ float sums[4];float sum=0;
    for(unsigned i=tid;i<dim;i+=128)sum+=x[i]*x[i];
    for(int s=16;s;s/=2)sum+=__shfl_down_sync(0xffffffff,sum,s);
    if(!lane)sums[warp]=sum;__syncthreads();
    float inv=1.0f/sqrtf((sums[0]+sums[1]+sums[2]+sums[3])/dim+epsilon);
    for(unsigned i=tid;i<dim;i+=128)out[i]=(x[i]*weights[i])*inv;
}
__global__ void qwen_neox_rope(float *q,float *k,const float *freq,const unsigned *position,
    unsigned heads,unsigned kv_heads,unsigned dim,unsigned rotary) {
    unsigned pair=blockIdx.x*blockDim.x+threadIdx.x,half=rotary/2;
    if(pair>=(heads+kv_heads)*half)return;
    unsigned h=pair/half,i=pair%half;
    q+=size_t(blockIdx.y)*heads*dim;k+=size_t(blockIdx.y)*kv_heads*dim;
    float *x=h<heads?q+size_t(h)*dim:k+size_t(h-heads)*dim;
    float angle=(*position+blockIdx.y)*freq[i],s=sinf(angle),c=cosf(angle),a=x[i],b=x[i+half];
    x[i]=a*c-b*s;x[i+half]=a*s+b*c;
}
__global__ void qwen_write_kv(const float *k,const float *v,float *cache_k,float *cache_v,
    const unsigned *position,unsigned stride) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<stride) {size_t p=size_t(*position+blockIdx.y)*stride+i;
        cache_k[p]=k[size_t(blockIdx.y)*stride+i];cache_v[p]=v[size_t(blockIdx.y)*stride+i];}
}
__global__ void qwen_attention_gate(float *attention,const float *q,unsigned heads,unsigned dim,unsigned gated) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    attention+=size_t(blockIdx.y)*heads*dim;q+=size_t(blockIdx.y)*heads*dim*(1+gated);
    if(i<heads*dim && gated) {unsigned h=i/dim,j=i%dim;attention[i]*=1.0f/(1.0f+expf(-q[size_t(h)*2*dim+dim+j]));}
}
struct ResidentQwenAttention {
    RbitnetQwenAttentionConfig cfg;RbitnetLlamaMatrix matrices[7];
    std::vector<void*> allocations;
    float *x=nullptr,*h=nullptr,*q_full=nullptr,*q=nullptr,*k_raw=nullptr,*k=nullptr,*v=nullptr;
    float *attn=nullptr,*projection=nullptr,*gate=nullptr,*up=nullptr,*kv_k=nullptr,*kv_v=nullptr;
    float *attn_norm=nullptr,*ffn_norm=nullptr,*q_norm=nullptr,*k_norm=nullptr,*frequency=nullptr;
    unsigned *position=nullptr,filled=0;
    float *attention_scratch=nullptr;
    bool split_kv=false;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    ~ResidentQwenAttention() {
        if(stream)cudaStreamSynchronize(stream);
        if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        for(auto p:allocations)cudaFree(p);if(stream)cudaStreamDestroy(stream);
    }
    template<typename T> bool alloc(T *&p,size_t n,const T *host=nullptr) {
        if(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(p);
        return !host || (cudaMemcpyAsync(p,host,n*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess && cudaStreamSynchronize(stream)==cudaSuccess);
    }
    template<typename T> bool alloc_kv(T *&p,size_t n) {
        MemoryCategoryScope category(MemoryKv);return alloc(p,n);
    }
    void matrix(unsigned i,const float *input,float *out) {
        const auto &m=matrices[i];QuantKind kind;resident_kind(m.type,kind);
        launch_quant_kernel(kind,m.weights,m.row_bytes,input,m.cols,m.rows,m.rows,out,stream);
    }
    void enqueue() {
        const auto &c=cfg;unsigned qs=c.heads*c.head_dim,ks=c.kv_heads*c.head_dim;
        resident_norm<<<1,256,0,stream>>>(x,attn_norm,c.epsilon,c.embd,h);
        matrix(0,h,q_full);matrix(1,h,k_raw);matrix(2,h,v);
        qwen_head_norm<<<c.heads,128,0,stream>>>(q_full,q_norm,c.head_dim,c.head_dim*(1+c.gated),c.epsilon,q);
        qwen_head_norm<<<c.kv_heads,128,0,stream>>>(k_raw,k_norm,c.head_dim,c.head_dim,c.epsilon,k);
        if(c.rotary)qwen_neox_rope<<<((c.heads+c.kv_heads)*(c.rotary/2)+255)/256,256,0,stream>>>(q,k,frequency,position,c.heads,c.kv_heads,c.head_dim,c.rotary);
        qwen_write_kv<<<(ks+255)/256,256,0,stream>>>(k,v,kv_k,kv_v,position,ks);
        if(split_kv)launch_split_attention(kv_k,kv_v,q,position,c.kv_heads,c.heads,c.head_dim,0,c.scale,c.capacity,1,attention_scratch,attn,stream);
        else resident_attention<<<c.heads,128,c.capacity*sizeof(float),stream>>>(kv_k,kv_v,q,position,c.kv_heads,c.heads,c.head_dim,0,c.scale,attn);
        qwen_attention_gate<<<(qs+255)/256,256,0,stream>>>(attn,q_full,c.heads,c.head_dim,c.gated);
        matrix(3,attn,projection);
        resident_norm<<<1,256,0,stream>>>(x,ffn_norm,c.epsilon,c.embd,h,projection);
        matrix(4,h,gate);matrix(5,h,up);
        resident_silu<<<(c.ffn+255)/256,256,0,stream>>>(gate,up,c.ffn);
        matrix(6,gate,projection);resident_add<<<(c.embd+255)/256,256,0,stream>>>(x,projection,c.embd);
    }
};
struct QwenAttentionSnapshot {
    unsigned length,heads,dim;float *k=nullptr,*v=nullptr;
    ~QwenAttentionSnapshot() {if(k)cudaFree(k);if(v)cudaFree(v);}
};
#include "qwen_prefill.cuh"
// Private bounded Qwen all-position verification and GDN-only rollback.
// Included inside the anonymous namespace immediately before ResidentQwenFull.
struct QwenSpeculativeWorkspace {
    static constexpr unsigned limit=9;
    float *h=nullptr,*logits=nullptr,*maxima=nullptr,*maximum=nullptr,*backup=nullptr;
    unsigned *ids=nullptr,*tokens=nullptr;
    struct Plane {float *source;size_t offset,elements;};
    std::vector<Plane> planes;
    std::vector<void*> allocations;
    size_t backup_elements=0;
    unsigned saved_length=0;
    unsigned q8_tile=0;
    uint64_t nonce=0;
    bool valid=false,poisoned=false;
    cudaGraph_t graphs[3][limit+1]={};cudaGraphExec_t executable[3][limit+1]={};
    ~QwenSpeculativeWorkspace() {
        for(auto &mode:executable)for(auto e:mode)if(e)cudaGraphExecDestroy(e);
        for(auto &mode:graphs)for(auto g:mode)if(g)cudaGraphDestroy(g);
        for(auto p:allocations)cudaFree(p);
    }
    template<typename T> bool alloc(T *&p,size_t elements) {
        if(!elements || elements>SIZE_MAX/sizeof(T))return false;
        if(cudaMalloc(reinterpret_cast<void**>(&p),elements*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(p);return true;
    }
    bool init(unsigned embd,unsigned vocab,const std::vector<RbitnetQwenFullLayer> &layers) {
        MemoryCategoryScope category(MemoryScratch);
        for(const auto &layer:layers)if(layer.kind==0) {
            auto *r=static_cast<ResidentQwenRecurrent*>(layer.context);const auto &c=r->cfg;
            size_t history=size_t(2*c.num_k+c.num_v)*c.head*c.conv,state=size_t(c.num_v)*c.head*c.head;
            if(history>SIZE_MAX-backup_elements)return false;
            planes.push_back({r->history,backup_elements,history});backup_elements+=history;
            if(state>SIZE_MAX-backup_elements)return false;
            planes.push_back({r->state,backup_elements,state});backup_elements+=state;
        }
        unsigned blocks=(vocab+255)/256;
        return alloc(h,size_t(limit)*embd) && alloc(logits,size_t(limit)*vocab)
            && alloc(maxima,size_t(limit)*blocks) && alloc(ids,size_t(limit)*blocks)
            && alloc(maximum,limit) && alloc(tokens,limit) && (!backup_elements || alloc(backup,backup_elements));
    }
    bool copy(cudaStream_t stream,bool restore) {
        for(const auto &plane:planes) {
            float *dst=restore?plane.source:backup+plane.offset;
            const float *src=restore?backup+plane.offset:plane.source;
            if(cudaMemcpyAsync(dst,src,plane.elements*sizeof(float),cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return false;
        }
        return true;
    }
    void invalidate() {valid=false;nonce=0;}
};

struct ResidentQwenFull {
    unsigned embd,vocab,capacity,filled=0;bool use_graphs;
    std::vector<RbitnetQwenFullLayer> layers;RbitnetLlamaMatrix head;
    std::vector<void*> allocations;
    float *x=nullptr,*h=nullptr,*norm=nullptr,*logits=nullptr,*maxima=nullptr,*maximum=nullptr;
    unsigned *position=nullptr,*ids=nullptr,*token=nullptr;
    cudaStream_t stream=nullptr;cudaGraph_t graphs[3]={};cudaGraphExec_t executable[3]={};float epsilon;
    QwenSpeculativeWorkspace *spec=nullptr;QwenPrefill *block=nullptr;bool tf32_prefill=false,ordered_prefill=false;unsigned tensor_gemm_calls=0;
    ~ResidentQwenFull() {
        if(stream)cudaStreamSynchronize(stream);
        for(auto e:executable)if(e)cudaGraphExecDestroy(e);for(auto g:graphs)if(g)cudaGraphDestroy(g);
        delete spec;delete block;
        for(auto p:allocations)cudaFree(p);if(stream)cudaStreamDestroy(stream);
    }
    template<typename T> bool alloc(T *&p,size_t n,const T *host=nullptr) {
        if(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(p);
        return !host || (cudaMemcpyAsync(p,host,n*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess && cudaStreamSynchronize(stream)==cudaSuccess);
    }
    template<typename T> void layer_enqueue(T *r) {
        // Contexts are exclusively owned by one Rust runtime. Captured kernel
        // parameters use the shared hidden vector; restore fields before returning.
        auto previous_stream=r->stream;auto *previous_x=r->x;
        r->stream=stream;r->x=x;r->enqueue();r->x=previous_x;r->stream=previous_stream;
    }
    bool reset() {
        if(spec){spec->invalidate();spec->poisoned=false;}
        for(const auto &layer:layers) {
            if(layer.kind==0) {
                auto *b=static_cast<ResidentQwenRecurrent*>(layer.context);const auto &c=b->cfg;
                if(cudaMemsetAsync(b->history,0,size_t(2*c.num_k+c.num_v)*c.head*c.conv*sizeof(float),stream)!=cudaSuccess
                    || cudaMemsetAsync(b->state,0,size_t(c.num_v)*c.head*c.head*sizeof(float),stream)!=cudaSuccess)return false;
                b->filled=0;
            } else static_cast<ResidentQwenAttention*>(layer.context)->filled=0;
        }
        filled=0;return true;
    }
    void advance(unsigned length) {
        filled=length;
        for(const auto &layer:layers) {
            if(layer.kind==0)static_cast<ResidentQwenRecurrent*>(layer.context)->filled=length;
            else static_cast<ResidentQwenAttention*>(layer.context)->filled=length;
        }
    }
    void enqueue(unsigned mode) {
        for(const auto &layer:layers) {
            if(layer.kind==0)layer_enqueue(static_cast<ResidentQwenRecurrent*>(layer.context));
            else {
                auto *r=static_cast<ResidentQwenAttention*>(layer.context);auto *previous=r->position;
                r->position=position;layer_enqueue(r);r->position=previous;
            }
        }
        output(mode,x);
    }
    void output(unsigned mode,float *hidden) {
        if(mode) {
            resident_norm<<<1,256,0,stream>>>(hidden,norm,epsilon,embd,h);
            QuantKind kind;resident_kind(head.type,kind);
            launch_quant_kernel(kind,head.weights,head.row_bytes,h,head.cols,head.rows,head.rows,logits,stream);
        }
        if(mode==2) {
            unsigned blocks=(vocab+255)/256;
            resident_argmax<<<blocks,256,0,stream>>>(logits,nullptr,vocab,maxima,ids);
            resident_argmax<<<1,256,0,stream>>>(maxima,ids,blocks,maximum,token);
        }
    }
    void enqueue_verify(unsigned count,unsigned mode) {
        bool previous=ordered_prefill;ordered_prefill=true;
        enqueue_block(count,0,spec->q8_tile);ordered_prefill=previous;
        if(!mode)return;
        resident_norm<<<count,256,0,stream>>>(block->x,norm,epsilon,embd,spec->h);
        QuantKind kind;resident_kind(head.type,kind);
        launch_ordered_gemm(kind,static_cast<const uint8_t*>(head.weights),head.row_bytes,spec->h,head.cols,head.rows,count,spec->logits,stream,spec->q8_tile);
        if(mode==2) {
            unsigned blocks=(vocab+255)/256;
            resident_argmax<<<dim3(blocks,count),256,0,stream>>>(spec->logits,nullptr,vocab,spec->maxima,spec->ids);
            resident_argmax<<<dim3(1,count),256,0,stream>>>(spec->maxima,spec->ids,blocks,spec->maximum,spec->tokens);
        }
    }
    void enqueue_block(unsigned count,unsigned mode,unsigned ordered_q8_tile=0) {
        for(const auto &layer:layers) {
            if(layer.kind==0)qwen_recurrent_block(static_cast<ResidentQwenRecurrent*>(layer.context),block,count,stream,tf32_prefill,ordered_prefill,ordered_q8_tile);
            else qwen_attention_block(static_cast<ResidentQwenAttention*>(layer.context),block,position,count,stream,tf32_prefill,ordered_prefill,ordered_q8_tile);
        }
        output(mode,block->x+size_t(count-1)*embd);
    }
};
}
extern "C" {
void *rbitnet_cuda_qwen_full_attention_create(const RbitnetQwenAttentionConfig *c,const RbitnetLlamaMatrix *m,
    const float *an,const float *fn,const float *qn,const float *kn,const float *freq) {
    if(!c || !m || !an || !fn || !qn || !kn || (c->rotary && !freq) || !c->embd || c->embd>32768
        || !c->ffn || c->ffn>65536 || !c->heads || c->heads>128 || !c->kv_heads || c->heads%c->kv_heads
        || !c->head_dim || c->head_dim>512 || !c->capacity || c->capacity>8192 || c->gated>1
        || c->rotary>c->head_dim || c->rotary%2 || !isfinite(c->epsilon) || c->epsilon<=0 || !isfinite(c->scale) || c->scale<=0)return nullptr;
    unsigned qs=c->heads*c->head_dim,ks=c->kv_heads*c->head_dim;
    unsigned cols[]={c->embd,c->embd,c->embd,qs,c->embd,c->embd,c->ffn};
    unsigned rows[]={qs*(1+c->gated),ks,ks,c->embd,c->ffn,c->ffn,c->embd};
    for(unsigned i=0;i<7;i++)if(!qwen_matrix_valid(m[i],cols[i],rows[i]))return nullptr;
    auto *r=new(std::nothrow) ResidentQwenAttention;if(!r)return nullptr;r->cfg=*c;r->split_kv=split_attention_enabled();
    for(unsigned i=0;i<7;i++)r->matrices[i]=m[i];
    if(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking)!=cudaSuccess
        || !r->alloc(r->x,c->embd) || !r->alloc(r->h,c->embd) || !r->alloc(r->q_full,qs*(1+c->gated))
        || !r->alloc(r->q,qs) || !r->alloc(r->k_raw,ks) || !r->alloc(r->k,ks) || !r->alloc(r->v,ks)
        || !r->alloc(r->attn,qs) || !r->alloc(r->projection,c->embd) || !r->alloc(r->gate,c->ffn) || !r->alloc(r->up,c->ffn)
        || !r->alloc_kv(r->kv_k,size_t(c->capacity)*ks) || !r->alloc_kv(r->kv_v,size_t(c->capacity)*ks) || !r->alloc(r->position,1)
        || !r->alloc(r->attn_norm,c->embd,an) || !r->alloc(r->ffn_norm,c->embd,fn)
        || !r->alloc(r->q_norm,c->head_dim,qn) || !r->alloc(r->k_norm,c->head_dim,kn)
        || (c->rotary && !r->alloc(r->frequency,c->rotary/2,freq))) {delete r;return nullptr;}
    if(r->split_kv && !r->alloc(r->attention_scratch,size_t(c->heads)*((c->capacity+attention_tile-1)/attention_tile)*(size_t(c->head_dim)+2))) {delete r;return nullptr;}
    return r;
}
void rbitnet_cuda_qwen_full_attention_destroy(void *p) {delete static_cast<ResidentQwenAttention*>(p);}
int rbitnet_cuda_qwen_full_attention_step(void *p,const float *input,unsigned pos,float *out) {
    auto *r=static_cast<ResidentQwenAttention*>(p);if(!r || !input || !out || pos>=r->cfg.capacity)return 1;
    NativeCallCompletion completion(r->stream);
    if(pos==0)r->filled=0;if(pos!=r->filled)return 2;
    if(cudaMemcpyAsync(r->x,input,r->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->position,&pos,sizeof(pos),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 3;
    if(r->cfg.graphs) {
        if(!r->executable) {
            if(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 4;
            r->enqueue();if(cudaStreamEndCapture(r->stream,&r->graph)!=cudaSuccess || cudaGraphInstantiate(&r->executable,r->graph,0)!=cudaSuccess)return 5;
        }
        if(cudaGraphLaunch(r->executable,r->stream)!=cudaSuccess)return 6;
    } else r->enqueue();
    if(cudaGetLastError()!=cudaSuccess || cudaMemcpyAsync(out,r->x,r->cfg.embd*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess
        || cudaStreamSynchronize(r->stream)!=cudaSuccess)return 7;r->filled=pos+1;completion.dismiss();return 0;
}
void *rbitnet_cuda_qwen_full_attention_snapshot(void *p) {
    auto *r=static_cast<ResidentQwenAttention*>(p);if(!r || !r->filled)return nullptr;
    auto *s=new(std::nothrow) QwenAttentionSnapshot;if(!s)return nullptr;
    s->length=r->filled;s->heads=r->cfg.kv_heads;s->dim=r->cfg.head_dim;
    size_t bytes=size_t(s->length)*s->heads*s->dim*sizeof(float);
    MemoryCategoryScope category(MemoryPrefix);
    if(cudaMalloc(reinterpret_cast<void**>(&s->k),bytes)!=cudaSuccess || cudaMalloc(reinterpret_cast<void**>(&s->v),bytes)!=cudaSuccess
        || cudaMemcpyAsync(s->k,r->kv_k,bytes,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(s->v,r->kv_v,bytes,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess
        || cudaStreamSynchronize(r->stream)!=cudaSuccess) {cudaStreamSynchronize(r->stream);delete s;return nullptr;}return s;
}
void rbitnet_cuda_qwen_full_attention_snapshot_destroy(void *p) {delete static_cast<QwenAttentionSnapshot*>(p);}
int rbitnet_cuda_qwen_full_attention_restore(void *p,const void *snapshot,unsigned length) {
    auto *r=static_cast<ResidentQwenAttention*>(p);auto *s=static_cast<const QwenAttentionSnapshot*>(snapshot);
    if(!r || !s || !length || length>s->length || length>r->cfg.capacity || s->heads!=r->cfg.kv_heads || s->dim!=r->cfg.head_dim)return 1;
    NativeCallCompletion completion(r->stream);
    size_t bytes=size_t(length)*s->heads*s->dim*sizeof(float);
    if(cudaMemcpyAsync(r->kv_k,s->k,bytes,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->kv_v,s->v,bytes,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess || cudaStreamSynchronize(r->stream)!=cudaSuccess)return 2;
    r->filled=length;completion.dismiss();return 0;
}
void *rbitnet_cuda_qwen_full_create(unsigned embd,unsigned vocab,unsigned capacity,unsigned layers,unsigned graphs,
    const RbitnetQwenFullLayer *blocks,const RbitnetLlamaMatrix *head,const float *norm,float epsilon) {
    if(!embd || embd>32768 || !vocab || vocab>1048576 || !capacity || capacity>8192 || !layers || layers>256
        || !blocks || !head || !norm || !isfinite(epsilon) || epsilon<=0 || !qwen_matrix_valid(*head,embd,vocab))return nullptr;
    for(unsigned i=0;i<layers;i++) {
        if(!blocks[i].context || blocks[i].kind>1)return nullptr;
        if(blocks[i].kind==0) {if(static_cast<ResidentQwenRecurrent*>(blocks[i].context)->cfg.embd!=embd)return nullptr;}
        else {const auto &c=static_cast<ResidentQwenAttention*>(blocks[i].context)->cfg;if(c.embd!=embd || c.capacity<capacity)return nullptr;}
    }
    auto *r=new(std::nothrow) ResidentQwenFull;if(!r)return nullptr;
    r->embd=embd;r->vocab=vocab;r->capacity=capacity;r->use_graphs=graphs!=0;r->epsilon=epsilon;r->head=*head;r->layers.assign(blocks,blocks+layers);
    unsigned maxima=(vocab+255)/256;
    if(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking)!=cudaSuccess || !r->alloc(r->x,embd) || !r->alloc(r->h,embd)
        || !r->alloc(r->norm,embd,norm) || !r->alloc(r->logits,vocab) || !r->alloc(r->maxima,maxima) || !r->alloc(r->ids,maxima)
        || !r->alloc(r->maximum,1) || !r->alloc(r->token,1) || !r->alloc(r->position,1)) {delete r;return nullptr;}return r;
}
void rbitnet_cuda_qwen_full_destroy(void *p) {delete static_cast<ResidentQwenFull*>(p);}
// Test-only host-array entrypoints use the same layer enqueue functions as the
// whole-model hot path. Temporary allocation/copies are excluded from throughput.
int rbitnet_cuda_qwen_recurrent_prefill_check(void *p,const float *input,unsigned pos,unsigned count,float *out) {
    auto *r=static_cast<ResidentQwenRecurrent*>(p);
    if(!r || !input || !out || !count || count>QwenPrefill::capacity || pos>=1048576 || count>1048576-pos)return 1;
    if(pos!=0 && pos!=r->filled)return 2;
    QwenPrefill b;std::vector<RbitnetQwenFullLayer> layers={{0,p}};
    if(!b.init(layers,r->cfg.embd))return 3;
    NativeCallCompletion completion(r->stream);
    if(pos==0) {
        unsigned inner=(2*r->cfg.num_k+r->cfg.num_v)*r->cfg.head;
        if(cudaMemsetAsync(r->history,0,size_t(inner)*r->cfg.conv*sizeof(float),r->stream)!=cudaSuccess
            || cudaMemsetAsync(r->state,0,size_t(r->cfg.num_v)*r->cfg.head*r->cfg.head*sizeof(float),r->stream)!=cudaSuccess)return 4;
    }
    size_t bytes=size_t(count)*r->cfg.embd*sizeof(float);
    if(cudaMemcpyAsync(b.x,input,bytes,cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 5;
    qwen_recurrent_block(r,&b,count,r->stream,false);
    if(cudaGetLastError()!=cudaSuccess || cudaMemcpyAsync(out,b.x,bytes,cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess
        || cudaStreamSynchronize(r->stream)!=cudaSuccess)return 6;
    r->filled=pos+count;completion.dismiss();return 0;
}
int rbitnet_cuda_qwen_attention_prefill_check(void *p,const float *input,unsigned pos,unsigned count,float *out) {
    auto *r=static_cast<ResidentQwenAttention*>(p);
    if(!r || !input || !out || !count || count>QwenPrefill::capacity || pos>=r->cfg.capacity || count>r->cfg.capacity-pos)return 1;
    if(pos!=0 && pos!=r->filled)return 2;
    QwenPrefill b;std::vector<RbitnetQwenFullLayer> layers={{1,p}};
    if(!b.init(layers,r->cfg.embd))return 3;
    NativeCallCompletion completion(r->stream);
    size_t bytes=size_t(count)*r->cfg.embd*sizeof(float);
    if(cudaMemcpyAsync(b.x,input,bytes,cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->position,&pos,sizeof(pos),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 4;
    qwen_attention_block(r,&b,r->position,count,r->stream,false);
    if(cudaGetLastError()!=cudaSuccess || cudaMemcpyAsync(out,b.x,bytes,cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess
        || cudaStreamSynchronize(r->stream)!=cudaSuccess)return 5;
    r->filled=pos+count;completion.dismiss();return 0;
}
int rbitnet_cuda_qwen_configure_prefill(void *p,unsigned enabled,unsigned tensor) {
    auto *r=static_cast<ResidentQwenFull*>(p);if(!r || enabled>1 || tensor>1 || r->filled || r->block)return 1;
    if(!enabled)return 0;
    auto *b=new(std::nothrow) QwenPrefill;if(!b)return 2;
    if(!b->init(r->layers,r->embd)) {delete b;cudaGetLastError();return 2;}
    r->block=b;r->tf32_prefill=tensor && tf32_prefill_supported();return 0;
}
int rbitnet_cuda_qwen_configure_ordered_prefill(void *p,unsigned enabled) {
    auto *r=static_cast<ResidentQwenFull*>(p);if(!r || enabled>1 || r->filled || !r->block || (enabled && r->tf32_prefill))return 1;
    for(auto &mode:r->block->executable)for(auto graph:mode)if(graph)return 2;
    r->ordered_prefill=enabled!=0;return 0;
}
unsigned rbitnet_cuda_qwen_prefill_capacity(void *p) {
    auto *r=static_cast<ResidentQwenFull*>(p);return r && r->block?QwenPrefill::capacity:0;
}
unsigned rbitnet_cuda_qwen_tensor_gemm_calls(void *p) {
    auto *r=static_cast<ResidentQwenFull*>(p);return r?r->tensor_gemm_calls:0;
}
unsigned rbitnet_cuda_qwen_split_attention_layers(void *p) {
    auto *r=static_cast<ResidentQwenFull*>(p);unsigned count=0;if(!r)return 0;
    for(auto &layer:r->layers)if(layer.kind==1 && static_cast<ResidentQwenAttention*>(layer.context)->split_kv)count++;
    return count;
}
int rbitnet_cuda_qwen_full_restored(void *p,unsigned length) {
    auto *r=static_cast<ResidentQwenFull*>(p);if(!r || length>r->capacity)return 1;
    for(const auto &layer:r->layers) {
        unsigned filled=layer.kind==0?static_cast<ResidentQwenRecurrent*>(layer.context)->filled:static_cast<ResidentQwenAttention*>(layer.context)->filled;
        if(filled!=length)return 2;
    }
    if(r->spec)r->spec->invalidate();r->filled=length;return 0;
}
int rbitnet_cuda_qwen_full_step(void *p,const float *input,unsigned pos,unsigned mode,float *logits,unsigned *token) {
    auto *r=static_cast<ResidentQwenFull*>(p);if(!r || !input || pos>=r->capacity || mode>2 || (mode==1 && !logits) || (mode==2 && !token))return 1;
    NativeCallCompletion completion(r->stream);
    if(pos==0 && !r->reset())return 2;
    if(pos!=r->filled)return 3;
    if(cudaMemcpyAsync(r->x,input,r->embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->position,&pos,sizeof(pos),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 4;
    if(r->use_graphs) {
        if(!r->executable[mode]) {
            if(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 5;
            r->enqueue(mode);if(cudaStreamEndCapture(r->stream,&r->graphs[mode])!=cudaSuccess || cudaGraphInstantiate(&r->executable[mode],r->graphs[mode],0)!=cudaSuccess)return 6;
        }
        if(cudaGraphLaunch(r->executable[mode],r->stream)!=cudaSuccess)return 7;
    } else r->enqueue(mode);
    if(cudaGetLastError()!=cudaSuccess)return 8;
    if(mode==1 && cudaMemcpyAsync(logits,r->logits,r->vocab*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 9;
    if(mode==2 && cudaMemcpyAsync(token,r->token,sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 10;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 11;
    r->advance(pos+1);r->tensor_gemm_calls=0;completion.dismiss();return 0;
}
int rbitnet_cuda_qwen_full_prefill(void *p,const float *input,unsigned pos,unsigned count,unsigned mode,float *logits,unsigned *token) {
    auto *r=static_cast<ResidentQwenFull*>(p);
    if(!r || !r->block || !input || !count || count>QwenPrefill::capacity || pos>=r->capacity || count>r->capacity-pos
        || mode>2 || (mode==1 && !logits) || (mode==2 && !token))return 1;
    NativeCallCompletion completion(r->stream);
    if(pos==0 && !r->reset())return 2;
    if(pos!=r->filled)return 3;
    if(cudaMemcpyAsync(r->block->x,input,size_t(count)*r->embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->position,&pos,sizeof(pos),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 4;
    unsigned tensor_calls=0;
    for(const auto &layer:r->layers) {
        const auto *m=layer.kind==0?static_cast<ResidentQwenRecurrent*>(layer.context)->matrices:static_cast<ResidentQwenAttention*>(layer.context)->matrices;
        for(unsigned i=0;i<(layer.kind==0?8u:7u);i++)tensor_calls+=use_tf32_prefill(m[i].cols,m[i].rows,count,r->tf32_prefill);
    }
    if(r->use_graphs) {
        if(!r->block->executable[mode][count]) {
            if(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 5;
            r->enqueue_block(count,mode);
            if(cudaStreamEndCapture(r->stream,&r->block->graphs[mode][count])!=cudaSuccess
                || cudaGraphInstantiate(&r->block->executable[mode][count],r->block->graphs[mode][count],0)!=cudaSuccess)return 6;
        }
        if(cudaGraphLaunch(r->block->executable[mode][count],r->stream)!=cudaSuccess)return 7;
    } else r->enqueue_block(count,mode);
    if(cudaGetLastError()!=cudaSuccess)return 8;
    if(mode==1 && cudaMemcpyAsync(logits,r->logits,r->vocab*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 9;
    if(mode==2 && cudaMemcpyAsync(token,r->token,sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 10;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 11;
    r->advance(pos+count);r->tensor_gemm_calls=tensor_calls;completion.dismiss();return 0;
}
// Private ABI; every entrypoint is synchronous with bounded/preallocated device storage.
int rbitnet_cuda_qwen_spec_configure(void *p,unsigned enabled) {
    auto *r=static_cast<ResidentQwenFull*>(p);
    if(!r || enabled>1 || r->filled || r->spec || (enabled && (!r->block || r->tf32_prefill)))return 1;
    if(!enabled)return 0;
    auto *s=new(std::nothrow) QwenSpeculativeWorkspace;if(!s)return 2;
    if(!s->init(r->embd,r->vocab,r->layers)) {delete s;cudaGetLastError();return 2;}
    r->spec=s;return 0;
}
// Configure before prompt/state/graphs exist; never mutate a captured graph.
int rbitnet_cuda_qwen_spec_q8_tile(void *p,unsigned tile) {
    auto *r=static_cast<ResidentQwenFull*>(p);
    if(!r || !r->spec || r->filled || (tile!=0 && tile!=4) || r->spec->valid || r->spec->poisoned)return 1;
    for(auto &mode:r->spec->graphs)for(auto graph:mode)if(graph)return 2;
    r->spec->q8_tile=tile;return 0;
}
uint64_t rbitnet_cuda_qwen_spec_bytes(void *p) {
    auto *r=static_cast<ResidentQwenFull*>(p);if(!r || !r->spec)return 0;
    auto *s=r->spec;uint64_t n=QwenSpeculativeWorkspace::limit,blocks=(r->vocab+255)/256;
    return 4*(n*r->embd+n*r->vocab+2*n*blocks+2*n+s->backup_elements);
}
size_t rbitnet_cuda_qwen_spec_state_elements(void *p) {
    auto *r=static_cast<ResidentQwenFull*>(p);return r && r->spec?r->spec->backup_elements:0;
}
// Test evidence only, excluded from every throughput run.
int rbitnet_cuda_qwen_spec_state_read(void *p,float *out,size_t elements) {
    auto *r=static_cast<ResidentQwenFull*>(p);
    if(!r || !r->spec || r->spec->poisoned || !out || elements!=r->spec->backup_elements)return 1;
    NativeCallCompletion completion(r->stream);
    for(const auto &plane:r->spec->planes)if(cudaMemcpyAsync(out+plane.offset,plane.source,plane.elements*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 2;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 3;completion.dismiss();return 0;
}
int rbitnet_cuda_qwen_spec_save(void *p,uint64_t *nonce) {
    auto *r=static_cast<ResidentQwenFull*>(p);
    if(!r || !r->spec || r->spec->poisoned || !nonce || !r->filled)return 1;
    for(const auto &layer:r->layers) {
        unsigned length=layer.kind==0?static_cast<ResidentQwenRecurrent*>(layer.context)->filled:static_cast<ResidentQwenAttention*>(layer.context)->filled;
        if(length!=r->filled)return 2;
    }
    auto *s=r->spec;s->invalidate();uint64_t id=llama_identity();if(!id)return 3;
    NativeCallCompletion completion(r->stream);
    if(!s->copy(r->stream,false) || cudaStreamSynchronize(r->stream)!=cudaSuccess) {
        s->poisoned=true;r->advance(0);return 4;
    }
    s->saved_length=r->filled;s->nonce=id;s->valid=true;*nonce=id;completion.dismiss();return 0;
}
int rbitnet_cuda_qwen_spec_restore(void *p,uint64_t nonce) {
    auto *r=static_cast<ResidentQwenFull*>(p);
    if(!r || !r->spec || r->spec->poisoned || !nonce || !r->spec->valid || r->spec->nonce!=nonce
        || !r->spec->saved_length || r->spec->saved_length>r->filled)return 1;
    auto *s=r->spec;unsigned length=s->saved_length;s->invalidate();
    NativeCallCompletion completion(r->stream);
    if(!s->copy(r->stream,true) || cudaStreamSynchronize(r->stream)!=cudaSuccess) {
        s->poisoned=true;r->advance(0);return 2;
    }
    // Attention prefix planes have never been overwritten: future KV becomes
    // inaccessible after advancing every layer back to the checkpoint length.
    r->advance(length);completion.dismiss();return 0;
}
int rbitnet_cuda_qwen_spec_discard(void *p,uint64_t nonce) {
    auto *r=static_cast<ResidentQwenFull*>(p);
    if(!r || !r->spec || !nonce || !r->spec->valid || nonce!=r->spec->nonce)return 1;
    r->spec->invalidate();return 0;
}
int rbitnet_cuda_qwen_spec_verify(void *p,const float *input,size_t input_elements,unsigned pos,unsigned count,
    unsigned mode,float *logits,size_t logits_elements,unsigned *ids,size_t id_elements) {
    auto *r=static_cast<ResidentQwenFull*>(p);
    if(!r || !r->spec || !r->block || r->tf32_prefill || !input || !count
        || count>QwenSpeculativeWorkspace::limit || pos>=r->capacity || count>r->capacity-pos
        || input_elements!=size_t(count)*r->embd || mode>2
        || (mode==0 && (logits || logits_elements || ids || id_elements))
        || (mode==1 && (!logits || logits_elements!=size_t(count)*r->vocab || ids || id_elements))
        || (mode==2 && (!ids || id_elements!=count || logits || logits_elements)))return 1;
    for(size_t i=0;i<input_elements;i++)if(!isfinite(input[i]))return 2;
    if(pos!=0 && (r->spec->poisoned || pos!=r->filled))return 3;
    NativeCallCompletion completion(r->stream);
    struct Guard {
        ResidentQwenFull *r;bool good=false;
        ~Guard() {if(!good){r->spec->invalidate();r->spec->poisoned=true;r->advance(0);}}
    } guard{r};
    if(pos==0 && !r->reset())return 4;
    if(cudaMemcpyAsync(r->block->x,input,input_elements*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->position,&pos,sizeof(pos),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 5;
    auto *s=r->spec;unsigned index=mode;
    if(r->use_graphs) {
        if(!s->executable[index][count]) {
            if(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 6;
            r->enqueue_verify(count,mode);
            if(cudaStreamEndCapture(r->stream,&s->graphs[index][count])!=cudaSuccess
                || cudaGraphInstantiate(&s->executable[index][count],s->graphs[index][count],0)!=cudaSuccess)return 7;
        }
        if(cudaGraphLaunch(s->executable[index][count],r->stream)!=cudaSuccess)return 8;
    } else r->enqueue_verify(count,mode);
    if(cudaGetLastError()!=cudaSuccess)return 9;
    if(mode==1 && cudaMemcpyAsync(logits,s->logits,logits_elements*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 10;
    if(mode==2 && cudaMemcpyAsync(ids,s->tokens,id_elements*sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 11;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 12;
    r->advance(pos+count);r->tensor_gemm_calls=0;guard.good=true;completion.dismiss();return 0;
}

}
