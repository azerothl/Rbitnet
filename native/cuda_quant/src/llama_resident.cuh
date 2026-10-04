// SPDX-License-Identifier: MIT
// Private streams and state per runtime: no mutable model state is process-global.
#include <vector>
#include <new>
#include "split_attention.cuh"
#include "kv_storage.cuh"
#include "paged_pool.cuh"
#include "paged_kernels.cuh"

namespace {
void launch_ordered_gemm(QuantKind,const uint8_t*,size_t,const float*,unsigned,unsigned,unsigned,float*,cudaStream_t,unsigned);

__global__ void resident_norm(float *x,const float *weights,float epsilon,unsigned n,float *y,const float *residual=nullptr) {
    x+=size_t(blockIdx.x)*n;y+=size_t(blockIdx.x)*n;if(residual)residual+=size_t(blockIdx.x)*n;
    __shared__ float sums[8];
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid/32;
    float total=0;
    for(unsigned i=tid;i<n;i+=blockDim.x) {float value=x[i];if(residual) {value+=residual[i];x[i]=value;} total+=value*value;}
    for(int shift=16;shift;shift/=2)total+=__shfl_down_sync(0xffffffff,total,shift);
    if(lane==0)sums[warp]=total;
    __syncthreads();
    if(tid==0) {total=0;for(unsigned i=0;i<blockDim.x/32;i++)total+=sums[i];sums[0]=1.0f/sqrtf(total/n+epsilon);}
    __syncthreads();
    for(unsigned i=tid;i<n;i+=blockDim.x)y[i]=(x[i]*sums[0])*weights[i];
}
__global__ void resident_add(float *x,const float *y,unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)x[i]+=y[i];
}
__global__ void resident_silu(float *gate,const float *up,unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {float g=gate[i];gate[i]=(g/(1.0f+expf(-g)))*up[i];}
}
__global__ void resident_rope_kv(float *q,float *k,const float *v,float *cache_k,float *cache_v,
    const float *frequency,const unsigned *position,unsigned heads,unsigned kv_heads,unsigned dim,unsigned rotary) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned pos=*position+blockIdx.y, pairs=dim/2;
    q+=size_t(blockIdx.y)*heads*dim;k+=size_t(blockIdx.y)*kv_heads*dim;v+=size_t(blockIdx.y)*kv_heads*dim;
    if(i<(heads+kv_heads)*pairs) {
        unsigned head=i/pairs,j=i%pairs;
        float *src=head<heads?q+head*dim:k+(head-heads)*dim;
        float a=src[2*j],b=src[2*j+1];
        if(2*j<rotary) {float angle=pos*frequency[j],s=sinf(angle),c=cosf(angle);src[2*j]=a*c-b*s;src[2*j+1]=a*s+b*c;}
        if(head>=heads) {
            size_t offset=size_t(pos)*kv_heads*dim+(head-heads)*dim+2*j;
            cache_k[offset]=src[2*j];cache_k[offset+1]=src[2*j+1];
            cache_v[offset]=v[(head-heads)*dim+2*j];cache_v[offset+1]=v[(head-heads)*dim+2*j+1];
        }
    }
}
__global__ void resident_attention(const float *k,const float *v,const float *q,const unsigned *position,
    unsigned kv_heads,unsigned heads,unsigned dim,unsigned window,float scale,float *y,const float *sinks=nullptr) {
    extern __shared__ float scores[];
    __shared__ float reductions[4];
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid/32;
    const unsigned head=blockIdx.x,kh=head/(heads/kv_heads),seq=*position+blockIdx.y+1;
    q+=size_t(blockIdx.y)*heads*dim;y+=size_t(blockIdx.y)*heads*dim;
    const unsigned first=window && seq>window?seq-window:0;
    for(unsigned p=first+warp;p<seq;p+=4) {
        float s=0;
        for(unsigned i=lane;i<dim;i+=32)s=fmaf(q[head*dim+i],k[size_t(p)*kv_heads*dim+kh*dim+i],s);
        for(int shift=16;shift;shift/=2)s+=__shfl_down_sync(0xffffffff,s,shift);
        if(lane==0)scores[p-first]=s*scale;
    }
    __syncthreads();
    float maximum=-CUDART_INF_F;
    for(unsigned i=tid;i<seq-first;i+=128)maximum=fmaxf(maximum,scores[i]);
    for(int shift=16;shift;shift/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,shift));
    if(lane==0)reductions[warp]=maximum;
    __syncthreads();maximum=fmaxf(fmaxf(reductions[0],reductions[1]),fmaxf(reductions[2],reductions[3]));
    if(sinks)maximum=fmaxf(maximum,sinks[head]);
    __syncthreads();
    float sum=0;
    for(unsigned i=tid;i<seq-first;i+=128) {scores[i]=expf(scores[i]-maximum);sum+=scores[i];}
    for(int shift=16;shift;shift/=2)sum+=__shfl_down_sync(0xffffffff,sum,shift);
    if(lane==0)reductions[warp]=sum;
    __syncthreads();sum=reductions[0]+reductions[1]+reductions[2]+reductions[3];
    if(sinks)sum+=expf(sinks[head]-maximum);
    for(unsigned i=tid;i<dim;i+=128) {
        float output=0;
        for(unsigned p=first;p<seq;p++)output=fmaf(scores[p-first]/sum,v[size_t(p)*kv_heads*dim+kh*dim+i],output);
        y[head*dim+i]=output;
    }
}
__device__ bool resident_better(float value,unsigned id,float previous,unsigned previous_id) {
    if(isnan(value))value=-CUDART_INF_F;
    unsigned bits=__float_as_uint(value),old=__float_as_uint(previous);
    unsigned key=(bits&0x80000000u)?~bits:bits^0x80000000u;
    unsigned old_key=(old&0x80000000u)?~old:old^0x80000000u;
    return key>old_key || (key==old_key && id>previous_id);
}
__global__ void resident_argmax(const float *values,const unsigned *indices,unsigned n,float *maxima,unsigned *ids) {
    values+=size_t(blockIdx.y)*n;if(indices)indices+=size_t(blockIdx.y)*n;
    maxima+=size_t(blockIdx.y)*gridDim.x;ids+=size_t(blockIdx.y)*gridDim.x;
    __shared__ float scores[256];
    __shared__ unsigned tokens[256];
    unsigned tid=threadIdx.x,i=blockIdx.x*blockDim.x+tid;
    scores[tid]=-CUDART_INF_F; tokens[tid]=0;
    for(;i<n;i+=gridDim.x*blockDim.x) {
        float value=isnan(values[i])?-CUDART_INF_F:values[i];unsigned id=indices?indices[i]:i;
        if(resident_better(value,id,scores[tid],tokens[tid])) {scores[tid]=value;tokens[tid]=id;}
    }
    __syncthreads();
    // Match Rust total_cmp and Iterator::max_by (the last token wins ties).
    for(unsigned stride=blockDim.x/2;stride;stride/=2) {
        if(tid<stride && resident_better(scores[tid+stride],tokens[tid+stride],scores[tid],tokens[tid])) {scores[tid]=scores[tid+stride];tokens[tid]=tokens[tid+stride];}
        __syncthreads();
    }
    if(tid==0) {maxima[blockIdx.x]=scores[0];ids[blockIdx.x]=tokens[0];}
}

bool resident_kind(unsigned ty,QuantKind &kind) {
    switch(ty) {
        case 0:kind=QuantKind::F32;break;case 2:kind=QuantKind::Q4_0;break;
        case 6:kind=QuantKind::Q5_0;break;case 8:kind=QuantKind::Q8_0;break;
        case 12:kind=QuantKind::Q4_K;break;case 13:kind=QuantKind::Q5_K;break;
        case 14:kind=QuantKind::Q6_K;break;case 39:kind=QuantKind::MXFP4;break;
        default:return false;
    }
    return true;
}
uint64_t llama_identity() {
    static std::mutex mutex;static uint64_t next=0;
    std::lock_guard<std::mutex> lock(mutex);
    if(next==std::numeric_limits<uint64_t>::max())return 0;
    return ++next;
}
struct ResidentLlama {
    const uint64_t identity=llama_identity();
    LlamaBlock *block=nullptr;
    std::unique_ptr<PagedKvState> paged;
    RbitnetLlamaConfig cfg;
    std::vector<RbitnetLlamaLayer> layers;
    std::vector<unsigned char> batch_model_key;
    RbitnetLlamaMatrix output;
    std::vector<void*> allocations;
    float *x=nullptr,*h=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*attn=nullptr,*projection=nullptr,*gate=nullptr,*up=nullptr,*logits=nullptr,*kv_k=nullptr,*kv_v=nullptr,*out_norm=nullptr,*frequency=nullptr,*maxima=nullptr,*maximum=nullptr;
    unsigned *position=nullptr,*ids=nullptr,*token=nullptr;
    float *attention_scratch=nullptr,*block_attention_scratch=nullptr;
    size_t attention_scratch_size=0;
    bool split_kv=false;
    bool tf32_prefill=false;
    unsigned tensor_gemm_calls=0;
    unsigned filled=0,kv_format=0;size_t kv_layer_bytes=0;unsigned *kv_invalid=nullptr;bool kv_poisoned=false;
    cudaStream_t stream=nullptr;
    cudaGraph_t graphs[3]={};
    cudaGraphExec_t executable[3]={};
    cudaGraph_t verify_graphs[2][LlamaBlock::verify_capacity+1]={};
    cudaGraphExec_t verify_executable[2][LlamaBlock::verify_capacity+1]={};
    bool use_graphs=true;
    ~ResidentLlama() {
        if(stream)cudaStreamSynchronize(stream);
        delete block;
        for(auto &modes:verify_executable)for(auto exec:modes)if(exec)cudaGraphExecDestroy(exec);
        for(auto &modes:verify_graphs)for(auto graph:modes)if(graph)cudaGraphDestroy(graph);
        for(auto exec:executable)if(exec)cudaGraphExecDestroy(exec);
        for(auto graph:graphs)if(graph)cudaGraphDestroy(graph);
        for(auto p:allocations)cudaFree(p);
        paged.reset();
        if(stream)cudaStreamDestroy(stream);
    }
    template<typename T> bool alloc(T *&ptr,size_t count,const T *host=nullptr) {
        if(cudaMalloc(reinterpret_cast<void**>(&ptr),count*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(ptr);
        return !host || (cudaMemcpyAsync(ptr,host,count*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess
            && cudaStreamSynchronize(stream)==cudaSuccess);
    }
    template<typename T> bool alloc_kv(T *&ptr,size_t count) {
        MemoryCategoryScope category(MemoryKv);return alloc(ptr,count);
    }
    bool alloc_kv_bytes(float *&ptr,size_t bytes) {
        MemoryCategoryScope category(MemoryKv);
        if(!bytes || cudaMalloc(reinterpret_cast<void**>(&ptr),bytes)!=cudaSuccess)return false;
        allocations.push_back(ptr);return true;
    }
    template<unsigned Format,bool Paged> void encoded_layer_t(float *q,float *k,const float *v,unsigned layer,unsigned count,float *scratch,float *out) {
        EncodedKvView<Format,Paged> view{paged?nullptr:reinterpret_cast<float*>(reinterpret_cast<char*>(kv_k)+layer*kv_layer_bytes),
            paged?nullptr:reinterpret_cast<float*>(reinterpret_cast<char*>(kv_v)+layer*kv_layer_bytes),paged?paged->table_k:nullptr,paged?paged->table_v:nullptr,
            layer,cfg.layers,cfg.capacity,cfg.kv_heads,cfg.head_dim};
        launch_encoded_kv(view,q,k,v,frequency,position,cfg.heads,cfg.rotary,cfg.window,count,split_kv,scratch,out,kv_invalid,stream);
    }
    void encoded_layer(float *q,float *k,const float *v,unsigned layer,unsigned count,float *scratch,float *out) {
        if(kv_format==1) {if(paged)encoded_layer_t<1,true>(q,k,v,layer,count,scratch,out);else encoded_layer_t<1,false>(q,k,v,layer,count,scratch,out);}
        else {if(paged)encoded_layer_t<2,true>(q,k,v,layer,count,scratch,out);else encoded_layer_t<2,false>(q,k,v,layer,count,scratch,out);}
    }
    void matrix(const RbitnetLlamaMatrix &m,const float *input,float *result) {
        QuantKind kind; resident_kind(m.type,kind);
        launch_quant_kernel(kind,m.weights,m.row_bytes,input,m.cols,m.rows,m.rows,result,stream);
    }
    void enqueue(unsigned mode) {
        size_t stride=size_t(cfg.kv_heads)*cfg.head_dim,layer_stride=stride*cfg.capacity;
        resident_norm<<<1,256,0,stream>>>(x,layers[0].attn_norm,cfg.epsilon,cfg.embd,h);
        for(unsigned il=0;il<cfg.layers;il++) {
            const auto &layer=layers[il];
            matrix(layer.q,h,q);matrix(layer.k,h,k);matrix(layer.v,h,v);
            if(kv_format)encoded_layer(q,k,v,il,1,attention_scratch,attn);
            else if(paged) {
                paged_resident_rope_kv<<<((cfg.heads+cfg.kv_heads)*(cfg.head_dim/2)+255)/256,256,0,stream>>>(q,k,v,paged->table_k,paged->table_v,il,frequency,position,cfg.heads,cfg.kv_heads,cfg.head_dim,cfg.rotary);
                if(split_kv)launch_paged_split_attention(paged->table_k,paged->table_v,il,q,position,cfg.kv_heads,cfg.heads,cfg.head_dim,cfg.window,1.0f/sqrtf(float(cfg.head_dim)),cfg.capacity,1,attention_scratch,attn,stream);
                else paged_resident_attention<<<cfg.heads,128,cfg.capacity*sizeof(float),stream>>>(paged->table_k,paged->table_v,il,q,position,cfg.kv_heads,cfg.heads,cfg.head_dim,cfg.window,1.0f/sqrtf(float(cfg.head_dim)),attn);
            } else {
            resident_rope_kv<<<((cfg.heads+cfg.kv_heads)*(cfg.head_dim/2)+255)/256,256,0,stream>>>(q,k,v,kv_k+il*layer_stride,kv_v+il*layer_stride,frequency,position,cfg.heads,cfg.kv_heads,cfg.head_dim,cfg.rotary);
            if(split_kv)launch_split_attention(kv_k+il*layer_stride,kv_v+il*layer_stride,q,position,cfg.kv_heads,cfg.heads,cfg.head_dim,cfg.window,1.0f/sqrtf(float(cfg.head_dim)),cfg.capacity,1,attention_scratch,attn,stream);
            else resident_attention<<<cfg.heads,128,cfg.capacity*sizeof(float),stream>>>(kv_k+il*layer_stride,kv_v+il*layer_stride,q,position,cfg.kv_heads,cfg.heads,cfg.head_dim,cfg.window,1.0f/sqrtf(float(cfg.head_dim)),attn);
            }
            matrix(layer.out,attn,projection);
            resident_norm<<<1,256,0,stream>>>(x,layer.ffn_norm,cfg.epsilon,cfg.embd,h,projection);
            matrix(layer.gate,h,gate);matrix(layer.up,h,up);
            resident_silu<<<(cfg.ffn+255)/256,256,0,stream>>>(gate,up,cfg.ffn);
            matrix(layer.down,gate,projection);
            if(il+1<cfg.layers)resident_norm<<<1,256,0,stream>>>(x,layers[il+1].attn_norm,cfg.epsilon,cfg.embd,h,projection);
            else if(mode)resident_norm<<<1,256,0,stream>>>(x,out_norm,cfg.epsilon,cfg.embd,h,projection);
            else resident_add<<<(cfg.embd+255)/256,256,0,stream>>>(x,projection,cfg.embd);
        }
        if(mode)matrix(output,h,logits);
        if(mode==2) {
            unsigned blocks=(cfg.vocab+255)/256;
            resident_argmax<<<blocks,256,0,stream>>>(logits,nullptr,cfg.vocab,maxima,ids);
            resident_argmax<<<1,256,0,stream>>>(maxima,ids,blocks,maximum,token);
        }
    }
};
bool llama_paged_variant(ResidentLlama *r) {
    if(!r->paged)return true;
    auto &pool=*r->paged->pool;std::lock_guard<std::mutex> lock(pool.mutex);
    if(pool.split!=r->split_kv)return false;
    if(pool.tf32_mode<0)pool.tf32_mode=r->tf32_prefill;
    return pool.tf32_mode==int(r->tf32_prefill);
}
struct LlamaSnapshot {
    std::unique_ptr<PagedKvSnapshot> paged;
    uint64_t owner=0;
    unsigned layers=0,kv_heads=0,head_dim=0,length=0,format=0;
    float *k=nullptr,*v=nullptr;
    ~LlamaSnapshot() {if(k)cudaFree(k);if(v)cudaFree(v);}
};
}

std::vector<unsigned char> llama_paged_model_key(const RbitnetLlamaConfig &c,
    const RbitnetLlamaLayer *layers,const RbitnetLlamaMatrix &head,const float *norm,const float *freq) {
    std::vector<unsigned char> key;
    auto append=[&](const auto &v) {const auto *p=reinterpret_cast<const unsigned char*>(&v);key.insert(key.end(),p,p+sizeof(v));};
    for(unsigned v:{c.embd,c.ffn,c.vocab,c.layers,c.heads,c.kv_heads,c.head_dim,c.rotary,c.window})append(v);
    append(c.epsilon);
    auto matrix=[&](const RbitnetLlamaMatrix &m) {append(m.weights);append(m.row_bytes);append(m.type);append(m.cols);append(m.rows);};
    matrix(head);
    for(unsigned i=0;i<c.embd;i++)append(norm[i]);
    for(unsigned i=0;i<c.rotary/2;i++)append(freq[i]);
    for(unsigned i=0;i<c.layers;i++) {
        const auto &l=layers[i];for(const auto &m:{l.q,l.k,l.v,l.out,l.gate,l.up,l.down})matrix(m);
        for(unsigned j=0;j<c.embd;j++) {append(l.attn_norm[j]);append(l.ffn_norm[j]);}
    }
    return key;
}

static void *llama_create_impl(const RbitnetLlamaConfig *cfg,const RbitnetLlamaLayer *layers,
    const RbitnetLlamaMatrix *output,const float *out_norm,const float *frequency,unsigned page_limit,const ResidentLlama *peer,unsigned variants,unsigned format=0) {
    if(!cfg || !layers || !output || !out_norm || !frequency || !cfg->embd || !cfg->ffn || !cfg->vocab || !cfg->layers || !cfg->heads || !cfg->kv_heads || cfg->heads%cfg->kv_heads || cfg->head_dim%2 || cfg->heads*cfg->head_dim!=cfg->embd || !cfg->rotary || cfg->rotary%2 || cfg->rotary>cfg->head_dim || !cfg->capacity || cfg->capacity>8192)return nullptr;
    if(cfg->embd>32768 || cfg->ffn>1048576 || cfg->vocab>1048576 || cfg->layers>256
        || cfg->heads>1024 || cfg->kv_heads>cfg->heads || cfg->head_dim>4096
        || size_t(cfg->heads)*cfg->head_dim!=cfg->embd || !output->weights)return nullptr;
    for(unsigned i=0;i<cfg->layers;i++)if(!layers[i].attn_norm || !layers[i].ffn_norm)return nullptr;
    if(format>2 || (format && (cfg->head_dim%32 || (variants&2))) || variants>3 || page_limit>65536 || (peer && (!page_limit || !peer->paged)))return nullptr;
    auto *r=new(std::nothrow) ResidentLlama;
    if(!r)return nullptr;
    if(!r->identity) {delete r;return nullptr;}
    if(page_limit) {
        try {
            auto key=llama_paged_model_key(*cfg,layers,*output,out_norm,frequency);key.push_back(static_cast<unsigned char>(format));
            std::shared_ptr<PhysicalKvPool> pool;
            if(peer) {
                pool=peer->paged->pool;
                if(pool->limit!=page_limit || pool->model_key!=key || pool->split!=bool(variants&1)
                    || pool->tf32_mode!=int(bool(variants&2)&&tf32_prefill_supported())) {delete r;return nullptr;}
            } else {
                pool=std::make_shared<PhysicalKvPool>(cfg->layers,cfg->kv_heads*cfg->head_dim,page_limit,format,cfg->head_dim);
                pool->model_key=std::move(key);
                pool->split=bool(variants&1);
                pool->tf32_mode=bool(variants&2)&&tf32_prefill_supported();
            }
            r->paged=std::unique_ptr<PagedKvState>(new(std::nothrow) PagedKvState(cfg->capacity));
            if(!r->paged || !r->paged->init(pool)) {delete r;return nullptr;}
        } catch(const std::bad_alloc&) {delete r;return nullptr;}
    }
    r->kv_format=format;r->kv_layer_bytes=encoded_kv_plane_bytes(format,size_t(cfg->capacity)*cfg->kv_heads*cfg->head_dim,cfg->head_dim);
    try {r->batch_model_key=llama_paged_model_key(*cfg,layers,*output,out_norm,frequency);}
    catch(const std::bad_alloc&) {delete r;return nullptr;}
    r->cfg=*cfg;r->output=*output;r->layers.assign(layers,layers+cfg->layers);r->use_graphs=cfg->graphs!=0;
    r->split_kv=bool(variants&1);
    r->tf32_prefill=bool(variants&2)&&tf32_prefill_supported();
    r->attention_scratch_size=size_t(cfg->heads)*((cfg->capacity+attention_tile-1)/attention_tile)*(size_t(cfg->head_dim)+2);
    QuantKind kind;
    if(!resident_kind(output->type,kind)) {delete r;return nullptr;}
    for(auto &l:r->layers)for(auto m:{l.q,l.k,l.v,l.out,l.gate,l.up,l.down})if(!m.weights || !resident_kind(m.type,kind)) {delete r;return nullptr;}
    if(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking)!=cudaSuccess) {delete r;return nullptr;}
    size_t kv=size_t(cfg->layers)*cfg->capacity*cfg->kv_heads*cfg->head_dim;
    if(!r->alloc(r->x,cfg->embd) || !r->alloc(r->h,cfg->embd) || !r->alloc(r->q,cfg->embd)
        || !r->alloc(r->k,cfg->kv_heads*cfg->head_dim) || !r->alloc(r->v,cfg->kv_heads*cfg->head_dim)
        || !r->alloc(r->attn,cfg->embd) || !r->alloc(r->projection,cfg->embd) || !r->alloc(r->gate,cfg->ffn)
        || !r->alloc(r->up,cfg->ffn) || !r->alloc(r->logits,cfg->vocab) || (!r->paged && (!r->alloc_kv_bytes(r->kv_k,r->kv_layer_bytes*cfg->layers) || !r->alloc_kv_bytes(r->kv_v,r->kv_layer_bytes*cfg->layers)))
        || !r->alloc(r->out_norm,cfg->embd,out_norm) || !r->alloc(r->frequency,cfg->rotary/2,frequency)
        || !r->alloc(r->position,1) || !r->alloc(r->maxima,(cfg->vocab+255)/256) || !r->alloc(r->ids,(cfg->vocab+255)/256)
        || !r->alloc(r->maximum,1) || !r->alloc(r->token,1)) {delete r;return nullptr;}
    if(format && !r->alloc(r->kv_invalid,1)) {delete r;return nullptr;}
    if(r->split_kv && !r->alloc(r->attention_scratch,r->attention_scratch_size)) {delete r;return nullptr;}
    for(auto &layer:r->layers) {
        float *norm=nullptr;
        if(!r->alloc(norm,cfg->embd,layer.attn_norm)) {delete r;return nullptr;}
        layer.attn_norm=norm;
        if(!r->alloc(norm,cfg->embd,layer.ffn_norm)) {delete r;return nullptr;}
        layer.ffn_norm=norm;
    }
    return r;
}
extern "C" {
void *rbitnet_cuda_llama_create(const RbitnetLlamaConfig *cfg,const RbitnetLlamaLayer *layers,
    const RbitnetLlamaMatrix *output,const float *norm,const float *freq) {
    return llama_create_impl(cfg,layers,output,norm,freq,0,nullptr,unsigned(split_attention_enabled())|(unsigned(tf32_prefill_requested())<<1));
}
void *rbitnet_cuda_llama_create_paged(const RbitnetLlamaConfig *cfg,const RbitnetLlamaLayer *layers,
    const RbitnetLlamaMatrix *output,const float *norm,const float *freq,unsigned limit,const void *peer,unsigned variants) {
    if(!limit)return nullptr;
    return llama_create_impl(cfg,layers,output,norm,freq,limit,static_cast<const ResidentLlama*>(peer),variants);
}
void *rbitnet_cuda_llama_create_kv(const RbitnetLlamaConfig *cfg,const RbitnetLlamaLayer *layers,
    const RbitnetLlamaMatrix *output,const float *norm,const float *freq,unsigned limit,const void *peer,unsigned variants,unsigned format) {
    return llama_create_impl(cfg,layers,output,norm,freq,limit,static_cast<const ResidentLlama*>(peer),variants,format);
}
int rbitnet_cuda_llama_paged_stats(const void *context,RbitnetPagedKvStats *out) {
    auto *r=static_cast<const ResidentLlama*>(context);if(!r || !r->paged || !out)return 1;
    auto &pool=*r->paged->pool;std::lock_guard<std::mutex> lock(pool.mutex);
    *out={};out->allocated_pages=pool.pages.size();out->peak_pages=pool.peak_pages;
    out->limit_pages=pool.limit;out->bytes_per_page=2*encoded_kv_plane_bytes(pool.format,size_t(pool.layers)*pool.stride*llama_page_tokens,pool.dim);
    out->allocations=pool.allocations;out->reuses=pool.reuses;out->cow_pages=pool.copies;out->refusals=pool.refusals;
    for(auto &p:pool.pages) {auto refs=p.use_count()-1;if(refs)out->referenced_pages++;out->references+=refs;}
    for(auto &p:r->paged->active)if(p)out->active_pages++;
    out->tokens=r->filled;return 0;
}
int rbitnet_cuda_llama_paged_trim(void *context) {
    auto *r=static_cast<ResidentLlama*>(context);if(!r || !r->paged)return 1;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 2;
    r->paged->pool->trim();return 0;
}
void rbitnet_cuda_llama_destroy(void *context) {delete static_cast<ResidentLlama*>(context);}
unsigned rbitnet_cuda_llama_split_attention_layers(void *p) {
    auto *r=static_cast<ResidentLlama*>(p);return r && r->split_kv?r->cfg.layers:0;
}
unsigned rbitnet_cuda_llama_tensor_gemm_calls(void *p) {
    auto *r=static_cast<ResidentLlama*>(p);return r?r->tensor_gemm_calls:0;
}
int rbitnet_cuda_llama_configure_tensor_prefill(void *p,unsigned enabled) {
    auto *r=static_cast<ResidentLlama*>(p);if(!r || enabled>1 || (r->kv_format && enabled) || r->block || r->filled)return 1;
    const bool wanted=enabled && tf32_prefill_supported();
    if(r->paged) {
        auto &pool=*r->paged->pool;std::lock_guard<std::mutex> lock(pool.mutex);
        if(pool.tf32_mode>=0 && pool.tf32_mode!=int(wanted))return 2;
        pool.tf32_mode=wanted;
    }
    r->tf32_prefill=wanted;return 0;
}
static int llama_prefill_impl(void *context,const float *embeddings,unsigned pos,unsigned count,
    unsigned mode,float *logits,unsigned *token,bool all) {
    auto *r=static_cast<ResidentLlama*>(context);
    if(!r || !embeddings || !count || count>LlamaBlock::capacity || pos>=r->cfg.capacity
        || count>r->cfg.capacity-pos || mode>2 || (mode==1 && !logits) || (mode==2 && !token)
        || (all && (!mode || count>LlamaBlock::verify_capacity)))return 1;
    NativeCallCompletion completion(r->stream);
    if(!llama_paged_variant(r))return 15;
    if(pos==0) {
        if(r->paged && !r->paged->reset(r->stream))return 13;
        r->filled=0;r->kv_poisoned=false;
    }if(pos!=r->filled)return 2;
    if(r->kv_format) {if(r->kv_poisoned)return 16;r->kv_poisoned=true;if(cudaMemsetAsync(r->kv_invalid,0,sizeof(unsigned),r->stream)!=cudaSuccess)return 16;}
    if(r->paged && !r->paged->prepare(pos,count,r->stream))return 14;
    const auto &c=r->cfg;unsigned stride=c.kv_heads*c.head_dim;
    if(!r->block) {
        auto *b=new(std::nothrow) LlamaBlock;if(!b)return 3;
        if(!b->init(c.embd,c.ffn,stride)) {delete b;return 3;}r->block=b;
    }
    auto &p=r->block->p;
    if(r->split_kv && !r->block_attention_scratch && !r->alloc(r->block_attention_scratch,r->attention_scratch_size*LlamaBlock::capacity))return 3;
    if(all && !r->block->init_verify(c.vocab))return 3;
    if(cudaMemcpyAsync(p[0],embeddings,size_t(count)*c.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->position,&pos,sizeof(pos),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 4;
    unsigned tensor_calls=0;
    auto uses_tensor=[&](const RbitnetLlamaMatrix &m) {return use_tf32_prefill(m.cols,m.rows,count,r->tf32_prefill);};
    for(const auto &layer:r->layers)for(const auto &m:{layer.q,layer.k,layer.v,layer.out,layer.gate,layer.up,layer.down})tensor_calls+=!r->kv_format && uses_tensor(m);
    if(mode && all)tensor_calls+=!r->kv_format && uses_tensor(r->output);
    bool use_graph=all && r->use_graphs;
    if(!use_graph || !r->verify_executable[mode-1][count]) {
    if(use_graph && cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 9;
    auto matrix=[&](const RbitnetLlamaMatrix &m,const float *x,float *y) {
        QuantKind kind;resident_kind(m.type,kind);
        if(r->kv_format)launch_ordered_gemm(kind,static_cast<const uint8_t*>(m.weights),m.row_bytes,x,m.cols,m.rows,count,y,r->stream,0);
        else launch_prefill_gemm(kind,m.weights,m.row_bytes,x,m.cols,m.rows,count,y,r->stream,uses_tensor(m));
    };
    size_t layer_stride=size_t(stride)*c.capacity;
    resident_norm<<<count,256,0,r->stream>>>(p[0],r->layers[0].attn_norm,c.epsilon,c.embd,p[1]);
    for(unsigned il=0;il<c.layers;il++) {
        const auto &l=r->layers[il];matrix(l.q,p[1],p[2]);matrix(l.k,p[1],p[3]);matrix(l.v,p[1],p[4]);
        dim3 rope(((c.heads+c.kv_heads)*(c.head_dim/2)+255)/256,count);
        if(r->kv_format)r->encoded_layer(p[2],p[3],p[4],il,count,r->block_attention_scratch,p[5]);
        else if(r->paged) {
            paged_resident_rope_kv<<<rope,256,0,r->stream>>>(p[2],p[3],p[4],r->paged->table_k,r->paged->table_v,il,r->frequency,r->position,c.heads,c.kv_heads,c.head_dim,c.rotary);
            if(r->split_kv)launch_paged_split_attention(r->paged->table_k,r->paged->table_v,il,p[2],r->position,c.kv_heads,c.heads,c.head_dim,c.window,1.0f/sqrtf(float(c.head_dim)),c.capacity,count,r->block_attention_scratch,p[5],r->stream);
            else paged_resident_attention<<<dim3(c.heads,count),128,c.capacity*sizeof(float),r->stream>>>(r->paged->table_k,r->paged->table_v,il,p[2],r->position,c.kv_heads,c.heads,c.head_dim,c.window,1.0f/sqrtf(float(c.head_dim)),p[5]);
        } else {
        resident_rope_kv<<<rope,256,0,r->stream>>>(p[2],p[3],p[4],r->kv_k+il*layer_stride,r->kv_v+il*layer_stride,r->frequency,r->position,c.heads,c.kv_heads,c.head_dim,c.rotary);
        if(r->split_kv)launch_split_attention(r->kv_k+il*layer_stride,r->kv_v+il*layer_stride,p[2],r->position,c.kv_heads,c.heads,c.head_dim,c.window,1.0f/sqrtf(float(c.head_dim)),c.capacity,count,r->block_attention_scratch,p[5],r->stream);
        else resident_attention<<<dim3(c.heads,count),128,c.capacity*sizeof(float),r->stream>>>(r->kv_k+il*layer_stride,r->kv_v+il*layer_stride,p[2],r->position,c.kv_heads,c.heads,c.head_dim,c.window,1.0f/sqrtf(float(c.head_dim)),p[5]);
        }
        matrix(l.out,p[5],p[6]);
        resident_norm<<<count,256,0,r->stream>>>(p[0],l.ffn_norm,c.epsilon,c.embd,p[1],p[6]);
        matrix(l.gate,p[1],p[7]);matrix(l.up,p[1],p[8]);
        resident_silu<<<(count*c.ffn+255)/256,256,0,r->stream>>>(p[7],p[8],count*c.ffn);
        matrix(l.down,p[7],p[6]);
        if(il+1<c.layers)resident_norm<<<count,256,0,r->stream>>>(p[0],r->layers[il+1].attn_norm,c.epsilon,c.embd,p[1],p[6]);
        else resident_add<<<(count*c.embd+255)/256,256,0,r->stream>>>(p[0],p[6],count*c.embd);
    }
    if(mode) {
        if(all) {
            resident_norm<<<count,256,0,r->stream>>>(p[0],r->out_norm,c.epsilon,c.embd,p[1]);
            matrix(r->output,p[1],r->block->verify_logits);
        } else {
            resident_norm<<<1,256,0,r->stream>>>(p[0]+size_t(count-1)*c.embd,r->out_norm,c.epsilon,c.embd,r->h);
            r->matrix(r->output,r->h,r->logits);
        }
    }
    if(mode==2) {
        unsigned blocks=(c.vocab+255)/256;
        if(all) {
            resident_argmax<<<dim3(blocks,count),256,0,r->stream>>>(r->block->verify_logits,nullptr,c.vocab,r->block->verify_maxima,r->block->verify_ids);
            resident_argmax<<<dim3(1,count),256,0,r->stream>>>(r->block->verify_maxima,r->block->verify_ids,blocks,p[1],r->block->verify_tokens);
        } else {
            resident_argmax<<<blocks,256,0,r->stream>>>(r->logits,nullptr,c.vocab,r->maxima,r->ids);
            resident_argmax<<<1,256,0,r->stream>>>(r->maxima,r->ids,blocks,r->maximum,r->token);
        }
    }
    if(use_graph) {
        if(cudaStreamEndCapture(r->stream,&r->verify_graphs[mode-1][count])!=cudaSuccess)return 10;
        if(cudaGraphInstantiate(&r->verify_executable[mode-1][count],r->verify_graphs[mode-1][count],0)!=cudaSuccess)return 11;
    }
    }
    if(use_graph && cudaGraphLaunch(r->verify_executable[mode-1][count],r->stream)!=cudaSuccess)return 12;
    if(cudaGetLastError()!=cudaSuccess)return 5;
    if(mode==1 && cudaMemcpyAsync(logits,all?r->block->verify_logits:r->logits,size_t(all?count:1)*c.vocab*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 6;
    if(mode==2 && cudaMemcpyAsync(token,all?r->block->verify_tokens:r->token,size_t(all?count:1)*sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 7;
    unsigned invalid=0;
    if(r->kv_format && cudaMemcpyAsync(&invalid,r->kv_invalid,sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 16;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 8;
    if(invalid)return 16;r->kv_poisoned=false;
    r->filled=pos+count;r->tensor_gemm_calls=tensor_calls;completion.dismiss();return 0;
}
int rbitnet_cuda_llama_prefill(void *context,const float *embeddings,unsigned pos,unsigned count,unsigned mode,float *logits,unsigned *token) {
    return llama_prefill_impl(context,embeddings,pos,count,mode,logits,token,false);
}
int rbitnet_cuda_llama_verify(void *context,const float *embeddings,unsigned pos,unsigned count,unsigned mode,float *logits,unsigned *tokens) {
    return llama_prefill_impl(context,embeddings,pos,count,mode,logits,tokens,true);
}
int rbitnet_cuda_llama_truncate(void *context,unsigned length) {
    auto *r=static_cast<ResidentLlama*>(context);
    if(!r || length>r->filled || r->kv_poisoned)return 1;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 2;
    // Dense causal attention never reads the discarded tail. Later writes replace it.
    if(r->paged && !r->paged->truncate(length,r->stream))return 3;
    r->filled=length;return 0;
}
void rbitnet_cuda_llama_snapshot_destroy(void *snapshot) {delete static_cast<LlamaSnapshot*>(snapshot);}
void *rbitnet_cuda_llama_snapshot(void *context,unsigned length) {
    auto *r=static_cast<ResidentLlama*>(context);
    if(!r || !length || length>r->filled || r->kv_poisoned)return nullptr;
    auto *s=new(std::nothrow) LlamaSnapshot;if(!s)return nullptr;
    s->format=r->kv_format;s->owner=r->identity;s->layers=r->cfg.layers;s->kv_heads=r->cfg.kv_heads;s->head_dim=r->cfg.head_dim;s->length=length;
    if(r->paged) {
        if(!llama_paged_variant(r)) {delete s;return nullptr;}
        if(cudaStreamSynchronize(r->stream)!=cudaSuccess) {delete s;return nullptr;}
        s->paged=r->paged->snapshot(length);
        if(!s->paged) {delete s;return nullptr;}return s;
    }
    size_t span=size_t(length)*s->kv_heads*s->head_dim,plane=encoded_kv_plane_bytes(s->format,span,s->head_dim),bytes=plane*s->layers;
    MemoryCategoryScope category(MemoryPrefix);
    if(cudaMalloc(reinterpret_cast<void**>(&s->k),bytes)!=cudaSuccess || cudaMalloc(reinterpret_cast<void**>(&s->v),bytes)!=cudaSuccess) {delete s;return nullptr;}
    for(unsigned i=0;i<s->layers;i++) {
        size_t capacity_elements=size_t(r->cfg.capacity)*s->kv_heads*s->head_dim;
        if(!encoded_kv_copy_prefix(reinterpret_cast<char*>(s->k)+i*plane,span,reinterpret_cast<char*>(r->kv_k)+i*r->kv_layer_bytes,capacity_elements,span,s->head_dim,s->format,r->stream)
            || !encoded_kv_copy_prefix(reinterpret_cast<char*>(s->v)+i*plane,span,reinterpret_cast<char*>(r->kv_v)+i*r->kv_layer_bytes,capacity_elements,span,s->head_dim,s->format,r->stream)) {
            cudaStreamSynchronize(r->stream);delete s;return nullptr;
        }
    }
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess) {delete s;return nullptr;}
    return s;
}
int rbitnet_cuda_llama_restore(void *context,const void *snapshot,unsigned length) {
    auto *r=static_cast<ResidentLlama*>(context);auto *s=static_cast<const LlamaSnapshot*>(snapshot);
    if(!r || !s || ((!r->paged || !s->paged) && s->owner!=r->identity) || !length || length>s->length || length>r->cfg.capacity
        || s->format!=r->kv_format || s->layers!=r->cfg.layers || s->kv_heads!=r->cfg.kv_heads || s->head_dim!=r->cfg.head_dim)return 1;
    NativeCallCompletion completion(r->stream);
    if(r->paged || s->paged) {
        if(!llama_paged_variant(r))return 5;
        if(!r->paged || !s->paged || !r->paged->restore(*s->paged,length,r->stream))return 4;
        r->filled=length;r->kv_poisoned=false;completion.dismiss();return 0;
    }
    size_t span=size_t(length)*s->kv_heads*s->head_dim,saved_elements=size_t(s->length)*s->kv_heads*s->head_dim;
    size_t capacity_elements=size_t(r->cfg.capacity)*s->kv_heads*s->head_dim,saved_plane=encoded_kv_plane_bytes(s->format,saved_elements,s->head_dim);
    r->kv_poisoned=r->kv_format!=0;
    for(unsigned i=0;i<s->layers;i++) {
        if(!encoded_kv_copy_prefix(reinterpret_cast<char*>(r->kv_k)+i*r->kv_layer_bytes,capacity_elements,reinterpret_cast<char*>(s->k)+i*saved_plane,saved_elements,span,s->head_dim,s->format,r->stream)
            || !encoded_kv_copy_prefix(reinterpret_cast<char*>(r->kv_v)+i*r->kv_layer_bytes,capacity_elements,reinterpret_cast<char*>(s->v)+i*saved_plane,saved_elements,span,s->head_dim,s->format,r->stream)) {
            cudaStreamSynchronize(r->stream);return 2;
        }
    }
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 3;
    r->filled=length;r->kv_poisoned=false;
    completion.dismiss();return 0;
}
int rbitnet_cuda_llama_step(void *context,const float *embedding,unsigned pos,unsigned mode,float *logits,unsigned *token) {
    auto *r=static_cast<ResidentLlama*>(context);
    if(!r || !embedding || pos>=r->cfg.capacity || mode>2 || (mode==1 && !logits) || (mode==2 && !token))return 1;
    NativeCallCompletion completion(r->stream);
    if(!llama_paged_variant(r))return 15;
    if(pos==0) {
        if(r->paged && !r->paged->reset(r->stream))return 13;
        r->filled=0;r->kv_poisoned=false;
    }
    if(pos!=r->filled)return 2;
    if(r->kv_format) {if(r->kv_poisoned)return 16;r->kv_poisoned=true;if(cudaMemsetAsync(r->kv_invalid,0,sizeof(unsigned),r->stream)!=cudaSuccess)return 16;}
    if(r->paged && !r->paged->prepare(pos,1,r->stream))return 14;
    if(cudaMemcpyAsync(r->x,embedding,r->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess || cudaMemcpyAsync(r->position,&pos,sizeof(pos),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 3;
    if(r->use_graphs) {
        if(!r->executable[mode]) {
            if(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 4;
            r->enqueue(mode);
            if(cudaStreamEndCapture(r->stream,&r->graphs[mode])!=cudaSuccess)return 5;
            if(cudaGraphInstantiate(&r->executable[mode],r->graphs[mode],0)!=cudaSuccess)return 6;
        }
        if(cudaGraphLaunch(r->executable[mode],r->stream)!=cudaSuccess)return 7;
    } else r->enqueue(mode);
    if(cudaGetLastError()!=cudaSuccess)return 8;
    if(mode==1 && cudaMemcpyAsync(logits,r->logits,r->cfg.vocab*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 9;
    if(mode==2 && cudaMemcpyAsync(token,r->token,sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 10;
    // Host input lifetimes and cancellation boundaries remain explicit, including prefill.
    unsigned invalid=0;
    if(r->kv_format && cudaMemcpyAsync(&invalid,r->kv_invalid,sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 16;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 11;
    if(invalid)return 16;r->kv_poisoned=false;
    r->filled=pos+1;
    completion.dismiss();return 0;
}
}
