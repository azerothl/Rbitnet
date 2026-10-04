// SPDX-License-Identifier: MIT
// Private streams and state per runtime: no mutable model state is process-global.
#include <vector>
#include <new>
#include "split_attention.cuh"

namespace {
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
    unsigned kv_heads,unsigned heads,unsigned dim,unsigned window,float scale,float *y) {
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
    __syncthreads();
    float sum=0;
    for(unsigned i=tid;i<seq-first;i+=128) {scores[i]=expf(scores[i]-maximum);sum+=scores[i];}
    for(int shift=16;shift;shift/=2)sum+=__shfl_down_sync(0xffffffff,sum,shift);
    if(lane==0)reductions[warp]=sum;
    __syncthreads();sum=reductions[0]+reductions[1]+reductions[2]+reductions[3];
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
struct ResidentLlama {
    LlamaBlock *block=nullptr;
    RbitnetLlamaConfig cfg;
    std::vector<RbitnetLlamaLayer> layers;
    RbitnetLlamaMatrix output;
    std::vector<void*> allocations;
    float *x=nullptr,*h=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*attn=nullptr,*projection=nullptr,*gate=nullptr,*up=nullptr,*logits=nullptr,*kv_k=nullptr,*kv_v=nullptr,*out_norm=nullptr,*frequency=nullptr,*maxima=nullptr,*maximum=nullptr;
    unsigned *position=nullptr,*ids=nullptr,*token=nullptr;
    float *attention_scratch=nullptr,*block_attention_scratch=nullptr;
    size_t attention_scratch_size=0;
    bool split_kv=false;
    bool tf32_prefill=false;
    unsigned tensor_gemm_calls=0;
    unsigned filled=0;
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
        if(stream)cudaStreamDestroy(stream);
    }
    template<typename T> bool alloc(T *&ptr,size_t count,const T *host=nullptr) {
        if(cudaMalloc(reinterpret_cast<void**>(&ptr),count*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(ptr);
        return !host || (cudaMemcpyAsync(ptr,host,count*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess
            && cudaStreamSynchronize(stream)==cudaSuccess);
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
            resident_rope_kv<<<((cfg.heads+cfg.kv_heads)*(cfg.head_dim/2)+255)/256,256,0,stream>>>(q,k,v,kv_k+il*layer_stride,kv_v+il*layer_stride,frequency,position,cfg.heads,cfg.kv_heads,cfg.head_dim,cfg.rotary);
            if(split_kv)launch_split_attention(kv_k+il*layer_stride,kv_v+il*layer_stride,q,position,cfg.kv_heads,cfg.heads,cfg.head_dim,cfg.window,1.0f/sqrtf(float(cfg.head_dim)),cfg.capacity,1,attention_scratch,attn,stream);
            else resident_attention<<<cfg.heads,128,cfg.capacity*sizeof(float),stream>>>(kv_k+il*layer_stride,kv_v+il*layer_stride,q,position,cfg.kv_heads,cfg.heads,cfg.head_dim,cfg.window,1.0f/sqrtf(float(cfg.head_dim)),attn);
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
struct LlamaSnapshot {
    unsigned layers=0,kv_heads=0,head_dim=0,length=0;
    float *k=nullptr,*v=nullptr;
    ~LlamaSnapshot() {if(k)cudaFree(k);if(v)cudaFree(v);}
};
}

extern "C" {
void *rbitnet_cuda_llama_create(const RbitnetLlamaConfig *cfg,const RbitnetLlamaLayer *layers,
    const RbitnetLlamaMatrix *output,const float *out_norm,const float *frequency) {
    if(!cfg || !layers || !output || !out_norm || !frequency || !cfg->embd || !cfg->ffn || !cfg->vocab || !cfg->layers || !cfg->heads || !cfg->kv_heads || cfg->heads%cfg->kv_heads || cfg->head_dim%2 || cfg->heads*cfg->head_dim!=cfg->embd || !cfg->rotary || cfg->rotary%2 || cfg->rotary>cfg->head_dim || !cfg->capacity || cfg->capacity>8192)return nullptr;
    auto *r=new(std::nothrow) ResidentLlama;
    if(!r)return nullptr;
    r->cfg=*cfg;r->output=*output;r->layers.assign(layers,layers+cfg->layers);r->use_graphs=cfg->graphs!=0;
    r->split_kv=split_attention_enabled();
    r->tf32_prefill=tf32_prefill_requested();
    r->attention_scratch_size=size_t(cfg->heads)*((cfg->capacity+attention_tile-1)/attention_tile)*(size_t(cfg->head_dim)+2);
    QuantKind kind;
    if(!resident_kind(output->type,kind)) {delete r;return nullptr;}
    for(auto &l:r->layers)for(auto m:{l.q,l.k,l.v,l.out,l.gate,l.up,l.down})if(!m.weights || !resident_kind(m.type,kind)) {delete r;return nullptr;}
    if(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking)!=cudaSuccess) {delete r;return nullptr;}
    size_t kv=size_t(cfg->layers)*cfg->capacity*cfg->kv_heads*cfg->head_dim;
    if(!r->alloc(r->x,cfg->embd) || !r->alloc(r->h,cfg->embd) || !r->alloc(r->q,cfg->embd)
        || !r->alloc(r->k,cfg->kv_heads*cfg->head_dim) || !r->alloc(r->v,cfg->kv_heads*cfg->head_dim)
        || !r->alloc(r->attn,cfg->embd) || !r->alloc(r->projection,cfg->embd) || !r->alloc(r->gate,cfg->ffn)
        || !r->alloc(r->up,cfg->ffn) || !r->alloc(r->logits,cfg->vocab) || !r->alloc(r->kv_k,kv) || !r->alloc(r->kv_v,kv)
        || !r->alloc(r->out_norm,cfg->embd,out_norm) || !r->alloc(r->frequency,cfg->rotary/2,frequency)
        || !r->alloc(r->position,1) || !r->alloc(r->maxima,(cfg->vocab+255)/256) || !r->alloc(r->ids,(cfg->vocab+255)/256)
        || !r->alloc(r->maximum,1) || !r->alloc(r->token,1)) {delete r;return nullptr;}
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
void rbitnet_cuda_llama_destroy(void *context) {delete static_cast<ResidentLlama*>(context);}
unsigned rbitnet_cuda_llama_split_attention_layers(void *p) {
    auto *r=static_cast<ResidentLlama*>(p);return r && r->split_kv?r->cfg.layers:0;
}
unsigned rbitnet_cuda_llama_tensor_gemm_calls(void *p) {
    auto *r=static_cast<ResidentLlama*>(p);return r?r->tensor_gemm_calls:0;
}
int rbitnet_cuda_llama_configure_tensor_prefill(void *p,unsigned enabled) {
    auto *r=static_cast<ResidentLlama*>(p);if(!r || enabled>1 || r->block)return 1;
    r->tf32_prefill=enabled && tf32_prefill_supported();return 0;
}
static int llama_prefill_impl(void *context,const float *embeddings,unsigned pos,unsigned count,
    unsigned mode,float *logits,unsigned *token,bool all) {
    auto *r=static_cast<ResidentLlama*>(context);
    if(!r || !embeddings || !count || count>LlamaBlock::capacity || pos>=r->cfg.capacity
        || count>r->cfg.capacity-pos || mode>2 || (mode==1 && !logits) || (mode==2 && !token)
        || (all && (!mode || count>LlamaBlock::verify_capacity)))return 1;
    if(pos==0)r->filled=0;if(pos!=r->filled)return 2;
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
    for(const auto &layer:r->layers)for(const auto &m:{layer.q,layer.k,layer.v,layer.out,layer.gate,layer.up,layer.down})tensor_calls+=uses_tensor(m);
    if(mode && all)tensor_calls+=uses_tensor(r->output);
    bool use_graph=all && r->use_graphs;
    if(!use_graph || !r->verify_executable[mode-1][count]) {
    if(use_graph && cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 9;
    auto matrix=[&](const RbitnetLlamaMatrix &m,const float *x,float *y) {
        QuantKind kind;resident_kind(m.type,kind);
        launch_prefill_gemm(kind,m.weights,m.row_bytes,x,m.cols,m.rows,count,y,r->stream,uses_tensor(m));
    };
    size_t layer_stride=size_t(stride)*c.capacity;
    resident_norm<<<count,256,0,r->stream>>>(p[0],r->layers[0].attn_norm,c.epsilon,c.embd,p[1]);
    for(unsigned il=0;il<c.layers;il++) {
        const auto &l=r->layers[il];matrix(l.q,p[1],p[2]);matrix(l.k,p[1],p[3]);matrix(l.v,p[1],p[4]);
        dim3 rope(((c.heads+c.kv_heads)*(c.head_dim/2)+255)/256,count);
        resident_rope_kv<<<rope,256,0,r->stream>>>(p[2],p[3],p[4],r->kv_k+il*layer_stride,r->kv_v+il*layer_stride,r->frequency,r->position,c.heads,c.kv_heads,c.head_dim,c.rotary);
        if(r->split_kv)launch_split_attention(r->kv_k+il*layer_stride,r->kv_v+il*layer_stride,p[2],r->position,c.kv_heads,c.heads,c.head_dim,c.window,1.0f/sqrtf(float(c.head_dim)),c.capacity,count,r->block_attention_scratch,p[5],r->stream);
        else resident_attention<<<dim3(c.heads,count),128,c.capacity*sizeof(float),r->stream>>>(r->kv_k+il*layer_stride,r->kv_v+il*layer_stride,p[2],r->position,c.kv_heads,c.heads,c.head_dim,c.window,1.0f/sqrtf(float(c.head_dim)),p[5]);
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
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 8;
    r->filled=pos+count;r->tensor_gemm_calls=tensor_calls;return 0;
}
int rbitnet_cuda_llama_prefill(void *context,const float *embeddings,unsigned pos,unsigned count,unsigned mode,float *logits,unsigned *token) {
    return llama_prefill_impl(context,embeddings,pos,count,mode,logits,token,false);
}
int rbitnet_cuda_llama_verify(void *context,const float *embeddings,unsigned pos,unsigned count,unsigned mode,float *logits,unsigned *tokens) {
    return llama_prefill_impl(context,embeddings,pos,count,mode,logits,tokens,true);
}
int rbitnet_cuda_llama_truncate(void *context,unsigned length) {
    auto *r=static_cast<ResidentLlama*>(context);
    if(!r || length>r->filled)return 1;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 2;
    // Dense causal attention never reads the discarded tail. Later writes replace it.
    r->filled=length;return 0;
}
void rbitnet_cuda_llama_snapshot_destroy(void *snapshot) {delete static_cast<LlamaSnapshot*>(snapshot);}
void *rbitnet_cuda_llama_snapshot(void *context,unsigned length) {
    auto *r=static_cast<ResidentLlama*>(context);
    if(!r || !length || length>r->filled)return nullptr;
    auto *s=new(std::nothrow) LlamaSnapshot;if(!s)return nullptr;
    s->layers=r->cfg.layers;s->kv_heads=r->cfg.kv_heads;s->head_dim=r->cfg.head_dim;s->length=length;
    size_t stride=size_t(s->kv_heads)*s->head_dim,span=size_t(length)*stride,bytes=span*s->layers*sizeof(float);
    if(cudaMalloc(reinterpret_cast<void**>(&s->k),bytes)!=cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&s->v),bytes)!=cudaSuccess) {delete s;return nullptr;}
    for(unsigned i=0;i<s->layers;i++) {
        size_t src=size_t(i)*r->cfg.capacity*stride,dst=size_t(i)*span;
        if(cudaMemcpyAsync(s->k+dst,r->kv_k+src,span*sizeof(float),cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess
            || cudaMemcpyAsync(s->v+dst,r->kv_v+src,span*sizeof(float),cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess) {
            cudaStreamSynchronize(r->stream);delete s;return nullptr;
        }
    }
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess) {delete s;return nullptr;}
    return s;
}
int rbitnet_cuda_llama_restore(void *context,const void *snapshot,unsigned length) {
    auto *r=static_cast<ResidentLlama*>(context);auto *s=static_cast<const LlamaSnapshot*>(snapshot);
    if(!r || !s || !length || length>s->length || length>r->cfg.capacity
        || s->layers!=r->cfg.layers || s->kv_heads!=r->cfg.kv_heads || s->head_dim!=r->cfg.head_dim)return 1;
    size_t stride=size_t(s->kv_heads)*s->head_dim,bytes=size_t(length)*stride*sizeof(float);
    for(unsigned i=0;i<s->layers;i++) {
        size_t src=size_t(i)*s->length*stride,dst=size_t(i)*r->cfg.capacity*stride;
        if(cudaMemcpyAsync(r->kv_k+dst,s->k+src,bytes,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess
            || cudaMemcpyAsync(r->kv_v+dst,s->v+src,bytes,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess) {
            cudaStreamSynchronize(r->stream);return 2;
        }
    }
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 3;
    r->filled=length;
    return 0;
}
int rbitnet_cuda_llama_step(void *context,const float *embedding,unsigned pos,unsigned mode,float *logits,unsigned *token) {
    auto *r=static_cast<ResidentLlama*>(context);
    if(!r || !embedding || pos>=r->cfg.capacity || mode>2 || (mode==1 && !logits) || (mode==2 && !token))return 1;
    if(pos==0)r->filled=0;
    if(pos!=r->filled)return 2;
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
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 11;
    r->filled=pos+1;
    return 0;
}
}
