// SPDX-License-Identifier: MIT
// Private streams and state per runtime: no mutable model state is process-global.
#include <vector>
#include <new>

namespace {
__global__ void resident_norm(float *x,const float *weights,float epsilon,unsigned n,float *y,const float *residual=nullptr) {
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
    unsigned pos=*position, pairs=dim/2;
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
    const unsigned head=blockIdx.x,kh=head/(heads/kv_heads),seq=*position+1;
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
    RbitnetLlamaConfig cfg;
    std::vector<RbitnetLlamaLayer> layers;
    RbitnetLlamaMatrix output;
    std::vector<void*> allocations;
    float *x=nullptr,*h=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*attn=nullptr,*projection=nullptr,*gate=nullptr,*up=nullptr,*logits=nullptr,*kv_k=nullptr,*kv_v=nullptr,*out_norm=nullptr,*frequency=nullptr,*maxima=nullptr,*maximum=nullptr;
    unsigned *position=nullptr,*ids=nullptr,*token=nullptr;
    unsigned filled=0;
    cudaStream_t stream=nullptr;
    cudaGraph_t graphs[3]={};
    cudaGraphExec_t executable[3]={};
    bool use_graphs=true;
    ~ResidentLlama() {
        if(stream)cudaStreamSynchronize(stream);
        for(auto exec:executable)if(exec)cudaGraphExecDestroy(exec);
        for(auto graph:graphs)if(graph)cudaGraphDestroy(graph);
        for(auto p:allocations)cudaFree(p);
        if(stream)cudaStreamDestroy(stream);
    }
    template<typename T> bool alloc(T *&ptr,size_t count,const T *host=nullptr) {
        if(cudaMalloc(reinterpret_cast<void**>(&ptr),count*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(ptr);
        return !host || cudaMemcpy(ptr,host,count*sizeof(T),cudaMemcpyHostToDevice)==cudaSuccess;
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
            resident_attention<<<cfg.heads,128,cfg.capacity*sizeof(float),stream>>>(kv_k+il*layer_stride,kv_v+il*layer_stride,q,position,cfg.kv_heads,cfg.heads,cfg.head_dim,cfg.window,1.0f/sqrtf(float(cfg.head_dim)),attn);
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
}

extern "C" {
void *rbitnet_cuda_llama_create(const RbitnetLlamaConfig *cfg,const RbitnetLlamaLayer *layers,
    const RbitnetLlamaMatrix *output,const float *out_norm,const float *frequency) {
    if(!cfg || !layers || !output || !out_norm || !frequency || !cfg->embd || !cfg->ffn || !cfg->vocab || !cfg->layers || !cfg->heads || !cfg->kv_heads || cfg->heads%cfg->kv_heads || cfg->head_dim%2 || cfg->heads*cfg->head_dim!=cfg->embd || !cfg->rotary || cfg->rotary%2 || cfg->rotary>cfg->head_dim || !cfg->capacity || cfg->capacity>8192)return nullptr;
    auto *r=new(std::nothrow) ResidentLlama;
    if(!r)return nullptr;
    r->cfg=*cfg;r->output=*output;r->layers.assign(layers,layers+cfg->layers);r->use_graphs=cfg->graphs!=0;
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
