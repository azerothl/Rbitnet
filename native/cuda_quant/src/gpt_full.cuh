// SPDX-License-Identifier: MIT
// Fixed-bank GPT-OSS graph. No layer activation/router traffic crosses the host.
namespace {
__device__ unsigned gpt_total_key(float value) {
    unsigned bits=__float_as_uint(value);
    return bits&0x80000000u?~bits:bits^0x80000000u;
}
__global__ void gpt_norm_ordered(float *x,const float *weights,float epsilon,unsigned n,float *y,const float *residual=nullptr) {
    __shared__ float inv;
    if(!threadIdx.x) {
        float total=0;
        for(unsigned i=0;i<n;i++) {
            float value=residual?__fadd_rn(x[i],residual[i]):x[i];
            if(residual)x[i]=value;
            total=__fadd_rn(total,__fmul_rn(value,value));
        }
        inv=1.0f/sqrtf(total/float(n)+epsilon);
    }
    __syncthreads();
    for(unsigned i=threadIdx.x;i<n;i+=256)y[i]=__fmul_rn(__fmul_rn(x[i],inv),weights[i]);
}
__global__ void gpt_router_matrix_ordered(const float *weights,size_t row_bytes,const float *x,unsigned cols,unsigned rows,unsigned lanes,float *out) {
    unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;
    if(row>=rows)return;
    const float *w=reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(weights)+size_t(row)*row_bytes);
    // Match the eight/sixteen FMA accumulators of the existing AVX2/AVX512 F32
    // router, followed by its sequential lane sum. No expert selection heuristic.
    float sum=0;
    if(lane<lanes)for(unsigned i=lane;i<cols/lanes*lanes;i+=lanes)sum=fmaf(w[i],x[i],sum);
    float total=0;
    for(unsigned i=0;i<lanes;i++)total=__fadd_rn(total,__shfl_sync(0xffffffff,sum,i));
    if(!lane) {
        for(unsigned i=cols/lanes*lanes;i<cols;i++)total=__fadd_rn(total,__fmul_rn(w[i],x[i]));
        out[row]=total;
    }
}
__global__ void gpt_router(const float *raw,const float *bias,unsigned count,unsigned used,
    float scale,unsigned *ids,float *probabilities,bool ordered=false) {
    // GPT-OSS has 32 experts and selects four. A serial selection avoids a
    // nondeterministic parallel tie reduction and is small beside expert GEMVs.
    if(threadIdx.x || blockIdx.x)return;
    for(unsigned s=0;s<used;s++) {
        unsigned best=0,key=0;bool found=false;
        for(unsigned e=0;e<count;e++) {
            bool selected=false;for(unsigned j=0;j<s;j++)selected|=ids[j]==e;
            if(selected)continue;
            // CUDA arithmetic canonicalizes NaNs; the CPU router's total_cmp
            // distinguishes their sign/payload. Preserve the source quiet NaN
            // when adding selection bias (ordinary finite arithmetic stays RN).
            float b=bias?bias[e]:0.0f;
            float selection=isnan(raw[e])?raw[e]:(isnan(b)?b:__fadd_rn(raw[e],b));
            unsigned candidate=gpt_total_key(selection);
            if(!found || candidate>key) {best=e;key=candidate;found=true;}
        }
        ids[s]=best;probabilities[s]=raw[best];
    }
    float maximum=-CUDART_INF_F;
    for(unsigned s=0;s<used;s++)maximum=fmaxf(maximum,probabilities[s]);
    float total=0;
    for(unsigned s=0;s<used;s++) {float delta=__fsub_rn(probabilities[s],maximum);probabilities[s]=ordered?float(exp(double(delta))):expf(delta);total=__fadd_rn(total,probabilities[s]);}
    for(unsigned s=0;s<used;s++)probabilities[s]=__fmul_rn(probabilities[s]/total,scale);
}
__global__ void gpt_rope(float *q,float *k,const float *frequency,const unsigned *position,
    unsigned heads,unsigned kv_heads,unsigned dim,unsigned rotary,float magnitude,const float *phases=nullptr) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,half=rotary/2;
    if(i>=(heads+kv_heads)*half)return;
    unsigned head=i/half,j=i%half;
    float *row=head<heads?q+head*dim:k+(head-heads)*dim;
    float angle=float(*position)*frequency[j],s=sinf(angle),c=cosf(angle),a=row[j],b=row[j+half];
    if(phases) {size_t offset=(size_t(*position)*half+j)*2;s=phases[offset];c=phases[offset+1];}
    row[j]=__fmul_rn(__fsub_rn(__fmul_rn(a,c),__fmul_rn(b,s)),magnitude);
    row[j+half]=__fmul_rn(__fadd_rn(__fmul_rn(a,s),__fmul_rn(b,c)),magnitude);
}
__global__ void gpt_write_kv(const float *k,const float *v,float *cache_k,float *cache_v,
    const unsigned *position,unsigned width) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<width) {size_t offset=size_t(*position)*width+i;cache_k[offset]=k[i];cache_v[offset]=v[i];}
}
__global__ void gpt_trace_layer(const float *x,const float *router,const unsigned *ids,
    unsigned embd,unsigned experts,unsigned used,float *out) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<embd)out[i]=x[i];
    if(i<experts)out[embd+i]=router[i];
    if(i<used)out[embd+experts+i]=float(ids[i]);
}
struct ResidentGpt {
    RbitnetGptConfig cfg;
    RbitnetLlamaMatrix head;
    std::vector<RbitnetGptLayer> layers;
    std::vector<void*> allocations;
    std::vector<float*> keys,values;
    float *x=nullptr,*h=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*attn=nullptr,*projection=nullptr,*router=nullptr;
    float *norm=nullptr,*frequency=nullptr,*phases=nullptr,*logits=nullptr,*maxima=nullptr,*maximum=nullptr,*scratch=nullptr;
    unsigned *position=nullptr,*ids=nullptr,*token=nullptr;
    unsigned filled=0;
    cudaStream_t stream=nullptr;
    cudaGraph_t graphs[3]={};cudaGraphExec_t executable[3]={};
    ~ResidentGpt() {
        if(stream)cudaStreamSynchronize(stream);
        for(auto p:executable)if(p)cudaGraphExecDestroy(p);
        for(auto p:graphs)if(p)cudaGraphDestroy(p);
        for(auto p:allocations)cudaFree(p);
        if(stream)cudaStreamDestroy(stream);
    }
    template<typename T> bool alloc(T *&p,size_t n,const T *host=nullptr) {
        if(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(p);
        return !host || (cudaMemcpyAsync(p,host,n*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess
            && cudaStreamSynchronize(stream)==cudaSuccess);
    }
    template<typename T> bool alloc_kv(T *&p,size_t n) {
        MemoryCategoryScope category(MemoryKv);return alloc(p,n);
    }
    bool copy(const float *&p,size_t n) {float *device=nullptr;if(!p)return false;if(!alloc(device,n,p))return false;p=device;return true;}
    void matrix(const RbitnetLlamaMatrix &m,const float *in,float *out) {
        QuantKind kind;resident_kind(m.type,kind);
        launch_quant_kernel(kind,m.weights,m.row_bytes,in,m.cols,m.rows,m.rows,out,stream);
    }
    void bias(float *out,const float *b,unsigned n) {resident_add<<<(n+255)/256,256,0,stream>>>(out,b,n);}
    void normalize(float *input,const float *norm,float *out,const float *residual=nullptr) {
        if(cfg.ordered)gpt_norm_ordered<<<1,256,0,stream>>>(input,norm,cfg.epsilon,cfg.embd,out,residual);
        else resident_norm<<<1,256,0,stream>>>(input,norm,cfg.epsilon,cfg.embd,out,residual);
    }
    void enqueue(unsigned mode,float *trace=nullptr) {
        const auto &c=cfg;unsigned qs=c.heads*c.head_dim,ks=c.kv_heads*c.head_dim;
        for(unsigned il=0;il<c.layers;il++) {
            const auto &l=layers[il];
            normalize(x,l.attn_norm,h);
            matrix(l.q,h,q);bias(q,l.q_bias,qs);
            matrix(l.k,h,k);bias(k,l.k_bias,ks);
            matrix(l.v,h,v);bias(v,l.v_bias,ks);
            if(c.rotary)gpt_rope<<<((c.heads+c.kv_heads)*(c.rotary/2)+255)/256,256,0,stream>>>(q,k,frequency,position,c.heads,c.kv_heads,c.head_dim,c.rotary,c.rope_magnitude,phases);
            gpt_write_kv<<<(ks+255)/256,256,0,stream>>>(k,v,keys[il],values[il],position,ks);
            unsigned window=il%2==0?c.window:0;
            float scale=1.0f/sqrtf(float(c.head_dim));
            if(c.split)launch_split_attention(keys[il],values[il],q,position,c.kv_heads,c.heads,c.head_dim,window,scale,c.capacity,1,scratch,attn,stream,l.sinks);
            else resident_attention<<<c.heads,128,size_t(c.capacity)*sizeof(float),stream>>>(keys[il],values[il],q,position,c.kv_heads,c.heads,c.head_dim,window,scale,attn,l.sinks);
            matrix(l.out,attn,projection);bias(projection,l.out_bias,c.embd);
            normalize(x,l.ffn_norm,h,projection);
            if(c.ordered && l.router.type==0)gpt_router_matrix_ordered<<<(c.experts+7)/8,256,0,stream>>>(static_cast<const float*>(l.router.weights),l.router.row_bytes,h,c.embd,c.experts,c.ordered,router);
            else matrix(l.router,h,router);
            bias(router,l.router_bias,c.experts);
            auto *moe=static_cast<ResidentMoe*>(l.moe);
            gpt_router<<<1,1,0,stream>>>(router,l.selection_bias,c.experts,c.used,c.weight_scale,moe->experts,moe->probabilities,c.ordered!=0);
            auto previous_stream=moe->stream;auto *previous_input=moe->input,*previous_output=moe->output;
            moe->stream=stream;moe->input=h;moe->output=projection;moe->enqueue();
            moe->stream=previous_stream;moe->input=previous_input;moe->output=previous_output;
            resident_add<<<(c.embd+255)/256,256,0,stream>>>(x,projection,c.embd);
            if(trace)gpt_trace_layer<<<(c.embd+255)/256,256,0,stream>>>(x,router,moe->experts,c.embd,c.experts,c.used,trace+size_t(il)*(c.embd+c.experts+c.used));
        }
        if(mode) {
            normalize(x,norm,h);matrix(head,h,logits);
        }
        if(mode==2) {
            unsigned blocks=(c.vocab+255)/256;
            resident_argmax<<<blocks,256,0,stream>>>(logits,nullptr,c.vocab,maxima,ids);
            resident_argmax<<<1,256,0,stream>>>(maxima,ids,blocks,maximum,token);
        }
    }
};
int gpt_step(ResidentGpt *r,const float *input,unsigned pos,unsigned mode,float *out,unsigned *id,bool hidden=false) {
    if(!r || !input || mode>2 || pos>=r->cfg.capacity || (mode==1 && !out) || (mode==2 && !id) || (hidden && !out))return 1;
    NativeCallCompletion completion(r->stream);
    if(pos==0)r->filled=0;if(pos!=r->filled)return 2;
    if(cudaMemcpyAsync(r->x,input,r->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->position,&pos,sizeof(pos),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 3;
    if(r->cfg.graphs) {
        if(!r->executable[mode]) {
            if(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 4;
            r->enqueue(mode);
            if(cudaStreamEndCapture(r->stream,&r->graphs[mode])!=cudaSuccess
                || cudaGraphInstantiate(&r->executable[mode],r->graphs[mode],0)!=cudaSuccess)return 5;
        }
        if(cudaGraphLaunch(r->executable[mode],r->stream)!=cudaSuccess)return 6;
    } else r->enqueue(mode);
    if(cudaGetLastError()!=cudaSuccess
        || (hidden && cudaMemcpyAsync(out,r->x,r->cfg.embd*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)
        || (mode==1 && cudaMemcpyAsync(out,r->logits,r->cfg.vocab*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)
        || (mode==2 && cudaMemcpyAsync(id,r->token,sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)
        || cudaStreamSynchronize(r->stream)!=cudaSuccess)return 7;
    r->filled=pos+1;completion.dismiss();return 0;
}
}
extern "C" {
void *rbitnet_cuda_gpt_full_create(const RbitnetGptConfig *c,const RbitnetGptLayer *layers,
    const RbitnetLlamaMatrix *head,const float *norm,const float *frequency) {
    if(!c || !layers || !head || !norm || (c->rotary && !frequency) || !c->embd || c->embd>32768
        || !c->vocab || c->vocab>1048576 || !c->layers || c->layers>256 || !c->heads || c->heads>128
        || !c->kv_heads || c->heads%c->kv_heads || !c->head_dim || c->head_dim>512 || c->rotary>c->head_dim
        || c->rotary%2 || !c->capacity || c->capacity>8192 || !c->experts || c->experts>128
        || !c->used || c->used>c->experts || c->graphs>1 || c->split>1 || (c->ordered!=0 && c->ordered!=8 && c->ordered!=16)
        || !isfinite(c->epsilon) || c->epsilon<=0 || !isfinite(c->rope_magnitude) || c->rope_magnitude<=0
        || !isfinite(c->weight_scale) || !qwen_matrix_valid(*head,c->embd,c->vocab))return nullptr;
    unsigned qs=c->heads*c->head_dim,ks=c->kv_heads*c->head_dim;
    for(unsigned i=0;i<c->layers;i++) {
        const auto &l=layers[i];auto *m=static_cast<ResidentMoe*>(l.moe);
        if(!m || m->dynamic || m->cfg.embd!=c->embd || m->cfg.experts!=c->experts || m->cfg.used!=c->used || !m->cfg.oai
            || !l.attn_norm || !l.ffn_norm || !l.q_bias || !l.k_bias || !l.v_bias || !l.out_bias || !l.router_bias || !l.sinks
            || !qwen_matrix_valid(l.q,c->embd,qs) || !qwen_matrix_valid(l.k,c->embd,ks) || !qwen_matrix_valid(l.v,c->embd,ks)
            || !qwen_matrix_valid(l.out,qs,c->embd) || !qwen_matrix_valid(l.router,c->embd,c->experts))return nullptr;
        // Complete any work on the borrowed stream before capturing on ours.
        if(cudaStreamSynchronize(m->stream)!=cudaSuccess)return nullptr;
    }
    auto *r=new(std::nothrow) ResidentGpt;if(!r)return nullptr;r->cfg=*c;r->head=*head;r->layers.assign(layers,layers+c->layers);
    unsigned maxima=(c->vocab+255)/256;
    if(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking)!=cudaSuccess || !r->alloc(r->x,c->embd) || !r->alloc(r->h,c->embd)
        || !r->alloc(r->q,qs) || !r->alloc(r->k,ks) || !r->alloc(r->v,ks) || !r->alloc(r->attn,qs)
        || !r->alloc(r->projection,c->embd) || !r->alloc(r->router,c->experts) || !r->alloc(r->norm,c->embd,norm)
        || (c->rotary && !r->alloc(r->frequency,c->rotary/2,frequency)) || !r->alloc(r->position,1)
        || !r->alloc(r->logits,c->vocab) || !r->alloc(r->maxima,maxima) || !r->alloc(r->ids,maxima)
        || !r->alloc(r->maximum,1) || !r->alloc(r->token,1)
        || (c->split && !r->alloc(r->scratch,size_t(c->heads)*((c->capacity+attention_tile-1)/attention_tile)*(c->head_dim+2)))) {delete r;return nullptr;}
    if(c->ordered && c->rotary) {
        std::vector<float> phase(size_t(c->capacity)*c->rotary);
        for(unsigned pos=0;pos<c->capacity;pos++)for(unsigned i=0;i<c->rotary/2;i++) {
            float angle=float(pos)*frequency[i];size_t offset=(size_t(pos)*(c->rotary/2)+i)*2;
            phase[offset]=sinf(angle);phase[offset+1]=cosf(angle);
        }
        if(!r->alloc(r->phases,phase.size(),phase.data())) {delete r;return nullptr;}
    }
    for(auto &l:r->layers) {
        float *key=nullptr,*value=nullptr;
        if(!r->alloc_kv(key,size_t(c->capacity)*ks) || !r->alloc_kv(value,size_t(c->capacity)*ks)
            || !r->copy(l.attn_norm,c->embd) || !r->copy(l.ffn_norm,c->embd) || !r->copy(l.q_bias,qs)
            || !r->copy(l.k_bias,ks) || !r->copy(l.v_bias,ks) || !r->copy(l.out_bias,c->embd)
            || !r->copy(l.router_bias,c->experts) || !r->copy(l.sinks,c->heads)
            || (l.selection_bias && !r->copy(l.selection_bias,c->experts))) {delete r;return nullptr;}
        r->keys.push_back(key);r->values.push_back(value);
    }
    return r;
}
void rbitnet_cuda_gpt_full_destroy(void *p) {delete static_cast<ResidentGpt*>(p);}
int rbitnet_cuda_gpt_full_step(void *p,const float *input,unsigned pos,unsigned mode,float *out,unsigned *id) {
    return gpt_step(static_cast<ResidentGpt*>(p),input,pos,mode,out,id);
}
int rbitnet_cuda_gpt_full_hidden_check(void *p,const float *input,unsigned pos,float *out) {
    return gpt_step(static_cast<ResidentGpt*>(p),input,pos,0,out,nullptr,true);
}
int rbitnet_cuda_gpt_full_layers_check(void *p,const float *input,unsigned pos,float *out) {
    auto *r=static_cast<ResidentGpt*>(p);if(!r || !input || !out || pos>=r->cfg.capacity || (pos && pos!=r->filled))return 1;
    if(!pos)r->filled=0;
    float *trace=nullptr;size_t bytes=size_t(r->cfg.layers)*(r->cfg.embd+r->cfg.experts+r->cfg.used)*sizeof(float);
    if(cudaMalloc(reinterpret_cast<void**>(&trace),bytes)!=cudaSuccess)return 2;
    int status=0;
    if(cudaMemcpyAsync(r->x,input,r->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->position,&pos,sizeof(pos),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)status=3;
    if(!status) {
        r->enqueue(0,trace);
        if(cudaGetLastError()!=cudaSuccess || cudaMemcpyAsync(out,trace,bytes,cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess
            || cudaStreamSynchronize(r->stream)!=cudaSuccess)status=4;
    }
    cudaFree(trace);if(!status)r->filled=pos+1;return status;
}
int rbitnet_cuda_gpt_router_check(const float *raw,const float *bias,unsigned count,unsigned used,float scale,unsigned *ids,float *probs) {
    if(!raw || !ids || !probs || !count || count>128 || !used || used>count || !isfinite(scale))return 1;
    ResidentGpt r;
    float *device_raw=nullptr,*device_bias=nullptr,*device_prob=nullptr;unsigned *device_ids=nullptr;
    if(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking)!=cudaSuccess || !r.alloc(device_raw,count,raw)
        || (bias && !r.alloc(device_bias,count,bias)) || !r.alloc(device_prob,used) || !r.alloc(device_ids,used))return 2;
    gpt_router<<<1,1,0,r.stream>>>(device_raw,device_bias,count,used,scale,device_ids,device_prob);
    if(cudaGetLastError()!=cudaSuccess || cudaMemcpyAsync(ids,device_ids,used*sizeof(unsigned),cudaMemcpyDeviceToHost,r.stream)!=cudaSuccess
        || cudaMemcpyAsync(probs,device_prob,used*sizeof(float),cudaMemcpyDeviceToHost,r.stream)!=cudaSuccess
        || cudaStreamSynchronize(r.stream)!=cudaSuccess)return 3;return 0;
}
}
