// SPDX-License-Identifier: MIT
// Segmented MLA graph with leased experts and a routed CPU fallback.
// Compressed MLA stores [normalized latent, rotary key] once per position.
namespace {
__global__ void mla_rope_queries(float *q,const float *phase,const unsigned *position,
    unsigned heads,unsigned dim,unsigned rotary,float magnitude) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,half=rotary/2;
    if(i>=heads*half)return;
    unsigned head=i/half,j=i%half;float *row=q+size_t(head)*dim+dim-rotary;
    size_t offset=(size_t(*position)*half+j)*2;
    float s=phase[offset],c=phase[offset+1],a=row[2*j],b=row[2*j+1];
    row[2*j]=__fmul_rn(__fsub_rn(__fmul_rn(a,c),__fmul_rn(b,s)),magnitude);
    row[2*j+1]=__fmul_rn(__fadd_rn(__fmul_rn(a,s),__fmul_rn(b,c)),magnitude);
}
__global__ void mla_router(const float *raw,const float *bias,unsigned count,unsigned used,
    unsigned groups,unsigned groups_used,unsigned sigmoid,unsigned normalize,float scale,
    unsigned *ids,float *probabilities) {
    if(threadIdx.x || blockIdx.x)return;
    float p[128],selection[128],group_scores[128];unsigned selected_groups[128];
    if(sigmoid) {
        for(unsigned e=0;e<count;e++)p[e]=1.0f/__fadd_rn(1.0f,float(exp(double(-raw[e]))));
    } else {
        float maximum=-CUDART_INF_F;
        for(unsigned e=0;e<count;e++)maximum=fmaxf(maximum,raw[e]);
        float total=0;
        for(unsigned e=0;e<count;e++) {p[e]=float(exp(double(__fsub_rn(raw[e],maximum))));total=__fadd_rn(total,p[e]);}
        for(unsigned e=0;e<count;e++)p[e]/=total;
    }
    for(unsigned e=0;e<count;e++) {
        float b=bias?bias[e]:0.0f;
        selection[e]=isnan(p[e])?p[e]:(isnan(b)?b:__fadd_rn(p[e],b));
    }
    if(groups>1) {
        unsigned width=count/groups;
        for(unsigned g=0;g<groups;g++) {
            const unsigned begin=g*width,end=begin+width;
            unsigned first=begin,second=begin;
            for(unsigned e=begin+1;e<end;e++)if(gpt_total_key(selection[e])>gpt_total_key(selection[first]))first=e;
            bool found=false;
            for(unsigned e=g*width;e<(g+1)*width;e++)if(e!=first && (!found || gpt_total_key(selection[e])>gpt_total_key(selection[second]))) {second=e;found=true;}
            group_scores[g]=__fadd_rn(selection[first],found?selection[second]:0.0f);
        }
        for(unsigned s=0;s<groups_used;s++) {
            unsigned best=0;bool found=false;
            for(unsigned g=0;g<groups;g++) {
                bool seen=false;for(unsigned j=0;j<s;j++)seen|=selected_groups[j]==g;
                if(!seen && (!found || gpt_total_key(group_scores[g])>gpt_total_key(group_scores[best]))) {best=g;found=true;}
            }
            selected_groups[s]=best;
        }
        for(unsigned g=0;g<groups;g++) {
            bool allowed=false;for(unsigned s=0;s<groups_used;s++)allowed|=selected_groups[s]==g;
            if(!allowed)for(unsigned e=g*width;e<(g+1)*width;e++)selection[e]=-CUDART_INF_F;
        }
    }
    for(unsigned s=0;s<used;s++) {
        unsigned best=0;bool found=false;
        for(unsigned e=0;e<count;e++) {
            bool seen=false;for(unsigned j=0;j<s;j++)seen|=ids[j]==e;
            if(!seen && (!found || gpt_total_key(selection[e])>gpt_total_key(selection[best]))) {best=e;found=true;}
        }
        ids[s]=best;probabilities[s]=p[best];
    }
    float total=0;for(unsigned s=0;s<used;s++)total=__fadd_rn(total,probabilities[s]);
    total=fmaxf(total,1.0f/16384.0f);
    for(unsigned s=0;s<used;s++)probabilities[s]=__fmul_rn(normalize?probabilities[s]/total:probabilities[s],scale);
}
__global__ void mla_write_latent(const float *normalized,const float *kv,const float *phase,
    const unsigned *position,unsigned rank,unsigned rotary,float magnitude,float *cache) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    float *out=cache+size_t(*position)*(rank+rotary);
    if(i<rank)out[i]=normalized[i];
    if(i<rotary/2) {
        size_t offset=(size_t(*position)*(rotary/2)+i)*2;
        float s=phase[offset],c=phase[offset+1],a=kv[rank+2*i],b=kv[rank+2*i+1];
        out[rank+2*i]=__fmul_rn(__fsub_rn(__fmul_rn(a,c),__fmul_rn(b,s)),magnitude);
        out[rank+2*i+1]=__fmul_rn(__fadd_rn(__fmul_rn(a,s),__fmul_rn(b,c)),magnitude);
    }
}
template<QuantKind kind>
__global__ void mla_head_matrix(const uint8_t *weights,size_t row_bytes,unsigned cols,unsigned rows,
    const float *x,unsigned input_stride,float *y,unsigned output_stride) {
    unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/32,head=blockIdx.y;
    if(row>=rows)return;
    const uint8_t *w=weights+(size_t(head)*rows+row)*row_bytes;
    float dot=quant_row_dot<kind>(w,x+size_t(head)*input_stride,cols);
    if(!(threadIdx.x&31))y[size_t(head)*output_stride+row]=dot;
}
void mla_launch_head(const RbitnetLlamaMatrix &m,unsigned heads,unsigned rows,const float *x,
    unsigned input_stride,float *y,unsigned output_stride,cudaStream_t stream) {
    QuantKind kind;resident_kind(m.type,kind);dim3 grid((rows+7)/8,heads);
#define MLA_HEAD(K) case QuantKind::K: mla_head_matrix<QuantKind::K><<<grid,256,0,stream>>>(static_cast<const uint8_t*>(m.weights),m.row_bytes,m.cols,rows,x,input_stride,y,output_stride);break
    switch(kind) {MLA_HEAD(F32);MLA_HEAD(Q4_0);MLA_HEAD(Q5_0);MLA_HEAD(Q8_0);MLA_HEAD(Q4_K);MLA_HEAD(Q5_K);MLA_HEAD(Q6_K);MLA_HEAD(MXFP4);}
#undef MLA_HEAD
}
__global__ void mla_query_tail(const float *q,unsigned heads,unsigned dim,unsigned rotary,unsigned rank,float *absorbed) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<heads*rotary)absorbed[size_t(i/rotary)*(rank+rotary)+rank+i%rotary]=q[size_t(i/rotary)*dim+dim-rotary+i%rotary];
}
// Different key and value widths, both read from the same compressed cache.
__global__ void mla_attention_partials(const float *cache,const float *q,const unsigned *position,
    unsigned heads,unsigned rank,unsigned rotary,unsigned parts,float scale,float *scratch) {
    __shared__ float scores[attention_tile],reductions[4];
    unsigned tid=threadIdx.x,lane=tid&31,warp=tid/32,head=blockIdx.x,part=blockIdx.y;
    unsigned width=rank+rotary,seq=*position+1,begin=part*attention_tile,end=min(seq,begin+attention_tile);
    float *out=scratch+(size_t(head)*parts+part)*(rank+2);
    if(begin>=end) {if(!tid) {out[0]=-CUDART_INF_F;out[1]=0;}return;}
    q+=size_t(head)*width;
    for(unsigned p=begin+warp;p<end;p+=4) {
        float dot=0;for(unsigned i=lane;i<width;i+=32)dot=fmaf(q[i],cache[size_t(p)*width+i],dot);
        for(int shift=16;shift;shift/=2)dot+=__shfl_down_sync(0xffffffff,dot,shift);
        if(!lane)scores[p-begin]=dot*scale;
    }
    __syncthreads();float maximum=-CUDART_INF_F;
    for(unsigned p=tid;p<end-begin;p+=128)maximum=fmaxf(maximum,scores[p]);
    for(int shift=16;shift;shift/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,shift));
    if(!lane)reductions[warp]=maximum;
    __syncthreads();maximum=fmaxf(fmaxf(reductions[0],reductions[1]),fmaxf(reductions[2],reductions[3]));
    __syncthreads();float sum=0;
    for(unsigned p=tid;p<end-begin;p+=128) {scores[p]=expf(scores[p]-maximum);sum+=scores[p];}
    for(int shift=16;shift;shift/=2)sum+=__shfl_down_sync(0xffffffff,sum,shift);
    if(!lane)reductions[warp]=sum;
    __syncthreads();sum=reductions[0]+reductions[1]+reductions[2]+reductions[3];
    if(!tid) {out[0]=maximum;out[1]=sum;}
    for(unsigned i=tid;i<rank;i+=128) {
        float value=0;
        for(unsigned p=begin;p<end;p++)value=fmaf(scores[p-begin],cache[size_t(p)*width+i],value);
        out[i+2]=value;
    }
}
__global__ void mla_attention(const float *cache,const float *q,const unsigned *position,
    unsigned rank,unsigned rotary,float scale,float *out) {
    extern __shared__ float scores[];__shared__ float reductions[4];
    unsigned tid=threadIdx.x,lane=tid&31,warp=tid/32,head=blockIdx.x,seq=*position+1,width=rank+rotary;
    q+=size_t(head)*width;
    for(unsigned p=warp;p<seq;p+=4) {
        float dot=0;for(unsigned i=lane;i<width;i+=32)dot=fmaf(q[i],cache[size_t(p)*width+i],dot);
        for(int shift=16;shift;shift/=2)dot+=__shfl_down_sync(0xffffffff,dot,shift);
        if(!lane)scores[p]=dot*scale;
    }
    __syncthreads();float maximum=-CUDART_INF_F;
    for(unsigned p=tid;p<seq;p+=128)maximum=fmaxf(maximum,scores[p]);
    for(int shift=16;shift;shift/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,shift));
    if(!lane)reductions[warp]=maximum;
    __syncthreads();maximum=fmaxf(fmaxf(reductions[0],reductions[1]),fmaxf(reductions[2],reductions[3]));
    __syncthreads();float sum=0;
    for(unsigned p=tid;p<seq;p+=128) {scores[p]=expf(scores[p]-maximum);sum+=scores[p];}
    for(int shift=16;shift;shift/=2)sum+=__shfl_down_sync(0xffffffff,sum,shift);
    if(!lane)reductions[warp]=sum;
    __syncthreads();sum=reductions[0]+reductions[1]+reductions[2]+reductions[3];
    out+=size_t(head)*rank;
    for(unsigned i=tid;i<rank;i+=128) {
        float value=0;for(unsigned p=0;p<seq;p++)value=fmaf(scores[p]/sum,cache[size_t(p)*width+i],value);
        out[i]=value;
    }
}
uint64_t mla_identity() {
    static std::mutex mutex;static uint64_t next=0;
    std::lock_guard<std::mutex> lock(mutex);
    if(next==std::numeric_limits<uint64_t>::max())return 0;
    return ++next;
}
struct MlaSnapshot {
    uint64_t owner=0;
    unsigned layers=0,width=0,length=0;
    float *data=nullptr;
    ~MlaSnapshot() {if(data)cudaFree(data);}
};
void mla_block_destroy(void *block);
struct ResidentMla {
    void *block=nullptr;
    RbitnetMlaConfig cfg;
    std::vector<RbitnetMlaLayer> layers;
    RbitnetLlamaMatrix head;
    std::vector<void*> allocations;
    std::vector<float*> cache;
    std::vector<cudaGraph_t> graphs;
    std::vector<cudaGraphExec_t> executable;
    float *x=nullptr,*h=nullptr,*qa=nullptr,*qa_normed=nullptr,*q=nullptr,*kv=nullptr,*latent=nullptr;
    float *absorbed=nullptr,*values=nullptr,*attended=nullptr,*projection=nullptr,*shared=nullptr,*sg=nullptr,*su=nullptr;
    float *router=nullptr,*probabilities=nullptr,*scratch=nullptr,*phases=nullptr,*norm=nullptr,*logits=nullptr;
    float *maxima=nullptr,*maximum=nullptr;
    unsigned *ids=nullptr,*token=nullptr,*position=nullptr;
    uint64_t identity=mla_identity();
    unsigned filled=0,next_layer=0,host_position=0;bool prepared=false,token_started=false,output_ready=false;
    cudaStream_t stream=nullptr;
    ~ResidentMla() {
        if(stream)cudaStreamSynchronize(stream);
        mla_block_destroy(block);
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
    bool copy(const float *&p,size_t n) {float *device=nullptr;if(!p || !alloc(device,n,p))return false;p=device;return true;}
    void matrix(const RbitnetLlamaMatrix &m,const float *input,float *output) {
        QuantKind kind;resident_kind(m.type,kind);
        launch_quant_kernel(kind,m.weights,m.row_bytes,input,m.cols,m.rows,m.rows,output,stream);
    }
    void normalize(float *input,const float *weight,unsigned width,float *output,const float *residual=nullptr) {
        // Stage independent loads/products in parallel, retaining the original
        // scalar RN sum and the large-width fallback used by GPT-OSS.
        launch_gpt_ordered_norm(input,weight,cfg.epsilon,width,output,residual,1,stream);
    }
    void enqueue_prepare(unsigned il) {
        const auto &c=cfg;const auto &l=layers[il];unsigned width=c.rank+c.rotary;
        normalize(x,l.attn_norm,c.embd,h);
        matrix(l.qa,h,qa);normalize(qa,l.qa_norm,l.qa.rows,qa_normed);matrix(l.qb,qa_normed,q);
        matrix(l.kva,h,kv);normalize(kv,l.kv_norm,c.rank,latent);
        if(c.rotary)mla_rope_queries<<<(c.heads*(c.rotary/2)+255)/256,256,0,stream>>>(q,phases,position,c.heads,c.head_dim,c.rotary,c.rope_magnitude);
        mla_write_latent<<<(max(c.rank,c.rotary/2)+255)/256,256,0,stream>>>(latent,kv,phases,position,c.rank,c.rotary,c.rope_magnitude,cache[il]);
        mla_launch_head(l.kb,c.heads,c.rank,q,c.head_dim,absorbed,width,stream);
        if(c.rotary)mla_query_tail<<<(c.heads*c.rotary+255)/256,256,0,stream>>>(q,c.heads,c.head_dim,c.rotary,c.rank,absorbed);
        float scale=1.0f/sqrtf(float(c.head_dim));
        if(c.split) {
            unsigned parts=(c.capacity+attention_tile-1)/attention_tile;
            mla_attention_partials<<<dim3(c.heads,parts),128,0,stream>>>(cache[il],absorbed,position,c.heads,c.rank,c.rotary,parts,scale,scratch);
            attention_merge<<<dim3(c.heads,1),128,parts*sizeof(float),stream>>>(scratch,c.heads,c.rank,parts,values);
        } else mla_attention<<<c.heads,128,c.capacity*sizeof(float),stream>>>(cache[il],absorbed,position,c.rank,c.rotary,scale,values);
        mla_launch_head(l.vb,c.heads,c.value_dim,values,c.rank,attended,c.value_dim,stream);
        matrix(l.out,attended,projection);
        normalize(x,l.ffn_norm,c.embd,h,projection);
        if(il>=c.dense_layers) {
            if(c.ordered && l.router.type==0)gpt_router_matrix_ordered<<<(c.experts+7)/8,256,0,stream>>>(static_cast<const float*>(l.router.weights),l.router.row_bytes,h,c.embd,c.experts,c.ordered,router);
            else matrix(l.router,h,router);
            mla_router<<<1,1,0,stream>>>(router,l.selection_bias,c.experts,c.used,c.groups,c.groups_used,c.sigmoid,c.weight_norm,c.weight_scale,ids,probabilities);
        }
        if(l.shared_gate.weights) {
            matrix(l.shared_gate,h,sg);matrix(l.shared_up,h,su);
            resident_silu<<<(l.shared_gate.rows+255)/256,256,0,stream>>>(sg,su,l.shared_gate.rows);
            matrix(l.shared_down,sg,shared);
        }
    }
    void enqueue_finish(unsigned il,bool cpu) {
        const auto &l=layers[il];const auto &c=cfg;
        if(il<c.dense_layers) {
            resident_add<<<(c.embd+255)/256,256,0,stream>>>(x,shared,c.embd);
            return;
        }
        if(!cpu) {
            auto *m=static_cast<ResidentMoe*>(l.moe);
            cudaMemcpyAsync(m->experts,ids,c.used*sizeof(unsigned),cudaMemcpyDeviceToDevice,stream);
            cudaMemcpyAsync(m->probabilities,probabilities,c.used*sizeof(float),cudaMemcpyDeviceToDevice,stream);
            auto previous_stream=m->stream;auto *previous_input=m->input,*previous_output=m->output;
            m->stream=stream;m->input=h;m->output=projection;m->enqueue();
            m->stream=previous_stream;m->input=previous_input;m->output=previous_output;
        }
        if(l.shared_gate.weights)resident_add<<<(c.embd+255)/256,256,0,stream>>>(projection,shared,c.embd);
        resident_add<<<(c.embd+255)/256,256,0,stream>>>(x,projection,c.embd);
    }
    void enqueue_head(unsigned mode) {
        if(mode) {normalize(x,norm,cfg.embd,h);matrix(head,h,logits);}
        if(mode==2) {
            unsigned blocks=(cfg.vocab+255)/256;
            resident_argmax<<<blocks,256,0,stream>>>(logits,nullptr,cfg.vocab,maxima,ids);
            resident_argmax<<<1,256,0,stream>>>(maxima,ids,blocks,maximum,token);
        }
    }
    template<typename Fn> bool launch(unsigned slot,Fn enqueue) {
        if(!cfg.graphs) {enqueue();return cudaGetLastError()==cudaSuccess;}
        if(!executable[slot]) {
            if(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return false;
            enqueue();
            if(cudaStreamEndCapture(stream,&graphs[slot])!=cudaSuccess
                || cudaGraphInstantiate(&executable[slot],graphs[slot],0)!=cudaSuccess)return false;
        }
        return cudaGraphLaunch(executable[slot],stream)==cudaSuccess && cudaGetLastError()==cudaSuccess;
    }
};
}
extern "C" {
void *rbitnet_cuda_mla_full_create(const RbitnetMlaConfig *c,const RbitnetMlaLayer *layers,
    const RbitnetLlamaMatrix *head,const float *norm,const float *phases) {
    if(!c || !layers || !head || !norm || (c->rotary && !phases) || !c->embd || c->embd>16384
        || !c->vocab || c->vocab>1048576 || !c->layers || c->layers>256 || !c->heads || c->heads>128
        || !c->head_dim || c->head_dim>1024 || !c->value_dim || c->value_dim>1024
        || !c->rank || c->rank>1024 || !c->capacity || c->capacity>8192
        || c->rotary>=c->head_dim || c->rotary%2 || !c->experts || c->experts>128
        || !c->used || c->used>c->experts || !c->groups || c->groups>c->experts
        || c->experts%c->groups || !c->groups_used || c->groups_used>c->groups
        || c->dense_layers>c->layers || c->sigmoid>1 || c->weight_norm>1
        || (c->ordered!=8 && c->ordered!=16) || c->graphs>1 || c->split>1
        || !isfinite(c->epsilon) || c->epsilon<=0 || !isfinite(c->rope_magnitude) || c->rope_magnitude<=0
        || !isfinite(c->weight_scale) || !qwen_matrix_valid(*head,c->embd,c->vocab))return nullptr;
    unsigned qrank=0,shared_width=0;
    for(unsigned il=0;il<c->layers;il++) {
        const auto &l=layers[il];
        if(!l.attn_norm || !l.qa_norm || !l.kv_norm || !l.ffn_norm || !l.qa.rows || l.qa.rows>16384
            || !qwen_matrix_valid(l.qa,c->embd,l.qa.rows) || !qwen_matrix_valid(l.qb,l.qa.rows,c->heads*c->head_dim)
            || !qwen_matrix_valid(l.kva,c->embd,c->rank+c->rotary)
            || !qwen_matrix_valid(l.kb,c->head_dim-c->rotary,c->rank*c->heads)
            || !qwen_matrix_valid(l.vb,c->rank,c->value_dim*c->heads)
            || !qwen_matrix_valid(l.out,c->value_dim*c->heads,c->embd)
            || (il>=c->dense_layers && !qwen_matrix_valid(l.router,c->embd,c->experts)))return nullptr;
        if(l.shared_gate.weights) {
            if(!l.shared_gate.rows || l.shared_gate.rows>65536
                || !qwen_matrix_valid(l.shared_gate,c->embd,l.shared_gate.rows)
                || !qwen_matrix_valid(l.shared_up,c->embd,l.shared_gate.rows)
                || !qwen_matrix_valid(l.shared_down,l.shared_gate.rows,c->embd))return nullptr;
            shared_width=max(shared_width,l.shared_gate.rows);
        } else if(il<c->dense_layers || l.shared_up.weights || l.shared_down.weights)return nullptr;
        if(l.moe) {
            auto *m=static_cast<ResidentMoe*>(l.moe);
            if(m->cfg.embd!=c->embd || m->cfg.used!=c->used || m->cfg.experts!=c->experts || m->cfg.oai)return nullptr;
        }
        qrank=max(qrank,l.qa.rows);
    }
    auto *r=new(std::nothrow) ResidentMla;if(!r)return nullptr;
    if(!r->identity) {delete r;return nullptr;}
    r->cfg=*c;r->head=*head;r->layers.assign(layers,layers+c->layers);
    r->graphs.resize(size_t(c->layers)*3+3,nullptr);r->executable.resize(r->graphs.size(),nullptr);
    unsigned blocks=(c->vocab+255)/256;
    if(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking)!=cudaSuccess
        || !r->alloc(r->x,c->embd) || !r->alloc(r->h,c->embd) || !r->alloc(r->qa,qrank)
        || !r->alloc(r->qa_normed,qrank) || !r->alloc(r->q,size_t(c->heads)*c->head_dim)
        || !r->alloc(r->kv,c->rank+c->rotary) || !r->alloc(r->latent,c->rank)
        || !r->alloc(r->absorbed,size_t(c->heads)*(c->rank+c->rotary))
        || !r->alloc(r->values,size_t(c->heads)*c->rank) || !r->alloc(r->attended,size_t(c->heads)*c->value_dim)
        || !r->alloc(r->projection,c->embd) || !r->alloc(r->shared,c->embd)
        || (shared_width && (!r->alloc(r->sg,shared_width) || !r->alloc(r->su,shared_width)))
        || !r->alloc(r->router,c->experts) || !r->alloc(r->probabilities,c->used)
        || !r->alloc(r->ids,max(c->used,blocks)) || !r->alloc(r->position,1)
        || !r->alloc(r->norm,c->embd,norm) || !r->alloc(r->logits,c->vocab)
        || !r->alloc(r->maxima,blocks) || !r->alloc(r->maximum,1) || !r->alloc(r->token,1)
        || (c->rotary && !r->alloc(r->phases,size_t(c->capacity)*c->rotary,phases))
        || (c->split && !r->alloc(r->scratch,size_t(c->heads)*((c->capacity+attention_tile-1)/attention_tile)*(c->rank+2)))) {delete r;return nullptr;}
    for(unsigned il=0;il<c->layers;il++) {
        float *cache=nullptr;
        {MemoryCategoryScope category(MemoryKv);
            if(!r->alloc(cache,size_t(c->capacity)*(c->rank+c->rotary))) {delete r;return nullptr;}}
        r->cache.push_back(cache);auto &l=r->layers[il];
        if(!r->copy(l.attn_norm,c->embd) || !r->copy(l.qa_norm,l.qa.rows) || !r->copy(l.kv_norm,c->rank)
            || !r->copy(l.ffn_norm,c->embd) || (l.selection_bias && !r->copy(l.selection_bias,c->experts))) {delete r;return nullptr;}
    }
    return r;
}
void rbitnet_cuda_mla_full_destroy(void *context) {delete static_cast<ResidentMla*>(context);}
int rbitnet_cuda_mla_full_begin(void *context,const float *embedding,unsigned pos) {
    auto *r=static_cast<ResidentMla*>(context);if(!r || !embedding || pos>=r->cfg.capacity || pos>r->filled)return 1;
    NativeCallCompletion completion(r->stream);
    // Starting at zero also discards a partly completed cancelled token.
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 2;
    r->host_position=pos;r->next_layer=0;r->prepared=false;r->token_started=false;r->output_ready=false;r->filled=pos;
    if(cudaMemcpyAsync(r->x,embedding,r->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->position,&r->host_position,sizeof(unsigned),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 2;
    int status=completion.complete(0,2);if(!status)r->token_started=true;return status;
}
int rbitnet_cuda_mla_full_prepare(void *context,unsigned il,unsigned *ids,float *probabilities) {
    auto *r=static_cast<ResidentMla*>(context);
    if(!r || !r->token_started || il!=r->next_layer || il>=r->cfg.layers || r->prepared
        || (il>=r->cfg.dense_layers && (!ids || !probabilities)))return 1;
    NativeCallCompletion completion(r->stream);
    if(!r->launch(il*3,[&]{r->enqueue_prepare(il);}))return 2;
    if(il>=r->cfg.dense_layers
        && (cudaMemcpyAsync(ids,r->ids,r->cfg.used*sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess
            || cudaMemcpyAsync(probabilities,r->probabilities,r->cfg.used*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess))return 3;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 3;
    r->prepared=true;completion.dismiss();return 0;
}
int rbitnet_cuda_mla_full_ffn_input(void *context,float *input) {
    auto *r=static_cast<ResidentMla*>(context);if(!r || !input || !r->prepared)return 1;
    NativeCallCompletion completion(r->stream);
    return completion.complete(cudaMemcpyAsync(input,r->h,r->cfg.embd*sizeof(float),cudaMemcpyDeviceToHost,r->stream)==cudaSuccess?0:2,2);
}
int rbitnet_cuda_mla_full_finish(void *context,unsigned il,const void *const *selected,const float *cpu_routed) {
    auto *r=static_cast<ResidentMla*>(context);if(!r || il!=r->next_layer || il>=r->cfg.layers || !r->prepared)return 1;
    NativeCallCompletion completion(r->stream);
    if(il>=r->cfg.dense_layers) {
        auto *m=static_cast<ResidentMoe*>(r->layers[il].moe);
        if(cpu_routed) {
            if(cudaMemcpyAsync(r->projection,cpu_routed,r->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 2;
        } else {
            if(!m || (m->dynamic && !selected))return 1;
            if(m->dynamic) {
                for(unsigned i=0;i<3*r->cfg.used;i++)if(!selected[i])return 1;
                if(cudaMemcpyAsync(m->selected,selected,3*r->cfg.used*sizeof(void*),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 2;
            }
        }
    }
    if(!r->launch(il*3+(cpu_routed?2:1),[&]{r->enqueue_finish(il,cpu_routed!=nullptr);})
        || cudaStreamSynchronize(r->stream)!=cudaSuccess)return 3;
    r->prepared=false;r->next_layer++;if(r->next_layer==r->cfg.layers)r->filled=r->host_position+1;
    completion.dismiss();return 0;
}
int rbitnet_cuda_mla_full_end(void *context,unsigned mode,float *logits,unsigned *token) {
    auto *r=static_cast<ResidentMla*>(context);if(!r || (!r->token_started && !r->output_ready) || mode>2 || r->prepared || r->next_layer!=r->cfg.layers
        || (mode==1 && !logits) || (mode==2 && !token))return 1;
    NativeCallCompletion completion(r->stream);
    if(!r->launch(r->cfg.layers*3+mode,[&]{r->enqueue_head(mode);}))return 2;
    if(mode==1 && cudaMemcpyAsync(logits,r->logits,r->cfg.vocab*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 3;
    if(mode==2 && cudaMemcpyAsync(token,r->token,sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 3;
    int status=completion.complete(0,3);if(!status) {r->token_started=false;r->output_ready=true;}return status;
}
void rbitnet_cuda_mla_snapshot_destroy(void *snapshot) {delete static_cast<MlaSnapshot*>(snapshot);}
void *rbitnet_cuda_mla_snapshot(void *context,unsigned length) {
    auto *r=static_cast<ResidentMla*>(context);
    if(!r || !length || length>r->filled || r->prepared || r->token_started || r->next_layer!=r->cfg.layers)return nullptr;
    auto *s=new(std::nothrow) MlaSnapshot;if(!s)return nullptr;
    s->owner=r->identity;s->layers=r->cfg.layers;s->width=r->cfg.rank+r->cfg.rotary;s->length=length;
    size_t words=size_t(length)*s->width;
    {MemoryCategoryScope category(MemoryPrefix);
        if(cudaMalloc(reinterpret_cast<void**>(&s->data),size_t(s->layers)*words*sizeof(float))!=cudaSuccess) {delete s;return nullptr;}}
    for(unsigned il=0;il<s->layers;il++) {
        if(cudaMemcpyAsync(s->data+size_t(il)*words,r->cache[il],words*sizeof(float),cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess) {
            cudaStreamSynchronize(r->stream);delete s;return nullptr;
        }
    }
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess) {delete s;return nullptr;}
    return s;
}
int rbitnet_cuda_mla_restore(void *context,const void *snapshot,unsigned length) {
    auto *r=static_cast<ResidentMla*>(context);auto *s=static_cast<const MlaSnapshot*>(snapshot);
    if(!r || !s || s->owner!=r->identity || s->layers!=r->cfg.layers || s->width!=r->cfg.rank+r->cfg.rotary
        || !length || length>s->length || length>r->cfg.capacity)return 1;
    NativeCallCompletion completion(r->stream);
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 2;
    size_t words=size_t(length)*s->width,source_words=size_t(s->length)*s->width;
    for(unsigned il=0;il<s->layers;il++)if(cudaMemcpyAsync(r->cache[il],s->data+size_t(il)*source_words,words*sizeof(float),cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess)return 2;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 2;
    r->filled=length;r->next_layer=r->cfg.layers;r->prepared=false;r->token_started=false;r->output_ready=false;completion.dismiss();return 0;
}
// Diagnostic host-array helpers use the production kernels. They are never
// called from hot model inference and do not change any captured addresses.
int rbitnet_cuda_mla_hidden_check(void *context,float *hidden) {
    auto *r=static_cast<ResidentMla*>(context);
    if(!r || !hidden || r->prepared || r->token_started || !r->output_ready || r->next_layer!=r->cfg.layers)return 1;
    NativeCallCompletion completion(r->stream);
    return completion.complete(cudaMemcpyAsync(hidden,r->x,r->cfg.embd*sizeof(float),cudaMemcpyDeviceToHost,r->stream)==cudaSuccess?0:2,2);
}
int rbitnet_cuda_mla_router_check(const float *raw,const float *bias,unsigned count,unsigned used,
    unsigned groups,unsigned groups_used,unsigned sigmoid,unsigned normalize,float scale,unsigned *ids,float *probabilities) {
    if(!raw || !ids || !probabilities || !count || count>128 || !used || used>count || !groups
        || groups>count || count%groups || !groups_used || groups_used>groups || sigmoid>1 || normalize>1 || !isfinite(scale))return 1;
    ResidentLlama b;float *dr=nullptr,*db=nullptr,*dp=nullptr;unsigned *di=nullptr;
    if(cudaStreamCreateWithFlags(&b.stream,cudaStreamNonBlocking)!=cudaSuccess)return 2;
    if(!b.alloc(dr,count) || (bias && !b.alloc(db,count)) || !b.alloc(dp,used) || !b.alloc(di,used))return 2;
    if(cudaMemcpyAsync(dr,raw,count*sizeof(float),cudaMemcpyHostToDevice,b.stream)!=cudaSuccess
        || (bias && cudaMemcpyAsync(db,bias,count*sizeof(float),cudaMemcpyHostToDevice,b.stream)!=cudaSuccess))return 3;
    mla_router<<<1,1,0,b.stream>>>(dr,db,count,used,groups,groups_used,sigmoid,normalize,scale,di,dp);
    return cudaGetLastError()==cudaSuccess
        && cudaMemcpyAsync(ids,di,used*sizeof(unsigned),cudaMemcpyDeviceToHost,b.stream)==cudaSuccess
        && cudaMemcpyAsync(probabilities,dp,used*sizeof(float),cudaMemcpyDeviceToHost,b.stream)==cudaSuccess
        && cudaStreamSynchronize(b.stream)==cudaSuccess?0:4;
}
int rbitnet_cuda_mla_attention_check(const float *cache,const float *queries,unsigned capacity,unsigned heads,
    unsigned rank,unsigned rotary,float scale,unsigned position,unsigned split,float *output) {
    if(!cache || !queries || !output || !capacity || capacity>8192 || !heads || heads>128 || !rank || rank>1024
        || rotary>1024 || rotary%2 || position>=capacity || split>1 || !isfinite(scale))return 1;
    ResidentLlama b;float *dc=nullptr,*dq=nullptr,*dy=nullptr,*ds=nullptr;unsigned *dp=nullptr;
    if(cudaStreamCreateWithFlags(&b.stream,cudaStreamNonBlocking)!=cudaSuccess)return 2;
    unsigned width=rank+rotary,parts=(capacity+attention_tile-1)/attention_tile;
    if(!b.alloc(dc,size_t(capacity)*width) || !b.alloc(dq,size_t(heads)*width) || !b.alloc(dy,size_t(heads)*rank)
        || !b.alloc(dp,1) || (split && !b.alloc(ds,size_t(heads)*parts*(rank+2))))return 2;
    if(cudaMemcpyAsync(dc,cache,size_t(capacity)*width*sizeof(float),cudaMemcpyHostToDevice,b.stream)!=cudaSuccess
        || cudaMemcpyAsync(dq,queries,size_t(heads)*width*sizeof(float),cudaMemcpyHostToDevice,b.stream)!=cudaSuccess
        || cudaMemcpyAsync(dp,&position,sizeof(unsigned),cudaMemcpyHostToDevice,b.stream)!=cudaSuccess)return 3;
    if(split) {
        mla_attention_partials<<<dim3(heads,parts),128,0,b.stream>>>(dc,dq,dp,heads,rank,rotary,parts,scale,ds);
        attention_merge<<<dim3(heads,1),128,parts*sizeof(float),b.stream>>>(ds,heads,rank,parts,dy);
    } else mla_attention<<<heads,128,capacity*sizeof(float),b.stream>>>(dc,dq,dp,rank,rotary,scale,dy);
    return cudaGetLastError()==cudaSuccess
        && cudaMemcpyAsync(output,dy,size_t(heads)*rank*sizeof(float),cudaMemcpyDeviceToHost,b.stream)==cudaSuccess
        && cudaStreamSynchronize(b.stream)==cudaSuccess?0:4;
}
}
