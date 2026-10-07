// SPDX-License-Identifier: MIT
// Fixed-bank GPT prefill with exact ordered projections and grouped
// routed FFNs. Include after the GPT token graph, with its block-owner hook.
// Segmented/cache block prefill lives in gpt_segmented_prefill.cuh.
namespace {
__global__ void gpt_block_bias(float *values,const float *bias,unsigned width,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<width*count)values[i]+=bias[i%width];
}
struct GptBlockWorkspace {
    ResidentGpt *runtime=nullptr;
    unsigned capacity=0,tile=0;
    std::vector<void*> allocations;
    float *x=nullptr,*h=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*attn=nullptr,*projection=nullptr,*router=nullptr,*scratch=nullptr;
    float *probabilities=nullptr,*all_logits=nullptr,*all_maxima=nullptr,*all_maximum=nullptr;
    unsigned *ids=nullptr,*all_ids=nullptr,*all_tokens=nullptr;
    std::unique_ptr<MoeGroupedWorkspace> moe;
    cudaGraph_t graphs[2][3][33]={};cudaGraphExec_t executable[2][3][33]={};
    ~GptBlockWorkspace() {
        if(runtime)cudaStreamSynchronize(runtime->stream);
        for(auto &all:executable)for(auto &mode:all)for(auto e:mode)if(e)cudaGraphExecDestroy(e);
        for(auto &all:graphs)for(auto &mode:all)for(auto g:mode)if(g)cudaGraphDestroy(g);
        moe.reset();for(auto p:allocations)cudaFree(p);
    }
    template<typename T> bool alloc(T *&pointer,size_t n) {
        if(!n || n>std::numeric_limits<size_t>::max()/sizeof(T))return false;
        if(cudaMalloc(reinterpret_cast<void**>(&pointer),n*sizeof(T))!=cudaSuccess)return false;
        try {allocations.push_back(pointer);}
        catch(const std::bad_alloc&) {cudaFree(pointer);pointer=nullptr;return false;}
        return true;
    }
    bool init(ResidentGpt *r,unsigned count,unsigned tile_mode) {
        if(!r || r->segmented || !count || count>32 || tile_mode>1 || r->layers.empty())return false;
        runtime=r;capacity=count;tile=tile_mode;
        auto *first=static_cast<ResidentMoe*>(r->layers[0].moe);
        if(!first || first->dynamic)return false;
        for(auto &layer:r->layers) {
            auto *m=static_cast<ResidentMoe*>(layer.moe);
            if(!m || m->dynamic || m->cfg.ffn!=first->cfg.ffn || m->cfg.embd!=first->cfg.embd
                || m->cfg.experts!=first->cfg.experts || m->cfg.used!=first->cfg.used || m->cfg.oai!=first->cfg.oai)return false;
        }
        moe=std::unique_ptr<MoeGroupedWorkspace>(new(std::nothrow) MoeGroupedWorkspace);
        if(!moe || !moe->init(first,count))return false;
        const auto &c=r->cfg;const size_t qs=size_t(count)*c.heads*c.head_dim,ks=size_t(count)*c.kv_heads*c.head_dim;
        const size_t heads=size_t(count)*((c.vocab+255)/256);
        return alloc(x,size_t(count)*c.embd) && alloc(h,size_t(count)*c.embd)
            && alloc(q,qs) && alloc(k,ks) && alloc(v,ks) && alloc(attn,qs)
            && alloc(projection,size_t(count)*c.embd) && alloc(router,size_t(count)*c.experts)
            && alloc(ids,size_t(count)*c.used) && alloc(probabilities,size_t(count)*c.used)
            && alloc(all_logits,size_t(count)*c.vocab) && alloc(all_maxima,heads)
            && alloc(all_ids,heads) && alloc(all_maximum,count) && alloc(all_tokens,count)
            && (!c.split || alloc(scratch,size_t(count)*c.heads*((c.capacity+attention_tile-1)/attention_tile)*(c.head_dim+2)));
    }
    void matrix(const RbitnetLlamaMatrix &m,const float *input,float *output,unsigned count) {
        QuantKind kind;resident_kind(m.type,kind);
        launch_ordered_gemm(kind,static_cast<const uint8_t*>(m.weights),m.row_bytes,input,
            m.cols,m.rows,count,output,runtime->stream,tile);
    }
    void bias(float *values,const float *weights,unsigned width,unsigned count) {
        gpt_block_bias<<<(width*count+255)/256,256,0,runtime->stream>>>(values,weights,width,count);
    }
    void norm(float *input,const float *weights,float *output,unsigned count,const float *residual=nullptr) {
        auto *r=runtime;
        if(r->cfg.ordered)launch_gpt_ordered_norm(input,weights,r->cfg.epsilon,r->cfg.embd,output,residual,count,r->stream);
        else resident_norm<<<count,256,0,r->stream>>>(input,weights,r->cfg.epsilon,r->cfg.embd,output,residual);
    }
    bool enqueue(unsigned count,unsigned mode,bool all) {
        auto *r=runtime;const auto &c=r->cfg;const auto stream=r->stream;
        const unsigned qs=c.heads*c.head_dim,ks=c.kv_heads*c.head_dim;
        for(unsigned il=0;il<c.layers;il++) {
            const auto &l=r->layers[il];
            norm(x,l.attn_norm,h,count);
            matrix(l.q,h,q,count);bias(q,l.q_bias,qs,count);
            matrix(l.k,h,k,count);bias(k,l.k_bias,ks,count);
            matrix(l.v,h,v,count);bias(v,l.v_bias,ks,count);
            dim3 rope(((c.heads+c.kv_heads)*(c.rotary/2)+255)/256,count);
            if(c.rotary)gpt_rope<<<rope,256,0,stream>>>(q,k,r->frequency,r->position,c.heads,c.kv_heads,c.head_dim,c.rotary,c.rope_magnitude,r->phases);
            gpt_write_kv<<<dim3((ks+255)/256,count),256,0,stream>>>(k,v,r->keys[il],r->values[il],r->position,ks);
            const unsigned window=il%2==0?c.window:0;const float scale=1.0f/sqrtf(float(c.head_dim));
            if(c.split)launch_split_attention(r->keys[il],r->values[il],q,r->position,c.kv_heads,c.heads,c.head_dim,window,scale,c.capacity,count,scratch,attn,stream,l.sinks);
            else resident_attention<<<dim3(c.heads,count),128,size_t(c.capacity)*sizeof(float),stream>>>(r->keys[il],r->values[il],q,r->position,c.kv_heads,c.heads,c.head_dim,window,scale,attn,l.sinks);
            matrix(l.out,attn,projection,count);bias(projection,l.out_bias,c.embd,count);
            norm(x,l.ffn_norm,h,count,projection);
            if(c.ordered && l.router.type==0)gpt_router_matrix_ordered<<<dim3((c.experts+7)/8,count),256,0,stream>>>(static_cast<const float*>(l.router.weights),l.router.row_bytes,h,c.embd,c.experts,c.ordered,router);
            else matrix(l.router,h,router,count);
            bias(router,l.router_bias,c.experts,count);
            gpt_router<<<count,1,0,stream>>>(router,l.selection_bias,c.experts,c.used,c.weight_scale,ids,probabilities,c.ordered!=0);
            auto *source=static_cast<ResidentMoe*>(l.moe);const auto previous=source->stream;
            source->stream=stream;moe->source=source;
            const bool okay=moe->enqueue(h,ids,probabilities,projection,count,tile);
            source->stream=previous;if(!okay)return false;
            resident_add<<<(count*c.embd+255)/256,256,0,stream>>>(x,projection,count*c.embd);
        }
        // Token graph continuation/prefix checks must see the final hidden vector.
        if(cudaMemcpyAsync(r->x,x+size_t(count-1)*c.embd,c.embd*sizeof(float),cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return false;
        if(mode) {
            if(all) {norm(x,r->norm,h,count);matrix(r->head,h,all_logits,count);}
            else {r->normalize(r->x,r->norm,r->h);r->matrix(r->head,r->h,r->logits);}
        }
        if(mode==2) {
            const unsigned blocks=(c.vocab+255)/256;
            if(all) {
                resident_argmax<<<dim3(blocks,count),256,0,stream>>>(all_logits,nullptr,c.vocab,all_maxima,all_ids);
                resident_argmax<<<dim3(1,count),256,0,stream>>>(all_maxima,all_ids,blocks,all_maximum,all_tokens);
            } else {
                resident_argmax<<<blocks,256,0,stream>>>(r->logits,nullptr,c.vocab,r->maxima,r->ids);
                resident_argmax<<<1,256,0,stream>>>(r->maxima,r->ids,blocks,r->maximum,r->token);
            }
        }
        return true;
    }
};
void gpt_block_destroy(void *block) {delete static_cast<GptBlockWorkspace*>(block);}
int gpt_prefill_impl(void *context,const float *embeddings,unsigned pos,unsigned count,
    unsigned mode,float *logits,unsigned *tokens,bool all) {
    auto *r=static_cast<ResidentGpt*>(context);
    auto *b=r?static_cast<GptBlockWorkspace*>(r->block):nullptr;
    if(!r || !b || !embeddings || !count || count>b->capacity || mode>2
        || pos>=r->cfg.capacity || count>r->cfg.capacity-pos || (mode==1 && !logits)
        || (mode==2 && !tokens) || (all && !mode))return 1;
    if(pos==0)r->filled=0;if(pos!=r->filled || r->token_started)return 2;
    r->output_ready=false;
    NativeCallCompletion completion(r->stream);
    if(cudaMemcpyAsync(b->x,embeddings,size_t(count)*r->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->position,&pos,sizeof(pos),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 3;
    if(r->cfg.graphs) {
        auto &graph=b->graphs[all][mode][count];auto &exec=b->executable[all][mode][count];
        if(!exec) {
            if(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 4;
            if(!b->enqueue(count,mode,all) || cudaStreamEndCapture(r->stream,&graph)!=cudaSuccess
                || cudaGraphInstantiate(&exec,graph,0)!=cudaSuccess)return 4;
        }
        if(cudaGraphLaunch(exec,r->stream)!=cudaSuccess)return 5;
    } else if(!b->enqueue(count,mode,all))return 5;
    if(cudaGetLastError()!=cudaSuccess
        || (mode==1 && cudaMemcpyAsync(logits,all?b->all_logits:r->logits,size_t(all?count:1)*r->cfg.vocab*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)
        || (mode==2 && cudaMemcpyAsync(tokens,all?b->all_tokens:r->token,size_t(all?count:1)*sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess))return 6;
    if(completion.complete(0,7)!=0)return 7;
    r->filled=pos+count;r->output_ready=true;return 0;
}
}
extern "C" {
// Configure once before inference. An explicit refusal retains the serial graph.
int rbitnet_cuda_gpt_configure_prefill(void *context,unsigned capacity,unsigned tile) {
    auto *r=static_cast<ResidentGpt*>(context);
    if(!r || r->segmented || r->filled || r->token_started || r->block || !capacity || capacity>32 || tile>1)return 1;
    auto *b=new(std::nothrow) GptBlockWorkspace;if(!b)return 2;
    if(!b->init(r,capacity,tile)) {delete b;return 2;}
    r->block=b;return 0;
}
unsigned rbitnet_cuda_gpt_prefill_capacity(void *context) {
    auto *r=static_cast<ResidentGpt*>(context);auto *b=r?static_cast<GptBlockWorkspace*>(r->block):nullptr;return b?b->capacity:0;
}
int rbitnet_cuda_gpt_full_prefill(void *context,const float *input,unsigned pos,unsigned count,unsigned mode,float *out,unsigned *ids) {
    return gpt_prefill_impl(context,input,pos,count,mode,out,ids,false);
}
int rbitnet_cuda_gpt_full_verify(void *context,const float *input,unsigned pos,unsigned count,unsigned mode,float *out,unsigned *ids) {
    return gpt_prefill_impl(context,input,pos,count,mode,out,ids,true);
}
}
