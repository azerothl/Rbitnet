// SPDX-License-Identifier: MIT
// Segmented/cache GPT block prefill: batched attention and router, host-admitted FFN.
namespace {
struct GptSegmentedBlockWorkspace {
    ResidentGpt *runtime=nullptr;
    unsigned capacity=0,count=0,tile=0,pos=0,active_layer=0;
    bool active=false;
    std::vector<void*> allocations;
    float *x=nullptr,*h=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*attn=nullptr,*projection=nullptr,*router=nullptr,*scratch=nullptr;
    unsigned *ids=nullptr;
    float *probabilities=nullptr,*all_logits=nullptr,*all_maxima=nullptr,*all_maximum=nullptr;
    unsigned *all_ids=nullptr,*all_tokens=nullptr;
    bool init(ResidentGpt *r,unsigned block_count,unsigned tile_mode) {
        if(!r || !r->segmented || !block_count || block_count>32 || tile_mode>1 || r->layers.empty())return false;
        runtime=r;capacity=block_count;tile=tile_mode;
        const auto &c=r->cfg;const size_t qs=size_t(block_count)*c.heads*c.head_dim,ks=size_t(block_count)*c.kv_heads*c.head_dim;
        const size_t heads=size_t(block_count)*((c.vocab+255)/256);
        return alloc(x,size_t(block_count)*c.embd) && alloc(h,size_t(block_count)*c.embd)
            && alloc(q,qs) && alloc(k,ks) && alloc(v,ks) && alloc(attn,qs)
            && alloc(projection,size_t(block_count)*c.embd) && alloc(router,size_t(block_count)*c.experts)
            && alloc(ids,size_t(block_count)*c.used) && alloc(probabilities,size_t(block_count)*c.used)
            && alloc(all_logits,size_t(block_count)*c.vocab) && alloc(all_maxima,heads)
            && alloc(all_ids,heads) && alloc(all_maximum,block_count) && alloc(all_tokens,block_count)
            && (!c.split || alloc(scratch,size_t(block_count)*c.heads*((c.capacity+attention_tile-1)/attention_tile)*(c.head_dim+2)));
    }
    template<typename T> bool alloc(T *&pointer,size_t n) {
        if(!n || n>std::numeric_limits<size_t>::max()/sizeof(T))return false;
        if(cudaMalloc(reinterpret_cast<void**>(&pointer),n*sizeof(T))!=cudaSuccess)return false;
        try {allocations.push_back(pointer);}
        catch(const std::bad_alloc&) {cudaFree(pointer);pointer=nullptr;return false;}
        return true;
    }
    void matrix(const RbitnetLlamaMatrix &m,const float *input,float *output,unsigned block_count) {
        QuantKind kind;resident_kind(m.type,kind);
        launch_ordered_gemm(kind,static_cast<const uint8_t*>(m.weights),m.row_bytes,input,
            m.cols,m.rows,block_count,output,runtime->stream,tile);
    }
    void bias(float *values,const float *weights,unsigned width,unsigned block_count) {
        gpt_block_bias<<<(width*block_count+255)/256,256,0,runtime->stream>>>(values,weights,width,block_count);
    }
    void norm(float *input,const float *weights,float *output,unsigned block_count,const float *residual=nullptr) {
        auto *r=runtime;
        if(r->cfg.ordered)launch_gpt_ordered_norm(input,weights,r->cfg.epsilon,r->cfg.embd,output,residual,block_count,r->stream);
        else resident_norm<<<block_count,256,0,r->stream>>>(input,weights,r->cfg.epsilon,r->cfg.embd,output,residual);
    }
    bool prepare_layer(unsigned il) {
        auto *r=runtime;const auto &c=r->cfg;const auto stream=r->stream;
        if(!active || il>=c.layers || il!=active_layer)return false;
        const auto &l=r->layers[il];
        const unsigned qs=c.heads*c.head_dim,ks=c.kv_heads*c.head_dim;
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
        for(unsigned t=0;t<count;t++) {
            gpt_router<<<1,1,0,stream>>>(router+size_t(t)*c.experts,l.selection_bias,c.experts,c.used,c.weight_scale,
                ids+size_t(t)*c.used,probabilities+size_t(t)*c.used,c.ordered!=0);
        }
        return cudaGetLastError()==cudaSuccess;
    }
    bool finish_token(unsigned il,unsigned token,const void *const *selected,const float *cpu_routed) {
        auto *r=runtime;const auto &c=r->cfg;const auto stream=r->stream;
        if(!active || il!=active_layer || token>=count)return false;
        const size_t embd=c.embd;
        if(cudaMemcpyAsync(r->x,x+size_t(token)*embd,embd*sizeof(float),cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return false;
        if(cudaMemcpyAsync(r->h,h+size_t(token)*embd,embd*sizeof(float),cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return false;
        if(cudaMemcpyAsync(r->projection,projection+size_t(token)*embd,embd*sizeof(float),cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return false;
        if(cpu_routed) {
            if(cudaMemcpyAsync(r->projection,cpu_routed,embd*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess)return false;
        } else {
            auto *m=static_cast<ResidentMoe*>(r->layers[il].moe);
            if(!m || (m->dynamic && !selected))return false;
            if(m->dynamic) {
                for(unsigned i=0;i<3*c.used;i++)if(!selected[i])return false;
                if(cudaMemcpyAsync(m->selected,selected,3*c.used*sizeof(void*),cudaMemcpyHostToDevice,stream)!=cudaSuccess)return false;
            }
            if(cudaMemcpyAsync(m->experts,ids+size_t(token)*c.used,c.used*sizeof(unsigned),cudaMemcpyDeviceToDevice,stream)!=cudaSuccess
                || cudaMemcpyAsync(m->probabilities,probabilities+size_t(token)*c.used,c.used*sizeof(float),cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return false;
            auto previous_stream=m->stream;auto *previous_input=m->input,*previous_output=m->output;
            m->stream=stream;m->input=r->h;m->output=r->projection;
            m->enqueue();
            m->stream=previous_stream;m->input=previous_input;m->output=previous_output;
        }
        resident_add<<<(c.embd+255)/256,256,0,stream>>>(r->x,r->projection,c.embd);
        if(cudaMemcpyAsync(x+size_t(token)*embd,r->x,embd*sizeof(float),cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return false;
        if(token+1==count) {active_layer++;if(active_layer==c.layers)active=false;}
        return cudaGetLastError()==cudaSuccess;
    }
    bool begin_block(const float *embeddings,unsigned position,unsigned block_count) {
        auto *r=runtime;
        if(!r || !embeddings || !block_count || block_count>capacity || position>=r->cfg.capacity
            || block_count>r->cfg.capacity-position)return false;
        if(position==0)r->filled=0;
        if(position!=r->filled || r->token_started || r->prepared)return false;
        r->output_ready=false;r->host_position=position;r->next_layer=0;r->prepared=false;r->token_started=true;
        count=block_count;pos=position;active_layer=0;active=true;
        if(cudaMemcpyAsync(x,embeddings,size_t(block_count)*r->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return false;
        if(cudaMemcpyAsync(r->position,&position,sizeof(position),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return false;
        return cudaGetLastError()==cudaSuccess;
    }
    bool end_block(unsigned mode,bool all) {
        auto *r=runtime;const auto &c=r->cfg;const auto stream=r->stream;
        if(!r || active || r->prepared || active_layer!=c.layers || !r->token_started)return false;
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
        r->filled=pos+count;r->token_started=false;r->output_ready=true;
        return cudaGetLastError()==cudaSuccess;
    }
};
void gpt_segmented_block_destroy(void *block) {delete static_cast<GptSegmentedBlockWorkspace*>(block);}
int gpt_segmented_prefill_impl(void *context,const float *embeddings,unsigned pos,unsigned count,
    unsigned mode,float *logits,unsigned *tokens,bool all) {
    auto *r=static_cast<ResidentGpt*>(context);
    auto *b=r?static_cast<GptSegmentedBlockWorkspace*>(r->block):nullptr;
    if(!r || !b || !embeddings || !count || count>b->capacity || mode>2
        || pos>=r->cfg.capacity || count>r->cfg.capacity-pos || (mode==1 && !logits)
        || (mode==2 && !tokens) || (all && !mode))return 1;
    NativeCallCompletion completion(r->stream);
    if(!b->begin_block(embeddings,pos,count))return 3;
    for(unsigned il=0;il<r->cfg.layers;il++) {
        if(!b->prepare_layer(il))return 5;
        for(unsigned t=0;t<count;t++) {
            auto *m=static_cast<ResidentMoe*>(r->layers[il].moe);
            if(!m || m->dynamic)return 8;
            if(!b->finish_token(il,t,nullptr,nullptr))return 5;
        }
    }
    if(!b->end_block(mode,all))return 5;
    if(cudaGetLastError()!=cudaSuccess
        || (mode==1 && cudaMemcpyAsync(logits,all?b->all_logits:r->logits,size_t(all?count:1)*r->cfg.vocab*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)
        || (mode==2 && cudaMemcpyAsync(tokens,all?b->all_tokens:r->token,size_t(all?count:1)*sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess))return 6;
    return completion.complete(0,7);
}
}
extern "C" {
int rbitnet_cuda_gpt_segmented_configure_prefill(void *context,unsigned capacity,unsigned tile) {
    auto *r=static_cast<ResidentGpt*>(context);
    if(!r || !r->segmented || r->filled || r->token_started || r->block || !capacity || capacity>32 || tile>1)return 1;
    auto *b=new(std::nothrow) GptSegmentedBlockWorkspace;if(!b)return 2;
    if(!b->init(r,capacity,tile)) {delete b;return 2;}
    r->block=b;return 0;
}
unsigned rbitnet_cuda_gpt_segmented_prefill_capacity(void *context) {
    auto *r=static_cast<ResidentGpt*>(context);
    auto *b=r?static_cast<GptSegmentedBlockWorkspace*>(r->block):nullptr;
    return b?b->capacity:0;
}
int rbitnet_cuda_gpt_segmented_block_begin(void *context,const float *embeddings,unsigned pos,unsigned count) {
    auto *r=static_cast<ResidentGpt*>(context);
    auto *b=r?static_cast<GptSegmentedBlockWorkspace*>(r->block):nullptr;
    if(!b)return 1;
    NativeCallCompletion completion(r->stream);
    return completion.complete(b->begin_block(embeddings,pos,count)?0:3,3);
}
int rbitnet_cuda_gpt_segmented_block_prepare(void *context,unsigned il,unsigned *ids,float *probabilities) {
    auto *r=static_cast<ResidentGpt*>(context);
    auto *b=r?static_cast<GptSegmentedBlockWorkspace*>(r->block):nullptr;
    if(!b || !ids || !probabilities)return 1;
    NativeCallCompletion completion(r->stream);
    if(!b->prepare_layer(il))return completion.complete(2,2);
    const auto &c=r->cfg;
    if(cudaMemcpyAsync(ids,b->ids,size_t(b->count)*c.used*sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(probabilities,b->probabilities,size_t(b->count)*c.used*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return completion.complete(3,3);
    return completion.complete(0,3);
}
int rbitnet_cuda_gpt_segmented_block_ffn_input(void *context,unsigned token,float *input) {
    auto *r=static_cast<ResidentGpt*>(context);
    auto *b=r?static_cast<GptSegmentedBlockWorkspace*>(r->block):nullptr;
    if(!b || !input || token>=b->count)return 1;
    NativeCallCompletion completion(r->stream);
    return completion.complete(cudaMemcpyAsync(input,b->h+size_t(token)*r->cfg.embd,r->cfg.embd*sizeof(float),cudaMemcpyDeviceToHost,r->stream)==cudaSuccess?0:2,2);
}
int rbitnet_cuda_gpt_segmented_block_finish(void *context,unsigned il,unsigned token,const void *const *selected,const float *cpu_routed) {
    auto *r=static_cast<ResidentGpt*>(context);
    auto *b=r?static_cast<GptSegmentedBlockWorkspace*>(r->block):nullptr;
    if(!b)return 1;
    NativeCallCompletion completion(r->stream);
    return completion.complete(b->finish_token(il,token,selected,cpu_routed)?0:2,2);
}
int rbitnet_cuda_gpt_segmented_block_end(void *context,unsigned mode,float *logits,unsigned *token) {
    auto *r=static_cast<ResidentGpt*>(context);
    auto *b=r?static_cast<GptSegmentedBlockWorkspace*>(r->block):nullptr;
    if(!b)return 1;
    NativeCallCompletion completion(r->stream);
    if(!b->end_block(mode,false))return completion.complete(2,2);
    if(mode==1 && cudaMemcpyAsync(logits,r->logits,r->cfg.vocab*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return completion.complete(3,3);
    if(mode==2 && cudaMemcpyAsync(token,r->token,sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return completion.complete(3,3);
    return completion.complete(0,3);
}
int rbitnet_cuda_gpt_segmented_prefill(void *context,const float *input,unsigned pos,unsigned count,unsigned mode,float *out,unsigned *ids) {
    return gpt_segmented_prefill_impl(context,input,pos,count,mode,out,ids,false);
}
int rbitnet_cuda_gpt_segmented_verify(void *context,const float *input,unsigned pos,unsigned count,unsigned mode,float *out,unsigned *ids) {
    return gpt_segmented_prefill_impl(context,input,pos,count,mode,out,ids,true);
}
}
