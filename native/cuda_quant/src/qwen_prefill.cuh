// SPDX-License-Identifier: MIT
// Included after the recurrent/attention contexts. One bounded workspace is
// shared by all layers; each projection uses its own packed token stride.
struct QwenPrefill {
    static constexpr unsigned capacity=128;
    float *x=nullptr,*h=nullptr,*projection=nullptr,*gate=nullptr,*up=nullptr;
    float *mixed=nullptr,*activated=nullptr,*z=nullptr,*beta=nullptr,*alpha=nullptr;
    float *recurrent_out=nullptr,*normed=nullptr,*q_full=nullptr,*q=nullptr;
    float *k_raw=nullptr,*k=nullptr,*v=nullptr,*attention=nullptr,*scratch=nullptr;
    std::vector<void*> allocations;
    cudaGraph_t graphs[3][capacity+1]={};cudaGraphExec_t executable[3][capacity+1]={};
    ~QwenPrefill() {
        for(auto &mode:executable)for(auto e:mode)if(e)cudaGraphExecDestroy(e);
        for(auto &mode:graphs)for(auto g:mode)if(g)cudaGraphDestroy(g);
        for(auto p:allocations)cudaFree(p);
    }
    bool alloc(float *&p,size_t width) {
        if(!width)return true;
        if(cudaMalloc(reinterpret_cast<void**>(&p),width*capacity*sizeof(float))!=cudaSuccess)return false;
        allocations.push_back(p);return true;
    }
    bool init(const std::vector<RbitnetQwenFullLayer> &layers,unsigned embd) {
        size_t ffn=0,inner=0,values=0,nv=0,qs=0,qraw=0,ks=0,workspace=0;
        for(const auto &layer:layers) {
            if(layer.kind==0) {
                const auto &c=static_cast<ResidentQwenRecurrent*>(layer.context)->cfg;
                ffn=std::max(ffn,size_t(c.ffn));inner=std::max(inner,size_t(2*c.num_k+c.num_v)*c.head);
                values=std::max(values,size_t(c.num_v)*c.head);nv=std::max(nv,size_t(c.num_v));
            } else {
                const auto *r=static_cast<ResidentQwenAttention*>(layer.context);const auto &c=r->cfg;
                ffn=std::max(ffn,size_t(c.ffn));qs=std::max(qs,size_t(c.heads)*c.head_dim);
                qraw=std::max(qraw,size_t(c.heads)*c.head_dim*(1+c.gated));ks=std::max(ks,size_t(c.kv_heads)*c.head_dim);
                if(r->split_kv)workspace=std::max(workspace,size_t(c.heads)*((c.capacity+attention_tile-1)/attention_tile)*(c.head_dim+2));
            }
        }
        return alloc(x,embd) && alloc(h,embd) && alloc(projection,embd) && alloc(gate,ffn) && alloc(up,ffn)
            && alloc(mixed,inner) && alloc(activated,inner) && alloc(z,values) && alloc(beta,nv) && alloc(alpha,nv)
            && alloc(recurrent_out,values) && alloc(normed,values) && alloc(q_full,qraw) && alloc(q,qs)
            && alloc(k_raw,ks) && alloc(k,ks) && alloc(v,ks) && alloc(attention,qs) && alloc(scratch,workspace);
    }
};
void qwen_block_matrix(const RbitnetLlamaMatrix &m,const float *x,float *y,unsigned count,cudaStream_t stream,bool tensor) {
    QuantKind kind;resident_kind(m.type,kind);
    launch_prefill_gemm(kind,m.weights,m.row_bytes,x,m.cols,m.rows,count,y,stream,use_tf32_prefill(m.cols,m.rows,count,tensor));
}
void qwen_recurrent_block(ResidentQwenRecurrent *r,QwenPrefill *b,unsigned count,cudaStream_t stream,bool tensor,bool ordered=false) {
    const auto &c=r->cfg;unsigned inner=(2*c.num_k+c.num_v)*c.head,values=c.num_v*c.head;
    auto matrix=[&](unsigned i,const float *x,float *y) {if(ordered){QuantKind kind;resident_kind(r->matrices[i].type,kind);const auto &m=r->matrices[i];launch_ordered_gemm(kind,static_cast<const uint8_t*>(m.weights),m.row_bytes,x,m.cols,m.rows,count,y,stream);}else qwen_block_matrix(r->matrices[i],x,y,count,stream,tensor);};
    resident_norm<<<count,256,0,stream>>>(b->x,r->attn_norm,c.epsilon,c.embd,b->h);
    matrix(0,b->h,b->mixed);matrix(1,b->h,b->z);matrix(2,b->h,b->beta);matrix(3,b->h,b->alpha);
    qwen_conv_block<<<(inner+255)/256,256,0,stream>>>(b->mixed,r->history,r->conv,c.conv,inner,b->activated,count);
    qwen_l2<<<dim3(2*c.num_k,count),128,0,stream>>>(b->activated,c.head,c.epsilon,inner);
    qwen_delta_block<<<(values+3)/4,128,0,stream>>>(r->state,b->activated,b->alpha,b->beta,r->dt,r->a,c.head,c.num_k,c.num_v,b->recurrent_out,count);
    qwen_norm_gate<<<dim3(c.num_v,count),128,0,stream>>>(b->recurrent_out,b->z,r->ssm_norm,c.head,c.epsilon,b->normed,values);
    matrix(4,b->normed,b->projection);
    resident_norm<<<count,256,0,stream>>>(b->x,r->ffn_norm,c.epsilon,c.embd,b->h,b->projection);
    matrix(5,b->h,b->gate);matrix(6,b->h,b->up);
    resident_silu<<<(count*c.ffn+255)/256,256,0,stream>>>(b->gate,b->up,count*c.ffn);
    matrix(7,b->gate,b->projection);resident_add<<<(count*c.embd+255)/256,256,0,stream>>>(b->x,b->projection,count*c.embd);
}
void qwen_attention_block(ResidentQwenAttention *r,QwenPrefill *b,unsigned *position,unsigned count,cudaStream_t stream,bool tensor,bool ordered=false) {
    const auto &c=r->cfg;unsigned qs=c.heads*c.head_dim,ks=c.kv_heads*c.head_dim;
    auto matrix=[&](unsigned i,const float *x,float *y) {if(ordered){QuantKind kind;resident_kind(r->matrices[i].type,kind);const auto &m=r->matrices[i];launch_ordered_gemm(kind,static_cast<const uint8_t*>(m.weights),m.row_bytes,x,m.cols,m.rows,count,y,stream);}else qwen_block_matrix(r->matrices[i],x,y,count,stream,tensor);};
    resident_norm<<<count,256,0,stream>>>(b->x,r->attn_norm,c.epsilon,c.embd,b->h);
    matrix(0,b->h,b->q_full);matrix(1,b->h,b->k_raw);matrix(2,b->h,b->v);
    qwen_head_norm<<<dim3(c.heads,count),128,0,stream>>>(b->q_full,r->q_norm,c.head_dim,c.head_dim*(1+c.gated),c.epsilon,b->q,c.heads);
    qwen_head_norm<<<dim3(c.kv_heads,count),128,0,stream>>>(b->k_raw,r->k_norm,c.head_dim,c.head_dim,c.epsilon,b->k,c.kv_heads);
    if(c.rotary)qwen_neox_rope<<<dim3(((c.heads+c.kv_heads)*(c.rotary/2)+255)/256,count),256,0,stream>>>(b->q,b->k,r->frequency,position,c.heads,c.kv_heads,c.head_dim,c.rotary);
    qwen_write_kv<<<dim3((ks+255)/256,count),256,0,stream>>>(b->k,b->v,r->kv_k,r->kv_v,position,ks);
    // All block keys may already be written, but each query's softmax reads
    // only [0, position + token], excluding the future part of the block.
    if(r->split_kv)launch_split_attention(r->kv_k,r->kv_v,b->q,position,c.kv_heads,c.heads,c.head_dim,0,c.scale,c.capacity,count,b->scratch,b->attention,stream);
    else resident_attention<<<dim3(c.heads,count),128,c.capacity*sizeof(float),stream>>>(r->kv_k,r->kv_v,b->q,position,c.kv_heads,c.heads,c.head_dim,0,c.scale,b->attention);
    qwen_attention_gate<<<dim3((qs+255)/256,count),256,0,stream>>>(b->attention,b->q_full,c.heads,c.head_dim,c.gated);
    matrix(3,b->attention,b->projection);
    resident_norm<<<count,256,0,stream>>>(b->x,r->ffn_norm,c.epsilon,c.embd,b->h,b->projection);
    matrix(4,b->h,b->gate);matrix(5,b->h,b->up);
    resident_silu<<<(count*c.ffn+255)/256,256,0,stream>>>(b->gate,b->up,count*c.ffn);
    matrix(6,b->gate,b->projection);resident_add<<<(count*c.embd+255)/256,256,0,stream>>>(b->x,b->projection,count*c.embd);
}
