// SPDX-License-Identifier: MIT
// Group (token, selected-slot) work by expert, without changing
// router IDs, probabilities, dot ordering or the final selected-slot reduction.
// Include after moe_resident.cuh. Workspaces are allocated before inference.
#include <limits>
#include <memory>
namespace {
struct MoeGroupChunk { unsigned expert,count,item[4]; };

__global__ void moe_group_worklist(const unsigned *ids,unsigned items,unsigned experts,
    MoeGroupChunk *chunks,unsigned *chunk_count) {
    if(threadIdx.x || blockIdx.x)return;
    unsigned count=0;
    // Stable expert order and stable original item order within each expert.
    // Only existing groups launch useful work; there is no experts*items padding.
    for(unsigned expert=0;expert<experts;expert++) {
        unsigned local=0;
        for(unsigned item=0;item<items;item++)if(ids[item]==expert) {
            if(!local) {chunks[count].expert=expert;chunks[count].count=0;}
            chunks[count].item[local++]=item;chunks[count].count=local;
            if(local==4) {count++;local=0;}
        }
        if(local)count++;
    }
    *chunk_count=count;
}

template<QuantKind kind>
__global__ void moe_group_ordered_matrix(const uint8_t *weights,size_t row_bytes,
    const float *input,unsigned cols,unsigned rows,unsigned used,bool independent,
    const MoeGroupChunk *chunks,const unsigned *chunk_count,float *output) {
    if(blockIdx.y>=*chunk_count)return;
    const auto &chunk=chunks[blockIdx.y];
    const unsigned warp=threadIdx.x/32,lane=threadIdx.x&31;
    const unsigned row=blockIdx.x*2+warp/4,local=warp%4;
    // Conditions are uniform within a warp. No lane can miss a shuffle.
    if(row>=rows || local>=chunk.count)return;
    const unsigned item=chunk.item[local];
    const unsigned source=independent?item:item/used;
    const uint8_t *w=weights+(size_t(chunk.expert)*rows+row)*row_bytes;
    const float value=quant_row_dot<kind>(w,input+size_t(source)*cols,cols);
    if(!lane)output[size_t(item)*rows+row]=value;
}

__global__ void moe_group_mxfp4_matrix(const uint8_t *weights,size_t row_bytes,
    const float *input,unsigned cols,unsigned rows,unsigned used,bool independent,
    const MoeGroupChunk *chunks,const unsigned *chunk_count,float *output) {
    if(blockIdx.y>=*chunk_count)return; // uniform CTA condition before barriers
    const auto &chunk=chunks[blockIdx.y];
    __shared__ __align__(16) uint8_t ws[4][8*17];
    __shared__ __align__(16) float xs[4][8*48];
    const unsigned tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const unsigned local_row=warp/4,local_item=warp%4;
    const unsigned row=blockIdx.x*4+local_row;
    const unsigned group=lane/4,j=(lane%4)*4,blocks=cols/32;
    float sum=0;
    for(unsigned base=0;base<blocks;base+=8) {
        for(unsigned i=tid;i<4*8*17;i+=512) {
            const unsigned r=i/(8*17),byte=i%(8*17),qb=byte/17;
            const unsigned source_row=blockIdx.x*4+r;
            ws[r][byte]=(source_row<rows && base+qb<blocks)
                ?weights[(size_t(chunk.expert)*rows+source_row)*row_bytes+size_t(base)*17+byte]:0;
        }
        for(unsigned i=tid;i<4*8*32;i+=512) {
            const unsigned local=i/(8*32),column=i%(8*32),qb=column/32,k=column%32;
            if(local<chunk.count && base+qb<blocks) {
                const unsigned item=chunk.item[local],source=independent?item:item/used;
                xs[local][qb*48+k]=input[size_t(source)*cols+size_t(base+qb)*32+k];
            } else xs[local][qb*48+k]=0;
        }
        __syncthreads();
        if(row<rows && local_item<chunk.count && base+group<blocks) {
            const uint8_t *b=ws[local_row]+group*17;
            const float d=__uint_as_float(b[0]<2?(0x00200000u<<b[0]):((uint32_t(b[0])-1)<<23));
            const uint32_t packed=uint32_t(b[1+j])|(uint32_t(b[2+j])<<8)
                |(uint32_t(b[3+j])<<16)|(uint32_t(b[4+j])<<24);
            #pragma unroll
            for(unsigned half=0;half<2;half++) {
                const float4 x=*reinterpret_cast<const float4*>(xs[local_item]+group*48+half*16+j);
                #pragma unroll
                for(unsigned k=0;k<4;k++) {
                    const unsigned nibble=(packed>>(k*8+half*4))&15;
                    const int magnitude=(0xC8643210u>>((nibble&7)*4))&15;
                    const int q=(nibble&8)?-magnitude:magnitude;
                    const float value=k==0?x.x:k==1?x.y:k==2?x.z:x.w;
                    sum=fmaf(d*float(q),value,sum);
                }
            }
        }
        __syncthreads();
    }
    for(int shift=16;shift>0;shift/=2)sum+=__shfl_down_sync(0xffffffff,sum,shift);
    if(!lane && row<rows && local_item<chunk.count)
        output[size_t(chunk.item[local_item])*rows+row]=sum;
}

void launch_moe_group_matrix(const RbitnetLlamaMatrix &matrix,unsigned experts,
    const float *input,unsigned used,bool independent,const MoeGroupChunk *chunks,
    const unsigned *chunk_count,unsigned max_chunks,float *output,cudaStream_t stream,unsigned tile) {
    QuantKind kind;resident_kind(matrix.type,kind);const unsigned rows=matrix.rows/experts;
    if(kind==QuantKind::MXFP4 && tile) {
        dim3 grid((rows+3)/4,max_chunks);
        moe_group_mxfp4_matrix<<<grid,512,0,stream>>>(static_cast<const uint8_t*>(matrix.weights),
            matrix.row_bytes,input,matrix.cols,rows,used,independent,chunks,chunk_count,output);return;
    }
    dim3 grid((rows+1)/2,max_chunks);
#define GROUP_MOE(K) case QuantKind::K: moe_group_ordered_matrix<QuantKind::K><<<grid,256,0,stream>>>(static_cast<const uint8_t*>(matrix.weights),matrix.row_bytes,input,matrix.cols,rows,used,independent,chunks,chunk_count,output);break
    switch(kind) {GROUP_MOE(F32);GROUP_MOE(Q4_0);GROUP_MOE(Q5_0);GROUP_MOE(Q8_0);GROUP_MOE(Q4_K);GROUP_MOE(Q5_K);GROUP_MOE(Q6_K);GROUP_MOE(MXFP4);}
#undef GROUP_MOE
}

__global__ void moe_group_combine(const float *values,const float *bias,const unsigned *ids,
    const float *probabilities,unsigned rows,unsigned used,unsigned count,float *output) {
    const unsigned row=blockIdx.x*blockDim.x+threadIdx.x,token=blockIdx.y;
    if(row>=rows || token>=count)return;
    float sum=0;
    for(unsigned slot=0;slot<used;slot++) {
        const unsigned item=token*used+slot;
        const float value=values[size_t(item)*rows+row]+(bias?bias[size_t(ids[item])*rows+row]:0.0f);
        sum=__fadd_rn(sum,__fmul_rn(probabilities[item],value));
    }
    output[size_t(token)*rows+row]=sum;
}

struct MoeGroupedWorkspace {
    ResidentMoe *source=nullptr;
    cudaStream_t last_stream=nullptr;
    unsigned capacity=0,max_items=0;
    std::vector<void*> allocations;
    float *g=nullptr,*u=nullptr,*d=nullptr;
    MoeGroupChunk *chunks=nullptr;unsigned *chunk_count=nullptr;
    ~MoeGroupedWorkspace() {
        if(last_stream)cudaStreamSynchronize(last_stream);
        for(auto pointer:allocations)cudaFree(pointer);
    }
    template<typename T> bool alloc(T *&pointer,size_t count) {
        if(!count || count>std::numeric_limits<size_t>::max()/sizeof(T))return false;
        if(cudaMalloc(reinterpret_cast<void**>(&pointer),count*sizeof(T))!=cudaSuccess)return false;
        try {allocations.push_back(pointer);}
        catch(const std::bad_alloc&) {cudaFree(pointer);pointer=nullptr;return false;}
        return true;
    }
    bool init(ResidentMoe *moe,unsigned count) {
        if(!moe || moe->dynamic || !count || count>32 || !moe->cfg.used
            || moe->cfg.used>16 || !moe->cfg.experts || moe->cfg.experts>128)return false;
        source=moe;last_stream=moe->stream;capacity=count;max_items=count*moe->cfg.used;
        return alloc(g,size_t(max_items)*moe->cfg.ffn)
            && alloc(u,size_t(max_items)*moe->cfg.ffn)
            && alloc(d,size_t(max_items)*moe->cfg.embd)
            && alloc(chunks,max_items) && alloc(chunk_count,1);
    }
    // Caller retains input/ID/probability/output device buffers and borrowed
    // fixed expert bank until the encompassing stream has completed.
    bool enqueue(const float *input,const unsigned *ids,const float *probabilities,
        float *output,unsigned count,unsigned tile) {
        if(!source || !input || !ids || !probabilities || !output || !count || count>capacity || tile>1)return false;
        const auto &c=source->cfg;const unsigned items=count*c.used;const auto stream=source->stream;
        last_stream=stream;
        moe_group_worklist<<<1,1,0,stream>>>(ids,items,c.experts,chunks,chunk_count);
        launch_moe_group_matrix(source->gate,c.experts,input,c.used,false,chunks,chunk_count,items,g,stream,tile);
        launch_moe_group_matrix(source->up,c.experts,input,c.used,false,chunks,chunk_count,items,u,stream,tile);
        // Original elementwise arithmetic and bias lookup, flattened items.
        moe_activate<<<(c.ffn*items+255)/256,256,0,stream>>>(g,u,source->gb,source->ub,ids,c.ffn,items,c.oai!=0);
        launch_moe_group_matrix(source->down,c.experts,g,c.used,true,chunks,chunk_count,items,d,stream,tile);
        dim3 combine((c.embd+255)/256,count);
        moe_group_combine<<<combine,256,0,stream>>>(d,source->db,ids,probabilities,c.embd,c.used,count,output);
        return true;
    }
};

struct MoeGroupDiagnostic {
    ResidentMoe *source=nullptr;
    std::unique_ptr<MoeGroupedWorkspace> workspace;
    std::vector<void*> allocations;
    cudaEvent_t start=nullptr,stop=nullptr;
    float *input=nullptr,*probabilities=nullptr,*output=nullptr,*reference=nullptr;
    unsigned *ids=nullptr;
    ~MoeGroupDiagnostic() {
        if(source)cudaStreamSynchronize(source->stream);
        workspace.reset();
        if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);
        delete source;
        for(auto p:allocations)cudaFree(p);
    }
    template<typename T> bool alloc(T *&pointer,size_t n,unsigned category=MemoryActivation) {
        if(!n || n>std::numeric_limits<size_t>::max()/sizeof(T))return false;
        MemoryCategoryScope scope(category);
        if(cudaMalloc(reinterpret_cast<void**>(&pointer),n*sizeof(T))!=cudaSuccess)return false;
        try {allocations.push_back(pointer);}
        catch(const std::bad_alloc&) {cudaFree(pointer);pointer=nullptr;return false;}
        return true;
    }
    void serial(float *out,unsigned count) {
        // The production single-token kernels are the numerical baseline.
        auto *m=source;
        auto *old_input=m->input,*old_output=m->output,*old_probabilities=m->probabilities;
        auto *old_ids=m->experts;
        for(unsigned token=0;token<count;token++) {
            m->input=input+size_t(token)*m->cfg.embd;m->output=out+size_t(token)*m->cfg.embd;
            m->experts=ids+token*m->cfg.used;m->probabilities=probabilities+token*m->cfg.used;
            m->enqueue();
        }
        m->input=old_input;m->output=old_output;m->experts=old_ids;m->probabilities=old_probabilities;
    }
};
}

extern "C" {
// Host-array oracle. mode 0=grouped ordered warps, 1=shared-byte MXFP4 tiles,
// 2=production serial FFNs. graphs 0/1; elapsed excludes allocation/copies/capture.
int rbitnet_cuda_moe_group_check(const RbitnetMoeConfig *c,const RbitnetLlamaMatrix *matrices,
    const float *gate_bias,const float *up_bias,const float *down_bias,const float *input,
    const unsigned *ids,const float *probabilities,unsigned count,unsigned mode,unsigned graphs,
    unsigned repeats,float *out,float *reference,float *elapsed_ms) {
    if(!c || !matrices || !input || !ids || !probabilities || !out || !reference || !elapsed_ms
        || !count || count>32 || mode>2 || graphs>1 || !repeats || repeats>100
        || !c->embd || c->embd>8192 || !c->ffn || c->ffn>8192 || !c->experts || c->experts>128
        || !c->used || c->used>16 || c->used>c->experts || c->oai>1)return 1;
    for(unsigned i=0;i<count*c->used;i++)if(ids[i]>=c->experts || !isfinite(probabilities[i]))return 1;
    for(unsigned token=0;token<count;token++)for(unsigned slot=0;slot<c->used;slot++)
        for(unsigned earlier=0;earlier<slot;earlier++)if(ids[token*c->used+slot]==ids[token*c->used+earlier])return 1;
    const unsigned columns[3]={c->embd,c->embd,c->ffn},rows[3]={c->ffn,c->ffn,c->embd};
    for(unsigned i=0;i<3;i++)if(!qwen_matrix_valid(matrices[i],columns[i],rows[i]*c->experts)
        || size_t(matrices[i].rows)*matrices[i].row_bytes>256u*1024u*1024u)return 1;
    MoeGroupDiagnostic d;RbitnetLlamaMatrix gpu[3];
    for(unsigned i=0;i<3;i++) {
        gpu[i]=matrices[i];uint8_t *w=nullptr;
        const size_t bytes=size_t(gpu[i].rows)*gpu[i].row_bytes;
        if(!d.alloc(w,bytes,MemoryWeights) || cudaMemcpy(w,gpu[i].weights,bytes,cudaMemcpyHostToDevice)!=cudaSuccess)return 2;
        gpu[i].weights=w;
    }
    d.source=static_cast<ResidentMoe*>(moe_create(c,&gpu[0],&gpu[1],&gpu[2],gate_bias,up_bias,down_bias,false));
    if(!d.source)return 2;
    const auto stream=d.source->stream;NativeCallCompletion completion(stream);
    d.workspace=std::unique_ptr<MoeGroupedWorkspace>(new(std::nothrow) MoeGroupedWorkspace);
    const size_t inputs=size_t(count)*c->embd,items=size_t(count)*c->used;
    if(!d.workspace || !d.workspace->init(d.source,count)
        || !d.alloc(d.input,inputs) || !d.alloc(d.ids,items) || !d.alloc(d.probabilities,items)
        || !d.alloc(d.output,inputs) || !d.alloc(d.reference,inputs)
        || cudaEventCreate(&d.start)!=cudaSuccess || cudaEventCreate(&d.stop)!=cudaSuccess
        || cudaMemcpyAsync(d.input,input,inputs*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess
        || cudaMemcpyAsync(d.ids,ids,items*sizeof(unsigned),cudaMemcpyHostToDevice,stream)!=cudaSuccess
        || cudaMemcpyAsync(d.probabilities,probabilities,items*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess)return 2;
    if(completion.complete(0,3)!=0)return 3;
    auto enqueue=[&] {
        if(mode==2) {d.serial(d.output,count);return true;}
        return d.workspace->enqueue(d.input,d.ids,d.probabilities,d.output,count,mode);
    };
    // Destruction drains and also abandons a failed capture before releasing
    // device buffers. Successful graphs are destroyed before their workspace.
    struct Captured {
        cudaStream_t stream;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        ~Captured() {NativeCallCompletion finish(stream);finish.complete(0,1);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
    } captured{stream};
    if(graphs) {
        if(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 4;
        if(!enqueue())return 4;
        if(cudaStreamEndCapture(stream,&captured.graph)!=cudaSuccess
            || cudaGraphInstantiate(&captured.executable,captured.graph,0)!=cudaSuccess)return 4;
    }
    auto launch=[&] {
        if(graphs)return cudaGraphLaunch(captured.executable,stream);
        if(!enqueue())return cudaErrorInvalidValue;
        return cudaGetLastError();
    };
    if(launch()!=cudaSuccess || cudaEventRecord(d.start,stream)!=cudaSuccess)return 5;
    for(unsigned i=0;i<repeats;i++)if(launch()!=cudaSuccess)return 5;
    if(cudaGetLastError()!=cudaSuccess || cudaEventRecord(d.stop,stream)!=cudaSuccess
        || cudaEventSynchronize(d.stop)!=cudaSuccess
        || cudaEventElapsedTime(elapsed_ms,d.start,d.stop)!=cudaSuccess)return 5;
    *elapsed_ms/=float(repeats);
    d.serial(d.reference,count);
    NativeCallCompletion download(stream);
    if(cudaGetLastError()!=cudaSuccess
        || cudaMemcpyAsync(out,d.output,inputs*sizeof(float),cudaMemcpyDeviceToHost,stream)!=cudaSuccess
        || cudaMemcpyAsync(reference,d.reference,inputs*sizeof(float),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)return 6;
    return download.complete(0,6);
}
}
