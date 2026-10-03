// SPDX-License-Identifier: MIT
// Routed experts share one private stream and one host synchronization per layer.
namespace {
template<QuantKind kind>
__global__ void moe_matrix(const uint8_t *w,size_t row_bytes,unsigned cols,unsigned rows,
    const unsigned *experts,const float *x,bool independent,float *y) {
    unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/32,slot=blockIdx.y;
    if(row>=rows)return;
    const uint8_t *weights=w+(size_t(experts[slot])*rows+row)*row_bytes;
    float sum=quant_row_dot<kind>(weights,x+(independent?size_t(slot)*cols:0),cols);
    if((threadIdx.x&31)==0)y[size_t(slot)*rows+row]=sum;
}
__global__ void moe_activate(float *gate,const float *up,const float *gb,const float *ub,
    const unsigned *experts,unsigned rows,unsigned used,bool oai) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*used)return;
    size_t b=size_t(experts[i/rows])*rows+i%rows;
    float g=gate[i]+(gb?gb[b]:0.0f),u=up[i]+(ub?ub[b]:0.0f);
    if(oai) {g=fminf(g,7.0f);u=fminf(fmaxf(u,-7.0f),7.0f)+1.0f;}
    gate[i]=(g/(1.0f+expf(-(oai?1.702f:1.0f)*g)))*u;
}
__global__ void moe_combine(const float *values,const float *bias,const unsigned *experts,
    const float *probabilities,unsigned rows,unsigned used,float *out) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows)return;
    float sum=0;
    for(unsigned s=0;s<used;s++) {
        float value=values[size_t(s)*rows+i]+(bias?bias[size_t(experts[s])*rows+i]:0.0f);
        // Keep the per-expert rounding used by the Rust graph.
        sum=__fadd_rn(sum,__fmul_rn(probabilities[s],value));
    }
    out[i]=sum;
}
struct ResidentMoe {
    RbitnetMoeConfig cfg;
    RbitnetLlamaMatrix gate,up,down;
    std::vector<void*> allocations;
    float *input=nullptr,*g=nullptr,*u=nullptr,*d=nullptr,*output=nullptr,*gb=nullptr,*ub=nullptr,*db=nullptr,*probabilities=nullptr;
    unsigned *experts=nullptr;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    ~ResidentMoe() {
        if(stream)cudaStreamSynchronize(stream);
        if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        for(auto p:allocations)cudaFree(p);if(stream)cudaStreamDestroy(stream);
    }
    template<typename T> bool alloc(T *&p,size_t n,const T *host=nullptr) {
        if(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(p);
        return !host || cudaMemcpy(p,host,n*sizeof(T),cudaMemcpyHostToDevice)==cudaSuccess;
    }
    void matrix(const RbitnetLlamaMatrix &m,const float *x,bool independent,float *y) {
        QuantKind kind;resident_kind(m.type,kind);
        unsigned rows=m.rows/cfg.experts;dim3 grid((rows+7)/8,cfg.used);
#define MOE_QUANT(K) case QuantKind::K: moe_matrix<QuantKind::K><<<grid,256,0,stream>>>(static_cast<const uint8_t*>(m.weights),m.row_bytes,m.cols,rows,experts,x,independent,y);break
        switch(kind) {MOE_QUANT(F32);MOE_QUANT(Q4_0);MOE_QUANT(Q5_0);MOE_QUANT(Q8_0);MOE_QUANT(Q4_K);MOE_QUANT(Q5_K);MOE_QUANT(Q6_K);MOE_QUANT(MXFP4);}
#undef MOE_QUANT
    }
    void enqueue() {
        matrix(gate,input,false,g);matrix(up,input,false,u);
        moe_activate<<<(cfg.ffn*cfg.used+255)/256,256,0,stream>>>(g,u,gb,ub,experts,cfg.ffn,cfg.used,cfg.oai!=0);
        matrix(down,g,true,d);
        moe_combine<<<(cfg.embd+255)/256,256,0,stream>>>(d,db,experts,probabilities,cfg.embd,cfg.used,output);
    }
};
}
extern "C" {
void *rbitnet_cuda_moe_create(const RbitnetMoeConfig *c,const RbitnetLlamaMatrix *gate,
    const RbitnetLlamaMatrix *up,const RbitnetLlamaMatrix *down,const float *gb,const float *ub,const float *db) {
    if(!c || !gate || !up || !down || !c->embd || !c->ffn || !c->experts || !c->used || c->used>c->experts || c->used>128)return nullptr;
    QuantKind kind;
    for(auto m:{*gate,*up,*down})if(!m.weights || !resident_kind(m.type,kind))return nullptr;
    if(gate->cols!=c->embd || up->cols!=c->embd || down->cols!=c->ffn
        || gate->rows!=c->ffn*c->experts || up->rows!=gate->rows || down->rows!=c->embd*c->experts)return nullptr;
    auto *r=new(std::nothrow) ResidentMoe;if(!r)return nullptr;
    r->cfg=*c;r->gate=*gate;r->up=*up;r->down=*down;
    if(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking)!=cudaSuccess
        || !r->alloc(r->input,c->embd) || !r->alloc(r->g,size_t(c->used)*c->ffn)
        || !r->alloc(r->u,size_t(c->used)*c->ffn) || !r->alloc(r->d,size_t(c->used)*c->embd)
        || !r->alloc(r->output,c->embd) || !r->alloc(r->experts,c->used) || !r->alloc(r->probabilities,c->used)
        || (gb && !r->alloc(r->gb,size_t(c->experts)*c->ffn,gb))
        || (ub && !r->alloc(r->ub,size_t(c->experts)*c->ffn,ub))
        || (db && !r->alloc(r->db,size_t(c->experts)*c->embd,db))) {delete r;return nullptr;}
    return r;
}
void rbitnet_cuda_moe_destroy(void *context) {delete static_cast<ResidentMoe*>(context);}
int rbitnet_cuda_moe_step(void *context,const float *input,const unsigned *experts,const float *probabilities,float *output) {
    auto *r=static_cast<ResidentMoe*>(context);if(!r || !input || !experts || !probabilities || !output)return 1;
    for(unsigned s=0;s<r->cfg.used;s++)if(experts[s]>=r->cfg.experts)return 2;
    if(cudaMemcpyAsync(r->input,input,r->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->experts,experts,r->cfg.used*sizeof(unsigned),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->probabilities,probabilities,r->cfg.used*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 3;
    if(!r->executable) {
        if(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 4;
        r->enqueue();
        if(cudaStreamEndCapture(r->stream,&r->graph)!=cudaSuccess || cudaGraphInstantiate(&r->executable,r->graph,0)!=cudaSuccess)return 5;
    }
    if(cudaGraphLaunch(r->executable,r->stream)!=cudaSuccess || cudaGetLastError()!=cudaSuccess)return 6;
    if(cudaMemcpyAsync(output,r->output,r->cfg.embd*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess
        || cudaStreamSynchronize(r->stream)!=cudaSuccess)return 7;
    return 0;
}
}
