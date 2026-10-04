// SPDX-License-Identifier: MIT
// Resident output head, preserving Rust's NaN, total_cmp and last-ID tie policy.
namespace {
struct ResidentHead {
    RbitnetLlamaMatrix matrix;float epsilon;
    float *x=nullptr,*h=nullptr,*norm=nullptr,*logits=nullptr,*maxima=nullptr,*maximum=nullptr;
    unsigned *ids=nullptr,*token=nullptr;
    std::vector<void*> allocations;cudaStream_t stream=nullptr;
    cudaGraph_t graphs[2]={};cudaGraphExec_t executable[2]={};
    ~ResidentHead() {
        if(stream)cudaStreamSynchronize(stream);
        for(auto e:executable)if(e)cudaGraphExecDestroy(e);
        for(auto g:graphs)if(g)cudaGraphDestroy(g);
        for(auto p:allocations)cudaFree(p);if(stream)cudaStreamDestroy(stream);
    }
    template<typename T> bool alloc(T *&p,size_t n,const T *host=nullptr) {
        if(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(p);return !host || (cudaMemcpyAsync(p,host,n*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess
            && cudaStreamSynchronize(stream)==cudaSuccess);
    }
    void enqueue(unsigned mode) {
        resident_norm<<<1,256,0,stream>>>(x,norm,epsilon,matrix.cols,h);
        QuantKind kind;resident_kind(matrix.type,kind);
        launch_quant_kernel(kind,matrix.weights,matrix.row_bytes,h,matrix.cols,matrix.rows,matrix.rows,logits,stream);
        if(mode) {
            unsigned blocks=(matrix.rows+255)/256;
            resident_argmax<<<blocks,256,0,stream>>>(logits,nullptr,matrix.rows,maxima,ids);
            resident_argmax<<<1,256,0,stream>>>(maxima,ids,blocks,maximum,token);
        }
    }
};
}
extern "C" {
void *rbitnet_cuda_head_create(const RbitnetLlamaMatrix *m,const float *norm,float epsilon) {
    if(!m || !norm || !m->cols || m->cols>65536 || !m->rows || m->rows>2097152
        || !isfinite(epsilon) || epsilon<=0 || !qwen_matrix_valid(*m,m->cols,m->rows))return nullptr;
    auto *r=new(std::nothrow) ResidentHead;if(!r)return nullptr;r->matrix=*m;r->epsilon=epsilon;
    unsigned blocks=(m->rows+255)/256;
    if(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking)!=cudaSuccess
        || !r->alloc(r->x,m->cols) || !r->alloc(r->h,m->cols) || !r->alloc(r->norm,m->cols,norm)
        || !r->alloc(r->logits,m->rows) || !r->alloc(r->maxima,blocks) || !r->alloc(r->ids,blocks)
        || !r->alloc(r->maximum,1) || !r->alloc(r->token,1)) {delete r;return nullptr;}
    return r;
}
void rbitnet_cuda_head_destroy(void *context) {delete static_cast<ResidentHead*>(context);}
int rbitnet_cuda_head_step(void *context,const float *input,unsigned mode,float *logits,unsigned *token) {
    auto *r=static_cast<ResidentHead*>(context);
    if(!r || !input || mode>1 || (!mode && !logits) || (mode && !token))return 1;
    if(cudaMemcpyAsync(r->x,input,r->matrix.cols*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 2;
    if(!r->executable[mode]) {
        if(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 3;
        r->enqueue(mode);
        if(cudaStreamEndCapture(r->stream,&r->graphs[mode])!=cudaSuccess || cudaGraphInstantiate(&r->executable[mode],r->graphs[mode],0)!=cudaSuccess)return 4;
    }
    if(cudaGraphLaunch(r->executable[mode],r->stream)!=cudaSuccess || cudaGetLastError()!=cudaSuccess)return 5;
    if(!mode && cudaMemcpyAsync(logits,r->logits,r->matrix.rows*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 6;
    if(mode && cudaMemcpyAsync(token,r->token,sizeof(unsigned),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess)return 7;
    if(cudaStreamSynchronize(r->stream)!=cudaSuccess)return 8;
    return 0;
}
}
