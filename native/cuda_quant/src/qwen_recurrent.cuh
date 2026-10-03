// SPDX-License-Identifier: MIT
// Qwen3.5 dense recurrent block. Private state, allocations and stream per layer.
// Follow Rbitnet's GGUF/F32 equations; Q/K L2 uses max(sum, epsilon), not sum+eps.
namespace {
// Keep the single-token kernels unchanged; block loops have separate code.
__global__ void qwen_conv(const float *input,float *history,const float *weights,
    unsigned taps,unsigned width,float *activated) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width)return;
    float sum=0;
    for(unsigned t=0;t+1<taps;t++) {
        float value=history[size_t(t)*width+i];
        sum=__fadd_rn(sum,__fmul_rn(value,weights[size_t(i)*taps+t]));
        history[size_t(t)*width+i]=(t+2<taps)?history[size_t(t+1)*width+i]:input[i];
    }
    sum=__fadd_rn(sum,__fmul_rn(input[i],weights[size_t(i)*taps+taps-1]));
    activated[i]=sum/(1.0f+expf(-sum));
}
__global__ void qwen_delta(float *state,const float *qkv,const float *alpha,const float *beta,
    const float *dt,const float *a,unsigned head,unsigned num_k,unsigned num_v,float *out) {
    unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;
    if(row>=num_v*head)return;
    unsigned vh=row/head,kh=vh%num_k;
    float t=alpha[vh]+dt[vh];
    float sp=t>35.0f?t:(t<-35.0f?0.0f:log1pf(expf(t)));
    float decay=expf(sp*a[vh]),b=1.0f/(1.0f+expf(-beta[vh]));
    float cells[8],keys[8],queries[8],dot=0;
    // At most 256 keys per row: one warp owns each value row. No state copy.
    #pragma unroll
    for(unsigned j=0;j<8;j++) {
        unsigned col=lane+j*32;
        if(col<head) {
            size_t index=size_t(row)*head+col;
            cells[j]=__fmul_rn(state[index],decay);
            keys[j]=qkv[size_t(num_k+kh)*head+col];
            queries[j]=qkv[size_t(kh)*head+col];
            dot=fmaf(cells[j],keys[j],dot);
        }
    }
    for(int s=16;s;s/=2)dot+=__shfl_down_sync(0xffffffff,dot,s);
    dot=__shfl_sync(0xffffffff,dot,0);
    float delta=(qkv[size_t(2*num_k)*head+row]-dot)*b,output=0;
    #pragma unroll
    for(unsigned j=0;j<8;j++) {
        unsigned col=lane+j*32;
        if(col<head) {
            float value=__fadd_rn(cells[j],__fmul_rn(keys[j],delta));
            state[size_t(row)*head+col]=value;
            output=fmaf(value,queries[j],output);
        }
    }
    for(int s=16;s;s/=2)output+=__shfl_down_sync(0xffffffff,output,s);
    if(!lane)out[row]=output/sqrtf(float(head));
}
__global__ void qwen_conv_block(const float *input,float *history,const float *weights,
    unsigned taps,unsigned width,float *activated,unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width)return;
    float previous[16];
    for(unsigned t=0;t+1<taps;t++)previous[t]=history[size_t(t)*width+i];
    for(unsigned token=0;token<count;token++) {
        float value=input[size_t(token)*width+i],sum=0;
        for(unsigned t=0;t+1<taps;t++) {
            sum=__fadd_rn(sum,__fmul_rn(previous[t],weights[size_t(i)*taps+t]));
            previous[t]=(t+2<taps)?previous[t+1]:value;
        }
        sum=__fadd_rn(sum,__fmul_rn(value,weights[size_t(i)*taps+taps-1]));
        activated[size_t(token)*width+i]=sum/(1.0f+expf(-sum));
    }
    for(unsigned t=0;t+1<taps;t++)history[size_t(t)*width+i]=previous[t];
}
__global__ void qwen_l2(float *qkv,unsigned head,float epsilon,unsigned stride=0) {
    float *x=qkv+size_t(blockIdx.y)*stride+size_t(blockIdx.x)*head;
    unsigned lane=threadIdx.x&31,warp=threadIdx.x/32;
    __shared__ float sums[4];
    float sum=0;
    for(unsigned i=threadIdx.x;i<head;i+=128)sum+=x[i]*x[i];
    for(int s=16;s;s/=2)sum+=__shfl_down_sync(0xffffffff,sum,s);
    if(!lane)sums[warp]=sum;
    __syncthreads();
    float inv=1.0f/sqrtf(fmaxf(sums[0]+sums[1]+sums[2]+sums[3],epsilon));
    for(unsigned i=threadIdx.x;i<head;i+=128)x[i]*=inv;
}
__global__ void qwen_delta_block(float *state,const float *qkv,const float *alpha,const float *beta,
    const float *dt,const float *a,unsigned head,unsigned num_k,unsigned num_v,float *out,unsigned count) {
    unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;
    if(row>=num_v*head)return;
    unsigned vh=row/head,kh=vh%num_k;
    float cells[8];
    #pragma unroll
    for(unsigned j=0;j<8;j++)if(lane+j*32<head)cells[j]=state[size_t(row)*head+lane+j*32];
    // One warp retains its value row across a causal block. Only the final
    // state is written to device memory; each intermediate output is retained.
    for(unsigned token=0;token<count;token++) {
    const float *current=qkv+size_t(token)*(2*num_k+num_v)*head;
    const float *al=alpha+size_t(token)*num_v,*be=beta+size_t(token)*num_v;
    float t=al[vh]+dt[vh];
    float sp=t>35.0f?t:(t<-35.0f?0.0f:log1pf(expf(t)));
    float decay=expf(sp*a[vh]),b=1.0f/(1.0f+expf(-be[vh]));
    float keys[8],queries[8],dot=0;
    // At most 256 keys per row: one warp owns each value row. No state copy.
    #pragma unroll
    for(unsigned j=0;j<8;j++) {
        unsigned col=lane+j*32;
        if(col<head) {
            cells[j]=__fmul_rn(cells[j],decay);
            keys[j]=current[size_t(num_k+kh)*head+col];
            queries[j]=current[size_t(kh)*head+col];
            dot=fmaf(cells[j],keys[j],dot);
        }
    }
    for(int s=16;s;s/=2)dot+=__shfl_down_sync(0xffffffff,dot,s);
    dot=__shfl_sync(0xffffffff,dot,0);
    float delta=(current[size_t(2*num_k)*head+row]-dot)*b,output=0;
    #pragma unroll
    for(unsigned j=0;j<8;j++) {
        unsigned col=lane+j*32;
        if(col<head) {
            float value=__fadd_rn(cells[j],__fmul_rn(keys[j],delta));
            cells[j]=value;
            output=fmaf(value,queries[j],output);
        }
    }
    for(int s=16;s;s/=2)output+=__shfl_down_sync(0xffffffff,output,s);
    if(!lane)out[size_t(token)*num_v*head+row]=output/sqrtf(float(head));
    }
    #pragma unroll
    for(unsigned j=0;j<8;j++)if(lane+j*32<head)state[size_t(row)*head+lane+j*32]=cells[j];
}
__global__ void qwen_norm_gate(const float *x,const float *z,const float *weights,
    unsigned head,float epsilon,float *out,unsigned stride=0) {
    x+=size_t(blockIdx.y)*stride;z+=size_t(blockIdx.y)*stride;out+=size_t(blockIdx.y)*stride;
    unsigned offset=blockIdx.x*head,tid=threadIdx.x,lane=tid&31,warp=tid/32;
    __shared__ float sums[4];float sum=0;
    for(unsigned i=tid;i<head;i+=128)sum+=x[offset+i]*x[offset+i];
    for(int s=16;s;s/=2)sum+=__shfl_down_sync(0xffffffff,sum,s);
    if(!lane)sums[warp]=sum;
    __syncthreads();
    float inv=1.0f/sqrtf((sums[0]+sums[1]+sums[2]+sums[3])/head+epsilon);
    for(unsigned i=tid;i<head;i+=128) {
        float gate=z[offset+i]/(1.0f+expf(-z[offset+i]));
        out[offset+i]=((x[offset+i]*weights[offset+i])*inv)*gate;
    }
}
struct ResidentQwenRecurrent {
    RbitnetQwenRecurrentConfig cfg;
    RbitnetLlamaMatrix matrices[8];
    std::vector<void*> allocations;
    float *x=nullptr,*h=nullptr,*mixed=nullptr,*z=nullptr,*beta=nullptr,*alpha=nullptr;
    float *history=nullptr,*conv=nullptr,*state=nullptr,*activated=nullptr,*attn=nullptr,*normed=nullptr;
    float *dt=nullptr,*a=nullptr,*ssm_norm=nullptr,*attn_norm=nullptr,*ffn_norm=nullptr;
    float *projection=nullptr,*gate=nullptr,*up=nullptr;
    unsigned filled=0;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    ~ResidentQwenRecurrent() {
        if(stream)cudaStreamSynchronize(stream);
        if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        for(auto p:allocations)cudaFree(p);if(stream)cudaStreamDestroy(stream);
    }
    template<typename T> bool alloc(T *&p,size_t n,const T *host=nullptr) {
        if(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))!=cudaSuccess)return false;
        allocations.push_back(p);
        return !host || (cudaMemcpyAsync(p,host,n*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess
            && cudaStreamSynchronize(stream)==cudaSuccess);
    }
    void matrix(unsigned i,const float *input,float *output) {
        const auto &m=matrices[i];QuantKind kind;resident_kind(m.type,kind);
        launch_quant_kernel(kind,m.weights,m.row_bytes,input,m.cols,m.rows,m.rows,output,stream);
    }
    void enqueue() {
        const auto &c=cfg;unsigned inner=(2*c.num_k+c.num_v)*c.head,values=c.num_v*c.head;
        resident_norm<<<1,256,0,stream>>>(x,attn_norm,c.epsilon,c.embd,h);
        matrix(0,h,mixed);matrix(1,h,z);matrix(2,h,beta);matrix(3,h,alpha);
        qwen_conv<<<(inner+255)/256,256,0,stream>>>(mixed,history,conv,c.conv,inner,activated);
        qwen_l2<<<2*c.num_k,128,0,stream>>>(activated,c.head,c.epsilon);
        qwen_delta<<<(values+3)/4,128,0,stream>>>(state,activated,alpha,beta,dt,a,c.head,c.num_k,c.num_v,attn);
        qwen_norm_gate<<<c.num_v,128,0,stream>>>(attn,z,ssm_norm,c.head,c.epsilon,normed);
        matrix(4,normed,projection);
        resident_norm<<<1,256,0,stream>>>(x,ffn_norm,c.epsilon,c.embd,h,projection);
        matrix(5,h,gate);matrix(6,h,up);
        resident_silu<<<(c.ffn+255)/256,256,0,stream>>>(gate,up,c.ffn);
        matrix(7,gate,projection);
        resident_add<<<(c.embd+255)/256,256,0,stream>>>(x,projection,c.embd);
    }
};
bool qwen_matrix_valid(const RbitnetLlamaMatrix &m,unsigned cols,unsigned rows) {
    if(!m.weights || m.cols!=cols || m.rows!=rows)return false;
    QuantKind kind;if(!resident_kind(m.type,kind))return false;
    size_t block=32,bytes=0;
    switch(kind) {
        case QuantKind::F32:block=1;bytes=4;break;
        case QuantKind::Q4_0:bytes=18;break;case QuantKind::Q5_0:bytes=22;break;
        case QuantKind::Q8_0:bytes=34;break;case QuantKind::MXFP4:bytes=17;break;
        case QuantKind::Q4_K:block=256;bytes=144;break;
        case QuantKind::Q5_K:block=256;bytes=176;break;
        case QuantKind::Q6_K:block=256;bytes=210;break;
    }
    return cols%block==0 && m.row_bytes==size_t(cols)/block*bytes;
}
struct QwenRecurrentSnapshot {
    unsigned head,num_k,num_v,conv,length;
    float *state=nullptr,*history=nullptr;
    ~QwenRecurrentSnapshot() { if(state)cudaFree(state);if(history)cudaFree(history); }
};
}
extern "C" {
void *rbitnet_cuda_qwen_recurrent_create(const RbitnetQwenRecurrentConfig *c,
    const RbitnetLlamaMatrix *m,const float *an,const float *fn,const float *conv,
    const float *dt,const float *a,const float *sn) {
    if(!c || !m || !an || !fn || !conv || !dt || !a || !sn || !c->embd || c->embd>32768
        || !c->ffn || c->ffn>65536 || !c->head || c->head>256 || !c->num_k || c->num_k>128
        || !c->num_v || c->num_v>128 || c->num_v%c->num_k || !c->conv || c->conv>16
        || !isfinite(c->epsilon) || c->epsilon<=0)return nullptr;
    unsigned inner=(2*c->num_k+c->num_v)*c->head,values=c->num_v*c->head;
    unsigned cols[]={c->embd,c->embd,c->embd,c->embd,values,c->embd,c->embd,c->ffn};
    unsigned rows[]={inner,values,c->num_v,c->num_v,c->embd,c->ffn,c->ffn,c->embd};
    for(unsigned i=0;i<8;i++)if(!qwen_matrix_valid(m[i],cols[i],rows[i]))return nullptr;
    auto *r=new(std::nothrow) ResidentQwenRecurrent;if(!r)return nullptr;
    r->cfg=*c;for(unsigned i=0;i<8;i++)r->matrices[i]=m[i];
    if(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking)!=cudaSuccess
        || !r->alloc(r->x,c->embd) || !r->alloc(r->h,c->embd) || !r->alloc(r->mixed,inner)
        || !r->alloc(r->z,values) || !r->alloc(r->beta,c->num_v) || !r->alloc(r->alpha,c->num_v)
        || !r->alloc(r->activated,inner) || !r->alloc(r->history,size_t(inner)*c->conv)
        || !r->alloc(r->state,size_t(values)*c->head) || !r->alloc(r->attn,values)
        || !r->alloc(r->normed,values) || !r->alloc(r->projection,c->embd)
        || !r->alloc(r->gate,c->ffn) || !r->alloc(r->up,c->ffn)
        || !r->alloc(r->conv,size_t(inner)*c->conv,conv) || !r->alloc(r->dt,c->num_v,dt)
        || !r->alloc(r->a,c->num_v,a) || !r->alloc(r->ssm_norm,values,sn)
        || !r->alloc(r->attn_norm,c->embd,an) || !r->alloc(r->ffn_norm,c->embd,fn)) {delete r;return nullptr;}
    return r;
}
void rbitnet_cuda_qwen_recurrent_destroy(void *context) {delete static_cast<ResidentQwenRecurrent*>(context);}
int rbitnet_cuda_qwen_recurrent_step(void *context,const float *input,unsigned pos,float *output) {
    auto *r=static_cast<ResidentQwenRecurrent*>(context);
    if(!r || !input || !output || pos>=1048576)return 1;
    if(pos!=0 && pos!=r->filled)return 2;
    if(pos==0) {
        unsigned inner=(2*r->cfg.num_k+r->cfg.num_v)*r->cfg.head;
        if(cudaMemsetAsync(r->history,0,size_t(inner)*r->cfg.conv*sizeof(float),r->stream)!=cudaSuccess
            || cudaMemsetAsync(r->state,0,size_t(r->cfg.num_v)*r->cfg.head*r->cfg.head*sizeof(float),r->stream)!=cudaSuccess)return 3;
        r->filled=0;
    }
    if(cudaMemcpyAsync(r->x,input,r->cfg.embd*sizeof(float),cudaMemcpyHostToDevice,r->stream)!=cudaSuccess)return 4;
    if(r->cfg.graphs) {
        if(!r->executable) {
            if(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 5;
            r->enqueue();
            if(cudaStreamEndCapture(r->stream,&r->graph)!=cudaSuccess || cudaGraphInstantiate(&r->executable,r->graph,0)!=cudaSuccess)return 6;
        }
        if(cudaGraphLaunch(r->executable,r->stream)!=cudaSuccess)return 7;
    } else r->enqueue();
    if(cudaGetLastError()!=cudaSuccess || cudaMemcpyAsync(output,r->x,r->cfg.embd*sizeof(float),cudaMemcpyDeviceToHost,r->stream)!=cudaSuccess
        || cudaStreamSynchronize(r->stream)!=cudaSuccess)return 8;
    r->filled=pos+1;return 0;
}
void *rbitnet_cuda_qwen_recurrent_snapshot(void *context) {
    auto *r=static_cast<ResidentQwenRecurrent*>(context);
    if(!r || !r->filled)return nullptr;
    auto *s=new(std::nothrow) QwenRecurrentSnapshot;if(!s)return nullptr;
    s->head=r->cfg.head;s->num_k=r->cfg.num_k;s->num_v=r->cfg.num_v;s->conv=r->cfg.conv;s->length=r->filled;
    size_t state=size_t(s->num_v)*s->head*s->head*sizeof(float);
    size_t history=size_t(2*s->num_k+s->num_v)*s->head*s->conv*sizeof(float);
    if(cudaMalloc(reinterpret_cast<void**>(&s->state),state)!=cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&s->history),history)!=cudaSuccess
        || cudaMemcpyAsync(s->state,r->state,state,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(s->history,r->history,history,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess
        || cudaStreamSynchronize(r->stream)!=cudaSuccess) {delete s;return nullptr;}
    return s;
}
void rbitnet_cuda_qwen_recurrent_snapshot_destroy(void *snapshot) {delete static_cast<QwenRecurrentSnapshot*>(snapshot);}
int rbitnet_cuda_qwen_recurrent_restore(void *context,const void *snapshot,unsigned length) {
    auto *r=static_cast<ResidentQwenRecurrent*>(context);auto *s=static_cast<const QwenRecurrentSnapshot*>(snapshot);
    // Recurrent state is a checkpoint, never a truncatable KV prefix.
    if(!r || !s || !length || length!=s->length || s->head!=r->cfg.head
        || s->num_k!=r->cfg.num_k || s->num_v!=r->cfg.num_v || s->conv!=r->cfg.conv)return 1;
    size_t state=size_t(s->num_v)*s->head*s->head*sizeof(float);
    size_t history=size_t(2*s->num_k+s->num_v)*s->head*s->conv*sizeof(float);
    if(cudaMemcpyAsync(r->state,s->state,state,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess
        || cudaMemcpyAsync(r->history,s->history,history,cudaMemcpyDeviceToDevice,r->stream)!=cudaSuccess
        || cudaStreamSynchronize(r->stream)!=cudaSuccess)return 2;
    r->filled=length;return 0;
}
}
