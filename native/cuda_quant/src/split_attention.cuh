// SPDX-License-Identifier: MIT
// Exact split-KV softmax. Fixed grids and per-context scratch remain valid in
// CUDA graphs while the device position changes. No token eviction or quantization.
#include <cstdio>
#include <cstdlib>
#include <mutex>
namespace {
constexpr unsigned attention_tile=256;
bool split_attention_enabled() {
    const char *flag=std::getenv("RBITNET_CUDA_SPLIT_KV");
    return flag && flag[0]=='1' && flag[1]=='\0';
}
__global__ void attention_partials(const float *k,const float *v,const float *q,const unsigned *position,
    unsigned kv_heads,unsigned heads,unsigned dim,unsigned window,float scale,unsigned parts,float *scratch) {
    __shared__ float scores[attention_tile],reductions[4];
    unsigned tid=threadIdx.x,lane=tid&31,warp=tid/32,head=blockIdx.x,part=blockIdx.y,token=blockIdx.z;
    unsigned seq=*position+token+1,first=window && seq>window?seq-window:0;
    unsigned begin=max(first,part*attention_tile),end=min(seq,(part+1)*attention_tile);
    float *out=scratch+((size_t(token)*heads+head)*parts+part)*(dim+2);
    if(begin>=end) {if(!tid) {out[0]=-CUDART_INF_F;out[1]=0;}return;}
    unsigned kh=head/(heads/kv_heads);q+=size_t(token)*heads*dim+head*dim;
    const bool dim128=dim==128;
    float q0=0,q1=0,q2=0,q3=0;
    if(dim128) {q0=q[lane];q1=q[lane+32];q2=q[lane+64];q3=q[lane+96];}
    for(unsigned p=begin+warp;p<end;p+=4) {
        float dot=0;
        if(dim128) {
            const float *kp=k+size_t(p)*kv_heads*dim+kh*dim+lane;
            float k0=kp[0],k1=kp[32],k2=kp[64],k3=kp[96];
            dot=fmaf(q0,k0,dot);dot=fmaf(q1,k1,dot);dot=fmaf(q2,k2,dot);dot=fmaf(q3,k3,dot);
        } else {
            for(unsigned i=lane;i<dim;i+=32)dot=fmaf(q[i],k[size_t(p)*kv_heads*dim+kh*dim+i],dot);
        }
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
    const unsigned stride=kv_heads*dim;
    for(unsigned i=tid;i<dim;i+=128) {
        float value=0;
        const float *vp=v+size_t(begin)*stride+kh*dim+i;
        const unsigned n=end-begin;
        unsigned t=0;
        for(;t+7<n;t+=8) {
            float v0=vp[(t+0)*stride],v1=vp[(t+1)*stride],v2=vp[(t+2)*stride],v3=vp[(t+3)*stride];
            float v4=vp[(t+4)*stride],v5=vp[(t+5)*stride],v6=vp[(t+6)*stride],v7=vp[(t+7)*stride];
            value=fmaf(scores[t+0],v0,value);value=fmaf(scores[t+1],v1,value);
            value=fmaf(scores[t+2],v2,value);value=fmaf(scores[t+3],v3,value);
            value=fmaf(scores[t+4],v4,value);value=fmaf(scores[t+5],v5,value);
            value=fmaf(scores[t+6],v6,value);value=fmaf(scores[t+7],v7,value);
        }
        for(;t<n;t++)value=fmaf(scores[t],vp[t*stride],value);
        out[i+2]=value;
    }
}
__global__ void attention_merge(const float *scratch,unsigned heads,unsigned dim,unsigned parts,float *y,const float *sinks=nullptr) {
    extern __shared__ float weights[];
    const float *row=scratch+(size_t(blockIdx.y)*heads+blockIdx.x)*parts*(dim+2);
    if(!threadIdx.x) {
        float maximum=-CUDART_INF_F;
        for(unsigned p=0;p<parts;p++)if(row[size_t(p)*(dim+2)+1]>0)maximum=fmaxf(maximum,row[size_t(p)*(dim+2)]);
        if(sinks)maximum=fmaxf(maximum,sinks[blockIdx.x]);
        float sum=sinks?expf(sinks[blockIdx.x]-maximum):0;
        for(unsigned p=0;p<parts;p++) {
            const float *src=row+size_t(p)*(dim+2);
            weights[p]=src[1]>0?expf(src[0]-maximum):0;
            sum=fmaf(weights[p],src[1],sum);
        }
        for(unsigned p=0;p<parts;p++)weights[p]=sum>0?weights[p]/sum:0;
    }
    __syncthreads();y+=(size_t(blockIdx.y)*heads+blockIdx.x)*dim;
    for(unsigned i=threadIdx.x;i<dim;i+=128) {
        float value=0;
        // Empty tiles have no numerator. Do not read their stale scratch bytes.
        for(unsigned p=0;p<parts;p++)if(weights[p]>0)value=fmaf(weights[p],row[size_t(p)*(dim+2)+i+2],value);
        y[i]=value;
    }
}
void launch_split_attention(const float *k,const float *v,const float *q,const unsigned *position,
    unsigned kv_heads,unsigned heads,unsigned dim,unsigned window,float scale,unsigned capacity,
    unsigned count,float *scratch,float *out,cudaStream_t stream,const float *sinks=nullptr) {
    unsigned parts=(capacity+attention_tile-1)/attention_tile;
    attention_partials<<<dim3(heads,parts,count),128,0,stream>>>(k,v,q,position,kv_heads,heads,dim,window,scale,parts,scratch);
    attention_merge<<<dim3(heads,count),128,parts*sizeof(float),stream>>>(scratch,heads,dim,parts,out,sinks);
}
bool gqa_grow(float *&ptr,size_t &cap,size_t n) {
    if(cap>=n && ptr) return true;
    if(ptr) cudaFree(ptr);
    ptr=nullptr;cap=0;
    if(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(float))!=cudaSuccess) return false;
    cap=n;return true;
}
} // namespace
extern "C" RBITNET_CUDA_API int rbitnet_cuda_gqa_prefill(const float *k,const float *v,const float *q,
    unsigned base_pos,unsigned n_tokens,unsigned n_kv,unsigned n_head,unsigned head_dim,
    float scale,float *y) {
    unsigned end=base_pos+n_tokens;
    unsigned parts=(end+attention_tile-1)/attention_tile;
    size_t kv_n=size_t(end)*n_kv*head_dim,q_n=size_t(n_tokens)*n_head*head_dim;
    size_t scratch_n=size_t(n_tokens)*n_head*parts*(head_dim+2);
    if(!k || !v || !q || !y || !n_tokens || !n_kv || !n_head || n_head%n_kv || !head_dim
        || end<n_tokens || end>8192 || n_tokens>2048 || n_head>128 || head_dim>256
        || scratch_n>size_t(128)<<20) return 1;
    static std::mutex guard;
    static float *dk=nullptr,*dv=nullptr,*dq=nullptr,*dy=nullptr,*scratch=nullptr;
    static unsigned *dpos=nullptr;
    static size_t ck=0,cv=0,cq=0,cy=0,cs=0;
    std::lock_guard<std::mutex> lock(guard);
    if(!dpos && cudaMalloc(reinterpret_cast<void**>(&dpos),sizeof(unsigned))!=cudaSuccess) return 2;
    if(!gqa_grow(dk,ck,kv_n) || !gqa_grow(dv,cv,kv_n) || !gqa_grow(dq,cq,q_n)
        || !gqa_grow(dy,cy,q_n) || !gqa_grow(scratch,cs,scratch_n)) return 2;
    if(cudaMemcpy(dk,k,kv_n*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess
        || cudaMemcpy(dv,v,kv_n*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess
        || cudaMemcpy(dq,q,q_n*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess
        || cudaMemcpy(dpos,&base_pos,sizeof(unsigned),cudaMemcpyHostToDevice)!=cudaSuccess) return 3;
    launch_split_attention(dk,dv,dq,dpos,n_kv,n_head,head_dim,0,scale,end,n_tokens,scratch,dy,nullptr);
    if(cudaGetLastError()!=cudaSuccess || cudaMemcpy(y,dy,q_n*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess
        || cudaDeviceSynchronize()!=cudaSuccess) return 4;
    static int once=0;
    if(!once) {std::fprintf(stderr,"qwen3 gqa prefill tokens=%u end=%u\n",n_tokens,end);once=1;}
    return 0;
}
