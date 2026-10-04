// SPDX-License-Identifier: MIT
// Exact split-KV softmax. Fixed grids and per-context scratch remain valid in
// CUDA graphs while the device position changes. No token eviction or quantization.
#include <cstdlib>
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
    for(unsigned p=begin+warp;p<end;p+=4) {
        float dot=0;
        for(unsigned i=lane;i<dim;i+=32)dot=fmaf(q[i],k[size_t(p)*kv_heads*dim+kh*dim+i],dot);
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
    for(unsigned i=tid;i<dim;i+=128) {
        float value=0;
        for(unsigned p=begin;p<end;p++)value=fmaf(scores[p-begin],v[size_t(p)*kv_heads*dim+kh*dim+i],value);
        out[i+2]=value;
    }
}
__global__ void attention_merge(const float *scratch,unsigned heads,unsigned dim,unsigned parts,float *y) {
    extern __shared__ float weights[];
    const float *row=scratch+(size_t(blockIdx.y)*heads+blockIdx.x)*parts*(dim+2);
    if(!threadIdx.x) {
        float maximum=-CUDART_INF_F;
        for(unsigned p=0;p<parts;p++)if(row[size_t(p)*(dim+2)+1]>0)maximum=fmaxf(maximum,row[size_t(p)*(dim+2)]);
        float sum=0;
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
    unsigned count,float *scratch,float *out,cudaStream_t stream) {
    unsigned parts=(capacity+attention_tile-1)/attention_tile;
    attention_partials<<<dim3(heads,parts,count),128,0,stream>>>(k,v,q,position,kv_heads,heads,dim,window,scale,parts,scratch);
    attention_merge<<<dim3(heads,count),128,parts*sizeof(float),stream>>>(scratch,heads,dim,parts,out);
}
}
