// SPDX-License-Identifier: MIT
// Draft: F32 page indirection; original arithmetic/reduction order.
namespace {
__global__ void paged_resident_rope_kv(float *q,float *k,const float *v,float *const *pages_k,float *const *pages_v,unsigned layer,
    const float *frequency,const unsigned *position,unsigned heads,unsigned kv_heads,unsigned dim,unsigned rotary) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned pos=*position+blockIdx.y, pairs=dim/2;
    q+=size_t(blockIdx.y)*heads*dim;k+=size_t(blockIdx.y)*kv_heads*dim;v+=size_t(blockIdx.y)*kv_heads*dim;
    if(i<(heads+kv_heads)*pairs) {
        unsigned head=i/pairs,j=i%pairs;
        float *src=head<heads?q+head*dim:k+(head-heads)*dim;
        float a=src[2*j],b=src[2*j+1];
        if(2*j<rotary) {float angle=pos*frequency[j],s=sinf(angle),c=cosf(angle);src[2*j]=a*c-b*s;src[2*j+1]=a*s+b*c;}
        if(head>=heads) {
            float *cache_k=pages_k[pos/llama_page_tokens],*cache_v=pages_v[pos/llama_page_tokens];
            size_t offset=(size_t(layer)*llama_page_tokens+pos%llama_page_tokens)*kv_heads*dim+(head-heads)*dim+2*j;
            cache_k[offset]=src[2*j];cache_k[offset+1]=src[2*j+1];
            cache_v[offset]=v[(head-heads)*dim+2*j];cache_v[offset+1]=v[(head-heads)*dim+2*j+1];
        }
    }
}
__global__ void paged_resident_attention(float *const *k,float *const *v,unsigned layer,const float *q,const unsigned *position,
    unsigned kv_heads,unsigned heads,unsigned dim,unsigned window,float scale,float *y,const float *sinks=nullptr) {
    extern __shared__ float scores[];
    __shared__ float reductions[4];
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid/32;
    const unsigned head=blockIdx.x,kh=head/(heads/kv_heads),seq=*position+blockIdx.y+1;
    q+=size_t(blockIdx.y)*heads*dim;y+=size_t(blockIdx.y)*heads*dim;
    const unsigned first=window && seq>window?seq-window:0;
    for(unsigned p=first+warp;p<seq;p+=4) {
        float s=0;
        for(unsigned i=lane;i<dim;i+=32)s=fmaf(q[head*dim+i],k[p/llama_page_tokens][(size_t(layer)*llama_page_tokens+p%llama_page_tokens)*kv_heads*dim+kh*dim+i],s);
        for(int shift=16;shift;shift/=2)s+=__shfl_down_sync(0xffffffff,s,shift);
        if(lane==0)scores[p-first]=s*scale;
    }
    __syncthreads();
    float maximum=-CUDART_INF_F;
    for(unsigned i=tid;i<seq-first;i+=128)maximum=fmaxf(maximum,scores[i]);
    for(int shift=16;shift;shift/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,shift));
    if(lane==0)reductions[warp]=maximum;
    __syncthreads();maximum=fmaxf(fmaxf(reductions[0],reductions[1]),fmaxf(reductions[2],reductions[3]));
    if(sinks)maximum=fmaxf(maximum,sinks[head]);
    __syncthreads();
    float sum=0;
    for(unsigned i=tid;i<seq-first;i+=128) {scores[i]=expf(scores[i]-maximum);sum+=scores[i];}
    for(int shift=16;shift;shift/=2)sum+=__shfl_down_sync(0xffffffff,sum,shift);
    if(lane==0)reductions[warp]=sum;
    __syncthreads();sum=reductions[0]+reductions[1]+reductions[2]+reductions[3];
    if(sinks)sum+=expf(sinks[head]-maximum);
    for(unsigned i=tid;i<dim;i+=128) {
        float output=0;
        for(unsigned p=first;p<seq;p++)output=fmaf(scores[p-first]/sum,v[p/llama_page_tokens][(size_t(layer)*llama_page_tokens+p%llama_page_tokens)*kv_heads*dim+kh*dim+i],output);
        y[head*dim+i]=output;
    }
}
__global__ void paged_attention_partials(float *const *k,float *const *v,unsigned layer,const float *q,const unsigned *position,
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
        for(unsigned i=lane;i<dim;i+=32)dot=fmaf(q[i],k[p/llama_page_tokens][(size_t(layer)*llama_page_tokens+p%llama_page_tokens)*kv_heads*dim+kh*dim+i],dot);
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
        for(unsigned p=begin;p<end;p++)value=fmaf(scores[p-begin],v[p/llama_page_tokens][(size_t(layer)*llama_page_tokens+p%llama_page_tokens)*kv_heads*dim+kh*dim+i],value);
        out[i+2]=value;
    }
}
void launch_paged_split_attention(float *const *k,float *const *v,unsigned layer,const float *q,const unsigned *position,
    unsigned kv_heads,unsigned heads,unsigned dim,unsigned window,float scale,unsigned capacity,
    unsigned count,float *scratch,float *out,cudaStream_t stream,const float *sinks=nullptr) {
    unsigned parts=(capacity+attention_tile-1)/attention_tile;
    paged_attention_partials<<<dim3(heads,parts,count),128,0,stream>>>(k,v,layer,q,position,kv_heads,heads,dim,window,scale,parts,scratch);
    attention_merge<<<dim3(heads,count),128,parts*sizeof(float),stream>>>(scratch,heads,dim,parts,out,sinks);
}
}
