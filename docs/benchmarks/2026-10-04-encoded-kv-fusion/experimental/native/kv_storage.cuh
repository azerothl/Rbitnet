// SPDX-License-Identifier: MIT
// Private integration draft: F16/Q8 K/V only; no GDN or convolution state.
#include <cuda_fp16.h>
#include <limits>
namespace {
size_t encoded_kv_plane_bytes(unsigned format,size_t elements,unsigned dim) {
    if(!dim || elements%dim || elements>std::numeric_limits<size_t>::max()/4)return 0;
    if(format==0)return elements*4;
    if(format==1)return elements*2;
    if(format==2)return elements+(elements/dim)*4;
    return 0;
}
template<unsigned Format,bool Paged> struct EncodedKvView {
    float *k,*v;float *const *pages_k,*const *pages_v;
    unsigned layer,layers,capacity,heads,dim;
    __device__ void *plane(bool value,unsigned position)const {
        if constexpr(Paged)return value?pages_v[position/32]:pages_k[position/32];
        else return value?v:k;
    }
    __device__ size_t elements()const {
        return size_t(Paged?layers:1)*(Paged?32:capacity)*heads*dim;
    }
    __device__ size_t index(unsigned position,unsigned head,unsigned i)const {
        size_t token=Paged?size_t(layer)*32+position%32:position;
        return (token*heads+head)*dim+i;
    }
    __device__ float read(bool value,unsigned position,unsigned head,unsigned i)const {
        const void *p=plane(value,position);size_t at=index(position,head,i);
        if constexpr(Format==1)return __half2float(static_cast<const __half*>(p)[at]);
        else {
            float scale=reinterpret_cast<const float*>(static_cast<const char*>(p)+elements())[at/dim];
            return float(static_cast<const signed char*>(p)[at])*scale;
        }
    }
};
__global__ void encoded_kv_rope(float *q,float *k,const float *frequency,const unsigned *position,
    unsigned heads,unsigned kv_heads,unsigned dim,unsigned rotary) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,pairs=dim/2,pos=*position+blockIdx.y;
    q+=size_t(blockIdx.y)*heads*dim;k+=size_t(blockIdx.y)*kv_heads*dim;
    if(i<(heads+kv_heads)*pairs) {
        unsigned head=i/pairs,j=i%pairs;float *src=head<heads?q+head*dim:k+(head-heads)*dim;
        float a=src[2*j],b=src[2*j+1];
        if(2*j<rotary) {float angle=pos*frequency[j],s=sinf(angle),c=cosf(angle);src[2*j]=a*c-b*s;src[2*j+1]=a*s+b*c;}
    }
}
template<unsigned Format,bool Paged>
__global__ void encoded_kv_store(const float *k,const float *v,EncodedKvView<Format,Paged> cache,
    const unsigned *position,unsigned *invalid) {
    unsigned head=blockIdx.x,token=blockIdx.y,tid=threadIdx.x,pos=*position+token;
    unsigned lane=tid&31,warp=tid/32;__shared__ float maxima[4];
    size_t input=(size_t(token)*cache.heads+head)*cache.dim,at=cache.index(pos,head,0);
    for(unsigned value=0;value<2;value++) {
        const float *source=(value?v:k)+input;void *plane=cache.plane(value,pos);
        if constexpr(Format==1) {
            for(unsigned i=tid;i<cache.dim;i+=128) {
                float x=source[i];if(!isfinite(x)||fabsf(x)>65504.0f) {atomicExch(invalid,1);x=0;}
                static_cast<__half*>(plane)[at+i]=__float2half_rn(x);
            }
        } else {
            float maximum=0;
            for(unsigned i=tid;i<cache.dim;i+=128) {
                float x=source[i];if(!isfinite(x))atomicExch(invalid,1);else maximum=fmaxf(maximum,fabsf(x));
            }
            for(unsigned shift=16;shift;shift/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,shift));
            if(!lane)maxima[warp]=maximum;
            __syncthreads();maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
            float scale=maximum/127.0f;if(scale==0)scale=1;
            if(!isfinite(scale*127.0f))scale=nextafterf(scale,0.0f);
            if(!tid)reinterpret_cast<float*>(static_cast<char*>(plane)+cache.elements())[at/cache.dim]=scale;
            for(unsigned i=tid;i<cache.dim;i+=128) {
                float x=source[i];int quant=isfinite(x)?__float2int_rn(x/scale):0;
                static_cast<signed char*>(plane)[at+i]=static_cast<signed char>(max(-127,min(127,quant)));
            }
            // Every warp must finish reading shared maxima before the next plane.
            __syncthreads();
        }
    }
}
template<unsigned Format,bool Paged>
__global__ void encoded_kv_fused_rope_store(float *q,float *k,const float *v,EncodedKvView<Format,Paged> cache,
    const float *frequency,const unsigned *position,unsigned query_heads,unsigned rotary,unsigned *invalid) {
    unsigned head=blockIdx.x,token=blockIdx.y,tid=threadIdx.x,pos=*position+token;
    unsigned lane=tid&31,warp=tid/32;__shared__ float maxima[4];
    const unsigned pairs=cache.dim/2,group=query_heads/cache.heads;
    float *query=q+size_t(token)*query_heads*cache.dim;
    float *key=k+(size_t(token)*cache.heads+head)*cache.dim;
    // Each block owns one KV head and exactly its associated query heads.
    // Pair writes do not overlap. The barrier publishes key rotation to every
    // participating warp before the original quantizer reads it.
    for(unsigned i=tid;i<(group+1)*pairs;i+=128) {
        unsigned local_head=i/pairs,j=i%pairs;
        float *src=local_head<group?query+(head*group+local_head)*cache.dim:key;
        if(2*j<rotary) {
            float a=src[2*j],b=src[2*j+1];
            float angle=pos*frequency[j],sn=sinf(angle),cs=cosf(angle);
            src[2*j]=a*cs-b*sn;src[2*j+1]=a*sn+b*cs;
        }
    }
    __syncthreads();
    size_t input=(size_t(token)*cache.heads+head)*cache.dim,at=cache.index(pos,head,0);
    for(unsigned value=0;value<2;value++) {
        const float *source=(value?v:k)+input;void *plane=cache.plane(value,pos);
        if constexpr(Format==1) {
            for(unsigned i=tid;i<cache.dim;i+=128) {
                float x=source[i];if(!isfinite(x)||fabsf(x)>65504.0f) {atomicExch(invalid,1);x=0;}
                static_cast<__half*>(plane)[at+i]=__float2half_rn(x);
            }
        } else {
            float maximum=0;
            for(unsigned i=tid;i<cache.dim;i+=128) {
                float x=source[i];if(!isfinite(x))atomicExch(invalid,1);else maximum=fmaxf(maximum,fabsf(x));
            }
            for(unsigned shift=16;shift;shift/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,shift));
            if(!lane)maxima[warp]=maximum;
            __syncthreads();maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
            float scale=maximum/127.0f;if(scale==0)scale=1;
            if(!isfinite(scale*127.0f))scale=nextafterf(scale,0.0f);
            if(!tid)reinterpret_cast<float*>(static_cast<char*>(plane)+cache.elements())[at/cache.dim]=scale;
            for(unsigned i=tid;i<cache.dim;i+=128) {
                float x=source[i];int quant=isfinite(x)?__float2int_rn(x/scale):0;
                static_cast<signed char*>(plane)[at+i]=static_cast<signed char>(max(-127,min(127,quant)));
            }
            // Every warp must finish reading shared maxima before the next plane.
            __syncthreads();
        }
    }
}

// Copy valid prefix payload and Q8 scales separately: capacity changes their
// offsets, even when the first payload bytes have an identical layout.
bool encoded_kv_copy_prefix(void *dst,size_t dst_elements,const void *src,size_t src_elements,
    size_t elements,unsigned dim,unsigned format,cudaStream_t stream) {
    size_t width=format==0?4:format==1?2:1;
    if(elements>dst_elements || elements>src_elements || !dim || elements%dim)return false;
    if(cudaMemcpyAsync(dst,src,elements*width,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return false;
    return format!=2 || cudaMemcpyAsync(static_cast<char*>(dst)+dst_elements,static_cast<const char*>(src)+src_elements,
        elements/dim*4,cudaMemcpyDeviceToDevice,stream)==cudaSuccess;
}
}

namespace {
template<unsigned Format,bool Paged>
__global__ void encoded_resident_attention(EncodedKvView<Format,Paged> cache,const float *q,const unsigned *position,
    unsigned kv_heads,unsigned heads,unsigned dim,unsigned window,float scale,float *y,const float *sinks=nullptr) {
    extern __shared__ float scores[];
    __shared__ float reductions[4];
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid/32;
    const unsigned head=blockIdx.x,kh=head/(heads/kv_heads),seq=*position+blockIdx.y+1;
    q+=size_t(blockIdx.y)*heads*dim;y+=size_t(blockIdx.y)*heads*dim;
    const unsigned first=window && seq>window?seq-window:0;
    for(unsigned p=first+warp;p<seq;p+=4) {
        float s=0;
        for(unsigned i=lane;i<dim;i+=32)s=fmaf(q[head*dim+i],cache.read(false,p,kh,i),s);
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
        for(unsigned p=first;p<seq;p++)output=fmaf(scores[p-first]/sum,cache.read(true,p,kh,i),output);
        y[head*dim+i]=output;
    }
}
}

namespace {
template<unsigned Format,bool Paged>
__global__ void encoded_attention_partials(EncodedKvView<Format,Paged> cache,const float *q,const unsigned *position,
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
        for(unsigned i=lane;i<dim;i+=32)dot=fmaf(q[i],cache.read(false,p,kh,i),dot);
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
        for(unsigned p=begin;p<end;p++)value=fmaf(scores[p-begin],cache.read(true,p,kh,i),value);
        out[i+2]=value;
    }
}
}

namespace {
template<unsigned Format,bool Paged>
void launch_encoded_kv(EncodedKvView<Format,Paged> cache,float *q,float *k,const float *v,
    const float *frequency,const unsigned *position,unsigned heads,unsigned rotary,unsigned window,
    unsigned count,bool split,float *scratch,float *output,unsigned *invalid,cudaStream_t stream) {
    encoded_kv_fused_rope_store<<<dim3(cache.heads,count),128,0,stream>>>(q,k,v,cache,frequency,position,heads,rotary,invalid);
    float scale=1.0f/sqrtf(float(cache.dim));
    if(split) {
        unsigned parts=(cache.capacity+attention_tile-1)/attention_tile;
        encoded_attention_partials<<<dim3(heads,parts,count),128,0,stream>>>(cache,q,position,cache.heads,heads,cache.dim,window,scale,parts,scratch);
        attention_merge<<<dim3(heads,count),128,parts*sizeof(float),stream>>>(scratch,heads,cache.dim,parts,output);
    } else encoded_resident_attention<<<dim3(heads,count),128,cache.capacity*sizeof(float),stream>>>(cache,q,position,cache.heads,heads,cache.dim,window,scale,output);
}
}
