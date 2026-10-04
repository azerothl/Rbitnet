// SPDX-License-Identifier: MIT
// Private decode experiment: parallel selected experts, original slot reduction.
namespace {
template<QuantKind kind>
__global__ void moe_parallel_down_combine(const uint8_t *down,size_t row_bytes,unsigned cols,
    unsigned rows,const unsigned *ids,const float *input,const float *bias,const float *probabilities,
    const uint8_t *const *selected,unsigned used,float *output) {
    const unsigned row=blockIdx.x,slot=threadIdx.x/32,lane=threadIdx.x&31;
    __shared__ float values[16];
    // Inactive warps still participate in the block barrier. Each selected
    // expert retains its original warp dot/FMA and shuffle order.
    if(slot<used) {
        const unsigned expert=ids[slot];
        const uint8_t *weights=selected?selected[slot]+size_t(row)*row_bytes:down+(size_t(expert)*rows+row)*row_bytes;
        const float dot=quant_row_dot<kind>(weights,input+size_t(slot)*cols,cols);
        if(!lane)values[slot]=dot+(bias?bias[size_t(expert)*rows+row]:0.0f);
    }
    __syncthreads();
    if(!threadIdx.x) {
        float sum=0;
        for(unsigned i=0;i<used;i++)sum=__fadd_rn(sum,__fmul_rn(probabilities[i],values[i]));
        output[row]=sum;
    }
}
void launch_moe_parallel_down(QuantKind kind,const RbitnetLlamaMatrix &down,const unsigned *ids,
    const float *input,const float *bias,const float *probabilities,unsigned used,unsigned rows,
    const uint8_t *const *selected,float *output,cudaStream_t stream) {
    unsigned warps=1;while(warps<used)warps*=2;
#define PARALLEL_DOWN(K) case QuantKind::K: moe_parallel_down_combine<QuantKind::K><<<rows,32*warps,0,stream>>>(static_cast<const uint8_t*>(down.weights),down.row_bytes,down.cols,rows,ids,input,bias,probabilities,selected,used,output);break
    switch(kind){PARALLEL_DOWN(F32);PARALLEL_DOWN(Q4_0);PARALLEL_DOWN(Q5_0);PARALLEL_DOWN(Q8_0);PARALLEL_DOWN(Q4_K);PARALLEL_DOWN(Q5_K);PARALLEL_DOWN(Q6_K);PARALLEL_DOWN(MXFP4);}
#undef PARALLEL_DOWN
}
}
