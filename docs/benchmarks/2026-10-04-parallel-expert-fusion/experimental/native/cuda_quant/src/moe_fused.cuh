// SPDX-License-Identifier: MIT
// Optional exact decode FFN fusion. Original quant bytes and dot/reduction order.
// Include before ResidentMoe; configurable only before its first enqueue.
namespace {
template<QuantKind kind>
__global__ void moe_fused_gate_up(const uint8_t *gate,const uint8_t *up,size_t gate_rb,size_t up_rb,
    unsigned cols,unsigned rows,const unsigned *ids,const float *input,const float *gb,const float *ub,
    bool oai,const uint8_t *const *selected,unsigned used,float *hidden) {
    const unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/32,slot=blockIdx.y;
    if(row>=rows)return;
    const unsigned expert=ids[slot];
    const uint8_t *gw=selected?selected[slot]+size_t(row)*gate_rb:gate+(size_t(expert)*rows+row)*gate_rb;
    const uint8_t *uw=selected?selected[used+slot]+size_t(row)*up_rb:up+(size_t(expert)*rows+row)*up_rb;
    const float gd=quant_row_dot<kind>(gw,input,cols),ud=quant_row_dot<kind>(uw,input,cols);
    if(!(threadIdx.x&31)) {
        const size_t bias=size_t(expert)*rows+row;
        float g=gd+(gb?gb[bias]:0.0f),u=ud+(ub?ub[bias]:0.0f);
        if(oai){g=fminf(g,7.0f);u=fminf(fmaxf(u,-7.0f),7.0f)+1.0f;}
        hidden[size_t(slot)*rows+row]=(g/(1.0f+expf(-(oai?1.702f:1.0f)*g)))*u;
    }
}
template<QuantKind kind>
__global__ void moe_fused_down_combine(const uint8_t *down,size_t row_bytes,unsigned cols,
    unsigned rows,const unsigned *ids,const float *input,const float *bias,const float *probabilities,
    const uint8_t *const *selected,unsigned used,float *output) {
    const unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(row>=rows)return;
    float sum=0.0f;
    // Every lane executes every shuffle. Selected-slot order and separate
    // multiply/add rounding match moe_combine, including expert bias lookup.
    for(unsigned slot=0;slot<used;slot++) {
        const unsigned expert=ids[slot];
        const uint8_t *weights=selected?selected[slot]+size_t(row)*row_bytes:down+(size_t(expert)*rows+row)*row_bytes;
        const float dot=quant_row_dot<kind>(weights,input+size_t(slot)*cols,cols);
        if(!(threadIdx.x&31)) {
            const float value=dot+(bias?bias[size_t(expert)*rows+row]:0.0f);
            sum=__fadd_rn(sum,__fmul_rn(probabilities[slot],value));
        }
    }
    if(!(threadIdx.x&31))output[row]=sum;
}
}
#include "moe_parallel_fused.cuh"
namespace {
void launch_moe_fused_gate_up(QuantKind kind,const RbitnetLlamaMatrix &gate,const RbitnetLlamaMatrix &up,
    const unsigned *ids,const float *input,const float *gb,const float *ub,unsigned used,unsigned rows,
    bool oai,const uint8_t *const *selected,float *hidden,cudaStream_t stream) {
    dim3 grid((rows+7)/8,used);
#define MOE_FUSED_GU(K) case QuantKind::K: moe_fused_gate_up<QuantKind::K><<<grid,256,0,stream>>>(static_cast<const uint8_t*>(gate.weights),static_cast<const uint8_t*>(up.weights),gate.row_bytes,up.row_bytes,gate.cols,rows,ids,input,gb,ub,oai,selected,used,hidden);break
    switch(kind){MOE_FUSED_GU(F32);MOE_FUSED_GU(Q4_0);MOE_FUSED_GU(Q5_0);MOE_FUSED_GU(Q8_0);MOE_FUSED_GU(Q4_K);MOE_FUSED_GU(Q5_K);MOE_FUSED_GU(Q6_K);MOE_FUSED_GU(MXFP4);}
#undef MOE_FUSED_GU
}
void launch_moe_fused_down(QuantKind kind,const RbitnetLlamaMatrix &down,const unsigned *ids,
    const float *input,const float *bias,const float *probabilities,unsigned used,unsigned rows,
    const uint8_t *const *selected,float *output,cudaStream_t stream,bool parallel=false) {
    if(parallel) {launch_moe_parallel_down(kind,down,ids,input,bias,probabilities,used,rows,selected,output,stream);return;}
    dim3 grid((rows+7)/8);
#define MOE_FUSED_D(K) case QuantKind::K: moe_fused_down_combine<QuantKind::K><<<grid,256,0,stream>>>(static_cast<const uint8_t*>(down.weights),down.row_bytes,down.cols,rows,ids,input,bias,probabilities,selected,used,output);break
    switch(kind){MOE_FUSED_D(F32);MOE_FUSED_D(Q4_0);MOE_FUSED_D(Q5_0);MOE_FUSED_D(Q8_0);MOE_FUSED_D(Q4_K);MOE_FUSED_D(Q5_K);MOE_FUSED_D(Q6_K);MOE_FUSED_D(MXFP4);}
#undef MOE_FUSED_D
}
}
