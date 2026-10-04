// SPDX-License-Identifier: MIT
// Keep the scalar RN fold; parallelise its independent loads/multiplies only.
__global__ void gpt_norm_staged(float *x,const float *weights,float epsilon,unsigned n,
    float *y,const float *residual=nullptr) {
    const size_t base=size_t(blockIdx.x)*n;x+=base;y+=base;if(residual)residual+=base;
    extern __shared__ float squared[];
    __shared__ float inv;
    for(unsigned i=threadIdx.x;i<n;i+=blockDim.x) {
        const float value=residual?__fadd_rn(x[i],residual[i]):x[i];
        if(residual)x[i]=value;
        squared[i]=__fmul_rn(value,value);
    }
    __syncthreads();
    if(!threadIdx.x) {
        float total=0;
        for(unsigned i=0;i<n;i++)total=__fadd_rn(total,squared[i]);
        inv=1.0f/sqrtf(total/float(n)+epsilon);
    }
    __syncthreads();
    for(unsigned i=threadIdx.x;i<n;i+=blockDim.x)y[i]=__fmul_rn(__fmul_rn(x[i],inv),weights[i]);
}
void launch_gpt_ordered_norm(float *x,const float *weights,float epsilon,unsigned n,
    float *y,const float *residual,unsigned count,cudaStream_t stream) {
    // Stay below 32 KiB on every supported architecture. Larger model widths
    // retain the original scalar path, without additional state or allocations.
    if(n<=8192) {gpt_norm_staged<<<count,256,size_t(n)*sizeof(float),stream>>>(x,weights,epsilon,n,y,residual);return;}
    gpt_norm_ordered<<<count,256,0,stream>>>(x,weights,epsilon,n,y,residual);
}

