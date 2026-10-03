// Draft for #95: share original MXFP4 bytes across four token warps while
// retaining quant_row_dot's eight-block/four-lane FMA and reduction order.
// Not compiled, integrated or benchmarked. No activation quantization.
template<unsigned Rows=4,unsigned Tokens=4>
__global__ void mxfp4_ordered_gemm(const uint8_t *weights,size_t row_bytes,
    const float *input,unsigned cols,unsigned rows,unsigned tokens,float *output) {
    // Pad each 32-column input block by 16 floats. A half-warp's two blocks
    // occupy disjoint shared banks for float4 loads; verify with Nsight.
    __shared__ __align__(16) uint8_t ws[Rows][8*17];
    __shared__ __align__(16) float xs[Tokens][8*48];
    const unsigned tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const unsigned local_row=warp/Tokens,local_token=warp%Tokens;
    const unsigned row=blockIdx.x*Rows+local_row,token=blockIdx.y*Tokens+local_token;
    const unsigned group=lane/4,j=(lane%4)*4,blocks=cols/32;
    float sum=0;
    for(unsigned base=0;base<blocks;base+=8) {
        for(unsigned i=tid;i<Rows*8*17;i+=Rows*Tokens*32) {
            unsigned r=i/(8*17),byte=i%(8*17),qb=byte/17;
            unsigned source_row=blockIdx.x*Rows+r;
            ws[r][byte]=(source_row<rows && base+qb<blocks)
                ?weights[size_t(source_row)*row_bytes+size_t(base)*17+byte]:0;
        }
        for(unsigned i=tid;i<Tokens*8*32;i+=Rows*Tokens*32) {
            unsigned t=i/(8*32),column=i%(8*32),qb=column/32,k=column%32;
            unsigned source_token=blockIdx.y*Tokens+t;
            xs[t][qb*48+k]=(source_token<tokens && base+qb<blocks)
                ?input[size_t(source_token)*cols+size_t(base+qb)*32+k]:0;
        }
        __syncthreads();
        if(row<rows && token<tokens && base+group<blocks) {
            const uint8_t *b=ws[local_row]+group*17;
            float d=__uint_as_float(b[0]<2?(0x00200000u<<b[0]):((uint32_t(b[0])-1)<<23));
            uint32_t packed=uint32_t(b[1+j])|(uint32_t(b[2+j])<<8)
                |(uint32_t(b[3+j])<<16)|(uint32_t(b[4+j])<<24);
            #pragma unroll
            for(unsigned half=0;half<2;half++) {
                const float4 x=*reinterpret_cast<const float4*>(xs[local_token]+group*48+half*16+j);
                #pragma unroll
                for(unsigned k=0;k<4;k++) {
                    unsigned nibble=(packed>>(k*8+half*4))&15;
                    int magnitude=(0xC8643210u>>((nibble&7)*4))&15;
                    int q=(nibble&8)?-magnitude:magnitude;
                    float value=k==0?x.x:k==1?x.y:k==2?x.z:x.w;
                    sum=fmaf(d*float(q),value,sum);
                }
            }
        }
        __syncthreads();
    }
    // Every lane participates, including partial row/token tiles; output is
    // guarded only after the warp reduction. Invalid blocks perform no FMAs.
    for(int shift=16;shift>0;shift/=2)sum+=__shfl_down_sync(0xffffffff,sum,shift);
    if(!lane && row<rows && token<tokens)output[size_t(token)*rows+row]=sum;
}
