"""Private ablation of a single RoPE/encoded-store launch per attention layer."""
from pathlib import Path
import hashlib,json,shutil
root=Path.cwd();base=root/'target/performance-cache';source=base/'quantized-kv-canonical-native';out=base/'encoded-store-fusion-native';out.mkdir(exist_ok=True)
manifest=json.loads((base/'quantized-kv-canonical-proof/manifest.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in source.iterdir():
    if p.is_file():
        assert sha(p)==manifest['native_source_sha256'][p.name],p.name
        shutil.copy2(p,out/p.name)
p=out/'kv_storage.cuh';s=p.read_text(encoding='utf-8')
start=s.index('template<unsigned Format,bool Paged>\n__global__ void encoded_kv_store(')
end=s.index('// Copy valid prefix payload',start)
store=s[start:end]
header='''template<unsigned Format,bool Paged>
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
'''
body=store[store.index('    for(unsigned value=0;value<2;value++)'):]
fused=header+body
s=s[:end]+fused+'\n'+s[end:]
a='''    encoded_kv_rope<<<dim3(((heads+cache.heads)*(cache.dim/2)+255)/256,count),256,0,stream>>>(q,k,frequency,position,heads,cache.heads,cache.dim,rotary);
    encoded_kv_store<<<dim3(cache.heads,count),128,0,stream>>>(k,v,cache,position,invalid);'''
assert s.count(a)==1
s=s.replace(a,'    encoded_kv_fused_rope_store<<<dim3(cache.heads,count),128,0,stream>>>(q,k,v,cache,frequency,position,heads,rotary,invalid);')
p.write_text(s,encoding='utf-8',newline='\n')
# Exercise the fused quantizer through the independent F64 decoder/attention
# oracle as well. A zero-RoPE scratch query has the required sequence shape.
p=out/'kv_attention_oracle.cuh';s=p.read_text(encoding='utf-8')
a='    encoded_kv_store<<<dim3(view.heads,view.capacity),128,0,stream>>>(k,v,view,position,invalid);'
assert s.count(a)==1
s=s.replace(a,'    cudaMemsetAsync(decoded_k,0,size_t(view.capacity)*view.heads*view.dim*sizeof(float),stream);\n    encoded_kv_fused_rope_store<<<dim3(view.heads,view.capacity),128,0,stream>>>(decoded_k,const_cast<float*>(k),v,view,nullptr,position,view.heads,0,invalid);')
p.write_text(s,encoding='utf-8',newline='\n')
# Close the mutable ABI configuration loophole as well as the constructor.
p=out/'llama_resident.cuh';s=p.read_text(encoding='utf-8')
a='if(!r || enabled>1 || r->block || r->filled)return 1;';assert s.count(a)==1
s=s.replace(a,'if(!r || enabled>1 || (r->kv_format && enabled) || r->block || r->filled)return 1;')
p.write_text(s,encoding='utf-8',newline='\n')
print('Private fused RoPE/F16-Q8 store and encoded TF32 ABI guard prepared; no build, numerical or performance result yet.')
