"""Independent NumPy F64 oracle for encoded decoded K/V and attention."""
from pathlib import Path
import ctypes as c,hashlib,json,sys
library=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
print('Encoded attention oracle: initializing CUDA before NumPy.',flush=True)
dll=c.CDLL(str(library.resolve()));f=dll.rbitnet_cuda_encoded_attention_oracle
print('Encoded attention oracle: DLL loaded; configuring managed allocation ceiling.',flush=True)
class Memory(c.Structure):
 _fields_=[(k,c.c_uint64)for k in ['version','limit','live','peak','allocations','refusals']]+[('categories',c.c_uint64*7)]
configure=dll.rbitnet_cuda_memory_configure;configure.argtypes=[c.c_uint64,c.c_uint64];configure.restype=c.c_int;assert configure(64*2**20,0)==0
stats=dll.rbitnet_cuda_memory_stats;stats.argtypes=[c.POINTER(Memory)];stats.restype=c.c_int
def state():
 m=Memory();assert stats(c.byref(m))==0;return {'live':int(m.live),'peak':int(m.peak),'categories':list(m.categories)}
initial=state()
# Initialize the actual device ledger before the independent CPU calculation.
# Native launch failures are diagnosed separately from NumPy import behavior.
import numpy as np
print('Encoded attention oracle: CUDA and NumPy initialized.',flush=True)
f.argtypes=[c.c_uint]*7+[c.POINTER(c.c_float)]*8+[c.POINTER(c.c_size_t)];f.restype=c.c_int
ptr=lambda x:x.ctypes.data_as(c.POINTER(c.c_float))
def encode(x,format):
 if format==1:return x.astype(np.float16).astype(np.float32),None
 scale=(np.max(np.abs(x),axis=-1)/np.float32(127)).astype(np.float32)
 scale[scale==0]=1
 with np.errstate(over='ignore'):bad=~np.isfinite(scale*np.float32(127))
 scale[bad]=np.nextafter(scale[bad],np.float32(0))
 values=np.clip(np.rint(x/scale[...,None]),-127,127).astype(np.int8)
 return values.astype(np.float32)*scale[...,None],scale
rng=np.random.default_rng(9384);report={'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'cases':[]}
for seq in [1,31,32,33,255,256,257,1025]:
 for format in [1,2]:
  for split in [0,1]:
   for window in [0,17]:
    heads,kv_heads,dim=4,2,64
    q=(rng.normal(size=(heads,dim))*.4).astype(np.float32)
    k=(rng.normal(size=(seq,kv_heads,dim))*.7).astype(np.float32);v=(rng.normal(size=k.shape)*.8).astype(np.float32)
    # Include a zero vector and constant vectors to exercise scale/half layout.
    k[0,0]=0;v[0,1]=np.float32(-.125)
    expected_k,scale_k=encode(k,format);expected_v,scale_v=encode(v,format)
    decoded_k=np.empty_like(k);decoded_v=np.empty_like(v);result=np.empty_like(q)
    sk=np.empty((seq,kv_heads),dtype=np.float32);sv=np.empty_like(sk);bytes=c.c_size_t()
    status=f(format,seq,heads,kv_heads,dim,window,split,ptr(q),ptr(k),ptr(v),ptr(result),ptr(decoded_k),ptr(decoded_v),ptr(sk),ptr(sv),c.byref(bytes));assert status==0,status
    assert np.array_equal(decoded_k.view(np.uint32),expected_k.view(np.uint32))and np.array_equal(decoded_v.view(np.uint32),expected_v.view(np.uint32))
    if format==2:assert np.array_equal(sk.view(np.uint32),scale_k.view(np.uint32))and np.array_equal(sv.view(np.uint32),scale_v.view(np.uint32))
    physical=2*(k.size*(2 if format==1 else 1)+(seq*kv_heads*4 if format==2 else 0));assert bytes.value==physical
    first=max(0,seq-window)if window else 0;oracle=np.empty_like(q,dtype=np.float64)
    for head in range(heads):
     kv=head//(heads//kv_heads);scores=expected_k[first:,kv].astype(np.float64)@q[head].astype(np.float64)/np.sqrt(dim)
     probability=np.exp(scores-scores.max());probability/=probability.sum()
     oracle[head]=probability@expected_v[first:,kv].astype(np.float64)
    error=float(np.max(np.abs(result.astype(np.float64)-oracle)));assert error<=3e-6,(format,seq,split,window,error)
    after=state();assert after['live']==initial['live']and after['categories']==initial['categories']
    report['cases'].append({'format':format,'seq':seq,'heads':heads,'kv_heads':kv_heads,'dim':dim,'window':window,'split':split,'max_abs_attention_error':error,'physical_kv_bytes':physical,'memory_after':after})
# Probe the Q8 scale overflow branch independently from large attention scores.
q=np.zeros((4,64),np.float32);k=np.full((1,2,64),np.finfo(np.float32).max,np.float32);k[0,1]*=-1;v=np.zeros_like(k)
dk=np.empty_like(k);dv=np.empty_like(v);result=np.empty_like(q);sk=np.empty((1,2),np.float32);sv=np.empty_like(sk);bytes=c.c_size_t()
assert f(2,1,4,2,64,0,0,ptr(q),ptr(k),ptr(v),ptr(result),ptr(dk),ptr(dv),ptr(sk),ptr(sv),c.byref(bytes))==0
expected,scales=encode(k,2);assert np.isfinite(dk).all()and np.array_equal(expected.view(np.uint32),dk.view(np.uint32))and np.array_equal(sk.view(np.uint32),scales.view(np.uint32))and np.all(result==0)
assert state()['live']==initial['live']and state()['categories']==initial['categories']
report['finite_extreme_q8']={'scale_overflow_branch':True,'finite_decode':True,'bit_exact_decode':True,'memory_after':state()}
(out/'attention-oracle.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('Encoded K/V decode/scales/physical bytes and actual F64 attention oracle:',len(report['cases']),'cases passed.',flush=True)
