"""Read the completed warmed Nsight capture; no inference or throughput claims."""
from pathlib import Path
import bisect,hashlib,json,sqlite3

base=Path.cwd()/'target/async-expert-cache/production-proof'
db=base/'warmed-async-cuda.sqlite';con=sqlite3.connect(db)
names=dict(con.execute('select id,value from StringIds'))
kernels=con.execute('select start,end,deviceId,contextId,streamId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
copies=con.execute('select start,end,deviceId,contextId,streamId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start').fetchall()
apis=con.execute('select start,end,nameId,returnValue from CUPTI_ACTIVITY_KIND_RUNTIME order by start').fetchall()
assert kernels and copies and apis
completed=json.loads((base/'profile-child-proof.json').read_text(encoding='utf-8'))
assert completed['same_output'] and completed['profiler_stop_succeeded'] and completed['async_failed']==0

def union(intervals):
 out=[]
 for start,end in sorted(intervals):
  if out and start<=out[-1][1]:out[-1][1]=max(end,out[-1][1])
  else:out.append([start,end])
 return out

by_context={}
for start,end,device,context,stream,name in kernels:by_context.setdefault((device,context,stream),[]).append((start,end,name))
groups={};examples=[]
for device,context,stream in sorted({(c[2],c[3],c[4])for c in copies if c[6]==1}):
 work=sorted((a,b,n,s)for (d,c,s),v in by_context.items()if d==device and c==context and s!=stream for a,b,n in v)
 spans=union([(a,b)for a,b,_,_ in work]);ends=[b for _,b in spans]
 selected=[c for c in copies if c[2:5]==(device,context,stream)and c[6]==1]
 overlap=0;overlapping_copies=0
 for a,b,_,_,_,size,_ in selected:
  i=bisect.bisect_right(ends,a);local=0
  while i<len(spans)and spans[i][0]<b:
   local+=max(0,min(b,spans[i][1])-max(a,spans[i][0]));i+=1
  overlap+=local;overlapping_copies+=local>0
  if local and len(examples)<12:
   first=next((x for x in work if x[0]<b and x[1]>a),None)
   assert first
   examples.append({'copy_start_ns':a,'copy_end_ns':b,'copy_bytes':size,'copy_stream':stream,'overlap_ns':local,'kernel_start_ns':first[0],'kernel_end_ns':first[1],'kernel':names[first[2]],'kernel_stream':first[3]})
 total=sum(c[1]-c[0]for c in selected)
 groups[str((device,context,stream))]={'device':device,'context':context,'stream':stream,'h2d_copies':len(selected),'h2d_bytes':sum(c[5]for c in selected),'h2d_duration_ns':total,'other_stream_compute_union_ns':sum(b-a for a,b in spans),'copy_compute_overlap_ns':overlap,'copies_overlapping_other_stream_compute':overlapping_copies,'fraction_copy_time_overlapped':overlap/total if total else 0}

alloc={};errors={};counts={}
for start,end,name,status in apis:
 label=names[name];counts[label]=counts.get(label,0)+1
 if any(x in label for x in ['Malloc','Free','HostAlloc','HostRegister','HostUnregister','AllocAsync']):alloc[label]=alloc.get(label,0)+1
 if status:errors[label]=errors.get(label,0)+1
assert not errors,errors
report={'capture':'completed, warmed second GPT generation, 512 MiB expert pool, previous-pass predictor','capture_artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [db,base/'warmed-async-cuda.nsys-rep',base/'profile-child-proof.json']},'child_completion':completed,'kernel_events':len(kernels),'kernel_duration_sum_ns':sum(k[1]-k[0]for k in kernels),'copy_events':len(copies),'h2d_groups':groups,'observed_runtime_api_counts':counts,'observed_warm_allocation_or_release_api_counts':alloc,'runtime_errors':errors,'overlap_examples':examples,'limits':['Profiler timing is perturbed and is not a throughput benchmark.','Absence of allocation APIs applies only to this captured warmed generation.','The predictor uses the previous pass; this is not a trained expert predictor.','Copy overlap is the interval intersection with kernels on other streams on the same GPU and CUDA context; intervals are merged to avoid double counting.']}
(base/'trace-analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'h2d_groups':groups,'warm_allocation_api_counts':alloc,'runtime_errors':errors,'example_count':len(examples)},indent=2))
