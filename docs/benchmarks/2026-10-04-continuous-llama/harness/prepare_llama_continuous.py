"""Build an isolated integration after Native batch and real finish-reason proof."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
root=Path.cwd();base=root/'target/performance-cache';crate=base/'check-llama-continuous';crate.mkdir(exist_ok=True)
def rep(s,a,b,n=1):
    assert s.count(a)==n,(a[:100],s.count(a),n);return s.replace(a,b)
for name in ['Cargo.toml','Cargo.lock']:shutil.copy2(root/name,crate/name)
for name in ['crates','recipes','.cargo','tests']:
    if(root/name).exists():shutil.copytree(root/name,crate/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('target','__pycache__'))
draft_only='--draft-only'in sys.argv
finish=base/'finish-reason-proof';manifest=None if draft_only else json.loads((finish/'manifest.json').read_text(encoding='utf-8'))
frozen_finish=base/'check-finish-reason'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# These exact corrected runtime/statistics files are immutable proof inputs.
for p in(base/'finish-reason-integration').rglob('*.rs'):
    relative=p.relative_to(base/'finish-reason-integration');source=p if draft_only else frozen_finish/relative
    if not draft_only:assert sha(source)==manifest['sources'][relative.as_posix()],relative
    dest=crate/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
core=crate/'crates/bitnet-core/src'
for name,target in [('llama_transient_batch_draft.rs','transient_batch.rs'),('llama_continuous_draft.rs','continuous.rs'),('llama_batch_controller_draft.rs','controller.rs'),('llama_continuous_tests_draft.rs','continuous_tests.rs')]:
    shutil.copy2(base/name,core/'llama/resident'/target)
p=core/'llama/resident.rs';p.write_text(p.read_text(encoding='utf-8')+'''
#[path="resident/transient_batch.rs"] mod transient_batch;
#[path="resident/continuous.rs"] mod continuous;
#[path="resident/controller.rs"] mod controller;
pub(crate) use controller::{BatchController,BatchOptions};
#[cfg(test)] #[path="resident/continuous_tests.rs"] mod continuous_tests;
''',encoding='utf-8',newline='\n')
p=core/'llama/mod.rs';p.write_text(p.read_text(encoding='utf-8')+'\npub(crate) use resident::{BatchController,BatchOptions};\n',encoding='utf-8',newline='\n')
p=core/'model/executor.rs';s=p.read_text(encoding='utf-8')
start=s.index('pub struct LlamaExecutor');end=s.index('pub struct BitNetNativeExecutor');part=s[start:end]
part=rep(part,'    runtime: Mutex<Option<LlamaRuntime>>,','    runtime: Mutex<Option<LlamaRuntime>>,\n    batch_options:Option<crate::llama::BatchOptions>,\n    batch:Mutex<Option<Arc<crate::llama::BatchController>>>,')
part=rep(part,'            runtime: Mutex::new(None),','            runtime: Mutex::new(None),\n            batch_options:crate::llama::BatchOptions::configured(backend_kind)?,\n            batch:Mutex::new(None),',2)
anchor='impl ModelExecutor for LlamaExecutor {'
helper='''impl LlamaExecutor {
    fn native_batch_controller(&self)->Result<Option<Arc<crate::llama::BatchController>>> {
        let Some(options)=self.batch_options else{return Ok(None)};
        let mut slot=self.batch.lock().map_err(|e|BitNetError::Inference(format!("batch startup lock poisoned: {e}")))?;
        if slot.is_none() {
            let model=Arc::new(crate::llama::LlamaModel::from_gguf_arc_with_config(Arc::clone(&self.gguf),self.backend_kind,self.config.clone())?);
            *slot=Some(Arc::new(crate::llama::BatchController::start(model,Arc::clone(&self.prompt_tokenizer),options)?));
        }
        Ok(slot.as_ref().cloned())
    }
}
'''
part=rep(part,anchor,helper+anchor)
for name,result,call in [('generate_with_timings','(String, PhaseTimings)','generate_with_timings'),('generate_output','crate::scheduler::InferenceOutput','generate_output')]:
    a=part.index('    fn '+name+'(');b=part.index('        let mut slot = self',a)
    assert 'Result<'+result+'>'in part[a:b] or 'Result<'+result.replace(' ','')+'>'in part[a:b],(name,part[a:b])
    part=part[:b]+'        if let Some(batch)=self.native_batch_controller()? {return batch.'+call+'(prompt,max_tokens,sampling);}\n'+part[b:]
a=part.index('    fn generate_streaming(');b=part.index('        let mut slot = self',a)
part=part[:b]+'        if let Some(batch)=self.native_batch_controller()? {return batch.generate_streaming(prompt,max_tokens,sampling,on_event);}\n'+part[b:]
# Public metadata distinguishes configured capacity from an initialized pipeline.
a=part.index('    fn offload_metadata(&self) -> Option<String> {')+len('    fn offload_metadata(&self) -> Option<String> {')
part=part[:a]+'''
        if let Some(options)=self.batch_options {
            let initialized=self.batch.lock().ok().is_some_and(|slot|slot.is_some());
            return Some(format!("experimental CUDA continuous Llama: configured=true, initialized={initialized}, slots={}, queued={}, real_prefill_budget={}, F32 mono",options.slots,options.queued,options.token_budget));
        }
'''+part[a:]
s=s[:start]+part+s[end:];p.write_text(s,encoding='utf-8',newline='\n')
p=core/'perf.rs';s=p.read_text(encoding='utf-8')
fields=['gpu_llama_batch_waves','gpu_llama_batch_rows','gpu_llama_batch_projections','gpu_llama_batch_max_rows']
s=rep(s,'    pub gpu_mla_full_tokens: u64,','    pub gpu_mla_full_tokens: u64,\n'+''.join('    pub '+f+': u64,\n'for f in fields))
s=rep(s,'    gpu_mla_full_tokens: AtomicU64,','    gpu_mla_full_tokens: AtomicU64,\n'+''.join('    '+f+': AtomicU64,\n'for f in fields))
s=rep(s,'        gpu_mla_full_tokens: p.gpu_mla_full_tokens.load(Ordering::Relaxed),','        gpu_mla_full_tokens: p.gpu_mla_full_tokens.load(Ordering::Relaxed),\n'+''.join('        '+f+':p.'+f+'.load(Ordering::Relaxed),\n'for f in fields))
s=rep(s,'pub fn snapshot() -> PerfSnapshot {','''/// Successful blocking Native forwards; distinct from virtual scheduler waves.
pub(crate) fn record_native_llama_batch(rows:usize,projections:u64) {
    let p=perf();p.gpu_llama_batch_waves.fetch_add(1,Ordering::Relaxed);
    p.gpu_llama_batch_rows.fetch_add(rows as u64,Ordering::Relaxed);
    p.gpu_llama_batch_projections.fetch_add(projections,Ordering::Relaxed);
    p.gpu_llama_batch_max_rows.fetch_max(rows as u64,Ordering::Relaxed);
}
pub fn snapshot() -> PerfSnapshot {''')
anchor='    let managed = crate::backend::cuda_managed_memory_stats();'
metrics=''.join('    counter!("rbitnet_core_'+f+'_total","Successful shared Native Llama forwards evaluated '+description+'",snap.'+f+');\n'for f,description in zip(fields[:3],['waves','activation rows','matrix projections']))
metrics+='    writeln!(s,"# TYPE rbitnet_core_gpu_llama_batch_max_rows gauge\\nrbitnet_core_gpu_llama_batch_max_rows {}",snap.gpu_llama_batch_max_rows).unwrap();\n'
s=rep(s,anchor,metrics+anchor);p.write_text(s,encoding='utf-8',newline='\n')
paths=[core/'llama/resident.rs',core/'llama/mod.rs',core/'model/executor.rs',*[core/'llama/resident'/name for name in ['transient_batch.rs','continuous.rs','controller.rs','continuous_tests.rs']]]
paths.append(core/'perf.rs')
for p in paths:subprocess.run(['rustfmt','--edition','2021','--config','skip_children=true',str(p)],check=True)
print('Private continuous Llama driver/controller and executor routing prepared from '+('unverified draft'if draft_only else'sealed finish-reason')+' sources; compilation, actual independent requests and network proofs pending.')
