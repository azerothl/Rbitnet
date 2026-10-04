"""Propagate actual termination without guessing from decoded text or token counts."""
from pathlib import Path
import re,subprocess
root=Path.cwd();base=root/'target/performance-cache';out=base/'finish-reason-integration'
out.mkdir(exist_ok=True)

def rep(s,a,b,n=1):
    assert s.count(a)==n,(a[:100],s.count(a),n)
    return s.replace(a,b)

def store(path,s):
    dest=out/path;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(s,encoding='utf-8',newline='\n')
    subprocess.run(['rustfmt','--edition','2021','--config','skip_children=true',str(dest)],check=True)

def literal_defaults(s,name):
    # These known literals contain only scalar arithmetic, no string braces.
    replacements=[]
    for m in re.finditer(r'\b'+name+r' \{',s):
        prefix=s[max(0,m.start()-12):m.start()]
        if re.search(r'(?:struct|impl)\s*$',prefix):continue
        start=m.end();depth=1;end=start
        while depth:
            ch=s[end];depth+=(ch=='{')-(ch=='}');end+=1
        inner=s[start:end-1]
        if '..' not in inner and 'finish_reason:' not in inner:
            replacements.append((end-1,'finish_reason: Default::default(),\n'))
    for position,value in reversed(replacements):s=s[:position]+value+s[position:]
    return s

paths=[p.relative_to(root)for p in(root/'crates').rglob('*.rs')if 'PhaseTimings {'in p.read_text(encoding='utf-8')or 'InferenceStats {'in p.read_text(encoding='utf-8')]
sources={p:literal_defaults(literal_defaults((root/p).read_text(encoding='utf-8'),'PhaseTimings'),'InferenceStats')for p in paths}
p=Path('crates/bitnet-core/src/timings.rs');s=sources[p]
s=rep(s,'/// Millisecond-resolution timings plus token counts from the tokenizer/runtime.','''/// Why the model stopped. Unknown is reserved for executors that only expose
/// approximate text statistics, never inferred from their output length.
#[derive(Debug,Clone,Copy,Default,PartialEq,Eq)]
pub enum GenerationFinishReason {
    Stop,
    Length,
    #[default]
    Unknown,
}
impl GenerationFinishReason {
    pub fn openai(self)->Option<&'static str> {
        match self {Self::Stop=>Some("stop"),Self::Length=>Some("length"),Self::Unknown=>None}
    }
}

/// Millisecond-resolution timings plus token counts from the tokenizer/runtime.''')
s=rep(s,'    pub completion_tokens: u32,','    pub completion_tokens: u32,\n    pub finish_reason: GenerationFinishReason,')
s=rep(s,'            completion_tokens: completion_tokens_est,','            completion_tokens: completion_tokens_est,\n            finish_reason: GenerationFinishReason::Unknown,')
sources[p]=s
p=Path('crates/bitnet-core/src/scheduler.rs');s=sources[p]
s=rep(s,'    pub speculative_attempted: bool,','    pub speculative_attempted: bool,\n    pub finish_reason: crate::timings::GenerationFinishReason,')
s=rep(s,'            speculative_attempted,','            speculative_attempted,\n            finish_reason: p.finish_reason,')
s=rep(s,'            speculative_attempted: true,','            speculative_attempted: true,\n            finish_reason: verify.finish_reason,')
# Legacy two-burst scheduling must not start another turn after observing EOS,
# and the final burst owns the termination reason if continuation is needed.
s=rep(s,'        if remaining > 0 {','        if remaining > 0 && phases_acc.finish_reason != crate::timings::GenerationFinishReason::Stop {')
s=rep(s,'            text.push_str(&tail);','            text.push_str(&tail);\n            phases_acc.finish_reason=tail_phases.finish_reason;')
# An EOS step can produce zero tokens. The previous queue could retry forever.
needle='                    if entry.1.completion_tokens >= orig.request.max_tokens {\n                        queue.mark_done(id);\n                    }'
replacement='''                    if phases.finish_reason==crate::timings::GenerationFinishReason::Stop {
                        entry.1.finish_reason=phases.finish_reason;
                        queue.mark_done(id);
                    } else if entry.1.completion_tokens>=orig.request.max_tokens {
                        entry.1.finish_reason=crate::timings::GenerationFinishReason::Length;
                        queue.mark_done(id);
                    } else if phases.completion_tokens==0 {
                        return Err(crate::error::BitNetError::Inference("decode step made no progress without an explicit stop reason".into()));
                    }'''
s=rep(s,needle,replacement)
needle='            if entry.1.completion_tokens >= orig.request.max_tokens {\n                queue.mark_done(id);\n            }'
s=rep(s,needle,replacement.replace('                    ','            '))
s=rep(s,'                    if entry.1.completion_tokens >= orig.request.max_tokens {\n                        queue.mark_done(id);\n                        continue;',
    '                    if entry.1.completion_tokens >= orig.request.max_tokens {\n                        entry.1.finish_reason=crate::timings::GenerationFinishReason::Length;\n                        queue.mark_done(id);\n                        continue;')
s=rep(s,'            if entry.1.completion_tokens >= orig.request.max_tokens {\n                done_ids.push(id);',
    '            if entry.1.completion_tokens >= orig.request.max_tokens {\n                entry.1.finish_reason=crate::timings::GenerationFinishReason::Length;\n                done_ids.push(id);')
# Outer zero-token requests are a budget termination, no EOS was sampled.
s=rep(s,'stats: InferenceStats::from_phases(PhaseTimings::default(), false),','stats: InferenceStats::from_phases(PhaseTimings {finish_reason:crate::timings::GenerationFinishReason::Length,..Default::default()}, false),')
sources[p]=s
finish_decision_tests='''\n#[cfg(test)]
mod finish_tests {
    #[test]
    fn eos_inside_verification_and_budget_exhaustion_are_distinct() {
        let eos=super::decide(&[2,3],2,&[99],&[],|i,_|[2,99][i]);
        assert!(eos.ended_on_eos);assert_eq!(eos.confirmed,[2]);assert_eq!(eos.pending,None);
        let budget=super::decide(&[2,3],2,&[99],&[],|i,_|[2,3][i]);
        assert!(!budget.ended_on_eos);assert_eq!(budget.confirmed,[2,3]);assert_eq!(budget.pending,None);
        let zero=super::decide(&[2],0,&[99],&[],|_,_|panic!("zero budget must not consume a sample"));
        assert!(!zero.ended_on_eos);assert!(zero.confirmed.is_empty());
    }
}
'''

for family in ['llama','qwen35','qwen3','mixtral']:
    p=Path(f'crates/bitnet-core/src/{family}/runtime.rs');s=sources[p]
    if family=='llama':
        s=rep(s,'        for step in 0..max_tokens {','        let mut finish_reason=crate::timings::GenerationFinishReason::Length;\n        for step in 0..max_tokens {')
        s=rep(s,'        while generated.len() < limit as usize {','        let mut finish_reason=crate::timings::GenerationFinishReason::Length;\n        while generated.len() < limit as usize {')
        s=rep(s,'        for step in 0..limit {','        let mut finish_reason=crate::timings::GenerationFinishReason::Length;\n        for step in 0..limit {')
        for clause in ['eos_ids.contains(&next_id)','eos.contains(&next)','stop.contains(&next)']:
            s=rep(s,'            if '+clause+' {\n                break;','            if '+clause+' {\n                finish_reason=crate::timings::GenerationFinishReason::Stop;\n                break;')
        s=rep(s,'            let Some(pending) = decision.pending else {\n                break;','            let Some(pending) = decision.pending else {\n                if decision.ended_on_eos {finish_reason=crate::timings::GenerationFinishReason::Stop;}\n                break;')
    else:
        loop='        for step in 0..max_tokens {'if family=='qwen35'else'        for _ in 0..max_tokens {'
        s=rep(s,loop,'        let mut finish_reason=crate::timings::GenerationFinishReason::Length;\n'+loop)
        clause='eos_ids.contains(&next_id)'if family=='qwen35'else'Some(next_id) == eos_id'
        s=rep(s,'            if '+clause+' {\n                break;','            if '+clause+' {\n                finish_reason=crate::timings::GenerationFinishReason::Stop;\n                break;')
    # Only complete-generation literals carry the observed result.
    s,n=re.subn(r'(completion_tokens: (?:gen|generated)\.len\(\) as u32,\s*)finish_reason: Default::default\(\),',r'\1finish_reason,',s)
    assert n==(3 if family=='llama'else 1),(family,n)
    sources[p]=s
p=Path('crates/bitnet-core/src/native/graph.rs');s=sources[p]
s=rep(s,'        for step in 0..limit {','        let mut finish_reason=crate::timings::GenerationFinishReason::Length;\n        for step in 0..limit {')
s=rep(s,'            if stop.contains(&next) {\n                break;','            if stop.contains(&next) {\n                finish_reason=crate::timings::GenerationFinishReason::Stop;\n                break;')
s,n=re.subn(r'(completion_tokens: generated\.len\(\) as u32,\s*)finish_reason: Default::default\(\),',r'\1finish_reason,',s);assert n==1
sources[p]=s
p=Path('crates/bitnet-core/src/llama/speculative.rs');s=(root/p).read_text(encoding='utf-8')
s=rep(s,'    pub accepted: usize,','    pub accepted: usize,\n    pub ended_on_eos: bool,')
s=rep(s,'        accepted: 0,','        accepted: 0,\n        ended_on_eos: false,')
s=rep(s,'        if eos.contains(&target) {\n            break;','        if eos.contains(&target) {\n            decision.ended_on_eos=true;\n            break;')
sources[p]=s+finish_decision_tests

p=Path('crates/bitnet-core/src/inference.rs');s=sources[p]
# Static stub completions are complete by definition. Toy/legacy estimates
# retain Unknown; no artificial EOS or length is inferred from whitespace.
for start in sorted([s.index('        if self.inner.stub {',s.index('    pub fn '+name+'('))for name in ['complete_detailed_with_options','complete_streaming']],reverse=True):
    candidates=[s.find(value,start)for value in ['        if let Some(ref ','        if self.inner.toy.is_some() {']]
    end=min(value for value in candidates if value>=0)
    section=s[start:end];section=rep(section,'finish_reason: Default::default(),','finish_reason: crate::timings::GenerationFinishReason::Stop,');s=s[:start]+section+s[end:]
sources[p]=s

p=Path('crates/bitnet-server/src/stream_stop.rs');s=(root/p).read_text(encoding='utf-8')
s=rep(s,'    pub fn finish(&mut self) -> String {','    pub fn stopped(&self)->bool {self.stopped}\n    pub fn finish(&mut self) -> String {')
sources[p]=s
p=Path('crates/bitnet-server/src/lib.rs');s=(root/p).read_text(encoding='utf-8')
# Completion route must preserve the chat route's reason.
first=s.index('"finish_reason": "stop"');s=s[:first]+s[first:].replace('"finish_reason": "stop"','"finish_reason": chat_body["choices"][0]["finish_reason"].clone()',1)
s=rep(s,'    let text = apply_stop_sequences(output.text, req.stop.as_ref());','''    let original_len=output.text.len();
    let text = apply_stop_sequences(output.text, req.stop.as_ref());
    let mut stats=output.stats;
    if text.len()<original_len {stats.finish_reason=bitnet_core::timings::GenerationFinishReason::Stop;}''')
s=rep(s,'json_completion(&request_model, &text, &output.stats)','json_completion(&request_model, &text, &stats)')
start=s.index('fn json_completion(');end=s.index('async fn live_stream_chat_completion',start)
section=s[start:end];section=rep(section,'"finish_reason": "stop"','"finish_reason": stats.finish_reason.openai()');s=s[:start]+section+s[end:]
start=s.index('Poll::Ready(Some(Ok(StreamEvent::Done(output))))');end=s.index('Poll::Ready(Some(Err(msg)))',start)
section=s[start:end];section=rep(section,'"finish_reason": "stop"','"finish_reason": if stop_filter.stopped(){Some("stop")}else{output.stats.finish_reason.openai()}');s=s[:start]+section+s[end:]
# Unused text-only helper knows nothing about token termination.
s=rep(s,'"finish_reason": "stop"','"finish_reason": serde_json::Value::Null')
sources[p]=s
p=Path('crates/bitnet-server/src/anthropic.rs');s=(root/p).read_text(encoding='utf-8')
s=rep(s,"    pub stop_reason: &'static str,","    pub stop_reason: Option<&'static str>,")
s=rep(s,'        stop_reason: "end_turn",','        stop_reason: match output.stats.finish_reason {bitnet_core::timings::GenerationFinishReason::Stop=>Some("end_turn"),bitnet_core::timings::GenerationFinishReason::Length=>Some("max_tokens"),bitnet_core::timings::GenerationFinishReason::Unknown=>None},')
s=rep(s,'"stop_reason": "end_turn"','"stop_reason": match output.stats.finish_reason {bitnet_core::timings::GenerationFinishReason::Stop=>Some("end_turn"),bitnet_core::timings::GenerationFinishReason::Length=>Some("max_tokens"),bitnet_core::timings::GenerationFinishReason::Unknown=>None}')
sources[p]=s
for p,s in sources.items():store(p,s)
server_fixture=(root/'crates/bitnet-server/tests/context_capacity.rs').read_text(encoding='utf-8').split('#[tokio::test]')[0]
server_fixture=rep(server_fixture,'fn fixture(path: &std::path::Path) {','fn fixture(path: &std::path::Path, eos:bool) {')
server_fixture=rep(server_fixture,'if i / 8 == 1 {','if i / 8 == (if eos {2}else{1}) {')
server_fixture+='\n'+(base/'finish_reason_http_draft.rs').read_text(encoding='utf-8')
store(Path('crates/bitnet-server/tests/finish_reason.rs'),server_fixture)
timings=sources[Path('crates/bitnet-core/src/timings.rs')]+'''\n#[cfg(test)]
mod finish_tests {
    use super::*;
    #[test]
    fn legacy_statistics_are_unknown_and_phase_reason_survives_both_combiners() {
        assert_eq!(PhaseTimings::from_total_wall_ms(10,128).finish_reason,GenerationFinishReason::Unknown);
        assert_eq!(GenerationFinishReason::Unknown.openai(),None);
        for reason in [GenerationFinishReason::Stop,GenerationFinishReason::Length] {
            let phases=PhaseTimings {finish_reason:reason,completion_tokens:0,..Default::default()};
            assert_eq!(crate::scheduler::InferenceStats::from_phases(phases,false).finish_reason,reason);
            #[allow(deprecated)]
            let stats=crate::scheduler::InferenceStats::from_speculative_phases(PhaseTimings::default(),phases);
            assert_eq!(stats.finish_reason,reason);
        }
    }
}
'''
store(Path('crates/bitnet-core/src/timings.rs'),timings)
s=sources[Path('crates/bitnet-core/tests/scheduler_speculative.rs')]+ '\n'+(base/'finish_reason_scheduler_draft.rs').read_text(encoding='utf-8')
store(Path('crates/bitnet-core/tests/scheduler_speculative.rs'),s)
print(f'Private finish-reason propagation prepared in {len(sources)} files; compilation and actual runtime/HTTP proofs pending.')
