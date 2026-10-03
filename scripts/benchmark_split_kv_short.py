#!/usr/bin/env python3
"""Same-revision short-prompt decode ablation, with actual split-kernel counters."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time
from unittest.mock import patch
import requests
from benchmark_engines import Server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','binary','library','output-dir']: parser.add_argument('--'+name,type=pathlib.Path,required=True)
    parser.add_argument('--port',type=int,default=18109)
    args=parser.parse_args(); root=args.output_dir; root.mkdir(parents=True,exist_ok=True)
    binary=root/'rbitnet.exe'; binary.write_bytes(args.binary.read_bytes())
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    config.update(rbitnet=str(binary.resolve()),cuda_quant_library=str(args.library.resolve()),port=args.port)
    config.pop('rbitnet_env',None)
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),warmup_cycle=0,rows=[],memory={})
    original=subprocess.Popen
    def save(): (root/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for model in [m for m in config['models'] if m['id'] in ['llama32-1b','qwen35-2b']]:
        baseline={}
        for split in ['0','1']:
            label=model['id']+'-split'+split
            def popen(*a,**kw):
                if kw.get('env'):
                    kw['env']=kw['env'].copy()
                    kw['env'].update(RBITNET_CUDA_SPLIT_KV=split,RBITNET_CUDA_QWEN_FULL='1' if model['id']=='qwen35-2b' else '0',
                                     RBITNET_REQUIRE_QWEN_FULL='1' if model['id']=='qwen35-2b' else '0',RBITNET_PREFIX_KV='0',
                                     RBITNET_SPECULATIVE='0',RBITNET_SPECULATIVE_PLD='0',RBITNET_CUDA_PREFILL='0')
                    for key in ['RBITNET_CHAT_TEMPLATE','RBITNET_CHAT_FORMAT']: kw['env'].pop(key,None)
                return original(*a,**kw)
            server=Server(config,model,'rbitnet','gpu',root); server.log_path=root/(label+'.log')
            try:
                with patch('subprocess.Popen',popen): server.start()
                for cycle in range(4):
                    body=dict(model=model['id'],messages=[dict(role='user',content='Write a long story about a robot exploring a library.')],max_tokens=32,temperature=0)
                    before=server.metrics(); start=time.perf_counter()
                    response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=120); response.raise_for_status()
                    wall=1000*(time.perf_counter()-start); raw=response.json(); after=server.metrics()
                    delta={k:after.get(k,0)-before.get(k,0) for k in after}
                    text=raw['choices'][0]['message']['content']
                    if split=='0': baseline[cycle]=text
                    row=dict(model=model['id'],split=split,cycle=cycle,request=body,response=raw,wall_ms=wall,metrics_delta=delta,matches_baseline=text==baseline[cycle])
                    report['rows'].append(row); save()
                    assert row['matches_baseline'] and text and '\ufffd' not in text
                    print(label,cycle,delta['rbitnet_inference_decode_ms_sum'],wall,flush=True)
            finally: report['memory'][label]=server.close(); save()


if __name__=='__main__': main()
