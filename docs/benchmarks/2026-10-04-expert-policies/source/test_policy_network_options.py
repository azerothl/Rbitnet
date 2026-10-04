"""Exercise the final child options, without starting CUDA or a server."""
import contextlib,importlib.util,io,json,pathlib,sys,tempfile,unittest
from unittest.mock import patch

class BoundaryReached(Exception): pass

class NetworkOptions(unittest.TestCase):
    def test_old_unrecorded_async_capture_is_archived_instead_of_reused(self):
        from resume_policy import reuse_or_archive
        source=pathlib.Path(__file__).resolve().parent.parent/'diagnostics/network-pre-correction.json'
        helper=pathlib.Path(__file__).resolve().parent/'live.py'
        capture=json.loads(source.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);target=root/'live/gpt-oss-20b-lru';target.mkdir(parents=True)
            result=target/'results.json';result.write_bytes(source.read_bytes());records=[]
            self.assertFalse(reuse_or_archive('network-gpt-oss-20b-lru',[sys.executable,str(helper),'--output-dir',str(target),'--policy','lru'],root,capture['binary_sha256'],capture['library_sha256'],records))
            self.assertFalse(target.exists())
            self.assertEqual(records[0]['action'],'preserved_incomplete_capture')
            archived=list((root/'interrupted').glob('*/0-gpt-oss-20b-lru/results.json'))
            self.assertEqual(len(archived),1)
            self.assertEqual(archived[0].read_bytes(),source.read_bytes())

    def test_final_child_options_and_capture_match_each_sync_policy(self):
        path=pathlib.Path(__file__).resolve().parent/'live.py'
        spec=importlib.util.spec_from_file_location('policy_live_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for family in ['gpt','mla']:
            for policy in ['lru','lfu','least-stale']:
                with self.subTest(family=family,policy=policy),tempfile.TemporaryDirectory() as directory:
                    root=pathlib.Path(directory);binary=root/'input.exe';library=root/'input.dll'
                    binary.write_bytes(b'not executed');library.write_bytes(b'not loaded')
                    config=root/'config.json';config.write_text(json.dumps(dict(models=[dict(id='gpt-oss-20b'),dict(id='glm47-flash')],port=18138)),encoding='utf-8')
                    argv=[str(path),'--config',str(config),'--binary',str(binary),'--library',str(library),'--output-dir',str(root/'capture'),
                          '--moe-cache','512','--policy',policy,'--split-kv']
                    argv+=['--gpt-full','--gpt-segmented']if family=='gpt'else['--mla-full']
                    observed={};auxiliary=[]
                    def child(*args,**kwargs):
                        if 'env'not in kwargs:
                            auxiliary.append(args);return object()
                        observed.update(kwargs['env']);raise BoundaryReached()
                    class FakeServer:
                        def __init__(self,*args): pass
                        def start(self):
                            module.subprocess.Popen(['memory sampler without env'])
                            # Deliberately hostile inherited values must be overwritten.
                            module.subprocess.Popen(['never launched'],env=dict(RBITNET_MOE_ASYNC='1',RBITNET_MOE_PREFETCH='previous-pass',RBITNET_CHAT_TEMPLATE='{user}',RBITNET_CHAT_FORMAT='raw'))
                        def close(self): return {}
                    with patch.object(module,'Server',FakeServer),patch.object(module.subprocess,'Popen',child),patch.object(sys,'argv',argv):
                        with self.assertRaises(BoundaryReached):module.main()
                    wanted=dict(RBITNET_MOE_CACHE_POLICY=policy,RBITNET_MOE_ASYNC='0',RBITNET_MOE_PREFETCH='off',RBITNET_MOE_CACHE_MB='512')
                    self.assertEqual({k:observed[k] for k in wanted},wanted)
                    self.assertEqual(len(auxiliary),1)
                    self.assertNotIn('RBITNET_CHAT_TEMPLATE',observed)
                    report=json.loads((root/'capture/results.json').read_text(encoding='utf-8'))
                    self.assertEqual(report['effective_moe_env'],wanted)
                    self.assertEqual(report['policy'],policy)
                    self.assertIs(report['moe_async'],False)
                    self.assertEqual(report['prefetch'],'off')

    def test_async_option_is_refused_in_this_sync_study(self):
        path=pathlib.Path(__file__).resolve().parent/'live.py'
        spec=importlib.util.spec_from_file_location('policy_live_argument_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        argv=[str(path),'--config','unused','--binary','unused','--library','unused','--output-dir','unused','--policy','lru','--async']
        with patch.object(sys,'argv',argv),contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as result:module.main()
        self.assertEqual(result.exception.code,2)
        self.assertIn('unrecognized arguments: --async',errors.getvalue())

if __name__=='__main__':unittest.main()
