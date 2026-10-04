"""Corruption and identity changes must never reuse the wrong counter replay."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from policy_replay_cache import ReplayCache,canonical_sha
class ReplayCacheTests(unittest.TestCase):
 def test_reuse_and_corruption(self):
  calls=[]
  def simulate(rows,budget,policy):calls.append((budget,policy));return dict(hits=len(rows),misses=0,phases={rows[0]['phase']:dict(hits=len(rows))})
  rows=[dict(pass_=1,position=0,phase='prefill',layer=0,selected=[2,3],group_bytes=4)]
  rows[0]['pass']=rows[0].pop('pass_');digest=canonical_sha(rows)
  with tempfile.TemporaryDirectory()as folder:
   cache=ReplayCache(folder,simulate,'source-a');expected=cache.replay(rows,8,'lru',digest)
   self.assertEqual(cache.replay(rows,8,'lru',digest),expected);self.assertEqual(len(calls),1)
   path=next(Path(folder).glob('*.json'));record=json.loads(path.read_text());record['result']['hits']=999
   path.write_text(json.dumps(record));self.assertEqual(cache.replay(rows,8,'lru',digest),expected);self.assertEqual(len(calls),2)
   path.write_text('{');self.assertEqual(cache.replay(rows,8,'lru',digest),expected);self.assertEqual(len(calls),3)
   self.assertFalse(list(Path(folder).glob('*.tmp')))
 def test_all_semantic_identity_fields_invalidate(self):
  calls=[]
  def simulate(rows,budget,policy):calls.append(1);return dict(hits=0)
  rows=[{'pass':1,'position':0,'phase':'prefill','layer':0,'selected':[2,3],'group_bytes':4}]
  with tempfile.TemporaryDirectory()as folder:
   cache=ReplayCache(folder,simulate,'source-a');digest=canonical_sha(rows)
   cache.replay(rows,8,'lru',digest);cache.replay(rows,16,'lru',digest);cache.replay(rows,8,'lfu',digest)
   ReplayCache(folder,simulate,'source-b').replay(rows,8,'lru',digest)
   for key,value in [('pass',2),('position',1),('phase','decode'),('layer',1),('selected',[3,2]),('group_bytes',8)]:
    changed=[rows[0]|{key:value}];new_digest=canonical_sha(changed);self.assertNotEqual(new_digest,digest)
    cache.replay(changed,8,'lru',new_digest)
   self.assertEqual(len(calls),10)
   # Placement-only metadata has no effect on the simulator's router input.
   self.assertEqual(canonical_sha([rows[0]|{'budget_bytes':100}]),digest)
if __name__=='__main__':unittest.main()
