"""Synthetic saved-record recovery reports, never scientific trials."""
import copy
import importlib.util
from pathlib import Path
import json
import subprocess
import sys
import pytest
from test_width_rate_recovery import source,module,passed,write


def reporter():
    spec=importlib.util.spec_from_file_location('recovery_report','scripts/report_width_rate_recovery.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def saved(tmp_path):
    v=source(tmp_path);root=tmp_path/'run';module()._recover(v['config'],root,lambda:v,lambda root,timeout:passed(root,v));return v,root


def test_distinct_valid_copy_never_promotes_failed_original(tmp_path):
    v,root=saved(tmp_path);result=reporter()._generate(v,root,tmp_path/'out')
    assert result['verification_valid'] and result['role']=='copied_checkpoint_verification_only'
    assert not result['original_outcome']['execution_valid'] and not result['original_effort']['actual_counts']['work_counts_complete']
    assert result['new_work']['training_updates']==0 and result['full_pilot']=='pending' and not result['contrasts_available']

@pytest.mark.parametrize('damage',['promote','count','time','code','oldwork','oldtime','receipt','copy','completion'])
def test_coherently_edited_report_receipts_or_claims_rejected(tmp_path,damage):
    v,root=saved(tmp_path);summary=json.loads((root/'summary.json').read_text())
    if damage=='promote':summary['original_outcome']['execution_valid']=True
    elif damage=='count':summary['new_work']['prediction_frames']=999
    elif damage=='time':summary['elapsed_seconds']=999
    elif damage=='code':summary['code']['files_sha256']['uv.lock']='0'*64
    elif damage=='oldwork':summary['original_effort']['actual_counts']['work_counts_complete']=True
    elif damage=='oldtime':summary['original_effort']['per_arm']['tiny']['derived_probe_seconds']=3.
    elif damage=='receipt':
        p=root/'verification-copy/checkpoint-verification.json';r=json.loads(p.read_text());r['adam_count']=10;write(p,r)
    elif damage=='copy':(root/'verification-copy/reference.npz').write_bytes(b'changed')
    else:(root/'controller-completion.json').unlink()
    write(root/'summary.json',summary)
    # Also rehash completion for summary mutations: source-derived semantics must win.
    marker=root/'controller-completion.json'
    if marker.exists():
        import hashlib
        b=json.loads(marker.read_text());b['summary_sha256']=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest();write(marker,b)
    out=reporter()._generate(v,root,tmp_path/'out');assert not out['verification_valid'] and not out['contrasts_available']


def test_failed_probe_work_and_old_incompleteness_are_preserved(tmp_path):
    v=source(tmp_path);root=tmp_path/'run'
    def fail(*args):raise subprocess.CalledProcessError(1,'probe',stderr=b'failure')
    module()._recover(v['config'],root,lambda:v,fail)
    result=reporter()._generate(v,root,tmp_path/'out')
    assert not result['verification_valid'] and result['probe_work_status']=='unknown' and result['new_work']['prediction_frames'] is None
    assert not result['original_effort']['actual_counts']['work_counts_complete']


def test_saved_only_report_import_denies_numerics(tmp_path):
    v,root=saved(tmp_path);v['source_root']=str(v['source_root']);write(tmp_path/'source-object.json',v)
    code='''import importlib.abc,sys,importlib.util,json
from pathlib import Path
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('jax','flax','optax','e3x','numpy'):raise AssertionError(name)
sys.meta_path.insert(0,Deny())
s=importlib.util.spec_from_file_location('report','scripts/report_width_rate_recovery.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
v=json.load(open(sys.argv[1]));v['source_root']=Path(v['source_root']);m._generate(v,Path(sys.argv[2]),Path(sys.argv[3]))
'''
    subprocess.run([sys.executable,'-c',code,str(tmp_path/'source-object.json'),str(root),str(tmp_path/'out')],capture_output=True,check=True)
