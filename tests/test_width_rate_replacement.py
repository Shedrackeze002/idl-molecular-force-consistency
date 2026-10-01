"""Replacement boundaries: mutations must fail before numerical work."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

CONFIG = Path('configs/aspirin-width-rate-replacement.json')

def module(name='width_rate_replacement_contract'):
    spec=importlib.util.find_spec('force_consistency.'+name)
    assert spec is not None, 'Replacement boundary not implemented'
    return importlib.import_module('force_consistency.'+name)

@pytest.mark.parametrize('key,value',[
    ('max_updates',6001),('backend','gpu'),('run_directory','runs/aspirin-width-rate-2026-09-30'),
    ('total_wall_seconds',11401),('loss_weights',{'energy':1.,'force':1.}),('extra',True),
])
def test_unreviewed_controls_rejected(key,value):
    config=json.loads(CONFIG.read_text());config[key]=value
    with pytest.raises(ValueError):module().validate_replacement(config)

def test_all_pins_and_ancestry_bind_original_facts():
    out=module().verify_replacement_sources(json.loads(CONFIG.read_text()))
    assert out['config_identity']=='d0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36'
    assert out['pin_count']==518
    assert len(out['historical'])==63
    assert out['ancestry']['original_status']=='failed'
    assert out['ancestry']['copy_status']=='passed'
    assert out['original_baselines']==json.loads(Path('runs/aspirin-loss-balance-2026-09-30/balanced/history.json').read_text())[0]['metrics']

def test_changed_pinned_file_is_rejected_without_array_import(monkeypatch):
    contract=module();actual=contract.digest
    monkeypatch.setattr(contract,'digest',lambda path:'0'*64 if str(path).endswith('uv.lock') else actual(path))
    with pytest.raises(ValueError,match='source'):contract.verify_replacement_sources(json.loads(CONFIG.read_text()))

@pytest.mark.parametrize('kind',['existing','symlink','race'])
def test_root_creation_never_writes_unowned_root(tmp_path,kind):
    boundary=module();root=tmp_path/'run';other=tmp_path/'other';other.mkdir()
    if kind=='existing':root.mkdir();(root/'keep').write_text('untouched')
    elif kind=='symlink':root.symlink_to(other,target_is_directory=True)
    else:root.mkdir()
    with pytest.raises((ValueError,FileExistsError)):boundary.claim_root(root)
    assert not (root/'invocation.json').exists()
    assert not (other/'invocation.json').exists()
    if kind=='existing':assert (root/'keep').read_text()=='untouched'

def test_original_preflight_receives_only_original_config(monkeypatch):
    replacement=module('width_rate_replacement');old=json.loads(Path('configs/aspirin-width-rate-diagnostic.json').read_text())
    class Engine:
        @staticmethod
        def width_rate_preflight(config,context,save_partial):
            assert config==old
            return {'sources':module().verify_replacement_sources(json.loads(CONFIG.read_text()))['original_sources']}
    monkeypatch.setattr(replacement,'_engine',lambda:Engine)
    out=replacement.replacement_preflight(json.loads(CONFIG.read_text()),None,lambda r:None)
    assert out['sources']['pin_count']==518

def test_bad_public_config_rejected_before_numerical_import(tmp_path):
    bad=json.loads(CONFIG.read_text());bad['backend']='gpu';path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
    code='''
import sys, importlib.abc, runpy
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('jax','numpy','flax','optax','e3x'):raise AssertionError('NUMERICAL_IMPORT')
sys.meta_path.insert(0,Deny())
sys.argv=['replacement','--config',sys.argv[1]]
runpy.run_module('force_consistency.width_rate_replacement',run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Unreviewed replacement configuration' in result.stderr
    assert 'NUMERICAL_IMPORT' not in result.stderr
