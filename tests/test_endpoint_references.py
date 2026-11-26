"""Explicit classified branch targets, separate from historical filename tokens."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPLIT=ROOT/'manifests/landscape17-aspirin-split.json'
CLASSES=ROOT/'reports/landscape17-endpoints.json'

def test_generator_checks_and_protects_existing_companion(tmp_path):
 output=tmp_path/'companion.json'
 command=[sys.executable,str(ROOT/'scripts/build_branch_references.py'),'--output',str(output)]
 generated=subprocess.run(command,capture_output=True,text=True)
 assert generated.returncode==0,generated.stderr
 assert json.loads(generated.stdout)['branches']==74
 original=output.read_bytes()
 checked=subprocess.run(command+['--check'],capture_output=True,text=True)
 assert checked.returncode==0,checked.stderr
 assert output.read_bytes()==original
 refused=subprocess.run(command,capture_output=True,text=True)
 assert refused.returncode!=0 and 'FileExistsError' in refused.stderr
 assert output.read_bytes()==original
 corrupted=json.loads(original);corrupted['branch_references']['aspirin/12/plus']['reference_class']='wrong'
 write(output,corrupted)
 rejected=subprocess.run(command+['--check'],capture_output=True,text=True)
 assert rejected.returncode!=0 and 'EndpointReferenceError' in rejected.stderr

@pytest.mark.parametrize('schema',[True,1.0])
@pytest.mark.parametrize('rehash',[False,True])
def test_companion_schema_and_actual_payload_identity_are_strict(tmp_path,schema,rehash):
 m=api();side=m.build_reference_manifest(SPLIT,CLASSES);side['schema']=schema
 if rehash:
  side['identity']=m.canonical_hash({k:v for k,v in side.items() if k!='identity'})
 path=write(tmp_path/'malformed.json',side)
 with pytest.raises(m.EndpointReferenceError):m.load_reference_manifest(path,SPLIT,CLASSES)

def api():return importlib.import_module('force_consistency.data.endpoint_references')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');return p

def real_inputs():return json.loads(SPLIT.read_text()),json.loads(CLASSES.read_text())

@pytest.fixture
def inputs(tmp_path):
 m=api();s,e=real_inputs();key='aspirin/12/plus'
 # A compact source fixture with an unchanged complete plus/minus connection.
 s['connections']['aspirin']={'12':s['connections']['aspirin']['12']}
 s['groups']={'aspirin':{'train':['12'],'validation':[],'test':[]}}
 s.pop('identity');s['identity']=m.canonical_hash(s)
 e={k:v for k,v in e.items() if k.startswith('aspirin/12/')}
 return write(tmp_path/'split.json',s),write(tmp_path/'classes.json',e)

def test_all_74_real_targets_and_known_filename_counterexamples(tmp_path):
 m=api();before=(SPLIT.read_bytes(),CLASSES.read_bytes());side=m.build_reference_manifest(SPLIT,CLASSES)
 path=write(tmp_path/'references.json',side);verified=m.load_reference_manifest(path,SPLIT,CLASSES)
 _,classification=real_inputs();assert len(verified.references)==74
 for key,target in verified.references.items():assert target==classification[key]['reference_class']
 expected={'aspirin/12/plus':'2','aspirin/12/minus':'1','aspirin/44/minus':'10','aspirin/66/plus':'9','aspirin/66/minus':'6'}
 for key,number in expected.items():assert m.reference_for_branch(verified,key)==f'Landscape17/aspirin/Conformations/coordsmin_{number}.xyz'
 for group in ('00','11','22','55','77','88'):assert verified.references[f'aspirin/{group}/plus']==verified.references[f'aspirin/{group}/minus']
 assert side['filename_minimum_pairs']['aspirin/66']==['6','6']
 assert side['source_sha256']=={'split_manifest':digest(SPLIT),'endpoint_classification':digest(CLASSES)}
 assert (SPLIT.read_bytes(),CLASSES.read_bytes())==before
 assert not hasattr(verified,'endpoint_pair')
 with pytest.raises(m.EndpointReferenceError):m.reference_for_branch(verified,'aspirin/unknown/plus')
 with pytest.raises(TypeError):verified.references['aspirin/12/plus']='wrong'

@pytest.mark.parametrize('mutation',['missing','unmatched','ambiguous','accepted_wrong','wrong_molecule','not_minimum','terminal','incomplete','unresolved','representative_wrong_molecule','split_identity','branch_missing'])
def test_invalid_source_never_falls_back_to_filename_pair(inputs,mutation):
 m=api();sp,ep=inputs;s=json.loads(sp.read_text());e=json.loads(ep.read_text());key='aspirin/12/plus'
 if mutation=='missing':del e[key]
 elif mutation=='unmatched':e[key]['status']='unmatched'
 elif mutation=='ambiguous':e[key]['accepted_classes'].append(e['aspirin/12/minus']['reference_class'])
 elif mutation=='accepted_wrong':e[key]['accepted_classes']=[e['aspirin/12/minus']['reference_class']]
 elif mutation=='wrong_molecule':e[key]['reference_class']='Landscape17/paracetamol/Conformations/coordsmin_2.xyz'
 elif mutation=='not_minimum':e[key]['reference_class']='Landscape17/aspirin/Conformations/coordsts_1_2.xyz'
 elif mutation=='terminal':e[key]['record_id']=s['connections']['aspirin']['12']['branches']['plus'][0]
 elif mutation=='incomplete':e[key]['incomplete']=['uncertified']
 elif mutation=='unresolved':e[key]['reference_unresolved']=['uncertified']
 elif mutation=='representative_wrong_molecule':e[key]['reference_record']='Landscape17/paracetamol/Conformations/coordsmin_2.xyz'
 elif mutation=='split_identity':s['identity']='0'*64
 else:
  del s['connections']['aspirin']['12']['branches']['minus'];s.pop('identity');s['identity']=m.canonical_hash(s)
 write(sp,s);write(ep,e)
 with pytest.raises(m.EndpointReferenceError):m.build_reference_manifest(sp,ep)

def test_distinct_matching_representative_keeps_canonical_target(inputs,tmp_path):
 m=api();sp,ep=inputs;e=json.loads(ep.read_text());key='aspirin/12/plus';e[key]['reference_record']='Landscape17/aspirin/Conformations/coordsmin_10.xyz';write(ep,e)
 # Supplied classification certifies a canonical class and an equivalent representative.
 # This pure interface copies that metadata; it does not recompute equivalence.
 side=m.build_reference_manifest(sp,ep);p=write(tmp_path/'side.json',side);v=m.load_reference_manifest(p,sp,ep)
 assert m.reference_for_branch(v,key).endswith('coordsmin_2.xyz')
 assert v.representatives[key].endswith('coordsmin_10.xyz')

@pytest.mark.parametrize('mutation',['wrong_target_rehashed','stale_identity','wrong_split_hash','wrong_class_hash','missing_branch','extra_branch','raw_dictionary'])
def test_sidecar_cannot_override_classified_source(inputs,tmp_path,mutation):
 m=api();sp,ep=inputs;side=m.build_reference_manifest(sp,ep);key='aspirin/12/plus'
 if mutation=='raw_dictionary':
  with pytest.raises(m.EndpointReferenceError):m.reference_for_branch(side,key)
  return
 if mutation=='wrong_target_rehashed':side['branch_references'][key]['reference_class']=side['branch_references']['aspirin/12/minus']['reference_class']
 elif mutation=='stale_identity':side['identity']='0'*64
 elif mutation=='wrong_split_hash':side['source_sha256']['split_manifest']='0'*64
 elif mutation=='wrong_class_hash':side['source_sha256']['endpoint_classification']='0'*64
 elif mutation=='missing_branch':del side['branch_references'][key]
 else:side['branch_references']['aspirin/fake/plus']=copy.deepcopy(side['branch_references'][key])
 if mutation!='stale_identity':side.pop('identity');side['identity']=m.canonical_hash(side)
 p=write(tmp_path/'bad.json',side)
 with pytest.raises(m.EndpointReferenceError):m.load_reference_manifest(p,sp,ep)

def test_source_change_is_detected_after_companion_was_frozen(inputs,tmp_path):
 m=api();sp,ep=inputs;p=write(tmp_path/'side.json',m.build_reference_manifest(sp,ep));e=json.loads(ep.read_text())
 e['aspirin/12/plus']['reference_record']='Landscape17/aspirin/Conformations/coordsmin_10.xyz';write(ep,e)
 with pytest.raises(m.EndpointReferenceError):m.load_reference_manifest(p,sp,ep)

def test_duplicate_json_key_is_not_a_valid_source_or_sidecar(inputs,tmp_path):
 m=api();sp,ep=inputs;p=tmp_path/'duplicate.json';p.write_text('{"schema":1,"schema":1}')
 with pytest.raises(m.EndpointReferenceError):m.load_reference_manifest(p,sp,ep)
