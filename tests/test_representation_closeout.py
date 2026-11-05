"""Independent saved-data reductions and refusal boundaries; no models."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import msgpack
import numpy as np
import pytest

from close_representation_study import (
    BoundInputs, checkpoint_metadata, history_summary, paired_summaries,
    reduce_predictions, timing_summary, write_artifacts,
    verify_score_identity, verify_recipe, canonical, SPLIT,
)


def history():
    return [{'update': u, 'equal_connection_force_mae_eV_per_A': 1 + u/1e6}
            for u in range(0, 200001, 500)]


def test_schedule_first_tie_and_exact_twenty_check_window():
    rows = history()
    rows[4]['equal_connection_force_mae_eV_per_A'] = 1.
    result = history_summary(rows)
    assert result['best_update'] == 0
    assert result['late20_updates'] == list(range(190500, 200001, 500))
    assert result['late20'] == pytest.approx(np.mean([r['equal_connection_force_mae_eV_per_A'] for r in rows[-20:]]))
    assert result['last'] == 1.2


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'future', 'boolean', 'nan', 'negative'])
def test_invalid_histories_refuse(change):
    rows = history()
    if change == 'missing': rows.pop(20)
    if change == 'duplicate': rows[1]['update'] = 0
    if change == 'future': rows[-1]['update'] = 200500
    if change == 'boolean': rows[0]['equal_connection_force_mae_eV_per_A'] = True
    if change == 'nan': rows[0]['equal_connection_force_mae_eV_per_A'] = float('nan')
    if change == 'negative': rows[0]['equal_connection_force_mae_eV_per_A'] = -1
    with pytest.raises(ValueError): history_summary(rows)


def labels():
    return {'record_ids': ['a', 'b', 'c'], 'groups': ['18', '18_2', '18_2'],
            'energy_residual': np.zeros(3), 'forces': np.zeros((3, 1, 3))}


def test_independent_id_join_unequal_group_mae_and_component_rmse():
    # Group18 has one zero-error record; group18_2 has two error3 records.
    forces = np.array([[[3., 3, 3]], [[0., 0, 0]], [[3., 3, 3]]])
    result = reduce_predictions(['c', 'a', 'b'], ['18_2', '18', '18_2'],
                                np.array([4., 0, 2]), forces, labels())
    assert result['per_connection_force_mae_eV_per_A'] == {'18': 0., '18_2': 3.}
    assert result['equal_connection_force_mae_eV_per_A'] == 1.5
    assert result['pooled_force_mae_eV_per_A'] == 2.
    assert result['pooled_component_force_rmse_eV_per_A'] == pytest.approx(np.sqrt(6))
    assert result['energy_mae_eV'] == 2.
    assert result['record_count'] == 3 and result['connection_count'] == 2


@pytest.mark.parametrize('ids,groups', [(['a','a','c'], ['18','18_2','18_2']),
    (['a','b','missing'], ['18','18_2','18_2']),
    (['a','b','c'], ['18','18','18'])])
def test_bad_prediction_support_or_group_join_refuses(ids, groups):
    with pytest.raises(ValueError):
        reduce_predictions(ids, groups, np.zeros(3), np.zeros((3,1,3)), labels())


def test_nonfinite_prediction_and_wrong_shape_refuse():
    for f in [np.full((3,1,3), np.nan), np.zeros((3,3))]:
        with pytest.raises(ValueError):
            reduce_predictions(['a','b','c'], ['18','18_2','18_2'], np.zeros(3), f, labels())


def test_pairing_keeps_both_seeds_and_sign_reversal_visible():
    rows=[]
    for family, scores in [('E64',(.03,.04,.05)),('SP',(.04,.06,.04)),('S64',(.035,.045,.055))]:
        for seed in (0,1):
            rows.append({'candidate':f'{family}-{seed}', 'seed':seed,
                         **dict(zip(('best','last','late20'),scores))})
    result=paired_summaries(rows)
    assert result['SP']['best']['seed_deltas']==pytest.approx([.01,.01])
    assert result['SP']['late20']['mean_delta']==pytest.approx(-.01)
    assert result['SP']['interpretation']=='mixed'
    assert result['S64']['interpretation']=='descriptive fixed-recipe comparison'
    with pytest.raises(ValueError): paired_summaries(rows[:-1])
    with pytest.raises(ValueError): paired_summaries(rows+[rows[0]])


def test_timing_never_turns_missing_or_bad_values_into_zero():
    assert timing_summary([{'timing_status':'unknown','elapsed_seconds':None}])['elapsed_seconds'] is None
    assert timing_summary([{'timing_status':'recorded','elapsed_seconds':2.},
                           {'timing_status':'recorded','elapsed_seconds':3.}])['elapsed_seconds']==5.
    assert timing_summary([{'timing_status':'recorded','elapsed_seconds':float('nan')}])['elapsed_seconds'] is None


@pytest.mark.parametrize('field,value', [('seed',1),('parameter_count',1),
    ('source_fingerprint','different'),('model',{'max_degree':0}),('analysis_source_sha256','different'),
    ('backend','cuda'),('training_revision','different')])
def test_score_identity_cannot_override_the_admitted_recipe(field,value):
    config={'seed':0,'model':{'max_degree':1},'backend':'cpu'}
    provenance={'parameter_count':43521,'code_fingerprint':'source','revision':'original','backend':'cpu'}
    score={'seed':0,'model':config['model'],'parameter_count':43521,
           'source_fingerprint':'source','analysis_source_sha256':'scorer','backend':'cpu','training_revision':'original'}
    verify_score_identity(score,config,provenance,'scorer')
    score[field]=value
    with pytest.raises(ValueError): verify_score_identity(score,config,provenance,'scorer')


@pytest.mark.parametrize('field,value', [('mode','diagnostic'),('consistency_strength',1.),
                                        ('expected_learning_subset_hash','other')])
def test_self_consistent_config_cannot_change_fixed_recipe(field,value):
    config={'model':{'features':64,'max_degree':1,'iterations':2,'radial_basis':8,
                     'cutoff_A':5.,'energy_readout_init':'zero'},
        'seed':0,'arm':'energy','backend':'cpu','batch_size':4,'learning_rate':.003,
        'max_updates':200000,'validation_interval':500,'expected_split_hash':SPLIT,
        'loss_weights':{'energy':.014584365670390719,'force':1.9854156343296092},
        'mode':'integration','consistency_strength':0.,'expected_learning_subset_hash':None}
    config[field]=value
    computation={k:v for k,v in config.items() if k not in ('data_directory','run_directory','resume_checkpoint','max_updates','wall_seconds')}
    manifest={'identity':'dataset','normalization':{}}
    provenance={'config_computation_hash':canonical(computation),'dataset_identity':'dataset',
        'normalization':{},'split_hash':SPLIT,'parameter_count':43521}
    summary={'provenance':provenance,'status':'passed','stop_reason':'max_updates','updates':200000,
        'checkpoint_new_process_verified':True,'best_checkpoint_new_process_verified':True}
    with pytest.raises(ValueError):verify_recipe('E64-0',config,provenance,summary,manifest)


def test_bound_inputs_detect_corruption_and_never_open_unlisted_paths(tmp_path):
    repo=tmp_path/'repository';retained=tmp_path/'retained';repo.mkdir();retained.mkdir()
    path=retained/'item.json';path.write_bytes(b'original')
    inputs=BoundInputs(repo,retained,{'files':{'retained:item.json':{
        'sha256':hashlib.sha256(b'original').hexdigest(),'bytes':8}}})
    assert inputs.read('retained:item.json')==b'original'
    with pytest.raises(ValueError): inputs.read('retained:test.npz')
    path.write_bytes(b'edited!!')
    with pytest.raises(ValueError,match='identity'): inputs.read('retained:item.json')


def envelope(payload=b'opaque model bytes', control=None):
    control = control or {'validation_history': history(), 'best_update': 0,
                          'best_score': 1., 'best_state': b'selected opaque bytes'}
    encoded=msgpack.packb(control,use_bin_type=True)
    return msgpack.packb({'payload':payload,'sha256':hashlib.sha256(payload).hexdigest(),
        'control_payload':encoded,'control_sha256':hashlib.sha256(encoded).hexdigest(),
        'provenance':json.dumps({'fixed':'identity'},sort_keys=True,separators=(',',':'))},use_bin_type=True)


def test_checkpoint_metadata_preserves_opaque_payload_and_authenticates_control():
    result=checkpoint_metadata(envelope(),{'fixed':'identity'})
    assert result['payload']==b'opaque model bytes'
    assert len(result['control']['validation_history'])==401
    wrong=msgpack.unpackb(envelope(),raw=False);wrong['control_payload']+=b'x'
    with pytest.raises(ValueError): checkpoint_metadata(msgpack.packb(wrong,use_bin_type=True),{'fixed':'identity'})
    with pytest.raises(ValueError): checkpoint_metadata(envelope(),{'fixed':'different'})


def test_output_refusal_preserves_existing_evidence(tmp_path):
    (tmp_path/'keep.txt').write_text('original')
    with pytest.raises((ValueError,FileExistsError)): write_artifacts({},tmp_path)
    assert (tmp_path/'keep.txt').read_text()=='original'


def test_auditor_import_cannot_reach_model_frameworks():
    module=Path(__file__).resolve().parents[1]/'scripts/close_representation_study.py'
    code=("import importlib.util,sys; "
          f"s=importlib.util.spec_from_file_location('closeout_guard',{str(module)!r}); "
          "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
          "assert not any(k.split('.')[0] in {'jax','torch','flax','force_consistency'} for k in sys.modules)")
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
