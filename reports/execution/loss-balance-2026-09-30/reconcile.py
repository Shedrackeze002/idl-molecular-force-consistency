"""Independent saved-artifact reconciliation; reads no validation/test data."""
import hashlib,json
from pathlib import Path
import numpy as np
from flax import serialization

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=read('configs/aspirin-loss-balance-diagnostic.json');root=Path(c['run_directory']);result=read(root/'summary.json')
assert read(root/'config.json')==c
assert all(sha(Path(c['source_run'])/p)==h for p,h in c['source_sha256'].items())
assert sha(c['calibration_path'])==c['calibration_sha256']
calib=read(c['calibration_path'])
assert all(sha(Path(c['calibration_run'])/p)==h for p,h in calib['sources'].items())
source=read(Path(c['source_run'])/'diverse128/summary.json');scheduled=sorted([0,*range(100,2001,100),250])
checks={'coverage_sources':len(c['source_sha256']),'calibration_raw':len(calib['sources']),'calibration_compact':1,'arms':{}}
for name,a in result['arms'].items():
    assert a==read(root/name/'summary.json')
    checks['arms'][name]={'status':a['status'],'executed':a.get('executed',False),'updates':a.get('updates',0)}
    if not a.get('execution_valid'):continue
    armc=read(root/name/'config.json');prov=read(root/name/'provenance.json');verification=read(root/name/'checkpoint-verification.json')
    assert armc['loss_weights']==prov['loss_weights']==c['arms'][name]
    assert a['initial_state_hashes']==prov['expected_initial_hashes']==source['initial_state_hashes']
    assert a['updates']==2000 and a['frame_exposures']==256000 and a['warmup_frame_exposures']==128
    assert a['elapsed_seconds']<1800 and a['checkpoint_new_process_verified'] and verification['status']=='passed'
    history=read(root/name/'history.json');assert history==a['history'] and [h['update'] for h in history]==scheduled
    envelope=serialization.msgpack_restore((root/name/'last.msgpack').read_bytes())
    assert hashlib.sha256(envelope['payload']).hexdigest()==envelope['sha256']
    checkpoint=serialization.msgpack_restore(envelope['payload'])
    assert int(checkpoint['update'])==int(checkpoint['optimizer']['0']['count'])==2000
    assert a['fitting_frame_count']==128 and verification['anchor_record_count']==16
    for role,gate in [('fitting','fit_gate'),('anchor','anchor_gate')]:
        assert a['initial_metrics'][role]['record_ids']==source['initial_metrics'][role]['record_ids']
        for phase in ('initial_metrics','final_metrics'):
            assert a[phase][role]['record_count']==(128 if role=='fitting' else 16)
            assert a[phase][role]['force_component_count']==(8064 if role=='fitting' else 1008)
        assert a['final_metrics'][role]==history[-1]['metrics'][role]
        recomputed=[]
        for key in ('energy_mse_eV2','force_mse_eV2_per_A2'):
            initial=a['initial_metrics'][role][key];assert np.isclose(initial,source['initial_metrics'][role][key],atol=1e-10,rtol=1e-8)
            fraction=a['final_metrics'][role][key]/initial
            assert np.isclose(fraction,a[gate]['fraction_remaining'][key],atol=1e-14,rtol=1e-12)
            recomputed.append(fraction<.1)
        assert a[gate]['accepted']==all(recomputed)
    assert all(a['final_metrics']['anchor_fitted_flags'].values())
    assert a['final_metrics']['anchor_not_fitted']['record_count']==0
    checks['arms'][name].update(history_points=len(history),fresh_process=verification['status'],elapsed_seconds=a['elapsed_seconds'])
if result['execution_valid']:
    assert all(a['execution_valid'] for a in result['arms'].values()) and result['replay']['status']=='passed'
    replay=result['arms']['replay']
    for h,s in zip(replay['history'],source['history'],strict=True):
        for role in ('fitting','anchor'):
            for key in ('energy_mse_eV2','force_mse_eV2_per_A2'):
                assert np.isclose(h['metrics'][role][key],s['metrics'][role][key],atol=1e-10,rtol=1e-8)
    b=result['arms']['balanced'];f=result['arms']['force-only']
    assert result['balanced_joint_accepted']==(b['fit_gate']['accepted'] and b['anchor_gate']['accepted'])
    assert result['force_only_force_accepted']==all(f[g]['fraction_remaining']['force_mse_eV2_per_A2']<.1 for g in ('fit_gate','anchor_gate'))
checks['matrix_execution_valid']=result['execution_valid'];checks['summary_sha256']=sha(root/'summary.json')
Path('.superpowers/sdd/2026-09-30-loss-balance-diagnostic/reconciliation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
