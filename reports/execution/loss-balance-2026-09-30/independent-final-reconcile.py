import json,hashlib
from pathlib import Path
R=Path('runs/aspirin-loss-balance-2026-09-30')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=read('configs/aspirin-loss-balance-diagnostic.json');s=read(R/'summary.json');report=read('reports/loss-balance-diagnostics.json')
assert read(R/'config.json')==c
assert len(c['source_sha256'])==31
for f,h in c['source_sha256'].items():assert sha(Path(c['source_run'])/f)==h,f
assert sha(c['calibration_path'])==c['calibration_sha256']
cal=read(c['calibration_path']);assert len(cal['sources'])==38
for f,h in cal['sources'].items():assert sha(Path(c['calibration_run'])/f)==h,f
assert report['source_summary_sha256']==sha(R/'summary.json')
assert report['report_generator_sha256']==sha('scripts/report_loss_balance.py')
for f,h in report['artifacts_sha256'].items():assert sha(R/f)==h,f
for f,h in report['figures_sha256'].items():assert sha(Path('reports')/f)==h,f
source=read(Path(c['source_run'])/'diverse128/summary.json');out={}
for n in ('replay','balanced','force-only'):
 a=s['arms'][n];assert a==read(R/n/'summary.json')
 assert report['arms'][n]=={k:v for k,v in a.items() if k!='provenance'}
 assert a['updates']==2000 and a['frame_exposures']==256000 and a['warmup_frame_exposures']==128
 assert a['execution_valid'] and a['checkpoint_new_process_verified'] and a['elapsed_seconds']<1800
 assert a['history']==read(R/n/'history.json')
 assert [r['update'] for r in a['history']]==sorted([0,250,*range(100,2001,100)])
 assert a['initial_state_hashes']==source['initial_state_hashes']
 ac=read(R/n/'config.json');p=read(R/n/'provenance.json');v=read(R/n/'checkpoint-verification.json')
 assert ac['loss_weights']==p['loss_weights']==c['arms'][n]
 assert v['status']=='passed' and v['update']==2000 and v['fitting_record_count']==128 and v['anchor_record_count']==16
 assert p['inputs']==read(Path(c['source_run'])/'diverse128/provenance.json')['inputs']
 for role,gate in [('fitting','fit_gate'),('anchor','anchor_gate')]:
  fractions=a[gate]['fraction_remaining']
  for key in ('energy_mse_eV2','force_mse_eV2_per_A2'):
   assert fractions[key]==a['final_metrics'][role][key]/source['initial_metrics'][role][key]
  assert a[gate]['accepted']==all(x<.1 for x in fractions.values())
 out[n]={k:a[k] for k in ('updates','frame_exposures','elapsed_seconds','fit_gate','anchor_gate')}
for a,b in zip(s['arms']['replay']['history'],source['history']):
 for role in ('fitting','anchor'):
  for key in ('energy_mse_eV2','force_mse_eV2_per_A2'):assert a['metrics'][role][key]==b['metrics'][role][key]
assert s['execution_valid'] and not s['balanced_joint_accepted'] and not s['force_only_force_accepted']
print(json.dumps({'result':'passed','source_hashes':31,'raw_calibration_hashes':38,'compact_hashes':1,'run_artifact_hashes':len(report['artifacts_sha256']),'replay_exact_comparisons':88,'arms':out},indent=2))
