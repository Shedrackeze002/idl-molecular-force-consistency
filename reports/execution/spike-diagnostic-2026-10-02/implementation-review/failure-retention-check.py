"""Engineering-only fault injection: no model construction or model calls."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import numpy as np
from force_consistency import coverage_runs as cov, metrics
spec=importlib.util.spec_from_file_location('spike_review','scripts/diagnose_capacity_spikes.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=Path('reports/execution/spike-diagnostic-2026-10-02/implementation-review')
cov._predict=lambda model: None
metrics.partition_batch=lambda part, ix: {'indices':ix}
results={}
for mode,n in [('second_call_failure',32),('history_mismatch',7)]:
    root=base/mode;root.mkdir(exist_ok=False)
    ids=tuple(str(i) for i in range(n))
    part=SimpleNamespace(record_ids=ids,positions=np.zeros((n,1,3)),forces=np.zeros((n,1,3)),energy_residual=np.zeros(n),memberships={i:{'groups':['a']} for i in ids})
    class StubJournal:
        def __init__(self): self.root=root;self.calls=0
        def call(self,kind,count,fn,params,batch):
            self.calls+=1
            if mode=='second_call_failure' and self.calls==2: raise TimeoutError('injected second-call cap')
            return np.zeros(count),np.zeros((count,1,3))
    journal=StubJournal()
    old=cov._group_score
    cov._group_score=lambda *a: {'actual':0.}
    try:
        m.analyze_case({'params':{},'optimizer':{},'update':3600},{'fitting':part},None,None,{}, {'metrics':{'fitting':{'expected':1.}}},journal,lambda d:m.write(root/'partial.json',d))
    except Exception as e:
        results[mode]={'exception':str(e),'attempted_stub_calls':journal.calls,'predictions_retained':(root/'predictions.npz').exists(),'state_audit_retained':(root/'state-audit.json').exists(),'files':sorted(p.name for p in root.iterdir())}
    finally: cov._group_score=old
root=base/'interrupted_report';root.mkdir(exist_ok=False)
(root/'config.json').write_text(json.dumps({'cases':[{'name':'a'}]}))
try: m.render(root,base/'interrupted_report_output')
except Exception as e: results['interrupted_report']={'exception':f'{type(e).__name__}: {e}'}
(base/'failure-retention-results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
