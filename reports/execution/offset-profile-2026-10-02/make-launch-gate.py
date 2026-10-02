"""Build concrete immutable launch gate after approved implementation/test identities."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path('scripts').absolute()))
from train_offset_profile import cc,verify_inputs
from force_consistency.runs import code_provenance
records=Path(__file__).parent;focused=sys.argv[1];suite=sys.argv[2];review=sys.argv[3]
c=cc.read('configs/aspirin-offset-profile.json');source=verify_inputs(c);code=code_provenance(Path.cwd());cc.verify_code(code)
for n in (focused,suite):
    last=Path(n).read_text().strip().splitlines()[-1]
    if ' passed' not in last or 'failed' in last or 'error' in last.lower():raise ValueError('Verification log not passed: '+n)
files=['scripts/offset_profile_core.py','scripts/train_offset_profile.py','scripts/run_offset_profile.py','scripts/report_offset_profile.py',
 'tests/test_offset_profile_core.py','tests/test_offset_profile_runs.py','tests/test_offset_profile_report.py','configs/aspirin-offset-profile.json',
 c['source_pins'],c['plan_gate'],'docs/superpowers/specs/2026-10-02-offset-profile-design.md','docs/superpowers/plans/2026-10-02-offset-profile.md',
 'docs/reviews/2026-10-02-offset-profile-scientific-plan-followup.md','docs/reviews/2026-10-02-offset-profile-engineering-plan-followup.md',
 focused,suite,review]
followup=records/'implementation-review-resolution.json'
if followup.exists():files.append(str(followup))
gate={'status':'cleared','launch_revision':code['revision'],'code_fingerprint':code['code_fingerprint'],
 'code':code,'config_identity':cc.canonical_hash(c),'source_pin_count':len(source['pins']),
 'artifacts':{n:cc.digest(cc.no_symlinks(n)) for n in files},
 'plan_reviews':'two_final_PASS','implementation_review':'PASS_resolved','focused':Path(focused).read_text().strip().splitlines()[-1],
 'suite':Path(suite).read_text().strip().splitlines()[-1],'run_ordinal':1,'research_trial':'not_started_at_gate_creation'}
with (records/'prelaunch-gate.json').open('x') as f:json.dump(gate,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:gate[k] for k in ('status','launch_revision','focused','suite','source_pin_count')},indent=2))
