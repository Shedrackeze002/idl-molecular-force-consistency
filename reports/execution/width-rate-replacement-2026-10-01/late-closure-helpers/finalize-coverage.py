"""Final acyclic evidence closure after saved-result review, archives, navigation and snapshot commit."""
from pathlib import Path
import hashlib,json,importlib.util,subprocess
E=Path('reports/execution/width-rate-replacement-2026-10-01');r=json.loads(Path('reports/width-rate-replacement-diagnostics.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('replacement_report','scripts/report_width_rate_replacement.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
inv=module.scoped_inventory(r);(E/'scope-inventory.json').write_text(json.dumps(inv,indent=2,sort_keys=True)+'\n')
required={}
for key in ('raw_files_sha256','source_pins','test_logs_sha256'):required.update(inv[key])
for a in json.loads(Path(inv['roles']['ancestry']).read_text())['artifacts']:required[a['path']]=a['sha256']
for p in inv['additional_required_paths']+list(inv['roles'].values()):required[p]=sha(p)
for p,h in json.loads((E/'production-code.json').read_text())['files_sha256'].items():required[p]=h
for p,h in json.loads((E/'test-results.json').read_text())['code_sha256'].items():required[p]=h
excluded_files={str(E/'log-completeness-input.json'),str(E/'log-completeness.json'),str(E/'log-completeness.md'),str(E/'coverage-check-stdout.txt'),str(E/'coverage-check-stderr.txt')}
for p in E.rglob('*'):
 if p.is_file() and str(p) not in excluded_files:required[str(p)]=sha(p)
for p in Path('.superpowers/sdd/2026-10-01-width-rate-replacement').glob('*.py'):required[str(p)]=sha(p)
evidence=Path('reports/PAPER_EVIDENCE_LOG.md').read_bytes()
inputs={'stage':'final','evidence_prefix_bytes':len(evidence),'evidence_prefix_sha256':hashlib.sha256(evidence).hexdigest(),'required_artifacts':required,'required_roles':inv['roles'],'case_evidence':{name:'WIDTHRATE-REPLACEMENT-OUTCOME-'+name+'-001' for name in r['arms']},'exclusions':['coverage_outputs','coverage_manifest','evidence_after_cutoff','closing_commit_sha']}
(E/'log-completeness-input.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n')
result=module.reconcile_evidence(r,'reports/PAPER_EVIDENCE_LOG.md',inputs)
result['unmodified_core_checker_errors']=list(result['errors'])
# Explicit append-only resolution of the sole observed short-revision failure.
try:
 initial=json.loads((E/'results-commit.json').read_text());resolved_path=E/'resolved-results-commit.json';resolved=json.loads(resolved_path.read_text())
 correction=json.loads((E/'closure-correction.json').read_text())
 full=subprocess.check_output(['git','rev-parse','--verify',initial['revision']+'^{commit}'],text=True).strip()
 assert resolved=={**initial,'revision':full} and len(full)==40
 assert correction['initial_record_sha256']==sha(E/'results-commit.json') and correction['resolved_record_sha256']==sha(resolved_path)
 assert required.get(str(resolved_path))==sha(resolved_path)
 assert module.structured_evidence(evidence.decode(),'WIDTHRATE-REPLACEMENT-RESULTS-COMMIT-002')==resolved
 assert module.structured_evidence(evidence.decode(),'WIDTHRATE-REPLACEMENT-CLOSURE-CORRECTION-001')==correction
 assert module.artifact_role_valid('results_commit',str(resolved_path),r,required)
 assert result['errors']==['Required artifact purpose/content invalid: results_commit']
 result['errors']=[];result['log_completeness']=True;result['full_results_commit_resolution_verified']=True
 result['scope']+=' The retained abbreviated snapshot reference is resolved via a separately validated full actual revision and structured correction; unchanged core checker errors are exposed separately.'
except (OSError,ValueError,KeyError,AssertionError,subprocess.SubprocessError) as exc:
 result['log_completeness']=False;result['full_results_commit_resolution_verified']=False;result['errors'].append('Full commit resolution failed: '+repr(exc))

# The preserved first report and its fixed index remain immutable. Reconcile the
# appended corrected current summary separately rather than rewriting history.
try:
 correction=json.loads((E/'report-invocation-correction.json').read_text());current_path=Path(correction['corrected_report_path']);current=json.loads(current_path.read_text())
 index=module.outcome_index(current,current_path)
 assert correction['initial_report_sha256']==sha('reports/width-rate-replacement-diagnostics.json')
 assert correction['corrected_report_sha256']==sha(current_path)
 assert current['config']==r['config'] and current['arms']==r['arms'] and current['artifacts_sha256']==r['artifacts_sha256']
 assert Path(current['source_root']).resolve()==Path(r['source_root']).resolve()
 assert current['execution_valid'] and current['interpretation_valid'] and current['actual_counts']['work_counts_complete'] and not current['verification_errors']
 assert json.loads(Path(correction['corrected_outcome_index_path']).read_text())==index
 assert module.structured_evidence(evidence.decode(),'WIDTHRATE-REPLACEMENT-OUTCOME-INDEX-002')==index
 assert module.structured_evidence(evidence.decode(),'WIDTHRATE-REPLACEMENT-REPORT-CORRECTION-001')==correction
 for path in (current_path,Path(correction['corrected_outcome_index_path']),E/'report-invocation-correction.json'):
  assert required.get(str(path))==sha(path)
 result['corrected_current_summary_reconciled']=True
 result['current_scientific_summary']=str(current_path)
 result['scope']+=' The initial relative-path reporting failure is retained; appended corrected absolute-path current summary and index are separately reconciled against identical raw/arm records.'
except (OSError,ValueError,KeyError,AssertionError) as exc:
 result['log_completeness']=False;result['corrected_current_summary_reconciled']=False;result['errors'].append('Corrected report reconciliation failed: '+repr(exc))
(E/'log-completeness.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
(E/'log-completeness.md').write_text('# Replacement research-log completeness\n\n'+json.dumps(result,indent=2,sort_keys=True)+'\n\nThis certifies the named structured artifact trail at the byte-count/hash cutoff. It does not independently rerun or certify all historical scientific correctness. Later evidence and closing commit identity are explicitly excluded to avoid circular hashes.\n')
print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['log_completeness'] else 1)
