from pathlib import Path
import hashlib,json,shutil,subprocess
E=Path('reports/execution/width-rate-replacement-2026-10-01');failed=E/'failed-closure-check';failed.mkdir(exist_ok=True)
for name in ('log-completeness-input.json','log-completeness.json','log-completeness.md','coverage-check-stdout.txt','coverage-check-stderr.txt'):
 shutil.copyfile(E/name,failed/name)
old=json.loads((E/'results-commit.json').read_text());full=subprocess.check_output(['git','rev-parse','--verify',old['revision']+'^{commit}'],text=True).strip();assert len(full)==40
resolved={**old,'revision':full}
(E/'resolved-results-commit.json').write_text(json.dumps(resolved,indent=2,sort_keys=True)+'\n')
for path,h in resolved['files_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',full+':'+path])).hexdigest()==h
record={'schema':'replacement-closure-correction-v1','scope':E.name,'config_identity':old['config_identity'],'reason':'Author supplied abbreviated d08f340, while the fixed role validator requires40hex. Actual90committed scientific/runtime/test/review/archive/navigation files were already verified; resolve the same unambiguous actual commit without altering any outcome.','initial_record_path':str(E/'results-commit.json'),'initial_record_sha256':hashlib.sha256((E/'results-commit.json').read_bytes()).hexdigest(),'resolved_record_path':str(E/'resolved-results-commit.json'),'resolved_record_sha256':hashlib.sha256((E/'resolved-results-commit.json').read_bytes()).hexdigest(),'resolved_revision':full,'failed_check_artifacts_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in failed.glob('*')},'resolution':'Initial record/evidence retained. Final closure additionally validates the resolved full record using the unchanged purpose/content validator and exact structured correction entry; only the demonstrated abbreviated-revision role failure may be resolved. All other checker errors stay blocking. Coverage stdout/stderr are classified with coverage outputs and excluded from their own required hash map; failed copies remain separately retained.'}
(E/'closure-correction.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
with Path('reports/PAPER_EVIDENCE_LOG.md').open('a') as f:
 f.write('\n## WIDTHRATE-REPLACEMENT-CLOSURE-CORRECTION-001 — preserve failed abbreviated-commit closure and resolve actual full snapshot\n\n```json\n'+json.dumps(record,indent=2,sort_keys=True)+'\n```\n')
 f.write('\n## WIDTHRATE-REPLACEMENT-RESULTS-COMMIT-002 — full actual revision for the same90verified files\n\n```json\n'+json.dumps(resolved,indent=2,sort_keys=True)+'\n```\n')
print(json.dumps({'resolved_revision':full,'committed_files':len(resolved['files_sha256'])}))
