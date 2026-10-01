from pathlib import Path
import hashlib,json,re
work=Path(__file__).parent;result=json.loads(Path('reports/width-rate-recovery-verification.json').read_text());invocation=json.loads((work/'production-invocation.json').read_text());summary=json.loads((Path(result['recovery_root'])/'summary.json').read_text()) if (Path(result['recovery_root'])/'summary.json').exists() else result
log=(work/'full-suite.txt').read_text();match=re.search(r'(\d+) passed in ([\d.]+)s',log);assert match and 'failed' not in log
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
child=result.get('child_receipt',{});t=json.loads((Path(result['recovery_root'])/'copy-transformation.json').read_text());code=result['code'];valid=result['verification_valid']
text=f'''# Verification-only copied-checkpoint execution record

The twice-reviewed amended plan ran natively on retained branch `codex/landscape17-baseline`, with one production invocation at `{invocation['revision']}`. Frozen recovery configuration canonical SHA256: `55c63996e555fad2c65edb38525dcd27f022188650ee255353b24e20a83c487c`; recovery manifest SHA256: `f7fff5b24a9e2eae365a97924d720d996d9c60321c39fc00711d307a03e5a4d4`. All163recovery pins and326historical pins were checked before execution and after the child; all145original raw files remain byte-identical.

**Separate copied verification: {result['status']}; validity: {valid}.** Exactly{result['probe_attempts']} fresh verifier attempt; zero training updates, optimizer proposals and warmup. Copied endpoint update/Adam count:{child.get('update')}/{child.get('adam_count')}; CPU/float64, prediction atol1e-10/rtol0. Maximum absolute reproduced prediction error:{child.get('maximum_absolute_error')}. Measured verifier duration:{result.get('probe_seconds')}s; controller duration:{result.get('elapsed_seconds')}s, including guards/copy/audit/terminal publication checks under180s. Completion marker duration is saved separately. Later reporting, evidence append and archiving are outside this cap. Parent UTC start/end:{result.get('started_at_utc')} / {result.get('ended_at_utc')}; monotonic clock determines durations.

New completed work: `{json.dumps(result['new_work'],sort_keys=True)}`. The verifier restores the original6000-update parameter/Adam state; its one template initializer consumes16anchor frames and its saved-endpoint predictions consume128fitting+16anchor frames. This validates endpoint reproducibility under the pinned implementation, not DFT accuracy, generalization or physical shape recovery.

## Exact transformation and immutable failed trial

Only embedded provenance serialization and the two copied checkpoint-hash bindings changed. Original source checkpoint SHA256: `{t['source_checkpoint_sha256']}`; separate copied checkpoint: `{t['copied_checkpoint_sha256']}`. Opaque numerical payload:`{t['payload_sha256']}`; opaque control payload:`{t['control_sha256']}`; history:`{t['source_history_sha256']}`. Source-derived full ten-file audit passed before launch and after completion. Semantic provenance, payload/control bytes, optimizer/parameter states, config/case/normalization, input/reference bytes and history remain unchanged. No original checkpoint/report was edited or re-probed.

Original matrix remains **failed**, both original strict joint learning gates failed, three cells blocked, no contrasts/winner/capacity/optimizer-cause inference, full pilot pending. Original fitting E/F fractions remain0.3220330430785254/0.45318885284127763; fully fitted anchor fractions0.10905757620503544/0.42746961258301264. Successful copied verification never retroactively promotes the original trial. Full original effort and all per-arm/shared/oracle receipts remain inherited in the separate JSON; unknown original child work/unavailable duration remain unknown. Combined accounting is incomplete; no efficiency claim.

## Review and engineering evidence

Two independent scientific/engineering reviews and material follow-up cleared the saved amendment before implementation. A fresh branch reviewer found five Important failure-path issues plus a required timestamp Minor, all accepted and repaired.17new RED regressions plus41existing passes established the five failures;58then passed. Additional known-work and truthful unknown-work checks gave60passes. Nonfinite malformed terminal JSON additionally reproduced RED; the first full suite was intentionally interrupted after358passes203.16s and retained as incomplete. Final focused61passed11.36s. **Final full suite:{match[1]}passed in{match[2]}s**, logSHA256`{h(work/'full-suite.txt')}`. Engineering fixtures are not research trials. [Final review and author rulings](../docs/reviews/2026-10-01-width-rate-recovery-final-review.md) retain all findings and the reviewer's deferred saved-result scope; fix correctness is author-verified, without a redundant code re-review or production retry.

## Preservation and next step

Protected recovery root:`runs/aspirin-width-rate-recovery-2026-10-01`; original root:`runs/aspirin-width-rate-2026-09-30`. All workspaces and synthetic reviewer repros retained; supplemental archive:`reports/execution/width-rate-recovery-2026-10-01` with ZIP CRC, exact entry sets and per-file SHA verification. Exact committed production code is saved alongside raw recovery output and all failed/incomplete engineering logs. All existing user files/changes remain untouched; no push, merge or cleanup.

Obsidian CLI remains unavailable (application not found/running; launch failures previously recorded). Pending navigation content stays in the repository workspace; vault has not been edited directly. Repository evidence remains numerical authority.

Next research milestone: a separately prospective replacement/completion width/rate matrix plan, independently reviewed by two subagents before numerical work. This phase performed zero new training and supplies no four-cell comparison or pilot clearance. A valid copy narrows the checkpoint failure to metadata serialization for this endpoint; learning failure remains unresolved.
'''
Path('reports/width-rate-recovery-execution-record.md').write_text(text)
print(json.dumps({'status':result['status'],'verification_valid':valid,'suite_count':int(match[1]),'suite_seconds':float(match[2]),'full_suite_sha256':h(work/'full-suite.txt')},indent=2))
