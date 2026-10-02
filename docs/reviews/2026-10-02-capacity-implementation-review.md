# Capacity preproduction implementation review

Date: 2026-10-02. Independent read-only implementation reviewer. **Initial verdict: NEEDS REPAIR; production not cleared.** Three material findings were reported promptly to the author. Author acceptance is recorded below; a separate follow-up must verify the repairs before production.

## Scope and evidence

Read `AGENTS.md`, the exact frozen configuration, amended capacity design/plan, both cleared independent plan follow-ups, all three new capacity modules and all three focused test files. Inspected the reused `DirectoryLease`, code provenance capture, input loader, actual independent boundary helper and immutable checkpoint publication helper. No implementation or production file was edited. No training, model initialization, prediction or test suite was run by this reviewer.

The author was concurrently completing focused checks; this review does not certify an unfinished full-suite result. Source line references below identify the version inspected before author repairs and should be interpreted with the named functions if later lines move. User PDFs and all prior review records were untouched.

## Material findings

### CAPACITY-IMPL-P2-001 — cap-invalidated reports do not canonically regenerate

`capacity_report.py:292-314`, `finalize`, inserts `Global cap exceeded during finalization` into the saved report when substantive persistence/marker exceeds the cap. `capacity_report.py:199-204` and `241-243`, `_closure_errors` / `_build_report`, instead produce `Finalization: Final report/evidence/marker cap closure invalid` on saved regeneration. These reports have different verification-error lists and therefore different JSON/Markdown bytes. The existing closure test at `tests/test_capacity_report.py:110-137` checks exact regeneration only for successful closure; its overrun branch checks contrast invalidation but omits the canonical equality assertion.

The cap is correctly disqualifying the scientific comparison, but the promised saved-only, byte-identical closeout fails precisely for a preserved unsuccessful trial. Use one deterministic closed-status representation/error across initial finalization and saved reconstruction, while distinguishing real hash corruption from an honestly recorded cap failure. Add RED/GREEN assertions of report equality and byte equality after both substantive-write and post-marker overrun.

Author response: accepted; independently identified the mismatch and began a RED assertion. No repaired behavior was certified in this initial review.

### CAPACITY-IMPL-P2-002 — root-retained returned work is discarded when local summary persistence fails

`capacity_runs.py:204-210`, `_run_owned_case`, returns known computed/attempted counts and any already completed child work when case-directory ownership is lost. The controller preserves the returned result in root `cases.json` at `capacity_report.py:354` and retains it in the journal/evidence. However, `_build_report` at `capacity_report.py:231-234` constructs its partial case solely from local `summary.json`; when this file is absent, `_partial_case` at `177-195` converts the known returned counts to unknown and loses the root-retained child record. The root record is not checked against local summaries for complete cases either.

The balanced research trail must retain controller-reported computation separately from independently verified local receipts, including known completed child work when local ownership/publication fails. Bind complete root case records to actual local summaries. For partial cases, expose the intact controller-reported result even if local files are unavailable; do not promote it to receipt verification. Add a missing/replaced-local-directory or failed-local-publication test with known root-retained counters and child work.

Author response: accepted; plans to bind complete root cases and retain controller-returned reported work separately from receipt-verified counts.

### CAPACITY-IMPL-P2-003 — supplied-key-only code validation accepts omitted participants and dependencies

`capacity_contract.py:38-47`, `verify_code`, checks only caller-supplied code files and dependencies. A pure reproduction successfully validated `code_files={'uv.lock': <actual hash>}`, `dependencies={}`, a matching fingerprint, current Python and revision `'0'*40`. Every implementation file and numerical dependency identity was omitted, while the public guard/report would regard this record as verified. The actual capture normally produces a full record, but a mutually consistent local metadata rewrite can weaken this boundary.

Reproduction is preserved in `reports/execution/capacity-2026-10-02/implementation-review-incomplete-code-guard.json`. It imported neither JAX nor NumPy and performed no numerical work. Require a complete authoritative participant set (or independently enumerated exact source set), mandatory runtime dependency keys and a genuine resolvable launch commit. A later documentation commit may legitimately change live HEAD; do not reject an honest archived launch revision merely because current HEAD differs. Add omission/nonexistent-revision RED/GREEN guard tests before numerical imports.

Author was notified promptly. Resolution/follow-up pending in this initial artifact.

## Partial-accounting advisory

`capacity_runs.py:244` writes `probe-phase.json` after the child template successfully initializes, retaining `known_template_initializations=1` and initializer frames before endpoint prediction starts. `_partial_case` reads only parent `phase.json` and exposes verified `child_work` only after full child completion. For interrupted/failed predictions, also retain this child phase and known template work separately from unknown remaining prediction work. Raw evidence exists but is currently absent from the report. This supplements the material completed-child/root-record finding.

## Controls that were correctly implemented in the inspected version

- Public frozen configuration and exact case membership; source pins and source-derived norm/support/baselines bound before public child numerical imports.
- Exclusive attempt reservation and child entry, unpredictable launch token, bound directory identity, all-component symlink rejection and fd-backed entry writes. The public CLI exposes no fixture bypass; private tiny tests remain private.
- Separate fresh width-specific initialization, zero moments/readout baseline verification, immutable saved0 blueprint and finite/nonnegative serialized state checks. The child initializes an independent declared template before terminal restoration.
- Actual accumulated training proposals are checked against independently computed full-gradient/NumPy proposals at update1 and the specified nonzero-moment boundary; discarded oracle work is separate from committed updates.
- Exact score sequence, terminal count/state/checkpoint/child binding, direct raw endpoint metric reconstruction from saved arrays, fixed raw gates and width64-minus-width32 / width64-divided-by-width32 orientation.
- A valid numerical execution may fail learning. Fitted-anchor, sparse-score, one-seed, initialization-shape/optimization and pending-main-pilot limitations are explicit. No result has been claimed for the unexecuted capacity production run.

These positive observations do not clear the three material findings or replace execution tests.

## Preserved review inspection trials

1. Batched source/config/review reads exceeded the tool output budget; focused line-range reads recovered the complete relevant source. No numerical conclusion was inferred from truncated output.
2. A broad temporary-directory inventory encountered macOS-protected subdirectories and truncated diagnostic output; a visible pytest capacity fixture was located. This was unnecessary exploration, did not mutate files and did not establish any missing artifact claim. No further broad inventory was used.
3. Pure incomplete-code guard reproduction succeeded unexpectedly; saved as the JSON trial above. It establishes the validation omission, not a scientific/model outcome.

Preserve this initial artifact and all trials. Follow-up should name each finding, inspect its repair and meaningful test evidence, and state a fresh verdict. Production remains gated on material resolution and the author's full-suite verification.
