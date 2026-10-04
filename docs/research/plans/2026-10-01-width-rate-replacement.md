> Research protocol copy. Dates, settings and checkboxes describe the saved protocol, not current experiment completion or a new launch approval. See [the experiment index](../../../reports/PAPER_EVIDENCE_LOG.md) for outcomes. Historical launch records referenced below are retained separately.

# Fresh Replacement Width/Rate Implementation Plan


**Goal:** One isolated fresh four-cell diagnostic with unchanged arithmetic/controls and a balanced complete research trail.
**Architecture:** pure frozen new contract → guarded wrappers around existing preflight/private arm/controller and new public probe → explicit saved-only report/log reconciliation. Existing pinned modules unchanged.
**Tech Stack:** locked CPU/float64JAX/Flax/Optax/E3x; stdlibcontract/report/journal, existing plot helper; no dependency install.
**Spec:** docs/research/specs/2026-10-01-width-rate-replacement-design.md.

## Global Constraints

Configd0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36;518pin map557daf51680f48a9c852bde83a133ee6bac0796bb36103bc2ae22c833ca66cb6. Old326/recovery163guards, original145raw/recovery19raw immutable. Newroot runs/aspirin-width-rate-replacement-2026-10-01, rolefresh_four_cell_width_rate_diagnostic. One matrix invocation/one child per completecell. Scientific controls equal old16/32×.001/.0003/6000/128fit/16fittedanchor/seed0/CPUfloat64/weights/Adam/backbone/64scores/63replay/<.1gates; caps1800/1800/3600/3600/global11400. No graft/continuation/retry/heldout/pilot. Existing sources unchanged/no globals rebinding. Pure guards before arrays; fixture adapters never relax public contract. Full failed/partial/blocked receipts; local featurecheckout/userPDFs preserved, no push/merge/cleanup.

## Review Focus

1. New config entering old frozen validator/fake old identity — reject boundary, Task1.
2. Coherently changed source/case/width/input/reference/checkpoint/receipt/rootrace — source-derived no-child guards/ownership, Tasks1/2.
3. Deadline crossed after last check/failure/timeout/valid receipt thenIOfailure — zeroorone attempt/full logs/knownunknownwork, Task2.
4. Missing/corrupt/nonfinite terminal/fake failed counts/time/baseline rebase — pure incomplete report/no invention, Task3.
5. Omitted negative/blocked/test/audit/artifact evidence — explicit completeness failure, Task3.

### Task1: Frozen contract and guarded reusable preflight

**Files:** create src/force_consistency/width_rate_replacement_contract.py, tests/test_width_rate_replacement.py; begin src/force_consistency/width_rate_replacement.py.
**Interfaces:** validate_replacement(config:dict)->None; verify_replacement_sources(config:dict)->dict (original numerical facts plus new identity/separate ancestry); replacement_preflight(config,context,save_partial)->dict. Private injection adapters only for tiny tests.

- [ ] RED test_frozen_controls_and_extra_keys rejects rates/caps/weights/backend/root/extra keys before numeric sentinel; test_all518pins_and_oldcontracts rejects missing/coherently rebased pins; test_disjoint_existing_symlink_and_raced_root guards oldroot/copy/currentroot and safe failure persistence.
- [ ] Run .venv/bin/python -m pytest tests/test_width_rate_replacement.py -q; retain meaningful RED.
- [ ] Implement pure canonical/config/pin/path guards and original contract checks. All original scientifickeys equal except reviewed administrative fields. Return guarded original baselines/inputs/history and normalizednewsource/ancestry. Pass exact oldconfig to existing width_rate_preflight after newguard; no numeric import before pure success.
- [ ] GREEN import/array-load-denying subprocess; sentinel verifies exact originalconfig reachesoldpreflight. Commit isolated helpers/tests.

### Task2: Fresh matrix, strict child and honest journal

**Files:** extend src/force_consistency/width_rate_replacement.py and tests/test_width_rate_replacement.py.
**Interfaces:** run_replacement_arm(config,case,prepared,root,provenance,context)->dict; audit_replacement_cell(config,root)->dict(metadataonly); diagnose_width_rate_replacement(config)->dict; module --config PATH or --probe PATH --attempt-sha256 HASH; journal events(seq,phase,status,artifacts_sha256,known/unknownwork), append-only.

- [ ] RED nontrivialtiny200/1000provenance/freshrestore, pairedfreshness/firststeporacle, fourcase sequence learningfailcontinues/executionfailblocks, rootownership, committedprefix/persistencefailure, coherentcase/reference/sourcechange nochild, positive allowance beforeissue, full failed/timeoutlogs, knownchildafterlaterfailure, finalpublicationcap. Retain all RED attempts.
- [ ] Implement public actualarray/case/normalization/baseline/history validation then unchanged _run_arm. Only16/.001gets63historyoracle. Reuse initialize/preflight/controller without disguising identity; new guarded sources attach only after originalnumericfacts match. Exclusive root ownership/safe external failure records. New public child calls the SAME full audit_replacement_cell (518+old contracts/all actual metadata/input/envelope/reference/committed history guards) before numerical imports/arrays, then atomically claims exclusive probe-entry.json against parent probe-attempt.json SHA/ordinal1. Missing/mismatched reservation or repeated entry rejected; then unchanged _probe_saved.
- [ ] Metadata prelaunch audit bindsactual config/case/provenance canonicalization/model/norm/sourceinputs/commit/historyprefix/payload-control/reference/update6000. One finite-positive-allowance child; completeavailablelogs. Strict receipt/count/tolerance bindings. Add direct-public-entry import/array-denying coherent metadata mutations, missing reservation and repeated-entry RED tests. Known child receipt survives later failedexecution. UTC/monotonic/code/source/run/phasejournal beforeoperations; allterminal/blockedresults hash-linked. Parent/shared complete/unknown receipts retained.
- [ ] Focused GREEN/importguard; commit wrapper/probe/journal, no productionyet.

### Task3: Saved report, log completeness, once-only production, archive

**Files:** create scripts/report_width_rate_replacement.py, tests/test_width_rate_replacement_report.py; generate reports/width-rate-replacement-diagnostics.json/md, reports/figures/width-rate-replacement-curves.png, reports/width-rate-replacement-execution-record.md, reports/width-rate-replacement-log-completeness.json/md, docs/reviews/2026-10-01-width-rate-replacement-final-review.md, reports/execution/width-rate-replacement-2026-10-01; append evidence/navigation.
**Interfaces:** generate(root:Path,output:Path)->dict; reconcile_evidence(result:dict,evidence_path:Path,artifacts:dict)->dict. Explicit new config/source arm/global validator; only config-independent oldmetric/gate/accounting/contrast/figure helpers imported. No numerical/array reads in report.

- [ ] RED validsynthetic4cells, coherentbaseline/history/gate/config/child/state/code/oracle/shared/global edits; partial/missing/corrupt/nonfinite terminal, failed known/unknownwork, orphan/committed labels, incompletecontrasts, omittednegative/blocked/test/audit/artifactlink. All64raw/group/overlaprows and oldfailed/recoverypassedunknownaccountingretained. Import/np.load-denying reportsubprocess.
- [ ] Implement explicitcompletevalidators and partialderivation from publishedcommit/oracle/child receipts. Invalid comparisons unavailable butdata/status/unknowncounts preserved. Contrastonlyallfour; terminalwindowonlyvalidfixedendpoint. Plot observedupdate/partiallabels; no phantomsummary. Evidence checksum index coverscases/controls/metrics/stops/engineering/reviews/archives; omission makescompletenessfalse.
- [ ] Focused GREEN then .venv/bin/python -m pytest -q; exactfullcount/time/rawSHA retained withallfailures. One fresh finalbranchreviewer beforeproduction; author Importantrepairs meaningfulRED→GREEN/fullsuite; no secondcodefixreview. Source/config/newcode/revision verification/scopedcommitbeforeaction.
- [ ] Exactlyonce JAX_PLATFORMS=cpu .venv/bin/python -m force_consistency.width_rate_replacement --config configs/aspirin-width-rate-replacement.json afterreviews/checks. Exclusiveinvocationmarker/stdout/stderr/exit; monitor savedcommits descriptively only. All actualfailed/blocked/overrun/unreceiptedoutcomes append; no numericalretry.
- [ ] Savedreport/PRELIMINARYcompleteness (later evidence explicitlypending); samefreshreviewer'sdeferredresultreconciliation. Archive raw/finalworkspace/audit/fail/exactproductioncode CRC/entry-set/SHA; originalsretained.  scopedresultssnapshotcommit and durable commitrecord. Then append terminalevidence links, freeze evidenceprefixbytecount/SHA plusrequiredartifactmap and run FINALcompleteness afterall review/preservation/navigation/commitrecords exist. Exclude onlyself-referencing completeness outputs/manifest, postcutoffclosingevidence and subsequentclosingcommitSHA, explicitly enumeratecutoffs. Test omission/edit of eachrequiredlaterartifact and validfailed/blockedclosure. Closingmetadata localcommit, no scientific rerun. Pilotpending; nextprotocoldecisionagainstresults, noautogrid.

## Self-review and authorization ruling

Tasks1–3cover config/arithmetic/sourceidentity, freshstates/originalbaselines, one-shotdeadlines/failurework, pureall-measurementreport and user'scomplete-logrequirement. Interface names/types agree; sourceguards notsilently relaxed. This is bounded reuse, not architecturalrefactor/newscientificsetting. Latest human lets-proceed follows named nextcomparison; preserve established native execution aftertwo review/check gates ratherthanrequestgenericapproval. If review materially changesscientificscope, create prospectiveamendmentbeforeaction. Saved plan is not an executedresult.

Material review incorporated: direct public child uses full source-derived pre-numeric audit and exclusive SHA-bound attempt reservation; final completeness after review/archive/navigation/results-snapshot-commit, acyclic explicit self-reference exclusions. Original fixed-rate/late-decay/width-geometry/order-cache-load caveats retained. Config/518pin identity unchanged; both reviewers must re-review these changes before implementation.
