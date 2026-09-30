# Controlled Fixed Loss-Balance Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL superpowers:executing-plans, native implementation task by task. Two independent scientific/engineering plan reviews before code, material revisions re-reviewed; preserve workspaces per AGENTS.md.

**Goal:** Test one frozen loss balance and a force-only control against an exact replay at matched data/model/update budget, preserving raw joint gates and unsuccessful evidence.
**Architecture:** Isolated weighted objectives/gradient accumulation; optional trainer step-factory injection; frozen calibration/source controller using existing train-only/checkpoint/coverage infrastructure; JSON-only report.
**Tech Stack:** Locked Python3.12/JAX/Flax/E3x/Optax/NumPy/Matplotlib, float64CPU, no dependencies added.
**Spec:** docs/superpowers/specs/2026-09-30-loss-balance-diagnostic-design.md.

## Global Constraints

- configs/aspirin-loss-balance-diagnostic.json pins31coverage sources and compact fresh-C gradient evidence. Complete31source and38calibration-raw gates precede prepared arrays. Same prepared/split/subset IDs and128fit/16anchor; anchor entirely fitted, train-only.
- seed0 zero-head16feature/degree1/2iteration/8basis/5Å model, Adam.001, microbatch16; weights replay1/1, balanced0.014584365670390719/1.9854156343296092, force-only0/2 fixed before new trials. Historical final params never initialized.
- All arms fresh, same initial param/optimizer hashes;2000updates/256000exposures, each1800s, warmup128 separately counted/discarded. Score every100 plus250 and0/final; no success early stop/resume/retry/sweep/newseed/GPU/test/fullpilot.
- Preserve raw common-baseline E/F errors/gates, both<.1initial, balanced joint success requires fitting AND anchor. Force-only force success is separate from joint success. No best-checkpoint selection or causal/capacity/held-out claim.
- Replay both partitions/E/F/all22source observations1e-10/1e-8; fresh-process source/checkpoint/input identity. Operational/replay failure retains partials and blocks pending arms, nextstep repair.
- Add optional step_factory=None to run_coverage_arm only; default unchanged, baseline normalized_loss unchanged. One Adam update after sequential fixed-param frame-weighted gradients.
- Native branch/local commits, preserve proposal.pdf deletion/Group9.pdf, all reviews/logs/evidence and Obsidian navigation. No merge/push/cleanup.

## Review Focus

1. Whole-tree calibration includes bias/zero-head structure; only initial gradient balance, no adaptive/Adam causal claim.
2. Force-only arm can pass force yet fail energy: never report joint success or use its energy as tuning selection.
3. Source/calibration or replay mismatch before candidate training: reject incomplete pins and block later arms, never use saved final initialization.
4. Weighted microbatches/change of builder default must not alter baseline/replay semantics, optimizer application count or force mixed derivatives.
5. Partial/nonfinite/wall/preflight failure must preserve attempted artifacts/status and report actual counts/verified outcomes.

### Task 1: Isolated weighted losses/updates and trainer injection

**Files:** create src/force_consistency/loss_balance.py, tests/test_loss_balance.py; modify src/force_consistency/coverage_runs.py run_coverage_arm signature/step selection only.
**Interfaces:** validate_weights(weights:dict)->tuple[float,float]; weighted_loss(params,batch,model,normalization,weights)->(scalar,raw_metrics); accumulated_weighted_loss_and_grad(params,batches,model,normalization,weights)->((scalar,raw_metrics),grad_tree); build_weighted_step(model,normalization,learning_rate,weights)->step callable with prepare hook; run_coverage_arm(...,*,step_factory=None).

- [x] Write failing independent weighted objective/gradient and3/3/1chunk tests, witness mixed force-backbone gradients, negative/nonfinite/allzero rejection, positive coefficient sum2, single Adam count/update equality and force-only bias stability. (1,1) exact delegated baseline; incoming trees unchanged.
- [x] Watch RED missing helper; implement isolated functions/cache JIT weighted loss, sequential chunk weighting then one Adam apply;1/1 delegate existing builder exactly, no baseline loss changes. Test actual factory injection with tiny2-update/fresh-process run and default monkeypatch regression.
- [x] Run .venv/bin/pytest -q tests/test_loss_balance.py tests/test_coverage.py; expect GREEN. Log meaningful failed checks/full suite appropriately; commit scoped.

### Task 2: Frozen calibration, replay, matrix controller and CLI

**Files:** create src/force_consistency/loss_balance_runs.py, tests/test_loss_balance_runs.py; modify cli.py.
**Interfaces:** balance_preflight(config)->(train,manifest,norm,selected,historical,calibration); verify_balance_replay(current,source,tolerance)->dict; execute_balance_matrix(root,runner,replay)->dict; diagnose_loss_balance(config)->dict. CLI diagnose-loss-balance --config.

- [x] Tests stale/incomplete31pins or calibration hash precede loader, bad ratio/count/update/IDs/state/weights rejects, historical final not passed as initial; engineering tiny-run fresh hashes/checkpoint probe. Replay both partition/errors/allsource22updates, missing/duplicate/mismatch blocks balanced/control. Protected dirs, partial exceptions retained without collisions, wall/nonfinite/downstream blocked and correct repair nextstep. RED then implementation.
- [x] Reuse load_gradient_sources for existing full source checks; verify calibration JSON hash plus complete38raw artifact index under calibration_run before loader and exact formula/source metadata against verified fresh state/subsets. Save current versus historical provenance/config/subsets/source calibration. No calibration rerun or adaptation. Each new arm invokes run_coverage_arm with isolated weights factory, immutable fixed-baseline gates and fresh expected hashes.
- [x] Log per-arm rationale/config/controls/gates before running, outcomes/stops/hashes immediately. Incremental root summaries, failed attempted versus blocked distinct. Controller validity all arms finite/fullcap/freshprocess/replay, never defined by learning success. Freshprocess predictions via existing coverage_probe.
- [x] Run .venv/bin/pytest -q tests/test_loss_balance_runs.py tests/test_loss_balance.py tests/test_coverage_matrix.py; expect GREEN. Full suite117prior+new before production; commit code/prospective evidence.

### Task 3: Once-only execution, balanced report, review and navigation

**Files:** create scripts/report_loss_balance.py, tests/test_loss_balance_report.py; reports/loss-balance-diagnostics.md/json/figures/execution record and final review after trials.
**Interfaces:** report generate(root:Path,output:Path)->None reads savedJSON only; no arrays/model execution.

- [x] Report tests partial/preflight/blocked counts, force-only force success without joint success, no fictitious causal/held-out claims. RED then derive raw endpoint/gate/status/compute tables and all22curves with scheduled checkpoints; initial+final common baseline and source/calibration/generator/run hashes. Operational invalidity gates scientific interpretation.
- [x] Full suite GREEN, then exact reviewed config ONCE: replay -> balanced -> force-only. All outcomes unknown; failures/blocked preserved, no extraattempt. Source/calibration hashes checked again after each arm/end.
- [x] Reconcile histories/metrics/gates/counts/init/checkpoints/replay/provenance and raw/compact/hash totals. Generate measured scientific plots, visually inspect, record every failed trial/step and limitations. Update Obsidian via CLI with backups/readback.
- [x] Fresh independent final review using most capable model; resolve Important/Critical through RED-to-GREEN/full suite without unreviewed research rerun. Mark plan/ledger complete, archive durable evidence locally, preserve workspaces/branch/user changes.

## Decision after this milestone

Balanced full gate pass only motivates separately reviewed pilot amendment; mixed E/F outcomes are tradeoffs; force-only success not balanced acceptance. Operational failure requires repair before interpretation. Failure under fixed arms is not a capacity theorem or a negative verdict on other weights/optimizers. No automatic broader training or parameter tuning.

Execution complete30September2026: exactmatrixonce, learninggatesfailed, finalreviewfindingrepaired,152tests pass. Evidence/report/executionrecord retain alloutcomes; no broadertraining authorized.
