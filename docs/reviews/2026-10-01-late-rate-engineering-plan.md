# Late-rate continuation: independent engineering plan review

Date: 2026-10-01. Scope: preimplementation review only. No implementation edits, model evaluation, training, or numerical experiment performed by this reviewer.

Reviewed `AGENTS.md`, the saved late-rate design and implementation plan, `configs/aspirin-late-rate.json`, the proposed source-pin map, and the relevant checkpoint, weighted-step, width/rate, coverage, and continuation APIs. I also inspected the actual parent metadata/history and verified all 16 proposed file pins against current bytes.

## Gate recommendation

**Amend the plan before implementation, then obtain follow-up review of the amendments.** No critical scientific or engineering contradiction was found. Three important specification gaps below should be resolved before code is written. These are bounded additions to the focused runner, not a request for a new execution framework.

## Verified facts and useful existing APIs

- The pin-map SHA256 is `55fb10bef1abc25783d8b6376dbfcb4896da71db1f6216210a01249224b42e30`; all 16 source files currently match. The pinned parent history contains 64 observations from update0 through6000, including1900/2000/2100. Thus the proposed saved-only spike analysis has the necessary source observations already pinned.
- Parent summary says completed/execution-valid. Model is width32, two message iterations, degree1, eight radial functions, cutoff5Å, zero-initialized energy readout. Parent provenance's original baselines equal history[0].metrics exactly.
- `runs.save_checkpoint` stores a Flax-serialized state in envelope `payload`, plus SHA256, serialized JSON provenance, and hashed `control_payload`. There is **no `.meta.json` sidecar**. Width/rate committed snapshots were saved without selection controls, so `load_run_control` deliberately rejects them; it is not the correct continuation API.
- `load_checkpoint(..., template=...)` can restore the full parameters and Optax state. A template initializer is structural scaffolding, not a reset of the restored optimizer. `build_weighted_step` constructs a rate-specific transformation but accepts the incoming Adam state, so different scalar rates can preserve count/moments.
- `width_rate_runs.first_step_oracle` is explicitly hard-coded to fresh count0/update1 and cannot be reused unchanged for this count6000 continuation. Its independent NumPy Adam arithmetic is reusable. The new boundary helper must verify absolute count6001 and8001 against the incoming nonzero moments.
- The versioned `commit_state` pattern publishes the receipt last and refuses existing immutable checkpoint/history versions. The new runner should retain this behavior, including retaining orphan files on failure.

## Important amendments

### E1. Freeze and validate the whole reviewed numerical contract

The config currently carries a source-pin-file hash, but Task1 only explicitly tests schedule values and file pins. The plan does not explicitly require runtime rejection of substituted loss weights, seed, support, bounds, caps, or observation schedule. A pin check alone would still pass if these reviewed settings were changed.

**Exact amendment:** add a focused `validate_contract(config)` used by launch, parent load, probe, and reporting. Require the exact reviewed two arms and order, schedule entries, start/end, score updates, weights, seed, CPU/float64, microbatch16, gate fraction, tolerance, caps/reserve, no early stopping/retry, and pinned parent identity. Bind the launch config hash into each arm's provenance and verification receipt. Validate source-derived model/normalization/128-fit/16-anchor identities against the parent rather than adding new independent adjustable settings. Add tamper tests for one rate, weights, end update, and missing boundary observation.

### E2. Bind current code and actual checkpoint ancestry across both subprocesses and reporting

Input pins are sound, but the current source-pin map only includes `uv.lock` among executable environment files. The plan says to use provenance, yet does not explicitly specify whether the original parent code fingerprint must equal the new runner fingerprint, or how the child verifies it. Blind equality is wrong because the new module changes code; no comparison at all is insufficient.

**Exact amendment:** retain the exact original parent provenance for `load_checkpoint`; verify its envelope through existing IO, then compare restored state/count, contract and commit hashes, parent history endpoint, probe completion and prediction receipt, copied inputs, and immutable original baselines. Record parent provenance identity separately from the new execution code identity. Before production snapshot/hash the current participating source files, `pyproject.toml`, lock, and new CLI/report code; verify that identity in each child and after execution. JSON-normalize provenance once before writing files and checkpoint envelopes, preserving the existing fix for integer dictionary keys. Test a modified current code fingerprint and a mismatched parent commit/provenance pair. No historical source file needs modification.

### E3. Make interrupted-work accounting durable and keep completion checks independent of summaries

The plan requires honest interrupted accounting and robust reports but does not specify a durable record for an abrupt process death. In-memory counts plus the last scored checkpoint cannot establish the exact work done after that receipt. Similarly, a successful summary cannot establish that the requested observation schedule and boundary verification actually occurred.

**Exact amendment:** persist a small phase/progress record before each bounded training block and before warmup/oracle/probe work. On normal return record actual attempted/computed/receipt-persisted *additional* updates; on abrupt interruption report the last exact receipt/count and explicitly unknown in-flight interval, not zero or a fabricated exact exposure count. Preserve any orphan newer files without treating them as committed. Tests should cover interrupted unscored work and a publication failure after checkpoint creation but before the receipt. Apply the finalization reserve before dispatching work, give probes the minimum arm/global remaining timeout, and invalidate any completion that exceeds either cap after persistence.

Saved-only reporting must reconstruct endpoint metrics/gates/counts from the committed history; require the exact configured score-update sequence for a complete arm, final receipt at10000, both boundary oracle receipts with incoming/outgoing state identities, and the fresh-process receipt bound to config, source code, inputs, references, checkpoint/history hashes and144 frames. Require identical parent6000 state hashes/initial metrics in both arms. Missing, failed, timed-out or tampered records must retain partial arm results but suppress complete schedule contrasts. Test deletion of an interior observation/boundary receipt and disagreement between summary and raw committed history. This can remain in the focused module.

## Nonblocking clarification

Make oracle tolerances explicit: prediction reproduction remains absolute `1e-10` with zero relative tolerance; independent NumPy-vs-Optax state comparisons can use the established `atol=1e-10, rtol=1e-8`, reporting the maximum discrepancy. Before computing either boundary proposal, validate actual count and nonzero incoming moments; verify the committed update against the independently proposed state, not merely metric equality. Count all discarded gradient/proposal/prediction work separately from the expected8000 new training updates and1,024,000 fitting exposures.

## Limits

This is a plan/API/artifact review, not a claim that the unimplemented runner works. I did not run the existing 476-test suite or deserialize/evaluate the parent checkpoint. Those operations belong to implementation verification/preflight. Scientific inference and schedule choice remain the separate scientific review's remit.
