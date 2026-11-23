# Idris → Olufunke and Aaron: MACE integration handoff

Prepared 18 November 2026 by Muhammed Idris (`imhd866`). This is a proposed
integration transfer for review, not teammate acceptance or launch authorization.

## Code and evidence to review

- Implementation PR: [#18](https://github.com/Shedrackeze002/idl-molecular-force-consistency/pull/18).
- Reviewed candidate publication: `4eaeb7b96f330ebdfaf44c78d8baea13b998b140`.
- Tested implementation: `dd2f3637c440e7a38364bcd944091d6a8a9a534c`.
- Latest [nine-file verification](../../reports/evidence/2026-11-18/trusted-dispatch-review/review-regression-suite.json):
  253 passed, 3 expected skips, 45 warnings, 19 subtests, exit 0.
- [Launch instructions and external provisioning](../run-mace-parity.md).
- [Report contribution draft](../../reports/IDRIS_MACE_REPRODUCTION_DRAFT.md).

PR acceptance, external trust-root provisioning, approved WSL admission and
bounded launch authorization are pending. No accepted environment/launch receipt,
real prediction output or accepted parity audit is available to transfer yet.

## Loader and evaluator interface

`scripts/mace_loader.py` exposes:

```python
verify_packet(packet_root)
load_admitted_arm(packet_root, attempt_root, *, expected_species)
```

The first returns authenticated packet identities. The second returns
`(arm, admission_receipt)` after environment/source/launch checks and one admitted
construction. It is the numerical API, not an onboarding or preapproval command.
Use it only inside the approved execution path after validating the fixed inputs.

The arm reuses Shedrack's shared `calculator_arm` bridge:

| Accessor / field | Contract |
| --- | --- |
| `arm['forces'](positions, species, record_id)` | Returns `(forces, info)` with float64 shape `(N,3)` or nonempty `(B,N,3)` |
| `arm['energy'](positions, species, record_id)` | One unbatched structure; returns `(scalar_eV, info)` |
| Position/force units | Angstrom / eV per angstrom |
| Energy gauge | Residual molecular eV; never subtract the offset again |
| Atom order | Exact validated 21-symbol order, preserved for every structure |
| Periodicity | Nonperiodic ASE structures |
| Identity tag | `checkpoint_identity_verified=False` remains on the generic bridge |

The separate admission receipt, not a caller-supplied tag, binds the checkpoint
snapshot, package/source comparisons, constructor and launch. Integration must
retain this receipt and its digest alongside evaluator results. Do not relabel
the generic bridge flag as verified or overwrite historical result directories.

## Primary artifact and fixed inputs

Use final EMA epoch index 2047 only: 23,397,415 model bytes, SHA-256
`fbd128215c0d74f5d9e5d3ccde778763221f987535dcc8ce4f4f8471a5b35407`.
The released v2 manifest is
`3a872e1886c37b0e069e4ecaaa426bbab6a4a771db7c805f21741ee67e22717c`;
saved final validation predictions are
`22bf7d88815ae3a84ecb8717c497f11311f058c1a862f8efd0e08a5efda0d3cd`.
The full artifact contract remains in the [accepted specification](../research/MACE_LOADER_PARITY_SPEC_2026-10-26.md).

Prediction reproduction is limited to the original 974 validation records with
exact ID joins and the connection counts in the task brief. Do not substitute
the selected checkpoint, retrain, use training/test inputs, align atoms or fit an
energy correction. A recovery driver remains Olufunke's separate integration work.

## Proposed shared synthetic check

Before a real checkpoint is handed over, agree this existing analytic bridge
check. It uses ASE harmonic fixtures and no learned model:

```bash
PYTHONPATH=.:src:scripts "$python_bin" -m unittest discover \
  -s tests -p test_mace_reference_adapter.py -v
```

Idris ran it in the existing WSL environment: **11 tests passed**, exit 0.
[Captured check](../../reports/evidence/2026-11-18/idris-delivery/bridge-check.json).
It covers units, species/order, shapes, energy gauge, identity isolation,
nonfinite/malformed returns, propagated errors and unchanged-relaxer neutrality.
Olufunke's agreement/rerun and acceptance are still pending; an Idris-side pass
does not fill those fields.

## Numerical transfer after approval

The operator uses the protected external launcher and independently supplied
policy digest. Do not invoke the checkout bootstrap or its `main()` directly.
The approved launch record determines the actual absolute paths and command;
see [the run guide](../run-mace-parity.md) rather than inventing another wrapper.

Retain locally: external policy and approved digests, launch/WSL/admission
receipts, supervisor and worker statuses, private constructor snapshot, full and
partial predictions, log files, independent audit, and both saved-result reviews.
Publish only small redacted summaries. Require complete admitted outputs,
one/974/974 construction/force/energy counts, 61,362 force components, and both
prediction maxima <=1e-10 with relative tolerance zero. Separately reproduce all
four score formulas before recording acceptance.

## Aaron's runtime-fact transfer

The existing proposal records a candidate WSL CPU environment and a Torch CPU
build difference from the original Mac scorer. It is not approved portability
evidence. At the authorized attempt, supply actual CPU/OS/Python/package and
source identities, setup/admission/construction/prediction/persistence timing
scope, request counts, completion/resource status, and device/thread/dtype flags.
Preserve distinctions between accessor calls and model-forward counts. No
inference cost, GPU speedup or matched-hardware throughput result exists yet.
Aaron determines any subsequent separately authorized measurement plan.

## Acceptance record — deliberately unfilled

| Decision | Required owner | Current state |
| --- | --- | --- |
| Implementation and exact launch reviewed | Shedrack / two independent reviews | Pending acceptance |
| WSL differences and external provisioning approved | Coordinator | Pending |
| Prediction and four-score reproduction accepted | Two saved-result reviewers | Not run |
| Synthetic check agreed and interface accepted | Olufunke | Awaiting confirmation |
| Real-model evaluator integration accepted | Olufunke | Blocked on accepted reproduction |
| Runtime-fact handoff received | Aaron | Not yet confirmed |

Record each actual decision with a dated reviewer/owner link; do not replace it
with this proposed handoff. Idris remains responsible for reproduction evidence;
Olufunke retains integration and recovery ownership.
