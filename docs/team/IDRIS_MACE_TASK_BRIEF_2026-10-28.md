# Idris: complete MACE loader and reproduction task

28 October 2026. This is the task specification from access through reviewed delivery. Use the [Windows quickstart](WINDOWS_MACE_QUICKSTART_2026-10-28.md) for setup commands and the linked technical contracts for implementation detail.

## Goal, ownership and limits

Deliver an authored, tested loader for the completed MACE Run A model, an auditable Windows/WSL environment, and independently checked validation prediction/score reproduction. Deliver the resulting code, evidence, report contribution and integration handoff.

Idris owns implementation and independent reproduction. Shedrack coordinates review, environment admission and bounded launch authorization. Olufunke owns evaluator integration and its acceptance. Aaron owns later physical-consistency and matched-hardware cost analysis, with Idris supplying runtime facts.

The model already completed 2,048 epochs. Do not retrain, add seeds, substitute the selected checkpoint/Run B, change the relaxer or open training/test partitions. Recovery execution and throughput benchmarking are later work packages.

**Current status:** packet/source access is available; actual teammate receipt, WSL installation, loader implementation, platform admission and numerical reproduction are not verified. This brief specifies the work; it does not declare it completed.

## Ordered completion checklist

All boxes describe Idris's future evidence, rather than work completed by this document.

| ID | Deliverable | Owner and completion evidence |
|---|---|---|
| I01 | Verified packet and platform facts | Idris: quickstart steps 1–3; packet PASS, Windows/WSL/system outputs |
| I02 | Runtime and test environments recorded | Idris: generated runtime/test locks, source commit, package inventory, build/install/check logs and original-Mac comparison |
| I03 | WSL admission proposal reviewed | Idris submits; Shedrack coordinates independent reviews and records the approved differences or refusal |
| I04 | Loader, worker, supervisor and independent auditor implemented | Idris: authored files, meaningful synthetic tests and documented commands |
| I05 | Exact implementation and launch record reviewed | Two implementation reviews resolved; Shedrack authorizes the specific bounded attempt |
| I06 | One complete or explicitly failed/incomplete validation attempt | Idris: raw outputs, identities, counts, maxima, resource/failure status; no hidden retry |
| I07 | Independent prediction and score audit | Two independent saved-result reviews: exact ID/count/hash checks and separate parity/score verdicts; resolve material findings, without model loading |
| I08 | PR, report contribution and integration transfer | Idris: code/evidence/report PR; Olufunke records bridge/integration acceptance and any remaining blockers |

I01 is the first return checkpoint. I04 synthetic implementation can progress while I03 is reviewed; I05–I07 cannot bypass platform or numerical authorization.

## 1. Use the correct source and packet

| Revision | Meaning |
|---|---|
| `3fa5969e5a0c2fdeaebc76a05bace20799715203` | Preserved scientific source baseline |
| `47b727b4efa4be91ada585be343e1ddb509b7116` | Packet release source; it predates the Windows configuration |
| `adf1b13819015cbfb1f2fbb79a79df132e7a4545` | Owner-development starting point, including Windows configuration |
| Your reviewed implementation commit | The later launch identity; record its full SHA and source hashes |

The baseline `src/`, `evaluation/`, shared bridge and bridge test are unchanged between the first and third revisions. New authored files and approved environment records must have their own identities.

The quickstart creates the owner branch from `adf1b13819015cbfb1f2fbb79a79df132e7a4545`. In an existing clean clone, the equivalent commands are:

```bash
git fetch origin refs/heads/post-midterm-mace-readiness:refs/remotes/origin/post-midterm-mace-readiness
git switch -c idris-mace-loader adf1b13819015cbfb1f2fbb79a79df132e7a4545
```

If the owner branch exists or you have uncommitted work, preserve it and report its current commit. Do not reset or replace work to repeat these commands.

Use the [released v2 packet](https://github.com/Shedrackeze002/idl-molecular-force-consistency/releases/tag/mace-idris-packet-v2-2026-10-28). Required identities:

- Archive: `ff92c74b83d3103a2542f1564d93be3b153f5342faef2675f9845d9d99958119`.
- Extracted manifest: `3a872e1886c37b0e069e4ecaaa426bbab6a4a771db7c805f21741ee67e22717c`.
- Primary final EMA model: `fbd128215c0d74f5d9e5d3ccde778763221f987535dcc8ce4f4f8471a5b35407`, 23,397,415 bytes, epoch index 2047.

The manifest authenticates eleven files, with itself authenticated separately. Preserve original packet bytes. Record actual download and verification; GitHub visibility alone does not establish receipt.

## 2. Complete Windows runtime setup and add test tools

Follow quickstart steps 1–4. The WSL recipe is proposed, not an installation verified on a teammate PC. Its source-only hostlist exception and CPU Torch build difference must stay visible.

The 44 runtime pins do not include pytest. After the runtime lock and installation exist, add test tools through a separate constrained lock. These versions come from the repository's `uv.lock`.

Run from the onboarding folder created by the quickstart:

```bash
(
set -euo pipefail
cd "$HOME/group9-mace-onboarding-2026-10-28"
test -s wsl-requirements.lock
if [ -e requirements-tests.in ] || [ -e requirements-tests.lock ]; then
  printf '%s\n' 'Preserve the existing test requirements and report their stage.' >&2
  exit 1
fi
cat > requirements-tests.in <<'REQ'
pytest==9.1.1
iniconfig==2.3.0
pluggy==1.6.0
pygments==2.21.0
REQ
uv_bin="$PWD/tools/uv"
python_bin="$PWD/.venv-mace-wsl/bin/python"
"$uv_bin" pip compile requirements-tests.in --constraint wsl-requirements.lock --python "$python_bin" --only-binary :all: --generate-hashes --output-file requirements-tests.lock
"$uv_bin" pip install --python "$python_bin" --require-hashes --no-deps --only-binary :all: -r requirements-tests.lock
"$uv_bin" pip check --python "$python_bin" | tee package-check-after-tests.txt
"$uv_bin" pip freeze --python "$python_bin" > environment-after-tests.txt
)
```

Use `install`, rather than syncing the test-only lock, to retain runtime packages. The runtime constraint preserves `packaging==26.3` and the model's other core versions. Retain both locks and the final full package inventory.

Do not project-sync the unrelated E3x/JAX environment into this MACE environment. No model construction or inference belongs to setup or synthetic verification.

## 3. Submit a WSL admission and launch-amendment proposal

The immutable packet verifier describes the original Mac scoring environment. Keep its actual comparison output, including the CPU Torch-version/platform differences. Do not edit `ACCEPTED_IDENTITY.json` or convert a Mac comparison failure into a Linux PASS.

Submit a separate `wsl-admission-proposal.json`, initially `PROPOSED`, with:

- Packet, manifest, model, runtime-lock, test-lock and implementation/source hashes.
- Actual Python patch, Windows/WSL/Linux versions, x86_64 architecture, glibc, CPU identity and complete installed package inventory.
- Original and proposed values for each changed platform/package field; explicitly identify original Torch `2.14.1` versus proposed `2.14.1+cpu`.
- Original and actual hashes for the eight authenticated source files, with artifact provenance and an explanation for each difference.
- Test-only package additions, setup/build/check logs and all original-verifier outputs, without omitted failures.

Shedrack coordinates independent review before accepting any difference. The approved receipt must enumerate exact permitted differences and proposed source hashes; every unapproved mismatch is a refusal. Do not automatically approve whatever the local machine reports.

The loader checks the approved WSL receipt and its pinned hash in the launch record. It also checks every unchanged packet/model/source requirement. The original Mac identity remains historical evidence, and the separate approved WSL receipt governs that specific platform attempt.

Include this prospective numerical amendment in the review request:

| Field | Required proposal |
|---|---|
| Input scope | Same 974 validation records, no training/test access |
| Calculator | Primary final EMA; CPU; one Torch thread; float64; no cueq/oeq acceleration; no compile |
| Requests | One construction, 974 force accessor requests and 974 energy accessor requests |
| Prediction acceptance | Every saved-prediction difference at most `1e-10 eV` or `1e-10 eV/angstrom`; relative tolerance zero |
| Added saved audit | Four score formulas and absolute bounds in section 6, using the same saved outputs and validation references |
| Resources | Shared 1,800-second setup/admission/construction/prediction/persistence cap; 256 MiB attempt directory; 16 MiB per log |
| Failure | Preserve partial outputs, timeout and errors; no retry/fallback/substitution; a new attempt requires a new reviewed decision |
| Execution evidence | Fresh absolute output path, exact command/interpreter, approved receipt hashes, implementation commit and resolved review records |

The original termination policy permits ten additional seconds for cleanup, with five seconds before killing remaining owned children. A run crossing 1,800 seconds cannot pass. Accessor counts do not establish model-forward counts.

The accepted 26 October Mac-only specification remains unchanged. This worksheet is not an approved WSL amendment or launch authorization. Obtain both the reviewed amendment and explicit bounded authorization before constructing the real model.

## 4. Implement and test the assigned modules

Read the [original detailed implementation plan](../superpowers/plans/2026-10-26-mace-loader-parity.md), [specification](../research/MACE_LOADER_PARITY_SPEC_2026-10-26.md) and [shared contract](../mace-evaluator-contract.md). Their technical controls remain, with WSL differences handled through section 3.

Create these files in your owner branch; they are implementation deliverables, not tools already supplied:

| File | Required interface or responsibility |
|---|---|
| `scripts/mace_loader.py` | `verify_packet(packet_root)` and `load_admitted_arm(packet_root, attempt_root)`; return shared arm plus separate admission receipt |
| `scripts/check_mace_parity.py` | `validate_inputs`, `compare_predictions`, `run_parity`; worker CLI accepts `--packet-root` and `--attempt-root` |
| `scripts/supervise_mace_parity.py` | Supervisor CLI accepts `--packet-root` and `--output`; owns fresh output, deadlines, child group and file/log bounds |
| `scripts/audit_mace_parity.py` | Independent saved-output audit; CLI accepts `--packet-root` and `--attempt-root`; no Torch/MACE model import |
| Four corresponding test files | `test_mace_loader.py`, `test_mace_parity.py`, `test_mace_parity_supervisor.py`, `test_mace_parity_audit.py` |
| `docs/run-mace-parity.md` | Reproducible commands, artifacts, statuses, accepted platform record and limits |

Preserve `src/`, `evaluation/` and `scripts/mace_reference_adapter.py`. Do not call the old scorer CLI: its full workflow also accesses training partitions. Reimplement only the bounded validation task described here.

Production order is packet authentication, complete input validation, approved environment/source admission, verified model snapshot construction, then prediction. Wrong identities or malformed inputs must cause zero constructor calls.

The constructor recipe below is an implementation reference inside the admitted loader. It is not an onboarding command to execute before authorization:

```python
import torch
from mace.calculators import MACECalculator

torch.set_num_threads(1)
calculator = MACECalculator(
    model_paths=[str(verified_snapshot_path)],
    device="cpu",
    default_dtype="float64",
    enable_cueq=False,
    enable_oeq=False,
    compile_mode=None,
)
```

Copy the authenticated model into the fresh attempt folder, hash before construction, pass that exact path, and hash again after prediction. Bind it through `calculator_arm(calculator, species, arm_id=..., checkpoint_identity=...)`.

The arm exposes `forces` returning `(array, info)` and `energy` returning `(scalar, info)`. Keep its `checkpoint_identity_verified=False` tag; the separately reviewed admission receipt proves model/environment identity.

Cover wrong/tampered manifest/model/source, missing/symlink files, constructor-path binding, duplicate/missing/reordered IDs, literal `18_2`, species/order, dtype/shape/nonfinite values and pickle refusal. Test absolute bounds below/at/above the limit, accessor counts, retained partial output, timeout/group cleanup and disk/log overflow.

Use analytic fields and fake constructors for those checks, never the real checkpoint. Keep injected test factories outside the production CLI. After creating the implementation/test files, run from `source/`:

```bash
python_bin="$HOME/group9-mace-onboarding-2026-10-28/.venv-mace-wsl/bin/python"
PYTHONPATH=.:src:scripts "$python_bin" -m pytest -q tests/test_mace_loader.py tests/test_mace_parity.py tests/test_mace_parity_supervisor.py tests/test_mace_parity_audit.py tests/test_mace_reference_adapter.py tests/test_shape_recovery_runner.py tests/test_shape_recovery_outcomes.py tests/test_shape_recovery_report.py -p no:cacheprovider
```

Require zero unexpected failures. Preserve expected unavailable-archive skips and warning counts; do not fetch training/test archives to eliminate a skip. Retain the command, output and commit, then obtain two implementation reviews before launch.

## 5. Run the one authorized validation comparison

Validate exactly 974 unique record IDs and the original per-frame atomic order, joining inputs, memberships and saved predictions by ID. Require float64 finite energy `(974,)` and forces `(974,21,3)`, giving 61,362 force components.

Connection record counts must be `13:165`, `18:217`, `18_2:227`, `66:199`, `78:166`. Preserve `18_2` as a distinct string.

Coordinates are angstrom, forces eV/angstrom and saved energies residual molecular eV. `REF_energy` is already residual; never subtract the offset again. `REF_forces` is the validation reference-force array.

Use unchanged positions and atom order: no alignment, rotation, translation, projection, re-centering or fitted correction. Save record IDs, full returned predictions, per-record/global differences, actual call counts, admission/environment/code/input hashes, elapsed time and final failure/status evidence.

The CLI names below describe your required implementation. Fill the variables with actual reviewed absolute paths and interpreter in the launch receipt; these templates do not authorize execution:

```bash
PYTHONPATH=.:src:scripts "$python_bin" scripts/supervise_mace_parity.py --packet-root "$packet_root" --output "$attempt_root"
PYTHONPATH=.:src:scripts "$python_bin" scripts/audit_mace_parity.py --packet-root "$packet_root" --attempt-root "$attempt_root"
```

Freeze the reviewed implementation and source identities during the attempt. No pull, commit, source/dependency change or automatic retry while it runs. Audit saved outputs without loading the model, writing its report separately from the protected original outputs.

## 6. Independently reproduce prediction agreement and scores

Obtain two independent saved-result reviews of the admitted outputs and independent audit. Resolve material findings before recording accepted reproduction.

Report two verdicts. `PARITY_PASS` requires complete admitted outputs, all counts/identities and both maximum absolute prediction differences within the unchanged limits. Partial agreement, timeout, operational failure or missing admission cannot pass.

The separate saved-output score audit joins predictions to validation `REF_energy` and `REF_forces` by exact record ID. It uses no model imports, extra learned calls, fitting data or test data.

For component error `dF = predicted_force - reference_force`, use:

- Pooled force MAE: `sum(abs(dF)) / 61362`, in eV/angstrom.
- Force component RMSE: `sqrt(sum(dF**2) / 61362)`, in eV/angstrom.
- Equal-connection force MAE: arithmetic mean of the five connection component MAEs, each divided by its own component count.
- Residual molecular energy MAE: `sum(abs(predicted_energy - REF_energy)) / 974`, in eV per molecule.

These are component metrics, not per-atom vector norms. Do not average connection RMSEs or treat frames as independent connections.

| Unrounded original final-validation metric | Reference value |
|---|---:|
| Pooled force MAE, eV/angstrom | 0.00525983286381519 |
| Force component RMSE, eV/angstrom | 0.02641038976153195 |
| Equal-connection force MAE, eV/angstrom | 0.005348956868537277 |
| Residual energy MAE, eV/molecule | 0.0074618421825989 |

Source: `models.final.validation` in the [public accepted report](../../reports/evidence/2026-10-15/mace-runA/report.json), file SHA-256 `85f70bb8865c23df86fa94303471f77bd76b03083fb184f6dbf513874b22c998`. This normalized public report has its own identity, distinct from the original accepted-report hash inside the packet.

The prospective `SCORE_REPRODUCED` rule requires all four unrounded absolute differences at most `1e-10` in the corresponding table units, relative tolerance zero, with exact counts and finite float64 reductions. Include this rule in the WSL amendment before execution.

Keep the residual-energy target; the separately reconstructed absolute-energy metric has different floating-point cancellation. Record both verdicts and all differences independently. Do not change bounds after viewing outputs.

## 7. Deliver evidence, report and integration handoff

Retain original packet, model snapshot, full predictions and protected execution outputs locally with checksums. Publish bounded code/tests, small result summaries, commands and limitations through your own authored PR.

Your delivery checklist is:

- Authored implementation/test commits and `docs/run-mace-parity.md`, with final synthetic commands and outputs.
- Runtime/test lock identities, package inventory, original Mac comparison, reviewed WSL admission and exact launch receipt.
- Input/model/source/output hashes, counts, maxima, resource status and independent parity/score audit reports.
- A summary table with original values, your measured WSL values, differences and separate verdicts; label missing results explicitly.
- Your report section explaining loader/environment/reproduction methods, outcomes, failures and limits. Credit Shedrack's original training and your independently authored contribution accurately.
- A handoff to Olufunke containing the loader API, accepted environment/receipt identities, arm/units/gauge contract, commands and one agreed synthetic bridge check. She records integration acceptance or the remaining blocker.
- Actual PR/commit and evidence links in your contribution record after they exist. Use your own GitHub-linked author identity.

A successful full task requires reviewed admission, complete accepted comparison/audit, two resolved independent saved-result reviews, reproducible authored delivery and integration acceptance. That acceptance is not a recovery result or throughput benchmark.

A failed or incomplete attempt still needs a valid delivery: retained errors/partial outputs, verified identities, failed criterion, completed stages and a proposed next bounded decision. It must not be labelled successful admission or reproduction. No silent retry, checkpoint switch, tolerance relaxation or solver change.

For help, send Shedrack the milestone ID, exact command, exit status, error/output, source commit and relevant small receipt files. Do not replace evidence with an unsupported PASS statement. Recovery goes to Olufunke after accepted reproduction; matched-hardware cost work is coordinated with Aaron separately.
