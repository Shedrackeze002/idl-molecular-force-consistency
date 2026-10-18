# MACE evaluator contract and remaining integration

Prepared 18 October 2026. The shared calculator bridge is synthetic-tested;
the checkpoint loader and real-model admission are still pending. Actual
MACE inference and recovery follow the midterm draft.

## Ownership

Shedrack prepared the shared ASE-to-evaluator bridge and its synthetic
tests. Idris retains the MACE checkpoint loader, independent reproduction,
and verification of checkpoint/environment identities. Olufunke owns
integration with the relaxation runner and reporter. These are distinct
contributions; preparation does not count the remaining assignments as
completed.

## Shared interface

`scripts/mace_reference_adapter.py` supplies:

```python
calculator_arm(calculator, species, *, arm_id, checkpoint_identity)
```

Pass an already constructed ASE calculator supporting both energy and
forces, the fixed element-symbol sequence, and a nonempty provenance dict.
The returned arm has the existing callable interface:

- `forces(positions, species=None, record_id=None)` returns `(forces, info)`
  for `(N,3)` or nonempty `(B,N,3)` inputs, preserving the shape and atom
  order. Coordinates are Å; forces are eV/Å; output dtype is float64.
- `energy(positions, species=None, record_id=None)` accepts one unbatched
  molecule and returns `(scalar_eV, info)` in the supplied calculator's
  energy gauge. No offset or per-atom normalization is added.

Every structure uses fresh nonperiodic ASE Atoms. Input validation happens
before calculator access, across the entire batch. No force projection,
rescaling, periodic images, retry or fallback is introduced. Force requests
do not call the energy accessor. A calculator may compute energy internally
or cache results, so force requests are not model-forward counts.

Caller provenance is deep-copied into private state and each output.
`checkpoint_identity_verified=False` is deliberate: the generic bridge
cannot prove which weights the supplied calculator contains. The loader
must independently bind verified bytes to its constructed calculator and
save that admission receipt; a provenance tag is insufficient. The frozen
runner records only its explicit arm audit fields, so integration must
also persist the verified checkpoint identity in a separate receipt or the
existing result-driver audit extension, without editing historical results.

## Primary artifact for the future loader

Use Run A's **final primary EMA model**, epoch index 2047 after 2,048 epochs.
Do not substitute the secondary selected model or Run B based on recovery.

| Item | Identity |
|---|---|
| File in coordinator's verified payload | `models/final_epoch.model` |
| File bytes | 23,397,415 |
| SHA-256 | `fbd128215c0d74f5d9e5d3ccde778763221f987535dcc8ce4f4f8471a5b35407` |
| Original accepted score report SHA-256 | `a14938dc9a1b9fe6f8498ee8f7dd59c7208741b1888062456542091bf63fc433` |
| Saved final validation predictions SHA-256 | `22bf7d88815ae3a84ecb8717c497f11311f058c1a862f8efd0e08a5efda0d3cd` |

These are original artifact identities, not hashes of path-normalized
publication exports. The weights and original scorer package are retained
separately by the coordinator and are not shipped by this clone.

Reuse the accepted scorer's calculator recipe in the existing MACE
environment: CPU, one Torch thread, `default_dtype='float64'`,
`enable_cueq=False`, `enable_oeq=False`, `compile_mode=None`, one ASE
structure per prediction. Accepted versions are Python 3.12.13,
MACE 0.3.16, Torch 2.14.1, ASE 3.29.0 and NumPy 2.5.3. The loader must
verify actual package/source identities against the packet's
`ACCEPTED_IDENTITY.json` with `verify_environment.py`, not just version
labels. The full original scorer package remains an optional historical
audit, separately retained by the coordinator. It also establishes the model's
residual-energy gauge and atom/Z sequence from an admitted frame.

## Verification performed and deferred

Synthetic tests use an analytic harmonic ASE calculator with a constant
energy offset. They check shape, units, species/Z order, nonperiodicity,
input and nested-provenance isolation, no extra energy accessor, failures
before calculation, declared malformed/nonfinite returns, operational
error propagation, and neutrality through the unchanged relaxer. Run in
either existing environment without installing packages:

```bash
PYTHONPATH="$PWD:$PWD/src:$PWD/scripts" \
  python -m unittest discover -s tests -p test_mace_reference_adapter.py -v
```

No checkpoint was deserialized, no trained-model inference ran, and no
archive payload was opened during this preparation. Metadata verification
checks the primary file bytes and the accepted report hash only. Synthetic
success establishes bridge behavior; it does not establish real MACE
parity, recovery success, throughput or hardware portability.

After the draft, complete and review the loader, then specify a bounded
validation-only parity check against the saved final predictions, with
record IDs, coordinates, atom order, tolerances and resource budget fixed
before execution. If parity is valid, evaluate MACE through the unchanged
procedure on the same ten validation starts and six own-reference controls.
Keep setup, force requests, internal computations and timings distinct.
Preserve errors and incomplete outcomes. No test scoring or solver tuning
is authorized by this preparation.

MACE is an external predictive reference with different architecture and
training exposure. Its recovery is not established, and a comparison does
not isolate a causal effect of force consistency. Decide on any revised
solver only after this control; apply a reviewed revision identically to
the existing checkpoints and starts.

## Rebuild and check the environment

The accepted scorer ran on **macOS arm64, Python 3.12.13** (full platform
identity in `ACCEPTED_IDENTITY.json`). This is the validated scoring host,
not proof of the lock's original resolution host. The retained lock header
records `uv pip compile --generate-hashes --python 3.12`; it does not prove
a universal cross-platform lock. The following commands target a fresh
local environment, from the directory containing the extracted packet:

```bash
uv venv --python 3.12.13 .venv-mace
uv pip sync --python .venv-mace/bin/python --require-hashes mace-idris-handoff/environment/requirements.lock
.venv-mace/bin/python mace-idris-handoff/verify_environment.py
```

These commands were checked against the installed CLI; no fresh install
was performed for this handover. On another OS, preserve any installation
failure and propose a separately reviewed compatible lock and parity
check. Do not silently replace pinned versions. A platform difference is
reported separately and requires review; it is not an automatic GPU ban.

`verify_environment.py` uses distribution metadata without importing Torch
or MACE. It checks Python, seven package versions and eight source files.
For every pin, `source_files` gives the distribution, installed-relative
path and expected digest. The exact operation is:

```python
from importlib import metadata
import hashlib
path = metadata.distribution(pin['distribution']).locate_file(pin['relative_path'])
actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
```

These are raw file hashes, including the calculator source and seven
training-related module files, not directory hashes. A PASS establishes
only installed metadata/source-byte agreement, not model admission,
hardware portability or prediction parity. Save its JSON output with the
loader's separate admission receipt.

## Review, approval and transfer

Idris drafts the loader, admission receipt and bounded parity plan. Shedrack
coordinates two independent reviews and approves the plan and admission
receipt after their findings are resolved; scientific execution still
needs the coordinator's explicit bounded authorization. Olufunke is the
assigned runner/reporter integration owner; her acceptance and timing are
not yet confirmed. Agree timing at the next team circulation rather than
assuming a member deadline.

Shedrack will transfer the packet privately as an attachment or controlled
link. The packet is prepared locally; delivery and receipt are not yet
claimed. Keep its weights/data out of GitHub.
