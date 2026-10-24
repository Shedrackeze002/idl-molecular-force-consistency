# Idris: MACE loader and reproduction handover

18 October 2026. The shared calculator bridge is ready. Your remaining
contribution is the checkpoint loader, model/environment admission and
independent reproduction. Olufunke owns integration with the relaxation
runner and reporter. No repeat training or additional seeds are needed.

## Access and first actions

The bridge and contract are on `mace-evaluator-preparation`, commit
`26884dac3713178b55080a3c333873b95e937ecf`, in
[PR #9](https://github.com/Shedrackeze002/idl-molecular-force-consistency/pull/9).
It depends on [PR #8](https://github.com/Shedrackeze002/idl-molecular-force-consistency/pull/8).
Both were open when this handover was prepared. Do not assume the default
branch contains these changes.

From an existing clean clone, fetch the branch and create your own branch:

```bash
git fetch origin mace-evaluator-preparation
git switch -c idris-mace-loader origin/mace-evaluator-preparation
```

If you already have uncommitted work or that branch name exists, preserve
your work and use your existing branch rather than resetting it.

Ask Shedrack for **`mace-idris-handoff-2026-10-18-v2.tar.gz`**, a prepared local
bundle of about 15.5 MB. It contains the primary model, validation inputs
and memberships, saved predictions, environment lock, source pins,
contract and a standalone checksum verifier. Model/data bytes are retained
separately from GitHub. The packet contains no test partition.

Archive SHA-256:
`ff92c74b83d3103a2542f1564d93be3b153f5342faef2675f9845d9d99958119`.
Verify this digest, extract into a new folder, then run:

```bash
python3 mace-idris-handoff/verify_bundle.py
```

Expected: `PASS: 11 inventoried files; no model loaded`. The manifest itself
is excluded from its member inventory and authenticated by the archive
digest. Do not commit the packet, weights or validation data to your PR.

## What to implement

Read [the shared contract](../mace-evaluator-contract.md). Implement your
loader outside the frozen `src/` and `evaluation/` inventories, construct
one admitted ASE MACE calculator, and bind it with
`scripts/mace_reference_adapter.py:calculator_arm`. Do not change the
relaxation algorithm to make the model pass.

Use the primary final EMA model at epoch index 2047 after 2,048 epochs:

- `model/final_epoch.model`, 23,397,415 bytes;
- SHA-256 `fbd128215c0d74f5d9e5d3ccde778763221f987535dcc8ce4f4f8471a5b35407`.

The accepted recipe uses CPU, one Torch thread, float64,
`enable_cueq=False`, `enable_oeq=False`, `compile_mode=None`, and one ASE
structure per prediction. `ACCEPTED_IDENTITY.json` lists accepted versions
and original calculator/installed-source hashes. Verify actual installed
source bytes and record environment differences. The original scorer
package is separately retained by Shedrack if you need the full historical
audit; it is not silently replaced by this compact packet.

The bridge preserves caller tags but marks
`checkpoint_identity_verified=False`. Your separate admission receipt must
prove that the checked model bytes were used to construct the calculator;
a copied provenance dict is not enough. Preserve errors and incomplete
attempts without an automatic retry or fallback.

## Units, energy gauge and row mapping

The validation input is `data/valid.extxyz`, with 974 structures. Preserve
each frame's atomic row order and read `record_id`. The membership sidecar
maps those IDs to the five validation connections. Atom symbols determine
Z; coordinates are Å and forces eV/Å. Do not rotate, translate, reorder or
project predictions for a parity comparison.

`predictions/final-valid-predictions.npz` contains:

| Field | Shape | Meaning |
|---|---|---|
| `record_ids` | `(974,)` | Exact record identities |
| `energy` | `(974,)` | Residual molecular energy, eV |
| `forces` | `(974,21,3)` | Float64 forces, eV/Å, original atom order |

Align input, membership and prediction rows by exact record ID, not by
assuming their storage orders agree. Reject duplicates or missing IDs.
The saved prediction file SHA-256 is
`22bf7d88815ae3a84ecb8717c497f11311f058c1a862f8efd0e08a5efda0d3cd`.

The model was trained with zero E0s and residual energies:

`E_absolute = E_residual + (-17647.258992926312 eV)` per molecule.

`REF_energy` in the extxyz is already residual. Do not subtract the offset
again. The force arrays need no energy offset. The export receipt also
inventories training files omitted from this validation-only packet;
`MANIFEST.json` defines the packet's actual members.

## Completion evidence and execution boundary

Your PR should include loader code, synthetic refusal/identity tests,
environment/source admission, documented commands and retained failure
evidence. Author your own commits. The packet identity receipt extracts
facts from the original accepted report and records its original hash;
the extracted receipt has its own distinct packet hash.

Prepare the loader first. Actual checkpoint inference and validation
parity follow the midterm draft under a reviewed record list, tolerances
and resource budget. Once parity is valid, coordinate the unchanged
procedure on the same ten validation starts and six own-reference
controls with Olufunke. Persist checkpoint identity separately from the
frozen runner's limited arm audit fields.

Do not substitute the secondary selected model or Run B, retrain, add
seeds, tune the solver or score test data. MACE's predictive reference is
established; its relaxation recovery is still unmeasured.

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
