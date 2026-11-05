# Idris: MACE loader and reproduction handover

Created 18 October; access instructions updated 28 October 2026. The shared calculator bridge is ready. Your remaining
contribution is the checkpoint loader, model/environment admission and
independent reproduction. Olufunke owns integration with the relaxation
runner and reporter. No repeat training or additional seeds are needed.

## Access and first actions

**Complete assignment:** use the [end-to-end task brief](IDRIS_MACE_TASK_BRIEF_2026-10-28.md) for milestones I01–I08, implementation contracts, acceptance checks and final delivery. The Windows quickstart below covers setup.

**Windows users: start with the [Windows quickstart](WINDOWS_MACE_QUICKSTART_2026-10-28.md).** Its first milestone is packet verification plus Windows/WSL details. The proposed CPU setup and its known source-only dependency are documented there. A Mac is not required to begin onboarding or loader development.

The packet is now available through the [GitHub release](https://github.com/Shedrackeze002/idl-molecular-force-consistency/releases/tag/mace-idris-packet-v2-2026-10-28). Download [the original v2 archive](https://github.com/Shedrackeze002/idl-molecular-force-consistency/releases/download/mace-idris-packet-v2-2026-10-28/mace-idris-handoff-2026-10-18-v2.tar.gz), [checksums](https://github.com/Shedrackeze002/idl-molecular-force-consistency/releases/download/mace-idris-packet-v2-2026-10-28/SHA256SUMS), [current instructions](https://github.com/Shedrackeze002/idl-molecular-force-consistency/releases/download/mace-idris-packet-v2-2026-10-28/DOWNLOAD_README.md) and [dataset attribution](https://github.com/Shedrackeze002/idl-molecular-force-consistency/releases/download/mace-idris-packet-v2-2026-10-28/ATTRIBUTION.md).

The archive is 15,504,217 bytes and contains the primary model, validation inputs, memberships, saved predictions, environment requirements and verification scripts. It contains no training or test partition. Its accepted bytes and digest are unchanged.

The [accepted 26 October specification](../research/MACE_LOADER_PARITY_SPEC_2026-10-26.md) and [implementation plan](../superpowers/plans/2026-10-26-mace-loader-parity.md) supersede earlier checkout, environment-rebuild and future-plan directions. Private-transfer references inside the immutable archive are historical.

The owner-development source is pinned at `adf1b13819015cbfb1f2fbb79a79df132e7a4545` on `post-midterm-mace-readiness`. The older release source predates the Windows configuration; the full task brief distinguishes each revision. From an existing clean clone, fetch it and create your own branch:

```bash
git fetch origin refs/heads/post-midterm-mace-readiness:refs/remotes/origin/post-midterm-mace-readiness
git switch -c idris-mace-loader adf1b13819015cbfb1f2fbb79a79df132e7a4545
```

If you already have uncommitted work or that branch name exists, preserve your work and use your existing branch rather than resetting it.

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

Prepare the loader under the accepted specification first. Implementation reviews and explicit bounded launch authorization precede real checkpoint inference and validation parity. Recovery remains a later, separately reviewed work package with Olufunke. Persist checkpoint identity separately from the frozen runner's limited arm audit fields.

Do not substitute the secondary selected model or Run B, retrain, add
seeds, tune the solver or score test data. MACE's predictive reference is
established; its relaxation recovery is still unmeasured.

## Check the accepted environment

The accepted scorer ran on **macOS arm64, Python 3.12.13**, with its full identity in `ACCEPTED_IDENTITY.json`. The accepted 26 October numerical specification retains that scoring environment; it does not describe an executed attempt. Team onboarding now follows the Windows quickstart. That setup work is separate from numerical launch permission.

Run the packet's metadata-only `verify_environment.py` with the accepted environment's Python. Save its JSON output with the loader's separate admission receipt.

Use the proposed Windows/WSL setup recipe to collect platform and installation evidence. A reviewed, bounded platform amendment is still needed before real WSL model loading and prediction comparison. The archive's older environment-build commands do not override the current attempt's scope. The original lock-resolution platform remains unestablished.

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

The loader/parity plan is accepted. Idris authors the implementation and admission evidence; Shedrack coordinates two implementation reviews before explicit bounded launch authorization. This publication does not approve a numerical run.

The coordinator confirms Olufunke as runner/reporter integration owner. Her implementation progress and a member deadline are not established here.

The release establishes public availability. Idris must still confirm documentation access, download the packet and verify its checksum before access is recorded as complete. Publication is not proof of receipt, successful loading or prediction parity.
