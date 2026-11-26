# Branch references for endpoint evaluation

Use `reference_class` from the committed endpoint classification to choose each branch's target. The split's legacy `endpoint_pair` contains transition-state filename tokens. It has no guaranteed plus/minus orientation and is deprecated for target selection. It remains unchanged to preserve the frozen split identity, partitions, frame order and paired start IDs.

`manifests/landscape17-aspirin-branch-references.json` supplies all 74 explicit branch targets. Its reader checks the actual split and classification file hashes, split identity, companion identity, terminal record binding and every target. A modified target still fails verification if its companion identity is recomputed. Use the protocol's trusted source files; this check establishes consistency with committed classifications, rather than independently repeating their geometric matching.

```python
from force_consistency.data.endpoint_references import (
    load_reference_manifest, reference_for_branch,
)

references = load_reference_manifest(
    "manifests/landscape17-aspirin-branch-references.json",
    "manifests/landscape17-aspirin-split.json",
    "reports/landscape17-endpoints.json",
)
plus_target = reference_for_branch(references, "aspirin/12/plus")
minus_target = reference_for_branch(references, "aspirin/12/minus")
```

Targets are fully qualified minimum record IDs. Unknown keys and missing, unmatched, ambiguous, incomplete, unresolved or cross-molecule classifications raise an error. There is no filename fallback. `reference_record` is the matching representative and may identify a different equivalent minimum; the canonical target remains `reference_class`. Representative metadata is preserved separately.

| Connection | Filename tokens | Classified plus | Classified minus |
| --- | --- | --- | --- |
| 12 | 1, 2 | min_2 | min_1 |
| 44 | 4, 4 | min_4 | min_10 |
| 66 | 6, 6 | min_9 | min_6 |

Across 37 connections, 23 filename pairs agree with the branch classes, 12 are reversed, and two contain different classes. Positional lookup therefore selects the wrong reference for 26 of 74 branches. These are interface discrepancies, not new model results. Connections 00, 11, 22, 55, 77 and 88 have equal classified targets for both branches. Repetition alone is not a reason to reject a connection; use the recorded classification status and matching policy.

The companion inherits the classifications' geometric tolerances and completeness limits. It does not establish continuous path behaviour, independently certify minima, or change data roles. Existing preparation uses classified references already; no new prepared arrays or training results are introduced by this repair.

Check the committed companion:

```sh
uv run --frozen python scripts/build_branch_references.py --check
```

To reproduce it, select a fresh output path with `--output`. Generation refuses to overwrite an existing file. Changing either source requires a reviewed protocol change and a new companion. Preserve old evaluation evidence.
