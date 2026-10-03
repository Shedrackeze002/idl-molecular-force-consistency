# Force-isolation scientific plan followup — 3 October 2026

Verdict: **PASS for final amended design SHA256 `1a61f0087b0baf147d07e8a5e5ed5dca02199910195bcf21f15e58132625ba21` and final amended plan SHA256 `dbac79dd3d57f5599094cb8abe9e13661d5e596ecfcfb53af64c359b0ab5b1ce`.** The initial scientific review remains unchanged. Its fixture clarification is resolved in the specification and Task2 requirements. No remaining material scientific or evaluation issue was identified in this bounded followup.

## Exact amended scope

Compared both final files against the checksum-verified preserved copies under `reports/execution/force-isolation-2026-10-03/drafts/before-amendment-A/`. The exact differences are one insertion in each document, with no deletion or replacement. They explicitly require nonzero readout kernels, demonstrated nonzero shared-body force-gradient norms, actual nonzero incoming body Adam moments, and retained per-block assertion/norm evidence in the future energy-label independence and gauge-invariance fixtures. The plan also specifies readout-kernel trainability and scalar-bias-only projection. The specification labels these requirements future engineering work, not measured results.

This resolves the zero-head risk: an initial state with zero readout kernel can make the body force-gradient comparison trivially zero. Demonstrating nonzero body gradients and incoming moments makes preservation and label/gauge independence checks meaningful. These checks still must be implemented, run, and independently reviewed later; this followup does not claim they have passed.

The remaining scientific protocol and interpretation are byte-for-byte unchanged: energy coefficient0, fixed force coefficient1.9854156343296092, same original source-zero state and exact projected-control identity, fitting-energy calibration, width32/rate0.0003/6000updates, same fitted support/normalization, strict original-baseline force-only and separate joint gates,63/62 grid INCLUDING3001, once-only child/oracles/budget, adverse energy/group/variance reporting and one-seed/historical/all-fitted/sparse limits. Thus the earlier scientific assessment and saved-only context checks remain applicable.

The amendment record includes the append-only LIT007 section3.3 locator correction and preserved review lookup/assertion attempts. This corrects evidence navigation without expanding a literature claim or altering old review/log bytes. Source-map and proposed-settings hashes remain identical. The previous 18,963-source audit was not repeated because this followup changes only prospective fixture requirements, and its source map is unchanged.

## Hash-bound evidence and limitations

| Artifact | SHA256 |
|---|---|
| Final design | `1a61f0087b0baf147d07e8a5e5ed5dca02199910195bcf21f15e58132625ba21` |
| Final plan | `dbac79dd3d57f5599094cb8abe9e13661d5e596ecfcfb53af64c359b0ab5b1ce` |
| Amendment-A record | `e60cafd5c55106d2ae265ac26b604c8c6fa44f548254728f717ccce1be2d6846` |
| Unchanged proposed-settings | `48d6bd6070904d348f250e1a9e8fd4e09cfe0fe44685af9b195e4c749fcfe2f8` |
| Unchanged source map | `eda3ae20c34d16a5fea4260d6a15a5ac31356e2a384df00e4dd08e29a0e30664` |
| Preserved initial scientific review | `445ee46a7d06cc425ac6ea1485757ea0fa3433b0e5e15366cdf0a57b631b792b` |

Exact diff/check results and artifact hashes are retained in `reports/reviews/force-isolation-scientific-plan-2026-10-03/followup-amendment-A-verification.json` and `followup-evidence-manifest.json`. This followup made zero model calls, predictions, parameter initializations, training updates or test reruns. Only this separate review and own followup evidence were written; primary documents, initial review, old sources/raw artifacts, evidence log, user PDFs and Obsidian were untouched.

The proposal remains UNEXECUTED. Two final plan reviews enable the primary's next recorded planning decision; implementation verification, independent implementation followup and the complete frozen prelaunch gate remain required before any public numerical invocation.
