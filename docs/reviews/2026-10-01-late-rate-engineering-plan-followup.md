# Late-rate engineering plan follow-up

Date: 2026-10-01. Reviewed amendment A in the design, the added implementation checklists, and `2026-10-01-late-rate-author-rulings.json`. Follow-up scope is resolution of E1/E2/E3; no implementation, model evaluation or training occurred.

**Gate: the amended plan is ready for implementation from the engineering-review perspective.** All three important preimplementation gaps are explicitly resolved. No new material issue found. This does not clear unimplemented code or authorize production before the specified tests and final implementation review.

| Finding | Resolution verified in revised documents |
|---|---|
| E1: numerical contract substitution | Exact contract validation is required at launch/load/probe/report, with rate/weight/endpoint/observation tamper tests and config-hash receipt binding. Source-derived model, normalization and support must match parent. |
| E2: checkpoint ancestry versus new executable identity | Original provenance is preserved separately from new code fingerprints; all parent artifact identities reconcile. JSON normalization is explicit. Current executable/dependency/revision identities are recorded and verified in children and after execution, with corruption tests. |
| E3: abrupt work and summary-only completion | Durable records precede bounded blocks and nontraining work; exact returned counts remain distinct from unknown interrupted intervals. Orphans are retained without committed credit. Reserve, bounded child timeout and post-persistence caps are explicit. Reports reconstruct metrics/gates/counts from raw history, require complete observation/boundary/child identities, and suppress contrasts for partial or inconsistent arms. |

The amendment also resolves the nonblocking oracle clarification: both arms verify absolute6001 and8001 state/moment proposals against actual committed state, retaining incoming moments. State oracle tolerance is `atol=1e-10, rtol=1e-8`; source/endpoint predictions require absolute1e-10 with rtol0. Numerical arms, budgets, acceptance criteria and datasets remain unchanged.

Implementation review should confirm these are enforced in code, particularly nonzero-count boundary handling and durable publication order. The existing count0-only oracle must not be reused unchanged. No additional general execution machinery is requested.
