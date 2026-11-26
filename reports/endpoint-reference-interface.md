# Endpoint reference interface audit and repair

Using the frozen filename pair as plus/minus targets disagrees with committed classifications for 26 of 74 aspirin branches. The connection counts are 23 agreeing, 12 reversed and two different, across 37 connections. Connection 12 reverses orientation; 44/minus and 66/plus have different classes. Equal target classes alone are not invalid: six other connections classify both branches to the same minimum class.

The separately versioned branch-reference companion and pure reader expose all 74 classified targets explicitly. Verification checks both source files, terminal records, unique matched classes and the entire companion. The original split and classification files remain byte-identical. Paired start IDs and partition assignments are preserved. The team handoff is also unchanged. Existing classification-driven preparation remains valid; the repair adds no physical classification or model result.

Validation: the original three data modules passed 32 checks before repair; the final data suite passed 60 checks, including 28 new reference-interface cases. Default collection retained all 243 original nodes and added 28, for 271 collected tests. The broader model-containing suite was collected only, not executed. No new model initialization, training or geometry matching was performed for this repair. The failure-first interface check (1 failure and 22 errors because the module did not exist) and intermediate 55-pass check are retained with final logs in the local evidence directory. Machine-readable validation is in reports/endpoint-reference-interface-verification.json.

Olufunke's evaluator branch was inspected separately and already uses classified references. Its tests were not executed here and its authored code was not changed. Two outdated 12/74 docstrings and the description of connection 66 as a reversal should be corrected in her own subsequent commit.

The first engineering implementation review exposed permissive Python JSON equality for boolean/float schema versions. Four adversarial cases reproduced two failures before the repair and all pass with strict integer-schema and independent actual-payload hashing. The original 56-pass receipt and this failed check remain retained.

The team-facing verification summary preserves counts, scope, timings and hashes. Exact machine-local commands, fixtures and full receipts remain separately retained; they are not included in this shared change.
