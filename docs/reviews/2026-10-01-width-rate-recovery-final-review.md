# Fresh final implementation review and author rulings

Review base 0e98c63, implementation b2ef4b0; independent reviewer recovery_final_review, static branch review and preserved opaque synthetic reproductions only. No reviewer numerical actions or suite run. Production outcome reconciliation deferred until the once-only result exists.

Verdict: no Critical, five Important, one Minor; changes required before production. Author assessed and accepted all five Important findings:

1. Root absent check followed by guard/mkdir was not exclusive ownership; fallback wrote whenever root existed. A raced directory or symlink could receive a failure record despite never being created by this controller. Author repair: capture successful exclusive directory inode/device and resolved location; persist locally only when that ownership remains safe, otherwise retain a named temporary external failure record. Regression asserts original synthetic source byte/file identities unchanged and no child launched.
2. Clock could cross deadline between final check and computing child timeout; a negative allowance still reached the runner. Repair: compute and require finite positive allowance before attempt count/issuance, and independently reject nonpositive/nonfinite allowance inside subprocess launcher. Regression proves zero child calls/work for deadline crossing.
3. Failed saved summaries checked only attempt count and accepted fabricated completed work and negative durations. Repair: all outcomes validate finite nonnegative duration/attempt/work state machine; completed work binds the saved child receipt and work receipt. Invalid claims remain quarantined as claimed terminal evidence; only independently bound available work is authoritative.
4. Missing/corrupt terminal summaries raised before reporting. Repair: explicitly incomplete retained-evidence report, unknown issuance/work for launch intent alone, known zero when no launch evidence, and completed work only with independently bound child/copy receipts. No invented elapsed time or success.
5. Failed/timeout subprocess output was discarded except a stderr tail. Repair: preserve full available stdout/stderr for either exception, preserving original exception and logging persistence notes, no retry. A completed saved child receipt before later log failure retains known validated work.

Minor accepted: actual parent and child start/end UTC timestamps alongside monotonic durations, explicitly identifying clock provenance. This is a requirement already present in the spec; no protocol/config/pin amendment.

Preserved reviewer reproductions: /private/tmp/width-rate-recovery-review-b2ef4b0/reproduce.py, results.json, ownership.py, ownership-result.json. Synthetic opaque metadata only. Additional author unowned-root external failure records retained in /private/tmp and will be archived with engineering evidence. Original source untouched.

Author engineering evidence: 17 new regression failures with 41 existing focused checks passing (11.53s); all 58 then passed (10.79s). Logs under .superpowers/sdd/2026-10-01-width-rate-recovery. Further checks/full suite and production result recorded below when executed. No second code review of the author fix pass; same reviewer completes only its deferred saved-outcome reconciliation.

Ruling: retain and fix all Important findings before production, as existing approved boundary implementation repairs. Config, 163 recovery pins, 326 historical pins, unchanged numerical verifier, one attempt and zero training remain fixed. No replacement trial or plan scope extension.

Author supplemental check:60focused checks passed11.15s; malformed nonfinite terminal JSON regression additionally reproduced RED. First full suite intentionally interrupted after358passes203.16s, preserved as incomplete. Strict terminal parser repair followed; final61focused checks passed11.36s. Final full suite pending; no production numerical action yet.

Final author verification:61focused passed11.36s; full388passed253.19s; full-logSHA256`f5be49a55a0e1712fc0efb41f6b84c8c6180e7905fa32d7108ff238f37006b29`.163recovery/326historical pins pass; no production attempt yet. One authorized copied verifier next, zero training.
