# Olufunke: Stage 2 report generator (plan milestone M5) handover

2 November 2026. Shedrack assigned this milestone to you. It is the reporter side of the
Stage 2 movement-cap study, the part of your evaluator and reporter work that turns
saved rows into labelled tables. Nothing here is a completed contribution until it is
reviewed and recorded in the [contribution ledger](CONTRIBUTIONS.md).

## The study in one paragraph

All five accepted E3x checkpoints recovered 0 of 10 original validation starts under the
frozen relaxer, and the relaxer accepted per-atom jumps of up to 436.7 Å. Stage 2 asks two
things: does a per-step movement cap (0.05, 0.10, 0.25 and 1.0 Å, plus the uncapped control)
help the relaxer, and do consistency-trained models then relax to their reference minima
better than direct-force models? Five checkpoints are primary (energy C seed 0, direct seeds
0 and 1, consistency λ = 0.1 seeds 0 and 1); two λ = 1 consistency checkpoints are a
secondary, exploratory, post-hoc arm. The design, rules and claim limits are in the spec
(section 8 in particular) and the decision log in the bundle. **No real-model run exists
yet.** Everything you build is tested on synthetic tables and on the saved historical
results.

## Access and first actions

The code is on `stage2-movement-cap`, at commit **`5ea6f61443f252b719a708e0c3a021db8b0a2b34`**
(this handover is the commit after it). It depends on no open PR. From an existing clean clone:

```bash
git fetch origin stage2-movement-cap
git switch -c olufunke-stage2-report origin/stage2-movement-cap
```

If you already have uncommitted work or that branch name exists, keep your work and use your
existing branch.

Ask Shedrack for **`stage2-olufunke-handoff-2026-11-02.tar.gz`**, about 0.98 MB (981,315 bytes), SHA-256
`d859d7930353b863d05deac41853c013d740986e982808d53741990b7acc2198`. It holds
the spec, the plan, the decision log, and a read-only subset of the accepted historical
results (50 original-start rows and 30 reference-start relaxations). It contains no
checkpoint, no model, no test partition. Verify the digest, extract into a new folder, then:

```bash
python3 stage2-olufunke-handoff/verify_bundle.py
```

Do not commit the bundle or any saved result to your PR.

Environment and tests, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src:scripts python -m pytest -q -p no:cacheprovider stage2_capped/tests
```

Without the saved Group9 files, the tests that read them skip (the admission and driver
real-file tests). That is expected for you. To run your historical-table test, point the
code at the bundle: `export STAGE2_GROUP9_ROOT=<extracted folder>/group9-subset`. Do not set
`STAGE2_REQUIRE_REAL_FILES=1`; it turns every skip into a failure and is for the launch machine.

## What to build

`stage2_capped/report.py` and `stage2_capped/tests/test_report.py`. The report is built **from
saved rows only**: result rows (schema in `stage2_capped/row_schema.py`) and the label records
that `stage2_capped/rules.py` produces (`rules.analyze(rows)`). It never loads a model,
never relaxes anything and never opens the test partition. You choose the API; one shape that
fits is `render_report(rows) -> {section name: markdown}` plus a writer that refuses to
overwrite. Output must be byte-identical on a second run.

### Must print, beside every label

1. For each comparison (primary λ = 0.1; exploratory λ = 1) and each cap: the label; interpretable
   items out of 20; connections and own minima those items cover; the adequacy result (energy C
   seed 0 recoveries at the cap against its control count); reference-start recoveries per
   checkpoint out of 6; the set E; the caps listed "not evaluable".
2. **The favoured arm, always, including `direct`.** For a single-cap observation read
   `r_direction_holds_at_the_cap` and `favors_at_the_cap`; the sweep's `favors` is `None` there.
3. Beside every `direction stated`: the not-evaluable caps and each such cap's note (a cap with
   zero contrast no longer vetoes a direction; the reader must see why a cap is missing).
4. For a cap with status `zero_contrast`: "at <cap>, neither arm recovered any interpretable
   original-start item (0 of N)". Never write "the arms do not differ". Key this off
   `per_cap[cap]["status"] == "zero_contrast"`, not off `zero_contrast_caps` (which also lists
   floor and gate caps).
5. A placebo-flagged cap prints "not distinguishable from seed noise" and is never an informative null.
   A floor or gate result is never a null.
6. The three `caveats` strings from the sweep record (five connections, not ten independent
   starts; the sweep is a robustness check on cap choice, not replication; descriptive difference only).
7. Attribution: match on `base_status`, never on `status` (it carries a suffix when λ = 1 runs
   contribute); print `lambda1_included` beside it; list `non_qualifying_flips` with the failed clause.
8. Everything from λ = 1 carries the exact text `exploratory, post-hoc, not MAE-matched`, in every
   table and caption, and a test must prove it.
9. Wording about consistency versus direct: "descriptive difference". Never "effect" or
   "significant". The unit is five connections. (The spec's attribution phrase, "consistent with
   step-size effects", is quoted verbatim and is the one allowed exception; `test_rules.py` shows
   how the project tests this.)
10. Per cap and checkpoint: travel and rejection counts, the fraction of capped steps and first
    capped attempt (spec section 3 asks for them), and for the controls the fraction of accepted steps
    above 0.25 Å and above 1.0 Å and whether c = 1.0 binds anywhere (if it never binds, say that
    `c1p00` duplicates the control).

### The historical table

Reproduce the accepted 50-row original-start recovery table (five models, ten starts each, 0 of 10
for every model) **from the accepted records, read-only**, and pin it in a test. The loader
`stage2_capped/stage0_compare.py:load_saved_original` reads exactly those rows. This is the check that
the reporter agrees with what was accepted before any new number is produced.

## What not to touch

- `stage2_capped/rules.py` is signed off against its SHA-256. Do not edit it; a change voids the sign-off. If you
  need a field it does not emit, ask Shedrack. Likewise `row_schema.py`, `relax_capped.py`,
  `restore_fork.py`, `admission.py`, `driver.py`, the diff gates and their allow-lists.
- Everything under `evaluation/`, `src/`, `configs/` and the pinned helper scripts is frozen. Your own
  `evaluation/reporter.py` can be imported; do not modify it.
- No model loads, no relaxation, no test-partition access, no network. Use the synthetic tables in
  `stage2_capped/tests/table_builder.py`; they are shaped like the real study (ten starts, five
  connections, six reference starts, seven checkpoints).
- Do not report a measured model result: none exists. Keep illustrative text clearly labelled.

## Verification we expect

- Tests that can fail: for each printing rule above, show it fails when the rule is broken (change the
  code, watch the test fail, restore it). The project does this for every module and records the
  count of broken variants caught.
- Determinism (two runs, identical bytes) and a test that reading rows never mutates them.
- A table with no rows, an incomplete table and a λ = 1-only table each give a clear refusal or an
  explicit "incomplete, not labelled" line, never a silent partial report.
- Keep the diff small and in your own files.

## Process

Use your own branch and commit identity. Open a PR into `stage2-movement-cap` when it is ready;
Shedrack arranges the two independent technical reviews TEAM.md requires (one on the scientific
claims and wording, one on implementation and verification) and merges after both accept. Record the
accepted PR, commits and checks in the contribution ledger afterwards. Proposed assignment is not
completed credit.

**Proposed return date: 8 November 2026**, so the traceability matrix and the full synthetic dry run
(M6, 6 to 10 November) can include the report. This is a proposal; confirm the date with Shedrack.
Questions at any point: ask Shedrack before changing an interface.
