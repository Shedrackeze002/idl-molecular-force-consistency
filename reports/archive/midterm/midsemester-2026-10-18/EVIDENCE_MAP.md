# Report evidence and requirements map

All paths below are repository-relative. Published exports carry their own
hashes; original retained artifacts have separate identities. Figure data
records the hashes of the actual files used. The accompanying
`claim-sources.json` binds nonfigure claims to exact source hashes and
field/function pointers, distinguishing published exports from retained
original metadata. No model or archive inference
is needed to reproduce this report's tables and figures.

| Report claim | Source and definition |
|---|---|
| Authors, title and proposed study | Submitted Landscape17 proposal, retained by the coordinator; historical proposal is not a result |
| 26/5/6 connections; 4,891/974/1,127 records | `manifests/landscape17-aspirin-split.json`, prepared-data records and accepted baseline report |
| Endpoint assignment correction | `docs/team/ENDPOINT_REFERENCES.md`; classifications determine own reference, not legacy filename pairs |
| Model/loss/optimizer | `src/force_consistency/models/`, `objectives.py`, `train.py`; accepted controlled `config.json` and `summary.json` per arm; 43,521/43,649 parameters |
| C/B/A BEST errors, A1, zero-force value | `reports/evidence/2026-10-18/baseline/{C,B,A}/score.json` and `acceptance.json`; force-selected checkpoints, equal connection weighting |
| Figure 1, all 401 points per candidate | Original retained `runs/aspirin-e3x-v3-scientific-recovery-2026-10-15/{A,B,C}/validation.json`; compact exact column extraction in `figures/plot-data.json`; every point independently reduced from five group sums/counts |
| Two-seed force/energy means and ranges | `reports/evidence/2026-10-18/controlled/acceptance.json:two_seed_arm_summaries`; C0 reused plus nine subsequent fits |
| Figure 2 | Same accepted per-seed force scores; marks and means, no confidence intervals |
| Selection sensitivity and seen targets | `selection-sensitivity.json`: population SD over 101 correlated checks at updates150000–200000, nine final-minus-BEST gaps, exact matching gap and accepted original-branch seen flags; not measurement uncertainty |
| Matching rule | `scripts/launch_e3x_comparison.py:matching_decision`; mean gap bounded by 10% of smaller mean and each arm's seed range; not individual-seed matching |
| 1.985 fit-plus-score ratio | `reports/evidence/2026-10-18/controlled/completion-audit.json:rows`; sum two lambda0.1 fit+score times divided by sum two direct fit+score times; excludes diagnostics and other overhead |
| Finite-record consistency improvement | Same completion audit and eight saved diagnostic records; does not establish global conservativity |
| MACE 5.349 equal-connection, 5.260 pooled, 26.410 RMSE | `reports/evidence/2026-10-15/mace-runA/report.json`, primary final model only; external reference with unmatched architecture/exposure |
| Original 0/10, no-relax 0/10, A2 failed | `reports/evidence/2026-10-18/recovery/acceptance.json` and raw branch results; five validation connections, five specific checkpoints |
| 25 force-converged wrong geometries | Accepted recovery report/branch rows; not 50 independent observations |
| Actual relaxation rule | Frozen `evaluation/relax.py`, known-issues report and dated diagnosis addendum; caps scale, convergence first, force agreement, no energy-decrease test |
| 155 probes, 11-reference mismatch, endpoint flags/graphs | `reports/evidence/2026-10-18/diagnosis/acceptance.json`, `raw/SUMMARY.json`, raw diagnostic arrays and addendum; exact-reference mismatch does not exclude nearby valid points |
| Four reproduced traces, extreme jumps and six-reference controls | `reports/evidence/2026-10-18/paired-trace/result_acceptance.json`, accepted trace report and binary query journals; selected post-hoc diagnostic, not new A2 or population estimate |
| Test-access scope | `reports/evidence/2026-10-18/verification-scope.json`; prior model-free fixtures parsed test payloads, no trained test scoring or selection |
| Contribution statements | `docs/team/CONTRIBUTIONS.md`, original evaluator PRs and authored commits; remaining assignments require member confirmation |

## Saved midterm rubric coverage

The coordinator's saved grading PDF, pages 1-3, specifies a four-to-five-page
midterm in the required Overleaf template. Its live sheet was inaccessible
during preparation; confirm live instructions before submitting.

| Requirement | Draft coverage |
|---|---|
| Nonexpert abstract, motivation and objective | Page 1 |
| Relevant literature and baseline/metric connections | Page 1, five primary references |
| Inputs/outputs, data preparation, metric, loss | Page 2 |
| Implemented baseline, two or more iterations | Page 3, failed short fits and three baseline configurations |
| Original baseline outputs/plots and preliminary results | Figures 1-2 and predictive table |
| Challenges, failure analysis, limitations | Recovery/diagnosis and discussion |
| Two or three concrete next actions | Three next actions on final page |
| Bibliography, basic roles, GitHub link | Final page |
| Changes from proposal | Explicit final-page paragraph |

Working-draft review, contribution confirmation and live submission checks
remain open; an evidence-backed draft is not a submitted final report.
