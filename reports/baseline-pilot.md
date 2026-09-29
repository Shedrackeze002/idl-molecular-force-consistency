# Baseline milestone: learning gate still failed

29 September 2026. The data pipeline, frozen aspirin preparation, untrained physics checks and checkpoint/training contracts passed. **The real learning gate failed in both reviewed attempts; the full capped pilot has not run.** The first baseline milestone is incomplete. Paracetamol remains separately blocked by its unmatched endpoint. No test metrics were computed.

## Fixed-subset engineering results

Both attempts used seed0, the same16 frozen IDs from connection01, identical backbone/loss/Adam1e-3/batch4 settings,2000-update/1800-second caps and the same10% own-initial-error criterion. The second changed only energy-readout initialization, after two reviews of the prospective amendment. It used an unchanged subset SHA-256338e75a921b30820b70cfba47d1f06e7356a6d2232905d88975d088844403a3c.

| Quantity | Random readout | Zero readout |
| --- | ---: | ---: |
| Initial energy MSE, eV² | 32.2325607746 | 0.0612983585313 |
| Final energy MSE, eV² | 0.00775175850061 | 0.000489709457747 |
| Initial force-component MSE, eV²/A² | 0.0435718463615 | 0.0101335610811 |
| Final force-component MSE, eV²/A² | 0.00977933668687 | 0.00757216205811 |
| Force fraction remaining | 0.22444164 | 0.74723604 |
| Updates | 2000 | 2000 |
| CPU elapsed seconds | 26.599732 | 26.417110 |
| Explicit compilation seconds | 1.360633 | 1.378251 |
| Mean synchronized update seconds | 0.00359734 | 0.00354506 |
| Full validation seconds total | 7.303931 | 7.219315 |

The zero-force MSE is0.0101335610811 eV²/A². Final improvement over that baseline is3.4956% for the original attempt and25.2764% for zero readout. Therefore a large energy or random-initialization loss reduction does not establish adequate force fitting. The second check has a stricter absolute force gate of0.00101335610811, which was not met. Initialization alone did not resolve acceptance. More updates/capacity, conditioning or an implementation issue remain hypotheses; no extra tuning was run.

Both saved checkpoints reproduced predictions in a new process. Every primary validation pass covered974 records/61362 force components across5 groups, with equal-connection and pooled MAE both recorded. These validation metrics come from a learning diagnostic fitted only on16 frames; they are not full-pilot or full-training generalization results. Zero-readout full-subset MSE history is saved every100 updates; original history includes batch losses and initial/final full-subset MSE only. Reference-minimum diagnostics are empty because all11 minima are training-seen.

Device was JAX CPU on the local Apple M5 Max, float64,4737 parameters. GPU-hours and peak device memory are unavailable, explicitly marked in each summary. Initialization, compilation/warmup, steady updates, validation and total elapsed costs are separated. Core versions stayed pinned; the optional GPU runtime extra was added to the lock before the second attempt. Actual CMU/Colab/Kaggle execution remains unverified. This is an engineering cost record; no scientific speed comparison follows.

## Traceability and next gate

Original run: `runs/aspirin-learning-check-seed0`; prospective run: `runs/aspirin-learning-check-zero-readout-seed0`. Both preserve source/config/code/lock/split/unit identities, exact subset IDs, full validation coverage, last/best checkpoints, losses and new-process checks. Compact reviewable copies, all local artifact SHA-256 hashes, full-subset learning history and validation metric history are in `reports/baseline-learning-checks.json`. Large checkpoints remain ignored and retained locally.

The initialization amendment explicitly allows only this one additional attempt and leaves the full pilot pending if it fails. A future training adjustment needs a new saved plan with two independent reviews. Diagnose the weak force fit with a predeclared control before adding direct/consistency arms; retain absolute errors, trivial baselines and failed attempts. No model-comparison hypothesis or publication effect has been established.

## Follow-up diagnosis, 30 September 2026

The two-review diagnostic plan was executed with training-only access. A deliberately representable teacher passed at300updates, while the real-label full-batch control failed at2000updates with force MSE0.007240959200598679, fraction0.7145522825221747 remaining. Frozen-body force-only readout fits also left substantial error. See `reports/force-learning-diagnostics.md` and evidence DIAGNOSTIC-002/003/004. No further tuning or full pilot followed. These new outcomes supplement and preserve both original failures; they do not satisfy the original batch4 gate or establish the scientific hypothesis.
