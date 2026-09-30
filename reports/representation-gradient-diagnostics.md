# Frozen representation and loss-gradient diagnostics

Execution status: **completed**; 8.961s. Learning acceptance remains false; full pilot pending.

The fixed16/16/128 training subsets, source2000-update checkpoints, model and original normalized loss were unchanged. Eight prespecified cases were attempted in order under600s; no optimizer update, training, test/validation access or label substitution occurred.

## Necessary force constraints

| Set | Zero-force MSE | Translation floor | Rigid floor | Original force gate | Obstructs gate |
|---|---:|---:|---:|---:|---|
| narrow16 | 0.0101335610811 | 4.64464121009e-09 | 7.74901342941e-09 | 0.00101335610811 | False |
| diverse16 | 0.0179396642219 | 5.86885208552e-09 | 9.52759197564e-09 | 0.00179396642219 | False |
| diverse128 | 0.0140839482181 | 6.15260455855e-09 | 9.03261621934e-09 | 0.00140839482181 | False |

MSE units are eV²/Å², pooled over force components. Equal-group values, per-frame net forces/centered torques, ranks and SVD/graph checks are retained separately in phase-A.json. The rigid floor is a numerical necessary-condition bound. A floor below the gate rules out these constraints alone as the explanation; it does not prove representability, integrability or sufficient capacity. A floor above it identifies incompatibility under the enforced symmetry, not corrupted labels or a particular calculator error. The raw labels and learning gates remain authoritative.

Cutoff connectivity and two-hop reach are descriptive support measures; they do not establish expressive sufficiency. The model sums scalar atom energies and obtains forces from negative coordinate derivatives, following the [official E3x force example](https://e3x.readthedocs.io/stable/examples/md17_ethanol.html); its benchmark accuracy/configuration is not project evidence.

## Frozen gradients

| Case | Status | Scaled E loss | Scaled F loss | E gradient norm | F gradient norm | F/E norm | Cosine | E slope | F slope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh-A-fitting | completed | 1.3157865 | 0.6294083 | 247.20812 | 0.58777878 | 0.0023776677 | 0.231787 | -61145.535 | -34.02499 |
| fresh-B-fitting | completed | 0.77752598 | 1.1142552 | 47.755156 | 0.98164429 | 0.020555776 | 0.40788804 | -2299.6761 | -20.084836 |
| fresh-C-fitting | completed | 0.80530789 | 0.87477184 | 72.411483 | 0.5319166 | 0.0073457494 | 0.35701823 | -5257.1741 | -14.03416 |
| final-A-fitting | completed | 0.0046266434 | 0.44974514 | 0.3148464 | 0.34601755 | 1.0990043 | -0.96459682 | 0.005957217 | -0.014642671 |
| final-B-fitting | completed | 0.020753515 | 0.99922031 | 0.8028158 | 0.7906237 | 0.98481333 | -0.99245995 | -0.014573868 | 0.0048535072 |
| final-C-fitting | completed | 0.070368275 | 0.85591203 | 1.3307488 | 0.90105063 | 0.67710046 | -0.70563313 | -0.92478742 | 0.034212736 |
| final-B-anchor | completed | 0.024194033 | 0.59086286 | 11.306375 | 1.5102467 | 0.1335748 | -0.35357627 | -121.79665 | 3.7566167 |
| final-C-anchor | completed | 0.032502583 | 0.55199084 | 5.1427581 | 1.0556031 | 0.20526011 | 0.3297979 | -28.238339 | -2.9046755 |

Norms, ratios and cosines depend on this parameterization and the fixed training scales. Slopes are local directional derivatives along ordinary **minus the joint gradient**, without normalizing that direction: −(||gE||²+gE·gF), −(||gF||²+gE·gF). Negative slopes imply a local first-order decrease for that objective. They are not the actual Adam direction, a training trajectory or a causal explanation. Zero-norm blocks have undefined cosine; the force head-bias gradient and initial zero-head backbone force gradient are structurally zero. Nonzero later derivatives do not establish adequate capacity.

Both central-difference steps1e-4 and1e-5 are retained for every completed objective; at least one must agree within1e-8+1e-4 relative tolerance. These checks verify selected directions, not every gradient component. The separate gradient sum is checked against the unchanged joint objective within1e-10/1e-8. Parameter and optimizer hashes before/after are retained.

## Measured observations

Initial energy-gradient norms exceed force-gradient norms by420.580,48.648,136.133times for A/B/C. At the final fitting snapshots the cosines are-0.964596819717,-0.992459954849,-0.705633134799. Under ordinary minus-joint-gradient motion, A's energy slope is positive while B/C's force slopes are positive. This is local objective opposition; no Adam direction or causal explanation was measured. The final C anchor has positive cosine, so conflict is not uniform across cases.

Failed individual finite-difference steps: final-C-fitting/energy/h=0.0001: error0.000224986181149. Every objective nevertheless meets the prespecified at-least-one-step criterion; all failed steps remain preserved.

All observed rigid ranks are6. Every geometry graph is connected, has no isolated atoms, and reaches all ordered nonself atom pairs within two hops at5Å (see companion JSON for exact ranges). These observations do not establish that this width/degree/message representation can express the target force field.

## Outcome and limits

Next decision: `separately_review_intervention_from_local_evidence`. Static observations do not clear the previous failed force gates. Any architecture, weighting, optimization, data-convention intervention or larger training run requires its own saved plan and two independent reviews. No additional attempt was selected after this result. Unresolved, failed or blocked observations remain in the raw artifacts and paper evidence log.

## Reproduction and evidence

Analysis artifacts: `runs/aspirin-representation-gradient-2026-09-30`; historical source run: `runs/aspirin-data-coverage-2026-09-30`. Frozen config: `configs/aspirin-representation-gradient-diagnostic.json`; source/prepared/split/subset identities and historical/current provenance are in the run artifacts. This report generator reads JSON only; it does not load data or execute a model. Raw observations and every partial/failure artifact are indexed by SHA256 in the companion JSON. See `reports/PAPER_EVIDENCE_LOG.md`, entries PLAN-GRADIENT/IMPL-GRADIENT/GRADIENT-*. Results are training-only diagnostics, with no held-out scientific comparison.
