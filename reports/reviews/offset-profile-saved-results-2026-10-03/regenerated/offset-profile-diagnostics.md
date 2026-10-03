# Controlled energy-offset profiling

Execution verified: True. Learning gate: False.

Historical width32 comparison; original unprojected zero-state MSE baselines.

| Role | Metric | Historical endpoint fraction | Profiled endpoint fraction | Relative MSE change |
|---|---|---:|---:|---:|
| fitting | energy_mse_eV2 | 0.319409433487 | 0.290005768525 | -0.0920563448646 |
| fitting | force_mse_eV2_per_A2 | 0.465705272479 | 0.451005974878 | -0.0315635198274 |
| anchor | energy_mse_eV2 | 0.11607064219 | 0.114180555308 | -0.016283935773 |
| anchor | force_mse_eV2_per_A2 | 0.447115047671 | 0.423690622454 | -0.0523901517955 |

62 shared postzero observations; dense correction statistics reported separately.

Mean-zero fitting bias is enforced by construction.
One seed/rate, historical reference and changed optimizer algorithm limit causal claims.
All128 fitting examples and16 fitted anchors provide no held-out generalization evidence.
Sparse score observations do not prove continuous stability or convergence.
Dense intervention corrections have no matching dense historical reference.
Full pilot remains pending.
