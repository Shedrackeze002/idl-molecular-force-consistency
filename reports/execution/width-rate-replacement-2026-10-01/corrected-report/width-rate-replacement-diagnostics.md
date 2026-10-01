# Fresh replacement width/rate diagnostic

Status: completed; interpretation valid: True; full pilot: pending.

| Cell | Status | Observed persisted update | Fitting E fraction | Fitting F fraction | Fitting joint gate | Anchor joint gate |
|---|---|---:|---:|---:|---|---|
| width16-lr1e-3 | completed | 6000 | 0.3220330430785254 | 0.45318885284127763 | False | False |
| width16-lr3e-4 | completed | 6000 | 0.3419217539400875 | 0.5926503654720645 | False | False |
| width32-lr1e-3 | completed | 6000 | 0.20377621120957126 | 0.3898121837590496 | False | False |
| width32-lr3e-4 | completed | 6000 | 0.3194094334870974 | 0.4657052724792342 | False | False |

Four-cell contrasts available: True.
Measured elapsed seconds: 4295.419170167006.

Every raw/per-group/equal-group/overlap observation, objective, gate, counter, phase time and immutable source identity is retained in the JSON and raw files.
![Saved prescribed observations](figures/width-rate-replacement-curves.png)

- One seed and fully fitted anchors; no held-out/generalization, capacity or optimizer-causal conclusion.
- Fresh fixed-rate endpoints do not test late-stage rate decay. Width changes initialization shape and optimization geometry.
- Sequential times depend on order, compiler cache and load; no efficiency ranking.
- Original failed trial and zero-update copy stay separate; original unreceipted child effort remains unknown.
- Partial terminal counts are raw claims; known persisted observations and child work require independent saved bindings.
- No automatic winner or full-pilot clearance. All failures and blocked cases stay indexed.
