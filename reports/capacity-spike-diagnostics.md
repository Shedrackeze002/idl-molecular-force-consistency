# Restored capacity spike diagnostic

Execution valid: True; source integrity: True; status: completed.

| Saved state | Status | Fitting raw E MSE | Mean error² | Centered variance | Offset share |
|---|---|---:|---:|---:|---:|
| width32-u3600 | verified | 0.017976363969 | 5.78235602588e-07 | 0.0179757857334 | 3.216643830668664e-05 |
| width32-u3700 | verified | 0.135137953386 | 0.116441003234 | 0.018696950152 | 0.8616454542671367 |
| width32-u3800 | verified | 0.0173033360209 | 5.01501290983e-06 | 0.017298321008 | 0.0002898292504844046 |
| width32-u6000 | verified | 0.0119832084235 | 4.49113145612e-08 | 0.0119831635122 | 3.747853911407392e-06 |
| width64-u4900 | verified | 0.00955538949606 | 3.72609416459e-08 | 0.00955535223512 | 3.899468636129593e-06 |
| width64-u5000 | verified | 4.88250678443 | 4.87001681894 | 0.0124899654907 | 0.9974418949031484 |
| width64-u5100 | verified | 0.00932241419186 | 3.48508598595e-05 | 0.009287563332 | 0.003738394276658725 |
| width64-u6000 | verified | 0.00794833348223 | 2.42818897427e-08 | 0.00794830920034 | 3.0549661507039164e-06 |

Energy MSE and its components are in eV². Signed errors use prediction minus target. Groups, anchors, full geometry, verification and signed increment decompositions are retained in JSON.

| Saved state | Weighted E gradient norm | Weighted F gradient norm | Backbone cosine | Head-kernel cosine | Weighted head-bias E gradient |
|---|---:|---:|---:|---:|---:|
| width32-u3600 | 0.3414387795 | 0.4435961991 | -0.8505316385904642 | -0.8058647090830453 | 0.009998292322 |
| width32-u3700 | 56.94514316 | 17.09925361 | 0.6557963389656083 | 0.6669436522612661 | -4.486693322 |
| width32-u3800 | 0.3844773012 | 0.4772648473 | -0.4323070251319819 | -0.546230539569556 | 0.02944485661 |
| width32-u6000 | 0.285536557 | 0.3512180741 | -0.8105755561889156 | -0.6378900447613025 | 0.00278645029 |
| width64-u4900 | 0.5412478598 | 0.5642166279 | -0.951607681471604 | -0.6241680404213391 | -0.002538050909 |
| width64-u5000 | 624.6571621 | 139.6159149 | 0.7010270670585239 | 0.6306152003206111 | 29.01607376 |
| width64-u5100 | 1.756162923 | 0.5158980357 | -0.07882513404718931 | 0.007302574594378368 | -0.07762116647 |
| width64-u6000 | 0.4996845848 | 0.5277601181 | -0.9367141256230225 | -0.5625960209801922 | -0.002048871198 |

Posthoc maxima and saved neighbors; no continuous-time or causal mechanism claim.
All supports fitted; one seed/rate and width changes optimization geometry.
Gradients/slopes are scaled Euclidean derivatives, not historical Adam directions.
Directional finite differences do not check every gradient component. Full pilot pending.

Zero training updates, zero random parameter initializations, zero optimizer proposals. Restored-state prediction, derivative and finite-difference calls are separately counted in per-case work.json. Raw learning gates remain unchanged.


Historical source hashes identify the original research artifacts. Raw run archives, original launch packages and historical audit records referenced here are retained separately; this report is a research export. See [the reproduction note](../docs/REPRODUCTION.md).
