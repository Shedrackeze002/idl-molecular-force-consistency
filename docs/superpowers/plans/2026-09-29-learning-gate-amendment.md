# Prospective learning-gate amendment

Status: approved for native execution after two independent reviews on 29 September 2026. Both returned ready for execution with no remaining blocker. Parent plan: `2026-09-29-data-baseline-implementation.md`, Task 7. The first seed-0 check is retained as failed acceptance, not overwritten.

## Evidence and limit

The first 16-frame check reached 2000 updates with force MSE 0.00977934, above the fixed 10% threshold 0.00435718. Zero-force target MSE is 0.01013356, so the final model improves it by only about 3.50%. This is a weak force fit, not merely an inconvenient acceptance label. Physics checks pass; source/prepared labels match; adjacent energy/work diagnostics support the sign/scale; label net-force/torque residuals do not explain the error. No architecture defect or inadequate budget has been established.

On one prespecified four-frame diagnostic batch, scaled initial energy/force objective gradients have L2 norms 15064.9/19.53; at the last checkpoint 242.23/1.818. These norms motivate testing extensive-energy readout initialization; they do not establish optimizer causality. The official E3x force-field example initializes its energy readout to zero (`kernel_init=zeros`), unlike the first attempt's default random Dense readout. This is a proposed implementation initialization choice, not an accuracy claim.

## One prospective attempt

1. Add an explicit energy-readout initialization option (`lecun_normal` or `zero`) to model configuration, preserving the first attempt's default/recorded behavior. Keep the shared backbone, 16 channels, degree one, two passes, eight radial functions and 5 A cutoff unchanged.
2. Test that zero readout produces zero initial residual energy/force, permits finite nonzero readout gradients, and learns nonzero derivatives after an update. Re-run physical tests on a deliberately nontrivial seeded readout state even when the production initialization is zero, so symmetry/derivative tests cannot pass vacuously. Retain fixed numerical tolerances.
3. Run exactly one new seed-0 learning check in a distinct directory with zero readout, identical frozen 16 IDs and their order, Adam1e-3, batch4, maximum2000 updates/1800seconds and the unchanged requirement that both subset MSEs fall below10% of their own initial values. Initial force MSE is now the zero-force baseline, making the force gate stronger. Report both attempts, including zero/mean baselines and the initialization caveat. No seed, subset, loss-weight or threshold search.
4. Save the original subset hash as a required input and verify it before fitting. Learning-check checkpoint continuation is disabled in this amendment because it currently resets the acceptance baseline; generic pilot checkpoint resume remains tested. Any future learning continuation requires an immutable original acceptance baseline and separately reviewed cumulative budget.
5. Only if the second check passes may the original capped1000-update/3600-second aspirin pilot use this explicitly recorded initialization. Otherwise retain the failed gate and deliver audited data/code/diagnostics with pilot pending; no further training adjustment is authorized by this amendment.

## Verification and reporting

Tests RED then GREEN for the new initialization and subset/resume gates; full physical and training suite. Preserve old configuration, run artifacts, hashes and ENGINEERING-001. Record the new result as a linked evidence entry. A passing check remains an engineering check; initial-error reduction must be accompanied by absolute MSE and zero-force baseline. No comparative scientific inference follows.
