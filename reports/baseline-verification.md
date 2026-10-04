# Baseline physics verification

29 September 2026. Untrained seed-19 model, CPU, all coordinates and parameters float64. E3x 1.0.2 natural-parity features have one parity channel: degree zero is even scalar and degree one is polar vector; no pseudotensors. All two message passes retain degree one; only the energy head truncates to even scalars. Atom energies are summed per molecule in eV and force is the negative coordinate derivative in eV/A. The offset is reporting-only.

Configuration: 16 features, degree one, two passes, eight reciprocal-Bernstein radial functions, smooth 5 A cutoff, all ordered nonself intramolecular edges. Parameter count: 4737. Six physics tests passed in 6.69 s, including a nontrivial seeded state, O(3)/translation/permutation transformations, batch agreement, scalar/vector nonzero checks, cutoff limits and finite nonzero force-loss parameter derivatives. Symmetry/batch tolerance: absolute 1e-8, relative 1e-7.

Central-difference max-component force errors for h=1e-3, 1e-4, 1e-5 A: [1.5030025307671835e-07, 9.454223792726424e-09, 8.784594329735995e-09] eV/A. Net-force residual maximum: 8.326672684688674e-17 eV/A; net-torque residual maximum: 3.8163916471489756e-17 eV. Acceptance min difference error <=1e-5 eV/A was met. Exact numbers and lock identity are saved in `reports/baseline-physics.json`. These checks establish tested implementation behavior, not learned accuracy or the scientific hypothesis.

## Reviewed initialization amendment

The original seeded readout check above is preserved. After the failed first learning check, the energy readout now has explicit `lecun_normal` and `zero` options, with default random behavior preserved. Production zero initialization is separately tested to produce zero energy/force and finite nonzero readout gradients that learn nonzero forces. Physics checks now instantiate zero initialization and deliberately perturb the head with seed191, scale0.1, preserving the seeded nontrivial backbone. Seven physics tests and seven training tests passed (14 total in26.69s). Updated difference errors: [1.840800183117164e-08, 7.37725186494842e-09, 7.325904053528953e-09] eV/A; unchanged tolerances. See `reports/baseline-physics-zero-readout.json`. Optional NVIDIA dependencies were added to the same lock while CPU core versions remained unchanged; the new lock identity is recorded with this check.


Historical source hashes identify the original research artifacts. Raw run archives, original launch packages and historical audit records referenced here are retained separately; this report is a research export. See [the reproduction note](../docs/REPRODUCTION.md).
