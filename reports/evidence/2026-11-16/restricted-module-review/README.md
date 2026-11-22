Verification of implementation 95830294179ef521560a6b8d6a631c832ae71dbf.

The WSL eight-file suite returned 228 passed, 3 expected skips, 45 warnings,
19 subtests passed, exit 0. Output is automatically captured with host roots
redacted. Synthetic modules and fake constructors only were used.

The isolated bootstrap exposes exactly mace_loader, mace_reference_adapter,
check_mace_parity, and evaluation.adapter through a restricted path-entry finder.
The evaluation parent is a sealed package created by the reviewed bootstrap;
neither a project nor an installed evaluation initializer executes.
No scripts/root/src directory is placed on the general import search path.
Python's standard PathFinder remains the meta-path finder.

Fresh-process temporary-venv regressions cover project-local e3nn, numpy, ase,
torch and scipy shadows in scripts/root/src across all three CLI entry points.
They also retain coverage for installed shadows of the project loader, bridge
and evaluation package, as well as startup hooks and package redirection.

The existing WSL proposal and its historical evidence are unchanged pending
review of this execution boundary. No environment approval or launch is claimed.
