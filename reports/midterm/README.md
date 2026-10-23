# Group 9 midterm report

This is the canonical midterm report. Read [the current PDF](group9_Midterm_IDL.pdf), edit [the source](group9_Midterm_IDL.tex), and use [team review assignments](TEAM_REVIEW.md) for remaining checks.

The current export has **five main-text pages and references on page 6**. MACE Run A's completed 2,048-epoch result is stated in the abstract, separated as a reference row in Table 1, explained in Section 5.3, and included in the conclusion. Its independently scored validation force MAE is 5.349 meV/Å. The table identifies MACE as a single-seed reference rather than an additional controlled E3x fit.

The comparison retains the different training budgets, architecture and hardware, and states that MACE recovery is unmeasured. All original E3x prediction and recovery table values, the loss, figure asset, course style and references are preserved. Paragraph consolidation retains the Introduction definitions and gives one Literature Review paragraph and two Data paragraphs. Figure 1 remains on page 3; Table 2 remains on page 5. Both main prediction results and the MACE subsection appear on page 4.

The current source compiles and all six exported pages were rendered and visually inspected. There are no overfull boxes, missing characters or undefined references. The current export has no layout warnings. [Verification](verification.json) records exact artifact identities and checks.

The report adds absolute scoped computing times, selection optimism, reference-force units and connected reference-start successes. The secondary energy/consistency-1 numerical match is explicitly post-hoc; consistency-1 recovery is untested. [Additional checks](additional-checks.json) bind these statements to saved source files. The figure legend uses the public 50 meV/Å criterion; [the figure builder](build_baseline_figure.py) reads the archived plot data without changing it.

Research code and evidence are linked to fixed snapshot `3fa5969e5a0c2fdeaebc76a05bace20799715203`. Its relaxation evaluator contains the reported acceptance tolerance and step-growth constants. This snapshot identifies the research artifacts; later report revisions are published separately in this directory. Earlier editions remain in the [midterm archive](../archive/midterm/README.md), with original files and hashes in the [relocation inventory](../archive/midterm/relocations.json).

Anonymous public access to the repository, cited research snapshot, evaluator and current PDF is verified; see [the access check](public-access-check.json). Remaining [release checks](RELEASE_CHECKLIST.md) are contribution confirmations, live Canvas instructions and deadline, and approval of the exact PDF. Page-limit compliance is established for this export; no submission is implied.
