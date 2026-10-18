# Midsemester course-rubric audit — 18 October 2026

**Verdict: the current report satisfies the substantive document requirements in the saved course midterm rubric. It remains a working draft, not a submission-ready or submitted deliverable.** This is a requirements assessment, not a prediction of the grade.

Audited report: `reports/midsemester-2026-10-18/main.pdf`, five pages including references, SHA-256 `8e965fc0ae7b05fe60250132d80c2e8b529e20cc2d5ecef31057a9b1cbc44521`. The source, five rendered pages and embedded repository link were checked. The required `neurips_2019.sty` is byte-identical to the saved course template. The report PDF has not been changed by this audit.

## Requirement-by-requirement coverage

The following applies the **Midterm (4–5p)** column, not proposal or final-report criteria.

| Course midterm criterion | Current report evidence | Assessment |
|---|---|---|
| Length and required report template | Five pages including bibliography; unmodified required course style | Met |
| Problem understandable to nonexperts | Abstract explains accurate stored-example predictions versus wrong molecular shapes and the all-model recovery failure, page 1 | Met |
| Motivation and at least one objective | Cost of quantum-chemistry calculations; reference-shape recovery at comparable force error/cost, section 1, page 1 | Met |
| 1–2 key related papers and brief connections | Section 2 connects Landscape17 and nonconservative-force work to recovery; also explains E3x, MACE and consistency-loss background | Met; five relevant references |
| At least one paper connects a metric or baseline to the problem | Landscape17 motivates structural recovery; MACE supplies an external predictive reference, page 1 | Met |
| Basic model inputs and outputs | Atomic numbers/coordinates to scalar molecular energy and atomic vector forces; shared structure and parameter counts, section 3.2, page 2 | Met; an architecture diagram is encouraged, not mandatory |
| Identify data and basic preparation | Aspirin/Landscape17 citation, atom count, grouped splits and record counts, training-only normalization, target mapping and symmetry alignment, section 3.1, page 2 | Met |
| State metric and explain relevance | Equal-connection component force MAE and structural/force recovery criteria, section 3.3, page 2 | Met; predictive accuracy is explicitly distinguished from recovery |
| Mention applicable loss | Supervised energy/force and consistency losses, force heads, scales/weights and optimizer, equation 1, page 2 | Met |
| Evidence of at least two runs, adjustments or configurations | Failed 1,000-update fits, A/B/C baseline configurations and ten final-setting fits, section 4, pages 2–3 | Met |
| Mandatory implemented and explained baseline | Implemented energy-derived E3x baseline, its architecture/training settings and measured results; independently scored MACE is additional context | Met; pending MACE evaluator loading does not erase the implemented E3x baseline |
| High-level extensions/experiments | Direct and consistency arms plus declared MACE procedure control, potential solver revision and physical/cost work | Met; completed versus planned work distinguished |
| Initial baseline plots/results confirming own work | Original baseline trajectories in Figure 1, measured BEST values and baseline configuration comparison, page 3 | Met; saved output is bound to accepted records |
| Preliminary results/progress metrics encouraged | Two-seed table and Figure 2; original recovery and reference-start controls, pages 3–4 | Included |
| Challenges so far and risks moving forward | All-model zero recovery, disconnected endpoints, selected extreme steps, selection sensitivity, limited connections/seeds and unresolved generalization, pages 3–5 | Met |
| 2–3 specific next actions | Three actions covering loader/parity and unchanged MACE control, conditional solver revision, closed-loop work/latency and a separately budgeted wall-time plan, page 5 | Met |
| Bibliography, 2–3 citations minimum | Five cited references, all listed in bibliography, pages 1 and 5 | Met |
| Basic role breakdown | Four named members, completed work separated from remaining assignments, page 5 | Present; members must confirm actual contributions before release |
| GitHub link | Clickable PR #10 link, page 5; identifies reviewed stack #10 → #9 → #8 | Present; grader access remains open because repository is private |

## Additional integrity checks and limits

The report discloses departures from the proposal: two seeds rather than three, deferred paracetamol, revised adequacy protocol and redistributed responsibilities. It preserves negative results and limits post-hoc interpretation. A positive recovery result is not required by the saved midterm column; no extra model run is needed merely to satisfy its document criteria. No standalone conclusion section is required in that column. Error/failure and sensitivity rows carry final-report criteria; this draft nevertheless includes both kinds of analysis. Full final-project experiments and the separate December video are not treated as missing midterm deliverables.

## Submission checks still open

1. Confirm the live Canvas assignment, exact due time/timezone, upload/group instructions and any later changes. The saved course post states October 23, 11:59 p.m. EST. A course announcement posted October 17, read during this audit, independently confirms **October 23 as the midterm date**, but does not confirm the time or Canvas instructions. No timezone conversion is inferred.
2. Confirm actual contributions and authored evidence with all four members, then approve the final text and remove the working-draft label. A role assignment is not proof of completed contribution; the course notes allow GitHub and document histories to inform contribution assessment.
3. Ensure graders can access the PRIVATE repository and the specific reviewed version. As checked during this audit, PR #10 remains open on `midsemester-report-draft`, stacked on unmerged changes. Its baseline checked revision is `8ed977b718aae811e40ff76ba4bae3ba76a92509`; this audit is a subsequent documentation update. Do not assume default `main` includes these results. Final integration/tag/link and staff access need resolution before submission.

## Sources and verification scope

- Coordinator-retained course `11-785_Project_Grading_Information.pdf`, pages 1–3, SHA-256 `0ad90675f62911b5266ae2168fcec20e414fe43ada5ebf44a04b5f94a4447a71`: actual Midterm column and contribution note, visually checked. The course PDF itself is not republished here.
- Saved required `course_docs/overleaf_template/neurips_2019.sty`, SHA-256 `2db0d258efc74c644e02ae4694c94cefb2c84e97cb8e804a5911ae7a02523b42`, compared byte-for-byte with the report copy.
- Coordinator-retained updated course deadline post, SHA-256 `faa31e1f70e8353e68bf8723fdd96c8da27d42bcf215519aac38553614ea404e`; later October 17 announcement checked through the connected course inbox.
- Live rubric-sheet URLs and template link were attempted but unavailable through web retrieval. The current public course landing page exposed no usable assignment details. The October 9 guided-project notice was distinguished from this team's Path 3 project; its topic-specific deliverables were not imported into this audit.
- Existing source/claim verification and independent report reviews remain applicable to the unchanged PDF. This audit checks requirements; it does not rerun models, independently grade the project, confirm staff permissions or submit anything.
