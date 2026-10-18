# Midsemester report working draft

Started 18 October 2026 at the coordinator's request, superseding the earlier
20 October writing start. `main.tex` is the editable report; `main.pdf` is
the compiled draft. It uses the unmodified course `neurips_2019.sty`.
The old 15 October outline remains preserved separately.

This draft is for team review. It is not submitted to Canvas. Completed
results, failed experiments, diagnostic findings and pending work are
separated. No new model training, checkpoint inference or recovery was run
to prepare it.

## Rebuild

From this directory, using an environment with Matplotlib and NumPy:

```bash
python build_figures.py
tectonic main.tex
```

Figure rendering reads only the compact saved `figures/plot-data.json`.
The data records original source hashes. All 1,203 baseline trajectory
points were checked against the five per-connection sums/counts; their
BEST indices match the accepted independently scored checkpoints. The
maximum BEST metric difference is about 6.94e-18 eV/Å, within the explicit
1e-12 extraction check. This is report arithmetic, not a fresh rescore.

## Before submission

- Confirm the live Canvas deadline and any updated midterm instructions.
  The saved course notice says 23 October, 11:59 p.m. EST; this has not
  been verified against live Canvas. Do not infer a local-time conversion.
- Keep the report within the saved rubric's four-to-five-page requirement.
  The generic template's eight-page guidance does not override this.
- Each member reviews their section and confirms actual completed
  contributions and authored GitHub links. Assigned work is not completed
  credit. Confirm whether there is newer Emmanuel/Idris work to add.
- Check the two numerical figures, all table entries and the explicit
  changes from the submitted proposal against `EVIDENCE_MAP.md`.
- Resolve the report's working-draft label only after team/content approval.
- Ensure graders can reach the intended reviewed code/results revision.
  The results and MACE preparation PRs were open when this draft began;
  publication is not the same as merging into the default branch.

The course's maximum-five-minute video is a separate December deliverable,
not asserted here as a midterm requirement. MACE real-model admission and
recovery are still pending. This report does not require them to be
invented or silently launched.
