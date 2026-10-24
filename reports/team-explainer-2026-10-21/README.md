# Group 9 meeting explainer, 21 October 2026

Read [the 25-page PDF](meeting-brief.pdf) for the team and professor discussion. Start with pages 2, 9, 12, 16, 19 and 20. Pages 21-22 give questions and answers. The guide includes successful, failed, mixed and incomplete work. It distinguishes a historical training plateau, the zero-recovery floor, and advice about research positioning.

This is a meeting guide, not the course-format midsemester submission. The [five-page midsemester draft](../midsemester-2026-10-18/README.md) retains its own release checks. Grader access, member contribution approval and live course instructions still need attention.

The completed scalar fits are included. Saved-result identity and arithmetic passed independent checks; the broader scientific interpretation remains limited to the fixed recipe and needs independent result review. No trained main-test or MACE recovery result is implied.

## Files

- `meeting-brief.pdf`: searchable, linked, A4 meeting guide.
- `brief.md`: editable reading text with the same page order.
- `brief.json`: structured source for the exact PDF layout.
- `SOURCES.md` and `sources.json`: exact source paths, primary links and hashes. Original and shared exports have separate identities.
- `build.py`: source-to-PDF renderer. It uses Python and ReportLab. The supplied PDF uses Arial Unicode from the local Mac; other systems need a font with the necessary scientific and author-name characters.
- `verification.json`: final artifact, page, length and source checks.

## Language

The brief uses ASD-STE100 Issue 9 writing rules and scientific terms given in the glossary. The checks include short sentences, paragraph length, consistent terms and a bounded vocabulary/meaning review. This is not independent language-conformance certification or ASD endorsement. This package does not include the official standard or its dictionary.

## Rebuild

From this directory, use `python3 build.py`. Install ReportLab in an appropriate document environment first. This command does not fit a model, score a checkpoint, change a training environment or access the numerical main-test data.

The source snapshot uses shared commit `3fa5969e5a0c2fdeaebc76a05bace20799715203`. Later publication of this guide does not alter its underlying experiment identities.
