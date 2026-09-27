# Landscape17 pathway sample

This folder contains **unmodified raw XYZ files** extracted from the public Landscape17 v1 archive. It is a small exploration sample for the current proposal, not the full training or evaluation dataset.

## Included files

- `aspirin/SteepestDescent/01/`: all 234 XYZ frames in archive path group `01`.
- `paracetamol/SteepestDescent/01/`: all 165 XYZ frames in archive path group `01`.
- Each molecule's `Conformations/` folder contains the archive's `coordsmin_0.xyz`, `coordsmin_1.xyz`, and `coordsts_0_1.xyz` reference structures.
- `SHA256SUMS` records the SHA-256 digest of every included XYZ file, relative to this folder.

Each frame contains atom species, three-dimensional positions, force vectors, and an energy in its XYZ header. No unit conversion or numerical modification was made during extraction. Path group `01` is an illustrative sample; it has **not** been designated as a training, validation, or test group. The final study must freeze whole-path splits and audit shared minima before fitting any model.

## Source and attribution

- Dataset: **Landscape17**, by Vlad Carare, Fabian Thiemann, Joe Morrow, David Wales, Edward Pyzer-Knapp, and Luke Dicks.
- Versioned record: [Figshare DOI 10.6084/m9.figshare.29949230.v1](https://doi.org/10.6084/m9.figshare.29949230.v1).
- Original file: `Landscape17.zip`, Figshare file ID `57341180`, 53,019,632 bytes.
- Original archive SHA-256: `3870ecfd6a5404c6654efba22f003f645e2f4dffffad1b5a22770d3437b3027b`.
- License shown on the Figshare record: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the original dataset and associated [Landscape17 paper](https://www.nature.com/articles/s41524-025-01878-x) when using these files.

The [full archive](https://ndownloader.figshare.com/files/57341180) is needed for the proposed whole-path experiment. Its other path groups are intentionally not copied into this sample folder.
