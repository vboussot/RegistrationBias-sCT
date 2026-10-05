# Data layout

No medical images are in this repository. `scripts/download.py` puts the
SynthRAD cases under `data/raw/synthrad2023` and `data/raw/synthrad2025`, the
simulated CBCT under `data/raw/sim_cbct` and the transform parameter files
under `data/transforms/`; `make pairs` writes the
KonfAI datasets to `data/processed/synthrad{year}/Task_{task}/{region}/{patient}`.

`splits/` holds, for each task, the held-out list and the five validation
folds. The lists also contain the identifiers of the external sets; KonfAI
keeps those present in the datasets of a given configuration.

The layout and the preparation are detailed in `docs/DATA.md`.
