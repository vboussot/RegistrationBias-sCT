# RegistrationBias-sCT

Code, configurations and patient-level results of **“When Misalignment Becomes
Supervision: Structured Label Noise in Supervised Synthetic CT Generation”**
([arXiv:2609.29387](https://arxiv.org/abs/2609.29387)).

The paper measures how the registration used to build paired training data
affects supervised MR-to-CT (Task 1) and CBCT-to-CT (Task 2) synthesis. It
compares Elastix-based pairs (**ELX**) with feature-based IMPACT-Reg pairs
(**IMPACT**), under MAE-only, VGG-perceptual and SAM-perceptual supervision.

## What you can do

| Goal | You need | Time | Start here |
|---|---|---|---|
| Read the reported numbers and build the tables from the patient-level metrics | Python, no data, no GPU | a minute | [Quick check](#quick-check-no-data) |
| Run prediction and evaluation with the 50 checkpoints | one 10 GB GPU, about 130 GB of disk | about a day and a half | [Reproduce the paper](#reproduce-the-paper) |
| Train the models | the training sets, more than 10 GB of GPU memory | days per model | [docs/REPRODUCE.md](docs/REPRODUCE.md#8-training-optional) |

## Quick check (no data)

```bash
git clone https://github.com/vboussot/RegistrationBias-sCT && cd RegistrationBias-sCT
pip install pytest pyyaml scipy

make test      # repository integrity
make tables    # tables/from_metrics/: Tables 4, 9 and 14-19 from the patient-level metrics
```

## Reproduce the paper

One file per step. Each command is detailed in
[docs/REPRODUCE.md](docs/REPRODUCE.md), with what it needs, what it writes and
how long it takes.

```bash
python scripts/download.py            # 1. held-out images, Sim-CBCT, transforms, checkpoints (23 GB)
make pairs                            # 2. ELX and IMPACT pairs for every downloaded case, Sim-CBCT included
python scripts/predict.py --gpu 0     # 3. predictions: ensembles and single folds
python scripts/evaluate.py --gpu 0    # 4. evaluations: MAE/PSNR/SSIM, Dice, d_SAM, LPIPS, uncertainty
python scripts/compare_metrics.py     # 5. your metrics against the paper's, patient by patient
python scripts/compare_tables.py      #    your tables against the paper's, cell by cell
```

Steps 3 and 4 skip what is already done, so they can be interrupted and
relaunched, and they take `--task`, `--mode`, `--loss` and `--fold` filters.
`--check` restricts them to 10 held-out cases per table row.

Tables and figures each have their own script:

| Result | Script |
|---|---|
| Table 4 | `scripts/tables/table_04_uncertainty.py` |
| Table 9 | `scripts/tables/table_09_registration_rows.py`, `table_09_registration_dice.py` |
| Tables 10–13 | `scripts/tables/table_10_13_leaderboards.py` |
| Tables 14, 15 (and 2, 3) | `scripts/tables/table_14_task1_registration.py`, `table_15_task2_registration.py` |
| Tables 16, 17 (and 5) | `scripts/tables/table_16_17_sam.py` |
| Tables 18, 19 (and 6) | `scripts/tables/table_18_task1_perceptual.py`, `table_19_task2_perceptual.py` |
| Figure 3 | `scripts/figures/figure_3.py` |
| Figure 4 | `scripts/figures/figure_4.py` |

The full correspondence between the paper and the files is in
[docs/PAPER_MAP.md](docs/PAPER_MAP.md).

## Repository layout

```text
scripts/download.py              step 1: every public input
scripts/preprocessing/           step 2: registrations and pairs; external sets; Sim-CBCT simulator
scripts/predict.py               step 3: predictions
scripts/evaluate.py              step 4: evaluations
scripts/compare_*.py             step 5: comparison with the paper
scripts/tables/                  one script per table
scripts/figures/                 one script per figure
scripts/registration_controls/   the registration controls of Table 9
scripts/train.py                 training
configs/                         training, inference and evaluation configurations
src/impactsynth/                 U-Net++, perceptual losses, evaluation metrics
data/splits/                     held-out and five-fold patient lists
models/manifest.csv              the 50 checkpoints, with size and SHA-256
results/raw/                     patient-level metrics of the paper
tables/                          the 19 tables of the paper; from_metrics/ holds those built by scripts/tables/
figures/                         the four figures of the paper
paper/                           source of the paper
tests/                           repository integrity tests
```

Images, predictions and model weights are not in the repository:
`scripts/download.py` fetches the public ones.

## Documentation

| File | Content |
|---|---|
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Step-by-step guide: requirements, commands, outputs, durations |
| [docs/PAPER_MAP.md](docs/PAPER_MAP.md) | Each table and figure of the paper: script and metrics |
| [docs/RESULTS.md](docs/RESULTS.md) | The result files of the repository |
| [docs/DATA.md](docs/DATA.md) | Datasets, layout on disk, splits |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Software versions and hardware notes |

## External test sets

Besides SynthRAD, Table 3 uses two independent sets
([docs/REPRODUCE.md](docs/REPRODUCE.md#7-external-sets-optional)):

- **Sim-CBCT** (Task 2): 103 CBCT simulated from the held-out CT with
  `scripts/preprocessing/cbct_synthesis.py`; step 1 downloads them.
- **Ext-T2** (Task 1): 24 expert-aligned T2 MRI/CT cases, to request from their
  authors; `scripts/preprocessing/prepare_external.py` prepares them.

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff).

## License

The code is released under the [Apache License 2.0](LICENSE). The SynthRAD
images, the third-party weights and the IMPACT transformations keep their own
licenses.
