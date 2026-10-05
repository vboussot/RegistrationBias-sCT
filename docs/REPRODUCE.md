# Reproducing the paper, step by step

Every command runs from the repository root. Steps 1 to 5 reproduce the
synthesis experiments (Tables 2 to 6 and 14 to 19) from the 50 checkpoints;
steps 6 to 8 are optional.

| Step | File | Writes |
|---|---|---|
| [1. Download](#1-download) | `scripts/download.py` | `data/raw/`, `data/transforms/`, `models/` |
| [2. Pairs](#2-pairs) | `scripts/preprocessing/prepare_pairs.py` | `data/processed/` |
| [3. Predictions](#3-predictions) | `scripts/predict.py` | `predictions/` |
| [4. Evaluations](#4-evaluations) | `scripts/evaluate.py` | `results/generated/Evaluations/` |
| [5. Comparison, tables, figures](#5-comparison-tables-and-figures) | `scripts/compare_*.py`, `scripts/tables/`, `scripts/figures/` | `tables/generated/`, `figures/generated/` |
| [6. Table 9 controls](#6-table-9-registration-controls-optional) | `scripts/registration_controls/` | `results/generated/` |
| [7. External sets](#7-external-sets-optional) | `scripts/preprocessing/cbct_synthesis.py`, `prepare_external.py` | `data/processed/external/` |
| [8. Training](#8-training-optional) | `scripts/train.py` | `artifacts/checkpoints/` |

## Before you start

| | |
|---|---|
| GPU | one GPU with 10 GB of memory |
| Disk | about 130 GB: 23 GB of downloads, 33 GB of pairs, 35 GB of predictions and up to 35 GB of temporary files during step 3 |
| Time | about a day and a half for steps 3 and 4 on an RTX 3080: about 20 hours of prediction and 13 hours of evaluation. `--check` (10 cases per table row) takes about a third of that |
| Accounts | none: every download is public |

Two Python environments are used. The main one runs everything except step 2:

```bash
conda env create -f environment.yml      # Python 3.12, PyTorch 2.7.1, KonfAI 1.8.7
conda activate registration-bias-sct
pip install -e .
```

Step 2 applies Elastix transforms and needs SimpleITK-SimpleElastix, which
cannot share an environment with SimpleITK:

```bash
python3.10 -m venv .venv-elx
.venv-elx/bin/pip install -r requirements-preprocessing.txt
```

Versions and hardware notes are in [ENVIRONMENT.md](ENVIRONMENT.md).

## 1. Download

```bash
python scripts/download.py --dry-run     # list what will be fetched
python scripts/download.py
```

| What | From | Size |
|---|---|---|
| The 266 held-out cases of SynthRAD2023 and SynthRAD2025 | Zenodo; extracted from the remote archives, which are not downloaded whole | 6.5 GB |
| IMPACT transforms of those cases | Hugging Face datasets `VBoussot/synthrad{2023,2025}-impact-registration` | 0.3 GB |
| ELX parameter files and ELX transforms of the 169 held-out SynthRAD2025 cases | SynthRAD2025 preprocessing repository; Hugging Face `VBoussot/RegistrationBias-sCT` | 0.2 GB |
| The 103 Sim-CBCT of the held-out SynthRAD2025 Task 2 cases | Hugging Face `VBoussot/RegistrationBias-sCT` | 0.5 GB |
| The 50 checkpoints, checked against `models/manifest.csv` | Hugging Face `VBoussot/RegistrationBias-sCT` | 15.7 GB |
| SAM 2.1 weights | Meta; Hugging Face `VBoussot/ImpactSynth` | 0.3 GB |

`--only data transforms elx sim-cbct checkpoints models` selects groups, `--task` and
`--year` restrict them. Files already present are kept.

## 2. Pairs

```bash
make pairs
```

This runs `scripts/preprocessing/prepare_pairs.py` for every year, task and
region with the `.venv-elx` environment (`make pairs ELX_PYTHON=...` for another
one). For each case it writes, under
`data/processed/synthrad{2023,2025}/Task_{1,2}/{region}/{case}/`:

| File | Content |
|---|---|
| `CT.mha` | planning CT, the target of the IMPACT convention |
| `CT_ELX.mha` | CT warped with the ELX transform, the target of the ELX convention (the rigid CT for 2023) |
| `MR.mha` or `CBCT.mha` | source image, the input of the ELX convention |
| `MR_IMPACT.mha` or `CBCT_IMPACT.mha` | source image warped with the IMPACT transform, the input of the IMPACT convention |
| `MASK.mha` | patient mask |

It then prepares the Sim-CBCT with `scripts/preprocessing/prepare_external.py`,
under `data/processed/external/Task_2/{AB,HN,TH}_ood/`: the simulated CBCT is the
input of both conventions and the CT their common target.

It takes about an hour. One region alone:
`python scripts/preprocessing/prepare_pairs.py --year 2025 --task 1 --region AB`.

### Computing the ELX registrations (optional)

Step 1 downloads the ELX transforms of the held-out cases. `register_elx.py`
computes them from the images, one region at a time; the training cases need it
before training with ELX pairs:

```bash
.venv-elx/bin/python scripts/preprocessing/register_elx.py --task 1 --region AB
```

Transforms already present are kept unless `--overwrite` is given.

## 3. Predictions

```bash
python scripts/predict.py --gpu 0
```

For each task, the script predicts:

| Models | Input | Predictions per case |
|---|---|---|
| ELX-trained and IMPACT-trained MAE ensembles | both conventions | 5 folds × 3 views (original, two flips), averaged |
| ELX-trained SAM ensemble, IMPACT-trained SAM and VGG ensembles | their own convention | 5 folds × 3 views, averaged |
| ELX-trained MAE and SAM single folds 0 to 4 | ELX convention | 1 |

Output: `predictions/Task_{t}/Supervised/{MODE}/{FOLD}/{CRITERION}/Output/{case}/sCT.mha`,
where `MODE` is the training convention, `FOLD` is `CV` for an ensemble or
`CV_i` for a fold, and `CRITERION` is the loss, with `_IMPACT` when the model
reads the IMPACT input. The ensembles used by the uncertainty analysis also get
`Uncertainty.mha`, the variance of their 15 predictions.

The script works one dataset at a time and skips what exists, so it can be
stopped and relaunched. During an ensemble it holds the 15 predictions of every
case of the current dataset (up to 35 GB for Task 1 thorax) and deletes them at
the end of the dataset.

Useful options:

```bash
python scripts/predict.py --gpu 0 --check                      # 10 cases per table row
python scripts/predict.py --gpu 0 --task 2 --loss MAE --fold CV # one part of the plan
python scripts/predict.py --dry-run                            # print the KonfAI commands
```

## 4. Evaluations

```bash
python scripts/evaluate.py --gpu 0
```

| `--kind` | Metric | Scored on | Feeds |
|---|---|---|---|
| `image` | MAE, PSNR, SSIM in the patient mask | every prediction | Tables 2, 3, 14, 15 and all others |
| `SEG` | TotalSegmentator Dice between sCT and CT segmentations | ensembles | Tables 5, 16, 17 |
| `PERCEPTUAL` | calibrated SAM distance `d_SAM` | ELX-trained MAE and SAM models | Tables 6, 18, 19 |
| `PERCEPTUAL1` | LPIPS | same | Tables 6, 18, 19 |
| `Uncertainty` | mean variance of the 15 predictions | MAE, SAM and VGG ensembles | Table 4 |

A prediction with the IMPACT input is scored against `CT`, the others against
`CT_ELX`. Output: `results/generated/Evaluations/`, one `Metric_TRAIN.json` per
branch and region, with the same tree as the paper's metrics in
`results/raw/Evaluations/`. Existing files are skipped; the options of step 3
apply (`--check`, `--task`, `--mode`, `--loss`, `--fold`).

## 5. Comparison, tables and figures

```bash
python scripts/compare_metrics.py     # every metric, patient by patient
python scripts/compare_tables.py      # every table, cell by cell
```

`compare_metrics.py` prints, for each branch and metric, the largest absolute
and relative difference with the paper's values. `compare_tables.py` rebuilds
the tables from your metrics into `tables/generated/` and prints, for each, the
number of identical cells and the largest difference. Rows of datasets you did
not evaluate stay empty. The metric files are described in [RESULTS.md](RESULTS.md).

One table at a time, from the paper's metrics (default) or from yours:

```bash
python scripts/tables/table_14_task1_registration.py
IMPACTSYNTH_EVALUATIONS=results/generated/Evaluations IMPACTSYNTH_TABLES=tables/generated \
  python scripts/tables/table_14_task1_registration.py
```

Tables 10 to 13 are the official SynthRAD2025 rankings:

```bash
python scripts/tables/table_10_13_leaderboards.py      # --offline for the saved CSVs
```

Figures 3 and 4 read `data/processed/` and `predictions/` and write to
`figures/generated/`:

```bash
python scripts/figures/figure_3.py     # also needs the predictions of an MSE model
python scripts/figures/figure_4.py
```

## 6. Table 9 registration controls (optional)

The "Reg." rows of Table 9 score a CT warped by one registration against the CT
aligned by the other, without any synthesis model.

```bash
# SimpleElastix environment: CT_ELX warped by IMPACT for the 266 held-out cases, then the dataset
.venv-elx/bin/python scripts/registration_controls/evaluate_ct_elx_impact_perturbations.py --tasks 1 2 \
  --regions AB HN TH brain pelvis --samples-per-level 0 --save-images \
  --output-dir results/generated/Registration/CT_ELX_IMPACT_baseline
.venv-elx/bin/python scripts/registration_controls/assemble_registration_dataset.py
# main environment: MAE/PSNR/SSIM, d_SAM, LPIPS and Dice
python scripts/registration_controls/evaluate_registration_controls.py --gpu 0
python scripts/compare_tables.py
```

## 7. External sets (optional)

Table 3 uses two independent sets besides SynthRAD.

**Sim-CBCT (Task 2).** Steps 1 and 2 download and prepare the 103 simulated
CBCT, and steps 3 to 5 include them. They are simulated from the held-out
SynthRAD2025 CT with `scripts/preprocessing/cbct_synthesis.py`, which needs the
RTK command-line tools (`pip install itk-rtk` and
`scripts/preprocessing/rtk_tools/`, or a compiled RTK). To simulate others, name
the input file after the case, from which the simulator picks the acquisition
settings of its centre:

```bash
mkdir -p sim_in
for case in data/raw/synthrad2025/Train/Task_2/*/*/; do cp $case/ct.mha sim_in/$(basename $case).mha; done
for image in sim_in/*.mha; do
  python scripts/preprocessing/cbct_synthesis.py --input $image --outdir sim --rtk-bin-dir scripts/preprocessing/rtk_tools
done
python scripts/preprocessing/prepare_external.py --sim-cbct sim --sim-suffix sim
```

`--sim-suffix sim` keeps this second set apart, under
`data/processed/external/Task_2/{AB,HN,TH}_sim/`: a new simulation is not
voxel-identical to the downloaded one.

**Ext-T2 (Task 1).** 24 expert-aligned T2 MRI/CT cases (Dowling et al.), to
request from their authors:

```bash
python scripts/preprocessing/prepare_external.py --ext-t2 DIR
```

Steps 3 to 5 then include it.

## 8. Training (optional)

```bash
python scripts/download.py --scope all                                # the full training sets, 52 GB
.venv-elx/bin/python scripts/preprocessing/register_elx.py --task 1 --region AB   # ELX transforms of the training cases
make pairs
python scripts/train.py --gpu 0 --task 1 --mode impact --loss mae --fold 0
```

`register_elx.py` is needed for the ELX convention only, for each SynthRAD2025
task and region; it takes one to seven minutes per case.
`prepare_pairs.py --without-elx` skips `CT_ELX` when only the IMPACT convention
is trained. KonfAI writes the checkpoints under `artifacts/checkpoints/`; the
paper keeps, for each fold, the one with the lowest validation MAE. Predict with
it through `python scripts/predict.py --checkpoints DIR`, where `DIR` has the
layout of `models/checkpoints/`.

The configurations ask for more than a 10 GB GPU and for about 165 GB of RAM;
see [ENVIRONMENT.md](ENVIRONMENT.md#hardware-notes).

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `miss ... CV_i.pt` in step 3 | A checkpoint is missing: rerun `python scripts/download.py --only checkpoints` |
| Step 3 or 4 stopped | Relaunch the same command: finished datasets and metrics are skipped |
| `SAM_Perceptual needs KonfAI >= 1.8.7` | An older KonfAI is installed: `pip install -r requirements.txt` |
| `This preprocessing step requires SimpleITK-SimpleElastix` | Step 2 was run with the main environment: use `.venv-elx` |
