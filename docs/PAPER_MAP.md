# From the paper to the files

For each table and figure of the paper: the script that produces it and the
patient-level metrics it reads. The tables of the paper are in
`tables/table_NN_*.tex`, those built by the scripts in `tables/from_metrics/`,
the paper's metrics in `results/raw/Evaluations/` and the metrics of a new run
in `results/generated/Evaluations/`.

## The experiment in brief

- **Data.** SynthRAD2025 (abdomen, head and neck, thorax) for training and
  in-distribution tests; SynthRAD2023 (brain, pelvis), the 24-case Ext-T2 MRI set
  and CT-derived simulated CBCT for out-of-distribution and independent tests.
  15 % of the cases are held out; the rest is split into five folds.
- **Pairs.** `ELX`: the organizers' Elastix registration. `IMPACT`: the released
  IMPACT-Reg registration.
- **Model.** 2.5D U-Net++ with a ResNet-34 encoder, five adjacent axial slices,
  26,084,881 parameters (`src/impactsynth/models.py`).
- **Training.** 320×320 patches, batch 32, random flips, AdamW (learning rate
  1e-3, weight decay 1e-3), validation every 2500 iterations, learning rate
  ×0.75 every ten validations, best validation MAE checkpoint
  (`configs/training/`).
- **Losses.** MAE; MAE + SAM 2.1 Hiera-small features, layer weights
  `(0, 1, 1, 0)`; MAE + VGG as comparator, IMPACT pairs only
  (`src/impactsynth/perceptual.py`).
- **Inference.** Five folds × three views (original and two flips), voxel-wise
  mean (`configs/inference/`).
- **Metrics.** MAE, PSNR, SSIM, TotalSegmentator Dice, calibrated SAM distance
  `d_SAM`, LPIPS, variance of the 15 predictions (`src/impactsynth/metrics.py`,
  `scripts/evaluate.py`).

## Tables

| Table | Content | Script | Metrics (under `Task_{t}/Supervised/`) |
|---|---|---|---|
| 1 | Registration Dice, Rigid / ELX / IMPACT | — | |
| 2 | Registration consistency, aggregate | from Tables 14 and 15 | `{ELX,IMPACT}/CV/{MAE,MAE_IMPACT}` |
| 3 | Independent sets: Ext-T2, Sim-CBCT | `table_14_task1_registration.py`, `table_15_task2_registration.py` | same, regions `ood`, `{AB,HN,TH}_ood` |
| 4 | Prediction uncertainty, ELX against IMPACT | `table_04_uncertainty.py` | `{ELX,IMPACT}/CV_Uncertainty/MAE` |
| 5 | Dice by training loss, compact | from Tables 16 and 17 | `IMPACT/CV_SEG` |
| 6 | Perceptual metrics, folds and ensemble, compact | from Tables 18 and 19 | `ELX/CV*_PERCEPTUAL*` |
| 7 | IMPACT-Reg anatomical consistency | — | |
| 8 | Public SynthRAD2025 validation split | — ([note](#table-8)) | |
| 9 | Registration-induced metric sensitivity | `table_09_registration_rows.py`, `table_09_registration_dice.py` ([note](#table-9)) | `Registration_ELX_IMPACT{,_Perceptual,_SEG}/Task_{t}` |
| 10–13 | Official SynthRAD2025 test rankings | `table_10_13_leaderboards.py` | `results/leaderboards/*.csv` |
| 14, 15 | Registration consistency by region | `table_14_task1_registration.py`, `table_15_task2_registration.py` | `{ELX,IMPACT}/CV/{MAE,MAE_IMPACT}` |
| 16, 17 | MAE, VGG and SAM losses by region | `table_16_17_sam.py` | `IMPACT/CV/{MAE,VGG,SAM}_IMPACT`, `IMPACT/CV_SEG` |
| 18, 19 | Mean of the folds against the ensemble, perceptual metrics | `table_18_task1_perceptual.py`, `table_19_task2_perceptual.py` | `ELX/{CV,CV_i}/{MAE,SAM}`, `ELX/{CV,CV_i}_PERCEPTUAL{,1}` |

All table scripts are in `scripts/tables/`. Tables 1, 7 and 8 have no script;
like the others, they are in `tables/` as LaTeX, written from the paper source
by `published_tables.py`.

## Figures

| Figure | Content | Script | Needs |
|---|---|---|---|
| 1 | Study protocol | — | |
| 2 | ELX and IMPACT synthesis geometry | — | |
| 3 | Qualitative comparison of the losses | `scripts/figures/figure_3.py` | the prepared images, the predictions, and those of an MSE model |
| 4 | Qualitative comparison of the registrations | `scripts/figures/figure_4.py` | the prepared images |

## Which predictions feed which table

| Predictions (`scripts/predict.py`) | Evaluations (`scripts/evaluate.py --kind`) | Tables |
|---|---|---|
| MAE ensembles, both training conventions, both inputs | `image` | 2, 3, 14, 15 |
| MAE ensembles, both training conventions, ELX input | `Uncertainty` | 4 |
| IMPACT-trained MAE, VGG and SAM ensembles, IMPACT input | `image`, `SEG` | 5, 16, 17 |
| ELX-trained MAE and SAM ensembles and single folds, ELX input | `image`, `PERCEPTUAL`, `PERCEPTUAL1` | 6, 18, 19 |

## Notes

### Table 8

The ELX columns are the BreizhCT entries of the public validation leaderboards
([Task 1](https://synthrad2025.grand-challenge.org/evaluation/validation-task-1-mri/leaderboard/):
68.2033, 29.8113, 0.9268, 0.7238, 8.4204;
[Task 2](https://synthrad2025.grand-challenge.org/evaluation/validation-task-2-cbct/leaderboard/):
52.8789, 32.3624, 0.9655, 0.8312, 5.4006), truncated to two decimals; the "SSIM"
row is MS-SSIM. The IMPACT columns are submissions that are not listed on the
public leaderboards.

### Table 9

The "Reg." rows are CT-only controls on the 266 held-out cases (Task 1: 66
AB/HN/TH and 50 brain/pelvis; Task 2: 103 and 47). `CT_deformed` is `CT_ELX`
resampled with the IMPACT transform, scored against `CT` and left unmasked
outside the patient. For SynthRAD2025 this is an ELX-then-IMPACT cycle; for
SynthRAD2023, where `CT_ELX` is the rigid CT, a forward IMPACT warp. The steps
are in [REPRODUCE.md](REPRODUCE.md#6-table-9-registration-controls-optional).

- The "Sup." rows come from the leaderboards, the challenge paper and Tables 16
  to 19.
- `results/raw` has no LPIPS file for the Task 1 AB and HN controls; step 6
  writes them.
