# Result files

| Path | Content |
|---|---|
| `tables/table_NN_*.tex` | the 19 tables of the paper, written from `paper/arxiv_v1.tex` by `scripts/tables/published_tables.py` |
| `results/raw/Evaluations/` | 828 `Metric_TRAIN.json` files: one per model branch and region, with the value of every metric for every patient |
| `tables/from_metrics/` | Tables 4, 9 and 14 to 19 built from those files by `scripts/tables/` |
| `results/leaderboards/` | the public SynthRAD2025 test leaderboards of Tables 10 to 13 |

## Metric files

The tree is
`Task_{t}/Supervised/{MODE}/{FOLD}{KIND}/{CRITERION}/{REGION}/Metric_TRAIN.json`:

| Element | Values |
|---|---|
| `MODE` | training pairs: `ELX` or `IMPACT` |
| `FOLD` | `CV` for the five-fold ensemble, `CV_i` for one fold |
| `KIND` | empty for MAE, PSNR and SSIM; `_SEG` for the Dice; `_PERCEPTUAL` for `d_SAM`; `_PERCEPTUAL1` for LPIPS; `_Uncertainty` |
| `CRITERION` | training loss (`MAE`, `SAM`, `VGG`), with `_IMPACT` when the model reads the IMPACT input |
| `REGION` | `AB`, `HN`, `TH`, `brain`, `pelvis`; `ood` is Ext-T2 (Task 1); `{AB,HN,TH}_ood` are the Sim-CBCT (Task 2) |

`Registration_ELX_IMPACT{,_Perceptual,_SEG}/Task_{t}/{REGION}/` hold the Table 9
controls. A run of `scripts/evaluate.py` writes the same tree under
`results/generated/Evaluations/`.

`tests/test_repository.py` checks that every cell of Tables 14 to 19 of the
paper is in the tables built by the scripts.

## Comparing a new run

```bash
python scripts/compare_metrics.py     # every metric, patient by patient
python scripts/compare_tables.py      # every table, cell by cell
```

`d_SAM` and LPIPS are computed on one 512×512 frame per axial slice (`d_SAM`:
zero-padded or cropped at the end of the axes; LPIPS: centred).
