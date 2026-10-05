# Data layout and preparation

Medical images are not in this repository: `scripts/download.py` fetches the
public ones, which remain under the terms of their datasets.

## Expected layout

```text
data/
├── raw/
│   ├── synthrad2023/Train/Task_1/{brain,pelvis}/PATIENT/
│   ├── synthrad2023/Train/Task_2/{brain,pelvis}/PATIENT/
│   ├── synthrad2025/Train/Task_{1,2}/{AB,HN,TH}/PATIENT/
│   └── sim_cbct/PATIENT.mha
├── transforms/
│   ├── synthrad2023-impact-registration/Task_*/REGION/PATIENT.txt
│   ├── synthrad2025-impact-registration/Task_*/REGION/PATIENT.txt
│   ├── synthrad2025-elx-parameters/param_def_{mr,cbct}_{AB,HN,TH}.txt
│   └── synthrad2025-elx-registration/Task_*/REGION/PATIENT.txt
├── processed/
│   ├── synthrad2023/Task_*/REGION/PATIENT/
│   ├── synthrad2025/Task_*/REGION/PATIENT/
│   └── external/Task_*/REGION/PATIENT/
└── splits/
    └── task_{1,2}/{Validation,CrossValidation_0,...}.txt
```

The raw 2025 case convention is lower-case `ct.mha`, `mr.mha` or `cbct.mha`
and `mask.mha`, exactly as in the Zenodo archives. The 2023 volumes use NIfTI
(`.nii.gz`). `scripts/download.py` extracts both into this layout.

Processed cases expose KonfAI groups:

- `CT.mha`: planning CT in the IMPACT reference geometry;
- `CT_ELX.mha`: CT warped with the organizers' ELX transform (the rigid CT for 2023);
- `MR.mha` or `CBCT.mha`: source modality for ELX training;
- `MR_IMPACT.mha` or `CBCT_IMPACT.mha`: source transformed with IMPACT;
- `MASK.mha`: evaluation/training mask.

## ELX registrations (SynthRAD2025)

The public archives contain the images but not the organizers' deformable
transforms. `scripts/preprocessing/register_elx.py` computes them with the
registration call of `stage2.py` in
<https://github.com/SynthRAD2025/preprocessing> (commit `8a5b125`): CT moving,
MR/CBCT fixed, patient mask as fixed mask, and the organizers' parameter file
for the task and region. The parameter files (GPL-3.0) are downloaded, not
vendored.

The transforms of the 169 held-out cases (Task 1: 66, Task 2: 103) are in the
Hugging Face repository `VBoussot/RegistrationBias-sCT`, under
`transforms/synthrad2025-elx-registration/`, and `scripts/download.py` fetches
them. Training cases are registered locally: one to seven minutes per case on
16 threads. A given SimpleITK-SimpleElastix build always returns the same
coefficients; builds `2.0.0rc2.dev910` and `2.4.0.dev67` differ by at most
1e-4 mm.

```bash
python scripts/download.py --only elx                               # parameters and held-out transforms
python scripts/preprocessing/register_elx.py --task 1 --region AB   # cases without a transform yet
```

## IMPACT registrations

- <https://huggingface.co/datasets/VBoussot/synthrad2023-impact-registration>
- <https://huggingface.co/datasets/VBoussot/synthrad2025-impact-registration>

These Elastix B-spline parameter files are released under CC BY-NC 4.0. The
centre whose data are restricted to challenge use is not included. The SynthRAD
images keep their own license.

## Preparation

`make pairs` runs `scripts/preprocessing/prepare_pairs.py` for every year, task
and region with the SimpleElastix environment. One region alone:

```bash
.venv-elx/bin/python scripts/preprocessing/prepare_pairs.py --year 2025 --task 1 --region AB
```

Cases already prepared are skipped.

## Split semantics

`Validation.txt` is the final held-out list. In training YAML, KonfAI's leading
`~` means that this list is excluded from the training subset. Each
`CrossValidation_i.txt` specifies the validation fold within the remaining
development patients.

A list holds the identifiers of every dataset of its task (SynthRAD2025,
SynthRAD2023 and the external sets); KonfAI keeps those present in the datasets
of a given configuration.

## External and simulated data

- **Ext-T2:** 24 expert-aligned T2 MRI/CT cases (Dowling et al., CSIRO), to
  request from their authors. `scripts/preprocessing/prepare_external.py
  --ext-t2 DIR` prepares them under `external/Task_1/CSIRO`.
- **Sim-CBCT:** 103 CBCT simulated from the held-out SynthRAD2025 CT by
  `scripts/preprocessing/cbct_synthesis.py` (RTK forward projection, noise,
  scatter, FDK reconstruction). They are in the Hugging Face repository
  `VBoussot/RegistrationBias-sCT`, under `sim-cbct/`, with the license of the
  SynthRAD2025 images they derive from (CC BY-NC 4.0). `scripts/download.py`
  fetches them and `make pairs` prepares them under
  `external/Task_2/{AB,HN,TH}_ood`.
