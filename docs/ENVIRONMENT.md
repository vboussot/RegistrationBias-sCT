# Environment

## Versions

The repository runs with the versions pinned in `requirements.txt`:

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.7.1 (CUDA 12.8 wheels) |
| torchvision | 0.22.1 |
| KonfAI, konfai-apps | 1.8.7 |
| segmentation-models-pytorch | 0.4.0 |
| SimpleITK | 2.5.6 |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| scikit-image | 0.26.0 |
| LPIPS | 0.1.4 |
| PyYAML | 6.0.3 |

The TotalSegmentator model behind the Dice is the KonfAI port
`VBoussot/TotalSegmentator-KonfAI`, revision `0718569`, fetched by KonfAI.

The SAM-supervised models are trained with the SAM 2 source at commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4` (in `requirements.txt`).

## Hardware notes

Prediction and evaluation fit one 10 GB GPU; the durations given in
[REPRODUCE.md](REPRODUCE.md) are for an NVIDIA RTX 3080 with 78 GB of RAM. Keep
one prediction process on the GPU at a time.

Training, as configured, needs more:

- **GPU memory.** Batch 32 at 320×320 in single precision needs more than 10 GB.
  With `autocast: true` it uses 9 GB.
- **RAM.** KonfAI caches the dataset in RAM when it fits: about 165 GiB for the
  345 Task 1 training cases. With less RAM it reads the patches from disk. A
  patch is read as a disk region only from uncompressed `.mha` and only if no
  transform needs the whole volume, which the `percentile:99.5` clip of the
  input does: 4 s per iteration on compressed files, against 3.3 iterations per
  second (GPU-bound) on uncompressed inputs clipped beforehand (`max_value: max`
  in the configuration).

## Preprocessing environment

The registration preprocessing needs APIs absent from ordinary SimpleITK:
`ReadParameterFile` and `TransformixImageFilter`, provided by
`SimpleITK-SimpleElastix==2.0.0rc2.dev910` (Python 3.10) in an environment
separate from the main one (`requirements-preprocessing.txt`).

This prerelease installs from PyPI on Linux with Python 3.10; on other
platforms it may need to be rebuilt from SimpleElastix. The scripts check for
the required APIs and fail early with a clear message.

`scripts/preprocessing/cbct_synthesis.py` additionally needs the RTK command-line
tools (`rtkforwardprojections`, `rtksimulatedgeometry`, `rtkfdk`).

## Installation notes

- Install a PyTorch build compatible with the local driver.
- SAM training downloads no weights automatically: `scripts/download.py --only
  models` puts the documented file in `models/external/`.
- Table extraction and repository tests need neither a GPU nor medical images.
