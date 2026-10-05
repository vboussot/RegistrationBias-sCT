# Model files

Model weights are intentionally not tracked: the 50 paper checkpoints total
roughly 15.7 GB, about 313 MB each. They are in the Hugging Face repository
`VBoussot/RegistrationBias-sCT`, under `checkpoints/`;
`scripts/download.py --only checkpoints` places them under `models/checkpoints/`
with the layout of `manifest.csv` and checks the SHA-256 digest recorded there.

The SAM 2.1 Hiera-small checkpoint is also excluded. Download the official
`sam2.1_hiera_small.pt` and place it at
`models/external/sam2.1_hiera_small.pt`. Its SHA-256 is
`6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`.

The TorchScript SAM encoder used by the calibrated d_SAM evaluation
(`SAM2.1_Small.pt`, 136 MB, SHA-256
`10c2fb000e57fa4adcb5f1c23aa7dfe6f622e1f649643ceb7e43e8188f024899`) is public in
the Hugging Face model repository `VBoussot/ImpactSynth`. KonfAI's `SAM_Perceptual` fetches it automatically, and
`scripts/download.py --only models` also places it in `models/external/`.

The TotalSegmentator model used for the Dice is fetched by KonfAI from
`VBoussot/TotalSegmentator-KonfAI`, pinned to revision `0718569`.

The VGG comparator uses torchvision's
`VGG16_Weights.IMAGENET1K_V1`. Torchvision downloads that third-party weight to
its normal cache on first use; it is not redistributed here.
