# Configurations

`training/` contains the 50 training configurations of the paper: two tasks ×
(ELX pairs: MAE, SAM; IMPACT pairs: MAE, SAM, VGG) × five folds. The model is
validated and its learning-rate scheduler stepped every 2500 iterations
(`it_validation`, `it_lr_update`). A leading `~` on a training subset is KonfAI
syntax for excluding the held-out list.

`inference/*_tta.yaml` are the inference configurations: the original image and
two flips are predicted by each of the five fold models (15 predictions per
patient) and averaged.

`evaluation/predictions/` holds one KonfAI evaluation configuration per metric
(`image`: MAE, PSNR and SSIM; `seg`: Dice; `sam`: `d_SAM`; `lpips`;
`uncertainty`). `scripts/evaluate.py` fills in the reference image, the dataset,
the predictions folder and the run name. `evaluation/registration_controls/`
holds the same four metrics for the Table 9 controls, filled in by
`scripts/registration_controls/evaluate_registration_controls.py`.
